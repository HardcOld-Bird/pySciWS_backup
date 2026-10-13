"""失败三分类器 + provider 报错原文入台账（种子 backlog 20261013-180000-leader）

用户设计 2026-10-13：所有提供商的并发带宽限制与项目需求同量级（mimo 明确 10 RPM /
10M TPM），「带宽不足」与「渠道真不可用」必须分开——**这是一切延期/回退策略的前置**。
而当前台账只存 stop_reason 短码，provider 报错原文根本没入账（本次实测），分类器无原料。

success_criteria 逐条对上：

- **台账含 error_text（构造失败跳实测）** → ④ 组；
- **分类器用真实样本 + 边界样例单测全绿（四类各 ≥2 例）** → ① 组（真实样本取自 12 个
  真实会话档：mimo「custom model service error…retry later」28 次、credit-usage 10 次）；
- **既有 quota/channel 识别行为回归不变** → ③ 组 + 既有 `test_quota_awareness.py` /
  `test_channel_error.py` 原样照绿；
- **词表可配置且单一源** → ② 组。
"""

from __future__ import annotations

import json

from pysci.skills.orchestration.tools import dispatch
from pysci.skills.orchestration.tools import failure_classes as fc
from pysci.skills.orchestration.tools import registry as reg
from pysci.skills.orchestration.tools.runner import Envelope

from .conftest import QUOTA_TEXT, envelope, fake_headless

#: 真实会话档实测原文（12 档案采样：28 / 10 次）——mimo 平台侧带宽/过载文案。
MIMO_TEXT = (
    "Custom model service error, please retry later or switch to another model. "
    "The custom model service is temporarily unavailable."
)


# ---------------------------------------------------------------------------
# ① 分类器：真实样本 + 边界样例（四类各 ≥2）
# ---------------------------------------------------------------------------
def test_real_corpus_samples_classify_correctly():
    """真实样本两类各归其位：mimo 文案→bandwidth；credit 文案→quota。"""
    assert fc.classify_failure(MIMO_TEXT) == "bandwidth"
    assert fc.classify_failure(QUOTA_TEXT) == "quota"


def test_bandwidth_variants():
    """带宽/限流簇（≥2 例）：429 与并发/速率并列，措辞各异都要吃住。"""
    for text in (
        "HTTP 429 Too Many Requests",
        "rate limit exceeded for this api key",
        "Concurrency limit reached: max organization concurrency: 1",
        "TPM quota exceeded, please retry",
        "the model is overloaded, try again later",
    ):
        assert fc.classify_failure(text) == "bandwidth", text


def test_outage_variants():
    """渠道真不可用簇（≥2 例）：鉴权 / 模型不存在 / 5xx / 连接被拒。"""
    for text in (
        "401 Unauthorized: invalid api key",
        "model not found: qwen-3.8-max",
        "HTTP 503 Service Unavailable",
        "connection refused by upstream",
        "no such model 'foo'",
    ):
        assert fc.classify_failure(text) == "outage", text


def test_quota_variants():
    """额度耗尽簇（≥2 例）：与原 QUOTA_ERROR_RE 同模式。"""
    for text in (
        "You've reached your credit usage limit.",
        "out of credits",
        "insufficient credits for this request",
    ):
        assert fc.classify_failure(text) == "quota", text


def test_unknown_and_empty():
    """无匹配 → unknown（不是「第四类」，是「判不出」）；空原文同理。"""
    assert fc.classify_failure("internal boom while running the task") == fc.UNKNOWN
    assert fc.classify_failure("") == fc.UNKNOWN
    assert fc.classify_failure(MIMO_TEXT, is_error=True) == "bandwidth"


def test_success_prose_is_never_classified_as_failure():
    """**成功跳**的正文可能只是提及这些词（复盘任务就是活样本）→ 不许判成故障。

    ``is_error=False`` 是调用方的明确声明：「这跳成功了，只是想拿这段文字问问」——
    此时一律 unknown。没有这道闸，「组员复盘额度事故」那一跳会被自己的复盘正文判成额度耗尽。
    """
    recap = f"复盘完成：原文是 {MIMO_TEXT}，已钉住策略。"
    assert fc.classify_failure(recap, is_error=False) == fc.UNKNOWN
    assert fc.classify_failure("", stop_reason="end_turn", is_error=False) == fc.UNKNOWN
    # 同一段文字在失败跳上照常分类（闸只认 is_error，不认文字内容）
    assert fc.classify_failure(recap, is_error=True) == "bandwidth"


def test_stop_reason_is_secondary_signal_only_when_error():
    """原文缺席时退回 stop_reason 作次级线索——但仅在该跳确为失败时。"""
    assert fc.classify_failure("", "rate_limit_error", True) == "bandwidth"
    assert fc.classify_failure("", "rate_limit_error", False) == fc.UNKNOWN


def test_precedence_prefers_quota_when_both_present():
    """两类同现取更硬者：额度耗尽要充值、带宽只要等——与 _detect_systemic 的次序一致。"""
    both = f"{MIMO_TEXT}｜{QUOTA_TEXT}"
    assert fc.classify_failure(both) == "quota"
    assert fc.classify_failure(MIMO_TEXT) == "bandwidth"


# ---------------------------------------------------------------------------
# ② 词表：可配置 + 单一源
# ---------------------------------------------------------------------------
def test_config_file_matches_registry_defaults():
    """随附词表与模块兜底**键齐**（改一边必改另一边）。"""
    table = fc.load_classes()
    assert set(fc.DEFAULT_CLASSES) <= set(table)
    assert set(fc.PRECEDENCE) == set(fc.DEFAULT_CLASSES)
    assert all(table[cls] for cls in fc.DEFAULT_CLASSES), "不许有空类"


def test_missing_or_broken_config_falls_back():
    """词表缺失/坏 JSON → 回落兜底（分类器不能在配置坏掉时失效）。"""
    missing = fc.load_classes(path=__import__("pathlib").Path("/nonexistent/x.json"))
    assert missing == fc.DEFAULT_CLASSES
    broken = fc.CLASSES_PATH.parent / "__broken_classes.json"
    broken.write_text("{not json", encoding="utf-8")
    try:
        assert fc.load_classes(path=broken) == fc.DEFAULT_CLASSES
        assert fc.precedence(path=broken) == fc.PRECEDENCE
    finally:
        broken.unlink()


def test_config_extension_takes_effect(tmp_path):
    """devops 可扩词表（附定义）——扩一条新 pattern，分类器立即可见。"""
    p = tmp_path / "classes.json"
    p.write_text(
        json.dumps(
            {
                "version": 1,
                "discipline": "t",
                "precedence": ["quota", "bandwidth", "outage"],
                "classes": {
                    "bandwidth": {"definition": "d", "patterns": ["throttled"]},
                },
            }
        ),
        encoding="utf-8",
    )
    table = fc.load_classes(path=p)
    assert table["bandwidth"] == ("throttled",)
    assert fc.combined_re("bandwidth", path=p).search("upstream throttled us")
    assert table["quota"] == fc.DEFAULT_CLASSES["quota"], "未覆盖的类回落兜底"


def test_dispatch_regexes_come_from_the_same_word_table():
    """**单一真源**：dispatch 的两个 regex 都由词表拼出（不是各存一份关键词）。"""
    table = fc.load_classes()
    for pat in table["quota"]:
        assert pat in dispatch.QUOTA_ERROR_RE.pattern, pat
    # channel 文本支 = 带宽 ∪ 故障（有意放宽）
    for pat in table["bandwidth"] + table["outage"]:
        assert pat in dispatch.CHANNEL_ERROR_RE.pattern, pat


def test_config_edit_reaches_dispatch_view():
    """词表扩一条 → dispatch 的联合视图立刻包含它（同一个 combined_re 缓存键）。"""
    import pathlib

    injected = "throttled by gateway"
    table = fc.load_classes()
    assert injected not in dispatch.CHANNEL_ERROR_RE.pattern
    extended = fc.combined_re("bandwidth", extra=(injected,))
    assert extended.search("upstream throttled by gateway")
    assert pathlib.Path(fc.CLASSES_PATH).exists(), (
        "随附词表必须在盘（dispatch 启动时从它编译）"
    )
    del table


def test_provider_error_re_matches_measured_wording():
    """既有用例的实测原文仍命中（CHANNEL_ERROR_RE 的对外行为不回归）。"""
    assert dispatch.CHANNEL_ERROR_RE.search(MIMO_TEXT)
    assert dispatch.CHANNEL_ERROR_RE.search(
        "custom model service error, please retry later or switch to another model"
    )
    assert dispatch.QUOTA_ERROR_RE.search(QUOTA_TEXT)


# ---------------------------------------------------------------------------
# ③ 既有识别行为不回归（原用例的另一半：直接对检测函数）
# ---------------------------------------------------------------------------
def test_detectors_unchanged_for_measured_samples(iso_dispatch):
    """两个检测函数对实测样本的判定与重构前一致。"""
    member = reg.Registry.load().member("quotamember")
    parsed = Envelope.parse(envelope(QUOTA_TEXT)[1])
    assert dispatch._detect_quota_exhaustion(parsed, "", "") is True

    mimo_env = Envelope.parse(envelope(MIMO_TEXT)[1])
    assert (
        dispatch._detect_channel_exhaustion(
            mimo_env, "", "", member=member, sid="ghost"
        )
        is True
    ), "会话档读不出时采信文案单特征（原行为）"


def test_ordinary_failure_not_systemic(iso_dispatch):
    """普通失败（不含 provider 签名）仍是 run_failed，不被新分类器误升格。"""
    member = reg.Registry.load().member("quotamember")
    env = Envelope(result="internal boom while running the task", is_error=True)
    assert dispatch._detect_systemic(env, "", "", member=member, sid="ghost") == ""


# ---------------------------------------------------------------------------
# ④ 台账 error_text（构造失败跳实测）
# ---------------------------------------------------------------------------
def _last_row(state):
    lines = (state / "ledger.jsonl").read_text(encoding="utf-8").strip().splitlines()
    return json.loads(lines[-1])


def test_failed_hop_records_provider_error_text(iso_dispatch, monkeypatch):
    """失败跳把 provider 原文写进台账新字段（本次实测的缺口）。"""
    fake_headless(monkeypatch, lambda n, cmd: envelope(QUOTA_TEXT))
    dispatch.do_dispatch("quotamember", text="做某事", quiet=True)

    row = _last_row(iso_dispatch)
    assert row["kind"] == "quota_exhausted"
    assert QUOTA_TEXT in row["error_text"], "台账必须留下可分类的原文"
    assert fc.classify_failure(row["error_text"], is_error=True) == "quota"


def test_error_text_is_capped(iso_dispatch, monkeypatch):
    """原文截 2000 字符（台账行不许爆）。"""
    fake_headless(
        monkeypatch, lambda n, cmd: envelope("x" * 9000, stop="stop_sequence")
    )
    dispatch.do_dispatch("quotamember", text="做某事", quiet=True)

    row = _last_row(iso_dispatch)
    assert len(row["error_text"]) == dispatch.ERROR_TEXT_MAX == 2000


def test_success_hop_leaves_error_text_empty(iso_dispatch, monkeypatch):
    """成功跳不写 error_text（没有失败原文可写）。"""
    body = f"<result>完工。原文引用 {MIMO_TEXT}</result>"
    fake_headless(
        monkeypatch,
        lambda n, cmd: envelope(body, is_error=False, stop="end_turn", rc=0),
    )
    dispatch.do_dispatch("quotamember", text="复盘", quiet=True, no_checks=True)

    assert _last_row(iso_dispatch)["error_text"] == ""


def test_ledger_field_defaults_for_old_rows():
    """旧台账行没有该字段 → dataclass 缺省空串（读旧账不炸）。"""
    from pysci.skills.orchestration.tools.ledger import LedgerEntry

    assert LedgerEntry(ts="", member="m", session_id="s").error_text == ""
