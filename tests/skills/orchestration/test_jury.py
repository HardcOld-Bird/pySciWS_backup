"""E3-prep 仲裁团全链路单测（mock transport，零真实请求）。

覆盖批准单第 8 点逐条：双序一致计票 / 单裁判不一致剔票 / 多数裁决 / 平票 escalate /
超时入队 / 缺 Key 降级 / 材料哈希冻结 / 非 Qwen 型号校验，外加 Key 永不落盘、简报
未批准只警告不拦、CLI 退出码三档。夹具见 conftest 的 ``iso_jury`` 一族。
"""

from __future__ import annotations

import json
import re
from argparse import Namespace

import pytest
import requests

from pysci.paths import ORCH_STATE_ROOT
from pysci.skills.orchestration.tools import jury

from .conftest import (
    ai_reply,
    biased_first_juror,
    display_label_for_liked,
    prefer_juror,
    seed_case,
    set_jury_panel,
)

LIKE = "实施改动，收益是机械可验收"
DISLIKE = "维持现状"
KEY = "sk-TESTKEY0123456789abcdef"
OPTION_BLOCK = re.compile(r"## 选项 A\n(.*?)\n\n## 选项 B", re.S)


def content_driven(*against: str):
    """裁判桩：``against`` 里的型号偏好「维持现状」，其余偏好实施改动（内容驱动 → 双序必一致）。"""

    def transport(model, messages, **kw):
        del kw
        liked = DISLIKE if model in against else LIKE
        return ai_reply(
            "A"
            if liked in OPTION_BLOCK.search(messages[-1]["content"]).group(1)
            else "B"
        )

    return transport


def biased_by_name(model: str) -> bool:
    return model.startswith("biased")


def open_court(iso_state, monkeypatch, panel, transport, *, key=KEY):
    """开庭准备：写 panel、注 Key、换 transport 桩，返回解析好的案卷。"""
    set_jury_panel(iso_state.parent / "registry.json", panel)
    if key is not None:
        monkeypatch.setenv(jury.KEY_ENV, key)
    if transport is not None:
        monkeypatch.setattr(jury, "chat_once", transport)
    return jury.parse_case(seed_case(iso_state))


# ---------------------------------------------------------------------------
# 案卷解析与自动附着材料（--dry 面）
# ---------------------------------------------------------------------------
def test_parse_case_requires_question_and_both_options(iso_jury):
    p = iso_jury / "evals" / "broken.md"
    p.write_text("## 问题\n只有一问。\n\n## 选项 A\n只有 A。\n", encoding="utf-8")
    with pytest.raises(ValueError, match="选项 B"):
        jury.parse_case(p)


def test_parse_case_reads_targets_and_sections(iso_jury):
    case = jury.parse_case(seed_case(iso_jury))
    assert case.case_id == "case-1" and case.targets == ["guards/target.txt"]
    assert LIKE in case.option_a and DISLIKE in case.option_b


def test_dossier_attaches_deterministic_material(iso_jury):
    """成功判据「--dry 输出完整组装材料」：git 读数 + 涉事文件字节读数 + 台账行。"""
    dossier = jury.build_dossier(jury.parse_case(seed_case(iso_jury)))
    assert "## 自动附着材料" in dossier and "stub-git-output" in dossier
    assert "guards/target.txt: 12 B / 2 行 / sha256=" in dossier
    assert "是否采纳该基建改动" in dossier


def test_relevant_ledger_rows_follow_targets(iso_jury):
    """「相关台账行」按涉事文件名命中：裁判看得见既往交付，材料仍是确定性抓取。"""
    from pysci.skills.orchestration.tools import ledger

    ledger.append(
        ledger.LedgerEntry(
            ts="2026-10-12T00:00:00+08:00",
            member="devops",
            session_id="s1",
            task_file="target.txt",
            kind="blocked",
            tags=["env-flaky"],
        )
    )
    rows = jury.relevant_ledger_rows(jury.parse_case(seed_case(iso_jury)))
    assert rows and "devops" in rows[-1] and "env-flaky" in rows[-1]


def test_diff_material_is_capped(iso_jury, monkeypatch):
    """git diff 是 prompt 体积的大头：超限截断并标注原大小（案卷不许被一次重构吞掉）。"""
    monkeypatch.setattr(jury, "run_git", lambda *a: "x" * (jury.DIFF_CAP_BYTES + 500))
    assert "（截断：git diff 全文 4500 B）" in jury.build_material(
        jury.parse_case(seed_case(iso_jury))
    )


# ---------------------------------------------------------------------------
# 双序、剔票、多数、平票
# ---------------------------------------------------------------------------
def test_dual_order_runs_each_juror_twice_and_agrees(iso_jury, monkeypatch):
    calls = []

    def transport(model, messages, **kw):
        del model, kw
        calls.append(messages)
        return ai_reply(display_label_for_liked(messages, LIKE))

    run = jury.adjudicate(open_court(iso_jury, monkeypatch, ["m1"], transport))
    assert run.status == "recorded"
    assert len(calls) == 2, "双序：每裁判 A/B 与 B/A 各一次"
    assert run.record["jurors"][0]["consistent"] is True
    assert run.record["votes"] == {"m1": "A"} and run.record["verdict"] == "A"
    assert run.record["escalate"] == ""


def test_position_biased_juror_is_not_counted(iso_jury, monkeypatch):
    """永远投标签 A＝位置偏好者：BA 局映射回来变成 B，自相矛盾 → 剔票不计。"""
    rec = jury.adjudicate(
        open_court(iso_jury, monkeypatch, ["m1"], biased_first_juror())
    ).record
    assert rec["jurors"][0]["consistent"] is False
    assert rec["inconclusive"] == ["m1"] and rec["votes"] == {}
    assert rec["verdict"] == "" and rec["escalate"] == "twin"


def test_majority_decides_and_minority_stays_in_record(iso_jury, monkeypatch):
    rec = jury.adjudicate(
        open_court(iso_jury, monkeypatch, ["m1", "m2", "m3"], content_driven("m3"))
    ).record
    assert rec["tally"] == {"A": 2, "B": 1}
    assert rec["votes"] == {"m1": "A", "m2": "A", "m3": "B"}
    assert rec["verdict"] == "A", "少数服从多数（§4 解析规则）"


def test_even_panel_tie_escalates_to_twin(iso_jury, monkeypatch):
    """偶数评审团平票＝模棱两可 → escalate=twin：挂起呈组长，不静默丢弃也不硬判。"""
    run = jury.adjudicate(
        open_court(iso_jury, monkeypatch, ["m1", "m2"], content_driven("m2"))
    )
    assert run.status == "recorded" and "escalate=twin" in run.message
    assert run.record["tally"] == {"A": 1, "B": 1}
    assert run.record["verdict"] == "" and run.record["escalate"] == "twin"


def test_inconclusive_majority_escalates(iso_jury, monkeypatch):
    """有效票不过半（多数裁判双序自相矛盾）同样判模棱两可——即便剩余那票是唯一的。"""

    def transport(model, messages, **kw):
        del kw
        if biased_by_name(model):
            return ai_reply("A")
        return ai_reply(display_label_for_liked(messages, LIKE))

    rec = jury.adjudicate(
        open_court(iso_jury, monkeypatch, ["honest", "biased-1", "biased-2"], transport)
    ).record
    assert rec["inconclusive"] == ["biased-1", "biased-2"]
    assert rec["verdict"] == "" and rec["escalate"] == "twin"


def test_transient_single_failure_recovers_on_retry(iso_jury, monkeypatch):
    """单局瞬时失败（HTTP 500）经一轮重试自愈：整案照常出裁决，attempts=2 留痕。"""
    n = []

    def flaky(model, messages, **kw):
        del model, kw
        n.append(1)
        if len(n) == 2:
            raise jury.JuryError("HTTP 500: upstream reset")
        return ai_reply(display_label_for_liked(messages, LIKE))

    run = jury.adjudicate(open_court(iso_jury, monkeypatch, ["m1"], flaky))
    assert run.status == "recorded" and run.record["verdict"] == "A"
    assert run.record["attempts"] == 2 and len(n) == 4


# ---------------------------------------------------------------------------
# 渠道降级：不崩、不烧、入队
# ---------------------------------------------------------------------------
def test_missing_key_degrades_without_any_call(iso_jury, monkeypatch):
    """缺 Key 的成功判据：可读降级 + 入队，零请求（不崩不烧）。"""
    set_jury_panel(iso_jury.parent / "registry.json", ["m1"])
    monkeypatch.delenv(jury.KEY_ENV, raising=False)
    calls = []
    monkeypatch.setattr(jury, "chat_once", lambda *a, **k: calls.append(a))
    run = jury.adjudicate(jury.parse_case(seed_case(iso_jury)))
    assert calls == [], "缺 Key 绝不能真发请求"
    assert run.status == "queued" and jury.KEY_ENV in run.message
    entry = json.loads(run.path.read_text(encoding="utf-8"))
    assert entry["case_id"] == "case-1" and jury.KEY_ENV in entry["reason"]


def test_key_from_dotenv_is_picked_up(iso_jury, monkeypatch):
    """用户早晨把 Key 写进项目根 .env 即可验收，不必动 shell 配置。"""
    monkeypatch.delenv(jury.KEY_ENV, raising=False)
    (iso_jury.parent / ".env").write_text(
        f'# SiliconFlow 账号 Key\n{jury.KEY_ENV}="{KEY}"\n', encoding="utf-8"
    )
    assert jury.read_key() == KEY


def test_empty_panel_degrades(iso_jury, monkeypatch):
    set_jury_panel(iso_jury.parent / "registry.json", [])
    monkeypatch.setenv(jury.KEY_ENV, KEY)
    run = jury.adjudicate(jury.parse_case(seed_case(iso_jury)))
    assert run.status == "queued" and "panel 未选型" in run.message
    assert not jury.record_path("case-1").exists()


def test_timeout_retries_once_then_enqueues(iso_jury, monkeypatch):
    """超时链（§4）：整体上限 → 重试一次 → 仍败入队留痕，不 raise、不落裁决。"""
    n = []

    def boom(model, messages, **kw):
        del model, messages, kw
        n.append(1)
        raise requests.Timeout("connect timeout")

    run = jury.adjudicate(open_court(iso_jury, monkeypatch, ["m1"], boom))
    assert len(n) == 4, "2 局 × 2 轮尝试"
    assert run.status == "queued" and run.record is None
    assert "已尝试 2 轮" in run.message
    entry = json.loads(run.path.read_text(encoding="utf-8"))
    assert "Timeout" in entry["reason"]
    assert not (iso_jury / "evals" / "jury" / "case-1.json").exists()


def test_successful_readjudication_clears_queue_entry(iso_jury, monkeypatch):
    """渠道恢复后重裁须销掉队列条目，不留僵尸案卷。"""
    case = jury.parse_case(seed_case(iso_jury))
    jury.enqueue(case, reason="先入队")
    assert jury.queue_path(case.case_id).exists()
    run = jury.adjudicate(open_court(iso_jury, monkeypatch, ["m1"], prefer_juror(LIKE)))
    assert run.status == "recorded"
    assert not jury.queue_path(case.case_id).exists()


# ---------------------------------------------------------------------------
# 非 Qwen 硬规则与配置面
# ---------------------------------------------------------------------------
def test_qwen_in_panel_refuses_to_open_court(iso_jury, monkeypatch):
    """家族外是硬规则：清单里混进 Qwen 系就拒开庭（违约要被看见，不静默剔除）。"""
    calls = []
    case = open_court(
        iso_jury,
        monkeypatch,
        ["deepseek-ai/DeepSeek-V3", "Qwen/Qwen3-235B"],
        lambda *a, **k: calls.append(a),
    )
    run = jury.adjudicate(case)
    assert run.status == "declined" and calls == [] and "Qwen" in run.message
    assert not jury.queue_path(case.case_id).exists(), "配置违规不是渠道故障，不入队"
    assert jury.qwen_violations(["moonshotai/Kimi-K2", "deepseek-ai/DeepSeek-V3"]) == []


def test_registry_jury_config_fail_open(iso_registry):
    set_jury_panel(iso_registry[0], ["m1"], deadline_s=-5, pricing="not-a-dict")
    cfg = jury.load_config()
    assert cfg["panel"] == ["m1"]
    assert cfg["deadline_s"] == jury.DEFAULT_JURY["deadline_s"], "非法值回落默认"
    assert cfg["base_url"] == "https://api.siliconflow.cn/v1"
    assert cfg["pricing"] == {}, "pricing 类型不符 → 空表（成本记 null，不猜价）"


# ---------------------------------------------------------------------------
# 材料冻结与 Key 安全
# ---------------------------------------------------------------------------
def test_dossier_hash_tracks_material(iso_jury, monkeypatch):
    """哈希＝「裁判当时看到什么」的指纹：材料不变则同、变则不同（回访计数的依据）。"""
    case = jury.parse_case(seed_case(iso_jury))
    h1 = jury.dossier_hash(jury.build_dossier(case))
    assert h1 == jury.dossier_hash(jury.build_dossier(case))
    monkeypatch.setattr(jury, "run_git", lambda *a: "different")
    assert jury.dossier_hash(jury.build_dossier(case)) != h1


def test_line_endings_do_not_shift_hash():
    """行尾是检出环境产物（与 sync.tree_hash 同一课），不算材料变更。"""
    assert jury.dossier_hash("a\r\nb") == jury.dossier_hash("a\nb")


def test_key_echoed_by_juror_is_redacted_in_record(iso_jury, monkeypatch):
    """裁判把 Key 回显进理由：留痕正文只留掩码（§4 Key 永不入库）。"""

    def leaky(model, messages, *, base_url, key, timeout_s):
        del model, messages, base_url, timeout_s
        return ai_reply("A", reason=f"依据 api_key {key} 判定")

    rec = jury.adjudicate(
        open_court(
            iso_jury, monkeypatch, ["m1"], leaky, key="sk-LEAKME0123456789abcdef"
        )
    ).record
    blob = json.dumps(rec, ensure_ascii=False)
    assert "sk-LEAKME0123456789abcdef" not in blob
    assert "[REDACTED]" in blob


def test_key_in_channel_error_is_redacted_in_queue(iso_jury, monkeypatch):
    """异常文本常带请求上下文：入队 reason 同样必须脱敏。"""
    secret = "sk-LEAKME0123456789abcdef"

    def boom(model, messages, **kw):
        del model, messages, kw
        raise RuntimeError(f"auth failed: bearer {secret}")

    run = jury.adjudicate(open_court(iso_jury, monkeypatch, ["m1"], boom, key=secret))
    assert run.status == "queued"
    blob = run.path.read_text(encoding="utf-8") + run.message
    assert secret not in blob and "[REDACTED]" in blob


def test_record_shape_matches_spec(iso_jury, monkeypatch):
    """留痕形状（批准单第 4 点）：问题、材料哈希、各裁判输出、双序一致性、计票、usage、成本。"""
    set_jury_panel(
        iso_jury.parent / "registry.json",
        ["m1"],
        pricing={"m1": {"input": 2.0, "output": 8.0}},
    )
    monkeypatch.setenv(jury.KEY_ENV, KEY)
    monkeypatch.setattr(jury, "chat_once", prefer_juror(LIKE))
    run = jury.adjudicate(jury.parse_case(seed_case(iso_jury)))
    rec = json.loads(run.path.read_text(encoding="utf-8"))
    for field in (
        "case_id",
        "question",
        "dossier_hash",
        "dossier",
        "brief",
        "panel",
        "jurors",
        "votes",
        "tally",
        "inconclusive",
        "verdict",
        "escalate",
        "usage",
        "cost",
    ):
        assert field in rec, field
    assert rec["cost"]["amount"] == pytest.approx((200 * 2.0 + 100 * 8.0) / 1e6)
    assert rec["usage"] == {
        "prompt_tokens": 200,
        "completion_tokens": 100,
        "total_tokens": 300,
    }
    assert [b["order"] for b in rec["jurors"][0]["ballots"]] == ["AB", "BA"]
    assert rec["brief"]["hash"], "简报哈希入档（冻结的是哪一版可回溯）"


def test_cost_unknown_without_pricing(iso_jury, monkeypatch):
    """无价目表 → cost.amount=None 并附 note：留痕里的成本数字必须能对外负责。"""
    rec = jury.adjudicate(
        open_court(iso_jury, monkeypatch, ["m1"], prefer_juror(LIKE))
    ).record
    assert rec["cost"]["amount"] is None and "价目表" in rec["cost"]["note"]


# ---------------------------------------------------------------------------
# 简报效力、解析兜底与 CLI 面
# ---------------------------------------------------------------------------
def test_unapproved_brief_warns_but_still_adjudicates(iso_jury, monkeypatch):
    """护栏级草案：未批准只警告（照样可用），出现 approved: 行才闭嘴。"""
    jury.JURY_BRIEF_PATH.write_text("# 草案 v1\n\n状态：待用户确认\n", encoding="utf-8")
    text, approved = jury.load_brief()
    assert approved is False and "草案" in text
    run = jury.adjudicate(open_court(iso_jury, monkeypatch, ["m1"], prefer_juror(LIKE)))
    assert run.status == "recorded" and "简报未批准" in run.message

    jury.JURY_BRIEF_PATH.write_text(
        "# 已批\n\napproved: 2026-10-12T07:00:00+08:00\n", encoding="utf-8"
    )
    assert jury.load_brief()[1] is True
    assert (
        "简报未批准"
        not in jury.adjudicate(jury.parse_case(seed_case(iso_jury))).message
    )


def test_shipped_jury_brief_is_draft_marked_for_user_confirmation():
    """随代码入库的简报草案：标明「待用户确认」、不含生效标记，直通规则写在简报而非代码里。"""
    text = (ORCH_STATE_ROOT / "evals" / "jury-brief.md").read_text(encoding="utf-8")
    assert "待用户确认" in text
    assert not re.search(r"^\s*>?\s*#*\s*approved\s*:", text, re.M)
    assert "自身专有基础设施" in text, "直通规则（§4 v0.6）须由简报承载"


def test_parse_vote_falls_back_on_unusable_answers():
    """强制 JSON 只是请求参数：解析侧兜底，坏 JSON/越界选项都算无效票。"""
    assert jury.parse_vote('{"choice": "A", "reason": "可", "confidence": 0.7}') == {
        "choice": "A",
        "reason": "可",
        "confidence": 0.7,
    }
    assert jury.parse_vote("我选 A，理由略")["choice"] == ""
    assert jury.parse_vote('{"choice": "C"}')["choice"] == ""
    assert jury.parse_vote('{"choice": "b", "confidence": 9}')["confidence"] == 1.0
    assert jury.flip("A", "BA") == "B" and jury.flip("", "AB") == ""


def ns(case_file, **kw):
    return Namespace(case_file=str(case_file), dry=kw.get("dry", False), timeout=None)


def test_cmd_jury_exit_codes(iso_jury, monkeypatch, capsys):
    """退出码三档（0 留痕 / 1 降级入队 / 2 拒开庭或案卷不可用）——脚本分流靠它。"""
    case = seed_case(iso_jury)
    set_jury_panel(iso_jury.parent / "registry.json", ["m1"])
    monkeypatch.setenv(jury.KEY_ENV, KEY)
    monkeypatch.setattr(jury, "chat_once", prefer_juror(LIKE))
    assert jury.cmd_jury(ns(case)) == 0
    assert "裁决=A" in capsys.readouterr().out

    set_jury_panel(iso_jury.parent / "registry.json", ["Qwen/Qwen3"])
    assert jury.cmd_jury(ns(case)) == 2
    assert "Qwen" in capsys.readouterr().out

    set_jury_panel(iso_jury.parent / "registry.json", [])
    assert jury.cmd_jury(ns(case)) == 1
    assert "队列" in capsys.readouterr().out

    bad = iso_jury / "evals" / "broken2.md"
    bad.write_text("## 问题\n缺一题。\n", encoding="utf-8")
    assert jury.cmd_jury(ns(bad)) == 2
    assert "案卷不可用" in capsys.readouterr().out


def test_cmd_jury_dry_prints_dossier_and_calls_nothing(iso_jury, monkeypatch, capsys):
    """--dry 判据：打印完整案卷（含自动附着段与材料哈希），零调用、零留痕。"""
    set_jury_panel(iso_jury.parent / "registry.json", ["m1"])
    monkeypatch.setenv(jury.KEY_ENV, KEY)
    calls = []
    monkeypatch.setattr(jury, "chat_once", lambda *a, **k: calls.append(a))
    rc = jury.cmd_jury(ns(seed_case(iso_jury), dry=True))
    out = capsys.readouterr().out
    assert rc == 0 and calls == []
    assert "## 自动附着材料" in out and "材料哈希" in out
    assert not jury.record_path("case-1").exists()
