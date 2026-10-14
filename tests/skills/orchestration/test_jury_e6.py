"""PSEP_3 pc：裁决书（verdict_summary）四落点 + E6 twin 终裁（PSEP_3 §4.4）。

钉新面：

① 裁决书：胜方 + 多数理由汇总 + 各席 confidence + text，落 jury ``<case>.json``、
   preference-journal（channel 记账）、``pysci decisions`` 数据面（journal 行带 finding_id）、
   TUI（mirror 读数）；
② E6：平票/inconclusive 过半 → **自动** twin 终裁收敛确定选项（decided_by=twin + 理由入账）；
③ twin 不可达/未表态 → 保守回退（未通过/挂起 + 出声，不静默放行）。

夹具复用 conftest 的 ``iso_jury`` 一族（mock transport，零真实请求）。
"""

from __future__ import annotations

import json
import re

from pysci.skills.orchestration.tools import journal as journal_tool
from pysci.skills.orchestration.tools import jury, mirror, taste, verdict
from pysci.skills.orchestration.tools.dispatch import DispatchOutcome

from .conftest import SEAT_KEY_ENV, ai_reply, seed_case, set_jury_panel

LIKE = "实施改动，收益是机械可验收"
DISLIKE = "维持现状"
KEY = "sk-TESTKEY0123456789abcdef"
OPTION_BLOCK = re.compile(r"## 选项 A\n(.*?)\n\n## 选项 B", re.S)


def content_driven(*against: str):
    """裁判桩：``against`` 型号偏好「维持现状」，其余偏好实施改动（内容驱动 → 双序一致）。"""

    def transport(model, messages, **kw):
        del kw
        liked = DISLIKE if model in against else LIKE
        return ai_reply(
            "A"
            if liked in OPTION_BLOCK.search(messages[-1]["content"]).group(1)
            else "B"
        )

    return transport


def open_court(iso_state, monkeypatch, panel, transport, *, key=KEY):
    """开庭准备：写 seats、注 Key、换 transport 桩，返回解析好的案卷。"""
    set_jury_panel(iso_state.parent / "registry.json", panel)
    if key is not None:
        monkeypatch.setenv(SEAT_KEY_ENV, key)
    if transport is not None:
        monkeypatch.setattr(jury, "chat_once", transport)
    return jury.parse_case(seed_case(iso_state))


def enable_twin(iso_state, *, enabled: bool = True):
    """就地打开/关闭 twin 开关并挂一个含宪章的 twin pod（供 E6 终裁）。"""
    reg = iso_state.parent / "registry.json"
    data = (
        json.loads(reg.read_text(encoding="utf-8"))
        if reg.exists()
        else {"version": 1, "exe": None, "models": {}, "members": {}, "checks": {}}
    )
    pod = iso_state.parent / "pods" / "twin"
    pod.mkdir(parents=True, exist_ok=True)
    (pod / "PREFERENCE.md").write_text(
        "# 判断宪章（测试夹具）\n\n- P-01 客观优质 > 审美\n", encoding="utf-8"
    )
    data["taste_enabled"] = enabled
    data.setdefault("members", {})["twin"] = {
        "pod": str(pod),
        "model_tier": "flash",
        "max_turns": 4,
        "timeout_s": 60,
        "cold_start": True,
        "readonly_extra": ["PREFERENCE.md"],
    }
    reg.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return pod


def twin_hop(choice="A", *, body=None, calls=None):
    """假 twin 派发：交付正文为判断 JSON（或指定 body）。"""

    def fake(member_id, **kw):
        if calls is not None:
            calls.append({"member": member_id, **kw})
        text = body
        if text is None:
            text = json.dumps(
                {"choice": choice, "reason": "P-01 优质优先", "confidence": 0.66},
                ensure_ascii=False,
            )
        return DispatchOutcome(
            code=0, kind="result", member=member_id, sid="sid-twin", body=text
        )

    return fake


# ===========================================================================
#  1. 裁决书四落点
# ===========================================================================
def test_verdict_summary_lands_four_places(iso_jury, monkeypatch):
    """多数裁决 → 裁决书齐备并落 jury JSON + journal（channel 记账）+ decisions 数据面 + TUI。"""
    run = jury.adjudicate(
        open_court(iso_jury, monkeypatch, ["m1", "m2", "m3"], content_driven("m3"))
    )
    rec = run.record
    vs = rec["verdict_summary"]
    assert vs["winner"] == "A" and vs["decided_by"] == "jury"
    assert vs["tally"] == {"A": 2, "B": 1}
    assert len(vs["seats"]) == 3
    assert vs["majority_reasons"], "多数理由汇总非空"
    assert vs["twin"] is None and "裁决=A" in vs["text"]

    # 落点 1：jury <case>.json 结构化字段
    on_disk = json.loads((jury.record_path(rec["case_id"])).read_text(encoding="utf-8"))
    assert on_disk["verdict_summary"]["text"] == vs["text"]

    # 落点 2：preference-journal（channel 记账；同一判决只记一次 kind=verdict_summary）
    rows = [
        r for r in journal_tool.read_records() if r.get("kind") == "verdict_summary"
    ]
    assert len(rows) == 1
    jr = rows[0]
    assert jr["channel"] == "jury" and jr["case_id"] == rec["case_id"]
    assert jr["finding_id"] == rec["case_id"], "decisions 面按 finding_id 关联可查"
    assert jr["reason"] == vs["text"]

    # 落点 3：decisions 数据源（journal.find_records 按 finding_id 命中）
    rulings = {
        str(r.get("finding_id")): r
        for r in journal_tool.read_records()
        if r.get("finding_id")
    }
    assert rulings[rec["case_id"]]["reason"] == vs["text"]

    # 落点 4：TUI（mirror 读数）
    assert any(vs["text"] in ln for ln in mirror.verdict_summary_lines())
    assert "裁决=A" in mirror._line_verdict_summary()


def test_verdict_summary_ignored_by_health_counting(iso_jury):
    """裁决书行是文书面、非又一票：health 通道用量不因它翻倍。"""
    journal_tool.append(
        {
            "ts": "2026-10-14T00:00:00+08:00",
            "situation": "x",
            "options": "",
            "choice": "A",
            "reason": "裁决=A",
            "source": "jury",
            "kind": "verdict_summary",
            "channel": "jury",
        }
    )
    assert verdict.health()["channels"]["jury"] == 0


def test_mirror_verdict_summary_graceful_when_empty(iso_jury):
    assert mirror.verdict_summary_lines() == ["  （无 jury 裁决书）"]
    assert "（无 jury 裁决记录）" in mirror._line_verdict_summary()


def test_build_verdict_summary_pure():
    """纯推导：胜方/各席/多数理由/twin 终裁，无需落盘。"""
    record = {
        "tally": {"A": 1, "B": 0},
        "inconclusive": [],
        "decided_by": "twin",
        "final_verdict": "B",
        "twin": {"reason": "外推：宪章无既例", "confidence": 0.55},
        "jurors": [
            {
                "seat": "s1",
                "model": "m1",
                "consistent": True,
                "vote": "A",
                "ballots": [{"real": "A", "reason": "理由甲", "confidence": 0.9}],
            }
        ],
    }
    vs = jury.build_verdict_summary(record)
    assert vs["winner"] == "B" and vs["decided_by"] == "twin"
    assert vs["twin"]["reason"] == "外推：宪章无既例"
    assert vs["majority_reasons"] == [], (
        "胜方为 twin 时多数理由按 jury 票面（无 B 票）为空"
    )


# ===========================================================================
#  2. 平票 → E6 twin 终裁收敛
# ===========================================================================
def test_tie_resolved_by_twin_converges(iso_jury, monkeypatch):
    """平票 → twin 终裁给出 B → 收敛到确定选项，decided_by/理由入账（不给平局）。"""
    case = open_court(iso_jury, monkeypatch, ["m1", "m2"], content_driven("m2"))
    enable_twin(iso_jury)
    calls: list[dict] = []
    monkeypatch.setattr(taste, "do_dispatch", twin_hop("B", calls=calls))

    run = jury.adjudicate(case)
    rec = run.record
    assert rec["tally"] == {"A": 1, "B": 1} and rec["escalate"] == "twin"
    assert rec["verdict"] == "B" and rec["final_verdict"] == "B"
    assert rec["decided_by"] == "twin"
    assert rec["twin"]["choice"] == "B" and rec["twin"]["reason"] == "P-01 优质优先"
    assert rec["twin"]["available"] is True
    assert "twin 终裁收敛：B" in run.message
    vs = rec["verdict_summary"]
    assert vs["winner"] == "B" and vs["decided_by"] == "twin"
    assert vs["twin"]["choice"] == "B"

    # 冷启动：twin 一跳必为 new 会话（无记忆纪律）
    assert [c["session"] for c in calls] == ["new"]
    # journal：twin 定案 → channel=twin（通道记账）
    jr = [r for r in journal_tool.read_records() if r.get("kind") == "verdict_summary"][
        -1
    ]
    assert jr["channel"] == "twin" and jr["decided_by"] == "twin"
    assert jr["final_verdict"] == "B"


def test_twin_choice_a_converges_symmetrically(iso_jury, monkeypatch):
    case = open_court(iso_jury, monkeypatch, ["m1", "m2"], content_driven("m2"))
    enable_twin(iso_jury)
    monkeypatch.setattr(taste, "do_dispatch", twin_hop("A"))
    rec = jury.adjudicate(case).record
    assert rec["verdict"] == "A" and rec["decided_by"] == "twin"


# ===========================================================================
#  3. twin 不可达/未表态 → 保守回退
# ===========================================================================
def test_twin_disabled_conservative_fallback(iso_jury, monkeypatch, capsys):
    """taste 未生效（twin 不可达）→ 不派发、判挂起、出声（不静默放行）。"""
    case = open_court(iso_jury, monkeypatch, ["m1", "m2"], content_driven("m2"))
    enable_twin(iso_jury, enabled=False)
    calls: list[dict] = []
    monkeypatch.setattr(taste, "do_dispatch", twin_hop("A", calls=calls))

    run = jury.adjudicate(case)
    rec = run.record
    assert calls == [], "twin 未生效不得派发"
    assert rec["verdict"] == "" and rec["decided_by"] == ""
    assert rec["escalate"] == "twin"
    assert rec["twin"]["available"] is False
    assert "保守回退" in run.message and "不可达" in run.message
    vs = rec["verdict_summary"]
    assert vs["winner"] == "" and vs["decided_by"] == ""

    # CLI 呈现：挂起 + 出声
    jury.cmd_jury(_ns(case.path))
    out = capsys.readouterr().out
    assert "保守回退挂起" in out


def test_twin_reachable_but_no_choice_falls_back(iso_jury, monkeypatch):
    """twin 可达但读不出选择（如未按格式）→ 同样保守回退（不把「没读出来」当表态）。"""
    case = open_court(iso_jury, monkeypatch, ["m1", "m2"], content_driven("m2"))
    enable_twin(iso_jury)
    monkeypatch.setattr(taste, "do_dispatch", twin_hop(body="没有 JSON 的交付"))
    rec = jury.adjudicate(case).record
    assert rec["verdict"] == "" and rec["decided_by"] == ""
    assert rec["twin"]["available"] is True, "通道可达，只是没给出确定选择"
    assert "未给出确定选择" in rec["twin"]["reason"]


# ===========================================================================
#  4. taste.ruling_for_case 面
# ===========================================================================
def test_ruling_for_case_returns_unavailable_when_disabled(iso_jury):
    case = jury.parse_case(seed_case(iso_jury))
    enable_twin(iso_jury, enabled=False)
    code, rec = taste.ruling_for_case(case)
    assert code == 2 and rec["available"] is False
    assert "taste_enabled" in rec["reason"] or "宪章" in rec["reason"]


def _ns(case_file):
    from argparse import Namespace

    return Namespace(case_file=str(case_file), dry=False, timeout=None)
