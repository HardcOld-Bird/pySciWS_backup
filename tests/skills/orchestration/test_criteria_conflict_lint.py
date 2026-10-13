"""批准单判据矛盾检测（backlog 20261013-071759-devops）。

反例是真实档案的形状：一份批准单里 ``success_criteria`` 段写「``taste_enabled`` 仍 false」，
其后又追加一段 ``**补充（用户指令 …）**`` 写「将 ``registry.taste_enabled`` 置 true」
（``state/suggestions/approved/20261012-164601-leader.md``）。机械验收只读前者，执行者只能
自己猜「后到者优先」——猜错被打回，猜对也留下一条与判据不符的交付读数。

三道闸各测各的落点，缺一就漏相应的回归：

1. **判定纯函数面**：只认「同一配置键在两处被要求取相反的布尔值」，同向 / 无追加段 /
   同句讲切换 / 键不同 / 键不带下划线 一律不报（误报的告警和没有告警一样没人读）；
2. **组长侧**：``orch suggestions`` 的呈现面标注 + ``cmd_approve`` 点名——**告警不阻断**，
   入队照做（改判据是组长的一笔，不是代码的权限）；
3. **组员侧**：任务书正文注入。stdout 只有组长读得到，受害的是执行端，所以取舍规则
   必须写进任务书本身。
"""

from __future__ import annotations

import argparse

from pysci.skills.orchestration.tools import dispatch, workflow

from .conftest import read_backlog, seed_raw_suggestion

#: 真实反例的同形样本（不读档案本身：判据面当前值是运行态，锁它=合法上线一次就红一次）。
CONFLICT_BODY = """# 改进建议（devops，待审批）

kind: usability
**批准单可被写成自相矛盾**

success_criteria：
- PREFERENCE v2 diff 草案落盘且 taste_enabled 仍 false、生效文件未动；
  全量无回归。

**补充（用户指令 2026-10-12 17:1x）**：完成 v2 蒸馏并落 diff 后，**将 registry.taste_enabled 置
true**（授权依据=用户原话）。
"""

#: 同一文本、追加段与判据同向（都要求 false）——不该报。
AGREE_BODY = CONFLICT_BODY.replace(
    "registry.taste_enabled 置\ntrue", "registry.taste_enabled 仍 false"
)

#: 同一句里同时出现 false 与 true：那是「把键由 false 改成 true」的切换陈述，不矛盾。
SWITCH_BODY = """# 改进建议（devops，待审批）

kind: usability
切换陈述。

success_criteria：
- taste_enabled 由 false 置 true 后有回归测试。

**补充（用户指令 2026-10-13）**：taste_enabled 由 false 置 true 即为本单目标。
"""


def _args(sid: str) -> argparse.Namespace:
    """approve 入参：判决通道走 override（组长自裁），不与被测的告警面混口径。"""
    return argparse.Namespace(
        suggestion_id=sid,
        note="",
        no_wake=True,
        jury=False,
        bench="",
        override="测试自裁",
    )


# ---------------------------------------------------------------------------
# 1. 判定纯函数面
# ---------------------------------------------------------------------------
def test_detects_boolean_conflict_between_criteria_and_amend():
    got = dispatch.criteria_conflicts(CONFLICT_BODY)
    assert [c["key"] for c in got] == ["taste_enabled"], got
    assert got[0]["criteria_value"] == "false" and got[0]["amend_value"] == "true"
    # 告警引用原文，不编造
    assert "taste_enabled 仍 false" in got[0]["criteria"]
    assert "registry.taste_enabled 置" in got[0]["amend"]
    assert "判据须单一源" in dispatch.conflict_notice(got[0])


def test_point_name_match_is_last_dotted_segment():
    """判据面写裸键、追加段写 ``registry.键`` ——点分名末段必须能对上，否则反例漏报。"""
    assert dispatch.criteria_conflicts(AGREE_BODY) == [], "同向不报"
    body = CONFLICT_BODY.replace(
        "taste_enabled 仍 false、生效文件未动", "enabled_flag 仍 false"
    ).replace("registry.taste_enabled 置\ntrue", "nested.enabled_flag 置 true")
    assert [c["key"] for c in dispatch.criteria_conflicts(body)] == ["enabled_flag"]


def test_no_amend_paragraph_is_not_a_conflict():
    body = CONFLICT_BODY.split("**补充（")[0]
    assert dispatch.criteria_conflicts(body) == []


def test_switch_statement_in_one_clause_is_not_a_conflict():
    assert dispatch.criteria_conflicts(SWITCH_BODY) == []


def test_different_keys_on_two_sides_are_not_a_conflict():
    body = CONFLICT_BODY.replace("registry.taste_enabled", "inspector_enabled")
    assert dispatch.criteria_conflicts(body) == []


def test_bare_word_is_not_treated_as_a_config_key():
    """无下划线的词（flag/kind/true）不进键候选——否则任何两段文本都能「同键」。"""
    body = (
        "# 改进建议（devops，待审批）\n\nkind: usability\n正文。\n\n"
        "success_criteria：\n- flag 仍 false\n\n"
        "**补充（用户指令）**：flag 置 true。\n"
    )
    assert dispatch.criteria_conflicts(body) == []


def test_conflicts_fail_open_on_empty_and_markerless_text():
    assert dispatch.criteria_conflicts("") == []
    assert dispatch.criteria_conflicts("# 标题\n\n只有散文，没有判据标记行。\n") == []


def test_superseded_registration_dissolves_the_conflict():
    """追加段里标 `superseded` 并**点名该键** = 单一源已裁决，lint 必须认账。

    不认账的话登记等于没登记，反例档案永远修不干净（判据「不再给相反要求」就无从验收）。
    登记必须是追加段的**续行**——另起一段就落在被检测的追加段之外（同 real 档案的写法）。
    """
    named = (
        CONFLICT_BODY.rstrip("\n")
        + "（本段 superseded 判据面的 taste_enabled 一条。）\n"
    )
    assert dispatch.criteria_conflicts(named) == []
    # 没点名键的泛泛 superseded 不算登记——否则任何一句免责声明都能消解所有告警
    vague = CONFLICT_BODY.rstrip("\n") + "（本段 superseded 一切旧文。）\n"
    assert [c["key"] for c in dispatch.criteria_conflicts(vague)] == ["taste_enabled"]


# ---------------------------------------------------------------------------
# 2. 组长侧：呈现面标注 + approve 点名（不阻断）
# ---------------------------------------------------------------------------
def test_suggestion_marks_flags_conflict():
    marks = workflow.suggestion_marks(CONFLICT_BODY)
    assert "[!] 判据矛盾（taste_enabled）" in marks, marks


def test_suggestion_marks_stay_quiet_when_criteria_is_single_source():
    body = CONFLICT_BODY.split("**补充（")[0]
    assert not any("判据矛盾" in m for m in workflow.suggestion_marks(body))


def test_approve_warns_but_still_enqueues(iso_state, capsys):
    sid = seed_raw_suggestion(iso_state, CONFLICT_BODY)

    rc = workflow.cmd_approve(_args(sid))

    out = capsys.readouterr().out
    assert rc == 0, out
    assert "[判据矛盾] taste_enabled" in out, out
    item = read_backlog(iso_state)[0]
    assert item["id"] == sid, "告警不阻断：入队照做（改判据的权限在组长，不在代码）"


# ---------------------------------------------------------------------------
# 3. 组员侧：矛盾与取舍规则进任务书正文（不写进任务书=执行者照旧得自己猜）
# ---------------------------------------------------------------------------
_APPROVED = "orchestration/state/suggestions/approved/20261013-071759-devops.md"


def _taskbook(evidence: str) -> str:
    return (
        f"基础设施改进任务（backlog FIFO 队首，id=20261013-071759-devops）：\n\n"
        f"摘要：批准单可被写成自相矛盾\n\n证据：{evidence}\n\n组长附注：（无）\n\n"
        "要求：以 worktree 隔离实施。\n"
    )


def test_write_taskbook_embeds_notice_and_still_dispatches(
    tmp_path, monkeypatch, capsys
):
    (tmp_path / _APPROVED).parent.mkdir(parents=True)
    (tmp_path / _APPROVED).write_text(CONFLICT_BODY, encoding="utf-8")
    monkeypatch.setattr(dispatch, "PROJECT_ROOT", tmp_path)
    pod = tmp_path / "pods" / "devops"

    dest = dispatch.write_taskbook(
        pod, text=_taskbook(_APPROVED), task_file=None, slug="s"
    )

    raw = dest.read_text(encoding="utf-8")
    out = capsys.readouterr().out
    assert "⚠ **判据矛盾" in raw, raw  # 组员可见
    assert "后到者优先" in raw and "taste_enabled" in raw
    assert "[!] 任务书 lint：[判据矛盾]" in out  # 组长侧同一条，两面同源
    assert raw.endswith("\n") and not raw.endswith("\n\n")  # 不破坏 end-of-file 约定


def test_write_taskbook_untouched_when_no_conflict(tmp_path, monkeypatch, capsys):
    (tmp_path / _APPROVED).parent.mkdir(parents=True)
    (tmp_path / _APPROVED).write_text(AGREE_BODY, encoding="utf-8")
    monkeypatch.setattr(dispatch, "PROJECT_ROOT", tmp_path)
    text = _taskbook(_APPROVED)

    dest = dispatch.write_taskbook(
        tmp_path / "pods" / "devops", text=text, task_file=None, slug="s"
    )

    raw = dest.read_text(encoding="utf-8")
    assert "判据矛盾" not in raw, raw[-200:]
    assert raw.rstrip("\n").endswith(text.rstrip("\n")), "干净批准单不得被改写"
    assert "任务书 lint" not in capsys.readouterr().out


def test_evidence_conflicts_fail_open(tmp_path, monkeypatch):
    monkeypatch.setattr(dispatch, "PROJECT_ROOT", tmp_path)
    assert dispatch.evidence_conflicts(_taskbook("（见建议归档）")) == []
    assert dispatch.evidence_conflicts(_taskbook(_APPROVED)) == [], (
        "档案读不到 → 不报也不炸"
    )
