"""lint 幻影告警防御条款快照（backlog 20261013-045000-leader / 20261013-061000-leader）。

Qoder harness lint 引擎两单实证会注入**捏造/陈旧** file:line 引用（orch.py:1040 越界、
幽灵文件 test_session_pool.py、§7.5 无中生有），仓内无法修源头，只能给受害者加抗体：
引用 file:line 的 lint/告警，处置前必须先对盘核验，不符即视为噪声丢弃、不得据其改代码。

判据口径=9 处（8 charter + 1 orchestration 技能真本，组长 2026-10-13 05:0x 更正注记）。
路径 2 先落 devops 写权限面内的 2 处（技能真本 + devops AGENTS.md）；charter 8 处属只读层
红线，用户 2026-10-13 05:5x「完全授权 charter 修改」后由 20261013-061000-leader 补齐，
本测试随之从「待授权」升格为全集断言（含同批落地的 #89 remedy_shape 纪律句）。
"""

from __future__ import annotations

import re

from pysci.paths import PROJECT_ROOT

#: 条款锚：只要求「先对盘核验」「不符」两个稳定词，措辞可微调不破断言。
ANCHOR = re.compile(r"对盘核验|不符即")

#: 条款应落在「读交付/判读」这一语境里——蒸馏时若把它挪进无人读的角落，断言会咬。
_SECTION = re.compile(r"^#{1,3}\s.*(交付|作业纪律|判读)", re.M)

#: 8 处 charter（只读层，用户 2026-10-13 05:5x 授权）——判据 9 处 = 这 8 处 + 技能真本；
#: SURFACES 另含路径 2 已落的 devops AGENTS.md（自维护层，不计入 9 处口径）。
CHARTERS = [
    "orchestration/pods/devops/.qoder/rules/charter.md",
    "orchestration/pods/inspector/.qoder/rules/charter.md",
    "orchestration/pods/figure/.qoder/rules/charter.md",
    "orchestration/pods/sim/.qoder/rules/charter.md",
    "orchestration/pods/theory/.qoder/rules/charter.md",
    "orchestration/pods/lit/.qoder/rules/charter.md",
    "orchestration/pods/drawing/.qoder/rules/charter.md",
    "orchestration/pods/model3d/.qoder/rules/charter.md",
]

SURFACES = [
    "orchestration/skills/orchestration/SKILL.md",
    "orchestration/pods/devops/AGENTS.md",
    *CHARTERS,
]


def _read(rel: str) -> str:
    return (PROJECT_ROOT / rel).read_text(encoding="utf-8")


def test_all_surfaces_carry_the_defense_clause():
    """判据 9 处（8 charter + 1 技能真本）各含条款，另加路径 2 的 devops AGENTS.md。"""
    assert len(CHARTERS) == len(set(CHARTERS)) == 8, "charter 清单须恰为授权 8 处"
    assert len(SURFACES) == len(set(SURFACES)) == 10
    for rel in SURFACES:
        assert ANCHOR.search(_read(rel)), f"缺 lint 幻影防御条款：{rel}"


def test_charter_pending_list_is_empty():
    """授权已到位：不得再有「待授权」挂起面（残留即清单腐化）。"""
    pending = [
        rel
        for rel in CHARTERS
        if not (PROJECT_ROOT / rel).exists() or not ANCHOR.search(_read(rel))
    ]
    assert pending == [], f"charter 面缺失条款：{pending}"


def test_clause_sits_in_delivery_reading_context():
    """条款要挂在「交付/作业纪律/判读」语境下——放错段落等于没放（没人会读到）。"""
    for rel in CHARTERS:
        text = _read(rel)
        hit = ANCHOR.search(text)
        assert hit, f"{rel} 缺条款"
        headings = [m for m in _SECTION.finditer(text) if m.start() < hit.start()]
        assert headings, (
            f"{rel} 条款位于任何交付段之前——落点错误：{text[: hit.start()]!r}"
        )


def test_devops_charter_carries_remedy_shape_discipline():
    """#89 同批落点：devops charter 有 remedy_shape 分流纪律句（与任务书注入同义）。"""
    text = _read("orchestration/pods/devops/.qoder/rules/charter.md")
    assert "remedy_shape" in text, "devops charter 缺修复形状纪律句"
    assert "restore" in text and "<blocked>" in text, (
        "纪律句须点明 restore/change 两态与 blocked 上报"
    )
