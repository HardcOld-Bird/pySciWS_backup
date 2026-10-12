"""lint 幻影告警防御条款快照（backlog 20261013-045000-leader，路径 2 子集）。

Qoder harness lint 引擎两单实证会注入**捏造/陈旧** file:line 引用（orch.py:1040 越界、
幽灵文件 test_session_pool.py、§7.5 无中生有），仓内无法修源头，只能给受害者加抗体：
引用 file:line 的 lint/告警，处置前必须先对盘核验，不符即视为噪声丢弃、不得据其改代码。

判据口径=9 处（8 charter + 1 orchestration 技能真本，组长 2026-10-13 05:0x 更正注记）。
路径 2 先落 devops 写权限面内的 2 处（orchestration 技能真本 + devops AGENTS.md）；
charter 8 处属只读层红线，**挂起待用户授权**，由后续单补齐——故本测试把 charter 面单列
成 ``CHARTER_PENDING``，授权落地后并入 ``SURFACES`` 即全集生效，不改判据结构。
"""

from __future__ import annotations

import re

from pysci.paths import PROJECT_ROOT

#: 条款锚：只要求「先对盘核验」「不符」两个稳定词，措辞可微调不破断言。
ANCHOR = re.compile(r"对盘核验|不符即")


def _read(rel: str) -> str:
    return (PROJECT_ROOT / rel).read_text(encoding="utf-8")


#: 路径 2 已落面（devops 写权限面内）。
SURFACES = [
    "orchestration/skills/orchestration/SKILL.md",
    "orchestration/pods/devops/AGENTS.md",
]

#: charter 8 处：只读层红线，挂起待用户授权；授权到位后移入 SURFACES。
CHARTER_PENDING = [
    "orchestration/pods/devops/.qoder/rules/charter.md",
    "orchestration/pods/inspector/.qoder/rules/charter.md",
    "orchestration/pods/figure/.qoder/rules/charter.md",
    "orchestration/pods/sim/.qoder/rules/charter.md",
    "orchestration/pods/theory/.qoder/rules/charter.md",
    "orchestration/pods/lit/.qoder/rules/charter.md",
    "orchestration/pods/drawing/.qoder/rules/charter.md",
    "orchestration/pods/model3d/.qoder/rules/charter.md",
]


def test_path2_surfaces_carry_the_defense_clause():
    """路径 2 两面均含条款（成功判据「真本含该句」的快照断言）。"""
    for rel in SURFACES:
        assert ANCHOR.search(_read(rel)), f"缺 lint 幻影防御条款：{rel}"


def test_pending_charter_surface_count_is_eight():
    """charter 待授权面恰为 8 处（判据口径 9 = 2 已落 + 8 待落，防清单腐化）。"""
    assert len(CHARTER_PENDING) == 8
    for rel in CHARTER_PENDING:
        # 待授权面此刻**不得**已含条款：若已含，说明授权单已落地，应并入 SURFACES。
        assert not ANCHOR.search(_read(rel)), (
            f"{rel} 已含条款——请移入 SURFACES 并从 CHARTER_PENDING 删除"
        )
