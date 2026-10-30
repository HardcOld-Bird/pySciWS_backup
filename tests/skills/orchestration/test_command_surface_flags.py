"""`command-surface.md` 的 `plan advance` 旗标表与 `-h` 实测一致
（backlog 20261015-202557-devops；20261016-220500 起 `plan run` 已废，其独有旗标并入 advance）。

现场：文档曾把两子命令的旗标写成「两者公共旗标」但枚举是**并集**，与两者实际**均不符**——
`plan advance … --name tb3-i2` 直接 `unrecognized arguments`（一次派发失败）；`plan run`
并没有 `--dry`。枚举既然存在就会被照抄，故钉成机械比对：**文档集合 == `-h` 集合**。
"""

from __future__ import annotations

import re

import pytest

from pysci.paths import ORCHESTRATION_ROOT
from pysci.skills.orchestration.tools import orch

CMDSURF = (
    ORCHESTRATION_ROOT
    / "skills"
    / "orchestration"
    / "references"
    / "command-surface.md"
)

#: argparse help 里「选项行」的形状：行首缩进后紧跟 `--flag`。
_OPT_LINE = re.compile(r"^\s+(--[a-z][a-z0-9-]*)", re.M)


def _help_flags(capsys, argv: list[str]) -> set[str]:
    with pytest.raises(SystemExit) as ei:
        orch.main(argv)
    assert ei.value.code == 0
    return set(_OPT_LINE.findall(capsys.readouterr().out))


def _doc_flags(label: str) -> set[str]:
    md = CMDSURF.read_text(encoding="utf-8")
    m = re.search(rf"^\s*#\s*{label} 旗标：(.*)$", md, re.M)
    assert m is not None, f"command-surface.md 缺「{label} 旗标」行"
    return set(re.findall(r"--[a-z][a-z0-9-]*", m.group(1)))


def test_plan_advance_flags_match_help(capsys):
    assert _doc_flags("advance") == _help_flags(capsys, ["plan", "advance", "-h"])


def test_plan_advance_absorbed_run_only_flags(capsys):
    """20261016-220500：`plan run` 的独有旗标 `--name`/`--max-turns` 由 advance 承接。"""
    flags = _help_flags(capsys, ["plan", "advance", "-h"])
    assert {"--name", "--max-turns"} <= flags, flags


def test_no_union_phrasing_left_in_command_surface():
    """不再出现「两者公共旗标」式并集表述（它就是上一版的失真源）。"""
    md = CMDSURF.read_text(encoding="utf-8")
    assert "公共旗标" not in md


def test_documented_tables_are_exactly_two_known_commands():
    """守卫：文档里带「X 旗标：」的行只剩 advance 一张表（新增命令须同步）。"""
    md = CMDSURF.read_text(encoding="utf-8")
    labels = re.findall(r"^\s*#\s*([a-z][a-z0-9-]*) 旗标：", md, re.M)
    assert labels == ["advance"], labels


# ---------------------------------------------------------------------------
# `plan run` 已废弃（backlog 20261016-220500）：命令面移除 + 活文档零残留
# ---------------------------------------------------------------------------
#: 已废同步子命令名（**运行期拼接**：防常量折叠把字面量折进 .pyc，令含二进制的
#: grep 误命中、判据把自己判红——任务书 grep 约定）。
_DEAD_SYNC = "plan " + "run"


def test_dead_sync_subcommand_removed(capsys):
    """命令面移除：argparse 直接拒收（用户明示无需向后兼容）。"""
    with pytest.raises(SystemExit) as ei:
        orch.main(["plan", "run", "p1", "--text", "t"])
    assert ei.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_dead_sync_subcommand_absent_from_plan_usage(capsys):
    with pytest.raises(SystemExit):
        orch.main(["plan", "-h"])
    out = capsys.readouterr().out
    m = re.search(r"\{([^}]*)\}", out, re.S)
    assert m, out
    choices = [c.strip() for c in m.group(1).replace("\n", " ").split(",")]
    assert "advance" in choices, choices
    assert _DEAD_SYNC.split()[-1] not in choices, choices


def test_live_docs_have_no_dead_sync_command():
    """活文档面（真本技能全文 + 现行设计引用）不得再提已废命令。

    历史留档（`references/legacy-readme-v2.2.md`、`decisions.md`、`history-and-risks.md`）
    如实记录当时状态，**不在**扫描面内。
    """
    faces = sorted((ORCHESTRATION_ROOT / "skills" / "orchestration").rglob("*.md")) + [
        ORCHESTRATION_ROOT / "references" / "plan-and-routing.md",
        ORCHESTRATION_ROOT / "references" / "wake-and-events.md",
    ]
    hits = [
        p.relative_to(ORCHESTRATION_ROOT).as_posix()
        for p in faces
        if _DEAD_SYNC in p.read_text(encoding="utf-8")
    ]
    assert not hits, f"活文档面仍提已废命令：{hits}"
