"""`command-surface.md` 的 `plan run`/`plan advance` 旗标表与 `-h` 实测一致
（backlog 20261015-202557-devops）。

现场：文档把两子命令的旗标写成「两者公共旗标」但枚举是**并集**，与两者实际**均不符**——
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


def test_plan_run_flags_match_help(capsys):
    assert _doc_flags("run") == _help_flags(capsys, ["plan", "run", "-h"])


def test_plan_advance_flags_match_help(capsys):
    assert _doc_flags("advance") == _help_flags(capsys, ["plan", "advance", "-h"])


def test_no_union_phrasing_left_in_command_surface():
    """不再出现「两者公共旗标」式并集表述（它就是上一版的失真源）。"""
    md = CMDSURF.read_text(encoding="utf-8")
    assert "公共旗标" not in md


def test_documented_tables_are_exactly_two_known_commands():
    """守卫：文档里带「X 旗标：」的行只有 run/advance 两张表（新增命令须同步）。"""
    md = CMDSURF.read_text(encoding="utf-8")
    labels = re.findall(r"^\s*#\s*([a-z][a-z0-9-]*) 旗标：", md, re.M)
    assert labels == ["run", "advance"], labels
