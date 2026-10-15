"""`command-surface.md` 的**顶层命令覆盖**与建议单口径（backlog 20261015-205905）。

现场：`requeue`/`resolve`/`watchdog`/`drain-stop`/`refresh-models`/`twin-review` 六个**真实存在**
的顶层命令在任何文档面都零命中——只能靠别的报错偶然看到 argparse 的 usage 行。本文件把
「顶层命令全部有档」钉成机械判据（同类缺口此后再现即红），并钉建议单的两条口径。
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

#: 205905 点名的六个（本轮补齐）。
NAMED = (
    "requeue",
    "resolve",
    "watchdog",
    "drain-stop",
    "refresh-models",
    "twin-review",
)

_USAGE_CHOICES = re.compile(r"\{(.*?)\}", re.S)


def _doc() -> str:
    return CMDSURF.read_text(encoding="utf-8")


def _top_level_commands(capsys) -> list[str]:
    """从 `orch -h` 的 usage 里取顶层命令集合（argparse 会把花括号组**折行**，故跨行匹配）。"""
    with pytest.raises(SystemExit):
        orch.main(["-h"])
    out = capsys.readouterr().out
    m = _USAGE_CHOICES.search(out)
    assert m, "未能从 orch -h 的 usage 解析出顶层命令集合"
    cmds = [c.strip() for c in m.group(1).replace("\n", " ").split(",")]
    cmds = [c for c in cmds if re.fullmatch(r"[a-z][a-z0-9-]*", c)]
    assert len(cmds) >= 10, f"顶层命令集合看着不对：{cmds}"
    return cmds


def _documented(doc: str, cmd: str) -> bool:
    """命令在文档里出现——**含 `a|b` 合并写法**（如 `inspector-on|off` 即覆盖 `inspector-off`）。"""
    return cmd in doc or f"|{cmd}" in doc or f"{cmd}|" in doc


def test_every_top_level_command_is_documented(capsys):
    """机械护栏：`pysci-orch -h` 的每个顶层命令都要在 command-surface.md 里有档。"""
    doc = _doc()
    missing = [c for c in _top_level_commands(capsys) if not _documented(doc, c)]
    assert not missing, f"顶层命令缺档：{missing}（command-surface.md）"


@pytest.mark.parametrize("cmd", NAMED)
def test_named_commands_present(cmd):
    assert _documented(_doc(), cmd), f"{cmd} 未在 command-surface.md 出现"


def test_named_commands_have_semantics_not_just_the_word():
    """不只是「出现过」：各自带一句语义/作用域。"""
    doc = _doc()
    assert "needs_leader → pending" in doc  # requeue
    assert "确认已达标的 needs_leader 条目 → done" in doc  # resolve
    assert "广播纪元停全部实例" in doc  # watchdog
    assert "协作式停止" in doc  # drain-stop
    assert "模型映射" in doc  # refresh-models
    assert "复核 twin" in doc  # twin-review


def test_requeue_resolve_scope_is_documented():
    """两者作用域写清（backlog 条目 + needs_leader），免被误当建议单销账口。"""
    doc = _doc()
    assert "**不收建议单 id**" in doc


def test_duplicate_suggestion_policy_documented():
    """重复建议单的处置口径 + 历史面审计读法（criteria ②）。"""
    doc = _doc()
    assert "重复建议单" in doc
    assert "合并至哪单" in doc
    assert "以 `--override` 正文为准" in doc


def test_suggestion_member_column_semantics_documented():
    """组长代补单在 `suggestions` 列表的 `[member]` 显示口径（criteria ③）。"""
    doc = _doc()
    assert "`[member]` 列" in doc
    assert "代补单" in doc and "目标" in doc
