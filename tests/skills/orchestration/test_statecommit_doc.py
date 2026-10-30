"""`statecommit` docstring 的「state 收尾扫描」不得复活（backlog 20261015-133401-devops）。

缺陷（usability）：docstring 声称余下的跟踪面改写「交给既有的 state 收尾扫描」，而**仓里没有
那个实现**——读它的人以为有安全网，实际运行期改写的跟踪面长期滞留 main（实测未提交 16 项 →
合并交集闸 rc=2 停链）。按组长 2026-10-15 的取舍取「文档写清责任人与时机」一路：docstring 改成
点名的三处收口，并在 `references/decisions.md` 登记；本文件把这两件事钉住。

判据串按 backlog 20261014-075230-devops 约定**运行期拼接**（防常量折叠进 .pyc 让含二进制的
grep 也命中、判据把自己判红），且全树扫描限 `*.py` 并排除 `__pycache__`。
"""

from __future__ import annotations

from pathlib import Path

from pysci.paths import PROJECT_ROOT
from pysci.skills.orchestration.tools import statecommit

#: 误导表述（运行期拼接：字面量不进源码也不进字节码）。
MISLEADING = "收尾" + "扫描"

SRC = PROJECT_ROOT / "src"
DECISIONS = PROJECT_ROOT / "orchestration" / "references" / "decisions.md"


def _py_files() -> list[Path]:
    return [
        p
        for p in SRC.rglob("*.py")
        if "__pycache__" not in p.parts and ".venv" not in p.parts
    ]


def test_misleading_phrase_is_gone_from_src():
    """判据一（负向 grep）：`src/**.py` 里不得再出现该表述——它要么指向**真存在**的落点，
    要么已被改写成「由谁在何时提交」。"""
    offenders = [
        p.relative_to(PROJECT_ROOT).as_posix()
        for p in _py_files()
        if MISLEADING in p.read_text(encoding="utf-8", errors="ignore")
    ]

    assert offenders == [], f"该表述已无对应实现，不得复活：{offenders}"


def test_docstring_names_the_real_actors_and_timing():
    """判据一的另一面：docstring 必须**点名**谁在何时提交（三处收口），不留下「有安全网」的暗示。"""
    doc = statecommit.__doc__ or ""

    assert "commit_face" in doc, "须说明本模块负责哪张面（新增即提交）"
    for token in ("_commit_ledger", "chore(state)", "git add"):
        assert token in doc, f"须点名一处收口来源：{token}"
    assert "留在人手里" in doc or "人写的在飞编辑" in doc, (
        "须写明为何不做「每次改写即自动提交」（人写的在飞编辑会被钉进历史）"
    )


def test_decision_is_registered_in_the_verdict_register():
    """登记面：`references/decisions.md`（README §13 的权威表）含本单裁决行。"""
    text = DECISIONS.read_text(encoding="utf-8")

    assert "20261015-133401-devops" in text
    assert "不再" in text or "不" in text  # 行内须写明「不做自动提交」的取舍
