"""审批口 `success_criteria` 形态解析（backlog adhoc-criteria-parse）。

缺陷（组长 2026-10-15 实测）：建议文件把判据写在**列表项**行（`- success_criteria: …`），
解析器只认行首裸标记 → 整条漏读 → 被判「缺可证伪成功判据，拒绝入队」，真判据被误拒
（实案 `20261015-130838-writing`，组长被迫 `--criteria` 手工补录绕过）。

本文件钉：三种书写形态（行首/列表项/粗体）**都能读出判据**、**行内值**也算判据、
以及同族的 `kind` 前缀盲区一并修好。
"""

from __future__ import annotations

from pysci.skills.orchestration.tools import delivery

CRIT = "- 主树 `pytest tests` 全绿；工作树零未跟踪项"


def test_reads_line_start_form():
    _kind, crit = delivery.parse_suggestion_fields(
        f"kind: defect\n\nsuccess_criteria:\n{CRIT}\n"
    )
    assert crit == CRIT


def test_reads_list_item_form():
    """**本缺陷的原件形态**：列表项前缀 + 其后要点行。"""
    _kind, crit = delivery.parse_suggestion_fields(
        f"kind: defect\n\n- success_criteria:\n  {CRIT}\n"
    )
    assert crit == CRIT


def test_reads_bold_form():
    _kind, crit = delivery.parse_suggestion_fields(
        f"kind: defect\n\n**success_criteria**:\n{CRIT}\n"
    )
    assert crit == CRIT


def test_reads_markdown_heading_form():
    """**markdown 标题形态**（``## success_criteria:``）也算——文档原话是「段」，读者自然
    这么写；旧口径整条漏读、真实判据被当「缺判据」误拒（20261016-203600-devops 首写实锤）。"""
    _kind, crit = delivery.parse_suggestion_fields(
        f"kind: defect\n\n## success_criteria:\n{CRIT}\n"
    )
    assert crit == CRIT


def test_reads_markdown_heading_with_kind_heading():
    """同族顺检：``## kind: usability`` 标题形态也能读出分类。"""
    kind, _crit = delivery.parse_suggestion_fields(
        "## kind: usability\n\nsuccess_criteria:\n- x\n"
    )
    assert kind == "usability"


def test_reads_inline_value_on_the_marker_line():
    """**行内值**也算判据（原件把全文直接写在标记行上——旧口径视而不见）。"""
    _kind, crit = delivery.parse_suggestion_fields(
        "kind: defect\n\n- success_criteria: 任务书声明的目录（含中文路径）能被注入 PYSCI_TASK_DIRS\n"
    )
    assert crit.startswith("任务书声明的目录") and "PYSCI_TASK_DIRS" in crit


def test_kind_marker_also_survives_a_list_item():
    """同族顺检：`- kind: defect` 不再因前缀而漏读整份分类。"""
    kind, _crit = delivery.parse_suggestion_fields(
        "- kind: defect\n\nsuccess_criteria:\n- x\n"
    )
    assert kind == "defect"


def test_replays_the_originally_rejected_suggestion_file():
    """**原案重放**：`state/suggestions/…130838-writing` 的原件形态 → 判据被正确读出。"""
    from pysci.paths import ORCHESTRATION_ROOT

    f = (
        ORCHESTRATION_ROOT
        / "state"
        / "suggestions"
        / "approved"
        / "20261015-130838-writing.md"
    )
    if not f.is_file():  # 留档不在时跳过（不把用例绑死在某个具体文件上）
        import pytest

        pytest.skip("原件留档缺席")
    text = f.read_text(encoding="utf-8")

    _kind, crit = delivery.parse_suggestion_fields(text)

    assert crit.strip(), "原件里判据完整——必须被读出，不得再判「缺判据」"
    assert "PYSCI_TASK_DIRS" in crit


# ---------------------------------------------------------------------------
# 残余缺口（backlog 20261016-224400）：标记行后隔**空行**（markdown 标题惯例）仍读空
# ---------------------------------------------------------------------------
def test_reads_heading_then_blank_line_then_bullets():
    """**第五例形态**：标题形态＋markdown 惯例空行＋要点行——旧口径「标记行后直到空行」
    把标题后那个空行读成判据结束，于是标准写法仍被判「缺判据」（本次实证的第五例）。"""
    _kind, crit = delivery.parse_suggestion_fields(
        f"kind: defect\n\n## success_criteria:\n\n{CRIT}\n"
    )
    assert crit == CRIT


def test_reads_bare_marker_then_blank_line_then_bullets():
    """同族的裸标记形态也跨空行（不是只有标题形态受益）。"""
    _kind, crit = delivery.parse_suggestion_fields(
        f"kind: defect\n\nsuccess_criteria：\n\n{CRIT}\n"
    )
    assert crit == CRIT


def test_marker_with_nothing_after_is_still_rejected():
    """**反例**：标记行后除空行外无任何正文（到 EOF）→ 仍判「缺判据」。"""
    _kind, crit = delivery.parse_suggestion_fields(
        "kind: defect\n\n## success_criteria:\n\n"
    )
    assert crit.strip() == ""


def test_marker_followed_by_another_section_is_rejected():
    """**反例护栏**：空行后紧跟下一节标题 → 判据段终止，不把后文当判据（防真空被冒名）。"""
    _kind, crit = delivery.parse_suggestion_fields(
        "kind: defect\n\n## success_criteria:\n\n## 现象\n\n这一节不是判据\n"
    )
    assert crit.strip() == ""


def test_inline_marker_does_not_absorb_following_paragraph():
    """**反例护栏**：行内值形态按旧口径「空行即止」——不越过空行把后文段落吞成判据。"""
    _kind, crit = delivery.parse_suggestion_fields(
        "success_criteria: 行内判据全文\n\n后文段落（不是判据）\n"
    )
    assert crit == "行内判据全文"
