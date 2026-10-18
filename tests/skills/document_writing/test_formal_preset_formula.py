"""学院预设胶囊框宽公式的 **c 口径** 回归（tb-5 返工 b_rv 问题①）。

实现（b2 `chip_width`）对引用号传的是 ``"1,2"``/``"1–3"``（**不含方括号**），故公式里的
``c`` 必须按「方括号内字符数」计；文档若写「含括号」，每例系统性多算 0.28 in。
"""

from __future__ import annotations

from pysci.paths import ORCHESTRATION_ROOT


def _doc() -> str:
    return (
        ORCHESTRATION_ROOT
        / "skills"
        / "document-writing"
        / "references"
        / "ppt-preset-formal.md"
    ).read_text(encoding="utf-8")


def test_c_counts_inside_brackets_only():
    doc = _doc()
    assert "不含" in doc and "方括号内" in doc
    assert "含" + "括号" not in doc, "旧口径（含括号）不得回流"
    assert "`[1]`→1" in doc and "`[12]`→2" in doc and "`[1,2]`→3" in doc


def test_worked_examples_match_the_formula():
    def width(n: int, c: int) -> float:
        return 0.27 * n + 0.30 + 0.55 / n + (0.14 * c + 0.08 if c else 0)

    assert round(width(4, 1), 4) == 1.7375
    assert round(width(4, 0), 4) == 1.5175
    doc = _doc()
    assert "1.7375" in doc and "1.5175" in doc and "Δ=0.000" in doc


def test_quick_table_uses_same_definition():
    doc = _doc()
    row = next(
        line for line in doc.splitlines() if line.startswith("| 胶囊框宽（非线性）")
    )
    assert "方括号内" in row and "不含" in row and "c=引用号字符数" not in row
