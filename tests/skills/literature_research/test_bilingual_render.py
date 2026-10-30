"""bilingual render 渲染层回归（任务书 20261023-011225 §三）。

两层：预处理/HTML 组装是纯函数，无浏览器也跑；PDF 打印用例需要系统 Chrome，
按 ``test_pdf_render.py`` 先例 importorskip + 启动探针 guard，不可用即跳过。
"""

from __future__ import annotations

import base64
from pathlib import Path

import pytest

from pysci.skills.literature_research.tools import bilingual_render as br

pymupdf = pytest.importorskip("pymupdf", reason="需要 pymupdf（随 pymupdf4llm 提供）")
pytest.importorskip("playwright", reason="需要 playwright")

#: 1x1 PNG（测试用最小真图，pymupdf 可抽）。
_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8"
    "z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)

#: 合成 fixture：契约形态 bilingual.md（CJK+图+公式+表格+多段）。
MINI_BILINGUAL = """# 标题一
*Heading One*

---
中文句一。
*English sentence one with $x^2$ math.*

---
![x](images/pic.png)

---
图注句子。
*Fig. 1. The caption.*

---
$$
E = mc^2
$$

---
表格句。
*Table sentence.*

---
<table>
<tr><td>1</td><td>2</td></tr>
</table>

---
中译尾段句。
*Tail sentence.*
"""


@pytest.fixture(scope="session")
def browser_available() -> bool:
    """系统 Chrome/Edge 启动探针（不可用按先例 skip，不红）。"""
    try:
        from playwright.sync_api import sync_playwright

        from pysci.skills.literature_research.tools.browser_fetch import _launch

        with sync_playwright() as p:
            b, _ch = _launch(p, "auto", True)
            b.close()
        return True
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"playwright/系统浏览器不可用：{exc}")


@pytest.fixture
def workdir(tmp_path: Path) -> Path:
    """落合成 bilingual.md 与最小图片到临时工作区。"""
    (tmp_path / "images").mkdir()
    (tmp_path / "images" / "pic.png").write_bytes(_PNG)
    (tmp_path / "bilingual.md").write_text(MINI_BILINGUAL, encoding="utf-8")
    return tmp_path


# ---------------------------------------------------------------------------
# 预处理（纯函数）
# ---------------------------------------------------------------------------
def test_paragraph_boundary_becomes_thematic_break() -> None:
    """``---`` 段落边界换 ``***``——pandoc 会把空行后的 ``---`` 当 YAML 元数据吞。"""
    out = br.to_pandoc_md(MINI_BILINGUAL)
    assert "\n---\n" not in out
    assert out.count("\n***\n") == 7


def test_english_lines_become_em_with_content_kept() -> None:
    """斜体英文行改写为 ``<em>``，行内容逐字保留（含内嵌 markdown/数学）。"""
    out = br.to_pandoc_md(MINI_BILINGUAL)
    assert "<em>English sentence one with $x^2$ math.</em>" in out
    assert "<em>Fig. 1. The caption.</em>" in out
    assert "<em>Heading One</em>" in out


def test_text_lines_escaped_but_math_kept() -> None:
    """文本行 markdown 转义防误解析；``$...$`` 行内数学原样保留给 pandoc。"""
    md = "中*文* 与 _下划_ 与 [引] 与 $a+b$。\n"
    out = br.to_pandoc_md(md)
    assert out.rstrip("\n") == "中\\*文\\* 与 \\_下划\\_ 与 \\[引\\] 与 $a+b$。"


def test_blocks_pass_through_verbatim() -> None:
    """``$$`` 公式块与 ``<table>`` 块整块透传；图片行原样。"""
    out = br.to_pandoc_md(MINI_BILINGUAL)
    assert "\n$$\nE = mc^2\n$$\n" in out
    assert "<table>\n<tr><td>1</td><td>2</td></tr>\n</table>" in out
    assert "![x](images/pic.png)" in out


def test_heading_keeps_level_and_escapes_text() -> None:
    """标题保留 ``#`` 层级；标题文字转义。"""
    out = br.to_pandoc_md("## 标题 *带星*\n*x*\n")
    assert out.startswith("## 标题 \\*带星\\*\n")


# ---------------------------------------------------------------------------
# HTML 组装（纯函数）
# ---------------------------------------------------------------------------
def test_build_html_is_self_contained(workdir: Path) -> None:
    """自包含：样式/KaTeX/字体全部内联，图片保持相对路径。"""
    html = br.build_html(MINI_BILINGUAL, title="t")
    assert "<style>" in html and "katex" in html.lower()
    assert "url(fonts/" not in html  # 字体已内联为 data: URI
    assert "data:font/woff2;base64," in html
    assert "renderMathInElement" in html
    assert 'src="images/pic.png"' in html
    assert "<script src=" not in html and '<link rel="stylesheet"' not in html


# ---------------------------------------------------------------------------
# PDF 打印（需系统浏览器，不可用按先例跳过）
# ---------------------------------------------------------------------------
def test_render_pdf_end_to_end(workdir: Path, browser_available: bool) -> None:
    """合成契约 md（CJK+图+公式+表格+多段）→ PDF：页数>0、图已嵌入、公式无裸 $$/$。"""
    pdf = br.render_pdf(workdir / "bilingual.md", workdir / "bilingual.pdf")
    assert pdf.is_file()
    doc = pymupdf.open(str(pdf))
    try:
        assert doc.page_count > 0
        assert doc[0].get_images()  # 图片已嵌入
        text = "".join(p.get_text() for p in doc)
    finally:
        doc.close()
    assert "$" not in text  # 公式已排版，无裸 $$/$ 残留
    assert "标题一" in text and "Tail sentence." in text
    assert "E = mc" in text  # 公式内容已渲染为字形
    # 结构顺序与 md 一致（数学字形在抽取流里偏移，只对普通文本断言顺序）
    order = [
        "标题一",
        "Heading One",
        "中文句一",
        "The caption.",
        "Table sentence.",
        "Tail sentence.",
    ]
    idx = [text.find(m) for m in order]
    assert all(i >= 0 for i in idx), f"缺标记：{order} → {idx}"
    assert idx == sorted(idx)
    assert (workdir / "bilingual.html").is_file()  # 中间 HTML 留工作区作调试图
