"""bilingual render 渲染层回归（任务书 20261023-011225 §三）。

两层：预处理/HTML 组装是纯函数，无浏览器也跑；PDF 打印用例需要系统 Chrome，
按 ``test_pdf_render.py`` 先例 importorskip + 启动探针 guard，不可用即跳过。
"""

from __future__ import annotations

import base64
import re
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
# 打印版式（2026-11-03 用户裁决四项）：双栏 / 窄页边距 / 页码 / 标注修复
# ---------------------------------------------------------------------------
def test_print_css_two_columns_and_layout_guards() -> None:
    """样式真源：双栏、body margin 清零、图片栏宽、图组不断裂、宽元素防线、长串可断行。"""
    css = (br.TEMPLATE_DIR / "bilingual.css").read_text(encoding="utf-8")
    assert "column-count: 2" in css, "全文默认双栏"
    assert "column-gap: 0.7cm" in css
    assert re.search(r"body\s*\{[^}]*margin: 0;", css), (
        "body margin 须清零（浏览器默认 8px 会蚕食页边距）"
    )
    assert "max-width: 100%" in css, "图片适应栏宽"
    assert ".figure" in css and "break-inside: avoid" in css, "图与图注不被栏/页断裂"
    assert "column-span: all" in css, "宽公式/表格跨双栏防线"
    assert "overflow-wrap: anywhere" in css, "长串（DOI/URL）栏内可断行"
    assert "column-span" in br._INIT_JS and "CSS.supports" in br._INIT_JS, (
        "宽元素须 DOM 量宽后加 .wide（不支持列跨时退化等比缩放）"
    )


def test_page_margin_below_one_cm() -> None:
    """页边距 <1cm 硬线（2026-11-03 用户裁决「1cm 以下理想」）。"""
    assert 0 < br.PAGE_MARGIN_CM < 1.0


def test_add_page_numbers_odd_right_even_left_in_bottom_band(tmp_path: Path) -> None:
    """页码叠加：每页纯数字、奇右偶左、落于下页边距带内（不与正文栏相交）。"""
    pdf = tmp_path / "blank.pdf"
    doc = pymupdf.open()
    for _ in range(3):
        doc.new_page(width=595.276, height=841.89)  # A4
    doc.save(str(pdf))
    doc.close()

    br.add_page_numbers(pdf, margin_cm=br.PAGE_MARGIN_CM)
    doc = pymupdf.open(str(pdf))
    try:
        assert doc.page_count == 3
        m_pt = br.PAGE_MARGIN_CM * 72.0 / 2.54
        for i, page in enumerate(doc):
            n = i + 1
            words = [w for w in page.get_text("words") if w[4].strip()]
            assert len(words) == 1 and words[0][4] == str(n), f"第 {n} 页页码缺失/不纯"
            x0, y0, x1, y1 = words[0][:4]
            assert (
                y0 >= page.rect.height - m_pt - 0.5 and y1 <= page.rect.height + 0.5
            ), f"第 {n} 页页码须落在下页边距带内：{(x0, y0, x1, y1)}"
            if n % 2:
                assert x1 > page.rect.width / 2, f"奇数页 {n} 页码须偏右下"
            else:
                assert x0 < page.rect.width / 2, f"偶数页 {n} 页码须偏左下"
    finally:
        doc.close()


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
