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
# 打印版式（2026-11-03 用户裁决四项 + 二轮三项）：双栏 / 窄页边距 / 页码 / 标注修复
# + 单栏强制（禁跨栏、超宽等比缩放）/ 栏间分割线
# ---------------------------------------------------------------------------
def test_print_css_two_columns_and_layout_guards() -> None:
    """样式真源：双栏、栏间分割线、body margin 清零、图片/表格栏宽、图组不断裂、长串可断行；
    **无**跨栏（column-span/.wide）——2026-11-03 二轮：超栏宽元素一律等比缩进栏。"""
    css = (br.TEMPLATE_DIR / "bilingual.css").read_text(encoding="utf-8")
    assert "column-count: 2" in css, "全文默认双栏"
    assert "column-gap: 0.7cm" in css
    assert "column-rule" in css, "栏间分割线（全页生效）"
    assert re.search(r"body\s*\{[^}]*margin: 0;", css), (
        "body margin 须清零（浏览器默认 8px 会蚕食页边距）"
    )
    assert "max-width: 100%" in css, "图片/表格适应栏宽"
    assert ".figure" in css and "break-inside: avoid" in css, (
        "图＋图注首对不被栏/页断裂"
    )
    assert ".caption-cont" in css, "图注余对容器（自由流布）"
    assert not re.search(r"\.caption-cont\s*\{[^}]*break-inside", css), (
        "图注余对不得钉死不断裂（长图注整组原子会跳栏首留洞）"
    )
    assert "overflow-wrap: anywhere" in css, "长串（DOI/URL）栏内可断行"
    assert "column-span" not in css and ".wide" not in css, "禁跨栏（CSS 侧）"
    assert "column-span" not in br._INIT_JS and ".wide" not in br._INIT_JS, (
        "禁跨栏（JS 侧）——量宽只做等比缩放接线"
    )
    assert "min-content" in br._INIT_JS and "zoom" in br._INIT_JS, (
        "超宽元素须按 min-content 量宽后等比缩放（只缩不放）"
    )
    assert "__COL_W_PX__" in br._INIT_JS, "判据用打印栏宽常量（build_html 注入）"


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


# ---------------------------------------------------------------------------
# 单栏强制（2026-11-03 二轮）：超宽公式/表格等比缩进栏（只缩不放），禁跨栏
# ---------------------------------------------------------------------------
_WIDE_COL_CELLS = "".join(f"<td>c{i}</td>" for i in range(30))

#: 宽元素行为夹具：小公式/窄表（不缩）+ 宽公式/宽表（缩放进栏）。
WIDE_BILINGUAL = """# 宽元素夹具
*Wide fixture*

---
小公式 $a + b$。
*Small formula.*

---
$$
E = mc^2
$$

---
宽公式。
*Wide formula.*

---
$$
a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 + a_8 + a_9 + a_{10} + a_{11} + a_{12} + a_{13} + a_{14}
$$

---
窄表格。
*Narrow table.*

---
<table>
<tr><td>1</td><td>2</td></tr>
</table>

---
宽表格。
*Wide table.*

---
<table>
<tr>__WIDE_CELLS__</tr>
</table>

---
尾段。
*Tail.*
""".replace("__WIDE_CELLS__", _WIDE_COL_CELLS)


def _col_w_px(html: str) -> float:
    """从构建产物里取打印栏宽常量（与 JS 判据同一真源）。"""
    m = re.search(r"var colW = ([\d.]+);", html)
    assert m, "build_html 未注入打印栏宽常量"
    return float(m.group(1))


def test_wide_formula_and_table_shrink_only_in_dom(
    tmp_path: Path, browser_available: bool
) -> None:
    """HTML/JS 层：超宽公式/表格等比缩放至栏宽（zoom<1），不超宽者原尺寸（只缩不放）。"""
    html = br.build_html(WIDE_BILINGUAL, title="wide")
    (tmp_path / "wide.html").write_text(html, encoding="utf-8")
    col_w = _col_w_px(html)

    from playwright.sync_api import sync_playwright

    from pysci.skills.literature_research.tools.browser_fetch import _launch

    with sync_playwright() as p:
        b, _ch = _launch(p, "auto", True)
        try:
            pg = b.new_page()
            pg.goto(
                (tmp_path / "wide.html").resolve().as_uri(), wait_until="networkidle"
            )
            pg.wait_for_function(
                "document.fonts ? document.fonts.status === 'loaded' : true"
            )
            got = pg.evaluate(
                """() => Array.from(
                    document.querySelectorAll(".katex-display, table")
                ).map(el => {
                    const zoom = parseFloat(el.style.zoom || "1");
                    const w0 = el.style.width, m0 = el.style.maxWidth,
                          s0 = el.style.whiteSpace, z0 = el.style.zoom;
                    el.style.zoom = "";
                    el.style.width = "min-content";
                    el.style.maxWidth = "none";
                    if (el.classList.contains("katex-display")) {
                        el.style.whiteSpace = "nowrap";
                    }
                    const natural = el.getBoundingClientRect().width;
                    el.style.width = w0; el.style.maxWidth = m0;
                    el.style.whiteSpace = s0; el.style.zoom = z0;
                    return {tag: el.tagName, zoom: zoom, natural: natural};
                })"""
            )
        finally:
            b.close()

    assert [g["tag"] for g in got] == ["SPAN", "SPAN", "TABLE", "TABLE"], got
    small, wide, narrow, big = got
    assert small["zoom"] == 1.0, f"小公式不得缩放（只缩不放）：{small}"
    assert narrow["zoom"] == 1.0, f"窄表格不得缩放（只缩不放）：{narrow}"
    for g in (wide, big):
        assert g["natural"] > col_w, f"夹具前提：该元素确实超栏宽：{g}"
        assert 0.0 < g["zoom"] < 1.0, f"超宽元素须等比缩放：{g}"
        scaled = g["natural"] * g["zoom"]
        assert scaled <= col_w + 2, f"缩放后须 ≤ 打印栏宽 {col_w}：{g} → {scaled}"


def test_print_smoke_no_cross_column_ink_and_rule_visible(
    tmp_path: Path, browser_available: bool
) -> None:
    """打印层冒烟（pymupdf 读数）：无任何墨迹跨栏缝、缩放公式/表格印出宽 ≤ 栏宽、
    栏间分割线可见。"""
    (tmp_path / "bilingual.md").write_text(WIDE_BILINGUAL, encoding="utf-8")
    pdf = br.render_pdf(tmp_path / "bilingual.md", tmp_path / "bilingual.pdf")
    html = (tmp_path / "bilingual.html").read_text(encoding="utf-8")
    col_w_pt = _col_w_px(html) * 0.75
    m_pt = br.PAGE_MARGIN_CM * 72.0 / 2.54
    gap = br.COLUMN_GAP_CM * 72.0 / 2.54
    gap_l, gap_r = m_pt + col_w_pt, m_pt + col_w_pt + gap
    gap_c = (gap_l + gap_r) / 2.0

    doc = pymupdf.open(str(pdf))
    try:
        assert doc.page_count > 0
        for pn, page in enumerate(doc, 1):
            for blk in page.get_text("dict")["blocks"]:
                for line in blk.get("lines", []):
                    b = line["bbox"]
                    assert not (b[0] < gap_l - 1 and b[2] > gap_r + 1), (
                        f"第 {pn} 页有跨栏缝墨迹（超栏宽溢出）：{b}"
                    )
            for img in page.get_images(full=True):
                b = page.get_image_bbox(img)
                assert b.x1 - b.x0 <= col_w_pt + 2, f"第 {pn} 页图片超栏宽：{b}"
            rules = [
                d["rect"]
                for d in page.get_drawings()
                if abs(d["rect"].x0 - gap_c) < 3
                and abs(d["rect"].x1 - gap_c) < 3
                and d["rect"].y1 - d["rect"].y0 > 100
            ]
            assert rules, f"第 {pn} 页缺栏间分割线"
    finally:
        doc.close()


# ---------------------------------------------------------------------------
# 图注粒度（2026-11-03 三轮）：图＋图注首对原子（.figure 不断裂），余对
# .caption-cont 自由流布；短图注组（无余对）行为不变
# ---------------------------------------------------------------------------
# 图片行须用**空 alt**（`![](...)`，与 bilingual.md 契约一致）：带 alt 的独图会被
# pandoc implicit_figures 转成 <figure>（不入 <p>），归组 JS 不生效——夹具曾因
# `![x](…)` 静默测空。
CAPTION_GRANULARITY = """# 图注粒度夹具
*Caption granularity fixture*

---
![](images/pic.png)

首对句对。
*First pair.*

余对一句对。
*Second pair.*

余对二句对。
*Third pair.*

---
![](images/pic.png)

唯一句对。
*Only pair.*

---
尾段。
*Tail.*
"""


def test_caption_granularity_first_pair_atomic_rest_flows(
    tmp_path: Path, browser_available: bool
) -> None:
    """多对图注：首对在 .caption（.figure 内、原子），余对在 .caption-cont（.figure
    外、可断）；短图注组无 .caption-cont、行为不变。"""
    (tmp_path / "images").mkdir()
    (tmp_path / "images" / "pic.png").write_bytes(_PNG)
    (tmp_path / "bilingual.md").write_text(CAPTION_GRANULARITY, encoding="utf-8")
    html = br.build_html(CAPTION_GRANULARITY, title="cap")
    (tmp_path / "cap.html").write_text(html, encoding="utf-8")

    from playwright.sync_api import sync_playwright

    from pysci.skills.literature_research.tools.browser_fetch import _launch

    with sync_playwright() as p:
        b, _ch = _launch(p, "auto", True)
        try:
            pg = b.new_page()
            pg.goto(
                (tmp_path / "cap.html").resolve().as_uri(), wait_until="networkidle"
            )
            pg.wait_for_function(
                "document.fonts ? document.fonts.status === 'loaded' : true"
            )
            got = pg.evaluate(
                """() => Array.from(document.querySelectorAll(".figure")).map(f => {
                    const cap = f.querySelector(".caption");
                    const nxt = f.nextElementSibling;
                    return {
                        figKids: Array.from(f.children).map(
                            k => k.className || k.tagName
                        ),
                        capKids: cap ? cap.children.length : 0,
                        contInFig: !!f.querySelector(".caption-cont"),
                        contKids: nxt && nxt.classList.contains("caption-cont")
                            ? nxt.children.length : 0
                    };
                })"""
            )
        finally:
            b.close()

    assert len(got) == 2, got
    multi, short = got
    assert multi["figKids"] == ["P", "caption"], (
        f"图＋首对原子：余对不得并入 .figure：{multi}"
    )
    assert multi["capKids"] == 1, f"首对恰好一对：{multi}"
    assert multi["contKids"] == 2 and not multi["contInFig"], (
        f"余对入 .caption-cont 且在 .figure 外自由流布：{multi}"
    )
    assert short["figKids"] == ["P", "caption"] and short["capKids"] == 1, (
        f"短图注组（无余对）行为不变：{short}"
    )
    assert short["contKids"] == 0 and not short["contInFig"], (
        f"短图注组不生成 .caption-cont：{short}"
    )
