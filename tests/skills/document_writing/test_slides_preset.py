"""组会 PPT 预设方案的三项机械检查（tb-20261017-212027-4-1016ppt-2 A.a5）。

三检查并入 `compose slides geometry`（预设单源：document-writing ``references/ppt-preset.md``）：

- (a) 内容页标题文本须为「·…·」；首页豁免、无标题的整页复制页自然跳过；
- (b) 标题框宽须随文本适应（程序化估算 vs 实际框宽，±10%）；
- (c) 非整页图须有描边（``<a:ln>`` 缺失即报）；≥ 画布 95% 的整页图豁免。

正例 = 用户原件 `20260918.pptx` 的标题**零误报**（四页标题都被识别，断言非空）；三负例
各由临时构造的迷你 deck 制造。真实 deck 用例在 data/ 缺席时跳过（worktree 内无数据）。
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

pptx = pytest.importorskip("pptx", reason="需要 python-pptx")
Image = pytest.importorskip("PIL.Image", reason="需要 Pillow")

from pptx import Presentation  # noqa: E402
from pptx.dml.color import RGBColor  # noqa: E402
from pptx.util import Inches, Pt  # noqa: E402

from pysci.skills.document_writing.tools import compose, slides_visual  # noqa: E402

_REL_0918 = "data/research/1_gain_ep/汇报PPT/20260918/20260918.pptx"
_REL_1016 = "data/research/1_gain_ep/汇报PPT/20261016/20261016.pptx"


def _project_data_root() -> Path:
    """**主树**项目根（含 data/）；worktree 里的 data/ 是分支快照，可能落后于主树真数据。"""
    from pysci.paths import PROJECT_ROOT

    try:
        out = subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        root = Path(out).parent
        if (root / "data").is_dir():
            return root
    except (OSError, subprocess.CalledProcessError):
        pass
    return PROJECT_ROOT


def _real_deck(rel: str) -> Path:
    p = _project_data_root() / rel
    if not p.exists():
        pytest.skip(f"缺少真实 deck（data/ 不在本树）：{rel}")
    return p


def _png(path: Path) -> Path:
    Image.new("RGB", (60, 40), (30, 90, 200)).save(path)
    return path


def _blank(prs: Presentation) -> None:
    prs.slides.add_slide(prs.slide_layouts[6])


def _titled(
    prs: Presentation,
    text: str,
    *,
    width: float | None = None,
    size: int = 20,
    spacing_pt: float | None = None,
) -> tuple[object, float]:
    """在预设标题区 (0.60,0.44) 加标题文本框；width=None 时按估算宽自适应。"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    box = slide.shapes.add_textbox(
        Inches(0.60), Inches(0.44), Inches(6.01), Inches(0.44)
    )
    box.text_frame.text = text
    for run in box.text_frame.paragraphs[0].runs:
        run.font.size = Pt(size)
        if spacing_pt is not None:
            run._r.get_or_add_rPr().set("spc", str(int(spacing_pt * 100)))
    est = slides_visual.estimate_title_width_in(box)
    box.width = Inches(width if width is not None else est)
    return slide, est


def _pic(
    slide,
    img: Path,
    *,
    left: float = 1.0,
    top: float = 1.5,
    width: float = 2.0,
    height: float = 1.5,
    border: bool = False,
) -> None:
    pic = slide.shapes.add_picture(
        str(img), Inches(left), Inches(top), Inches(width), Inches(height)
    )
    if border:
        pic.line.color.rgb = RGBColor(0, 0, 0)


def _save(prs: Presentation, tmp_path: Path, name: str = "deck.pptx") -> Path:
    out = tmp_path / name
    prs.save(str(out))
    return out


# ---------------------------------------------------------------------------
# 正例：用户原件标题零误报
# ---------------------------------------------------------------------------
def test_user_original_titles_have_zero_false_positives():
    """`20260918.pptx` 四页标题全部被识别，且形态/框宽零误报（非空断言防「没检测」）。"""
    deck = _real_deck(_REL_0918)
    prs = Presentation(str(deck))
    titles = {
        i: slides_visual._title_text(slides_visual._preset_title_shape(s))
        for i, s in enumerate(prs.slides, 1)
    }
    titles = {i: t for i, t in titles.items() if t}
    assert titles == {
        2: "·现状总结·",
        3: "·今天的AI能做什么·",
        4: "·一些准备·",
        5: "·pySci·",
    }, titles

    rep = slides_visual.geometry_report(deck)
    assert all(not s.title_form_issues for s in rep.slides), [
        s.title_form_issues for s in rep.slides if s.title_form_issues
    ]
    assert all(not s.title_width_issues for s in rep.slides), [
        s.title_width_issues for s in rep.slides if s.title_width_issues
    ]


def test_1016_deck_is_clean_after_adaptive_fix():
    """修复后的 `20261016.pptx`（页 3 固定 1.39、页 2/6 自适应）→ 0 违规。

    backlog 20261018-085835-writing ①：W2 不再把「框宽随文本自适应」判成标题不一致。
    """
    deck = _real_deck(_REL_1016)
    rep = slides_visual.geometry_report(deck)
    form = {s.index for s in rep.slides if s.title_form_issues}
    width = {s.index for s in rep.slides if s.title_width_issues}
    border = {s.index for s in rep.slides if s.picture_border_issues}
    assert form == set(), form
    assert width == set(), width
    assert border == set(), border
    assert rep.title_diffs == [], rep.title_diffs
    assert rep.violations == 0, rep.summary()


def test_fixed_and_adaptive_widths_comply(tmp_path):
    """合规形状：页 3 固定 1.39 in（3 pt 字距），页 2/6 按估算自适应 → 0 违规。"""
    prs = Presentation()
    _blank(prs)  # 页 1（首页）
    _titled(prs, "·AI COMSOL·", spacing_pt=1.5)  # 页 2：自适应
    _titled(prs, "·pySci·", width=1.39, spacing_pt=3.0)  # 页 3：用户认可固定宽
    _titled(prs, "·pySci 与 Agent Loop·", spacing_pt=1.5)  # 页 6 形态：自适应
    deck = _save(prs, tmp_path)

    rep = slides_visual.geometry_report(deck)
    assert rep.title_diffs == [], rep.title_diffs
    assert rep.violations == 0, rep.summary()


# ---------------------------------------------------------------------------
# 三负例（合成 deck）+ 正例对照
# ---------------------------------------------------------------------------
def test_clean_preset_deck_has_no_issues(tmp_path):
    """标题「·…·」且框宽自适应、图带描边 → 三项零报（首页豁免亦被覆盖）。"""
    prs = Presentation()
    _blank(prs)  # 页 1（首页）
    slide, est = _titled(prs, "·现状总结·")
    _pic(slide, _png(tmp_path / "a.png"), border=True)
    deck = _save(prs, tmp_path)

    rep = slides_visual.geometry_report(deck)
    box = next(sh for sh in slide.shapes if getattr(sh, "has_text_frame", False))
    assert slides_visual._glyph_width_in(box) == pytest.approx(1.75, abs=0.02)
    assert est == pytest.approx(1.95, abs=0.02), est  # 字形宽 + 左右内边距 0.2
    for s in rep.slides:
        assert not s.title_form_issues and not s.title_width_issues, s
        assert not s.picture_border_issues, s
    assert rep.ok, rep.summary()


def test_estimate_accounts_for_spacing_and_insets(tmp_path):
    """② 回归：估算宽含内边距与字距，「估算宽 +10%」设框在 3 pt 字距下不折行。"""
    from pptx.util import Inches as In

    prs = Presentation()
    _blank(prs)
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    box = slide.shapes.add_textbox(In(0.60), In(0.44), In(6.01), In(0.44))
    box.text_frame.text = "·A Very Long Latin Title For Spacing Check·"
    run = box.text_frame.paragraphs[0].runs[0]
    run.font.size = Pt(20)
    run._r.get_or_add_rPr().set("spc", "300")  # 3 pt 字距

    glyph = slides_visual._glyph_width_in(box)
    text_w = slides_visual.title_text_width_in(box)
    est = slides_visual.estimate_title_width_in(box)
    assert text_w > glyph, "字距未计入文字宽"
    assert est == pytest.approx(text_w + 0.2, abs=0.01), "内边距未计入框宽"
    assert glyph * 1.1 - 0.2 < text_w, "前提：旧口径（只估字形宽）会折行"
    box.width = In(est * 1.1)
    assert est * 1.1 - 0.2 >= text_w, "新口径 +10% 设框仍折行"

    sg = slides_visual.geometry_report(_save(prs, tmp_path)).slides[1]
    assert not sg.title_width_issues, sg.title_width_issues


def test_w2_ignores_adaptive_title_widths(tmp_path):
    """① 回归：同格式、不同文本的标题框宽刻意不同 → W2 不得报「不一致」。"""
    prs = Presentation()
    titles = ("·短·", "·中等长度标题·", "·pySci 与 Agent Loop·")
    for text in titles:
        s = prs.slides.add_slide(prs.slide_layouts[5])  # Title Only
        t = s.shapes.title
        t.text = text
        for r in t.text_frame.paragraphs[0].runs:
            r.font.size = Pt(20)
            r.font.name = "Heiti SC Light"
        t.width = Inches(slides_visual.estimate_title_width_in(t))
        t.height = Inches(0.44)
    deck = _save(prs, tmp_path)

    prs2 = Presentation(str(deck))
    widths = {round(s.shapes.title.width / 914400, 3) for s in prs2.slides}
    assert len(widths) == 3, f"前提：三页框宽确实不同 {widths}"
    rep = slides_visual.geometry_report(deck)
    assert rep.title_diffs == [], rep.title_diffs


def test_missing_dot_form_is_flagged(tmp_path):
    """负例 (a)：内容页标题无前后「·」→ 形态违规。"""
    prs = Presentation()
    _blank(prs)
    _titled(prs, "无装饰标题")
    deck = _save(prs, tmp_path)
    sg = slides_visual.geometry_report(deck).slides[1]
    assert sg.title_form_issues and "·…·" in sg.title_form_issues[0]
    assert not sg.title_width_issues, sg.title_width_issues
    assert sg.violations >= 1


def test_overwide_title_box_is_flagged(tmp_path):
    """负例 (b)：短标题被统一拉成 6.01 in 框宽 → 适配违规（预设的返工事故原型）。"""
    prs = Presentation()
    _blank(prs)
    _titled(prs, "·pySci·", width=6.01)
    deck = _save(prs, tmp_path)
    sg = slides_visual.geometry_report(deck).slides[1]
    assert sg.title_width_issues and "6.01 in" in sg.title_width_issues[0]
    assert not sg.title_form_issues, sg.title_form_issues


def test_picture_without_border_is_flagged(tmp_path):
    """负例 (c)：非整页图无 <a:ln> → 描边违规。"""
    prs = Presentation()
    _blank(prs)
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _pic(slide, _png(tmp_path / "b.png"), border=False)
    deck = _save(prs, tmp_path)
    sg = slides_visual.geometry_report(deck).slides[1]
    assert sg.picture_border_issues and "无描边" in sg.picture_border_issues[0]


def test_full_page_picture_is_border_exempt(tmp_path):
    """整页满幅图（= 画布尺寸）免描边检查。"""
    prs = Presentation()
    _blank(prs)
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _pic(
        slide,
        _png(tmp_path / "bg.png"),
        left=0.0,
        top=0.0,
        width=10.0,
        height=7.5,
        border=False,
    )
    deck = _save(prs, tmp_path)
    assert slides_visual.geometry_report(deck).slides[1].picture_border_issues == []


def test_first_page_and_explicit_exemption(tmp_path):
    """首页默认豁免；`title_exempt_pages` 可把指定页一并豁免。"""
    prs = Presentation()
    _titled(prs, "首页标题允许无点")  # 第 1 页：首页豁免
    _titled(prs, "第二页无点")
    deck = _save(prs, tmp_path)

    rep = slides_visual.geometry_report(deck)
    assert rep.slides[0].title_form_issues == []
    assert rep.slides[1].title_form_issues

    rep2 = slides_visual.geometry_report(deck, title_exempt_pages={1, 2})
    assert all(not s.title_form_issues for s in rep2.slides)


# ---------------------------------------------------------------------------
# CLI / JSON / 退出码
# ---------------------------------------------------------------------------
def test_cli_strict_fails_on_preset_violation(tmp_path):
    prs = Presentation()
    _blank(prs)
    _titled(prs, "无装饰标题")
    deck = _save(prs, tmp_path)
    assert compose.main(["slides", "geometry", str(deck)]) == 0  # 默认只报告
    assert compose.main(["slides", "geometry", str(deck), "--strict"]) == 1


def test_cli_json_exposes_preset_issues(tmp_path, capsys):
    prs = Presentation()
    _blank(prs)
    _titled(prs, "无装饰标题")
    deck = _save(prs, tmp_path)
    assert compose.main(["slides", "geometry", str(deck), "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["slides"][1]["title_form_issues"]
    assert "title_width_issues" in data["slides"][1]
    assert "picture_border_issues" in data["slides"][1]
