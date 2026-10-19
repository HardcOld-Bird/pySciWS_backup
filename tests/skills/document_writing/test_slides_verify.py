"""slides_visual 视觉验收的三态回归（机制性视觉检查闸的生产/验收链）。

用户硬要求：**不让组员在不经视觉检查的情况下提交 ppt 产物**。`compose slides verify`
是那条机制的验收端——本文件用**临时生成的迷你 pptx**构造三态：

- 齐备（渲染存在 + 页数一致 + 非空白）→ PASS；
- 缺渲染（渲染目录缺席/为空）→ FAIL；
- 页数不符（渲染张数 ≠ 幻灯片页数）→ FAIL；
- 空白页（某页渲染成纯色）→ FAIL。

渲染图用 Pillow 直接构造（不依赖 LibreOffice），故三态用例在任何环境都确定性可跑；
`slides render` 真渲染链另有 :func:`test_render_pptx_pages_real_chain`（无 LibreOffice 时跳过）。
"""

from __future__ import annotations

import pytest

pytest.importorskip("pptx", reason="需要 python-pptx")
Image = pytest.importorskip("PIL.Image", reason="需要 Pillow")

from pysci.skills.document_writing.tools import (  # noqa: E402
    compose,
    pptx_io,
    slides_visual,
)


def _mini_pptx(path, n_slides: int = 2):
    """生成 n_slides 页的迷你 pptx（无标题页，页数 = n_slides）。"""
    specs = [
        pptx_io.SlideSpec(title=f"Slide {i}", bullets=[f"point {i}"])
        for i in range(1, n_slides + 1)
    ]
    pptx_io.build_pptx(specs, path)
    assert len(pptx_io.read_pptx(path)) == n_slides
    return path


def _write_png(path, *, blank: bool):
    """写一张 200x120 的 PNG：blank=False 加一条黑墨迹（非空白），True 则纯白。"""
    im = Image.new("RGB", (200, 120), "white")
    if not blank:
        for x in range(20, 180):
            im.putpixel((x, 60), (0, 0, 0))
    im.save(path)


def _seed_render(pptx_path, *, n_pngs: int, blank_idx: tuple[int, ...] = ()):
    """按约定目录播种 n_pngs 张渲染图（blank_idx 里的页码写成纯白）。"""
    d = slides_visual.default_render_dir(pptx_path)
    d.mkdir(parents=True, exist_ok=True)
    for i in range(1, n_pngs + 1):
        _write_png(d / f"page-{i:03d}.png", blank=i in blank_idx)
    return d


def test_齐备_passes(tmp_path):
    """渲染存在 + 页数一致 + 非空白 → ok，problems 为空。"""
    deck = _mini_pptx(tmp_path / "deck.pptx", 2)
    _seed_render(deck, n_pngs=2)
    rep = slides_visual.verify_pptx_render(deck)
    assert rep.ok, rep.problems
    assert rep.slide_count == 2 and rep.png_count == 2
    assert rep.problems == [] and rep.blank_pages == []


def test_缺渲染_fails(tmp_path):
    """渲染目录缺席（组员没渲染）→ FAIL，理由含「缺渲染」。"""
    deck = _mini_pptx(tmp_path / "deck.pptx", 2)
    rep = slides_visual.verify_pptx_render(deck)
    assert not rep.ok
    assert any("缺渲染" in p for p in rep.problems), rep.problems
    assert rep.png_count == 0


def test_空渲染目录也判缺渲染(tmp_path):
    """渲染目录存在但为空 → 同样判缺渲染（不是「页数不符」）。"""
    deck = _mini_pptx(tmp_path / "deck.pptx", 2)
    slides_visual.default_render_dir(deck).mkdir(parents=True, exist_ok=True)
    rep = slides_visual.verify_pptx_render(deck)
    assert not rep.ok
    assert any("缺渲染" in p for p in rep.problems), rep.problems


def test_页数不符_fails(tmp_path):
    """渲染张数 ≠ 幻灯片页数 → FAIL，理由含「页数不符」。"""
    deck = _mini_pptx(tmp_path / "deck.pptx", 2)
    _seed_render(deck, n_pngs=3)
    rep = slides_visual.verify_pptx_render(deck)
    assert not rep.ok
    assert any("页数不符" in p for p in rep.problems), rep.problems
    assert rep.slide_count == 2 and rep.png_count == 3


def test_空白页_fails(tmp_path):
    """某页渲染成纯色（渲染失败/空页）→ FAIL，理由含「空白页」。"""
    deck = _mini_pptx(tmp_path / "deck.pptx", 2)
    _seed_render(deck, n_pngs=2, blank_idx=(2,))
    rep = slides_visual.verify_pptx_render(deck)
    assert not rep.ok
    assert any("空白页" in p for p in rep.problems), rep.problems
    assert rep.blank_pages == ["page-002.png"]


def test_page_ink_ratio_separates_blank_from_inked(tmp_path):
    """墨迹占比：纯白页 ≈ 0，带墨迹页远高于空白阈值。"""
    blank = tmp_path / "b.png"
    inked = tmp_path / "i.png"
    _write_png(blank, blank=True)
    _write_png(inked, blank=False)
    assert slides_visual.page_ink_ratio(blank) == pytest.approx(0.0, abs=1e-6)
    assert slides_visual.page_ink_ratio(inked) > slides_visual.DEFAULT_MIN_INK_RATIO
    assert slides_visual.page_is_blank(blank) and not slides_visual.page_is_blank(inked)


# ---------------------------------------------------------------------------
# CLI 退出码（派发侧 check 只认退出码 0/非 0）
# ---------------------------------------------------------------------------
def test_cli_verify_exit_zero_when_complete(tmp_path):
    deck = _mini_pptx(tmp_path / "deck.pptx", 1)
    _seed_render(deck, n_pngs=1)
    assert compose.main(["slides", "verify", str(deck)]) == 0


def test_cli_verify_nonzero_when_render_missing(tmp_path):
    deck = _mini_pptx(tmp_path / "deck.pptx", 1)
    assert compose.main(["slides", "verify", str(deck)]) == 1


def test_cli_verify_render_dir_override(tmp_path):
    """`--render-dir` 指向别处时按它核验（约定目录缺席也不算缺渲染）。"""
    deck = _mini_pptx(tmp_path / "deck.pptx", 1)
    alt = tmp_path / "elsewhere"
    alt.mkdir()
    _write_png(alt / "page-001.png", blank=False)
    rc = compose.main(["slides", "verify", str(deck), "--render-dir", str(alt)])
    assert rc == 0


# ---------------------------------------------------------------------------
# 真渲染链（pptx → pdf → PNG）：需 LibreOffice，缺席即跳过
# ---------------------------------------------------------------------------
def _libreoffice_available() -> bool:
    from pysci.skills.document_writing.tools.config import settings

    return bool(settings.find_libreoffice())


@pytest.mark.skipif(
    not _libreoffice_available(), reason="未检测到 LibreOffice（渲染链依赖）"
)
def test_render_pptx_pages_real_chain(tmp_path):
    """真链路：迷你 pptx → 逐页 PNG（页数 = 幻灯片数），且渲染图非空白。"""
    deck = _mini_pptx(tmp_path / "deck.pptx", 2)
    res = slides_visual.render_pptx_pages(deck)
    assert res.slide_count == 2
    assert len(res.pngs) == 2
    assert all(p.exists() for p in res.pngs)
    assert not any(slides_visual.page_is_blank(p) for p in res.pngs)
    assert slides_visual.verify_pptx_render(deck).ok


# ---------------------------------------------------------------------------
# 渲染目录解析（backlog 20261016-042501-devops）：用户指定目录**单份**即过关
# ---------------------------------------------------------------------------
def test_render_in_a_custom_sibling_dir_passes(tmp_path):
    """渲染件只落用户指定的 `render/` 一份（无 `<stem>_render/`）→ PASS（不再要双份）。"""
    deck = _mini_pptx(tmp_path / "deck.pptx", 2)
    custom = tmp_path / "render"
    custom.mkdir()
    for i in (1, 2):
        _write_png(custom / f"page-{i:03d}.png", blank=False)
    rep = slides_visual.verify_pptx_render(deck)
    assert rep.ok, rep.problems
    assert rep.render_dir == custom and rep.render_dir_source == "discovered"


def test_convention_dir_still_wins_when_present(tmp_path):
    """默认行为兼容：约定目录非空时优先用它（不去猜同级别的目录）。"""
    deck = _mini_pptx(tmp_path / "deck.pptx", 1)
    _seed_render(deck, n_pngs=1)
    other = tmp_path / "render"
    other.mkdir()
    _write_png(other / "page-001.png", blank=False)
    rep = slides_visual.verify_pptx_render(deck)
    assert rep.ok and rep.render_dir_source == "default", rep.problems


def test_discovery_prefers_the_dir_matching_slide_count(tmp_path):
    """同级多目录但恰有一个页数吻合 → 用它（歧义只在无人吻合时）。"""
    deck = _mini_pptx(tmp_path / "deck.pptx", 2)
    for name, n in (("render_ok", 2), ("render_x", 5)):
        d = tmp_path / name
        d.mkdir()
        for i in range(1, n + 1):
            _write_png(d / f"page-{i:03d}.png", blank=False)
    rep = slides_visual.verify_pptx_render(deck)
    assert rep.ok and rep.render_dir_source == "discovered", rep.problems


def test_ambiguous_sibling_dirs_fail_with_candidates(tmp_path):
    """同级多个含 PNG 目录且无一页数吻合 → 报歧义并列候选（不猜）。"""
    deck = _mini_pptx(tmp_path / "deck.pptx", 2)
    for name, n in (("render_a", 3), ("render_b", 5)):
        d = tmp_path / name
        d.mkdir()
        for i in range(1, n + 1):
            _write_png(d / f"page-{i:03d}.png", blank=False)
    rep = slides_visual.verify_pptx_render(deck)
    assert not rep.ok and any("歧义" in p for p in rep.problems), rep.problems
    assert rep.render_dir_source == "ambiguous"


def test_single_mismatched_sibling_dir_is_skipped(tmp_path):
    """唯一旁路目录张数不符（素材目录 `插图/` 4 张 ≠ 15 页）→ 跳过，不误报页数不符。

    实案 20261019-192536-reviewer：默认 verify 把 `插图/`（4 PNG）当渲染件，
    报「页数不符：渲染 4 页 ≠ 幻灯片 15 页」——缺渲染被伪装成版面缺陷。
    """
    deck = _mini_pptx(tmp_path / "deck.pptx", 2)
    ext = tmp_path / "插图"
    ext.mkdir()
    for i in range(1, 5):
        _write_png(ext / f"page-{i:03d}.png", blank=False)
    rep = slides_visual.verify_pptx_render(deck)
    assert not rep.ok
    assert rep.render_dir_source == "mismatch", rep.render_dir_source
    assert any("已跳过" in p and "插图" in p for p in rep.problems), rep.problems
    assert not any("页数不符" in p for p in rep.problems), rep.problems
    assert rep.png_count == 0


def test_cli_verify_single_mismatched_bypass_is_rc1(tmp_path, capsys):
    deck = _mini_pptx(tmp_path / "deck.pptx", 2)
    ext = tmp_path / "插图"
    ext.mkdir()
    for i in range(1, 4):
        _write_png(ext / f"page-{i:03d}.png", blank=False)
    assert compose.main(["slides", "verify", str(deck)]) == 1
    out = capsys.readouterr().out
    assert "已跳过" in out and "页数不符" not in out
