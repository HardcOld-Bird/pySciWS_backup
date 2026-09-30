"""postprocess 单测：对微小合成图做 Pillow 调整 / 抽色 / contact sheet。

全部离线、纯 CPU；用 tmp_path 造合成图，不触碰技能数据区。
"""

from __future__ import annotations

import pytest
from PIL import Image

from pysci.skills.ai_drawing.tools import postprocess as pp


def _solid(path, w=40, h=30, color=(200, 60, 60)) -> str:
    Image.new("RGB", (w, h), color).save(path)
    return str(path)


def _two_tone(path, w=40, h=40):
    """上半红、下半蓝——供抽色断言两种主色都被检出。"""
    im = Image.new("RGB", (w, h))
    for y in range(h):
        for x in range(w):
            im.putpixel((x, y), (220, 30, 30) if y < h // 2 else (30, 30, 220))
    im.save(path)
    return str(path)


def test_open_image_and_convert(tmp_path):
    src = _solid(tmp_path / "a.png")
    assert pp.open_image(src).mode == "RGB"
    assert pp.open_image(src, mode="L").mode == "L"


def test_adjust_resize(tmp_path):
    src = _solid(tmp_path / "a.png", 40, 30)
    out = pp.adjust_image(src, tmp_path / "o.png", resize=(20, 15))
    with Image.open(out) as im:
        assert im.size == (20, 15)


def test_adjust_resize_keep_aspect(tmp_path):
    """某一边为 0 → 按原长宽比自动计算。"""
    src = _solid(tmp_path / "a.png", 40, 20)
    out = pp.adjust_image(src, tmp_path / "o.png", resize=(0, 10))
    with Image.open(out) as im:
        assert im.size == (20, 10)


def test_adjust_resize_both_zero_raises(tmp_path):
    src = _solid(tmp_path / "a.png")
    with pytest.raises(ValueError, match="不能同时为 0"):
        pp.adjust_image(src, tmp_path / "o.png", resize=(0, 0))


def test_adjust_crop(tmp_path):
    src = _solid(tmp_path / "a.png", 40, 30)
    out = pp.adjust_image(src, tmp_path / "o.png", crop=(5, 5, 25, 20))
    with Image.open(out) as im:
        assert im.size == (20, 15)


def test_adjust_rotate_expands(tmp_path):
    src = _solid(tmp_path / "a.png", 40, 10)
    out = pp.adjust_image(src, tmp_path / "o.png", rotate=90)
    with Image.open(out) as im:
        # 90° 旋转 + expand → 宽高互换
        assert im.size == (10, 40)


def test_adjust_pad_centers(tmp_path):
    src = _solid(tmp_path / "a.png", 20, 20)
    out = pp.adjust_image(src, tmp_path / "o.png", pad=(60, 60), pad_color=(0, 0, 0))
    with Image.open(out) as im:
        assert im.size == (60, 60)
        # 左上角是加边填充（黑）
        assert im.getpixel((0, 0)) == (0, 0, 0)


def test_adjust_convert_mode_and_jpeg_flattens_alpha(tmp_path):
    src = _solid(tmp_path / "a.png", 20, 20)
    out = pp.adjust_image(src, tmp_path / "o.jpg", mode="RGBA")
    # JPEG 不支持 alpha → 落盘应为 RGB
    with Image.open(out) as im:
        assert im.format == "JPEG"
        assert im.mode == "RGB"


def test_resolve_format_unknown_suffix_raises(tmp_path):
    src = _solid(tmp_path / "a.png")
    with pytest.raises(ValueError, match="无法从后缀"):
        pp.adjust_image(src, tmp_path / "o.xyz")


def test_extract_palette_single_color(tmp_path):
    src = _solid(tmp_path / "a.png", 32, 32, color=(18, 52, 86))
    pal = pp.extract_palette(src, n=3)
    assert pal
    # 主色应接近纯色（median-cut 量化后）
    assert pal[0].startswith("#")
    assert len(pal[0]) == 7


def test_extract_palette_two_tones(tmp_path):
    src = _two_tone(tmp_path / "t.png")
    pal = pp.extract_palette(src, n=2)
    assert len(pal) == 2
    # 两种主色应可区分
    assert pal[0] != pal[1]


def test_extract_palette_clamps_n(tmp_path):
    src = _solid(tmp_path / "a.png")
    # n 夹到 [1,16]
    assert len(pp.extract_palette(src, n=0)) >= 1
    assert len(pp.extract_palette(src, n=99)) <= 16


def test_contact_sheet(tmp_path):
    imgs = [_solid(tmp_path / f"{i}.png", 20, 20) for i in range(4)]
    out = pp.contact_sheet(imgs, tmp_path / "sheet.png", cols=2, thumb=32)
    assert out.exists() and out.stat().st_size > 0
    with Image.open(out) as im:
        assert im.width > 0 and im.height > 0


def test_contact_sheet_empty_raises(tmp_path):
    with pytest.raises(ValueError, match="没有可用的源图"):
        pp.contact_sheet([], tmp_path / "sheet.png")


def test_contact_sheet_skips_missing(tmp_path):
    good = _solid(tmp_path / "g.png")
    out = pp.contact_sheet([good, str(tmp_path / "missing.png")], tmp_path / "s.png")
    assert out.exists()
