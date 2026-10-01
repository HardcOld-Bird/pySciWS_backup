"""imaging 单测：本地位图算子（scikit-image + OpenCV，optional extra）。

依赖策略是"惰性导入 + optional extra"，所以测试分三档：
1. **无守卫**：只用 PIL/numpy 的路径（``split_alpha`` / ``composite`` / I/O 辅助 / 依赖探测），
   任何环境都跑；
2. ``needs_cv2``：OpenCV 侧（泊松融合 / inpaint / mask / 透视）；
3. ``needs_skimage``：scikit-image 侧（形态学 / 连通域测量 / 配准）。

只装了一个库时，另一侧的用例 skip 而非 fail——这正是 optional extra 该有的行为。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pysci.skills.ai_drawing.tools import imaging

_AVAIL = imaging.available()
needs_cv2 = pytest.mark.skipif(_AVAIL.get("opencv") is None, reason=imaging.INSTALL_HINT)
needs_skimage = pytest.mark.skipif(
    _AVAIL.get("scikit-image") is None, reason=imaging.INSTALL_HINT
)


# ---------------------------------------------------------------------------
# 造图辅助
# ---------------------------------------------------------------------------
def _save_rgb(path, arr) -> None:
    Image.fromarray(np.asarray(arr, dtype=np.uint8)).save(path)


def _square_on_black(path, size=(60, 80), box=(10, 5, 30, 25), value=255):
    """黑底上一块白矩形（rows box[0]:box[2], cols box[1]:box[3]）。"""
    h, w = size
    arr = np.zeros((h, w), dtype=np.uint8)
    arr[box[0]:box[2], box[1]:box[3]] = value
    _save_rgb(path, arr)
    return path


# ---------------------------------------------------------------------------
# 依赖探测与纯 numpy/PIL 路径（无守卫）
# ---------------------------------------------------------------------------
def test_available_reports_both_libraries():
    """available() 必须两个键都在（doctor 依赖它展示），缺则为 None 而非 KeyError。"""
    got = imaging.available()
    assert set(got) == {"scikit-image", "opencv"}
    for v in got.values():
        assert v is None or isinstance(v, str)


def test_install_hint_gives_exact_command():
    assert 'uv pip install -e ".[imaging]"' in imaging.INSTALL_HINT


def test_load_rgb_missing_raises_imaging_error(tmp_path):
    """统一用 ImagingError（而非 FileNotFoundError）→ CLI 一处 except 就能兜住。"""
    with pytest.raises(imaging.ImagingError, match="图像不存在"):
        imaging._load_rgb(tmp_path / "nope.png")


def test_load_rgb_modes(tmp_path):
    p = tmp_path / "x.png"
    Image.new("RGBA", (6, 4), (1, 2, 3, 200)).save(p)
    assert imaging._load_rgb(p).shape == (4, 6, 3)
    assert imaging._load_rgb(p, mode="RGBA").shape == (4, 6, 4)
    assert imaging._load_rgb(p, mode="L").shape == (4, 6)


def test_save_jpeg_drops_alpha(tmp_path):
    """JPEG 无 alpha 通道 → 自动转 RGB，而不是让 PIL 抛异常。"""
    arr = np.zeros((8, 8, 4), dtype=np.uint8)
    arr[:, :, 3] = 128
    out = imaging._save(arr, tmp_path / "x.jpg")
    with Image.open(out) as im:
        assert im.mode == "RGB"


def test_save_creates_parent_dirs(tmp_path):
    out = imaging._save(np.zeros((4, 4, 3), dtype=np.uint8), tmp_path / "a" / "b" / "x.png")
    assert out.is_file()


def test_bgr_roundtrip():
    arr = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    assert np.array_equal(imaging._from_bgr(imaging._to_bgr(arr)), arr)
    gray = np.arange(4, dtype=np.uint8).reshape(2, 2)
    assert np.array_equal(imaging._to_bgr(gray), gray)   # 2D 不翻通道


def test_parse_points_string_and_sequence():
    a = imaging._parse_points("1,2;3,4|5,6")
    assert a.shape == (3, 2) and a.dtype == np.float32
    assert a.tolist() == [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    assert imaging._parse_points([(1, 2), (3, 4)]).tolist() == [[1.0, 2.0], [3.0, 4.0]]
    assert imaging._parse_points(None) is None


def test_parse_points_bad_shape_raises():
    with pytest.raises(imaging.ImagingError, match=r"\(N,2\)"):
        imaging._parse_points("1,2,3;4,5,6")


# ---------------------------------------------------------------------------
# 图层：alpha 分离与合成（纯 PIL，无守卫）
# ---------------------------------------------------------------------------
def test_split_alpha(tmp_path):
    src = tmp_path / "layer.png"
    arr = np.zeros((10, 12, 4), dtype=np.uint8)
    arr[:, :, 0] = 200                       # 全红
    arr[:, :6, 3] = 255                      # 左半不透明
    Image.fromarray(arr).save(src)

    res = imaging.split_alpha(src, tmp_path / "out", stem="L")
    assert res["rgb"].name == "L_rgb.png" and res["alpha"].name == "L_alpha.png"
    with Image.open(res["rgb"]) as im:
        assert im.mode == "RGB"
        assert im.getpixel((0, 0)) == (200, 0, 0)
    with Image.open(res["alpha"]) as im:
        assert im.mode == "L"
        assert im.getpixel((0, 0)) == 255      # 左半
        assert im.getpixel((11, 0)) == 0       # 右半


def test_split_alpha_default_stem(tmp_path):
    src = tmp_path / "hero.png"
    Image.new("RGBA", (4, 4), (0, 0, 0, 0)).save(src)
    res = imaging.split_alpha(src, tmp_path)
    assert res["rgb"].name == "hero_rgb.png"


def test_composite_places_layer(tmp_path):
    base = tmp_path / "base.png"
    Image.new("RGBA", (20, 20), (255, 255, 255, 255)).save(base)
    layer = tmp_path / "layer.png"
    Image.new("RGBA", (8, 8), (255, 0, 0, 255)).save(layer)

    out = imaging.composite(base, [layer], tmp_path / "c.png", positions=[(5, 5)])
    with Image.open(out) as im:
        assert im.getpixel((0, 0))[:3] == (255, 255, 255)     # 底图未被覆盖处
        assert im.getpixel((8, 8))[:3] == (255, 0, 0)         # 图层落点


def test_composite_opacity_blends(tmp_path):
    base = tmp_path / "base.png"
    Image.new("RGBA", (10, 10), (255, 255, 255, 255)).save(base)
    layer = tmp_path / "layer.png"
    Image.new("RGBA", (10, 10), (255, 0, 0, 255)).save(layer)

    out = imaging.composite(base, [layer], tmp_path / "c.png", opacities=[0.5])
    with Image.open(out) as im:
        r, g, b = im.convert("RGB").getpixel((5, 5))
    assert r == 255
    assert 100 < g < 160 and 100 < b < 160      # 半透明红叠白 → 粉


def test_composite_stacks_in_order(tmp_path):
    base = tmp_path / "base.png"
    Image.new("RGBA", (10, 10), (0, 0, 0, 255)).save(base)
    lo = tmp_path / "lo.png"
    Image.new("RGBA", (10, 10), (255, 0, 0, 255)).save(lo)
    hi = tmp_path / "hi.png"
    Image.new("RGBA", (10, 10), (0, 0, 255, 255)).save(hi)

    out = imaging.composite(base, [lo, hi], tmp_path / "c.png")
    with Image.open(out) as im:
        assert im.convert("RGB").getpixel((5, 5)) == (0, 0, 255)   # 后者在上


def test_composite_validation(tmp_path):
    base = tmp_path / "base.png"
    Image.new("RGBA", (10, 10), (0, 0, 0, 255)).save(base)
    layer = tmp_path / "l.png"
    Image.new("RGBA", (10, 10), (255, 0, 0, 255)).save(layer)

    with pytest.raises(imaging.ImagingError, match="至少一张图层"):
        imaging.composite(base, [], tmp_path / "o.png")
    with pytest.raises(imaging.ImagingError, match="数量不匹配"):
        imaging.composite(base, [layer], tmp_path / "o.png", opacities=[1.0, 0.5])
    with pytest.raises(imaging.ImagingError, match="0–1"):
        imaging.composite(base, [layer], tmp_path / "o.png", opacities=[1.5])


# ---------------------------------------------------------------------------
# OpenCV 侧
# ---------------------------------------------------------------------------
def _striped_base(path, size=60, period=4, lo=60, hi=180):
    """竖条纹底图（强纹理），用于区分 normal / mixed。"""
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    for c in range(size):
        arr[:, c] = hi if (c // period) % 2 == 0 else lo
    _save_rgb(path, arr)
    return path


def _flat_patch(path, size=20, value=120, rgb=None):
    """均匀色素材（alpha 全开）——梯度处处为零。"""
    arr = np.zeros((size, size, 4), dtype=np.uint8)
    if rgb is None:
        arr[:, :, :3] = value
    else:
        arr[:, :, :3] = rgb
    arr[:, :, 3] = 255
    Image.fromarray(arr).save(path)
    return path


@needs_cv2
def test_fuse_output_follows_base_size(tmp_path):
    """最基本的接线：输出尺寸随 base（src 只贡献梯度），且确实落盘。"""
    src = _flat_patch(tmp_path / "patch.png", value=120)
    base = tmp_path / "base.png"
    Image.new("RGB", (60, 60), (120, 120, 120)).save(base)
    out = imaging.fuse(src, base, tmp_path / "f.png")
    assert out.is_file()
    with Image.open(out) as im:
        assert im.size == (60, 60)


@needs_cv2
def test_fuse_flat_patch_on_flat_base_vanishes(tmp_path):
    """**误用警示**：泊松融合迁移的是梯度，不是绝对颜色。

    均匀色素材（∇src≡0）+ 均匀底图 → 方程无源项，解恒等于边界值，素材的“颜色”
    完全消失。想把一块纯色贴上去应该用 ``composite``，而不是 ``fuse``。
    """
    src = _flat_patch(tmp_path / "green.png", rgb=(0, 255, 0))
    base = tmp_path / "base.png"
    Image.new("RGB", (60, 60), (120, 120, 120)).save(base)

    out = imaging.fuse(src, base, tmp_path / "f.png")
    with Image.open(out) as im:
        got = np.asarray(im.convert("RGB"), dtype=int)[30, 30]
    assert np.abs(got - 120).max() <= 2          # 仍是底图色，没有变绿


@needs_cv2
def test_fuse_transfers_src_gradient(tmp_path):
    """素材自带梯度时，该梯度会被迁移到融合区（这才是 fuse 的正常用途）。"""
    src = tmp_path / "grad.png"
    arr = np.zeros((20, 20, 4), dtype=np.uint8)
    for c in range(20):
        arr[:, c, :] = (40 + c * 9, 40 + c * 9, 40 + c * 9, 255)   # 水平渐变
    Image.fromarray(arr).save(src)
    base = tmp_path / "base.png"
    Image.new("RGB", (60, 60), (120, 120, 120)).save(base)

    out = imaging.fuse(src, base, tmp_path / "f.png")
    with Image.open(out) as im:
        reg = np.asarray(im.convert("L"), dtype=int)[25:36, 25:36]
    assert reg.max() - reg.min() > 10            # 均匀底图上长出了明暗变化


@needs_cv2
def test_fuse_mixed_lets_base_texture_show_through(tmp_path):
    """mixed 取 max(|∇src|,|∇dst|) → 底图强纹理透出；normal 只用 |∇src| → 纹理被抹平。

    实测（条纹 60/180，周期 4px）：mixed 的融合区极差≈条纹本身幅度，normal 只剩
    边界值的调和过渡。两者仅在**底图有纹理**时可区分（均匀底图上完全相同）。
    """
    src = _flat_patch(tmp_path / "flat.png", value=120)
    base = _striped_base(tmp_path / "stripes.png")

    spreads = {}
    for mode in ("normal", "mixed"):
        out = imaging.fuse(src, base, tmp_path / f"s_{mode}.png", mode=mode)
        with Image.open(out) as im:
            reg = np.asarray(im.convert("L"), dtype=int)[25:36, 25:36]
        spreads[mode] = int(reg.max() - reg.min())

    assert spreads["mixed"] > spreads["normal"] + 20


@needs_cv2
def test_fuse_uses_alpha_as_mask(tmp_path):
    """未给 mask 时取 src 的 alpha 通道；alpha 为 0 的区域不参与融合。"""
    src = tmp_path / "patch.png"
    arr = np.zeros((20, 20, 4), dtype=np.uint8)
    arr[:, :, 2] = 255
    arr[5:15, 5:15, 3] = 255                  # 只有中心方块参与融合
    Image.fromarray(arr).save(src)
    base = tmp_path / "base.png"
    Image.new("RGB", (60, 60), (200, 200, 200)).save(base)
    out = imaging.fuse(src, base, tmp_path / "f.png")
    with Image.open(out) as im:
        corner = np.asarray(im.convert("RGB"), dtype=int)[2, 2]
    assert np.allclose(corner, [200, 200, 200], atol=2)   # mask 外原样保留


@needs_cv2
def test_fuse_rejects_oversized_src(tmp_path):
    src = tmp_path / "big.png"
    Image.new("RGBA", (80, 80), (0, 255, 0, 255)).save(src)
    base = tmp_path / "small.png"
    Image.new("RGB", (40, 40), "white").save(base)
    with pytest.raises(imaging.ImagingError, match="大于 base"):
        imaging.fuse(src, base, tmp_path / "o.png")


@needs_cv2
def test_fuse_mask_size_mismatch(tmp_path):
    src = tmp_path / "p.png"
    Image.new("RGBA", (20, 20), (0, 255, 0, 255)).save(src)
    base = tmp_path / "b.png"
    Image.new("RGB", (60, 60), "white").save(base)
    bad = tmp_path / "m.png"
    Image.new("L", (10, 10), 255).save(bad)
    with pytest.raises(imaging.ImagingError, match="mask 尺寸"):
        imaging.fuse(src, base, tmp_path / "o.png", mask=bad)


@needs_cv2
def test_fuse_empty_mask_raises(tmp_path):
    src = tmp_path / "p.png"
    Image.new("RGBA", (20, 20), (0, 255, 0, 255)).save(src)
    base = tmp_path / "b.png"
    Image.new("RGB", (60, 60), "white").save(base)
    black = tmp_path / "m.png"
    Image.new("L", (20, 20), 0).save(black)
    with pytest.raises(imaging.ImagingError, match="融合区域为空"):
        imaging.fuse(src, base, tmp_path / "o.png", mask=black)


@needs_cv2
def test_inpaint_removes_defect(tmp_path):
    src = tmp_path / "s.png"
    arr = np.full((40, 40, 3), 180, dtype=np.uint8)
    arr[18:22, 18:22] = 0                     # 一块黑瑕疵
    _save_rgb(src, arr)
    mask = tmp_path / "m.png"
    marr = np.zeros((40, 40), dtype=np.uint8)
    marr[17:23, 17:23] = 255
    _save_rgb(mask, marr)

    out = imaging.inpaint(src, tmp_path / "o.png", mask=mask)
    with Image.open(out) as im:
        got = np.asarray(im.convert("RGB"), dtype=int)
    assert got[20, 20].min() > 100              # 瑕疵被周围色填充


@needs_cv2
def test_inpaint_ns_method(tmp_path):
    src = tmp_path / "s.png"
    _save_rgb(src, np.full((30, 30, 3), 150, dtype=np.uint8))
    mask = tmp_path / "m.png"
    marr = np.zeros((30, 30), dtype=np.uint8)
    marr[10:20, 10:20] = 255
    _save_rgb(mask, marr)
    assert imaging.inpaint(src, tmp_path / "o.png", mask=mask, method="ns").is_file()


@needs_cv2
def test_inpaint_requires_mask(tmp_path):
    src = tmp_path / "s.png"
    _save_rgb(src, np.full((20, 20, 3), 150, dtype=np.uint8))
    with pytest.raises(imaging.ImagingError, match="必须提供"):
        imaging.inpaint(src, tmp_path / "o.png")


@needs_cv2
def test_inpaint_black_mask_raises(tmp_path):
    src = tmp_path / "s.png"
    _save_rgb(src, np.full((20, 20, 3), 150, dtype=np.uint8))
    mask = tmp_path / "m.png"
    _save_rgb(mask, np.zeros((20, 20), dtype=np.uint8))
    with pytest.raises(imaging.ImagingError, match="全黑"):
        imaging.inpaint(src, tmp_path / "o.png", mask=mask)


@needs_cv2
def test_make_mask_otsu(tmp_path):
    src = _square_on_black(tmp_path / "s.png", size=(40, 40), box=(10, 10, 30, 30))
    out = imaging.make_mask(src, tmp_path / "m.png", method="otsu")
    with Image.open(out) as im:
        got = np.asarray(im.convert("L"))
    assert got.max() == 255 and got.min() == 0
    assert int((got > 127).sum()) == 400        # 20x20 白块


@needs_cv2
def test_make_mask_invert(tmp_path):
    src = _square_on_black(tmp_path / "s.png", size=(40, 40), box=(10, 10, 30, 30))
    out = imaging.make_mask(src, tmp_path / "m.png", method="otsu", invert=True)
    with Image.open(out) as im:
        got = np.asarray(im.convert("L"))
    assert int((got > 127).sum()) == 40 * 40 - 400


@needs_cv2
def test_make_mask_manual_needs_thresh(tmp_path):
    src = _square_on_black(tmp_path / "s.png", size=(20, 20))
    with pytest.raises(imaging.ImagingError, match="--thresh"):
        imaging.make_mask(src, tmp_path / "m.png", method="manual")
    out = imaging.make_mask(src, tmp_path / "m.png", method="manual", thresh=100)
    assert out.is_file()


@needs_cv2
def test_make_mask_canny_and_blur(tmp_path):
    src = _square_on_black(tmp_path / "s.png", size=(40, 40), box=(10, 10, 30, 30))
    out = imaging.make_mask(src, tmp_path / "m.png", method="canny", blur=1)
    with Image.open(out) as im:
        got = np.asarray(im.convert("L"))
    assert got.max() == 255
    assert 0 < int((got > 127).sum()) < 40 * 40   # 只有边缘，不是整块


@needs_cv2
def test_make_mask_grabcut(tmp_path):
    src = _square_on_black(tmp_path / "s.png", size=(60, 60), box=(15, 15, 45, 45))
    out = imaging.make_mask(src, tmp_path / "m.png", method="grabcut",
                            rect=[10, 10, 40, 40], iterations=3)
    with Image.open(out) as im:
        got = np.asarray(im.convert("L"))
    assert int((got > 127).sum()) > 100


@needs_cv2
def test_make_mask_grabcut_needs_rect(tmp_path):
    src = _square_on_black(tmp_path / "s.png", size=(40, 40))
    with pytest.raises(imaging.ImagingError, match="--rect"):
        imaging.make_mask(src, tmp_path / "m.png", method="grabcut")


@needs_cv2
def test_make_mask_grabcut_rect_out_of_bounds(tmp_path):
    src = _square_on_black(tmp_path / "s.png", size=(40, 40))
    with pytest.raises(imaging.ImagingError, match="超出图像范围"):
        imaging.make_mask(src, tmp_path / "m.png", method="grabcut", rect=[30, 30, 20, 20])


@needs_cv2
def test_make_mask_unknown_method(tmp_path):
    src = _square_on_black(tmp_path / "s.png", size=(20, 20))
    with pytest.raises(imaging.ImagingError, match="未知 method"):
        imaging.make_mask(src, tmp_path / "m.png", method="magic")


@needs_cv2
def test_perspective_rectifies(tmp_path):
    """把梯形拉回矩形：输出尺寸与指定 size 一致。"""
    src = tmp_path / "s.png"
    _save_rgb(src, np.full((60, 60, 3), 200, dtype=np.uint8))
    out = imaging.perspective(
        src, tmp_path / "w.png",
        src_pts="5,5;55,10;50,55;8,50", size=(40, 40),
    )
    with Image.open(out) as im:
        assert im.size == (40, 40)


@needs_cv2
def test_perspective_rotate_only(tmp_path):
    """rotate 只做旋转（忽略点列），且不改画布尺寸。"""
    src = tmp_path / "s.png"
    arr = np.zeros((40, 40, 3), dtype=np.uint8)
    arr[:20, :, 0] = 255                       # 上半红
    _save_rgb(src, arr)
    out = imaging.perspective(src, tmp_path / "r.png", src_pts="0,0;1,0;1,1;0,1", rotate=90)
    with Image.open(out) as im:
        assert im.size == (40, 40)
        got = np.asarray(im.convert("RGB"))
    # 水平分界线转 90° 后变成垂直分界线（左半红）
    assert got[20, 5, 0] > 200
    assert got[20, 35, 0] < 60


@needs_cv2
def test_perspective_needs_four_points(tmp_path):
    src = tmp_path / "s.png"
    _save_rgb(src, np.full((20, 20, 3), 100, dtype=np.uint8))
    with pytest.raises(imaging.ImagingError, match="4 个 src 点"):
        imaging.perspective(src, tmp_path / "o.png", src_pts="0,0;1,0;1,1")


# ---------------------------------------------------------------------------
# scikit-image 侧
# ---------------------------------------------------------------------------
@needs_skimage
def test_morphology_open_removes_specks(tmp_path):
    src = tmp_path / "s.png"
    arr = np.zeros((40, 40), dtype=np.uint8)
    arr[10:30, 10:30] = 255                    # 主体
    arr[2, 2] = 255                            # 单像素噪点
    _save_rgb(src, arr)
    out = imaging.morphology(src, tmp_path / "m.png", op="open", radius=2)
    with Image.open(out) as im:
        got = np.asarray(im.convert("L"))
    assert got[2, 2] == 0                      # 噪点被开运算抹掉
    assert got[20, 20] == 255                  # 主体保留


@needs_skimage
def test_morphology_all_ops(tmp_path):
    src = _square_on_black(tmp_path / "s.png", size=(30, 30), box=(10, 10, 20, 20))
    for op in ("open", "close", "erode", "dilate"):
        assert imaging.morphology(src, tmp_path / f"{op}.png", op=op).is_file(), op


@needs_skimage
def test_morphology_unknown_op(tmp_path):
    src = _square_on_black(tmp_path / "s.png", size=(20, 20))
    with pytest.raises(imaging.ImagingError, match="未知 op"):
        imaging.morphology(src, tmp_path / "o.png", op="squash")


@needs_skimage
def test_measure_regions_reports_geometry(tmp_path):
    src = tmp_path / "s.png"
    arr = np.zeros((60, 80), dtype=np.uint8)
    arr[10:30, 5:25] = 255                     # 400 px 方块
    arr[40:45, 50:55] = 255                    # 25 px 小方块
    _save_rgb(src, arr)

    regions = imaging.measure_regions(src, min_area=10)
    assert len(regions) == 2
    big = regions[0]                           # 按面积降序
    assert big["area_px"] == 400
    assert big["bbox_xyxy"] == (5, 10, 25, 30)     # x0,y0,x1,y1（不是 row/col）
    assert big["centroid"] == pytest.approx((14.5, 19.5), abs=0.05)
    assert big["solidity"] == pytest.approx(1.0, abs=0.01)
    assert regions[1]["area_px"] == 25


@needs_skimage
def test_measure_regions_min_area_filter(tmp_path):
    src = tmp_path / "s.png"
    arr = np.zeros((60, 80), dtype=np.uint8)
    arr[10:30, 5:25] = 255
    arr[40:45, 50:55] = 255
    _save_rgb(src, arr)
    assert len(imaging.measure_regions(src, min_area=100)) == 1


@needs_skimage
def test_measure_regions_from_mask(tmp_path):
    src = tmp_path / "s.png"
    _save_rgb(src, np.full((30, 30, 3), 128, dtype=np.uint8))    # 均匀图，阈值化无意义
    mask = tmp_path / "m.png"
    marr = np.zeros((30, 30), dtype=np.uint8)
    marr[5:15, 5:15] = 255
    _save_rgb(mask, marr)
    regions = imaging.measure_regions(src, mask=mask, min_area=10)
    assert len(regions) == 1 and regions[0]["area_px"] == 100


@needs_skimage
def test_measure_regions_uniform_image_is_empty(tmp_path):
    """常量图无法 Otsu → 内部退回中值，应返回空列表而不是抛异常。"""
    src = tmp_path / "s.png"
    _save_rgb(src, np.full((20, 20), 100, dtype=np.uint8))
    assert imaging.measure_regions(src) == []


@needs_skimage
def test_align_recovers_known_shift(tmp_path):
    """已知平移量要被准确估出。

    只断言**幅值**不断言符号：skimage 各版本的 shift 符号约定改过，而本层要保证的是
    “量对”与 ``(y,x) → (x,y)`` 的转置正确（这才是调用方最容易算错的地方）。
    """
    rng = np.random.default_rng(0)
    ref = (rng.random((64, 64)) * 255).astype(np.uint8)
    ref_p = tmp_path / "ref.png"
    _save_rgb(ref_p, ref)
    moved = np.roll(ref, shift=(3, 5), axis=(0, 1))     # dy=3, dx=5
    src_p = tmp_path / "src.png"
    _save_rgb(src_p, moved)

    res = imaging.align(src_p, ref_p, upsample=10)
    assert [abs(v) for v in res["shift_yx"]] == pytest.approx([3.0, 5.0], abs=0.2)
    # (x,y) 形式必须是 (y,x) 的反转，不能把两个分量写反
    assert res["shift_xy"] == pytest.approx(list(reversed(res["shift_yx"])), abs=1e-9)
    assert res["rms_error"] is not None
    assert "out" not in res                            # 未给 dest 就不写图


@needs_skimage
def test_align_writes_corrected_image(tmp_path):
    rng = np.random.default_rng(1)
    ref = (rng.random((48, 48)) * 255).astype(np.uint8)
    ref_p = tmp_path / "ref.png"
    _save_rgb(ref_p, ref)
    src_p = tmp_path / "src.png"
    _save_rgb(src_p, np.roll(ref, shift=(0, 4), axis=(0, 1)))

    out = tmp_path / "aligned.png"
    res = imaging.align(src_p, ref_p, out)
    assert Path(res["out"]) == out
    assert out.is_file()
    with Image.open(out) as im:
        assert im.size == (48, 48)


@needs_skimage
def test_align_size_mismatch_raises(tmp_path):
    a = tmp_path / "a.png"
    _save_rgb(a, np.zeros((20, 20), dtype=np.uint8))
    b = tmp_path / "b.png"
    _save_rgb(b, np.zeros((30, 30), dtype=np.uint8))
    with pytest.raises(imaging.ImagingError, match="尺寸一致"):
        imaging.align(a, b)


# ---------------------------------------------------------------------------
# CLI 接线（依赖缺失时给安装命令，而不是 traceback）
# ---------------------------------------------------------------------------
def test_cli_reports_missing_extra(tmp_path, monkeypatch, capsys):
    from pysci.skills.ai_drawing.tools import imagine

    def boom():
        raise imaging.ImagingError(imaging.INSTALL_HINT)

    monkeypatch.setattr(imaging, "_cv2", boom)
    src = tmp_path / "a.png"
    Image.new("RGB", (10, 10), "white").save(src)
    rc = imagine.main([
        "img", "fuse", str(src), "--base", str(src), "--out", str(tmp_path / "o.png"),
    ])
    assert rc == 1
    err = capsys.readouterr().err
    assert "img fuse 失败" in err
    assert 'uv pip install -e ".[imaging]"' in err


@needs_skimage
def test_cli_img_measure_end_to_end(tmp_path, capsys):
    from pysci.skills.ai_drawing.tools import imagine

    src = tmp_path / "s.png"
    arr = np.zeros((40, 40), dtype=np.uint8)
    arr[10:30, 10:30] = 255
    _save_rgb(src, arr)
    assert imagine.main(["img", "measure", str(src), "--min-area", "10"]) == 0
    out = capsys.readouterr().out
    assert "连通域测量" in out and "area_px" in out and "400" in out


@needs_cv2
def test_cli_img_mask_end_to_end(tmp_path, capsys):
    from pysci.skills.ai_drawing.tools import imagine

    src = _square_on_black(tmp_path / "s.png", size=(40, 40), box=(10, 10, 30, 30))
    out = tmp_path / "m.png"
    assert imagine.main(["img", "mask", str(src), "--out", str(out)]) == 0
    assert out.is_file()
    assert "视觉校验" in capsys.readouterr().out


def test_cli_img_split_end_to_end(tmp_path, capsys):
    """split 只依赖 PIL → 无 optional extra 也能跑（云端图层拆分的下游第一步）。"""
    from pysci.skills.ai_drawing.tools import imagine

    src = tmp_path / "layer.png"
    Image.new("RGBA", (8, 8), (255, 0, 0, 128)).save(src)
    assert imagine.main(["img", "split", str(src), "--dest", str(tmp_path / "o")]) == 0
    assert (tmp_path / "o" / "layer_rgb.png").is_file()
    assert (tmp_path / "o" / "layer_alpha.png").is_file()
