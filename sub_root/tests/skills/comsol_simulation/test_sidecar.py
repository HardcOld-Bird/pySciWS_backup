"""sidecar 机械断言（纯数学）：auto-zoom 反演 / data↔pixel 变换 / 契约读写 / 旋转角点。"""

from __future__ import annotations

import json
from importlib.util import find_spec

import pytest

from pysci.skills.comsol_simulation import sidecar as sc

_HAS_MPL = find_spec("matplotlib") is not None


# ---------------------------------------------------------------------------
# comsol_auto_window（auto-zoom 反演）
# ---------------------------------------------------------------------------


def test_auto_window_aspect_matched_keeps_bbox():
    # 几何 2:1，像素框 2:1 → 窗口恰好=几何 bbox
    win = sc.comsol_auto_window((0, 2, 0, 1), (0, 0, 200, 100))
    assert win == pytest.approx((0, 2, 0, 1))


def test_auto_window_height_limited_expands_width():
    # 几何 1:1，像素框 2:1 → 宽受限方向展开：xspan=2*yspan
    win = sc.comsol_auto_window((0, 1, 0, 1), (0, 0, 200, 100))
    assert win == pytest.approx((-0.5, 1.5, 0, 1))


def test_auto_window_width_limited_expands_height():
    # 几何 4:1，像素框 2:1 → 高受限方向展开：yspan=xspan/2
    win = sc.comsol_auto_window((0, 4, 0, 1), (0, 0, 200, 100))
    assert win == pytest.approx((0, 4, -0.5, 1.5))


def test_auto_window_preserves_center():
    win = sc.comsol_auto_window((1, 3, 2, 4), (0, 0, 200, 100))
    xmin, xmax, ymin, ymax = win
    assert ((xmin + xmax) / 2, (ymin + ymax) / 2) == pytest.approx((2, 3))


def test_auto_window_rejects_degenerate():
    with pytest.raises(ValueError, match="退化"):
        sc.comsol_auto_window((0, 0, 0, 1), (0, 0, 100, 100))
    with pytest.raises(ValueError, match="退化"):
        sc.comsol_auto_window((0, 1, 0, 1), (0, 0, 0, 100))


# ---------------------------------------------------------------------------
# Transform（data↔pixel，y 翻转显式）
# ---------------------------------------------------------------------------


def test_transform_maps_extent_corners_to_crop_corners():
    tr = sc.pixel_transform((200, 100), (10, 20, 110, 90), (0, 1, 0, 1))
    assert tr.to_pixel(0, 1) == pytest.approx((10, 20))  # (xmin, ymax) → 顶左
    assert tr.to_pixel(1, 0) == pytest.approx((110, 90))  # (xmax, ymin) → 底右


def test_transform_y_flips():
    tr = sc.pixel_transform((200, 100), (0, 0, 100, 100), (0, 1, 0, 1))
    _, py_top = tr.to_pixel(0, 1)
    _, py_bottom = tr.to_pixel(0, 0)
    assert py_top < py_bottom  # 数据 y 向上 ⇒ 像素 y 向下


def test_transform_roundtrip():
    tr = sc.pixel_transform((200, 100), (10, 20, 110, 90), (-2, 3, 5, 9))
    x, y = 0.123, 7.456
    assert tr.to_data(*tr.to_pixel(x, y)) == pytest.approx((x, y))


def test_transform_dict_roundtrip():
    tr = sc.pixel_transform((200, 100), (0, 0, 100, 100), (0, 1, 0, 1))
    assert sc.Transform.from_dict(tr.to_dict()) == tr


def test_pixel_transform_rejects_degenerate():
    with pytest.raises(ValueError):
        sc.pixel_transform((0, 100), (0, 0, 1, 1), (0, 1, 0, 1))
    with pytest.raises(ValueError):
        sc.pixel_transform((100, 100), (0, 0, 1, 1), (0, 0, 0, 1))


# ---------------------------------------------------------------------------
# sidecar 契约（组装 / 读写 / 校验）
# ---------------------------------------------------------------------------


def test_build_sidecar_defaults_crop_to_full_image():
    data = sc.build_sidecar(
        image_path="out.png", image_size=(200, 100), extent=(0, 1, 0, 1),
    )
    assert data["schema"] == sc.SIDECAR_SCHEMA
    assert data["crop_box_px"] == [0, 0, 200, 100]
    assert data["extent"] == [0, 1, 0, 1]


def test_build_sidecar_inverts_extent_from_geom_bbox():
    data = sc.build_sidecar(
        image_path="out.png", image_size=(200, 100),
        geom_bbox=(0, 2, 0, 1), source={"plot_group": "pg1"},
    )
    assert data["extent"] == pytest.approx([0, 2, 0, 1])
    assert data["source"]["plot_group"] == "pg1"


def test_build_sidecar_requires_extent_or_geom_bbox():
    with pytest.raises(ValueError, match="extent 或 geom_bbox"):
        sc.build_sidecar(image_path="o.png", image_size=(10, 10))


def test_write_and_read_sidecar_roundtrip(tmp_path):
    data = sc.build_sidecar(
        image_path=tmp_path / "out.png", image_size=(200, 100),
        geom_bbox=(0, 2, 0, 1), interior={"blank": False, "flat_frac": 0.1},
    )
    path = sc.write_sidecar(tmp_path / "out.png", data, guard_data=False)
    assert path.name == "out.sidecar.json"
    loaded = sc.read_sidecar(path)
    assert loaded["transform"] == data["transform"]
    assert loaded["interior"]["blank"] is False


def test_read_sidecar_rejects_wrong_schema(tmp_path):
    p = tmp_path / "bad.sidecar.json"
    p.write_text(json.dumps({"schema": "x", "image": {}, "crop_box_px": [],
                             "extent": [], "transform": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        sc.read_sidecar(p)


def test_read_sidecar_rejects_missing_field(tmp_path):
    p = tmp_path / "bad.sidecar.json"
    p.write_text(json.dumps({"schema": sc.SIDECAR_SCHEMA}), encoding="utf-8")
    with pytest.raises(ValueError, match="缺关键字段"):
        sc.read_sidecar(p)


# ---------------------------------------------------------------------------
# 消费端：extent 解析 / 像素旋转矩形（锚点旋转修正）
# ---------------------------------------------------------------------------


def test_raster_extent_reads_back():
    data = sc.build_sidecar(image_path="o.png", image_size=(10, 10), extent=(-1, 1, -2, 2))
    assert sc.raster_extent(data) == pytest.approx((-1, 1, -2, 2))


def test_rect_corners_pixel_unrotated():
    corners = sc.rect_corners_pixel((10, 20), 4, 6)
    flat = [c for pair in corners for c in pair]
    assert flat == pytest.approx([8, 17, 12, 17, 12, 23, 8, 23])


def test_rect_corners_pixel_rotates_about_center():
    corners = sc.rect_corners_pixel((0, 0), 4, 2, angle_deg=90)
    # 旋转 90° 后角点应落在 (±1, ±2)（宽高互换占位）
    xs = sorted(px for px, _ in corners)
    ys = sorted(py for _, py in corners)
    assert xs == pytest.approx([-1, -1, 1, 1])
    assert ys == pytest.approx([-2, -2, 2, 2])


def test_rect_corners_pixel_center_is_mean_of_corners():
    corners = sc.rect_corners_pixel((7, 9), 3, 5, angle_deg=33)
    cx = sum(px for px, _ in corners) / 4
    cy = sum(py for _, py in corners) / 4
    assert (cx, cy) == pytest.approx((7, 9))


def test_rect_corners_pixel_rejects_degenerate():
    with pytest.raises(ValueError):
        sc.rect_corners_pixel((0, 0), 0, 5)


@pytest.mark.skipif(not _HAS_MPL, reason="matplotlib 未安装")
def test_overlay_rect_pixel_lands_in_data_coords():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = sc.build_sidecar(
        image_path="o.png", image_size=(100, 100), extent=(0, 1, 0, 1),
    )
    fig, ax = plt.subplots()
    patch = sc.overlay_rect_pixel(
        ax, data, center_px=(50, 50), width_px=20, height_px=10,
        angle_deg=0, edgecolor="red",
    )
    xy = patch.get_xy()[:4]  # Polygon 闭合点 5 点，取 4 角
    # 像素矩形 (40..60, 45..55) → 数据 (0.4..0.6, 0.45..0.55)（y 翻转）
    flat = [c for pair in xy for c in pair]
    assert flat == pytest.approx([0.4, 0.55, 0.6, 0.55, 0.6, 0.45, 0.4, 0.45])
    plt.close(fig)
