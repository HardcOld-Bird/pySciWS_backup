"""半圆显示区后处理：ROI 掩膜/裁剪、clip path、直径边以下遮盖。"""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from matplotlib.patches import Rectangle  # noqa: E402

from pysci.research.gain_ep.simulation import semicircle as sc  # noqa: E402


def test_mask_upper_half_disc_and_diameter_inclusive():
    roi = sc.SemicircleROI(radius=1.0, center=(2.0, 3.0))
    pts = np.array(
        [
            [2.0, 3.0],  # 圆心
            [3.0, 3.0],  # 直径右端（含）
            [1.0, 3.0],  # 直径左端（含）
            [2.0, 4.0],  # 顶点（含）
            [2.0, 2.9],  # 直径边以下
            [3.01, 3.0],  # 圆外
        ]
    )
    assert roi.mask(pts).tolist() == [True, True, True, True, False, False]
    assert roi.bounds() == (1.0, 3.0, 3.0, 4.0)


def test_clip_points_filters_and_preserves_order():
    roi = sc.SemicircleROI(radius=1.0)
    xy = np.array([[0.0, 0.5], [0.0, -0.5], [0.5, 0.0], [2.0, 2.0]])
    vals = np.array([10.0, 20.0, 30.0, 40.0])
    got_xy, got_vals = sc.clip_points(xy, vals, roi)
    assert got_vals.tolist() == [10.0, 30.0]
    assert got_xy[:, 1].tolist() == [0.5, 0.0]


def test_outline_endpoints_and_arc():
    roi = sc.SemicircleROI(radius=2.0, center=(1.0, -1.0))
    out = roi.outline(181)
    assert out.shape == (181, 2)
    assert np.allclose(out[0], [3.0, -1.0])
    assert np.allclose(out[-1], [-1.0, -1.0])
    assert np.all(out[:, 1] >= -1.0 - 1e-12)
    assert np.allclose(np.hypot(out[:, 0] - 1.0, out[:, 1] + 1.0), 2.0)


def test_invalid_roi_rejected():
    with pytest.raises(ValueError):
        sc.SemicircleROI(radius=0.0)
    with pytest.raises(ValueError):
        sc.SemicircleROI(radius=1.0, center=(0.0,))


def test_clip_artist_sets_semicircle_clip_path():
    import matplotlib.pyplot as plt  # noqa: PLC0415

    fig, ax = plt.subplots()
    try:
        art = ax.imshow(np.zeros((4, 4)), extent=(0, 1, 0, 1))
        roi = sc.SemicircleROI(radius=0.5, center=(0.5, 0.5))
        assert sc.clip_artist(art, ax, roi) is art
        assert art.get_clip_path() is not None
    finally:
        plt.close(fig)


def test_below_diameter_mask_covers_view_below_edge():
    import matplotlib.pyplot as plt  # noqa: PLC0415

    fig, ax = plt.subplots()
    try:
        ax.set_xlim(-1.0, 3.0)
        ax.set_ylim(-2.0, 2.0)
        roi = sc.SemicircleROI(radius=1.0, center=(1.0, 0.0))
        patch = sc.add_below_diameter_mask(ax, roi, zorder=9.0)
        assert isinstance(patch, Rectangle)
        assert patch.get_y() == -2.0
        assert patch.get_height() == 2.0
        assert patch.get_x() <= -1.0 and patch.get_x() + patch.get_width() >= 3.0
        assert patch.get_zorder() == 9.0
        assert patch in ax.patches
    finally:
        plt.close(fig)


def test_below_diameter_mask_none_when_view_above_edge():
    import matplotlib.pyplot as plt  # noqa: PLC0415

    fig, ax = plt.subplots()
    try:
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(0.5, 2.0)
        roi = sc.SemicircleROI(radius=1.0, center=(0.0, 0.0))
        assert sc.add_below_diameter_mask(ax, roi) is None
    finally:
        plt.close(fig)


def test_apply_semicircle_clips_all_artists_and_masks():
    import matplotlib.pyplot as plt  # noqa: PLC0415

    fig, ax = plt.subplots()
    try:
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        a = ax.imshow(np.zeros((4, 4)), extent=(0, 1, 0, 1))
        b = ax.imshow(np.ones((4, 4)), extent=(0, 1, 0, 1))
        roi = sc.SemicircleROI(radius=0.5, center=(0.5, 0.5))
        patch = sc.apply_semicircle(ax, roi, [a, b], mask_zorder=7.0)
        assert a.get_clip_path() is not None and b.get_clip_path() is not None
        assert isinstance(patch, Rectangle) and patch.get_zorder() == 7.0
    finally:
        plt.close(fig)
