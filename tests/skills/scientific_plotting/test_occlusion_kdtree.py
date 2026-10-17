"""``occlusion.min_distance_px`` 的精确最近邻加速（backlog 20261017-120835）。

旧实现是分块全对全 O(N·M)：2880×1440 真图对实测 31.2 s（占整次判据 98.7%）。改用
``scipy.spatial.cKDTree`` 后须满足——**读数完全一致**（<1e-9）、耗时降到 ≤1/10。
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from pysci.skills.scientific_plotting.tools import occlusion as occ


def _brute(a: np.ndarray, b: np.ndarray) -> float:
    """参考实现（原分块全对全），仅用于对拍。"""
    ys, xs = np.nonzero(a)
    ys2, xs2 = np.nonzero(b)
    q = np.column_stack([xs2, ys2]).astype(float)
    p = np.column_stack([xs, ys]).astype(float)
    best = float("inf")
    for i in range(0, len(p), 300):
        d = np.sqrt(((p[i : i + 300, None, :] - q[None, :, :]) ** 2).sum(-1))
        best = min(best, float(d.min()))
    return best


def _mask(
    h: int, w: int, box: tuple[int, int, int, int], color: float = 1
) -> np.ndarray:
    m = np.zeros((h, w), dtype=bool)
    m[box[0] : box[1], box[2] : box[3]] = True
    return m


# ---------------------------------------------------------------------------
# ① 读数完全一致（KD 与暴力同解）
# ---------------------------------------------------------------------------
def test_kdtree_matches_bruteforce_exactly():
    rng = np.random.default_rng(7)
    for _ in range(5):
        a = rng.random((120, 140)) > 0.97  # 稀疏「墨迹」
        b = rng.random((120, 140)) > 0.95  # 稀疏「图线」
        if not a.any() or not b.any():
            continue
        assert occ.min_distance_px(a, b) == pytest.approx(_brute(a, b), abs=1e-9)


def test_returns_none_when_either_side_empty():
    a = _mask(40, 40, (1, 5, 1, 5))
    empty = np.zeros((40, 40), dtype=bool)
    assert occ.min_distance_px(a, empty) is None
    assert occ.min_distance_px(empty, a) is None


# ---------------------------------------------------------------------------
# ② 耗时 ≤1/10（同一对稠密蒙版）
# ---------------------------------------------------------------------------
def test_kdtree_at_least_10x_faster():
    ink = _mask(500, 500, (10, 210, 10, 210))  # 4 万「墨迹」像素
    line = _mask(500, 500, (400, 403, 10, 490))  # 一条「图线」
    t = time.perf_counter()
    ref = _brute(ink, line)
    t_brute = time.perf_counter() - t
    t = time.perf_counter()
    got = occ.min_distance_px(ink, line)
    t_kd = time.perf_counter() - t
    assert got == pytest.approx(ref, abs=1e-9)
    assert t_kd <= t_brute / 10, f"未达 10×：brute={t_brute:.2f}s kd={t_kd:.3f}s"
