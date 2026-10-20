"""gain_ep 扫描区几何单一真源（用户口头反馈：菱形永远正方形，0.31 @ 近边 0.25）。"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

from pysci.paths import PROJECT_ROOT, research_asset_dir
from pysci.research.gain_ep import scan_geometry as sg
from pysci.research.gain_ep.simulation import field_schema as fs


def _gain_ep_root() -> Path:
    """worktree 不含 gitignored 重数据 → 只读回退到主树（git common dir 的父级）。"""
    root = research_asset_dir("gain_ep")
    if (root / "simulation" / "field_8period_1p5m" / "field_schema.json").exists():
        return root
    common = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "--git-common-dir"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return Path(common).resolve().parent / "data" / "research" / "1_gain_ep"


SCHEMA_15 = _gain_ep_root() / "simulation" / "field_8period_1p5m" / "field_schema.json"


def _edges(vertices: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    pts = sg.ordered_vertices(vertices)
    return [(pts[i], pts[(i + 1) % 4]) for i in range(4)]


def _point_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    t = float(np.dot(p - a, ab) / np.dot(ab, ab))
    t = min(1.0, max(0.0, t))
    return float(np.linalg.norm(p - (a + t * ab)))


def _edge_distances(vertices, center=sg.METASURFACE_CENTER_M) -> list[float]:
    c = np.asarray(center, dtype=float)
    return sorted(_point_segment_distance(c, a, b) for a, b in _edges(vertices))


def test_registry_constants_square():
    assert sg.SCAN_SIDE_M == 0.31
    assert sg.SCAN_RADIAL_M == 0.31
    assert sg.SCAN_TRANSVERSE_M == 0.31
    assert sg.SCAN_NEAR_EDGE_M == 0.25
    assert sg.SCAN_SHAPE == "square"
    assert sg.DIAMOND_IS_SQUARE is True
    assert sg.SCAN_RADIAL_DEG == (135.0, 45.0)
    assert sg.SCAN_ALIGNED_RADIAL is True


def test_region_vertices_square_near_far_transverse():
    for deg in sg.SCAN_RADIAL_DEG:
        verts = sg.scan_region_vertices(deg)
        sides = [float(np.linalg.norm(b - a)) for a, b in _edges(verts)]
        assert np.allclose(sides, 0.31, atol=1e-9), "四边必须等长 0.31 m"
        near, far = _edge_distances(verts)[0], _edge_distances(verts)[-1]
        assert np.isclose(near, 0.25, atol=1e-9)
        assert np.isclose(far, 0.56, atol=1e-9)
        # 对角线 = 0.31*sqrt(2)（正方形）
        diag = float(np.linalg.norm(verts[0] - verts[2]))
        assert np.isclose(diag, 0.31 * np.sqrt(2), atol=1e-9)
        # 横向 ±0.155
        u = sg.radial_unit(deg)
        v = np.array([-u[1], u[0]])
        offsets = [(p - np.asarray(sg.METASURFACE_CENTER_M)) @ v for p in verts]
        assert np.isclose(max(offsets), 0.155, atol=1e-9)
        assert np.isclose(min(offsets), -0.155, atol=1e-9)


def test_square_helper_matches_region_vertices():
    rect = sg.scan_square_vertices(135.0)
    assert np.allclose(
        np.array(sorted(np.round(rect, 9).tolist())),
        np.array(sorted(np.round(sg.scan_region_vertices(135.0), 9).tolist())),
    )
    explicit = sg.scan_square_vertices(135.0, side_m=0.31)
    assert np.allclose(
        np.array(sorted(np.round(explicit, 9).tolist())),
        np.array(sorted(np.round(rect, 9).tolist())),
    )


def test_counterexample_60x31_rectangle_is_rejected():
    """twin A 误读的 60×31 矩形钉为反例：非正方形直接拒绝。"""
    with pytest.raises(ValueError, match="菱形永远正方形"):
        sg.scan_region_vertices(135.0, radial_m=0.60, transverse_m=0.31)
    # 旧矩形的远边 0.85 m ≠ 用户口径 0.56 m
    assert not np.isclose(0.25 + 0.60, 0.56)


def test_scan_geometry_block_square():
    block = sg.scan_geometry_block(135.0)
    assert block["shape"] == "square"
    assert block["radial_m"] == 0.31
    assert block["transverse_m"] == 0.31
    assert block["near_edge_m"] == 0.25
    assert block["edges_aligned_radial"] is True
    center = np.asarray(block["center_m"])
    origin = np.asarray(sg.METASURFACE_CENTER_M)
    assert np.isclose(np.linalg.norm(center - origin), 0.405, atol=1e-9)


def test_historical_1p5m_schema_is_old_square_near_30cm():
    """1.5m schema 仍是旧正方形（side 0.31/近边 0.30）——旧近边不是新 0.25。"""
    schema = fs.load_field_schema(SCHEMA_15)
    assert len(schema.diamonds) == 2
    for d in schema.diamonds:
        assert np.isclose(d.side_m, 0.31)
        near = _edge_distances(np.asarray(d.vertices_m, dtype=float))[0]
        assert np.isclose(near, 0.30, atol=1e-9)
        assert near != 0.25
