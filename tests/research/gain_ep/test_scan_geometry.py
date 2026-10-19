"""gain_ep 扫描区几何单一真源：常量值 + 与 1.5m field_schema 输出一致（F3）。"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

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


def _near_edge_distance(vertices: np.ndarray, center=sg.METASURFACE_CENTER_M) -> float:
    c = np.asarray(center, dtype=float)
    return min(_point_segment_distance(c, a, b) for a, b in _edges(vertices))


def test_registry_constants():
    assert sg.SCAN_SIDE_M == 0.31
    assert sg.SCAN_NEAR_EDGE_M == 0.30
    assert sg.SCAN_RADIAL_DEG == (135.0, 45.0)
    assert sg.SCAN_EDGES_PARALLEL_RADIAL is True


def test_vertices_side_near_edge_and_orientation():
    for deg in sg.SCAN_RADIAL_DEG:
        verts = sg.scan_square_vertices(deg)
        sides = [float(np.linalg.norm(b - a)) for a, b in _edges(verts)]
        assert np.allclose(sides, sg.SCAN_SIDE_M, atol=1e-9)
        assert np.isclose(_near_edge_distance(verts), sg.SCAN_NEAR_EDGE_M, atol=1e-9)
        u = sg.radial_unit(deg)
        v = np.array([-u[1], u[0]])
        for a, b in _edges(verts):
            e = (b - a) / np.linalg.norm(b - a)
            assert max(abs(float(e @ u)), abs(float(e @ v))) > 1 - 1e-9


def test_registry_matches_1p5m_field_schema():
    """验收：登记值与 field_8period_1p5m/field_schema.json 输出一致。"""
    schema = fs.load_field_schema(SCHEMA_15)
    assert len(schema.diamonds) == 2
    assert sorted(d.radial_deg for d in schema.diamonds) == sorted(sg.SCAN_RADIAL_DEG)
    for d in schema.diamonds:
        assert d.side_m is not None and np.isclose(d.side_m, sg.SCAN_SIDE_M, atol=1e-9)
        verts = np.asarray(d.vertices_m, dtype=float)
        assert np.isclose(_near_edge_distance(verts), sg.SCAN_NEAR_EDGE_M, atol=1e-9)
        want = sg.scan_square_vertices(d.radial_deg)
        got = np.array(sorted(np.round(verts, 9).tolist()))
        exp = np.array(sorted(np.round(want, 9).tolist()))
        assert np.allclose(got, exp, atol=1e-9)


def test_scan_geometry_block_keys():
    block = sg.scan_geometry_block(135.0)
    assert block["radial_deg"] == 135.0
    assert block["side_m"] == sg.SCAN_SIDE_M
    assert block["near_edge_m"] == sg.SCAN_NEAR_EDGE_M
    assert block["edges_parallel_radial"] is True
    center = np.asarray(block["center_m"])
    origin = np.asarray(sg.METASURFACE_CENTER_M)
    assert np.isclose(np.linalg.norm(center - origin), 0.455, atol=1e-9)
    assert len(block["vertices_m"]) == 4
