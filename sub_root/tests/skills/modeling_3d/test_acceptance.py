"""验收门机械断言：STL 度量（纯 Python 解析）+ 打印/渲染分开判据。

合成 STL（二进制/ASCII 立方体及病态变体）恒跑，无外部依赖。
"""

from __future__ import annotations

import struct

import pytest

from pysci.skills.modeling_3d.acceptance import (
    MeshMetrics,
    judge_print,
    judge_render,
    metrics_from_markers,
    stl_metrics,
)

# 单位立方体 12 三角面片（外向法线、右手缠绕），缩放 s 后即 s³ 体积立方体
_CUBE_TRIS = (
    ((0, 0, 0), (0, 1, 0), (1, 0, 0)),
    ((0, 1, 0), (1, 1, 0), (1, 0, 0)),  # -z
    ((0, 0, 1), (1, 0, 1), (0, 1, 1)),
    ((0, 1, 1), (1, 0, 1), (1, 1, 1)),  # +z
    ((0, 0, 0), (1, 0, 0), (0, 0, 1)),
    ((1, 0, 0), (1, 0, 1), (0, 0, 1)),  # -y
    ((0, 1, 0), (0, 1, 1), (1, 1, 0)),
    ((0, 1, 1), (1, 1, 1), (1, 1, 0)),  # +y
    ((0, 0, 0), (0, 0, 1), (0, 1, 0)),
    ((0, 0, 1), (0, 1, 1), (0, 1, 0)),  # -x
    ((1, 0, 0), (1, 1, 0), (1, 0, 1)),
    ((1, 1, 0), (1, 1, 1), (1, 0, 1)),  # +x
)


def _scaled(tris, s: float):
    return tuple(
        tuple((x * s, y * s, z * s) for x, y, z in tri) for tri in tris
    )


def _write_binary_stl(path, tris) -> None:
    """写二进制 STL（法线取 0，解析端不依赖法线字段）。"""
    buf = bytearray(b"\0" * 80)
    buf += struct.pack("<I", len(tris))
    for tri in tris:
        buf += struct.pack("<12f", 0.0, 0.0, 0.0, *[c for v in tri for c in v])
        buf += struct.pack("<H", 0)
    path.write_bytes(bytes(buf))


def _write_ascii_stl(path, tris) -> None:
    lines = ["solid test"]
    for tri in tris:
        lines.append("  facet normal 0 0 0")
        lines.append("    outer loop")
        for v in tri:
            lines.append(f"      vertex {v[0]} {v[1]} {v[2]}")
        lines.append("    endloop")
        lines.append("  endfacet")
    lines.append("endsolid test")
    path.write_text("\n".join(lines), encoding="utf-8")


# ── stl_metrics：度量正确性 ──────────────────────────────────────────────


def test_binary_cube_metrics(tmp_path):
    stl = tmp_path / "cube.stl"
    _write_binary_stl(stl, _scaled(_CUBE_TRIS, 2.0))
    m = stl_metrics(stl)
    assert m.faces == 12
    assert m.verts == 8
    assert m.watertight is True
    assert m.winding_consistent is True
    assert m.degenerate_faces == 0
    assert m.bbox_mm == pytest.approx((2.0, 2.0, 2.0))
    assert m.volume_mm3 == pytest.approx(8.0, rel=1e-6)


def test_ascii_cube_metrics(tmp_path):
    stl = tmp_path / "cube_ascii.stl"
    _write_ascii_stl(stl, _scaled(_CUBE_TRIS, 3.0))
    m = stl_metrics(stl)
    assert m.faces == 12
    assert m.watertight is True
    assert m.volume_mm3 == pytest.approx(27.0, rel=1e-6)
    assert m.bbox_mm == pytest.approx((3.0, 3.0, 3.0))


def test_open_mesh_not_watertight(tmp_path):
    stl = tmp_path / "open.stl"
    _write_binary_stl(stl, _scaled(_CUBE_TRIS, 1.0)[:-1])  # 去掉一个面
    m = stl_metrics(stl)
    assert m.watertight is False


def test_flipped_winding_detected(tmp_path):
    tris = list(_scaled(_CUBE_TRIS, 1.0))
    v0, v1, v2 = tris[0]
    tris[0] = (v0, v2, v1)  # 翻一个面
    stl = tmp_path / "flipped.stl"
    _write_binary_stl(stl, tris)
    m = stl_metrics(stl)
    assert m.winding_consistent is False


def test_degenerate_face_counted(tmp_path):
    tris = list(_scaled(_CUBE_TRIS, 1.0)) + [((0, 0, 0), (0, 0, 0), (0, 0, 0))]
    stl = tmp_path / "degen.stl"
    _write_binary_stl(stl, tris)
    assert stl_metrics(stl).degenerate_faces == 1


def test_empty_stl_raises(tmp_path):
    stl = tmp_path / "bad.stl"
    stl.write_bytes(b"\0" * 80 + struct.pack("<I", 0))
    with pytest.raises(ValueError):
        stl_metrics(stl)


# ── judge_print：打印件严判据 ───────────────────────────────────────────


def test_print_gate_pass_on_good_cube(tmp_path):
    stl = tmp_path / "cube.stl"
    _write_binary_stl(stl, _scaled(_CUBE_TRIS, 20.0))
    r = judge_print(stl_metrics(stl), expect_bbox_mm=(20.0, 20.0, 20.0))
    assert r.ok and r.criteria == "print"
    assert "expect_bbox_mm" not in "".join(r.warnings)
    r2 = judge_print(
        stl_metrics(stl),
        expect_bbox_mm=(20.0, 20.0, 20.0),
        expect_volume_mm3=8000.0,
    )
    assert r2.ok


def test_print_gate_rejects_open_mesh(tmp_path):
    stl = tmp_path / "open.stl"
    _write_binary_stl(stl, _scaled(_CUBE_TRIS, 1.0)[:-1])
    r = judge_print(stl_metrics(stl), expect_bbox_mm=(1.0, 1.0, 1.0))
    assert not r.ok
    assert any("watertight" in f for f in r.failures)


def test_print_gate_rejects_bbox_mismatch():
    m = MeshMetrics(
        faces=12, bbox_mm=(20.0, 20.0, 21.0), volume_mm3=8000.0,
        watertight=True, winding_consistent=True,
    )
    r = judge_print(m, expect_bbox_mm=(20.0, 20.0, 20.0))
    assert not r.ok
    assert any("尺寸不符" in f for f in r.failures)


def test_print_gate_bbox_sorted_compare():
    m = MeshMetrics(
        faces=12, bbox_mm=(8.0, 40.0, 20.0), volume_mm3=1.0,
        watertight=True, winding_consistent=True,
    )
    assert judge_print(m, expect_bbox_mm=(40.0, 20.0, 8.0)).ok  # 轴向打乱仍过


def test_print_gate_rejects_missing_metrics():
    r = judge_print(MeshMetrics(faces=12))  # 流形/法线未度量
    assert not r.ok
    assert any("度量缺失" in f for f in r.failures)


def test_print_gate_face_budget_and_degenerate():
    m = MeshMetrics(
        faces=500_000, volume_mm3=1.0, watertight=True,
        winding_consistent=True, degenerate_faces=2,
    )
    r = judge_print(m)
    assert not r.ok
    assert any("预算" in f for f in r.failures)
    assert any("退化面片" in f for f in r.failures)


def test_print_gate_warns_without_expect_bbox():
    m = MeshMetrics(
        faces=12, volume_mm3=1.0, watertight=True, winding_consistent=True
    )
    r = judge_print(m)
    assert r.ok  # 只警告不判负
    assert any("expect_bbox_mm" in w for w in r.warnings)


# ── judge_render：渲染件宽判据（与打印分开）─────────────────────────────


def test_render_gate_accepts_open_mesh():
    m = MeshMetrics(faces=5_000, watertight=False, degenerate_faces=1)
    r = judge_render(m)
    assert r.ok and r.criteria == "render"
    assert any("开面" in w for w in r.warnings)  # 非闭合仅提示不可打印
    assert any("退化面片" in w for w in r.warnings)


def test_render_gate_face_budget_is_redline():
    r = judge_render(MeshMetrics(faces=300_000))
    assert not r.ok
    assert any("渲染预算" in f for f in r.failures)


def test_render_gate_bbox_only_warns():
    m = MeshMetrics(faces=100, bbox_mm=(5.0, 5.0, 5.0))
    r = judge_render(m, expect_bbox_mm=(20.0, 20.0, 20.0))
    assert r.ok
    assert any("仅提示" in w for w in r.warnings)


def test_metrics_from_markers():
    m = metrics_from_markers({"faces": "1234", "volume_mm3": "8000.0"})
    assert m.faces == 1234
    assert m.volume_mm3 == pytest.approx(8000.0)
    assert m.watertight is None  # 申报值不做流形断言
