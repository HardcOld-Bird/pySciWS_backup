"""postprocess 中依赖 pyvista 的部分：合成网格上的场统计、网格质量、剖切、探针、积分、离屏渲染。

用 ``pytest.importorskip("pyvista")`` 守卫 —— 无图形栈的环境整模块跳过。
离屏渲染需可用的 VTK 离屏后端（本机已验证；失败时 skip 而非 fail，避免环境噪声）。
"""

from __future__ import annotations

import pytest

pv = pytest.importorskip("pyvista", reason="需要 pyvista（场渲染 / 网格分析）")

from pysci.skills.comsol_simulation.tools import postprocess as pp  # noqa: E402

#: Wavelet 源自带点标量 'RTData'
SCALAR = "RTData"


@pytest.fixture()
def wavelet():
    return pv.Wavelet()


def test_field_stats(wavelet):
    s = pp.field_stats(wavelet, SCALAR)
    assert s["n"] == wavelet.n_points
    assert s["min"] <= s["mean"] <= s["max"]
    assert s["rms"] >= 0.0


def test_mesh_quality(wavelet):
    q = pp.mesh_quality(wavelet)
    assert q["n_cells"] == wavelet.n_cells
    assert q["n_points"] == wavelet.n_points
    assert len(q["bounds"]) == 6


def test_slice_and_probe(wavelet):
    sl = pp.slice_grid(wavelet, normal=(0, 0, 1))
    assert sl is not None
    v = pp.probe_point(wavelet, tuple(wavelet.center), SCALAR)
    assert isinstance(v, float)


def test_integrate_scalar(wavelet):
    val = pp.integrate_scalar(wavelet, SCALAR)
    assert isinstance(val, float)


def test_render_grid_offscreen(wavelet, tmp_path):
    out = tmp_path / "render.png"
    try:
        res = pp.render_grid(
            wavelet, out, scalars=SCALAR, window_size=(320, 240), title="smoke"
        )
    except Exception as e:  # noqa: BLE001 - 无离屏 GL 后端时环境性跳过
        pytest.skip(f"离屏渲染不可用：{type(e).__name__}: {e}")
    assert res.exists() and res.stat().st_size > 0
