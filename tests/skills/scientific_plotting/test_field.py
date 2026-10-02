"""field 数据驱动场面板纯 Python 单测（CSV/VTK 读取、tripcolor 渲染、compose_grid 集成）。

覆盖 P1-2：场面板以显式 cmap+norm 渲染，与共享 colorbar 严格一致。
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from pysci.skills.scientific_plotting.tools import field, raster  # noqa: E402


def _synth_field_csv(path, n=6):
    """造 COMSOL 风格场 CSV：% 注释头 + 列名行 + x,y,value 数据（value=x*y）。"""
    xs, ys = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
    lines = ["% comment header to skip", "x [m], y [m], p_t [Pa]"]
    for x, y in zip(xs.ravel(), ys.ravel(), strict=True):
        lines.append(f"{x}, {y}, {x * y}")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def test_read_csv_skips_comments_and_header(tmp_path):
    xy, vals = field.load_field_points(_synth_field_csv(tmp_path / "f.csv"))
    assert xy.shape == (36, 2) and vals.shape == (36,)
    assert np.allclose(vals, xy[:, 0] * xy[:, 1])


def test_load_field_points_bad_suffix(tmp_path):
    p = tmp_path / "f.dat"
    p.write_text("1,2,3", encoding="utf-8")
    with pytest.raises(ValueError):
        field.load_field_points(p)


def test_read_csv_insufficient_cols(tmp_path):
    p = tmp_path / "f.csv"
    p.write_text("1,2\n3,4\n", encoding="utf-8")
    with pytest.raises(ValueError):
        field.load_field_points(p, cols=(0, 1, 5))


def test_add_field_panel_returns_mappable(tmp_path):
    xy, vals = field.load_field_points(_synth_field_csv(tmp_path / "f.csv"))
    fig, ax = plt.subplots()
    m = field.add_field_panel(ax, xy, vals, cmap="bwr", vmin=0, vmax=1)
    assert m is not None
    plt.close(fig)


def test_render_field_panel_to_png(tmp_path):
    out = field.render_field_panel(
        _synth_field_csv(tmp_path / "f.csv"),
        tmp_path / "panel.png",
        cmap="bwr",
        vmin=0,
        vmax=1,
        dpi=80,
    )
    assert out.exists() and out.stat().st_size > 0


def test_read_vtk(tmp_path):
    pv = pytest.importorskip("pyvista")
    mesh = pv.ImageData(dimensions=(5, 5, 1))
    mesh["p"] = np.arange(mesh.n_points, dtype=float)
    f = tmp_path / "f.vtk"
    mesh.save(str(f))
    xy, vals = field.load_field_points(f, scalars="p")
    assert xy.shape[0] == vals.size == 25


def test_compose_grid_field_panel_strict_colorbar(tmp_path):
    spec = {
        "rows": 1,
        "cols": 1,
        "figsize": (5, 4),
        "colorbar": {"cmap": "bwr", "vmin": 0, "vmax": 1, "label": "p"},
        "panels": [
            {"kind": "field", "source": str(_synth_field_csv(tmp_path / "f.csv"))}
        ],
    }
    res = raster.compose_grid(spec)
    assert res.field_axes and res.field_axes[0] is res.axes[0][0]
    res.close()
