"""raster 栅格面板纯 Python 单测：extent/crop 解析、叠加原语、单面板合成（Agg，不需 COMSOL）。"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib.image import imsave  # noqa: E402

from pysci.skills.scientific_plotting.tools import raster  # noqa: E402

OVERLAYS = [
    {"type": "rotbox", "cx": 0.3, "cy": 0.5, "w": 0.2, "h": 0.08, "angle": 135},
    {
        "type": "panel",
        "cx": 0.1,
        "cy": 0.85,
        "w": 0.12,
        "h": 0.12,
        "angle": 135,
        "text": "Exp.\n(TBD)",
    },
    {"type": "dashed", "x0": 0.2, "y0": 0.6, "x1": 0.12, "y1": 0.8},
    {"type": "text", "x": 0.5, "y": 0.1, "s": "Sim."},
]


def _synth_png(path, *, w=120, h=100):
    rng = np.random.default_rng(1)
    imsave(path, rng.uniform(0.3, 1.0, (h, w, 3)))
    return path


def test_resolve_extent_and_crop_from_sidecar(tmp_path):
    sc = {
        "extent_requested": [0.0, 1.0, 0.0, 0.5],
        "extent_applied": {"xmin": "0", "xmax": "1", "ymin": "0", "ymax": "0.5"},
        "crop_box_px": [10, 20, 110, 90],
    }
    extent, crop = raster.resolve_extent_and_crop(sc)
    assert extent == (0.0, 1.0, 0.0, 0.5)
    assert crop == (10, 20, 110, 90)


def test_resolve_extent_untrusted_when_partially_applied():
    sc = {"extent_requested": [0.0, 1.0, 0.0, 0.5], "extent_applied": {"xmin": "0"}}
    extent, crop = raster.resolve_extent_and_crop(sc)
    assert extent is None and crop is None


def test_resolve_extent_explicit_wins():
    sc = {
        "extent_requested": [0.0, 1.0, 0.0, 0.5],
        "extent_applied": dict.fromkeys(("xmin", "xmax", "ymin", "ymax")),
        "crop_box_px": [1, 2, 3, 4],
    }
    extent, crop = raster.resolve_extent_and_crop(
        sc, extent=(9.0, 9.0, 9.0, 9.0), crop_box=(0, 0, 5, 5)
    )
    assert extent == (9.0, 9.0, 9.0, 9.0) and crop == (0, 0, 5, 5)


def test_apply_overlays_and_unknown_type():
    fig, ax = plt.subplots()
    raster.apply_overlays(ax, OVERLAYS)
    assert len(ax.patches) == 2 and len(ax.texts) == 2
    with pytest.raises(ValueError):
        raster.apply_overlays(ax, [{"type": "nope"}])
    plt.close(fig)


def test_compose_raster_panel(tmp_path):
    img = _synth_png(tmp_path / "r.png")
    spec = tmp_path / "ov.json"
    spec.write_text(json.dumps(OVERLAYS), encoding="utf-8")
    out = raster.compose_raster_panel(
        img,
        tmp_path / "out.png",
        extent=(0.0, 1.0, 0.0, 0.5),
        overlays=raster.load_overlays(spec),
    )
    assert out.exists() and out.stat().st_size > 0


def test_compose_raster_panel_requires_extent(tmp_path):
    img = _synth_png(tmp_path / "r.png")
    with pytest.raises(ValueError):
        raster.compose_raster_panel(img, tmp_path / "out.png")


def test_load_overlays_yaml(tmp_path):
    p = tmp_path / "ov.yaml"
    p.write_text(
        "- type: dashed\n  x0: 0\n  y0: 0\n  x1: 1\n  y1: 1\n", encoding="utf-8"
    )
    assert raster.load_overlays(p) == [
        {"type": "dashed", "x0": 0, "y0": 0, "x1": 1, "y1": 1}
    ]


# ---------------------------------------------------------------------------
# extent_recovered 回退（comsol export image --geom-bbox 写入）
# ---------------------------------------------------------------------------
def test_resolve_extent_recovered_fallback():
    sc = {"extent_recovered": [-0.4, 0.4, -0.2, 0.4], "crop_box_px": [5, 5, 100, 90]}
    extent, crop = raster.resolve_extent_and_crop(sc)
    assert extent == (-0.4, 0.4, -0.2, 0.4) and crop == (5, 5, 100, 90)


# ---------------------------------------------------------------------------
# compose_grid（spec 驱动多面板组装）
# ---------------------------------------------------------------------------
def test_compose_grid_image_blank_and_axes(tmp_path):
    img = _synth_png(tmp_path / "a.png")
    spec = {
        "rows": 2,
        "cols": 2,
        "figsize": (6, 6),
        "panels": [
            {"kind": "image", "image": str(img)},
            {"kind": "blank"},
            {"kind": "axes", "name": "myax"},
            {"kind": "image", "image": str(img)},
        ],
    }
    res = raster.compose_grid(spec)
    assert len(res.axes) == 2 and len(res.axes[0]) == 2
    assert res.axes[0][1] is None  # blank → None
    assert "myax" in res.axes_map  # axes 面板记入 axes_map（供调用方填充）
    res.close()


def test_compose_grid_raster_needs_extent(tmp_path):
    img = _synth_png(tmp_path / "r.png")
    spec = {"rows": 1, "cols": 1, "panels": [{"kind": "raster", "image": str(img)}]}
    with pytest.raises(ValueError):
        raster.compose_grid(spec)


def test_compose_grid_raster_from_sidecar_recovered(tmp_path):
    img = _synth_png(tmp_path / "r.png")
    sc = tmp_path / "r.sidecar.json"
    sc.write_text(
        json.dumps({"extent_recovered": [0, 1, 0, 1], "crop_box_px": [0, 0, 100, 80]}),
        encoding="utf-8",
    )
    spec = {
        "rows": 1,
        "cols": 1,
        "panels": [{"kind": "raster", "image": str(img), "sidecar": str(sc)}],
    }
    res = raster.compose_grid(spec)
    assert res.raster_axes and res.raster_axes[0] is res.axes[0][0]
    res.close()


def test_compose_grid_labels_colorbar_to_file(tmp_path):
    img = _synth_png(tmp_path / "f.png")
    sc = tmp_path / "f.sidecar.json"
    sc.write_text(
        json.dumps(
            {
                "extent_requested": [0, 1, 0, 1],
                "extent_applied": dict.fromkeys(("xmin", "xmax", "ymin", "ymax")),
                "crop_box_px": [0, 0, 100, 80],
            }
        ),
        encoding="utf-8",
    )
    spec = {
        "rows": 1,
        "cols": 2,
        "figsize": (8, 4),
        "panel_labels": True,
        "col_titles": ["field", "far"],
        "colorbar": {"cmap": "bwr", "vmin": -160, "vmax": 160, "label": "|p|"},
        "suptitle": "test",
        "panels": [
            {"kind": "raster", "image": str(img), "sidecar": str(sc)},
            {"kind": "image", "image": str(img)},
        ],
    }
    out = raster.compose_grid_to_file(spec, tmp_path / "grid.png", dpi=80)
    assert out.exists() and out.stat().st_size > 0


def test_load_grid_spec_yaml(tmp_path):
    p = tmp_path / "grid.yaml"
    p.write_text("rows: 1\ncols: 1\npanels:\n  - kind: blank\n", encoding="utf-8")
    spec = raster.load_grid_spec(p)
    assert spec["rows"] == 1 and spec["panels"] == [{"kind": "blank"}]


def test_figures_cli_compose_grid_parser():
    from pysci.skills.scientific_plotting.tools import figures

    args = figures.build_parser().parse_args(
        ["compose-grid", "--spec", "s.yaml", "--out", "o.png", "--dpi", "200"]
    )
    assert args.func is figures.cmd_compose_grid
    assert args.spec.name == "s.yaml" and args.out.name == "o.png" and args.dpi == 200
