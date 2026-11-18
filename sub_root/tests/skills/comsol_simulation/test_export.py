"""出图机械断言（mock mph）：尺寸换算 / 属性拼装顺序 / 首中试设 / 空白判定。"""

from __future__ import annotations

from importlib.util import find_spec

import pytest

from pysci.skills.comsol_simulation import export

_HAS_MPL = find_spec("matplotlib") is not None


# ---------------------------------------------------------------------------
# pixels_from_size（尺寸契约 → 像素）
# ---------------------------------------------------------------------------


def test_pixels_from_size_publication_width():
    # 170mm @600dpi ≈ 4016px（期刊 double 栏宽口径）
    w, h = export.pixels_from_size(width_mm=170.0, height_mm=100.0, dpi=600)
    assert w == round(170.0 / 25.4 * 600)
    assert h == round(100.0 / 25.4 * 600)


def test_pixels_from_size_aspect_fills_missing_side():
    w, h = export.pixels_from_size(width_mm=25.4, dpi=100, aspect=2.0)
    assert (w, h) == (100, 50)


def test_pixels_from_size_requires_one_side():
    with pytest.raises(ValueError, match="至少其一"):
        export.pixels_from_size(dpi=600)


def test_pixels_from_size_rejects_bad_dpi():
    with pytest.raises(ValueError, match="dpi"):
        export.pixels_from_size(width_mm=10, height_mm=10, dpi=0)


# ---------------------------------------------------------------------------
# 属性拼装（纯函数；属性名真源=COMSOL 6.4 手册 Image(Export)）
# ---------------------------------------------------------------------------


def test_build_image_ops_property_names():
    ops = export.build_image_ops("pg1", "out.png", 800, 600, dpi=300)
    props = {op.candidates[0]: op.value for op in ops}
    assert props["sourcetype"] == "plotgroup"
    assert props["sourceobject"] == "pg1"
    assert props["size"] == "manualweb"
    assert props["unit"] == "px"
    assert props["width"] == "800"
    assert props["height"] == "600"
    assert props["resolution"] == "300"  # DPI 属性名（mph 官方 demo 真源）
    assert props["lockratio"] == "off"
    assert props["background"] == "color"
    assert props["antialias"] == "on"
    # 输出路径属性：filename（mph）优先、pngfilename（手册）兜底
    file_op = next(op for op in ops if "filename" in op.candidates)
    assert file_op.candidates == ("filename", "pngfilename")
    assert file_op.value.endswith("out.png")
    assert all(op.target == "export" for op in ops)


def test_build_brand_ops_options_before_logo():
    ops = export.build_brand_ops(strip=True)
    names = [op.candidates[0] for op in ops]
    # 顺序契约：先开 options*d（logo*d 仅在其开启时受控），再关 logo*d；
    # 1d/2d/3d 是各自独立属性，全部要设
    assert names == ["options1d", "options2d", "options3d",
                     "logo1d", "logo2d", "logo3d"]
    assert [op.value for op in ops] == ["on", "on", "on", "off", "off", "off"]


def test_build_brand_ops_disabled_is_empty():
    assert export.build_brand_ops(strip=False) == []


def test_build_clean_ops_candidates():
    ops = export.build_clean_ops()
    by_name = {op.candidates[0]: op for op in ops}
    # 导出节点侧：title/legend 全维度关；子特征侧 colorbar；绘图组标题 none
    assert by_name["title2d"].value == "off"
    assert by_name["legend2d"].value == "off"
    assert by_name["colorlegend"].candidates == ("colorlegend", "legend", "showcolorbar")
    assert by_name["titletype"].value == "none"
    assert by_name["titletype"].target == "plot_group"
    assert by_name["colorlegend"].target == "features"


def test_build_color_range_ops():
    ops = export.build_color_range_ops(-1.0, 1.0)
    assert [op.value for op in ops] == ["on", "-1.0", "1.0"]


def test_build_polar_rmax_ops():
    ops = export.build_polar_rmax_ops(2.5)
    assert {op.candidates[0]: op.value for op in ops} == {
        "axislimits": "on", "rmin": "0.0", "rmax": "2.5",
    }


# ---------------------------------------------------------------------------
# build_export_plan 总装
# ---------------------------------------------------------------------------


def test_plan_clean_implies_brand_strip():
    plan = export.build_export_plan("pg1", "o.png", 10, 10, clean=True)
    values = [op.value for op in plan.ops]
    assert "off" in values and "on" in values  # options=on + logo=off
    assert any(op.candidates[0].startswith("logo") for op in plan.ops)


def test_plan_no_clean_keeps_brand_by_default():
    plan = export.build_export_plan("pg1", "o.png", 10, 10, clean=False)
    assert not any(op.candidates[0].startswith("logo") for op in plan.ops)
    assert not any(op.candidates[0].startswith("options") for op in plan.ops)


def test_plan_explicit_strip_brand_overrides():
    plan = export.build_export_plan(
        "pg1", "o.png", 10, 10, clean=True, strip_brand=False
    )
    assert not any(op.candidates[0].startswith("logo") for op in plan.ops)


def test_plan_extent_warns_2d_limits_not_settable():
    plan = export.build_export_plan(
        "pg1", "o.png", 10, 10, extent=(0, 1, 0, 1)
    )
    assert any("轴限" in w for w in plan.warnings)


def test_plan_with_range_and_rmax():
    plan = export.build_export_plan(
        "pg_far", "o.png", 10, 10, color_range=(-1, 1), polar_rmax=3.0
    )
    kinds = [op.candidates[0] for op in plan.ops]
    assert "rangecolormin" in kinds and "rmax" in kinds


# ---------------------------------------------------------------------------
# apply_plan（首中试设；mock 节点）
# ---------------------------------------------------------------------------


class _FakeNode:
    """鸭子类型节点（mph Node 口径）：只认 allowed 属性名，其余 property 抛异常。"""

    def __init__(self, allowed: tuple[str, ...] = ()):
        self.allowed = set(allowed)
        self.calls: list[tuple[str, str]] = []

    def property(self, prop: str, value: str):
        self.calls.append((prop, value))
        if prop not in self.allowed:
            raise ValueError(f"no such property: {prop}")


def test_apply_plan_first_candidate_wins():
    node = _FakeNode(allowed=("legend",))
    plan = export.ExportPlan(
        ops=[export.PropOp("plot_group", ("colorlegend", "legend"), "off")]
    )
    applied = export.apply_plan(plan, export_node=_FakeNode(), plot_group_node=node)
    assert applied == ["plot_group.legend=off"]
    assert node.calls == [("colorlegend", "off"), ("legend", "off")]


def test_apply_plan_all_candidates_fail_recorded_as_missing():
    node = _FakeNode(allowed=())
    plan = export.ExportPlan(ops=[export.PropOp("export", ("nope",), "x")])
    applied = export.apply_plan(plan, export_node=node, plot_group_node=_FakeNode())
    assert applied == []


def test_apply_plan_features_target_uses_feature_nodes():
    feat_ok = _FakeNode(allowed=("colorlegend",))
    feat_bad = _FakeNode(allowed=())
    plan = export.ExportPlan(
        ops=[export.PropOp("features", ("colorlegend", "legend"), "off")]
    )
    applied = export.apply_plan(
        plan, export_node=_FakeNode(), plot_group_node=_FakeNode(),
        feature_nodes=[feat_bad, feat_ok],
    )
    assert applied == ["features.colorlegend=off"]


# ---------------------------------------------------------------------------
# 空白自检（_flat_fraction 纯函数 + blank_check 驱动）
# ---------------------------------------------------------------------------


def test_flat_fraction_uniform_is_one():
    assert export._flat_fraction([(1.0, 1.0, 1.0)] * 10) == 1.0


def test_flat_fraction_mixed():
    samples = [(0.0, 0.0, 0.0)] * 3 + [(1.0, 1.0, 1.0)]
    assert export._flat_fraction(samples) == 0.75


def test_flat_fraction_empty_counts_as_flat():
    assert export._flat_fraction([]) == 1.0


@pytest.mark.skipif(not _HAS_MPL, reason="matplotlib 未安装")
def test_blank_check_flags_flat_png(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    png = tmp_path / "flat.png"
    fig, ax = plt.subplots(figsize=(1, 1))
    ax.axis("off")  # 纯色图：无轴框/刻度/图线
    fig.savefig(png, dpi=32, facecolor="white")
    plt.close(fig)
    res = export.blank_check(png, sample_step=4)
    assert res["blank"] is True
    assert res["flat_frac"] >= export.BLANK_FLAT_FRACTION


@pytest.mark.skipif(not _HAS_MPL, reason="matplotlib 未安装")
def test_blank_check_flags_content_png(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    png = tmp_path / "content.png"
    fig, ax = plt.subplots(figsize=(1, 1))
    ax.plot([0, 1], [0, 1], lw=3)  # 有图线
    fig.savefig(png, dpi=32)
    plt.close(fig)
    res = export.blank_check(png, sample_step=2)
    assert res["blank"] is False
