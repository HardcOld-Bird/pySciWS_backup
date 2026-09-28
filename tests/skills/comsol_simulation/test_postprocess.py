"""postprocess 纯 Python 单测：COMSOL CSV 解析、收敛阶、参考解比对、守恒、传递函数、解析解、场验证器、
渲染栅格自检（轴框检测/空白度量）。

不依赖 pyvista/VTK/COMSOL —— 只测数值与解析逻辑（用假 grid 走 _scalar_array 的 getattr 路径）。
渲染 / 网格质量 / 剖切 / 探针在 test_postprocess_vista.py（需 pyvista）。
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pysci.skills.comsol_simulation.tools import postprocess as pp

# 两种真实 COMSOL 导出形态（取自 Phase 0 冒烟产物 pg10.csv / evl2.csv 的头结构）
DATASET_CSV = """\
% Model,4 pySweep model.mph
% Version,COMSOL 6.4.0.293
% Date,"Sep 26 2026, 11:29"
% Dimension,2
% Nodes,3
% Expressions,0
% Description,
% Length unit,m
% X,Y
0.0,0.0
1.0,2.0
3.0,4.0
"""

TABLE_CSV = """\
% Model,4 pySweep model.mph
% Version,COMSOL 6.4.0.293
% Table,Evaluation 2D
% x,y,Value
0.1,0.2,421.5
0.3,0.4,-461.4
"""


# ---------------------------------------------------------------------------
# COMSOL CSV 解析
# ---------------------------------------------------------------------------
def test_read_dataset_csv(tmp_path):
    p = tmp_path / "pg.csv"
    p.write_text(DATASET_CSV, encoding="utf-8")
    t = pp.read_comsol_csv(p)
    assert t.columns == ["X", "Y"]
    assert t.shape == (3, 2)
    assert t.meta["Nodes"] == "3"
    assert t.meta["Expressions"] == "0"
    assert t.meta["Dimension"] == "2"
    assert "Model" not in t.columns  # 元信息行未被误当列名（回归保护）
    np.testing.assert_allclose(t.column("X"), [0.0, 1.0, 3.0])
    np.testing.assert_allclose(t.column("Y"), [0.0, 2.0, 4.0])
    assert t.has("X") and not t.has("Z")


def test_read_table_csv_lowercase_header(tmp_path):
    p = tmp_path / "evl.csv"
    p.write_text(TABLE_CSV, encoding="utf-8")
    t = pp.read_comsol_csv(p)
    assert t.columns == ["x", "y", "Value"]  # 小写坐标列 + 值列
    assert t.shape == (2, 3)
    np.testing.assert_allclose(t.column("Value"), [421.5, -461.4])
    assert t.meta["Table"] == "Evaluation 2D"


def test_read_csv_missing_column_raises(tmp_path):
    p = tmp_path / "pg.csv"
    p.write_text(DATASET_CSV, encoding="utf-8")
    t = pp.read_comsol_csv(p)
    with pytest.raises(ValueError):
        t.column("nope")


def test_read_csv_no_header_autonames(tmp_path):
    p = tmp_path / "bare.csv"
    p.write_text("1.0,2.0\n3.0,4.0\n", encoding="utf-8")
    t = pp.read_comsol_csv(p)
    assert t.shape == (2, 2)
    assert t.columns == ["col0", "col1"]


# ---------------------------------------------------------------------------
# 收敛阶
# ---------------------------------------------------------------------------
def test_convergence_order_second_order():
    hs = [0.4, 0.2, 0.1, 0.05]
    errs = [3.0 * h**2 for h in hs]
    assert pp.convergence_order(hs, errs) == pytest.approx(2.0, abs=1e-6)


def test_convergence_order_first_order():
    hs = [0.4, 0.2, 0.1]
    errs = [0.5 * h for h in hs]
    assert pp.convergence_order(hs, errs) == pytest.approx(1.0, abs=1e-6)


def test_convergence_order_degenerate():
    assert np.isnan(pp.convergence_order([0.1], [0.01]))  # <2 有效点
    assert np.isnan(pp.convergence_order([0.1, 0.2], [0.0, 0.0]))  # 误差全 0 被 mask


# ---------------------------------------------------------------------------
# 参考解比对
# ---------------------------------------------------------------------------
def test_compare_to_reference_pass():
    a = np.array([1.0, 2.0, 3.0])
    r = pp.compare_to_reference(a, a.copy())
    assert r.passed
    assert "max_rel_dev" in r.detail
    assert r.report().startswith("[PASS]")


def test_compare_to_reference_fail():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.0, 2.0, 3.5])
    r = pp.compare_to_reference(a, b, rtol=1e-3)
    assert not r.passed
    assert r.report().startswith("[FAIL]")


def test_compare_to_reference_shape_mismatch():
    r = pp.compare_to_reference(np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0]))
    assert not r.passed
    assert "形状" in r.detail


def test_compare_to_reference_complex():
    a = np.array([1 + 1j, 2 - 1j])
    assert pp.compare_to_reference(a, a.copy()).passed


# ---------------------------------------------------------------------------
# 守恒
# ---------------------------------------------------------------------------
def test_check_conservation_pass():
    assert pp.check_conservation(100.0, 99.5, rtol=1e-2).passed


def test_check_conservation_fail():
    assert not pp.check_conservation(100.0, 80.0, rtol=1e-2).passed


def test_check_conservation_zero_inflow():
    assert pp.check_conservation(0.0, 0.0).passed
    assert not pp.check_conservation(0.0, 5.0).passed


# ---------------------------------------------------------------------------
# 传递函数 / dB
# ---------------------------------------------------------------------------
def test_transfer_function_and_db():
    out = np.array([2 + 0j, 0 + 2j])
    inp = np.array([1 + 0j, 1 + 0j])
    H = pp.transfer_function(out, inp)
    np.testing.assert_allclose(H, [2 + 0j, 0 + 2j])
    np.testing.assert_allclose(pp.db(H), [20 * np.log10(2), 20 * np.log10(2)])
    assert pp.db(1.0) == pytest.approx(0.0)
    assert pp.db(10.0) == pytest.approx(20.0)


def test_transfer_function_div_zero_no_raise():
    H = pp.transfer_function(np.array([1 + 0j]), np.array([0 + 0j]))
    assert H.shape == (1,)  # errstate 抑制警告，不抛异常


# ---------------------------------------------------------------------------
# 解析参考解：刚壁波导平面波
# ---------------------------------------------------------------------------
def test_rigid_waveguide_unit_magnitude():
    x = np.linspace(0, 1, 11)
    p = pp.rigid_waveguide_pressure(x, freq=1000.0, amplitude=2.0)
    assert np.iscomplexobj(p)
    np.testing.assert_allclose(np.abs(p), 2.0, atol=1e-12)  # 平面波幅值恒定


def test_rigid_waveguide_at_origin():
    p = pp.rigid_waveguide_pressure(np.array([0.0]), freq=3430.0, amplitude=1.0)
    assert abs(p[0] - 1.0) < 1e-12


# ---------------------------------------------------------------------------
# 场统计 / 验证器（假 grid，避开 pyvista）
# ---------------------------------------------------------------------------
def _fake_grid(point_arrays):
    return SimpleNamespace(
        point_data=point_arrays, cell_data={}, array_names=list(point_arrays)
    )


# ---------------------------------------------------------------------------
# 渲染栅格自检（轴框检测 + 空白度量）
# ---------------------------------------------------------------------------
def _synth_render(blank: bool, *, w: int = 260, h: int = 200, box=(40, 30, 219, 179)):
    """合成仿 COMSOL 渲染灰度图：白底 + 轴框 + 右侧 colorbar 竖条；内部空白或彩色。"""
    rng = np.random.default_rng(0)
    g = np.full((h, w), 255.0)
    x0, y0, x1, y1 = box
    g[y0, x0 : x1 + 1] = 0.0
    g[y1, x0 : x1 + 1] = 0.0
    g[y0 : y1 + 1, x0] = 0.0
    g[y0 : y1 + 1, x1] = 0.0
    # colorbar：右侧窄竖条（竖直边暗、水平帽短），干扰帧检测的对照物
    g[40:170, 235] = 0.0
    g[40:170, 245] = 0.0
    g[40, 235:246] = 0.0
    g[169, 235:246] = 0.0
    if blank:
        pass  # 内部纯白
    else:
        g[y0 + 1 : y1, x0 + 1 : x1] = rng.uniform(120, 255, (y1 - y0 - 1, x1 - x0 - 1))
    return g, box


def test_detect_frame_box_array_finds_axis_frame():
    g, box = _synth_render(blank=True)
    assert pp.detect_frame_box_array(g) == box  # colorbar 竖边不被误认为轴框左右界


def test_detect_frame_box_array_no_frame_returns_none():
    g = np.full((100, 100), 255.0)  # 无轴框
    assert pp.detect_frame_box_array(g) is None


def test_interior_blank_metrics_blank_vs_field():
    g_blank, box = _synth_render(blank=True)
    m_blank = pp.interior_blank_metrics(g_blank, box)
    assert m_blank["blank"] is True and m_blank["unique_q"] <= 4
    g_field, box2 = _synth_render(blank=False)
    m_field = pp.interior_blank_metrics(g_field, box2)
    assert m_field["blank"] is False and m_field["std"] > 8.0


def test_read_gray_png_and_detect_frame_box_file(tmp_path):
    from matplotlib.image import imsave

    g, box = _synth_render(blank=True)
    p = tmp_path / "render.png"
    imsave(p, g.astype(np.uint8))
    gray = pp.read_gray_png(p)
    assert gray is not None and gray.shape == g.shape
    assert pp.detect_frame_box(p) == box
    assert pp.interior_blank_metrics(gray, box)["blank"] is True


def test_comsol_auto_window_width_limited():
    # 几何 bbox 宽>高（半圆），轴框为竖长（portrait）→ 宽受限：x 贴几何 bbox，y 居中展开
    bbox = (-0.117, 0.683, -0.057, 0.400)   # gw=0.80, gh=0.457
    crop = (48, 6, 807, 877)                # w_px=759, h_px=871
    x0, x1, y0, y1 = pp.comsol_auto_window(bbox, crop)
    assert x0 == pytest.approx(-0.117, abs=1e-9) and x1 == pytest.approx(0.683, abs=1e-9)
    s = 0.80 / 759
    cy = (-0.057 + 0.400) / 2
    assert y0 == pytest.approx(cy - 871 * s / 2) and y1 == pytest.approx(cy + 871 * s / 2)
    assert (y1 - y0) > (x1 - x0)  # portrait 框 → y 范围被拉大


def test_field_stats():
    g = _fake_grid({"p": np.array([1.0, 2.0, 3.0, 4.0])})
    s = pp.field_stats(g, "p")
    assert s["min"] == 1.0 and s["max"] == 4.0
    assert s["mean"] == pytest.approx(2.5)
    assert s["n"] == 4
    assert s["rms"] == pytest.approx(np.sqrt(np.mean([1.0, 4.0, 9.0, 16.0])))


def test_field_stats_missing_scalar():
    g = _fake_grid({"p": np.array([1.0])})
    with pytest.raises(KeyError):
        pp.field_stats(g, "nope")


def test_validate_field_finite_ok():
    assert pp.validate_field(_fake_grid({"p": np.array([1.0, 2.0])}), "p").passed


def test_validate_field_nan_fails():
    assert not pp.validate_field(_fake_grid({"p": np.array([1.0, np.nan])}), "p").passed


def test_validate_field_max_abs():
    g = _fake_grid({"p": np.array([1.0, 100.0])})
    assert not pp.validate_field(g, "p", max_abs=10.0).passed
    assert pp.validate_field(g, "p", max_abs=1000.0).passed
