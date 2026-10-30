"""H7 vmax 自适应硬项：判定口径（v2 端标阶梯默认 / tight 遗留窗）、显示区实测、CLI 与 rubric 接线。"""

from __future__ import annotations

import json

import numpy as np
import pytest

from pysci.paths import PODS_ROOT
from pysci.skills.scientific_plotting.tools import field, figures

RUBRIC = PODS_ROOT / "reviewer" / "rubrics" / "scientific-plotting.md"


def test_boundary_ratio_40_over_36_05_fails():
    """组 i 现值 40/36.05=1.109 恰越界 → FAIL（边界用例）。"""
    xy = np.array([[0.0, 0.1], [0.2, 0.3]])
    vals = np.array([36.05, -20.0])
    rep = field.check_vmax_adaptive(xy, vals, 40.0, center=(0.0, 0.0), radius=1.0)
    assert not rep.ok
    assert rep.measured_max == pytest.approx(36.05)
    assert rep.ratio == pytest.approx(40.0 / 36.05)


def test_ratio_one_exact_full_scale_passes():
    """vmax=35、实测 max=35 → 恰满标，PASS。"""
    xy = np.array([[0.0, 0.1], [0.2, 0.3]])
    vals = np.array([35.0, -20.0])
    rep = field.check_vmax_adaptive(xy, vals, 35.0, center=(0.0, 0.0), radius=1.0)
    assert rep.ok and rep.ratio == pytest.approx(1.0)


def test_display_region_excludes_outside_values():
    """显示区外（y<0 或 r>ROI）的大值不计入实测 max。"""
    xy = np.array(
        [
            [0.0, 0.1],  # 区内
            [0.0, -0.5],  # y<0
            [5.0, 0.0],  # r>ROI
        ]
    )
    vals = np.array([30.0, 999.0, 999.0])
    rep = field.check_vmax_adaptive(xy, vals, 32.0, center=(0.0, 0.0), radius=1.0)
    assert rep.measured_max == pytest.approx(30.0)
    assert rep.ok and rep.n_points == 1


def test_y_zero_caliber_line_excluded_strictly():
    """严格 y>0：y=0 口径线行的大值不计入（M1 收紧）。"""
    xy = np.array([[0.0, 0.0], [0.0, 0.1]])
    vals = np.array([99.0, 1.0])
    rep = field.check_vmax_adaptive(xy, vals, 2.0, center=(0.0, 0.0), radius=1.0)
    assert rep.measured_max == pytest.approx(1.0)
    assert rep.region.startswith("y>0")


def test_air_domain_baseline_synthetic_strict_y_positive(tmp_path):
    """全局技能测试（合成 fixture，不依赖任务数据）：严格 y>0 气域 max 排除 y=0 口径线行与区外大值。

    原 EP 真数据基线（依赖任务数据）在此改为合成点集：意图不变——严格 y>0 口径下只取
    y>0、r<=ROI 内的点；口径线（y=0）行与显示区外的大值均不计入。
    """
    rows = [
        (0.28, 0.10, 1.71),  # 气域内（y>0）最大
        (0.30, 0.20, 1.20),
        (0.25, 0.30, 0.80),
        (0.28, 0.00, 2.50),  # y=0 口径线行：大值须被严格 y>0 排除
        (0.30, 0.00, 2.20),
        (5.00, 0.10, 99.0),  # 显示区外：须被 ROI 排除
    ]
    path = _write_field_csv(tmp_path, rows)
    xy, vals = field.load_field_points(path)

    rep = field.check_vmax_adaptive(
        xy, vals, 2.0, center=(0.282842712474619, 0.0), radius=0.6
    )

    assert rep.measured_max == pytest.approx(1.71)
    assert rep.region.startswith("y>0")


def test_invalid_inputs_raise():
    xy = np.array([[0.0, 0.1]])
    vals = np.array([1.0])
    with pytest.raises(ValueError):
        field.check_vmax_adaptive(xy, vals, 0.0)
    with pytest.raises(ValueError):
        field.check_vmax_adaptive(xy, vals, 1.0, radius=0.0)
    with pytest.raises(ValueError):
        field.check_vmax_adaptive(np.array([[0.0, -1.0]]), vals, 1.0, radius=1.0)


def _write_field_csv(tmp_path, rows, name: str = "field.csv") -> str:
    path = tmp_path / name
    np.savetxt(path, np.asarray(rows, dtype=float), delimiter=",", fmt="%.10g")
    return str(path)


def test_cli_vmax_check_pass_fail_and_json(tmp_path, capsys):
    src = _write_field_csv(tmp_path, [(0.0, 0.1, 36.05), (0.2, 0.3, -20.0)])
    # 默认 ceil-one-sig 口径（v2 端标阶梯）：40 = ceil_label(36.05) 合规端标 →
    # PASS——旧 tight 窗 40/36.05=1.109 恰越界的假 FAIL 由此消除。
    ok = figures.main(
        [
            "vmax-check",
            "--source",
            src,
            "--vmax",
            "40",
            "--center",
            "0",
            "0",
            "--radius",
            "1",
        ]
    )
    out = capsys.readouterr().out
    assert ok == 0
    assert "PASS" in out and "ceil-one-sig" in out

    # 显著偏离取整规则 → FAIL（真量程错误仍拦）。
    bad = figures.main(
        [
            "vmax-check",
            "--source",
            src,
            "--vmax",
            "45",
            "--center",
            "0",
            "0",
            "--radius",
            "1",
        ]
    )
    out = capsys.readouterr().out
    assert bad == 1
    assert "FAIL" in out and "期望 vmax=40" in out

    # --rounding tight 保留旧比值窗（40/36.05=1.109 越界 → FAIL）。
    legacy = figures.main(
        [
            "vmax-check",
            "--source",
            src,
            "--vmax",
            "40",
            "--rounding",
            "tight",
            "--center",
            "0",
            "0",
            "--radius",
            "1",
        ]
    )
    out = capsys.readouterr().out
    assert legacy == 1
    assert "FAIL" in out and "口径 tight" in out

    rc = figures.main(
        [
            "vmax-check",
            "--source",
            src,
            "--vmax",
            "40",
            "--center",
            "0",
            "0",
            "--radius",
            "1",
            "--json",
        ]
    )
    data = json.loads(capsys.readouterr().out)
    assert rc == 0 and data["ok"] is True
    assert data["rounding"] == "ceil-one-sig"
    assert data["expected_vmax"] == pytest.approx(40.0)
    assert data["measured_max"] == pytest.approx(36.05)
    assert data["region"].startswith("y>0,")


def test_cli_default_rounding_is_v2_ladder(tmp_path, capsys):
    """默认口径即 v2 端标阶梯：2.11 的合规端标是 2.5（旧一位有效数字规则会给 3）。"""
    src = _write_field_csv(tmp_path, [(0.0, 0.1, 2.11), (0.2, 0.3, -0.5)])
    ok = figures.main(["vmax-check", "--source", src, "--vmax", "2.5"])
    out = capsys.readouterr().out
    assert ok == 0 and "PASS" in out and "期望 vmax=2.5" in out

    bad = figures.main(["vmax-check", "--source", src, "--vmax", "3"])
    out = capsys.readouterr().out
    assert bad == 1 and "FAIL" in out and "期望 vmax=2.5" in out


def test_ceil_label_rounding_judgment():
    """v2 端标口径：端标=ceil_label(实测 max) 即合格；偏离/裁峰均拦。"""
    xy = np.array([[0.0, 0.1], [0.2, 0.3]])
    vals = np.array([36.05, -20.0])
    ok = field.check_vmax_adaptive(xy, vals, 40.0, rounding="ceil-one-sig")
    assert ok.ok and ok.expected_vmax == pytest.approx(40.0)
    dev = field.check_vmax_adaptive(xy, vals, 45.0, rounding="ceil-one-sig")
    assert not dev.ok
    clip = field.check_vmax_adaptive(xy, vals, 35.0, rounding="ceil-one-sig")
    assert not clip.ok
    tight = field.check_vmax_adaptive(xy, vals, 40.0, rounding="tight")
    assert not tight.ok and tight.expected_vmax is None
    with pytest.raises(ValueError):
        field.check_vmax_adaptive(xy, vals, 40.0, rounding="nope")


def test_ceil_label_v2_ladder_in_vmax_check():
    """v2 放宽后可接受的端标：1.4655→1.5、2.11→2.5（旧一位有效数字规则下会判 1.5/2.5 FAIL）。"""
    xy = np.array([[0.0, 0.1], [0.2, 0.3]])
    near = field.check_vmax_adaptive(
        xy, np.array([1.4655, -0.2]), 1.5, rounding="ceil-one-sig"
    )
    assert near.ok and near.expected_vmax == pytest.approx(1.5)

    two = field.check_vmax_adaptive(
        xy, np.array([2.11, -0.5]), 2.5, rounding="ceil-one-sig"
    )
    assert two.ok and two.expected_vmax == pytest.approx(2.5)

    # 非 v2 合法值仍拦：3（d1=3,d2=0 高于期望 2.5）与 2.0（低于 2.11，裁峰）。
    assert not field.check_vmax_adaptive(
        xy, np.array([2.11, -0.5]), 3.0, rounding="ceil-one-sig"
    ).ok
    assert not field.check_vmax_adaptive(
        xy, np.array([2.11, -0.5]), 2.0, rounding="ceil-one-sig"
    ).ok


def test_group_rounding_ceil_label_driver():
    """组共享口径同归 judge_vmax：driver=ceil_label(实测 max) 合格才整组 PASS。"""
    xy = np.array([[0.0, 0.1]])
    strong = field.check_vmax_adaptive(
        xy, np.array([38.79]), 40.0, rounding="ceil-one-sig"
    )
    weak = field.check_vmax_adaptive(
        xy, np.array([2.339]), 40.0, rounding="ceil-one-sig"
    )
    group = field.check_vmax_group(
        [strong, weak], sources=["L", "R"], rounding="ceil-one-sig"
    )
    assert group.ok and group.driver.source == "L"
    off = field.check_vmax_adaptive(xy, np.array([30.0]), 40.0, rounding="ceil-one-sig")
    assert not field.check_vmax_group([off], rounding="ceil-one-sig").ok


def test_rubric_h7_declares_vmax_and_ep_subquestion():
    text = RUBRIC.read_text(encoding="utf-8")
    assert "vmax 自适应" in text and "[1.0,1.1]" in text and "vmax-check" in text
    assert "ceil_label" in text, (
        "H7 判据须与 D2 端标口径同源（v2 阶梯，裁决 2026-10-27）"
    )
    assert "工作点 (cr,ci)" in text and "EP" in text
    assert "严格 y>0" in text
    assert RUBRIC.stat().st_size <= 8192


def test_vmax_check_radius_help_documents_strict_caliper(capsys):
    """③ 帮助文本同步：--radius 说明须为严格 y>0（不再写 y>=0）。"""
    with pytest.raises(SystemExit):
        figures.main(["vmax-check", "-h"])
    out = capsys.readouterr().out
    assert "严格 y>0" in out and "y>=0" not in out


# ---------------------------------------------------------------------------
# 组共享量程口径（backlog 20261026-231148-reviewer）
# ---------------------------------------------------------------------------
def _group_csvs(tmp_path, *, left_max: float, right_max: float) -> tuple[str, str]:
    left = _write_field_csv(tmp_path, [(0.0, 0.1, left_max)], name="L.csv")
    right = _write_field_csv(tmp_path, [(0.0, 0.1, right_max)], name="R.csv")
    return left, right


def test_group_shared_single_weak_member_is_informational(tmp_path, capsys):
    """单文件 + --group-shared：弱侧 ratio 远超窗也只作 informational，不再 FAIL。"""
    src = _write_field_csv(tmp_path, [(0.0, 0.1, 2.339)])
    rc = figures.main(["vmax-check", "--source", src, "--vmax", "40", "--group-shared"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "PASS" in out and "弱侧" in out and "：FAIL" not in out


def test_group_shared_single_clipped_member_still_fails(tmp_path, capsys):
    """ratio<1 是裁峰（vmax 装不下本面板峰值），任何口径下都 FAIL。"""
    src = _write_field_csv(tmp_path, [(0.0, 0.1, 45.0)])
    rc = figures.main(["vmax-check", "--source", src, "--vmax", "40", "--group-shared"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "FAIL" in out and "裁峰" in out


def test_group_two_sources_weak_side_not_fail(tmp_path, capsys):
    """L/R 共用 ±vmax：driver（L，实测最大）在窗内 → 整组 PASS，R 侧只列 informational。"""
    left, right = _group_csvs(tmp_path, left_max=38.79, right_max=2.339)
    rc = figures.main(
        ["vmax-check", "--source", left, "--source", right, "--vmax", "40"]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "组共享量程" in out and "PASS" in out
    assert "driver=" in out and "L.csv" in out and "R.csv" in out
    assert "weak" in out and "不计 FAIL" in out


def test_group_driver_over_window_fails(tmp_path, capsys):
    """vmax 对组内最大面板也过大（ratio>hi）→ FAIL（真错误仍被拦）。"""
    left, right = _group_csvs(tmp_path, left_max=30.0, right_max=2.3)
    rc = figures.main(
        ["vmax-check", "--source", left, "--source", right, "--vmax", "40"]
    )
    out = capsys.readouterr().out
    assert rc == 1
    assert "FAIL" in out and "driver 越窗" in out


def test_group_driver_clipping_fails(tmp_path, capsys):
    left, right = _group_csvs(tmp_path, left_max=45.0, right_max=2.3)
    rc = figures.main(
        ["vmax-check", "--source", left, "--source", right, "--vmax", "40"]
    )
    assert rc == 1
    assert "FAIL" in capsys.readouterr().out


def test_group_json_shape(tmp_path, capsys):
    left, right = _group_csvs(tmp_path, left_max=38.79, right_max=2.339)
    rc = figures.main(
        [
            "vmax-check",
            "--source",
            left,
            "--source",
            right,
            "--vmax",
            "40",
            "--json",
        ]
    )
    data = json.loads(capsys.readouterr().out)
    assert rc == 0 and data["ok"] is True and data["mode"] == "group"
    assert data["allow"] == [1.0, 1.1]
    assert data["rounding"] == "ceil-one-sig"
    assert data["expected_vmax"] == pytest.approx(40.0)
    assert data["driver"]["source"].endswith("L.csv")
    roles = {
        m["source"].split("\\")[-1].split("/")[-1]: m["role"] for m in data["members"]
    }
    assert roles == {"L.csv": "driver", "R.csv": "weak"}


def test_single_fail_hints_group_shared(tmp_path, capsys):
    """未加开关的单文件弱侧 FAIL 需自动提示组共享口径。"""
    src = _write_field_csv(tmp_path, [(0.0, 0.1, 2.339)])
    rc = figures.main(["vmax-check", "--source", src, "--vmax", "40"])
    out = capsys.readouterr().out
    assert rc == 1 and "FAIL" in out
    assert "组共享量程" in out and "--group-shared" in out


def test_check_vmax_group_driver_selection():
    xy = np.array([[0.0, 0.1]])
    strong = field.check_vmax_adaptive(xy, np.array([38.79]), 40.0)
    weak = field.check_vmax_adaptive(xy, np.array([2.339]), 40.0)
    group = field.check_vmax_group([strong, weak], sources=["L", "R"])
    assert group.ok and group.driver.source == "L"
    assert group.members[1].role == "weak" and not group.members[1].in_window
    assert group.members[0].in_window


def test_check_vmax_group_rejects_mixed_vmax_and_empty():
    xy = np.array([[0.0, 0.1]])
    a = field.check_vmax_adaptive(xy, np.array([1.0]), 1.0)
    b = field.check_vmax_adaptive(xy, np.array([1.0]), 2.0)
    with pytest.raises(ValueError):
        field.check_vmax_group([])
    with pytest.raises(ValueError):
        field.check_vmax_group([a, b])
