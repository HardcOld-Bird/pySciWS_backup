"""H7 vmax 自适应硬项：比值 ∈[1.0,1.1]、显示区实测、CLI 与 rubric 接线。"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from pysci.paths import PODS_ROOT, PROJECT_ROOT, research_asset_dir
from pysci.skills.scientific_plotting.tools import field, figures

RUBRIC = PODS_ROOT / "reviewer" / "rubrics" / "scientific-plotting.md"


def _gain_ep_root() -> Path:
    root = research_asset_dir("gain_ep")
    if (root / "simulation" / "field_8period_1p5m_ep").is_dir():
        return root
    common = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "--git-common-dir"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return Path(common).resolve().parent / "data" / "research" / "1_gain_ep"


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


def test_ep_data_air_domain_baseline_1605():
    """EP 数据：口径线行 2.535 排除；y>0 气域 max=1.605±1%。"""
    path = (
        _gain_ep_root()
        / "simulation"
        / "field_8period_1p5m_ep"
        / "field_groove_nogain_L.csv"
    )
    xy, vals = field.load_field_points(path)
    rep = field.check_vmax_adaptive(
        xy, vals, 2.0, center=(0.282842712474619, 0.0), radius=0.7
    )
    assert rep.measured_max == pytest.approx(1.605, rel=0.01)


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
    src_pass = _write_field_csv(
        tmp_path, [(0.0, 0.1, 35.0), (0.2, 0.3, -20.0)], name="field_pass.csv"
    )
    fail = figures.main(
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
    assert fail == 1
    assert "FAIL" in capsys.readouterr().out

    ok = figures.main(
        [
            "vmax-check",
            "--source",
            src_pass,
            "--vmax",
            "35",
            "--center",
            "0",
            "0",
            "--radius",
            "1",
        ]
    )
    assert ok == 0
    capsys.readouterr()

    rc = figures.main(
        [
            "vmax-check",
            "--source",
            src_pass,
            "--vmax",
            "35",
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
    assert data["measured_max"] == pytest.approx(35.0)
    assert data["region"].startswith("y>0,")


def test_rubric_h7_declares_vmax_and_ep_subquestion():
    text = RUBRIC.read_text(encoding="utf-8")
    assert "vmax 自适应" in text and "[1.0,1.1]" in text and "vmax-check" in text
    assert "工作点 (cr,ci)" in text and "EP" in text
    assert "严格 y>0" in text
    assert RUBRIC.stat().st_size <= 8192


def test_vmax_check_radius_help_documents_strict_caliper(capsys):
    """③ 帮助文本同步：--radius 说明须为严格 y>0（不再写 y>=0）。"""
    with pytest.raises(SystemExit):
        figures.main(["vmax-check", "-h"])
    out = capsys.readouterr().out
    assert "严格 y>0" in out and "y>=0" not in out
