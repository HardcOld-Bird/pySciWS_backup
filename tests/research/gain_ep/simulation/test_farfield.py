"""远场雷达 1° 导出：三组批量 CSV/MAT + schema / 网格校验 / COMSOL 表读取。"""

from __future__ import annotations

import json

import numpy as np
import pytest

from pysci.research.gain_ep.simulation import farfield as ff


def _series(name: str, n: int = 361, scale: float = 1.0) -> ff.RadarSeries:
    angles = np.arange(n, dtype=float)
    amp = scale * (1.0 + np.sin(np.radians(angles)))
    return ff.RadarSeries(name, angles, amp)


def _groups(n: int = 361) -> tuple[ff.RadarGroup, ...]:
    return (
        ff.RadarGroup("groove_gain", (_series("L", n, 2.0), _series("R", n, 1.5))),
        ff.RadarGroup("groove_nogain", (_series("L", n, 0.2), _series("R", n, 0.1))),
        ff.RadarGroup("gain_nogroove", (_series("L", n, 0.3), _series("R", n, 0.25))),
    )


def test_export_three_groups_csv_mat_and_schema(tmp_path):
    written = ff.export_radar_groups(_groups(), tmp_path)
    assert set(written) == {
        "groove_gain.csv",
        "groove_gain.mat",
        "groove_nogain.csv",
        "groove_nogain.mat",
        "gain_nogroove.csv",
        "gain_nogroove.mat",
        "schema",
    }
    csv = np.loadtxt(written["groove_gain.csv"], delimiter=",", skiprows=1)
    assert csv.shape == (361, 3)
    assert csv[0, 0] == 0.0 and csv[-1, 0] == 360.0
    assert np.allclose(csv[:, 1], 2.0 * (1.0 + np.sin(np.radians(csv[:, 0]))))

    from scipy.io import loadmat  # noqa: PLC0415 - 仅在 MAT 断言时依赖

    mat = loadmat(written["groove_nogain.mat"])
    assert mat["angle_deg"].ravel().shape == (361,)
    assert mat["amp_L"].ravel().shape == (361,)
    assert np.isclose(mat["amp_R"].max(), 0.2)

    schema = json.loads(written["schema"].read_text(encoding="utf-8"))
    assert schema["angle"]["points"] == 361
    assert schema["angle"]["step"] == 1.0
    assert schema["value"]["unit"] == "Pa"
    assert set(schema["groups"]) == {
        "groove_gain",
        "groove_nogain",
        "gain_nogroove",
    }
    assert schema["groups"]["groove_gain"]["series"] == ["L", "R"]
    assert schema["legacy_case_keys"]["full"] == "groove_gain"


def test_export_legacy_half_circle_grid(tmp_path):
    groups = _groups(181)
    written = ff.export_radar_groups(
        groups, tmp_path, expected_points=181, full_circle=False
    )
    header = written["groove_gain.csv"].read_text(encoding="utf-8").splitlines()[0]
    assert header == "angle_deg,L,R"
    schema = json.loads(written["schema"].read_text(encoding="utf-8"))
    assert schema["angle"]["stop"] == 180.0


@pytest.mark.parametrize(
    "angles",
    [
        np.arange(360, dtype=float),  # 点数不足
        np.arange(361, dtype=float) * 0.99 + 0.0,  # 步长不统一（末点也偏）
        np.r_[0.0, np.arange(360, dtype=float)][::-1],  # 倒序
    ],
)
def test_validate_angle_grid_rejects_bad_grids(angles):
    with pytest.raises(ValueError):
        ff.validate_angle_grid(angles)


def test_export_rejects_negative_amplitude(tmp_path):
    bad = ff.RadarGroup(
        "groove_gain",
        (ff.RadarSeries("L", np.arange(361, dtype=float), np.full(361, -1.0)),),
    )
    with pytest.raises(ValueError, match="非负"):
        ff.export_radar_groups((bad,), tmp_path)


def _comsol_file(tmp_path, name: str, n: int = 361):
    path = tmp_path / name
    angles = np.arange(n, dtype=float)
    amp = 1.0 + np.sin(np.radians(angles))
    data = np.column_stack(
        (np.cos(np.radians(angles)), np.sin(np.radians(angles)), angles, amp)
    )
    np.savetxt(
        path,
        data,
        delimiter=",",
        header="% Model,test",
        comments="",
        fmt="%.10g",
    )
    return path, angles, amp


def test_cli_export(tmp_path, capsys):
    src, _angles, _amp = _comsol_file(tmp_path, "far.csv")
    out = tmp_path / "out"
    rc = ff.main(
        [
            "--source",
            "groove_gain",
            "L",
            str(src),
            "--out",
            str(out),
            "--formats",
            "csv",
        ]
    )
    assert rc == 0
    assert (out / "radar_1deg_groove_gain.csv").exists()
    assert (out / "radar_1deg_schema.json").exists()
    assert "[radar-export]" in capsys.readouterr().out


def test_groups_from_comsol_csvs(tmp_path):
    path = tmp_path / "far_full_L.csv"
    angles = np.arange(361, dtype=float)
    amp = 1.0 + np.sin(np.radians(angles))
    data = np.column_stack(
        (np.cos(np.radians(angles)), np.sin(np.radians(angles)), angles, amp)
    )
    np.savetxt(
        path,
        data,
        delimiter=",",
        header="% Model,test\n% Expr,abs(acpr.efc1.pext)",
        comments="",
        fmt="%.10g",
    )
    groups = ff.groups_from_comsol_csvs({"groove_gain": {"L": path}})
    assert len(groups) == 1 and groups[0].key == "groove_gain"
    got_angles, got_amp = ff.load_comsol_csv(path)
    assert np.allclose(got_angles, angles)
    assert np.allclose(got_amp, amp)


def test_wide_loader_reads_own_export(tmp_path):
    """导出→读取闭环：宽表 CSV 可直接用 load_radar_csv 读回。"""
    written = ff.export_radar_groups(_groups(), tmp_path, formats=("csv",))
    angles, series = ff.load_radar_csv(written["groove_gain.csv"])
    assert angles.shape == (361,)
    assert list(series) == ["L", "R"]
    assert np.allclose(angles, np.arange(361, dtype=float))
    assert np.allclose(series["L"], 2.0 * (1.0 + np.sin(np.radians(angles))))
    assert np.allclose(series["R"], 1.5 * (1.0 + np.sin(np.radians(angles))))


def test_load_comsol_csv_skips_wide_header(tmp_path):
    """backlog 20261020-030949：宽表表头不再让 load_comsol_csv 崩。"""
    written = ff.export_radar_groups(_groups(), tmp_path, formats=("csv",))
    angles, amp = ff.load_comsol_csv(
        written["groove_gain.csv"], angle_col=0, value_col=1
    )
    assert angles.shape == (361,) and amp.shape == (361,)
    assert np.allclose(amp, 2.0 * (1.0 + np.sin(np.radians(angles))))


def test_wide_loader_rejects_headerless_table(tmp_path):
    src, _angles, _amp = _comsol_file(tmp_path, "far.csv", n=181)
    with pytest.raises(ValueError, match="load_comsol_csv"):
        ff.load_radar_csv(src)


def test_schema_points_loader_hint(tmp_path):
    written = ff.export_radar_groups(_groups(), tmp_path, formats=("csv",))
    schema = json.loads(written["schema"].read_text(encoding="utf-8"))
    assert schema["csv"]["loader"].endswith("load_radar_csv")
