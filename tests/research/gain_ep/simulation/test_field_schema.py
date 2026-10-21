"""field_schema 契约：两代 schema 同源解析 + 新同义键点名（F6）。"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from pysci.paths import PROJECT_ROOT, research_asset_dir
from pysci.research.gain_ep.simulation import field_schema as fs


def _gain_ep_root() -> Path:
    """worktree 不含 gitignored 重数据 → 只读回退到主树（git common dir 的父级）。"""
    root = research_asset_dir("gain_ep")
    if (root / "simulation" / "field_8period_1m" / "field_schema.json").exists():
        return root
    common = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "--git-common-dir"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return Path(common).resolve().parent / "data" / "research" / "1_gain_ep"


SIM = _gain_ep_root() / "simulation"
SCHEMA_1M = SIM / "field_8period_1m" / "field_schema.json"
SCHEMA_15 = SIM / "field_8period_1p5m" / "field_schema.json"


def test_parse_1m_legacy_schema():
    schema = fs.load_field_schema(SCHEMA_1M)
    assert schema.kind == fs.SCHEMA_KIND
    assert len(schema.diamonds) == 1
    diamond = schema.diamonds[0]
    assert diamond.radial_deg is None
    assert diamond.side_m is not None and np.isclose(diamond.side_m, 0.31)
    assert diamond.source_key == "geometry.inner_diamond"
    assert schema.semicircle is not None
    assert np.isclose(schema.semicircle.radius_m, 0.5)
    assert schema.semicircle.center_m is None
    assert schema.semicircle.source_key == "geometry.display_semicircle"


def test_parse_1p5m_canonical_schema():
    schema = fs.load_field_schema(SCHEMA_15)
    assert schema.kind == fs.SCHEMA_KIND
    assert sorted(d.radial_deg for d in schema.diamonds) == [45.0, 135.0]
    assert all(np.isclose(d.side_m, 0.31) for d in schema.diamonds)
    assert all(d.source_key == "geometry.diamonds" for d in schema.diamonds)
    assert schema.semicircle is not None
    assert np.isclose(schema.semicircle.radius_m, 0.7)
    assert np.allclose(schema.semicircle.center_m, (0.282842712474619, 0.0))
    assert schema.semicircle.source_key == "display_semicircle"


def test_no_key_drift_on_real_schemas():
    for path in (SCHEMA_1M, SCHEMA_15):
        doc = json.loads(path.read_text(encoding="utf-8"))
        assert fs.key_drift_warnings(doc) == [], path


def test_key_drift_names_new_synonyms():
    doc = {"geometry": {"diamond_list": []}, "semicircle_data": {}}
    warns = fs.key_drift_warnings(doc)
    assert any("geometry.diamond_list" in w for w in warns)
    assert any("root.semicircle_data" in w for w in warns)


def test_unknown_diamond_key_does_not_parse_as_diamonds():
    """新同义键不被解析器当正口径（禁再增新同义键的消费面）。"""
    doc = {"geometry": {"diamond_list": [{"side_m": 0.31}]}}
    schema = fs.parse_field_schema(doc)
    assert schema.diamonds == ()


def test_parse_scan_regions_contract():
    """R6 新口径：geometry.scan_regions 可被同源解析。"""
    doc = {
        "kind": fs.SCHEMA_KIND,
        "version": 1,
        "geometry": {
            "scan_regions": [
                {
                    "radial_deg": 135,
                    "shape": "square",
                    "radial_m": 0.31,
                    "transverse_m": 0.31,
                    "near_edge_m": 0.25,
                    "center_m": [-0.106066, 0.388909],
                    "vertices_m": [[0, 0], [0, 1], [1, 1], [1, 0]],
                }
            ]
        },
    }
    schema = fs.parse_field_schema(doc)
    assert len(schema.scan_regions) == 1
    reg = schema.scan_regions[0]
    assert reg.shape == "square"
    assert reg.radial_m == 0.31 and reg.transverse_m == 0.31
    assert reg.near_edge_m == 0.25 and reg.radial_deg == 135.0
    assert reg.source_key == "geometry.scan_regions"


def test_schema_contract_policy():
    contract = fs.schema_contract()
    assert contract["kind"] == fs.SCHEMA_KIND
    assert "diamonds" in contract["geometry"]
    assert "scan_regions" in contract["geometry"]
    assert "display_semicircle" in contract["geometry"]
    caliber = contract["scan_region"]["caliber"]
    assert caliber.startswith("用户口头反馈 2026-10-20")
    assert "菱形永远正方形" in caliber
    assert "禁再增新同义键" in contract["policy"]


def test_parse_rejects_non_mapping():
    with pytest.raises(ValueError):
        fs.parse_field_schema([])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# F9：显示区严格 y>0 口径单一真源（1p8m_ep2 三组 L = 1001.87/1.368/1.984）
# ---------------------------------------------------------------------------
from pysci.skills.scientific_plotting.tools import field  # noqa: E402

CENTER = (0.282842712474619, 0.0)
RADIUS = 0.90
EP2 = _gain_ep_root() / "simulation" / "field_8period_1p8m_ep2"
EXPECTED_STRICT = {
    "groove_gain": 1001.87,
    "groove_nogain": 1.368,
    "gain_nogroove": 1.984,
}
LEGACY_SCHEMA = {
    "groove_gain": 1005.6568,
    "groove_nogain": 2.1720,
    "gain_nogroove": 2.5264,
}


def test_display_declaration_strict_values_and_same_source():
    """声明值 = strict y>0 实测 = vmax-check 读数（同源函数）。"""
    for group, want in EXPECTED_STRICT.items():
        xy, vals = field.load_field_points(EP2 / f"field_{group}_L.csv")
        decl = fs.display_declaration(xy, vals, center=CENTER, radius=RADIUS)
        got = decl[fs.KEY_DISPLAY_MAX]
        assert got == pytest.approx(want, rel=0.01), (group, got)
        chk = field.check_vmax_adaptive(xy, vals, got, center=CENTER, radius=RADIUS)
        assert chk.measured_max == got, "声明值必须与 vmax-check 同源"
        assert chk.n_points == decl[fs.KEY_DISPLAY_POINTS]


def test_counterexample_stale_y_ge_zero_caliber_pinned():
    """旧口径反例：y>=0（含口径线）＝1005.66/2.172/2.526——比 strict y>0 显著偏高。"""
    for group, old in LEGACY_SCHEMA.items():
        xy, vals = field.load_field_points(EP2 / f"field_{group}_L.csv")
        y = xy[:, 1]
        mask = (y >= 0) & ((xy[:, 0] - CENTER[0]) ** 2 + y**2 <= RADIUS**2)
        legacy = float(np.abs(vals[mask]).max())
        assert legacy == pytest.approx(old, rel=0.01)
        strict = fs.display_declaration(xy, vals, center=CENTER, radius=RADIUS)[
            fs.KEY_DISPLAY_MAX
        ]
        assert strict < legacy, "strict y>0 必须小于含口径线的旧读数"


# ---------------------------------------------------------------------------
# 20261020-114233-figure：真实 schema 与 vmax-check 口径一致（f2 可直接引用）
# ---------------------------------------------------------------------------
EP3 = _gain_ep_root() / "simulation" / "field_8period_1p2m_ep3"


def _schema_pairs():
    for path in (EP2 / "field_schema.json", EP3 / "field_schema.json"):
        doc = json.loads(path.read_text(encoding="utf-8"))
        semi = doc["display_semicircle"]
        yield path, doc, (tuple(semi["center_m"]), float(semi["radius_m"]))


def test_real_schemas_display_values_match_vmax_check():
    """两代真实 schema 的 files.* 声明值 == vmax-check 严格 y>0 实测（f2 直接引用）。"""
    for path, doc, (center, radius) in _schema_pairs():
        assert doc["files"], path
        for key, entry in doc["files"].items():
            xy, vals = field.load_field_points(path.parent / f"field_{key}.csv")
            chk = field.check_vmax_adaptive(
                xy, vals, entry[fs.KEY_DISPLAY_MAX], center=center, radius=radius
            )
            assert chk.measured_max == entry[fs.KEY_DISPLAY_MAX], (path.name, key)
            assert chk.n_points == entry[fs.KEY_DISPLAY_POINTS], (path.name, key)
    # 1p8m_ep2 三组 L 的目标值（f2 验收读数）
    doc = json.loads((EP2 / "field_schema.json").read_text(encoding="utf-8"))
    for group, want in EXPECTED_STRICT.items():
        assert doc["files"][f"{group}_L"][fs.KEY_DISPLAY_MAX] == pytest.approx(
            want, rel=0.01
        )


def test_recompute_display_declarations_roundtrip(tmp_path):
    """重算助手：把旧值改回 strict y>0，且二次运行幂等（不写盘）。"""
    path, doc, (center, radius) = next(iter(_schema_pairs()))
    key = "groove_nogain_L"
    stale = dict(doc)
    stale["files"] = {key: dict(doc["files"][key])}
    stale["files"][key][fs.KEY_DISPLAY_MAX] = 2.172
    stale["files"][key][fs.KEY_DISPLAY_POINTS] = 108937
    tmp = tmp_path / "field_schema.json"
    tmp.write_text(
        json.dumps(stale, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    changed = fs.recompute_display_declarations(tmp, csv_dir=EP2)
    assert key in changed and changed[key]["old"] == pytest.approx(2.172, rel=1e-6)
    fixed = json.loads(tmp.read_text(encoding="utf-8"))
    assert fixed["files"][key][fs.KEY_DISPLAY_MAX] == pytest.approx(1.3676, rel=0.01)
    assert fixed["files"][key][fs.KEY_DISPLAY_POINTS] == 108476
    assert fs.recompute_display_declarations(tmp, csv_dir=EP2) == {}, (
        "二次运行须幂等不写盘"
    )
