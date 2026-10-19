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


def test_schema_contract_policy():
    contract = fs.schema_contract()
    assert contract["kind"] == fs.SCHEMA_KIND
    assert "diamonds" in contract["geometry"]
    assert "display_semicircle" in contract["geometry"]
    assert "禁再增新同义键" in contract["policy"]


def test_parse_rejects_non_mapping():
    with pytest.raises(ValueError):
        fs.parse_field_schema([])  # type: ignore[arg-type]
