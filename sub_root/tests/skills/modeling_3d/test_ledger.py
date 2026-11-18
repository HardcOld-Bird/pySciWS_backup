"""交付冻结对账机械断言：标记行协议 + 指纹对账（漂移/缺失）+ data/ 护栏。"""

from __future__ import annotations

import json
import shutil

import pytest

from pysci.paths import research_asset_dir
from pysci.skills.modeling_3d import ledger

TEST_NAME = "zz_modeling3d_selftest"


@pytest.fixture()
def session_dir():
    """在 data/ 建临时建模会话；结束整目录清理（不污染 data/）。"""
    base = research_asset_dir(TEST_NAME)
    sess = base / "models" / "s1"
    sess.mkdir(parents=True, exist_ok=True)
    yield sess
    shutil.rmtree(base, ignore_errors=True)


def _mk(tmp_path, name: str, content: str):
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


# ── [model3d] 标记行协议 ────────────────────────────────────────────────


def test_parse_markers_basic():
    out = (
        "some noise\n"
        "[model3d] stl: C:\\data\\a.stl\n"
        "[model3d] faces: 12\n"
        "more noise [not a marker]\n"
        "[model3d] volume_mm3: 8000.0000\n"
    )
    m = ledger.parse_markers(out)
    assert m == {
        "stl": "C:\\data\\a.stl",
        "faces": "12",
        "volume_mm3": "8000.0000",
    }


def test_parse_markers_last_wins_and_empty():
    m = ledger.parse_markers("[model3d] faces: 1\n[model3d] faces: 2\n")
    assert m["faces"] == "2"
    assert ledger.parse_markers("clean output") == {}


# ── 指纹对账 ────────────────────────────────────────────────────────────


def test_freeze_and_reconcile_ok(tmp_path):
    script = _mk(tmp_path, "part.py", "print('hi')\n")
    art = _mk(tmp_path, "part.stl", "stl-bytes")
    rec = ledger.build_record(script, [art], ledger_dir=tmp_path, meta={"slug": "s"})
    assert rec["schema"] == ledger.LEDGER_SCHEMA
    assert rec["meta"]["slug"] == "s"

    out = tmp_path / "part.ledger.json"
    out.write_text(
        json.dumps(rec, ensure_ascii=False), encoding="utf-8"
    )
    loaded = ledger.load_record(out)
    r = ledger.reconcile(loaded, base_dir=out.parent)
    assert r.ok
    assert r.script_status == "ok"
    assert r.artifact_status == {"part.stl": "ok"}


def test_reconcile_detects_artifact_tamper(tmp_path):
    script = _mk(tmp_path, "part.py", "print('hi')\n")
    art = _mk(tmp_path, "part.stl", "stl-bytes")
    rec = ledger.build_record(script, [art], ledger_dir=tmp_path)
    art.write_text("hand-edited!", encoding="utf-8")  # 手改模型
    r = ledger.reconcile(rec, base_dir=tmp_path)
    assert not r.ok
    assert r.artifact_status["part.stl"] == "drift"
    assert any("禁止交付" in f for f in r.failures)


def test_reconcile_detects_script_drift_and_missing(tmp_path):
    script = _mk(tmp_path, "part.py", "print('hi')\n")
    art = _mk(tmp_path, "part.stl", "stl-bytes")
    rec = ledger.build_record(script, [art], ledger_dir=tmp_path)
    script.write_text("print('edited')\n", encoding="utf-8")
    art.unlink()
    r = ledger.reconcile(rec, base_dir=tmp_path)
    assert not r.ok
    assert r.script_status == "drift"
    assert r.artifact_status["part.stl"] == "missing"
    assert any("复现链" in f or "脚本" in f for f in r.failures)


def test_reconcile_missing_script(tmp_path):
    script = _mk(tmp_path, "part.py", "print('hi')\n")
    art = _mk(tmp_path, "part.stl", "stl-bytes")
    rec = ledger.build_record(script, [art], ledger_dir=tmp_path)
    script.unlink()
    r = ledger.reconcile(rec, base_dir=tmp_path)
    assert r.script_status == "missing"
    assert not r.ok


def test_build_record_requires_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        ledger.build_record(tmp_path / "nope.py", [], ledger_dir=tmp_path)
    script = _mk(tmp_path, "part.py", "x")
    with pytest.raises(FileNotFoundError):
        ledger.build_record(script, [tmp_path / "nope.stl"], ledger_dir=tmp_path)


# ── freeze：data/ 防散落护栏 + 会话内落盘 ────────────────────────────────


def test_freeze_guard_rejects_outside_data(tmp_path):
    script = _mk(tmp_path, "part.py", "x")
    art = _mk(tmp_path, "part.stl", "y")
    with pytest.raises(ValueError):
        ledger.freeze(script, [art], tmp_path / "x.ledger.json")


def test_freeze_inside_session_and_roundtrip(session_dir, tmp_path):
    script = session_dir / "scripts" / "part.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text("print('model')\n", encoding="utf-8")
    art = session_dir / "stl" / "part.stl"
    art.parent.mkdir(parents=True, exist_ok=True)
    art.write_bytes(b"stl")
    out = ledger.freeze(
        script, [art], session_dir / "ledger" / "part.ledger.json",
        meta={"slug": "s1"},
    )
    assert out.is_file()
    r = ledger.reconcile(ledger.load_record(out), base_dir=out.parent)
    assert r.ok and r.script_status == "ok"
    assert r.format().count("ok") >= 2
