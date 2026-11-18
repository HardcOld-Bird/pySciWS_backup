"""session 会话脚手架机械断言：目录骨架、npz 落盘、时间线日志、data/ 护栏。"""

from __future__ import annotations

import shutil

import numpy as np
import pytest

from pysci.paths import research_asset_dir
from pysci.skills.theory_calculation import session as theory_session

TEST_NAME = "zz_theorycalc_selftest"


@pytest.fixture()
def session_dir():
    """建立测试会话；结束后整目录清理（不污染 data/）。"""
    base = research_asset_dir(TEST_NAME)
    sess = theory_session.ensure_session(TEST_NAME, "s1")
    yield sess
    shutil.rmtree(base, ignore_errors=True)


def test_ensure_session_scaffold(session_dir):
    assert (session_dir / "results").is_dir()
    assert (session_dir / "plots").is_dir()
    notes = session_dir / "notes.md"
    assert notes.is_file()
    assert notes.read_text(encoding="utf-8").startswith("# 理论计算日志")


def test_save_results_npz_roundtrip(session_dir):
    out = theory_session.save_results(
        session_dir, {"eig": np.array([1.0, 2.0])}, name="eig"
    )
    assert out.name == "eig.npz"
    with np.load(out) as data:
        assert np.allclose(data["eig"], [1.0, 2.0])


def test_append_note_appends_timestamped_entries(session_dir):
    theory_session.append_note(session_dir, "发现 EP 在 k=0.3")
    theory_session.append_note(session_dir, "第二条")
    text = (session_dir / "notes.md").read_text(encoding="utf-8")
    assert "发现 EP 在 k=0.3" in text
    assert "第二条" in text
    assert text.count("## ") >= 2  # 每条一个时间戳小节


def test_writes_outside_data_are_rejected(tmp_path):
    with pytest.raises(ValueError):
        theory_session.save_results(tmp_path, {"a": np.array([0.0])})
    with pytest.raises(ValueError):
        theory_session.append_note(tmp_path, "stray")
