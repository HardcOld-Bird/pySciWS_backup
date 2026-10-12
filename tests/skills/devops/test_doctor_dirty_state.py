"""doctor「跟踪 state 脏面哨兵」回归（backlog 20261012-080410-devops 方案 C）。

钉住两件事：① 脏面被列出并给出归因指引（组员合并前检查被挡 / 自愈改脏=缺陷）；
② **advisory 不判失败**——脏跟踪 state 常常是组长在飞的改动或本单自己要提交的文件，
计入 bad 会把正常作业挡成事故，故脏与净两次 doctor 的 rc 相同。
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import pysci.skills.orchestration.tools.sync as sync_mod
from pysci.skills.devops.tools import budget as budget_mod
from pysci.skills.devops.tools import dev, guard_probe, keyscan, probe

DIRTY = ["orchestration/state/registry.json", "orchestration/state/backlog.json"]


@pytest.fixture
def quiet_doctor(tmp_path, monkeypatch):
    """PODS_ROOT/预算两根隔离 + 各段消音，只看脏面段的有无与措辞。"""
    monkeypatch.setattr(dev, "PODS_ROOT", tmp_path / "empty-pods")
    monkeypatch.setattr(budget_mod, "SKILLS_ROOT", tmp_path / "no-skills")
    monkeypatch.setattr(budget_mod, "RULES_ROOT", tmp_path / "no-rules")
    monkeypatch.setattr(budget_mod, "PODS_ROOT", tmp_path / "empty-pods")
    monkeypatch.setattr(sync_mod, "sync", lambda check_only=False: ["[=] 一致"])
    monkeypatch.setattr(probe, "doctor_status", lambda: ("ok", "隔离"))
    monkeypatch.setattr(guard_probe, "run_all", lambda: [])
    monkeypatch.setattr(guard_probe, "report", lambda results: [])
    monkeypatch.setattr(keyscan, "scan", lambda: ([], None))
    monkeypatch.setattr(keyscan, "report", lambda findings, error: [])


def test_doctor_lists_dirty_tracked_state(quiet_doctor, monkeypatch, capsys):
    monkeypatch.setattr(dev, "dirty_tracked_state", lambda **kw: list(DIRTY))
    rc = dev.cmd_doctor(SimpleNamespace(pod=None))
    out = capsys.readouterr().out
    assert "跟踪 state 脏面哨兵" in out
    assert "[!] orchestration/state/registry.json" in out
    assert "先提交或还原" in out and "那是缺陷" in out
    assert rc == 0, "脏面 advisory，不得把 doctor 打成失败"


def test_doctor_reports_clean_face(quiet_doctor, capsys):
    rc = dev.cmd_doctor(SimpleNamespace(pod=None))
    out = capsys.readouterr().out
    assert "跟踪 state 脏面哨兵" in out
    assert "无未提交改动" in out
    assert "[!] orchestration/state" not in out
    assert rc == 0
