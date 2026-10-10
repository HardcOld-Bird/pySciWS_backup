"""doctor「部署台账巡检（sync --check）」联动回归（backlog 20261010-003127-devops）。

锁定：cmd_doctor 跑 sync(check_only=True)，把 [*] 记录哈希漂移（worktree-sync 污染类
问题的症状）与 [!] 副本漂移计为问题（rc=2）并提示重生成；全一致时 rc=0、报「无漂移」。
以 monkeypatch 假 sync.sync 与空 PODS_ROOT 隔离，不触碰真实 manifest/state。
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import pysci.skills.orchestration.tools.sync as sync_mod
from pysci.skills.devops.tools import budget as budget_mod
from pysci.skills.devops.tools import dev


@pytest.fixture
def no_pods(tmp_path, monkeypatch):
    """PODS_ROOT 指向不存在目录 → pod 列表为空；预算两根同样隔离，只看 sync 台账段。"""
    monkeypatch.setattr(dev, "PODS_ROOT", tmp_path / "empty-pods")
    monkeypatch.setattr(budget_mod, "SKILLS_ROOT", tmp_path / "no-skills")
    monkeypatch.setattr(budget_mod, "RULES_ROOT", tmp_path / "no-rules")
    monkeypatch.setattr(budget_mod, "PODS_ROOT", tmp_path / "empty-pods")


def _fake_sync(report: list[str]):
    return lambda check_only=False: list(report)


def test_doctor_flags_recorded_hash_drift(no_pods, monkeypatch, capsys):
    monkeypatch.setattr(
        sync_mod,
        "sync",
        _fake_sync(
            [
                "[*] 真本变更：document-writing（f97e5e09 → dbac3b85）",
                "[=] orchestration → .qoder/skills/orchestration（一致）",
            ]
        ),
    )
    rc = dev.cmd_doctor(SimpleNamespace(pod=None))
    assert rc == 2
    out = capsys.readouterr().out
    assert "部署台账巡检" in out
    assert "真本变更" in out
    assert "pysci-dev sync" in out  # 给出重生成指引


def test_doctor_flags_copy_drift(no_pods, monkeypatch, capsys):
    monkeypatch.setattr(
        sync_mod,
        "sync",
        _fake_sync(["[!] 漂移：X → pod/.qoder/skills/X（副本 aa ≠ 真本 bb）"]),
    )
    rc = dev.cmd_doctor(SimpleNamespace(pod=None))
    assert rc == 2
    assert "漂移" in capsys.readouterr().out


def test_doctor_clean_when_no_drift(no_pods, monkeypatch, capsys):
    monkeypatch.setattr(
        sync_mod,
        "sync",
        _fake_sync(
            [
                "[=] orchestration → .qoder/skills/orchestration（一致）",
                "[=] devops → orchestration/pods/devops/.qoder/skills/devops（一致）",
            ]
        ),
    )
    rc = dev.cmd_doctor(SimpleNamespace(pod=None))
    assert rc == 0
    assert "无漂移" in capsys.readouterr().out
