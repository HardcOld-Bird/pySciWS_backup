"""sync 台账即写即提交（backlog 20261013-203724-devops）。

病灶：`orchestration/state/skills-deployed.json` 是 git 跟踪的**生成物**，改任一技能真本
都会重写它。不提交就在 main 留一个台账脏项，而台账又恰是下次「改真本」的单必须更新的
共享面——分支内提交它会被合并交集闸拦（git 自己也会拒覆盖脏文件），不提交则要拆两笔。
本组钉住新口径：写盘后**确有变更即提交**、无变更不提交、`check_only` 不提交、非仓/未跟踪
一律 fail-soft（部署已成事实，记账失败不该连带失败）。
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from pysci.skills.orchestration.tools import sync

LEDGER = "orchestration/state/skills-deployed.json"


def _git(root: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@e", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
        check=True,
    )
    return proc.stdout.strip()


def _scaffold(root: Path) -> None:
    """技能真本 + manifest + 已跟踪的空台账（gitignored 副本目录）。"""
    (root / "orchestration" / "skills" / "demo").mkdir(parents=True)
    (root / "orchestration" / "skills" / "demo" / "SKILL.md").write_text(
        "v1\n", encoding="utf-8"
    )
    (root / "orchestration" / "skills" / "manifest.toml").write_text(
        'version = 1\n\n[[skills]]\nname = "demo"\n'
        'source = "orchestration/skills/demo"\n'
        'targets = ["orchestration/pods/p/.qoder/skills"]\n',
        encoding="utf-8",
    )
    (root / "orchestration" / "state").mkdir(parents=True)
    (root / "orchestration" / "state" / "skills-deployed.json").write_text(
        '{\n  "version": 1,\n  "skills": {}\n}\n', encoding="utf-8"
    )
    (root / ".gitignore").write_text("orchestration/pods/\n", encoding="utf-8")


def _repo(tmp_path: Path) -> Path:
    root = tmp_path / "proj"
    _scaffold(root)
    _git(root, "init", "-q", "-b", "main")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "init")
    return root


def _bind(monkeypatch, root: Path) -> None:
    """把 sync 的三个路径常量指到临时仓（否则会动真仓库——测试绝不能碰它）。"""
    monkeypatch.setattr(sync, "PROJECT_ROOT", root)
    monkeypatch.setattr(
        sync, "MANIFEST_PATH", root / "orchestration" / "skills" / "manifest.toml"
    )
    monkeypatch.setattr(
        sync, "DEPLOYED_PATH", root / "orchestration" / "state" / "skills-deployed.json"
    )


# ---------------------------------------------------------------------------
# 核心：变更即提交 → 不留脏面（于是不再与合并交集闸互斥）
# ---------------------------------------------------------------------------
def test_sync_commits_ledger_on_change(tmp_path, monkeypatch):
    root = _repo(tmp_path)
    _bind(monkeypatch, root)
    report = sync.sync()
    assert any("台账已提交" in ln for ln in report), report
    # 台账不再是 main 上的脏项——正是本单要消除的那类
    assert LEDGER not in _git(root, "status", "--porcelain")
    assert "台账同步" in _git(root, "log", "-1", "--format=%s")


def test_sync_no_commit_when_nothing_changed(tmp_path, monkeypatch):
    root = _repo(tmp_path)
    _bind(monkeypatch, root)
    sync.sync()  # 首次：部署 + 提交
    report = sync.sync()  # 再次：真本与台账都没变
    assert not any("台账已提交" in ln for ln in report), report


def test_check_only_never_commits(tmp_path, monkeypatch):
    root = _repo(tmp_path)
    _bind(monkeypatch, root)
    before = _git(root, "rev-list", "--count", "HEAD")
    report = sync.sync(check_only=True)
    assert not any("台账已提交" in ln for ln in report)
    assert _git(root, "rev-list", "--count", "HEAD") == before  # 未新增提交


def test_autocommit_false_leaves_ledger_dirty(tmp_path, monkeypatch):
    """autocommit=False 是测试/检视用的退出口：部署照做、台账不提交。"""
    root = _repo(tmp_path)
    _bind(monkeypatch, root)
    report = sync.sync(autocommit=False)
    assert not any("台账已提交" in ln for ln in report)
    assert LEDGER in _git(root, "status", "--porcelain")


# ---------------------------------------------------------------------------
# fail-soft：记账失败不该让部署失败
# ---------------------------------------------------------------------------
def test_fail_soft_when_ledger_untracked(tmp_path, monkeypatch):
    root = _repo(tmp_path)
    # 有意不跟踪台账（.gitignore 掉并提交该规则）
    (root / ".gitignore").write_text(
        "orchestration/pods/\n" + LEDGER + "\n", encoding="utf-8"
    )
    _git(root, "rm", "--cached", "-q", "--", LEDGER)
    _git(root, "add", ".gitignore")
    _git(root, "commit", "-q", "-m", "untrack ledger")
    _bind(monkeypatch, root)
    before = _git(root, "rev-list", "--count", "HEAD")
    report = sync.sync()
    assert any("未被跟踪" in ln for ln in report), report
    assert _git(root, "rev-list", "--count", "HEAD") == before  # 没擅自入库


def test_fail_soft_outside_git(tmp_path, monkeypatch):
    root = tmp_path / "nogit"
    _scaffold(root)  # 有真本/manifest/台账，但不是 git 仓
    _bind(monkeypatch, root)
    report = sync.sync()
    assert any("非 git 工作树" in ln for ln in report), report


def test_fail_soft_on_commit_error(tmp_path, monkeypatch):
    """记账被 git 挡下（index.lock 占位）：打提示、部署仍成功、不抛。"""
    root = _repo(tmp_path)
    _bind(monkeypatch, root)
    (root / ".git" / "index.lock").write_text("", encoding="utf-8")
    report = sync.sync()  # 不应抛
    assert any("请手动提交" in ln for ln in report), report
