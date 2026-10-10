"""mdgen-check（backlog 20261010-191922-devops）——钩子终态治理工具。

设计意图（问题）：脚本批量产出 `orchestration/skills/**` 下 .md 后，首次提交必被
pre-commit 的 `end-of-file-fixer` / `trailing-whitespace` 改写而失败——钩子打印
`Fixing <file>` 后中止提交，改动留在工作区（`AM`），必须再 `git add` + 提交一次。
更隐蔽的是：脚本作者若在生成后立即做逐字节/哈希断言、随后才提交，则钩子改写让
那条断言**当场过期**——报告的是一个不再为真的证据。

本工具委托 pre-commit 自身跑到收敛（零算法漂移），报每文件 pre/post sha256；
「钩子终态哈希」即内容等价性验证的唯一基准。测试双轨：端到端 1 例真跑 pre-commit
锁定行为，其余用 monkeypatch 覆盖控制流（去重、缺失、失败码、不收敛、CLI 参数）。
"""

from __future__ import annotations

import hashlib
import importlib.util
import shutil
import subprocess
from pathlib import Path

import pytest

from pysci.skills.devops.tools import mdgen

CFG = """\
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v6.0.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
"""


def _init_repo(root: Path) -> None:
    """在 tmp_path 造一个自足 git 仓库（pre-commit 需 git 上下文 + config）。"""
    subprocess.run(["git", "init", "-q", "."], cwd=str(root), check=True)
    (root / ".pre-commit-config.yaml").write_text(CFG, encoding="utf-8")


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


_HAS_GIT = shutil.which("git") is not None
_HAS_PRECOMMIT = importlib.util.find_spec("pre_commit") is not None
e2e = pytest.mark.skipif(
    not (_HAS_GIT and _HAS_PRECOMMIT),
    reason="端到端测试需 git 与 pre_commit 模块；缺任一即跳过（控制流单测仍跑）",
)


# --------------------------------------------------------------------------
# 端到端（真跑 pre-commit；一次锁定 trailing-whitespace + end-of-file-fixer 行为）
# --------------------------------------------------------------------------
@e2e
def test_e2e_dirty_file_converges_to_hook_state(tmp_path):
    _init_repo(tmp_path)
    p = tmp_path / "x.md"
    p.write_bytes(b"hello   \n\n\n")  # 行末空白 + 多余末尾换行

    results = mdgen.check_files([p], cwd=tmp_path)

    assert len(results) == 1
    r = results[0]
    assert r.pre_size == 11 and r.post_size == 6
    assert r.delta == -5
    assert r.changed is True
    # 钩子终态：行末空白削掉、末尾保留一个 \n
    assert p.read_bytes() == b"hello\n"
    # post_hash 与磁盘字节一致（供事后校验用）
    assert r.post_hash == _sha(p.read_bytes())


@e2e
def test_e2e_clean_file_reports_no_change(tmp_path):
    _init_repo(tmp_path)
    p = tmp_path / "y.md"
    p.write_bytes(b"already fine\n")

    results = mdgen.check_files([p], cwd=tmp_path)

    r = results[0]
    assert r.changed is False
    assert r.pre_hash == r.post_hash
    assert p.read_bytes() == b"already fine\n"


@e2e
def test_e2e_dry_run_restores_bytes(tmp_path):
    _init_repo(tmp_path)
    p = tmp_path / "z.md"
    original = b"trailing   \n\n"
    p.write_bytes(original)

    results = mdgen.check_files([p], cwd=tmp_path, dry_run=True)

    # 报告"钩子会改写"
    assert results[0].changed is True
    # 但磁盘字节**未改**——dry_run 的核心承诺
    assert p.read_bytes() == original


# --------------------------------------------------------------------------
# 控制流单测（monkeypatch _run_hook；不真起 pre-commit，毫秒级）
# --------------------------------------------------------------------------
def _patch_run_hook(monkeypatch, side_effect):
    """把 mdgen._run_hook 替换为可控 fake；side_effect(hook_id, files) -> rc。"""
    calls: list[tuple[str, int]] = []

    def fake(hook_id, files, *, cwd):
        calls.append((hook_id, len(files)))
        return side_effect(hook_id, files)

    monkeypatch.setattr(mdgen, "_run_hook", fake)
    return calls


def test_missing_paths_are_skipped(tmp_path, monkeypatch):
    calls = _patch_run_hook(monkeypatch, lambda h, f: 0)
    p = tmp_path / "onDisk.md"
    p.write_bytes(b"a\n")

    results = mdgen.check_files([p, tmp_path / "ghost.md"], cwd=tmp_path)

    assert len(results) == 1 and results[0].path == p
    # 缺失路径不进 argv；只跑两支钩子
    assert calls == [("trailing-whitespace", 1), ("end-of-file-fixer", 1)]


def test_duplicate_paths_are_deduped(tmp_path, monkeypatch):
    calls = _patch_run_hook(monkeypatch, lambda h, f: 0)
    p = tmp_path / "same.md"
    p.write_bytes(b"a\n")

    results = mdgen.check_files([p, p, p], cwd=tmp_path)

    assert len(results) == 1
    # 去重后 files 长度=1
    assert all(n == 1 for _, n in calls)


def test_all_empty_input_returns_empty(tmp_path, monkeypatch):
    calls = _patch_run_hook(monkeypatch, lambda h, f: 0)
    assert mdgen.check_files([], cwd=tmp_path) == []
    assert calls == []  # 空输入根本不该调 pre-commit


def test_hook_error_rc_raises_runtime_error(tmp_path, monkeypatch):
    """rc 不属于 {0,1} → RuntimeError（钩子状态不确定，宁拒不误报终态哈希）。"""

    def side(hook, files):
        return 3 if hook == "trailing-whitespace" else 0

    _patch_run_hook(monkeypatch, side)
    p = tmp_path / "a.md"
    p.write_bytes(b"a\n")

    with pytest.raises(RuntimeError, match="意外退出码"):
        mdgen.check_files([p], cwd=tmp_path)


def test_non_convergence_raises_after_max_rounds(tmp_path, monkeypatch):
    """两支钩子每轮都 rc=1 → 3 轮上限后 RuntimeError（防呆钩子互相打架）。"""

    _patch_run_hook(monkeypatch, lambda h, f: 1)
    p = tmp_path / "b.md"
    p.write_bytes(b"a\n")

    with pytest.raises(RuntimeError, match="仍未收敛"):
        mdgen.check_files([p], cwd=tmp_path)


def test_second_round_needed_when_first_round_rewrites(tmp_path, monkeypatch):
    """round1 有改写、round2 全 rc=0 → 正常收敛返回（不触发 MAX_ROUNDS 抛错）。"""

    state = {"calls": 0}

    def side(hook, files):
        state["calls"] += 1
        # 前两次调用（round1 两支钩子）返 1；round2 两支返 0
        return 1 if state["calls"] <= 2 else 0

    _patch_run_hook(monkeypatch, side)
    p = tmp_path / "c.md"
    p.write_bytes(b"a\n")

    results = mdgen.check_files([p], cwd=tmp_path)

    assert len(results) == 1
    assert state["calls"] == 4  # 两支 × 两轮


# --------------------------------------------------------------------------
# format_results + main CLI
# --------------------------------------------------------------------------
from pysci.skills.devops.tools.mdgen import FileResult


def test_format_results_empty_summary(tmp_path):
    txt = mdgen.format_results([])
    assert "无在盘文件" in txt


def test_format_results_shows_fixed_and_ok(tmp_path):
    a = tmp_path / "a.md"
    b = tmp_path / "b.md"
    results = [
        FileResult(
            path=a, pre_hash="x" * 64, post_hash="y" * 64, pre_size=10, post_size=6
        ),
        FileResult(
            path=b, pre_hash="z" * 64, post_hash="z" * 64, pre_size=4, post_size=4
        ),
    ]
    txt = mdgen.format_results(results)
    assert "[FIXED]" in txt and "[OK   ]" in txt
    assert "Δ-4" in txt
    assert "合计 2 文件，钩子改写 1 个" in txt
    assert "post_sha256" in txt


@e2e
def test_main_dry_run_reports_changes_without_touching(tmp_path, monkeypatch, capsys):
    """CLI 路径：--dry-run 报告 changed 数，字节还原；rc=0。"""
    _init_repo(tmp_path)
    p = tmp_path / "cli.md"
    original = b"cli   \n\n"
    p.write_bytes(original)

    monkeypatch.chdir(tmp_path)
    rc = mdgen.main(["--dry-run", str(p)])

    assert rc == 0
    assert p.read_bytes() == original


def test_main_runtime_error_maps_to_rc_2(tmp_path, monkeypatch, capsys):
    def side(hook, files):
        return 5

    _patch_run_hook(monkeypatch, side)
    p = tmp_path / "e.md"
    p.write_bytes(b"a\n")
    monkeypatch.chdir(tmp_path)

    rc = mdgen.main([str(p)])

    assert rc == 2
    assert "mdgen-check 失败" in capsys.readouterr().err
