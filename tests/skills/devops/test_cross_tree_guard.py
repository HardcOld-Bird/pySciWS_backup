"""pysci-dev 跨树自检（backlog 20261012-010508-devops）。

``pysci`` 以 editable 装入，``pysci.paths.PROJECT_ROOT`` 由**包文件位置**决定、与 cwd
无关：在 worktree 目录里漏设 ``PYTHONPATH=<wt>/src`` 裸跑 ``pysci-dev sync``，命令照旧
退出 0、逐行输出「（一致）」，读写的却全是 main 的真本与 pod 副本——「看起来跑了但什么
都没发生」；``doctor`` 更糟，它据 main 状态给出「健康」的假结论。

本测试钉住三条：跨树必须 **rc=2 且先于任何动手**、同树照旧放行、``--force`` 明示知情
后放行。
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from pysci.skills.devops.tools import dev


def _git(root: Path, *args: str) -> str:
    """在 root 跑 git（测试夹具需要真 git 上下文，带 identity 防无配置报错）。"""
    proc = subprocess.run(
        [
            "git",
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@example.invalid",
            *args,
        ],
        cwd=str(root),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return proc.stdout


@pytest.fixture
def trees(tmp_path, monkeypatch):
    """造两棵树并把代码面钉在 main：``main`` 与 ``main/.qoder/worktrees/t1``。"""
    main = tmp_path / "main"
    wt = main / ".qoder" / "worktrees" / "t1"
    wt.mkdir(parents=True)
    for root in (main, wt):
        (root / "pyproject.toml").write_text(
            '[project]\nname = "pysci"\n', encoding="utf-8"
        )
    monkeypatch.setattr(dev, "PROJECT_ROOT", main)
    return main, wt


def test_cross_tree_warning_names_both_roots_and_fix(trees):
    """告警必须把两棵树都摆出来并给出可抄的修法（只说「错了」等于没帮上忙）。"""
    main, wt = trees
    warn = dev.cross_tree_warning(wt)
    assert "[!] 跨树执行" in warn
    assert str(wt) in warn and str(main) in warn
    assert "PYTHONPATH" in warn and "--force" in warn


def test_same_tree_is_silent(trees):
    """main 本身及其任意子目录都算同树——pod 目录里跑巡检不该被拦。"""
    main, _ = trees
    assert dev.cross_tree_warning(main) == ""
    assert dev.cross_tree_warning(main / "orchestration" / "pods" / "devops") == ""


def test_worktree_subdir_and_foreign_tree_still_flagged(trees, tmp_path):
    """判据是**脚下有没有自己的 pyproject.toml**：worktree 就摆在 main/.qoder/worktrees
    里，「不在项目根之下」根本判不出跨树；而 <wt>/src 深处同样要报。"""
    main, wt = trees
    (wt / "src").mkdir()
    assert "跨树" in dev.cross_tree_warning(wt / "src")
    # 无主之地（Temp）不是静默失真面：命令确定地作用于代码面所指项目根，放行
    outside = tmp_path / "no-project"
    outside.mkdir()
    assert dev.cross_tree_warning(outside) == ""
    monkey_root = tmp_path / "other-project"
    monkey_root.mkdir()
    (monkey_root / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    assert dev.cross_tree_warning(monkey_root) != ""
    assert main != monkey_root  # 夹具自证：不是同一棵树


def test_main_rejects_cross_tree_before_dispatch(trees, monkeypatch, capsys):
    """rc=2 之外更要紧的是**不派活**：cmd_sync 一次都不被调用，main 侧文件不被碰。"""
    _, wt = trees
    called: list[str] = []
    monkeypatch.setattr(
        dev, "cmd_sync", lambda args: called.append("sync") or 0, raising=True
    )
    monkeypatch.setattr(dev, "cmd_doctor", lambda args: called.append("doctor") or 0)
    monkeypatch.chdir(wt)

    assert dev.main(["sync"]) == 2
    assert dev.main(["doctor", "--pod", "devops"]) == 2
    assert called == []
    out = capsys.readouterr().out
    assert out.count("[!] 跨树执行") == 2
    assert "--force" in out


def test_force_dispatches_after_warning(trees, monkeypatch, capsys):
    """--force 是知情放行：仍然打印告警（留痕），但把命令跑完。"""
    _, wt = trees
    monkeypatch.setattr(dev, "cmd_sync", lambda args: 7)
    monkeypatch.chdir(wt)

    assert dev.main(["sync", "--force"]) == 7
    out = capsys.readouterr().out
    assert "[!] 跨树执行" in out and "继续执行" in out


def test_same_tree_dispatches_without_noise(trees, monkeypatch, capsys):
    main, _ = trees
    seen: list[str] = []
    monkeypatch.setattr(dev, "cmd_sync", lambda args: seen.append("sync") or 0)
    monkeypatch.chdir(main)

    assert dev.main(["sync", "--check"]) == 0
    assert seen == ["sync"]
    assert "跨树" not in capsys.readouterr().out


def test_every_subcommand_accepts_force(trees, monkeypatch):
    """自检是全局护栏：所有子命令都得有 --force 出口，否则新命令一上线就堵死人。"""
    main, _ = trees
    seen: list[str] = []
    for func in (
        "cmd_doctor",
        "cmd_sync",
        "cmd_venv_sync",
        "cmd_worktree",
        "cmd_skilltax",
        "cmd_probe",
        "cmd_mdgen_check",
    ):
        monkeypatch.setattr(
            dev, func, lambda args, _n=func: seen.append(_n) or 0, raising=True
        )
    monkeypatch.chdir(main)

    for argv in (
        ["doctor", "--force"],
        ["sync", "--force"],
        ["venv-sync", "--force"],
        ["worktree", "list", "--force"],
        ["skilltax", "--force"],
        ["probe", "--force"],
        ["mdgen-check", str(main / "x.md"), "--force"],
    ):
        assert dev.main(argv) == 0, f"{argv} 不接受 --force"
    assert seen == [
        "cmd_doctor",
        "cmd_sync",
        "cmd_venv_sync",
        "cmd_worktree",
        "cmd_skilltax",
        "cmd_probe",
        "cmd_mdgen_check",
    ]


# ---------------------------------------------------------------------------
# 标准 worktree 配方不该被自家护栏误拦（backlog 20261012-094646-devops）
# ---------------------------------------------------------------------------
def _linked(root: Path) -> Path:
    """把 root 标成 linked worktree：``.git`` 是文件（内容为 gitdir: 指针）。"""
    (root / ".git").write_text("gitdir: ../../.git/worktrees/t1\n", encoding="utf-8")
    return root


@pytest.fixture
def recipe(tmp_path, monkeypatch):
    """配方形状：脚下=主仓 main，代码面=其下 linked worktree（``PYTHONPATH=<wt>/src``）。"""
    main = tmp_path / "main"
    wt = main / ".qoder" / "worktrees" / "t1"
    wt.mkdir(parents=True)
    for root in (main, wt):
        (root / "pyproject.toml").write_text('[project]\nname = "pysci"\n', "utf-8")
    _linked(wt)
    monkeypatch.setattr(dev, "PROJECT_ROOT", wt)
    return main, wt


def test_standard_recipe_is_allowed_without_force(recipe, monkeypatch, capsys):
    """护栏要防的是「脚下 worktree、代码面指 main」，配方恰好相反——判据方向不能搞反。"""
    main, wt = recipe
    assert dev.cross_tree_warning(main) == ""

    seen: list[str] = []
    monkeypatch.setattr(dev, "cmd_sync", lambda args: seen.append("sync") or 0)
    monkeypatch.chdir(main)
    assert dev.main(["sync"]) == 0
    assert seen == ["sync"] and "--force" not in capsys.readouterr().out
    assert dev.PROJECT_ROOT == wt  # 夹具自证：操作对象确实是 worktree 那棵树


def test_allowance_needs_real_worktree_marker(tmp_path, monkeypatch):
    """放行只认 linked worktree 的 ``.git`` **文件**：内嵌的另一棵树（无 .git）仍拦，
    否则「PROJECT_ROOT 恰好在脚下目录里随便某处」也会被当配方放过。"""
    main = tmp_path / "main"
    inner = main / "vendored" / "other"
    inner.mkdir(parents=True)
    for root in (main, inner):
        (root / "pyproject.toml").write_text('[project]\nname = "x"\n', "utf-8")
    monkeypatch.setattr(dev, "PROJECT_ROOT", inner)
    assert dev.cross_tree_warning(main) != ""


def test_reverse_accident_shape_still_blocked(recipe, monkeypatch, capsys):
    """反向事故：cwd 在 worktree 里、代码面却指外层 main——照旧 rc=2 拒执行。"""
    main, wt = recipe
    monkeypatch.setattr(dev, "PROJECT_ROOT", main)
    monkeypatch.chdir(wt)
    assert dev.main(["sync"]) == 2
    assert "跨树执行" in capsys.readouterr().out


def test_guard_agrees_with_real_linked_worktree(tmp_path):
    """`.git` 是文件这一判据须与真 git 一致（造不出 linked worktree 则跳过）。"""
    if shutil.which("git") is None:
        pytest.skip("需 git")
    main = tmp_path / "repo"
    main.mkdir()
    (main / "pyproject.toml").write_text('[project]\nname = "pysci"\n', "utf-8")
    _git(main, "init", "-q", "-b", "main")
    _git(main, "add", "pyproject.toml")
    _git(main, "commit", "-q", "-m", "i")
    wt = main / ".qoder" / "worktrees" / "t1"
    _git(main, "worktree", "add", "-q", "-b", "wt-t1", str(wt))

    assert (wt / ".git").is_file() and (main / ".git").is_dir()
    old = dev.PROJECT_ROOT
    dev.PROJECT_ROOT = wt
    try:
        assert dev.cross_tree_warning(main) == "", (
            "真 linked worktree + 主仓脚下 = 配方"
        )
    finally:
        dev.PROJECT_ROOT = old
    dev.PROJECT_ROOT = main
    try:
        assert dev.cross_tree_warning(wt) != "", "事故形状（代码面指外层 main）仍拦"
    finally:
        dev.PROJECT_ROOT = old
