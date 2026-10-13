"""`pysci-dev venv-sync`（backlog 20261012-114239-devops）——uv sync 的 Windows 安全配方。

缺陷：``uv sync`` 遇运行中 exe（本机实发=组长挂着 ``pysci-orch.exe watch``）报
「拒绝访问 (os error 5)」而失败，**失败前已把同目录其余入口删光**——裸命令集体
command not found 的半坏 venv，会被误诊成 PATH/安装问题，并让在跑的 drain 会话当场
失去工具面。手工对策（已验证）：改名让路 → sync → 核对入口。本工具把它固化。

锁形态只能在本机真复现（POSIX 的 unlink 看目录权限，模拟不出），故涉锁两例带 skipif。
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

from pysci.skills.devops.tools import dev

_PING = Path("C:/Windows/System32/ping.exe")
image_lock = pytest.mark.skipif(
    os.name != "nt" or not _PING.exists(),
    reason="运行中镜像锁（改名成、删除败）是 Windows 专有形态",
)


@pytest.fixture
def tree(tmp_path, monkeypatch):
    """造 ``PROJECT_ROOT`` + ``.venv/Scripts`` + 三个声明入口（其中 pysci-dev 在盘）。"""
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "pysci"\n\n[project.scripts]\n'
        'pysci-dev = "pysci.a:main"\n'
        'pysci-orch = "pysci.b:main"\n'
        'pysci-bench = "pysci.c:main"\n',
        encoding="utf-8",
    )
    scripts = tmp_path / ".venv" / "Scripts"
    scripts.mkdir(parents=True)
    (scripts / "pysci-dev.exe").write_bytes(b"MZ-old")
    monkeypatch.setattr(dev, "PROJECT_ROOT", tmp_path)
    return tmp_path, scripts


def _fake_uv(scripts: Path, recreate: list[str], rc: int = 0):
    """假 uv：先删光在盘入口（真实 uv 的破坏顺序），再重建 recreate 名单。"""

    def run(extra):
        for f in scripts.glob("pysci-*.exe"):
            f.unlink()
        for name in recreate:
            (scripts / f"{name}.exe").write_bytes(b"MZ-new")
        return rc, "Fake: uv sync 完成"

    return run


# ---------------------------------------------------------------------------
# 判据：声明入口读得准、锁形态判得准
# ---------------------------------------------------------------------------
def test_project_scripts_sorted(tmp_path):
    pp = tmp_path / "pyproject.toml"
    pp.write_text(
        '[project]\nname="p"\n\n[project.scripts]\nzb="m:a"\nab="m:b"\n',
        encoding="utf-8",
    )
    assert dev.project_scripts(pp) == ["ab", "zb"]


def test_project_scripts_broken_or_absent_is_empty(tmp_path):
    """解析不了/没声明 → 空表，调用方据此拒绝核对（不是静当「无需核对」）。"""
    assert dev.project_scripts(tmp_path / "ghost.toml") == []
    bad = tmp_path / "bad.toml"
    bad.write_text("= not toml =\n", encoding="utf-8")
    assert dev.project_scripts(bad) == []


def test_move_aside_restores_unoccupied_file(tmp_path):
    """未被占用：改名+删除都成 ⇒ 探测无害，原文件必须还原在原位。"""
    exe = tmp_path / "pysci-dev.exe"
    exe.write_bytes(b"MZ")

    assert dev._move_aside_locked(exe, "TS") == ""
    assert exe.read_bytes() == b"MZ"
    assert not list(tmp_path.glob("*.locked-TS"))


@image_lock
def test_move_aside_ordinary_handle_is_stuck(tmp_path):
    """普通打开句柄：连改名都被拒 ⇒ 判 stuck，文件原样未动（不能谎报「已让路」）。"""
    exe = tmp_path / "pysci-orch.exe"
    exe.write_bytes(b"MZ")
    handle = open(exe, "rb")
    try:
        assert dev._move_aside_locked(exe, "TS") == "stuck"
    finally:
        handle.close()
    assert exe.exists()


@image_lock
def test_move_aside_running_image_clears_the_name(tmp_path):
    """真运行中镜像：删除被拒（os error 5）但改名能成 ⇒ 原路径让开，uv 得以原地重建。"""
    exe = tmp_path / "pysci-orch.exe"
    shutil.copy2(_PING, exe)
    proc = subprocess.Popen(
        [str(exe), "-t", "127.0.0.1"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        time.sleep(0.6)
        assert proc.poll() is None, "镜像必须真在跑，否则本例什么都没测"
        with pytest.raises(OSError):
            exe.unlink()  # 前提自证：这就是 uv 撞的那把锁
        assert dev._move_aside_locked(exe, "TS") == "moved"
        assert not exe.exists(), "原名必须已让开"
        assert (tmp_path / "pysci-orch.exe.locked-TS").exists()
    finally:
        proc.terminate()
        proc.wait()


def test_locked_holder_reads_wmic_csv(monkeypatch):
    """占用者身份来自 wmic（镜像路径在创建时定格，改名后仍显示原名）。"""

    class Proc:
        returncode = 0
        stdout = (
            "Node,ExecutablePath,ProcessId\r\n"
            "PC-LEADER,C:\\proj\\.venv\\Scripts\\pysci-orch.exe,22676\r\n"
        )

    monkeypatch.setattr(dev.subprocess, "run", lambda *a, **k: Proc())
    got = dev._locked_holder(Path("C:/proj/.venv/Scripts/pysci-orch.exe"))
    assert "PC-LEADER" in got and "22676" in got


def test_locked_holder_silent_when_wmic_unavailable(monkeypatch):
    class Proc:
        returncode = 1
        stdout = ""

    monkeypatch.setattr(dev.subprocess, "run", lambda *a, **k: Proc())
    assert dev._locked_holder(Path("x/pysci-dev.exe")) == ""
    monkeypatch.setattr(
        dev.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(OSError("no wmic"))
    )
    assert dev._locked_holder(Path("x/pysci-dev.exe")) == ""


# ---------------------------------------------------------------------------
# 命令面：不出现半坏态，出现时必报 rc=2
# ---------------------------------------------------------------------------
def test_cmd_venv_sync_needs_declared_scripts(tree, capsys):
    """没有 [project.scripts] 就无从核对——拒绝执行而不是「校验通过」。"""
    root, _ = tree
    (root / "pyproject.toml").write_text('[project]\nname="p"\n', encoding="utf-8")
    assert dev.cmd_venv_sync(_Args(uv_args=[])) == 2
    assert "读不到 [project.scripts]" in capsys.readouterr().out


def test_cmd_venv_sync_ok_when_all_entries_present(tree, monkeypatch, capsys):
    root, scripts = tree
    monkeypatch.setattr(
        dev,
        "run_uv_sync",
        _fake_uv(scripts, ["pysci-dev", "pysci-orch", "pysci-bench"]),
    )
    assert dev.cmd_venv_sync(_Args(uv_args=[])) == 0
    assert "入口齐备" in capsys.readouterr().out


def test_cmd_venv_sync_reports_half_broken_as_rc2(tree, monkeypatch, capsys):
    """缺陷的正解：入口少了必须 **rc=2 + 列名**，绝不能像旧 uv 那样退 0 或只报 import 错。"""
    root, scripts = tree
    monkeypatch.setattr(dev, "run_uv_sync", _fake_uv(scripts, ["pysci-dev"]))
    assert dev.cmd_venv_sync(_Args(uv_args=[])) == 2
    out = capsys.readouterr().out
    assert "入口缺失" in out and "pysci-orch" in out and "pysci-bench" in out


def test_cmd_venv_sync_forwards_uv_args(tree, monkeypatch):
    root, scripts = tree
    seen: list[list[str]] = []

    def run(extra):
        seen.append(list(extra))
        for n in ("pysci-dev", "pysci-orch", "pysci-bench"):
            (scripts / f"{n}.exe").write_bytes(b"MZ")
        return 0, "ok"

    monkeypatch.setattr(dev, "run_uv_sync", run)
    assert dev.cmd_venv_sync(_Args(uv_args=["--offline"])) == 0
    assert seen == [["--offline"]]


@image_lock
def test_cmd_venv_sync_survives_locked_entry(tree, monkeypatch, capsys):
    """端到端（真镜像锁）：入口被占住时仍把三个入口做齐、退出 0，且不让原名挡住 uv。"""
    root, scripts = tree
    locked = scripts / "pysci-orch.exe"
    shutil.copy2(_PING, locked)
    proc = subprocess.Popen(
        [str(locked), "-t", "127.0.0.1"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(0.6)

    # 假 uv 复刻真实破坏顺序：先全删（被锁的那个会抛错）再重建声明过的入口
    def run(extra):
        for f in list(scripts.glob("pysci-*.exe")):
            try:
                f.unlink()
            except OSError:
                raise AssertionError("被锁入口应先由让路步骤处理掉") from None
        for n in ("pysci-dev", "pysci-orch", "pysci-bench"):
            (scripts / f"{n}.exe").write_bytes(b"MZ-new")
        return 0, "Fake: uv sync 完成"

    monkeypatch.setattr(dev, "run_uv_sync", run)
    try:
        assert dev.cmd_venv_sync(_Args(uv_args=[])) == 0
        out = capsys.readouterr().out
        assert "已改名让路" in out and "入口齐备" in out
        assert all(
            (scripts / f"{n}.exe").read_bytes() == b"MZ-new"
            for n in ("pysci-dev", "pysci-orch", "pysci-bench")
        )
    finally:
        proc.terminate()
        proc.wait()


def test_recycle_residue_copy_vs_orphan(tmp_path):
    """原名在盘 ⇒ 残留只是多余副本，删；原名缺失 ⇒ 残留是唯一原件，改回原名。

    反了就会吞掉入口：无 pyproject 变更的 `uv sync` 判定无事可做，**不会**重建被让路的
    exe（2026-10-12 main 实测），此时残留就是那个入口的全部家当。
    """
    (tmp_path / "pysci-dev.exe").write_bytes(b"live")
    (tmp_path / "pysci-dev.exe.locked-20260101000000").write_bytes(b"copy")
    (tmp_path / "pysci-orch.exe.locked-20260101000001").write_bytes(b"only-copy")

    assert dev._recycle_aside_residue(tmp_path) == (1, 1)
    assert not (tmp_path / "pysci-dev.exe.locked-20260101000000").exists()
    assert (tmp_path / "pysci-dev.exe").read_bytes() == b"live"
    assert (tmp_path / "pysci-orch.exe").read_bytes() == b"only-copy"


def test_recycle_residue_tolerates_still_held(tmp_path, monkeypatch):
    """占用进程没退 ⇒ 删不动也不还原（原名在盘），静默留给下一次。"""
    (tmp_path / "pysci-dev.exe").write_bytes(b"live")
    (tmp_path / "pysci-dev.exe.locked-1").write_bytes(b"held")
    monkeypatch.setattr(
        dev.os, "unlink", lambda *a, **k: (_ for _ in ()).throw(OSError("still held"))
    )
    assert dev._recycle_aside_residue(tmp_path) == (0, 0)
    assert (tmp_path / "pysci-dev.exe.locked-1").exists()


def test_cmd_venv_sync_recycles_stale_residue(tree, monkeypatch, capsys):
    """命令面第一步就回收垃圾：让路造的是带本次 ts 的新名字，不会与往次残留撞车。"""
    root, scripts = tree
    stale = scripts / "pysci-dev.exe.locked-20260101000000"
    stale.write_bytes(b"old")
    monkeypatch.setattr(
        dev,
        "run_uv_sync",
        _fake_uv(scripts, ["pysci-dev", "pysci-orch", "pysci-bench"]),
    )
    assert dev.cmd_venv_sync(_Args(uv_args=[])) == 0
    out = capsys.readouterr().out
    assert "回收往次让路：删副本 1 个，还原原件 0 个" in out
    assert not stale.exists()


@pytest.mark.skipif(os.name != "nt", reason="让路分支只在 Windows exe 锁形态下走")
def test_cmd_venv_sync_puts_back_when_uv_skips_rebuild(tree, monkeypatch, capsys):
    """让了路而 uv 没重建 ⇒ 自己把原件改回原名，别让解药变成肇事者。"""
    root, scripts = tree
    for n in ("pysci-orch", "pysci-bench"):  # 本例模拟「入口都在盘、uv 无事可做」
        (scripts / f"{n}.exe").write_bytes(b"MZ-old")

    def fake_aside(exe, ts):
        if exe.name != "pysci-dev.exe":
            return ""
        os.replace(exe, exe.with_name(f"{exe.name}.locked-{ts}"))
        return "moved"

    monkeypatch.setattr(dev, "_move_aside_locked", fake_aside)
    monkeypatch.setattr(dev, "_locked_holder", lambda exe: "占用者=PC PID 1")
    monkeypatch.setattr(dev, "run_uv_sync", lambda extra: (0, "Checked 206 packages"))
    assert dev.cmd_venv_sync(_Args(uv_args=[])) == 0
    out = capsys.readouterr().out
    assert "uv 未重建 ['pysci-dev.exe']" in out and "入口齐备" in out
    assert "仍被占用" not in out  # 让回原地的不是残留，别报假提示
    assert (scripts / "pysci-dev.exe").read_bytes() == b"MZ-old"
    assert not list(scripts.glob("*.locked-*"))


def test_run_uv_sync_uses_project_root(tmp_path, monkeypatch):
    """真实 uv 调用面：cwd 必须是 PROJECT_ROOT（否则可能在 worktree 里建第二套 venv）。"""
    seen: dict = {}

    class Proc:
        returncode = 0
        stdout = "Resolved 40 packages"
        stderr = ""

    def fake_run(argv, **kw):
        seen["argv"] = argv
        seen["cwd"] = kw.get("cwd")
        return Proc()

    monkeypatch.setattr(dev, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(dev.subprocess, "run", fake_run)
    rc, out = dev.run_uv_sync(["--offline"])
    assert rc == 0 and "Resolved" in out
    assert seen["argv"][:3] == ["uv", "sync", "--offline"]
    assert Path(seen["cwd"]) == tmp_path


class _Args:
    def __init__(self, uv_args=None):
        self.uv_args = uv_args or []
