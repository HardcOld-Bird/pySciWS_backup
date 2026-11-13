"""pysci.toolchain 解析器单测：vendored 绝对路径优先，缺失回退裸命令（fail-open）。

覆盖两类口径：
- 路径解析：``<tool>/current.txt`` → ``<tool>/<ver>/…`` 可执行文件；指针/目录缺失、
  uv tool 未安装时回退裸名字（交 PATH 解析），不抛异常；
- 落盘一致性：真机上 ``current.txt`` 的 git/node/uv 版本须与 ``toolchain.json`` 一致
  （版本真源），且解析出的可执行文件如实存在。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from pysci import toolchain

_TOOLCHAIN_JSON = toolchain.TOOLCHAIN_JSON
_needs_vendored = pytest.mark.skipif(
    not _TOOLCHAIN_JSON.is_file(), reason="vendored 工具链未落盘（未跑 init-deps）"
)


def _fake_deps(root: Path) -> Path:
    deps = root / "deps"
    (deps / "git" / "9.9.9" / "cmd").mkdir(parents=True)
    (deps / "git" / "current.txt").write_text("9.9.9\n", encoding="utf-8")
    (deps / "git" / "9.9.9" / "cmd" / "git.exe").write_text("", encoding="utf-8")
    (deps / "uv").mkdir()
    (deps / "uv" / "current.txt").write_text("0.0.1\r\n", encoding="utf-8")
    (deps / "uv" / "0.0.1").mkdir()
    (deps / "uv" / "0.0.1" / "uv.exe").write_text("", encoding="utf-8")
    (deps / "node" / "runtime").mkdir(parents=True)
    (deps / "node" / "runtime" / "current.txt").write_text("1.0.0", encoding="utf-8")
    (deps / "node" / "runtime" / "1.0.0").mkdir()
    (deps / "node" / "runtime" / "1.0.0" / "node.exe").write_text("", encoding="utf-8")
    return deps


@pytest.fixture()
def fake_deps(tmp_path, monkeypatch):
    deps = _fake_deps(tmp_path)
    monkeypatch.setattr(toolchain, "DEPS_ROOT", deps)
    monkeypatch.setattr(toolchain, "UV_TOOL_BIN_DIR", deps / "uv" / "bin")
    return deps


def test_git_exe_resolves_vendored_absolute_path(fake_deps):
    got = toolchain.git_exe()
    assert got == str(fake_deps / "git" / "9.9.9" / "cmd" / "git.exe")


@pytest.mark.parametrize(
    ("resolver", "name"),
    [(toolchain.uv_exe, "uv.exe"), (toolchain.node_exe, "node.exe")],
    ids=["uv", "node"],
)
def test_uv_node_resolve_vendored(fake_deps, resolver, name):
    got = Path(resolver())
    assert got.is_absolute() and got.is_file() and got.name == name


def test_missing_pointer_falls_back_to_bare_name(tmp_path, monkeypatch):
    empty = tmp_path / "deps"
    empty.mkdir()
    monkeypatch.setattr(toolchain, "DEPS_ROOT", empty)
    assert toolchain.git_exe() == "git"
    assert toolchain.uv_exe() == "uv"
    assert toolchain.bash_exe() == "bash"
    assert toolchain.base_python_exe() == "python"


def test_pointer_without_version_dir_falls_back(fake_deps):
    (fake_deps / "git" / "current.txt").write_text("0.0.0", encoding="utf-8")
    assert toolchain.git_exe() == "git"


def test_uv_tool_exe_resolves_then_falls_back(fake_deps):
    assert toolchain.uv_tool_exe("zotero-cli") == "zotero-cli"
    bin_dir = fake_deps / "uv" / "bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / "zotero-cli.exe").write_text("", encoding="utf-8")
    got = toolchain.uv_tool_exe("zotero-cli")
    assert Path(got) == bin_dir / "zotero-cli.exe"


def test_resolve_command_maps_known_names_and_passes_through_unknown(fake_deps):
    assert toolchain.resolve_command("git") == toolchain.git_exe()
    assert toolchain.resolve_command("perl") == "perl"


def test_prepend_toolchain_path_prefixes_existing_dirs_only(fake_deps):
    os_dir = fake_deps / "git" / "9.9.9" / "usr" / "bin"
    os_dir.mkdir(parents=True)
    got = toolchain.prepend_toolchain_path("BASE")
    parts = got.split(os.pathsep)
    assert parts[-1] == "BASE"
    assert parts[0] == str(fake_deps / "git" / "9.9.9" / "cmd")
    assert str(os_dir) in parts
    assert str(fake_deps / "git" / "9.9.9" / "usr") not in parts  # 未落盘目录不得混入


def test_venv_exe_paths_live_under_project_venv():
    scripts = toolchain.venv_scripts_dir()
    assert toolchain.venv_python_exe() == scripts / "python.exe"
    assert toolchain.venv_pythonw_exe() == scripts / "pythonw.exe"


@_needs_vendored
def test_pointer_versions_match_toolchain_json():
    spec = json.loads(_TOOLCHAIN_JSON.read_text(encoding="utf-8"))
    for tool in ("git", "node", "uv"):
        assert toolchain._current_version(tool) == spec[tool]["version"], tool


@_needs_vendored
def test_resolved_executables_exist_on_disk():
    for exe in (
        toolchain.git_exe(),
        toolchain.bash_exe(),
        toolchain.uv_exe(),
        toolchain.uvx_exe(),
        toolchain.node_exe(),
        toolchain.base_python_exe(),
    ):
        assert Path(exe).is_file(), exe
