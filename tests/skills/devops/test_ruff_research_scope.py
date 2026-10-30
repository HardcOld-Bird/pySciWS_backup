"""研究脚本 ruff 口径：``data/research/**`` 显式路径检查按研究线同一口径豁免纯风格规则。

backlog 20261028-215208-theory：``data/**`` 在 ruff ``exclude`` 内只对目录遍历生效——
显式路径（``ruff check data/research/.../study.py``，各成员自查常用）仍会报
N803/N806/B905（物理符号 S/A/B、``zip()`` 写法），实测 12 条。本文件钉两件事：
配置在册（per-file-ignores 与整树 exclude 并存）+ 显式路径行为（研究树宽、src 树严）。
"""

from __future__ import annotations

import shutil
import subprocess
import tomllib

import pytest

from pysci.paths import PROJECT_ROOT

PYPROJECT = PROJECT_ROOT / "pyproject.toml"


def _ruff_cfg() -> dict:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["tool"]["ruff"]


def test_research_tree_has_per_file_ignores():
    ignores = _ruff_cfg()["lint"]["per-file-ignores"]
    assert ignores["data/research/**"] == [
        "N",
        "SIM",
        "B",
        "C4",
        "E722",
        "E402",
        "E731",
    ], "研究脚本须与 src 研究线同一豁免口径（含 N、B）"


def test_data_tree_still_excluded_from_traversal():
    assert "data/**" in _ruff_cfg()["exclude"], "data/ 非代码资产整树不参与遍历 lint"


PROBE = "def f(S, A, B):\n    return list(zip(S, A, B))\n"


def _ruff_stdin(filename: str) -> int:
    exe = shutil.which("ruff")
    if exe is None:
        pytest.skip("ruff 不在 PATH（裸环境跳过行为探针，配置在册钉已覆盖）")
    proc = subprocess.run(
        [exe, "check", "--stdin-filename", filename, "-"],
        input=PROBE,
        text=True,
        capture_output=True,
        cwd=str(PROJECT_ROOT),
        check=False,
    )
    return proc.returncode


def test_explicit_research_path_ignores_style_rules():
    """物理符号命名 + 无 strict 的 zip()：研究树显式路径检查应为 0。"""
    assert _ruff_stdin("data/research/1_example/probe.py") == 0


def test_explicit_src_path_still_strict():
    """同一片段在 src 树仍按严格口径报 N803/B905（豁免不外溢）。"""
    assert _ruff_stdin("src/pysci/probe.py") == 1
