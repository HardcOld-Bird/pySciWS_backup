"""pod-guard work.md 编辑钩子的回归（任务书 20261023-001431 §三「钩子」面）。

方法学同 ``tests/skills/orchestration/test_pod_guard_blocks.py``：subprocess 用
``node <GUARD>`` + stdin JSON 载荷驱动**线上接线的那份 pod-guard.mjs**，不 mock。
坏编辑（增删行/改 ID）必须 exit 2，好编辑与非 work.md 目标必须放行。
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from pysci.paths import ORCHESTRATION_ROOT

GUARD = ORCHESTRATION_ROOT / "guards" / "pod-guard.mjs"
NODE_REQUIRED = pytest.mark.skipif(
    shutil.which("node") is None, reason="需要 node 运行 guard"
)

#: 磁盘现状：头部注释 + 三个 ID 行（钩子只认行结构，内容任意）。
WORK_MD = (
    "# 翻译规则：把每行 TAB 后的英文整行替换为中文翻译；保留行首 ID；不增删行。\n"
    "# 行首 ID 与行数不许改动（pod-guard 钩子硬校验）。\n"
    "\n"
    "S0001\tHello world.\n"
    "S0002\tSecond sentence.\n"
    "C0001\tFig. 1. The setup.\n"
)


def run_guard(
    tool_input: dict, *, project_root: str, task_dirs: list[str]
) -> tuple[int, str]:
    """以真实 PreToolUse 姿势调用 pod-guard：stdin 收 tool_use JSON，env 注入白名单。"""
    env = dict(os.environ)
    env["PYSCI_PROJECT_ROOT"] = project_root
    env["PYSCI_POD"] = str(Path(project_root) / "some-pod")
    env["PYSCI_READONLY"] = ""
    env["PYSCI_DEPLOYED_SKILLS"] = ""
    env["PYSCI_TASK_DIRS"] = ";".join(task_dirs)
    env["PYSCI_GUARD_LOG"] = ""  # 关闭 guard 日志，不污染真实 state
    proc = subprocess.run(
        ["node", str(GUARD)],
        input=json.dumps({"tool_input": tool_input}),
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
    )
    return proc.returncode, proc.stderr


@pytest.fixture
def work_md(tmp_path: Path) -> Path:
    """``<tmp>/bilingual/demo/work.md``：命中钩子的 bilingual/*/work.md 形态。"""
    p = tmp_path / "bilingual" / "demo" / "work.md"
    p.parent.mkdir(parents=True)
    p.write_text(WORK_MD, encoding="utf-8")
    return p


@NODE_REQUIRED
def test_good_edit_passes(work_md: Path, tmp_path: Path) -> None:
    """好编辑（只换 TAB 后译文，ID 与行数不动）放行。"""
    rc, err = run_guard(
        {
            "file_path": str(work_md),
            "old_string": "S0001\tHello world.",
            "new_string": "S0001\t你好世界。",
        },
        project_root=str(tmp_path),
        task_dirs=[str(tmp_path)],
    )
    assert rc == 0, err


@NODE_REQUIRED
def test_edit_line_count_change_blocked(work_md: Path, tmp_path: Path) -> None:
    """Edit 增删行（old/new 行数不等）硬拦 exit 2。"""
    rc, err = run_guard(
        {
            "file_path": str(work_md),
            "old_string": "S0001\tHello world.",
            "new_string": "S0001\t你好世界。\n新增一行",
        },
        project_root=str(tmp_path),
        task_dirs=[str(tmp_path)],
    )
    assert rc == 2
    assert "行结构硬校验失败" in err and "行数不等" in err


@NODE_REQUIRED
def test_edit_id_prefix_change_blocked(work_md: Path, tmp_path: Path) -> None:
    """Edit 改行首 ID 前缀（S0001→S0009）硬拦 exit 2。"""
    rc, err = run_guard(
        {
            "file_path": str(work_md),
            "old_string": "S0001\tHello world.",
            "new_string": "S0009\t你好世界。",
        },
        project_root=str(tmp_path),
        task_dirs=[str(tmp_path)],
    )
    assert rc == 2
    assert "ID 前缀" in err


@NODE_REQUIRED
def test_edit_partial_id_fragment_blocked(work_md: Path, tmp_path: Path) -> None:
    """ID 前缀的碎片改写（行内 ``001\tHello``→``009\t你好``，拼出 S0009）也拦——
    按落盘结果核验 ID 序列，逐行前缀比对拦不住的碎片替换同样逃不掉。"""
    rc, err = run_guard(
        {
            "file_path": str(work_md),
            "old_string": "001\tHello",
            "new_string": "009\t你好",
        },
        project_root=str(tmp_path),
        task_dirs=[str(tmp_path)],
    )
    assert rc == 2
    assert "ID 前缀" in err


@NODE_REQUIRED
def test_write_same_structure_passes(work_md: Path, tmp_path: Path) -> None:
    """Write 整篇覆盖但行数与 ID 序列不变（假翻译场景）放行。"""
    translated = WORK_MD.replace("Hello world.", "你好世界。")
    rc, err = run_guard(
        {"file_path": str(work_md), "content": translated},
        project_root=str(tmp_path),
        task_dirs=[str(tmp_path)],
    )
    assert rc == 0, err


@NODE_REQUIRED
def test_write_line_count_change_blocked(work_md: Path, tmp_path: Path) -> None:
    """Write 增删行（行数与磁盘现状不一致）硬拦 exit 2。"""
    rc, err = run_guard(
        {"file_path": str(work_md), "content": WORK_MD + "S0003\tExtra line.\n"},
        project_root=str(tmp_path),
        task_dirs=[str(tmp_path)],
    )
    assert rc == 2
    assert "行数与磁盘现状不一致" in err


@NODE_REQUIRED
def test_write_id_sequence_change_blocked(work_md: Path, tmp_path: Path) -> None:
    """Write 调换/改写 ID 前缀序列（行数不变也拦）。"""
    swapped = WORK_MD.replace("S0001\t", "X0000\t")
    rc, err = run_guard(
        {"file_path": str(work_md), "content": swapped},
        project_root=str(tmp_path),
        task_dirs=[str(tmp_path)],
    )
    assert rc == 2
    assert "ID 前缀序列与磁盘现状不一致" in err


@NODE_REQUIRED
def test_write_missing_disk_blocked(tmp_path: Path) -> None:
    """磁盘没有现成 work.md（prepare 生成物）时 Write 拒绝手建。"""
    fresh = tmp_path / "bilingual" / "fresh" / "work.md"
    fresh.parent.mkdir(parents=True)
    rc, err = run_guard(
        {"file_path": str(fresh), "content": "S0001\tHello.\n"},
        project_root=str(tmp_path),
        task_dirs=[str(tmp_path)],
    )
    assert rc == 2
    assert "bilingual prepare" in err


@NODE_REQUIRED
def test_non_work_md_behavior_unchanged(tmp_path: Path) -> None:
    """非 work.md 目标不受此闸影响（好编辑坏行数也放行，白名单自己管）。"""
    other = tmp_path / "notes.md"
    other.write_text("a\nb\n", encoding="utf-8")
    rc, err = run_guard(
        {
            "file_path": str(other),
            "old_string": "a\nb",
            "new_string": "a\nb\nc",
        },
        project_root=str(tmp_path),
        task_dirs=[str(tmp_path)],
    )
    assert rc == 0, err


@NODE_REQUIRED
def test_inbox_work_md_not_gated(tmp_path: Path) -> None:
    """用户专属区（data/用户的inbox/）先于钩子放行——即使路径命中 bilingual/*/work.md。"""
    inbox_md = tmp_path / "data" / "用户的inbox" / "bilingual" / "demo" / "work.md"
    inbox_md.parent.mkdir(parents=True)
    inbox_md.write_text(WORK_MD, encoding="utf-8")
    rc, err = run_guard(
        {"file_path": str(inbox_md), "content": "任意内容\n"},
        project_root=str(tmp_path),
        task_dirs=[str(tmp_path)],
    )
    assert rc == 0, err


@NODE_REQUIRED
def test_good_edit_on_absent_old_string_passes(work_md: Path, tmp_path: Path) -> None:
    """old_string 在盘上不存在（Edit 会在工具层报错）：结构无从改变，钩子放行。"""
    rc, err = run_guard(
        {
            "file_path": str(work_md),
            "old_string": "S9999\tNot on disk.",
            "new_string": "S9999\t别的。",
        },
        project_root=str(tmp_path),
        task_dirs=[str(tmp_path)],
    )
    assert rc == 0, err
