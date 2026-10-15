"""任务书「白名单（写权限）」段**声明即注入** pod-guard（backlog 20261015-132939-fix3）。

实案 b2：任务书明明声明了白名单目录，guard 却照拦成 ``whitelist_conflict``——**声明面**
（lint 的 ``whitelist`` 规则读它、只告警）与**注入面**（``PYSCI_TASK_DIRS`` 只来自
``--dirs``/plan step ``dirs``）脱节。本单把两面接成单点：派发时用 lint 的**同一**解析器
（:func:`dispatch._whitelist_entries`）读出声明，并入 ``PYSCI_TASK_DIRS``。

末两例跑**真 pod-guard**（``node`` + stdin JSON + env，同 ``test_pod_guard_blocks`` 的
方法学）：声明目录放行、声明外目录照旧拦——证明注入既生效又不过宽。
"""

from __future__ import annotations

from pathlib import Path

from pysci.paths import PROJECT_ROOT
from pysci.skills.orchestration.tools import dispatch

from .test_pod_guard_blocks import NODE_REQUIRED, run_guard

TB = """# 任务书 demo

## 要求

改 `src/pysci/skills/orchestration/tools/` 下的东西，测试放 `tests/`。

## 白名单（写权限）

- src/pysci/skills/orchestration/tools/
- tests/

（白名单外功能必需才动、最小改动、显式声明。）
"""


def _taskbook(tmp_path: Path) -> Path:
    p = tmp_path / "task-demo.md"
    p.write_text(TB, encoding="utf-8")
    return p


def test_parses_declared_dirs_from_the_taskbook(tmp_path):
    assert dispatch.taskbook_whitelist_dirs(_taskbook(tmp_path)) == [
        "src/pysci/skills/orchestration/tools",
        "tests",
    ]


def test_missing_or_absent_taskbook_fails_soft(tmp_path):
    assert dispatch.taskbook_whitelist_dirs(tmp_path / "nope.md") == []
    assert dispatch.taskbook_whitelist_dirs(None) == []


def test_compose_keeps_explicit_dirs_first_and_dedupes(tmp_path):
    out = dispatch.dirs_with_taskbook_whitelist(
        ["tests", "data/x"], _taskbook(tmp_path)
    )
    assert out == ["tests", "data/x", "src/pysci/skills/orchestration/tools"]


@NODE_REQUIRED
def test_declared_dir_is_allowed_by_the_real_guard(tmp_path):
    """b2 同款场景：只有任务书声明、没有 ``--dirs`` → 注入后真 guard **放行**。"""
    dirs = dispatch.dirs_with_taskbook_whitelist([], _taskbook(tmp_path))
    pod = tmp_path / "pod"
    pod.mkdir()
    target = PROJECT_ROOT / "tests" / "_probe_declared.md"

    rc, err = run_guard(
        file_path=str(target),
        pod=str(pod),
        project_root=str(PROJECT_ROOT),
        task_dirs=dirs,
    )

    assert rc == 0, err


@NODE_REQUIRED
def test_undeclared_dir_is_still_blocked(tmp_path):
    """注入不过宽：白名单没声明的路径照旧拦（否则等于把 guard 关掉）。"""
    dirs = dispatch.dirs_with_taskbook_whitelist([], _taskbook(tmp_path))
    pod = tmp_path / "pod"
    pod.mkdir()
    target = PROJECT_ROOT / "src" / "pysci" / "paths.py"

    rc, err = run_guard(
        file_path=str(target),
        pod=str(pod),
        project_root=str(PROJECT_ROOT),
        task_dirs=dirs,
    )

    assert rc == 2 and "白名单目录之外" in err


def test_env_carries_the_injected_dirs(tmp_path):
    """声明面 → env 面：``build_env`` 把同一列表写进 ``PYSCI_TASK_DIRS``（guard 的唯一输入）。"""
    from pysci.skills.orchestration.tools import registry, runner

    dirs = dispatch.dirs_with_taskbook_whitelist(["data/x"], _taskbook(tmp_path))
    env = runner.build_env(
        registry.Registry.load().member("devops"), dirs, deployed_skills=[]
    )

    assert env["PYSCI_TASK_DIRS"].split(";") == dirs


def test_injection_point_is_wired_into_dispatch():
    """接线在场：``do_dispatch`` 组装 ``all_dirs`` 时走的是这条通路（改回旧式即红）。"""
    src = Path(dispatch.__file__).read_text(encoding="utf-8")
    assert "dirs_with_taskbook_whitelist(" in src
    assert "all_dirs = list(dirs or []) + extra_dirs\n" not in src, "旧式裸拼接已退役"
