"""pod-guard 的路径纪律硬闸回归（backlog 20261010-193601-devops）。

**为什么单开一份**：`test_delivery_gate_budget.py` 立了「pytest 直跑真 guard」的模式，
但只覆盖 Stop 侧；pod-guard 是 PreToolUse 侧，此前**没有**阳性/阴性 pytest——首版
delivery-gate 的 TDZ 事故（13 例违规被外层 catch 静默吞掉）证明：约定不等于机制，
每 guard 都要有能"注入违规并断言被拦"的用例，否则 fail-open 会让 guard 悄悄退化成
空壳。本文件补齐 pod-guard 的**阳性**（拦截 + 留痕）与**阴性**（放行）双侧。

**同时锁 fail-open 留痕纪律**（20261010-193601-devops 采纳的三件套之一）：任何吞异常
分支必须 stderr 打 `[pod-guard fail-open] <原因>`，`guards-in-force` 元哨兵与
`pysci-dev doctor` 活体探针都以此为据。若将来重构改回静默 catch，本文件的
`test_fail_open_on_bad_stdin_prints_trace` 会立即红。

**方法学**：subprocess 用 `node <GUARD>` + stdin JSON 载荷 + env PYSCI_*；不 mock，
测的就是线上接线的那份代码。
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from pysci.paths import ORCHESTRATION_ROOT, PROJECT_ROOT

GUARD = ORCHESTRATION_ROOT / "guards" / "pod-guard.mjs"
NODE_REQUIRED = pytest.mark.skipif(
    shutil.which("node") is None, reason="需要 node 运行 guard"
)


def run_guard(
    *,
    file_path: str = "",
    pod: str | None = None,
    project_root: str = "",
    readonly: list[str] | None = None,
    deployed: list[str] | None = None,
    task_dirs: list[str] | None = None,
    stdin: str | None = None,
) -> tuple[int, str]:
    """以真实 PreToolUse 姿势调用 pod-guard：stdin 收 tool_use JSON，env 注入白名单。"""
    env = dict(os.environ)
    if pod is not None:
        env["PYSCI_POD"] = pod
    else:
        env.pop("PYSCI_POD", None)
    env["PYSCI_PROJECT_ROOT"] = project_root
    env["PYSCI_READONLY"] = ";".join(readonly or [])
    env["PYSCI_DEPLOYED_SKILLS"] = ";".join(deployed or [])
    env["PYSCI_TASK_DIRS"] = ";".join(task_dirs or [])
    payload = json.dumps({"tool_input": {"file_path": file_path}})
    proc = subprocess.run(
        ["node", str(GUARD)],
        input=stdin if stdin is not None else payload,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
    )
    return proc.returncode, proc.stderr


# ===========================================================================
#  阳性：违规必须被拦（exit 2 + stderr 含具体理由）
# ===========================================================================


@NODE_REQUIRED
def test_blocks_pod_outside_and_untasked(tmp_path):
    """pod 外、也不在 taskDirs 的路径 → exit 2，理由含 '白名单目录之外'。

    故意选 PROJECT_ROOT 下的**不存在的**兄弟路径（不依赖 tmp——tmp 分支已放行 pod 外
    Temp 写入）；guard 只做字符串前缀匹配，路径是否实际存在不影响判定。
    """
    pod = tmp_path / "pod"
    pod.mkdir()
    outside = PROJECT_ROOT / "_pysci_probe_pod_outside" / "x.md"
    rc, err = run_guard(
        file_path=str(outside),
        pod=str(pod),
        project_root=str(tmp_path),
        task_dirs=[],
    )
    assert rc == 2
    assert "白名单目录之外" in err


@NODE_REQUIRED
def test_blocks_readonly_layer_in_pod(tmp_path):
    """pod 内只读层（.qoder/rules/charter.md）→ exit 2，理由含 '只读层'。"""
    pod = tmp_path / "pod"
    (pod / ".qoder" / "rules").mkdir(parents=True)
    rc, err = run_guard(
        file_path=str(pod / ".qoder" / "rules" / "charter.md"),
        pod=str(pod),
        readonly=[".qoder/rules/charter.md"],
    )
    assert rc == 2
    assert "只读层" in err and "charter.md" in err


@NODE_REQUIRED
def test_blocks_deployed_skill_copy(tmp_path):
    """pod 内 `.qoder/skills/<deployed>/…` → exit 2，理由含 '部署副本'。"""
    pod = tmp_path / "pod"
    (pod / ".qoder" / "skills" / "devops").mkdir(parents=True)
    rc, err = run_guard(
        file_path=str(pod / ".qoder" / "skills" / "devops" / "SKILL.md"),
        pod=str(pod),
        deployed=["devops"],
    )
    assert rc == 2
    assert "部署副本" in err


# ===========================================================================
#  阴性：合规路径必须放行（exit 0 + 无 stderr）
# ===========================================================================


@NODE_REQUIRED
def test_allows_pod_maintained_layer(tmp_path):
    """pod 内 AGENTS.md（自维护层，非只读）→ 放行。"""
    pod = tmp_path / "pod"
    pod.mkdir()
    rc, err = run_guard(
        file_path=str(pod / "AGENTS.md"),
        pod=str(pod),
        readonly=[".qoder/rules/charter.md"],
    )
    assert (rc, err) == (0, "")


@NODE_REQUIRED
def test_allows_task_whitelist_dir(tmp_path):
    """taskDirs 白名单内（项目相对）→ 放行。"""
    pod = tmp_path / "pod"
    pod.mkdir()
    work = tmp_path / "src" / "app"
    work.mkdir(parents=True)
    rc, err = run_guard(
        file_path=str(work / "main.py"),
        pod=str(pod),
        project_root=str(tmp_path),
        task_dirs=["src/app"],
    )
    assert (rc, err) == (0, "")


@NODE_REQUIRED
def test_allows_system_tmp(tmp_path, monkeypatch):
    """os.tmpdir 前缀 → 放行（探针 fixture 常写 Temp）。"""
    pod = tmp_path / "pod"
    pod.mkdir()
    tmp_root = Path(tempfile.gettempdir())
    probe = tmp_root / "pysci_probe_target.md"
    rc, err = run_guard(file_path=str(probe), pod=str(pod))
    assert (rc, err) == (0, "")


@NODE_REQUIRED
def test_allows_empty_file_path(tmp_path):
    """tool_input.file_path 缺失 → 静默放行（非 Write/Edit 工具不匹配本 matcher）。"""
    rc, err = run_guard(file_path="", pod=str(tmp_path))
    assert (rc, err) == (0, "")


# ===========================================================================
#  fail-open 留痕纪律（backlog 20261010-193601-devops 三件套之一）
# ===========================================================================


@NODE_REQUIRED
def test_fail_open_on_bad_stdin_prints_trace(tmp_path):
    """stdin 非 JSON → rc=0 + stderr 打 `[pod-guard fail-open]`——留痕不得静默。

    这条断言**专门**为「未来重构把 catch 改回静默」而设：若 fail-open 又变成空壳，
    `guards-in-force` 元哨兵与本用例都会红。"""
    rc, err = run_guard(stdin="{not json", pod=str(tmp_path))
    assert rc == 0
    assert "[pod-guard fail-open]" in err
