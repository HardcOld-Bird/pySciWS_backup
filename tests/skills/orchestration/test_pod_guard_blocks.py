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
import re
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
    guard_log: str | None = None,
    node_tmp: str | None = None,
    stdin: str | None = None,
) -> tuple[int, str]:
    """以真实 PreToolUse 姿势调用 pod-guard：stdin 收 tool_use JSON，env 注入白名单。

    ``guard_log`` 缺省置空串＝**关闭启动日志**，避免测试往真实
    ``orchestration/state/guard-log.jsonl`` 追加行；要验日志时传临时路径。
    """
    env = dict(os.environ)
    if pod is not None:
        env["PYSCI_POD"] = pod
    else:
        env.pop("PYSCI_POD", None)
    env["PYSCI_PROJECT_ROOT"] = project_root
    env["PYSCI_READONLY"] = ";".join(readonly or [])
    env["PYSCI_DEPLOYED_SKILLS"] = ";".join(deployed or [])
    env["PYSCI_TASK_DIRS"] = ";".join(task_dirs or [])
    env["PYSCI_GUARD_LOG"] = guard_log if guard_log is not None else ""
    if node_tmp is not None:
        # Node 的 os.tmpdir() 读 TEMP/TMP：把「系统临时目录」挪走，才能让 tmp 下的
        # fixture 真正走白名单判定（否则 tmp 兜底把所有测试写入都放行了）。
        env["TEMP"] = node_tmp
        env["TMP"] = node_tmp
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


@NODE_REQUIRED
def test_startup_log_records_resolved_task_dirs(tmp_path):
    """guard 启动即把 PYSCI_TASK_DIRS 的实际解析结果写日志（相对路径按项目根）。"""
    log = tmp_path / "guard-log.jsonl"
    pod = tmp_path / "pod"
    pod.mkdir()
    fake_tmp = tmp_path / "node-tmp"
    fake_tmp.mkdir()
    rc, err = run_guard(
        file_path=str(tmp_path / "data" / "x" / "note.md"),
        pod=str(pod),
        project_root=str(tmp_path),
        task_dirs=["data/x"],
        guard_log=str(log),
        node_tmp=str(fake_tmp),
    )
    assert rc == 0, err
    rows = [json.loads(ln) for ln in log.read_text(encoding="utf-8").splitlines()]
    row = rows[-1]
    assert row["task_dirs_env"] == "data/x"
    assert row["task_dirs_resolved"][0].replace("\\", "/").endswith("/data/x")

    # 空集也要留痕（「环境变量为空」正是 whitelist_conflict 的常见现场）
    rc, _ = run_guard(
        file_path=str(tmp_path / "src" / "y.md"),
        pod=str(pod),
        project_root=str(tmp_path),
        task_dirs=[],
        guard_log=str(log),
        node_tmp=str(fake_tmp),
    )
    assert rc == 2
    last = json.loads(log.read_text(encoding="utf-8").splitlines()[-1])
    assert last["task_dirs_resolved"] == []


# ===========================================================================
#  用户专属区新语义（2026-10-22 用户裁决）：增删改放行，删除须用户同意
# ===========================================================================


@NODE_REQUIRED
def test_inbox_write_is_allowed(tmp_path):
    """组员对 data/用户的inbox/ 可增删改——写入（创建/修改）面放行。"""
    pod = tmp_path / "pod"
    pod.mkdir()
    rc, err = run_guard(
        file_path=str(PROJECT_ROOT / "data" / "用户的inbox" / "note.md"),
        pod=str(pod),
        project_root=str(PROJECT_ROOT),
    )
    assert rc == 0, err


@NODE_REQUIRED
def test_inbox_outbox_write_still_allowed(tmp_path):
    """outbox 不再是唯一例外（2026-10-22 取代只读版），与任意 inbox 写入同权。"""
    pod = tmp_path / "pod"
    pod.mkdir()
    rc, err = run_guard(
        file_path=str(
            PROJECT_ROOT
            / "data"
            / "用户的inbox"
            / "大任务"
            / "x"
            / "outbox"
            / "交付.md"
        ),
        pod=str(pod),
        project_root=str(PROJECT_ROOT),
    )
    assert rc == 0, err


@NODE_REQUIRED
def test_inbox_delete_via_bash_is_blocked(tmp_path):
    """删除须征求用户同意——Bash 删除动词 × 专属区路径同现即拦。"""
    pod = tmp_path / "pod"
    pod.mkdir()
    stdin = json.dumps({"tool_input": {"command": "rm data/用户的inbox/旧稿.md"}})
    rc, err = run_guard(pod=str(pod), project_root=str(PROJECT_ROOT), stdin=stdin)
    assert rc == 2 and "征求用户同意" in err


@NODE_REQUIRED
def test_inbox_delete_via_python_unlink_is_blocked(tmp_path):
    """python 侧删除（.unlink()/rmtree）同样在保守网内。"""
    pod = tmp_path / "pod"
    pod.mkdir()
    stdin = json.dumps(
        {
            "tool_input": {
                "command": "python -c \"from pathlib import Path; Path('data/用户的inbox/x.txt').unlink()\""
            }
        }
    )
    rc, err = run_guard(pod=str(pod), project_root=str(PROJECT_ROOT), stdin=stdin)
    assert rc == 2 and "征求用户同意" in err


@NODE_REQUIRED
def test_inbox_non_delete_bash_is_allowed(tmp_path):
    """非删除的 Bash（读取/列目录）不拦——保守网只针对删除动词。"""
    pod = tmp_path / "pod"
    pod.mkdir()
    stdin = json.dumps({"tool_input": {"command": "ls data/用户的inbox"}})
    rc, err = run_guard(pod=str(pod), project_root=str(PROJECT_ROOT), stdin=stdin)
    assert rc == 0, err


@NODE_REQUIRED
def test_inbox_rename_via_bash_is_allowed(tmp_path):
    """改名/移动＝「改」，不在「删除须同意」范围（2026-10-22 裁决）。"""
    pod = tmp_path / "pod"
    pod.mkdir()
    stdin = json.dumps(
        {"tool_input": {"command": "mv data/用户的inbox/a.md data/用户的inbox/b.md"}}
    )
    rc, err = run_guard(pod=str(pod), project_root=str(PROJECT_ROOT), stdin=stdin)
    assert rc == 0, err


@NODE_REQUIRED
def test_delete_outside_inbox_not_in_scope(tmp_path):
    """专属区外的删除不归本门禁管（任务白名单面由既有规则覆盖）。"""
    pod = tmp_path / "pod"
    pod.mkdir()
    stdin = json.dumps({"tool_input": {"command": "rm /tmp/junk.txt"}})
    rc, err = run_guard(pod=str(pod), project_root=str(PROJECT_ROOT), stdin=stdin)
    assert rc == 0, err


def test_src_keeps_no_dependency_on_user_inbox():
    """通用基建禁依赖用户专属区（2026-10-22 裁决）：src/ 内仅 settings.py 可**构造**
    ``用户的inbox`` 路径（用户自管配置真源的显式例外，读侧 fail-open）；消息文案里
    提及路径不计（判别＝独立路径段字面量 ``"用户的inbox"``）。需用用户区文件时
    复制到 data/research、data/skills 用副本。"""
    seg = re.compile(r"['\"]用户的inbox['\"]")
    hits = []
    for p in sorted((PROJECT_ROOT / "src").rglob("*.py")):
        try:
            text = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if seg.search(text):
            hits.append(p.relative_to(PROJECT_ROOT).as_posix())
    assert hits == ["src/pysci/skills/orchestration/tools/settings.py"], (
        f"src/ 出现用户专属区路径构造（除 settings.py 显式例外外禁止）：{hits}"
    )
