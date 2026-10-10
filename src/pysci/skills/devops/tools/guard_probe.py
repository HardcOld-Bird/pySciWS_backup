"""guard 在法活体探针（backlog 20261010-193601-devops）。

**要解决的具体坑**：`delivery-gate` / `pod-guard` 用 fail-open 语义——异常一律
exit 0 放行。首版 delivery-gate 因 ESM 顶层 TDZ 抛 ReferenceError 被外层 catch 吞掉，
13 例应拦的违规全部静默通过；`pysci-dev doctor` 报"钩子接线齐全"、"手动喂一个大
AGENTS.md exit code 是 0"——观测面完全看不出 guard 已退化成一个空壳。

本模块把「违规真的被拦」固化为 doctor 的活体哨兵：Temp 造自足 fixture，`node`
跑**真 guard**，断言两件事——

1. **在法**（阳性）：违规入参 → exit 2 + stderr 含具体理由关键字（`AGENTS.md` /
   `只读层` 等）；
2. **留痕**（fail-open 纪律）：坏 stdin → exit 0 + stderr 含 `[<name> fail-open]`
   标签。留痕是「静默退化」唯一的可观测面——若哪天重构改回裸 `catch {}`，本项
   立即红；单靠「rc=0」验不出 guard 究竟是在执法还是压根没跑。

方法学与 `tests/skills/orchestration/test_delivery_gate_budget.py` 一致：subprocess
`node <GUARD>` + stdin JSON 载荷 + `PYSCI_*` 环境注入，不 mock。整段探针 <1s（无
headless 一跳），故**不需要** agentsMdExcludes 那种无时间戳台账——每次 doctor
现场跑即可。node 不在 PATH 或 fixture 出意外 → **不假装绿**，用 `inconclusive`
等级让 doctor 显示 `[!]`（advisory，rc 不硬失败；与 fail-open 语义一致，观测层
不阻塞生产，但也不掩盖）。

用法：由 `pysci-dev doctor` 自动调用；单跑：
`PYTHONPATH=<repo>/src python -m pysci.skills.devops.tools.guard_probe`
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from pysci.paths import ORCHESTRATION_ROOT

DELIVERY_GATE = ORCHESTRATION_ROOT / "guards" / "delivery-gate.mjs"
POD_GUARD = ORCHESTRATION_ROOT / "guards" / "pod-guard.mjs"

#: pod 自维护层尺寸硬闸（与 budget-layers.md §1、basic.md §1 同口径）
LIMIT = 8192


@dataclass(frozen=True)
class GuardProbeResult:
    """单 guard 的活体探针结论。

    ``status`` 三档：``ok`` 在法且留痕；``bad`` 至少一项断言失败（硬告警）；
    ``inconclusive`` 探针环境异常（缺 node / guard 文件不存在等，advisory 告警，
    不计入 rc）。
    """

    name: str
    status: str  # ok | bad | inconclusive
    in_force: bool  # 阳性：违规被拦
    fail_open_traced: bool  # 阴性 + 留痕：坏 stdin 有 [name fail-open] 标签
    detail: str


def _which_node() -> str | None:
    return shutil.which("node")


def _run(
    node: str, guard: Path, *, payload: str, env: dict[str, str]
) -> tuple[int, str]:
    full_env = dict(os.environ)
    full_env.update(env)
    proc = subprocess.run(
        [node, str(guard)],
        input=payload,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=full_env,
        timeout=30,
    )
    return proc.returncode, proc.stderr or ""


def _mk_pod_fixture(root: Path, *, oversize_agents: bool) -> Path:
    """造一个最小自足 pod：AGENTS.md + .qoder/rules/charter.md；oversize 时 AGENTS 超 8192B。"""
    pod = root / "probe_pod"
    (pod / ".qoder" / "rules").mkdir(parents=True, exist_ok=True)
    agents_content = "x" * (LIMIT + 4096) if oversize_agents else "# 记忆\n稳定事实。\n"
    (pod / "AGENTS.md").write_text(agents_content, encoding="utf-8")
    (pod / ".qoder" / "rules" / "charter.md").write_text(
        "---\ntrigger: always_on\n---\n\n# charter\n", encoding="utf-8"
    )
    return pod


def probe_delivery_gate() -> GuardProbeResult:
    """Stop 侧：违规 AGENTS.md 必被拦（exit 2）+ 坏 stdin 必留痕（[delivery-gate fail-open]）。"""
    name = "delivery-gate"
    node = _which_node()
    if node is None:
        return GuardProbeResult(name, "inconclusive", False, False, "node 不在 PATH")
    if not DELIVERY_GATE.exists():
        return GuardProbeResult(
            name, "inconclusive", False, False, f"guard 文件缺失：{DELIVERY_GATE}"
        )

    with tempfile.TemporaryDirectory(prefix="pysci-gp-dgate-") as td:
        pod = _mk_pod_fixture(Path(td), oversize_agents=True)
        stop_ok = json.dumps(
            {
                "last_assistant_message": "<result>已交付</result>",
                "stop_hook_active": False,
                "cwd": str(pod),
            }
        )
        env = {"PYSCI_POD": str(pod), "PYSCI_DEPLOYED_SKILLS": ""}
        try:
            rc, err = _run(node, DELIVERY_GATE, payload=stop_ok, env=env)
        except subprocess.TimeoutExpired:
            return GuardProbeResult(
                name, "inconclusive", False, False, "guard 30s 超时"
            )
        in_force = rc == 2 and "AGENTS.md" in err

        bad_rc, bad_err = _run(
            node, DELIVERY_GATE, payload="{not json", env={"PYSCI_POD": str(pod)}
        )
        traced = bad_rc == 0 and "[delivery-gate fail-open]" in bad_err

    if in_force and traced:
        return GuardProbeResult(
            name, "ok", True, True, "violation exit=2 / bad-stdin exit=0 且留痕"
        )
    detail_bits: list[str] = []
    if not in_force:
        detail_bits.append(
            f"违规 AGENTS.md 未被拦（rc={rc}，stderr 头 80：{err[:80]!r}）"
        )
    if not traced:
        detail_bits.append(
            f"fail-open 未留痕（rc={bad_rc}，stderr 头 80：{bad_err[:80]!r}）"
        )
    return GuardProbeResult(
        name, "bad", in_force, traced, "；".join(detail_bits) or "未知"
    )


def probe_pod_guard() -> GuardProbeResult:
    """PreToolUse 侧：违规写只读层必被拦（exit 2）+ 坏 stdin 必留痕（[pod-guard fail-open]）。"""
    name = "pod-guard"
    node = _which_node()
    if node is None:
        return GuardProbeResult(name, "inconclusive", False, False, "node 不在 PATH")
    if not POD_GUARD.exists():
        return GuardProbeResult(
            name, "inconclusive", False, False, f"guard 文件缺失：{POD_GUARD}"
        )

    with tempfile.TemporaryDirectory(prefix="pysci-gp-pguard-") as td:
        pod = _mk_pod_fixture(Path(td), oversize_agents=False)
        charter = pod / ".qoder" / "rules" / "charter.md"
        payload = json.dumps({"tool_input": {"file_path": str(charter)}})
        env = {
            "PYSCI_POD": str(pod),
            "PYSCI_PROJECT_ROOT": str(td),
            "PYSCI_READONLY": ".qoder/rules/charter.md",
            "PYSCI_DEPLOYED_SKILLS": "",
            "PYSCI_TASK_DIRS": "",
        }
        try:
            rc, err = _run(node, POD_GUARD, payload=payload, env=env)
        except subprocess.TimeoutExpired:
            return GuardProbeResult(
                name, "inconclusive", False, False, "guard 30s 超时"
            )
        in_force = rc == 2 and "只读层" in err

        bad_rc, bad_err = _run(
            node, POD_GUARD, payload="{not json", env={"PYSCI_POD": str(pod)}
        )
        traced = bad_rc == 0 and "[pod-guard fail-open]" in bad_err

    if in_force and traced:
        return GuardProbeResult(
            name, "ok", True, True, "写只读层 exit=2 / bad-stdin exit=0 且留痕"
        )
    detail_bits = []
    if not in_force:
        detail_bits.append(f"写只读层未被拦（rc={rc}，stderr 头 80：{err[:80]!r}）")
    if not traced:
        detail_bits.append(
            f"fail-open 未留痕（rc={bad_rc}，stderr 头 80：{bad_err[:80]!r}）"
        )
    return GuardProbeResult(
        name, "bad", in_force, traced, "；".join(detail_bits) or "未知"
    )


def run_all() -> list[GuardProbeResult]:
    """双 guard 活体探针。返回顺序稳定（delivery-gate 先，pod-guard 后）。"""
    return [probe_delivery_gate(), probe_pod_guard()]


def report(results: list[GuardProbeResult]) -> list[str]:
    """doctor 用：逐 guard 一行 [√]/[✗]/[!] + 结论；空入参返回单行提示。"""
    lines: list[str] = []
    for r in results:
        if r.status == "ok":
            lines.append(f"  [√] {r.name}：{r.detail}")
        elif r.status == "bad":
            lines.append(
                f"  [✗] {r.name} 未执法或 fail-open 未留痕：{r.detail}"
                "（见 orchestration/skills/devops/references/budget-layers.md §5）"
            )
        else:
            lines.append(
                f"  [!] {r.name} 探针无法运行：{r.detail}（advisory，不计入 rc）"
            )
    return lines


def main(argv: list[str] | None = None) -> int:
    """`python -m …guard_probe` 独立入口；rc=0 全绿、rc=2 有 bad、rc=0 且仅 inconclusive。"""
    results = run_all()
    for ln in report(results):
        print(ln)
    return 2 if any(r.status == "bad" for r in results) else 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
