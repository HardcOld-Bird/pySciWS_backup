"""pysci-dev 统一 CLI 入口——devops 组员专属工具（与 pysci-orch 严格分离）。

子命令::

    uv run pysci-dev doctor [--pod <id>]     # pod 健康巡检
    ... dev sync [--check]                   # 技能真本→部署副本（复用 orchestration.sync 引擎）
    ... dev worktree add|remove|list [<name>]
    ... dev skilltax [--apply|--probe]       # 平台技能清单税：量、落关停配置、复测键义
    ... dev probe [--json|--no-ledger]       # agentsMdExcludes 生效性机械探针：因果 A/B、写台账

作业规程全文见 devops 技能 SKILL.md（部署于 devops pod）；设计依据 orchestration/README.md §7。
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import subprocess
import sys
from pathlib import Path

from pysci.paths import PODS_ROOT, PROJECT_ROOT

#: worktree 统一放置目录（与 Qoder 原生 --worktree 约定一致，避免两套位置）。
WORKTREES_ROOT = PROJECT_ROOT / ".qoder" / "worktrees"

#: 组长专属根规则（组员 pod 须经 settings.agentsMdExcludes 排除）与全员公约数（必须可见）。
LEADER_RULE = "leader-only.md"
BASIC_RULE = "basic.md"


def _run_git(*args: str) -> tuple[int, str]:
    """在项目根运行 git 命令并返回 (rc, 合并输出)。"""
    proc = subprocess.run(
        ["git", *args],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
    )
    return proc.returncode, (proc.stdout + proc.stderr).strip()


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------
def exclude_problems(content: str) -> list[str]:
    """校验 pod settings 的 ``agentsMdExcludes`` 是否恰好排除组长专属根规则。

    两侧都是护栏：缺排除 → 组员注入组长规程（越权面，见 leader-only.md 头注）；
    glob 过宽连 ``basic.md`` 一起命中 → 组员丢掉全员公约数（8KB 预算、上报纪律等）。
    按每条 pattern 的**文件名段**做 fnmatch，故 ``**/leader-only.md``、
    ``**/rules/leader-only.md`` 与裸 ``leader-only.md`` 等价判为已排除。
    """
    try:
        excludes = json.loads(content).get("agentsMdExcludes") or []
    except (json.JSONDecodeError, AttributeError):
        return [
            f"settings.json 无法解析 {LEADER_RULE} 排除项（非法 JSON 或缺顶层对象）"
        ]
    pats = [str(e).replace("\\", "/") for e in excludes]
    names = [Path(p).name for p in pats]
    if not any(fnmatch.fnmatch(LEADER_RULE, n) for n in names):
        return [f"leader 规则未排除（缺 agentsMdExcludes：'**/{LEADER_RULE}'）"]
    wide = next((p for p, n in zip(pats, names) if fnmatch.fnmatch(BASIC_RULE, n)), "")
    return (
        [f"agentsMdExcludes 过宽：'{wide}' 会连 {BASIC_RULE} 一并排除"] if wide else []
    )


def _doctor_pod(pod: Path) -> list[str]:
    """巡检单个 pod，返回问题/状态行列表。"""
    lines: list[str] = []
    name = pod.name
    if not pod.exists():
        return [f"  {name:<10} [缺失]"]
    problems: list[str] = []
    settings = pod / ".qoder" / "settings.json"
    if not settings.exists():
        problems.append("settings.json 缺失（hooks 未接线）")
    else:
        content = settings.read_text(encoding="utf-8")
        for hook in ("delivery-gate", "pod-guard"):
            if hook not in content:
                problems.append(f"{hook} 未接线")
        problems += exclude_problems(content)
    rules_dir = pod / ".qoder" / "rules"
    if not (rules_dir / "charter.md").exists():
        problems.append("charter.md 缺失")
    clash = sorted(p.name for p in rules_dir.glob(f"{Path(LEADER_RULE).stem}*"))
    if clash:
        problems.append(
            f"pod 自建规则 {clash[0]} 撞名组长规则（leader-only* 命名应避开排除 glob）"
        )
    agents_md = pod / "AGENTS.md"
    if not agents_md.exists():
        problems.append("AGENTS.md 缺失（记忆层未初始化）")
    elif (size := agents_md.stat().st_size) > 8 * 1024:
        problems.append(f"AGENTS.md 体量 {size // 1024}KB > 8KB（提请成员精简）")
    inbox_n = len(list((pod / "inbox").glob("*.md"))) if (pod / "inbox").exists() else 0
    # 部署漂移
    from pysci.skills.orchestration.tools import sync as _sync

    skills_dir = pod / ".qoder" / "skills"
    for spec in _sync.load_manifest():
        if any(Path(t).resolve() == skills_dir.resolve() for t in spec.targets):
            deployed = skills_dir / spec.name
            if not deployed.exists():
                problems.append(f"部署缺失：{spec.name}")
            elif _sync.tree_hash(deployed) != _sync.tree_hash(spec.source):
                problems.append(f"部署漂移：{spec.name}（跑 pysci-dev sync）")
    mark = "✗" if problems else "√"
    lines.append(
        f"  {name:<10} [{mark}] inbox={inbox_n}"
        + ("；".join(problems) if problems else "")
    )
    return lines


def cmd_doctor(args: argparse.Namespace) -> int:
    """pod 健康巡检：hooks 接线 / leader 规则排除 / charter / AGENTS.md 体量 / 部署漂移 /
    inbox 积压；外加「部署台账巡检」（联动 ``sync --check``，捕获 skills-deployed.json
    记录哈希相对真本的漂移（[*]）与副本漂移（[!]））与「harness 预算审计」（basic.md §1
    三档尺寸：真本与根 rules 每文件、pod 自维护层、技能 description 行合计）。"""
    print("== pod 巡检 ==")
    pods = (
        [PODS_ROOT / args.pod]
        if args.pod
        else sorted(p for p in PODS_ROOT.iterdir() if p.is_dir())
        if PODS_ROOT.exists()
        else []
    )
    bad = 0
    for pod in pods:
        for line in _doctor_pod(pod):
            print(line)
            if "[✗]" in line or "[缺失]" in line:
                bad += 1
    print()
    # 部署台账巡检（sync --check 联动）：记录哈希 vs live 真本（[*]）+ 副本漂移（[!]）
    print("== 部署台账巡检（sync --check）==")
    from pysci.skills.orchestration.tools import sync as _sync

    drift = [ln for ln in _sync.sync(check_only=True) if ln.startswith(("[!", "[*]"))]
    for ln in drift:
        print(ln)
    bad += len(drift)
    if not drift:
        print("  [√] 无漂移（记录哈希与真本/副本一致）")
    elif any(ln.startswith("[*]") for ln in drift):
        print(
            "  → 记录哈希过期：在 main 上跑 pysci-dev sync 重生成 skills-deployed.json"
        )
    print()
    # harness 预算三档审计（basic.md §1）：真本/根 rules/pod 自维护层尺寸 + description 合计
    print("== harness 预算审计（三档 ≤8192B）==")
    from pysci.skills.devops.tools import budget as _budget

    findings = _budget.audit()
    desc = _budget.description_findings()
    for ln in _budget.report(findings, desc):
        print(ln)
    bad += sum(1 for f in findings if f.over)
    bad += sum(1 for d in desc if d.over)
    print()
    # 平台注入税关停核查（backlog 20261010-plugin-tax-probe）：每 pod settings 须
    # 声明关掉跨 cwd 注入的平台技能，新 pod 漏配 = 每跳多交一份税。
    print("== 平台技能税关停核查 ==")
    from pysci.skills.devops.tools import skilltax as _tax

    gaps: list[str] = []
    for pod in pods:
        gap = _tax.disabled_gap(pod)
        if gap:
            gaps.append(pod.name)
            print(f"  [✗] {pod.name}：应关未关 {len(gap)} 项 ({_tax.short_names(gap)})")
    bad += len(gaps)
    if not gaps:
        print(
            f"  [√] 全部 pod 已声明关停 {len(_tax.recommended_disabled())} 项平台技能"
            f"（保留 {list(_tax.KEEP_ON_PODS)}）"
        )
    print()
    # agentsMdExcludes 生效性哨兵（backlog 20261010-165408-devops）：静态段只证明 settings
    # 写了排除 glob；此段读机械探针台账，证明 CLI 真据此把组长专属根规则挡在组员会话外。
    # 机制判假=真越权回归（硬失败）；未跑/CLI 或排除面变更=advisory（headless 昂贵，不硬失败）。
    print("== agentsMdExcludes 生效性哨兵 ==")
    from pysci.skills.devops.tools import probe as _probe

    level, msg = _probe.doctor_status()
    print(
        f"  [{'✗' if level == 'fail' else '!' if level == 'advisory' else '√'}] {msg}"
    )
    bad += 1 if level == "fail" else 0
    print()
    print(
        f"[{'√ 全部健康' if not bad else f'✗ {bad} 项问题（pod/部署台账/预算/平台税/排除哨兵）'}]"
    )
    return 0 if not bad else 2


# ---------------------------------------------------------------------------
# sync（复用 orchestration 引擎——库级复用，非 CLI 混用）
# ---------------------------------------------------------------------------
def cmd_sync(args: argparse.Namespace) -> int:
    """技能真本 → 部署副本同步。"""
    from pysci.skills.orchestration.tools import sync as _sync

    report = _sync.sync(check_only=args.check)
    print("\n".join(report))
    drift = [line for line in report if line.startswith("[!]")]
    return 2 if (drift and args.check) else 0


# ---------------------------------------------------------------------------
# worktree
# ---------------------------------------------------------------------------
def cmd_worktree(args: argparse.Namespace) -> int:
    """worktree 生命周期：add / remove / list（统一放 .qoder/worktrees/，分支 wt-<name>）。"""
    action = args.action
    if action == "list":
        rc, out = _run_git("worktree", "list")
        print(out)
        return rc
    if not args.name:
        print("[!] add/remove 需要 <name>")
        return 2
    wt_path = WORKTREES_ROOT / args.name
    if action == "add":
        if wt_path.exists():
            print(f"[!] 已存在：{wt_path}（复用即可）")
            return 2
        rc, out = _run_git("worktree", "add", "-b", f"wt-{args.name}", str(wt_path))
        print(out)
        if rc == 0:
            print()
            print("[NEXT] 测试策略（PYTHONPATH 覆盖，免建环境）：")
            print(
                f'  PYTHONPATH="{wt_path.as_posix()}/src" uv run --no-sync pytest '
                f'"{wt_path.as_posix()}/tests" -x -q'
            )
        return rc
    if action == "remove":
        rc, out = _run_git("worktree", "remove", str(wt_path))
        print(out or f"[√] worktree 已移除：{wt_path.name}")
        rc2, out2 = _run_git("branch", "-D", f"wt-{args.name}")
        if rc2 == 0:
            print(f"[√] 分支已删除：wt-{args.name}")
        else:
            print(f"[i] 分支删除跳过：{out2.splitlines()[-1] if out2 else ''}")
        return rc
    print(f"[!] 未知 action：{action}")
    return 2


# ---------------------------------------------------------------------------
# skilltax（平台注入税）
# ---------------------------------------------------------------------------
def cmd_skilltax(args: argparse.Namespace) -> int:
    """量各 pod 的平台技能清单税与关停缺口（``--probe`` 则实跑 headless 一跳复测键义）。"""
    from pysci.skills.devops.tools import skilltax as _tax

    argv: list[str] = []
    if args.pod:
        argv += ["--pod", args.pod]
    if args.apply:
        argv += ["--apply"]
    if args.probe:
        argv += ["--probe"]
    if args.overlay:
        argv += ["--overlay", args.overlay]
    return _tax.main(argv)


# ---------------------------------------------------------------------------
# probe（agentsMdExcludes 生效性哨兵）
# ---------------------------------------------------------------------------
def cmd_probe(args: argparse.Namespace) -> int:
    """跑 agentsMdExcludes 因果 A/B 探针、写台账、打印机制结论。"""
    from pysci.skills.devops.tools import probe as _probe

    argv: list[str] = []
    if args.json:
        argv += ["--json"]
    if args.no_ledger:
        argv += ["--no-ledger"]
    return _probe.main(argv)


# ---------------------------------------------------------------------------
# 门面
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    """CLI 入口。"""
    parser = argparse.ArgumentParser(
        prog="pysci-dev",
        description="pySci devops 组员工具（worktree/部署/巡检；设计见 orchestration/README.md §7）",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("doctor", help="pod 健康巡检")
    p.add_argument("--pod", help="只巡检指定成员")
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("sync", help="技能真本 → 部署副本同步")
    p.add_argument("--check", action="store_true")
    p.set_defaults(func=cmd_sync)

    p = sub.add_parser("worktree", help="worktree add/remove/list")
    p.add_argument("action", choices=["add", "remove", "list"])
    p.add_argument("name", nargs="?")
    p.set_defaults(func=cmd_worktree)

    p = sub.add_parser(
        "skilltax",
        help="量平台注入税（技能清单/agent 清单）与按 pod 关停缺口；--apply 落配置，"
        "--probe 实跑一跳复测键义",
    )
    p.add_argument("--pod", help="只处理指定成员")
    p.add_argument("--probe", action="store_true", help="造探针 pod 实跑 headless 一跳")
    p.add_argument("--overlay", help="探针 settings 叠加层（JSON 串）")
    p.add_argument(
        "--apply", action="store_true", help="把应关平台技能写进各 pod settings"
    )
    p.set_defaults(func=cmd_skilltax)

    p = sub.add_parser(
        "probe",
        help="agentsMdExcludes 生效性机械探针：Temp 因果 A/B（hide token 在 control 现、"
        "在 withexclude 消），结论写 git 跟踪台账；doctor 据此报警",
    )
    p.add_argument("--json", action="store_true", help="只打印 verdict JSON")
    p.add_argument(
        "--no-ledger",
        action="store_true",
        help="只复测不落台账（临时核查，不改动仓库）",
    )
    p.set_defaults(func=cmd_probe)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
