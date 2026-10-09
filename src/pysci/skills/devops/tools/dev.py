"""pysci-dev 统一 CLI 入口——devops 组员专属工具（与 pysci-orch 严格分离）。

子命令::

    uv run pysci-dev doctor [--pod <id>]     # pod 健康巡检
    ... dev sync [--check]                   # 技能真本→部署副本（复用 orchestration.sync 引擎）
    ... dev worktree add|remove|list [<name>]

作业规程全文见 devops 技能 SKILL.md（部署于 devops pod）；设计依据 orchestration/README.md §7。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from pysci.paths import PODS_ROOT, PROJECT_ROOT

#: worktree 统一放置目录（与 Qoder 原生 --worktree 约定一致，避免两套位置）。
WORKTREES_ROOT = PROJECT_ROOT / ".qoder" / "worktrees"


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
    if not (pod / ".qoder" / "rules" / "charter.md").exists():
        problems.append("charter.md 缺失")
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
    """pod 健康巡检：hooks 接线 / charter / AGENTS.md 体量 / 部署漂移 / inbox 积压。"""
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
    print(f"[{'√ 全部健康' if not bad else f'✗ {bad} 个 pod 有问题'}]")
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

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
