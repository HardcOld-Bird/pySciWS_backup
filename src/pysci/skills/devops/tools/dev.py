"""pysci-dev 统一 CLI 入口——devops 组员专属工具（与 pysci-orch 严格分离）。

子命令::

    uv run --no-sync pysci-dev doctor [--pod <id>]     # pod 健康巡检
    ... dev sync [--check]                   # 技能真本→部署副本（复用 orchestration.sync 引擎）
    ... dev venv-sync [uv sync 参数…]         # Windows 安全的 uv sync：让路→sync→核对入口
    ... dev worktree add|merge|remove|list [<name>]
    ... dev skilltax [--apply|--probe]       # 平台技能清单税：量、落关停配置、复测键义
    ... dev probe [--json|--no-ledger]       # agentsMdExcludes 生效性机械探针：因果 A/B、写台账

作业规程全文见 devops 技能 SKILL.md（部署于 devops pod）；设计依据 orchestration/README.md §7。
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from pysci.paths import PODS_ROOT, PROJECT_ROOT, WORKTREES_ROOT
from pysci.skills.orchestration.tools.registry import dirty_tracked_state

#: worktree 统一放置目录取自 ``pysci.paths.WORKTREES_ROOT``（与 Qoder 原生 --worktree
#: 约定一致，也与 drain 的孤儿回收/世代判定同一份事实——两处各写一遍必漂）。

#: 组长专属根规则（组员 pod 须经 settings.agentsMdExcludes 排除）与全员公约数（必须可见）。
LEADER_RULE = "leader-only.md"
BASIC_RULE = "basic.md"


def _git_proc(*args: str) -> subprocess.CompletedProcess[str]:
    """在项目根跑 git，返回原始 CompletedProcess（不裁剪输出）。"""
    return subprocess.run(
        ["git", *args],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
    )


def _run_git(*args: str) -> tuple[int, str]:
    """在项目根运行 git 命令并返回 (rc, 合并输出)。"""
    proc = _git_proc(*args)
    return proc.returncode, (proc.stdout + proc.stderr).strip()


def _owning_root(cwd: Path) -> Path | None:
    """cwd 及其祖先中最近的 ``pyproject.toml`` 所在目录（一个都没有则返回 None）。"""
    for cur in (cwd, *cwd.parents):
        if (cur / "pyproject.toml").exists():
            return cur
    return None


def _is_linked_worktree(root: Path) -> bool:
    """root/.git 是**文件**（而非目录）即 linked worktree；纯 stat，不起子进程。"""
    return (root / ".git").is_file()


def cross_tree_warning(cwd: Path | None = None) -> str:
    """脚下那棵树与代码面所指的项目根不是同一棵时返回告警文本，可信时返回空串。

    ``pysci`` 以 editable 装入，``pysci.paths.PROJECT_ROOT`` 由**包文件位置**决定、与
    cwd 无关：在 worktree 目录里裸跑 ``pysci-dev sync``（漏设 ``PYTHONPATH=<wt>/src``）
    时命令照旧退出 0、逐行输出「（一致）」，读写的却全是 main 的真本与 pod 副本——
    「看起来跑了但什么都没发生」；``doctor`` 更糟，它据 main 的状态给出「健康」的假结论。
    worktree 就摆在 ``PROJECT_ROOT/.qoder/worktrees/`` 里，所以「不在项目根之下」判不出
    跨树，判据是**脚下有没有自己的 pyproject.toml**：有且不是 PROJECT_ROOT 即另一棵树。

    但**标准 worktree 配方**（技能 §worktree 作业规程：cwd 停在主仓、代码面用
    ``PYTHONPATH=<wt>/src`` 覆盖）恰好也是 own ≠ PROJECT_ROOT，方向却相反——此时
    PROJECT_ROOT 位于脚下树之内、且自身是 linked worktree（``.git`` 是文件），正是本次
    要操作的那棵树，故放行。反向事故（脚下在 worktree、代码面指外层 main）与两树无隶属
    （真·跑错项目）仍照旧拦。

    Args:
        cwd: 待判定目录，默认 ``Path.cwd()``（测试注入用）。

    Returns:
        告警正文（多行）；同树或标准配方可信时为空串。
    """
    cur = (cwd if cwd is not None else Path.cwd()).resolve()
    root = PROJECT_ROOT.resolve()
    own = _owning_root(cur)
    if own is None or own == root:
        return ""
    if own in root.parents and _is_linked_worktree(root):
        return ""
    lines = [
        "[!] 跨树执行：脚下是另一棵树，而本进程的代码面/项目根指向别处——",
        f"    cwd 所属项目根 = {own}",
        f"    PROJECT_ROOT  = {root}（由包文件位置决定，与 cwd 无关）",
        "    不加拦则读写后者的真本/副本/台账、对前者什么都不做，且照样退出 0"
        "（doctor 会拿别处的状态给出「健康」的假结论）。",
        '    操作 worktree 请站在主仓外面带代码面覆盖：PYTHONPATH="<wt>/src" pysci-dev …，'
        "或确认自己确实要操作上面那个项目根后加 --force。",
    ]
    return "\n".join(lines)


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
    三档尺寸：真本与根 rules 每文件、pod 自维护层、技能 description 行合计）与
    「API Key 泄漏哨兵」（扫 git 跟踪文件找疑似密钥形态，README §14）。"""
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
    # guard 在法活体探针（backlog 20261010-193601-devops）：Temp fixture + node 直跑真 guard，
    # 双断言（违规被拦 + fail-open 留痕），把「fail-open 静默吞掉 guard 自伤」这一结构性坑
    # 固化到观测面。bad 计入 rc=2；inconclusive（node 缺失/guard 文件缺失）advisory 不硬失败。
    print("== guard 在法活体探针 ==")
    from pysci.skills.devops.tools import guard_probe as _gp

    gp_results = _gp.run_all()
    for ln in _gp.report(gp_results):
        print(ln)
    bad += sum(1 for r in gp_results if r.status == "bad")
    print()
    # 脏跟踪面哨兵（backlog 20261012-080410-devops 方案 C）：orchestration/state/** 里被
    # git 跟踪的文件有未提交改动时，碰同一文件的组员单会被「合并前 git status 检查」拦下
    # （只能交 <blocked>）。此段只观测不计 bad——脏多半是组长在飞的改动或待提交的正常
    # 积压；「无人操作却自己变脏」才是缺陷（自愈写运行态面，跟踪面只留人写的覆盖）。
    print("== 跟踪 state 脏面哨兵 ==")
    dirty = dirty_tracked_state()
    if dirty:
        for p in dirty:
            print(f"  [!] {p}")
        print("  → 派发改 registry/backlog/deliveries 的单前，先提交或还原这些；")
        print(
            "    若它们不是你或组长改的（运行时自愈改脏跟踪面），那是缺陷，提请 devops。"
        )
    else:
        print("  [√] orchestration/state/** 跟踪面无未提交改动")
    print()
    # API Key 泄漏哨兵（README §14、E3-prep）：Key 只能由用户本人提供且永不入库，
    # 而「永不入库」一旦只是纪律就会随某次提交进 git 历史。扫 git 跟踪文件找疑似密钥
    # 形态（tests/ 豁免：假 Key 是阳性用例素材），命中即硬失败；git 不可用只 advisory。
    print("== API Key 泄漏哨兵 ==")
    from pysci.skills.devops.tools import keyscan as _ks

    findings, ks_error = _ks.scan()
    for ln in _ks.report(findings, ks_error):
        print(ln)
    bad += len(findings)
    print()
    print(
        f"[{'√ 全部健康' if not bad else f'✗ {bad} 项问题（pod/部署台账/预算/平台税/排除哨兵/在法探针/Key 泄漏）'}]"
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
# venv-sync（uv sync 的 Windows 安全配方，backlog 20261012-114239-devops）
# ---------------------------------------------------------------------------
def project_scripts(pyproject: Path) -> list[str]:
    """读 ``[project.scripts]`` 的命令名（排序）；读不到/解析不了返回空表。"""
    import tomllib

    try:
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    scripts = (data.get("project") or {}).get("scripts") or {}
    return sorted(scripts)


def _move_aside_locked(exe: Path, ts: str) -> str:
    """把入口 exe 改名让路，同时判明它是否被**运行中进程**占住。

    Windows 语义（2026-10-12 实测）：运行中的镜像以 FILE_SHARE_DELETE 打开 ⇒ **改名能成、
    删除失败**；普通打开句柄则两者都失败。``uv sync`` 要删这些 exe，所以只有前者是
    「改名让开、sync 就能原地重建」的可救形态——后者只能报给组长处置。

    删除就是判据本身，故未被占用的文件会被真删掉：先取字节快照、探测后原样写回
    （代价仅 mtime 变新，而 uv sync 本来就会重写这些入口）。

    Returns:
        ``"moved"``：被占住，已改名为 ``<name>.locked-<ts>``，原路径让开；
        ``""``：未被占用（改名+删除都成），已按原字节还原到原名；
        ``"stuck"``：连改名都被拒（或读不出内容），文件原样未动。
    """
    try:
        payload = exe.read_bytes()
    except OSError:
        return "stuck"
    scratch = exe.with_name(f"{exe.name}.locked-{ts}")
    try:
        os.replace(exe, scratch)
    except OSError:
        return "stuck"
    try:
        os.remove(scratch)
    except OSError:
        return "moved"
    exe.write_bytes(payload)
    return ""


def _locked_holder(exe: Path) -> str:
    """best-effort 找出占住该 exe 的进程，找不到返回空串。

    进程镜像路径在创建时定格，**改名后 wmic 仍显示原名**，故按原名匹配。参数经 argv
    直传（不经 shell），免疫 MSYS 把 ``/format:csv`` 改写成路径。
    """
    try:
        proc = subprocess.run(
            ["wmic", "process", "get", "ProcessId,ExecutablePath", "/format:csv"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    if proc.returncode != 0:
        return ""
    tail = f"\\{exe.name.lower()}"
    for line in proc.stdout.splitlines():
        parts = [p.strip() for p in line.strip().split(",")]
        if len(parts) < 3:
            continue
        node, path, pid = parts[0], parts[-2], parts[-1]
        if path.lower().endswith(tail):
            return f"占用者={node} PID {pid}（{path}）"
    return ""


def run_uv_sync(extra: list[str]) -> tuple[int, str]:
    """在项目根跑 ``uv sync``，返回 (rc, 合并输出)。测试替身挂这一层。"""
    proc = subprocess.run(
        ["uv", "sync", *extra],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=1800,
    )
    return proc.returncode, (proc.stdout + proc.stderr).strip()


def _aside_name(exe: Path, ts: str) -> Path:
    """让路文件的名字（与 `_move_aside_locked` 保持一致）。"""
    return exe.with_name(f"{exe.name}.locked-{ts}")


def _recycle_aside_residue(scripts_dir: Path) -> tuple[int, int]:
    """回收往次让路留下的 ``<name>.exe.locked-<ts>``，返回 (删掉的副本数, 还原的原件数)。

    判据必须先查原名：原名**在盘** ⇒ 残留只是多余副本，删；原名**缺失** ⇒ 残留是唯一的
    原件（上次让路后 uv 判定无事可做、没重建它），必须改回原名。**反过来做就会吞掉唯一的
    入口文件**（2026-10-12 main 实测踩过：无 pyproject 变更的一跑把 pysci-dev.exe 弄没了）。
    还被占用进程捏着的删不动，静默留给下一次。
    """
    deleted = restored = 0
    for stale in sorted(scripts_dir.glob("*.exe.locked-*")):
        orig = stale.with_name(stale.name.split(".locked-")[0])
        try:
            if orig.exists():
                stale.unlink()
                deleted += 1
            else:
                os.replace(stale, orig)
                restored += 1
        except OSError:
            continue
    return deleted, restored


def cmd_venv_sync(args: argparse.Namespace) -> int:
    """Windows 安全的 ``uv sync``：先让被占住的入口 exe 改名让路，sync 后逐个核对入口。

    动机：``uv sync`` 遇运行中 exe（典型=组长挂着的 ``pysci-orch.exe watch``）会以
    「拒绝访问 (os error 5)」失败，而**失败前已把同目录其余入口删光**——裸命令集体
    command not found 的半坏 venv，容易被误诊成 PATH/安装问题，并让在跑的 drain 会话
    当场失去工具面。本命令把「回收残留 → 改名让路 → sync → 让回原地 → 核对入口」固化成一条。

    Returns:
        0 成功且入口齐备；2 入口仍缺失（半坏态未消除，须上报）；否则透传 uv 的 rc。
    """
    pyproject = PROJECT_ROOT / "pyproject.toml"
    scripts_dir = PROJECT_ROOT / ".venv" / "Scripts"
    names = project_scripts(pyproject)
    if not names:
        print(f"[!] 读不到 [project.scripts]（{pyproject}）——无法核对入口")
        return 2
    if not scripts_dir.exists():
        print(f"[!] 无 {scripts_dir}（非 Windows 布局或尚未 uv sync）")
        return 2

    ts = datetime.now().astimezone().strftime("%Y%m%d%H%M%S")
    deleted, revived = _recycle_aside_residue(scripts_dir)
    if deleted or revived:
        print(f"[i] 回收往次让路：删副本 {deleted} 个，还原原件 {revived} 个")
    moved: list[Path] = []
    if os.name == "nt":
        print("== 让路：探测声明入口是否被运行中进程占住 ==")
        for name in names:
            exe = scripts_dir / f"{name}.exe"
            if not exe.exists():
                continue
            state = _move_aside_locked(exe, ts)
            if state == "moved":
                moved.append(exe)
                holder = _locked_holder(exe)
                print(
                    f"  [√] {exe.name} 被占住 → 已改名让路"
                    f"（{exe.name}.locked-{ts}）" + (f"；{holder}" if holder else "")
                )
            elif state == "stuck":
                print(
                    f"  [!] {exe.name} 连改名都被拒（被非镜像句柄占用），原样未动——"
                    "它仍会挡住 uv sync，请组长关掉占用程序（勿擅 kill 组长进程）"
                )
    else:
        print("[i] 非 Windows：无 exe 锁形态，直接 uv sync")

    print("== uv sync ==")
    rc, out = run_uv_sync(list(args.uv_args or []))
    print(out)

    put_back: list[str] = []
    for exe in moved:
        scratch = _aside_name(exe, ts)
        if not exe.exists() and scratch.exists():
            # uv 判定无事可做就不会重建入口——让路的原件必须自己改回去，否则让路成了肇事者
            os.replace(scratch, exe)
            put_back.append(exe.name)
    if put_back:
        print(f"  [√] uv 未重建 {put_back} → 已把让路原件改回原名")

    missing = [n for n in names if not (scripts_dir / f"{n}.exe").exists()]
    if missing:
        print(
            f"[✗] 入口缺失 {missing}（uv rc={rc}）——半坏态未消除；"
            "把上面报错原文与缺失清单一并报组长"
        )
        return 2
    for exe in moved:
        scratch = _aside_name(exe, ts)
        if not scratch.exists():
            continue  # 已被上面「让回原地」用掉了，不是待清残留
        try:
            scratch.unlink()
        except OSError:
            print(f"  [i] 残留 {scratch.name} 仍被占用（占用进程退出后可删）")
    print(f"[√] 入口齐备（[project.scripts] {len(names)} 个全在盘）")
    return rc


# ---------------------------------------------------------------------------
# worktree
# ---------------------------------------------------------------------------
#: 跨合并必搬的 gitignored 运行态清单（backlog 20261010-worktree-merge-cp）。
#: --no-ff 只带 git 面，这些文件不随迁；registry-runtime-split 那次靠 devops 人工 cp
#: 才保住 5 成员 6 会话。此清单是**保底**——真实搬运集 = 本清单 ∪ git 侧 ignored 发现，
#: 再挖掉 :data:`MERGE_STATE_EXCLUDE_SUFFIXES`。
MERGE_STATE_FILES = ("sessions-runtime.json", "registry.presplit.json", "watch.json")

#: **永不跨树搬**的运行态后缀。``*.lock``（devops/inspector/bencher）不是数据而是
#: 「谁在跑」的活体读数，其归属判据 = 文件里的 pid + run_log 双锚；把 worktree 侧的锁
#: cp 进 main，main 就认到一把 pid 与 run_log **都不属于它**的锁——``orch status`` 据此
#: 误报「worker 在跑」，而真在跑的 drain 反而被这把假锁挡住不唤醒（backlog
#: 20261012-191512-devops 实测：两次合并各拖进一把 inspector.lock，人工 rm 才解封）。
#: 锁的现场性由判活面负责（drain ``lock_status`` 认墓碑/残骸），合并面一律不搬。
MERGE_STATE_EXCLUDE_SUFFIXES = (".lock",)

#: **备份文件同样永不跨树**：cp 阶段会在目标侧留 ``<name>.bak-<ts>``，它是**本地面**的
#: 历史副本，搬过去只在对岸造垃圾——而它又恰好是下一次合并的「交集」候选，与「交集类
#: 停链/交付形状」直接自相矛盾（backlog 20261013-143233-devops 实测：两次合并各留一个
#: ``*.bak-<ts>``，且历史上有 6 个已被误提交进仓库）。带时间戳的名字 match 不上固定后缀，
#: 故用**中缀**判。
MERGE_STATE_EXCLUDE_INFIX = ".bak-"


def _state_dir(root: Path) -> Path:
    """某仓根（main 或 worktree）下的 orchestration/state 目录。"""
    return root / "orchestration" / "state"


def _ignored_state_names(wt: Path) -> set[str]:
    """向 git 问 worktree 的 orchestration/state 顶层哪些是 ignored（分支侧 .gitignore 规则）。"""
    sd = _state_dir(wt)
    if not sd.exists():
        return set()
    candidates = [p.name for p in sd.iterdir() if p.is_file()]
    if not candidates:
        return set()
    payload = "\n".join(f"orchestration/state/{n}" for n in candidates)
    try:
        # 用 bytes 通道：Windows subprocess text=True 会把 input 里的 \n 译成 \r\n，
        # 而 git check-ignore --stdin 只识别裸 LF（实测 \r 让整批查询静默返空）。
        proc = subprocess.run(
            ["git", "-C", str(wt), "check-ignore", "--stdin"],
            input=payload.encode("utf-8"),
            capture_output=True,
            timeout=30,
        )
    except (subprocess.TimeoutExpired, OSError):
        return set()
    stdout = proc.stdout.decode("utf-8", errors="replace")
    return {Path(line.strip()).name for line in stdout.splitlines() if line.strip()}


def _cp_ignored_state(wt: Path, main: Path, ts: str) -> list[tuple[str, str]]:
    """纯 cp：worktree→main 同名；main 侧存在的先字节级备份 ``<name>.bak-<ts>``。

    Returns:
        ``[(name, tag)]``——tag=``new`` 表目标原本不存在，否则 ``bak=<备份名>``。
        fail-open：worktree 侧文件缺失/读不动跳过本条不抛。
    """
    names = {
        n
        for n in set(MERGE_STATE_FILES) | _ignored_state_names(wt)
        if not n.endswith(MERGE_STATE_EXCLUDE_SUFFIXES)
        and MERGE_STATE_EXCLUDE_INFIX not in n
    }
    sd_wt = _state_dir(wt)
    sd_main = _state_dir(main)
    sd_main.mkdir(parents=True, exist_ok=True)
    log: list[tuple[str, str]] = []
    for name in sorted(names):
        src = sd_wt / name
        if not src.is_file():
            continue
        try:
            data = src.read_bytes()
        except OSError:
            continue
        dst = sd_main / name
        tag = "new"
        if dst.exists():
            bak = dst.with_name(f"{name}.bak-{ts}")
            try:
                bak.write_bytes(dst.read_bytes())
                tag = f"bak={bak.name}"
            except OSError as exc:
                print(f"[!] 备份 {dst.name} 失败（{exc}），仍会覆盖")
        dst.write_bytes(data)
        log.append((name, tag))
    return log


def _seed_runtime_from_main(wt: Path) -> str:
    """把 main 的运行态 ``sessions-runtime.json`` 播种进新 worktree（add 收尾自动做）。

    **为什么必须播种**（backlog 20261013-131505-devops 实测）：``git worktree add`` 只给出
    main 的**跟踪面**，运行态是 gitignored 的——于是新 worktree 的 ``sessions-runtime.json``
    是空壳（1 member / 0 sessions），而 merge 的「会话数不减」自检（base=45）必判它「比
    main 少」而中止。实测该中止**每次都要人工 ``cp`` 一次**才能过闸，且报错文案把成因说成
    「worktree 是否 stale（应从最新 main 起 add）」——而当事人恰恰就是从最新 main 起的，
    指引与真因不符。

    只在 worktree 侧**确实更稀疏**时覆盖（等价于 merge 那条闸的判据），故重复调用无副作用、
    也不会把已推进的 worktree 运行态打回去。

    Returns:
        一行读数（未播种时说明原因），供 add 的输出面打印。
    """
    src = _state_dir(PROJECT_ROOT) / "sessions-runtime.json"
    if not src.exists():
        return "[i] main 无运行态可播种（sessions-runtime.json 缺席）——跳过"
    main_n = _runtime_session_total(PROJECT_ROOT)
    wt_n = _runtime_session_total(wt)
    if main_n is None:
        return "[i] main 运行态读不出会话数——跳过播种"
    if wt_n is not None and wt_n >= main_n:
        return f"[=] worktree 运行态已不稀疏（{wt_n} ≥ main {main_n}）——跳过播种"
    dst = _state_dir(wt) / "sessions-runtime.json"
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(src.read_bytes())
    except OSError as exc:
        # fail-open：播种失败不该让 add 失败——merge 侧那条闸仍会出声，只是又要人工 cp
        return f"[!] 运行态播种失败（{exc}）——merge 前请人工 cp sessions-runtime.json"
    return f"[√] 已播种运行态（会话数 {main_n}）→ 新 worktree 的首次 merge 无需人工 cp"


def _runtime_session_total(root: Path) -> int | None:
    """读 root 侧 sessions-runtime.json 的会话总数。

    直接用运行态文件而非 Registry.load：后者会触发 cutover 或警告副作用；自检要**无副作用**
    读数，且合并前后都拿同一把尺子量。

    Returns:
        会话总数；**运行态文件不存在返回 None**（区别于「存在但 0 条」——前者不该被当作
        「比 main 少」而误挡合并：一个从未派发/加载过的 worktree 本就没有运行态）。
        文件损坏按 0 计（存在但读不出）。
    """
    rt = _state_dir(root) / "sessions-runtime.json"
    if not rt.exists():
        return None
    try:
        data = json.loads(rt.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0
    total = 0
    for entry in data.get("members", {}).values():
        sessions = entry.get("sessions", []) if isinstance(entry, dict) else []
        total += len(sessions)
    return total


def porcelain_paths(out: str) -> set[str]:
    """``git status --porcelain`` 输出 → 工作区未提交面（含未跟踪；目录以 ``/`` 结尾）。

    重命名/复制取**新**路径（合并会覆盖的是新路径上的工作）。行格式是定长的
    ``XY<空格>path``，故非该形态的行（别的命令混进来的输出）一律跳过——宁漏不误，
    误报会把「无关脏项」也算成冲突而挡掉本该放行的合并。
    """
    paths: set[str] = set()
    for line in out.splitlines():
        if len(line) < 4 or line[2] != " ":
            continue
        xy, rest = line[:2], line[3:]
        if xy[0] in "RC" and " -> " in rest:
            rest = rest.split(" -> ", 1)[1]
        paths.add(rest.strip('"'))
    return paths


def committed_face(branch: str) -> set[str] | None:
    """本次合并会触碰的文件面 = **merge-base 起算**的分支提交面；算不出返回 None。

    判据必须是 ``merge-base HEAD <branch>..branch``，不是 ``main..HEAD``：分支落后 main 时
    两点 diff 列出的是**别人新提交**的文件（实测两次假阳性，其中一次据此交出了错误的
    <blocked>）。
    """
    rc, base = _run_git("merge-base", "HEAD", branch)
    if rc != 0 or not base.strip():
        return None
    rc, out = _run_git(
        "-c",
        "core.quotepath=false",
        "diff",
        "--name-only",
        f"{base.strip()}..{branch}",
    )
    if rc != 0:
        return None
    return {line.strip() for line in out.splitlines() if line.strip()}


def intersect_faces(committed: set[str], dirty: set[str]) -> list[str]:
    """提交面 ∩ 未提交面；未跟踪**目录**脏项命中其下的提交文件（前缀匹配）。"""
    hits: list[str] = []
    for f in sorted(committed):
        if f in dirty or any(
            d.endswith("/") and (f + "/").startswith(d) for d in dirty
        ):
            hits.append(f)
    return hits


def dirty_face() -> set[str]:
    """main 工作区未提交面（含未跟踪）。

    走 ``_git_proc`` 而非 ``_run_git``：后者把输出 ``strip()`` 了，会吃掉 porcelain **首行的
    前导空格**（``" M a.txt"`` → ``"M a.txt"``），解析器便把首项判成非 status 行而漏掉。
    """
    proc = _git_proc("-c", "core.quotepath=false", "status", "--porcelain")
    return porcelain_paths(proc.stdout)


def merge_faces(branch: str) -> tuple[set[str], set[str]] | None:
    """``(本次提交面, main 未提交面)``；任一算不出返回 None（调用方据此跳过检查）。"""
    try:
        committed = committed_face(branch)
        if committed is None:
            return None
        return committed, dirty_face()
    except (OSError, subprocess.SubprocessError):
        return None


def cmd_worktree_merge(name: str, wt: Path) -> int:
    """worktree merge 主体：交集检查 → cp gitignored 运行态 → 会话数校验 → merge --no-ff → 自检。

    Returns:
        0 成功；2 硬阻塞（与 main 未提交面有交集 / worktree 缺失 / 会话数会倒退）；
        merge 自身失败则透传其 rc。
    """
    branch = f"wt-{name}"
    ts = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S")

    print("== 合并前交集检查（本次提交面 ∩ main 未提交面）==")
    faces = merge_faces(branch)
    if faces is None:
        # 求交失败不该挡合并：那是把 usability 缺陷换成另一种误报。git merge 自身仍会
        # 拒绝覆盖未提交工作，本检查的价值在于**先于 cp** 给出可读清单。
        print("  [!] 求交失败（git 不可用 / merge-base·diff 非零）——跳过检查，继续合并")
    else:
        committed, dirty = faces
        offenders = intersect_faces(committed, dirty)
        print(
            f"  本次提交面 {len(committed)} 文件（merge-base 起算）"
            f" / main 未提交 {len(dirty)} 项"
        )
        if offenders:
            print(
                f"  [✗] 交集 {len(offenders)} 项——合并会覆盖 main 工作区的未提交改动，"
                "中止（未 cp、未 merge）："
            )
            for f in offenders:
                print(f"      {f}")
            print(
                "  处置：组长先提交或还原上述文件后重跑；与本单无关的脏项不会走到这里"
            )
            return 2
        print("  [√] 交集为空（无关脏项不拦）")

    base_n = _runtime_session_total(PROJECT_ROOT) or 0
    wt_n = _runtime_session_total(wt)
    # 仅当 worktree 确实带来一个更稀疏的运行态（会覆盖 main 的更丰富数据）才中止；
    # worktree 无运行态（从未派发/加载）时 cp 对 sessions-runtime.json 是 no-op，不该误挡。
    if wt_n is not None and wt_n < base_n:
        print(
            f"[!] worktree 侧会话数 {wt_n} < main baseline {base_n}——cp 会覆盖更丰富的 main "
            f"运行态、丢会话，中止。**成因＝worktree 的运行态比 main 稀疏**，处置："
            f"① 该面若来自 `pysci-dev worktree add` 之后，add 会自动播种，走到这里说明"
            f"worktree 侧运行态后来被清空/回退过——查它在盘内容；"
            f"② 或 main 在 add 之后新增了大量会话——以 main 为准，直接合并（本闸要防的是"
            f"反向：拿稀疏面盖掉 richer 面）；"
            f"③ 手动 `cp orchestration/state/sessions-runtime.json <wt>/orchestration/state/` 后重跑"
        )
        return 2

    log = _cp_ignored_state(wt, PROJECT_ROOT, ts)
    print("== cp gitignored 运行态（worktree → main）==")
    if log:
        for nm, tag in log:
            print(f"  [√] orchestration/state/{nm}（{tag}）")
    else:
        print("  （无候选文件需搬）")

    print(f"== git merge --no-ff {branch} ==")
    rc, out = _run_git("merge", "--no-ff", branch)
    print(out)
    if rc != 0:
        print(
            f"[!] merge 失败 rc={rc}——cp 已落；main 侧冲突请人工处理，运行态文件不受影响"
        )
        return rc

    after_n = _runtime_session_total(PROJECT_ROOT) or 0
    if after_n < base_n:
        print(
            f"[✗] 合并后会话数倒退：base={base_n} after={after_n}——请核对 sessions-runtime.json"
        )
        return 2
    print(f"[√] 会话数不减（base={base_n} → after={after_n}）")
    return 0


def _worktree_exists(*, name: str, wt_path: Path) -> None:
    """同名 worktree 目录已存在时的**世代守卫**断言（backlog 20261012-184000-leader）。

    为什么不再只说一句「复用即可」：孤儿回收走的是 ``git worktree remove --force``，
    而 Windows 上只要旧 headless 会话还开着目录里的句柄，remove 完全可能 rc=0 而目录
    依旧趴在原地——那是**残骸**，不是「你的隔离面还在」。把它当成可复用的面，就是请第二
    个写者进同一个房间（2026-10-12 实测：两个会话并发写同一 worktree，一边整跳白跑）。
    故此处区分两种事实，并把「换新一代名字」这条路明确指出来。
    """
    rc, out = _run_git("worktree", "list")
    registered = rc == 0 and any(
        line.split(maxsplit=1)[0].strip('"').lower() == wt_path.as_posix().lower()
        for line in out.splitlines()
        if line.strip()
    )
    if registered:
        print(f"[!] 已存在且登记在册的 worktree：{wt_path}")
        print("  同一隔离面继续用即可（add 不重复建，rc=2 只是把话说明白）")
        return
    print(f"[!] 目录存在但**未登记为 worktree**（残骸，非你的面）：{wt_path}")
    print(
        "  常见成因：孤儿回收 remove --force 后旧会话句柄仍占用，目录没真被摘掉——"
        "同名重建 = 两个写者并发写同一目录"
    )
    print(
        f"  处置：先 `git worktree prune` 复验；若目录还在，改用新一代名 "
        f"`pysci-dev worktree add {name}-r2`（drain 的异名重建就是这么做的），"
        "旧目录留给组长处置，别在这里 rm -rf"
    )


def cmd_worktree(args: argparse.Namespace) -> int:
    """worktree 生命周期：add / merge / remove / list（统一 .qoder/worktrees/，分支 wt-<name>）。"""
    action = args.action
    if action == "list":
        rc, out = _run_git("worktree", "list")
        print(out)
        return rc
    if not args.name:
        print(f"[!] {action} 需要 <name>")
        return 2
    wt_path = WORKTREES_ROOT / args.name
    if action == "add":
        if wt_path.exists():
            _worktree_exists(name=args.name, wt_path=wt_path)
            return 2
        rc, out = _run_git("worktree", "add", "-b", f"wt-{args.name}", str(wt_path))
        print(out)
        if rc == 0:
            # add 收尾播种运行态：否则首个 merge 必撞「会话数 0 < baseline」闸（backlog
            # 20261013-131505-devops）——gitignore 的运行态不会随 worktree add 过来。
            print(_seed_runtime_from_main(wt_path))
            print()
            print(
                "[NEXT] 测试策略（PYTHONPATH 覆盖，免建环境；**cwd 须在 worktree 之外**）："
            )
            print(
                f'  PYTHONPATH="{wt_path.as_posix()}/src" uv run --no-sync pytest '
                f'"{wt_path.as_posix()}/tests" -x -q'
            )
            print(
                "  已 cd 进 worktree 时别用 uv（会就地建空 .venv，缺依赖即崩），"
                "改用主解释器："
            )
            print(
                f'  PYTHONUTF8=1 PYTHONPATH="{wt_path.as_posix()}/src" '
                f'"{PROJECT_ROOT.as_posix()}/.venv/Scripts/python.exe" '
                f'-m pytest "{wt_path.as_posix()}/tests" -q'
            )
        return rc
    if action == "merge":
        if not wt_path.exists():
            print(f"[!] worktree 不存在：{wt_path}")
            return 2
        return cmd_worktree_merge(args.name, wt_path)
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
# mdgen-check（钩子终态治理，backlog 20261010-191922-devops）
# ---------------------------------------------------------------------------
def cmd_mdgen_check(args: argparse.Namespace) -> int:
    """把 pre-commit 对文件的改写提前做完；报 pre/post sha256（供内容等价性以终态为准）。"""
    from pysci.skills.devops.tools import mdgen as _mdgen

    argv: list[str] = []
    if args.dry_run:
        argv += ["--dry-run"]
    argv += list(args.paths)
    return _mdgen.main(argv)


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

    p = sub.add_parser(
        "venv-sync",
        help="Windows 安全的 uv sync：先把被运行中进程占住的入口 exe 改名让路，"
        "sync 后按 [project.scripts] 核对入口一个不缺（防半坏 venv）",
    )
    p.add_argument(
        "uv_args", nargs="*", help="转发给 uv sync 的额外参数（如 --offline）"
    )
    p.set_defaults(func=cmd_venv_sync)

    p = sub.add_parser(
        "worktree",
        help="worktree add/merge/remove/list（merge=机械化跨合并搬运行态 + --no-ff + 会话数自检）",
    )
    p.add_argument("action", choices=["add", "merge", "remove", "list"])
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

    p = sub.add_parser(
        "mdgen-check",
        help="脚本批量产出/改写 .md 后，把 pre-commit（trailing-whitespace + "
        "end-of-file-fixer 等）的改写提前做完；逐文件打印 pre/post sha256，"
        "内容等价性断言以 post 为准（首提必过、无验证过期窗口）",
    )
    p.add_argument(
        "paths",
        nargs="+",
        help="待治理文件路径（不存在的静默跳过，与生成脚本常带模板路径的行为一致）",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="只报告钩子会改写哪些文件、跑完还原字节（预览用，不改仓库）",
    )
    p.set_defaults(func=cmd_mdgen_check)

    # 每个子命令都吃 --force：跨树自检是全局护栏，不该有子命令例外。
    for sp in sub.choices.values():
        sp.add_argument(
            "--force",
            action="store_true",
            help="cwd 与项目根不同树时仍然执行（默认拒绝并 rc=2）",
        )

    args = parser.parse_args(argv)
    warn = cross_tree_warning()
    if warn:
        print(warn)
        if not getattr(args, "force", False):
            print("    已拒绝执行（rc=2）。修好代码面再来，或加 --force 明示知情。")
            return 2
        print("    [--force] 明知跨树，继续执行。")
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
