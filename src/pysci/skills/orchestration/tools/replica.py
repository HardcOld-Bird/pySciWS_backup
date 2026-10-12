"""基线捕获与副本 A/B（REFACTOR v1.0 §3「副本 A/B 与基线捕获」；E4）。

**要解决的问题**：E2 的 ``orch eval run`` 只能测「现在这棵树的 harness」。要判「这次改动
有没有让 harness 变好」，必须让**同一个案卷**在改动前/改动后各跑一次——而 main 只有一个
HEAD，工作区还在被别的单推着走。§3 的解法是 git worktree 副本：两个 ref 各materialize
一份树，各跑各的案，最后 :func:`eval.compare` 出二元判决。

**副本怎么「变成另一棵树」**（§13 待决 2 的 E4 实测裁定，写进技能 references）：
子进程带 ``PYTHONPATH=<replica>/src``，于是 ``import pysci`` 命中副本源码，而
:func:`pysci.paths._find_project_root` 从 ``__file__`` 就近上溯 ——
``pysci.paths.PROJECT_ROOT`` 就落在**副本根**。这一条把 state/、pods/、guards/、技能真本
全部重锚到副本，派发链无需任何「副本模式」参数。代价与对策：

- 副本没有 ``.venv``（gitignored）→ 组员裸 ``pysci-X`` 会失效。对策：父进程注入
  :data:`runner` 侧新旋钮 ``PYSCI_VENV_SCRIPTS``（指向主树 venv），而 PYTHONPATH 被
  headless 进程继承，于是**脚本来自主 venv、代码来自副本**——正是 A/B 要的形状。
- 副本缺 pods 的 gitignored 运行目录（``bench/ outbox/ inbox/ .qoder/skills/``）→
  :func:`_prep_replica` 就地重建并跑技能 sync（副本自己的真本 → 副本 pods）。
- 副本缺 ``.env``（凭据 gitignored）→ 从主树复制（副本用后即弃，不外流）。
- 副本缺「git 保不住的组员面」（自建 rules/skills、部署副本）→ 由
  :func:`capture_baseline` 的归档在 :func:`build_replica` 时还原；还原发生在 sync
  **之前**，好让副本真本覆盖部署副本（A/B 测的就是两个 ref 的真本差异）。

**基线捕获为什么不建 worktree**（§3 写的是「git worktree @ 审批时 HEAD」，此处按 §3
末句「实现细节 E4 实测商榷」定案）：tracked 面 git 本来就保得住，一个常驻 detached
worktree 只是把同一份内容多占一份盘，还会随 refs 移动而失真；真正**不可再生**的是
pods 里 gitignored 的组员面，所以捕获 = **commit 记录 + gitignored 归档**，worktree
交给 :func:`run_ab` 按需建、用完弃（判据 4「零残留」因此可满足）。
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from pysci.paths import ORCH_STATE_ROOT, PROJECT_ROOT

from . import eval as _eval
from .registry import orchestration_rel

#: 基线区（§3：bencher 完全写权）。``manifest.json`` 入库，物理归档不入库。
BASELINES_DIR: Path = ORCH_STATE_ROOT / "evals" / "baselines"
#: A/B 判决落盘区（择机查看面的结构化证据）。
AB_DIR: Path = ORCH_STATE_ROOT / "evals" / "ab"
#: 副本工作树根（与 devops worktree 同一落点，孤儿回收/清理纪律共用一套认知）。
WORKTREES_ROOT: Path = PROJECT_ROOT / ".qoder" / "worktrees"
#: 副本名前缀（``git worktree list`` 里一眼可辨，也是清理闸的识别面）。
REPLICA_PREFIX: str = "bench"

#: pod 内**不归档**的目录：运行期产物与草稿区（A/B 要求每轮全新，残留=复核而非生产）。
ARCHIVE_SKIP: tuple[str, ...] = ("bench", "outbox", "inbox")

#: 判「窗口期污染」看的知识面——只算会影响测量的那部分（§5 二元判决的归因面）。
HARNESS_FACE: tuple[str, ...] = (
    "src/pysci/skills/orchestration",
    "src/pysci/skills/devops",
    "src/pysci/skills/bencher",
    "orchestration/guards",
    "orchestration/skills",
    "orchestration/state/evals",
    "orchestration/state/failure-tags.json",
    "orchestration/pods",
    "pyproject.toml",
)


def _no_log(_msg: str) -> None:
    """默认进度回调：静默（库调用方自理日志）。"""


_NO_LOG: Callable[[str], None] = _no_log


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _git_proc(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    """直连 git（argv 列表，不经 shell——MSYS 路径转换坑免疫，见 devops AGENTS.md）。"""
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd or PROJECT_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )


def _git(*args: str, cwd: Path | None = None) -> tuple[int, str]:
    try:
        proc = _git_proc(*args, cwd=cwd)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 1, str(exc)
    return proc.returncode, (proc.stdout or "").strip()


def resolve_commit(ref: str) -> str:
    """ref → 完整 commit；解析失败抛 ValueError（副本 A/B 的 ref 写错必须当场听见）。"""
    rc, out = _git("rev-parse", "--verify", f"{ref}^{{commit}}")
    if rc != 0 or not out:
        raise ValueError(
            f"ref 解析失败：{ref!r}（{out.splitlines()[-1] if out else 'git 无输出'}）"
        )
    return out


# ---------------------------------------------------------------------------
# 基线捕获
# ---------------------------------------------------------------------------
def _pod_gitignored_files(pod_rel: str) -> list[str]:
    """该 pod 下 git **保不住**的文件（ignored + 未跟踪），项目根相对路径。

    用 ``ls-files -ico --exclude-standard``（-o 未跟踪、-c ignored）——「按各 pod
    .gitignore 细则」正是这条命令在做的事，我们自己不再维护一张目录表，组员自建
    rules/skills 落在哪个目录都自动被覆盖。
    """
    rc, out = _git(
        "-c",
        "core.quotePath=false",
        "ls-files",
        "-ico",
        "--exclude-standard",
        "-z",
        "--",
        pod_rel,
    )
    if rc != 0:
        return []
    paths = [p.replace("\\", "/") for p in out.split("\0") if p.strip()]
    keep: list[str] = []
    for p in paths:
        rel = p[len(pod_rel) :].strip("/")
        top = rel.split("/")[0] if rel else ""
        if top in ARCHIVE_SKIP:
            continue
        keep.append(p)
    return sorted(keep)


def capture_baseline(
    case_id: str, *, ref: str = "HEAD", extra_pods: list[str] | None = None
) -> dict[str, Any]:
    """捕获一案的基线：记录 ref 的 commit + 归档案内组员 pod 的 gitignored 面。

    Args:
        case_id: 案卷 id（``state/evals/harness/<case_id>.md``）。
        ref: 要钉住的 commit（默认 HEAD=审批时点）。
        extra_pods: 追加归档的 pod id（默认只收案内 ``member:``）。

    Returns:
        写进 ``state/evals/baselines/<case_id>/manifest.json`` 的文档。

    Raises:
        ValueError: 案卷读不到（捕获的是「这一案当时依赖什么」，无案即无语义）。
    """
    commit = resolve_commit(ref)
    pods: list[str] = []
    try:
        case = _eval.parse_case(_eval.HARNESS_DIR / f"{case_id}.md")
        pods.append(case.member)
    except (OSError, ValueError):
        if not extra_pods:
            raise ValueError(
                f"案卷不可解析：{case_id}（捕获基线要知道案内 member，先补案或显式 --pods）"
            ) from None
    pods += [p for p in (extra_pods or []) if p]
    seen: set[str] = set()
    dest = BASELINES_DIR / case_id
    files: list[dict[str, Any]] = []
    for pod in [p for p in pods if not (p in seen or seen.add(p))]:
        pod_rel = f"orchestration/pods/{pod}"
        for rel in _pod_gitignored_files(pod_rel):
            src = PROJECT_ROOT / rel
            if not src.is_file():
                continue
            target = dest / "pods" / pod / rel[len(pod_rel) :].strip("/")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target)
            data = src.read_bytes()
            files.append(
                {
                    "path": rel,
                    "bytes": len(data),
                    "sha256": _sha(data),
                }
            )
    doc = {
        "version": 1,
        "case": case_id,
        "ref": ref,
        "commit": commit,
        "captured_at": _now(),
        "pods": sorted(seen),
        "files": files,
        # 物理归档是 gitignored 面（可再生性靠 git + 本清单）；清单入库即 §13-2 的答案。
        "archive_tracked": False,
    }
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "manifest.json").write_text(
        json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return doc


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def read_manifest(case_id: str) -> dict[str, Any]:
    """读某案基线清单；缺失/损坏返回空 dict（调用方据此判「还没捕获过」）。"""
    p = BASELINES_DIR / case_id / "manifest.json"
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


# ---------------------------------------------------------------------------
# 副本工作树
# ---------------------------------------------------------------------------
def replica_name(label: str, side: str) -> str:
    """副本名 ``bench-<label>-<side>``（side=a|b）。"""
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "-" for ch in label)
    return f"{REPLICA_PREFIX}-{safe}-{side}"


def replica_path(name: str) -> Path:
    return WORKTREES_ROOT / name


def build_replica(
    name: str,
    ref: str,
    *,
    cases: list[str] | None = None,
    log: Callable[[str], None] = _NO_LOG,
) -> dict[str, Any]:
    """按 ref 建一份 detached worktree 副本，并补齐「git 保不住」的面。

    步骤（顺序即因果，别调）：worktree add → 复制 ``.env`` → 还原基线归档 →
    子进程内重建 pods 运行目录 + 技能部署（:data:`PYTHONPATH` 重锚 PROJECT_ROOT）。

    Returns:
        ``{name, path, ref, commit, restored: [归档还原文件数]}``。

    Raises:
        RuntimeError: git worktree add 失败（ref 不存在/路径被占）。
    """
    path = replica_path(name)
    if path.exists():
        _git("worktree", "remove", "--force", str(path))
        shutil.rmtree(path, ignore_errors=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    commit = resolve_commit(ref)
    rc, out = _git("worktree", "add", "--detach", str(path), commit)
    if rc != 0:
        raise RuntimeError(f"副本建立失败（{name} @ {ref}）：{out.splitlines()[-1:]}")
    env_src = PROJECT_ROOT / ".env"
    if env_src.exists():
        shutil.copy2(env_src, path / ".env")
        log("[i] 副本已复制 .env（凭据 gitignored；副本用后即弃）")
    restored = 0
    for case in cases or []:
        restored += _restore_archive(case, path, log=log)
    log(f"[i] 副本就绪 {name}（{commit[:8]}），还原归档 {restored} 文件")
    _replica_call(path, ["_replica-prep"], timeout_s=600, log=log)
    return {
        "name": name,
        "path": str(path),
        "ref": ref,
        "commit": commit,
        "restored": restored,
    }


def _restore_archive(
    case_id: str, replica: Path, *, log: Callable[[str], None] = _NO_LOG
) -> int:
    """把该案基线归档里的 pod 文件放进副本（**先于** sync，让副本真本覆盖部署副本）。"""
    src_root = BASELINES_DIR / case_id / "pods"
    if not src_root.exists():
        log(f"[i] {case_id} 无基线归档（先跑 bench capture，A/B 只比 git 保得住的面）")
        return 0
    n = 0
    for src in sorted(p for p in src_root.rglob("*") if p.is_file()):
        rel = src.relative_to(src_root)  # <pod>/<pod 相对路径>
        dst = replica / "orchestration" / "pods" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        n += 1
    return n


def destroy_replica(name: str, *, log: Callable[[str], None] = _NO_LOG) -> bool:
    """摘除副本工作树（``--force`` 容忍脏面）+ ``prune`` 兜底。"""
    path = replica_path(name)
    rc, _ = _git("worktree", "remove", "--force", str(path))
    if rc != 0 and path.exists():
        shutil.rmtree(path, ignore_errors=True)
    _git("worktree", "prune")
    gone = not path.exists()
    log(f"[{'√' if gone else '!'}] 副本清理 {name}")
    return gone


def list_replicas() -> list[str]:
    """当前 :data:`REPLICA_PREFIX` 前缀的残留副本名（验收判据「零残留」的读数面）。"""
    rc, out = _git("worktree", "list", "--porcelain")
    if rc != 0:
        return []
    names: list[str] = []
    for line in out.splitlines():
        if line.startswith("worktree "):
            p = line[len("worktree ") :]
            if Path(p).name.startswith(f"{REPLICA_PREFIX}-"):
                names.append(Path(p).name)
    return names


def _replica_pythonpath(replica: Path) -> str:
    src = str(replica / "src")
    old = os.environ.get("PYTHONPATH", "")
    return f"{src}{os.pathsep}{old}" if old else src


#: 副本子进程的入口协议文件（**主树**的 ``pysci-bench`` 实现，按文件路径直跑）。
#:
#: 为什么不 ``-m pysci.skills.bencher.tools.bench``：``-m`` 让入口模块也从副本树解析，
#: 而 bencher 包 E4（8192235）才入库——对更早的 ref，副本里根本没有这个模块，
#: 子进程当场 ModuleNotFoundError，A/B 退化成「1.3s 空壳 done」（backlog
#: 20261012-170000-leader 实锤）。文件直跑则 ``sys.path[0]``=主树 bench tools 目录
#: （无 ``pysci`` 包可遮蔽），``import pysci`` 仍按 PYTHONPATH 命中副本——**测量协议
#: 来自主树（稳定），被测 harness 来自副本**，正是 A/B 要的切分。
BENCH_ENTRY: Path = (
    Path(__file__).resolve().parents[2] / "bencher" / "tools" / "bench.py"
)


def _replica_call(
    replica: Path,
    bench_args: list[str],
    *,
    timeout_s: int,
    log: Callable[[str], None] = _NO_LOG,
) -> tuple[int, str, str]:
    """在副本语义下跑一次 ``pysci-bench`` 内部命令（返回 rc/stdout/stderr）。

    关键在 :func:`_replica_pythonpath`：PYTHONPATH 让 ``pysci.paths.PROJECT_ROOT`` 落在
    副本根，于是这一次调用看到的 state/pods/技能真本全是副本的。cwd 留在主树——
    cd 进副本会让 uv 就地建空 ``.venv``（devops AGENTS.md 已记录的坑）。
    rc≠0 时把 stderr 尾巴写进调用方日志：静默失败正是空壳 done 的温床。
    """
    env = {
        **os.environ,
        "PYTHONPATH": _replica_pythonpath(replica),
        "PYTHONUTF8": "1",
        "PYSCI_VENV_SCRIPTS": str(_venv_scripts(PROJECT_ROOT)),
    }
    argv = [sys.executable, str(BENCH_ENTRY), *bench_args]
    try:
        proc = subprocess.run(
            argv,
            cwd=str(PROJECT_ROOT),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        log(f"[!] 副本调用失败：{exc}")
        return 1, "", str(exc)
    if proc.returncode != 0:
        tail = ((proc.stderr or "") + (proc.stdout or ""))[-400:]
        log(f"[!] 副本调用 rc={proc.returncode}（{' '.join(bench_args[:2])}…）：{tail}")
    return proc.returncode, proc.stdout or "", proc.stderr or ""


def _venv_scripts(root: Path) -> Path:
    """某树 venv 的命令目录（Windows ``Scripts`` / POSIX ``bin``）。"""
    win = root / ".venv" / "Scripts"
    return win if win.exists() else root / ".venv" / "bin"


def rows_from_stdout(stdout: str) -> list[dict[str, Any]]:
    """取副本 stdout 里 ``ROWS_BEGIN``/``ROWS_END`` 之间的台账行 JSON。

    用标记带而不是「解析最后一行」：组员输出里本来就可能有多行 JSON（结果正文），
    裸取末行会把交付正文当读数。
    """
    if "ROWS_BEGIN" not in stdout:
        return []
    body = stdout.split("ROWS_BEGIN", 1)[1].split("ROWS_END", 1)[0]
    rows: list[dict[str, Any]] = []
    for line in body.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and obj.get("case"):
            rows.append(obj)
    return rows


# ---------------------------------------------------------------------------
# 窗口期污染与 A/B 判决
# ---------------------------------------------------------------------------
def window_commits(ref_a: str, ref_b: str) -> list[dict[str, str]]:
    """``ref_a..ref_b`` 之间**碰过 harness 面**的提交（排除两端本身）。

    为什么这是判决的前置闸（§5）：A/B 归因说「a→b 的差异造成了读数差异」，若中间还夹着
    别的知识面提交，读数差就是那几个提交的合力——此时任何「改善/未改善」都是错的因果。
    识别出来就停手，标 ``inconclusive`` 并把证据交给终裁链（``escalate=twin``）。
    """
    rc, out = _git(
        "log",
        "--format=%h\t%s",
        f"{ref_a}..{ref_b}",
        "--",
        *HARNESS_FACE,
    )
    if rc != 0 or not out:
        return []
    out_rows: list[dict[str, str]] = []
    ends = {resolve_commit(ref_a), resolve_commit(ref_b)}
    for line in out.splitlines():
        parts = line.split("\t", 1)
        if len(parts) != 2:
            continue
        sha, subject = parts[0].strip(), parts[1].strip()
        full = resolve_commit(sha) if sha else ""
        if full in ends:
            continue
        out_rows.append({"sha": sha, "subject": subject})
    return out_rows


def run_ab(
    ref_a: str,
    ref_b: str,
    cases: list[str],
    *,
    label: str,
    tier: str = "",
    force: bool = False,
    log: Callable[[str], None] = _NO_LOG,
) -> dict[str, Any]:
    """副本 A/B：两 ref 各跑案内案 → 二元判决 → 落盘 ``state/evals/ab/<label>.json``。

    Args:
        ref_a / ref_b: 旧→新两端（``改善`` 意味着 b 优于 a）。
        cases: 案 id 列表；空 = 该 ref 下的全套。
        label: 本轮标签（同时是判决文件名与 ``evals.jsonl`` 的分组面）。
        tier: 覆盖案卷档位。
        force: 窗口期污染时也照跑（只信 git 的极端场合才用；默认停手）。
        log: 进度回调（worker 传 run 日志写入器）。

    Returns:
        ``{label, ref_a, ref_b, verdict, compared, pollution, escalate, sides,
        evidence, residual_worktrees}``；``verdict`` ∈ 改善 / 未改善 / inconclusive。
        任一侧零真实读数时置 ``error``（空壳闸）且不出判决——「没做」不许伪装成
        「做了且未改善」。
    """
    ids = [str(c) for c in cases or []]
    if not ids:
        ids = sorted(
            f.stem
            for f in (
                _eval.HARNESS_DIR if _eval.HARNESS_DIR.exists() else Path(".")
            ).glob("*.md")
        )
    pollution = window_commits(ref_a, ref_b)
    doc: dict[str, Any] = {
        "version": 1,
        "label": label,
        "started_at": _now(),
        "ref_a": ref_a,
        "ref_b": ref_b,
        "commit_a": "",
        "commit_b": "",
        "cases": ids,
        "pollution": pollution,
        "verdict": "",
        "escalate": "",
        "sides": {},
    }
    if pollution and not force:
        doc["verdict"] = "inconclusive"
        doc["escalate"] = "twin"
        doc["error"] = (
            f"窗口期污染：{ref_a}..{ref_b} 间另有 {len(pollution)} 个 harness 面提交，"
            "读数差无法归因于 a→b 这一改动——二元判决会给出错误的因果"
        )
        doc["finished_at"] = _now()
        _write_ab(doc)
        log(f"[!] {doc['error']}（escalate=twin，挂起留痕）")
        return doc
    names = [replica_name(label, "a"), replica_name(label, "b")]
    try:
        sides: dict[str, dict[str, Any]] = {}
        for side, ref, name in (("a", ref_a, names[0]), ("b", ref_b, names[1])):
            rep = build_replica(name, ref, cases=ids, log=log)
            doc[f"commit_{side}"] = rep["commit"]
            rows = _hop_cases(
                Path(rep["path"]), ids, label=f"{label}-{side}", tier=tier, log=log
            )
            sides[side] = {"commit": rep["commit"], "cases": rows}
            doc["sides"] = sides
        # 空壳闸（backlog 20261012-170000-leader）：占位 fail 行不是读数，不许进 compare。
        real = {
            side: {
                c: r
                for c, r in sides[side]["cases"].items()
                if r and not r.get("synthetic")
            }
            for side in ("a", "b")
        }
        doc["evidence"] = {
            "ledger_rows": sum(len(side_rows) for side_rows in real.values())
        }
        if not real["a"] or not real["b"]:
            doc["error"] = (
                "副本空壳完成闸：a 侧 "
                f"{len(real['a'])}/{len(ids)}、b 侧 {len(real['b'])}/{len(ids)}"
                " 行真实读数——副本未派发或环境失效（逐案原因见 sides 的 reasons），"
                "无读数不构成判决"
            )
            log(f"[✗] {doc['error']}")
        else:
            cmp = _eval.compare(real["b"], {"cases": real["a"]})
            doc["verdict"] = cmp["verdict"]
            doc["compare"] = cmp
    finally:
        for name in names:
            destroy_replica(name, log=log)
    doc["residual_worktrees"] = list_replicas()
    doc["finished_at"] = _now()
    _write_ab(doc)
    return doc


def _hop_cases(
    replica: Path,
    case_ids: list[str],
    *,
    label: str,
    tier: str,
    log: Callable[[str], None] = _NO_LOG,
) -> dict[str, dict[str, Any]]:
    """在副本里跑一批案，把读数取回主树台账（``runner=bencher``）。

    副本未返回读数的案补 ``synthetic`` 占位行（带 rc/stderr 尾巴）——只供人查原因，
    **不是判决证据**：run_ab 的空壳闸据此把这类行挡在 compare 之外。
    """
    rc, out, err = _replica_call(
        replica,
        [
            "_hop",
            "--cases",
            ",".join(case_ids),
            "--label",
            label,
            "--tier",
            tier,
        ],
        timeout_s=_hop_timeout(case_ids),
        log=log,
    )
    rows = rows_from_stdout(out)
    result: dict[str, dict[str, Any]] = {}
    for r in rows:
        result[str(r["case"])] = r
        _eval.EVAL_LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _eval.EVAL_LEDGER_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    for cid in case_ids:
        result.setdefault(
            cid,
            {
                "case": cid,
                "verdict": "fail",
                "reasons": [f"副本未返回读数（rc={rc}）：{(err or out)[-200:]}"],
                "num_turns": 0,
                "synthetic": True,
            },
        )
    return result


def _hop_timeout(case_ids: list[str]) -> int:
    """整批跳的墙钟上限 = Σ案内 ``timeout_s`` + 20% 余量（派发链自身也要时间）。"""
    total = 0
    for cid in case_ids:
        try:
            total += _eval.parse_case(_eval.HARNESS_DIR / f"{cid}.md").timeout_s
        except (OSError, ValueError):
            total += _eval.DEFAULT_TIMEOUT_S
    return int(total * 1.2) + 120


def _write_ab(doc: dict[str, Any]) -> None:
    AB_DIR.mkdir(parents=True, exist_ok=True)
    name = str(doc.get("label") or "unnamed")
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "-" for ch in name)
    path = AB_DIR / f"{safe}.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(str(tmp), str(path))


def latest_ab(label: str = "") -> dict[str, Any] | None:
    """最近一份 A/B 判决（``label`` 精确回取）；无则 None。"""
    if not AB_DIR.exists():
        return None
    files = sorted(AB_DIR.glob("*.json"))
    if label:
        files = [f for f in files if f.stem == label]
    if not files:
        return None
    try:
        data = json.loads(files[-1].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def ab_rel(doc: dict[str, Any]) -> str:
    """判决文件的 orchestration 相对路径（日志/status 里给可读定位）。"""
    return orchestration_rel(AB_DIR / f"{doc.get('label', '')}.json")
