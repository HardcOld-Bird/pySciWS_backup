"""``pysci-bench``——bencher 组员的测量工具面（REFACTOR v1.0 §3/§8；E4）。

四条命令面（**全部复用**共享库，本文件只做参数装配与读数呈现）：

- ``run`` —— 跑套件案（一案/全部），台账行带 ``runner=bencher``；二元判决 0/1，
  案卷不可用 2（与 ``orch eval run`` 同一口径，两条通道的差别只在**发起方**）。
- ``capture`` —— 基线捕获：记录 ref 的 commit + 归档案内 pod 的 gitignored harness 面
  （:func:`replica.capture_baseline`）。§13 待决 2 在 E4 的定案：**不建常驻 worktree**，
  理由见 :mod:`replica` 模块文档串。
- ``ab`` —— 副本 A/B：两 ref 各建 detached worktree、各跑案内案、出「改善/未改善」；
  窗口期污染 → ``inconclusive`` + ``escalate=twin``（§5.5 回退链）。退出码
  ``0=改善 / 1=未改善 / 2=用法错 / 3=inconclusive``。
- ``baseline`` / ``report`` —— 基线回填与读数表（``--runner`` 默认 bencher：移交后
  「谁测的」必须分得开）。

两个内部命令（下划线前缀，勿手调）：``_hop`` 在**副本语义**下跑一批案并把台账行以
``ROWS_BEGIN/ROWS_END`` 标记带吐回父进程；``_replica-prep`` 在副本里重建 pods 运行目录
并跑技能部署。二者由 :mod:`replica` 以 ``PYTHONPATH=<副本>/src`` 的子进程调用。
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from pysci.paths import PODS_ROOT, PROJECT_ROOT
from pysci.skills.orchestration.tools import eval as _eval
from pysci.skills.orchestration.tools import replica as _rep

ROWS_BEGIN = "ROWS_BEGIN"
ROWS_END = "ROWS_END"


def _split_csv(value: str | None) -> list[str]:
    return [v.strip() for v in (value or "").replace("，", ",").split(",") if v.strip()]


# ---------------------------------------------------------------------------
# run / baseline / report
# ---------------------------------------------------------------------------
def cmd_run(args: argparse.Namespace) -> int:
    """``pysci-bench run``：以 bencher 身份跑套件（测量通道，不是组长抽检）。"""
    try:
        cases = _eval.load_cases("" if args.all else args.case)
    except ValueError as exc:
        print(f"[✗] 案卷不可用：{exc}")
        print(
            "    [NEXT] 案卷=state/evals/harness/<case-id>.md（member:/tier:/dirs:/产物:/断言:）"
        )
        return 2
    ran: dict[str, dict[str, Any]] = {}
    print(
        f"== bencher 套件运行（{len(cases)} 案，档位 {args.tier or '案内 tier'}，"
        f"label={args.label or '-'}）=="
    )
    for case in cases:
        row = _eval.run_case(
            case,
            tier=args.tier or "",
            label=args.label or "",
            runner=_eval.RUNNER_BENCHER,
        )
        _eval._print_row(row)
        ran[row.case] = _asdict(row)
    return _verdict_exit(ran)


def _asdict(row: _eval.Row) -> dict[str, Any]:
    from dataclasses import asdict

    return asdict(row)


def cmd_baseline(args: argparse.Namespace) -> int:
    """``pysci-bench baseline``：把最近一轮（可按 label/runner 选）回填为基线快照。"""
    try:
        doc = _eval.write_baseline(args.label or "", args.runner or "")
    except ValueError as exc:
        print(f"[✗] {exc}")
        return 2
    print(f"[√] 基线已回填：{len(doc['cases'])} 案 → {_eval.BASELINE_PATH.name}")
    for cid, c in doc["cases"].items():
        print(
            f"  - {cid:<18} {c['verdict']:<4} turns={c['num_turns']} ctx={c['ctx_ratio']:.4f}"
        )
    print("    [NEXT] `pysci-bench ab --ref-a <旧> --ref-b <新>` 出二元判决。")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    """``pysci-bench report``：最近读数表（默认只看 bencher 通道）+ 基线对照。"""
    rows = _eval.latest_rows(args.label or "", args.runner or _eval.RUNNER_BENCHER)
    if not rows:
        print("  （台账里没有匹配的压测行）")
        return 0
    base = _eval.read_baseline()
    print(
        f"== 读数表（label={args.label or '-'}，runner={args.runner or 'bencher'}）=="
    )
    for cid in sorted(rows):
        r = rows[cid]
        b = (base.get("cases") or {}).get(cid, {})
        man = _rep.read_manifest(cid)
        print(
            f"  {cid:<18} {r.get('verdict', '?'):<4} turns={r.get('num_turns', 0)}"
            f"/{r.get('max_turns', 0)} dur={int(r.get('duration_ms', 0)) / 1000:.0f}s"
            + (f"  基线 {b.get('verdict', '-')}/{b.get('num_turns', '?')}" if b else "")
            + (
                f"  捕获@{str(man.get('commit', ''))[:7]}"
                f"（{len(man.get('files') or [])} 归档）"
                if man
                else "  [!] 无基线捕获"
            )
        )
    return 0


def _verdict_exit(ran: dict[str, dict[str, Any]]) -> int:
    """通过=0、有案失败=1（判决是测量结果，不是崩溃）。"""
    if not ran:
        return 2
    return 0 if all(r.get("verdict") == "pass" for r in ran.values()) else 1


# ---------------------------------------------------------------------------
# capture / ab
# ---------------------------------------------------------------------------
def cmd_capture(args: argparse.Namespace) -> int:
    """``pysci-bench capture``：基线捕获（commit 记录 + gitignored 面归档）。"""
    try:
        doc = _rep.capture_baseline(
            args.case, ref=args.ref or "HEAD", extra_pods=_split_csv(args.pods)
        )
    except (ValueError, OSError) as exc:
        print(f"[✗] 捕获失败：{exc}")
        return 2
    dest = _rep.BASELINES_DIR / args.case
    print(
        f"[√] 基线已捕获：{args.case} @ {doc['commit'][:8]}（{doc['ref']}），"
        f"归档 {len(doc['files'])} 个 gitignored 文件 → {dest.relative_to(PROJECT_ROOT)}"
    )
    for f in doc["files"][:12]:
        print(f"  - {f['path']}（{f['bytes']}B）")
    if len(doc["files"]) > 12:
        print(f"  … 其余 {len(doc['files']) - 12} 项见 manifest.json")
    return 0


def cmd_ab(args: argparse.Namespace) -> int:
    """``pysci-bench ab``：副本 A/B 二元判决（含窗口期污染闸）。"""
    try:
        doc = _rep.run_ab(
            args.ref_a,
            args.ref_b,
            _split_csv(args.cases),
            label=args.label or f"ab-{args.ref_a[:7]}-{args.ref_b[:7]}",
            tier=args.tier or "",
            force=args.force,
            log=lambda m: print(f"  {m}"),
        )
    except (ValueError, RuntimeError) as exc:
        print(f"[✗] A/B 未起跑：{exc}")
        return 2
    print(f"\n== A/B 判决：{doc['verdict'] or '未出'}（label={doc['label']}）==")
    print(
        f"  a={doc.get('commit_a', '')[:8]}（{doc['ref_a']}）  "
        f"b={doc.get('commit_b', '')[:8]}（{doc['ref_b']}）"
    )
    for side in ("a", "b"):
        cases = (doc.get("sides") or {}).get(side, {}).get("cases", {})
        for cid in sorted(cases):
            r = cases[cid] or {}
            print(
                f"  [{side}] {cid:<18} {r.get('verdict', '?'):<4} "
                f"turns={r.get('num_turns', 0)}/{r.get('max_turns', 0)}"
            )
            for reason in r.get("reasons") or []:
                print(f"        · {reason}")
    if doc.get("pollution"):
        print(f"  [!] 窗口期污染提交 {len(doc['pollution'])}：")
        for c in doc["pollution"][:8]:
            print(f"        - {c['sha']} {c['subject'][:60]}")
    if doc.get("escalate"):
        print(
            f"  [s] escalate={doc['escalate']}：§0.5 无平局档，模棱两可交终裁链"
            "（taste 未启用则挂起留痕，不静默丢弃）"
        )
    residual = doc.get("residual_worktrees") or []
    print(
        f"  副本残留：{'零（判据 4 达标）' if not residual else residual}"
        f"  判决落盘 {_rep.ab_rel(doc)}"
    )
    if residual:
        return 2
    return {
        "改善": 0,
        "未改善": 1,
        "inconclusive": 3,
    }.get(str(doc.get("verdict")), 2)


def cmd_replica(args: argparse.Namespace) -> int:
    """``pysci-bench replica``：列残留副本（清理纪律的自查面）。"""
    names = _rep.list_replicas()
    if not names:
        print("  （无 bench-* 副本残留）")
        return 0
    for n in names:
        print(f"  ▶ {n} → {_rep.replica_path(n)}")
    print(
        "    [NEXT] `pysci-bench ab` 正常收尾会自动摘除；手动清：git worktree remove --force"
    )
    return 1


# ---------------------------------------------------------------------------
# 内部命令（副本语义下由 orchestration.tools.replica 调用）
# ---------------------------------------------------------------------------
def cmd_hop(args: argparse.Namespace) -> int:
    """内部命令 ``_hop``：在**当前树**（副本 PYTHONPATH 已重锚）跑一批案，吐行。"""
    ids = _split_csv(args.cases)
    cases = _eval.load_cases()
    if ids:
        cases = [c for c in cases if c.case_id in ids]
    rows = []
    for case in cases:
        try:
            rows.append(
                _asdict(
                    _eval.run_case(
                        case,
                        tier=args.tier or "",
                        label=args.label or "",
                        runner=_eval.RUNNER_BENCHER,
                    )
                )
            )
        except Exception as exc:  # noqa: BLE001 — 标记带必须吐得出来，一案崩不许带走整批
            rows.append(
                {
                    "case": case.case_id,
                    "verdict": "fail",
                    "reasons": [f"副本跳异常：{type(exc).__name__}: {exc}"],
                    "num_turns": 0,
                    "duration_ms": 0,
                    "runner": _eval.RUNNER_BENCHER,
                    "label": args.label or "",
                }
            )
    print(ROWS_BEGIN)
    for r in rows:
        print(json.dumps(r, ensure_ascii=False))
    print(ROWS_END)
    return 0


def cmd_replica_prep(args: argparse.Namespace) -> int:
    """内部命令 ``_replica-prep``：在当前树重建 pods 运行目录并部署技能。"""
    del args
    from pysci.skills.orchestration.tools.sync import sync

    made = 0
    if PODS_ROOT.exists():
        for pod in sorted(p for p in PODS_ROOT.iterdir() if p.is_dir()):
            for rel in _rep.POD_RUNTIME_DIRS:
                target = pod / rel
                if not target.exists():
                    target.mkdir(parents=True, exist_ok=True)
                    made += 1
    report = sync()
    print(f"[i] pods 运行目录新建 {made} 个；技能部署 {len(report)} 行")
    for line in report:
        if not line.startswith("[=]"):
            print(f"  {line}")
    return 0


# ---------------------------------------------------------------------------
# 门面
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="pysci-bench",
        description=(
            "bencher 组员测量面：套件运行 / 基线捕获 / 副本 A/B / 读数表"
            "（复用 orchestration 共享库，不复制判据实现）"
        ),
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run", help="跑套件案（--all 或 --case）→ 二元判决")
    p.add_argument("--case", default="", help="案卷 id")
    p.add_argument("--all", action="store_true", help="跑全套（按文件名序）")
    p.add_argument("--tier", default="", help="覆盖档位（空=案内 tier，默认 minimum）")
    p.add_argument("--label", default="", help="本轮标签（供 baseline/ab 精确回取）")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("capture", help="基线捕获：commit 记录 + gitignored 面归档")
    p.add_argument("--case", required=True, help="案卷 id")
    p.add_argument("--ref", default="HEAD", help="要钉住的 ref（默认 HEAD）")
    p.add_argument(
        "--pods", default="", help="追加归档的 pod（逗号分隔；默认案内 member）"
    )
    p.set_defaults(fn=cmd_capture)

    p = sub.add_parser("ab", help="副本 A/B：两 ref 各跑案内案 → 二元判决")
    p.add_argument("--ref-a", required=True, help="旧端 ref")
    p.add_argument("--ref-b", required=True, help="新端 ref")
    p.add_argument("--cases", default="", help="案 id（逗号分隔；空=全套）")
    p.add_argument("--label", default="", help="本轮标签（判决文件名）")
    p.add_argument("--tier", default="", help="覆盖档位")
    p.add_argument(
        "--force",
        action="store_true",
        help="窗口期污染时仍照跑（默认停手判 inconclusive）",
    )
    p.set_defaults(fn=cmd_ab)

    p = sub.add_parser("baseline", help="把最近一轮回填为基线快照")
    p.add_argument("--label", default="", help="只取该标签的读数")
    p.add_argument(
        "--runner",
        default=_eval.RUNNER_BENCHER,
        help=f"只取该发起方的读数（默认 {_eval.RUNNER_BENCHER}；空=不分）",
    )
    p.set_defaults(fn=cmd_baseline)

    p = sub.add_parser("report", help="最近读数表（默认只看 bencher 通道）")
    p.add_argument("--label", default="", help="按标签过滤")
    p.add_argument(
        "--runner",
        default=_eval.RUNNER_BENCHER,
        help="按发起方过滤（空=不分 leader/bencher）",
    )
    p.set_defaults(fn=cmd_report)

    p = sub.add_parser("replica", help="列残留测量副本（清理纪律自查）")
    p.set_defaults(fn=cmd_replica)

    p = sub.add_parser("_hop", help="[内部] 在当前树跑一批案并吐台账行")
    p.add_argument("--cases", default="")
    p.add_argument("--label", default="")
    p.add_argument("--tier", default="")
    p.set_defaults(fn=cmd_hop)

    p = sub.add_parser("_replica-prep", help="[内部] 副本内重建 pods 目录 + 技能部署")
    p.set_defaults(fn=cmd_replica_prep)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.fn(args))
    except KeyboardInterrupt:
        print("\n[!] 已中断（台账只写已完成的行）")
        return 130


if __name__ == "__main__":
    sys.exit(main())
