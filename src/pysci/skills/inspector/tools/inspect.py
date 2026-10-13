"""``pysci-inspect``——inspector 组员的发现登记与分流工具面（README §6；E5a）。

四条命令面（**全部复用** :mod:`orchestration.tools.inspectd` 库，本文件只做参数装配与
读数呈现）：

- ``finding add`` —— 登记一条发现（``--class mechanical|judgment``）。机械证据类必须带
  ``--repro`` 与 ``--evidence``（别人能重跑才算证据，basic.md §2）；两类都必须
  ``--criteria`` 写可证伪判据，否则 :func:`inspectd.promote_finding` 会按 §6 总闸拒晋升。
- ``finding list`` —— 读发现台账（可按状态过滤）。
- ``promote <finding>`` —— 机械证据类免判决晋升进 backlog（``source=inspector``）。
  判断类调这条会被拒：它没有免判决通道。
- ``submit <finding> --jury`` —— 判断类**必经** J 仲裁后晋升。本命令的参数表里没有
  ``--override``：§5.2 的「inspector 无自裁面」是代码事实，不是纪律条款。

另附：

- ``gate`` 读当前资源闸门（开关/让路/迟滞）——组员据此知道「现在能不能产出」，
  但放行与否由引擎在 :func:`inspectd.activation_gate` 一处决定，本 CLI 无写权；
- ``transcript list|scan|fetch`` —— **会话记录审计工具链**（用户设计 2026-10-13，
  backlog 20261013-130000-leader）：list 按优先级列可检会话 → scan 拿简版定位疑点 →
  fetch 按行锚取原文。**纯 agent 导向**：返回值即阅读面（用户查聊天用 Qoder CLI 本身）。
  实现全在 :mod:`orchestration.tools.transcript`，本文件只装配参数与呈现读数。
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from typing import Any

from pysci.paths import PROJECT_ROOT
from pysci.skills.orchestration.tools import inspectd as _insp
from pysci.skills.orchestration.tools import registry as _reg
from pysci.skills.orchestration.tools import transcript as _tr


def _require(fid: str) -> dict[str, Any]:
    """按 id 取发现，取不到就抛给调用方打印可读提示。"""
    rec = _insp.finding_by_id(fid)
    if rec is None:
        raise ValueError(
            f"找不到发现 {fid!r}（或前缀不唯一）：`pysci-inspect finding list` 看现有 id"
        )
    return rec


# ---------------------------------------------------------------------------
# finding add / list
# ---------------------------------------------------------------------------
def cmd_finding_add(args: argparse.Namespace) -> int:
    """``pysci-inspect finding add``：登记一条发现（先落台账，分流由类别决定）。"""
    try:
        rec = _insp.append_finding(
            cls=args.cls,
            title=args.title,
            symptom=args.symptom,
            repro=args.repro,
            evidence=args.evidence,
            expect=args.expect,
            tag=args.tag,
            kind=args.kind,
            criteria=args.criteria,
            source=args.source,
            remedy_shape=args.remedy_shape,
        )
    except ValueError as exc:
        print(f"[✗] 登记被拒：{exc}")
        return 2
    print(
        f"[√] 已登记发现 {rec['id']}（{rec['class']}，kind={rec['kind']}，"
        f"remedy_shape={rec.get('remedy_shape', 'change')}）"
    )
    if rec["class"] == "mechanical":
        print(
            f"    [NEXT] `pysci-inspect promote {rec['id']}` 免判决晋升；"
            "引擎也会在全闲时自动扫 pending 机械类。"
        )
    else:
        print(
            f"    [NEXT] `pysci-inspect submit {rec['id']} --jury`（判断类必经仲裁，"
            "无自裁通道）。"
        )
    if not (rec.get("success_criteria") or "").strip():
        print("    [!] 缺 --criteria：无判据的提案不可证伪，晋升会被 §6 总闸拒绝。")
    return 0


def cmd_finding_list(args: argparse.Namespace) -> int:
    """``pysci-inspect finding list``：读发现台账。"""
    rows = _insp.findings(status=args.status) if args.status else _insp.findings()
    if not rows:
        print("  （发现台账为空：finding add 登记一条）")
        return 0
    for rec in rows:
        print(
            f"  {rec['id']}  [{rec.get('status', '?'):<9}] "
            f"{rec.get('class', '?'):<10} kind={rec.get('kind', '?'):<9} "
            f"{str(rec.get('title', ''))[:60]}"
        )
    print(
        f"  —— 共 {len(rows)} 条（台账 {_insp.orchestration_rel(_insp.FINDINGS_PATH)}）"
    )
    return 0


# ---------------------------------------------------------------------------
# promote / submit
# ---------------------------------------------------------------------------
def cmd_promote(args: argparse.Namespace) -> int:
    """``pysci-inspect promote``：机械证据类免判决晋升（判断类走这里会被拒）。"""
    try:
        rec = _require(args.finding)
    except ValueError as exc:
        print(f"[✗] {exc}")
        return 2
    if rec.get("class") != "mechanical":
        print(
            f"[✗] {rec['id']} 是 {rec.get('class')} 类：判断类没有免判决通道，"
            "请用 `submit --jury`。"
        )
        return 2
    out = _insp.promote_finding(rec, channel="mechanical", wake=not args.no_wake)
    action = out.get("action")
    if action == "promoted":
        print(
            f"[√] 已晋升 {out['backlog_id']} → backlog 第 {out.get('position')} 位"
            f"（source=inspector，证据 {out.get('evidence_file', '')}）"
        )
        if out.get("wake"):
            print(f"    [▶] {out['wake']}")
        return 0
    if action == "duplicate":
        print(f"[i] {out['backlog_id']} 已在 backlog 中（不重复立案）")
        return 0
    print(f"[✗] 未晋升（{action}）：{out.get('reason', '')}")
    return 1


def cmd_submit(args: argparse.Namespace) -> int:
    """``pysci-inspect submit --jury``：判断类经仲裁后晋升（唯一自决出口）。"""
    try:
        rec = _require(args.finding)
    except ValueError as exc:
        print(f"[✗] {exc}")
        return 2
    if rec.get("class") != "judgment":
        print(
            f"[✗] {rec['id']} 是 {rec.get('class')} 类：机械证据类免判决，用 `promote`。"
        )
        return 2
    out = _insp.submit_judgment(rec, dry=args.dry, timeout=args.timeout)
    outcome = out.get("outcome")
    if outcome == "promoted":
        print(
            f"[√] 仲裁采纳 → 已晋升 {out.get('backlog_id')}"
            f"（backlog 第 {out.get('position')} 位，计票 {out.get('tally')}）"
        )
        return 0
    if outcome == "rejected":
        print(
            f"[i] 仲裁判 B（维持现状），计票 {out.get('tally')}——不入账、不算 override"
        )
        return 0
    mark = "√" if outcome == "dry" else "!"
    print(f"[{mark}] {outcome}：{out.get('reason') or out.get('message') or out}")
    return 0 if outcome in ("dry", "queued", "escalated") else 1


# ---------------------------------------------------------------------------
# gate
# ---------------------------------------------------------------------------
def cmd_gate(args: argparse.Namespace) -> int:
    """``pysci-inspect gate``：读当前资源闸门（只读；放行与否由引擎决定）。"""
    del args
    for line in _insp.status_lines():
        print(line)
    return 0


# ---------------------------------------------------------------------------
# transcript：会话记录审计工具链（backlog 20261013-130000-leader）
# ---------------------------------------------------------------------------
def _leader_key() -> str:
    """leader 项目键 = 仓库根目录的会话键（leader 的 TUI 跑在仓库根）。"""
    return _reg.pod_session_key(PROJECT_ROOT)


def cmd_transcript_list(args: argparse.Namespace) -> int:
    """``transcript list``：按优先级列可检会话（leader 先、新先，头部 ≤ 预算）。"""
    refs = _tr.list_sessions(
        leader_key=_leader_key(),
        budget=args.limit,
        only_checkable=not args.all,
    )
    if not refs:
        print("  （无可检查会话：新增回合 <5 或已全检；`--all` 看全量状态）")
        return 0
    for ref in refs:
        when = (
            datetime.fromtimestamp(ref.last_active).astimezone().strftime("%m-%d %H:%M")
            if ref.last_active
            else "-"
        )
        print(
            f"  {ref.state:<9} {ref.kind:<6} {ref.sid[:8]}  回合={ref.rounds:<4} "
            f"最后活跃={when}  {ref.project}"
        )
    print(
        f"  —— 共 {len(refs)} 个"
        f"（预算 {args.limit if args.limit else '不限'}；scan 一个再 fetch 深挖）"
    )
    return 0


def cmd_transcript_scan(args: argparse.Namespace) -> int:
    """``transcript scan``：简版索引（+``--verbose`` 正文）——摘要先行。"""
    path = _tr.find_session(args.sid)
    if path is None:
        print(f"[✗] 找不到会话 {args.sid}（或前缀有歧义）：先 `transcript list` 选靶")
        return 2
    turns: tuple[int, int] | None = None
    if args.turns:
        try:
            lo, hi = (int(x) for x in str(args.turns).split("-", 1))
        except ValueError:
            print(f"[✗] --turns 形状非法：{args.turns!r}（应为 a-b，如 3-7）")
            return 2
        turns = (lo, hi)
    text = _tr.scan_session(path, turns=turns, verbose=args.verbose)
    print(text)
    if args.save:
        saved = _tr.save_cache(path.stem, text)
        print(f"  [i] 简版已缓存：{_insp.orchestration_rel(saved)}")
    # 检查过就该记「已检查」——否则下轮 list 还会把同一会话顶出来
    cursor = _tr.load_cursor()
    ref = _tr.SessionRef(
        sid=path.stem,
        project=path.parent.name,
        kind=_tr.session_kind(path.parent.name, leader_key=_leader_key()),
        rounds=len(_tr.build_rounds(path)),
        last_active=path.stat().st_mtime,
        state=_tr.CHECKED,
    )
    _tr.mark_checked(
        cursor, ref, now_iso=datetime.now().astimezone().isoformat(timespec="seconds")
    )
    _tr.save_cursor(cursor)
    return 0


def cmd_transcript_fetch(args: argparse.Namespace) -> int:
    """``transcript fetch``：按行锚（或按模式）从原始 jsonl 精确取段（有界返回）。"""
    if args.anchor:
        print(_tr.fetch_anchor(args.anchor))
        return 0
    if args.sid and args.grep:
        print(_tr.fetch_grep(args.sid, args.grep))
        return 0
    print("[✗] 需要 `--anchor <sid:L_a-L_b>`，或 `--sid <sid> --grep <模式>` 之一")
    return 2


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="pysci-inspect",
        description=(
            "inspector 组员的发现登记与分流面（README §6）。"
            "**无 override 参数**——判断类只能经 J 仲裁，机械类才免判决。"
        ),
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("finding", help="发现台账")
    fsub = p.add_subparsers(dest="fcmd", required=True)
    pa = fsub.add_parser("add", help="登记一条发现")
    pa.add_argument(
        "--class",
        dest="cls",
        required=True,
        choices=list(_insp.FINDING_CLASSES),
        help="mechanical=有可重跑证据（免判决）；judgment=须经 J 仲裁",
    )
    pa.add_argument("--title", required=True, help="一句话现象（进 backlog 摘要）")
    pa.add_argument("--symptom", default="", help="症状展开")
    pa.add_argument("--repro", default="", help="复现命令（机械类必填）")
    pa.add_argument("--evidence", default="", help="实际输出（机械类必填）")
    pa.add_argument("--expect", default="", help="期望输出")
    pa.add_argument(
        "--tag", default="", help="失败标签（state/failure-tags.json 词表）"
    )
    pa.add_argument(
        "--kind",
        default="usability",
        choices=["defect", "usability"],
        help="提案分类（进 backlog 条目的 kind）",
    )
    pa.add_argument(
        "--criteria", default="", help="可证伪成功判据（§6 总闸，缺则不晋升）"
    )
    pa.add_argument(
        "--remedy-shape",
        dest="remedy_shape",
        default="",
        choices=list(_insp.REMEDY_SHAPES),
        help="修复形状：restore=复原既有声明行为（免判决照旧）；change=改声明行为/策略"
        "（缺省 change；实施前须过判决包）",
    )
    pa.add_argument("--source", default="inspector", help="来源标记（默认 inspector）")
    pa.set_defaults(fn=cmd_finding_add)

    pl = fsub.add_parser("list", help="列发现")
    pl.add_argument(
        "--status",
        default="",
        help=f"按状态过滤（{', '.join(_insp.FINDING_STATUSES)}）",
    )
    pl.set_defaults(fn=cmd_finding_list)

    p = sub.add_parser("promote", help="机械证据类免判决晋升 backlog")
    p.add_argument("finding", help="发现 id（或其唯一前缀）")
    p.add_argument("--no-wake", action="store_true", help="晋升后不唤醒 devops worker")
    p.set_defaults(fn=cmd_promote)

    p = sub.add_parser("submit", help="判断类经 J 仲裁后晋升")
    p.add_argument("finding", help="发现 id（或其唯一前缀）")
    p.add_argument(
        "--jury",
        action="store_true",
        required=True,
        help="显式确认经仲裁（本 CLI 唯一的自决出口；无 override 面）",
    )
    p.add_argument("--dry", action="store_true", help="组卷但不打电话（演练案卷形状）")
    p.add_argument("--timeout", type=int, default=None, help="仲裁单调用超时秒数覆盖")
    p.set_defaults(fn=cmd_submit)

    p = sub.add_parser("gate", help="读当前资源闸门（开关/让路/迟滞）")
    p.set_defaults(fn=cmd_gate)

    # ---- transcript：会话记录审计工具链（摘要先行、锚点深挖）----
    p = sub.add_parser("transcript", help="会话记录审计工具链（list/scan/fetch）")
    tsub = p.add_subparsers(dest="tcmd", required=True)

    tl = tsub.add_parser("list", help="按优先级列可检会话（leader 先、新先）")
    tl.add_argument(
        "--limit",
        type=int,
        default=_tr.ROUND_BUDGET,
        help=f"头部保留数（缺省每轮预算 {_tr.ROUND_BUDGET}；0=不限）",
    )
    tl.add_argument(
        "--all", action="store_true", help="连同非可检会话一起列出（看全量状态）"
    )
    tl.set_defaults(fn=cmd_transcript_list)

    ts = tsub.add_parser("scan", help="会话简版：每回合一行索引（+行锚）")
    ts.add_argument("sid", help="会话 id（或其唯一前缀）")
    ts.add_argument("--turns", default="", help="回合区间 a-b（缺省全部）")
    ts.add_argument(
        "--verbose", action="store_true", help="附简版正文（折叠思考/工具调用）"
    )
    ts.add_argument(
        "--save", action="store_true", help="落缓存（state/inspector 内，可丢）"
    )
    ts.set_defaults(fn=cmd_transcript_scan)

    tf = tsub.add_parser("fetch", help="按行锚/模式取原始段（有界返回）")
    tf.add_argument("--anchor", default="", help="<sid>.jsonl:L<a>-L<b>")
    tf.add_argument("--sid", default="", help="配合 --grep：在哪个会话里找")
    tf.add_argument("--grep", default="", help="正则模式（配合 --sid）")
    tf.set_defaults(fn=cmd_transcript_fetch)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.fn(args))
    except KeyboardInterrupt:
        print("\n[!] 已中断（发现台账只写已完成的行）")
        return 130


if __name__ == "__main__":
    sys.exit(main())
