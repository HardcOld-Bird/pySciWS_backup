"""``pysci`` 用户控制台——**LLM 静态通道**（README §4.1/§6、§12 M7）。

`pysci` 是用户的唯一入口：**人类交互正道 = 裸 `pysci` 进 TUI**（`tools/tui/`）；
本模块承载其**静态子命令面**（``status`` / ``journal`` / ``decisions``），面向
**LLM / 脚本**（非 tty 裸调用亦回落到 `status`，见 `tui/app.py` 入口分派）——输出
向机器可读优化：稳定的 ``[key] value`` 行（:data:`STATUS_ITEMS` 顺序即契约）+ 语义化
退出码（0=正常、2=用法/输入错）。人类不需要背这些子命令。

用户侧「唯一关注入口」：``status`` 一次看全需要用户知道的事，``journal`` 将口述内容
机械分流。CLI 人体工学按 §0.9 范本（Qoder CLI / commitizen）：自然语言 + 编号菜单，
**永不要求用户打路径**；输出末行给 [NEXT]。

骨架范围（机械面先行，生效确认与上线宣告留用户）：``review``（盲审）/``distill``/
``overrides``/``stats`` 留 E6 后续单，本模块在 ``status`` 里为它们保留概览项位置并
如实标注「占位」。

分流法理（§5.4/§10）：① **偏好陈述** → 决策日志作训练语料，不进处理队列（不要求
行动）；② **行动指令** → 强制处理队列，组长必须写任务书派 devops、**不得 override**
（用户明确要求即授权）。队列读数是 ``orch status`` 的可见性来源——不入组长视野的
队列等于没有队列。

路径均为模块级常量且在函数内读全局（monkeypatch 可隔离，同 :mod:`journal`）。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from pysci.paths import ORCH_STATE_ROOT, PODS_ROOT, PROJECT_ROOT

from . import channels as _channels
from . import journal as _journal
from . import jury, routing
from .registry import Registry
from .taste import PREFERENCE_NAME

#: 行动指令队列（§5.4「强制派发」同等待遇）。与 backlog.json 同族**入库跟踪**：
#: 用户指令是授权凭证，「先决策后问责」（§0.4）要求它比一次会话活得更久。
MIRROR_QUEUE_PATH: Path = ORCH_STATE_ROOT / "mirror-queue.jsonl"

#: 分流选项编号（菜单文案与 ``--select`` 取值共用，防文案漂移）。
CHOICE_PREFERENCE = 1
CHOICE_DIRECTIVE = 2
MENU_ITEMS: tuple[tuple[int, str, str], ...] = (
    (
        CHOICE_PREFERENCE,
        "偏好陈述",
        "→ 决策日志（preference-journal.jsonl）作 twin 训练语料；不要求行动",
    ),
    (
        CHOICE_DIRECTIVE,
        "行动指令",
        "→ 强制处理队列（mirror-queue.jsonl）；组长须写任务书派 devops，不得 override",
    ),
)


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _rel(p: Path) -> str:
    """打印用相对路径；被 monkeypatch 到 tmp 时原样显示（不炸）。"""
    try:
        return str(p.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(p)


def charter_path(reg: Registry | None = None, member_id: str = "twin") -> Path:
    """宪章落点（twin pod 内）；registry 未注册该成员时回落到 PODS_ROOT 约定位置。"""
    reg = reg or Registry.load()
    try:
        pod = reg.member(member_id).pod
    except KeyError:
        pod = PODS_ROOT / member_id
    return pod / PREFERENCE_NAME


def charter_state(reg: Registry | None = None) -> dict[str, Any]:
    """宪章状态读数：存在性、落盘字节数、**结构化状态标记**（生效确认权归用户）。

    判据改**读顶部状态行**而非裸搜「草案」（backlog 20261013-192000-leader）：v2 顶注里有
    「审阅 v2 增量草案并同意转正」这样的**叙述性**用词，逐文搜「草案」会把它误判成未生效
    （实测：文件已是 v2 生效 8171 B，mirror 却报「草案 v1，待用户确认生效」）。约定标记形如
    ``> **状态：v2 生效**``；解析不到标记时**保底**回落原子串判据——无标记的旧文件不得因此
    被当成已生效。
    """
    p = charter_path(reg)
    if not p.is_file():
        return {"exists": False, "bytes": 0, "status": "", "draft": True, "path": p}
    text = p.read_text(encoding="utf-8")
    status = _charter_status(text)
    draft = ("草案" in status) if status else ("草案" in text)
    return {
        "exists": True,
        "bytes": len(text.encode("utf-8")),
        "status": status,
        "draft": draft,
        "path": p,
    }


def _charter_status(text: str) -> str:
    """取顶部状态行里的状态串（``v2 生效`` / ``v1 草案``…）；解析不到 → 空串。

    只认**结构化标记**：``**状态：<串>**``。扫前若干行即可（状态行在顶注里），不全文搜——
    正文里出现「草案」是叙述，不是状态。故意不用正则：这里要的字面是 ``**`` 与全角冒号，
    用 split 更直白且不给转义留坑。
    """
    for line in text.splitlines()[:12]:
        if "状态" not in line:
            continue
        tail = line.split("状态", 1)[1].lstrip("：:* ")
        if "**" in tail:
            return tail.split("**", 1)[0].strip()
    return ""


def read_queue(path: Path | None = None) -> list[dict[str, Any]]:
    """队列全量读回（缺文件即空表；坏行跳过——观察面 fail-soft，不堵派发链）。"""
    target = path if path is not None else MIRROR_QUEUE_PATH
    if not target.exists():
        return []
    out: list[dict[str, Any]] = []
    for raw in target.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        try:
            rec = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(rec, dict):
            out.append(rec)
    return out


def pending_directives(path: Path | None = None) -> list[dict[str, Any]]:
    """未处理的行动指令（``status != "done"``）——组长侧可见性的读数来源。"""
    return [r for r in read_queue(path) if r.get("status") != "done"]


def append_directive(text: str, *, path: Path | None = None) -> dict[str, Any]:
    """行动指令入队（追加一行 JSON；不进决策日志）。"""
    target = path if path is not None else MIRROR_QUEUE_PATH
    rec = {"ts": _now(), "text": text, "status": "pending", "source": "mirror"}
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


def append_preference(text: str, *, path: Path | None = None) -> dict[str, Any]:
    """偏好陈述入决策日志（复用 :func:`journal.append`，不另造一套行格式）。"""
    return _journal.append(
        {"situation": text, "source": "mirror", "kind": "preference"}, path=path
    )


def taste_cases() -> int:
    """twin 判断/裁决留痕数（``state/evals/taste/*.json``：盲审 taste + consult 代裁同目录）。

    目录取 **taste 模块的活属性**（非本模块 import 时的快照）——测试/隔离面 monkeypatch
    ``taste.TASTE_DIR`` 即刻生效（同「路径取模块级、函数内读全局」纪律）。
    """
    from . import taste as _taste

    base = _taste.TASTE_DIR
    return len(sorted(base.glob("*.json"))) if base.is_dir() else 0


def twin_ruling_lines(limit: int = 8) -> list[str]:
    """twin 判断/裁决**逐条明细**（审计面：夜航代裁须事后可复核；数据层纯函数）。

    ``pysci`` TUI「盲审与审查」详情与 `orch twin-review --list` 共用（同一能力不留两份
    实现：拼装在 :func:`taste.ruling_lines`，本函数只做数据层转发）。
    """
    from . import taste

    return taste.ruling_lines(limit=limit)


# ---------------------------------------------------------------- status 概览项


def _line_review() -> str:
    """twin 判断/裁决追溯读数（PSEP_2 配套：夜航代裁须事后可复核，含待复核计数）。"""
    from . import taste

    n = taste_cases()
    if not n:
        return "twin 判断/裁决 = 0（留痕尚未产生）"
    pending = taste.pending_review_count()
    return (
        f"twin 判断/裁决 = {n}（待用户复核 {pending}；明细 `orch twin-review --list`）"
    )


def _line_charter() -> str:
    c = charter_state()
    if not c["exists"]:
        return f"宪章缺失：{_rel(c['path'])}（twin 无依据不判断）"
    if c["status"]:
        state = (
            f"待确认 {c['status']}（草案态）" if c["draft"] else f"已生效 {c['status']}"
        )
    else:
        # 无结构化标记的旧文件：保留原文案（别把「没标记」读成「未生效」）
        state = "草案 v1，待用户确认生效" if c["draft"] else "已确认（草案标记已去）"
    return f"宪章 = {state}（{_rel(c['path'])}，{c['bytes']} B）"


def _line_switches() -> str:
    from .inspectd import (
        enabled as inspector_enabled,  # 延迟导入：inspectd 引整条 worker 链
    )

    on = Registry.load().taste_enabled
    insp = inspector_enabled()
    return (
        f"twin 判断通道 taste_enabled = {on}"
        f"{'（已生效）' if on else '（未生效：orch taste 拒绝派发）'}  "
        f"inspector 开关 = {insp}"
        + (
            "（默认态：orch 命令自动拉起巡检）"
            if insp
            else "（已关闭：拉起逻辑 no-op，orch inspector-on 恢复）"
        )
    )


def _line_queues() -> str:
    n_dir = len(pending_directives())
    jury_q = ORCH_STATE_ROOT / "evals" / "jury-queue"
    n_jury = len(sorted(jury_q.glob("*.json"))) if jury_q.is_dir() else 0
    return (
        f"行动指令队列待处理 = {n_dir}  jury 待重裁 = {n_jury}"
        "  偏差报告队列 = 后续单（twin 偏差报告动线，属审查面）"
    )


def _line_routing() -> str:
    """路由模式（路由 v3 规格 4）：当前生效预设、是否隐性 DeepSeek 态、判据读数、下次切换。

    隐性态是**时间驱动的覆盖**：toml 那一行没改而烧的渠道改了。用户只看 ``[model] preset``
    是看不出这件事的，所以它必须进「唯一关注入口」，而不是只进 ``orch status``。
    """
    return routing.summary_line()


def _line_jury() -> str:
    """仲裁团席位健康（README §5.3 逐席直调）：主席不可达即 ``[!]`` 点名并宣告回退上场。

    只读 registry 与 Key 就位性——不开庭、不发请求、不打 Key 值。
    """
    return jury.readiness_summary(jury.seat_readiness())


def _line_fallback() -> str:
    """渠道回退事件（用户裁决 2026-10-12 21:1x 规格 4b）：计数 + 最近几条。

    读事件流不判成败；空流也要出一行「尚无回退」——「没坏」与「没人记账」必须可辨。
    """
    return _channels.summary_line()


def _line_trend() -> str:
    return "一致率 / override 率趋势 = 后续单（属决策审计面：overrides/stats）"


#: 待决事项清单落点（gitignored 运行态：它是「此刻谁在等用户」的活体读数，不是长期
#: 资产；写它的是组长/用户的裁决动线）。
PENDING_PATH: Path = ORCH_STATE_ROOT / "mirror-pending.json"


def pending_items(path: Path | None = None) -> list[str]:
    """读待决清单（``["…", …]`` 或 ``{"items": [...]}``）；缺失/损坏 → 空表（fail-open）。"""
    target = PENDING_PATH if path is None else path
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    items = raw.get("items") if isinstance(raw, dict) else raw
    if not isinstance(items, list):
        return []
    return [str(x).strip() for x in items if str(x).strip()]


def _line_pending(path: Path | None = None) -> str:
    """待决事项：**数据驱动**（backlog 20261013-192000-leader）。

    原实现把两项历史待决（minimum 选型 / twin 宣告）硬编码在代码里，两项都已解决、读数却
    原地不动。真源＝``state/mirror-pending.json``（组长/用户增删），无真源或为空即显
    「（无待决）」——**不许**在代码里留历史项冒充现状。
    """
    items = pending_items(path)
    return "待决事项 = " + ("；".join(items) if items else "（无待决）")


def _line_acceptance() -> str:
    return "待用户验收的最终交付 = 后续单（用户验收动线，属任务书面）"


def _line_inspector_decisions() -> str:
    """inspector 决策水位（用户 2026-10-13 06:2x 裁决：重要决策必须汇总进概览）。

    本条目就是「免判决自动晋升」的教训面——ctx 水位闸类决策若早在这里亮出来，
    用户审计就不必等事后联系组长。免判决+change>0 时置顶高亮。
    """
    from .inspectd import decision_counts  # 延迟导入：inspectd 引整条 worker 链

    c = decision_counts()
    hot = (
        f"〔重点〕免判决+change={c['free_change']} 待审计"
        if c["free_change"]
        else "免判决+change=0"
    )
    return (
        f"inspector 决策：自动晋升={c['promoted']} 待审计={c['open']} {hot} "
        f"判决类={c['judged']}（全貌 `pysci decisions`）"
    )


#: 概览项登记表（§10 注册纪律·硬要求）：任何新增「需用户关注事项类型」必须加进这里。
def _line_defer() -> str:
    """带宽延期读数（用户裁决 2026-10-13；§3.3/M6 推广到**全部队列**）。

    带宽类失败改「延期而非停链」之后，用户在概览上必须能看见「有活压着等」——否则
    「队列看着空、其实有条目在退避」就是一块盲区。devops backlog 与**成员队列**两条链
    各挂各的 defer_until，读数合并呈现（bencher 无写侧，不入此计）。
    """
    from . import queues as _q  # 延迟导入：mirror 是读数层，queues 是运行态数据层
    from . import workflow as _wf  # 延迟导入：workflow 引 dispatch，顶层引会成环

    n = _wf.deferred_count() + _q.deferred_count()
    if not n:
        return "带宽延期 = （无）"
    waits = [w for w in (_wf.next_defer_s(), _q.next_defer_s()) if w > 0]
    wait = min(waits) if waits else 0.0
    return f"带宽延期 = {n} 条在等，最近 {wait:.0f}s 后到期"


STATUS_ITEMS: tuple[tuple[str, Callable[[], str]], ...] = (
    ("盲审", _line_review),
    ("宪章", _line_charter),
    ("开关", _line_switches),
    ("队列", _line_queues),
    ("路由", _line_routing),
    ("仲裁", _line_jury),
    ("回退", _line_fallback),
    ("决策", _line_inspector_decisions),
    ("延期", _line_defer),
    ("趋势", _line_trend),
    ("待决", _line_pending),
    ("验收", _line_acceptance),
)


def status_lines() -> list[str]:
    """全景读数（``pysci status`` 打印；每项独立容错，坏一项不塌全貌）。"""
    out: list[str] = []
    for name, fn in STATUS_ITEMS:
        try:
            out.append(f"  [{name}] {fn()}")
        except Exception as exc:  # noqa: BLE001 - 观察面绝不阻塞生产（§0.6）
            out.append(f"  [{name}] 读数失败：{type(exc).__name__}: {exc}")
    return out


def cmd_status(args: argparse.Namespace) -> int:
    """``pysci status``：用户侧唯一关注入口（§10 硬要求）——**LLM 静态通道**。

    输出为稳定的 ``[key] value`` 行（顺序即 :data:`STATUS_ITEMS` 契约），供 LLM/脚本逐行
    解析；人类无需敲它（裸 `pysci` 进 TUI 即得同一批读数的交互面）。退出码恒 0（读数面
    单项容错，坏一项不塌全貌）。
    """
    print("== pysci 全景 [LLM 通道]（用户侧唯一关注入口，README §4.1/§6）==")
    for line in status_lines():
        print(line)
    if not Registry.load().taste_enabled:
        print(
            "  → twin 通道休眠：确认 PREFERENCE.md 草案后置 taste_enabled=true 才对外服务"
        )
    print('[NEXT] 口述偏好或指令：pysci journal "……"（编号菜单分流）')
    return 0


# ---------------------------------------------------------------- decisions
def decisions_detail_lines(
    *, days: int = 0, cls: str = "", shape: str = ""
) -> list[str]:
    """inspector 决策链全貌的**只读拼装**（README §12 数据层纯函数）。

    ``pysci decisions`` 与 TUI「决策审计」详情视图**共用**本函数——同一能力不留两份实现。
    免判决+change 类置顶（最需审计者优先）：发现 → class → remedy_shape → 判决记录 →
    实施 commit → 核验状态一屏看全。零新状态面。
    """
    from .inspectd import decision_rows  # 延迟导入：inspectd 引整条 worker 链

    rows = decision_rows(days=days, cls=cls, shape=shape)
    if not rows:
        return ["（无匹配的 inspector 决策记录）"]
    out = [
        f"== inspector 决策全貌（{len(rows)} 条；只读拼装：findings × backlog "
        "× run 摘要 × jury 案卷 × 决策日志）=="
    ]
    for r in rows:
        flag = "[!]" if r["attention"] else "[i]"
        bl = r["backlog_status"] or "未晋升"
        out.append(
            f"{flag} {r['id']}  {r['ts']}  "
            f"[{r['class']}/remedy_shape={r['remedy_shape']}]"
        )
        out.append(f"     发现：{str(r['title'])[:80]}")
        ruling = f"｜{r['ruling'][:60]}" if r["ruling"] else ""
        out.append(f"     判决：{r['verdict_ref']}{ruling}")
        impl = f"实施：{r['commit'][:8]}" if r["commit"] else "实施：（无 commit 读数）"
        verify = f"核验：{r['verify']}" if r["verify"] else f"backlog={bl}"
        out.append(f"     {impl}  {verify}")
        if r.get("note"):
            out.append(f"     留痕：{str(r['note'])[:90]}")
    if any(r["attention"] for r in rows):
        out.append(
            "  [!] 置顶项=免判决+change（未经判决改变了声明行为/策略）——优先审计；"
            "改判后应补 J/twin 判决记录留痕"
        )
    out.append(
        "[NEXT] 过滤：--days N（近 N 天）/ --class mechanical|judgment / "
        "--shape restore|change"
    )
    return out


def decisions_count() -> int:
    """决策记录条数（TUI 更新徽标计数用；与 :func:`decisions_detail_lines` 同一数据源）。"""
    from .inspectd import decision_rows

    try:
        return len(decision_rows())
    except Exception:  # noqa: BLE001 - 观察面绝不阻塞生产（§0.6）
        return 0


def cmd_decisions(args: argparse.Namespace) -> int:
    """``pysci decisions``：inspector 决策链全貌（用户审计入口，§10 注册纪律）。"""
    for line in decisions_detail_lines(
        days=int(args.days or 0), cls=args.cls or "", shape=args.shape or ""
    ):
        print(line)
    return 0


# ---------------------------------------------------------------- journal 分流


def _parse_select(raw: Any) -> list[int]:
    """把 ``1`` / ``2`` / ``1,2`` 解析成编号表（非法 token 忽略、去重保序）。"""
    picks: list[int] = []
    for tok in str(raw).replace("，", ",").split(","):
        tok = tok.strip()
        if tok.isdigit() and int(tok) in (CHOICE_PREFERENCE, CHOICE_DIRECTIVE):
            if int(tok) not in picks:
                picks.append(int(tok))
    return picks


def _menu() -> list[int]:
    """交互菜单（回车=只选①）。非 tty 不阻塞等输入，直接要求 ``--select``。"""
    print("这条内容是——")
    for num, label, hint in MENU_ITEMS:
        print(f"  {num}) {label}：{hint}")
    if not sys.stdin.isatty():
        print("✗ 非交互环境：用 --select 1|2|1,2 明示分流（不猜）")
        return []
    try:
        raw = input("选择编号（如 1,2；回车=1）：").strip() or "1"
    except EOFError:
        # tty 但没有键盘（agent 会话里 stdin 常是空管道）：与「非 tty」同一处置——
        # 不猜编号、不写文件，把选择权交回 --select。抛出 traceback 会让人以为崩了。
        print("✗ 读不到输入（stdin 无内容）：用 --select 1|2|1,2 明示分流（不猜）")
        return []
    picks = _parse_select(raw)
    if not picks:
        print("✗ 没有有效编号（只接受 1 / 2），未写入任何内容")
    return picks


def route(text: str, selects: list[int]) -> dict[str, Any]:
    """按编号机械分流并落盘；返回各项落点（未选中的为 None）。"""
    out: dict[str, Any] = {"text": text, "preference": None, "directive": None}
    if CHOICE_PREFERENCE in selects:
        out["preference"] = append_preference(text)
    if CHOICE_DIRECTIVE in selects:
        out["directive"] = append_directive(text)
    return out


def cmd_journal(args: argparse.Namespace) -> int:
    """``pysci journal <文本> [--select 1,2]``：写入时机械分流（§10）。"""
    text = " ".join(args.text).strip() if args.text else ""
    if not text:
        print('✗ 缺少口述内容：pysci journal "……"')
        return 2
    selects = _parse_select(args.select) if args.select else _menu()
    if not selects:
        return 2
    r = route(text, selects)
    if r["preference"]:
        print(f"  ① 偏好陈述 → {_rel(_journal.JOURNAL_PATH)}（kind=preference）")
    if r["directive"]:
        print(f"  ② 行动指令 → {_rel(MIRROR_QUEUE_PATH)}（status=pending，强制派发面）")
    print(
        "[NEXT] "
        + (
            "组长 orch status 已可见该行动指令，须写任务书派 devops（不得 override）"
            if r["directive"]
            else "已作训练语料留存，不要求行动"
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="pysci",
        description=(
            "pysci 用户控制台 · **LLM 静态通道**（README §4.1/§6；人类交互请裸跑 `pysci` 进 TUI）："
            "status 全景 + journal 口述分流 + decisions 决策审计"
        ),
        epilog="输出为稳定的 `[key] value` 行（机器可读）；退出码：0=正常、2=用法/输入错。",
    )
    sub = ap.add_subparsers(dest="action", required=True)
    p = sub.add_parser(
        "status", help="[LLM 通道] 全景概览（待盲审/宪章/开关/队列/仲裁席位/待决事项）"
    )
    p.set_defaults(func=cmd_status)
    p = sub.add_parser(
        "journal", help="[LLM 通道] 口述内容编号分流：①偏好→日志 ②指令→强制队列"
    )
    p.add_argument("text", nargs="*", help="口述正文（可多词，合并为一行）")
    p.add_argument(
        "--select",
        help="跳过菜单直接分流：1=偏好 2=行动指令 1,2=两者（非交互环境必填）",
    )
    p.set_defaults(func=cmd_journal)
    p = sub.add_parser(
        "decisions",
        help="[LLM 通道] inspector 决策链全貌（发现→class→remedy_shape→判决→commit→核验）",
    )
    p.add_argument("--days", type=int, default=0, help="只看近 N 天（0=不限）")
    p.add_argument(
        "--class",
        dest="cls",
        default="",
        choices=["mechanical", "judgment"],
        help="按发现类别过滤",
    )
    p.add_argument(
        "--shape",
        default="",
        choices=["restore", "change"],
        help="按修复形状过滤",
    )
    p.set_defaults(func=cmd_decisions)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


__all__ = [
    "CHOICE_DIRECTIVE",
    "CHOICE_PREFERENCE",
    "MENU_ITEMS",
    "MIRROR_QUEUE_PATH",
    "STATUS_ITEMS",
    "append_directive",
    "append_preference",
    "charter_path",
    "charter_state",
    "cmd_decisions",
    "cmd_journal",
    "cmd_status",
    "decisions_count",
    "decisions_detail_lines",
    "main",
    "pending_directives",
    "read_queue",
    "route",
    "status_lines",
    "taste_cases",
]
