"""``pysci-mirror`` 用户控制台骨架（REFACTOR §10，E6-prep）。

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
from . import jury
from .registry import Registry
from .taste import PREFERENCE_NAME, TASTE_DIR

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
    """宪章状态读数：存在性、落盘字节数、是否仍标「草案」（生效确认权归用户）。"""
    p = charter_path(reg)
    if not p.is_file():
        return {"exists": False, "bytes": 0, "draft": True, "path": p}
    text = p.read_text(encoding="utf-8")
    return {
        "exists": True,
        "bytes": len(text.encode("utf-8")),
        "draft": "草案" in text,
        "path": p,
    }


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
    """已表态、待盲审的 twin 判断数（``state/evals/taste/*.json``）。"""
    return len(sorted(TASTE_DIR.glob("*.json"))) if TASTE_DIR.is_dir() else 0


# ---------------------------------------------------------------- status 概览项


def _line_review() -> str:
    n = taste_cases()
    return f"待盲审案件数 = {n}" + (
        "〔盲审即训练，review 面属 E6 后续单〕" if n else "（twin 判断留痕尚未产生）"
    )


def _line_charter() -> str:
    c = charter_state()
    if not c["exists"]:
        return f"宪章缺失：{_rel(c['path'])}（twin 无依据不判断）"
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
        "  偏差报告队列 = 占位（E6 后续单）"
    )


def _line_jury() -> str:
    """仲裁团席位健康（REFACTOR §4 逐席直调）：主席不可达即 ``[!]`` 点名并宣告回退上场。

    只读 registry 与 Key 就位性——不开庭、不发请求、不打 Key 值。
    """
    return jury.readiness_summary(jury.seat_readiness())


def _line_fallback() -> str:
    """渠道回退事件（用户裁决 2026-10-12 21:1x 规格 4b）：计数 + 最近几条。

    读事件流不判成败；空流也要出一行「尚无回退」——「没坏」与「没人记账」必须可辨。
    """
    return _channels.summary_line()


def _line_trend() -> str:
    return "一致率 / override 率趋势 = 占位（overrides·stats 属 E6 后续单）"


def _line_pending() -> str:
    return "待决事项 = minimum 压测档型号选型（用户）；twin 上线宣告（用户）"


def _line_acceptance() -> str:
    return "待用户验收的最终交付 = 占位（E6 后续单）"


#: 概览项登记表（§10 注册纪律·硬要求）：任何新增「需用户关注事项类型」必须加进这里。
STATUS_ITEMS: tuple[tuple[str, Callable[[], str]], ...] = (
    ("盲审", _line_review),
    ("宪章", _line_charter),
    ("开关", _line_switches),
    ("队列", _line_queues),
    ("仲裁", _line_jury),
    ("回退", _line_fallback),
    ("趋势", _line_trend),
    ("待决", _line_pending),
    ("验收", _line_acceptance),
)


def status_lines() -> list[str]:
    """全景读数（``pysci-mirror status`` 打印；每项独立容错，坏一项不塌全貌）。"""
    out: list[str] = []
    for name, fn in STATUS_ITEMS:
        try:
            out.append(f"  [{name}] {fn()}")
        except Exception as exc:  # noqa: BLE001 - 观察面绝不阻塞生产（§0.6）
            out.append(f"  [{name}] 读数失败：{type(exc).__name__}: {exc}")
    return out


def cmd_status(args: argparse.Namespace) -> int:
    """``pysci-mirror status``：用户侧唯一关注入口（§10 硬要求）。"""
    print("== mirror 全景（用户侧唯一关注入口，REFACTOR §10）==")
    for line in status_lines():
        print(line)
    if not Registry.load().taste_enabled:
        print(
            "  → twin 通道休眠：确认 PREFERENCE.md 草案后置 taste_enabled=true 才对外服务"
        )
    print('[NEXT] 口述偏好或指令：pysci-mirror journal "……"（编号菜单分流）')
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
    """``pysci-mirror journal <文本> [--select 1,2]``：写入时机械分流（§10）。"""
    text = " ".join(args.text).strip() if args.text else ""
    if not text:
        print('✗ 缺少口述内容：pysci-mirror journal "……"')
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
        prog="pysci-mirror",
        description="用户控制台（REFACTOR §10）：status 全景 + journal 口述分流",
    )
    sub = ap.add_subparsers(dest="action", required=True)
    p = sub.add_parser(
        "status", help="全景概览（待盲审/宪章/开关/队列/仲裁席位/待决事项）"
    )
    p.set_defaults(func=cmd_status)
    p = sub.add_parser("journal", help="口述内容编号分流：①偏好→日志 ②指令→强制队列")
    p.add_argument("text", nargs="*", help="口述正文（可多词，合并为一行）")
    p.add_argument(
        "--select",
        help="跳过菜单直接分流：1=偏好 2=行动指令 1,2=两者（非交互环境必填）",
    )
    p.set_defaults(func=cmd_journal)
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
    "cmd_journal",
    "cmd_status",
    "main",
    "pending_directives",
    "read_queue",
    "route",
    "status_lines",
    "taste_cases",
]
