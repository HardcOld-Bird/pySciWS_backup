"""组员会话池管理（README §3.2）：原生能力的面向 LLM 薄封装。

原生事实（Phase 0 实锤）：会话按 pod cwd 键分储于 ``~/.qoder-cn/projects/<键>/``；
``--list-sessions``/``--delete-session`` 是**跨项目聚合视图且按序号删除（序号漂移）**
——因此池索引以 registry.json 为权威，prune 先按 sid 匹配序号、匹配失败仅做标记。
"""

from __future__ import annotations

import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from .registry import Member, pod_sessions_dir, resolve_exe

_LINE_RE = re.compile(
    r"^\s*(?P<idx>\d+)\.\s+(?P<name>.*?)\((?P<when>[^)]*)\)\s+\[(?P<sid>[0-9a-f-]{36})\]"
)


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def register_session(member: Member, sid: str, name: str) -> None:
    """在成员会话池登记新会话（dispatch 侧调用；registry 落盘由调用方负责）。"""
    member.sessions.append(
        {
            "sid": sid,
            "name": name,
            "hops": 0,
            "created": _now(),
            "last_active": _now(),
            "status": "active",
        }
    )


def touch_session(member: Member, sid: str, *, hops_delta: int = 1) -> None:
    """一跳完成后更新会话元数据（跳数/最后活跃）。"""
    member.update_session(sid, last_active=_now())
    for s in member.sessions:
        if s.get("sid") == sid:
            s["hops"] = int(s.get("hops", 0)) + hops_delta


def session_jsonl(member: Member, sid: str) -> Path:
    """会话 jsonl 的磁盘路径（存在性 = 会话可 resume 的物理证据）。"""
    return pod_sessions_dir(member.pod) / f"{sid}.jsonl"


def format_pool(member: Member, limit: int = 5) -> list[str]:
    """渲染会话池清单（组长派发决策用：名称/跳数/最后活跃/磁盘状态）。

    Args:
        member: 成员条目。
        limit: 最多显示条数（默认策略=继续最近活跃会话，故最近的排最前）。

    Returns:
        显示行列表，首行为默认选择提示。
    """
    active = member.active_sessions
    lines: list[str] = []
    default = active[0]["sid"] if active else None
    lines.append(
        f"会话池（{len(active)} 活跃；默认继续最近会话"
        f"{f'：{default[:8]}' if default else '不存在→将新开会话'}）"
    )
    for s in active[:limit]:
        on_disk = "在盘" if session_jsonl(member, s["sid"]).exists() else "缺档!"
        lines.append(
            f"  - {s['sid'][:8]}  {s.get('name', '(未命名)'):<24} "
            f"跳数={s.get('hops', 0):<3} 最后活跃={s.get('last_active', '?')}  [{on_disk}]"
        )
    archived = [s for s in member.sessions if s.get("status") == "archived"]
    if archived:
        lines.append(f"  （另有 {len(archived)} 个归档会话，--all 查看）")
    return lines


def archive(member: Member, sid: str) -> str:
    """标记会话归档（蒸馏跳由 dispatch 侧先行完成；本函数只改状态）。"""
    sess = member.find_session(sid)
    if sess is None:
        return f"[!] 未找到会话 {sid[:8]}（或缩写有歧义）"
    sess["status"] = "archived"
    sess["archived_at"] = _now()
    return f"[√] 会话 {sid[:8]}（{sess.get('name', '')}）已归档"


def prune(member: Member, sid: str) -> str:
    """删除会话：解析 --list-sessions 聚合列表按 sid 匹配序号后 --delete-session。

    匹配失败（序号漂移/已不存在）时只做 registry 标记，不删文件——原生序号删除
    不可靠（Phase 0 坑位记录），fail-open 优先。
    """
    exe = resolve_exe()
    try:
        proc = subprocess.run(
            [str(exe), "--cwd", str(member.pod), "--list-sessions"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
        )
        idx: str | None = None
        for line in proc.stdout.splitlines():
            m = _LINE_RE.match(line)
            if m and m.group("sid") == sid:
                idx = m.group("idx")
                break
        if idx is None:
            member.update_session(sid, status="pruned")
            return (
                f"[!] 聚合列表未匹配到 {sid[:8]}，仅在 registry 标记 pruned（不删文件）"
            )
        subprocess.run(
            [str(exe), "--cwd", str(member.pod), "--delete-session", idx],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        return f"[!] 原生删除失败（{exc}），仅在 registry 标记 pruned"
    member.update_session(sid, status="pruned")
    return f"[√] 会话 {sid[:8]} 已删除（原生 delete-session 序号 {idx}）"


def pool_stats(member: Member) -> dict[str, Any]:
    """池摘要（status 命令用）。"""
    active = member.active_sessions
    return {
        "active": len(active),
        "archived": len([s for s in member.sessions if s.get("status") == "archived"]),
        "latest": active[0] if active else None,
    }
