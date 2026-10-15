"""组员会话池管理（README §3.2）：原生能力的面向 LLM 薄封装。

原生事实（Phase 0 实锤）：会话按 pod cwd 键分储于 ``~/.qoder-cn/projects/<键>/``；
``--list-sessions``/``--delete-session`` 是**跨项目聚合视图且按序号删除（序号漂移）**
——因此池索引以 registry.json 为权威，prune 先按 sid 匹配序号、匹配失败仅做标记。
"""

from __future__ import annotations

import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from . import procs
from .registry import Member, pod_sessions_dir, resolve_exe

_LINE_RE = re.compile(
    r"^\s*(?P<idx>\d+)\.\s+(?P<name>.*?)\((?P<when>[^)]*)\)\s+\[(?P<sid>[0-9a-f-]{36})\]"
)

#: 渠道级故障轮的模型名（20261012-194500-leader 实锤）：上游故障不是真实模型回合，
#: CLI 把它合成一条 assistant 记录写进会话档，``model`` 恒为此值。
SYNTHETIC_MODEL = "<synthetic>"

#: 末轮探针读取的尾部字节窗口。会话档可达数 MB（本跳实测 6.3 M 字符），为一判据全量读
#: 不值；故障轮总在档尾，256 KiB 足够覆盖其后追加的 active-leaf/summary 等簿记行。
TAIL_BYTES = 1 << 18


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


def touch_session(
    member: Member,
    sid: str,
    *,
    hops_delta: int = 1,
    chars_offset: int | None = None,
    ctx_ratio: float | None = None,
) -> None:
    """一跳完成后更新会话元数据（跳数/最后活跃/字符水位/上下文水位）。

    Args:
        member: 成员条目（就地改，落盘由调用方 ``Registry.save`` 负责）。
        sid: 会话 id。
        hops_delta: 本次计入的跳数增量。
        chars_offset: 本跳结束时会话 jsonl 的**字符水位**。持久化它是为了：会话 jsonl
            是唯一能客观反映「这一跳说了多少话」的本地存储，而每跳都在独立进程里跑，
            必须有跨进程的基准点（backlog 20261010-ledger-est-tokens）。水位与本跳实测
            起点不一致 = 期间存在 orch 记账外的增长（手动 --resume 等）。
        ctx_ratio: 本跳结束时的**上下文占用比**（envelope 的 ``context_usage_ratio``）。
            纯观测落池：ctx 是会话态只增不减，不落池就永远只能事后在台账里看到 81%/87%
            那种腰斩读数。**不参与会话选择**——原 FRESH_CTX_RATIO 水位闸经用户审查裁决
            已回退（策略变更不得绕判决通道）；重启类似机制须经判决包并以真实账单/缓存
            读数为依据。
    """
    member.update_session(sid, last_active=_now())
    for s in member.sessions:
        if s.get("sid") == sid:
            s["hops"] = int(s.get("hops", 0)) + hops_delta
            if chars_offset is not None:
                s["chars_offset"] = chars_offset
            if ctx_ratio is not None:
                s["ctx_ratio"] = round(float(ctx_ratio), 4)


def session_jsonl(member: Member, sid: str) -> Path:
    """会话 jsonl 的磁盘路径（存在性 = 会话可 resume 的物理证据）。"""
    return pod_sessions_dir(member.pod) / f"{sid}.jsonl"


def session_chars(member: Member, sid: str) -> int:
    """会话 jsonl 的**字符数**（est_tokens 相对用量尺子的测量面）。

    双渠道（BYOK / 内置）用同一测法 → 同一把尺子，跨档可比（用户裁决 2026-10-10：
    envelope 与 jsonl 逐轮 usage 在双渠道实测全为 0，绝对精度不可得，只做相对排名）。
    中文以原生 UTF-8 落盘（实测 6.28 M 字符仅 6 处 ``\\uXXXX`` 转义），故字符数不被
    转义膨胀扭曲。全量读+解码实测 23 ms（6.3 M 字符），够用即简单——不做字节寻址。

    Returns:
        字符数；**fail-open**：文件缺失/读不动 → 0（新会话首跳本就从 0 起算）。
    """
    path = session_jsonl(member, sid)
    if not path.exists():
        return 0
    try:
        return len(path.read_text(encoding="utf-8", errors="replace"))
    except OSError:
        return 0


def last_assistant_record(member: Member, sid: str) -> dict[str, Any] | None:
    """会话档尾部**最后一条** assistant 记录（读不出 → ``None``，不抛）。

    只读尾部 :data:`TAIL_BYTES`：判据问的是「这一跳是怎么收尾的」，档头几 M 字符的
    历史回合与答案无关。首行可能被窗口切断，故从第二条起取完整行。
    """
    path = session_jsonl(member, sid)
    try:
        size = path.stat().st_size
        with path.open("rb") as fh:
            fh.seek(max(0, size - TAIL_BYTES))
            chunk = fh.read().decode("utf-8", errors="replace")
    except OSError:
        return None
    lines = chunk.splitlines()
    if size > TAIL_BYTES and lines:
        lines = lines[1:]  # 切断的半行不是合法 JSON，留给下一条
    for line in reversed(lines):
        if '"assistant"' not in line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(rec, dict) and rec.get("type") == "assistant":
            msg = rec.get("message")
            return msg if isinstance(msg, dict) else None
    return None


def channel_failure_turn(
    member: Member, sid: str, *, text_re: re.Pattern[str]
) -> bool | None:
    """会话档末轮是否为**渠道级故障轮**——三特征在同一记录内合取。

    判据（2026-10-12 实测，devops 会话 6f0cc8e5 档尾）：上游故障时 CLI 合成的那条
    assistant 记录同时满足 ``model="<synthetic>"``、``stop_reason="stop_sequence"`` 而
    ``stop_sequence`` 为空（自称停在某个序列上、却没有任何序列）、正文是渠道固定句。
    真实模型回合的 ``model`` 是渠道名（实测健康档为 ``qfmodel``），故本判据不会把
    「模型自己说了那句话」的回合误读成故障——这正是文案单特征会误伤的地方。

    Args:
        member: 成员条目（决定会话档所在目录）。
        sid: 会话 id。
        text_re: 渠道固定句模式（由 dispatch 的模式表持有，此处只做读取与合取）。

    Returns:
        ``True``=末轮确为故障轮；``False``=末轮是真实回合（调用方据此**不**升级为
        系统性故障）；``None``=读不出（档缺失/半写/窗口内无 assistant 记录），判据缺席。
    """
    msg = last_assistant_record(member, sid)
    if msg is None:
        return None
    text = " ".join(
        str(c.get("text", ""))
        for c in msg.get("content", [])
        if isinstance(c, dict) and c.get("type") == "text"
    )
    return (
        str(msg.get("model", "")) == SYNTHETIC_MODEL
        and str(msg.get("stop_reason", "")) == "stop_sequence"
        and not msg.get("stop_sequence")
        and bool(text_re.search(text))
    )


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
        proc = procs.run_hidden(
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
        procs.run_hidden(
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
