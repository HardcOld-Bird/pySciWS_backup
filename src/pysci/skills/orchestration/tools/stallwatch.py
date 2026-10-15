"""worker **静默卡死**的判据与跳级 watchdog（task-20261015-worker-stall，用户点名高危）。

**实案（2026-10-15 08:02）**：devops 队列 worker（``_queue-worker devops``，pid 12960）获锁后
**静默卡死 31+ 分钟**——run 日志 0 字节、零会话登记、零 token、条目假挂 ``in_progress``；
**心跳线程照跳**（:class:`~.backpressure.LiveSession` ≤120s 刷一次），故既有的「pid 判活 /
心跳判活 / 锁龄」三道闸全判它「活着」。完全静默，只能等用户从 API 控制台偶然发现。

本模块补的是那三道闸**结构性看不见**的一维：**进程活着 ≠ 工作有进展**。判据两条（双闸，
任一成立即判卡死）：

1. ``no_spawn``：job 出队后 **N 分钟仍未派生** headless 进程。派生之前全是本地文件逻辑
   （席位解析 / 写任务书 / 组 argv），正常毫秒级；唯一合法的慢是**边界避让**的 ≤300s 主动
   睡（``routing.boundary_hold_s`` 硬钳），故 N 取 600s 仍有 2× 余量。
2. ``no_growth``：**已派生**后会话 jsonl **M 分钟字符零增长**。这是「exe 起来了但不动」的
   那一类（启动挂起 / 卡在握手）。M 取 1800s——**不取更短**是因为单次工具调用可能整段无
   输出（实测单跳 p90=41min），而误杀正常长跳的代价（丢掉整跳工作）远高于晚一步发现。

判死的**动作**由调用方（:mod:`~.queueworker`）落：条目回 ``pending``（复用 ``reset_orphans``
语义）→ 发事件出声 → 杀 worker 进程树。本模块只做**判据与监护**（纯函数 + 一条守护线程），
好让判据在假时钟下可测、动作在注入的杀器下可断言而不真杀。

跨进程面（巡更，见 :func:`patrol_reasons`）另用 **jsonl mtime** 探针：那里拿不到「派生瞬间的
字符水位」基线，而文件最后写入时刻本身就是「有没有进展」的等价读数。
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: ``no_spawn`` 阈值（秒）：job 出队到派生 headless 的上限。见模块 docstring。
NO_SPAWN_S: int = 600

#: ``no_growth`` 阈值（秒）：派生后会话字符零增长的上限。见模块 docstring。
NO_GROWTH_S: int = 1800

#: 守护线程的判定周期（秒）。判死阈值是分钟级，30s 一拍足够及时，也不吵。
PROBE_INTERVAL_S: int = 30

#: 巡更面判据的缺省「在途超期」门槛（秒）：与 ``watch.STALL_THRESHOLD_S`` 同量级。
PATROL_AGE_S: int = 1200


@dataclass
class JobProgress:
    """当前 job 的**进展读数**（watchdog 判据的输入面；全部字段由埋点喂入）。

    ``chars_at`` 是「最后一次被观测到有进展的时刻」——派生那一刻先记一次（起跑线），此后
    每次探测到字符水位上涨就刷新它。于是 ``now - chars_at`` 就是「静默了多久」。
    """

    item_id: str = ""
    member: str = ""
    started_at: float = 0.0  # job 出队时刻
    stage: str = ""  # 最近一次埋点阶段名
    stage_at: float = 0.0
    spawned_at: float | None = None  # headless 派生时刻（None = 还没派生）
    pid: int = 0
    sid: str = ""
    chars: int = 0  # 最近一次测得的会话字符水位
    chars_at: float = 0.0  # 该水位的观测时刻（= 最后一次「有进展」）
    probed: bool = False  # 是否成功探测过一次（探不到时不下 no_growth 判词）
    stages: list[str] = field(default_factory=list)


def job_verdict(
    prog: JobProgress,
    *,
    now: float,
    no_spawn_s: int = NO_SPAWN_S,
    no_growth_s: int = NO_GROWTH_S,
) -> str:
    """判据（**纯函数**，假时钟下可测）：``""`` / ``"no_spawn"`` / ``"no_growth"``。"""
    if prog.started_at <= 0:
        return ""
    if prog.spawned_at is None:
        return "no_spawn" if now - prog.started_at > no_spawn_s else ""
    if not prog.probed:
        return ""  # 探不到字符水位（注册表/文件面不可读）→ 不据此判死
    return "no_growth" if now - prog.chars_at > no_growth_s else ""


def verdict_text(verdict: str, prog: JobProgress, *, now: float) -> str:
    """判词的**人读**文本（进 run 日志/事件 detail，带时长读数）。"""
    if verdict == "no_spawn":
        return (
            f"job {prog.item_id}（{prog.member}）出队后 {(now - prog.started_at) / 60:.0f}min "
            f"仍未派生 headless 进程（末阶段={prog.stage or '?'}）——判卡死"
        )
    if verdict == "no_growth":
        return (
            f"job {prog.item_id}（{prog.member}）已派生（pid={prog.pid}，sid={prog.sid[:8]}）"
            f"但会话 {(now - prog.chars_at) / 60:.0f}min 字符零增长"
            f"（水位 {prog.chars}）——判卡死"
        )
    return ""


def inflight_sessions(member_id: str) -> list[dict[str, Any]]:
    """该成员**在途跳**的会话条目（会话池里带 ``pid`` 标记的那些，见 ``note_hop_pid``）。

    读不到（注册表坏 / 无此成员）→ 空表：判据侧把「查不到在途会话」当成**一种**卡死形态
    （未派生），故此处不抛、也不假装有。
    """
    try:
        from .registry import Registry

        reg = Registry.load()
        sessions = list(reg.member(member_id).sessions or [])
    except Exception:  # noqa: BLE001 - 观察面读失败不阻塞巡更
        return []
    out: list[dict[str, Any]] = []
    for s in sessions:
        try:
            pid = int(s.get("pid") or 0)
        except (TypeError, ValueError):
            pid = 0
        if pid > 0:
            out.append(dict(s))
    return out


def _member(member_id: str) -> Any | None:
    """解析成员条目（读不到 → ``None``：判据侧据此弃权，不据此判死）。"""
    try:
        from .registry import Registry

        return Registry.load().member(member_id)
    except Exception:  # noqa: BLE001 - 注册表读不动不阻塞巡更
        return None


def _session_path(member_id: str, sid: str) -> Path | None:
    """会话 jsonl 路径（成员解析不出 → ``None``）。

    路径约定只走 ``sessions.session_jsonl`` 这一处（与台账 est_tokens 的测量面**同一实现**），
    不自己拼目录——否则测试夹具换掉会话根时两边会读到不同文件。
    """
    member = _member(member_id)
    if member is None or not sid:
        return None
    try:
        from . import sessions as _sessions

        return Path(_sessions.session_jsonl(member, sid))
    except Exception:  # noqa: BLE001
        return None


def session_chars(member_id: str, sid: str) -> int | None:
    """在途会话的字符水位（成员解析不出 → ``None``，**不**回 0 谎报「没长」）。"""
    member = _member(member_id)
    if member is None or not sid:
        return None
    try:
        from . import sessions as _sessions

        return int(_sessions.session_chars(member, sid))
    except Exception:  # noqa: BLE001
        return None


def session_idle_s(member_id: str, sid: str, *, now: float) -> float | None:
    """会话 jsonl **距最后一次写入**的秒数（文件不存在/读不到 → ``None``）。

    跨进程面（巡更）用的就是它：没有「派生瞬间水位」基线可用时，文件 mtime 就是
    「这个会话还在动吗」的等价读数（每回合都落盘）。
    """
    path = _session_path(member_id, sid)
    if path is None:
        return None
    try:
        return max(0.0, float(now) - path.stat().st_mtime)
    except OSError:
        return None


def patrol_reasons(
    member_id: str,
    items: list[dict[str, Any]],
    *,
    now: float,
    age_threshold_s: int = PATROL_AGE_S,
    no_growth_s: int = NO_GROWTH_S,
) -> list[str]:
    """巡更面判据：该成员**在途（in_progress）超期且无进展**的条目 → 疑似卡死文本。

    只在成员**有心跳**（「慢而不死」面）时才有意义——无心跳那支已由 ``watch.queue_pressure``
    的既有 STALL 判据报出，两边合起来才是完整的一维（本函数补的正是心跳照跳却不动的那一小撮）。

    判据（任一成立即报）：
    - 有在途条目却**没有任何带 pid 的在途会话** ⇒ 这一跳根本没派生（实案形态）；
    - 在途会话的 jsonl **超过 ``no_growth_s`` 没被写过** ⇒ 派生出来了但不动。
    """

    def age_of(it: dict[str, Any]) -> float | None:
        raw = it.get("started_at")
        if not raw:
            return None
        try:
            from datetime import datetime

            return float(now) - datetime.fromisoformat(str(raw)).timestamp()
        except (TypeError, ValueError):
            return None

    running = [
        it for it in items if str(it.get("status")) == "in_progress" and age_of(it)
    ]
    if not running:
        return []
    inflight = inflight_sessions(member_id)
    out: list[str] = []
    for it in running:
        age = age_of(it) or 0.0
        if age <= age_threshold_s:
            continue
        iid = str(it.get("id"))
        if not inflight:
            out.append(
                f"{member_id}:{iid} in_progress {age / 60:.0f}min 且无任何在途会话进程"
                "（未派生）——疑似 worker 卡死"
            )
            continue
        idle = [
            session_idle_s(member_id, str(s.get("sid") or ""), now=now)
            for s in inflight
        ]
        known = [x for x in idle if x is not None]
        if known and min(known) > no_growth_s:
            sid8 = str(inflight[0].get("sid") or "")[:8]
            out.append(
                f"{member_id}:{iid} in_progress {age / 60:.0f}min 且在途会话 {sid8} "
                f"{min(known) / 60:.0f}min 无字符增长——疑似 worker 卡死"
            )
    return out


class StallWatch:
    """**跳级 watchdog**：job 起后无进展超时就判卡死（守护线程，跑在 worker 进程内）。

    为什么放在 worker 内而不是只做巡更：实案里队列已空、没人再拉 worker（鸡生蛋），
    外部队列判据根本没机会执行；而卡死的那一刻**唯一的活口**就是这个进程自己的线程。

    ``on_stall(verdict, reason, prog)`` 由调用方给（动作在 :mod:`~.queueworker`：复位条目 +
    发事件 + 杀进程树）。线程是守护线程且**吞掉自身异常**——监护面绝不能把 worker 弄崩。

    测试注入口：``now``（假时钟）、``probe``（假水位）、``start_thread=False`` + :meth:`tick`
    （手工推拍，免真起线程/真等）。
    """

    def __init__(
        self,
        *,
        on_stall: Callable[[str, str, JobProgress], None],
        interval_s: int = PROBE_INTERVAL_S,
        no_spawn_s: int = NO_SPAWN_S,
        no_growth_s: int = NO_GROWTH_S,
        probe: Callable[[str, str], int | None] | None = None,
        now: Callable[[], float] = time.time,
        start_thread: bool = True,
    ) -> None:
        self.prog = JobProgress()
        self._on_stall = on_stall
        self.interval_s = max(1, int(interval_s))
        self.no_spawn_s = int(no_spawn_s)
        self.no_growth_s = int(no_growth_s)
        self._probe = probe or session_chars
        self._now = now
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.armed = False
        self.fired = ""  # 已判出的 verdict（非空即已触发，防重复动作）

    # ---- 生命周期 ---------------------------------------------------------
    def start(self) -> StallWatch:
        if self._thread is None and self.interval_s > 0:
            self._thread = threading.Thread(
                target=self._loop, daemon=True, name="stall-watch"
            )
            self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            try:
                self.tick()
            except Exception:  # noqa: BLE001 - 监护线程绝不炸 worker（观察面）
                pass

    # ---- 埋点入口 ---------------------------------------------------------
    def arm(self, item_id: str, member: str) -> None:
        """一条 job 开始（出队即调用）：起跑线 + 清掉上一件的判词。"""
        self.prog = JobProgress(
            item_id=str(item_id),
            member=str(member),
            started_at=float(self._now()),
        )
        self.armed = True
        self.fired = ""

    def disarm(self) -> None:
        """一条 job 收尾（含异常路径）：不再为它判死。"""
        self.armed = False
        self.prog.started_at = 0.0

    def note(self, stage: str, **fields: Any) -> None:
        """阶段埋点（worker/派发链调用）：刷新阶段名与进展时刻。

        ``spawn`` 阶段额外记下 pid/sid 与会话字符基线（``no_growth`` 判据的起跑线）。
        """
        now = float(self._now())
        p = self.prog
        p.stage = stage
        p.stage_at = now
        p.stages.append(stage)
        if stage == "spawn":
            p.spawned_at = now
            p.chars_at = now
            p.pid = int(fields.get("pid") or 0)
            p.sid = str(fields.get("sid") or "")
            chars = self._probe_safe(p.member, p.sid)
            if chars is not None:
                p.chars = chars
                p.probed = True

    def _probe_safe(self, member: str, sid: str) -> int | None:
        if not sid:
            return None
        try:
            return self._probe(member, sid)
        except Exception:  # noqa: BLE001 - 探针坏掉只丢一条判据，不塌监护
            return None

    # ---- 判定 -------------------------------------------------------------
    def observe(self, *, now: float) -> None:
        """刷新会话水位（有增长即刷新「最后进展时刻」）。"""
        p = self.prog
        if p.spawned_at is None or not p.sid:
            return
        chars = self._probe_safe(p.member, p.sid)
        if chars is None:
            return
        p.probed = True
        if chars > p.chars:
            p.chars = chars
            p.chars_at = float(now)

    def tick(self) -> str:
        """推一拍：探水位 → 判词 → 命中即调 ``on_stall``（返回 verdict；未命中给空串）。"""
        if not self.armed or self.fired:
            return ""
        now = float(self._now())
        self.observe(now=now)
        verdict = job_verdict(
            self.prog, now=now, no_spawn_s=self.no_spawn_s, no_growth_s=self.no_growth_s
        )
        if not verdict:
            return ""
        self.fired = verdict
        self._on_stall(verdict, verdict_text(verdict, self.prog, now=now), self.prog)
        return verdict


def kill_tree(pid: int) -> tuple[bool, str]:
    """杀进程树（**测试注入口**：用例换掉它即可断言「杀了谁」而不真杀）。"""
    from . import procs

    return procs.terminate_tree(pid)


def hard_exit(code: int) -> None:
    """硬退出本进程（**测试注入口**：主线程正卡在 ``communicate`` 里，没有别的出路）。"""
    os._exit(code)


__all__ = [
    "NO_GROWTH_S",
    "NO_SPAWN_S",
    "PATROL_AGE_S",
    "PROBE_INTERVAL_S",
    "JobProgress",
    "StallWatch",
    "hard_exit",
    "inflight_sessions",
    "job_verdict",
    "kill_tree",
    "patrol_reasons",
    "session_chars",
    "session_idle_s",
    "verdict_text",
]
