"""背压原语：活跃会话注册表 + 延期纯函数库（backlog 20261013-193000-leader）。

**这是「带宽治理」的安全半**（原 #115 拆单 1/2）：只提供可测的原语，**不接 drain 主循环、
不改变任何现有行为**——现有 `channel_error` 全局停止语义原样保留（全量含该用例照绿即是
零行为变更的机械证明）。接线（背压优先于付费改道、睡到到期续跑、#105 分流）属拆单 2/2。

两件东西：

- **活跃会话注册表**（``state/active-sessions.json``）：谁在跑、跑哪个档、为什么跑。
  写侧由各 worker 在跳开始/结束调用，读侧供调度面判断「有没有更高优先级的活会话」。
  **>10min 无更新视为死条目**（:data:`STALE_S`）——它是「此刻谁在跑」的活体读数，
  与 pid 判活**互校**：两者不一致时以 pid 为准（本模块只给读数，不做杀伐）。
- **延期纯函数库**：``bandwidth`` 类失败的处置是**等**，等的时长按指数退避
  （2→4→8…分钟），并有**双闸**——次数 ≤ :data:`DEFER_MAX_ATTEMPTS`、累计 ≤
  :data:`DEFER_BUDGET_S`。纯函数：不读时钟（``now`` 由调用方给），不落盘。
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from pysci.paths import ORCH_STATE_ROOT

#: 活跃会话注册表落点（gitignored 运行态：心跳跳跳变，写进跟踪面会堵合并）。
REGISTRY_PATH: Path = ORCH_STATE_ROOT / "active-sessions.json"

#: 心跳过期阈值（秒）：超过它没更新的条目视为**死条目**（worker 被杀/进程没了）。
STALE_S: int = 600  # 10min

#: worker 心跳周期（秒）：**显著小于** :data:`STALE_S`（判死阈值 10min）——§3.3 建议 ≤3min。
#: 长 job 阻塞在 ``do_dispatch`` 里时由 :class:`LiveSession` 的守护线程按此周期刷心跳，
#: 故 >10min 的 job 也不会被判死。
HEARTBEAT_INTERVAL_S: int = 120

#: 延期退避基数（秒）：第一次延 2 分钟，其后翻倍。
DEFER_BASE_S: int = 120

#: 延期次数闸：单项最多延这么多次（超了走回退链或上报组长）。
DEFER_MAX_ATTEMPTS: int = 6

#: 延期**壁钟**预算闸（秒）：累计等待封顶（次数闸挡不住「真下线」——每次失败耗时数分钟时
#: 次数还没用完，队列已被拖住；双闸并用才是完整判据）。
DEFER_BUDGET_S: int = 1800  # 30min


# ---------------------------------------------------------------------------
# 活跃会话注册表
# ---------------------------------------------------------------------------
def _iso(epoch: float) -> str:
    return (
        datetime.fromtimestamp(float(epoch)).astimezone().isoformat(timespec="seconds")
    )


def _epoch(text: Any) -> float:
    try:
        return datetime.fromisoformat(str(text)).timestamp()
    except (TypeError, ValueError):
        return 0.0


def read_registry(path: Path | None = None) -> list[dict[str, Any]]:
    """读注册表条目表；缺失/损坏/形状不对 → 空表（**fail-open**：观察面不堵派发链）。"""
    target = REGISTRY_PATH if path is None else path
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    items = raw.get("sessions") if isinstance(raw, dict) else raw
    if not isinstance(items, list):
        return []
    return [x for x in items if isinstance(x, dict)]


def write_registry(entries: list[dict[str, Any]], path: Path | None = None) -> None:
    """整表落盘（临时文件 + 原子替换：半写的注册表会让读侧把在跑的会话当空气）。"""
    target = REGISTRY_PATH if path is None else path
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text(
        json.dumps({"version": 1, "sessions": entries}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    tmp.replace(target)


def _key(entry: dict[str, Any]) -> tuple[str, str]:
    return str(entry.get("worker", "")), str(entry.get("sid", ""))


def note_session(
    worker: str,
    *,
    sid: str,
    tier: str,
    purpose: str,
    now: float,
    path: Path | None = None,
) -> dict[str, Any]:
    """登记/更新一个活跃会话（跳开始与跳结束都调它）。

    已存在（同 worker+sid）则刷新 ``updated`` 并覆盖 tier/purpose，``started`` 保留
    首次值——这样「这一跳跑了多久」不被后续心跳抹掉。
    """
    entries = read_registry(path)
    stamp = _iso(now)
    record = {
        "worker": str(worker),
        "tier": str(tier),
        "purpose": str(purpose),
        "sid": str(sid),
        "started": stamp,
        "updated": stamp,
    }
    for entry in entries:
        if _key(entry) == _key(record):
            record["started"] = str(entry.get("started") or stamp)
            entry.update(record)
            write_registry(entries, path)
            return dict(entry)
    entries.append(record)
    write_registry(entries, path)
    return record


def heartbeat(worker: str, sid: str, *, now: float, path: Path | None = None) -> bool:
    """只刷新 ``updated``（跳内推进）；命中返回 True，没这条会话返回 False。"""
    entries = read_registry(path)
    stamp = _iso(now)
    hit = False
    for entry in entries:
        if _key(entry) == (str(worker), str(sid)):
            entry["updated"] = stamp
            hit = True
    if hit:
        write_registry(entries, path)
    return hit


def drop_session(worker: str, sid: str, *, path: Path | None = None) -> bool:
    """摘掉一个会话（跳正常收尾时调）；摘到返回 True。"""
    entries = read_registry(path)
    kept = [e for e in entries if _key(e) != (str(worker), str(sid))]
    if len(kept) == len(entries):
        return False
    write_registry(kept, path)
    return True


def live_entries(
    *, now: float, stale_s: int = STALE_S, path: Path | None = None
) -> list[dict[str, Any]]:
    """**未过期**的条目：「最后更新」在 ``stale_s`` 内的（死条目不上读侧）。"""
    out: list[dict[str, Any]] = []
    for entry in read_registry(path):
        updated = _epoch(entry.get("updated"))
        if updated and now - updated <= stale_s:
            out.append(entry)
    return out


def prune_stale(*, now: float, stale_s: int = STALE_S, path: Path | None = None) -> int:
    """清掉死条目并落盘，返回清掉的条数（读侧不必负责写，故单列一个动作）。"""
    entries = read_registry(path)
    kept = [
        e for e in entries if (u := _epoch(e.get("updated"))) and now - u <= stale_s
    ]
    dropped = len(entries) - len(kept)
    if dropped:
        write_registry(kept, path)
    return dropped


def active_count(
    *, now: float, stale_s: int = STALE_S, path: Path | None = None
) -> int:
    """当前**活跃**会话数（未过期条目）——「有 worker 在跑吗」的一行读数。"""
    return len(live_entries(now=now, stale_s=stale_s, path=path))


class LiveSession:
    """worker 的活跃会话生命周期（§3.3 **写侧接线**）：登记 + 周期心跳 + 退场注销。

    用法（持锁之后立刻包住整段消化循环）::

        sess = LiveSession("devops", sid=run_log.stem, tier="devops", purpose="drain").start()
        try:
            ...
        finally:
            sess.stop()

    进入即 :func:`note_session`；``start_thread`` 为真时起一个**守护线程**按
    :data:`HEARTBEAT_INTERVAL_S` 刷心跳——worker 阻塞在 ``do_dispatch``（长 job）里也有
    心跳，不被 :data:`STALE_S` 判死。退出（含异常）停线程并 :func:`drop_session`。

    ``start_thread=False`` + :meth:`heartbeat_now` 是**测试注入口**：假时钟下手工推进心跳，
    免真起线程/真等（长 job 续命的确定性验证）。
    """

    def __init__(
        self,
        worker: str,
        *,
        sid: str = "",
        tier: str = "",
        purpose: str = "",
        interval_s: int = HEARTBEAT_INTERVAL_S,
        path: Path | None = None,
        now: Any = time.time,
        start_thread: bool = True,
    ) -> None:
        self.worker = str(worker)
        self.sid = str(sid) or self.worker
        self.tier = str(tier)
        self.purpose = str(purpose)
        self.interval_s = max(1, int(interval_s))
        self.path = path
        self._now = now
        self.start_thread = start_thread
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = False

    def _stamp(self, now: Any = None) -> float:
        return float(self._now() if now is None else now)

    def start(self) -> LiveSession:
        """登记会话并按需起心跳线程（幂等：重复调用不重复起线程）。"""
        if self._started:
            return self
        self._started = True
        note_session(
            self.worker,
            sid=self.sid,
            tier=self.tier,
            purpose=self.purpose,
            now=self._stamp(),
            path=self.path,
        )
        if self.start_thread:
            self._thread = threading.Thread(
                target=self._loop, daemon=True, name=f"hb-{self.worker}"
            )
            self._thread.start()
        return self

    def heartbeat_now(self, now: Any = None) -> bool:
        """刷一次心跳（可注入 ``now``）。线程循环与测试注入口共用。"""
        return heartbeat(self.worker, self.sid, now=self._stamp(now), path=self.path)

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            try:
                self.heartbeat_now()
            except Exception:  # noqa: BLE001 - 心跳线程绝不炸 worker（观察面写）
                pass

    def stop(self) -> None:
        """停心跳线程并注销会话（幂等；工作线程阻塞时线程取守护，join 有超时不挂）。"""
        if not self._started:
            return
        self._started = False
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        drop_session(self.worker, self.sid, path=self.path)

    def __enter__(self) -> LiveSession:
        return self.start()

    def __exit__(self, *exc: Any) -> bool:
        self.stop()
        return False


# ---------------------------------------------------------------------------
# 延期读侧共享 helper（devops backlog / 成员队列 / bencher 共用，避免各写各的）
# ---------------------------------------------------------------------------
def item_deferred(item: dict[str, Any], now: float) -> bool:
    """该条目是否仍处在延期待重试窗口内（``defer_until`` 在未来）。"""
    try:
        return float(item.get("defer_until") or 0) > float(now)
    except (TypeError, ValueError):
        return False


def deferred_items(items: list[dict[str, Any]], *, now: float) -> list[dict[str, Any]]:
    """``pending`` 且仍在延期窗内的条目（调度侧据此跳过与睡眠）。"""
    return [
        it for it in items if it.get("status") == "pending" and item_deferred(it, now)
    ]


def deferred_count(items: list[dict[str, Any]], *, now: float) -> int:
    """仍在延期窗内的 ``pending`` 条目数（观察面读数口径）。"""
    return len(deferred_items(items, now=now))


def next_defer_s(items: list[dict[str, Any]], *, now: float) -> float:
    """到**最近一个**延期条目到期的秒数；没有 → ``0.0``（worker 靠它睡到最近到期再查）。"""
    waits = [
        float(it.get("defer_until") or 0) - float(now)
        for it in deferred_items(items, now=now)
    ]
    return max(0.0, min(waits)) if waits else 0.0


# ---------------------------------------------------------------------------
# 延期纯函数库（不读时钟、不落盘、不接主循环）
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DeferPlan:
    """一次延期的判决。``exhausted`` 为真时调用方应走回退链或上报（``delay_s`` 为 0）。"""

    delay_s: int
    until: float
    attempt: int  # 已发生的延期次数（0 = 本次是第一次延期）
    exhausted: bool
    reason: str  # 未触顶时空串；否则是**哪道闸**触的顶


def defer_delay_s(deferred: int, *, base_s: int = DEFER_BASE_S) -> int:
    """第 ``deferred`` 次延期应等多久（0 起；2→4→8…分钟＝基数逐次翻倍）。"""
    return int(base_s) * (2 ** max(0, int(deferred)))


def plan_defer(
    deferred: int,
    elapsed_s: float,
    *,
    now: float,
    base_s: int = DEFER_BASE_S,
    max_attempts: int = DEFER_MAX_ATTEMPTS,
    budget_s: int = DEFER_BUDGET_S,
) -> DeferPlan:
    """判决「还能不能再延一次」。

    Args:
        deferred: 此前已延期的次数（0 起）。
        elapsed_s: 此前延期累计耗掉的墙钟秒数。
        now: 当前 epoch 秒（**由调用方给**，纯函数不读时钟——测试才好注入）。
        base_s: 退避基数。
        max_attempts: 次数闸（``0`` = 不设限）。
        budget_s: 壁钟预算闸（``0`` = 不设限）。

    Returns:
        :class:`DeferPlan`。三道判据依次是「次数已满」「耗时已满」「再延一次会超预算」——
        最后一条是序列与预算的**交互**（退避是指数增长，越往后一次就吃掉越多预算）。
    """
    count = max(0, int(deferred))
    elapsed = max(0.0, float(elapsed_s))
    if max_attempts > 0 and count >= max_attempts:
        return DeferPlan(0, now, count, True, f"延期次数触顶（{count}/{max_attempts}）")
    if budget_s > 0 and elapsed >= budget_s:
        return DeferPlan(
            0, now, count, True, f"延期耗时触顶（{elapsed:.0f}s/{budget_s}s）"
        )
    delay = defer_delay_s(count, base_s=base_s)
    if budget_s > 0 and elapsed + delay > budget_s:
        return DeferPlan(
            0,
            now,
            count,
            True,
            f"再延一次会超预算（已 {elapsed:.0f}s + {delay}s > {budget_s}s）",
        )
    return DeferPlan(delay, now + delay, count, False, "")
