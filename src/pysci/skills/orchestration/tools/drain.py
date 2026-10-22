"""devops backlog：计划器（组包入队）/ 锁 / 后台唤醒（README §6）。

``orch approve`` 采纳建议入 backlog 后，orch 自动以**分离后台进程**唤醒 backlog **计划器**；
计划器组包成批次**入队 devops FIFO**（``state/queues/devops.json``），由队列 worker 执行
——组长零阻塞、零轮询（对齐 README §4.5 等待模型）。**执行只留队列 worker 一份实现**
（backlog-fifo-20261014：drain 自持的 do_dispatch 循环退役；账务单一实现在
:func:`_settle_outcome`，计划器 dry 支与 worker 真派发支共用）。

三块机制：

- **锁**（``state/devops.lock``，JSON: pid/started/run_log）：同刻至多一个**计划器**。
  原子 ``O_CREAT|O_EXCL`` 抢占；持有者进程已死或锁龄超 :data:`LOCK_STALE_S`
  判 stale 并接管（fail-open，防崩溃后永久死锁）。**收尾事实优先于锁的自称**：
  释放时 unlink 删不掉（Windows 共享句柄）则改写为 ``released`` **墓碑**并在 run 日志留
  ``[!]``；连墓碑都写不进时，「该 run 已有 ``.done``」仍单独把残骸判为非活
  （:func:`_run_concluded`）。
- **唤醒**（:func:`wake_devops`）：approve 成功后调用；计划器锁忙则不重复 spawn（本项在
  **持久 backlog** 里，由在跑那位一并计划，FIFO 不丢件），空闲则 Popen 分离进程运行内部
  命令 ``_drain-devops``。message 附「等待已自动化」（挂载由 Stop 钩子自动完成，README §4.5）。
- **计划器循环**（:func:`drain_backlog`）：获锁后先做启动扫尾（孤儿 in_progress 回收 +
  worktree 四面临场处置：**devops 队列 worker 在跑时跳过**——那些 in_progress 是它的在途；
  :func:`reclaim_orphan_sessions` 终止仍活着的派生会话进程 → :func:`cleanup_orphan_worktrees`
  先抢救再清理 → 确认不了干净就 :func:`workflow.bump_worktree_generation` 换 ``<id>-r<gen>``
  新一代隔离面）；然后 ``backlog_take_batch``（种子 + 同提请者组包 ≤ :data:`BATCH_MAX`）逐批
  ``queues.enqueue(kind=BACKLOG_BATCH_KIND)``（携种子/组包元数据与批次正文），计划毕
  ``queueworker.wake_member("devops")``。**账务**（销账/needs_leader/延期/系统性停止/抢救）
  在 worker 侧经 :func:`settle_backlog_batch` → :func:`_settle_outcome` 落定；计划日志与
  ``<ts>.done``（``kind="plan"``）留 ``state/devops-runs/``，执行日志与 .done 在
  ``state/queues-runs/devops/``；完成唤醒走统一事件流 ``QUEUE-EMPTY``（worker 退出时发）。
  分离进程无 kill 之外的规范停止手段，故补**协作式停止**：``orch devops stop`` 写计划器
  ``state/devops.cancel`` **与 devops 队列 worker 的 ``state/queues/devops.cancel``**；
  两侧都在间隙检查——指向本进程则干净退出（释放锁、写含 ``stopped=cooperative`` 的部分
  .done、cancel 自删）；指向他方的陈旧信号顺手清除，绝不误停新 worker。

内部命令 ``_drain-devops``（计划器）与 ``_queue-worker devops``（执行）均下划线前缀，
argparse help 标注 ``[内部]``（勿手调）。
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from pysci.paths import ORCH_STATE_ROOT, PROJECT_ROOT

from . import backpressure as _bp
from . import channels as _channels
from . import presets, procs, workflow
from .registry import Registry, note_hop_pid, orchestration_rel, refresh_models
from .workflow import (
    AWAITING_DECISION_KEY,
    backlog_batch_text,
    backlog_complete,
    backlog_needs_leader,
    backlog_reclaim_orphans,
    backlog_requeue,
    backlog_take_batch,
    bump_worktree_generation,
    worktree_name,
)

#: 单批消化上限（护栏）：种子 + 同提请者组包 ≤ 此数（backlog 20261010-batch-digest）。
BATCH_MAX: int = 4

#: devops 队列里承载「backlog 批次」的条目 kind：计划器入队、worker 执行 + 结算。
#: 执行只留队列 worker 一份实现（backlog-fifo-20261014：drain 退役为批量计划器）。
BACKLOG_BATCH_KIND: str = "backlog-batch"

#: drain 互斥锁（JSON: pid/started/run_log）；仅 orch 与 devops drain 读写。
LOCK_PATH: Path = ORCH_STATE_ROOT / "devops.lock"

#: run 日志与 .done 摘要目录。
DEVOPS_RUNS_DIR: Path = ORCH_STATE_ROOT / "devops-runs"

#: :func:`new_run_log` 同秒顺延上限（正常同秒并发 ≤ 2，留足余量给时钟回拨）。
RUN_LOG_MAX_TRIES: int = 100

#: 顺延分隔符与计数位宽。用 ``_`` 而**不是** ``-``：``-`` 的码位（0x2D）低于 ``.``
#: （0x2E），``<ts>-2.done`` 会排在 ``<ts>.done`` **之前**，:func:`latest_done` 的
#: ``sorted()[-1]`` 就退回读旧摘要——建议书示例名 ``<ts>-2.log`` 满足不了它自己附带的
#: 「``latest_done()`` 排序语义不变」要求，故此处按该要求取零填充的 ``_<n>``（backlog
#: 20261011-233248-devops）。位宽对齐上限 ``100`` 才使 ``_10 < _100`` 与 ``_2 < _10``。
RUN_LOG_SEP: str = "_"
RUN_LOG_PAD: int = len(str(RUN_LOG_MAX_TRIES))

#: 锁龄上限（秒）——超过即判 stale 接管（worker 崩溃兜底）。
#:
#: **2026-10-15 由 6h 下调为 45min**（task-20261015-worker-stall，用户点名「卡死无任何事件」）。
#: 依据是台账实测：208 跳的时长分布 p50=10.7min、p90=40.8min、p95=58.3min、max=129.7min
#: ——所以**不能**只按 30-60min 一刀切：单看年龄会把 10% 的正常跳判死并抢锁。故本阈值
#: 与 :func:`_holder_heartbeating` **配对**才成立：
#:
#: * 持有者仍**心跳新鲜**（``LiveSession`` ≤120s 刷新，成员队列/计划器/bencher 三方都登记）
#:   ⇒ 超龄也只说明「慢」，一律**不抢锁**——抢了就是双 worker 并发消费（本仓库最贵的事故）；
#: * 持有者**无心跳**（死进程 / 假活僵尸）⇒ 45min 即解套，不再白等 6h。
#:
#: 于是「心跳活着但工作卡死」这一类（2026-10-15 08:02 实案：31+ min 零进展、心跳照跳）
#: 不再由本判据兜——它由 :mod:`~.stallwatch` 的**跳级 watchdog**按「无进展」判据主动杀。
#: 两层分工：心跳管「慢而不死」，watchdog 管「不死但不动」，本阈值管「死了还占着锁」。
LOCK_STALE_S: int = 45 * 60

#: 干跑开关环境变量：置真值时 wake 派生的 drain 走 --dry（不真派发，演练/验证唤醒链用）。
DRY_ENV: str = "PYSCI_ORCH_DRAIN_DRY"

#: 协作式停止信号（JSON: pid/requested_at）；orch devops stop 写，drain 消费自删。
CANCEL_PATH: Path = ORCH_STATE_ROOT / "devops.cancel"

#: 锁记录的「已释放」标记键：unlink 删不掉锁时把锁改写为该标记的**墓碑**——
#: 残骸不再是互斥依据（:func:`_lock_live` 认它为非活锁）。backlog 20261011-225933-devops。
RELEASED_KEY: str = "released"

#: 广播停止信号的 scope 标记（watchdog 停机用；backlog 20261014-135935-devops）：
#: ``{"scope": "all", "requested_at": …}`` = **停止纪元**——按启动时刻比较，一次停止
#: 覆盖同刻全部实例（含不持锁的孤儿）；单 pid 语义只留 drain/队列的常规停止路径。
STOP_SCOPE_ALL: str = "all"

#: acquire 抢锁重试轮数（竞态窗口 + 残骸清除）。
ACQUIRE_RETRIES: int = 3

#: 「半写载荷」宽限（秒）：锁文件已由他方 ``O_EXCL`` 建成、payload 尚未落盘的窗口。
#: 竞争者在此窗口内不得据「读成空/损坏」立刻偷锁（见 :func:`_await_payload`）。真实窗口
#: 是同一进程内两次 syscall 之间（微秒级），宽限取 0.2s = 四个数量级余量；连宽限都读不出
#: 载荷才按残骸接管——残骸不得挡路。
LOCK_PAYLOAD_GRACE_S: float = 0.2

#: 上者的轮询步长（秒）。
LOCK_PAYLOAD_POLL_S: float = 0.01

_COMMIT_RE = re.compile(r"commit[^0-9a-f]{0,16}\b([0-9a-f]{7,40})\b", re.IGNORECASE)

#: 交付正文的 ``backlog id=<id1>,<id2>,…`` 清单（批量消化逐 id 销账依据）。id 仅含
#: 字母数字/-/_，故捕获在遇中文/标点即止；分隔容忍 ``,``/``、``/空白。
_BACKLOG_IDS_RE = re.compile(
    r"backlog\s+id\s*[:=]\s*([0-9A-Za-z_\-]+(?:\s*[,、]\s*[0-9A-Za-z_\-]+)*)",
    re.IGNORECASE,
)


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


@dataclass
class WorkerPaths:
    """一套 worker 生命周期落点（锁 / run 目录 / 协作停止信号 / 锁龄上限）。

    为什么要把它做成参数（README §1「共享逻辑做库，不复制实现」）：本模块的锁语义
    是被三次事故打磨出来的——墓碑（删不掉的锁不得挡路）、``.done`` 收尾证据（谎报在跑的
    残骸不得被当活锁）、run 名同秒顺延（撞名会让活锁被上一任的摘要判成残骸）。bencher
    的 FIFO worker（:mod:`benchd`，README §5.5）需要的正是同一套判活规则；复制一份等于
    把这些教训重踩一遍。
    """

    lock: Path
    runs: Path
    cancel: Path
    stale_s: int = LOCK_STALE_S


def _devops_paths() -> WorkerPaths:
    """devops drain 的默认落点。

    **每次调用现读模块常量**而不是在 import 期固化：测试用 ``monkeypatch.setattr(drain,
    "LOCK_PATH", tmp)`` 换隔离面，固化快照会让那些补丁打不中（锁语义的用例全部失效）。
    """
    return WorkerPaths(
        lock=LOCK_PATH, runs=DEVOPS_RUNS_DIR, cancel=CANCEL_PATH, stale_s=LOCK_STALE_S
    )


# ---------------------------------------------------------------------------
# 锁
# ---------------------------------------------------------------------------
def _pid_alive(pid: int) -> bool:
    """判断 pid 是否仍存活（Windows: tasklist；POSIX: os.kill 0 信号）。

    判活失败（命令不可用/超时）时**保守视为存活**——宁可让锁忙等下一轮，也不误抢
    正在运行的 worker（误抢会导致双 worker 并发消费 backlog）。
    """
    if pid <= 0:
        return False
    if sys.platform == "win32":
        try:
            proc = procs.run_hidden(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=15,
            )
        except (OSError, subprocess.TimeoutExpired):
            return True
        return re.search(rf"\b{pid}\b", proc.stdout or "") is not None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def read_lock(paths: WorkerPaths | None = None) -> dict[str, Any] | None:
    """读取锁记录；不存在或损坏返回 None（损坏锁将由 acquire 覆盖）。"""
    lock = (paths or _devops_paths()).lock
    if not lock.exists():
        return None
    try:
        data = json.loads(lock.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return data if isinstance(data, dict) else None


def _lock_age_s(data: dict[str, Any]) -> float:
    """锁龄（秒）；started 不可解析时返回 -1（视为未知，不按超龄处理）。"""
    try:
        started = datetime.fromisoformat(str(data.get("started", "")))
    except (ValueError, TypeError):
        return -1.0
    return (datetime.now().astimezone() - started).total_seconds()


def _lock_released(data: dict[str, Any] | None) -> bool:
    """锁记录是否为「已释放却删不掉」的墓碑（残骸，不作互斥依据）。"""
    return bool(data) and bool(data.get(RELEASED_KEY))


def _run_done_path(data: dict[str, Any] | None) -> Path | None:
    """锁记录里那个 run 对应的 ``.done`` 路径（无法定位时 None）。"""
    run_log = str((data or {}).get("run_log", ""))
    if not run_log:
        return None
    path = Path(run_log)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.with_suffix(".done")


def _done_predates_lock(body: Any, data: dict[str, Any]) -> bool:
    """``.done`` 的 ``completed`` **严格早于**锁的 ``started``（⇒ 不是本锁的收尾）。

    同一 run 的因果次序固定是「起锁 → 干活 → 释放 → 写摘要」，摘要若早于本锁诞生，
    它收的就是上一任的尾。取「严格早」而不是「不晚于」：秒级时间戳下同秒空跑（起锁与
    收尾落在同一秒）是合法的自收尾，判成未收尾就把 E4 要解的假 running 又挡回来了
    （:func:`_run_concluded` 与本函数是两道独立闸：前者认身份，后者认时序；命名面由
    :func:`new_run_log` 查重兜住——backlog 20261011-233248-devops）。

    任一时刻读不出（字段缺失/格式非 ISO）时返回 False——不确定就不否决。
    """
    completed = str(body.get("completed", "")) if isinstance(body, dict) else ""
    started = str(data.get("started", ""))
    if not completed or not started:
        return False
    try:
        done_at = datetime.fromisoformat(completed)
        lock_at = datetime.fromisoformat(started)
        return done_at < lock_at
    except (ValueError, TypeError):
        # TypeError：一侧带时区一侧不带（手写/迁移期的摘要）——比较不了就不否决
        return False


def _run_concluded(data: dict[str, Any] | None) -> bool:
    """锁自称在跑的 run 是否**已写出 ``.done``**——收尾硬证据（README §4.5）。

    ``.done`` 由 drain 在 :func:`release_lock` **之后**写出，故摘要存在即该 run 不可能
    还在跑，锁只剩「删不掉的残骸」这一种解释。判活若不采纳这条，墓碑也写不进的极端
    残骸（他方连写都不共享）就会把队列挡成假 running 直到 6 h 超龄（backlog
    20261011-231237-devops，收尾 E3 未竟格）。

    摘要读得出时先过两道独立闸：**身份自证**（``run_log`` 同名——万一是别的 run 的摘要
    落到这个文件名，不能被当作本锁的收尾证据而误抢活锁）与**时序核对**
    （:func:`_done_predates_lock`——摘要早于本锁诞生就收不了本锁的尾）。命名面另有护栏：
    :func:`new_run_log` 同秒顺延，run 身份不再撞名（backlog 20261011-233248-devops；
    此前秒级同名会让「同秒续跑的活锁」被上一任的 ``.done`` 判成残骸并被抢走，单 worker
    互斥击穿）。
    """
    done = _run_done_path(data)
    if done is None or not done.exists():
        return False
    try:
        body = json.loads(done.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return True  # 半写/损坏：文件已在即已收尾（写它在释放锁之后）
    if _done_predates_lock(body, data):
        return False
    declared = (
        str(body.get("run_log", "")) if isinstance(body, dict) else ""
    )  # 摘要未记身份 → 无更反的证据，按存在为准
    if not declared:
        return True
    return Path(declared).name == Path(str(data.get("run_log", ""))).name


def _overwrite_lock(payload: str, paths: WorkerPaths | None = None) -> bool:
    """就地改写锁文件内容（先原子 replace，兜底截断重写）。

    用于「文件删不掉」的 Windows 共享句柄场景：能改写就能标记墓碑 / 接管残骸。
    全失败返回 False（调用方留痕，不静默）。
    """
    lock = (paths or _devops_paths()).lock
    tmp = lock.with_name(lock.name + ".tmp")
    try:
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(payload, encoding="utf-8")
        os.replace(str(tmp), str(lock))
        return True
    except OSError:
        pass
    finally:
        try:
            if tmp.exists():
                os.unlink(str(tmp))
        except OSError:
            pass
    try:
        lock.write_text(payload, encoding="utf-8")
        return True
    except OSError:
        return False


def _holder_heartbeating(
    paths: WorkerPaths | None = None, data: dict[str, Any] | None = None
) -> bool:
    """锁的持有者**仍在心跳**吗（超龄锁的免抢判据，2026-10-15 worker-stall 单）。

    为什么需要它：锁龄上限（:data:`LOCK_STALE_S`）单看年龄无法区分「快」与「死」——实测
    单跳可跑 130min，一刀切 45min 会把 ~10% 的正常跳判死并让第二个 worker 抢锁，直接违反
    ``_pid_alive`` docstring 里那条纪律（「误抢会导致双 worker 并发消费」）。而活跃会话
    注册表（``backpressure.LiveSession``，≤120s 刷心跳）给出的正是「持有者还活着」的独立
    证据：**超龄 + 心跳 ⇒ 慢而不死，不抢**。

    身份对齐：锁记录里的 ``run_log`` 与心跳条目的 ``sid`` 同源（两侧都用 ``run_log.stem``，
    见 :func:`new_run_log`／各 worker 的 ``LiveSession(... sid=run_log.stem ...)``），故按
    sid 命中即认定为同一持有者；``worker == 锁文件名`` 作为兜底（旧记录 run_log 不可解析时）。

    读心跳失败**保守判为「有心跳」**：与 :func:`_pid_alive`「判不出来时宁可不抢」同口径——
    竞争条件的代价是双 worker，而多等一拍的代价只是慢。
    """
    d = data if data is not None else read_lock(paths)
    if not d:
        return False
    p = paths or _devops_paths()
    wanted = {Path(str(d.get("run_log") or "")).stem, p.lock.stem}
    try:
        entries = _bp.live_entries(now=time.time())
    except Exception:  # noqa: BLE001 - 心跳面读不动不改变「宁可不抢」的方向
        return True
    return any(
        str(e.get("sid") or "") in wanted or str(e.get("worker") or "") in wanted
        for e in entries
    )


def _lock_live(data: dict[str, Any] | None, paths: WorkerPaths | None = None) -> bool:
    """锁是否有效：未宣告收尾、持有者进程存活且锁龄未超上限。

    两条**收尾宣告**都优先于 pid 判活（判活本身可能说谎：pid 回收、tasklist 超时保守
    判活）：``released`` 墓碑，以及该 run 已有 ``.done`` 摘要（:func:`_run_concluded`）。

    超龄（``age > stale_s``）**不直接判死**：持有者仍心跳时它只是慢（见
    :func:`_holder_heartbeating`，2026-10-15 worker-stall 单）——抢走仍在干活的锁 = 双 worker。
    """
    if not data or _lock_released(data) or _run_concluded(data):
        return False
    p = paths or _devops_paths()
    age = _lock_age_s(data)
    if age > p.stale_s and not _holder_heartbeating(p, data):
        return False
    return _pid_alive(int(data.get("pid", 0) or 0))


def _await_payload(paths: WorkerPaths | None = None) -> bool:
    """等锁文件长出**可解析的载荷**（他方刚建成、正在写入的窄窗）。

    返回 ``True`` = 宽限内读到了载荷（调用方必须再判活决定是否让位）；``False`` = 空/损坏
    载荷一直读不出（真残骸，按既有语义接管）。文件消失同样返回 ``False``——空位本就该被抢占。

    为什么需要（2026-10-15 看门狗双生单）：``acquire_lock`` 先 ``O_EXCL`` 建文件、后写
    payload，两步之间文件是空的；``read_lock`` 把空文件与损坏文件一并读成 ``None``，
    竞争者据此立刻 ``unlink`` 即成**偷锁**——两个实例各自都以为自己持锁。
    """
    p = paths or _devops_paths()
    deadline = time.time() + LOCK_PAYLOAD_GRACE_S
    while True:
        if read_lock(p) is not None:
            return True
        if not p.lock.exists():
            return False
        if time.time() >= deadline:
            return False
        time.sleep(LOCK_PAYLOAD_POLL_S)


def acquire_lock(
    run_log: str, paths: WorkerPaths | None = None
) -> dict[str, Any] | None:
    """尝试获取 worker 锁。

    Args:
        run_log: 本次 run 的日志相对路径（写入锁记录供 status 展示）。
        paths: 生命周期落点；缺省=devops drain（:func:`_devops_paths`），bencher worker
            传 :data:`benchd.BENCH_PATHS` 复用同一套判活/墓碑/收尾语义。

    Returns:
        成功获取时返回锁记录 ``{pid, started, run_log}``；已被存活 worker 持有时
        返回 ``None``（调用方应优雅退出，不重复消费）。残骸一律接管：进程死/超龄的
        stale 锁、``released`` 墓碑、以及「该 run 已有 ``.done``」的已收尾残骸——删不掉
        就地改写接管（残骸不得挡路，backlog 20261011-225933-devops + 20261011-231237-devops）。
    """
    p = paths or _devops_paths()
    p.lock.parent.mkdir(parents=True, exist_ok=True)
    record = {"pid": os.getpid(), "started": _now(), "run_log": run_log}
    payload = json.dumps(record, ensure_ascii=False, indent=2) + "\n"
    for _ in range(ACQUIRE_RETRIES):
        if _lock_live(read_lock(p), p):
            return None
        try:
            fd = os.open(str(p.lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            # 竞态：他方刚抢占。若其存活则退出；若仍是 stale 残留则清除后重试。
            if _lock_live(read_lock(p), p):
                return None
            # **半写载荷窗口**：他方 ``O_EXCL`` 建成与 ``write(payload)`` 之间，锁文件是
            # 空的；``read_lock`` 把空文件读成 None（同「损坏锁」），若据此立刻 unlink 就
            # **偷走仍在写入的活锁** —— 两个实例各自以为自己持锁，双生（2026-10-15 看门狗
            # 双生单：并发 start 撞开这道窄窗即可复现）。先给一拍：载荷出现且判活就让位，
            # 仍读不出才认作残骸（残骸不得挡路，语义不变）。
            if _await_payload(p) and _lock_live(read_lock(p), p):
                return None
            try:
                os.unlink(str(p.lock))
                continue
            except OSError:
                # 残骸删不掉（共享句柄）：就地改写接管；改写后回读确认 pid 是自己的
                # 才算到手（他方同期改写则让位，保持单 worker 互斥）。
                if _overwrite_lock(payload, p):
                    held = read_lock(p) or {}
                    if int(held.get("pid", 0) or 0) == os.getpid():
                        return record
                continue
        except OSError:
            return None
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(payload)
        return record
    return None


def release_lock(
    run_log: Path | str | None = None, paths: WorkerPaths | None = None
) -> bool:
    """释放锁——**仅当持有者是本进程**（防未获取到锁的进程误删他人锁）。

    Args:
        run_log: 本次 run 日志；删除失败时在其中留一行 ``[!]`` 警示（None = 只标记、
            不留痕，供非 drain 调用方使用）。
        paths: 生命周期落点；缺省=devops drain。

    Returns:
        True = 锁已不占位（删除成功，或本来无锁）；False = 未删掉（删除失败已改写为
        :data:`RELEASED_KEY` 墓碑，或该锁不是本进程的）。

    **失败不可静默**：Windows 上他方仍持该文件句柄时 DeleteFile 报共享冲突，旧实现的
    ``except OSError: pass`` 把失败吞成「锁仍在、pid 仍活」的假 running——``_lock_live``
    遂把残骸判为活锁，下一任 drain 被 ``skipped=lock_busy`` 挡掉，最长阻塞到
    :data:`LOCK_STALE_S`（6h）超龄才解套（backlog 20261011-225933-devops，与
    20261011-220622-leader 的 watch 吞事件同源）。墓碑令锁面与观察面同说一席话；连墓碑
    都写不进（他方连写都不共享）时仍不挡路——紧随其后写出的 ``.done`` 会被判活单独当作
    收尾证据（:func:`_run_concluded`，backlog 20261011-231237-devops）。
    """
    data = read_lock(paths)
    if not data:
        return True
    if int(data.get("pid", 0) or 0) != os.getpid():
        return False  # 非自持锁：绝不改动（含墓碑——清除归下一任 acquire）
    try:
        os.unlink(str((paths or _devops_paths()).lock))
        return True
    except OSError as exc:
        marked = _mark_released(data, paths)
        _note_release_failure(run_log, exc, marked=marked)
        return False


def _mark_released(data: dict[str, Any], paths: WorkerPaths | None = None) -> bool:
    """把锁记录改写为 released 墓碑（保留 pid/run_log 供 status 溯源）。"""
    tombstone = {
        RELEASED_KEY: True,
        "pid": int(data.get("pid", 0) or 0),
        "started": data.get("started", ""),
        "run_log": data.get("run_log", ""),
        "released_at": _now(),
    }
    return _overwrite_lock(
        json.dumps(tombstone, ensure_ascii=False, indent=2) + "\n", paths
    )


def _note_release_failure(
    run_log: Path | str | None, exc: OSError, *, marked: bool
) -> None:
    """把释放失败写进 run 日志（无人值守的后台进程必须留痕；无 run_log 则只能沉默）。"""
    if run_log is None:
        return
    path = Path(run_log)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    tail = (
        "已写 released 标记" if marked else "released 标记亦写入失败（残骸待超龄解套）"
    )
    _log(path, f"[!] 锁释放失败（共享句柄），{tail}：{exc}")


def lock_status(paths: WorkerPaths | None = None) -> dict[str, Any]:
    """status 命令用的锁态摘要：``{state: idle|running|stale, pid, started, ...}``。

    stale 时附 ``crashed`` 字段：该 run 无对应 ``.done`` = worker 被中途杀死（崩溃）；
    有 ``.done`` = 上一次干净完成但锁残留（release 未及/异常）。供 orch status surfacing。

    released 墓碑与「该 run 已有 ``.done``」是两种**已宣告收尾的残骸**，各自单列读数且
    一律 ``crashed=False``（崩溃者走不到 release、也写不出 ``.done``）：前者锁面自己标了
    记号，后者连标号都写不进（backlog 20261011-231237-devops）——两者都不占队列，下一任
    drain 就地接管。
    """
    p = paths or _devops_paths()
    data = read_lock(p)
    if not data:
        return {"state": "idle"}
    pid = int(data.get("pid", 0) or 0)
    age = _lock_age_s(data)
    base = {
        "pid": pid,
        "started": data.get("started", ""),
        "run_log": data.get("run_log", ""),
        "age_s": round(age, 1) if age >= 0 else -1,
    }
    if _lock_released(data):
        return {
            "state": "stale",
            "released": True,
            "released_at": data.get("released_at", ""),
            "crashed": False,
            **base,
        }
    if _run_concluded(data):
        return {
            "state": "stale",
            "residue": "done-exists",
            "crashed": False,
            **base,
        }
    alive = _pid_alive(pid)
    # 锁龄未知（started 不可解析，:func:`_lock_age_s` 按既定语义返回 -1 = 未知≠超龄）
    # 时**不得**据此判 stale：那条与 :func:`_lock_live` 直接矛盾（它只问 `age > stale_s`，
    # 未知龄放行判活），于是同一份锁记录会「观察面报崩溃残骸、锁面拒起锁」，操作员照
    # 观察面去清锁/杀进程，杀的正是在跑的 worker（backlog insp-20261013-045910-mechanical：
    # 判读语义对齐，先例 #43/#44 家族）。未知龄与存活进程并存时按 running 报——判不出
    # 龄不等于判得出死。
    # 超龄但持有者仍心跳 ⇒ 报 running（慢而不死；与 :func:`_lock_live` 同判据同方向，
    # 两面都说「running」才不会有第二个人去抢锁，2026-10-15 worker-stall 单）。
    age_ok = age < 0 or age <= p.stale_s or _holder_heartbeating(p, data)
    state = "running" if (alive and age_ok) else "stale"
    out: dict[str, Any] = {"state": state, **base}
    if state == "stale":
        out["crashed"] = _stale_run_crashed(data)
    return out


def _stale_run_crashed(data: dict[str, Any]) -> bool:
    """stale 锁对应的 run 是否崩溃（无 ``.done``）。

    ``.done`` 由 drain_backlog 的 finally 块在正常/异常收尾时写出——存在即代表 worker
    跑到了收尾（干净完成），缺失即代表进程被中途杀死（OS 关机 / kill -9）没机会收尾。
    run_log 无法定位时保守判为崩溃。
    """
    done = _run_done_path(data)
    return True if done is None else not done.exists()


# ---------------------------------------------------------------------------
# 协作式停止（orch devops stop ↔ drain 循环间隙消费）
# ---------------------------------------------------------------------------
def request_stop(
    paths: WorkerPaths | None = None, *, broadcast: bool = False
) -> tuple[bool, str]:
    """写协作停止信号，目标为当前持锁 worker。供 ``orch devops stop`` 调用。

    ``broadcast=True``（watchdog 用，backlog 20261014-135935-devops）：写**停止纪元**
    （``scope="all"``）而非单个 pid——同刻**全部** ``_watchdog-loop`` 实例（含不持锁的
    孤儿）都会在各自下一个自查点（≤60s）退出；**锁空闲时照样投递**（旧实现在此返回
    「无需停止」且不写信号，孤儿永远收不到——2026-10-14 实测：两实例同跑、一次 stop
    只停一个）。纪元按 ``requested_at`` 与各实例启动时刻比较，晚于纪元启动的**新实例
    不受影响**（看门狗调度器/下一跳照常拉起）。

    Returns:
        ``(ok, message)``。非广播：无运行中 worker 时**不写信号**（返回 False——空闲
        队列不需要停，残留信号还会危及下一个 worker）；广播：**恒写**（孤儿可能无锁在跑）。
    """
    p = paths or _devops_paths()
    if broadcast:
        p.cancel.parent.mkdir(parents=True, exist_ok=True)
        stamp = _now()
        p.cancel.write_text(
            json.dumps(
                {"scope": STOP_SCOPE_ALL, "requested_at": stamp},
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return True, (
            f"已广播停止请求（停止纪元 {stamp}）：早于该时刻启动的全部 watchdog 实例将在"
            "下一个自查点（≤60s）干净退出；晚于它启动的新实例不受影响。"
        )
    st = lock_status(p)
    if st["state"] != "running":
        return False, "无运行中的 drain worker（锁空闲），无需停止。"
    p.cancel.parent.mkdir(parents=True, exist_ok=True)
    p.cancel.write_text(
        json.dumps(
            {"pid": st["pid"], "requested_at": _now()}, ensure_ascii=False, indent=2
        )
        + "\n",
        encoding="utf-8",
    )
    return True, (
        f"已请求协作停止（目标 pid {st['pid']}）：worker 将于下一条目间隙干净退出"
        f"（释放锁、写部分 .done），信号文件 {orchestration_rel(p.cancel)} 消费后自删。"
    )


def cancel_status(paths: WorkerPaths | None = None) -> dict[str, Any] | None:
    """status 展示用：待消费的停止信号（无/损坏则 None）。"""
    cancel = (paths or _devops_paths()).cancel
    if not cancel.exists():
        return None
    try:
        data = json.loads(cancel.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return data if isinstance(data, dict) else None


def _consume_stop_request(
    paths: WorkerPaths | None = None, *, started_at: str = ""
) -> bool:
    """drain 循环检查：停止信号是否指向本进程；无论指向与否信号文件均被删除。

    陈旧信号（pid≠本进程——被 kill 的 worker 未及消费而留下的）顺手清除并返回
    False，绝不误停接棒的下一个 worker。

    **广播纪元支**（``scope="all"``，watchdog 用）：按 ``requested_at`` 与调用方
    ``started_at`` 比较——纪元晚于本实例启动才消费（返回 True），**不删文件**（删了会
    抢掉其他实例的信号：第一个醒来的实例就会把纪元吞掉，其余照旧存活——2026-10-14
    实测的两实例场景）；纪元早于本实例（新实例）→ False 且保留，陈旧纪元不误停接棒者。
    """
    cancel = (paths or _devops_paths()).cancel
    if not cancel.exists():
        return False
    try:
        data = json.loads(cancel.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        data = {}
    if str(data.get("scope") or "") == STOP_SCOPE_ALL:
        req = str(data.get("requested_at") or "")
        return bool(req) and bool(started_at) and req > started_at
    try:
        pid = int(data.get("pid", 0))
    except (ValueError, TypeError):
        pid = 0
    try:
        cancel.unlink()
    except OSError:
        pass
    return pid == os.getpid()


# ---------------------------------------------------------------------------
# run 日志 / .done
# ---------------------------------------------------------------------------
def _log(run_log: Path | None, line: str) -> None:
    """向 run 日志追加一行（带时间戳）。日志写入失败不阻塞 drain（fail-open）。

    ``run_log=None`` = 静默：回收类函数允许被独立调用（非 drain 循环内）而不必编日志。
    """
    if run_log is None:
        return
    try:
        run_log.parent.mkdir(parents=True, exist_ok=True)
        with run_log.open("a", encoding="utf-8") as fh:
            fh.write(f"[{datetime.now().strftime('%H:%M:%S')}] {line}\n")
    except OSError:
        pass


def log_stage(run_log: Path | None, stage: str, **fields: Any) -> None:
    """**阶段埋点**：把 worker/派发链的推进点追加进 run 日志（带毫秒时间戳）。

    为什么必须有它（2026-10-15 08:02 实案，用户定性「非常危险」）：devops 队列 worker
    ``_queue-worker``(pid 12960) 获锁后**静默卡死 31+ 分钟**——run 日志 **0 字节**、零会话
    登记、零 token、条目假挂 ``in_progress``，只能等用户从 API 控制台偶然发现；事后无栈可查、
    钉不死根因。根因是「日志里什么阶段都可能，就是没有『我走到哪一步了』的记录」——
    而 ``quiet=True``（queue worker 一律静默派发）会把一切 stdout 提示也吃掉。

    本函数独立于 ``quiet``：**只写文件、不写 stdout**，故静默派发照样留痕。逐阶段一行，
    每行即时落盘（open→write→flush→close，不共享句柄），进程被硬杀也留得下最后一条。

    格式：``[2026-10-15T08:02:31.412+08:00] [stage] lock_acquired member=devops``。
    写失败 fail-open（观察面不阻塞生产，§0.6）。
    """
    if run_log is None:
        return
    stamp = datetime.now().astimezone().isoformat(timespec="milliseconds")
    tail = " ".join(f"{k}={v}" for k, v in fields.items() if v not in (None, ""))
    try:
        run_log.parent.mkdir(parents=True, exist_ok=True)
        with run_log.open("a", encoding="utf-8") as fh:
            fh.write(f"[{stamp}] [stage] {stage}{' ' + tail if tail else ''}\n")
            fh.flush()
    except (OSError, ValueError):  # 坏路径（含内嵌 NUL）也吞：取证面绝不改派发结局
        pass


def _log_exception(
    run_log: Path | None, exc: BaseException, head: str, *, frames: int = 20
) -> str:
    """异常入 run 日志：一行短消息 + **完整栈回溯**；返回短消息串（进 ``.done``）。

    三面派生路径（drain / bencher / inspector）同口径共用。只记 ``{type}: {exc}``
    时，日志里能看到的就是 ``TypeError: expected str, bytes or os.PathLike object,
    not list`` 这一句——不指文件、不指行、不指旗标，排查只能 git blame 猜落点
    （2026-10-13 run 211429 实证，backlog 20261013-003000-leader）。后台进程无人值守，
    栈必须由它自己留在日志里。

    Args:
        run_log: 本次 run 日志（None = 静默，仍返回短消息串）。
        exc: 捕获到的异常（栈取自 ``exc.__traceback__``）。
        head: 短消息前缀，如 ``drain 异常中断``。
        frames: 栈帧上限（防循环调用把日志刷爆）。

    Returns:
        ``"{异常类型}: {消息}"`` 短串——调用方原样塞进 ``summary["error"]``。
    """
    brief = f"{type(exc).__name__}: {exc}"
    if run_log is None:
        return brief
    stack = "".join(
        traceback.format_exception(type(exc), exc, exc.__traceback__, limit=frames)
    )
    _log(run_log, f"[!] {head}：{brief}\n{stack.rstrip()}")
    return brief


def new_run_log(paths: WorkerPaths | None = None) -> Path:
    """生成本次 run 的日志路径 ``<runs>/<ts>.log``，**撞名则补位顺延并当场占位**。

    run 名不只是日志文件名：``.done`` 判活（:func:`_run_concluded`）与 watch 的完成身份
    三元组都按名字取证。秒级时间戳在同秒两次调用时**同名**，于是快速收尾的上一任（空队列
    /仅 lock_busy，同秒就写出 ``.done``）会让紧随其后的活锁——锁新鲜、pid 存活、正在干活
    ——被判成 ``stale/done-exists`` 残骸并被人抢走：README §6 的单 worker 互斥就此击穿
    （backlog 20261011-233248-devops，建议书点 ①）。

    两道护栏的分工：**命名面**（本函数）保证 run 身份唯一，**时序面**
    （:func:`_done_predates_lock`）保证即便有人手工拿到同名锁、上一任的摘要也收不了本锁的
    尾。顺延名用 :data:`RUN_LOG_SEP` 补位（字典序仍晚于被占名，``latest_done`` 读的才是
    最新摘要）；占位用 ``O_CREAT|O_EXCL`` 原子创建空 ``.log``——只查存在不占位，同秒两个
    进程仍会同时选中同名（:func:`_log` 与 ``wake_devops`` 都以 append 打开，先建空文件
    无害）。顺延用尽（同秒撞满 :data:`RUN_LOG_MAX_TRIES`，实际只在时钟回拨时出现）或目录
    不可建时退到 pid 后缀名（fail-open：命名护栏不得挡住 drain 开工）。
    """
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    runs = (paths or _devops_paths()).runs
    fallback = runs / f"{stamp}{RUN_LOG_SEP}p{os.getpid()}.log"
    try:
        runs.mkdir(parents=True, exist_ok=True)
    except OSError:
        return fallback
    for n in range(1, RUN_LOG_MAX_TRIES + 1):
        stem = stamp if n == 1 else f"{stamp}{RUN_LOG_SEP}{n:0{RUN_LOG_PAD}}"
        cand = runs / f"{stem}.log"
        if cand.with_suffix(".done").exists():
            continue
        try:
            os.close(os.open(str(cand), os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        except OSError:
            continue  # 名字被占（含并发他方刚创建）：顺延下一个
        return cand
    return fallback


def latest_done(paths: WorkerPaths | None = None) -> dict[str, Any] | None:
    """最近一次 run 的 .done 摘要（无则 None）；附 ``name`` 字段。"""
    runs = (paths or _devops_paths()).runs
    if not runs.exists():
        return None
    dones = sorted(runs.glob("*.done"))
    if not dones:
        return None
    try:
        data = json.loads(dones[-1].read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if isinstance(data, dict):
        data["name"] = dones[-1].stem
    return data if isinstance(data, dict) else None


#: run 摘要 items 里「这一条目这次尝试失败」的状态读数。系统性故障（额度耗尽 /
#: 渠道级故障）**不在**此列：那是渠道坏了、条目无罪复位，计入会把「渠道坏了」误读成
#: 「任务书写坏了」（并污染 §2.4 的重复失败阈值立案）。
#: 带宽延期的睡眠上限（秒）：全队皆未到期时睡到最近到期，但单次不许超过它——
#: 过长会把「条目间隙检查协作停止」的那条纪律架空（10 分钟内必回来看一次信号）。
DEFER_MAX_SLEEP_S: int = 600


def _sleep(seconds: float) -> None:
    """等待落点（测试可整只替换——真睡 600s 的用例没人跑得动）。"""
    time.sleep(seconds)


FAILED_ITEM_STATUSES: tuple[str, ...] = (
    "needs_leader",
    "blocked",
    "run_failed",
    "parse_error",
    "check_fail",
)


def _systemic_kinds() -> tuple[str, ...]:
    """系统性故障 kind 清单（唯一真源在 dispatch 的模式表旁，此处延迟取免导入环）。"""
    from .dispatch import (  # 延迟导入：drain↔dispatch 互为上下游，只在用时取常量
        SYSTEMIC_KINDS,
    )

    return SYSTEMIC_KINDS


def _systemic_headline(kind: str) -> str:
    """系统性故障短标题（note 文案）：额度类失败 / 渠道级失败。"""
    from .dispatch import SYSTEMIC_TEXT

    return SYSTEMIC_TEXT.get(kind, (kind, kind))[0]


def _systemic_error_text(kind: str) -> str:
    """``.done`` 的 ``error`` 句——三个 worker 面（drain/bench/inspect）同文同调。"""
    from .dispatch import systemic_error_text

    return systemic_error_text(kind)


def _systemic_mark(kind: str) -> str:
    """run 日志行首标记：``[Q]``=额度、``[C]``=渠道。

    watch 的日志判据读的是 kind 串本身（不认标记），标记只为复盘时一眼分清这次是
    「该充值」还是「该等渠道恢复」——两类混成一个字母，黑匣子就白写了。
    """
    return "C" if kind in ("channel_error", "channel_saturated") else "Q"


def failure_counts(
    *, limit: int = 0, paths: WorkerPaths | None = None
) -> dict[str, int]:
    """按 backlog 条目 id 汇总 run 摘要中的失败次数（坏文件/坏行跳过，观察面 fail-soft）。

    ``limit>0`` 只看最近 N 份 ``.done``。重派前的关键读数：同一条目累计失败 ≥2 次时，
    原样重派大概率原样再败——该改的是任务书或补证据，而不是再派一次（backlog
    20261012-073700-leader 要求 4，源于夜航三次 run_failed 期间 status 只报
    needs_leader 不带次数）。
    """
    runs = (paths or _devops_paths()).runs
    if not runs.exists():
        return {}
    dones = sorted(runs.glob("*.done"))
    if limit > 0:
        dones = dones[-limit:]
    out: dict[str, int] = {}
    for p in dones:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        items = data.get("items") if isinstance(data, dict) else None
        if not isinstance(items, list):
            continue
        for it in items:
            if not isinstance(it, dict):
                continue
            if str(it.get("status", "")) not in FAILED_ITEM_STATUSES:
                continue
            cid = str(it.get("id", "") or "")
            if cid:
                out[cid] = out.get(cid, 0) + 1
    return out


def run_failed_streak(*, paths: WorkerPaths | None = None) -> int:
    """**倒序**遍历 run 摘要，返回尾部连续含 ``run_failed`` 条目的批次数（fail-soft）。

    「组包过载」的读数面（backlog 20261012-170301-leader）：多张大单并批 → 该跳
    max_turns 触顶 → 种子 run_failed——连续两批同形状就不是巧合。系统性故障不计入
    （渠道坏了条目无罪，语义同 :data:`FAILED_ITEM_STATUSES` 旁注）。坏文件断链不计
    也不续（无法读取的批次无资格证明任何事）。
    """
    runs = (paths or _devops_paths()).runs
    if not runs.exists():
        return 0
    streak = 0
    for p in reversed(sorted(runs.glob("*.done"))):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            break
        items = data.get("items") if isinstance(data, dict) else None
        if not isinstance(items, list):
            break
        if any(
            isinstance(it, dict) and str(it.get("status", "")) == "run_failed"
            for it in items
        ):
            streak += 1
        else:
            break
    return streak


def _extract_commit(body: str) -> str:
    """从交付正文尽力提取 commit hash（``commit ... <hex>``）；无则空串。"""
    m = _COMMIT_RE.search(body or "")
    return m.group(1) if m else ""


#: 正文提取与分支查询双双失败时写进 ``.done`` 的 commit 字段注记（不留空串——空串读起来
#: 像「没提交」，而实际多半是「报了但没被抓到」，backlog 20261011-220621-leader 要求 4）。
COMMIT_UNRESOLVED = "unresolved"


def merge_commit_of(worktree_name: str) -> str:
    """按 worktree/分支名回溯其并入 main 的 merge commit（短哈希）；查不到返回空串。

    ``wt-<backlog-id>`` 是任务书纪律（条目 id 即 worktree 名），``pysci-dev worktree
    merge`` 走 ``git merge --no-ff`` 产生默认消息 ``Merge branch 'wt-<id>'``，故 ``--grep``
    回溯即可定位——分支即便已被 ``worktree remove`` 删掉也照样查得到。
    """
    if not worktree_name:
        return ""
    rc, out = _run_git(
        "log", "--merges", "-n", "1", "--format=%h", f"--grep=wt-{worktree_name}"
    )
    if rc != 0:
        return ""
    for line in out.splitlines():
        token = line.strip()
        if re.fullmatch(r"[0-9a-f]{7,40}", token):
            return token
    return ""


def resolve_commit(body: str, worktree_name: str) -> str:
    """交付正文抓 hash → 抓不到按 ``wt-<id>`` 分支查 merge commit → 再查不到写注记。

    实锤失配形状（E0-B 条目）：正文写 ``合并 commit hash：f2e0840``，而 "hash" 里的 a 是
    十六进制字母，:data:`_COMMIT_RE` 的间隔类 ``[^0-9a-f]{0,16}`` 跨不过去 → 正则整体
    失配、账上留下 ``commit=""``（merge commit 其实一直在 ``git log`` 里可查）。
    """
    return _extract_commit(body) or merge_commit_of(worktree_name) or COMMIT_UNRESOLVED


def _extract_backlog_ids(body: str) -> list[str]:
    """从交付正文提取 ``backlog id=<id1>,<id2>,…`` 清单（本次完成的 backlog id）。

    多处出现合并去重（保序）；无匹配返回空列表。批量消化据此逐 id 销账。
    """
    ids: list[str] = []
    for m in _BACKLOG_IDS_RE.finditer(body or ""):
        for tok in re.split(r"[,、]", m.group(1)):
            tok = tok.strip()
            if tok and tok not in ids:
                ids.append(tok)
    return ids


# ---------------------------------------------------------------------------
# 孤儿 worktree 清理（reclaim 顺带；与 pysci-dev worktree 同布局：.qoder/worktrees/<name> + wt-<name>）
# ---------------------------------------------------------------------------
def _run_git(*args: str) -> tuple[int, str]:
    """在项目根运行 git，返回 (rc, 合并输出)。超时/异常一律 fail-open（返回非零、不抛）。

    **走 :func:`procs.run_hidden`**（2026-10-15 加急「空终端风暴」）：本函数是看门狗 tick 与
    各合并/回收路径的 git 唯一落点，而 tick 由**无控制台**的 pythonw 驱动——裸 ``subprocess.run``
    会让每次 git 都新开一个可见窗口。
    """
    try:
        proc = procs.run_hidden(
            ["git", *args],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 1, str(exc)
    return proc.returncode, ((proc.stdout or "") + (proc.stderr or "")).strip()


# ---------------------------------------------------------------------------
# 抢救面（backlog 20261012-234000-leader：未收尾进度先保住，再谈清理）
# ---------------------------------------------------------------------------
#: 抢救提交的消息标记：机器的保命提交必须与人的里程碑提交一眼可分——前者不该被评审，
#: 更不能被当成「这一单做完了」的销账证据。
SALVAGE_COMMIT_TAG = "wip(salvage)"

#: 抢救提交的 hash 长度（与 :func:`merge_commit_of` 的短 hash 口径一致，够任务书指认）。
SALVAGE_HEAD_LEN: int = 12


def worktree_path(name: str) -> Path:
    """条目隔离面的目录（与 ``pysci-dev worktree add`` 同一布局，唯一构造点）。"""
    return workflow.WORKTREES_ROOT / name


def _rel_path(path: Path) -> str:
    """项目根相对 posix 串（进任务书给组员看；根外路径退回绝对 posix 串）。"""
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _face_registered(path: Path) -> bool:
    """该目录是否**登记在册**（``git worktree list`` 里有它）。

    区分「你的隔离面还在」和「残骸还趴着」的唯一可靠依据。残骸目录里那个 ``.git``
    文件指向已删的管理项，进去跑 ``git -C`` 的读数不可信（最坏情形回落到父仓工作区，
    把主仓的脏面算成这个条目的进度）——所以判脏之前先问登记，未登记的一面绝不进去写。
    """
    rc, out = _run_git("worktree", "list", "--porcelain")
    if rc != 0:
        return False
    want = path.as_posix().lower()
    for line in out.splitlines():
        if line.startswith("worktree "):
            if Path(line[len("worktree ") :].strip()).as_posix().lower() == want:
                return True
    return False


def worktree_salvage_state(name: str, *, main_ref: str = "main") -> dict[str, Any]:
    """孤儿 worktree「还有什么没保住」的只读读数——回收动手前唯一的依据。

    读不出证据时一律往「有东西要救」的方向判（``unknown``）：``worktree remove --force``
    是这条链路上唯一不可逆的动作，误判干净＝销毁进度，误判脏＝只是多留一个面没人删，
    两个代价不对称（backlog 20261012-234000-leader 的裁定）。

    Args:
        name: 隔离面名（= 条目 id 或 ``<id>-r<gen>``）。
        main_ref: 比较基准分支，「未合并」＝这条分支领先它的提交数。

    Returns:
        ``{worktree, path, present, registered, branch, branch_exists, dirty, unmerged,
        head, unknown, salvageable}``；``salvageable`` 是「有东西要救或证据不足」的合取。
    """
    path = worktree_path(name)
    branch = f"wt-{name}"
    present = path.is_dir()
    state: dict[str, Any] = {
        "worktree": name,
        "path": path,
        "present": present,
        "registered": present and _face_registered(path),
        "branch": branch,
        "branch_exists": False,
        "dirty": False,
        "unmerged": 0,
        "head": "",
        "unknown": False,
    }
    rc, _out = _run_git("rev-parse", "--verify", "--quiet", f"refs/heads/{branch}")
    state["branch_exists"] = rc == 0
    rc, out = _run_git("rev-parse", "--short", branch)
    if rc == 0 and out.strip():
        state["head"] = out.strip()[:SALVAGE_HEAD_LEN]
    rc, out = _run_git("rev-list", "--count", f"{main_ref}..{branch}")
    if rc == 0 and out.strip().isdigit():
        state["unmerged"] = int(out.strip())
    else:
        state["unknown"] = True
    if state["registered"]:
        rc, out = _run_git("-C", str(path), "status", "--short")
        if rc == 0:
            state["dirty"] = bool(out.strip())
        else:
            state["dirty"] = True
            state["unknown"] = True
    elif present:
        # 在册与否分不清时按「目录里有东西」算：宁可多留一个面，不赌它干净
        state["dirty"] = True
        state["unknown"] = True
    state["salvageable"] = bool(
        (state["present"] or state["branch_exists"])
        and (state["dirty"] or state["unmerged"] > 0 or state["unknown"])
    )
    return state


def worktree_faces() -> list[str]:
    """在册 worktree 的**面名**（``git worktree list`` ∩ 本编排的面根）——检出与回收共用一份枚举。

    只认落在 ``workflow.WORKTREES_ROOT`` 下的面（别人的 worktree/主树不算），
    ``git`` 不可用或无面 → ``[]``（读不出就静默，不猜）。
    """
    rc, out = _run_git("worktree", "list", "--porcelain")
    if rc != 0:
        return []
    root = workflow.WORKTREES_ROOT
    names: list[str] = []
    for line in out.splitlines():
        if not line.startswith("worktree "):
            continue
        p = Path(line[len("worktree ") :].strip())
        try:
            rel = p.relative_to(root)
        except ValueError:
            continue  # 主树/仓外树：不是本编排的面
        if rel.parts and str(rel.parts[0]) not in names:
            names.append(str(rel.parts[0]))
    return names


#: 「已提交未合并」的 (branch, head) → (unmerged, 分叉点) 缓存（**进程内**；watchdog 长驻）。
#: 为什么要它（用户 2026-10-15 加急「空终端风暴」）：看门狗 tick 一分钟一次，逐面跑
#: `worktree_salvage_state`（每面约 4 条 git）＝ 一次 tick 十几到二十几个 console 子进程；
#: 它们落在**没有控制台**的 pythonw 里 ⇒ 每个都新开可见窗口闪现。改为「两次批量调用 + 结果
#: 缓存」：**空闲 tick 的 git 派生 = 2**，只有 head 变了的面才再花两条。
_UNMERGED_CACHE: dict[tuple[str, str], tuple[int, str]] = {}


def unmerged_worktree_alerts(*, main_ref: str = "main") -> list[dict[str, Any]]:
    """「**已提交、未合并**」的面 → 只读告警读数（**零新造判据**，backlog 20261015-145045）。

    实案（2026-10-15 两起，用户征询后裁定必修）：devops 尾段「提交→合并→推送」被进程杀死
    /跳截断/重启时会停在**已提交未合并**——成果卡在 worktree 里静默躺平（实案 6304 卡 3
    小时），零告警。判据**复用** :func:`worktree_salvage_state` 的 ``unmerged`` 口径
    （``git rev-list --count main..branch``），不新造第二份。

    **批量 + 缓存**（2026-10-15 加急 `adhoc-storm-fix`）：一次 ``worktree list --porcelain``
    拿全部面、一次 ``for-each-ref`` 拿全部面分支的 head；领先数与分叉点按 ``(branch, head)``
    缓存——**无变化的 tick 一条 git 都不再花**（空闲 tick 恒 2 条）。

    **只检出，不合并**（组长 2026-10-15 裁决）：patrol 自动 merge 进 main 是护栏级写操作
    （与「一个 worktree 一个写者」「合并前交集闸」同级）——故读数里附**现成合并命令**，
    由组长有意授权后一键执行。

    Returns:
        每面一条 ``{worktree, branch, head, unmerged, behind_point, merge_cmd, title}``。
    """
    rc, out = _run_git("worktree", "list", "--porcelain")
    if rc != 0:
        return []
    root = workflow.WORKTREES_ROOT
    faces: list[tuple[str, str]] = []
    for line in out.splitlines():
        if not line.startswith("worktree "):
            continue
        p = Path(line[len("worktree ") :].strip())
        try:
            rel = p.relative_to(root)
        except ValueError:
            continue  # 主树/仓外树：不是本编排的面
        if rel.parts:
            name = str(rel.parts[0])
            faces.append((name, f"wt-{name}"))
    if not faces:
        return []
    rc, refs = _run_git("for-each-ref", "--format=%(refname:short) %(objectname:short)")
    heads = {
        parts[0]: parts[1]
        for parts in (ln.split() for ln in refs.splitlines())
        if len(parts) >= 2
    }
    out_alerts: list[dict[str, Any]] = []
    for name, branch in faces:
        head = heads.get(branch)
        if not head:
            continue  # 分支不在（已合并删除）→ 不是「已提交未合并」
        cached = _UNMERGED_CACHE.get((branch, head))
        if cached is None:
            rc, cnt = _run_git("rev-list", "--count", f"{main_ref}..{branch}")
            n = int(cnt.strip()) if rc == 0 and cnt.strip().isdigit() else 0
            point = "?"
            if n > 0:
                rc, base = _run_git("merge-base", main_ref, branch)
                point = base.strip()[:SALVAGE_HEAD_LEN] if rc == 0 else "?"
            _UNMERGED_CACHE[(branch, head)] = (n, point)
        else:
            n, point = cached
        if n <= 0:
            continue  # 已合并（或无独有提交）→ 不误报
        out_alerts.append(
            {
                "worktree": name,
                "branch": branch,
                "head": head,
                "unmerged": n,
                "behind_point": point,
                "merge_cmd": f"pysci-dev worktree merge {name}",
                "title": f"面 {name} 有 {n} 个提交未合并（{branch} @ {head}）",
            }
        )
    return out_alerts


def announce_unmerged_worktrees(
    seen: set, *, main_ref: str = "main"
) -> list[dict[str, Any]]:
    """检出「已提交未合并」并**出声**（NEEDS_LEADER）；``seen`` 由调用方持有做进程内去重。

    ``seen`` 语义：``{(branch, head)}``——同一面同一 head 只喊一次（watchdog 是长驻进程，
    tick 一分钟一次，不去重就是每分钟一条噪声）；head 变了（又有新提交）会再喊一次。
    **不落盘**：本单白名单不含 ``.gitignore``，为去重新增 state 文件会在 main 留 ``??``。
    """
    from . import events as _events

    fresh: list[dict[str, Any]] = []
    for a in unmerged_worktree_alerts(main_ref=main_ref):
        key = (a["branch"], a["head"])
        if key in seen:
            continue
        seen.add(key)
        fresh.append(a)
        try:
            _events.emit(
                _events.NEEDS_LEADER,
                subject=str(a["branch"]),
                detail=(
                    f"「已提交未合并」：{a['title']}；与 {main_ref} 的分叉点 "
                    f"{a['behind_point']}。patrol **不自动合并**（护栏级写操作），"
                    "请有意授权后执行："
                ),
                action=str(a["merge_cmd"]),
            )
        except Exception:  # noqa: BLE001 - 出声失败不挡巡检
            pass
    return fresh


def salvage_worktree(
    name: str,
    run_log: Path | None = None,
    *,
    entry_id: str = "",
    reason: str = "",
    state: dict[str, Any] | None = None,
    quarantined: bool = False,
) -> dict[str, Any] | None:
    """把面上没保住的进度落成一条 **WIP-salvage 提交**，分支与工作树都不删。

    ``git add -A`` 收未提交改动 → ``commit`` 成抢救提交。**这里的 ``--no-verify`` 是
    必需品不是违规**：pre-commit 跑 ruff/测试，半制成品几乎必挂，钩子一失败抢救就静默
    不发生——那正是本单要消灭的「进度靠手动才保得住」。所以提交带
    :data:`SALVAGE_COMMIT_TAG` 标记，评审与销账面都不把它当交付（里程碑提交仍是常态纪律）。

    Args:
        name: 隔离面名。
        run_log: 写日志面（None 静默）。
        entry_id: 原条目 id（进提交消息，让续作的人知道这坨是谁的活儿）。
        reason: 成因读数（``crash`` / 交付 kind），进消息与任务书。
        state: 已读好的 :func:`worktree_salvage_state`（省一遍 git；缺省现场读）。
        quarantined: 旧会话进程没能确认离场（第四面判 hot）。此时**绝不往别人的树里写**
            ——并发 ``add``/``commit`` 会把它的半成品卷进我们的提交，只读记账不碰工作树。

    Returns:
        抢救记录（``workflow.backlog_attach_salvage`` 等的输入）；没什么可救 → None。
        抢救动作失败时**仍返回记录**（``uncommitted=True``），因为「没救成」恰恰意味着
        这个面更不能被删。
    """
    st = state or worktree_salvage_state(name)
    if not st["salvageable"]:
        return None
    uncommitted = bool(st["dirty"])
    wrote = False
    if st["present"] and not st["registered"] and not quarantined:
        _log(
            run_log,
            f"[!] 抢救 {name}：目录在但**未登记为 worktree**（remove --force 的残骸）"
            "——不进去写（git 读数不可信），面原样保留、进度靠分支取回",
        )
    elif st["registered"] and uncommitted and not quarantined:
        rc_add, out_add = _run_git("-C", str(st["path"]), "add", "-A")
        msg = (
            f"{SALVAGE_COMMIT_TAG}: 抢救 {entry_id or name} 的未收尾进度（{_now()}）\n\n"
            f"成因={reason or '未记录'}。本提交由 drain 孤儿回收自动打出：不是交付、"
            "勿据此销账，续作者应在其上补完。"
        )
        rc_cm, out_cm = _run_git(
            "-C", str(st["path"]), "commit", "--no-verify", "-m", msg
        )
        if rc_cm == 0:
            wrote = True
        elif rc_add != 0:
            _log(
                run_log,
                f"[!] 抢救 {name}：git add 失败（{out_add[:120]}）——面与分支原样保留",
            )
        else:
            # 「nothing to commit」也走这一支：读数说脏、落盘说不脏（行尾规范化/并发写），
            # 不重复动手，但把差异写清楚，别让人以为进度已经进提交。
            _log(
                run_log,
                f"[!] 抢救 {name}：commit 未发生（{out_cm[:120]}）——面与分支原样保留",
            )
    rc, head = _run_git("rev-parse", "--short", st["branch"])
    if rc == 0 and head.strip():
        st["head"] = head.strip()[:SALVAGE_HEAD_LEN]
    record: dict[str, Any] = {
        "worktree": name,
        "branch": st["branch"],
        "commit": st["head"],
        "path": _rel_path(Path(st["path"])),
        "reason": reason,
        "entry": entry_id,
        "registered": bool(st["registered"]),
        "present": bool(st["present"]),
        "uncommitted": bool(uncommitted and not wrote),
        "unmerged_before": int(st["unmerged"] or 0),
        "salvaged_at": _now(),
    }
    _log(
        run_log,
        f"[+] 抢救孤儿 worktree {name}：分支 {st['branch']} 保留、提交 "
        f"{record['commit'] or '(无)'}"
        + ("（未提交部分未能自动取回）" if record["uncommitted"] else "")
        + ("［隔离：旧会话进程未确认离场，未动工作树］" if quarantined else ""),
    )
    return record


def cleanup_orphan_worktrees(
    names: list[str],
    run_log: Path | None = None,
    *,
    salvage_sink: dict[str, dict[str, Any]] | None = None,
    quarantined: Any = (),
) -> list[str]:
    """回收死 worker 的 worktree/branch——**先抢救没能确认干净的**。

    每个 name 三分支：

    - **确认干净**（无未提交改动、分支不领先 main）：``worktree remove --force`` +
      ``branch -D``，与旧行为逐字相同（「不回归」的那一支）；
    - **有未收尾内容**：先 :func:`salvage_worktree` 落成 WIP 提交，**分支与工作树都不删**
      ——force-remove 只对被证明干净的面开放。抢救读数写进 ``salvage_sink``，调用方据此
      注记条目 / 自动续作（原行为是把这些直接销毁）；
    - **隔离中**（``quarantined``，旧会话没能确认离场）：不碰工作树，只读记账 + 保面。

    全程 fail-open（git 缺失/超时/该 worktree 不存在均忽略）；**只动传入的孤儿名**——
    活跃/手动 worktree（别名）不受影响，不会误删并发的手动作业。

    Args:
        names: 孤儿条目 id 列表（= worktree 名，见 :func:`workflow.backlog_take_first`）。
        run_log: 写日志面。
        salvage_sink: 可选字典，收 ``{name: 抢救记录}``（None 则只保不报）。
        quarantined: 须隔离对待的名字（旧写者还在/摘不掉），不往里写。

    Returns:
        实际清理到（worktree 或 branch 至少其一删除成功）的 name 列表——**抢救过的面
        不出现在这里**，它没被清掉，是有意保留的续作面。
    """
    cleaned: list[str] = []
    guard = {str(q) for q in quarantined if q}
    for name in names:
        if not name:
            continue
        state = worktree_salvage_state(name)
        if state["salvageable"]:
            record = salvage_worktree(
                name,
                run_log,
                entry_id=name,
                reason="crash",
                state=state,
                quarantined=name in guard,
            )
            if record is not None and salvage_sink is not None:
                salvage_sink[name] = record
            continue
        rc_wt, _ = _run_git("worktree", "remove", "--force", str(worktree_path(name)))
        rc_br, _ = _run_git("branch", "-D", f"wt-{name}")
        if rc_wt == 0 or rc_br == 0:
            cleaned.append(name)
            _log(
                run_log,
                f"[i] 清理孤儿 worktree：{name}（worktree rc={rc_wt}, branch rc={rc_br}）",
            )
    _run_git("worktree", "prune")  # 兜底：清悬挂管理项
    return cleaned


def worktree_dir_present(name: str) -> bool:
    """孤儿 worktree 的目录在 ``cleanup_orphan_worktrees`` 之后是否仍趴在磁盘上。

    为什么单独立一个判据：``git worktree remove --force`` 返回 0 **不等于**目录真没了。
    Windows 上只要有进程还开着该目录里的句柄（典型=还没被终止的 headless 会话），删除
    就是异步/半途的；把「rc=0」当「已清空」再重建同名 worktree，等于把两个写者请回同
    一个房间（backlog 20261012-184000-leader 的实证路径）。

    抢救过的面**该不该走到这里**看登记状态（:func:`_face_registered`）：在册的面留在原地是
    设计意图——进度已在分支上，同名续完正是「零手动」要的那条路；目录虽在却未在册的是
    上一代 remove 的残骸，必须升代，否则同名重建就是把两个写者请回同一个房间。
    """
    if not name:
        return False
    return worktree_path(name).exists()


# ---------------------------------------------------------------------------
# 孤儿会话进程回收（第四面；backlog 20261012-184000-leader）
# ---------------------------------------------------------------------------
#: 终止后复查存活的等待秒数（杀进程树是异步的，立即复查会假报「还活着」）。
KILL_GRACE_S: float = 1.5


def orphan_hop_sessions(orphans: list[str]) -> list[dict[str, Any]]:
    """从运行态会话池里捞出「还在为这些孤儿条目写盘」的在途跳。

    归属靠 :func:`registry.note_hop_pid` 在派生瞬间记下的两键：``pid``（会话进程 OS
    pid）与 ``pid_task``（任务书文件名 ``task-<派发时刻>-<slug>.md``，slug=种子条目 id）。
    跳正常收尾时该 pid 会被常规会话池写覆盖掉，所以**看得到 pid 就等于这跳没走完**——
    正是 worker 被中途杀死时唯一留下的活口线索。

    Returns:
        ``[{"member", "sid", "pid", "task", "entry_id", "entry"}…]``（``entry`` 是会话池
        里的原字典，供调用方就地改 pid；``entry_id`` 为反解出的条目 id）。
    """
    wanted = [str(i) for i in orphans if i]
    if not wanted:
        return []
    reg = Registry.load()
    hits: list[dict[str, Any]] = []
    members = reg.runtime.get("members", {})
    if not isinstance(members, dict):
        return []
    for mid, entry in members.items():
        for s in (entry or {}).get("sessions", []) or []:
            try:
                pid = int(s.get("pid") or 0)
            except (TypeError, ValueError):
                continue
            task = str(s.get("pid_task") or "")
            if pid <= 0 or not task:
                continue
            matched = next((oid for oid in wanted if task.endswith(f"-{oid}.md")), None)
            if matched is None:
                continue
            hits.append(
                {
                    "member": str(mid),
                    "sid": str(s.get("sid", "")),
                    "pid": pid,
                    "task": task,
                    "entry_id": matched,
                    "entry": s,
                }
            )
    return hits


def reclaim_orphan_sessions(
    orphans: list[str],
    run_log: Path | None = None,
    *,
    alive=None,
    is_session=None,
    terminate=None,
    sleep_s: float | None = None,
) -> list[str]:
    """孤儿回收的**第四面**：终止旧 worker 遗留、仍在写隔离面的 headless 会话进程。

    原本回收只动三面（锁、backlog 条目、worktree 名），漏了进程：worker 被杀时它派生
    的会话进程活在独立进程树里，继续往同名 worktree 写；下一次 drain 删掉那个 worktree
    再重派同单，新会话与旧会话就并发双写（2026-10-12 实测：新会话读 25 轮才发现「本
    会话零成功写入」并主动停手上报）。

    动手前必过两道闸，因为 pid 会被系统复用：

    1. :func:`procs.pid_alive` —— 不活就只是陈记账，擦掉即可；
    2. :func:`procs.pid_is_session` —— 命令行里得带着那个 sid。对不上就**绝不终止**
       （那可能是不相干的进程），宁可将该条目列为待异名重建。

    终止后复查仍存活（杀不动/孙进程赖着）→ 同样列为待异名重建。

    Args:
        orphans: 本次回收到的孤儿条目 id（= 任务书里嵌的种子 id）。
        run_log: 写日志面（None 静默）。
        alive / is_session / terminate: 注入点（测试钉两条通路用），默认走
            :mod:`procs` 与 :func:`_pid_alive` 的真实实现。
        sleep_s: 终止后复查前的等待秒数。

    Returns:
        仍可能被旧写者占据的条目 id（调用方据此走「异名隔离重建」，禁止复用旧名）。
    """
    alive = alive or _pid_alive
    is_session = is_session or procs.pid_is_session
    terminate = terminate or procs.terminate_tree
    wait = KILL_GRACE_S if sleep_s is None else sleep_s
    hits = orphan_hop_sessions(orphans)
    hot: list[str] = []
    for hit in hits:
        oid = hit["entry_id"]
        sid8 = hit["sid"][:8]
        label = f"{hit['member']}/{sid8} pid={hit['pid']} 任务书={hit['task']}"
        if not alive(hit["pid"]):
            _log(run_log, f"[i] 第四面回收：孤儿会话进程已自灭，清记账：{label}")
            _clear_hop_pid(hit)
            continue
        if not is_session(hit["pid"], hit["sid"]):
            _log(
                run_log,
                f"[!] 第四面回收：pid={hit['pid']} 已不是会话 {sid8}"
                "（pid 被复用/命令行不符）——**不动手**，条目改走异名重建",
            )
            _clear_hop_pid(hit)
            hot.append(oid)
            continue
        _log(run_log, f"[i] 第四面回收：终止孤儿会话进程 → {label}")
        done, echo = terminate(hit["pid"])
        if not done:
            _log(run_log, f"[!] 第四面回收：终止失败（{echo}）→ 条目改走异名重建")
            _clear_hop_pid(hit)
            hot.append(oid)
            continue
        if wait:
            time.sleep(wait)
        if alive(hit["pid"]):
            _log(
                run_log,
                f"[!] 第四面回收：已发终止但 pid={hit['pid']} 仍存活"
                "（孙进程/内核挂起）→ 条目改走异名重建",
            )
            _clear_hop_pid(hit)
            hot.append(oid)
        else:
            _clear_hop_pid(hit)
    return hot


def _clear_hop_pid(hit: dict[str, Any]) -> None:
    """抹掉一条会话的在途 pid 记账（处置完不留陈标记，免得下次对着复用的 pid 误动手）。"""
    note_hop_pid(hit["member"], hit["sid"], pid=0, task="")


# ---------------------------------------------------------------------------
# drain 循环
# ---------------------------------------------------------------------------

#: 交集类 blocked 的标签（词表 ``state/failure-tags.json``，backlog 20261013-102152-devops）。
INTERSECT_TAG = "merge_dirty_face"

#: ``pysci-dev worktree merge`` 的拒合并行：``[✗] 交集 N 项——…``，其后缩进逐行列出相交文件。
_INTERSECT_HEAD_RE = re.compile(r"\[✗\]\s*交集\s*(\d+)\s*项")

#: 看起来是仓内相对路径的行（含 ``/`` 才认；``处置：…`` 这类中文散文不算）。
_PATHISH_RE = re.compile(r"^[\w.\-]+(?:/[\w.\-]+)+$")


def merge_intersect_cause(outcome: Any) -> tuple[bool, list[str]]:
    """这一跳的 blocked 是不是「main 脏面与本次合并相交」——并尽量带上交集文件清单。

    为什么要在抢救之前问成因：抢救假设「上一跳干了活没干完，下一跳接着干」，而交集类
    成因里**下一跳改变不了 main 的脏面**。2026-10-13 上午实测三连跳
    （``072000-leader`` → ``101105`` → ``101705``）三跳树哈希逐字节相同（``b270e8cd``）、
    测试结果相同，纯烧 worktree 与测试轮次，第 4 跳才撞上 ``SALVAGE_MAX_CHAIN``。

    两个通道任一命中即认定（组员有词可标、代码有文可辨，缺一面就会漂移）：``tags`` 含
    :data:`INTERSECT_TAG`，或正文里有工具打印的 :data:`_INTERSECT_HEAD_RE` 行。清单只从
    拒合并行**之后**的缩进路径行取；读不到清单仍算交集类——认成因比凑清单重要。

    Args:
        outcome: 本跳 :class:`dispatch.DispatchOutcome`。

    Returns:
        ``(是否交集类, 相交文件清单)``；非 blocked 或非交集成因 → ``(False, [])``。
    """
    if getattr(outcome, "kind", "") != "blocked":
        return False, []
    body = outcome.body or ""
    tagged = INTERSECT_TAG in (outcome.tags or [])
    lines = body.splitlines()
    head_i = next(
        (i for i, ln in enumerate(lines) if _INTERSECT_HEAD_RE.search(ln)), -1
    )
    if not tagged and head_i < 0:
        return False, []
    offenders: list[str] = []
    # 清单从拒合并行的**下一行**起算：同一行的尾巴（「——本单提交与 main…」）不是文件。
    for ln in lines[head_i + 1 :] if head_i >= 0 else []:
        if not ln.strip():
            continue
        if not ln[:1].isspace():
            break  # 缩进块结束：后面是组员自己的正文，不是工具读数
        tok = ln.strip().strip("`*")
        if _PATHISH_RE.match(tok) and tok not in offenders:
            offenders.append(tok)
    return True, offenders


def _intersect_hold_text(face: str, offenders: list[str]) -> str:
    """停链结论 + 组长的最小处置（note 里必须可执行，不留「等组长看着办」）。"""
    files = (
        "、".join(offenders)
        if offenders
        else "（组员未贴清单，全文见 outbox 审计副本）"
    )
    return (
        f"\n【停链判定=成因是 main 脏面与本次合并相交，重跑无用】待清文件：{files}；"
        f"处置：在 main 提交或还原上述文件后 "
        f"`pysci-dev worktree merge {face}`"
        f"（分支 wt-{face} 与抢救提交已保住进度，无需再派 worker 重做）"
    )


def _salvage_progress(
    seed: dict[str, Any], run_log: Path, *, reason: str, stop_chain: bool = False
) -> dict[str, Any] | None:
    """失败/停止收场的种子：先保住面上的未收尾进度，再让队列自动接手续完。

    两条通路按条目此刻的状态分（组长裁决 2026-10-12 23:4x「全自动，不靠手动盘点」）：

    - 条目还在队列里（系统性故障 → 已复位 pending）：就地注记抢救事实，下一跳回到**同一
      棵树**接着干（:func:`workflow.backlog_attach_salvage`）；
    - 条目已落 needs_leader（blocked/run_failed/parse_error/验收 FAIL → 不会再自动重派）：
      另立 ``source=salvage`` 续作单插到 pending 队首，原条目仍是 needs_leader——
      「这一跳失败了」是组长要看到的事实，代码不替他忘掉。

    Args:
        seed: 本批种子条目。
        run_log: run 日志面。
        reason: 成因读数（进提交消息与任务书，续作者据此判读）。
        stop_chain: 成因对续作无效时置真（如 main 脏面交集：再派一个 worker 改变不了
            相交文件，只会多立一张空转单）。进度照样抢救成提交、面与分支都留着，
            只是**不立续作单**——处置权交回组长的 note。

    Returns:
        抢救记录（附 ``continuation`` 若有续作单）；面上没东西可救 / 抢救未成 → 记录或 None。
    """
    name = worktree_name(seed)
    state = worktree_salvage_state(name)
    if not state["salvageable"]:
        return None
    record = salvage_worktree(
        name,
        run_log,
        entry_id=str(seed.get("id", "")),
        reason=reason,
        state=state,
    )
    if record is None:
        return None
    if workflow.backlog_attach_salvage(name, record) is not None:
        _log(
            run_log,
            f"[+] 未收尾进度已抢救并就地续作：{name}（提交 "
            f"{record['commit'] or '无'}，成因 {reason}）",
        )
        return record
    if stop_chain:
        _log(
            run_log,
            f"[+] 未收尾进度已抢救：{name}（提交 {record['commit'] or '无'}，成因 {reason}）"
            f" → **停链**：成因是 {reason}，再派 worker 改变不了它，不立续作单"
            f"（进度在分支 wt-{name}，处置见条目 note）",
        )
        return record
    cont = workflow.backlog_append_salvage(seed, record)
    _log(
        run_log,
        f"[+] 未收尾进度已抢救：{name}（提交 {record['commit'] or '无'}，成因 {reason}）"
        + (
            f" → 自动续作单 {cont['id']} 已插到队首"
            if cont
            else " → 续作单未立（同一提交已在队或抢救链已到顶，见条目 note）"
        ),
    )
    if cont:
        record = {**record, "continuation": cont["id"]}
    return record


def _batch_faces(
    seed: dict[str, Any], group: list[dict[str, Any]]
) -> tuple[str, list[str], list[str], str]:
    """批次四件套：种子 id / 组包 id / 整批 id / 标签（``seed`` 或 ``seed(+N)``）。"""
    seed_id = str(seed.get("id", ""))
    group_ids = [str(g.get("id", "")) for g in group if g.get("id")]
    label = seed_id if not group_ids else f"{seed_id}(+{len(group_ids)})"
    return seed_id, group_ids, [seed_id, *group_ids], label


def _dry_settle_batch(
    seed: dict[str, Any], group: list[dict[str, Any]], run_log: Path
) -> list[dict[str, Any]]:
    """dry 支：模拟 devops 吃下整批（种子+组包），逐 id 销账（计划器演练用）。"""
    seed_id, _group_ids, batch_ids, label = _batch_faces(seed, group)
    supersede_batch_entries(set(batch_ids))  # dry 支同走去重口径
    t0 = time.time()
    for bid in batch_ids:
        backlog_complete(
            bid,
            note="dry 干跑（未实派 devops）",
            batch_size=len(batch_ids),
            batch_seed=seed_id,
        )
    _log(run_log, f"[dry] 批次 {label} 干跑销账 {len(batch_ids)} 项完成")
    return [
        {
            "id": bid,
            "status": "dry",
            "batch_size": len(batch_ids),
            "batch_seed": seed_id,
            "elapsed_s": round(time.time() - t0, 1),
        }
        for bid in batch_ids
    ]


def _settle_outcome(
    seed: dict[str, Any],
    group: list[dict[str, Any]],
    outcome: Any,
    run_log: Path,
    t0: float,
) -> list[dict[str, Any]]:
    """批次派发结果 → backlog 账务（销账 / needs_leader / 延期 / 系统性停止）。

    自 backlog-fifo-20261014 起，执行由队列 worker 承担、本函数是**唯一**的账务实现
    （计划器 dry 支与 worker 真派发支共用；逐支语义与旧 ``_drain_batch`` 完全一致）：

    - **result**（code 0）：解析交付 ``backlog id=<ids>``，与本批取交集逐 id 销账 done
      （种子必含——必做项，防御性兜底）；组包中未选的项保持 pending 留待下轮；每个 done
      项记 ``batch_size``/``batch_seed``（供 orch stats 批量分布）。
    - **blocked/run_failed/parse_error/验收FAIL**：仅**种子** → needs_leader，组包项保持
      pending（**不连坐**）。种子隔离面里还有未收尾内容时先走 :func:`_salvage_progress`。
    - **系统性故障（``quota_exhausted`` / ``channel_error``）**：种子复位 pending、组包项
      保持 pending；返回 ``status=<kind>`` 由调用方触发全局停止（渠道坏了，不逐项
      needs_leader）；带宽类先延期（``deferred``，不停链），触顶才回升 ``channel_saturated``。

    Returns:
        per-id 结果字典列表（写入 .done 的 items）：种子/每个完成组包项各一条
        ``{id, status, commit?, kind?, reason?, batch_size?, batch_seed?, elapsed_s}``。
    """
    seed_id, group_ids, batch_ids, label = _batch_faces(seed, group)
    # quiet 通路读数逐条抄进 run 日志（字段语义与原实现不变）：不抄就没人知道
    # 「这一跳等了多久/在兜底档上跑」——台账有数但没人逐行看。
    if outcome.sid_new_reason:
        _log(
            run_log, f"[i] sid 在途闸：{outcome.sid_new_reason}，本批已自动换新会话续跑"
        )
    if outcome.route:
        _log(run_log, f"{outcome.route}（批次 {label}）")
    if outcome.hold:
        _log(run_log, f"{outcome.hold}（批次 {label}）")
    if outcome.same_seat_retries:
        _log(
            run_log,
            f"[!] 渠道瞬态故障同席重试 {outcome.same_seat_retries} 次"
            f"（{outcome.same_seat_retry_ms / 1000:.0f}s）（批次 {label}）",
        )
    for event in outcome.fallbacks:
        _log(run_log, f"{_channels.alert_line(event)}（批次 {label}）")
    if outcome.warning:
        _log(run_log, f"[!] {outcome.warning}（{seed_id}）")
    elapsed = round(time.time() - t0, 1)

    if outcome.code == 0:
        listed = _extract_backlog_ids(outcome.body)
        # 只销账本批内的 id；种子必 done（result=成功且种子为必做项，防御漏报）
        completed = [seed_id]
        for did in listed:
            if did in batch_ids and did not in completed:
                completed.append(did)
        # commit 逐 id 定位：正文抓得到就用抓到的，抓不到按 wt-<id> 分支回溯 merge
        # commit（worktree 名 = 条目 id 的任务书纪律），再查不到才落注记
        commits = {
            cid: resolve_commit(
                outcome.body, worktree_name(seed) if cid == seed_id else cid
            )
            for cid in completed
        }
        for cid in completed:
            backlog_complete(
                cid,
                note=f"drain 批量销账（commit {commits[cid]}）",
                batch_size=len(completed),
                batch_seed=seed_id,
            )
        _log(
            run_log,
            f"[√] 批次 {label} 完成：销账 {len(completed)} 项（{', '.join(completed)}；"
            f"{elapsed}s，"
            f"commit={', '.join(f'{cid}:{commits[cid]}' for cid in completed)}）",
        )
        return [
            {
                "id": cid,
                "status": "done",
                "commit": commits[cid],
                "batch_size": len(completed),
                "batch_seed": seed_id,
                "elapsed_s": elapsed,
            }
            for cid in completed
        ]

    # 系统性故障（额度耗尽 / 渠道级故障）：渠道坏了、不是任务书坏了 → 种子无罪复位
    # pending，组包保持 pending；调用方据此全局停止（不再逐单烧下一项）
    # **带宽类失败先延期、不停链**（用户裁决 2026-10-13，backlog 20261013-193500-leader）：
    # 渠道熔断的授权反转**仅限 bandwidth** —— 瞬态带宽不足的正确动作是等（背压），不是
    # 换席/停链。但延期有兜底：预算（次数 / 壁钟）触顶仍不愈 → **回升为全局停止**
    # （channel_saturated），封死「分类器误判/文案漂移导致逐项烧跳」。
    if outcome.kind in _systemic_kinds():
        plan = _bandwidth_defer_plan(outcome, seed, time.time())
        if plan is not None and plan.exhausted:
            kind = "channel_saturated"
            headline = _systemic_headline(kind)
            err = (
                "带宽延期触顶仍未愈（"
                + plan.reason
                + "）："
                + (outcome.error or outcome.body[:120] or "")
                .strip()
                .replace(chr(10), " ")
            )
            backlog_requeue(seed_id, note=f"{headline}全局停止，复位待渠道恢复")
            salv = _salvage_progress(seed, run_log, reason=kind)
            _log(
                run_log,
                f"[{_systemic_mark(kind)}] 批次 {label} 带宽延期触顶 → 回升全局停止：{err[:120]}",
            )
            return [
                {
                    "id": seed_id,
                    "status": kind,
                    "error": err[:200],
                    "salvage": (salv or {}).get("commit", ""),
                    "elapsed_s": elapsed,
                }
            ]
        if plan is not None:
            nth = plan.attempt + 1
            used = float(seed.get("defer_elapsed_s") or 0.0) + plan.delay_s
            workflow.backlog_defer(
                seed_id,
                until=plan.until,
                count=nth,
                elapsed_s=used,
                note=(
                    f"带宽延期第 {nth} 次（等 {plan.delay_s}s 后重试）"
                    "——带宽不足是瞬态，故不换席不停链；预算触顶则回升全局停止"
                ),
            )
            _log(
                run_log,
                f"[i] 批次 {label} 带宽延期第 {nth} 次（等 {plan.delay_s}s）"
                "→ 条目回队，继续吃下一项，**不停链**",
            )
            return [
                {
                    "id": seed_id,
                    "status": "deferred",
                    "defer_s": plan.delay_s,
                    "defer_count": nth,
                    "elapsed_s": elapsed,
                }
            ]
        kind = outcome.kind
        headline = _systemic_headline(kind)
        err = (outcome.error or outcome.body[:160] or "").strip().replace("\n", " ")
        backlog_requeue(seed_id, note=f"{headline}全局停止，复位待渠道恢复")
        # 停止不等价于「这一跳没干过活」：额度/渠道在半路断掉时，面上可能已经写了
        # 一半。条目复位回队后下一跳回到同一棵树续完，别从 main 重开。
        salv = _salvage_progress(seed, run_log, reason=kind)
        _log(
            run_log,
            f"[{_systemic_mark(kind)}] 批次 {label} {kind} → 种子复位回队，"
            f"触发全局停止：{err[:120]}",
        )
        return [
            {
                "id": seed_id,
                "status": kind,
                "error": err[:200],
                "salvage": (salv or {}).get("commit", ""),
                "elapsed_s": elapsed,
            }
        ]

    # 失败（blocked / run_failed / parse_error / 验收 FAIL）→ 仅种子 needs_leader；
    # 组包项保持 pending（不连坐：未被承诺、未 in_progress，留待下轮）
    # ⑤ needs_leader note 不截断全文（184000 修复面⑤）：note 是**唯一**留在 backlog 上的
    # 证据索引，截 160 字让两案关键证据变残片（违背 §0.4 留痕）。全文另在 dispatch 落
    # outbox 审计副本；note 一并留全文（err 只给 run 日志一行摘要）。
    err = (outcome.error or outcome.body[:160] or "").strip().replace("\n", " ")
    full = (outcome.error or outcome.body or "").strip()
    # 交集类失败（main 脏面与本次合并相交）单独分流：成因不在被派者的树上，重派一个
    # worker 改变不了它 → 抢救进度但**停链**（不立续作单），待清清单挂回原条目。
    isect, offenders = merge_intersect_cause(outcome)
    extra: dict[str, Any] = {}
    if isect:
        extra["merge_intersect"] = offenders
    # 「因缺授权/缺裁决 blocked」闭环（backlog 20261013-193058-devops）：条目挂标记 + 存
    # 原点名原文，重排时任务书注入「原点名 ↔ 本次补齐」对照行。判据＝交付自带的成因档位
    # （reason=needs_decision，受控词表），不猜正文——其余成因（余量/环境/上游）不需要授权。
    if str(getattr(outcome, "reason", "") or "") == "needs_decision":
        extra[AWAITING_DECISION_KEY] = True
        extra["blocked_report"] = full[:400]
    backlog_needs_leader(
        seed_id,
        note=(
            f"drain 跳过（交付={outcome.kind}）：{full}\n"
            # 点名**被派发者**的 outbox（seed["member"] 是提请者，不是写交付的人）
            f"【全文已留审计副本：pods/{outcome.member or 'devops'}/outbox/delivery-*.xml】"
            + (_intersect_hold_text(worktree_name(seed), offenders) if isect else "")
        ),
        extra=extra or None,
    )
    salv = _salvage_progress(
        seed,
        run_log,
        reason=str(outcome.kind or "needs_leader"),
        stop_chain=isect,
    )
    _log(
        run_log,
        f"[!] 批次 {label} 失败 → 种子 needs_leader（{outcome.kind}）"
        + (f"，交集 {len(offenders)} 项已停链（待组长清面后手动合并）" if isect else "")
        + f"，组包 {len(group_ids)} 项保持 pending 不连坐：{err[:120]}",
    )
    item_out: dict[str, Any] = {
        "id": seed_id,
        "status": "needs_leader",
        "kind": outcome.kind,
        "error": err[:200],
        "elapsed_s": elapsed,
    }
    # blocked 的成因档位（backlog 20261013-191213-devops）：透传进 .done 逐条，
    # 供 orch status 分列计数「待裁决 / 待排期」。未标注（空串）不落字段——缺省合法。
    reason = str(getattr(outcome, "reason", "") or "")
    if reason:
        item_out["reason"] = reason
    if isect:
        item_out["merge_intersect"] = offenders
    if salv:
        item_out["salvage"] = salv["commit"]
        if salv.get("continuation"):
            item_out["continuation"] = salv["continuation"]
    return [item_out]


def stop_kind_of(results: list[dict[str, Any]]) -> str:
    """结算结果里的系统性停止 kind（无则空串）——调用方据此停队/全局停止。"""
    return next(
        (
            str(r.get("status"))
            for r in results
            if str(r.get("status")) in _systemic_kinds()
        ),
        "",
    )


def settle_backlog_batch(
    item: dict[str, Any], outcome: Any, run_log: Path, *, t0: float
) -> tuple[str, list[dict[str, Any]], str]:
    """队列 worker 侧：结算一条 backlog 批次（``kind=BACKLOG_BATCH_KIND``）的账务。

    批次条目由 :func:`drain_backlog`（计划器）入队，``item["backlog"]`` 携种子/组包元数据；
    本函数把它交给 :func:`_settle_outcome`（唯一账务实现），并把结果映射成队列 worker 的
    动作面：

    Returns:
        ``(work_status, results, stop_kind)``——``work_status`` ∈ ``done``（正常收尾）/
        ``failed``（种子 needs_leader 或系统性停止）/``deferred``（带宽延期，条目由 worker
        挂 ``defer_until`` 重取）；``results`` 为该批的账务条目列表（**旧 drain 的 items
        形状**：``{id, status, commit?, kind?, reason?, batch_size?, batch_seed?…}``，首条为
        种子）；``stop_kind`` 非空 = 系统性全局停止（worker 停队，剩余条目保持 pending）。
    """
    meta = dict(item.get("backlog") or {})
    seed = dict(meta.get("seed") or {})
    group = [dict(g) for g in (meta.get("group") or [])]
    results = _settle_outcome(seed, group, outcome, run_log, t0)
    status = str(results[0].get("status") or "") if results else ""
    if status == "done":
        work_status = "done"
    elif status == "deferred":
        work_status = "deferred"
    else:
        work_status = "failed"
    return work_status, results, stop_kind_of(results)


def _seed_tier(item: dict) -> str:
    """条目所属的**优先级档名**（优先级表用的是档名，不是成员 id）。

    表里三档是成员名（leader/devops/inspector），另有两档是聚合名（research = 7 专职 +
    reviewer + deputy；bench-pressure = bencher 派生的压测会话）。故：命中档名原样用，
    其余成员归 ``research``（表的第 2 档就是这个聚合档）。
    """
    from . import settings as _settings

    name = str(item.get("member") or "")
    known = {t for grp in _settings.priority_order() for t in grp}
    return name if name in known else "research"


def _higher_priority_live(seed: dict, now: float) -> bool:
    """注册表里是否**有更高优先级的活会话**（unknown 类的二级判据用）。

    与 pid 判活互校是读侧的事：这里只排除心跳过期的死条目（:func:`backpressure.live_entries`）。
    """
    from . import backpressure as _bp
    from . import settings as _settings

    mine = _settings.priority_rank(_seed_tier(seed))
    return any(
        _settings.priority_rank(str(e.get("tier") or "")) < mine
        for e in _bp.live_entries(now=now)
    )


def _bandwidth_defer_plan(outcome: Any, seed: dict, now: float):
    """带宽类失败 → 延期判决；非带宽（或判不出且无更高优先级在场）→ ``None``。

    分流口径（用户裁决 2026-10-13，backlog 20261013-193500-leader）：
    ``bandwidth`` → 延期（跑通即止，不换席、不停链）；``outage``/``quota`` → 维持原
    全局停止；``unknown`` → 二级判据：有更高优先级活会话在场按 bandwidth，否则按 outage。
    """
    if str(getattr(outcome, "kind", "")) != "channel_error":
        return None
    from . import backpressure as _bp
    from . import failure_classes as _fc

    text = str(
        getattr(outcome, "error_text", "") or getattr(outcome, "error", "") or ""
    )
    cls = _fc.classify_failure(text, is_error=True)
    if cls == _fc.UNKNOWN and _higher_priority_live(seed, now):
        cls = "bandwidth"
    if cls != "bandwidth":
        return None
    return _bp.plan_defer(
        int(seed.get("defer_count") or 0),
        float(seed.get("defer_elapsed_s") or 0.0),
        now=now,
    )


def _devops_worker_live() -> bool:
    """devops 队列 worker 是否在跑（计划器扫尾闸：在途 in_progress 不是孤儿）。"""
    try:
        from . import queueworker as _qw

        return lock_status(_qw.member_paths("devops")).get("state") == "running"
    except Exception:  # noqa: BLE001 - 读数失败按「不在跑」保守处理（扫尾有恢复语义）
        return False


def _batch_meta(item: dict[str, Any]) -> dict[str, Any]:
    """队列条目的批次元数据（``queues.enqueue(extra={"backlog": …})`` 平铺进条目）。"""
    meta = item.get("backlog")
    return dict(meta) if isinstance(meta, dict) else {}


def _covered_ids(item: dict[str, Any]) -> set[str]:
    """该队列条目承诺覆盖的 backlog id 集（非批次条目回空集）。"""
    return {str(x) for x in (_batch_meta(item).get("ids") or [])}


def inflight_batch_ids(member: str = "devops") -> set[str]:
    """**在飞批次**已覆盖的 backlog id 集（queued/in_progress 的 ``backlog-batch`` 条目）。

    双记账根治（backlog 20261016-224211）：组包项按设计**保持 pending**（不连坐），于是
    下一轮计划器会把它们当新种子再取一次——同一 id 各自生成独立条目、重复烧掉一跳容量
    （实测：215945/215946 被 22:01 的组包消化，却又在 22:08/22:17 各生成独立条目）。
    取批前排除本集合，重复条目**不再产生**。

    **延期待重试的条目不算在飞**：``queues.defer`` 把条目复位 ``pending`` 并挂
    ``defer_until``——它已把覆盖的 id 交还 backlog（等带宽恢复后重取），若仍计作承诺面，
    同一队列里**后续条目会被整体挡住**（实测：``test_bandwidth_defers_and_keeps_going``
    的 b2 不再被取，延期支停链）。
    """
    from . import queues as _queues
    from . import workq as _workq

    covered: set[str] = set()
    for it in _workq.load(_queues.queue_path(member)):
        if str(it.get("kind") or "") != BACKLOG_BATCH_KIND:
            continue
        if str(it.get("status") or "") not in (
            _workq.STATUS_PENDING,
            _workq.STATUS_IN_PROGRESS,
        ):
            continue
        if str(it.get("status")) == _workq.STATUS_PENDING and _bp.item_deferred(
            it, time.time()
        ):
            continue  # 延期回队＝已交还承诺面（等到期重取，不挡别人）
        covered |= _covered_ids(it)
    return covered


def supersede_batch_entries(covered: set[str], *, keep: str = "") -> list[str]:
    """把**仍在待跑**（``pending``）且与本批覆盖面相交的批次条目**就地销账**。

    方向①（建议单 backlog 20261016-224211）：组包入队时撤销/标记其覆盖的 backlog id
    对应的**独立条目**。只动 ``pending``——``in_progress`` 的条目其 worker 已在跑，杀了
    反而制造孤儿；本批与它争同一 id 属计划器双取，由 :func:`inflight_batch_ids` 的排除面
    在源头堵死。销账条目不派跳（不派生会话），故重复 id 只被执行一次。

    Returns:
        被销账的条目 id 列表。
    """
    from . import queues as _queues
    from . import workq as _workq

    if not covered:
        return []
    qp = _queues.queue_path("devops")
    dropped: list[str] = []
    for it in _workq.load(qp):
        iid = str(it.get("id") or "")
        if not iid or iid == keep:
            continue
        if str(it.get("kind") or "") != BACKLOG_BATCH_KIND:
            continue
        if str(it.get("status") or "") != _workq.STATUS_PENDING:
            continue
        common = covered & _covered_ids(it)
        if not common:
            continue
        _workq.mutate(
            qp,
            iid,
            status=_workq.STATUS_DONE,
            result={
                "code": 0,
                "kind": "result",
                "body": (
                    "已被新批次覆盖，就地销账（双记账去重，未派跳）："
                    + "、".join(sorted(common))
                ),
                "checks": [],
                "deduped": sorted(common),
            },
        )
        dropped.append(iid)
    return dropped


def _enqueue_batch(
    seed: dict[str, Any], group: list[dict[str, Any]], run_log: Path
) -> dict[str, Any]:
    """把一批（种子 + 组包菜单）入队 devops FIFO，返回计划记账条目。

    批次正文走 :func:`workflow.backlog_batch_text`（种子必做 + 菜单），元数据
    ``item["backlog"]`` 携种子/组包全文——worker 侧结算（:func:`settle_backlog_batch`）
    与 dry 演练（:func:`_dry_settle_batch`）都从它还原账务上下文。
    """
    from . import queues as _queues

    seed_id, _group_ids, batch_ids, label = _batch_faces(seed, group)
    effort = str(seed.get("effort") or "") or None
    eff_s = f"（effort={effort}）" if effort else ""
    _log(run_log, f"[→] 计划批次 {label}: {str(seed.get('summary', ''))[:50]}{eff_s}")
    item = _queues.enqueue(
        "devops",
        kind=BACKLOG_BATCH_KIND,
        text=backlog_batch_text(seed, group, max_batch=BATCH_MAX),
        slug=seed_id,
        session="latest",
        effort=str(effort or ""),
        extra={
            "backlog": {
                "seed": seed,
                "group": group,
                "ids": batch_ids,
                "seed_id": seed_id,
            }
        },
    )
    # 去重（方向①）：本批覆盖的 id 若还有**待跑**的独立条目，就地销账——否则同一 id
    # 会被执行两次（实测 backlog 20261016-224211）。
    dropped = supersede_batch_entries(set(batch_ids), keep=str(item.get("id", "")))
    if dropped:
        _log(
            run_log,
            f"[i] 去重：{len(dropped)} 条重复条目就地销账（不派跳）：{'、'.join(dropped)}",
        )
    _log(run_log, f"[i] 批次 {label} 已入队 devops FIFO（item={item.get('id', '')}）")
    return {
        "id": seed_id,
        "status": "queued",
        "queue_item": str(item.get("id", "")),
        "batch_size": len(batch_ids),
        "batch_seed": seed_id,
        "ids": batch_ids,
        "deduped": dropped,
    }


def replan_from_backlog(*, dry: bool = False) -> int:
    """队列 worker 的**续计划**：批次收尾后把仍 pending 的 backlog 条目再组包入队。

    这是「组包项留 pending 下轮再取」语义的落点——不续计划，被卡批次里未选中的组包项
    要等下一次 approve 才动（旧 drain 循环的「一项卡住不阻塞后续」会退化成「等唤醒」）。
    取**计划器锁**短暂序列化（防与在跑的计划器并发双取同一 pending 种子——take_batch
    的「读→标 in_progress→写」不是原子的，双取会把同一批入队两次）；锁忙即返回 0
    （在跑的计划器会把 pending 一并计划）。计划记账落 ``devops-runs``（计划日志）。

    Returns:
        本次续计划入队的批数。
    """
    run_log = new_run_log()
    run_log.parent.mkdir(parents=True, exist_ok=True)
    if acquire_lock(orchestration_rel(run_log)) is None:
        return 0  # 计划器在跑：pending 由它一并计划，不抢
    planned = 0
    try:
        _log(run_log, f"[▶] 续计划（worker 侧，pid={os.getpid()}，dry={dry}）")
        while True:
            # 排除在飞批次已覆盖的 id（双记账根治，backlog 20261016-224211）
            seed, group = backlog_take_batch(BATCH_MAX, exclude=inflight_batch_ids())
            if seed is None:
                break
            if dry:
                _dry_settle_batch(seed, group, run_log)
                planned += 1
                continue
            _enqueue_batch(seed, group, run_log)
            planned += 1
            break  # 一次一批：链由每个批次收尾后的下一次续计划接上（FIFO 依次处理）
    except Exception as exc:  # noqa: BLE001 - 续计划失败不炸在跑的批次
        _log_exception(run_log, exc, "续计划异常（worker 继续）")
    finally:
        _log(run_log, f"[√] 续计划结束：入队 {planned} 批")
        release_lock(run_log)
    return planned


def drain_backlog(run_log: Path, *, dry: bool = False) -> dict[str, Any]:
    """backlog **批量计划器**（backlog-fifo-20261014）：组包成批次 → 入队 devops FIFO。

    执行只留队列 worker 一份实现（``orch _queue-worker devops``）；本函数保留原 drain 的
    全部**计划侧**语义，**不再自持 do_dispatch 循环**：

    - 预设激活闸 / 计划器互斥锁（``state/devops.lock``，防双计划器）/ 模型映射自愈；
    - 启动扫尾：孤儿 in_progress 回收 + worktree 抢救/异名重建——**devops 队列 worker
      在跑时不回收**（此刻的 in_progress 是它的在途，不是孤儿；worker 启动时会做
      条目级的 ``workq.reset_orphans``）；
    - 循环 ``backlog_take_batch`` 组包（种子必做 + 同提请者菜单 ≤ ``BATCH_MAX``）→ 每条
      批次入队 ``state/queues/devops.json``（``kind=backlog-batch``，携种子/组包元数据与
      批次正文）→ 计划毕 ``queueworker.wake_member("devops")`` 拉起 FIFO 执行；
    - 带宽延期条目由 ``backlog_take_batch`` 自然跳过，计划器**不睡重试**——延期落在队列
      条目侧的 defer（worker 睡到到期自动重取），语义等价且不占计划器；
    - 记账：``devops-runs/<ts>.log``（计划日志）与 ``<ts>.done``（``kind="plan"``）；执行账
      在 ``queues-runs/devops/`` 的 worker 日志与 .done；完成唤醒走统一事件流的
      ``QUEUE-EMPTY``（worker 退出时发），watch 对 ``kind="plan"`` 的 .done 不报
      QUEUE-COMPLETE（记账不是消费事实）。

    Args:
        run_log: 本次计划日志路径；同名 ``.done`` 为计划摘要。
        dry: True 时不真入队，模拟逐 id 销账（唤醒链演练 / 测试，语义同旧 drain）。

    Returns:
        ``.done`` 摘要字典：``{kind:"plan", started, completed, count, elapsed_s, items,
        run_log}``；协作停止附 ``stopped="cooperative"``；预设未激活时附
        ``skipped="preset_unavailable"``。
    """
    done_path = run_log.with_suffix(".done")
    # 预设的 main 席表为空（没配席：跟踪覆盖在位以覆盖为准，否则看出厂表）即拒绝开工
    # （用户 2026-10-13 回填裁决，激活闸与派发共读 presets.preset_gate）：逐单派发只会把
    # 整队列标成失败，让「改一行 toml」的真处置淹没在重复失败读数里。此门在拿锁之前。
    gate = presets.preset_gate(Registry.load(), "main")
    if gate:
        reason = gate
        _log(
            run_log,
            f"[!] {reason}——计划器拒绝开工（改 data/用户的inbox/pysci_settings.toml [model] preset 后重跑）",
        )
        print(f"[!] {reason}")
        print(
            "    处置：改 data/用户的inbox/pysci_settings.toml 的 [model] preset（合法串见该文件注释）。"
        )
        return {
            "skipped": "preset_unavailable",
            "error": reason,
            "run_log": orchestration_rel(run_log),
        }
    lock = acquire_lock(orchestration_rel(run_log))
    if lock is None:
        _log(run_log, "[i] 另一计划器持锁运行中，本进程退出（不重复计划）。")
        return {"skipped": "lock_busy", "run_log": orchestration_rel(run_log)}

    t0 = time.time()
    summary: dict[str, Any] = {
        "kind": "plan",
        "started": _now(),
        "run_log": orchestration_rel(run_log),
        "dry": dry,
        "items": [],
    }
    _log(run_log, f"[▶] backlog 计划器启动（pid={os.getpid()}, dry={dry}）")
    # 模型映射自愈（用户裁决 2026-10-10）：--list-models 刷新档位映射的**运行态面**
    # （重配 BYOK 后下次 drain 即恢复正确 UUID）；跟踪 registry.json 不被写——写它会把
    # main 工作区每 drain 改脏并堵死碰 registry 的单（backlog 20261012-080410-devops）。
    # 任何失败仅告警不阻塞（宁旧勿空）。
    try:
        res = refresh_models()
        if res["changed"]:
            detail = "; ".join(
                f"{t}: {c['old'] or '(空)'} → {_channels.chain_text(c['new'])}"
                for t, c in res["changed"].items()
            )
            summary["models_refreshed"] = res["changed"]
            _log(run_log, f"[i] 模型映射已刷新（运行态面）：{detail}")
        if res.get("shadowed"):
            detail = "; ".join(
                f"{t}: 跟踪={_channels.chain_text(s['tracked'])} "
                f"遮蔽解析值={_channels.chain_text(s['resolved'])}"
                for t, s in sorted(res["shadowed"].items())
            )
            _log(
                run_log,
                f"[i] 模型映射跟踪面遮蔽解析值（人不写则不换）：{detail}",
            )
        if res.get("conflicts"):
            _log(
                run_log,
                "[!] 档位链与名→档规则冲突（串档，须人裁决）："
                + "; ".join(res["conflicts"]),
            )
        if res["missing"]:
            _log(
                run_log,
                f"[i] 模型映射未命中（保持现值）：{', '.join(res['missing'])}"
                "——BYOK/Mimo 改名则更新 registry.model_chains",
            )
    except Exception as exc:
        _log_exception(run_log, exc, "模型映射刷新失败（忽略，沿用 registry 现值）")
    # 启动扫尾（**devops 队列 worker 在跑时跳过**）：此刻的 in_progress 是它的在途，
    # 不是孤儿——worker 启动时已做条目级 workq.reset_orphans；在这里回收会把正在执行的
    # 批次复位 pending、被重复入队。无 worker 在场才谈「孤儿」。
    orphans: list[str] = []
    if _devops_worker_live():
        _log(
            run_log,
            "[i] devops 队列 worker 在跑 → 跳过启动扫尾（in_progress 是它的在途，非孤儿）",
        )
    else:
        orphans = backlog_reclaim_orphans()
    if orphans:
        summary["reclaimed"] = orphans
        _log(run_log, f"[i] 回收孤儿 in_progress 条目：{', '.join(orphans)}")
        if dry:
            _log(
                run_log,
                "[i] dry 模式：跳过孤儿会话进程终止与 worktree 清理（非破坏性演练）",
            )
        else:
            # 第四面（进程）先动手：旧会话还活着时删它的 worktree，只会让它对着已删目录
            # 继续写（Windows 上句柄不放，remove 也未必真清干净），随后重派就是双写。
            hot = reclaim_orphan_sessions(orphans, run_log)
            salvaged: dict[str, Any] = {}
            cleaned = cleanup_orphan_worktrees(
                orphans, run_log, salvage_sink=salvaged, quarantined=hot
            )
            if cleaned:
                summary["worktrees_cleaned"] = cleaned
            for name, record in salvaged.items():
                # 回收已把这些条目复位 pending，注记即「原地续完」；本轮循环就会再取到它
                workflow.backlog_attach_salvage(name, record)
            if salvaged:
                summary["worktrees_salvaged"] = sorted(salvaged)
            # 抢救过的面**不进「待升代」集合**——但只有**登记在册**的那部分才算数：
            # 目录还在却没登记（remove --force 的残骸）就不是「你的面」，同名重建=两个
            # 写者进同一房间（backlog 20261012-184000-leader 的实证路径）。残骸照样升代，
            # 只是进度不再被销毁：抢救记录里的分支/提交会被续作任务书指回。
            stuck = sorted(
                set(hot)
                | {
                    n
                    for n in orphans
                    if worktree_dir_present(n)
                    and not (salvaged.get(n) or {}).get("registered")
                }
            )
            if stuck:
                renamed = bump_worktree_generation(stuck)
                summary["worktrees_regenerated"] = renamed
                for oid, name in renamed.items():
                    _log(
                        run_log,
                        f"[!] 旧隔离面无法确认干净 → 异名重建：{oid} → {name}"
                        "（新一代隔离面，禁止复用旧名）",
                    )
    # §3.3 写侧接线：登记活跃会话 + 守护线程心跳（长计划不被判死）；死会话条目先清。
    _bp.prune_stale(now=time.time())
    sess = _bp.LiveSession(
        "devops", sid=run_log.stem, tier="devops", purpose="planner"
    ).start()
    try:
        planned = 0
        while True:
            if _consume_stop_request():
                summary["stopped"] = "cooperative"
                _log(run_log, "[s] 收到协作停止信号，干净退出（批次间隙，无在途消费）")
                break
            seed, group = backlog_take_batch(BATCH_MAX, exclude=inflight_batch_ids())
            if seed is None:
                # 空 or 仅剩带宽延期条目：计划器不睡——延期由队列条目侧 defer 承接
                # （worker 睡到到期自动重取），旧 drain 的睡觉分支在此退役。
                break
            if dry:
                summary["items"].extend(_dry_settle_batch(seed, group, run_log))
                continue
            summary["items"].append(_enqueue_batch(seed, group, run_log))
            planned += 1
            # **一次一批**（对齐旧 drain 的「一步一批次」节奏）：下一批由 worker 收尾后的
            # 续计划链上（replan_from_backlog）——未取项保持 pending，不被提前拉成在途。
            break
        if planned and not dry:
            from . import queueworker as _qw

            _, msg = _qw.wake_member("devops")
            summary["worker"] = msg
            _log(run_log, f"[→] 已入队 {planned} 批并唤醒 devops FIFO worker：{msg}")
    except Exception as exc:  # 记录后仍写 .done（后台进程无人值守，不留黑洞）
        summary["error"] = _log_exception(run_log, exc, "backlog 计划器异常中断")
    finally:
        summary["count"] = len(summary["items"])
        # blocked 成因档位分列计数（backlog 20261013-191213-devops）：让「待裁决」与
        # 「待排期」在 .done 与 orch status 上分开读——两类最优处置相反但都长成 blocked。
        # 只数**标注了** reason 的 items；未标注是合法缺省，不入任何档。
        by_reason: dict[str, int] = {}
        for it in summary["items"]:
            r = str(it.get("reason") or "")
            if r:
                by_reason[r] = by_reason.get(r, 0) + 1
        if by_reason:
            summary["by_reason"] = by_reason
        summary["completed"] = _now()
        summary["elapsed_s"] = round(time.time() - t0, 1)
        release_lock(run_log)
        try:
            done_path.parent.mkdir(parents=True, exist_ok=True)
            done_path.write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        except OSError as exc:
            _log(run_log, f"[!] .done 写入失败：{exc}")
        # 计划器**不发** QUEUE-EMPTY / CHANNEL-STOP：那是**消费**事实，归执行侧——
        # worker 退出（队列排空/全局停止）时发（queueworker.run_worker 的 finally）。
        sess.stop()  # 退场注销活跃会话（读数面不留幽灵）
    _log(
        run_log,
        f"[√] 计划器结束：计划 {summary['count']} 条，耗时 {summary['elapsed_s']}s "
        f"→ {orchestration_rel(done_path)}",
    )
    return summary


def cmd_drain_devops(args) -> int:
    """内部命令 ``_drain-devops``：跑 backlog 计划器（组包→入队→唤醒 FIFO），落计划日志。"""
    run_log = Path(args.run_log) if getattr(args, "run_log", None) else new_run_log()
    if not run_log.is_absolute():
        run_log = PROJECT_ROOT / run_log
    run_log.parent.mkdir(parents=True, exist_ok=True)
    summary = drain_backlog(run_log, dry=bool(getattr(args, "dry", False)))
    if summary.get("skipped") == "lock_busy":
        return 0
    return 0


# ---------------------------------------------------------------------------
# 唤醒（approve 调用）
# ---------------------------------------------------------------------------
def _drain_argv(run_log: Path, *, dry: bool) -> list[str]:
    """构造运行内部命令的 argv（``sys.executable -m`` → 从任意 cwd 可用）。"""
    argv = [
        sys.executable,
        "-m",
        "pysci.skills.orchestration.tools.orch",
        "_drain-devops",
        "--run-log",
        str(run_log),
    ]
    if dry:
        argv.append("--dry")
    return argv


#: 测试进程的判据：pytest 跑每个用例时注入 ``PYTEST_CURRENT_TEST``、收集期注入
#: ``PYTEST_VERSION``，两者都被子进程继承——用例里真起 CLI 子命令的派生同样挡得住。
TEST_PROCESS_ENV: tuple[str, ...] = ("PYTEST_CURRENT_TEST", "PYTEST_VERSION")


def in_test_process() -> bool:
    """当前进程是不是 pytest 测试进程（含测试派生的子进程）。"""
    return any(os.environ.get(name) for name in TEST_PROCESS_ENV)


def _spawn_detached(argv: list[str], *, env: dict[str, str], stdout: Any) -> None:
    """以分离后台进程启动 argv（Windows: DETACHED_PROCESS；POSIX: 新会话）。

    **测试进程在这里停住**。三入口（status/approve/dispatch）内嵌的拉起是「代码保证」的
    生命周期面，逐用例 patch :data:`inspectd._spawn_detached` 挡不住没挂 ``iso_state`` 的
    漏网用例——实测两个用例真起了 ``_inspect-worker``，run_log 落在当时所在的 worktree
    树里，紧跟着被 ``worktree merge`` 的 cp 面拖进 main（backlog 20261012-191512-devops）。
    判据取「我是测试进程」而非「我在哪棵树」：在主树里跑套件同样不该点着巡检队列。
    确需真 Popen 的用例自行清掉 :data:`TEST_PROCESS_ENV`（`test_drain_wake.py` 有反例，
    防这道闸被焊死后静默废掉生产面而 mock 全绿看不见）。注意闸在**派生面**不在判定面，
    故测试进程里 ``wake_*`` 返回的 ``spawned=True`` 只是空话——要断言派生 argv 的用例
    一律 patch 本函数（现有用例都这么做）。
    """
    if in_test_process():
        return
    kwargs: dict[str, Any] = {
        "stdin": subprocess.DEVNULL,
        "stdout": stdout,
        "stderr": subprocess.STDOUT,
        "cwd": str(PROJECT_ROOT),
        "env": env,
        "close_fds": True,
    }
    if sys.platform == "win32":
        kwargs["creationflags"] = (
            subprocess.DETACHED_PROCESS
            | subprocess.CREATE_NEW_PROCESS_GROUP
            | subprocess.CREATE_NO_WINDOW
        )
    else:
        kwargs["start_new_session"] = True
    # 并入**单一隐藏口径**（backlog adhoc-storm-fix 判据①）：本处原有旗标保留（DETACHED_PROCESS
    # 等），`popen_hidden` 再统一叠加 CREATE_NO_WINDOW——同一能力不留两份口径。
    procs.popen_hidden(argv, **kwargs)


def wake_devops(*, force_dry: bool | None = None) -> tuple[bool, str]:
    """approve 后尝试唤醒后台 backlog **计划器**（组包 → 入队 devops FIFO → 唤醒 worker）。

    Args:
        force_dry: 覆盖干跑判定（None → 读 :data:`DRY_ENV` 环境变量）。

    Returns:
        ``(spawned, message)``。``spawned=False`` 表示计划器锁忙未唤醒（在跑那位会把新
        条目一并计划；条目在**持久 backlog** 里，不会丢）；``True`` 表示已派生分离进程
        （message 含日志路径）。两种情况的 message 都附一行「等待已自动化」
        （:func:`watch.autowatch_lines`）：挂载自 2026-10-13 起由 Stop 钩子每回合自动完成
        （README §4.5 v2），不再催组长手动挂。
    """
    st = lock_status()
    if st["state"] == "running":
        return (
            False,
            "devops 计划器正在跑，本项将并入其本轮组包/交由 FIFO 队列消化（不重复唤醒）。"
            + _mount_block(),
        )
    DEVOPS_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    run_log = new_run_log()
    dry = bool(os.environ.get(DRY_ENV)) if force_dry is None else bool(force_dry)
    argv = _drain_argv(run_log, dry=dry)
    env = {**os.environ, "PYTHONUTF8": "1"}
    with open(run_log, "ab") as logf:
        _spawn_detached(argv, env=env, stdout=logf)
    return True, (
        f"devops 计划器已后台唤醒（日志 {orchestration_rel(run_log)}）；"
        "批次将入队 devops FIFO 由队列 worker 消化，结果见 orch status / queues-runs。"
        + _mount_block()
    )


def _mount_block() -> str:
    """唤醒 message 尾部追加的「等待已自动化」行（换行分隔，首行为空 = 与正文自然断行）。

    不再给挂载命令——挂载由 Stop 钩子自动完成（README §4.5 v2）；手动兜底见技能真本。
    """
    from .watch import autowatch_lines  # 延迟导入：watch↔drain 环

    return "\n" + "\n".join(autowatch_lines(indent="    "))
