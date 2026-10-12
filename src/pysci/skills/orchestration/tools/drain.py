"""devops backlog 自动消化：锁 / 后台唤醒 / drain 循环（README §6）。

``orch approve`` 采纳建议入 backlog 后，orch 自动以**分离后台进程**唤醒 devops，
FIFO 串行消化整个 backlog——组长零阻塞、零轮询（对齐 README §4.5 等待模型）。

三块机制：

- **锁**（``state/devops.lock``，JSON: pid/started/run_log）：同刻至多一个 drain
  worker。原子 ``O_CREAT|O_EXCL`` 抢占；持有者进程已死或锁龄超 :data:`LOCK_STALE_S`
  判 stale 并接管（fail-open，防 worker 崩溃后永久死锁）。**收尾事实优先于锁的自称**：
  释放时 unlink 删不掉（Windows 共享句柄）则改写为 ``released`` **墓碑**并在 run 日志留
  ``[!]``；连墓碑都写不进时，「该 run 已有 ``.done``」仍单独把残骸判为非活
  （:func:`_run_concluded`）——两条判活否决合起来，队列不会被一把删不掉的锁阻塞 6 h
  （:func:`release_lock`，backlog 20261011-225933-devops + 20261011-231237-devops）。
- **唤醒**（:func:`wake_devops`）：approve 成功后调用；锁忙则不重复 spawn（本项由
  当前 drain 循环接手），空闲则 Popen 分离进程运行内部命令 ``_drain-devops``。返回的
  message 固定附 watch 事件等待器挂载指令（:mod:`watch`，README §4.5）——工作交棒的
  当刻由代码把「怎么等」交给组长。
- **drain 循环**（:func:`drain_backlog`）：获锁后先回收孤儿 in_progress 条目（已死
  worker 遗留，重置 pending），并**四面**清干净它留下的写入者
  （:func:`reclaim_orphan_sessions` 终止仍活着的派生会话进程 →
  :func:`cleanup_orphan_worktrees` 摘 worktree/branch → 任一面确认不了干净就
  :func:`workflow.bump_worktree_generation` 换 ``<id>-r<gen>`` 新一代隔离面，绝不复用
  可能还有人写的旧名；worktree 名默认 = 条目 id，``--force`` 摘除 + ``prune``
  兜底，只清孤儿名不误删活跃/手动 worktree）；``backlog_take_batch`` 取队首为种子 +
  同提请者组包（≤ :data:`BATCH_MAX` 项）→ 批量任务书（种子必做 + 组包菜单，devops 自主
  选取合并实施）→ 一次 ``do_dispatch("devops")`` → 解析交付 ``backlog id=<ids>`` 逐 id
  销账（commit 字段三级定位：正文提取 → 按 ``wt-<id>`` 分支回溯 merge commit → 注记
  ``unresolved``，不留空串）；未选组包项留 pending 下轮；失败（blocked/run_failed）仅**种子** needs_leader、
  组包不连坐（一项卡住不阻塞全队列）；额度类失败（quota_exhausted）例外——视为系统性
  故障**全局停止**（当前条目复位 pending、剩余保持 pending、.done 注明）；run 日志
  逐行落 ``state/devops-runs/<ts>.log``，全部完成写 ``<ts>.done``（JSON 摘要）。run 名是
  身份而非标签：同秒撞名时 :func:`new_run_log` 顺延为 ``<ts>_002``…并当场占位。
  ``.done`` 是干净完成信号：stale 锁若缺对应 ``.done`` 即 worker 被中途杀死（崩溃），
  :func:`lock_status` 据此置 ``crashed``，orch status surfacing 供用户判断 devops 状态。

分离进程无 kill 之外的规范停止手段，故补**协作式停止**：``orch drain-stop`` 写
``state/devops.cancel``（JSON: pid/requested_at，指向当前锁持有者）；drain 在每项
间隙检查——信号指向本进程则干净退出（释放锁、写含 ``stopped=cooperative`` 的部分
.done、cancel 自删）；指向他方的陈旧信号顺手清除，绝不误停新 worker。

内部命令 ``_drain-devops`` 下划线前缀，argparse help 标注为 ``[内部]``（勿手调）。
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from pysci.paths import ORCH_STATE_ROOT, PROJECT_ROOT

from . import procs, workflow
from .registry import Registry, note_hop_pid, orchestration_rel, refresh_models
from .workflow import (
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

#: 锁龄上限（秒）——超过即判 stale 接管（worker 崩溃兜底）。6h 覆盖最长 devops 任务。
LOCK_STALE_S: int = 6 * 3600

#: 干跑开关环境变量：置真值时 wake 派生的 drain 走 --dry（不真派发，演练/验证唤醒链用）。
DRY_ENV: str = "PYSCI_ORCH_DRAIN_DRY"

#: 协作式停止信号（JSON: pid/requested_at）；orch drain-stop 写，drain 消费自删。
CANCEL_PATH: Path = ORCH_STATE_ROOT / "devops.cancel"

#: 锁记录的「已释放」标记键：unlink 删不掉锁时把锁改写为该标记的**墓碑**——
#: 残骸不再是互斥依据（:func:`_lock_live` 认它为非活锁）。backlog 20261011-225933-devops。
RELEASED_KEY: str = "released"

#: acquire 抢锁重试轮数（竞态窗口 + 残骸清除）。
ACQUIRE_RETRIES: int = 3

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

    为什么要把它做成参数（REFACTOR §0.9「共享逻辑做库，不复制实现」）：本模块的锁语义
    是被三次事故打磨出来的——墓碑（删不掉的锁不得挡路）、``.done`` 收尾证据（谎报在跑的
    残骸不得被当活锁）、run 名同秒顺延（撞名会让活锁被上一任的摘要判成残骸）。bencher
    的 FIFO worker（:mod:`benchd`，REFACTOR §8）需要的正是同一套判活规则；复制一份等于
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
            proc = subprocess.run(
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


def _lock_live(data: dict[str, Any] | None, paths: WorkerPaths | None = None) -> bool:
    """锁是否有效：未宣告收尾、持有者进程存活且锁龄未超上限。

    两条**收尾宣告**都优先于 pid 判活（判活本身可能说谎：pid 回收、tasklist 超时保守
    判活）：``released`` 墓碑，以及该 run 已有 ``.done`` 摘要（:func:`_run_concluded`）。
    """
    if not data or _lock_released(data) or _run_concluded(data):
        return False
    p = paths or _devops_paths()
    age = _lock_age_s(data)
    if age > p.stale_s:
        return False
    return _pid_alive(int(data.get("pid", 0) or 0))


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
    state = "running" if (alive and 0 <= age <= p.stale_s) else "stale"
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
# 协作式停止（orch drain-stop ↔ drain 循环间隙消费）
# ---------------------------------------------------------------------------
def request_stop(paths: WorkerPaths | None = None) -> tuple[bool, str]:
    """写协作停止信号，目标为当前持锁 worker。供 ``orch drain-stop`` 调用。

    Returns:
        ``(ok, message)``。无运行中 worker 时**不写信号**（返回 False——空闲队列
        不需要停，残留信号还会危及下一个 worker）。
    """
    p = paths or _devops_paths()
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


def _consume_stop_request(paths: WorkerPaths | None = None) -> bool:
    """drain 循环检查：停止信号是否指向本进程；无论指向与否信号文件均被删除。

    陈旧信号（pid≠本进程——被 kill 的 worker 未及消费而留下的）顺手清除并返回
    False，绝不误停接棒的下一个 worker。
    """
    cancel = (paths or _devops_paths()).cancel
    if not cancel.exists():
        return False
    try:
        pid = int(json.loads(cancel.read_text(encoding="utf-8")).get("pid", 0))
    except (json.JSONDecodeError, OSError, ValueError, TypeError):
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


#: run 摘要 items 里「这一条目这次尝试失败」的状态读数。``quota_exhausted`` **不在**
#: 此列：那是渠道级系统性故障、条目无罪复位，计入会把「额度没了」误读成「任务书写坏了」。
FAILED_ITEM_STATUSES: tuple[str, ...] = (
    "needs_leader",
    "blocked",
    "run_failed",
    "parse_error",
    "check_fail",
)


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
    """在项目根运行 git，返回 (rc, 合并输出)。超时/异常一律 fail-open（返回非零、不抛）。"""
    try:
        proc = subprocess.run(
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


def cleanup_orphan_worktrees(
    names: list[str], run_log: Path | None = None
) -> list[str]:
    """清理死 worker 残留的 worktree/branch（孤儿回收的顺带动作）。

    每个 name（= 孤儿条目 id = worktree 名，见 workflow.backlog_take_first 记录约定）：
    ``git worktree remove --force`` 摘工作树 + ``git branch -D wt-<name>`` 删分支；末尾
    ``git worktree prune`` 兜底清悬挂管理项（目录已删但元数据残留）。

    **只清传入的孤儿名**——活跃/手动 worktree（别名）不受影响，故不会误删并发的手动
    作业（对齐任务书要求）。全程 fail-open（git 缺失/超时/该 worktree 不存在均忽略）。

    Returns:
        实际清理到（worktree 或 branch 至少其一删除成功）的 name 列表。
    """
    cleaned: list[str] = []
    for name in names:
        if not name:
            continue
        wt_path = workflow.WORKTREES_ROOT / name  # 与 pysci-dev worktree add 同源
        rc_wt, _ = _run_git("worktree", "remove", "--force", str(wt_path))
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
    """
    if not name:
        return False
    return (workflow.WORKTREES_ROOT / name).exists()


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
def _drain_batch(
    seed: dict[str, Any], group: list[dict[str, Any]], run_log: Path, *, dry: bool
) -> list[dict[str, Any]]:
    """消化一批（种子 + 同提请者组包菜单）：一次派发 devops → 逐交付 id 销账。

    种子条目的 ``effort`` 标注透传给本跳（组包只收同档位条目，故一批一跳一标，
    见 :func:`workflow.backlog_take_batch`）。

    - **result**（code 0）：解析交付 ``backlog id=<ids>``，与本批取交集逐 id 销账 done
      （种子必含——必做项，防御性兜底）；组包中未选的项保持 pending 留待下轮；每个 done
      项记 ``batch_size``/``batch_seed``（供 orch stats 批量分布）。
    - **blocked/run_failed/parse_error/验收FAIL**：仅**种子** → needs_leader，组包项保持
      pending（**不连坐**——它们未被承诺、未 in_progress）。
    - **quota_exhausted**：种子复位 pending、组包项保持 pending，返回 quota 状态由调用方
      触发全局停止（系统性故障，不逐项 needs_leader）。

    Returns:
        per-id 结果字典列表（写入 .done 的 items）：种子/每个完成组包项各一条
        ``{id, status, commit?, kind?, batch_size?, batch_seed?, elapsed_s}``。
    """
    seed_id = str(seed.get("id", ""))
    group_ids = [str(g.get("id", "")) for g in group if g.get("id")]
    batch_ids = [seed_id, *group_ids]
    label = seed_id if not group_ids else f"{seed_id}(+{len(group_ids)})"
    effort = str(seed.get("effort") or "") or None
    eff_s = f"（effort={effort}）" if effort else ""
    _log(
        run_log,
        f"[→] 消化批次 {label}: {str(seed.get('summary', ''))[:50]}{eff_s}",
    )
    t0 = time.time()

    if dry:
        # 干跑：模拟 devops 吃下整批（种子+组包），逐 id 销账
        for bid in batch_ids:
            backlog_complete(
                bid,
                note="drain --dry 干跑（未实派 devops）",
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

    from .dispatch import (
        do_dispatch,  # 延迟导入：避免 drain↔dispatch 环（dispatch 不依赖 drain）
    )

    outcome = do_dispatch(
        "devops",
        text=backlog_batch_text(seed, group, max_batch=BATCH_MAX),
        slug=seed_id,
        session="latest",
        effort=effort,
        quiet=True,
    )
    elapsed = round(time.time() - t0, 1)
    if outcome.warning:
        # 交付是被宽松兜底面救回的（解析器版本落后于协议）：不改变判读，但必须进 run
        # 日志——组长侧看得到「这一跳的交付格式是兜底读出来的」，据此决定要不要重跑
        _log(run_log, f"[!] {outcome.warning}（{seed_id}）")

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

    # 额度类失败 = 系统性故障：种子无罪复位 pending，组包保持 pending；调用方据此全局停止
    if outcome.kind == "quota_exhausted":
        err = (outcome.error or outcome.body[:160] or "").strip().replace("\n", " ")
        backlog_requeue(seed_id, note="额度类全局停止，复位待渠道恢复")
        _log(
            run_log,
            f"[Q] 批次 {label} 额度耗尽 → 种子复位回队，触发全局停止：{err[:120]}",
        )
        return [
            {
                "id": seed_id,
                "status": "quota_exhausted",
                "error": err[:200],
                "elapsed_s": elapsed,
            }
        ]

    # 失败（blocked / run_failed / parse_error / 验收 FAIL）→ 仅种子 needs_leader；
    # 组包项保持 pending（不连坐：未被承诺、未 in_progress，留待下轮）
    err = (outcome.error or outcome.body[:160] or "").strip().replace("\n", " ")
    backlog_needs_leader(
        seed_id, note=f"drain 跳过（交付={outcome.kind}）：{err[:160]}"
    )
    _log(
        run_log,
        f"[!] 批次 {label} 失败 → 种子 needs_leader（{outcome.kind}），"
        f"组包 {len(group_ids)} 项保持 pending 不连坐：{err[:120]}",
    )
    return [
        {
            "id": seed_id,
            "status": "needs_leader",
            "kind": outcome.kind,
            "error": err[:200],
            "elapsed_s": elapsed,
        }
    ]


def drain_backlog(run_log: Path, *, dry: bool = False) -> dict[str, Any]:
    """FIFO 串行消化整个 backlog（持锁运行；异常/结束均释放锁并写 .done）。

    Args:
        run_log: 本次 run 日志路径；同名 ``.done`` 为完成摘要。
        dry: True 时不真派发，仅模拟销账（唤醒链演练 / 测试）。

    Returns:
        .done 摘要字典：``{started, completed, count, elapsed_s, items, run_log}``；
        协作停止时附 ``stopped="cooperative"``；额度类全局停止时附
        ``stopped="quota_exhausted"`` 与 ``error``（系统性故障说明）。
    """
    done_path = run_log.with_suffix(".done")
    lock = acquire_lock(orchestration_rel(run_log))
    if lock is None:
        _log(run_log, "[i] 另一 drain worker 持锁运行中，本进程退出（不重复消费）。")
        return {"skipped": "lock_busy", "run_log": orchestration_rel(run_log)}

    t0 = time.time()
    summary: dict[str, Any] = {
        "started": _now(),
        "run_log": orchestration_rel(run_log),
        "dry": dry,
        "items": [],
    }
    _log(run_log, f"[▶] drain 启动（pid={os.getpid()}, dry={dry}）")
    # 模型映射自愈（用户裁决 2026-10-10）：--list-models 刷新档位映射的**运行态面**
    # （重配 BYOK 后下次 drain 即恢复正确 UUID）；跟踪 registry.json 不被写——写它会把
    # main 工作区每 drain 改脏并堵死碰 registry 的单（backlog 20261012-080410-devops）。
    # 任何失败仅告警不阻塞（宁旧勿空）。
    try:
        res = refresh_models()
        if res["changed"]:
            detail = "; ".join(
                f"{t}: {c['old'] or '(空)'} → {c['new']}"
                for t, c in res["changed"].items()
            )
            summary["models_refreshed"] = res["changed"]
            _log(run_log, f"[i] 模型映射已刷新（运行态面）：{detail}")
        if res.get("shadowed"):
            detail = "; ".join(
                f"{t}: 跟踪={s['tracked'] or '(空)'} 遮蔽解析值={s['resolved'] or '(空)'}"
                for t, s in sorted(res["shadowed"].items())
            )
            _log(
                run_log,
                f"[i] 模型映射跟踪面遮蔽解析值（人不写则不换）：{detail}",
            )
        if res["missing"]:
            _log(
                run_log,
                f"[i] 模型映射未命中（保持现值）：{', '.join(res['missing'])}"
                "——BYOK 改名则更新 registry.model_patterns",
            )
    except Exception as exc:
        _log(
            run_log,
            f"[!] 模型映射刷新失败（忽略，沿用 registry 现值）：{type(exc).__name__}: {exc}",
        )
    # 刚获锁 → 此刻的 in_progress 必是已死 worker 的孤儿，回收为 pending
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
            cleaned = cleanup_orphan_worktrees(orphans, run_log)
            if cleaned:
                summary["worktrees_cleaned"] = cleaned
            stuck = sorted(set(hot) | {n for n in orphans if worktree_dir_present(n)})
            if stuck:
                renamed = bump_worktree_generation(stuck)
                summary["worktrees_regenerated"] = renamed
                for oid, name in renamed.items():
                    _log(
                        run_log,
                        f"[!] 旧隔离面无法确认干净 → 异名重建：{oid} → {name}"
                        "（新一代隔离面，禁止复用旧名）",
                    )
    try:
        while True:
            if _consume_stop_request():
                summary["stopped"] = "cooperative"
                _log(run_log, "[s] 收到协作停止信号，干净退出（条目间隙，无在途消费）")
                break
            seed, group = backlog_take_batch(BATCH_MAX)
            if seed is None:
                break
            results = _drain_batch(seed, group, run_log, dry=dry)
            summary["items"].extend(results)
            if any(r.get("status") == "quota_exhausted" for r in results):
                # 系统性故障 → 全局停止：种子已在 _drain_batch 内复位 pending，
                # 组包与剩余条目未承诺仍 pending；释放锁与写 .done 由 finally 统一处理
                summary["stopped"] = "quota_exhausted"
                summary["error"] = (
                    "系统性故障：模型渠道额度耗尽（quota_exhausted）——全局停止，"
                    "全部条目保持 pending 待渠道恢复"
                )
                _log(run_log, "[Q] 全局停止：额度耗尽，剩余条目保持 pending，释放锁")
                break
    except Exception as exc:  # 记录后仍写 .done（后台进程无人值守，不留黑洞）
        summary["error"] = f"{type(exc).__name__}: {exc}"
        _log(run_log, f"[!] drain 异常中断：{summary['error']}")
    finally:
        summary["count"] = len(summary["items"])
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
    _log(
        run_log,
        f"[√] drain 结束：处理 {summary['count']} 项，耗时 {summary['elapsed_s']}s "
        f"→ {orchestration_rel(done_path)}",
    )
    return summary


def cmd_drain_devops(args) -> int:
    """内部命令 ``_drain-devops``：持锁消化 backlog，落 run 日志 + .done。"""
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
    subprocess.Popen(argv, **kwargs)


def wake_devops(*, force_dry: bool | None = None) -> tuple[bool, str]:
    """approve 后尝试唤醒后台 drain worker。

    Args:
        force_dry: 覆盖干跑判定（None → 读 :data:`DRY_ENV` 环境变量）。

    Returns:
        ``(spawned, message)``。``spawned=False`` 表示锁忙未唤醒（message 说明本项由
        当前 drain 循环接手）；``True`` 表示已派生分离进程（message 含日志路径）。
        两种情况的 message 都**固定附带 watch 挂载指令行**（:func:`watch.mount_lines`）：
        「工作已交办、开始等待」这个时刻由代码把等待方式交给组长，不靠其记得查文档
        （backlog 20261010-orch-watch，README §4.5）。
    """
    st = lock_status()
    if st["state"] == "running":
        return (
            False,
            "devops 正在处理队列，本项将由当前 drain 循环接手（不重复唤醒）。"
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
        f"devops 已后台唤醒（日志 {orchestration_rel(run_log)}）；"
        "结果见 orch status / 该目录 .done 文件。" + _mount_block()
    )


def _mount_block() -> str:
    """唤醒 message 尾部追加的挂载指令块（换行分隔，首行为空 = 与正文自然断行）。"""
    from .watch import (
        mount_lines,  # 延迟导入：watch↔drain 环（watch 依赖 drain 的锁/摘要原语）
    )

    return "\n" + "\n".join(mount_lines(indent="    "))
