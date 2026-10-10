"""devops backlog 自动消化：锁 / 后台唤醒 / drain 循环（README §6）。

``orch approve`` 采纳建议入 backlog 后，orch 自动以**分离后台进程**唤醒 devops，
FIFO 串行消化整个 backlog——组长零阻塞、零轮询（对齐 README §4.5 等待模型）。

三块机制：

- **锁**（``state/devops.lock``，JSON: pid/started/run_log）：同刻至多一个 drain
  worker。原子 ``O_CREAT|O_EXCL`` 抢占；持有者进程已死或锁龄超 :data:`LOCK_STALE_S`
  判 stale 并接管（fail-open，防 worker 崩溃后永久死锁）。
- **唤醒**（:func:`wake_devops`）：approve 成功后调用；锁忙则不重复 spawn（本项由
  当前 drain 循环接手），空闲则 Popen 分离进程运行内部命令 ``_drain-devops``。返回的
  message 固定附 watch 事件等待器挂载指令（:mod:`watch`，README §4.5）——工作交棒的
  当刻由代码把「怎么等」交给组长。
- **drain 循环**（:func:`drain_backlog`）：获锁后先回收孤儿 in_progress 条目（已死
  worker 遗留，重置 pending），并顺带清理其残留 worktree/branch
  （:func:`cleanup_orphan_worktrees`，worktree 名 = 条目 id，``--force`` 摘除 + ``prune``
  兜底，只清孤儿名不误删活跃/手动 worktree）；``backlog_take_batch`` 取队首为种子 +
  同提请者组包（≤ :data:`BATCH_MAX` 项）→ 批量任务书（种子必做 + 组包菜单，devops 自主
  选取合并实施）→ 一次 ``do_dispatch("devops")`` → 解析交付 ``backlog id=<ids>`` 逐 id
  销账；未选组包项留 pending 下轮；失败（blocked/run_failed）仅**种子** needs_leader、
  组包不连坐（一项卡住不阻塞全队列）；额度类失败（quota_exhausted）例外——视为系统性
  故障**全局停止**（当前条目复位 pending、剩余保持 pending、.done 注明）；run 日志
  逐行落 ``state/devops-runs/<ts>.log``，全部完成写 ``<ts>.done``（JSON 摘要）。
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
from datetime import datetime
from pathlib import Path
from typing import Any

from pysci.paths import ORCH_STATE_ROOT, PROJECT_ROOT

from .registry import orchestration_rel, refresh_models
from .workflow import (
    backlog_batch_text,
    backlog_complete,
    backlog_needs_leader,
    backlog_reclaim_orphans,
    backlog_requeue,
    backlog_take_batch,
)

#: 单批消化上限（护栏）：种子 + 同提请者组包 ≤ 此数（backlog 20261010-batch-digest）。
BATCH_MAX: int = 4

#: drain 互斥锁（JSON: pid/started/run_log）；仅 orch 与 devops drain 读写。
LOCK_PATH: Path = ORCH_STATE_ROOT / "devops.lock"

#: run 日志与 .done 摘要目录。
DEVOPS_RUNS_DIR: Path = ORCH_STATE_ROOT / "devops-runs"

#: 锁龄上限（秒）——超过即判 stale 接管（worker 崩溃兜底）。6h 覆盖最长 devops 任务。
LOCK_STALE_S: int = 6 * 3600

#: 干跑开关环境变量：置真值时 wake 派生的 drain 走 --dry（不真派发，演练/验证唤醒链用）。
DRY_ENV: str = "PYSCI_ORCH_DRAIN_DRY"

#: 协作式停止信号（JSON: pid/requested_at）；orch drain-stop 写，drain 消费自删。
CANCEL_PATH: Path = ORCH_STATE_ROOT / "devops.cancel"

_COMMIT_RE = re.compile(r"commit[^0-9a-f]{0,16}\b([0-9a-f]{7,40})\b", re.IGNORECASE)

#: 交付正文的 ``backlog id=<id1>,<id2>,…`` 清单（批量消化逐 id 销账依据）。id 仅含
#: 字母数字/-/_，故捕获在遇中文/标点即止；分隔容忍 ``,``/``、``/空白。
_BACKLOG_IDS_RE = re.compile(
    r"backlog\s+id\s*[:=]\s*([0-9A-Za-z_\-]+(?:\s*[,、]\s*[0-9A-Za-z_\-]+)*)",
    re.IGNORECASE,
)


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


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


def read_lock() -> dict[str, Any] | None:
    """读取锁记录；不存在或损坏返回 None（损坏锁将由 acquire 覆盖）。"""
    if not LOCK_PATH.exists():
        return None
    try:
        data = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
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


def _lock_live(data: dict[str, Any] | None) -> bool:
    """锁是否有效：持有者进程存活且锁龄未超上限。"""
    if not data:
        return False
    age = _lock_age_s(data)
    if age > LOCK_STALE_S:
        return False
    return _pid_alive(int(data.get("pid", 0) or 0))


def acquire_lock(run_log: str) -> dict[str, Any] | None:
    """尝试获取 drain 锁。

    Args:
        run_log: 本次 run 的日志相对路径（写入锁记录供 status 展示）。

    Returns:
        成功获取时返回锁记录 ``{pid, started, run_log}``；已被存活 worker 持有时
        返回 ``None``（调用方应优雅退出，不重复消费）。stale 锁（进程死/超龄）被接管。
    """
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {"pid": os.getpid(), "started": _now(), "run_log": run_log}
    payload = json.dumps(record, ensure_ascii=False, indent=2) + "\n"
    for _ in range(3):
        if _lock_live(read_lock()):
            return None
        try:
            fd = os.open(str(LOCK_PATH), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            # 竞态：他方刚抢占。若其存活则退出；若仍是 stale 残留则清除后重试。
            if _lock_live(read_lock()):
                return None
            try:
                os.unlink(str(LOCK_PATH))
            except OSError:
                pass
            continue
        except OSError:
            return None
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(payload)
        return record
    return None


def release_lock() -> None:
    """释放锁——**仅当持有者是本进程**（防未获取到锁的进程误删他人锁）。"""
    data = read_lock()
    if data and int(data.get("pid", 0) or 0) == os.getpid():
        try:
            os.unlink(str(LOCK_PATH))
        except OSError:
            pass


def lock_status() -> dict[str, Any]:
    """status 命令用的锁态摘要：``{state: idle|running|stale, pid, started, ...}``。

    stale 时附 ``crashed`` 字段：该 run 无对应 ``.done`` = worker 被中途杀死（崩溃）；
    有 ``.done`` = 上一次干净完成但锁残留（release 未及/异常）。供 orch status surfacing。
    """
    data = read_lock()
    if not data:
        return {"state": "idle"}
    pid = int(data.get("pid", 0) or 0)
    age = _lock_age_s(data)
    alive = _pid_alive(pid)
    state = "running" if (alive and 0 <= age <= LOCK_STALE_S) else "stale"
    out: dict[str, Any] = {
        "state": state,
        "pid": pid,
        "started": data.get("started", ""),
        "run_log": data.get("run_log", ""),
        "age_s": round(age, 1) if age >= 0 else -1,
    }
    if state == "stale":
        out["crashed"] = _stale_run_crashed(data)
    return out


def _stale_run_crashed(data: dict[str, Any]) -> bool:
    """stale 锁对应的 run 是否崩溃（无 ``.done``）。

    ``.done`` 由 drain_backlog 的 finally 块在正常/异常收尾时写出——存在即代表 worker
    跑到了收尾（干净完成），缺失即代表进程被中途杀死（OS 关机 / kill -9）没机会收尾。
    run_log 无法定位时保守判为崩溃。
    """
    run_log_rel = str(data.get("run_log", ""))
    if not run_log_rel:
        return True
    done = (PROJECT_ROOT / run_log_rel).with_suffix(".done")
    return not done.exists()


# ---------------------------------------------------------------------------
# 协作式停止（orch drain-stop ↔ drain 循环间隙消费）
# ---------------------------------------------------------------------------
def request_stop() -> tuple[bool, str]:
    """写协作停止信号，目标为当前持锁 worker。供 ``orch drain-stop`` 调用。

    Returns:
        ``(ok, message)``。无运行中 worker 时**不写信号**（返回 False——空闲队列
        不需要停，残留信号还会危及下一个 worker）。
    """
    st = lock_status()
    if st["state"] != "running":
        return False, "无运行中的 drain worker（锁空闲），无需停止。"
    CANCEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    CANCEL_PATH.write_text(
        json.dumps(
            {"pid": st["pid"], "requested_at": _now()}, ensure_ascii=False, indent=2
        )
        + "\n",
        encoding="utf-8",
    )
    return True, (
        f"已请求协作停止（目标 pid {st['pid']}）：worker 将于下一条目间隙干净退出"
        f"（释放锁、写部分 .done），信号文件 {orchestration_rel(CANCEL_PATH)} 消费后自删。"
    )


def cancel_status() -> dict[str, Any] | None:
    """status 展示用：待消费的停止信号（无/损坏则 None）。"""
    if not CANCEL_PATH.exists():
        return None
    try:
        data = json.loads(CANCEL_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return data if isinstance(data, dict) else None


def _consume_stop_request() -> bool:
    """drain 循环检查：停止信号是否指向本进程；无论指向与否信号文件均被删除。

    陈旧信号（pid≠本进程——被 kill 的 worker 未及消费而留下的）顺手清除并返回
    False，绝不误停接棒的下一个 worker。
    """
    if not CANCEL_PATH.exists():
        return False
    try:
        pid = int(json.loads(CANCEL_PATH.read_text(encoding="utf-8")).get("pid", 0))
    except (json.JSONDecodeError, OSError, ValueError, TypeError):
        pid = 0
    try:
        CANCEL_PATH.unlink()
    except OSError:
        pass
    return pid == os.getpid()


# ---------------------------------------------------------------------------
# run 日志 / .done
# ---------------------------------------------------------------------------
def _log(run_log: Path, line: str) -> None:
    """向 run 日志追加一行（带时间戳）。日志写入失败不阻塞 drain（fail-open）。"""
    try:
        run_log.parent.mkdir(parents=True, exist_ok=True)
        with run_log.open("a", encoding="utf-8") as fh:
            fh.write(f"[{datetime.now().strftime('%H:%M:%S')}] {line}\n")
    except OSError:
        pass


def new_run_log() -> Path:
    """生成本次 run 的日志路径 ``devops-runs/<ts>.log``（不落盘）。"""
    return DEVOPS_RUNS_DIR / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"


def latest_done() -> dict[str, Any] | None:
    """最近一次 run 的 .done 摘要（无则 None）；附 ``name`` 字段。"""
    if not DEVOPS_RUNS_DIR.exists():
        return None
    dones = sorted(DEVOPS_RUNS_DIR.glob("*.done"))
    if not dones:
        return None
    try:
        data = json.loads(dones[-1].read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if isinstance(data, dict):
        data["name"] = dones[-1].stem
    return data if isinstance(data, dict) else None


def _extract_commit(body: str) -> str:
    """从交付正文尽力提取 commit hash（``commit ... <hex>``）；无则空串。"""
    m = _COMMIT_RE.search(body or "")
    return m.group(1) if m else ""


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
        wt_path = PROJECT_ROOT / ".qoder" / "worktrees" / name
        rc_wt, _ = _run_git("worktree", "remove", "--force", str(wt_path))
        rc_br, _ = _run_git("branch", "-D", f"wt-{name}")
        if rc_wt == 0 or rc_br == 0:
            cleaned.append(name)
            if run_log:
                _log(
                    run_log,
                    f"[i] 清理孤儿 worktree：{name}（worktree rc={rc_wt}, branch rc={rc_br}）",
                )
    _run_git("worktree", "prune")  # 兜底：清悬挂管理项
    return cleaned


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

    if outcome.code == 0:
        commit = _extract_commit(outcome.body)
        listed = _extract_backlog_ids(outcome.body)
        # 只销账本批内的 id；种子必 done（result=成功且种子为必做项，防御漏报）
        completed = [seed_id]
        for did in listed:
            if did in batch_ids and did not in completed:
                completed.append(did)
        for cid in completed:
            backlog_complete(
                cid,
                note=f"drain 批量销账（commit {commit or '未报告'}）",
                batch_size=len(completed),
                batch_seed=seed_id,
            )
        _log(
            run_log,
            f"[√] 批次 {label} 完成：销账 {len(completed)} 项（{', '.join(completed)}；"
            f"{elapsed}s，commit={commit or '-'}）",
        )
        return [
            {
                "id": cid,
                "status": "done",
                "commit": commit,
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
    # 模型映射自愈（用户裁决 2026-10-10）：--list-models 刷新 registry.models——
    # 重配 BYOK 后下次 drain 即恢复正确 UUID。任何失败仅告警不阻塞（宁旧勿空）。
    try:
        res = refresh_models()
        if res["changed"]:
            detail = "; ".join(
                f"{t}: {c['old'] or '(空)'} → {c['new']}"
                for t, c in res["changed"].items()
            )
            summary["models_refreshed"] = res["changed"]
            _log(run_log, f"[i] 模型映射已刷新：{detail}")
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
        # 顺带清理死 worker 残留的 worktree/branch（worktree 名 = 条目 id）；dry 演练不破坏
        if dry:
            _log(run_log, "[i] dry 模式：跳过孤儿 worktree 清理（非破坏性演练）")
        else:
            cleaned = cleanup_orphan_worktrees(orphans, run_log)
            if cleaned:
                summary["worktrees_cleaned"] = cleaned
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
        release_lock()
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


def _spawn_detached(argv: list[str], *, env: dict[str, str], stdout: Any) -> None:
    """以分离后台进程启动 argv（Windows: DETACHED_PROCESS；POSIX: 新会话）。"""
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
