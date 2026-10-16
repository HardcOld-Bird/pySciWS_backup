"""成员队列 worker（README §12，M4）：每成员一个**非常驻** FIFO worker。

推广 devops drain 的生命周期模式——**抽取共用，不复制粘贴**（§纪律）：锁/墓碑/收尾证据/
run 名同秒顺延全部走 :mod:`drain` 的 ``WorkerPaths`` + ``acquire_lock``/``release_lock``/
``new_run_log``/``lock_status``（bencher 亦复用同一套），派生走 ``drain._spawn_detached``
（自带「测试进程不真派生」的护栏）。

流程：入队且无存活 worker → :func:`wake_member` 派生 ``_queue-worker <member>`` → worker 获锁
（**per-member 锁防双 worker**）→ 孤儿回收（``workq.reset_orphans``：刚获锁时的 in_progress
必是上任被杀留下的）→ 循环 { 出队（``workq.take_next``——**出队点单点**，m6 在此接
backpressure 延期检查，本环节只留位置）→ ``do_dispatch`` 执行 → 记结果 → 下一条 } →
**队空写 ``.done`` 摘要并退出销毁**。``kind=backlog-batch``（backlog-fifo-20261014：devops
backlog 批次）不走通用记账，改经 :func:`_settle_batch_item` → ``drain.settle_backlog_batch``
落 backlog 账务（销账/needs_leader/延期/系统性停止），完成后**续计划**下一批；devops 队列
排空/全局停止时由本 worker 发 ``QUEUE-EMPTY``/``CHANNEL-STOP``（完成唤醒的执行侧落点）。

review 附属（§3.1 第 3 件）：宿主 job（``review=True``）交付成功后**自动**投 reviewer 子 job；
review 子 job 收尾时尽力回填宿主记录。宿主的「完成」由 :func:`queues.completion` 派生。
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from pysci.paths import ORCH_STATE_ROOT, PROJECT_ROOT

from . import backpressure, events, queues, stallwatch, statecommit, workq
from . import drain as _drain
from .drain import (
    BACKLOG_BATCH_KIND,
    DEFER_MAX_SLEEP_S,
    WorkerPaths,
    _consume_stop_request,  # drain 的实现——按 §纪律「抽取共用」直接引用，不复制
    _dry_settle_batch,
    _seed_tier,
    _spawn_detached,
    acquire_lock,
    lock_status,
    log_stage,
    new_run_log,
    release_lock,
    settle_backlog_batch,
)
from .registry import orchestration_rel
from .stallwatch import StallWatch

#: 成员 worker 的运行态根（锁/停止信号在 ``state/queues/`` 旁，run 日志独立成树）。
QUEUE_WORKER_ROOT: Path = ORCH_STATE_ROOT / "queues-runs"

#: 跳级 watchdog 判死后本进程的退出码（非 0：`.done` 不会写出，锁面据此可认出「被中途杀死」）。
STALL_EXIT_CODE: int = 3

#: 巡更自愈：孤儿复位后，若成员队列仍有 pending 且**最老者**超过这个岁数，顺手把 worker
#: 叫起来（鸡生蛋的另一半——复位只是把活放回队列，没人拉 worker 就等于没复位）。
RECLAIM_WAKE_MIN_AGE_S: int = 120

#: 巡更自愈的**唤醒冷却**（秒）：该成员的 run 目录里若有这个窗口内的日志，说明刚叫过，
#: 本轮不再叫——防「worker 起不来 → 每 60s 派生一次」的暴走（无状态判据，无需额外台账）。
RECLAIM_WAKE_COOLDOWN_S: int = 180

#: **交付正文全文落盘目录（入 git）**：长交付全文的唯一入库落点——队列条目 ``result.body``
#: 截断至 2000 字、``queues-runs/*.log`` 被 gitignore，长交付原文否则不入远端（用户裁决
#: 2026-10-15：黄金任务回归集需要全文参照）。一 job 一文件 ``<job_id>.md``。每次现读模块常量
#: 便于测试 monkeypatch 隔离（同 :data:`QUEUE_WORKER_ROOT`）。
DELIVERIES_DIR: Path = ORCH_STATE_ROOT / "deliveries"

#: 新留档的**落盘即提交**信息（:mod:`statecommit`）——「入 git」是声明，不是希望。
DELIVERY_COMMIT_MSG: str = "chore(state): 交付全文留档入库（新留档即提交）"

#: 双写者防护：同 plan/step 已有**活跃写者**时，本条**延期**回队多久（秒）再复核
#: （backlog 20261015-140209-f3-writer-guard）。够过期：写者要么已收尾（下次取出即正常跑），
#: 要么仍在跑（再延一轮，人工看得到那双写者证据）。
WRITER_HOLD_DEFER_S: int = 300


def _live_writer_of_sid(m: Any, sid: str) -> dict[str, Any] | None:
    """``sid`` 的上一跳进程是否仍活——**复用** sid 在途闸（pid 判活 + argv 核 sid 两闸）。

    不新造探活：判据与 :func:`dispatch._sid_inflight` 同一份（实案 20261012-184000 的原话
    「桩只给 pid 不给 argv 核对就把闸测成哑巴」）。
    """
    if not sid or sid in ("latest",):
        return None
    try:
        from . import dispatch as _dispatch

        return _dispatch._sid_inflight(m, sid)
    except Exception:  # noqa: BLE001 - 探活面 fail-open（不冤枉正常跑）
        return None


def writer_occupancy(
    member: str, item: dict[str, Any], *, root: Path | None = None
) -> dict[str, Any] | None:
    """同 plan/step 是否已有**活跃写者** → 占用证据 ``{pid, task, sid, via}``；无 → ``None``。

    为什么要它（backlog 20261015-140209-f3-writer-guard）：实案 job 20261015-122406 批次派发
    崩在「已派发、未结账」，条目坠落重派 → 同一份工单出现**两个活跃写者**（会话 6304 与重派
    会话抢写同一份 TUI 改进包），靠执行端自己认出双写者才没互踩。

    三条锚（任一给出「进程活着且 argv 里就是那个 sid」即判占用，全部 fail-open）：
    1. **会话池的 pid_task**——派发瞬间 ``note_hop_pid`` 把任务书名（``…-<plan>-<step>.md``）
       写进会话条目，按 ``-<plan>-<step>.md`` 后缀精确命中；
    2. **同队列里同 plan/step 的其它在途条目**的 ``session``；
    3. **plan state 该环节行**的 ``session_id``（上一次派发实际用的整条 sid）。
    """
    plan_id = str(item.get("plan") or "")
    step_id = str(item.get("step") or "")
    if not plan_id or not step_id:
        return None  # 计划外条目（adhoc / backlog 组包）无 plan/step 归属，不做此判
    try:
        from . import registry as _reg

        m = _reg.Registry.load().member(member)
    except Exception:  # noqa: BLE001 - 注册表读不到即不判（fail-open）
        return None

    # 锚 1：会话池里 pid_task 指向本 plan/step 任务书的活会话
    needle = f"-{plan_id}-{step_id}.md"
    try:
        for sess in getattr(m, "sessions", []) or []:
            if needle not in str(sess.get("pid_task") or ""):
                continue
            sid = str(sess.get("sid") or "")
            if occ := _live_writer_of_sid(m, sid):
                return {**occ, "sid": sid, "via": "session-pool"}
    except Exception:  # noqa: BLE001
        pass

    # 锚 2/3：队列同 plan/step 在途条目的 session ／ plan state 该行的 session_id
    sids: list[tuple[str, str]] = []
    try:
        for other in workq.load(queues.queue_path(member, root=root)):
            if str(other.get("id")) == str(item.get("id")):
                continue
            if (
                str(other.get("plan") or "") == plan_id
                and str(other.get("step") or "") == step_id
                and str(other.get("status") or "") == workq.STATUS_IN_PROGRESS
            ):
                sids.append((str(other.get("session") or ""), "queue-item"))
    except Exception:  # noqa: BLE001
        pass
    try:
        from . import plans as _plans

        row = next(
            (
                s
                for s in _plans._load_state(plan_id).get("steps", [])
                if str(s.get("id")) == step_id
            ),
            None,
        )
        if row:
            sids.append((str(row.get("session_id") or ""), "plan-state"))
    except Exception:  # noqa: BLE001
        pass
    seen: set[str] = set()
    for sid, via in sids:
        if not sid or sid in seen:
            continue
        seen.add(sid)
        if occ := _live_writer_of_sid(m, sid):
            return {**occ, "sid": sid, "via": via}
    return None


def _hold_for_writer(q: Path, item: dict[str, Any], occ: dict[str, Any]) -> None:
    """把本条**延期回队**（pending + ``defer_until``）并留证据 note——不丢、不抢、不互踩。"""
    note = (
        f"双写者防护：同 plan/step 已有活跃写者 pid={occ['pid']} sid={occ['sid']} "
        f"任务书 {occ['task']}（锚 {occ['via']}）——本条延期 {WRITER_HOLD_DEFER_S}s 再复核"
    )
    workq.mutate(
        q,
        str(item["id"]),
        status=workq.STATUS_PENDING,
        started_at="",
        defer_until=time.time() + WRITER_HOLD_DEFER_S,
        note=note,
    )


def announce_writer_hold(member: str, action: str, rows: list[dict[str, Any]]) -> None:
    """把「因活跃写者而暂缓」的决策**出声**到组长面（NEEDS_LEADER；fail-soft）。

    为什么走事件而不 `print`：worker 进程**不写 stdout**（它的可读面是 run 日志 + ``.done``，
    ``queueworker`` 全篇零 `print`）——出声要落到组长真能看见的地方。证据（pid/sid/任务书/
    锚）同时进了条目 note 与 run 日志（``writer_occupied_hold``/``orphan_held_writer_alive``）。
    """
    detail = "；".join(
        f"{r['id']}：写者 pid={r['pid']} sid={r['sid']} 任务书 {r['task']}（锚 {r['via']}）"
        for r in rows
    )
    try:
        events.emit(
            events.NEEDS_LEADER,
            subject=f"{member}:{rows[0]['id'] if rows else '?'}",
            detail=f"{member} {action} {len(rows)} 项（双写者防护，已延期回队不重派）：{detail}",
            action=(
                f"确认该写者是否真在跑（pid={rows[0]['pid'] if rows else '?'}）；"
                "人为放弃则手工清该条目 note 后改 pending"
            ),
        )
    except Exception:  # noqa: BLE001 - 出声失败不改派发结局
        pass


def orphans_guarded(
    q: Path, member: str, *, trace: Any = None
) -> tuple[list[str], list[dict[str, Any]]]:
    """孤儿复位**合一**写者核验（backlog 20261015-140209-f3-writer-guard）。

    获锁一刻仍在 ``in_progress`` 的条目**未必**是孤儿——也可能是**上一个 worker 被误判死**
    而它的会话进程还在写。老做法无条件复位 → 重派 → 双写者互踩（实案 20261015-122406 链）。
    故逐条核验上一写者死活：

    - **死** → 正常复位 ``pending``（复用 :func:`_reset_inflight`，字段语义同 ``reset_orphans``）；
    - **活** → **不复位不重派**，原地留证据 note 并**出声**（进 ``held`` 供调用方打日志/摘要），
      等下一轮巡更再核（写者收尾后自然放行）。

    Returns:
        ``(reset_ids, held_rows)``。
    """
    reset: list[str] = []
    held: list[dict[str, Any]] = []
    for it in workq.load(q):
        if str(it.get("status")) != workq.STATUS_IN_PROGRESS:
            continue
        occ = writer_occupancy(member, it, root=q.parent.parent)
        if occ is None:
            if _reset_inflight(
                q,
                str(it.get("id")),
                note=f"孤儿回收（{it.get('started_at', '?')} 起的在途项）",
            ):
                reset.append(str(it.get("id")))
            continue
        note = (
            f"孤儿回收**暂缓**：上一写者仍活 pid={occ['pid']} sid={occ['sid']} "
            f"任务书 {occ['task']}（锚 {occ['via']}）——不复位不重派，等其收尾后再核"
        )
        workq.mutate(q, str(it.get("id")), note=note)
        held.append({"id": str(it.get("id")), **occ})
        if trace is not None:
            trace("orphan_held_writer_alive", item=str(it.get("id")), pid=occ["pid"])
    return reset, held


def qodercli_error_tail(*, root: Path | None = None, max_lines: int = 15) -> str:
    """最近一次 qoderclicn run 日志的报错尾段（第一现场取证，用户明令 2026-10-15）。

    事故教训：网关 503 的报错只存在于 ``~/.qoder-cn/logs/runs/*/qodercli.log``，交付面
    零可见——组长与用户一起「蒙在鼓里」。跳失败时把 ERROR/WARN/异常行尾段并入交付留档。
    fail-soft：取证绝不阻塞 worker。``root`` 为测试注入口（默认 ``~/.qoder-cn/logs/runs``）。
    """
    try:
        runs = root or (Path.home() / ".qoder-cn" / "logs" / "runs")
        if not runs.is_dir():
            return ""
        latest = max(runs.iterdir(), key=lambda d: d.stat().st_mtime, default=None)
        if latest is None:
            return ""
        log = latest / "qodercli.log"
        if not log.is_file():
            return ""
        hits = [
            ln
            for ln in log.read_text(encoding="utf-8", errors="ignore").splitlines()
            if re.search(r"ERROR|WARN|Exception|5\d\d |timeout|refused|unavailable", ln)
        ]
        return "\n".join(hits[-max_lines:])
    except Exception:  # noqa: BLE001 - 取证面绝不反噬交付
        return ""


def archive_delivery(
    item: dict[str, Any], result: dict[str, Any], *, body: str, error: str = ""
) -> str:
    """把**完整交付正文**落 ``state/deliveries/<job_id>.md``（头部元数据 + 正文原文）→ 相对路径。

    - **失败交付同样留档**（blocked/failed 正文对返工分析有价值）：``error`` 一并写入「错误正文」段。
    - 写失败 **fail-soft**（留档面不阻塞 worker；返回空串，调用方不挂 ``body_file``）。
    """
    job_id = str(item.get("id") or "").strip()
    if not job_id:
        return ""
    body = str(body or "")
    err = str(error or "")
    try:
        DELIVERIES_DIR.mkdir(parents=True, exist_ok=True)
        p = DELIVERIES_DIR / f"{job_id}.md"
        head = [
            f"# 交付全文留档：{job_id}",
            "",
            f"- member: {result.get('member') or item.get('member') or ''}",
            f"- kind: {result.get('kind') or ''}",
            f"- code: {result.get('code')}",
            f"- plan: {item.get('plan') or ''}",
            f"- step: {item.get('step') or ''}",
            f"- task_file: {result.get('task_file') or ''}",
            f"- sid: {result.get('sid') or ''}",
            f"- duration_ms: {result.get('duration_ms')}",
            f"- ts: {datetime.now().astimezone().isoformat(timespec='seconds')}",
            "",
            "## 交付正文",
            "",
            body.rstrip() or "（空）",
        ]
        if err:
            head += ["", "## 错误正文", "", err.rstrip()]
            first_scene = qodercli_error_tail()
            if first_scene:
                head += [
                    "",
                    "## 第一现场（qoderclicn run 日志报错尾段）",
                    "",
                    first_scene,
                ]
        p.write_text("\n".join(head) + "\n", encoding="utf-8")
    except Exception:  # noqa: BLE001 - 留档面绝不阻塞 worker（正文仍在队列条目内务面）
        return ""
    # 声明「入 git」的面落盘即提交（backlog 20261015-085823-devops）：不提交则新留档永远
    # 停在 ``??``，远端/新克隆取不到交付全文——详情页引用的路径在远端悬空。fail-soft。
    statecommit.commit_face(DELIVERIES_DIR, DELIVERY_COMMIT_MSG)
    return orchestration_rel(p)


def member_paths(member: str, *, root: Path | None = None) -> WorkerPaths:
    """该成员 worker 的生命周期落点（锁 / run 目录 / 协作停止信号）。

    每次**现读**模块常量，便于测试 monkeypatch 隔离面（同 drain 的理由）。
    """
    base = (root or ORCH_STATE_ROOT) / "queues"
    runs = (root or ORCH_STATE_ROOT) / "queues-runs" / member
    return WorkerPaths(
        lock=base / f"{member}.lock",
        runs=runs,
        cancel=base / f"{member}.cancel",
    )


def _worker_argv(member: str, run_log: Path, *, dry: bool) -> list[str]:
    """内部命令 argv（``sys.executable -m`` → 从任意 cwd 可用）。"""
    argv = [
        sys.executable,
        "-m",
        "pysci.skills.orchestration.tools.orch",
        "_queue-worker",
        member,
        "--run-log",
        str(run_log),
    ]
    if dry:
        argv.append("--dry")
    return argv


def wake_member(
    member: str, *, dry: bool = False, root: Path | None = None
) -> tuple[bool, str]:
    """入队后拉起该成员的 worker；锁忙（已有存活 worker）则不重复派生。

    Returns:
        ``(spawned, message)``——``False`` = 已有 worker 在跑，本条由当前循环接手。
    """
    paths = member_paths(member, root=root)
    st = lock_status(paths)
    if st.get("state") == "running":
        return False, f"{member} worker 正在跑队列，本条由当前循环接手（不重复派生）。"
    paths.runs.mkdir(parents=True, exist_ok=True)
    run_log = new_run_log(paths)
    argv = _worker_argv(member, run_log, dry=dry)
    env = {**os.environ, "PYTHONUTF8": "1"}
    with open(run_log, "ab") as logf:
        _spawn_detached(argv, env=env, stdout=logf)
    return True, f"{member} worker 已后台唤醒（日志 {orchestration_rel(run_log)}）"


def _execute(
    item: dict[str, Any],
    *,
    dry: bool,
    trace: Any = None,
) -> tuple[str, dict[str, Any], Any]:
    """执行一条 job（复用 :func:`dispatch.do_dispatch`，会话策略沿派发惯例）。

    返回 ``(status, result, outcome)``；``outcome`` 原样带出（带宽类延期判据要用它的
    ``kind``/``error_text``，别在结果字典里丢掉分类原料）。

    ``trace``：阶段埋点/进展回调（:func:`run_worker` 给的那个）——透传给 ``do_dispatch``，
    让它把「走到哪一步了」写进 run 日志（2026-10-15 worker-stall 单：静默卡死的取证面）。
    """
    if dry:
        return workq.STATUS_DONE, {"dry": True}, None
    from .dispatch import do_dispatch  # 延迟导入：dispatch 引 runner 链

    out = do_dispatch(
        str(item.get("member")),
        text=str(item.get("task") or "") or None,
        task_file=str(item.get("task_file") or "") or None,
        session=str(item.get("session") or "latest"),
        effort=str(item.get("effort") or "") or None,
        model_tier=str(item.get("model") or "") or None,
        dirs=list(item.get("dirs") or []),
        plan=str(item.get("plan") or ""),
        step=str(item.get("step") or ""),
        max_turns=item.get("max_turns") or None,
        quiet=True,
        auto=True,
        trace=trace,
    )
    code = int(getattr(out, "code", 2))
    env = getattr(out, "envelope", None)
    # 结果字典要能**独立撑起阻塞推进原语的交接报告**（直派退役后它不再拿
    # DispatchOutcome，而是从本条目还原）——分类/正文/验收/建议/时长/会话一并落盘。
    result: dict[str, Any] = {
        "kind": str(getattr(out, "kind", "")),
        "body": str(getattr(out, "body", ""))[:2000],
        "error": str(getattr(out, "error", ""))[:500],
        "task_file": str(item.get("task_file") or ""),
        "code": code,
        "member": str(getattr(out, "member", "") or item.get("member") or ""),
        "sid": str(getattr(out, "sid", "") or ""),
        "session_name": str(getattr(out, "session_name", "") or ""),
        "checks": [dict(c) for c in (getattr(out, "checks", None) or [])],
        "suggestion_id": str(getattr(out, "suggestion_id", "") or ""),
        "duration_ms": int(getattr(env, "duration_ms", 0) or 0) if env else 0,
    }
    # 交付正文**全文留档**（入 git）：result.body 仍截 2000 字（内务读取面不破），
    # 新增 ``body_file`` 指向全文；截断时在 body 末尾注明（读取面据此顺带透出路径）。
    body_file = archive_delivery(
        item,
        result,
        body=str(getattr(out, "body", "")),
        error=str(getattr(out, "error", "")),
    )
    if body_file:
        result["body_file"] = body_file
        if len(str(getattr(out, "body", ""))) > 2000:
            result["body"] = f"{result['body']}\n…（截断，全文见 {body_file}）"
    return (workq.STATUS_DONE if code == 0 else workq.STATUS_FAILED), result, out


def _defer_plan(outcome: Any, item: dict[str, Any], now: float):
    """带宽类失败的延期判决（**复用** drain 的纯函数，不重写，§纪律）。

    成员队列与 devops backlog 走同一套分类/退避/双闸（``backpressure.plan_defer``）；
    devops 既有语义不动，此处只是把同一判据接到成员队列 worker 的出队处置上。
    """
    if outcome is None:
        return None
    from .drain import _bandwidth_defer_plan

    return _bandwidth_defer_plan(outcome, item, now)


def _delivery_note(result: dict[str, Any] | None) -> str:
    """事件细节里的「交付全文留档」段（**绝对路径**，可直接 Read）。

    为什么放进事件（backlog 20261015-205906）：组长若想在 `plan advance`（结算+派发）**之前**
    先读完整交付把关，此前只能猜目录/翻 ledger/`find`——而路径其实早就在结果里
    （``body_file``），只是没透到事件面。事件细节会原样进 watch 唤醒行与 `events drain`
    输出（``watch._wake_message``），故一处补齐两处可见。``events.jsonl`` 属 gitignore 的
    运行面，写机器绝对路径无碍；留档缺失（写失败 fail-soft）时返回空串、不加噪声。
    """
    rel = str((result or {}).get("body_file") or "").strip()
    if not rel:
        return ""
    p = Path(rel)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return f"｜交付全文留档：{p}"


def _host_delivery_note(host_id: str, host_member: str, *, root: Path | None) -> str:
    """合并单元（宿主+review）在 review 收尾时发的事件：优先给**宿主**的留档路径。

    取不到宿主条目就退回 review 自己那条的（best-effort，绝不抛）。
    """
    try:
        host = next(
            (
                i
                for i in queues.items(host_member, root=root)
                if str(i.get("id")) == str(host_id)
            ),
            None,
        )
    except Exception:  # noqa: BLE001 - 事件面绝不阻塞生产
        host = None
    return _delivery_note(host)


def _notify(
    q: Path,
    item: dict[str, Any],
    status: str,
    result: dict[str, Any],
    member: str,
    *,
    root: Path | None = None,
) -> None:
    """worker 消化处**直发**统一事件（§3.2）——JOB-DONE / STEP-REVIEW-DONE。

    合并单元（§3.1）：带 ``review`` 的宿主 job 交付后**不**发 JOB-DONE，等 review 子 job
    收尾时按 :func:`queues.completion` 判「宿主+review 均收尾」才发**一次**
    ``STEP-REVIEW-DONE``。无 review 的普通 job 每件发 JOB-DONE。写失败 fail-open。
    """
    try:
        if str(item.get("kind")) == "review":
            host_id = str(item.get("parent") or "")
            host_member = str(item.get("parent_member") or "")
            state, why = queues.completion(host_id, host_member=host_member, root=root)
            if state == queues.COMPLETE:
                events.emit(
                    events.STEP_REVIEW_DONE,
                    subject=host_id,
                    detail=f"{host_member} 环节 + review 合并完成（{why}）"
                    + (
                        _host_delivery_note(host_id, host_member, root=root)
                        or _delivery_note(result)
                    ),
                    action="orch events drain 后逐 plan 推进；orch status 看交付明细",
                )
            return
        if item.get("review"):
            state, why = queues.completion(
                str(item["id"]), host_member=member, root=root
            )
            if state == queues.COMPLETE:  # 无子 job 或子 job 已收尾（宿主失败/已完成）
                events.emit(
                    events.STEP_REVIEW_DONE,
                    subject=str(item["id"]),
                    detail=f"{member} 环节 + review 合并完成（{why}）"
                    + _delivery_note(result),
                    action="orch events drain 后逐 plan 推进；orch status 看交付明细",
                )
            return  # AWAITING_REVIEW → 等 review 子 job 收尾时那一支发
        events.emit(
            events.JOB_DONE,
            subject=str(item["id"]),
            detail=f"{member} job 收尾：{status}（{result.get('kind', '')}）"
            + _delivery_note(result),
            action="orch events drain 后接续下一份工作；orch plan advance 推进",
        )
    except Exception:  # noqa: BLE001 - 事件面绝不阻塞生产（§0.6）
        pass


def _settle_batch_item(
    q: Path,
    member: str,
    item: dict[str, Any],
    outcome: Any,
    run_log: Path,
    summary: dict[str, Any],
    *,
    t0: float,
    dry: bool,
    error: str = "",
    root: Path | None = None,
    result: dict[str, Any] | None = None,
) -> None:
    """backlog 批次条目的执行收尾：drain 账务（唯一实现）→ 队列条目终态。

    - done（销账完成）/ failed（needs_leader 或系统性停止）→ ``workq.finish``；
      systemic 时置 ``summary["stopped"]`` 由调用方停队（全局停止语义）。
    - deferred（带宽延期）→ 账务已把 backlog 条目写 ``defer_until``；本条目同步挂
      ``defer_until``（worker 睡到到期自动重取，**不停链**）。
    - dry（干跑演练）→ 模拟逐 id 销账（与计划器 dry 支同源）。
    - ``outcome is None``（**派发抛异常**，非 dry）→ 按 ``run_failed`` 合成结果结算——
      绝不能误走 dry 支把失败批次销账成 done（2026-10-14 子代理实测抓到的接线缺陷）。
    - ``result``（``_execute`` 的交付结果）仅供补挂 ``body_file``：批次条目终态存的是
      backlog 账务行，全文留档路径不带上就丢在读取面之外。

    账务行落**本 worker 的 run 日志**（执行账与计划账分列，见 drain 模块 docstring）。
    """
    if dry:
        meta = dict(item.get("backlog") or {})
        seed = dict(meta.get("seed") or {})
        group = [dict(g) for g in (meta.get("group") or [])]
        results = _dry_settle_batch(seed, group, run_log)
        workq.finish(
            q, str(item["id"]), workq.STATUS_DONE, dict(results[0]) if results else {}
        )
        summary["items"].append(
            {"id": str(item["id"]), "status": "dry", "backlog_results": results}
        )
        return
    if outcome is None:
        from .dispatch import DispatchOutcome  # 延迟导入：dispatch 引 runner 链

        outcome = DispatchOutcome(
            code=2,
            kind="run_failed",
            body="",
            error=error or "批次派发异常",
            member=member,
        )
    work_status, results, stop = settle_backlog_batch(item, outcome, run_log, t0=t0)
    first = dict(results[0]) if results else {}
    if work_status == "deferred":
        delay = float(first.get("defer_s") or 0.0) or 60.0
        meta = dict(item.get("backlog") or {})
        seed = dict(meta.get("seed") or {})
        used = float(seed.get("defer_elapsed_s") or 0.0) + delay
        queues.defer(
            member,
            str(item["id"]),
            until=time.time() + delay,
            count=int(first.get("defer_count") or 1),
            elapsed_s=used,
            note="devops 批次带宽延期：等瞬态带宽恢复后重取（backlog 条目侧已同步 defer）",
            root=root,
        )
        summary["items"].append(
            {
                "id": str(item["id"]),
                "status": "deferred",
                "backlog_results": results,
            }
        )
        return
    status = workq.STATUS_DONE if work_status == "done" else workq.STATUS_FAILED
    # 批次条目终态存的是 backlog 账务行（非交付正文），`_execute` 算出的 `body_file`
    # 会在这里被丢掉——补挂上去，读取面才拿得到全文路径（留档本身已由 `_execute` 写好）。
    if str((result or {}).get("body_file") or ""):
        first["body_file"] = str(result["body_file"])
    workq.finish(q, str(item["id"]), status, first)
    summary["items"].append(
        {"id": str(item["id"]), "status": work_status, "backlog_results": results}
    )
    if stop:
        summary["stopped"] = stop
        summary["error"] = str(first.get("error", "") or "")


def _reset_inflight(q: Path, item_id: str, *, note: str) -> bool:
    """把**单条**在途条目复位 ``pending``（字段语义与 :func:`workq.reset_orphans` 一致）。"""
    if not item_id:
        return False
    return (
        workq.mutate(
            q,
            item_id,
            status=workq.STATUS_PENDING,
            started_at="",
            reclaimed_at=workq._now(),
            note=note,
        )
        is not None
    )


def _make_stall_handler(member: str, q: Path, run_log: Path) -> Any:
    """造 watchdog 的 ``on_stall`` 回调：复位条目 → 出声 → 杀进程树 → 硬退出。

    为什么必须杀到**本进程**（而不只是那个 headless 子进程）：主线程此刻正卡在
    ``run_headless`` 的 ``communicate()`` 里，没有协作退出点可返回；不硬退就会「复活」成
    第二个消费者（条目已复位成 pending，worker 拿到锁却没死 → 双 worker 并发消费）。
    先杀已派生的会话子进程（它还在写 worktree），再杀自己（含后代），最后 ``os._exit``
    兜底 Windows 上 taskkill 与 os._exit 之间的竞态。

    出声两路：① ``NEEDS-LEADER`` 事件（记录面 + 巡检命名）；② 复位本身让
    ``watch.queue_pressure`` 的「队列有 pending 且无 worker」判据成立 → 一个巡更周期内
    ``PENDING-WORK`` 差分唤醒组长；本进程被杀且无 ``.done`` → 锁面 ``crashed`` →
    既有 ``CRASH`` 差分也报一次。三路并行，不留静默。
    """
    from .stallwatch import JobProgress  # noqa: F401  （类型注解用途）

    def handler(verdict: str, reason: str, prog: Any) -> None:
        log_stage(
            run_log, "stall_detected", verdict=verdict, item=prog.item_id, detail=reason
        )
        _reset_inflight(
            q, str(prog.item_id), note=f"watchdog 判卡死（{verdict}）：{reason}"
        )
        try:
            events.emit(
                events.NEEDS_LEADER,
                subject=str(prog.item_id or member),
                detail=f"{member} worker 卡死已被 watchdog 处置：{reason}",
                action=(
                    "条目已回 pending；orch status 看队列读数；本 worker 已自退，"
                    "下次唤醒会重新消费该条目（卡死成因见该 run 的 run 日志）"
                ),
            )
        except Exception:  # noqa: BLE001 - 出声失败也要继续处置（不能因它卡住）
            pass
        if int(prog.pid or 0) > 0:
            stallwatch.kill_tree(int(prog.pid))
        stallwatch.kill_tree(os.getpid())
        stallwatch.hard_exit(STALL_EXIT_CODE)

    return handler


def _member_recent_log(member: str, *, now: float, window_s: int) -> bool:
    """该成员的 run 目录里有没有 ``window_s`` 内写过的日志（= 刚叫过 worker，别重复叫）。"""
    runs = member_paths(member).runs
    try:
        logs = list(runs.glob("*.log"))
    except OSError:
        return False
    return any(
        (now - p.stat().st_mtime) <= window_s
        for p in logs
        if _safe_mtime(p) is not None
    )


def _safe_mtime(p: Path) -> float | None:
    try:
        return p.stat().st_mtime
    except OSError:
        return None


def reclaim_orphans(*, now: float | None = None, wake: bool = True) -> dict[str, Any]:
    """**巡更自愈**：把「``in_progress`` 但无活 worker」的孤儿条目复位 ``pending`` 并唤醒。

    补的缺口（task-20261015-worker-stall B2「鸡生蛋」）：``workq.reset_orphans`` 只在
    **下次获锁**时执行，而队列若已无 ``pending`` 就再没人拉 worker——孤儿条目永久假挂
    ``in_progress``、锁永久占位，正是 2026-10-15 08:02 实案（31+ min 无人处置）。本函数
    把同一回收动作搬到**定期巡更面**（:func:`watchdog.run_watchdog` 每 tick 调用），不再
    依赖「下次获锁」这个不存在的下次。

    判据：该成员队列有在途条目 **且** 该成员的 worker 锁非 ``running``（锁在跑=那些在途是
    它的，不碰）。复位之后若仍有**非延期**的 ``pending`` 且最老者已超
    :data:`RECLAIM_WAKE_MIN_AGE_S`，且该成员 :data:`RECLAIM_WAKE_COOLDOWN_S` 内没派生过
    worker，则顺手 ``wake_member``——把「活放回队列」补成「真的有人来干」。

    Returns:
        ``{"reclaimed": {member: [item_id, …]}, "woken": [member, …], "notes": [...]}``。
    """
    moment = time.time() if now is None else float(now)
    out: dict[str, Any] = {"reclaimed": {}, "woken": [], "notes": []}
    for member in queues.all_members():
        q = queues.queue_path(member)
        try:
            items = workq.load(q)
        except Exception:  # noqa: BLE001 - 单个队列坏不挡其余成员
            continue
        if not items:
            continue
        if lock_status(member_paths(member)).get("state") == "running":
            continue  # 有活 worker：在途条目是它的
        inflight = [
            str(i.get("id"))
            for i in items
            if str(i.get("status")) == workq.STATUS_IN_PROGRESS
        ]
        if inflight:
            done, held = orphans_guarded(q, member)
            out["reclaimed"][member] = done
            out["notes"].append(
                f"{member}：孤儿在途 {len(done)} 项已复位 pending（无活 worker）"
            )
            if held:
                out.setdefault("held", {})[member] = held
                out["notes"].append(
                    f"{member}：{len(held)} 项**未复位**——上一写者进程仍活（不重派，等其收尾后再核）"
                )
                announce_writer_hold(member, "孤儿复位暂缓", held)
            try:
                events.emit(
                    events.NEEDS_LEADER,
                    subject=f"{member}:{done[0] if done else '?'}",
                    detail=(
                        f"{member} worker 不在场却留下 {len(done)} 项 in_progress —— "
                        "已按孤儿回收复位为 pending 并重新唤醒（自愈，非静默）"
                    ),
                    action="orch status 看队列读数；本条仅需知悉（已自愈，卡死成因见该 run 日志）",
                )
            except Exception:  # noqa: BLE001 - 出声失败不挡自愈
                pass
        if not wake:
            continue
        pend = [
            i
            for i in workq.load(q)
            if str(i.get("status")) == workq.STATUS_PENDING
            and not backpressure.item_deferred(i, moment)
        ]
        if not pend:
            continue
        epochs = [workq.item_epoch(i) for i in pend]
        known = [e for e in epochs if e is not None]
        if not known or moment - min(known) < RECLAIM_WAKE_MIN_AGE_S:
            continue
        if _member_recent_log(member, now=moment, window_s=RECLAIM_WAKE_COOLDOWN_S):
            out["notes"].append(
                f"{member}：pending {len(pend)} 项但 {RECLAIM_WAKE_COOLDOWN_S}s 内刚派生过"
                " worker，本轮不重复叫"
            )
            continue
        spawned, msg = wake_member(member)
        if spawned:
            out["woken"].append(member)
            out["notes"].append(f"{member}：有活无人干 → 已唤醒 worker（{msg}）")
    return out


def run_worker(
    member: str,
    run_log: Path,
    *,
    dry: bool = False,
    root: Path | None = None,
    watch: Any = None,
) -> dict[str, Any]:
    """FIFO 消化该成员队列（持锁运行；异常/结束均释放锁并写 ``.done`` 摘要）。

    §3.3：写侧接线——登记活跃会话 + 守护线程心跳（长 job 阻塞在 ``do_dispatch`` 里也有
    心跳、不被判死）；出队单点先过延期判据；带宽类失败回队挂 ``defer_until``，到期自动重取，
    双闸触顶则停止本队列并上报（发 CHANNEL-STOP 事件）。

    **阶段埋点 + 跳级 watchdog**（2026-10-15 worker-stall 单）：本函数把「走到哪一步」逐段
    写进 run 日志（``drain.log_stage``，**不受 ``quiet`` 影响**——只写文件不写 stdout），
    并同时喂给 :class:`~.stallwatch.StallWatch` 作「无进展」判据；判死即复位条目、出声、
    杀进程树（见 :func:`_make_stall_handler`）。``watch`` 允许注入（测试用假监护）。
    """
    paths = member_paths(member, root=root)
    q = queues.queue_path(member, root=root)
    if acquire_lock(orchestration_rel(run_log), paths) is None:
        return {"skipped": "lock_busy"}
    summary: dict[str, Any] = {
        "member": member,
        "started": workq._now(),
        "items": [],
        "run_log": str(run_log),
    }
    t_worker = time.time()
    log_stage(run_log, "lock_acquired", member=member, pid=os.getpid(), dry=dry)
    # 进展面：一个回调同时供**埋点**（写 run 日志）与**判据**（StallWatch 的无进展判据）。
    guard = (
        watch
        if watch is not None
        else StallWatch(on_stall=_make_stall_handler(member, q, run_log))
    )
    guard.start()
    # 先按「worker 启动」占位起跑：获锁到取到第一条 job 之间的卡死（实案就发生在这段：
    # 零会话登记 ⇒ 根本没走到派生）同样要有判据可用。
    guard.arm("(worker-startup)", member)

    def trace(stage: str, **fields: Any) -> None:
        log_stage(run_log, stage, **fields)
        guard.note(stage, **fields)

    # devops 的 QUEUE-EMPTY = **backlog 消化完**的知会（旧 drain 语义）；plan 环节直派的
    # devops job 走通用 JOB-DONE 收账，不再叠发（task-20261014-devops-gap-d1 双通道对齐）。
    saw_batch = False
    # 死会话条目（注册表过期）与死队列条目（in_progress 孤儿）各清各的，不重复回收。
    trace("prune_stale")
    backpressure.prune_stale(now=time.time())
    trace("live_session_register")
    sess = backpressure.LiveSession(
        member,
        sid=run_log.stem,
        tier=_seed_tier({"member": member}),
        purpose="queue-worker",
    ).start()
    try:
        trace("reset_orphans_begin")
        orphans, orphan_held = orphans_guarded(q, member, trace=trace)
        if orphans:
            summary["reclaimed"] = orphans
            trace("reset_orphans_done", count=len(orphans))
        if orphan_held:
            summary["orphan_held"] = orphan_held
            announce_writer_hold(member, "孤儿复位暂缓", orphan_held)
        while True:
            # 每条新迭代先解除上一轮的监护：条目已结账（或已延期），它的「无进展」判词
            # 不再有意义——带着旧判据继续计时就会把已收尾的 job 判成卡死（误杀）。
            guard.disarm()
            if _consume_stop_request(paths):
                summary["stopped"] = "cooperative"
                trace("stop_requested")
                break
            trace("take_ready")
            item = queues.take_ready(member, root=root)  # 出队单点：过延期判据
            if item is None:
                guard.disarm()
                wait = 0.0 if dry else queues.next_defer_s(member, root=root)
                if wait > 0:
                    _drain._sleep(min(wait, float(DEFER_MAX_SLEEP_S)))
                    continue
                break
            # 双写者防护（backlog 20261015-140209-f3-writer-guard）：同 plan/step 已有活跃
            # 写者 → **不派发**，延期回队 + 留证据 note + 出声；等写者收尾后下一轮自然放行。
            occ = writer_occupancy(member, item, root=root)
            if occ is not None:
                _hold_for_writer(q, item, occ)
                trace(
                    "writer_occupied_hold",
                    item=str(item["id"]),
                    pid=occ["pid"],
                    sid=occ["sid"],
                    via=occ["via"],
                )
                announce_writer_hold(
                    member, "拒绝派发", [{"id": str(item["id"]), **occ}]
                )
                continue
            guard.arm(str(item["id"]), member)  # 本跳起跑线（无进展判据在此归零）
            trace("item_taken", item=str(item["id"]), kind=str(item.get("kind") or ""))
            # 单条目异常**就地结账**（2026-10-14 实测：`task=` 参数名写错 → TypeError 逃出
            # 循环 → 条目永远卡 in_progress、推进干等到超时）。失败也要 finish：发
            # JOB-DONE 让 plan advance 拿到事件而不是空等；本队列其余条目继续。
            t0 = time.time()
            try:
                status, result, outcome = _execute(item, dry=dry, trace=trace)
                trace(
                    "hop_settled",
                    item=str(item["id"]),
                    status=status,
                    code=result.get("code"),
                )
            except Exception as exc:  # noqa: BLE001 - 一条目炸不得炸整队
                if str(item.get("kind")) == BACKLOG_BATCH_KIND:
                    # 派发异常：栈进 run 日志（旧 drain 的异常留痕语义——不留黑洞）
                    _drain._log_exception(run_log, exc, "批次派发异常")
                status = workq.STATUS_FAILED
                result = {
                    "kind": "run_failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "code": 2,
                    "member": member,
                }
                outcome = None
                summary["item_error"] = result["error"]
                _drain._log_exception(run_log, exc, "job 执行异常")
            # devops backlog 批次（计划器入队）：账务=backlog 销账/needs_leader/延期/系统性
            # 停止，全部走 drain 的单一实现；本条目终态由结算结果映射（见 _settle_batch_item）。
            if str(item.get("kind")) == BACKLOG_BATCH_KIND:
                saw_batch = True
                _settle_batch_item(
                    q,
                    member,
                    item,
                    outcome,
                    run_log,
                    summary,
                    t0=t0,
                    dry=dry,
                    error=str(result.get("error") or ""),
                    root=root,
                    result=result,
                )
                last = (
                    str(summary["items"][-1].get("status", ""))
                    if summary["items"]
                    else ""
                )
                if last != "deferred":  # 延期不是终态，不发收尾事件（同普通条目路径）
                    _notify(q, item, last, result, member, root=root)
                if summary.get("stopped"):  # 系统性全局停止：停队，剩余条目保持 pending
                    break
                # 停止优先于续计划：协作停止信号在场时不把更多 backlog 条目拉成在途
                # （「条目间隙停止：未取项仍 pending」语义的落点）。
                if _consume_stop_request(paths):
                    summary["stopped"] = "cooperative"
                    break
                # 续计划（backlog-fifo-20261014）：被卡批次里未选中的组包项留在 pending，
                # 旧 drain 循环会在下一轮把它们当新种子再取——执行搬到 worker 后由此补齐，
                # 否则「一项卡住不阻塞后续」退化成「等下一次 approve」。
                _drain.replan_from_backlog(dry=dry)
                continue
            plan = _defer_plan(outcome, item, time.time())
            if plan is not None and not plan.exhausted:
                nth = plan.attempt + 1
                used = float(item.get("defer_elapsed_s") or 0.0) + plan.delay_s
                queues.defer(
                    member,
                    str(item["id"]),
                    until=plan.until,
                    count=nth,
                    elapsed_s=used,
                    note=(
                        f"带宽延期第 {nth} 次（等 {plan.delay_s}s 后重试）"
                        "——带宽不足是瞬态，不换席不停链；预算触顶则停止本队列并上报"
                    ),
                    root=root,
                )
                summary["items"].append({"id": str(item["id"]), "status": "deferred"})
                continue
            if plan is not None and plan.exhausted:
                # 双闸触顶仍未愈 → 停止本队列并上报（沿 devops 语义；发渠道级停止事件）
                workq.finish(
                    q,
                    str(item["id"]),
                    workq.STATUS_FAILED,
                    {**result, "saturated": True, "exhausted": plan.reason},
                )
                summary["items"].append(
                    {"id": str(item["id"]), "status": "channel_saturated"}
                )
                summary["stopped"] = "channel_saturated"
                events.emit(
                    events.CHANNEL_STOP,
                    subject=str(item["id"]),
                    detail=f"{member} job 带宽延期触顶仍未愈（{plan.reason}）→ 停止本队列",
                    action="渠道恢复后重新入队/唤醒；orch status 看延期读数",
                )
                break
            workq.finish(q, str(item["id"]), status, result)
            summary["items"].append({"id": str(item["id"]), "status": status})
            if status == workq.STATUS_DONE:
                if str(item.get("kind")) == "review":
                    queues.backfill_host(item, result, root=root)
                else:
                    child = queues.spawn_review(item, root=root)
                    if child:
                        summary["items"].append(
                            {"id": str(child["id"]), "status": "queued"}
                        )
            _notify(q, item, status, result, member, root=root)
        workq.prune(q)
    except Exception as exc:  # noqa: BLE001 - 异常也要写 .done（不留黑洞）
        summary["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        guard.disarm()
        guard.stop()  # 监护线程随 worker 退场：不留一条还在判死已收尾 job 的线
        sess.stop()  # 退场注销活跃会话（先于释放锁，读数面不留幽灵）
        release_lock(run_log, paths)
        summary["completed"] = workq._now()
        summary["count"] = len(summary["items"])
        summary["elapsed_s"] = round(time.time() - t_worker, 1)
        # blocked 成因档位分列（backlog 20261013-191213-devops 语义迁移到执行账）：
        # 计划器 items 已是「queued」无 reason，档位只在批次的账务行里——从 backlog_results
        # 聚合进 .done，供 orch status 分列「待裁决 / 待排期」。
        by_reason: dict[str, int] = {}
        for entry in summary["items"]:
            for row in entry.get("backlog_results") or []:
                r = str(row.get("reason") or "")
                if r:
                    by_reason[r] = by_reason.get(r, 0) + 1
        if by_reason:
            summary["by_reason"] = by_reason
        done = run_log.with_suffix(".done")
        done.parent.mkdir(parents=True, exist_ok=True)
        done.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        # devops 完成唤醒（backlog-fifo-20261014 迁移）：执行只留队列 worker 一份实现后，
        # 「消化完」的消费事实由**执行侧**发——统一事件流一条 QUEUE-EMPTY（旧 drain.finally
        # 的职责平移至此）；仅在消化过 backlog 批次的 run 发（plan 环节直派走 JOB-DONE 收账，
        # 双通道对齐见 task-20261014-devops-gap-d1）；带宽延期触顶回升的全局停止另发一条
        # 渠道级停止事件。写失败 fail-open。
        if member == "devops" and saw_batch:
            try:
                events.note_queue_empty(
                    "devops",
                    subject=run_log.stem,
                    count=summary["count"],
                    elapsed_s=summary.get("elapsed_s", "?"),
                    stopped=str(summary.get("stopped", "") or ""),
                )
            except Exception:  # noqa: BLE001 - 事件面绝不阻塞 worker（§0.6）
                pass
            if str(summary.get("stopped", "") or "") == "channel_saturated":
                try:
                    events.emit(
                        events.CHANNEL_STOP,
                        subject=run_log.stem,
                        detail=(
                            "devops 带宽延期触顶仍未愈 → 全局停止（"
                            + str(summary.get("error", "") or "")[:80]
                            + "）"
                        ),
                        action="渠道恢复后重新入队/唤醒；orch status 看延期读数",
                    )
                except Exception:  # noqa: BLE001
                    pass
    return summary


def cmd_queue_worker(args: Any) -> int:
    """内部命令 ``_queue-worker <member>``：持锁消化该成员队列，落 run 日志 + .done。"""
    run_log = Path(args.run_log) if getattr(args, "run_log", None) else None
    if run_log is None:
        paths = member_paths(str(args.member))
        paths.runs.mkdir(parents=True, exist_ok=True)
        run_log = new_run_log(paths)
    elif not run_log.is_absolute():
        run_log = Path(os.getcwd()) / run_log
    run_worker(str(args.member), run_log, dry=bool(getattr(args, "dry", False)))
    return 0
