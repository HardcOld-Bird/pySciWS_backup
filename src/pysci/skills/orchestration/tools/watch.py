"""orch watch 事件等待器：把 drain 观察闭环从「组长记得去查」硬化成代码（README §4.5/§6）。

理念（用户裁决 2026-10-10：能代码硬保证的不靠 LLM 纪律）：后台 drain worker 是**分离
进程**，组长派发完就没事了，体系此前只能靠组长自觉轮询 ``orch status`` 才知道队列跑完 /
崩了 / 卡在额度上——这正是「靠纪律」的软点。结构上限在于**唤醒注册只能由组长会话的
工具发起**（外部进程无法向 TUI 注入通知），故硬化到能硬的那一层：

- **事件判定全在代码**（本模块）：读状态文件 + Python 侧 ``_pid_alive``（不起 shell，
  故 MSYS 路径转换问题免疫），**七类**事件——CRASH / QUOTA-STOP / CHANNEL-STOP /
  NEEDS-LEADER / **TASKBOOK-NEW** / QUEUE-COMPLETE / SALVAGE（TASKBOOK-NEW 是 M3 的
  最小扩展：只加事件名 + 检测函数 `_taskbook_ids` + 基线字段 `taskbook_ids`；**m5 收编**
  进统一事件层，届时本支改为读事件流）。
- **挂载自 2026-10-13 起自动化**（README §4.5 v2）：组长回合结束由 ``Stop + asyncRewake``
  钩子（``.qoder/hooks/watch-arm.sh``）每回合抢「最新在听者」并挂 ``orch watch``——**无需
  组长手动挂**；:func:`autowatch_lines` 只在交办/唤醒当刻回报「已自动挂载」。

手动挂载降为**兜底**（仅钩子不可用时）：组长以 Bash ``run_in_background`` 启动
``pysci-orch watch``（阻塞轮询，默认 60 s、硬上限 60 s），命中即原生后台通知唤醒本会话；
**必须作为该后台调用的唯一顶层命令**（嵌入 ``&``/``&&``/管道会被连坐杀死）。watch 随组长
会话消亡、**不影响 detached drain**；漏挂/会话中断期间的账由下次 ``orch status`` 经
:func:`reconcile` 补账。

状态落 ``state/watch.json``（人类可读 JSON）：基线快照 + 心跳 + 已发事件 + 黑匣子
末帧（:data:`RECENT_MAX`），供 status 判断 watch 是否仍挂载（心跳新鲜=挂载中）。

QUEUE-COMPLETE 的判定纪律（backlog 20261011-220622-leader，源于 210557 那场
``fired=[]`` 脱靶）：**完成事实的存在性优先于锁的自称**——``.done`` 由 drain 在
``release_lock`` 之后写出，故「该 run 有 ``.done``」即收尾硬证据，此刻锁仍显示
running 只可能是 unlink 失败残留 + 判活说谎（pid 回收 / tasklist 超时保守判活）。
完成身份是 **(run 名, completed 时刻, count) 三元组**，且**抑制帧 / 瞬态空值一律
不得推进基线**（:func:`advance_base`）——否则被吞掉的事实永不参与下次比较，watch
与 status 补账会同时哑掉。判读所需现场每帧写进 ``recent``。

同一条纪律现已在**锁面**同样执法（backlog 20261011-225933-devops 的 released 墓碑 +
20261011-231237-devops 的 ``_run_concluded`` 判活否决），新代码不再产生「谎锁压事件」
的帧；本模块 :func:`_in_progress` 的抑制支与转移帧兜底因此退为**保险**：既护升级瞬间
残留的旧基线，也顶住 ``.done`` 半写这类读不出证据的瞬态。
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from pysci.paths import ORCH_STATE_ROOT

from . import drain as _drain
from . import workflow as _workflow

#: watch 自身状态文件（基线快照 / 心跳 / 已发事件）；仅 watch 写、status 读。
WATCH_PATH: Path = ORCH_STATE_ROOT / "watch.json"

#: 默认轮询间隔与超时（秒）。超时 4 h 覆盖一次完整 drain run（README §6）。
DEFAULT_INTERVAL_S: int = 60
DEFAULT_TIMEOUT_S: int = 4 * 3600

#: 轮询间隔硬上限——再长就会漏掉短命窗口的崩溃信号（用户裁决：≤60s）。
MAX_INTERVAL_S: int = 60
MIN_INTERVAL_S: int = 1

#: 心跳宽限：超过 interval × 此系数 + 余量即判 watch 已不在挂载（会话结束/被杀）。
HEARTBEAT_GRACE_FACTOR: int = 3
HEARTBEAT_GRACE_S: int = 60

#: 事件优先级（同刻多事件时取首个为主事件，其余以「同时」附注）。SALVAGE 排最后：
#: 它是「进度已保住、队列已自动接上」的知会，不要求组长动手，不该盖住真要处理的信号。
PRIORITY: tuple[str, ...] = (
    "CRASH",
    "QUOTA-STOP",
    "CHANNEL-STOP",
    "NEEDS-LEADER",
    # 用户投递大任务（REFACTOR.md §3.4）：**立即唤醒组长**（M3 最小扩展——只加事件名、检测
    # 函数与基线字段；m5 把它并入统一事件层，届时本支改为读事件流）。
    "TASKBOOK-NEW",
    "QUEUE-COMPLETE",
    "SALVAGE",
)

#: run 日志里的额度类全局停止痕迹（drain 在触发全局停止前打 ``[Q]`` 行）。日志判定
#: 早于 .done 落盘，能让 watch 在 drain 收尾的瞬间就唤醒，而不是等摘要写完。
QUOTA_RE = re.compile(r"quota_exhausted|额度耗尽|credit usage limit", re.IGNORECASE)

#: run 日志里的渠道级故障全局停止痕迹（drain 打 ``[C]`` 行并显式写出 kind，
#: backlog 20261012-194500-leader）。判据与 QUOTA 同构：**日志先于 .done**，早一拍唤醒。
CHANNEL_RE = re.compile(r"channel_error|渠道级故障|custom model service error", re.I)

#: 系统性故障族全表：(``.done`` 的 stopped 值, 快照身份字段, 事件名, 日志判据, 读数)。
#: 五列并置是有意为之——kind、字段、事件名、文案若分处五地各写一遍，加下一族时必漏一列
#: （本表新加的就是第二族）。事件名须都在 :data:`PRIORITY` 里（test_watch 钉住这一条）。
SYSTEMIC_FACES: tuple[tuple[str, str, str, re.Pattern[str], str], ...] = (
    ("quota_exhausted", "quota_run", "QUOTA-STOP", QUOTA_RE, "额度类失败"),
    ("channel_error", "channel_run", "CHANNEL-STOP", CHANNEL_RE, "渠道级故障"),
)

#: 快照字段 → (kind, 事件名)，供 :func:`detect` 逐族差分。
_SYSTEMIC_EVENTS: tuple[tuple[str, str, str], ...] = tuple(
    (field, kind, event) for kind, field, event, _rx, _read in SYSTEMIC_FACES
)

#: kind → 事件 detail 里的故障读数（两类停止动作相同、成因不同，须说得出差别）。
_SYSTEMIC_READINGS: dict[str, str] = {
    kind: read for kind, _f, _e, _rx, read in SYSTEMIC_FACES
}

#: 以 run 名为身份、须受 ``hold`` 保护的字段。系统性族一支由 :data:`SYSTEMIC_FACES`
#: 派生——加第三族时若还手写在下面这张表里，抑制帧就会把新事件的身份抹掉（同一 run
#: 重复报 CHANNEL-STOP，或反过来在瞬态空帧里丢掉停止事实）。
_HOLD_RUN_FIELDS: tuple[str, ...] = (
    "crash_run",
    *(field for _kind, field, _event, _rx, _read in SYSTEMIC_FACES),
)

#: 完成事件的身份三元组字段：任一变化即「新完成事实」——同 run 的 ``.done`` 被重写
#: （配额/协作停止后续跑完）也能再报（backlog 20261011-220622-leader 要求 1）。
DONE_IDENTITY_FIELDS: tuple[str, ...] = ("done", "done_completed", "done_count")

#: 黑匣子环形缓冲帧数。命中/超时之外，脱靶复盘只需要「最近几帧各自看见了什么」。
RECENT_MAX: int = 10

#: ``fired`` 审计留痕保留条数。
FIRED_KEEP: int = 20


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _age_s(iso: Any) -> float:
    """距某个 ISO 时刻的秒数；不可解析时返回 -1（视为未知）。"""
    try:
        dt = datetime.fromisoformat(str(iso))
    except (TypeError, ValueError):
        return -1.0
    return (datetime.now().astimezone() - dt).total_seconds()


def _stem(path_like: Any) -> str:
    """run 日志/摘要串 → 文件名 stem（``…/devops-runs/20261010-160413.log`` → 时间戳名）。"""
    s = str(path_like or "")
    return Path(s).stem if s else ""


# ---------------------------------------------------------------------------
# 快照：drain 观察面的单帧全貌（全部字段皆代码判定，无 LLM 参与）
# ---------------------------------------------------------------------------
def _systemic_runs(
    lock_run: str, done_name: str, done: dict[str, Any] | None
) -> dict[str, str]:
    """各类系统性故障发生在哪个 run（未发生的 kind 缺席）。

    两路取一：``.done`` 的 ``stopped=<kind>``（收尾后），或 run 日志现该族痕迹（收尾前）
    ——后者让 watch 早一拍唤醒。一次扫日志把两族都判掉，别为第二类再开一遍文件。
    """
    found: dict[str, str] = {}
    texts: dict[str, str] = {}
    for kind, _field, _event, rx, _read in SYSTEMIC_FACES:
        if done and done.get("stopped") == kind:
            found[kind] = done_name
            continue
        for name in (lock_run, done_name):
            if not name:
                continue
            if name not in texts:
                try:
                    texts[name] = (_drain.DEVOPS_RUNS_DIR / f"{name}.log").read_text(
                        encoding="utf-8", errors="replace"
                    )
                except OSError:
                    texts[name] = ""
            if rx.search(texts[name]):
                found[kind] = name
                break
    return found


def snapshot() -> dict[str, Any]:
    """当前 drain 观察面快照（六类事件的判据原料）。

    字段全部为「身份 + 状态」，供 :func:`detect` 做**基线差分**：崩溃与系统性故障以
    run 名为身份（同一 run 不重复报，新 run 再报），完成以 (名, completed, count)
    三元组为身份，needs_leader 与抢救续作分别以条目 id 集合为身份。
    """
    lock = _drain.lock_status()
    done = _drain.latest_done()
    crashed = lock["state"] == "stale" and bool(lock.get("crashed"))
    lock_run = _stem(lock.get("run_log", ""))
    done_name = str((done or {}).get("name", ""))
    needs = sorted(
        {
            str(i.get("id", ""))
            for i in _workflow.backlog_by_status("needs_leader")
            if i.get("id")
        }
    )
    salvaged = sorted(
        {str(i.get("id", "")) for i in _workflow.backlog_salvage() if i.get("id")}
    )
    systemic = _systemic_runs(lock_run, done_name, done)
    return {
        "ts": _now(),
        "lock_state": lock["state"],
        "lock_pid": int(lock.get("pid", 0) or 0),
        "lock_run": lock_run,
        "lock_done": _lock_done_key(lock_run),
        "crash_run": lock_run if crashed else "",
        "done": done_name,
        "done_stopped": str((done or {}).get("stopped", "") or ""),
        "done_completed": str((done or {}).get("completed", "") or ""),
        "done_count": int((done or {}).get("count", 0) or 0),
        "done_elapsed_s": (done or {}).get("elapsed_s", "?"),
        "needs_leader": needs,
        "taskbook_ids": _taskbook_ids(),
        "salvage": salvaged,
        "pending": len(_workflow.backlog_pending()),
        "quota_run": systemic.get("quota_exhausted", ""),
        "channel_run": systemic.get("channel_error", ""),
    }


#: ``.done`` 的 ``stopped`` 字段 → 完成方式的人类读数。
_HOW_DONE = {
    "cooperative": "协作停止",
    "quota_exhausted": "额度耗尽全局停止",
    "channel_error": "渠道级故障全局停止",
}


def _complete_event(
    name: str, data: dict[str, Any], pending: int, *, via: str = ""
) -> dict[str, Any]:
    """QUEUE-COMPLETE 事件体——``data`` 为该 run 的 ``.done`` 读数（或快照同名字段）。"""
    how = _HOW_DONE.get(str(data.get("stopped", "") or ""), "自然跑完")
    detail = f"处理 {int(data.get('count', 0) or 0)} 项 / {data.get('elapsed_s', '?')}s，{how}"
    return {
        "event": "QUEUE-COMPLETE",
        "subject": name,
        "detail": f"{detail}（{via}）" if via else detail,
        "action": f"剩余 pending={pending}；orch status 看交付明细与台账",
    }


def _complete_event_from_key(
    key: str, cur: dict[str, Any], *, via: str
) -> dict[str, Any]:
    """按完成身份三元组串补报（转移帧：该 run 的 ``.done`` 没占住 latest_done 名额）。"""
    name = key.split("|", 1)[0]
    return _complete_event(
        name, _done_file(name), int(cur.get("pending", 0) or 0), via=via
    )


def _done_file(name: str) -> dict[str, Any]:
    """按 run 名直读 ``devops-runs/<name>.done``（缺席/半写/损坏 → ``{}``）。

    与 :func:`_drain.latest_done` 的区别是**点名查询**：锁自称在跑的那个 run 是否已经
    留下收尾摘要，只能这么查——它是拆穿「锁还活着」这个谎的唯一硬证据。
    """
    if not name:
        return {}
    try:
        data = json.loads(
            (_drain.DEVOPS_RUNS_DIR / f"{name}.done").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _key(name: Any, completed: Any, count: Any) -> str:
    """完成身份三元组串（无名 → 空串＝无完成事实）。"""
    n = str(name or "")
    return (
        "" if not n else f"{n}|{completed or ''}|{count if count is not None else ''}"
    )


def _done_key(s: dict[str, Any]) -> str:
    """快照/基线里的完成身份（:data:`DONE_IDENTITY_FIELDS` 三字段的读数）。"""
    return _key(s.get("done"), s.get("done_completed"), s.get("done_count"))


def _lock_done_key(name: str) -> str:
    """锁自称在跑的那个 run 的收尾身份；无 ``.done`` 时为空串（= 真的还在跑）。"""
    data = _done_file(name)
    return "" if not data else _key(name, data.get("completed"), data.get("count"))


def _in_progress(s: dict[str, Any]) -> bool:
    """锁自称正在跑「那个 run」**且该 run 确实还没留下收尾摘要**，才叫进行中。

    ``.done`` 由 drain 在 ``release_lock`` 之后写出，所以「锁还在、摘要已存在」只能是
    释放失败残留 + 判活误真（pid 被 Windows 回收 / tasklist 超时保守判活）——210557 那场
    QUEUE-COMPLETE 就是被这种假「进行中」抑制后又被推进基线而永久吞掉的。

    锁面自 20261011-231237-devops 起把「该 run 已有 ``.done``」直接判为非活，故本支只
    可能在两种情况下为真：旧代码写下的锁/基线仍在场，或 ``.done`` 半写到读不出证据。
    保留它 = 观察面对锁面的不信任仍是防线，不是冗余。
    """
    return (
        s.get("lock_state") == "running"
        and bool(s.get("lock_run"))
        and s.get("lock_run") == s.get("done")
        and not s.get("lock_done")
    )


def _taskbook_ids() -> list[str]:
    """``data/taskbooks/`` 现有任务书 id（TASKBOOK-NEW 的**检测函数**；§3.4 M3 最小扩展）。

    增量判据与 NEEDS-LEADER 同构：新 id（不在基线里）即「新落盘的任务书」→ 立即唤醒。
    基线字段 = 快照的 ``taskbook_ids``（记进 ``state/watch.json``）。m5 收编进统一事件层。
    """
    from . import taskbooks as _tb  # 延迟导入：taskbooks 是叶子数据层

    try:
        return [str(d.get("id", "")) for d in _tb.list_all() if d.get("id")]
    except Exception:  # noqa: BLE001 - 观察面绝不阻塞生产（§0.6）
        return []


def detect(
    base: dict[str, Any], cur: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """基线 → 当前 的差分判定。返回 ``(events, verdict)``。

    ``verdict = {"hold": set[str], "why": str}``——``hold`` 是本帧**不得**并入基线的
    身份组（由 :func:`advance_base` 执行），``why`` 是抑制/无事的原因，进黑匣子留痕。

    判定规则（每条都是「新出现」而非「当前存在」——既存状态是基线的一部分，不重复
    打扰组长）：

    - **CRASH**：锁持有者 pid 已死且该 run 无 ``.done``，且该 run 未报过。
    - **QUOTA-STOP**：额度类全局停止出现在某个尚未报过的 run 上。
    - **CHANNEL-STOP**：渠道级故障全局停止，判据与上一条同族（:data:`SYSTEMIC_FACES`）。
    - **NEEDS-LEADER**：backlog 里新出现的 needs_leader 条目（逐个成事件）。
    - **QUEUE-COMPLETE**：完成身份三元组（名, completed, count）变化，且该 run 不是
      「锁自称在跑且尚无 ``.done``」的真进行中 run。**收尾事实优先于锁的自称**。
    - **转移帧兜底**（要求 2）：基线里那个自称在跑的 run 已留下 ``.done``，而本帧锁已
      不在它身上、其身份又从未报出（``.done`` 名额被更晚的 run 占了）→ 照报。锁面采纳
      收尾否决后新基线不会再带 ``lock_state=running`` + ``lock_done`` 的矛盾帧，此支转为
      护升级前旧基线的保险。

    瞬态空值（``.done`` 半写、latest_done 读不到）与抑制帧都走 ``hold``：身份保持
    基线原值，下一帧继续比较——这是 210557 那场「 fired=[] 且补账同哑」的反制。
    """
    out: list[dict[str, Any]] = []
    hold: set[str] = set()
    why = ""

    crash = str(cur.get("crash_run", "") or "")
    if crash:
        if crash != base.get("crash_run", ""):
            out.append(
                {
                    "event": "CRASH",
                    "subject": crash,
                    "detail": f"pid {cur.get('lock_pid', 0)} 已死且该 run 无 .done",
                    "action": (
                        "worker 被中途杀死：下次 approve/_drain-devops 自动接管并回收孤儿条目"
                    ),
                }
            )
    else:
        hold.add("crash_run")

    for field, kind, event in _SYSTEMIC_EVENTS:
        hit = str(cur.get(field, "") or "")
        if hit:
            if hit != base.get(field, ""):
                out.append(
                    {
                        "event": event,
                        "subject": hit,
                        "detail": f"{_SYSTEMIC_READINGS[kind]}触发全局停止",
                        "action": "条目均保持 pending，待渠道恢复后 approve/drain 续消化",
                    }
                )
        else:
            hold.add(field)

    seen = set(base.get("needs_leader") or [])
    for bid in cur.get("needs_leader") or []:
        if bid not in seen:
            out.append(
                {
                    "event": "NEEDS-LEADER",
                    "subject": bid,
                    "detail": "drain 判定该条目须组长处理（已跳过，不阻塞队列）",
                    "action": "查 state/backlog.json 的 note 后重派 / 改任务书 / 上报用户",
                }
            )

    # 用户投递的大任务（REFACTOR.md §3.4，M3 最小扩展）：新 id 落盘即**立即唤醒**组长。
    seen_tb = set(base.get("taskbook_ids") or [])
    for tid in cur.get("taskbook_ids") or []:
        if tid not in seen_tb:
            out.append(
                {
                    "event": "TASKBOOK-NEW",
                    "subject": tid,
                    "detail": "用户投递了新任务书（data/taskbooks/，落盘即唤醒）",
                    "action": (
                        "orch taskbooks list 查看 → taskbooks link <id> --plan <pid> "
                        "拆解推进（用户 accept/reject 只经 TUI）"
                    ),
                }
            )

    # 抢救知会（backlog 20261012-234000-leader）：上次尝试的未收尾进度已被机器保住并
    # 自动接上了——这是「不必动手」的知会，报一次即可，队列自己在续完。
    seen_salvage = set(base.get("salvage") or [])
    for bid in cur.get("salvage") or []:
        if bid not in seen_salvage:
            out.append(
                {
                    "event": "SALVAGE",
                    "subject": bid,
                    "detail": "上次尝试的未收尾进度已抢救为提交，续作已自动接上（零手动）",
                    "action": "无需处置；想核实现场：orch status 的抢救段 / state/backlog.json",
                }
            )

    key = _done_key(cur)
    base_key = _done_key(base)
    if not key:
        hold.add("done")
        why = "no-done"
    elif key != base_key:
        if _in_progress(cur):
            hold.add("done")
            why = "lock-claims-running"
        else:
            out.append(
                _complete_event(
                    str(cur.get("done", "")),
                    {
                        "count": cur.get("done_count", 0),
                        "elapsed_s": cur.get("done_elapsed_s", "?"),
                        "stopped": cur.get("done_stopped", ""),
                    },
                    int(cur.get("pending", 0) or 0),
                )
            )
    if (
        base.get("lock_state") == "running"
        and base.get("lock_done")
        and cur.get("lock_run") != base.get("lock_run")
        and base.get("lock_done") not in (base_key, key)
    ):
        out.append(
            _complete_event_from_key(str(base["lock_done"]), cur, via="锁释放转移帧")
        )

    order = {name: i for i, name in enumerate(PRIORITY)}
    return sorted(out, key=lambda e: order.get(e["event"], len(order))), {
        "hold": hold,
        "why": why,
    }


def detect_events(base: dict[str, Any], cur: dict[str, Any]) -> list[dict[str, Any]]:
    """只要事件列表的便捷面（不关心基线推进时用）；判定全在 :func:`detect`。"""
    return detect(base, cur)[0]


def advance_base(
    base: dict[str, Any], cur: dict[str, Any], hold: Any = ()
) -> dict[str, Any]:
    """把当前帧并入基线，但 ``hold`` 里的身份组保持基线原值。

    抑制帧/瞬态空值若照样推进基线，被吞掉的完成事实就再不参与比较——watch 与 ``orch
    status`` 补账会同时哑掉（backlog 20261011-220622-leader 要求 3）。
    """
    groups = hold if isinstance(hold, (set, frozenset)) else set(hold or ())
    nxt = dict(cur)
    if "done" in groups:
        for field in DONE_IDENTITY_FIELDS:
            nxt[field] = base.get(field, "" if field != "done_count" else 0)
    for field in _HOLD_RUN_FIELDS:
        if field in groups:
            nxt[field] = base.get(field, "")
    return nxt


def format_event(ev: dict[str, Any]) -> str:
    """事件的**单行**呈现（watch 命中时首行即此，供组长一眼判读）。"""
    return f"{ev['event']} {ev['subject']} —— {ev['detail']}｜建议处置：{ev['action']}"


# ---------------------------------------------------------------------------
# watch.json：基线 / 心跳 / 已发事件
# ---------------------------------------------------------------------------
def load_state() -> dict[str, Any]:
    """读 watch 状态；不存在或损坏返回 ``{}``（fail-open，status 会重建基线）。"""
    if not WATCH_PATH.exists():
        return {}
    try:
        data = json.loads(WATCH_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def save_state(data: dict[str, Any]) -> None:
    """落 watch 状态（可读 JSON）。写失败不抛（观察面塌不了主链路）。"""
    try:
        WATCH_PATH.parent.mkdir(parents=True, exist_ok=True)
        WATCH_PATH.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    except OSError:
        pass


# ---------------------------------------------------------------------------
# 黑匣子：每帧现场 + 退出原因（脱靶复盘的唯一线索来源）
# ---------------------------------------------------------------------------
#: 等待器没来得及写退出原因（被外部杀死 / 会话连坐崩溃）时补上的记号。
KILLED = "killed（无退出记录）"


def _frame(cur: dict[str, Any], why: str = "") -> dict[str, Any]:
    """黑匣子一帧：判读脱靶所需的最小现场（锁自称 + 完成身份 + 抑制原因）。"""
    return {
        "ts": cur.get("ts", ""),
        "lock_state": cur.get("lock_state", ""),
        "lock_run": cur.get("lock_run", ""),
        "done": _done_key(cur),
        "why": why,
    }


def fmt_frame(f: dict[str, Any]) -> str:
    """黑匣子帧的单行呈现（``orch status`` 判读用）。"""
    return (
        f"{str(f.get('ts', ''))[:19]} 锁={f.get('lock_state', '?')}"
        f"/{f.get('lock_run') or '-'} done={f.get('done') or '-'}"
        f"{' 抑制=' + str(f['why']) if f.get('why') else ''}"
    )


def push_recent(state: dict[str, Any], frame: dict[str, Any]) -> None:
    """帧进 ``recent`` 环形缓冲（只留最近 :data:`RECENT_MAX` 帧）。"""
    recent = list(state.get("recent") or [])
    recent.append(frame)
    state["recent"] = recent[-RECENT_MAX:]


def _legacy(prior: dict[str, Any]) -> dict[str, Any]:
    """把上一任等待器折成「遗言」块（:func:`arm` 继承——被杀死的那任没机会自己留话）。"""
    reason = str(prior.get("exit_reason", "") or "")
    if not reason:
        reason = KILLED if prior.get("heartbeat_at") else "unknown"
    block = {"armed_at": str(prior.get("armed_at", "") or ""), "exit_reason": reason}
    recent = prior.get("recent")
    if isinstance(recent, list) and recent:
        block["recent"] = recent[-RECENT_MAX:]
    return block


def last_words_lines(state: dict[str, Any] | None = None) -> list[str]:
    """上一任遗言：只有**非正常结束**才出声——脱靶复盘的入口，别在正常态刷屏。

    两个来源：本任状态被 :func:`reconcile` 追认成 ``KILLED``，或 :func:`arm` 从被杀
    的前任那里继承来的 ``prev`` 块。
    """
    st = state if state is not None else load_state()
    deceased: dict[str, Any] | None = None
    if st.get("exit_reason") == KILLED:
        deceased = st
    else:
        prev = st.get("prev")
        if isinstance(prev, dict) and prev.get("exit_reason") == KILLED:
            deceased = prev
    if deceased is None:
        return []
    armed = str(deceased.get("armed_at", ""))[:19] or "?"
    out = [
        f"  [!] 上一任等待器（起于 {armed}）没有退出记录——被外部杀死或连坐崩溃；"
        "黑匣子末帧："
    ]
    for frame in list(deceased.get("recent") or [])[-3:]:
        out.append(f"    · {fmt_frame(frame)}")
    return out


def unacked_fired(state: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """``fired`` 里尚未被 ``orch status`` 确认过的事件（水位线 :data:`fired_ack` 之上）。

    watch 命中即写 ``fired`` 然后退出——唤醒靠**后台任务完成通知**，而通知只随「顶层
    单命令挂载」这条通道存在。2026-10-12 夜航的形态是：挂载被 `&` 嵌进别的命令，
    事件命中、留痕完整，唤醒却永远没送出。故 ``fired`` 才是命中事实的持久面，水位线
    让它**至少被补报一次**而不是一直沉默、也不重复刷屏。

    ts 由 :func:`_now` 产出（同时区、秒精度 ISO 串），字典序即时间序；水位线缺失
    （旧版 watch.json 无 ``fired_ack`` 键）时全部视为未确认——宁可多报一次。
    """
    st = state if state is not None else load_state()
    ack = str(st.get("fired_ack", "") or "")
    return [e for e in (st.get("fired") or []) if str(e.get("ts", "")) > ack]


def ack_fired(events: list[dict[str, Any]], *, state: dict[str, Any] | None = None):
    """把水位线推到 ``events`` 末条并落盘（展示即确认，与补账的一次性清算同语义）。"""
    st = state if state is not None else load_state()
    if events:
        st["fired_ack"] = str(events[-1].get("ts", "") or _now())
        save_state(st)
    return events


def fired_lines() -> list[str]:
    """``orch status`` 的 fired 未确认明细段：**展示即确认**（就地推进水位线）。

    自读自写盘，调用方勿传自己的旧 state 字典——``reconcile()`` 会整体重写
    watch.json，拿旧副本回写会把刚推进的补账基线覆盖回去（水位与基线互相抹掉，
    正是本单要杜绝的「命中了但没人知道」的另一种复发形态）。
    """
    ev = unacked_fired()
    if not ev:
        return []
    out = [
        f"  [!] fired 未确认={len(ev)} 起（watch 已命中，唤醒未必送达；现补报明细）："
    ]
    out.extend(f"    · {format_event(e)}" for e in ev)
    ack_fired(ev)
    return out


def mounted(state: dict[str, Any] | None = None) -> bool:
    """watch 是否仍在挂载：心跳新鲜度在 interval×3+60 s 宽限内。

    watch 随组长会话消亡（会话结束/机器重启即死），故心跳是唯一可用的存活信号
    ——它不是「进程判活」（那需要 pid 与 shell），只是「最后一次轮询距今多久」。
    """
    st = state if state is not None else load_state()
    interval = int(st.get("interval_s", DEFAULT_INTERVAL_S) or DEFAULT_INTERVAL_S)
    age = _age_s(st.get("heartbeat_at", ""))
    if age < 0:
        return False
    return age <= interval * HEARTBEAT_GRACE_FACTOR + HEARTBEAT_GRACE_S


def arm(
    *,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    interval_s: int = DEFAULT_INTERVAL_S,
    heartbeat: bool = True,
) -> dict[str, Any]:
    """以当前快照为基线建立 watch 状态并落盘（``run_watch`` 启动即调用）。

    Args:
        timeout_s/interval_s: 记入状态的等待参数（status 呈现与心跳宽限用）。
        heartbeat: True 表示**真有等待器在跑**（run_watch 挂载）；False 用于 status
            首次建立基线——此时写新鲜心跳会把「没人等」谎报成「已挂载」。

    ``watch.json`` 是全局单例：新挂载会整体覆写，故先把上一任折进 ``prev``（遗言）。
    测试/诊断性挂载前须确认无 live 挂载，否则会覆写正在等待那任的状态文件。
    """
    cur = snapshot()
    now = _now()
    state = {
        "version": 1,
        "armed_at": now,
        "heartbeat_at": now if heartbeat else "",
        "timeout_s": timeout_s,
        "interval_s": interval_s,
        "polls": 0,
        "base": cur,
        "fired": [],
        "fired_ack": "",
        "recent": [],
    }
    prior = load_state()
    if prior.get("armed_at"):
        state["prev"] = _legacy(prior)
        # 未确认的命中跨任继承：watch.json 是全局单例、新挂载整体覆写，若前任的
        # fired 从未被 status 呈现过（嵌入式挂载丢唤醒的夜航形态），这些账必须活过
        # 覆写——否则「命中了但没人知道」会被下一次挂载永久抹掉。
        carry = unacked_fired(prior)
        if carry:
            state["fired"] = carry[-FIRED_KEEP:]
            state["fired_ack"] = str(prior.get("fired_ack", "") or "")
    save_state(state)
    return state


# ---------------------------------------------------------------------------
# 等待已自动化：不再催挂（手动挂载降为兜底，见 README §4.5 / references/drain-watch.md）
# ---------------------------------------------------------------------------
def watch_argv(*, timeout_s: int = DEFAULT_TIMEOUT_S) -> str:
    """手动**兜底**要挂的命令行串（裸名优先，README §11 CLI 范式）；自动挂载不经此串。"""
    return f"pysci-orch watch --timeout {timeout_s}"


def autowatch_lines(*, indent: str = "") -> list[str]:
    """「等待已自动化」提示行：交办/唤醒当刻由代码回报——组长**无需手动挂载**。

    自动挂载（README §4.5 v2：``Stop + asyncRewake`` 钩子每回合接管）之后，这里不再给
    挂载命令、也不再列事件名（那些属**手动兜底**，见技能真本 / ``references/drain-watch.md``）；
    只回报「已自动挂载」，免得组长照旧提示重复挂一条。
    """
    return [
        f"{indent}watch 等待已自动化（Stop 钩子每回合自动挂载，无需手动）；"
        "钩子不可用时照 README §4.5 手动兜底。"
    ]


def auto_remount_lines(*, indent: str = "") -> list[str]:
    """watch 命中/超时退出后的提示：下一空闲回合由钩子自动重挂，无需手动。"""
    return [f"{indent}（watch 已退；下一空闲回合由 Stop 钩子自动重挂，无需手动。）"]


# ---------------------------------------------------------------------------
# 阻塞等待
# ---------------------------------------------------------------------------
def clamp_interval(interval_s: int | None) -> int:
    """轮询间隔钳制到 [MIN, MAX]（None/非法 → 默认；超上限会漏短命崩溃窗口）。"""
    try:
        n = int(DEFAULT_INTERVAL_S if interval_s is None else interval_s)
    except (TypeError, ValueError):
        return DEFAULT_INTERVAL_S
    if n <= 0:
        return DEFAULT_INTERVAL_S
    return max(MIN_INTERVAL_S, min(n, MAX_INTERVAL_S))


def _touch(
    state: dict[str, Any],
    cur: dict[str, Any],
    base: dict[str, Any],
    verdict: dict[str, Any],
) -> None:
    """本帧落进状态：按 ``hold`` 约束推进基线 + 刷新心跳 + 轮次 + 黑匣子帧。"""
    state["base"] = base
    state["heartbeat_at"] = _now()
    state["polls"] = int(state.get("polls", 0)) + 1
    push_recent(state, _frame(cur, str(verdict.get("why", "") or "")))


def _emit(state: dict[str, Any], events: list[dict[str, Any]], echo: Any) -> None:
    """命中收场：``fired`` 审计留痕（最近 :data:`FIRED_KEEP` 条）+ ``exit_reason`` + 单行呈现。

    ``fired`` 只追加、不置水位线：本进程打出的单行**可能**永远到不了组长会话
    （挂载通道被 `&` 嵌死时的夜航形态），确认权交给 ``orch status`` 的展示动作。
    """
    fired = list(state.get("fired", []))
    for ev in events:
        fired.append({"ts": _now(), **ev})
    state["fired"] = fired[-FIRED_KEEP:]
    state["last_exit"] = "event"
    state["exit_reason"] = "event:" + ",".join(str(e["event"]) for e in events)
    save_state(state)
    echo(format_event(events[0]))
    for ev in events[1:]:
        echo(f"  同时：{format_event(ev)}")
    for line in auto_remount_lines():
        echo(line)


def run_watch(
    *,
    timeout_s: int | None = None,
    interval_s: int | None = None,
    sleep: Any = time.sleep,
    monotonic: Any = time.monotonic,
    echo: Any = print,
) -> int:
    """阻塞轮询直到命中事件或超时；无论命中或超时均返回 0（非零会被误读为故障）。

    Args:
        timeout_s: 总等待秒数（默认 4 h；``<=0`` 视作立即超时判定一轮）。
        interval_s: 轮询间隔，钳制到 :data:`MIN_INTERVAL_S`–:data:`MAX_INTERVAL_S`。
        sleep/monotonic: 注入口，测试用假时钟免真等。
        echo: 输出注入口。

    每轮按 :func:`advance_base` 把当前快照并入 ``base``（受 :func:`detect` 的 ``hold``
    约束：抑制帧与瞬态空值**不推进**，被压住的完成事实下一帧仍在比较），同时刷新心跳
    （status 判挂载用）并把该帧现场写进 ``recent`` 黑匣子。
    """
    interval = clamp_interval(interval_s)
    total = DEFAULT_TIMEOUT_S if timeout_s is None else int(timeout_s)
    state = arm(timeout_s=total, interval_s=interval)
    prev = state["base"]
    deadline = monotonic() + max(0, total)

    while True:
        left = deadline - monotonic()
        if left <= 0:
            break
        sleep(min(interval, left))
        cur = snapshot()
        events, verdict = detect(prev, cur)
        prev = advance_base(prev, cur, verdict["hold"])
        _touch(state, cur, prev, verdict)
        if events:
            _emit(state, events, echo)
            return 0
        save_state(state)

    cur = snapshot()
    events, verdict = detect(prev, cur)
    prev = advance_base(prev, cur, verdict["hold"])
    _touch(state, cur, prev, verdict)
    if events:  # 最后一拍才落地的收尾也照报——TIMEOUT 不该是「没看见」的托词
        _emit(state, events, echo)
        return 0
    state["last_exit"] = "timeout"
    state["exit_reason"] = "timeout"
    save_state(state)
    echo(
        f"TIMEOUT {_human(total)} 内无事件"
        f"（锁态={cur.get('lock_state', '?')}，pending={cur.get('pending', 0)}，"
        f"needs_leader={len(cur.get('needs_leader') or [])}）"
    )
    for line in auto_remount_lines():
        echo(line)
    return 0


def _human(seconds: int) -> str:
    """秒数的人类读数（超时行的时长描述）。"""
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m" if m else f"{h}h"
    return f"{m}m{s:02d}s" if m else f"{s}s"


# ---------------------------------------------------------------------------
# orch status 补账（一次性清算：报完即推进基线，幂等）
# ---------------------------------------------------------------------------
def reconcile() -> tuple[list[dict[str, Any]], bool]:
    """补账自上次基线以来漏掉的事件。

    Returns:
        ``(events, settled)``。``settled=True`` 表示本次已把基线推进到当前快照
        （事件只补一次，重复运行 status 不会刷屏）；watch 仍挂载时返回
        ``([], False)``——事件归 watch 的原生通知负责，此处不抢账、不动基线。
    """
    state = load_state()
    base = state.get("base")
    if not isinstance(base, dict) or not base:
        arm(
            timeout_s=DEFAULT_TIMEOUT_S,
            interval_s=clamp_interval(None),
            heartbeat=False,
        )
        return [], False
    if mounted(state):
        return [], False
    if (
        state.get("heartbeat_at")
        and not state.get("exit_reason")
        and not state.get("last_exit")
    ):
        # 有心跳却无退出记录 = 被外部杀死：追认遗言，供 status 读黑匣子
        state["exit_reason"] = KILLED
    cur = snapshot()
    events, verdict = detect(base, cur)
    state["base"] = advance_base(base, cur, verdict["hold"])
    push_recent(state, _frame(cur, str(verdict.get("why", "") or "")))
    state["heartbeat_at"] = ""
    state["reconciled_at"] = _now()
    state["settled"] = events
    save_state(state)
    return events, True


def status_lines() -> list[str]:
    """``orch status`` 的 watch 段文案（挂载态 + 补账 + 上一任遗言）。"""
    state = load_state()
    if not isinstance(state.get("base"), dict) or not state.get("base"):
        arm(
            timeout_s=DEFAULT_TIMEOUT_S,
            interval_s=clamp_interval(None),
            heartbeat=False,
        )
        return [
            "  （首次运行：已按当前状态建立补账基线——此后 drain 期间的事件要么由挂载的"
            "watch 实时报，要么下次 orch status 补账）"
        ]
    if mounted(state):
        interval = int(
            state.get("interval_s", DEFAULT_INTERVAL_S) or DEFAULT_INTERVAL_S
        )
        age = _age_s(state.get("heartbeat_at", ""))
        if state.get("last_exit"):
            # 刚命中/刚超时退出：心跳仍新鲜，但等待器已经不在了——别把「刚结束」读成「在等」
            return [
                f"  ⊘ 上次运行已结束（{state['last_exit']}，心跳 {age:.0f}s 前）——"
                f"继续等待由 Stop 钩子自动重挂（无需手动；兜底命令 {watch_argv()}）"
            ] + fired_lines()
        return [
            f"  ▶ 已挂载（心跳 {age:.0f}s 前，每 {interval}s 轮询，已跑 "
            f"{state.get('polls', 0)} 轮）——命中将以原生后台通知送达"
        ]
    events, _ = reconcile()
    armed = str(state.get("armed_at", ""))[:19] or "?"
    out: list[str] = []
    if state.get("last_exit"):
        out.append(f"  ⊘ 未挂载（上次运行至 {state['last_exit']}，起于 {armed}）")
    else:
        out.append(f"  ⊘ 未挂载（基线 {armed}——组长会话结束后无人观察）")
    out.extend(last_words_lines())  # reconcile 可能刚追认「被杀死」，重读磁盘为准
    if events:
        out.append(f"  [!] 补账 {len(events)} 起事件：")
        for ev in events:
            out.append(f"    · {format_event(ev)}")
    else:
        out.append("  （自上次基线以来无新事件）")
    out.extend(fired_lines())  # reconcile 之后：水位与基线各写各的，互不覆盖
    if _workflow.backlog_pending() or _drain.lock_status()["state"] == "running":
        out.append("  等待 drain：watch 由 Stop 钩子自动挂载（无需手动；退出即唤醒）")
    return out


def cmd_watch(args: Any) -> int:
    """``orch watch``：阻塞等待 drain 事件（手动**兜底**用；常态由 Stop 钩子自动挂载）。"""
    return run_watch(
        timeout_s=getattr(args, "timeout", None),
        interval_s=getattr(args, "interval", None),
    )


__all__ = [
    "DONE_IDENTITY_FIELDS",
    "FIRED_KEEP",
    "KILLED",
    "RECENT_MAX",
    "WATCH_PATH",
    "ack_fired",
    "advance_base",
    "arm",
    "cmd_watch",
    "detect",
    "detect_events",
    "fired_lines",
    "fmt_frame",
    "format_event",
    "last_words_lines",
    "load_state",
    "autowatch_lines",
    "push_recent",
    "reconcile",
    "auto_remount_lines",
    "run_watch",
    "snapshot",
    "status_lines",
    "unacked_fired",
    "watch_argv",
]
