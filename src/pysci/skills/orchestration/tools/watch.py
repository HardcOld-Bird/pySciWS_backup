"""orch watch 事件等待器：把 drain 观察闭环从「组长记得去查」硬化成代码（README §4.5/§6）。

理念（用户裁决 2026-10-10：能代码硬保证的不靠 LLM 纪律）：后台 drain worker 是**分离
进程**，组长派发完就没事了，体系此前只能靠组长自觉轮询 ``orch status`` 才知道队列跑完 /
崩了 / 卡在额度上——这正是「靠纪律」的软点。结构上限在于**唤醒注册只能由组长会话的
工具发起**（外部进程无法向 TUI 注入通知），故硬化到能硬的那一层：

- **事件判定全在代码**（本模块）：读状态文件 + Python 侧 ``_pid_alive``（不起 shell，
  故 MSYS 路径转换问题免疫），四类事件——CRASH / QUOTA-STOP / NEEDS-LEADER /
  QUEUE-COMPLETE。
- **挂载提示由代码在唤醒当刻注入**：``wake_devops`` / ``approve`` / ``--from-backlog``
  的输出固定追加 :func:`mount_lines`，组长在需要等待的那一刻被明确告知挂什么命令。

残余软点（已知且接受）：组长是否照 ``[NEXT]`` 真去后台起一条 ``orch watch``。弃
Monitor 工具方案（会话侧监视器注册更重、且同样需要组长发起）。

用法：组长以 Bash ``run_in_background`` 启动 ``pysci-orch watch``——进程阻塞轮询
（默认 60 s，硬上限 60 s）直到命中事件或超时，打印**单行**事件后退出；退出即原生后台
任务完成通知唤醒组长。watch 随组长会话消亡，**不影响 detached drain**；漏挂 / 会话
中断期间的账由下次 ``orch status`` 经 :func:`reconcile` 补账（一次性清算并推进基线）。

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

#: 事件优先级（同刻多事件时取首个为主事件，其余以「同时」附注）。
PRIORITY: tuple[str, ...] = ("CRASH", "QUOTA-STOP", "NEEDS-LEADER", "QUEUE-COMPLETE")

#: run 日志里的额度类全局停止痕迹（drain 在触发全局停止前打 ``[Q]`` 行）。日志判定
#: 早于 .done 落盘，能让 watch 在 drain 收尾的瞬间就唤醒，而不是等摘要写完。
QUOTA_RE = re.compile(r"quota_exhausted|额度耗尽|credit usage limit", re.IGNORECASE)

#: 完成事件的身份三元组字段：任一变化即「新完成事实」——同 run 的 ``.done`` 被重写
#: （配额/协作停止后续跑完）也能再报（backlog 20261011-220622-leader 要求 1）。
DONE_IDENTITY_FIELDS: tuple[str, ...] = ("done", "done_completed", "done_count")

#: 黑匣子环形缓冲帧数。命中/超时之外，脱靶复盘只需要「最近几帧各自看见了什么」。
RECENT_MAX: int = 10


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
def _quota_run(lock_run: str, done_name: str, done: dict[str, Any] | None) -> str:
    """额度类全局停止发生在哪个 run（无则空串）。

    两路取一：``.done`` 的 ``stopped=quota_exhausted``（收尾后），或 run 日志现
    :data:`QUOTA_RE` 痕迹（收尾前）——后者让 watch 早一拍唤醒。
    """
    if done and done.get("stopped") == "quota_exhausted":
        return done_name
    for name in (lock_run, done_name):
        if not name:
            continue
        log = _drain.DEVOPS_RUNS_DIR / f"{name}.log"
        try:
            if QUOTA_RE.search(log.read_text(encoding="utf-8", errors="replace")):
                return name
        except OSError:
            continue
    return ""


def snapshot() -> dict[str, Any]:
    """当前 drain 观察面快照（四类事件的判据原料）。

    字段全部为「身份 + 状态」，供 :func:`detect` 做**基线差分**：崩溃/额度以
    run 名为身份（同一 run 不重复报，新 run 再报），完成以 (名, completed, count)
    三元组为身份，needs_leader 以条目 id 集合为身份。
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
        "pending": len(_workflow.backlog_pending()),
        "quota_run": _quota_run(lock_run, done_name, done),
    }


#: ``.done`` 的 ``stopped`` 字段 → 完成方式的人类读数。
_HOW_DONE = {"cooperative": "协作停止", "quota_exhausted": "额度耗尽全局停止"}


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

    quota = str(cur.get("quota_run", "") or "")
    if quota:
        if quota != base.get("quota_run", ""):
            out.append(
                {
                    "event": "QUOTA-STOP",
                    "subject": quota,
                    "detail": "额度类失败触发全局停止",
                    "action": "条目均保持 pending，待渠道恢复后 approve/drain 续消化",
                }
            )
    else:
        hold.add("quota_run")

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
    for group, field in (("crash_run", "crash_run"), ("quota_run", "quota_run")):
        if group in groups:
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
        "recent": [],
    }
    prior = load_state()
    if prior.get("armed_at"):
        state["prev"] = _legacy(prior)
    save_state(state)
    return state


# ---------------------------------------------------------------------------
# 挂载指令文案（[NEXT] 注入与 watch 自述共用，防文案漂移）
# ---------------------------------------------------------------------------
def watch_argv(*, timeout_s: int = DEFAULT_TIMEOUT_S) -> str:
    """组长要挂的命令行串（裸名优先，README §11 CLI 范式）。"""
    return f"pysci-orch watch --timeout {timeout_s}"


def mount_lines(*, indent: str = "", header: bool = True) -> list[str]:
    """固定挂载指令行：工作交办/唤醒当刻由代码告知「该挂什么」，不靠组长记文档。

    Args:
        indent: 每行前缀（并入他方 [NEXT] 段时缩进用）。
        header: True 时首行自带 ``[NEXT]`` 标签（自成一段）；False 时作既有段落的
            续行（``orch dispatch`` 已自行打过标签，勿重复）。
    """
    first = (
        "[NEXT] 挂载事件等待器（Bash run_in_background 启动，退出即以原生通知唤醒"
        "本会话，等待期零 token）："
        if header
        else "挂载事件等待器（Bash run_in_background，退出即以原生通知唤醒本会话）："
    )
    lines = [
        first,
        f"  {watch_argv()}",
        "  命中单行退出：CRASH / QUOTA-STOP / NEEDS-LEADER <id> / QUEUE-COMPLETE；"
        "超时无事件打印 TIMEOUT。",
        "  watch 随本会话消亡、不影响后台 drain；漏挂由下次 orch status 补账。",
    ]
    return [f"{indent}{line}" for line in lines]


def remount_lines(*, indent: str = "") -> list[str]:
    """命中/超时后的重挂提示（一次 watch 只报一批事件即退，后续须按需再挂）。"""
    return [
        f"{indent}如需继续等待下一批事件，重挂：{watch_argv()}",
        f"{indent}（不挂也行：状态已落盘，下次 orch status 会补账。）",
    ]


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
    """命中收场：``fired`` 审计留痕（最近 20 条）+ ``exit_reason`` + 单行呈现 + 重挂提示。"""
    fired = list(state.get("fired", []))
    for ev in events:
        fired.append({"ts": _now(), **ev})
    state["fired"] = fired[-20:]
    state["last_exit"] = "event"
    state["exit_reason"] = "event:" + ",".join(str(e["event"]) for e in events)
    save_state(state)
    echo(format_event(events[0]))
    for ev in events[1:]:
        echo(f"  同时：{format_event(ev)}")
    for line in remount_lines():
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
    for line in remount_lines():
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
    """``orch status`` 的 watch 段文案（挂载态 + 补账 + 上一任遗言 + 挂载指令）。"""
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
                f"继续等待请重挂：{watch_argv()}"
            ]
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
    if _workflow.backlog_pending() or _drain.lock_status()["state"] == "running":
        out.append("  等待 drain：后台运行 pysci-orch watch（退出即唤醒）")
    return out


def cmd_watch(args: Any) -> int:
    """``orch watch``：阻塞等待 drain 事件（组长经 Bash run_in_background 启动）。"""
    return run_watch(
        timeout_s=getattr(args, "timeout", None),
        interval_s=getattr(args, "interval", None),
    )


__all__ = [
    "DONE_IDENTITY_FIELDS",
    "KILLED",
    "RECENT_MAX",
    "WATCH_PATH",
    "advance_base",
    "arm",
    "cmd_watch",
    "detect",
    "detect_events",
    "fmt_frame",
    "format_event",
    "last_words_lines",
    "load_state",
    "mount_lines",
    "push_recent",
    "reconcile",
    "remount_lines",
    "run_watch",
    "snapshot",
    "status_lines",
    "watch_argv",
]
