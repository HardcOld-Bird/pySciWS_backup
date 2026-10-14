"""边界避让派发（种子 backlog 20261013-144500-leader，用户裁决 2026-10-13）

席链随时间走（路由 v3 谷时隐性 DeepSeek），卡在峰谷切换点前几秒出发＝整跳落在
**将要过期**的窗口里（成本错配）；而等待期零 LLM 成本，5 分钟很值。

实现 ``dispatch._boundary_hold`` 在**解析角色之前**调用——等过界后再读，读到的才是
新窗口的链（顺序本身就是判据，故本组专设 ``test_hold_resolves_chain_after_switch``）。

任务书 success_criteria 逐条对上：

- **构造「距边界 <5min」→ 派发在过点后发生（时间戳实测）** → ① 组
  ``test_hold_waits_past_switch_point``（记录 ``_sleep`` 秒数，验 ``clock + wait ≥ 切换点``）
  与 ``test_hold_resolves_chain_after_switch``（跨界的那一跳真落进新窗口的席链）；
- **「距边界 >5min」→ 不等待（对照组）** → ① 组 ``test_far_from_boundary_does_not_wait``；
- **``--no-hold`` 放行** → ① 组 ``test_no_hold_flag_skips_wait``；
- **等待行入 run 日志** → ② 组 ``test_hold_line_reaches_run_log``；
- **等待上限硬钳制 ≤300s** → ③ 组 ``test_boundary_hold_s_is_clamped``（配置面）与
  ``test_over_threshold_never_waits``（行为面：钳制后仍 > 真实距离就不等）。

``quota``/重试那两条与本单无关的面在 ``test_same_seat_retry.py``；等待的抽象落点是
``dispatch._sleep``（退避与避让共用），autouse 夹具 ``no_real_backoff_sleep`` 已把真实
等待换成 no-op，故本组用例只 patch 记录、不真睡。
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from pysci.skills.orchestration.tools import dispatch, drain, registry, routing

from .conftest import (
    drain_once,
    envelope,
    fake_headless,
    seed_backlog,
    set_models,
    set_routing_clock,
)

#: 2026-10-14（周三）＝实测非假日工作日；峰段 9–12 与 14–18（出厂表）。
WORKDAY = date(2026, 10, 14)
OK_BODY = "<result>完工，commit abc1234 已合并。</result>"


def cn(day: date, hour: int, minute: int = 0, second: int = 0) -> datetime:
    """北京时间的某时刻（判定口径就是它，用例不与机器时区纠缠）。"""
    return datetime(
        day.year,
        day.month,
        day.day,
        hour,
        minute,
        second,
        tzinfo=ZoneInfo(routing.CN_TZ_NAME),
    )


def _record_sleep(monkeypatch):
    """记录 ``_sleep`` 收到的秒数——用例不真睡，但「等了多久」必须可断言。"""
    slept: list[float] = []
    monkeypatch.setattr(dispatch, "_sleep", lambda s: slept.append(float(s)))
    return slept


def _set_hold(state, **over):
    """改写隔离 registry.json 的 ``routing.boundary_hold_s``（其余键走出厂缺省）。"""
    reg_file = state / "registry.json"
    data = json.loads(reg_file.read_text(encoding="utf-8"))
    data.setdefault("routing", {}).update(over)
    reg_file.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _last_ledger_row(state):
    lines = (state / "ledger.jsonl").read_text(encoding="utf-8").strip().splitlines()
    return json.loads(lines[-1])


def _ok_dispatch(monkeypatch, **kw):
    """无故障派发（一次跑通），返回本次 argv 列表。"""
    calls = fake_headless(
        monkeypatch,
        lambda n, cmd: envelope(OK_BODY, is_error=False, stop="end_turn", rc=0),
    )
    dispatch.do_dispatch("quotamember", text="出图", quiet=True, **kw)
    return calls


# ---------------------------------------------------------------------------
# ① 避让本体：等 / 不等 / --no-hold / 等到过点
# ---------------------------------------------------------------------------
def test_hold_waits_past_switch_point(iso_state, iso_dispatch, monkeypatch):
    """距 12:00 入谷 100s（<300s）→ 等 100s，且等完真的跨过了切换点。

    这是 success_criteria 的「时间戳实测」：判据不是「调过 ``_sleep``」，而是
    ``时钟读数 + 等待秒数 ≥ 切换点``——等少了（没等到点）与等多了都不算避让。
    """
    slept = _record_sleep(monkeypatch)
    set_models(iso_dispatch, flash=["seat-a"])
    set_routing_clock(monkeypatch, cn(WORKDAY, 11, 58, 20))  # 距 12:00 入谷 100s

    _ok_dispatch(monkeypatch, no_checks=True)

    assert len(slept) == 1, "避让只该等一次（等完即过界，不再二次等）"
    assert slept[0] == 100.0, "等的秒数必须正好补到切换点，不是四舍五入到分钟"
    assert (
        cn(WORKDAY, 11, 58, 20).timestamp() + slept[0] >= cn(WORKDAY, 12, 0).timestamp()
    ), "等完必须真的跨过 12:00——等 90s 或等 120s 都不合格"


def test_hold_resolves_chain_after_switch(iso_state, iso_dispatch, monkeypatch):
    """避让的**意义**：等过界后再解析席链，读到的才是**新窗口**的链。

    钉的是顺序而非「席链变了」：``_boundary_hold`` 读的时钟与 ``presets`` 解析链读的
    是**同一份**冻结时钟（``set_routing_clock``），冻结下两者永远一致，故「换了席」
    这类断言测不到顺序——时钟只要在避让**之后**没推进，把避让挪到解析之后它照样绿。

    真正的判据是**推进条件的归属**：避让之后紧邻的下一步必须是路由读数
    （``note_switch``），角色/席链解析在其后。把 ``_boundary_hold`` 挪到
    ``resolve_role``/``plan_seats`` 之后，本例立即红。这条顺序正是「等过界后读到新窗口
    的链」的机械保证，也是任务书要防的那类回归。
    """
    slept = _record_sleep(monkeypatch)
    set_models(iso_dispatch, flash=["seat-a"])
    set_routing_clock(monkeypatch, cn(WORKDAY, 11, 58, 20))  # 12:00 前 100s

    _ok_dispatch(monkeypatch, no_checks=True)

    src = Path(dispatch.__file__).read_text(encoding="utf-8")
    hold_at = src.index("hold_line, hold_wait_s = _boundary_hold(")
    switch_at = src.index("route_line = routing.note_switch()")
    resolve_at = src.index("presets.resolve_role(")
    assert slept == [100.0], "避让确实等了——读数面先证成，再谈顺序"
    assert hold_at < switch_at < resolve_at, (
        "避让必须在路由读数之前、角色解析之前：等过界后再读，才是新窗口的链"
    )


def test_far_from_boundary_does_not_wait(iso_state, iso_dispatch, monkeypatch):
    """对照组：距切换 > 阈值（10:00 距 12:00 入谷 7200s）→ 一次都不等。"""
    slept = _record_sleep(monkeypatch)
    set_models(iso_dispatch, flash=["seat-a"])
    set_routing_clock(monkeypatch, cn(WORKDAY, 10, 0))

    _ok_dispatch(monkeypatch, no_checks=True)

    assert slept == [], "7200s 的距离去等 300s 是白拖队列，避让只在边界附近生效"


def test_no_hold_flag_skips_wait(iso_state, iso_dispatch, monkeypatch):
    """``--no-hold``（草稿/调试场景）：哪怕就在边界前也不等。"""
    slept = _record_sleep(monkeypatch)
    set_models(iso_dispatch, flash=["seat-a"])
    set_routing_clock(monkeypatch, cn(WORKDAY, 11, 59, 30))  # 距 12:00 仅 30s

    _ok_dispatch(monkeypatch, no_checks=True, no_hold=True)

    assert slept == [], "``no_hold=True`` 是硬开关，连 30s 的边界也不许等"


# ---------------------------------------------------------------------------
# ② 观察面：run 日志 + 台账
# ---------------------------------------------------------------------------
def test_hold_line_reaches_run_log(iso_state, iso_dispatch, monkeypatch):
    """等待行必须进 run 日志——否则批次账上「这一跳凭空多了 100s」无人可解释。"""
    _record_sleep(monkeypatch)
    set_models(iso_dispatch, flash=["seat-a"])
    set_routing_clock(monkeypatch, cn(WORKDAY, 11, 58, 20))
    seed_backlog(
        iso_state,
        [{"id": "h1", "status": "pending", "summary": "s1", "member": "quotamember"}],
    )
    fake_headless(
        monkeypatch,
        lambda n, cmd: envelope(OK_BODY, is_error=False, stop="end_turn", rc=0),
    )
    run_log = drain.new_run_log()
    drain_once(log=run_log)

    text = run_log.read_text(encoding="utf-8")
    assert "边界避让" in text, "run 日志必须点名避让，否则耗时读数无人可解释"
    assert "100" in text, "等待秒数要进日志——「等了多久」是统计避让节省的原料"


def test_hold_seconds_land_in_ledger(iso_state, iso_dispatch, monkeypatch):
    """台账 ``hold_s`` 记等待秒数（规格 4「台账记录避让事件」）。"""
    slept = _record_sleep(monkeypatch)
    set_models(iso_dispatch, flash=["seat-a"])
    set_routing_clock(monkeypatch, cn(WORKDAY, 11, 58, 20))

    _ok_dispatch(monkeypatch, no_checks=True)

    row = _last_ledger_row(iso_dispatch)
    assert row["hold_s"] == 100, "台账 hold_s 必须与实际等待一致"
    assert slept == [100.0], "台账读数与实等秒数同源，两处不许各说各话"


def test_no_wait_records_zero_hold(iso_state, iso_dispatch, monkeypatch):
    """没避让就记 0——字段必须在（缺字段会让统计面把「没等」误读成「数据缺失」）。"""
    _record_sleep(monkeypatch)
    set_models(iso_dispatch, flash=["seat-a"])
    set_routing_clock(monkeypatch, cn(WORKDAY, 10, 0))

    _ok_dispatch(monkeypatch, no_checks=True)

    assert _last_ledger_row(iso_dispatch)["hold_s"] == 0


# ---------------------------------------------------------------------------
# ③ 配置面：缺省 / 0=禁用 / 硬钳制 ≤300s
# ---------------------------------------------------------------------------
def test_boundary_hold_s_default(iso_state):
    """出厂缺省 300s（5 分钟，用户裁决的「稍等过界」尺度）。"""
    assert registry.Registry.load().routing_hold_s == registry.DEFAULT_BOUNDARY_HOLD_S


def test_zero_disables_boundary_hold(iso_state, iso_dispatch, monkeypatch):
    """``boundary_hold_s = 0`` → 整条避让通路关掉（不是「等 0 秒」，是不进分支）。"""
    _set_hold(iso_dispatch, boundary_hold_s=0)
    slept = _record_sleep(monkeypatch)
    set_models(iso_dispatch, flash=["seat-a"])
    set_routing_clock(monkeypatch, cn(WORKDAY, 11, 59, 59))  # 只差 1s 也不等

    _ok_dispatch(monkeypatch, no_checks=True)

    assert registry.Registry.load().routing_hold_s == 0
    assert slept == []


def test_boundary_hold_s_is_clamped(iso_state, iso_dispatch):
    """硬钳制 ≤300s（规格 3）：写 3600 也只等 300——「稍等过界」不是「排队等窗口」。"""
    _set_hold(iso_dispatch, boundary_hold_s=3600)
    assert (
        registry.Registry.load().routing_hold_s == registry.BOUNDARY_HOLD_CAP_S == 300
    )


def test_over_threshold_never_waits(iso_state, iso_dispatch, monkeypatch):
    """配置写 9999（钳到 300）而真实距离 7200s → 不等；钳制在行为面同样成立。"""
    _set_hold(iso_dispatch, boundary_hold_s=9999)
    slept = _record_sleep(monkeypatch)
    set_models(iso_dispatch, flash=["seat-a"])
    set_routing_clock(monkeypatch, cn(WORKDAY, 10, 0))

    _ok_dispatch(monkeypatch, no_checks=True)

    assert registry.Registry.load().routing_hold_s == 300
    assert slept == []


def test_negative_or_garbage_falls_back_to_default(iso_state, iso_dispatch):
    """非法值 fail-open 回缺省（与 ``fallback_policy`` 同口径）：负数/非数字都不炸派发。"""
    _set_hold(iso_dispatch, boundary_hold_s=-1)
    assert registry.Registry.load().routing_hold_s == registry.DEFAULT_BOUNDARY_HOLD_S
    _set_hold(iso_dispatch, boundary_hold_s="nope")
    assert registry.Registry.load().routing_hold_s == registry.DEFAULT_BOUNDARY_HOLD_S
