"""模型回退链的**行为面**（backlog 20261012-211500-leader，用户裁决 2026-10-12 21:1x）。

规格落点逐条钉住（同源/配置形态的断言在 ``test_flash_channel_swap.py`` 与
``test_model_refresh.py``）：

* 规格 3：当前席不可用（额度 / 渠道 / **启动即失败**）→ 沿有序链**一席一跳**自动降级；
  全链耗尽 → 保留系统性 kind，drain 全局停止（#77 语义，不逐单空烧）；
* 规格 4：回退**必须**被看见——调用者面（dispatch 输出 + drain run 日志）逐条告警行，
  观察面（``state/channel-events.jsonl`` 事件流 → mirror 概览项「回退」+ ``orch status``
  判决包健康段）计数与最近条目；
* 补充判据（用户 21:3x：BYOK Qwen 余额暂不充值、将持续不可用）：**降级粘滞窗**——首次
  探测失败即进窗；窗内同档直接落到下一席（零探测）但仍打一行「席 X 跳过中」；窗口过期
  自动恢复探测；全链都在窗内时强探最先到期席（不停摆）。

席位一律用占位串（``seat-a`` 等）：本组测的是**选席与告警**，与真实 modelID 无关。
"""

from __future__ import annotations

import json
import time

from pysci.skills.orchestration.tools import (
    channels,
    dispatch,
    drain,
    mirror,
    registry,
    verdict,
)
from pysci.skills.orchestration.tools.dispatch import DispatchOutcome

from .conftest import (
    CHANNEL_TEXT,
    QUOTA_TEXT,
    cmd_model,
    envelope,
    fake_headless,
    read_backlog,
    read_runtime,
    seed_backlog,
    set_models,
    write_runtime,
)

A = "seat-a"  # 首席（Mimo  analog）
B = "seat-b"  # 次席（BYOK Qwen analog）
C = "seat-c"  # 末席（内置免费 analog）
OK_BODY = "<result>完工，commit abc1234 已合并。</result>"


def _broken(state, sid_seen=None):
    """渠道级故障响应器：写末轮合成故障形状 + 返回该文案 envelope（#77 判据支）。"""

    def responder(n, cmd):
        from .conftest import append_turn, sid_from_argv

        append_turn(
            state,
            sid_from_argv(cmd),
            model="<synthetic>",
            text=CHANNEL_TEXT,
            stop_reason="stop_sequence",
            stop_sequence="",
        )
        return envelope(CHANNEL_TEXT)

    return responder


def _events(state):
    return channels.read_events(path=state / "channel-events.jsonl")


def _fail_window(
    state, tier: str, seat: str, *, ts: float, kind: str = "quota_exhausted"
):
    """直接预置粘滞窗（模拟「上一次派发探到该席坏了」），避免用例为了造态而多烧一跳。"""
    rt = read_runtime(state)
    rt[registry.RUNTIME_SEAT_FAIL_KEY] = {
        tier: {seat: {"ts": ts, "kind": kind, "member": "quotamember"}}
    }
    write_runtime(state, rt)


# ---------------------------------------------------------------------------
# ① 自动降级 + 调用者告警（规格 3 / 4a）
# ---------------------------------------------------------------------------
def test_chief_seat_broken_demotes_one_hop_and_alerts(
    iso_state, iso_dispatch, monkeypatch, capsys
):
    set_models(iso_dispatch, flash=[A, B, C])
    calls = fake_headless(
        monkeypatch,
        lambda n, cmd: (
            envelope(QUOTA_TEXT)
            if n == 1
            else envelope(OK_BODY, is_error=False, stop="end_turn", rc=0)
        ),
    )

    o = dispatch.do_dispatch("quotamember", text="出图", quiet=False, no_checks=True)
    out = capsys.readouterr().out

    assert o.kind == "result" and o.code == 0, "换席后跑通=正常交付，不是带伤交付"
    assert len(calls) == 2
    assert cmd_model(calls[0]) == A and cmd_model(calls[1]) == B, "一席一跳，不越过次席"
    assert "[!] 档位回退：main 席seat-a不可用 → 已切换 seat-b" in out, out
    assert "原因 quota_exhausted" in out, (
        "告警要带成因，否则读的人无法决定充值还是修渠道"
    )
    assert len(o.fallbacks) == 1
    line = (iso_dispatch / "ledger.jsonl").read_text(encoding="utf-8")
    assert B in line, (
        "台账 model 字段记**实际所用席位**（success_criteria 的真派判据同形）"
    )


def test_demote_alert_suppressed_by_quiet_but_still_logged(
    iso_state, iso_dispatch, monkeypatch, capsys
):
    """quiet=True 不打扰人，但事件照样落账——告警面与留痕面各自独立。"""
    set_models(iso_dispatch, flash=[A, B])
    fake_headless(
        monkeypatch,
        lambda n, cmd: (
            envelope(QUOTA_TEXT)
            if n == 1
            else envelope(OK_BODY, is_error=False, stop="end_turn", rc=0)
        ),
    )
    o = dispatch.do_dispatch("quotamember", text="出图", quiet=True, no_checks=True)

    assert capsys.readouterr().out == ""
    assert o.kind == "result" and len(o.fallbacks) == 1
    events = _events(iso_dispatch)
    assert len(events) == 1
    assert events[0]["tier"] == "main" and events[0]["from"] == A
    assert events[0]["to"] == B and events[0]["reason"] == "quota_exhausted"
    assert events[0]["member"] == "quotamember" and events[0]["ts"][:4] == "2026"


def test_three_seat_chain_walks_to_last(iso_state, iso_dispatch, monkeypatch):
    """前两席都坏 → 落到末席；每起回退各一行告警、各一条事件。"""
    set_models(iso_dispatch, flash=[A, B, C])
    calls = fake_headless(
        monkeypatch,
        lambda n, cmd: (
            envelope(QUOTA_TEXT)
            if n <= 2
            else envelope(OK_BODY, is_error=False, stop="end_turn", rc=0)
        ),
    )

    o = dispatch.do_dispatch("quotamember", text="出图", quiet=True, no_checks=True)

    assert o.kind == "result" and len(calls) == 3
    assert [cmd_model(c) for c in calls] == [A, B, C], "严格按链序下探，不回头、不跳席"
    assert [e["to"] for e in _events(iso_dispatch)] == [B, C]


# ---------------------------------------------------------------------------
# ② 全链耗尽 → 系统性停止（规格 3 后半 + #77 语义不变）
# ---------------------------------------------------------------------------
def test_chain_exhausted_keeps_systemic_kind_and_records_exhaustion(
    iso_state, iso_dispatch, monkeypatch
):
    set_models(iso_dispatch, flash=[A, B])
    calls = fake_headless(monkeypatch, _broken(iso_dispatch))

    o = dispatch.do_dispatch("quotamember", text="出图", quiet=True)

    assert o.kind == "channel_error", "全链坏了要说「渠道级故障」，drain 据此全局停止"
    # 同席重试（backlog 20261013-143000-leader，用户裁决 2026-10-13）：渠道级故障先在
    # 原席自救一次（测试口径压到 1 次，见 conftest.iso_dispatch），仍败才换席——故是
    # 首跳 + 1 次自救 + 次席一跳 = 3 跳，而不是旧语义的「两席各一跳」。
    assert [cmd_model(c) for c in calls] == [A, A, B], "自救不换席，换席才动链"
    assert o.same_seat_retries == 1, "自救次数必须在读数面上可见"
    events = _events(iso_dispatch)
    assert events[-1]["to"] == "" and events[-1]["to_label"] == "(全链耗尽)", (
        "「停止」这件事本身也要落账，否则观察面上它是隐形的"
    )
    assert [e["to"] for e in events] == [B, ""], "自救不进回退事件面，只有真实换席进"


def test_drain_stops_and_logs_each_demote(iso_state, monkeypatch):
    """run 日志逐条留痕回退（drain 不另立文案口径），且第二单不再被派发。"""
    seed_backlog(
        iso_state,
        [
            {"id": "k1", "status": "pending", "summary": "s1", "member": "devops"},
            {"id": "k2", "status": "pending", "summary": "s2", "member": "devops"},
        ],
    )
    event = {
        "ts": "2026-10-13T00:00:00+08:00",
        "tier": "flash",
        "from": A,
        "from_label": A,
        "to": B,
        "to_label": B,
        "reason": "channel_error",
        "member": "devops",
        "sid": "s",
    }
    dispatched = []

    def fake_dispatch(member, **kw):
        dispatched.append(kw.get("slug"))
        return DispatchOutcome(
            code=2,
            kind="channel_error",
            body="",
            error="渠道级故障：custom model service error",
            fallbacks=[event],
        )

    monkeypatch.setattr(dispatch, "do_dispatch", fake_dispatch)
    run_log = drain.new_run_log()
    summary = drain.drain_backlog(run_log)

    assert summary["stopped"] == "channel_error"
    assert dispatched == ["k1"], "渠道坏了不是任务书坏了，k2 不得再烧一跳"
    text = run_log.read_text(encoding="utf-8")
    assert "[!] 档位回退：flash 席seat-a不可用 → 已切换 seat-b" in text
    assert "（批次 k1" in text, "留痕要点名批次，否则事后查不到是哪一单换的席"
    assert "[C] 全局停止" in text
    by_id = {i["id"]: i for i in read_backlog(iso_state)}
    assert by_id["k1"]["status"] == "pending" and by_id["k2"]["status"] == "pending"


# ---------------------------------------------------------------------------
# ③ 启动即失败也算「席坏」（规格 3 的第三判据）
# ---------------------------------------------------------------------------
def test_startup_failure_switches_seat(iso_state, iso_dispatch, monkeypatch):
    """型号在账户里不存在/渠道起不来 → rc≠0 且 envelope 不可解析 = 席坏，换席。"""
    set_models(iso_dispatch, flash=[A, B])
    calls = fake_headless(
        monkeypatch,
        lambda n, cmd: (
            (1, "not json at all", "")
            if n == 1
            else envelope(OK_BODY, is_error=False, stop="end_turn", rc=0)
        ),
    )

    o = dispatch.do_dispatch("quotamember", text="出图", quiet=True, no_checks=True)

    assert o.kind == "result" and len(calls) == 2
    assert cmd_model(calls[1]) == B
    events = _events(iso_dispatch)
    assert events[0]["reason"] == "startup_failure", (
        "成因要如实记「启动即失败」，不冒充额度"
    )


def test_startup_failure_single_seat_chain_not_promoted(
    iso_state, iso_dispatch, monkeypatch
):
    """一席链（旧单值写法）启动即失败 → 说 run_failed，**不**升格 channel_error：

    没换过席就宣布「渠道级故障」会把一次性抖动说成停摆，drain 会因此整条队列让路。
    """
    set_models(iso_dispatch, flash=A)
    calls = fake_headless(monkeypatch, lambda n, cmd: (1, "boom", ""))

    o = dispatch.do_dispatch("quotamember", text="出图", quiet=True)

    assert len(calls) == 1, "席坏不做同席重试"
    assert o.kind == "run_failed"
    assert _events(iso_dispatch) == [], "没有换席就没有回退事件，观察面不凭空计数"


# ---------------------------------------------------------------------------
# ④ 降级粘滞窗（用户 21:3x 补充判据的三段）
# ---------------------------------------------------------------------------
def test_window_skips_broken_seat_without_probing(
    iso_state, iso_dispatch, monkeypatch, capsys
):
    """段一+二：首次探测失败并告警 → 窗内第二次派发**零探测**该席，直落下一席。"""
    set_models(iso_dispatch, flash=[A, B])
    ok = lambda n, cmd: envelope(OK_BODY, is_error=False, stop="end_turn", rc=0)  # noqa: E731

    first = fake_headless(
        monkeypatch,
        lambda n, cmd: envelope(QUOTA_TEXT) if n == 1 else ok(n, cmd),
    )
    assert (
        dispatch.do_dispatch(
            "quotamember", text="出图", quiet=True, no_checks=True
        ).kind
        == "result"
    )
    assert len(first) == 2, "首跳探了 A（失败）+ 次跳换 B"

    calls = fake_headless(monkeypatch, ok)
    o = dispatch.do_dispatch("quotamember", text="出图", quiet=False, no_checks=True)
    out = capsys.readouterr().out

    assert o.kind == "result" and len(calls) == 1, "窗内不再白烧 A 的探测跳"
    assert cmd_model(calls[0]) == B
    assert "席 seat-a 跳过中" in out and "窗口内不重试" in out, (
        "省跳数不能省掉可查性：跳过哪一席、什么时候坏的、为什么都得打出来"
    )


def test_window_expiry_resumes_probe(iso_state, iso_dispatch, monkeypatch):
    """段三：窗口过期 → 自动恢复探测首席（充值/修好后无需改代码即自愈）。"""
    set_models(iso_dispatch, flash=[A, B])
    _fail_window(iso_dispatch, "flash", A, ts=time.time() - 4000)
    calls = fake_headless(
        monkeypatch,
        lambda n, cmd: envelope(OK_BODY, is_error=False, stop="end_turn", rc=0),
    )

    o = dispatch.do_dispatch("quotamember", text="出图", quiet=True, no_checks=True)

    assert o.kind == "result" and cmd_model(calls[0]) == A, "过期后首席仍是首席"
    assert len(calls) == 1


def test_success_clears_window(iso_state, iso_dispatch):
    """跑通即销窗：一席的成功比一小时的等待更有资格恢复正常优先级。"""
    set_models(iso_dispatch, flash=[A, B])
    reg = registry.Registry.load()
    reg.note_seat_failure("flash", A, kind="quota_exhausted", member="m")

    channels.note_seat_ok(reg, tier="flash", seat=A)

    assert A not in reg.seat_failures.get("flash", {})
    assert channels.plan_seats(reg, "flash").usable == [A, B]


def test_whole_chain_in_window_forces_earliest_probe(iso_state, iso_dispatch):
    """全链都在窗内不能演变成「无席可用」：按最先到期的一席再探一次（宁可多烧一跳）。"""
    _fail_window(iso_dispatch, "flash", A, ts=time.time() - 3500)
    rt = read_runtime(iso_dispatch)
    rt[registry.RUNTIME_SEAT_FAIL_KEY]["flash"].update(
        {
            B: {"ts": time.time() - 200, "kind": "channel_error", "member": "m"},
            C: {"ts": time.time() - 100, "kind": "channel_error", "member": "m"},
        }
    )
    write_runtime(iso_dispatch, rt)
    set_models(iso_dispatch, flash=[A, B, C])

    plan = channels.plan_seats(registry.Registry.load(), "flash")

    assert plan.usable == [A], f"最先到期的 A 被强探：{plan.usable}"
    assert plan.forced_probe == A
    assert [s["seat"] for s in plan.skipped] == [B, C]
    assert "全部席位都在跳过窗内" in channels.forced_line(plan)


def test_window_zero_disables_stickiness(iso_state, iso_dispatch):
    """``model_fallback.skip_window_s=0`` → 不粘滞（每跳照探首席），参数面真要能关掉。"""
    reg_file = iso_dispatch / "registry.json"
    data = json.loads(reg_file.read_text(encoding="utf-8"))
    data["model_fallback"] = {"skip_window_s": 0, "event_tail": 5}
    reg_file.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    set_models(iso_dispatch, flash=[A, B])
    _fail_window(iso_dispatch, "flash", A, ts=time.time())

    plan = channels.plan_seats(registry.Registry.load(), "flash")

    assert plan.usable == [A, B] and plan.skipped == []


def test_non_systemic_failure_retries_same_seat(iso_state, iso_dispatch, monkeypatch):
    """故障不在渠道（普通 run_failed）→ 同席重试一次，**不**换席、**不**进窗。"""
    set_models(iso_dispatch, flash=[A, B])
    calls = fake_headless(monkeypatch, lambda n, cmd: envelope("internal boom"))

    o = dispatch.do_dispatch("quotamember", text="出图", quiet=True)

    assert o.kind == "run_failed" and len(calls) == 2
    assert all(cmd_model(c) == A for c in calls), "换席白烧：非系统性失败不碰链"
    assert _events(iso_dispatch) == []
    reg = registry.Registry.load()
    assert A not in reg.seat_failures.get("flash", {}), "普通失败不该把好席关进窗"


def test_no_retry_suppresses_demotion(iso_state, iso_dispatch, monkeypatch):
    """``no_retry``（单跳语义）同时抑制换席：只探首席，坏了就报系统性 kind。"""
    set_models(iso_dispatch, flash=[A, B, C])
    calls = fake_headless(monkeypatch, lambda n, cmd: envelope(QUOTA_TEXT))

    o = dispatch.do_dispatch("quotamember", text="出图", quiet=True, no_retry=True)

    assert o.kind == "quota_exhausted" and len(calls) == 1
    assert _events(iso_dispatch) == [], "不换席就不留回退事件，计数面别虚高"


# ---------------------------------------------------------------------------
# ⑤ 观察面读数（规格 4b：mirror 概览项 + orch status 健康段同源）
# ---------------------------------------------------------------------------
def test_mirror_status_item_reports_events(iso_state, iso_dispatch):
    """mirror 的「回退」项：有事件报计数+最近条目，无事件报「尚无」——缺席与零必须可辨。"""
    item = dict(mirror.STATUS_ITEMS)
    assert "回退" in item, "概览项没接线的话，事件流写了也没人看（规格 4b 只剩半面）"
    assert "（尚无渠道回退事件）" in item["回退"]()

    channels.append_event(
        {
            "ts": "2026-10-13T01:02:00+08:00",
            "tier": "flash",
            "from": A,
            "from_label": A,
            "to": B,
            "to_label": B,
            "reason": "channel_error",
        },
        path=iso_dispatch / "channel-events.jsonl",
    )
    line = item["回退"]()

    assert "累计 1 起" in line and "近 7 日 1 起" in line
    assert "flash seat-a→seat-b" in line and "channel_error" in line, (
        "条目要含时刻/档位/原席→新席/原因，光有计数查不到是哪一单"
    )


def test_orch_status_health_section_shows_fallbacks(iso_state, iso_dispatch):
    """``orch status`` 判决包健康段与 mirror 共读同一句话（单一口径，不各写各的）。"""
    channels.append_event(
        {
            "ts": "2026-10-13T02:03:00+08:00",
            "tier": "max",
            "from": "Mimo-V2.6-Pro",
            "from_label": "Mimo-V2.6-Pro",
            "to": "Qwen-3.8-Max",
            "to_label": "Qwen-3.8-Max",
            "reason": "quota_exhausted",
        },
        path=iso_dispatch / "channel-events.jsonl",
    )

    lines = verdict.health_lines()
    mirror_line = dict(mirror.STATUS_ITEMS)["回退"]()

    body = "\n".join(lines)
    assert "渠道回退" in body and "Mimo-V2.6-Pro→Qwen-3.8-Max" in body
    assert mirror_line.splitlines()[0] in body, (
        "两处读数是同一句话的两次呈现，不是两份口径"
    )


def test_summary_line_survives_bad_and_future_events(iso_state, iso_dispatch):
    """观察面 fail-open：坏行跳过、时间戳读不出不误计近窗（读数失败绝不能挡住 status/mirror）。"""
    p = iso_dispatch / "channel-events.jsonl"
    p.write_text(
        "{ not json\n"
        + json.dumps({"ts": "nonsense", "tier": "flash", "from": A, "to": B})
        + "\n",
        encoding="utf-8",
    )

    line = channels.summary_line(path=p)

    assert "累计 1 起" in line and "近 7 日 0 起" in line, (
        "读不出的时间戳按最旧处理 → 落进窗外，宁可少计也不虚报"
    )


def test_chain_text_is_one_presentation_source(iso_state, iso_dispatch):
    """链的一行呈现（drain 日志与 refresh-models 共用）：单值/列表都能读，空链有占位。"""
    reg = registry.Registry.load()

    assert channels.chain_text([A, B], reg=reg) == "seat-a → seat-b"
    assert channels.chain_text(A, reg=reg) == "seat-a", "旧单值写照样能呈现"
    assert channels.chain_text(["", B], reg=reg) == "(用户级默认) → seat-b"
    assert channels.chain_text([], reg=reg) == "(空)"
