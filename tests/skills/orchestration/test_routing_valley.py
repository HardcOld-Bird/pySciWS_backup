"""路由 v3：谷时隐性 DeepSeek 的判定/切换/换血三面测试（backlog 20261013-093000-leader）。

success_criteria 落点（任务书原文逐条对上）：

- **谷时判定单测矩阵** → ① 组（工作日各时段 × 周末 × 调休周六 × 法定节假日 × 覆盖外年份）；
- **chinesecalendar 覆盖外年份告警** → ``test_holiday_library_*``；
- **切换时机实测**（12:00/18:00 即切、8:00/13:00 早切、单跳在途不换模型＝台账 model
  字段一致性）→ ② 组 + ``test_inflight_hop_keeps_its_model_*``；
- **隐性态第三席链三例 + 缺 Key 告警实测** → ③ 组（席链与面板，终裁 v4 矩阵：隐性态
  固定 GLM+KIMI+Mimo、pro 席不被时间覆盖）+ ④ 组（缺 Key 减员 2 席运行的双面告警、
  同族回退席不上场）；
- **mirror status 路由段实时读数 + 下次切换预告** → ⑤ 组（mirror 概览 + ``orch status`` 段）；
- **minimum=Mimo-Flash** → ⑥ 组只钉机制（隐性态不劫持 minimum 独立链）。链首的**当前配置
  读数**不在测试里钉死——把运行面此刻值当默认值钉，合法上线一次就红一次；证据走交付里的
  实跑命令。

日期一律取真实日历读数（``chinese_calendar`` 1.11 实测），不造日期。
"""

from __future__ import annotations

import json
import sys
from argparse import Namespace
from datetime import date, datetime
from zoneinfo import ZoneInfo

from pysci.skills.orchestration.tools import (
    dispatch,
    journal,
    jury,
    ledger,
    mirror,
    orch,
    presets,
    registry,
    routing,
)

from .conftest import (
    SEAT_KEY_ENV,
    cmd_model,
    envelope,
    fake_headless,
    set_jury_panel,
    set_models,
    set_presets,
    set_routing_clock,
    write_preset,
    write_routing,
)

DS_UUID = "1a54cc57-e614-4521-80cc-447de31dc896"
DS_BUILTIN = "DeepSeek-Flash"
MIMO_MAIN = "seat-mimo-main"
QWEN_MAIN = "seat-qwen-main"
BUILTIN = "Qwen3.8-Flash"
OK_BODY = "<result>完工，commit abc1234 已合并。</result>"

#: 2026-10-14（周三）＝实测非假日工作日（get_holiday_detail → (False, None)）。
WORKDAY = date(2026, 10, 14)
#: 2026-10-17（周六）＝普通周末。
SATURDAY = date(2026, 10, 17)
#: 2026-10-10（周六）＝国庆**调休上班**日（实测 off=False、name=National Day）。
MAKEUP_SATURDAY = date(2026, 10, 10)
#: 2026-10-05（周一）＝落在工作日里的法定节假日（实测 off=True、name=National Day）。
HOLIDAY_WEEKDAY = date(2026, 10, 5)


def cn(day: date, hour: int, minute: int = 0) -> datetime:
    """北京时间的某时刻（判定口径就是它，用例不与机器时区纠缠）。"""
    return datetime(
        day.year, day.month, day.day, hour, minute, tzinfo=ZoneInfo(routing.CN_TZ_NAME)
    )


def config(preset: str = "mimo", **routing_kv) -> dict:
    """注入用的「配置面快照」（读盘替身）：``[model] preset`` 与 ``[routing]`` 同一份。"""
    model = {"preset": preset}
    model.update(routing_kv)
    return {"model": model}


def reading(moment: datetime, preset: str = "mimo", **routing_kv):
    """注入配置的路由读数（矩阵用例不碰盘，也就与开发者机器上的 toml 无关）。"""
    return routing.state(moment=moment, data=config(preset, **routing_kv))


# ---------------------------------------------------------------------------
# ① 谷/峰判定矩阵（规格 1）
# ---------------------------------------------------------------------------
def test_workday_peak_hours_are_not_hidden():
    """峰段内（9–12、14–18）＝原预设：2 倍价时段把主席交给官方 DeepSeek 不是本意。"""
    for hour, minute in ((9, 0), (10, 0), (11, 59), (14, 0), (17, 59)):
        r = reading(cn(WORKDAY, hour, minute))
        assert r.hidden is False, f"{hour:02d}:{minute:02d} 应在峰段"
        assert "峰段" in r.basis and "2 倍价" in r.basis


def test_workday_off_peak_hours_are_valley():
    """深夜/清晨与午休前半＝谷（规格 1 的 ③：工作日其余时段）。"""
    for hour, minute in ((0, 30), (2, 0), (12, 0), (12, 30), (18, 0), (23, 0)):
        r = reading(cn(WORKDAY, hour, minute))
        assert r.hidden is True, f"{hour:02d}:{minute:02d} 应判成谷"
        assert "谷段" in r.basis


def test_weekend_is_valley_all_day():
    """周六日全天谷（规格 1 的 ①），含落在「峰段钟点」上的周末午后。"""
    for hour in (10, 15, 21):
        r = reading(cn(SATURDAY, hour))
        assert r.hidden is True
        assert "周末全天" in r.basis


def test_makeup_work_saturday_is_still_valley():
    """调休上班的周六（实测 2026-10-10：off=False 却挂 National Day）**仍按星期几判谷**。

    用户裁决：``is_in_lieu`` 不参与。库里这天 ``off`` 是假的——若实现先查假日表就会把这
    天误判成工作日，于是峰段照烧 2 倍价，正是要钉住的那条。
    """
    r = reading(cn(MAKEUP_SATURDAY, 10))
    assert r.hidden is True, "调休周六按周末算谷"
    assert "周六（周末全天）" in r.basis


def test_legal_holiday_on_weekday_is_valley():
    """法定节假日落在工作日＝全天谷（规格 1 的 ②；实测 2026-10-05 周一 National Day）。"""
    r = reading(cn(HOLIDAY_WEEKDAY, 10))
    assert r.hidden is True
    assert "法定节假日：National Day" in r.basis


def test_holiday_library_out_of_coverage_falls_back_conservatively():
    """库不覆盖的年份（2027）→ 保守不切换 + 出声告警（规格 1 末段）。"""
    r = reading(cn(date(2027, 5, 10), 10))
    assert r.hidden is False, "判不出的日期一律维持原预设（宁可多付峰时价）"
    assert "保守沿用原预设" in r.basis
    assert any("不覆盖 2027" in w for w in r.warnings), r.warnings


def test_holiday_library_absent_warns_instead_of_crashing(monkeypatch):
    """库缺席（ImportError）同样走「判不出＋告警」，不许把读数面/派发拖崩。"""
    monkeypatch.setitem(sys.modules, "chinese_calendar", None)
    off, name, warn = routing.holiday_off(WORKDAY)
    assert (off, name) == (False, "")
    assert "chinesecalendar 未安装" in warn
    kind, basis, warn2 = routing.day_reading(WORKDAY, routing.Face())
    assert (kind, basis) == ("unknown", "工作日（节假日判不出）")
    assert warn2 == warn
    assert routing.state(moment=cn(WORKDAY, 10), data=config()).hidden is False


def test_installed_library_reads_known_dates():
    """已装库对覆盖内年份不出声，且读数与判定面一致（防「告警用例常绿、真库读歪」）。"""
    assert routing.holiday_off(WORKDAY) == (False, "", "")
    assert routing.holiday_off(HOLIDAY_WEEKDAY) == (True, "National Day", "")


def test_explicit_deepseek_preset_reports_no_override():
    """toml 本来就写着 deepseek 时不叫「覆盖」——它不是被覆盖方，报覆盖等于虚报切换。"""
    r = reading(cn(SATURDAY, 10), preset="deepseek")
    assert r.hidden is False
    assert "无需覆盖" in r.basis
    assert routing.producer_family(
        moment=cn(SATURDAY, 10), data=config("deepseek")
    ) == ("deepseek")
    assert (
        routing.hidden_valley_preset(moment=cn(SATURDAY, 10), data=config("deepseek"))
        == ""
    )


def test_routing_disabled_face_still_prints_factory_peak():
    """``enabled=false`` 关闭面：维持原预设，但判据口径照说（否则用户以为判定坏了）。"""
    r = reading(cn(SATURDAY, 10), enabled=False)
    assert r.hidden is False
    assert "enabled=false" in r.basis
    assert "09:00–12:00" in r.basis and "14:00–18:00" in r.basis


def test_illegal_routing_keys_fall_back_loudly():
    """非法键逐键回落出厂 + 警告（静默回落＝「用户改了却没生效」）；合法键照常生效。"""
    f = routing.face(
        config(
            valley_preset="doubao",
            leadtime_s=99999,
            peak=[[9, 12], [25, 3]],
            jury_panel=[],
        )
    )
    assert f.valley_preset == routing.DEFAULT_VALLEY_PRESET
    assert f.leadtime_s == routing.DEFAULT_LEADTIME_S
    assert f.peak == ((9, 12),), "非法条目丢弃、合法条目保留"
    assert f.jury_panel == routing.DEFAULT_JURY_PANEL
    assert len(f.warnings) == 4, f.warnings


def test_all_illegal_peak_windows_fall_back_to_factory():
    """peak 全非法 → 回落出厂双峰段（不能让「配错」读成「全天峰」或「全天谷」）。"""
    f = routing.face({"model": {"peak": [[12, 9], ["x", 3], [7]]}})
    assert f.peak == routing.DEFAULT_PEAK_WINDOWS
    assert len(f.warnings) == 4  # 三条丢弃 + 一条「无合法区间」整体回落
    assert any("无合法区间" in w for w in f.warnings)


def test_non_table_routing_block_warns_and_uses_factory():
    f = routing.face({"routing": "enabled = true"})
    assert f.enabled is True
    assert f.peak == routing.DEFAULT_PEAK_WINDOWS
    assert any("不是表" in w for w in f.warnings)


# ---------------------------------------------------------------------------
# ② 切换时机（规格 2）：早切 1 小时、即切、单跳在途不换模型
# ---------------------------------------------------------------------------
def test_switch_moments_on_workday_include_early_entry():
    """工作日切换时刻＝峰段起点**减提前量**（入峰）+ 峰段终点（入谷），按时间排序。"""
    moments = routing.switch_moments(WORKDAY, routing.Face())
    assert [(t.strftime("%H:%M"), label) for t, label in moments] == [
        ("08:00", "入峰"),
        ("12:00", "入谷"),
        ("13:00", "入峰"),
        ("18:00", "入谷"),
    ]


def test_leadtime_moves_entry_back_by_one_hour():
    """08:00 与 13:00＝进峰前一小时：这一跳会跨过峰界，所以此刻就得切回原预设。"""
    assert reading(cn(WORKDAY, 7, 59)).hidden is True
    for hour, minute in ((8, 0), (8, 30), (13, 0), (13, 59)):
        r = reading(cn(WORKDAY, hour, minute))
        assert r.hidden is False, f"{hour:02d}:{minute:02d} 应已提前切回"
        assert "已提前切回" in r.basis


def test_leadtime_is_configurable():
    """提前量改 0 → 08:00 仍在谷里（早切是可调参数，不是硬编码教条）。"""
    assert reading(cn(WORKDAY, 8, 0), leadtime_s=0).hidden is True
    assert reading(cn(WORKDAY, 8, 0), leadtime_s=3600).hidden is False


def test_peak_right_edges_switch_immediately():
    """12:00 与 18:00＝峰段右端点（半开区间）：跨过即入谷，提前量不参与。"""
    assert reading(cn(WORKDAY, 12, 0)).hidden is True
    assert reading(cn(WORKDAY, 18, 0)).hidden is True
    assert reading(cn(WORKDAY, 17, 59)).hidden is False


def test_weekend_has_no_switch_moments_of_its_own():
    """周末自身不产生切换点：周五 18:00 那次入谷已经把状态切过去了。

    任务书把「周末/节假日起点」列为切换时机，实现里却不出现这一条——原因就在这：状态机
    已在 18:00 切完，跨零点没有状态变化。下次切换因此指向周一 08:00（入峰，含提前量）。
    """
    assert routing.switch_moments(SATURDAY, routing.Face()) == []
    when, mode, warn = routing.next_switch(cn(SATURDAY, 20), routing.Face())
    assert (when.strftime("%m-%d %H:%M"), mode) == ("10-19 08:00", "入峰")
    assert warn == ()


def test_next_switch_directions_around_lunch():
    """午前一小时预告「入谷」，午后预告「入峰」：mirror 的下次切换方向串靠这个。"""
    assert routing.next_switch(cn(WORKDAY, 11), routing.Face())[:2] == (
        cn(WORKDAY, 12),
        "入谷",
    )
    assert routing.next_switch(cn(WORKDAY, 12, 30), routing.Face())[:2] == (
        cn(WORKDAY, 13),
        "入峰",
    )
    after = routing.next_switch(cn(WORKDAY, 18, 30), routing.Face())
    assert (after[0].strftime("%m-%d %H:%M"), after[1]) == ("10-15 08:00", "入峰")


def test_note_switch_first_reading_only_sets_baseline(iso_state):
    """首次读数只立基线不出声、不写决策日志（drain 一跳一新进程，否则每跳都虚报一次）。"""
    assert routing.note_switch(moment=cn(WORKDAY, 10)) == ""
    assert journal.read_records() == []
    raw = json.loads(routing.ROUTING_STATE_PATH.read_text(encoding="utf-8"))
    assert raw["mode"] == "mimo-原样"
    assert raw["switched_at"] == "", "立基线没有「切换」这件事，mirror 不得报出假切换"
    assert routing.last_switch_line().startswith("尚未跨过峰谷界")


def test_note_switch_records_crossing(iso_state):
    """跨峰谷界：一行文案 + 决策日志 kind=routing + 上次切换读数 + run 日志。"""
    routing.note_switch(moment=cn(WORKDAY, 10))
    run_log = iso_state / "devops-runs" / "run.log"
    line = routing.note_switch(moment=cn(SATURDAY, 20), run_log=run_log)
    assert "路由切换" in line and "mimo-原样 → deepseek-隐性" in line
    assert line in run_log.read_text(encoding="utf-8")

    records = journal.read_records()
    assert len(records) == 1 and records[0]["kind"] == routing.JOURNAL_KIND
    assert records[0]["source"] == "routing"
    assert "只影响新派发，在途单跳不换模型" in records[0]["reason"]
    at = routing.state(moment=cn(SATURDAY, 20), data=config()).at
    assert routing.last_switch_line() == f"{at} 由 mimo-原样 切至 deepseek-隐性"
    assert routing.note_switch(moment=cn(SATURDAY, 21)) == "", "同一状态不重复出声"
    assert len(journal.read_records()) == 1


def test_note_switch_journal_failure_does_not_break_dispatch(
    iso_state, monkeypatch, capsys
):
    """观察面写不进去不许炸派发（fail-open 红线）；但基线不推进——下一跳重试同一次切换。"""
    routing.note_switch(moment=cn(WORKDAY, 10))

    def boom(record):
        raise OSError("disk gone")

    monkeypatch.setattr(journal, "append", boom)
    line = routing.note_switch(moment=cn(SATURDAY, 20))
    assert "路由切换" in line, "告警照说，只是留痕缺失"
    assert routing.note_switch(moment=cn(SATURDAY, 21)) != "", (
        "没落成就还会再出声（不吞跨越）"
    )
    assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# ③ 隐性态换血：主席席链与仲裁团面板（规格 3）
# ---------------------------------------------------------------------------
def _seed_presets(state):
    set_presets(
        state,
        mimo={"main": [MIMO_MAIN, BUILTIN], "pro": ["seat-mimo-pro", BUILTIN]},
        qwen={"main": [QWEN_MAIN, BUILTIN], "pro": ["seat-qwen-pro", BUILTIN]},
    )


def test_daytime_chain_follows_configured_preset(iso_dispatch):
    """峰段基线：main/pro 各取本预设席（隐性态不许在峰时偷偷改任何东西）。"""
    _seed_presets(iso_dispatch)
    reg = registry.Registry.load()
    assert reg.model_chain("main") == [MIMO_MAIN, BUILTIN]
    assert reg.model_chain("pro") == ["seat-mimo-pro", BUILTIN]
    write_preset("qwen")
    assert reg.model_chain("main") == [QWEN_MAIN, BUILTIN]


def test_hidden_mode_chain_and_panel_follow_valley(iso_dispatch, valley_clock):
    """谷时（时钟已搬到周六夜）：main 取 deepseek 席，**pro 保持原预设席**，面板换固定三席。"""
    _seed_presets(iso_dispatch)
    write_preset("qwen")
    reg = registry.Registry.load()
    assert reg.model_chain("main") == [DS_UUID, DS_BUILTIN]
    assert reg.model_chain("pro") == ["seat-qwen-pro", BUILTIN], (
        "pro 席不被时间覆盖（用户终裁 2026-10-13：隐性态下 pro 保持原席，提高工作质量）"
    )
    assert routing.hidden_valley_preset() == "deepseek"
    assert routing.producer_family() == "deepseek"
    assert routing.jury_panel(("glm", "kimi", "qwen")) == ("glm", "kimi", "mimo")


def test_jury_panel_three_cases(iso_state):
    """终裁 v4 矩阵（用户 2026-10-13 定格）：隐性态固定 GLM+KIMI+Mimo，非隐性取预设自带。"""
    peak, valley = cn(WORKDAY, 10), cn(SATURDAY, 20)
    # 隐性态不随配置预设漂：qwen/mimo 两个被覆盖预设同答案（第三席统一 Mimo）
    assert routing.jury_panel(
        ("glm", "kimi", "mimo"), moment=valley, data=config("qwen")
    ) == ("glm", "kimi", "mimo")
    assert routing.jury_panel(
        ("glm", "kimi", "qwen"), moment=valley, data=config("mimo")
    ) == ("glm", "kimi", "mimo")
    # 峰段（非隐性）按预设自带矩阵：mimo→glm+kimi+qwen
    assert routing.jury_panel(
        ("glm", "kimi", "qwen"), moment=peak, data=config("mimo")
    ) == ("glm", "kimi", "qwen")
    # deepseek 预设自己在谷时不构成覆盖（被评方＝预设本身），面板按预设自带
    assert routing.jury_panel(
        ("glm", "kimi", "mimo"), moment=valley, data=config("deepseek")
    ) == ("glm", "kimi", "mimo")
    # 隐性态面板整副可配（[routing] jury_panel，终裁矩阵的机器落点）
    assert routing.jury_panel(
        ("glm", "kimi", "mimo"),
        moment=valley,
        data=config("qwen", jury_panel=["kimi", "glm", "mimo"]),
    ) == ("kimi", "glm", "mimo")


def test_dispatch_in_valley_uses_byok_deepseek_seat(
    iso_dispatch, monkeypatch, valley_clock
):
    """派发实测：谷时 ``-m`` 落 BYOK DeepSeek UUID（首席消歧，不烧内置同名席）。"""
    _seed_presets(iso_dispatch)
    write_preset("qwen")
    calls = fake_headless(
        monkeypatch,
        lambda n, cmd: envelope(OK_BODY, is_error=False, stop="end_turn", rc=0),
    )
    o = dispatch.do_dispatch("quotamember", text="谷时开工", quiet=True, no_checks=True)
    assert o.kind == "result"
    assert cmd_model(calls[0]) == DS_UUID
    assert ledger.read_all()[-1].model == DS_UUID, "台账 model 即谷时落点证据"


def test_inflight_hop_keeps_its_model_when_clock_crosses(iso_dispatch, monkeypatch):
    """单跳在途不换模型（规格 2）：这一跳**跑的过程中**时钟跨过峰谷界，落点不许变。

    判据不是「代码里没再读一次时钟」（那读的是链副本），而是端到端：argv 已把 ``-m``
    钉死、台账记的就是那个值；时钟搬谷只改**下一跳**的读数。
    """
    _seed_presets(iso_dispatch)
    write_preset("qwen")
    set_routing_clock(monkeypatch, cn(WORKDAY, 10))  # 起跑在峰段
    calls: list[list[str]] = []

    def run(cmd, cwd=None, env=None, timeout_s=None, on_spawn=None):
        calls.append(cmd)
        if len(calls) == 1:
            # 这一跳正在跑——此刻墙钟跨过界（入谷）也不许回头改本跳模型
            set_routing_clock(monkeypatch, cn(SATURDAY, 20))
        return envelope(OK_BODY, is_error=False, stop="end_turn", rc=0)

    monkeypatch.setattr(dispatch, "run_headless", run)
    o = dispatch.do_dispatch(
        "quotamember", text="峰段起跑的一跳", quiet=True, no_checks=True
    )
    assert o.kind == "result"
    assert cmd_model(calls[0]) == QWEN_MAIN, "在途单跳不许被谷时读数改模型"
    assert ledger.read_all()[-1].model == QWEN_MAIN
    assert routing.state().hidden is True, "界已跨过，否则这条用例什么都没测到"
    o2 = dispatch.do_dispatch("quotamember", text="下一跳", quiet=True, no_checks=True)
    assert o2.kind == "result"
    assert cmd_model(calls[1]) == DS_UUID, "覆盖只作用于新派发"
    assert "路由切换" in o2.route, "跨越必须由下一跳的切换记录出声"


def test_minimum_chain_is_not_hijacked_by_valley(
    iso_dispatch, monkeypatch, valley_clock
):
    """minimum 独立链不吃时间覆盖（规格 5）：压测席永远走自己的链。"""
    _seed_presets(iso_dispatch)
    set_models(iso_dispatch, minimum="min-seat")
    calls = fake_headless(
        monkeypatch,
        lambda n, cmd: envelope(OK_BODY, is_error=False, stop="end_turn", rc=0),
    )
    dispatch.do_dispatch(
        "quotamember",
        text="压测",
        quiet=True,
        no_checks=True,
        model_tier="minimum",
    )
    assert cmd_model(calls[0]) == "min-seat"
    assert registry.Registry.load().model_chain("minimum") == ["min-seat"]


def test_valley_status_lines_explain_the_override(iso_registry, valley_clock):
    """status 的预设段与覆盖同屏：只报型号不报「为什么 mimo 烧 DeepSeek」就是误导。"""
    lines = "\n".join(presets.status_lines(registry.Registry.load()))
    assert "谷时隐性态生效" in lines
    assert DS_UUID in lines or DS_BUILTIN in lines
    assert "仲裁团构成：glm+kimi+mimo" in lines
    assert "隐性态固定三席" in lines
    assert "隐性态不覆盖" in lines, "pro 席不被时间覆盖这件事必须与读数同屏"
    assert "Mimo-V2.6-Pro" in lines, "pro 席仍显示原席链（终裁：pro 保原席）"


# ---------------------------------------------------------------------------
# ④ 减员运行（终裁 2026-10-13：零 TokenPlan、缺 Key 双面告警、同族回退不上场）
# ---------------------------------------------------------------------------


def _seat(plan, name):
    return next(s for s in plan["seats"] if s.name == name)


#: mimo 席专有的 Key 变量名（永不 setenv → 恒为「缺 Key」态；glm/kimi 共用 SEAT_KEY_ENV）。
MIMO_KEY = "PYSCI_JURY_MIMO_KEY"


def _panel_with_missing_mimo():
    """glm+kimi（Key 就位）+ mimo（缺 Key）的三席配置。"""
    return [
        "glm",
        "kimi",
        {
            "name": "mimo",
            "model": "Mimo-V2.6-Pro",
            "base_url": "https://stub.invalid/mimo/v1",
            "key_env": MIMO_KEY,
        },
    ]


def test_missing_key_seat_runs_two_seat_court_with_dual_alert(
    iso_jury, monkeypatch, valley_clock
):
    """mimo 席缺 Key → 标 unavailable、本庭 2 席运行，**调用者面与读数面同文案告警**。"""
    set_jury_panel(registry.REGISTRY_PATH, _panel_with_missing_mimo())
    monkeypatch.delenv(MIMO_KEY, raising=False)
    monkeypatch.setenv(SEAT_KEY_ENV, "sk-test")
    plan = jury.seat_readiness(jury.load_config())
    mimo = _seat(plan, "mimo")
    assert mimo.usable is False and mimo.reason == f"缺 {MIMO_KEY}", mimo.reason
    assert [s.name for s in plan["voting"]] == ["glm", "kimi"], "减员照常开庭"
    note = jury._absence_note(plan)
    assert note and "本庭 2 席运行" in note and "twin" in note, note
    assert note in jury.readiness_summary(plan), "读数面（mirror/status）必见"
    assert note in jury._court_alert(plan), "调用者面（Run.message/留痕）必见"


def test_same_family_fallback_is_blocked_and_alerted(
    iso_jury, monkeypatch, valley_clock
):
    """被评模型（DeepSeek）同族的回退席不上场：宁可 2 席运行，也不让被评方进庭当裁判。

    否则「主席缺席 → 回退席顶上」会在谷时把整案变成 §4 declined——该降级的案子被误判成
    拒开庭，且那恰是隐性 DeepSeek 要开的庭。
    """
    set_jury_panel(
        registry.REGISTRY_PATH,
        _panel_with_missing_mimo(),
        fallback={
            "enabled": True,
            "name": "siliconflow",
            "model": "deepseek-ai/DeepSeek-V4-Pro",
            "base_url": "https://stub.invalid/ds/v1",
            "key_env": SEAT_KEY_ENV,
        },
    )
    monkeypatch.delenv(MIMO_KEY, raising=False)
    monkeypatch.setenv(SEAT_KEY_ENV, "sk-test")  # 回退席 Key 就位也照样挡
    plan = jury.seat_readiness(jury.load_config())
    assert plan["producer"] == "deepseek"
    assert plan["fallback_used"] is False
    assert "同族" in plan["fallback_blocked"] and "§4" in plan["fallback_blocked"]
    assert [s.name for s in plan["voting"]] == ["glm", "kimi"]
    note = jury._absence_note(plan)
    assert "回退席不上场" in note and "同族" in note, note
    assert note in jury._court_alert(plan) and note in jury.readiness_summary(plan)


def test_other_family_fallback_covers_without_absence_alert(
    iso_jury, monkeypatch, valley_clock
):
    """非同族回退席照常顶上（回满三席）：报 FALLBACK_NOTICE，不再报「缺席」。"""
    set_jury_panel(
        registry.REGISTRY_PATH,
        _panel_with_missing_mimo(),
        fallback={
            "enabled": True,
            "name": "kimi-backup",
            "model": "kimi-k2.6",
            "base_url": "https://stub.invalid/backup/v1",
            "key_env": SEAT_KEY_ENV,
        },
    )
    monkeypatch.delenv(MIMO_KEY, raising=False)
    monkeypatch.setenv(SEAT_KEY_ENV, "sk-test")
    plan = jury.seat_readiness(jury.load_config())
    assert plan["fallback_used"] is True and len(plan["voting"]) == 3
    assert jury._absence_note(plan) == "", "回退补上了就不叫缺席（免得双面自相矛盾）"
    assert jury.FALLBACK_NOTICE in jury._court_alert(plan)


def test_panel_absence_is_a_visible_prompt(iso_jury, valley_clock):
    """第三席家族在 ``jury.seats`` 里没有配置席 → 读数面点名催办（不猜端点不猜型号）。"""
    set_jury_panel(registry.REGISTRY_PATH, ["glm", "kimi"])
    line = jury.readiness_summary(jury.seat_readiness(jury.load_config()))
    assert "面板缺席：mimo" in line


# ---------------------------------------------------------------------------
# ⑤ 观测面（规格 4）：mirror 概览「路由」项 + orch status 路由段
# ---------------------------------------------------------------------------
def test_mirror_status_has_routing_line_with_next_switch(
    iso_state, iso_registry, monkeypatch, capsys
):
    """mirror 概览「路由」项：当前模式 + 判据读数 + 下次切换时间与方向 + 上次切换。"""
    set_routing_clock(monkeypatch, cn(WORKDAY, 11))
    mirror.cmd_status(Namespace())
    out = capsys.readouterr().out
    assert "[路由]" in out
    assert "〔mimo〕" in out, "峰段读数就是配置预设本身"
    assert "下次 10-14 12:00 入谷" in out
    assert "上次切换" in out


def test_mirror_status_routing_line_names_hidden_mode(
    iso_state, iso_registry, monkeypatch, capsys
):
    """谷时 mirror 读数点名「隐性 DeepSeek」——用户一天看一次的入口要能看出今天烧谁。"""
    set_routing_clock(monkeypatch, cn(SATURDAY, 20))
    mirror.cmd_status(Namespace())
    out = capsys.readouterr().out
    assert "〔隐性 DeepSeek〕" in out
    assert "下次 10-19 08:00 入峰" in out


def test_orch_status_has_routing_section_with_last_switch(
    iso_state, iso_registry, monkeypatch, capsys
):
    """``orch status`` 的「路由模式」段：预设/生效路由/判据/下次切换/上次切换/峰段口径。"""
    set_routing_clock(monkeypatch, cn(WORKDAY, 11))
    routing.note_switch(moment=cn(WORKDAY, 11))
    set_routing_clock(monkeypatch, cn(SATURDAY, 20))
    routing.note_switch(moment=cn(SATURDAY, 20))
    orch.cmd_status(Namespace())
    out = capsys.readouterr().out
    assert "路由模式" in out
    assert "生效路由 隐性 DeepSeek" in out
    assert "下次切换：10-19 08:00 入峰" in out
    assert "峰段口径 09:00–12:00、14:00–18:00｜提前量 60 分钟" in out
    assert "由 mimo-原样 切至 deepseek-隐性" in out


def test_routing_warnings_appear_on_both_reading_surfaces(
    iso_state, iso_registry, monkeypatch, capsys
):
    """保守不切换的告警必须与读数同屏（否则「没切」看起来像「路由没在工作」）。"""
    set_routing_clock(monkeypatch, cn(date(2027, 5, 10), 10))
    lines = routing.status_lines()
    assert any("不覆盖 2027" in line for line in lines), lines
    mirror.status_lines()  # 不炸（观察面逐项容错）


def test_routing_config_on_toml_is_persisted_surface(iso_state):
    """``[routing]`` 块真源就是 data/pysci_settings.toml：改文件即改判据（配置面闭环）。"""
    write_routing(enabled=False, valley_preset="qwen", leadtime_s=1800)
    f = routing.face()
    assert (f.enabled, f.valley_preset, f.leadtime_s) == (False, "qwen", 1800)
    assert f.warnings == ()
    write_routing(peak=[[10, 13]], jury_panel=["kimi"])
    f2 = routing.face()
    assert f2.peak == ((10, 13),) and f2.jury_panel == ("kimi",)


# ---------------------------------------------------------------------------
# [routing] 并入 [model]（用户 2026-10-13「模型类设置集中一处」）
# ---------------------------------------------------------------------------


def _isolate_settings(tmp_path, monkeypatch):
    """把 ``settings.SETTINGS_PATH`` 指到 tmp——**不隔离就会写掉所在树的真配置**。

    本组新用例直接造配置文件；漏了这一步，测试会把 worktree 里的
    ``data/pysci_settings.toml`` 改成用例夹具（实测：跑完一轮，真配置变成
    「[model] preset + enabled / [routing] leadtime_s」的半迁移壳）。
    """
    from pysci.skills.orchestration.tools import settings as _settings

    monkeypatch.setattr(_settings, "SETTINGS_PATH", tmp_path / "pysci_settings.toml")
    return _settings


def test_routing_keys_are_read_from_model_section(tmp_path, monkeypatch):
    """新位置：路由键住在 [model] 节内（合并后的唯一形状）→ 零迁移告警。"""
    _isolate_settings(tmp_path, monkeypatch)
    write_routing(enabled=False, peak=[[10, 11]])
    f = routing.face()
    assert f.enabled is False and f.peak == ((10, 11),)
    assert not any("已并入" in w for w in f.warnings)


def test_legacy_routing_section_is_read_once_with_migration_warning(
    tmp_path, monkeypatch
):
    """旧位置兼容读：仍写 [routing] 的文件照常生效，但**出声**请迁移（手写配置不该硬失败）。"""
    _settings = _isolate_settings(tmp_path, monkeypatch)
    _settings.SETTINGS_PATH.write_text(
        "[model]"
        + chr(10)
        + 'preset = "mimo"'
        + chr(10)
        + chr(10)
        + "[routing]"
        + chr(10)
        + 'valley_preset = "qwen"'
        + chr(10)
        + "leadtime_s = 60"
        + chr(10),
        encoding="utf-8",
    )
    f = routing.face()
    assert f.valley_preset == "qwen" and f.leadtime_s == 60, "旧位置的键仍须生效"
    assert any("已并入 [model]" in w and "valley_preset" in w for w in f.warnings)


def test_partial_migration_does_not_silently_drop_legacy_keys(tmp_path, monkeypatch):
    """**半迁移**：enabled 已进 [model]、leadtime_s 还在 [routing] → 两者都要生效。

    逐键回落（而非整块二选一）就是为了防这一格：按整块判的话，落在旧位置的那几个键
    会被静默丢掉——用户以为改了、实际没生效。"""
    _settings = _isolate_settings(tmp_path, monkeypatch)
    _settings.SETTINGS_PATH.write_text(
        "[model]"
        + chr(10)
        + 'preset = "mimo"'
        + chr(10)
        + "enabled = false"
        + chr(10)
        + chr(10)
        + "[routing]"
        + chr(10)
        + "leadtime_s = 120"
        + chr(10),
        encoding="utf-8",
    )
    f = routing.face()
    assert f.enabled is False and f.leadtime_s == 120, (
        "逐键回落：旧位置的键不许被静默丢掉"
    )
    assert any("leadtime_s" in w for w in f.warnings)
