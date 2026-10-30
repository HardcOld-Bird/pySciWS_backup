"""TUI 抛光 c（tui-polish-20261016-polish-c）：文案排版三件。

- **反馈 #6**：三处时长读数（倒计时／距峰／距边界）统一人性化格式（天/小时/分/秒，
  按量级省略前导整零段、**秒段恒显**）＋**统一每秒刷新**（旧实现只刷倒计时卡）；
- **反馈 #7**：渠道回退段改**列表式**（统计行 / 「最近：」小标 / 逐条一行缩进对齐，
  不再是一行「；」长串）；
- **反馈 #9**：任务书家族按 **root 创建序**自动编号（主项显示、子项不带、稳定不漂移）。
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest
from textual.widgets import ListView, Static

from pysci import paths as _paths
from pysci.skills.orchestration.tools import channels, routing, taskbooks
from pysci.skills.orchestration.tools.tui import app as tui
from pysci.skills.orchestration.tools.tui import cards as tui_cards
from pysci.skills.orchestration.tools.tui import modules as tui_modules


def _run(coro):
    return asyncio.run(coro)


async def _open_module(app, pilot, index: int) -> None:
    lv = app.query_one("#menu", ListView)
    lv.index = index
    await pilot.pause()
    await pilot.press("enter")
    await pilot.pause()


# ---------------------------------------------------------------------------
# ① humanize_duration：量级/整点/边界（单点格式化函数的判据面）
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("secs", "want"),
    [
        (0, "0 秒"),
        (27, "27 秒"),
        (59, "59 秒"),
        (60, "1 分 0 秒"),
        (600, "10 分 0 秒"),
        (3540, "59 分 0 秒"),
        (3600, "1 小时 0 分 0 秒"),
        (3661, "1 小时 1 分 1 秒"),
        (86400, "1 天 0 小时 0 分 0 秒"),
        (90061, "1 天 1 小时 1 分 1 秒"),
        # 3550 分 27 秒 = 213027s：3550 分 = 59 小时 10 分 → 2 天 11 小时 10 分 27 秒。
        # （任务书示例写「11 小时 50 分」是笔误：那是 3550+40 分的读数。）
        (213027, "2 天 11 小时 10 分 27 秒"),
    ],
)
def test_humanize_duration_magnitudes(secs, want):
    assert routing.humanize_duration(secs) == want


def test_humanize_seconds_segment_always_present():
    """**秒段恒显**：不许「10 分」与「10 分 0 秒」两可（统一精确到秒）。"""
    for s in (0, 59, 60, 600, 3600, 86400, 213027):
        assert routing.humanize_duration(s).endswith("秒"), s


def test_humanize_negative_carries_sign():
    assert routing.humanize_duration(-27) == "-27 秒"
    assert routing.humanize_duration(-3661) == "-1 小时 1 分 1 秒"


# ---------------------------------------------------------------------------
# ② 三处调用点：同一格式（旧读数「3550 分」「59.3 小时」不再出现）
# ---------------------------------------------------------------------------
def test_countdown_line_is_humanized(monkeypatch):
    monkeypatch.setattr(routing, "countdown_s", lambda _t: 213027)
    line = tui_modules._countdown_line("10-19 08:50", "入峰")
    assert "2 天 11 小时 10 分 27 秒" in line, line
    assert "3550 分" not in line


def test_countdown_line_past_keeps_wording(monkeypatch):
    monkeypatch.setattr(routing, "countdown_s", lambda _t: -3661)
    assert "已过 1 小时 1 分 1 秒" in tui_modules._countdown_line("10-19 08:50", "入峰")


def _fake_reading():
    return SimpleNamespace(next_at="10-19 08:50", next_mode="入峰")


def test_hold_line_within_window_is_humanized(monkeypatch):
    monkeypatch.setattr(routing, "state", lambda **_k: _fake_reading())
    monkeypatch.setattr(routing, "countdown_s", lambda _t: 3540)
    line = tui_modules._hold_line(SimpleNamespace(boundary_hold_s=3600))
    assert "距边界 59 分 0 秒" in line, line
    assert "临近边界将延迟派发" in line


def test_hold_line_outside_window_is_humanized(monkeypatch):
    monkeypatch.setattr(routing, "state", lambda **_k: _fake_reading())
    monkeypatch.setattr(routing, "countdown_s", lambda _t: 213027)
    line = tui_modules._hold_line(SimpleNamespace(boundary_hold_s=600))
    assert "距边界 2 天 11 小时 10 分 27 秒" in line, line
    assert "3550 分" not in line


def test_routing_basis_gap_is_humanized(iso_state):
    """距峰读数：`{gap/3600:.1f} 小时` → 人性化（跨天可读）。"""
    moment = datetime(2026, 10, 17, 0, 30)  # 谷段；出厂峰窗 09–12/14–18
    reading = routing.state(moment=moment)
    assert "距峰" in reading.basis, reading.basis
    assert "8 小时 30 分 0 秒" in reading.basis, reading.basis
    assert "小时）" not in reading.basis.replace("小时 30 分 0 秒）", ""), (
        "旧的 `{:.1f} 小时` 形态应已消失"
    )


# ---------------------------------------------------------------------------
# ③ tick：三处时长读数**每秒一起刷**（且不重建卡控件）
# ---------------------------------------------------------------------------
def test_tick_refreshes_all_three_duration_readings(monkeypatch):
    seq = {"n": 0}

    def _cd() -> str:
        seq["n"] += 1
        return f"  倒计时：距下次切换还有{seq['n']} 秒（模拟 → 入谷）"

    def _sw() -> list[str]:
        return ["开关：模拟", f"路由：距峰 10-19 09:00 还有 {seq['n']} 分 0 秒"]

    def _cf() -> list[str]:
        return [
            "路由配置生效值",
            f"  边界避让：窗口 600s（距边界 {seq['n']} 秒，窗口外不延迟）",
        ]

    monkeypatch.setattr(tui_cards, "countdown_text", _cd)
    monkeypatch.setattr(tui_cards, "switches_lines", _sw)
    monkeypatch.setattr(tui_cards, "routing_config_lines", _cf)

    async def _go():
        app = tui.PysciTUI()
        async with app.run_test(size=(120, 60)) as pilot:
            await pilot.pause()
            await _open_module(app, pilot, 4)  # 系统与路由
            before_ids = [id(c) for c in app.query("#detail-cards > Card")]
            first = (
                app.detail_countdown_text,
                app.detail_switches_text,
                app.detail_config_text,
            )
            app._tick_detail()  # 模拟一秒到达
            await pilot.pause()
            after = (
                app.detail_countdown_text,
                app.detail_switches_text,
                app.detail_config_text,
            )
            after_ids = [id(c) for c in app.query("#detail-cards > Card")]
            rendered = (
                str(app.query_one("#detail-countdown", Static).render()),
                str(app.query_one("#card-switches", Static).render()),
                str(app.query_one("#card-config", Static).render()),
            )
            return first, after, before_ids, after_ids, rendered

    first, after, before_ids, after_ids, rendered = _run(_go())
    assert before_ids == after_ids, "tick 不该重建卡控件（防闪烁）"
    for i, name in enumerate(("倒计时卡", "开关/路由卡", "配置卡")):
        assert first[i] != after[i], f"{name} 未随 tick 刷新"
    n = str(seq["n"])  # 首帧取 1，tick 再取一次（计数器即期望值，不写死数字）
    assert f"还有{n} 秒" in rendered[0], rendered[0]
    assert f"还有 {n} 分 0 秒" in rendered[1], rendered[1]
    assert f"距边界 {n} 秒" in rendered[2], rendered[2]


# ---------------------------------------------------------------------------
# ④ 回退段：逐条一行的列表式（不再是「；」长串、无孤立缩进）
# ---------------------------------------------------------------------------
def test_fallback_lines_are_structured_list(monkeypatch):
    monkeypatch.setattr(
        channels, "summary_head", lambda **_k: "渠道回退：近 7 日 2 起｜累计 3 起"
    )
    monkeypatch.setattr(
        channels,
        "recent_event_lines",
        lambda **_k: [
            "10-13 08:26 main A→B（channel_error）",
            "10-13 10:53 main B→C（channel_error）",
        ],
    )
    lines = tui_cards._fallback_lines()
    assert lines[0] == "回退：渠道回退：近 7 日 2 起｜累计 3 起", lines
    assert lines[1] == "  最近：", lines
    assert [ln[:6] for ln in lines[2:]] == ["    · ", "    · "], lines
    assert "10-13 08:26" in lines[2] and "10-13 10:53" in lines[3]
    assert all("；" not in ln for ln in lines), "不许再有一行「；」长串"
    assert len(lines) == 4, "两项事件 = 统计行 + 小标 + 两行（不是一行长串）"


def test_fallback_lines_empty_window(monkeypatch):
    monkeypatch.setattr(channels, "summary_head", lambda **_k: "（尚无渠道回退事件）")
    monkeypatch.setattr(channels, "recent_event_lines", lambda **_k: [])
    assert tui_cards._fallback_lines() == ["回退：（尚无渠道回退事件）"]


def test_system_arbitration_card_renders_fallback_as_list(iso_state):
    """端到端：真事件流 → 仲裁卡里回退段逐条一行。"""
    now = datetime.now().astimezone()
    for i, (when, frm, to) in enumerate(
        [
            (
                (now - timedelta(hours=2)).isoformat(timespec="seconds"),
                "seat-a",
                "seat-b",
            ),
            (
                (now - timedelta(hours=1)).isoformat(timespec="seconds"),
                "seat-b",
                "seat-c",
            ),
        ]
    ):
        channels.append_event(
            {
                "ts": when,
                "tier": "main",
                "from": frm,
                "from_label": frm,
                "to": to,
                "to_label": to,
                "reason": "channel_error",
            }
        )
    blocks = tui_cards.system_blocks()
    card = next(b for b in blocks if b.get("id") == "card-arbitration")
    lines = [str(ln) for ln in card["lines"]]
    assert any(ln.startswith("回退：渠道回退：近 7 日 2 起") for ln in lines), lines
    assert "  最近：" in lines, lines
    bullets = [ln for ln in lines if ln.startswith("    · ")]
    assert len(bullets) == 2, lines
    assert all("；" not in ln for ln in lines), lines


# ---------------------------------------------------------------------------
# ⑤ 家族编号：按 root 创建序、家族共享、稳定不漂移、只在主项
# ---------------------------------------------------------------------------
T0 = datetime(2026, 10, 17, 9, 0, 0)


@pytest.fixture
def face(tmp_path, monkeypatch):
    data = tmp_path / "data"
    monkeypatch.setattr(taskbooks, "TASKBOOKS_DIR", data / "taskbooks")
    taskbooks.TASKBOOKS_DIR.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(_paths, "DATA_ROOT", data)
    return data


def _mk(title: str, when: datetime) -> dict:
    return taskbooks.create(title, f"{title} 正文", now=when)


def _no_map() -> dict[str, int]:
    return {f["root"]: int(f["no"]) for f in taskbooks.families()}


def test_family_numbers_follow_root_creation_order(face):
    a = _mk("甲", T0)
    b = _mk("乙", T0 + timedelta(minutes=1))
    c = _mk("丙", T0 + timedelta(minutes=2))
    assert _no_map() == {a["id"]: 1, b["id"]: 2, c["id"]: 3}


def test_family_number_shared_by_root_and_rework(face):
    a = _mk("甲", T0)
    r1 = taskbooks.create_rework(
        a["id"], mode="full", note="第一轮", now=T0 + timedelta(minutes=5)
    )
    fams = taskbooks.families()
    assert len(fams) == 1, "本体与返工同族（沿 rework_of 链）"
    fam = fams[0]
    assert fam["no"] == 1 and fam["root"] == a["id"]
    assert {m["id"] for m in fam["members"]} == {a["id"], r1["id"]}


def test_family_numbers_do_not_drift_when_new_family_appears(face):
    a = _mk("甲", T0)
    b = _mk("乙", T0 + timedelta(minutes=1))
    before = _no_map()
    # 更晚创建的家族 → 递增新号；既有编号不动（编号锚 root 创建序，不随列表排序漂）
    c = _mk("丙", T0 + timedelta(hours=3))
    after = _no_map()
    assert after[a["id"]] == before[a["id"]], "既有家族编号漂移了"
    assert after[b["id"]] == before[b["id"]]
    assert after[c["id"]] == max(after.values())


def test_group_row_shows_number_and_sub_rows_do_not(face):
    a = _mk("甲", T0)
    b = _mk("乙", T0 + timedelta(minutes=1))
    taskbooks.create_rework(
        a["id"], mode="full", note="n", now=T0 + timedelta(minutes=9)
    )

    def _pairs(app) -> list[tuple[dict, str]]:
        """行模型 ↔ **已挂载**列表条目（同序；行文本＝行内各格 render 拼接）。"""
        items = list(app.query("#tb-list > ListItem"))
        assert len(items) == len(app.tb_rows), (len(items), len(app.tb_rows))
        return [
            (r, " ".join(str(c.render()) for c in li.query(Static)))
            for r, li in zip(app.tb_rows, items, strict=True)
        ]

    async def _go():
        app = tui.PysciTUI()
        async with app.run_test(size=(120, 45)) as pilot:
            await pilot.pause()
            await pilot.press("t")
            await pilot.pause()
            collapsed = _pairs(app)
            # 展开甲族 → 该族出次级行
            app.query_one("#tb-list", ListView).index = next(
                i for i, r in enumerate(app.tb_rows) if r["root"] == a["id"]
            )
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            expanded = _pairs(app)
            return collapsed, expanded

    collapsed, expanded = _run(_go())
    nos = {r["root"]: r["no"] for r, _ in collapsed if r["kind"] == "group"}
    texts = {r["root"]: txt for r, txt in collapsed if r["kind"] == "group"}
    assert nos[a["id"]] == 1 and nos[b["id"]] == 2, nos
    assert "#1" in texts[a["id"]] and "#2" in texts[b["id"]], texts
    subs = [txt for r, txt in expanded if r["kind"] == "sub"]
    assert subs, "家族应可展开出次级行"
    assert all("#" not in t for t in subs), f"子项不该带编号：{subs}"
