"""用户任务书：存储/状态机/程序化面/唤醒（README §12，M3）。

落盘用**临时 data 根**（monkeypatch ``taskbooks.TASKBOOKS_DIR`` 与 ``pysci.paths.DATA_ROOT``
——后者是 ``assert_within_data`` 防 stray 护栏的判据源），绝不碰真实 ``data/taskbooks/``。
"""

from __future__ import annotations

import asyncio
import json

import pytest
from textual.widgets import Input, TextArea

from pysci import paths as _paths
from pysci.skills.orchestration.tools import events, orch, taskbooks, watch
from pysci.skills.orchestration.tools.tui import app as tui


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture
def tbdir(tmp_path, monkeypatch):
    data = tmp_path / "data"
    d = data / "taskbooks"
    monkeypatch.setattr(taskbooks, "TASKBOOKS_DIR", d)
    monkeypatch.setattr(_paths, "DATA_ROOT", data)  # 防 stray 护栏的判据源
    return d


def _make(title="大任务", body="正文若干", **kw) -> dict:
    return taskbooks.create(title, body, **kw)


# ---------------------------------------------------------------------------
# ① 存储与元数据
# ---------------------------------------------------------------------------
def test_create_writes_two_files_with_metadata(tbdir):
    st = _make(title="重做图管线", body="第一行\n第二行")
    assert st["id"].startswith("tb-") and st["status"] == "new"
    md = (tbdir / f"{st['id']}.taskbook.md").read_text(encoding="utf-8")
    state = json.loads((tbdir / f"{st['id']}.state.json").read_text(encoding="utf-8"))
    for key in ("id", "ts", "title", "status", "source"):
        assert f"{key}:" in md, key
    assert state["source"] == "mirror-tui" and state["plan_id"] == ""
    assert taskbooks.read_body(st["id"]) == "第一行\n第二行"


def test_write_outside_data_root_is_refused(tmp_path, monkeypatch):
    """防 stray 护栏：任务书必须落在 data/ 里（§3.4 / basic.md §5）。"""
    monkeypatch.setattr(taskbooks, "TASKBOOKS_DIR", tmp_path / "elsewhere")
    with pytest.raises(ValueError):
        taskbooks.create("x", "y")


def test_id_shape_and_source_llm(tbdir):
    st = _make(title="Hello World!", source="llm")
    assert st["id"].split("-")[0] == "tb" and "hello-world" in st["id"]
    assert st["source"] == "llm"


# ---------------------------------------------------------------------------
# ② 状态机
# ---------------------------------------------------------------------------
def test_legal_and_illegal_transitions(tbdir):
    st = _make()
    tid = st["id"]
    assert taskbooks.link_plan(tid, "p1")["status"] == "planning"
    assert taskbooks.transition(tid, "active")["status"] == "active"
    assert taskbooks.transition(tid, "delivered")["status"] == "delivered"
    with pytest.raises(ValueError):  # delivered → planning 非法
        taskbooks.transition(tid, "planning")
    assert taskbooks.accept(tid)["status"] == "accepted"
    with pytest.raises(ValueError):  # accepted 终态
        taskbooks.transition(tid, "new")


def test_reject_records_note_and_returns_to_new(tbdir):
    """success_criteria：**rejected 带批语回 new**，批语持久保留于 state.json。"""
    tid = _make()["id"]
    taskbooks.link_plan(tid, "p1")
    taskbooks.transition(tid, "active")
    taskbooks.transition(tid, "delivered")
    st = taskbooks.reject(tid, "拆得不对，重来")
    assert st["status"] == "new"  # 返工重拆解
    again = taskbooks.load_state(tid)
    assert again["rejections"][-1]["note"] == "拆得不对，重来"  # 批语持久
    assert again["status"] == "new"
    assert str(again["plan_id"]) in ("p1", "")  # 链接字段不丢


def test_sync_delivered_follows_plan_completion(tbdir, monkeypatch):
    """所链 plan 完成 → 幂等推进 delivered（组长程序化面的联动）。"""
    from pysci.skills.orchestration.tools import plans as _plans

    tid = _make()["id"]
    taskbooks.link_plan(tid, "p-ok")
    monkeypatch.setattr(_plans, "load_state", lambda pid: {"completed_at": "T"})
    assert taskbooks.sync_delivered() == [tid]
    assert taskbooks.load_state(tid)["status"] == "delivered"
    assert taskbooks.sync_delivered() == []  # 幂等（已 delivered，不再重复推进）


def test_sync_delivered_leaves_incomplete_plans_alone(tbdir, monkeypatch):
    from pysci.skills.orchestration.tools import plans as _plans

    tid = _make()["id"]
    taskbooks.link_plan(tid, "p-run")
    monkeypatch.setattr(_plans, "load_state", lambda pid: {})
    assert taskbooks.sync_delivered() == []
    assert taskbooks.load_state(tid)["status"] == "planning"


def test_sections_split_three_ways(tbdir):
    a = _make(title="a")["id"]
    b = _make(title="b")["id"]
    c = _make(title="c")["id"]
    taskbooks.link_plan(a, "p")  # a: planning → 现有大任务
    for tid in (b, c):  # b/c: new → active → delivered
        taskbooks.transition(tid, "active")
        taskbooks.transition(tid, "delivered")
    taskbooks.accept(c)  # c: → 历史
    secs = taskbooks.sections()
    assert a in [d["id"] for d in secs["open"]]
    assert b in [d["id"] for d in secs["review"]]
    assert c in [d["id"] for d in secs["history"]]


def test_rejected_item_appears_in_history_with_note(tbdir):
    tid = _make()["id"]
    taskbooks.transition(tid, "active")
    taskbooks.transition(tid, "delivered")
    taskbooks.reject(tid, "拆得不对")
    assert tid in [d["id"] for d in taskbooks.sections()["history"]]  # 带批语进历史


# ---------------------------------------------------------------------------
# ③ 组长程序化面（orch taskbooks）
# ---------------------------------------------------------------------------
def test_orch_taskbooks_list_show_link(tbdir, capsys):
    tid = _make(title="目标大任务")["id"]
    assert orch.main(["taskbooks", "list"]) == 0
    out = capsys.readouterr().out
    assert tid in out and "现有大任务" in out
    assert orch.main(["taskbooks", "show", tid]) == 0
    assert "目标大任务" in capsys.readouterr().out
    assert orch.main(["taskbooks", "link", tid, "--plan", "p9"]) == 0
    assert taskbooks.load_state(tid)["plan_id"] == "p9"
    assert orch.main(["taskbooks", "show", "tb-none"]) == 2


# ---------------------------------------------------------------------------
# ④ TASKBOOK-NEW 唤醒（M5 收编进统一事件层 §3.2）
# ---------------------------------------------------------------------------
def test_taskbook_new_is_in_priority():
    """事件名仍在 watch 优先级表（多事件同刻排序）与统一事件层词表（§3.2）。"""
    assert "TASKBOOK-NEW" in watch.PRIORITY
    assert "TASKBOOK-NEW" in events.TYPES
    assert "TASKBOOK-NEW" in events.STREAM_TYPES


def test_create_emits_taskbook_new_to_stream(tbdir):
    """落盘即发 TASKBOOK-NEW（生产者直发统一事件流；M5 收编，替代旧基线差分）。"""
    st = _make(title="唤醒我")
    evs = events.read_all()
    assert [e["event"] for e in evs] == ["TASKBOOK-NEW"]
    assert evs[0]["subject"] == st["id"]


def test_watch_fires_on_stream_taskbook_new(tbdir):
    """旧差分路径已删（收编）：watch 改从统一事件流按 ``events_seq`` 水位读出即唤醒。"""
    _make(title="z")
    evs = watch.detect_events({"events_seq": 0}, {"events_seq": events.latest_seq()})
    assert [e["event"] for e in evs] == ["TASKBOOK-NEW"]
    assert evs[0]["subject"].startswith("tb-") and "taskbooks" in evs[0]["action"]
    # 水位线未推进（无新事件）→ 不报
    assert watch.detect_events({"events_seq": 1}, {"events_seq": 1}) == []


# ---------------------------------------------------------------------------
# ⑤ TUI 面板（Pilot）：投递 + 批复
# ---------------------------------------------------------------------------
def _deliver(tid: str) -> None:
    taskbooks.transition(tid, "active")
    taskbooks.transition(tid, "delivered")


def test_pilot_new_taskbook_lands_on_disk(tbdir):
    """success_criteria：Pilot 驱动 TUI 新增 → data/taskbooks/ 落盘两文件、元数据齐全。"""

    async def _go():
        app = tui.PysciTUI()
        async with app.run_test() as pilot:
            await pilot.press("t")
            assert app.view == "taskbooks"
            await pilot.press("n")
            assert app.view == "tb-new"
            app.query_one("#tb-new-title-input", Input).value = "TUI 投递的大任务"
            app.query_one("#tb-new-body", TextArea).text = "正文甲\n正文乙"
            await pilot.click("#tb-new-submit")
            assert app.last_created.startswith("tb-")
            # ③ 确认即关页回列表（backlog 20261015-230248）：消息落列表面
            assert app.view == "taskbooks"
            assert "已投递" in app.tb_msg

    _run(_go())
    st = taskbooks.list_all()
    assert len(st) == 1
    assert st[0]["title"] == "TUI 投递的大任务" and st[0]["source"] == "mirror-tui"
    assert st[0]["priority"] == "normal"  # 修订⑦：默认中
    assert taskbooks.read_body(st[0]["id"]) == "正文甲\n正文乙"
    assert (tbdir / f"{st[0]['id']}.taskbook.md").exists()
    assert (tbdir / f"{st[0]['id']}.state.json").exists()


def test_pilot_accept_and_reject_selected(tbdir):
    """批复 Pilot——accept 终态；reject 走**三段式**（模式→优先级→批语）生成返工任务书。"""
    a = _make(title="甲")["id"]
    b = _make(title="乙")["id"]
    for tid in (a, b):
        _deliver(tid)
    seen: dict[str, str] = {}

    async def _go():
        app = tui.PysciTUI()
        async with app.run_test() as pilot:
            await pilot.press("t")
            assert len(app.tb_items) == 2
            seen["first"] = str(app.tb_items[0]["id"])
            seen["second"] = str(app.tb_items[1]["id"])
            await pilot.press("a")  # 接受选中（第 0 项）
            assert "accepted" in app.tb_msg and seen["first"] in app.tb_msg
            await pilot.press("down")  # 重建后第 1 项
            await pilot.press("r")  # 开三段式驳回面板
            assert app.view == "tb-reject"
            app._set_radio("#tb-reject-mode", 1)  # 无监察改进
            app._set_radio("#tb-reject-priority", 2)  # 低
            app.query_one("#tb-note", TextArea).text = "拆得不对"
            await pilot.click("#tb-reject-submit")
            assert app.last_rework.startswith("tb-")
            assert "rework" in app.tb_reject_msg.lower() or "返工" in app.tb_reject_msg
            seen["rework"] = app.last_rework

    _run(_go())
    by_id = {d["id"]: d for d in taskbooks.list_all()}
    assert by_id[seen["first"]]["status"] == "accepted"
    assert by_id[seen["second"]]["status"] == "rejected"  # 原项终态留档（不再回 new）
    assert by_id[seen["second"]]["rejections"][-1]["note"] == "拆得不对"
    rw = by_id[seen["rework"]]
    assert rw["status"] == "new" and rw["source"] == "mirror-tui"
    assert rw["rework_of"] == seen["second"] and rw["rework_round"] == 1
    assert rw["mode"] == "no_inspector" and rw["priority"] == "low"
    assert rw["title"] == "乙（返工第 1 轮）"  # 继承原名 + 轮次后缀（同名返工件可辨）


def test_pilot_taskbook_panel_reachable_and_esc_returns(tbdir):
    async def _go():
        app = tui.PysciTUI()
        async with app.run_test() as pilot:
            await pilot.press("t")
            assert app.view == "taskbooks"
            await pilot.press("escape")
            assert app.view == "menu"

    _run(_go())
