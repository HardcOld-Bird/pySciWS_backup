"""TUI 大改版 s9（task tui-overhaul-20261016-s9-tests）：**跨面一致性钉**（九环收口）。

各环局部自测已随环落地（taskbook_groups／plans_panel／decisions_panel／boards_s7／review_blind
等）；本文件只补**全局缝合判据**，不加新功能：

1. footer 键位矩阵**全视图**（menu 广告 t/j；非 menu 一律不广告；`o` 只在 taskbooks；新增键有实效）；
2. 滚动指示**全覆盖**（`_SCROLL_PANES` 每条的视图打开后位置行在场，赋值 `scroll_y` 立即刷新）；
3. 禁语**全仓钉**（所有可断言读面）；
4. 图标空格**全仓钉**（tools/ 全目录，含 tui/）；
5. 徽标语义回归（plans/system 指纹不建详情、倒计时不进指纹、同 cursor 不写水位）；
6. 导航一致性冒烟（menu 七模块 → Esc 回菜单，含 plans/decisions/review 新分支）；
7. 契约读面重进一致（行模型/展开态/计数经 Esc 往返后与数据层读数一致）。

踩坑纪律：行选择一律走**行模型**（`tb_rows`/`plan_rows`/`decision_rows`），不碰裸下标语义。
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
import yaml
from textual.binding import Binding
from textual.widgets import ListView, Static

from pysci import paths as _paths
from pysci.skills.orchestration.tools import (
    inspectd,
    journal,
    mirror,
    plans,
    taskbooks,
    taste,
)
from pysci.skills.orchestration.tools.tui import app as tui
from pysci.skills.orchestration.tools.tui import modules as tui_modules


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# 数据面（隔离：计划/任务书/taste/mirror/journal；decisions 用替身行）
# ---------------------------------------------------------------------------
_DEC_ROW = {
    "id": "f-s9",
    "ts": "2026-10-16T10:00:00+08:00",
    "class": "judgment",
    "remedy_shape": "restore",
    "title": "s9 决策行",
    "verdict_ref": "J 案卷 x.md",
    "ruling": "裁示：恢复",
    "commit": "abc1234",
    "verify": "stage-b",
    "backlog_status": "done",
    "note": "",
    "attention": False,
    "finding_status": "open",
    "channel": "jury",
}


def _write_plan(pdir: Path, pid: str, *, created: str) -> None:
    pdir.mkdir(parents=True, exist_ok=True)
    (pdir / f"{pid}.plan.yaml").write_text(
        yaml.safe_dump(
            {
                "id": pid,
                "goal": f"目标-{pid}",
                "steps": [
                    {"id": "s1", "member": "sim", "brief": "环节一"},
                    {"id": "s2", "member": "figure", "brief": "环节二"},
                ],
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    (pdir / f"{pid}.state.json").write_text(
        json.dumps(
            {
                "plan_id": pid,
                "created": created,
                "goal": f"目标-{pid}",
                "steps": [
                    {"id": "s1", "status": "done"},
                    {"id": "s2", "status": "queued"},
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _write_taste(tdir: Path, cid: str, *, ts: str, review: dict | None = None) -> None:
    case = tdir / f"{cid}.md"
    case.write_text(
        f"## 问题\n选哪个？（{cid}）\n\n## 选项 A\n甲\n\n## 选项 B\n乙\n",
        encoding="utf-8",
    )
    rec = {
        "ts": ts,
        "case_id": cid,
        "case_file": str(case),
        "case_head": f"选哪个？（{cid}）",
        "choice": "A",
        "confidence": 0.6,
        "reason": "理由",
        "parsed": True,
        "body": "全文",
        "record_kind": "taste",
        "night": False,
        "authority": "",
    }
    if review is not None:
        rec["review"] = review
    taste.write_record(rec)


@pytest.fixture
def s9face(tmp_path, monkeypatch):
    """四面隔离＋造数据：2 plan／返工家族／1 待盲审＋1 已复核／1 决策行。"""
    pdir = tmp_path / "plans"
    monkeypatch.setattr(plans, "PLANS_DIR", pdir)
    _write_plan(pdir, "p-a", created="2026-10-16T11:00:00+08:00")
    _write_plan(pdir, "p-b", created="2026-10-16T09:00:00+08:00")

    data = tmp_path / "data"
    monkeypatch.setattr(taskbooks, "TASKBOOKS_DIR", data / "taskbooks")
    monkeypatch.setattr(_paths, "DATA_ROOT", data)
    root = taskbooks.create("S9 家族", "正文")["id"]
    r1 = taskbooks.create_rework(root, mode="full", note="一轮")["id"]

    tdir = tmp_path / "taste"
    tdir.mkdir()
    monkeypatch.setattr(taste, "TASTE_DIR", tdir)
    _write_taste(tdir, "c-pending", ts="2026-10-16T10:00:00+08:00")
    _write_taste(
        tdir,
        "c-done",
        ts="2026-10-15T10:00:00+08:00",
        review={
            "ts": "2026-10-15T12:00:00+08:00",
            "verdict": "confirmed",
            "note": "",
            "blind": {
                "user_choice": "A",
                "agree": True,
                "remedy": None,
                "requirements": "",
            },
        },
    )
    monkeypatch.setattr(mirror, "MIRROR_QUEUE_PATH", tmp_path / "mirror-queue.jsonl")
    monkeypatch.setattr(journal, "JOURNAL_PATH", tmp_path / "journal.jsonl")
    monkeypatch.setattr(inspectd, "decision_rows", lambda **kw: [dict(_DEC_ROW)])
    return {"root": root, "r1": r1}


# ---------------------------------------------------------------------------
# 导航小件（行模型驱动；禁裸下标语义猜测）
# ---------------------------------------------------------------------------
MENU_INDEX = {
    "taskbooks": 0,
    "plans": 1,
    "queues": 2,
    "review": 3,
    "system": 4,
    "decisions": 5,
    "pending": 6,
}


async def _enter_module(app, pilot, key: str) -> None:
    lv = app.query_one("#menu", ListView)
    lv.index = MENU_INDEX[key]
    await pilot.pause()
    await pilot.press("enter")
    await pilot.pause()


async def _back_to_menu(app, pilot) -> None:
    for _ in range(4):
        if app.view == "menu":
            return
        await pilot.press("escape")
        await pilot.pause()


async def _goto(app, pilot, view: str) -> None:
    """按用户路径导航到某视图（menu → 模块 → 视图内一步）。"""
    if view == "detail":
        await _enter_module(app, pilot, "queues")
    elif view in ("taskbooks", "tb-detail"):
        await _enter_module(app, pilot, "taskbooks")
        if view == "tb-detail":
            app.query_one("#tb-list", ListView).index = 0
            await pilot.pause()
            await pilot.press("o")
            await pilot.pause()
    elif view in ("plans", "plan-detail"):
        await _enter_module(app, pilot, "plans")
        if view == "plan-detail":
            app.query_one("#plans-list", ListView).index = 0
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
    elif view in ("decisions", "decision-detail"):
        await _enter_module(app, pilot, "decisions")
        if view == "decision-detail":
            app.query_one("#decisions-list", ListView).index = 0
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
    elif view in ("review-blind", "review-rulings", "ruling-detail"):
        await _enter_module(app, pilot, "review")
        if view != "review-blind" and app.view == "review-blind":
            # 有待盲审：先就地复核掉（选 A 提交）→ 自动转已裁决列表
            await pilot.press("1")
            await pilot.pause()
            await pilot.click("#blind-submit")
            await pilot.pause()
    if view == "ruling-detail" and app.view == "review-rulings":
        app.query_one("#rulings-list", ListView).index = 0
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()


# ===========================================================================
# ① footer 键位矩阵（全视图）
# ===========================================================================
ALL_VIEWS = (
    "menu",
    "detail",
    "journal",
    "taskbooks",
    "tb-detail",
    "tb-new",
    "tb-reject",
    "plans",
    "plan-detail",
    "decisions",
    "decision-detail",
    "review-blind",
    "review-rulings",
    "ruling-detail",
)


def test_footer_matrix_all_views(monkeypatch):
    """t/j 只在 menu 广告；`o` 只在 taskbooks 广告；受管 action 随视图闸控（含新视图）。"""
    app = tui.PysciTUI()
    for view in ALL_VIEWS:
        app.view = view
        expect_tj = view == "menu"
        assert app.check_action("taskbooks", ()) is expect_tj, (view, "t")
        assert app.check_action("journal", ()) is expect_tj, (view, "j")
        assert app.check_action("tb_open_latest", ()) is (view == "taskbooks"), (
            view,
            "o",
        )
    # `o` 的键位映射存在（BINDINGS 注册）
    keys = {b.action: b.key for b in Binding.make_bindings(tui.PysciTUI.BINDINGS)}
    assert keys.get("tb_open_latest") == "o"
    assert keys.get("blind_pick_a") == "1" and keys.get("blind_pick_b") == "2"


def test_new_keys_have_observable_effect(s9face):
    """新增键实效：taskbooks 的 `o` 进详情；review-blind 的 `1`/`2` 选中选项。"""

    async def _go():
        app = tui.PysciTUI()
        async with app.run_test(size=(120, 45)) as pilot:
            await pilot.pause()
            await _enter_module(app, pilot, "taskbooks")
            app.query_one("#tb-list", ListView).index = 0
            await pilot.pause()
            await pilot.press("o")
            await pilot.pause()
            after_o = app.view
            await _back_to_menu(app, pilot)
            await _enter_module(app, pilot, "review")
            picks = []
            await pilot.press("1")
            await pilot.pause()
            picks.append(app.blind_selected)
            await pilot.press("2")
            await pilot.pause()
            picks.append(app.blind_selected)
            return after_o, app.view, picks

    after_o, review_view, picks = _run(_go())
    assert after_o == "tb-detail", "`o` 在 taskbooks 必须有效果"
    assert review_view == "review-blind" and picks == ["A", "B"]


# ===========================================================================
# ② 滚动指示全覆盖
# ===========================================================================
def test_scroll_mark_present_for_every_registered_view(s9face):
    """`_SCROLL_PANES` 每条的视图打开后：位置行在场（文本以 `_SCROLL_MARK_HEAD` 开头）。"""
    panes = dict(tui._SCROLL_PANES)

    async def _go():
        app = tui.PysciTUI()
        async with app.run_test(size=(120, 45)) as pilot:
            await pilot.pause()
            seen: dict[str, str] = {}
            for view in sorted(panes):
                await _goto(app, pilot, view)
                assert app.view == view, f"导航未达 {view}（实到 {app.view}）"
                mark = app.query_one(panes[view][1], Static)
                app._refresh_scroll_mark()
                await pilot.pause()
                seen[view] = str(mark.render())
                await _back_to_menu(app, pilot)
            return seen

    seen = _run(_go())
    assert set(seen) == set(panes), "全部注册视图都要能被走到"
    for view, text in seen.items():
        assert text.startswith(tui._SCROLL_MARK_HEAD), (view, text)


def test_scroll_y_assignment_refreshes_mark_immediately(s9face):
    """滚轮/滚动条通路（`scroll_y` 赋值）立即刷新位置指示（含 taskbooks 列表）。"""

    async def _go():
        app = tui.PysciTUI()
        async with app.run_test(size=(120, 20)) as pilot:
            await pilot.pause()
            out: dict[str, tuple[str, str]] = {}
            for view in ("detail", "taskbooks", "plans", "decisions"):
                await _goto(app, pilot, view)
                pane = app.query_one(tui._SCROLL_PANES[view][0])
                pane.scroll_y = 1
                await pilot.pause()
                mark = str(app.query_one(tui._SCROLL_PANES[view][1], Static).render())
                out[view] = (mark, f"y={int(pane.scroll_y)}")
                await _back_to_menu(app, pilot)
            return out

    out = _run(_go())
    for view, (mark, y) in out.items():
        assert mark.startswith(tui._SCROLL_MARK_HEAD), view
        if y != "y=0":  # 内容可滚时，指示须反映真实行号
            assert f"（第 {y[2:]}/" in mark, (view, mark)


# ===========================================================================
# ③ 禁语全仓钉
# ===========================================================================
BANNED = (
    "Esc/← 返回",
    "批复入口",
    "口述动作",
    "推进：",
    "多 plan 并行",
    "reviewer 附属队列",
    "审查面（inspector 队列）超时项",
    "审查入口",
)


def test_all_read_faces_have_no_banned_prompts(s9face):
    """全部可断言读面（detail/tb/plan/decision/blind/ruling + 各列表行）无禁语。"""

    async def _go():
        app = tui.PysciTUI()
        async with app.run_test(size=(120, 45)) as pilot:
            await pilot.pause()
            faces: list[str] = []
            await _enter_module(app, pilot, "queues")
            faces += [
                app.detail_title,
                app.detail_body,
                app.detail_text,
                app.detail_hint,
            ]
            await _back_to_menu(app, pilot)
            await _enter_module(app, pilot, "taskbooks")
            faces += [
                "\n".join(
                    str(cell.render())
                    for li in app.query("#tb-list > ListItem")
                    for cell in li.query(Static)
                )
            ]
            app.query_one("#tb-list", ListView).index = 0
            await pilot.pause()
            await pilot.press("o")
            await pilot.pause()
            faces += [app.tb_detail_title, app.tb_detail_body, app.tb_detail_hint]
            await _back_to_menu(app, pilot)
            await _enter_module(app, pilot, "plans")
            faces += [
                "\n".join(str(cell.render()) for cell in c.query(Static))
                for c in app.query("#plans-list > ListItem")
            ]
            app.query_one("#plans-list", ListView).index = 0
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            faces += [app.plan_detail_title, app.plan_detail_head, app.plan_detail_md]
            await _back_to_menu(app, pilot)
            await _enter_module(app, pilot, "decisions")
            app.query_one("#decisions-list", ListView).index = 0
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            faces += [
                app.decision_detail_title,
                app.decision_detail_head,
                app.decision_detail_md,
            ]
            await _back_to_menu(app, pilot)
            await _enter_module(app, pilot, "review")
            faces += [
                app.blind_msg,
                str(app.query_one("#blind-question", Static).render()),
                str(app.query_one("#blind-evidence", Static).render()),
            ]
            await _back_to_menu(app, pilot)
            await _enter_module(app, pilot, "review")
            return "\n".join(faces), app.view

    text, view = _run(_go())
    for banned in BANNED:
        assert banned not in text, banned
    assert view in ("review-blind", "review-rulings")


def test_modules_detail_faces_have_no_banned_prompts():
    """数据层文本读面（modules 七模块 detail + 静态通道）同样无禁语。"""
    from pysci.skills.orchestration.tools import mirror as _mirror

    faces = ["\n".join(m.detail()) for m in tui_modules.modules()]
    faces.append("\n".join(_mirror.decisions_detail_lines()))
    text = "\n".join(faces)
    for banned in BANNED:
        assert banned not in text, banned


# ===========================================================================
# ④ 图标空格全仓钉（tools/ 全目录，含 tui/）
# ===========================================================================
_ICON_CHARS = "✔⊘▶◆●★"
_ICON_EXEMPT = frozenset(" ]）)\"'`")


def test_no_icon_glued_to_text_across_tools():
    """`tools/` 全目录（含 tui/）源码无「图标紧贴文字」（豁免成对括号/字符串末位）。"""
    import re

    tools_dir = Path(tui_modules.__file__).resolve().parents[1]
    hits: list[str] = []
    for p in sorted(tools_dir.rglob("*.py")):
        for no, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            for m in re.finditer(f"[{_ICON_CHARS}](.)", line):
                if m.group(1) in _ICON_EXEMPT:
                    continue
                hits.append(f"{p.name}:{no}: {line.strip()}")
    assert not hits, "图标后须隔一个空格再跟文字：\n" + "\n".join(hits)


# ===========================================================================
# ⑤ 徽标语义回归（§2.2）
# ===========================================================================
def test_fingerprint_cursors_avoid_full_detail_and_are_memoized(monkeypatch):
    """plans 的 `cursor()` 不建计划详情；system 的指纹走 **stable** 且被 TTL 复用（2s 一次）。

    （system 的 stable 指纹**由详情派生**（去掉倒计时行）是 s3 既定接线，故它的护栏不是
    「不调 detail」而是「走 stable + TTL 复用」——两条一起才等于「每秒 tick 不全量重建」。）
    """
    detail_calls = {"n": 0}

    def spy_plans_detail() -> list[str]:
        detail_calls["n"] += 1
        return []

    stable_calls = {"n": 0}
    real_stable = tui_modules._system_stable

    def spy_stable() -> list[str]:
        stable_calls["n"] += 1
        return real_stable()

    monkeypatch.setattr(tui_modules, "_plans_detail", spy_plans_detail)
    monkeypatch.setattr(tui_modules, "_system_stable", spy_stable)
    mods = {m.key: m for m in tui_modules.modules()}
    mods["plans"].cursor()
    mods["plans"].cursor()
    assert detail_calls["n"] == 0, "plans 指纹不得构建计划详情"
    mods["system"].cursor()
    mods["system"].cursor()
    assert stable_calls["n"] == 1, (
        "system 指纹须走 stable 且被 TTL 复用（2s 内只算一次）"
    )


def test_system_countdown_not_in_fingerprint(monkeypatch):
    """倒计时行不进 system 指纹（徽标不因每秒跳动而常亮）。"""
    monkeypatch.setattr(
        tui_modules,
        "_countdown_line",
        lambda *a, **k: "  倒计时：距下次切换还有 1 分 1 秒（x → 入谷）",
    )
    stable = tui_modules._system_stable()
    assert all(tui_modules._COUNTDOWN_MARK not in ln for ln in stable)
    assert tui_modules._system_detail()[0].startswith("  倒计时")


def test_mark_seen_same_cursor_skips_write(tmp_path, monkeypatch):
    """`_mark_seen`：cursor 未变则不写盘（每秒 tick 不重写 seen 文件）。"""
    seen = tmp_path / "seen.json"
    calls: list[int] = []
    real = tui.watermark.save_seen
    monkeypatch.setattr(
        tui.watermark,
        "save_seen",
        lambda data, path=None: (calls.append(1), real(data, path))[-1],
    )

    async def _go():
        app = tui.PysciTUI(seen_path=seen)
        async with app.run_test() as pilot:
            await pilot.pause()
            app._mark_seen(app._mods[0])
            app._mark_seen(app._mods[0])
            return len(calls)

    assert _run(_go()) == 1


# ===========================================================================
# ⑥ 导航一致性冒烟（七模块 → Esc 回菜单）
# ===========================================================================
def test_menu_every_module_enter_then_escape_returns(s9face):
    """七模块逐一 Enter→Esc：view 状态机全程正确、无崩溃（含 plans/decisions/review 新分支）。"""

    async def _go():
        app = tui.PysciTUI()
        async with app.run_test(size=(120, 45)) as pilot:
            await pilot.pause()
            walked: list[tuple[str, str]] = []
            for key in MENU_INDEX:
                await _enter_module(app, pilot, key)
                walked.append((key, app.view))
                await _back_to_menu(app, pilot)
                assert app.view == "menu", (key, app.view)
            return walked

    walked = _run(_go())
    assert [k for k, _ in walked] == list(MENU_INDEX)
    got = dict(walked)
    assert got["taskbooks"] == "taskbooks" and got["plans"] == "plans"
    assert got["decisions"] == "decisions"
    assert got["review"] in ("review-blind", "review-rulings")
    for key in ("queues", "system", "pending"):
        assert got[key] == "detail", key


# ===========================================================================
# ⑦ 契约读面：重进视图状态重建正确
# ===========================================================================
def test_taskbook_rows_and_expansion_survive_reentry(s9face):
    """`tb_rows`/`tb_expanded` 与渲染一致；Esc 往返后行模型重建正确、展开态保留。"""

    async def _go():
        app = tui.PysciTUI()
        async with app.run_test(size=(120, 45)) as pilot:
            await pilot.pause()
            await _enter_module(app, pilot, "taskbooks")
            first = (
                len(app.tb_rows),
                len(app.query("#tb-list > ListItem")),
                dict(app.tb_expanded),
            )
            app.query_one("#tb-list", ListView).index = 0
            await pilot.press("enter")  # 展开
            await pilot.pause()
            expanded = (
                len(app.tb_rows),
                len(app.query("#tb-list > ListItem")),
                dict(app.tb_expanded),
            )
            await _back_to_menu(app, pilot)
            await _enter_module(app, pilot, "taskbooks")
            again = (
                len(app.tb_rows),
                len(app.query("#tb-list > ListItem")),
                dict(app.tb_expanded),
            )
            return first, expanded, again, [r["kind"] for r in app.tb_rows]

    first, expanded, again, kinds = _run(_go())
    assert first[0] == first[1] == 1 and first[2] == {}, "初始收起：一组一行"
    assert expanded[0] == expanded[1] == 3, "展开后：组行 + 2 次级（本体+返工）"
    assert again == expanded, "Esc 往返后行模型/列表/展开态一致"
    assert kinds == ["group", "sub", "sub"]


def test_plan_and_decision_rows_match_data_layer(s9face):
    """`plan_rows`/`decision_rows` 与数据层读数一致；plan-detail 往返后列表完好。"""

    async def _go():
        app = tui.PysciTUI()
        async with app.run_test(size=(120, 45)) as pilot:
            await pilot.pause()
            await _enter_module(app, pilot, "plans")
            plans_before = [r["id"] for r in app.plan_rows]
            app.query_one("#plans-list", ListView).index = 0
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            in_detail = app.view
            await pilot.press("escape")
            await pilot.pause()
            back = (app.view, [r["id"] for r in app.plan_rows])
            await _back_to_menu(app, pilot)
            await _enter_module(app, pilot, "decisions")
            dec = [r["id"] for r in app.decision_rows]
            return plans_before, in_detail, back, dec

    plans_before, in_detail, back, dec = _run(_go())
    assert plans_before == [r["id"] for r in plans.overview_rows()]
    assert in_detail == "plan-detail" and back == ("plans", plans_before)
    assert dec == [r["id"] for r in inspectd.decision_rows()]


def test_blind_count_tracks_pending_across_reentry(s9face):
    """`blind_count` 随复核递减（进程内）；Esc 往返后与数据层待审数一致。"""

    async def _go():
        app = tui.PysciTUI()
        async with app.run_test(size=(120, 45)) as pilot:
            await pilot.pause()
            await _enter_module(app, pilot, "review")
            first = (app.view, app.blind_count, app.blind_case_id)
            await pilot.press("1")  # 选中选项 A（与替身案的 twin=A 一致）
            await pilot.pause()
            await pilot.click("#blind-submit")
            await pilot.pause()
            after = (app.view, app.blind_count)
            await _back_to_menu(app, pilot)
            await _enter_module(app, pilot, "review")
            return first, after, (app.view, len(taste.blind_queue()), app.blind_count)

    first, after, reentered = _run(_go())
    assert first == ("review-blind", 1, "c-pending")
    assert after == ("review-rulings", 0), "处理完 → 已裁决列表"
    assert reentered == ("review-rulings", 0, 0), "重进与数据层一致（无待审）"


def test_ruling_detail_matches_data_layer_row(s9face):
    """裁决列表/详情读面与 `taste.ruling_rows()` 一致（重进亦一致）。"""

    async def _go():
        app = tui.PysciTUI()
        async with app.run_test(size=(120, 45)) as pilot:
            await pilot.pause()
            await _enter_module(app, pilot, "review")  # 有 1 项待审 → 先处理才到列表
            await pilot.press("1")
            await pilot.pause()
            await pilot.click("#blind-submit")
            await pilot.pause()
            rows = [r["case_id"] for r in app.ruling_rows]
            await pilot.press("escape")
            await pilot.pause()
            await _enter_module(app, pilot, "review")
            again = [r["case_id"] for r in app.ruling_rows]
            app.query_one("#rulings-list", ListView).index = 0
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            return rows, again, app.view, app.ruling_detail_id, app.ruling_detail_md

    rows, again, view, did, md = _run(_go())
    expect = [r["case_id"] for r in taste.ruling_rows()]
    assert rows == expect and again == expect
    assert view == "ruling-detail" and did in expect
    assert "# 裁决" in md
