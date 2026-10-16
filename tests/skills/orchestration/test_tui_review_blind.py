"""TUI 改版 s8（task tui-overhaul-20261016-s8-review-blind）：盲审交互＋偏差报告＋已裁决列表。

钉用户定案（逐条）：

1. 入口：有待盲审 → **直接进盲审界面**（顶部「尚待盲审 N 项」）；否则 → 已裁决列表；
2. **盲审反转**（2_REFACTOR §10）：**提交前渲染面绝不出现 twin 的 choice/reason/body**；
   - 一致提交 → 复核 confirmed ＋自动呈现下一项（队列空 → 已裁决列表）；
   - 不一致 → 展开「① 原因批复」＋「② 是否需恢复」（需恢复/不需恢复/其他；**其他**才出具体要求框）；
3. 偏差报告（§5.4）：确认后落 `mirror-queue.jsonl`（kind=deviation、字段全）；**需恢复**额外
   建「偏差纠正」任务书（priority=high）并把 `taskbook_id` 回填进偏差行；不需恢复/其他不建件；
4. 已裁决列表/详情：行＝复核结论旗标＋ts＋案由；详情＝头卡＋Markdown（**可显 twin 答案**——盲审已过）；
5. 禁语：本板块渲染面无「reviewer 附属队列」「审查面（inspector 队列）超时项」「审查入口」「Esc/← 返回」。
"""

from __future__ import annotations

import asyncio
import json

import pytest
from textual.widgets import ListView, Static

from pysci import paths as _paths
from pysci.skills.orchestration.tools import journal, mirror, taskbooks, taste
from pysci.skills.orchestration.tools.tui import app as tui

Q_MARK = "S8_QUESTION_MARK"
A_MARK = "S8_OPTION_A_MARK"
B_MARK = "S8_OPTION_B_MARK"
EV_MARK = "S8_EVIDENCE_MARK"
TWIN_REASON_MARK = "S8_TWIN_REASON_MARK"
TWIN_BODY_MARK = "S8_TWIN_BODY_MARK"


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture
def face(tmp_path, monkeypatch):
    """隔离：taste 留痕 / 案卷文件 / mirror 队列 / 任务书目录 / journal。"""
    taste_dir = tmp_path / "taste"
    taste_dir.mkdir()
    monkeypatch.setattr(taste, "TASTE_DIR", taste_dir)
    monkeypatch.setattr(mirror, "MIRROR_QUEUE_PATH", tmp_path / "mirror-queue.jsonl")
    monkeypatch.setattr(taskbooks, "TASKBOOKS_DIR", tmp_path / "data" / "taskbooks")
    monkeypatch.setattr(_paths, "DATA_ROOT", tmp_path / "data")
    monkeypatch.setattr(journal, "JOURNAL_PATH", tmp_path / "journal.jsonl")
    return tmp_path


def _case_file(face, cid: str) -> str:
    p = face / f"{cid}.md"
    p.write_text(
        f"## 问题\n选哪个？（{Q_MARK}）\n\n"
        f"## 选项 A\n甲方案（{A_MARK}）\n\n"
        f"## 选项 B\n乙方案（{B_MARK}）\n\n"
        f"## 证据\n证据正文（{EV_MARK}）\n",
        encoding="utf-8",
    )
    return str(p)


def _record(
    face,
    cid: str,
    *,
    ts: str,
    choice: str = "A",
    review: dict | None = None,
) -> dict:
    rec = {
        "ts": ts,
        "case_id": cid,
        "case_file": _case_file(face, cid),
        "case_head": f"选哪个？（{Q_MARK}）",
        "choice": choice,
        "confidence": 0.62,
        "reason": f"援引 P-03（{TWIN_REASON_MARK}）",
        "parsed": True,
        "body": f"twin 交付全文（{TWIN_BODY_MARK}）",
        "record_kind": "taste",
        "night": False,
        "authority": "",
    }
    if review is not None:
        rec["review"] = review
    taste.write_record(rec)
    return rec


async def _enter_review(app, pilot) -> None:
    lv = app.query_one("#menu", ListView)
    lv.index = 3  # 盲审与审查
    await pilot.pause()
    await pilot.press("enter")
    await pilot.pause()


def _blind_face_text(app) -> str:
    """盲审界面**全部渲染文本**（Static/ChoiceCard/Button）。"""
    from pysci.skills.orchestration.tools.tui.widgets import ChoiceCard

    parts: list[str] = []
    for sel in ("#blind-count", "#blind-question", "#blind-evidence", "#blind-msg"):
        parts.append(str(app.query_one(sel, Static).render()))
    for sel in ("#blind-a", "#blind-b"):
        parts.append(str(app.query_one(sel, ChoiceCard).render()))
    return "\n".join(parts)


def _read_record(face, cid: str) -> dict:
    return json.loads((face / "taste" / f"{cid}.json").read_text(encoding="utf-8"))


def _deviation_rows(face) -> list[dict]:
    p = face / "mirror-queue.jsonl"
    if not p.exists():
        return []
    return [
        json.loads(ln)
        for ln in p.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]


# ---------------------------------------------------------------------------
# ① 入口分派
# ---------------------------------------------------------------------------
def test_entry_goes_to_blind_when_pending(face):
    _record(face, "c1", ts="2026-10-16T10:00:00+08:00", choice="A")
    _record(face, "c2", ts="2026-10-16T11:00:00+08:00", choice="B")

    async def _go():
        app = tui.PysciTUI()
        async with app.run_test(size=(120, 45)) as pilot:
            await pilot.pause()
            await _enter_review(app, pilot)
            return (
                app.view,
                app.blind_count,
                app.blind_case_id,
                str(app.query_one("#blind-count", Static).render()),
            )

    view, count, cid, line = _run(_go())
    assert view == "review-blind" and count == 2
    assert cid == "c1", "队首＝ts 最小（升序）"
    assert "尚待盲审 2 项" in line


def test_entry_goes_to_rulings_when_no_pending(face):
    _record(
        face,
        "c-done",
        ts="2026-10-16T10:00:00+08:00",
        review={"ts": "2026-10-16T12:00:00+08:00", "verdict": "confirmed", "note": ""},
    )

    async def _go():
        app = tui.PysciTUI()
        async with app.run_test(size=(120, 45)) as pilot:
            await pilot.pause()
            await _enter_review(app, pilot)
            return app.view, len(app.ruling_rows)

    view, n = _run(_go())
    assert view == "review-rulings" and n == 1


# ---------------------------------------------------------------------------
# ② 盲审界面：隐答案 / 一致 / 不一致
# ---------------------------------------------------------------------------
def test_blind_face_hides_twin_answers(face):
    """提交前渲染面**绝不出现** twin 的 choice/reason/body（反转设计 §10）。"""
    _record(face, "c1", ts="2026-10-16T10:00:00+08:00", choice="A")

    async def _go():
        app = tui.PysciTUI()
        async with app.run_test(size=(120, 45)) as pilot:
            await pilot.pause()
            await _enter_review(app, pilot)
            return app.view, _blind_face_text(app)

    view, text = _run(_go())
    assert view == "review-blind"
    for hidden in (TWIN_REASON_MARK, TWIN_BODY_MARK):
        assert hidden not in text, f"提交前泄露 twin 答案：{hidden}"
    assert Q_MARK in text, "案情（问题）须呈现"
    assert A_MARK in text and B_MARK in text, "两个选项都须呈现"


def test_keyboard_and_click_selection(face):
    _record(face, "c1", ts="2026-10-16T10:00:00+08:00", choice="A")

    async def _go():
        app = tui.PysciTUI()
        async with app.run_test(size=(120, 45)) as pilot:
            await pilot.pause()
            await _enter_review(app, pilot)
            await pilot.press("2")  # 键盘可选（选项 B）
            await pilot.pause()
            by_key = (
                app.blind_selected,
                bool(app.query_one("#blind-b").has_class("-selected")),
            )
            await pilot.click("#blind-a")  # 点击改选 A（点击只上报 value）
            await pilot.pause()
            return by_key, (
                app.blind_selected,
                bool(app.query_one("#blind-a").has_class("-selected")),
            )

    by_key, by_click = _run(_go())
    assert by_key == ("B", True)
    assert by_click == ("A", True)


def test_submit_agree_confirms_and_advances_then_rulings(face):
    _record(face, "c1", ts="2026-10-16T10:00:00+08:00", choice="A")
    _record(face, "c2", ts="2026-10-16T11:00:00+08:00", choice="A")

    async def _go():
        app = tui.PysciTUI()
        async with app.run_test(size=(120, 45)) as pilot:
            await pilot.pause()
            await _enter_review(app, pilot)
            await pilot.press("1")  # 选 A（与 c1 的 twin 一致）
            await pilot.pause()
            await pilot.click("#blind-submit")
            await pilot.pause()
            after_first = (app.view, app.blind_case_id, app.blind_count)
            await pilot.press("1")  # c2 也选 A
            await pilot.pause()
            await pilot.click("#blind-submit")
            await pilot.pause()
            return after_first, (app.view, len(app.ruling_rows))

    after_first, after_second = _run(_go())
    assert after_first == ("review-blind", "c2", 1), "一致 → 确认后自动下一项"
    assert _read_record(face, "c1")["review"]["verdict"] == "confirmed"
    assert _read_record(face, "c1")["review"]["note"] == "盲审与 twin 一致"
    assert _read_record(face, "c1")["review"]["blind"]["agree"] is True
    assert after_second[0] == "review-rulings", "全部处理完 → 已裁决列表"
    assert _read_record(face, "c2")["review"]["verdict"] == "confirmed"
    assert after_second[1] == 2


def test_submit_disagree_opens_ask_and_other_requires_requirements(face):
    _record(face, "c1", ts="2026-10-16T10:00:00+08:00", choice="B")

    async def _go():
        app = tui.PysciTUI()
        async with app.run_test(size=(120, 45)) as pilot:
            await pilot.pause()
            await _enter_review(app, pilot)
            await pilot.press("1")  # 选 A（与 twin 不一致）
            await pilot.pause()
            await pilot.click("#blind-submit")
            await pilot.pause()
            ask = (
                app.blind_ask_visible,
                app.query_one("#blind-ask").display,
                app.query_one("#blind-requirements").display,
            )
            app._set_radio("#blind-remedy", 2)  # 「其他」
            await pilot.pause()
            other = (
                app.blind_remedy,
                app.query_one("#blind-requirements").display,
            )
            app.query_one("#blind-reason").text = "与 P-07 冲突"
            app.query_one("#blind-requirements").text = "请恢复并补色标上限"
            await pilot.pause()
            await pilot.click("#blind-confirm")
            await pilot.pause()
            return ask, other, (app.view, app.blind_count, app.blind_msg)

    ask, other, after = _run(_go())
    assert ask[0] is True and ask[1] is True and ask[2] is False, (
        "追问区展开、具体要求框未显示"
    )
    assert other == ("other", True), "选「其他」才出具体要求框"
    assert after[0] in ("review-rulings", "review-blind") and after[1] == 0
    rows = _deviation_rows(face)
    assert len(rows) == 1
    row = rows[0]
    assert row["kind"] == "deviation" and row["source"] == "blind-review"
    assert (
        row["case_id"] == "c1"
        and row["user_choice"] == "A"
        and row["twin_choice"] == "B"
    )
    assert row["reason"] == "与 P-07 冲突" and row["remedy"] == "other"
    assert row["requirements"] == "请恢复并补色标上限"
    assert row["status"] == "pending" and "偏差报告 c1" in row["text"]
    assert "taskbook_id" not in row, "「其他」不建任务书"
    assert taskbooks.list_all() == [], "不建件"
    rev = _read_record(face, "c1")["review"]
    assert rev["verdict"] == "rejected" and rev["note"] == "与 P-07 冲突"
    assert rev["blind"]["agree"] is False and rev["blind"]["remedy"] == "other"


def test_restore_creates_taskbook_and_backfills_id(face):
    _record(face, "c1", ts="2026-10-16T10:00:00+08:00", choice="B")

    async def _go():
        app = tui.PysciTUI()
        async with app.run_test(size=(120, 45)) as pilot:
            await pilot.pause()
            await _enter_review(app, pilot)
            await pilot.press("1")  # 选 A（不一致）
            await pilot.pause()
            await pilot.click("#blind-submit")
            await pilot.pause()
            app._set_radio("#blind-remedy", 0)  # 「需恢复」（默认）
            app.query_one("#blind-reason").text = "理由：色标不符"
            await pilot.pause()
            await pilot.click("#blind-confirm")
            await pilot.pause()
            return app.view

    view = _run(_go())
    assert view in ("review-rulings", "review-blind"), (
        "确认后应自动呈现下一项/已裁决列表"
    )
    rows = _deviation_rows(face)
    assert len(rows) == 1
    tb_id = rows[0].get("taskbook_id") or ""
    assert tb_id, "需恢复 → 偏差行须回填 taskbook_id"
    st = taskbooks.load_state(tb_id) or {}
    assert st.get("priority") == "high" and st.get("status") == "new"
    assert str(st.get("title") or "").startswith("偏差纠正：c1")
    body = taskbooks.read_body(tb_id)
    assert "选择对照" in body and "用户盲审选择：A" in body and "twin 裁决：B" in body
    assert "理由：色标不符" in body and "实施要求" in body


def test_neednot_restore_only_enqueues(face):
    _record(face, "c1", ts="2026-10-16T10:00:00+08:00", choice="B")

    async def _go():
        app = tui.PysciTUI()
        async with app.run_test(size=(120, 45)) as pilot:
            await pilot.pause()
            await _enter_review(app, pilot)
            await pilot.press("1")
            await pilot.pause()
            await pilot.click("#blind-submit")
            await pilot.pause()
            app._set_radio("#blind-remedy", 1)  # 「不需恢复」
            await pilot.pause()
            await pilot.click("#blind-confirm")
            await pilot.pause()

    _run(_go())
    rows = _deviation_rows(face)
    assert len(rows) == 1 and rows[0]["remedy"] == "none"
    assert "taskbook_id" not in rows[0]
    assert taskbooks.list_all() == [], "不需恢复不建任务书"


# ---------------------------------------------------------------------------
# ③ 已裁决列表 + 详情（twin 答案此处可显）
# ---------------------------------------------------------------------------
def test_rulings_list_and_detail_show_twin_and_review(face):
    _record(
        face,
        "c-done",
        ts="2026-10-16T10:00:00+08:00",
        choice="A",
        review={
            "ts": "2026-10-16T12:00:00+08:00",
            "verdict": "rejected",
            "note": "原因：P-07 冲突",
            "blind": {
                "user_choice": "B",
                "agree": False,
                "remedy": "restore",
                "requirements": "恢复原实现",
            },
        },
    )

    async def _go():
        app = tui.PysciTUI()
        async with app.run_test(size=(120, 45)) as pilot:
            await pilot.pause()
            await _enter_review(app, pilot)
            lv_texts = [
                " ".join(str(cell.render()) for cell in li.query(Static))
                for li in app.query_one("#rulings-list", ListView).children
            ]
            app.query_one("#rulings-list", ListView).index = 0
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            detail = (
                app.view,
                app.ruling_detail_id,
                app.ruling_detail_head,
                app.ruling_detail_md,
            )
            await pilot.press("escape")
            await pilot.pause()
            back1 = app.view
            await pilot.press("escape")
            await pilot.pause()
            return lv_texts, detail, back1, app.view

    lv_texts, detail, back1, back2 = _run(_go())
    assert len(lv_texts) == 1 and lv_texts[0].startswith("rejected")
    assert "2026-10-16T10:00:00" in lv_texts[0] and Q_MARK in lv_texts[0]
    view, did, head, md = detail
    assert view == "ruling-detail" and did == "c-done"
    assert "rejected" in head and "12:00:00" in head
    assert "# 裁决 c-done" in md and Q_MARK in md
    assert f"- 选项 A：甲方案（{A_MARK}）" in md
    assert "## twin 裁决（盲审已过，此处可显）" in md and TWIN_REASON_MARK in md
    assert "## 你的盲审" in md and "- 选择：B" in md and "与 twin 一致：否" in md
    assert "需恢复" in md and "原因：P-07 冲突" in md and "恢复原实现" in md
    assert back1 == "review-rulings" and back2 == "menu"


def test_rulings_empty_state(face):
    async def _go():
        app = tui.PysciTUI()
        async with app.run_test(size=(120, 45)) as pilot:
            await pilot.pause()
            await _enter_review(app, pilot)
            return app.view, [
                str(li.query_one(Static).render())
                for li in app.query_one("#rulings-list", ListView).children
            ]

    view, texts = _run(_go())
    assert view == "review-rulings" and any("（暂无裁决记录）" in t for t in texts)


def test_back_from_blind_returns_menu(face):
    _record(face, "c1", ts="2026-10-16T10:00:00+08:00", choice="A")

    async def _go():
        app = tui.PysciTUI()
        async with app.run_test(size=(120, 45)) as pilot:
            await pilot.pause()
            await _enter_review(app, pilot)
            await pilot.press("escape")
            await pilot.pause()
            return app.view

    assert _run(_go()) == "menu"


# ---------------------------------------------------------------------------
# ④ 禁语钉
# ---------------------------------------------------------------------------
def test_review_faces_have_no_banned_prompts(face):
    _record(face, "c1", ts="2026-10-16T10:00:00+08:00", choice="A")
    _record(
        face,
        "c-done",
        ts="2026-10-15T10:00:00+08:00",
        review={"ts": "2026-10-15T12:00:00+08:00", "verdict": "confirmed", "note": ""},
    )

    async def _go():
        app = tui.PysciTUI()
        async with app.run_test(size=(120, 45)) as pilot:
            await pilot.pause()
            await _enter_review(app, pilot)
            blind = _blind_face_text(app)
            await pilot.press("escape")
            await pilot.pause()
            lv = app.query_one("#menu", ListView)
            await pilot.press("enter")  # 再进（仍有 1 项待审）
            await pilot.pause()
            return blind, _blind_face_text(app)

    faces = "\n".join(_run(_go()))
    for banned in (
        "reviewer 附属队列",
        "审查面（inspector 队列）超时项",
        "审查入口",
        "Esc/← 返回",
    ):
        assert banned not in faces, banned
