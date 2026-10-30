"""TUI 改版 s1-data：数据层读数/写入扩展（任务书 tui-overhaul-20261016-s1-data）。

钉住验收五面：

1. plans 概览/指纹**只读 state.json**（``_load_yaml`` 抛异常仍正常）+ 解析缓存不被
   调用方的就地改写污染；
2. taskbooks 家族归组（真实 ``data/taskbooks/`` 实测 + 合成分叉/孤儿）+ ``detail_markdown``
   与 ``detail_lines`` 共享同一组装件、「批复入口」提示行已清除；
3. queues.water_lines 每成员必出一行（含全零「（空）」）+ 在途明细行带成员名；
4. taste 盲审队列**绝不含 twin 答案**（choice/reason/body/confidence/parsed）；
5. mirror 偏差报告同队列同待遇（``pending_directives`` 自动可见）。
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from pysci.skills.orchestration.tools import (
    journal,
    mirror,
    plans,
    queues,
    taskbooks,
    taste,
    workq,
)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def pface(tmp_path, monkeypatch):
    """plans 目录隔离（缓存按解析后路径 keyed ⇒ tmp 面自然隔离）。"""
    plans_dir = tmp_path / "plans"
    plans_dir.mkdir()
    monkeypatch.setattr(plans, "PLANS_DIR", plans_dir)
    return plans_dir


def _write_plan(
    plans_dir: Path,
    plan_id: str,
    *,
    goal: str = "目标",
    steps: list | None = None,
    state_extra: dict | None = None,
) -> None:
    (plans_dir / f"{plan_id}.plan.yaml").write_text(
        yaml.safe_dump(
            {
                "id": plan_id,
                "goal": goal,
                "steps": steps
                or [
                    {"id": "s1", "member": "sim", "brief": "环节一"},
                    {"id": "s2", "member": "figure", "brief": "环节二"},
                ],
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    state = {
        "plan_id": plan_id,
        "created": "2026-10-16T10:00:00+08:00",
        "goal": goal,
        "steps": [{"id": "s1", "status": "done"}, {"id": "s2", "status": "pending"}],
    }
    state.update(state_extra or {})
    (plans_dir / f"{plan_id}.state.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# ① plans：概览/指纹只读 state + 缓存
# ---------------------------------------------------------------------------
def test_overview_rows_state_only_and_structure(pface, monkeypatch):
    """性能钉：``_load_yaml`` 抛异常时 overview/digest 仍正常（证明不解析 YAML）。"""
    _write_plan(pface, "p1")
    # 缺 state 的 plan：回落 _load_yaml 取 goal——抛异常也只塌 goal，不塌概览
    (pface / "p2.plan.yaml").write_text(
        yaml.safe_dump(
            {"id": "p2", "goal": "仅 YAML", "steps": [{"id": "s1", "member": "sim"}]},
            allow_unicode=True,
        ),
        encoding="utf-8",
    )

    def boom(_pid: str) -> dict:
        raise RuntimeError("不许解析 YAML")

    monkeypatch.setattr(plans, "_load_yaml", boom)

    rows = plans.overview_rows()
    by_id = {r["id"]: r for r in rows}
    row = by_id["p1"]
    assert row["goal"] == "目标" and row["mark"] == "▶ 进行中"
    assert (row["done"], row["total"]) == (1, 2)
    assert row["created"] == "2026-10-16T10:00:00+08:00"
    assert row["current"] == {
        "id": "s2",
        "status": "pending",
        "session_id": "",
        "task_file": "",
    }
    assert by_id["p2"]["goal"] == ""  # 回落失败 fail-soft（不抛即过）
    digest = plans.plan_digest()
    assert len(digest) == 12 and all(c in "0123456789abcdef" for c in digest)


def test_overview_mark_and_current_none_when_completed(pface):
    _write_plan(
        pface,
        "pc",
        state_extra={
            "completed_at": "2026-10-16T12:00:00+08:00",
            "steps": [{"id": "s1", "status": "done"}, {"id": "s2", "status": "done"}],
        },
    )
    (row,) = plans.overview_rows()
    assert row["mark"] == "✔ 完成" and row["current"] is None
    assert (row["done"], row["total"]) == (2, 2)


def test_overview_rows_sort_created_desc_missing_last(pface):
    _write_plan(pface, "pA", state_extra={"created": "2026-10-16T09:00:00+08:00"})
    _write_plan(pface, "pB", state_extra={"created": "2026-10-16T11:00:00+08:00"})
    _write_plan(pface, "pC", state_extra={"created": ""})
    _write_plan(pface, "pD", state_extra={"created": ""})
    assert [r["id"] for r in plans.overview_rows()] == ["pB", "pA", "pD", "pC"]


def test_plan_digest_tracks_step_status(pface):
    _write_plan(pface, "p1")
    d1 = plans.plan_digest()
    sp = pface / "p1.state.json"
    st = json.loads(sp.read_text(encoding="utf-8"))
    st["steps"][1]["status"] = "queued"
    sp.write_text(json.dumps(st, ensure_ascii=False), encoding="utf-8")
    assert plans.plan_digest() != d1, "步骤状态变化必须改变指纹"


def test_state_cache_not_poisoned_and_hit_avoids_reparsing(pface, monkeypatch):
    _write_plan(pface, "p1")
    calls = {"n": 0}
    real = plans._read_state_json

    def counting(path: Path) -> dict:
        calls["n"] += 1
        return real(path)

    monkeypatch.setattr(plans, "_read_state_json", counting)
    st = plans._load_state("p1")
    st["steps"].append({"id": "ghost", "status": "pending"})
    st["goal"] = "被污染"
    fresh = plans._load_state("p1")  # 命中缓存（不重解析）且不受改写污染
    assert calls["n"] == 1
    assert all(s.get("id") != "ghost" for s in fresh["steps"])
    assert fresh["goal"] == "目标"
    sp = pface / "p1.state.json"
    st["plan_id"] = "p1"
    sp.write_text(json.dumps(st, ensure_ascii=False), encoding="utf-8")
    reloaded = plans._load_state("p1")  # 落盘 → mtime 变 → 重新解析
    assert calls["n"] == 2


def test_plan_tree_nodes_superstep_structure(pface):
    steps = [
        {
            "id": "A",
            "mode": "serial",
            "brief": "综合改进",
            "children": [
                {"id": "a1", "kind": "self", "brief": "成因"},
                {"id": "a2", "member": "devops", "brief": "实施"},
            ],
        },
        {"id": "s1", "member": "sim", "brief": "扁平环节"},
    ]
    _write_plan(
        pface,
        "pt",
        steps=steps,
        state_extra={
            "steps": [
                {"id": "A.a1", "status": "done"},
                {"id": "A.a2", "status": "queued"},
                {"id": "s1", "status": "pending"},
            ]
        },
    )
    nodes = plans.plan_tree_nodes("pt")
    assert [n["path"] for n in nodes] == ["A", "s1"]
    a = nodes[0]
    assert a["kind"] == "super" and a["mode"] == "serial" and a["status"] == "started"
    assert [c["path"] for c in a["children"]] == ["A.a1", "A.a2"]
    assert a["children"][0]["kind"] == "leaf" and a["children"][0]["status"] == "done"
    assert a["children"][1]["member"] == "devops"
    assert nodes[1]["kind"] == "leaf" and nodes[1]["status"] == "pending"
    assert plans.plan_tree_nodes("nope") == []


# ---------------------------------------------------------------------------
# ② queues：water_lines 成员完整性
# ---------------------------------------------------------------------------
def test_water_lines_every_member_and_named_inflight(tmp_path, monkeypatch):
    qdir = tmp_path / "queues"
    qdir.mkdir()
    monkeypatch.setattr(queues, "QUEUES_DIR", qdir)
    queues.enqueue("sim", text="a", plan="p1", step="s1")
    qp = queues.queue_path("sim")
    workq.mutate(
        qp,
        workq.load(qp)[0]["id"],
        status=workq.STATUS_IN_PROGRESS,
        started_at="2026-10-16T10:00:00+08:00",
    )
    lines = queues.water_lines()
    for m in queues.QUEUE_MEMBERS:
        assert any(ln.lstrip().startswith(f"[{m}]") for ln in lines), f"缺成员行：{m}"
    zero_line = next(ln for ln in lines if ln.lstrip().startswith("[figure]"))
    assert "（空）" in zero_line, "全零计数成员须明示（空）"
    inflight = [ln for ln in lines if "在途 " in ln]
    assert inflight, "在途明细行缺席"
    assert inflight[0].strip().startswith("· [sim]"), "在途明细行必须带成员名"
    assert "（p1·s1）" in inflight[0] and "起于 2026-10-16T10:00:00" in inflight[0]
    assert "在途合计：pending=0 in_progress=1" in lines[0]


def test_water_lines_empty_dir_lists_all_members_as_zero(monkeypatch):
    monkeypatch.setattr(queues, "QUEUES_DIR", Path("no-such-dir"))
    lines = queues.water_lines()
    joined = "\n".join(lines)
    assert lines[0].lstrip().startswith("在途合计：pending=0 in_progress=0")
    for m in queues.QUEUE_MEMBERS:
        assert any(f"[{m}]" in ln and "（空）" in ln for ln in lines), m
    assert "双零" in joined


# ---------------------------------------------------------------------------
# ③ taste：盲审读数/复核扩展
# ---------------------------------------------------------------------------
@pytest.fixture
def tface(tmp_path, monkeypatch):
    tdir = tmp_path / "taste"
    tdir.mkdir()
    monkeypatch.setattr(taste, "TASTE_DIR", tdir)
    jpath = tmp_path / "preference-journal.jsonl"
    monkeypatch.setattr(journal, "JOURNAL_PATH", jpath)
    case = tmp_path / "blind-case.md"
    case.write_text(
        "## 问题\n选哪个方案？\n\n## 选项 A\n甲方案\n\n## 选项 B\n乙方案\n\n## 证据\n数据 X\n",
        encoding="utf-8",
    )
    return SimpleNamespace(dir=tdir, case=case, journal=jpath)


def _taste_rec(cid: str, ts: str, case_file: str, **over) -> dict:
    base = {
        "ts": ts,
        "case_id": cid,
        "case_file": case_file,
        "case_head": "选哪个方案？",
        "choice": "A",
        "confidence": 0.7,
        "reason": "援引 P-03",
        "parsed": True,
        "body": "全文",
        "record_kind": "taste",
        "night": False,
        "authority": "",
    }
    base.update(over)
    taste.write_record(base)
    return base


def test_blind_queue_hides_twin_answers(tface):
    _taste_rec("c-pending", "2026-10-16T10:00:00+08:00", str(tface.case))
    _taste_rec(
        "c-done",
        "2026-10-16T09:00:00+08:00",
        str(tface.case),
        review={"ts": "2026-10-16T11:00:00+08:00", "verdict": "confirmed", "note": ""},
    )
    q = taste.blind_queue()
    assert [it["case_id"] for it in q] == ["c-pending"]  # 已复核的不入队；ts 升序
    item = q[0]
    assert set(item) == {
        "case_id",
        "ts",
        "case_file",
        "question",
        "option_a",
        "option_b",
        "options",
        "evidence",
        "case_head",
    }
    for forbidden in ("choice", "reason", "body", "confidence", "parsed"):
        assert forbidden not in item
    assert item["question"] == "选哪个方案？"
    assert item["option_a"] == "甲方案" and item["option_b"] == "乙方案"
    assert item["options"] == [("A", "甲方案"), ("B", "乙方案")]
    assert item["evidence"] == "数据 X"


def test_blind_queue_missing_case_falls_back_to_head(tface):
    _taste_rec("c-gone", "2026-10-16T10:00:00+08:00", str(tface.dir / "nope.md"))
    (item,) = taste.blind_queue()
    assert item["question"] == "选哪个方案？"  # 回落 case_head（可审不崩）
    assert "案卷缺失" in item["evidence"]


def test_blind_case_returns_full_record(tface):
    _taste_rec("c-full", "2026-10-16T10:00:00+08:00", str(tface.case))
    rec = taste.blind_case("c-full")
    assert rec is not None and rec["choice"] == "A" and rec["reason"]
    assert taste.blind_case("nope") is None


def test_ruling_rows_reviewed_new_to_old(tface):
    _taste_rec(
        "c-old",
        "2026-10-16T09:00:00+08:00",
        str(tface.case),
        review={
            "ts": "2026-10-16T10:00:00+08:00",
            "verdict": "confirmed",
            "note": "n1",
        },
    )
    _taste_rec(
        "c-new",
        "2026-10-16T11:00:00+08:00",
        str(tface.case),
        review={"ts": "2026-10-16T12:00:00+08:00", "verdict": "rejected", "note": "n2"},
        night=True,
        authority="夜航预授权",
    )
    rows = taste.ruling_rows()
    assert [r["case_id"] for r in rows] == ["c-new", "c-old"]
    assert set(rows[0]) == {
        "case_id",
        "ts",
        "choice",
        "confidence",
        "reason",
        "review",
        "night",
        "authority",
        "case_head",
    }
    assert rows[0]["review"]["verdict"] == "rejected"
    assert rows[0]["night"] is True and rows[0]["authority"] == "夜航预授权"
    assert len(taste.ruling_rows(limit=1)) == 1


def test_apply_review_backward_compat_and_blind_payload(tface, capsys):
    _taste_rec("c-rev", "2026-10-16T10:00:00+08:00", str(tface.case))
    # 旧调用（不传 blind）照常工作
    rc, _ = taste.apply_review("c-rev", verdict="confirmed", note="对")
    assert rc == 0
    rec = json.loads((tface.dir / "c-rev.json").read_text(encoding="utf-8"))
    assert rec["review"]["verdict"] == "confirmed" and rec["review"]["note"] == "对"
    assert rec["review"]["blind"] is None
    row = json.loads(tface.journal.read_text(encoding="utf-8").splitlines()[-1])
    assert row["kind"] == "verdict" and row["channel"] == "twin"
    # 新载荷：blind 原样并入 review
    blind = {
        "user_choice": "B",
        "agree": False,
        "remedy": "restore",
        "requirements": "恢复原实现",
    }
    rc, _ = taste.apply_review("c-rev", verdict="rejected", note="不一致", blind=blind)
    assert rc == 0
    rec = json.loads((tface.dir / "c-rev.json").read_text(encoding="utf-8"))
    assert rec["review"]["blind"] == blind


# ---------------------------------------------------------------------------
# ④ mirror：偏差报告队列（同队列同待遇）
# ---------------------------------------------------------------------------
@pytest.fixture
def mface(tmp_path, monkeypatch):
    q = tmp_path / "mirror-queue.jsonl"
    monkeypatch.setattr(mirror, "MIRROR_QUEUE_PATH", q)
    return q


DEV = {
    "case_id": "c-dev",
    "user_choice": "A",
    "twin_choice": "B",
    "reason": "与 P-07 冲突",
    "remedy": "restore",
    "requirements": "恢复原实现",
}


def test_append_deviation_roundtrip(mface):
    line = mirror.append_deviation(dict(DEV))
    assert line["kind"] == "deviation" and line["status"] == "pending"
    assert line["source"] == "blind-review" and line["ts"]
    assert "偏差报告 c-dev" in line["text"] and "处置=需恢复" in line["text"]
    pend = mirror.pending_deviations()
    assert len(pend) == 1 and pend[0]["case_id"] == "c-dev"
    assert mirror.pending_directives(), "偏差行 schema 与行动指令兼容（自动可见）"
    (dl,) = mirror.deviation_lines()
    assert "c-dev" in dl and "需恢复" in dl and "与 P-07 冲突" in dl


def test_deviation_done_leaves_pending_view(mface):
    mirror.append_deviation(dict(DEV, case_id="c2", remedy="none"))
    lines = mface.read_text(encoding="utf-8").splitlines()
    lines[0] = json.dumps(
        {**json.loads(lines[0]), "status": "done"}, ensure_ascii=False
    )
    mface.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert mirror.pending_deviations() == []
    assert mirror.deviation_lines() == []


def test_line_queues_reports_real_deviation_count(mface):
    assert "偏差报告待处理 = 0" in mirror._line_queues()
    mirror.append_deviation(dict(DEV, case_id="c3", remedy="other"))
    line = mirror._line_queues()
    assert "偏差报告待处理 = 1" in line
    assert "行动指令队列待处理 = 0" in line, "偏差不计入行动指令数（分开报）"


# ---------------------------------------------------------------------------
# ⑤ taskbooks：家族归组 + Markdown 详情
# ---------------------------------------------------------------------------
def test_families_on_real_taskbook_data():
    """真实 ``data/taskbooks/`` 实测：既有返工家族 tb-20261015-194834-3 在册且归组自洽。

    **不钉死「latest 是哪个具体 id」**：该家族随返工轮次增长——实测 2026-10-17 组长新开
    返工第 2 轮（`tb-20261017-044604-3-2`），旧断言 `latest == tb-20261015-225652-3-1`
    随之失效变红（把「此刻的真实数据」当出厂判据的经典坑）。改为**数据自证**：latest 必须是
    族内 ``(rework_round, ts)`` 序最大者——判据仍咬「排序对不对」，但不随真实数据漂。
    """
    fams = taskbooks.families()
    assert fams
    by_root = {f["root"]: f for f in fams}
    fam = by_root.get("tb-20261015-194834-3")
    assert fam is not None, "既有返工家族 tb-20261015-194834-3 必须在册"

    newest = max(
        fam["members"],
        key=lambda m: (int(m.get("rework_round") or 0), str(m.get("ts") or "")),
    )
    assert fam["latest"] == str(newest["id"]), "latest 须＝族内轮次/时刻最大者"
    assert fam["members"][0]["id"] == fam["latest"], "展示序首项＝最新"
    assert fam["members"][-1]["id"] == fam["root"], "展示序末项＝本体"
    assert taskbooks.family_label(fam["members"][-1]) == "本体"
    assert fam["size"] == len(fam["members"]) >= 2
    assert all(
        taskbooks.family_label(m).startswith("返工第") for m in fam["members"][:-1]
    ), "非本体成员都该是返工轮次标签"
    # 每件任务书恰好属于一家；家族列表按 latest 的 ts 降序
    assert sum(f["size"] for f in fams) == len(taskbooks.list_all())
    tss = [
        str(next(m for m in f["members"] if m["id"] == f["latest"]).get("ts") or "")
        for f in fams
    ]
    assert tss == sorted(tss, reverse=True)


def test_families_orphan_and_multi_child(tmp_path, monkeypatch):
    tbdir = tmp_path / "taskbooks"
    tbdir.mkdir()
    monkeypatch.setattr(taskbooks, "TASKBOOKS_DIR", tbdir)

    def mk(tid: str, ts: str, ro: str = "", rnd: int = 0) -> None:
        state = {
            "id": tid,
            "ts": ts,
            "title": tid,
            "status": "new",
            "rework_of": ro,
            "rework_round": rnd,
            "rejections": [],
        }
        (tbdir / f"{tid}.state.json").write_text(
            json.dumps(state, ensure_ascii=False), encoding="utf-8"
        )

    mk("tb-root", "2026-10-16T08:00:00+08:00")
    mk("tb-r1", "2026-10-16T09:00:00+08:00", ro="tb-root", rnd=1)
    mk("tb-r2", "2026-10-16T10:00:00+08:00", ro="tb-r1", rnd=2)
    mk("tb-sib", "2026-10-16T11:00:00+08:00", ro="tb-root", rnd=1)  # 多子分叉
    mk("tb-orphan", "2026-10-16T12:00:00+08:00", ro="tb-gone", rnd=3)  # 孤儿

    fams = taskbooks.families()
    by_root = {f["root"]: f for f in fams}
    fam = by_root["tb-root"]
    assert [m["id"] for m in fam["members"]] == ["tb-r2", "tb-sib", "tb-r1", "tb-root"]
    assert fam["latest"] == "tb-r2" and fam["size"] == 4
    orphan = by_root["tb-orphan"]  # 指向不存在 id 的项自成一家
    assert orphan["size"] == 1 and orphan["latest"] == "tb-orphan"
    assert [f["root"] for f in fams] == ["tb-orphan", "tb-root"]  # latest.ts 降序


def test_md_path_public(tmp_path, monkeypatch):
    tbdir = tmp_path / "taskbooks"
    monkeypatch.setattr(taskbooks, "TASKBOOKS_DIR", tbdir)
    assert taskbooks.md_path("tb-x") == tbdir / "tb-x.taskbook.md"


@pytest.fixture
def tbface(tmp_path, monkeypatch):
    tbdir = tmp_path / "taskbooks"
    tbdir.mkdir()
    monkeypatch.setattr(taskbooks, "TASKBOOKS_DIR", tbdir)
    return tbdir


def _mk_tb(tbdir: Path, tid: str, status: str) -> None:
    (tbdir / f"{tid}.taskbook.md").write_text(
        f"---\nid: {tid}\n---\n\n正文内容\n", encoding="utf-8"
    )
    (tbdir / f"{tid}.state.json").write_text(
        json.dumps(
            {
                "id": tid,
                "ts": "2026-10-16T10:00:00+08:00",
                "title": "MD 大任务",
                "status": status,
                "plan_id": "",
                "rework_of": "",
                "rework_round": 0,
                "rejections": [{"ts": "2026-10-16T11:00:00+08:00", "note": "批语一"}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_detail_markdown_delivered_has_star_and_report(tbface):
    _mk_tb(tbface, "tb-md-1", "delivered")
    md = taskbooks.detail_markdown("tb-md-1")
    assert md.startswith("# 任务书 tb-md-1")
    assert "★" in md and "批复参考" in md
    assert "全流程报告" in md  # 流程报告（report_sections 输出内嵌）
    assert "## 任务书正文" in md and "正文内容" in md
    assert "## 批语历史" in md and "批语一" in md
    # 文本通道共享同一组装件；「批复入口」提示行已从数据面清除
    lines = taskbooks.detail_lines("tb-md-1")
    assert any("★ 批复参考" in ln for ln in lines)
    assert not any("批复入口" in ln for ln in lines)


def test_detail_markdown_undelivered_report_at_bottom(tbface):
    _mk_tb(tbface, "tb-md-2", "new")
    md = taskbooks.detail_markdown("tb-md-2")
    assert "## 流程报告" in md and "★" not in md
    assert not any("批复入口" in ln for ln in taskbooks.detail_lines("tb-md-2"))
