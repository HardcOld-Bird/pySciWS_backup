"""要求与历史总览（用户 2026-10-17 裁决的返工上下文修复）。

钉四件事：① 总览从**家族链**机械重建（原始要求 + 历轮批语带轮次归属 + 历轮收获），
存量脏正文同样受益；② 返工件正文＝规范化总览（不再复制堆叠）；③ 注入面——
详情置顶/批语历史全家族/报告历轮摘要归属/sync 自动补报告；④ 自动附送——派单任务书
与 jury 案卷（组长遗忘不得传递给组员/裁判）。
"""

from __future__ import annotations

import json

import pytest

from pysci import paths as _paths
from pysci.skills.orchestration.tools import dispatch, jury, plans, taskbooks


@pytest.fixture
def face(tmp_path, monkeypatch):
    """隔离任务书目录 / 计划目录 / 数据根。"""
    data = tmp_path / "data"
    monkeypatch.setattr(taskbooks, "TASKBOOKS_DIR", data / "taskbooks")
    monkeypatch.setattr(_paths, "DATA_ROOT", data)
    pdir = tmp_path / "plans"
    monkeypatch.setattr(plans, "PLANS_DIR", pdir)
    return tmp_path


def _family(face):
    """root（原始要求）→ rw1（批语 1）→ rw2（批语 2）；返回 (root, rw1, rw2)。"""
    root = taskbooks.create("大任务", "用户原始要求：修好 X 并保持 Y")["id"]
    rw1 = taskbooks.create_rework(root, improve={}, note="X 还是没修好")["id"]
    rw2 = taskbooks.create_rework(rw1, improve={}, note="Y 被破坏了")["id"]
    return root, rw1, rw2


# ---------------------------------------------------------------------------
# ① 总览重建
# ---------------------------------------------------------------------------
def test_digest_rebuilds_requirements_notes_and_harvest(face):
    root, rw1, rw2 = _family(face)
    digest = taskbooks.requirements_digest(rw2)
    assert digest.startswith("## 要求与历史总览（自动附送）")
    # ① 原始要求＝家族根正文（不被返工前缀/批语段淹没）
    assert "### ① 用户原始要求" in digest and "用户原始要求：修好 X 并保持 Y" in digest
    # ② 批语逐条带轮次归属 + 驳回对象
    assert "### ② 历轮驳回批语" in digest
    assert "#### 第 1 轮返工批语" in digest and "X 还是没修好" in digest
    assert f"，驳回 {root}）" in digest
    assert "#### 第 2 轮返工批语" in digest and "Y 被破坏了" in digest
    assert f"，驳回 {rw1}）" in digest
    # ③ 历轮收获（防复发）：本件（rw2）自己不入，前两轮入
    assert "### ③ 历轮收获（改进与交付摘要——防复发）" in digest
    assert f"（{root}）" in digest and f"（{rw1}）" in digest
    assert f"（{rw2}）" not in digest


def test_digest_from_dirty_legacy_data(face):
    """存量脏数据：正文复制堆叠 + rejections 无 round 字段 → 总览仍干净重建。"""
    root, rw1, _rw2 = _family(face)
    # 模拟旧格式：rw1 正文是嵌套堆叠的怪物
    (face / "data" / "taskbooks" / f"{rw1}.taskbook.md").write_text(
        "---\n"
        f"id: {rw1}\n"
        "---\n\n"
        "（返工第 2 轮，原任务书 tb-x；模式 full）\n"
        "（返工第 1 轮，原任务书 tb-y；模式 full）\n"
        "用户原始要求：修好 X 并保持 Y\n"
        "## 本轮驳回批语\nX 还是没修好\n"
        "## 本轮驳回批语\nY 被破坏了\n",
        encoding="utf-8",
    )
    # 模拟旧格式：批语条目无 round 字段
    p = face / "data" / "taskbooks" / f"{root}.state.json"
    st = json.loads(p.read_text(encoding="utf-8"))
    for r in st.get("rejections") or []:
        r.pop("round", None)
    p.write_text(json.dumps(st, ensure_ascii=False), encoding="utf-8")

    digest = taskbooks.requirements_digest(rw1)
    assert digest.count("X 还是没修好") == 1, "批语只出现一次（不被脏正文重复放大）"
    assert "#### 第 1 轮返工批语" in digest, "无 round 字段按「被驳回件轮次+1」推算"
    assert digest.count("## 本轮驳回批语") == 0, "脏正文的重复段不进总览"


def test_digest_has_history_flag(face):
    fresh = taskbooks.create("新件", "正文")["id"]
    assert taskbooks.digest_has_history(fresh) is False
    root, rw1, _ = _family(face)
    assert taskbooks.digest_has_history(root) is True
    assert taskbooks.digest_has_history(rw1) is True


# ---------------------------------------------------------------------------
# ② 返工正文规范化
# ---------------------------------------------------------------------------
def test_create_rework_body_is_canonical_digest(face):
    root = taskbooks.create("大任务", "原始要求")["id"]
    rw = taskbooks.create_rework(root, improve={"leader": True}, note="这版不行")
    body = taskbooks.read_body(rw["id"])
    assert body.startswith(f"（返工第 1 轮，原任务书 {root}；改进项 leader）")
    assert "## 要求与历史总览（自动附送）" in body
    assert body.count("这版不行") == 1, "批语只在总览 ② 出现一次"
    assert "## 本轮驳回批语" not in body, "旧式重复段不再堆叠"


# ---------------------------------------------------------------------------
# ③ 注入面
# ---------------------------------------------------------------------------
def test_detail_digest_top_and_family_rejections(face):
    root, rw1, rw2 = _family(face)
    lines = taskbooks.detail_lines(rw2)
    text = "\n".join(lines)
    assert "要求与历史总览" in text
    assert text.index("要求与历史总览") < text.index("任务书正文"), "总览置顶（正文前）"
    # 批语历史＝全家族（进行中返工件自己的 rejections 为空，不再显示「（无）」）
    assert "批语历史（2 条）" in text
    assert "第 1 轮批语（" in text and "X 还是没修好" in text
    assert "第 2 轮批语（" in text and "Y 被破坏了" in text


def test_report_round_summary_carry_note_rounds(face):
    root, rw1, rw2 = _family(face)
    sections = taskbooks.report_sections(rw2)
    hist = next(s for s in sections if s.startswith("## 历轮摘要"))
    assert f"初版 {root}" in hist
    assert f"第 1 轮 {rw1}" in hist
    assert "第 1 轮批语：X 还是没修好" in hist, "逐条批语带轮次归属"
    assert "第 2 轮批语：Y 被破坏了" in hist


def test_sync_delivered_auto_builds_report(face, monkeypatch):
    from pysci.skills.orchestration.tools import plans as _plans

    tid = taskbooks.create("交付件", "正文")["id"]
    taskbooks.link_plan(tid, "p-ok")
    monkeypatch.setattr(_plans, "load_state", lambda pid: {"completed_at": "T"})
    assert taskbooks.sync_delivered() == [tid]
    rp = taskbooks.report_path(tid)
    assert rp.exists(), "交付即补全流程报告（③历轮收获的取数源）"
    assert "全流程报告" in rp.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# ④ 自动附送（派单任务书 / jury 案卷）
# ---------------------------------------------------------------------------
def _seed_plan(tmp_path, monkeypatch, plan_id, steps):
    d = tmp_path / "plans"
    d.mkdir(parents=True, exist_ok=True)
    body = "goal: g\nsteps:\n" + "".join(
        f"  - id: {s}\n    member: devops\n" for s in steps
    )
    (d / f"{plan_id}.plan.yaml").write_text(body, encoding="utf-8")
    monkeypatch.setattr(plans, "PLANS_DIR", d)
    return d


def test_write_taskbook_attaches_digest_for_taskbook_plans(face, tmp_path, monkeypatch):
    _seed_plan(tmp_path, monkeypatch, "fix-20261014", ["f1"])
    root = taskbooks.create("大任务", "用户原始要求：修好 X")["id"]
    rw = taskbooks.create_rework(root, improve={}, note="没修好")["id"]
    taskbooks.link_plan(rw, "fix-20261014")
    dest = dispatch.write_taskbook(
        tmp_path / "pod", text="干这个活", task_file=None, slug="fix-20261014-f1"
    )
    text = dest.read_text(encoding="utf-8")
    assert "要求与历史总览（自动附送）" in text
    assert "用户原始要求：修好 X" in text and "没修好" in text
    assert text.index("干这个活") < text.index("要求与历史总览"), "总览附在任务书末尾"


def test_write_taskbook_skips_adhoc_without_taskbook(face, tmp_path, monkeypatch):
    _seed_plan(tmp_path, monkeypatch, "fix-20261014", ["f1"])
    dest = dispatch.write_taskbook(
        tmp_path / "pod", text="计划外单件", task_file=None, slug="fix-20261014-f1"
    )
    assert "要求与历史总览" not in dest.read_text(encoding="utf-8")


def test_jury_material_attaches_digest(face):
    # 唯一标记（uuid）：正例不得靠真实 git diff 里的同名字面量「假阳性」通过。
    import uuid

    mark = f"用户原始要求哨兵-{uuid.uuid4().hex}"
    root = taskbooks.create("大任务", mark)["id"]
    rw = taskbooks.create_rework(root, improve={}, note="Y 被破坏")["id"]
    case = jury.Case(case_id=f"{rw}-jury", question="q", option_a="a", option_b="b")
    material = jury.build_material(case)
    assert "要求与历史总览（自动附送）" in material
    assert mark in material and "Y 被破坏" in material


def test_jury_material_skips_unrelated_case(face):
    # 反例同理：材料含真实 git diff（可能恰好含同名字面量）→ 断言只认**唯一标记**。
    import uuid

    mark = f"用户原始要求哨兵-{uuid.uuid4().hex}"
    root = taskbooks.create("大任务", mark)["id"]
    taskbooks.create_rework(root, improve={}, note="批语")["id"]
    case = jury.Case(
        case_id="devops-proposal-20261017", question="q", option_a="a", option_b="b"
    )
    assert mark not in jury.build_material(case)
