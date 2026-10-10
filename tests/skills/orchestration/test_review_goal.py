"""review 任务书嵌入目标陈述回归（backlog 20261009-231455-reviewer）。

锁定：
1. ``_extract_goal_excerpt`` 剥离 write_taskbook 自动头、限长、不可读→""；
2. ``cmd_review`` 把 ``--goal-from`` 摘录 / ``--goal`` 文本嵌入审查任务书，并登记进
   review 记录（goal_source/goal_excerpt）；``--goal`` 优先于 ``--goal-from``；
3. 目标块含「勿以生产者 notes.md 替代」独立性指令，且 FAIL→复审第二轮仍携带；
4. 缺省（无 goal）向后兼容：不嵌目标块、告警、记录字段为空。

以 monkeypatch 假 do_dispatch 避免真派发；PODS_ROOT/REVIEWS_DIR 重定向到 tmp 隔离。
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pysci.paths import PROJECT_ROOT
from pysci.skills.orchestration.tools import workflow
from pysci.skills.orchestration.tools.dispatch import DispatchOutcome
from pysci.skills.orchestration.tools.workflow import _extract_goal_excerpt, cmd_review

# --- 夹具 ---------------------------------------------------------------


@pytest.fixture
def iso_review(tmp_path, monkeypatch):
    """隔离 REVIEWS_DIR 与 PODS_ROOT（造假 rubric），返回 (reviews_dir, pods_root)。"""
    reviews = tmp_path / "reviews"
    pods = tmp_path / "pods"
    (pods / "reviewer" / "rubrics").mkdir(parents=True)
    (pods / "reviewer" / "rubrics" / "generic.md").write_text(
        "# generic rubric\n", encoding="utf-8"
    )
    monkeypatch.setattr(workflow, "REVIEWS_DIR", reviews)
    import pysci.paths

    monkeypatch.setattr(pysci.paths, "PODS_ROOT", pods)
    return reviews, pods


def _fake_dispatch_factory(script: list[str], captured: list[dict]):
    """按 script 顺序为 reviewer 跳返回 VERDICT；origin 返工跳恒成功。"""

    def fake(member, **kw):
        captured.append({"member": member, "text": kw.get("text", "")})
        if member == "reviewer":
            verdict = script[
                len([c for c in captured if c["member"] == "reviewer"]) - 1
            ]
            return DispatchOutcome(
                code=0,
                kind="result",
                member=member,
                sid=f"sid-{len(captured)}",
                body=f"<verdict>{verdict}</verdict>",
            )
        return DispatchOutcome(code=0, kind="result", member=member, sid="rework-sid")

    return fake


def _args(**over):
    base = dict(
        artifact="data/research/1_gain_ep/article/figures/fig0",
        origin="figure",
        rubric="generic",
        goal=None,
        goal_from=None,
    )
    base.update(over)
    return SimpleNamespace(**base)


# --- 1. _extract_goal_excerpt ------------------------------------------


def test_extract_strips_auto_header(tmp_path):
    tb = tmp_path / "task-x.md"
    tb.write_text(
        "# 任务书 fig0\n\n（派发时间：2026-10-09T20:25:08+08:00）\n\n"
        "请完成收尾：\n1. 移植绘图逻辑；\n2. 走正规管线 build。\n",
        encoding="utf-8",
    )
    out = _extract_goal_excerpt(tb)
    assert out.startswith("请完成收尾")
    assert "# 任务书" not in out
    assert "派发时间" not in out
    assert "移植绘图逻辑" in out


def test_extract_bounds_length(tmp_path):
    tb = tmp_path / "big.md"
    tb.write_text("# 任务书 b\n\n" + "甲" * 5000, encoding="utf-8")
    out = _extract_goal_excerpt(tb, limit=100)
    assert len(out) <= 100 + len(" …（截断）")
    assert out.endswith("…（截断）")


def test_extract_missing_file_returns_empty(tmp_path):
    assert _extract_goal_excerpt(tmp_path / "nope.md") == ""


def test_extract_resolves_relative_to_project_root(monkeypatch, tmp_path):
    """相对路径按 PROJECT_ROOT 解析（与 write_taskbook 一致）。"""
    fake_root = tmp_path / "root"
    (fake_root / "orchestration" / "pods" / "figure" / "inbox").mkdir(parents=True)
    rel = Path("orchestration/pods/figure/inbox/task-rel.md")
    (fake_root / rel).write_text("# 任务书 r\n\n相对路径目标正文。\n", encoding="utf-8")
    monkeypatch.setattr(workflow, "PROJECT_ROOT", fake_root)
    assert "相对路径目标正文" in _extract_goal_excerpt(rel)


# --- 2. cmd_review 嵌入 -------------------------------------------------


def test_goal_from_embeds_excerpt_and_records(iso_review, monkeypatch, tmp_path):
    reviews, _ = iso_review
    tb = tmp_path / "origin-task.md"
    tb.write_text(
        "# 任务书 fig0\n\n（派发时间：2026-10-09T20:25:08+08:00）\n\n"
        "绘制避免交叉色散单面板 smoke 图，绑定 aps/single 85mm。\n",
        encoding="utf-8",
    )
    captured: list[dict] = []
    monkeypatch.setattr(
        workflow, "do_dispatch", _fake_dispatch_factory(["PASS"], captured)
    )
    rc = cmd_review(_args(goal_from=str(tb)))
    assert rc == 0
    task_text = captured[0]["text"]
    assert "绘制避免交叉色散单面板 smoke 图" in task_text
    assert "目标陈述" in task_text
    # 独立性指令：勿依赖生产者自述
    assert "notes.md" in task_text
    # 记录登记 goal_source/goal_excerpt
    rec = json.loads(
        (reviews / f"{_review_id(reviews)}.json").read_text(encoding="utf-8")
    )
    assert rec["goal_excerpt"].startswith("绘制避免交叉色散")
    assert rec["goal_source"].endswith("origin-task.md")


def test_goal_literal_embeds_and_marks_source(iso_review, monkeypatch):
    reviews, _ = iso_review
    captured: list[dict] = []
    monkeypatch.setattr(
        workflow, "do_dispatch", _fake_dispatch_factory(["PASS"], captured)
    )
    rc = cmd_review(_args(goal="一句话目标：单面板 smoke 图"))
    assert rc == 0
    assert "一句话目标：单面板 smoke 图" in captured[0]["text"]
    rec = json.loads(
        (reviews / f"{_review_id(reviews)}.json").read_text(encoding="utf-8")
    )
    assert rec["goal_source"] == "literal(--goal)"
    assert rec["goal_excerpt"] == "一句话目标：单面板 smoke 图"


def test_goal_literal_takes_precedence_over_goal_from(
    iso_review, monkeypatch, tmp_path
):
    tb = tmp_path / "o.md"
    tb.write_text("# 任务书 o\n\n来自文件的目标。\n", encoding="utf-8")
    captured: list[dict] = []
    monkeypatch.setattr(
        workflow, "do_dispatch", _fake_dispatch_factory(["PASS"], captured)
    )
    cmd_review(_args(goal="直接文本目标", goal_from=str(tb)))
    assert "直接文本目标" in captured[0]["text"]
    assert "来自文件的目标" not in captured[0]["text"]


def test_goal_block_persists_into_second_round(iso_review, monkeypatch):
    """FAIL→返工→复审：第二轮审查任务书仍携带目标块。"""
    captured: list[dict] = []
    monkeypatch.setattr(
        workflow, "do_dispatch", _fake_dispatch_factory(["FAIL", "PASS"], captured)
    )
    rc = cmd_review(_args(goal="目标陈述ABC"))
    assert rc == 0
    reviewer_texts = [c["text"] for c in captured if c["member"] == "reviewer"]
    assert len(reviewer_texts) == 2
    assert all("目标陈述ABC" in t for t in reviewer_texts)
    # 返工跳派给 origin，不含目标块（生产者已知目标）
    origin_texts = [c["text"] for c in captured if c["member"] == "figure"]
    assert origin_texts and "目标陈述ABC" not in origin_texts[0]


# --- 3. 缺省向后兼容 ----------------------------------------------------


def test_no_goal_is_backward_compatible(iso_review, monkeypatch, capsys):
    reviews, _ = iso_review
    captured: list[dict] = []
    monkeypatch.setattr(
        workflow, "do_dispatch", _fake_dispatch_factory(["PASS"], captured)
    )
    rc = cmd_review(_args())
    assert rc == 0
    assert "目标陈述" not in captured[0]["text"]
    rec = json.loads(
        (reviews / f"{_review_id(reviews)}.json").read_text(encoding="utf-8")
    )
    assert rec["goal_source"] == ""
    assert rec["goal_excerpt"] == ""
    # 告警提示缺独立目标依据
    assert "未提供目标陈述" in capsys.readouterr().out


def test_goal_from_unreadable_warns_and_continues(
    iso_review, monkeypatch, tmp_path, capsys
):
    captured: list[dict] = []
    monkeypatch.setattr(
        workflow, "do_dispatch", _fake_dispatch_factory(["PASS"], captured)
    )
    rc = cmd_review(_args(goal_from=str(tmp_path / "missing.md")))
    assert rc == 0  # 不中断审查
    assert "目标陈述" not in captured[0]["text"]
    assert "--goal-from 任务书不可读" in capsys.readouterr().out


# --- 工具 ---------------------------------------------------------------


def _review_id(reviews: Path) -> str:
    """取 reviews 目录下唯一记录文件的 stem。"""
    files = list(reviews.glob("rev-*.json"))
    assert len(files) == 1, files
    return files[0].stem


def test_project_root_import_is_used():
    """守卫：workflow 须能从 PROJECT_ROOT 解析相对任务书路径（导入面稳定）。"""
    assert workflow.PROJECT_ROOT == PROJECT_ROOT
