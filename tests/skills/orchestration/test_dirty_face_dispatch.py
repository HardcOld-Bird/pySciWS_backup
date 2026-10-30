"""派发前哨：声明面 ∩ main 未提交跟踪面 → 派发前告警（backlog 20261016-022657）。

现象：`orchestration/state/registry.json` 是高频脏面（成员注册/档位/check 调整都改），需改它的
基建单几乎必与它相交 → `pysci-dev worktree merge` 稳定 rc=2、组员白烧一跳（`cb36d9c` 前例：
reviewer 并发写 `agents.md` 时任何改该索引行的单也撞）。merge 工具已把交集清单摊开，缺的是
**派发前**把同一事实搬到组长眼前——组员侧无法自救（他们那里同样 rc=2、整单被抢救成 blocked）。
"""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

from pysci.skills.orchestration.tools import dispatch, plans, queues, queueworker

DIRTY = "orchestration/state/registry.json"


def _dirty(monkeypatch, paths: list[str]) -> None:
    """把脏面读数钉成给定清单（真 git 状态不可控，故注入）。"""
    monkeypatch.setattr(dispatch, "dirty_tracked_state", lambda **k: list(paths))


# ---------------------------------------------------------------------------
# ① 交集合与告警文案
# ---------------------------------------------------------------------------
def test_clash_lists_intersection(monkeypatch):
    _dirty(monkeypatch, [DIRTY])
    assert dispatch.dirty_face_clash(["orchestration/state"]) == [DIRTY]


def test_clash_empty_when_clean(monkeypatch):
    _dirty(monkeypatch, [])
    assert dispatch.dirty_face_clash(["src", "tests"]) == []


def test_warn_names_paths_and_gives_two_ways(monkeypatch, capsys):
    _dirty(monkeypatch, [DIRTY])
    capsys.readouterr()

    clash = dispatch.warn_dirty_face_clash(["orchestration/state"])

    out = capsys.readouterr().out
    assert clash == [DIRTY]
    assert DIRTY in out, out
    assert "先提交/还原" in out and "任务书写一行" in out, out
    assert dispatch.DIRTY_FACE_ACK_MARK in out, out


def test_ack_marker_silences_the_warning(monkeypatch, tmp_path, capsys):
    """任务书写了「本单由组长先清面」＝已知认账 → 不再重复告警。"""
    _dirty(monkeypatch, [DIRTY])
    tb = tmp_path / "t.task.md"
    tb.write_text(f"# 任务\n\n{dispatch.DIRTY_FACE_ACK_MARK}\n", encoding="utf-8")
    capsys.readouterr()

    assert dispatch.warn_dirty_face_clash(["orchestration/state"], tb) == []
    assert capsys.readouterr().out == ""


def test_ack_marker_detected_relative_to_project_root(monkeypatch, tmp_path):
    """相对路径的任务书（state 里存的就是相对路径）按项目根解析。"""
    monkeypatch.setattr(dispatch, "PROJECT_ROOT", tmp_path)
    (tmp_path / "x.task.md").write_text(dispatch.DIRTY_FACE_ACK_MARK, encoding="utf-8")
    assert dispatch.taskbook_acks_dirty_face(Path("x.task.md")) is True
    assert dispatch.taskbook_acks_dirty_face(Path("nope.md")) is False


# ---------------------------------------------------------------------------
# ② 组长面：plan advance / plan run 的 stdout 上必须先说
# ---------------------------------------------------------------------------
def _plan_args(plan_id: str, **over) -> Namespace:
    base = dict(
        id=plan_id,
        file="",
        force=True,
        text=None,
        task=None,
        name=None,
        session=None,
        model_tier=None,
        effort=None,
        max_turns=None,
        dirs=None,
        evidence=None,
        dry=False,
    )
    base.update(over)
    return Namespace(**base)


@pytest.fixture
def face(tmp_path, monkeypatch):
    d = tmp_path / "plans"
    d.mkdir(exist_ok=True)
    monkeypatch.setattr(plans, "PLANS_DIR", d)
    monkeypatch.setattr(queueworker, "wake_member", lambda m, **k: (True, "stub"))
    return d


def _mk_plan(face_dir: Path, plan_id: str, *, dirs: list[str]) -> str:
    import yaml

    p = face_dir / f"{plan_id}.plan.yaml"
    p.write_text(
        yaml.safe_dump(
            {
                "id": plan_id,
                "goal": "目标",
                "steps": [{"id": "s1", "member": "sim", "brief": "一", "dirs": dirs}],
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    plans.cmd_plan_new(_plan_args(plan_id, file=str(p)))
    return plan_id


def test_plan_advance_warns_on_dirty_declared_face(
    iso_state, face, monkeypatch, capsys
):
    _dirty(monkeypatch, [DIRTY])
    _mk_plan(face, "p-dir", dirs=["orchestration/state"])
    capsys.readouterr()

    assert plans.cmd_plan_advance(_plan_args("p-dir", text="t")) == 0

    out = capsys.readouterr().out
    assert DIRTY in out and "交集闸" in out, out


def test_plan_advance_silent_when_clean(iso_state, face, monkeypatch, capsys):
    _dirty(monkeypatch, [])
    _mk_plan(face, "p-clean", dirs=["orchestration/state"])
    capsys.readouterr()

    assert plans.cmd_plan_advance(_plan_args("p-clean", text="t")) == 0
    assert DIRTY not in capsys.readouterr().out


def test_plan_advance_silent_when_taskbook_acks(iso_state, face, monkeypatch, capsys):
    _dirty(monkeypatch, [DIRTY])
    _mk_plan(face, "p-ack", dirs=["orchestration/state"])
    tb = face / "p-ack.task.md"
    tb.write_text(f"# 任务\n\n{dispatch.DIRTY_FACE_ACK_MARK}\n", encoding="utf-8")
    capsys.readouterr()

    assert plans.cmd_plan_advance(_plan_args("p-ack", task=str(tb))) == 0
    assert DIRTY not in capsys.readouterr().out


def test_plan_run_warns_too(iso_state, face, monkeypatch, capsys):
    """阻塞原语（`plan run`/`plan adhoc` 共用）同样先哨。"""
    _dirty(monkeypatch, [DIRTY])
    _mk_plan(face, "p-run", dirs=["orchestration/state"])
    monkeypatch.setattr(plans, "_wait_all_inflight", lambda *a, **k: [])
    capsys.readouterr()

    plans.cmd_plan_run(_plan_args("p-run", text="t"))

    assert DIRTY in capsys.readouterr().out


def test_queues_enqueue_path_unaffected(iso_state):
    """回归护栏：本前哨只加告警，不改入队行为（条目照常入队）。"""
    queues.enqueue("sim", text="job")
    assert len(queues.items("sim")) == 1
