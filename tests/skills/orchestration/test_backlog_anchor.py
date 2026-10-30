"""销账锚点校验（backlog 20261015-145953-f4b）。

实案：组长裁量销账 TUI-UX 单，「账上 done 实际未交付」静默 3 小时（成果卡 worktree）。
本单把「账实相符」机械化：销账必经 :func:`workflow.delivery_anchor` 三选一——
① 合并提交在 main ② 留档/证据在盘 ③ 显式无产物标记；无锚即拒收并转 needs_leader 出声。
**生效边界**：只管销账这一刻，不回溯改判已 done 的历史条目。
"""

from __future__ import annotations

import json
from pathlib import Path

from pysci.skills.orchestration.tools import drain, workflow


def _seed(state: Path, item_id="item-a", status="pending", **extra) -> None:
    p = state / "backlog.json"
    data = (
        json.loads(p.read_text(encoding="utf-8"))
        if p.exists()
        else {"version": 1, "items": []}
    )
    data["items"].append(
        {"id": item_id, "member": "devops", "status": status, "summary": "s", **extra}
    )
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def _find(state: Path, item_id: str) -> dict:
    data = json.loads((state / "backlog.json").read_text(encoding="utf-8"))
    return next(i for i in data["items"] if i["id"] == item_id)


# ---------------------------------------------------------------------------
# 三类锚点各放行
# ---------------------------------------------------------------------------
def test_anchor_merge_commit_in_main(iso_state, monkeypatch):
    monkeypatch.setattr(
        drain, "_run_git", lambda *a: (0, "") if a[0] == "merge-base" else (1, "")
    )
    ok, why = workflow.delivery_anchor({"id": "x", "commit": "abcdef123456"})
    assert ok and "合并提交在 main" in why


def test_anchor_merge_commit_not_in_main_refused(iso_state, monkeypatch):
    monkeypatch.setattr(drain, "_run_git", lambda *a: (1, ""))
    ok, why = workflow.delivery_anchor({"id": "x", "commit": "abcdef123456"})
    assert not ok and "不在 main" in why, "已提交未合并 ≠ 已交付"


def test_anchor_delivery_archive_on_disk(iso_state, monkeypatch):
    from pysci.skills.orchestration.tools import queueworker

    d = iso_state / "deliveries"
    d.mkdir()
    monkeypatch.setattr(queueworker, "DELIVERIES_DIR", d)
    (d / "devops-20261015-000000-item-a.md").write_text("x", encoding="utf-8")
    ok, why = workflow.delivery_anchor({"id": "item-a"})
    assert ok and "交付留档在盘" in why, "批次派发落的是 <member>-<ts>-<item_id>.md"
    assert workflow.delivery_anchor({"id": "item-b"})[0] is False


def test_anchor_explicit_no_artifact_marker(iso_state):
    assert workflow.delivery_anchor({"id": "x", "no_artifact": True})[0]
    ok, why = workflow.delivery_anchor(
        {"id": "x", "note": "纯探针（no_artifact）：无产物"}
    )
    assert ok and "无产物标记" in why


# ---------------------------------------------------------------------------
# 销账路径接入：无锚拒收 + 出声；有锚放行
# ---------------------------------------------------------------------------
def test_complete_without_anchor_is_refused_and_needs_leader(iso_state, capsys):
    _seed(iso_state)
    assert workflow.backlog_complete("item-a", "做完了") is False
    row = _find(iso_state, "item-a")
    assert row["status"] == "needs_leader", "无锚 → 拒绝 done 并转 needs_leader"
    assert row["anchor_missing"] and "无锚点" in row["anchor_missing"]
    out = capsys.readouterr().out
    assert "拒绝销账 item-a" in out and "无锚点" in out, "必须出声点名缺什么"


def test_complete_with_anchor_passes(iso_state, monkeypatch):
    monkeypatch.setattr(workflow, "ORCH_STATE_ROOT", iso_state)
    d = iso_state / "deliveries"
    d.mkdir()
    (d / "item-a.md").write_text("x", encoding="utf-8")
    _seed(iso_state)
    assert workflow.backlog_complete("item-a", "做完了") is True
    assert _find(iso_state, "item-a")["status"] == "done"


def test_resolve_leader_path_also_gated(iso_state, capsys):
    """组长裁量销账（needs_leader → done）同样过闸：无锚即拒（条目留在 needs_leader）。"""
    _seed(iso_state, status="needs_leader")
    assert workflow.backlog_resolve("item-a", note="组长销账") is None
    assert _find(iso_state, "item-a")["status"] == "needs_leader"
    assert "拒绝销账 item-a" in capsys.readouterr().out


def test_resolve_passes_with_marker(iso_state):
    _seed(iso_state, status="needs_leader", note="无产物任务 no_artifact")
    assert workflow.backlog_resolve("item-a", note="组长销账") is not None
    assert _find(iso_state, "item-a")["status"] == "done"


# ---------------------------------------------------------------------------
# 爆炸半径：历史条目零误伤（只对新销账生效）
# ---------------------------------------------------------------------------
def test_already_done_history_is_untouched(iso_state, monkeypatch):
    monkeypatch.setattr(workflow, "ORCH_STATE_ROOT", iso_state)
    _seed(iso_state, item_id="old-1", status="done")
    # 校验只在销账那一刻跑：已 done 的历史条目不会被回溯改判
    assert _find(iso_state, "old-1")["status"] == "done"
    assert workflow.delivery_anchor(_find(iso_state, "old-1"))[0] is False, (
        "（对照）该历史条目确实无锚——但状态不被回溯改判"
    )
    assert _find(iso_state, "old-1")["status"] == "done"
