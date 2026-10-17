"""orch 失败/恢复指引二处的按因分型（backlog 20261017-104500 + 104502）。

① twin consult 的 rc=2 一码两义：冷启动拒复用 ≠「twin 不可达」——必须提示 `--session new`；
② plan advance 对 failed/run_failed 环节的会话建议与「先盘点半成品」提示。
"""

from __future__ import annotations

from argparse import Namespace


def _fake_item(result: dict):
    return lambda *a, **k: {"result": result}


# ---------------------------------------------------------------------------
# ① twin consult：冷启动拒复用给出可执行重试
# ---------------------------------------------------------------------------
def test_twin_consult_cold_start_hint(monkeypatch, tmp_path, capsys):
    from pysci.skills.orchestration.tools import dispatch, taste, verdict

    monkeypatch.setattr(verdict, "taste_enabled", lambda: True)
    # 留痕面重定向（哨兵禁止写真实 evals/）
    monkeypatch.setattr(taste, "write_record", lambda rec: tmp_path / "rec.json")
    monkeypatch.setattr(taste, "_journal_ruling", lambda rec, why: None)
    monkeypatch.setattr(taste, "_night_marker", lambda: {"night": False})
    monkeypatch.setattr(
        dispatch,
        "do_dispatch",
        lambda *a, **k: dispatch.DispatchOutcome(
            code=2,
            kind="run_failed",
            member="twin",
            error="cold_start 成员只允许 session='new'（收到 'latest'）",
        ),
    )
    case = tmp_path / "case.md"
    case.write_text("# 案卷：\n问题\n\n## 选项 A\n\n## 选项 B\n", encoding="utf-8")
    rc = verdict.cmd_consult_twin(Namespace(case_file=str(case), session=None))
    out = capsys.readouterr().out
    assert rc != 0
    assert "冷启动纪律拒绝复用会话" in out
    assert "加 `--session new` 重试" in out
    assert "twin 本身可用" in out


def test_twin_consult_disabled_still_says_not_enabled(monkeypatch, tmp_path, capsys):
    """未启用那一义保持不变（分型不能把「未启用」说成「冷启动」）。"""
    from pysci.skills.orchestration.tools import verdict

    monkeypatch.setattr(verdict, "taste_enabled", lambda: False)
    case = tmp_path / "case.md"
    case.write_text("# 案卷", encoding="utf-8")
    rc = verdict.cmd_consult_twin(Namespace(case_file=str(case), session=None))
    out = capsys.readouterr().out
    assert rc == 2
    assert "未启用" in out and "--session new" not in out


# ---------------------------------------------------------------------------
# ② plan advance：failed/run_failed 环节的会话建议 + 恢复提示
# ---------------------------------------------------------------------------
def test_run_failed_step_advises_new_session_and_recovery(monkeypatch, capsys):
    from pysci.skills.orchestration.tools import plans

    monkeypatch.setattr(
        plans,
        "_queue_item_for_step",
        _fake_item({"kind": "run_failed", "tags": [], "body": ""}),
    )
    plans._print_blocked_options(
        "p1", {"id": "s1", "session_id": "982c6a21dead"}, {"member": "figure"}
    )
    out = capsys.readouterr().out
    assert "建议 `--session new`" in out
    assert "run_failed 恢复清单" in out
    assert "盘点半成品" in out
    assert "续作自动复用" not in out


def test_content_blocked_keeps_auto_reuse(monkeypatch, capsys):
    """内容类补充（blocked，非 run_failed）→ 维持自动复用。"""
    from pysci.skills.orchestration.tools import plans

    monkeypatch.setattr(
        plans,
        "_queue_item_for_step",
        _fake_item({"kind": "blocked", "tags": [], "body": "补充信息"}),
    )
    plans._print_blocked_options(
        "p1", {"id": "s1", "session_id": "982c6a21dead"}, {"member": "figure"}
    )
    out = capsys.readouterr().out
    assert "续作自动复用" in out
    assert "建议 `--session new`" not in out
    assert "run_failed 恢复清单" not in out


# ---------------------------------------------------------------------------
# ③ 文档面
# ---------------------------------------------------------------------------
def test_docs_record_cold_start_and_recovery_checklist():
    from pysci.paths import ORCHESTRATION_ROOT

    ref = ORCHESTRATION_ROOT / "skills" / "orchestration" / "references"
    cs = (ref / "command-surface.md").read_text(encoding="utf-8")
    assert "冷启动须 --session new" in cs, "command-surface 未补 twin 冷启动口径"
    playbook = (ref / "dispatch-playbook.md").read_text(encoding="utf-8")
    assert "run_failed 恢复清单" in playbook
    assert "只核对与正式交付" in playbook
