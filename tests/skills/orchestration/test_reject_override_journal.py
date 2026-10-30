"""reject 强制 override 留痕 + 决策日志（backlog 20261011-210316-leader，E0-A）。

覆盖 workflow.cmd_reject 的硬化（无 ``--override`` → 非零退出且不改动建议状态）与 journal
追加语义（有 override → 落 source=override 行、理由完整、建议移入 rejected/）。含审计优先
序测（journal 写失败即中止，建议不被动）——workflow 改动须阳性覆盖（20261010-193601 教训）。
"""

from __future__ import annotations

import argparse

import pytest

from pysci.skills.orchestration.tools import journal, workflow

from .conftest import seed_suggestion


@pytest.fixture
def journal_on_state(iso_state, monkeypatch):
    """把决策日志重定向到隔离 state 目录，返回 (state, journal_path)。"""
    jpath = iso_state / "preference-journal.jsonl"
    monkeypatch.setattr(journal, "JOURNAL_PATH", jpath)
    return iso_state, jpath


def _args(sid, *, override=None, note=""):
    return argparse.Namespace(suggestion_id=sid, override=override, note=note)


# ---------------------------------------------------------------------------
# journal 库
# ---------------------------------------------------------------------------
def test_journal_append_roundtrip(tmp_path):
    p = tmp_path / "j.jsonl"
    rec = journal.append(
        {
            "situation": "S",
            "choice": "reject",
            "source": "override",
            "kind": "override",
        },
        path=p,
    )
    assert rec["ts"]  # 自动补时间
    assert rec["options"] == ""  # schema 缺键补空串
    back = journal.read_records(p)
    assert len(back) == 1
    assert back[0]["source"] == "override" and back[0]["reason"] == ""
    # 两次 append → 两行（追加而非覆盖）
    journal.append({"choice": "approve", "source": "jury"}, path=p)
    assert journal.line_count(p) == 2


def test_journal_read_skips_blank_and_bad(tmp_path):
    p = tmp_path / "j.jsonl"
    p.write_text('{"a": 1}\n\nnot-json\n{"b": 2}\n', encoding="utf-8")
    assert [list(r) for r in journal.read_records(p)] == [["a"], ["b"]]


# ---------------------------------------------------------------------------
# reject 无 override → 阻塞
# ---------------------------------------------------------------------------
def test_reject_without_override_blocks(journal_on_state, capsys):
    state, jpath = journal_on_state
    sid = seed_suggestion(state, member="devops")

    rc = workflow.cmd_reject(_args(sid, override=""))

    assert rc == 2
    assert "--override" in capsys.readouterr().out
    # 建议仍 pending、未移入 rejected/
    assert (state / "suggestions" / f"{sid}.md").exists()
    assert not (state / "suggestions" / "rejected" / f"{sid}.md").exists()
    assert not jpath.exists()  # 未留痕


def test_reject_override_hint_precedes_lookup(journal_on_state, capsys):
    """未知 id 且无 override → 先给 --override 提示（失败快，不报「未找到」掩盖纪律）。"""
    state, _ = journal_on_state
    rc = workflow.cmd_reject(_args("does-not-exist", override=None))
    assert rc == 2
    assert "--override" in capsys.readouterr().out


def test_reject_unknown_id_with_override_blocks(journal_on_state, capsys):
    state, _ = journal_on_state
    rc = workflow.cmd_reject(_args("does-not-exist", override="有理由但建议不存在"))
    assert rc == 2
    assert "未找到" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# reject 有 override → 留痕 + 移动
# ---------------------------------------------------------------------------
def test_reject_with_override_logs_journal_and_moves(journal_on_state):
    state, jpath = journal_on_state
    sid = seed_suggestion(state, member="devops")

    rc = workflow.cmd_reject(_args(sid, override="成本过高且与 §5.1 判决流冲突"))

    assert rc == 0
    tail = journal.read_records(jpath)[-1]
    assert tail["source"] == "override"
    assert tail["kind"] == "override"
    assert tail["choice"] == "reject"
    assert tail["reason"] == "成本过高且与 §5.1 判决流冲突"  # 理由完整
    assert tail["suggestion_id"] == sid
    # 建议移入 rejected/，pending 清空
    assert not (state / "suggestions" / f"{sid}.md").exists()
    assert (state / "suggestions" / "rejected" / f"{sid}.md").exists()
    # 回复文件含 override 理由
    reply = (state / "replies" / "devops" / f"{sid}.md").read_text(encoding="utf-8")
    assert "成本过高" in reply and "override" in reply


def test_reject_audits_before_renaming(journal_on_state, monkeypatch):
    """journal 写失败即中止：建议保持 pending，绝不留「已否决却无留痕」的半状态。"""

    def boom(_rec):
        raise OSError("disk full")

    state, jpath = journal_on_state
    sid = seed_suggestion(state, member="devops")
    monkeypatch.setattr(journal, "append", boom)

    with pytest.raises(OSError):
        workflow.cmd_reject(_args(sid, override="应留痕但日志写失败"))

    assert (state / "suggestions" / f"{sid}.md").exists()  # 未移动
    assert not jpath.exists()
