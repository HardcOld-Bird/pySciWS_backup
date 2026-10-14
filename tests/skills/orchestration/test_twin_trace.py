"""twin 仲裁盲测追溯板块（PSEP_2 §1 配套要求/§5，task-20261014-psep2-n3）。

全链：案卷 → twin 判断（选择/理由/置信度）→ 记录落盘 ``evals/taste/<case>.json``
（含**夜航预授权**标记）→ **单一账目源** preference-journal（channel=twin + case_id）→
审计面逐条可查（mirror.twin_ruling_lines = TUI「盲审与审查」详情的同一数据层）→
用户复核（``orch twin-review --confirm/--reject`` 留痕；否决入 override 账供后续改进追溯）。
"""

from __future__ import annotations

import json
from argparse import Namespace
from types import SimpleNamespace

import pytest

from pysci.skills.orchestration.tools import (
    dispatch,
    journal,
    mirror,
    orch,
    settings,
    taste,
    verdict,
)
from pysci.skills.orchestration.tools.dispatch import DispatchOutcome

RULING_BODY = (
    '<result>{"choice": "批准方案乙", "reason": "援引 P-03 与案卷数据", '
    '"confidence": 0.62}</result>'
)


@pytest.fixture
def face(tmp_path, monkeypatch):
    """追溯面全隔离：记录目录 / journal / 夜航开关文件都落 tmp。"""
    taste_dir = tmp_path / "taste"
    taste_dir.mkdir()
    monkeypatch.setattr(taste, "TASTE_DIR", taste_dir)
    jpath = tmp_path / "preference-journal.jsonl"
    monkeypatch.setattr(journal, "JOURNAL_PATH", jpath)
    toml = tmp_path / "pysci_settings.toml"
    monkeypatch.setattr(settings, "SETTINGS_PATH", toml)

    def night(on: bool) -> None:
        toml.write_text(
            f"[night]\nenabled = {'true' if on else 'false'}\n", encoding="utf-8"
        )

    def case(name: str = "night-case.md") -> object:
        p = tmp_path / name
        p.write_text(
            "# 案卷：高风险裁决\n是否批准把 X 规则级改动落地？\n（证据略）\n",
            encoding="utf-8",
        )
        return p

    return SimpleNamespace(night=night, case=case, dir=taste_dir, journal=jpath)


def _journal_rows(path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _consult(face, monkeypatch, body: str = RULING_BODY) -> int:
    monkeypatch.setattr(
        dispatch,
        "do_dispatch",
        lambda *a, **k: DispatchOutcome(
            code=0, kind="result", body=body, sid="twin-sid"
        ),
    )
    return verdict.cmd_consult_twin(Namespace(case_file=str(face.case()), session=None))


def test_night_twin_ruling_full_chain_traceable(face, monkeypatch, capsys):
    """验收主链：夜航一次 twin 代裁 → 记录 + 单一账目 + 审计面全要素。"""
    face.night(True)
    assert _consult(face, monkeypatch) == 0
    out = capsys.readouterr().out
    assert "批准方案乙" in out and "夜航预授权" in out and "twin-review" in out

    rec = json.loads((face.dir / "night-case.json").read_text(encoding="utf-8"))
    assert rec["record_kind"] == "consult" and rec["case_id"] == "night-case"
    assert rec["choice"] == "批准方案乙" and rec["confidence"] == 0.62
    assert rec["reason"] and rec["ts"] and rec["case_head"]
    assert rec["night"] is True and rec["authority"] == "夜航预授权"

    row = _journal_rows(face.journal)[-1]
    assert row["channel"] == "twin" and row["case_id"] == "night-case"
    assert row["source"] == "twin" and row["kind"] == "ruling" and row["night"] is True

    text = "\n".join(mirror.twin_ruling_lines())
    for needle in (
        "night-case",
        "批准方案乙",
        "0.62",
        "夜航预授权",
        "待复核",
        "案卷：",
        "援引 P-03",
    ):
        assert needle in text, needle
    assert "待用户复核 1" in mirror._line_review()


def test_review_reject_leaves_trace_and_override_account(face, monkeypatch, capsys):
    face.night(True)
    _consult(face, monkeypatch)
    capsys.readouterr()
    assert (
        orch.main(
            ["twin-review", "night-case", "--reject", "--note", "结论与 P-07 冲突"]
        )
        == 0
    )
    assert "否决" in capsys.readouterr().out

    rec = json.loads((face.dir / "night-case.json").read_text(encoding="utf-8"))
    assert rec["review"]["verdict"] == "rejected"
    assert "P-07" in rec["review"]["note"] and rec["review"]["ts"]

    row = _journal_rows(face.journal)[-1]
    assert row["source"] == "user" and row["kind"] == "override"
    assert row["reason"] == "结论与 P-07 冲突" and row["channel"] == "twin"
    assert row["case_id"] == "night-case" and row["night"] is True

    text = "\n".join(mirror.twin_ruling_lines())
    assert "已复核:rejected" in text and "待复核" not in text
    assert "待用户复核 0" in mirror._line_review()


def test_review_confirm_records_verdict_account(face, monkeypatch):
    face.night(False)
    _consult(face, monkeypatch)
    assert orch.main(["twin-review", "night-case", "--confirm"]) == 0
    row = _journal_rows(face.journal)[-1]
    assert row["source"] == "user" and row["kind"] == "verdict"
    assert row["choice"] == "confirmed"


def test_review_list_and_error_paths(face, monkeypatch, capsys):
    face.night(True)
    _consult(face, monkeypatch)
    capsys.readouterr()
    assert orch.main(["twin-review", "--list"]) == 0
    out = capsys.readouterr().out
    assert "night-case" in out and "待用户复核 = 1" in out
    assert orch.main(["twin-review", "nope", "--reject"]) == 2
    assert "无此裁决留痕" in capsys.readouterr().out
    assert orch.main(["twin-review", "night-case"]) == 2
    assert "--confirm" in capsys.readouterr().out


def test_daytime_ruling_has_no_authority_marker_and_unparsed_kept(face, monkeypatch):
    face.night(False)
    rc = _consult(face, monkeypatch, body="自由意见：建议先小范围试点（无 JSON）")
    assert rc == 1, "没读出裁示 → rc 非零（留痕仍写）"
    rec = json.loads((face.dir / "night-case.json").read_text(encoding="utf-8"))
    assert rec["night"] is False and rec["authority"] == "" and rec["parsed"] is False
    row = _journal_rows(face.journal)[-1]
    assert row["night"] is False and row["choice"] == ""
