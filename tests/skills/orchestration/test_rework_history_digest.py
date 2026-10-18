"""rework_round≥2 的历次改进展示：**摘要＋报告路径**，不再全文灌入（backlog 20261017-104503）。

旧实现把前轮报告全文塞进 advance 输出（实测 67.1 KB 被截断成 2000 字符预览——既进不了
上下文也读不全）。新口径：每轮只给关键段摘要（任务概述/批语/改进清单/裁决/结论）＋路径，
细节由组长自查该文件。
"""

from __future__ import annotations

FILLER = "全文标记FULLTEXT" * 200  # 旧实现会把这段灌进输出


def _bind(monkeypatch, tmp_path, states, reports, bodies=None):
    from pysci.skills.orchestration.tools import taskbooks

    monkeypatch.setattr(
        taskbooks, "load_state", lambda tid: dict(states.get(tid) or {})
    )
    monkeypatch.setattr(
        taskbooks,
        "report_path",
        lambda tid: reports.get(tid) or (tmp_path / f"{tid}.report.md"),
    )
    monkeypatch.setattr(taskbooks, "read_body", lambda tid: (bodies or {}).get(tid, ""))


def test_rework_history_is_digest_plus_path(monkeypatch, tmp_path):
    from pysci.skills.orchestration.tools import plans

    rep = tmp_path / "prev.report.md"
    rep.write_text(
        "# 全流程报告：前轮\n\n"
        "## 任务概述\n\n- 状态：delivered；轮次：第 1 轮\n\n"
        "## 批语\n\n1. 主瓣方向偏离 45°/135°；2. 内框应完整位于区域内\n\n"
        "## 改进清单\n\n- 补 ext 闸；- 结尾页惯例\n\n"
        "## 原始要求（大段全文）\n\n" + FILLER + "\n",
        encoding="utf-8",
    )
    _bind(
        monkeypatch,
        tmp_path,
        {
            "cur": {
                "id": "cur",
                "rework_round": 2,
                "rework_of": "prev",
                "title": "本轮",
            },
            "prev": {"id": "prev", "title": "前轮", "rework_of": ""},
        },
        {"prev": rep},
    )

    out = "\n".join(plans._rework_history_lines({"taskbook": "cur"}))
    assert "任务概述" in out and "批语" in out and "改进清单" in out
    assert "细节自查该文件" in out and "prev.report.md" in out
    assert "FULLTEXT" not in out, "前轮报告全文被灌入"
    assert len(out) < 2500, f"摘要仍过大：{len(out)}"


def test_single_round_has_no_history(monkeypatch, tmp_path):
    from pysci.skills.orchestration.tools import plans

    _bind(monkeypatch, tmp_path, {"cur": {"rework_round": 1}}, {})
    assert plans._rework_history_lines({"taskbook": "cur"}) == []


def test_missing_report_falls_back_to_path_hint(monkeypatch, tmp_path):
    """报告缺失 → 不给全文（读任务书正文只做兜底摘要），仍指明去路。"""
    from pysci.skills.orchestration.tools import plans

    _bind(
        monkeypatch,
        tmp_path,
        {
            "cur": {"id": "cur", "rework_round": 2, "rework_of": "prev"},
            "prev": {"id": "prev", "title": "前轮", "rework_of": ""},
        },
        {},
        bodies={"prev": "# 前轮任务书\n\n## 任务概述\n\n- 兜底正文摘要\n"},
    )
    out = "\n".join(plans._rework_history_lines({"taskbook": "cur"}))
    assert "兜底正文摘要" in out
    assert "无报告留痕" in out
