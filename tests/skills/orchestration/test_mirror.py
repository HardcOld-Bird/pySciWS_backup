# ---------------------------------------------------------------------------
# 宪章态判据与待决清单（backlog 20261013-192000-leader）
# ---------------------------------------------------------------------------
def test_charter_status_marker_beats_narrative_draft_word(tmp_path, monkeypatch):
    """**阳性用例**：正文含「草案」叙述词，但状态标记是 v2 生效 → 仍判已生效。

    实测缺陷：v2 顶注写「审阅 v2 增量草案并同意转正」，裸搜「草案」把它误判成未生效。
    """
    from pysci.skills.orchestration.tools import mirror

    f = tmp_path / "PREFERENCE.md"
    f.write_text(
        "# PREFERENCE.md — twin 判断宪章\n\n"
        "> **状态：v2 生效**（用户 2026-10-13 审阅 v2 增量草案并同意转正）\n\n"
        "正文里还会提到草案二字。\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(mirror, "charter_path", lambda reg=None, **kw: f)
    st = mirror.charter_state()
    assert st["status"] == "v2 生效" and st["draft"] is False
    assert "已生效 v2 生效" in mirror._line_charter()


def test_charter_status_draft_marker_still_reads_draft(tmp_path, monkeypatch):
    """显式草案标记 → 仍是草案态（判据换成标记，不是「一律放行」）。"""
    from pysci.skills.orchestration.tools import mirror

    f = tmp_path / "PREFERENCE.md"
    f.write_text("> **状态：v1 草案**\n", encoding="utf-8")
    monkeypatch.setattr(mirror, "charter_path", lambda reg=None, **kw: f)
    st = mirror.charter_state()
    assert st["draft"] is True and "待确认" in mirror._line_charter()


def test_charter_status_without_marker_falls_back_to_substring(tmp_path, monkeypatch):
    """无标记的旧文件 → 保底回落原子串判据（不许把「没标记」读成「未生效」）。"""
    from pysci.skills.orchestration.tools import mirror

    f = tmp_path / "PREFERENCE.md"
    f.write_text("# 宪章（草案）\n无状态行\n", encoding="utf-8")
    monkeypatch.setattr(mirror, "charter_path", lambda reg=None, **kw: f)
    st = mirror.charter_state()
    assert st["status"] == "" and st["draft"] is True
    assert "待用户确认生效" in mirror._line_charter()


def test_pending_items_are_data_driven(tmp_path):
    """待决清单可增删；无真源/为空 → 「（无待决）」，不再回放历史硬编码项。"""
    from pysci.skills.orchestration.tools import mirror

    p = tmp_path / "mirror-pending.json"
    assert mirror.pending_items(p) == []
    assert "（无待决）" in mirror._line_pending(p)

    p.write_text('{"items": ["甲待决", "乙待决"]}', encoding="utf-8")
    assert mirror.pending_items(p) == ["甲待决", "乙待决"]
    assert mirror._line_pending(p) == "待决事项 = 甲待决；乙待决"

    p.write_text('["裸数组也收"]', encoding="utf-8")
    assert mirror.pending_items(p) == ["裸数组也收"]

    p.write_text("{not json", encoding="utf-8")
    assert mirror.pending_items(p) == [] and "（无待决）" in mirror._line_pending(p)
