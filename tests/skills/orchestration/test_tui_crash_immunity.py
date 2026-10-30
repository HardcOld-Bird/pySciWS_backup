"""TUI 崩溃根治的三层（backlog 20261017-120500）：运行日志 / 重建指纹跳过 / 异常免疫。"""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("textual")

from pysci.skills.orchestration.tools.tui import app as appmod  # noqa: E402
from pysci.skills.orchestration.tools.tui import log as tlog  # noqa: E402


@pytest.fixture
def logdir(tmp_path, monkeypatch):
    monkeypatch.setattr(tlog, "LOG_DIR", tmp_path / "logs")
    return tmp_path / "logs"


# ---------------------------------------------------------------------------
# ① 运行日志：时间戳行 + traceback + 绝不抛
# ---------------------------------------------------------------------------
def test_log_writes_timestamped_line(logdir):
    tlog.log("TUI 启动")
    text = tlog.log_path().read_text(encoding="utf-8")
    assert "TUI 启动" in text
    assert text[:4].isdigit(), "行首须为时间戳"


def test_log_exception_records_traceback(logdir):
    try:
        1 / 0
    except ZeroDivisionError as e:
        tlog.log_exception(e, where="event")
    text = tlog.log_path().read_text(encoding="utf-8")
    assert "[event]" in text and "ZeroDivisionError" in text and "Traceback" in text


def test_log_never_raises(tmp_path, monkeypatch):
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("x", encoding="utf-8")
    monkeypatch.setattr(tlog, "LOG_DIR", blocker)  # mkdir 必失败
    tlog.log("x")  # 不抛
    tlog.log_exception(RuntimeError("y"))


# ---------------------------------------------------------------------------
# ② 重建指纹：内容未变 → 同一指纹（跳过整栈重建）
# ---------------------------------------------------------------------------
def test_detail_fingerprint_equal_when_unchanged():
    blocks = [
        {"kind": "section", "text": "标题"},
        {"kind": "card", "id": "a", "lines": ["1", "2"]},
    ]
    a = appmod._detail_fingerprint("system", blocks)
    b = appmod._detail_fingerprint("system", [dict(x) for x in blocks])
    assert a == b


def test_detail_fingerprint_differs_on_change():
    base = [{"kind": "card", "id": "a", "lines": ["1"]}]
    changed = [{"kind": "card", "id": "a", "lines": ["2"]}]
    assert appmod._detail_fingerprint("m", base) != appmod._detail_fingerprint(
        "m", changed
    )
    assert appmod._detail_fingerprint("m1", base) != appmod._detail_fingerprint(
        "m2", base
    )


def test_render_detail_cards_has_skip_branch_with_log():
    src = appmod.PysciTUI._render_detail_cards.__doc__ or ""
    import inspect

    body = inspect.getsource(appmod.PysciTUI._render_detail_cards)
    assert "_detail_fp" in body and "[skip]" in body and "return" in body, src


# ---------------------------------------------------------------------------
# ③ 异常免疫：记录后不退出
# ---------------------------------------------------------------------------
def test_handle_exception_logs_and_returns(monkeypatch):
    seen: list = []
    monkeypatch.setattr(tlog, "log_exception", lambda e, **kw: seen.append((e, kw)))
    app = appmod.PysciTUI()
    asyncio.run(app._handle_exception(RuntimeError("race")))  # 不抛
    assert seen and isinstance(seen[0][0], RuntimeError)
