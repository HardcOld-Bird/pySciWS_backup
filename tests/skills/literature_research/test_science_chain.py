"""Science 链路：CF 等待/headed 重试、重试退避、Zotero 兜底、逐渠道证据。"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from pysci.skills.literature_research.tools import browser_fetch, research, zotero_cli


# ---------------------------------------------------------------------------
# 适配器与 PDF 直链推导
# ---------------------------------------------------------------------------
def test_science_adapter_and_pdf_url_derivation():
    assert browser_fetch.detect_adapter(
        "https://www.science.org/doi/10.1126/x"
    ).name == ("science")
    assert (
        browser_fetch._derive_pdf_url(
            "https://www.science.org/doi/10.1126/science.1210713"
        )
        == "https://www.science.org/doi/pdf/10.1126/science.1210713"
    )
    assert (
        browser_fetch._derive_pdf_url(
            "https://www.science.org/doi/full/10.1126/science.1210713"
        )
        == "https://www.science.org/doi/pdf/10.1126/science.1210713"
    )
    assert (
        browser_fetch._derive_pdf_url("https://www.science.org/doi/pdf/10.1126/x")
        is None
    ), "已是 PDF 直链不重复改"
    assert (
        browser_fetch._derive_pdf_url("https://journals.aps.org/prl/abstract/10.1103/X")
        == "https://journals.aps.org/prl/pdf/10.1103/X"
    )


# ---------------------------------------------------------------------------
# CF 挑战等待
# ---------------------------------------------------------------------------
class _Page:
    def __init__(self, unblock_after: int | None) -> None:
        self.blocked = True
        self.waits = 0
        self.unblock_after = unblock_after

    def title(self) -> str:
        return "Just a moment..." if self.blocked else "Article"

    def content(self) -> str:
        return "just a moment" if self.blocked else "article body"

    def wait_for_timeout(self, _ms: int) -> None:
        self.waits += 1
        if self.unblock_after is not None and self.waits >= self.unblock_after:
            self.blocked = False

    def wait_for_load_state(self, *_a, **_k) -> None:
        return None


class _Ctx:
    def cookies(self) -> list[dict]:
        return []


def test_await_cloudflare_returns_when_challenge_clears():
    page = _Page(unblock_after=2)
    assert browser_fetch._await_cloudflare(page, _Ctx(), timeout_s=10) is True
    assert page.waits >= 2


def test_await_cloudflare_times_out_without_clearance():
    page = _Page(unblock_after=None)
    assert browser_fetch._await_cloudflare(page, _Ctx(), timeout_s=0) is False


# ---------------------------------------------------------------------------
# headless 被识别 → 自动降 headed 重试一次
# ---------------------------------------------------------------------------
def test_bundle_retries_headed_once_on_cloudflare(monkeypatch, tmp_path):
    calls: list[bool] = []

    def fake(_url, **kw):
        calls.append(bool(kw["headless"]))
        return SimpleNamespace(
            cloudflare=bool(kw["headless"]),
            ok=not kw["headless"],
            notes=[],
            seconds=0.1,
        )

    monkeypatch.setattr(browser_fetch, "_fetch_bundle_once", fake)
    res = browser_fetch.fetch_bundle("https://x", out_dir=tmp_path, use_cache=False)
    assert calls == [True, False]
    assert res.ok and res.notes and "headed" in res.notes[0]


# ---------------------------------------------------------------------------
# read 层指数退避
# ---------------------------------------------------------------------------
def test_fetch_with_retries_uses_exponential_backoff(monkeypatch):
    results = [
        SimpleNamespace(ok=False, retryable=True, failure_kind="cloudflare"),
        SimpleNamespace(ok=False, retryable=True, failure_kind="transient"),
        SimpleNamespace(ok=True, retryable=False),
    ]
    calls: list[int] = []
    sleeps: list[float] = []

    def fake(_u, _o, _a):
        calls.append(1)
        return results.pop(0)

    monkeypatch.setattr(research, "_safe_fetch", fake)
    monkeypatch.setattr(research.time, "sleep", lambda s: sleeps.append(s))
    args = SimpleNamespace(headed=False, retries=2)
    got = research._fetch_with_retries("u", Path("."), args)
    assert got.ok and len(calls) == 3
    assert sleeps == [2.0, 4.0], "2s → 4s 指数退避"


def test_fetch_with_retries_stops_on_non_retryable(monkeypatch):
    calls: list[int] = []
    monkeypatch.setattr(
        research,
        "_safe_fetch",
        lambda *a: (
            calls.append(1)
            or SimpleNamespace(ok=False, retryable=False, failure_kind="paywall")
        ),
    )
    got = research._fetch_with_retries(
        "u", Path("."), SimpleNamespace(headed=False, retries=3)
    )
    assert not got.ok and len(calls) == 1, "付费墙不是可重试类，不得空转"


# ---------------------------------------------------------------------------
# Zotero 本地 PDF 兜底
# ---------------------------------------------------------------------------
class _FakeZotero:
    def __init__(self, children: list[dict]) -> None:
        self.children = children

    def search_items(self, _q: str, limit: int = 25) -> list[dict]:
        return [{"data": {"key": "K1", "title": "Light Propagation"}}]

    def get_children(self, _key: str) -> list[dict]:
        return self.children


def test_zotero_fallback_copies_local_pdf(monkeypatch, tmp_path):
    pdf = tmp_path / "storage" / "A1" / "paper.pdf"
    pdf.parent.mkdir(parents=True)
    pdf.write_bytes(b"%PDF-1.7 zotero")
    monkeypatch.setattr(zotero_cli, "available", lambda: True)
    monkeypatch.setattr(
        zotero_cli,
        "ZoteroCli",
        lambda: _FakeZotero(
            [
                {
                    "data": {
                        "key": "A1",
                        "itemType": "attachment",
                        "filename": "paper.pdf",
                        "contentType": "application/pdf",
                    }
                }
            ]
        ),
    )
    monkeypatch.setattr(
        research, "settings", replace(research.settings, zotero_data_dir=tmp_path)
    )
    dest = tmp_path / "out" / "p.pdf"
    got, note = research._zotero_find_pdf(
        "10.1126/science.1210713", "Light Propagation", dest
    )
    assert got == dest and dest.read_bytes().startswith(b"%PDF")
    assert "A1" in note and "PDF" in note


def test_zotero_fallback_reports_no_attachment(monkeypatch, tmp_path):
    monkeypatch.setattr(zotero_cli, "available", lambda: True)
    monkeypatch.setattr(zotero_cli, "ZoteroCli", lambda: _FakeZotero([]))
    monkeypatch.setattr(
        research, "settings", replace(research.settings, zotero_data_dir=tmp_path)
    )
    got, note = research._zotero_find_pdf("10.x", "T", tmp_path / "p.pdf")
    assert got is None and "无本地 PDF 附件" in note


def test_zotero_fallback_absent_tool_is_soft(monkeypatch):
    monkeypatch.setattr(zotero_cli, "available", lambda: False)
    got, note = research._zotero_find_pdf("10.x", "T", Path("x.pdf"))
    assert got is None and "未安装" in note


def test_zotero_falls_back_to_title_when_doi_search_is_empty(monkeypatch, tmp_path):
    class _TitleOnly:
        def search_items(self, q: str, limit: int = 25) -> list[dict]:
            if "10." in q:
                return []
            return [{"data": {"key": "K9", "title": "Light Propagation"}}]

        def get_children(self, key: str) -> list[dict]:
            return []

    monkeypatch.setattr(zotero_cli, "available", lambda: True)
    monkeypatch.setattr(zotero_cli, "ZoteroCli", _TitleOnly)
    monkeypatch.setattr(
        research, "settings", replace(research.settings, zotero_data_dir=tmp_path)
    )
    got, note = research._zotero_find_pdf(
        "10.1126/x", "Light Propagation", tmp_path / "p.pdf"
    )
    assert got is None
    assert "K9" in note and "无本地 PDF 附件" in note, note
