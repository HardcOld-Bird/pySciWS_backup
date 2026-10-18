"""OA 回退（Semantic Scholar → arXiv / Europe PMC）与下载失败诊断（20261018-220003-lit）。

覆盖：JATS 全文 XML → Markdown；OA 回退顺序（arXiv → PMC → OA 直链）与失败说明；
`BundleResult` 的失败归类（Cloudflare/付费墙/瞬时/404/PDF 被拒）及可重试性。
"""

from __future__ import annotations

from pysci.skills.literature_research.tools import browser_fetch, oa_fallback

JATS = """<?xml version="1.0" encoding="UTF-8"?>
<article xmlns:xlink="http://www.w3.org/1999/xlink">
  <front><article-meta>
    <title-group><article-title>Acoustic metamaterials: a review</article-title></title-group>
    <abstract><p>We review acoustic metamaterials.</p></abstract>
  </article-meta></front>
  <body>
    <sec><title>Introduction</title><p>First paragraph.</p>
      <sec><title>Background</title><p>Nested paragraph.</p></sec>
    </sec>
    <fig id="f1"><caption><p>Fig. 1: a caption.</p></caption></fig>
  </body>
</article>
"""


def test_jats_to_markdown_extracts_title_abstract_sections_and_captions():
    md = oa_fallback.jats_to_markdown(JATS)
    assert md.startswith("# Acoustic metamaterials: a review")
    assert "## Abstract" in md and "We review acoustic metamaterials." in md
    assert "## Introduction" in md and "First paragraph." in md
    assert "### Background" in md and "Nested paragraph." in md
    assert "**Figure**: Fig. 1: a caption." in md


def test_jats_parse_failure_is_empty():
    assert oa_fallback.jats_to_markdown("<not-xml") == ""


def test_try_oa_prefers_arxiv_pdf(monkeypatch, tmp_path):
    monkeypatch.setattr(
        oa_fallback,
        "semantic_scholar",
        lambda doi, **_kw: {
            "arxiv_id": "1906.10862",
            "pmcid": "",
            "oa_url": "",
            "oa_status": "GREEN",
        },
    )
    called = {}

    def fake_download(url, dest, **_kw):
        called["url"] = url
        dest.write_bytes(b"%PDF-1.4")
        return dest

    monkeypatch.setattr(oa_fallback, "_download_pdf", fake_download)
    md, pdf, note = oa_fallback.try_oa("10.1103/x", pdf_dest=tmp_path / "x.pdf")
    assert pdf is not None and pdf.exists()
    assert called["url"] == "https://arxiv.org/pdf/1906.10862"
    assert "arXiv 1906.10862" in note


def test_try_oa_falls_back_to_europepmc_markdown(monkeypatch, tmp_path):
    monkeypatch.setattr(
        oa_fallback,
        "semantic_scholar",
        lambda doi, **_kw: {
            "arxiv_id": "",
            "pmcid": "PMC4771441",
            "oa_url": "",
            "oa_status": "GOLD",
        },
    )
    monkeypatch.setattr(
        oa_fallback,
        "europepmc_markdown",
        lambda pmcid, **_kw: "# Review\n\nbody\n",
    )
    md, pdf, note = oa_fallback.try_oa("10.1126/x", pdf_dest=tmp_path / "x.pdf")
    assert pdf is None and md.startswith("# Review")
    assert "PMC4771441" in note


def test_try_oa_no_metadata_is_fail_soft(monkeypatch, tmp_path):
    monkeypatch.setattr(oa_fallback, "semantic_scholar", lambda doi, **_kw: {})
    md, pdf, note = oa_fallback.try_oa("10.1/x", pdf_dest=tmp_path / "x.pdf")
    assert md == "" and pdf is None and "未给出副本" in note


# ---------------------------------------------------------------------------
# 失败诊断归类
# ---------------------------------------------------------------------------
def _bundle(**kw) -> browser_fetch.BundleResult:
    return browser_fetch.BundleResult(url="https://example.org/x", **kw)


def test_classify_cloudflare_is_retryable():
    b = _bundle(cloudflare=True, status=403)
    browser_fetch._classify_failure(b)
    assert b.failure_kind == "cloudflare" and b.retryable is True
    assert "HTTP 403" in b.error


def test_classify_paywall_requires_manual_access():
    b = _bundle(paywalled=True, status=403)
    browser_fetch._classify_failure(b)
    assert b.failure_kind == "paywall" and b.retryable is False
    assert "付费墙" in b.error


def test_classify_transient_5xx_retryable():
    b = _bundle(status=503)
    browser_fetch._classify_failure(b)
    assert b.failure_kind == "transient" and b.retryable is True


def test_classify_not_found_not_retryable():
    b = _bundle(status=404)
    browser_fetch._classify_failure(b)
    assert b.failure_kind == "not_found" and b.retryable is False


def test_classify_pdf_blocked_records_pdf_status():
    b = _bundle(status=200, pdf_status=403)
    browser_fetch._classify_failure(b)
    assert b.failure_kind == "pdf_blocked" and "HTTP 403" in b.error
    assert b.retryable is False


def test_bundle_fields_serialize_in_cache_roundtrip(monkeypatch, tmp_path):
    """缓存命中的旧清单没有新字段也必须有安全默认（不炸、不误报可重试）。"""
    res = _bundle(ok=True, slug="s", out_dir=tmp_path / "s", title="t")
    (tmp_path / "s").mkdir()
    md = tmp_path / "s" / "s.md"
    md.write_text("x", encoding="utf-8")
    res.html_path = md
    browser_fetch._write_bundle_cache(res, res.url, tmp_path)
    cached = browser_fetch._read_bundle_cache(
        res.url, tmp_path, max_age_days=None, want_html=True, want_pdf=False
    )
    assert cached is not None and cached.ok
    assert (
        cached.status == 0 and cached.failure_kind == "" and cached.retryable is False
    )
