"""browser_fetch 正文抽取离线快照测试（阶段 4：trafilatura 替换手写 innerText）。

覆盖 trafilatura 抽取层与逐级回退，全部离线（trafilatura 对 HTML 字符串解析不触网，
只有 fetch_url 才联网；本测试只喂字符串）：
- ``_wrap_document`` 片段包装 / 完整文档透传；
- ``_trafilatura_body`` 从真实 HTML 抽出 markdown、空输入 / 未安装 → None；
- ``_extract_body`` 优先 scoped→trafilatura，回退 full-page→trafilatura，最后回退 innerText；
- 传给 page.evaluate 的选择器来自 PublisherAdapter（保留适配器语义）。
"""

from __future__ import annotations

import sys

from pysci.skills.literature_research.tools import browser_fetch as bf

# 一段贴近真实论文正文的 HTML 片段（无 <html> 根，模拟适配器命中的 outerHTML）
ARTICLE_FRAGMENT = (
    "<article>"
    "<h1>Acoustic exceptional points in coupled resonators</h1>"
    "<p>Exceptional points are spectral singularities of non-Hermitian operators where "
    "both eigenvalues and eigenvectors coalesce. In acoustics they enable enhanced "
    "sensing and unidirectional topological energy transfer between coupled cavities.</p>"
    "<p>We report the experimental observation of an exceptional point in a system of "
    "two coupled acoustic cavities with balanced gain and loss, tuning the coupling "
    "strength to reach the coalescence of the two resonance eigenvalues and eigenvectors.</p>"
    "<p>The measured reflection spectrum shows the characteristic square-root topology "
    "of the eigenvalue surfaces in the vicinity of the exceptional point, in excellent "
    "agreement with coupled-mode theory predictions for the same parameter regime.</p>"
    "</article>"
)


class _FakePage:
    """最小 Playwright page 替身：按传入脚本返回预设的 outerHTML / innerText / 整页 HTML。"""

    def __init__(self, scoped_html="", full_html="", inner_text=""):
        self.scoped_html = scoped_html
        self.full_html = full_html
        self.inner_text = inner_text
        self.eval_calls: list[tuple] = []

    def evaluate(self, script, arg=None):
        self.eval_calls.append((script, arg))
        if script == bf._JS_EXTRACT_HTML:
            return self.scoped_html
        if script == bf._JS_EXTRACT:
            return self.inner_text
        return ""

    def content(self):
        return self.full_html


# ---------------------------------------------------------------------------
# _wrap_document —— 片段包装 / 完整文档透传
# ---------------------------------------------------------------------------
def test_wrap_document_wraps_bare_fragment():
    out = bf._wrap_document("<div>hello</div>")
    assert out.startswith("<html><body>")
    assert out.endswith("</body></html>")
    assert "<div>hello</div>" in out


def test_wrap_document_passthrough_full_html():
    doc = "<html><body><p>x</p></body></html>"
    assert bf._wrap_document(doc) == doc


def test_wrap_document_passthrough_doctype():
    doc = "<!DOCTYPE html><html><body><p>x</p></body></html>"
    assert bf._wrap_document(doc) == doc


def test_wrap_document_empty():
    assert bf._wrap_document("") == ""
    assert bf._wrap_document("   ") == ""


# ---------------------------------------------------------------------------
# _trafilatura_body —— 真实离线抽取
# ---------------------------------------------------------------------------
def test_trafilatura_body_extracts_markdown_from_article():
    md = bf._trafilatura_body(
        bf._wrap_document(ARTICLE_FRAGMENT), "https://example.org/a"
    )
    assert md is not None
    assert "exceptional point" in md.lower()
    assert len(md) > 200


def test_trafilatura_body_empty_returns_none():
    assert bf._trafilatura_body("", "https://example.org") is None
    assert bf._trafilatura_body("   ", "https://example.org") is None


def test_trafilatura_body_returns_none_when_unavailable(monkeypatch):
    """trafilatura 未安装（sys.modules[name]=None → import 抛 ImportError）→ 优雅返回 None。"""
    monkeypatch.setitem(sys.modules, "trafilatura", None)
    assert (
        bf._trafilatura_body(bf._wrap_document(ARTICLE_FRAGMENT), "https://x") is None
    )


# ---------------------------------------------------------------------------
# _extract_body —— 逐级回退阶梯
# ---------------------------------------------------------------------------
def test_extract_body_prefers_trafilatura_scoped():
    page = _FakePage(
        scoped_html=ARTICLE_FRAGMENT, inner_text="PLAIN_INNERTEXT_SENTINEL"
    )
    out = bf._extract_body(page, "https://example.org/a", bf.GENERIC_ADAPTER)
    assert "exceptional point" in out.lower()
    assert "PLAIN_INNERTEXT_SENTINEL" not in out  # 未走 innerText 兜底
    # 首步用 _JS_EXTRACT_HTML 取 scoped outerHTML，且选择器来自 adapter
    assert page.eval_calls[0][0] == bf._JS_EXTRACT_HTML
    assert page.eval_calls[0][1] == list(bf.GENERIC_ADAPTER.fulltext_selectors)


def test_extract_body_falls_back_to_full_page():
    page = _FakePage(
        scoped_html="",  # 适配器未命中
        full_html=bf._wrap_document(ARTICLE_FRAGMENT),
        inner_text="SENTINEL",
    )
    out = bf._extract_body(page, "https://example.org/a", bf.GENERIC_ADAPTER)
    assert "exceptional point" in out.lower()
    assert "SENTINEL" not in out


def test_extract_body_final_fallback_innertext():
    """scoped 空 + 整页正文太短（trafilatura 抽不出 >200 字）→ 回退浏览器 innerText。"""
    page = _FakePage(
        scoped_html="",
        full_html="<html><body><p>tiny</p></body></html>",
        inner_text="FALLBACK_SENTINEL",
    )
    out = bf._extract_body(page, "https://example.org/a", bf.GENERIC_ADAPTER)
    assert out == "FALLBACK_SENTINEL"
    assert any(s == bf._JS_EXTRACT for s, _ in page.eval_calls)


def test_extract_body_innertext_when_trafilatura_unavailable(monkeypatch):
    """trafilatura 缺失时保持旧行为：直接用浏览器 innerText（能力不塌方）。"""
    monkeypatch.setitem(sys.modules, "trafilatura", None)
    page = _FakePage(scoped_html=ARTICLE_FRAGMENT, inner_text="INNERTEXT_ONLY")
    out = bf._extract_body(page, "https://x", bf.GENERIC_ADAPTER)
    assert out == "INNERTEXT_ONLY"


def test_extract_body_swallows_page_errors():
    """page.evaluate / content 抛错时不冒泡，最终回退到空串（不使抓取整体崩溃）。"""

    class _Boom(_FakePage):
        def evaluate(self, script, arg=None):
            raise RuntimeError("boom")

        def content(self):
            raise RuntimeError("boom")

    out = bf._extract_body(_Boom(), "https://x", bf.GENERIC_ADAPTER)
    assert out == ""
