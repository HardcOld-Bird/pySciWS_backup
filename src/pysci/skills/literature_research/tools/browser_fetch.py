"""付费墙论文抓取（Playwright 驱动本机真实浏览器）：HTML 全文 / 正文 PDF / 补充材料。

背景
----
纯 httpx/requests 直接请求出版商页面会被 Cloudflare 拦截（403 "Just a moment"）。
但用 Playwright 驱动**本机已安装的真实浏览器**（Chrome/Edge），既能通过 Cloudflare，
又因脚本在本机运行而自带校园网 IP，从而激活机构订阅访问（页面显示
"Access Provided by <大学>"）。相比"会截图的浏览器子代理"，本模块是**纯脚本、无视觉、
可批量**的：一个进程循环 N 个 URL，提取 DOM 正文并存为本地 Markdown，之后直接 Read。

实测（APS Phys. Rev. Applied）：单篇约 4 秒、约 3.5 万字符、headless 无窗口、零视觉 Token。

适用与局限
----------
- 目前仅显式适配 **APS**（journals.aps.org），全文容器为 ``#fulltext-content``。
  其他出版商走 generic 兜底（article/main/body），需各自适配后再保证质量。
- 散文/正文/参考文献/图表标题提取质量高（正文经 trafilatura 抽为结构化 markdown）；
  **公式常丢失**——若出版商把公式渲染为图片/SVG 且无 MathML（APS 即如此，页面
  <math> 数为 0），正文抽取取不到。公式敏感的精读请改走 PDF + 云端公式识别，或 arXiv LaTeX 源码。

依赖
----
    uv add playwright trafilatura   # playwright 复用本机浏览器过 CF；trafilatura 从 HTML 抽正文 markdown
    # 若本机没有 Chrome/Edge，再执行： playwright install chromium

用法::

    from pysci.skills.literature_research.tools.browser_fetch import fetch_all, fetch_html, fetch_and_save

    bundle = fetch_all("https://journals.aps.org/prapplied/abstract/10.1103/...")
    # 一次拿全套：HTML 正文 + 正文 PDF + 补充材料，存到 cache/html_fulltext/<slug>/
    print(bundle.html_path, bundle.pdf_path, bundle.supp_paths)

    res = fetch_html(url)                        # 仅 HTML 全文（内存对象）
    res, path = fetch_and_save(url)              # 仅 HTML，存 cache/html_fulltext/<slug>.md

CLI::

    python -m pysci.skills.literature_research.tools.browser_fetch all <url> [--out-dir DIR] [--headed]   # 全套
    python -m pysci.skills.literature_research.tools.browser_fetch pdf <url> [--out-dir DIR]              # 仅正文 PDF
    python -m pysci.skills.literature_research.tools.browser_fetch fetch <url> [--output out.md] [--headed]
    python -m pysci.skills.literature_research.tools.browser_fetch batch <url1> <url2> ... [--out-dir DIR]
    python -m pysci.skills.literature_research.tools.browser_fetch batch --from-file urls.txt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from .cache_manager import bump_mtime
from .config import settings
from .notes import render_note


# ---------------------------------------------------------------------------
# 异常
# ---------------------------------------------------------------------------
class BrowserNotAvailable(RuntimeError):
    pass


class FetchFailed(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# 常量与出版商适配
# ---------------------------------------------------------------------------
DEFAULT_OUT_DIR: Path = settings.cache_dir / "html_fulltext"
CF_MARKERS = (
    "just a moment",
    "verifying you are human",
    "cf-challenge",
    "checking your browser",
    "attention required",
)
#: CF JS 挑战等待上限（秒）与轮询间隔（毫秒）——a2 方案定为 ~30s。
CF_WAIT_S = 30
CF_POLL_MS = 1000


@dataclass(frozen=True)
class PublisherAdapter:
    """把一个出版商的 URL 映射到其正文容器选择器。"""

    name: str
    hosts: tuple[str, ...]
    fulltext_selectors: tuple[str, ...]
    wait_selector: str | None = None


APS_ADAPTER = PublisherAdapter(
    name="aps",
    hosts=("journals.aps.org",),
    # 实测：全文位于 <div class="content" id="fulltext-content">，不含站点导航噪声
    fulltext_selectors=(
        "#fulltext-content",
        "section.fulltext div.content",
        "div.article-fulltext",
    ),
    wait_selector="#fulltext-content",
)
SCIENCE_ADAPTER = PublisherAdapter(
    name="science",
    hosts=("science.org",),
    # Science 2021+ 页面：正文在 article/.article__body；旧页回落 #fulltext。
    fulltext_selectors=(
        "article .article__body",
        "div.article__body",
        "section.article__body",
        "article",
        "#fulltext",
    ),
    wait_selector="article, .article__body, #fulltext",
)
GENERIC_ADAPTER = PublisherAdapter(
    name="generic",
    hosts=(),
    fulltext_selectors=("article", "main", "[role=main]", "#content", "body"),
    wait_selector=None,
)
ADAPTERS: tuple[PublisherAdapter, ...] = (APS_ADAPTER, SCIENCE_ADAPTER)


def detect_adapter(url: str) -> PublisherAdapter:
    host = (urlparse(url).hostname or "").lower()
    for a in ADAPTERS:
        if any(host == h or host.endswith("." + h) for h in a.hosts):
            return a
    return GENERIC_ADAPTER


# ---------------------------------------------------------------------------
# 浏览器内执行的 JS（提取正文 / 元数据 / MathML-LaTeX）
# ---------------------------------------------------------------------------
_JS_EXTRACT = r"""
(selectors) => {
  document.querySelectorAll('script,style,noscript,nav,footer,aside,form').forEach(e => e.remove());
  let root = null;
  for (const s of selectors) {
    const el = document.querySelector(s);
    if (el && (el.innerText || '').trim().length > 200) { root = el; break; }
  }
  if (!root) root = document.querySelector('article, main, [role=main]') || document.body;
  return root ? root.innerText : '';
}
"""

# 同上，但返回命中容器的 outerHTML（供 trafilatura 抽结构化 markdown 正文；
# 保留 DOM 结构，比 innerText 更利于标题/列表/表格还原）。
_JS_EXTRACT_HTML = r"""
(selectors) => {
  document.querySelectorAll('script,style,noscript,nav,footer,aside,form').forEach(e => e.remove());
  let root = null;
  for (const s of selectors) {
    const el = document.querySelector(s);
    if (el && (el.innerText || '').trim().length > 200) { root = el; break; }
  }
  if (!root) root = document.querySelector('article, main, [role=main]') || document.body;
  return root ? root.outerHTML : '';
}
"""

_JS_META = r"""
() => {
  const out = {};
  const metas = document.querySelectorAll('meta[name], meta[property]');
  for (const m of metas) {
    const k = m.getAttribute('name') || m.getAttribute('property') || '';
    if (!k.startsWith('citation_') && k !== 'dc.title' && k !== 'og:title') continue;
    const v = m.getAttribute('content');
    if (v == null) continue;
    (out[k] = out[k] || []).push(v);
  }
  out['__document_title'] = [document.title];
  return out;
}
"""

_JS_MATH = r"""
() => Array.from(document.querySelectorAll('math[alttext]'))
        .map(m => m.getAttribute('alttext')).filter(Boolean);
"""

# 正文 PDF 链接：优先文本为 "PDF" 的按钮，其次 APS 的 /pdf/<doi> 模式
_JS_PDF_LINK = r"""
() => {
  const as = Array.from(document.querySelectorAll('a[href]'));
  for (const a of as) {
    const t = (a.innerText || a.textContent || '').trim().toUpperCase();
    if (t === 'PDF' || t === 'DOWNLOAD PDF' || t === 'FULL TEXT PDF') return a.href;
  }
  for (const a of as) { if (/\/pdf\/10\.\d{4,}/.test(a.href)) return a.href; }
  return null;
}
"""

# 补充材料链接：仅站内、按 href 路径命中 /supplemental/、/media/ 等（实测 APS SI 位于
# /{journal}/supplemental/{doi}/<file>）。按路径而非文本匹配，避免把参考文献里
# "Prog. Theor. Phys. Suppl." 之类标题误判为补充材料；并排除正文 PDF 与引用导出。
_JS_SUPP_LINKS = r"""
() => {
  const out = []; const seen = new Set();
  const host = location.hostname;
  document.querySelectorAll('a[href]').forEach(a => {
    let u; try { u = new URL(a.href); } catch (e) { return; }
    if (u.hostname !== host) return;                    // 仅站内，排除 dx.doi.org 等外链
    const path = u.pathname.toLowerCase();
    const full = a.href.toLowerCase();
    const isMainPdf = /\/pdf\/10\.\d{4,}/.test(path);
    const isExport = /\/export\/|type=|download citation/.test(full);
    const isSupp = /\/supplemental\/|\/media\/|supp_|ancillary|\.(zip|mp4|avi|mov|wmv|wav|mp3)\b/i.test(path);
    if (isSupp && !isMainPdf && !isExport && !seen.has(a.href)) {
      seen.add(a.href);
      out.push({ text: (a.innerText || a.textContent || '').trim().replace(/\s+/g, ' ').slice(0, 120), href: a.href });
    }
  });
  return out;
}
"""


# ---------------------------------------------------------------------------
# 结果对象
# ---------------------------------------------------------------------------
@dataclass
class FetchResult:
    url: str
    ok: bool = False
    title: str = ""
    text: str = ""
    meta: dict[str, Any] = field(default_factory=dict)
    math_latex: list[str] = field(default_factory=list)
    cloudflare: bool = False
    institutional_access: bool = False
    seconds: float = 0.0
    adapter: str = ""
    browser: str = ""
    error: str = ""


# ---------------------------------------------------------------------------
# 浏览器启动（channel 复用本机浏览器，含回退）
# ---------------------------------------------------------------------------
def _channel_candidates(channel: str) -> list[str]:
    if channel and channel != "auto":
        return [channel]
    return ["chrome", "msedge"]


def _launch(p: Any, channel: str, headless: bool) -> tuple[Any, str]:
    tried: list[str] = []
    for ch in _channel_candidates(channel):
        try:
            return p.chromium.launch(channel=ch, headless=headless), ch
        except Exception as e:  # noqa: BLE001
            tried.append(f"{ch}: {e}")
    try:
        return p.chromium.launch(headless=headless), "chromium(bundled)"
    except Exception as e:  # noqa: BLE001
        tried.append(f"bundled-chromium: {e}")
    raise BrowserNotAvailable(
        "无法启动浏览器。已尝试:\n  "
        + "\n  ".join(tried)
        + "\n提示：确认本机已安装 Chrome/Edge，或运行 `playwright install chromium`。"
    )


# ---------------------------------------------------------------------------
# 元数据小工具
# ---------------------------------------------------------------------------
def _first(meta: dict[str, Any], key: str) -> str:
    v = meta.get(key)
    if isinstance(v, list):
        return v[0] if v else ""
    return v or ""


def _strip_tags(s: str) -> str:
    """去掉标题里可能内嵌的 MathML/HTML 标签（APS 部分标题 citation_title 带 <math>）。"""
    s = re.sub(r"<[^>]+>", "", s or "")
    return re.sub(r"\s+", " ", s).strip()


def _pick_title(meta: dict[str, Any]) -> str:
    for k in ("citation_title", "dc.title", "og:title"):
        if meta.get(k):
            return _strip_tags(_first(meta, k))
    dt = _first(meta, "__document_title")
    return _strip_tags(re.split(r"\s*\|\s*", dt)[0])


def _slug(res: FetchResult) -> str:
    base = _first(res.meta, "citation_doi") or res.title or res.url
    base = re.sub(r"https?://", "", base)
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("_").lower()
    return s[:80] or "article"


# ---------------------------------------------------------------------------
# 正文抽取（trafilatura 优先，浏览器 innerText 兜底）
# ---------------------------------------------------------------------------
def _wrap_document(html: str) -> str:
    """把 HTML 片段包成完整文档（trafilatura 要求根为 <html>，裸片段会返回 None）。"""
    h = (html or "").strip()
    if not h:
        return ""
    head = h[:512].lower()
    if "<html" in head or "<!doctype" in head:
        return h
    return f"<html><body>{h}</body></html>"


def _trafilatura_body(html: str, url: str) -> str | None:
    """用 trafilatura 从原始 HTML 提取结构化 markdown 正文。

    未安装 / 解析异常 / 结果为空时返回 None，由调用方回退。favor_precision 去噪
    （导航/页脚/链接堆），保留表格/图片/链接/格式；不取评论区。
    """
    if not html or not html.strip():
        return None
    try:
        from trafilatura import extract
    except ImportError:
        return None
    try:
        md = extract(
            html,
            url=url,
            output_format="markdown",
            include_comments=False,
            include_tables=True,
            include_images=True,
            include_links=True,
            include_formatting=True,
            favor_precision=True,
        )
    except Exception:  # noqa: BLE001
        return None
    return md.strip() if md and md.strip() else None


def _extract_body(page: Any, url: str, adapter: PublisherAdapter) -> str:
    """正文抽取：trafilatura（结构化 markdown）优先，逐级回退到浏览器 innerText。

    1. trafilatura 解析适配器命中的正文容器（outerHTML 包成完整文档）；
    2. trafilatura 解析整页 HTML；
    3. 浏览器内 innerText（旧 ``_JS_EXTRACT``）——保证 trafilatura 不可用/失配时仍出正文。
    """
    selectors = list(adapter.fulltext_selectors)
    try:
        scoped = page.evaluate(_JS_EXTRACT_HTML, selectors) or ""
    except Exception:  # noqa: BLE001
        scoped = ""
    try:
        full = page.content() or ""
    except Exception:  # noqa: BLE001
        full = ""
    for candidate in (_wrap_document(scoped), _wrap_document(full)):
        md = _trafilatura_body(candidate, url)
        if md and len(md) > 200:
            return md
    try:
        return page.evaluate(_JS_EXTRACT, selectors) or ""
    except Exception:  # noqa: BLE001
        return ""


# ---------------------------------------------------------------------------
# 核心抓取
# ---------------------------------------------------------------------------
def fetch_html(
    url: str,
    *,
    headless: bool = True,
    channel: str = "auto",
    timeout_ms: int = 60000,
    adapter: PublisherAdapter | None = None,
    retry_headed_on_cloudflare: bool = True,
) -> FetchResult:
    """抓取单个 URL 的 HTML 全文。

    Args:
        url: 文章页 URL（如 APS 的 abstract 页，其全文内嵌于 #fulltext-content）。
        headless: 无头模式（默认，可批量、无窗口）；被 Cloudflare 拦时会自动改 headed 重试。
        channel: 'auto'（依次试 chrome→msedge→bundled）| 'chrome' | 'msedge' | 'chromium'。
        timeout_ms: 导航/等待超时（毫秒）。
        adapter: 强制指定出版商适配器；默认按 URL 主机名自动识别。

    Returns:
        FetchResult（ok/title/text/meta/math_latex/cloudflare/institutional_access/...）。
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as e:  # pragma: no cover
        raise BrowserNotAvailable("Playwright 未安装。运行：uv add playwright") from e

    ad = adapter or detect_adapter(url)
    t0 = time.time()
    res = FetchResult(url=url, adapter=ad.name)

    with sync_playwright() as p:
        browser, used = _launch(p, channel, headless)
        res.browser = used
        try:
            ctx = browser.new_context(
                viewport={"width": 1400, "height": 2200}, locale="en-US"
            )
            page = ctx.new_page()
            page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)

            if ad.wait_selector:
                try:
                    page.wait_for_selector(ad.wait_selector, timeout=timeout_ms)
                except Exception:  # noqa: BLE001
                    pass
            try:
                page.wait_for_load_state("networkidle", timeout=15000)
            except Exception:  # noqa: BLE001
                pass
            page.wait_for_timeout(1200)

            html_low = page.content().lower()
            title_low = (page.title() or "").lower()
            res.cloudflare = (
                any(m in html_low for m in CF_MARKERS) or "just a moment" in title_low
            )
            res.institutional_access = (
                "access provided by" in html_low
                or "tongji" in html_low
                or 'data-auth-required="false"' in html_low
            )

            res.meta = page.evaluate(_JS_META) or {}
            res.title = _pick_title(res.meta)
            res.text = _extract_body(page, url, ad)
            try:
                res.math_latex = page.evaluate(_JS_MATH) or []
            except Exception:  # noqa: BLE001
                res.math_latex = []

            res.ok = (not res.cloudflare) and len(res.text) > 500
            if not res.ok and not res.error:
                if res.cloudflare:
                    res.error = "Cloudflare 拦截"
                else:
                    res.error = "未提取到正文（可能需登录、被拦截，或选择器不匹配）"
        finally:
            browser.close()

    res.seconds = round(time.time() - t0, 1)

    if res.cloudflare and headless and retry_headed_on_cloudflare:
        print(
            "[browser_fetch] headless 触发 Cloudflare，改用 headed 重试…",
            file=sys.stderr,
        )
        return fetch_html(
            url,
            headless=False,
            channel=channel,
            timeout_ms=timeout_ms,
            adapter=ad,
            retry_headed_on_cloudflare=False,
        )
    return res


# ---------------------------------------------------------------------------
# 序列化为 Markdown（带 YAML frontmatter）
# ---------------------------------------------------------------------------
def to_markdown(res: FetchResult) -> str:
    """把抓取结果序列化为带 YAML frontmatter 的 markdown（存于 ``cache/html_fulltext/``）。

    frontmatter 的序列化统一走 :func:`notes.dump_frontmatter`（PyYAML）。原先手写的
    ``_yaml_str`` 只处理双引号与反斜杠，遇到值里含换行、前导特殊字符（``*`` ``&`` ``#``
    ``: `` 等）或形如 ``2018-03-12`` 的日期串时会产出语义错误或非法的 YAML；
    交给 PyYAML 后引号策略由库负责，不必自己维护转义规则。
    """
    fm: dict[str, Any] = {
        "source_url": res.url,
        "title": res.title,
        "doi": _first(res.meta, "citation_doi") or None,
        "journal": _first(res.meta, "citation_journal_title") or None,
        "publication_date": _first(res.meta, "citation_publication_date") or None,
        "volume": _first(res.meta, "citation_volume") or None,
        "firstpage": _first(res.meta, "citation_firstpage") or None,
        "authors": list(res.meta.get("citation_author") or []),
        "fetched_at": datetime.now().isoformat(timespec="seconds"),
        "fetcher": (
            f"browser_fetch (playwright, adapter={res.adapter}, browser={res.browser})"
        ),
        "institutional_access": bool(res.institutional_access),
        "cloudflare_passed": not res.cloudflare,
        "char_count": len(res.text),
        "equations_note": (
            "公式可能以图片/SVG 呈现而未被提取；需公式请走 PDF + 云端识别或 arXiv LaTeX"
        ),
    }
    body = f"\n# {res.title}\n\n{res.text.strip()}\n"
    if res.math_latex:
        body += f"\n<!-- 另检测到 {len(res.math_latex)} 条 MathML alttext(LaTeX) 公式 -->\n\n"
        body += "".join(f"$$ {m} $$\n" for m in res.math_latex)
    return render_note(fm, body)


# ---------------------------------------------------------------------------
# 保存 / 批量
# ---------------------------------------------------------------------------
def fetch_and_save(
    url: str, *, out_dir: Path | str | None = None, **kw: Any
) -> tuple[FetchResult, Path | None]:
    res = fetch_html(url, **kw)
    if not res.ok:
        print(f"[browser_fetch] FAILED {url}: {res.error}", file=sys.stderr)
        return res, None
    od = Path(out_dir) if out_dir else DEFAULT_OUT_DIR
    od.mkdir(parents=True, exist_ok=True)
    path = od / f"{_slug(res)}.md"
    path.write_text(to_markdown(res), encoding="utf-8")
    print(f"[browser_fetch] OK {url} -> {path} ({len(res.text)} chars, {res.seconds}s)")
    return res, path


def fetch_many(
    urls: list[str], *, out_dir: Path | str | None = None, **kw: Any
) -> dict[str, Path | Exception]:
    """批量抓取，返回 {url: 保存路径 或 异常}。单篇失败不影响其他。"""
    out: dict[str, Path | Exception] = {}
    for u in urls:
        try:
            _, path = fetch_and_save(u, out_dir=out_dir, **kw)
            out[u] = path if path else FetchFailed("blocked/empty")
        except Exception as e:  # noqa: BLE001
            out[u] = e
    return out


# ---------------------------------------------------------------------------
# PDF / 补充材料下载（同一浏览器会话，复用校园网权限与 CF 通行）
# ---------------------------------------------------------------------------
@dataclass
class BundleResult:
    url: str
    ok: bool = False
    title: str = ""
    slug: str = ""
    out_dir: Path | None = None
    html_path: Path | None = None
    pdf_path: Path | None = None
    supp_paths: list[Path] = field(default_factory=list)
    supp_links: list[dict[str, Any]] = field(default_factory=list)
    cloudflare: bool = False
    institutional_access: bool = False
    seconds: float = 0.0
    adapter: str = ""
    browser: str = ""
    char_count: int = 0
    notes: list[str] = field(default_factory=list)
    #: 主页面 HTTP 状态（``page.goto`` 的 response；拿不到 → 0）。
    status: int = 0
    #: 正文 PDF 下载的 HTTP 状态（未尝试/未知 → 0）。
    pdf_status: int = 0
    #: 失败归类：cloudflare / paywall / transient / not_found / pdf_blocked / no_fulltext。
    failure_kind: str = ""
    #: 人读失败原因（含 HTTP 状态）。
    error: str = ""
    #: 检测到付费墙/登录墙标记。
    paywalled: bool = False
    #: 是否值得自动/手动重试（Cloudflare、429、5xx）。
    retryable: bool = False


def _looks_like_pdf(data: bytes | None) -> bool:
    return bool(data) and data[:5] == b"%PDF-"


def _download_bytes(ctx: Any, href: str, timeout_ms: int) -> tuple[bytes | None, int]:
    """用浏览器上下文的 request（共享 cookie / CF 通行）下载二进制。

    Returns:
        ``(bytes|None, http_status)``——状态码供失败诊断（403 付费墙/Cloudflare 与
        404 链接失效的处置相反）；异常/无响应 → ``(None, 0)``。
    """
    try:
        resp = ctx.request.get(href, timeout=timeout_ms)
        status = int(getattr(resp, "status", 0) or 0)
        if resp.ok:
            return resp.body(), status
        return None, status
    except Exception:  # noqa: BLE001
        pass
    return None, 0


def _download_via_click(page: Any, href: str) -> bytes | None:
    """回退：在页面内触发下载并捕获 download 事件。"""
    try:
        with page.expect_download(timeout=20000) as di:
            page.evaluate(
                "(h) => { const a=document.createElement('a'); a.href=h;"
                " a.download=''; document.body.appendChild(a); a.click(); }",
                href,
            )
        dl = di.value
        with open(dl.path(), "rb") as f:
            return f.read()
    except Exception:  # noqa: BLE001
        return None


def _derive_pdf_url(url: str) -> str | None:
    """出版商 PDF 直链推导：APS ``/abstract/``→``/pdf/``；Science ``/doi/…``→``/doi/pdf/…``。"""
    if "/abstract/" in url:
        return url.replace("/abstract/", "/pdf/")
    u = url.split("#", 1)[0].split("?", 1)[0]
    m = re.match(
        r"(https?://(?:www\.)?science\.org)/doi/(?:(?:full|abs|epdf)/)?(.+)$", u
    )
    if m and "/doi/pdf/" not in u:
        return f"{m.group(1)}/doi/pdf/{m.group(2)}"
    return None


def _filename_from_url(href: str, text: str | None = None) -> str:
    path = urlparse(href).path
    name = unquote(path.rsplit("/", 1)[-1]) if path else ""
    if name and "." in name:
        return name
    if text and "." in text and len(text) < 80:
        return text.strip()
    return ""


def _safe_filename(name: str, fallback: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", (name or "").strip()).strip("_")
    return name[:100] or fallback


# ---------------------------------------------------------------------------
# URL → bundle 清单缓存（Tier A 持久复用：同一 URL 命中即免起浏览器）
# ---------------------------------------------------------------------------
def _url_manifest_path(url: str, base_out: Path) -> Path:
    """``<base_out>/.by_url/<sha1(url)[:16]>.json``——因 slug 抓取后才确定，故按 URL 建索引。"""
    h = hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]
    return base_out / ".by_url" / f"{h}.json"


def _read_bundle_cache(
    url: str,
    base_out: Path,
    *,
    max_age_days: int | None,
    want_html: bool,
    want_pdf: bool,
) -> BundleResult | None:
    """查 URL 清单；若 bundle 仍在、新鲜且满足 want_* 要求，重建 BundleResult（不启动浏览器）。"""
    mp = _url_manifest_path(url, base_out)
    if not mp.exists():
        return None
    try:
        m = json.loads(mp.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    ts = float(m.get("ts") or 0.0)
    if (
        max_age_days is not None
        and max_age_days > 0
        and (time.time() - ts) > max_age_days * 86400
    ):
        return None
    html_p = Path(m["html_path"]) if m.get("html_path") else None
    pdf_p = Path(m["pdf_path"]) if m.get("pdf_path") else None
    if want_html and not (html_p and html_p.exists()):
        return None
    if want_pdf and not (pdf_p and pdf_p.exists()):
        return None
    supp = [Path(x) for x in (m.get("supp_paths") or []) if Path(x).exists()]
    out_dir_p = Path(m["out_dir"]) if m.get("out_dir") else None
    res = BundleResult(
        url=m.get("url", url),
        ok=bool(m.get("ok", True)),
        title=m.get("title", ""),
        slug=m.get("slug", ""),
        out_dir=out_dir_p,
        html_path=html_p,
        pdf_path=pdf_p,
        supp_paths=supp,
        supp_links=m.get("supp_links") or [],
        cloudflare=bool(m.get("cloudflare", False)),
        institutional_access=bool(m.get("institutional_access", False)),
        seconds=0.0,
        adapter=m.get("adapter", ""),
        browser="cache",
        char_count=int(m.get("char_count") or 0),
        notes=["命中本地缓存 bundle（未启动浏览器）；--force 可强制重抓"],
    )
    for p in [x for x in (html_p, pdf_p, out_dir_p) if x] + supp:
        bump_mtime(p)
    bump_mtime(mp)
    return res


def _write_bundle_cache(res: BundleResult, url: str, base_out: Path) -> None:
    """成功抓取后写入/更新 URL 清单，供下次同 URL 命中复用。失败静默。"""
    if not res.ok or not res.slug:
        return
    mp = _url_manifest_path(url, base_out)
    entry = {
        "url": url,
        "slug": res.slug,
        "out_dir": str(res.out_dir) if res.out_dir else None,
        "html_path": str(res.html_path) if res.html_path else None,
        "pdf_path": str(res.pdf_path) if res.pdf_path else None,
        "supp_paths": [str(x) for x in res.supp_paths],
        "supp_links": res.supp_links,
        "title": res.title,
        "adapter": res.adapter,
        "char_count": res.char_count,
        "ok": res.ok,
        "cloudflare": res.cloudflare,
        "institutional_access": res.institutional_access,
        "ts": time.time(),
    }
    try:
        mp.parent.mkdir(parents=True, exist_ok=True)
        mp.write_text(json.dumps(entry, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        pass


def _cf_blocked(page: Any) -> bool:
    """当前页是否仍是 Cloudflare 挑战页（标题或正文标记）。"""
    try:
        title = (page.title() or "").lower()
    except Exception:  # noqa: BLE001
        title = ""
    if "just a moment" in title:
        return True
    try:
        low = (page.content() or "")[:300000].lower()
    except Exception:  # noqa: BLE001
        return True  # 读不到内容＝不能断言已放行
    return any(m in low for m in CF_MARKERS)


def _await_cloudflare(page: Any, ctx: Any, timeout_s: float = CF_WAIT_S) -> bool:
    """等 CF JS 挑战完成（cf_clearance 出现 / 页面落定）；返回是否放行。"""
    deadline = time.time() + max(0.0, timeout_s)
    while True:
        if not _cf_blocked(page):
            try:
                page.wait_for_load_state("networkidle", timeout=8000)
            except Exception:  # noqa: BLE001
                pass
            return not _cf_blocked(page)
        has_clearance = any(c.get("name") == "cf_clearance" for c in ctx.cookies())
        if has_clearance:
            try:
                page.wait_for_timeout(CF_POLL_MS)
            except Exception:  # noqa: BLE001
                pass
            if not _cf_blocked(page):
                return True
        if time.time() >= deadline:
            return False
        page.wait_for_timeout(CF_POLL_MS)


def _fetch_bundle_once(
    url: str,
    *,
    want_html: bool = True,
    want_pdf: bool = True,
    want_supplements: bool = True,
    out_dir: Path | str | None = None,
    headless: bool = True,
    channel: str = "auto",
    timeout_ms: int = 60000,
    adapter: PublisherAdapter | None = None,
    challenge_wait_s: float = CF_WAIT_S,
) -> BundleResult:
    """单次浏览器会话抓取（不含缓存与 headed 重试，由 :func:`fetch_bundle` 编排）。"""
    base_out = Path(out_dir) if out_dir else DEFAULT_OUT_DIR

    try:
        from playwright.sync_api import sync_playwright
    except ImportError as e:  # pragma: no cover
        raise BrowserNotAvailable("Playwright 未安装。运行：uv add playwright") from e

    ad = adapter or detect_adapter(url)
    t0 = time.time()
    res = BundleResult(url=url, adapter=ad.name)

    with sync_playwright() as p:
        browser, used = _launch(p, channel, headless)
        res.browser = used
        try:
            ctx = browser.new_context(
                viewport={"width": 1400, "height": 2200},
                locale="en-US",
                accept_downloads=True,
            )
            page = ctx.new_page()
            response = page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
            res.status = int(getattr(response, "status", 0) or 0)
            # DOI 入口会 302 到出版商站点：按**最终 URL** 重认适配器（Science 等靠此命中）
            final_ad = detect_adapter(page.url or url)
            if final_ad.name != "generic":
                ad = final_ad
                res.adapter = ad.name
            # CF 挑战页没有适配器选择器：先判挑战并等待/早退，别把 60s 选择器超时白花掉
            if _cf_blocked(page):
                passed = _await_cloudflare(page, ctx, challenge_wait_s)
                res.notes.append(
                    f"Cloudflare JS 挑战：等待 {challenge_wait_s:.0f}s "
                    + ("已通过（cf_clearance/页面落定）" if passed else "未通过")
                )
                if not passed:
                    res.cloudflare = True
                    res.seconds = round(time.time() - t0, 1)
                    _classify_failure(res)
                    return res
            if ad.wait_selector:
                try:
                    page.wait_for_selector(ad.wait_selector, timeout=timeout_ms)
                except Exception:  # noqa: BLE001
                    pass
            try:
                page.wait_for_load_state("networkidle", timeout=15000)
            except Exception:  # noqa: BLE001
                pass
            page.wait_for_timeout(1000)

            html_low = page.content().lower()
            res.paywalled = any(m in html_low for m in PAYWALL_MARKERS)
            res.cloudflare = _cf_blocked(page)
            res.institutional_access = (
                "access provided by" in html_low
                or "tongji" in html_low
                or 'data-auth-required="false"' in html_low
            )
            if res.cloudflare:
                # 归类统一由 _classify_failure 产（含 Cloudflare 备注）；早退路径也要归。
                res.seconds = round(time.time() - t0, 1)
                _classify_failure(res)
                return res

            meta = page.evaluate(_JS_META) or {}
            res.title = _pick_title(meta)
            fr = FetchResult(
                url=url,
                ok=True,
                title=res.title,
                meta=meta,
                adapter=ad.name,
                browser=used,
            )
            fr.institutional_access = res.institutional_access
            res.slug = _slug(fr)
            art_dir = base_out / res.slug
            art_dir.mkdir(parents=True, exist_ok=True)
            res.out_dir = art_dir

            if want_html:
                fr.text = _extract_body(page, url, ad)
                try:
                    fr.math_latex = page.evaluate(_JS_MATH) or []
                except Exception:  # noqa: BLE001
                    fr.math_latex = []
                res.char_count = len(fr.text)
                if fr.text.strip():
                    md_path = art_dir / f"{res.slug}.md"
                    md_path.write_text(to_markdown(fr), encoding="utf-8")
                    res.html_path = md_path
                else:
                    res.notes.append("HTML 正文为空（选择器未命中或需登录）")

            if want_pdf:
                pdf_href = page.evaluate(_JS_PDF_LINK) or _derive_pdf_url(url)
                if pdf_href:
                    data, pdf_status = _download_bytes(ctx, pdf_href, timeout_ms)
                    res.pdf_status = pdf_status
                    if not _looks_like_pdf(data):
                        data = _download_via_click(page, pdf_href)
                    if _looks_like_pdf(data):
                        pdf_path = art_dir / f"{res.slug}.pdf"
                        pdf_path.write_bytes(data)
                        res.pdf_path = pdf_path
                    else:
                        res.notes.append(
                            f"PDF 下载失败或非 PDF 内容：{pdf_href}（HTTP {pdf_status or '?'}）"
                        )
                else:
                    res.notes.append("未定位到正文 PDF 链接")

            if want_supplements:
                links = page.evaluate(_JS_SUPP_LINKS) or []
                res.supp_links = links
                if not links:
                    res.notes.append("未发现补充材料（该文可能无 SI）")
                for i, lk in enumerate(links, 1):
                    href = lk.get("href")
                    if not href:
                        continue
                    data, _ = _download_bytes(
                        ctx, href, timeout_ms
                    ) or _download_via_click(page, href)
                    if data:
                        fname = (
                            _filename_from_url(href, lk.get("text"))
                            or f"{res.slug}_supp{i}.bin"
                        )
                        sp = art_dir / _safe_filename(fname, f"{res.slug}_supp{i}")
                        sp.write_bytes(data)
                        res.supp_paths.append(sp)
                    else:
                        res.notes.append(f"补充材料下载失败：{href}")

            res.ok = (res.html_path is not None) or (res.pdf_path is not None)
            if not res.ok:
                _classify_failure(res)
        finally:
            browser.close()

    res.seconds = round(time.time() - t0, 1)
    return res


def fetch_bundle(
    url: str,
    *,
    want_html: bool = True,
    want_pdf: bool = True,
    want_supplements: bool = True,
    out_dir: Path | str | None = None,
    headless: bool = True,
    channel: str = "auto",
    timeout_ms: int = 60000,
    adapter: PublisherAdapter | None = None,
    use_cache: bool = True,
    max_age_days: int | None = None,
    challenge_wait_s: float = CF_WAIT_S,
    retry_headed_on_challenge: bool = True,
) -> BundleResult:
    """一次浏览器会话拿全套；CF 挑战等待、headless 被识别自动降 headed 重试一次。

    缓存（Tier A 持久层）：``use_cache=True``（默认）先查 ``<out_dir>/.by_url/<hash>.json``；
    满足 want_* 即**不启动浏览器**直接复用；``use_cache=False``（``--force``）强制重抓。
    """
    base_out = Path(out_dir) if out_dir else DEFAULT_OUT_DIR
    kw = dict(
        want_html=want_html,
        want_pdf=want_pdf,
        want_supplements=want_supplements,
        out_dir=base_out,
        channel=channel,
        timeout_ms=timeout_ms,
        adapter=adapter,
        challenge_wait_s=challenge_wait_s,
    )

    if use_cache:
        cached = _read_bundle_cache(
            url,
            base_out,
            max_age_days=max_age_days,
            want_html=want_html,
            want_pdf=want_pdf,
        )
        if cached is not None:
            print(f"[browser_fetch] 命中缓存 bundle：{cached.slug}（未启动浏览器）")
            return cached

    res = _fetch_bundle_once(url, headless=headless, **kw)
    if res.cloudflare and headless and retry_headed_on_challenge:
        print("[browser_fetch] Cloudflare 未过：headless 被识别，降 --headed 重试一次…")
        headed = _fetch_bundle_once(url, headless=False, **kw)
        headed.notes = [
            "headless 挑战未过，已自动降 headed 重试一次",
            *headed.notes,
        ]
        headed.seconds = round(res.seconds + headed.seconds, 1)
        res = headed
    if use_cache:
        _write_bundle_cache(res, url, base_out)
    return res


#: 付费墙/登录墙的页面文案标记（仅在**未取得全文**时用于归因，不用于放行判断）。
PAYWALL_MARKERS: tuple[str, ...] = (
    "access denied",
    "purchase this article",
    "buy article",
    "buy this article",
    "get access",
    "subscribe to",
    "sign in to access",
    "sign in to view",
    "rent this article",
    "institutional subscription",
    "paywall",
)


def _classify_failure(res: BundleResult) -> None:
    """按 HTTP 状态 / Cloudflare / 付费墙标记给出**可定位**的失败原因与可重试性。

    - Cloudflare / 429 / 5xx：**可重试**（headed 或稍后）；
    - 401/403 或付费墙标记：**需人工**（机构访问/OA 副本）；404/410：链接失效；
    - PDF 链接状态非 200：单列 ``pdf_blocked``（主页可能可访问但 PDF 被拒）。
    """
    status = int(res.status or 0)
    if res.cloudflare:
        res.failure_kind = "cloudflare"
        res.error = f"HTTP {status or '?'} Cloudflare 拦截"
        res.retryable = True
        res.notes.append("Cloudflare 拦截：可加 --headed 重试，或走 arXiv/PMC OA 副本")
    elif res.paywalled or status in (401, 403):
        res.failure_kind = "paywall"
        res.error = f"HTTP {status or 403} 付费墙/无访问权限"
        res.retryable = False
        res.notes.append("非 OA 或机构访问未生效：需机构网络，或走 arXiv/PMC OA 副本")
    elif status == 429 or status >= 500:
        res.failure_kind = "transient"
        res.error = f"HTTP {status} 服务端瞬时错误/限流"
        res.retryable = True
        res.notes.append("瞬时错误：稍后重试（或换浏览器通道）")
    elif status in (404, 410):
        res.failure_kind = "not_found"
        res.error = f"HTTP {status} 页面不存在/已撤销"
        res.retryable = False
    elif res.pdf_status and res.pdf_status != 200:
        res.failure_kind = "pdf_blocked"
        res.error = f"PDF 下载 HTTP {res.pdf_status}"
        res.retryable = res.pdf_status == 429 or res.pdf_status >= 500
        res.notes.append("主页可访问但正文 PDF 被拒：需 OA 副本或机构访问")
    else:
        res.failure_kind = "no_fulltext"
        res.error = "页面可访问但未取得 PDF/正文"
        res.retryable = False
        res.notes.append("页面无正文 PDF：可试 OA 副本（arXiv/PMC）或 --headed")


def fetch_all(
    url: str, *, out_dir: Path | str | None = None, **kw: Any
) -> BundleResult:
    """一次拿全套（HTML 正文 + 正文 PDF + 补充材料）。"""
    return fetch_bundle(
        url, want_html=True, want_pdf=True, want_supplements=True, out_dir=out_dir, **kw
    )


def fetch_pdf(url: str, *, out_dir: Path | str | None = None, **kw: Any) -> Path | None:
    """仅下载正文 PDF，返回保存路径。"""
    r = fetch_bundle(
        url,
        want_html=False,
        want_pdf=True,
        want_supplements=False,
        out_dir=out_dir,
        **kw,
    )
    return r.pdf_path


def fetch_supplements(
    url: str, *, out_dir: Path | str | None = None, **kw: Any
) -> list[Path]:
    """仅下载补充材料，返回保存路径列表（无则空列表）。"""
    r = fetch_bundle(
        url,
        want_html=False,
        want_pdf=False,
        want_supplements=True,
        out_dir=out_dir,
        **kw,
    )
    return r.supp_paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(
        "[browser_fetch] 调试后门——`research read <id>` 内部就是调本模块的 fetch_all。"
        "直接跑它是为了把 Playwright / 付费墙 / Cloudflare 这一段单独隔离出来排查；"
        "其中 batch（批量抓取）与 pdf（只下正文 PDF）在 research CLI 里没有独立入口。",
        file=sys.stderr,
    )
    parser = argparse.ArgumentParser(
        description="付费墙论文 HTML 全文抓取（Playwright + 本机浏览器）"
    )
    sub = parser.add_subparsers(dest="cmd")

    pf = sub.add_parser("fetch", help="抓取单个 URL")
    pf.add_argument("url")
    pf.add_argument(
        "--output", type=Path, default=None, help="输出 .md；默认打印到 stdout"
    )
    pf.add_argument(
        "--headed", action="store_true", help="有头模式（Cloudflare 更稳，但弹窗）"
    )
    pf.add_argument("--channel", default="auto", help="auto|chrome|msedge|chromium")

    pb = sub.add_parser("batch", help="批量抓取并保存到 out-dir")
    pb.add_argument("urls", nargs="*")
    pb.add_argument(
        "--from-file", type=Path, default=None, help="每行一个 URL，# 开头忽略"
    )
    pb.add_argument("--out-dir", type=Path, default=None)
    pb.add_argument("--headed", action="store_true")
    pb.add_argument("--channel", default="auto")

    pa = sub.add_parser("all", help="一次拿全套：HTML 正文 + 正文 PDF + 补充材料")
    pa.add_argument("url")
    pa.add_argument("--out-dir", type=Path, default=None)
    pa.add_argument("--headed", action="store_true")
    pa.add_argument("--channel", default="auto")

    pp = sub.add_parser("pdf", help="仅下载正文 PDF")
    pp.add_argument("url")
    pp.add_argument("--out-dir", type=Path, default=None)
    pp.add_argument("--headed", action="store_true")
    pp.add_argument("--channel", default="auto")

    args = parser.parse_args()

    if args.cmd == "fetch":
        r = fetch_html(args.url, headless=not args.headed, channel=args.channel)
        md = to_markdown(r)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(md, encoding="utf-8")
            print(
                f"[browser_fetch] written {args.output} ({len(r.text)} chars, ok={r.ok})"
            )
        else:
            print(md)
    elif args.cmd == "batch":
        urls = list(args.urls)
        if args.from_file:
            for ln in args.from_file.read_text(encoding="utf-8").splitlines():
                ln = ln.strip()
                if ln and not ln.startswith("#"):
                    urls.append(ln)
        if not urls:
            print("no urls given")
            sys.exit(1)
        results = fetch_many(
            urls, out_dir=args.out_dir, headless=not args.headed, channel=args.channel
        )
        ok = sum(1 for v in results.values() if isinstance(v, Path))
        for u, v in results.items():
            if not isinstance(v, Path):
                print(f"  FAIL {u}: {v}", file=sys.stderr)
        print(f"[browser_fetch] batch done: {ok}/{len(results)} saved")
    elif args.cmd == "all":
        r = fetch_all(
            args.url,
            out_dir=args.out_dir,
            headless=not args.headed,
            channel=args.channel,
        )
        print(
            f"[browser_fetch] bundle ok={r.ok} ({r.seconds}s, "
            f"adapter={r.adapter}, browser={r.browser})"
        )
        print(f"  title: {r.title}")
        print(f"  dir:   {r.out_dir}")
        print(f"  html:  {r.html_path} ({r.char_count} chars)")
        print(f"  pdf:   {r.pdf_path}")
        print(f"  supp:  {len(r.supp_paths)} -> {[p.name for p in r.supp_paths]}")
        for n in r.notes:
            print(f"  note:  {n}", file=sys.stderr)
    elif args.cmd == "pdf":
        path = fetch_pdf(
            args.url,
            out_dir=args.out_dir,
            headless=not args.headed,
            channel=args.channel,
        )
        print(f"[browser_fetch] pdf -> {path}")
    else:
        parser.print_help()
