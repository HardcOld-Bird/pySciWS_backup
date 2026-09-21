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
- 散文/正文/参考文献/图表标题提取质量高；**公式常丢失**——若出版商把公式渲染为
  图片/SVG 且无 MathML（APS 即如此，页面 <math> 数为 0），inner_text 取不到。
  公式敏感的精读请改走 PDF + 云端公式识别，或 arXiv LaTeX 源码。

依赖
----
    uv add playwright      # 仅 Python 包；用 channel 复用本机浏览器，通常无需下载 Chromium
    # 若本机没有 Chrome/Edge，再执行： playwright install chromium

用法::

    from skills_src.literature_research.tools.browser_fetch import fetch_all, fetch_html, fetch_and_save

    bundle = fetch_all("https://journals.aps.org/prapplied/abstract/10.1103/...")
    # 一次拿全套：HTML 正文 + 正文 PDF + 补充材料，存到 cache/html_fulltext/<slug>/
    print(bundle.html_path, bundle.pdf_path, bundle.supp_paths)

    res = fetch_html(url)                        # 仅 HTML 全文（内存对象）
    res, path = fetch_and_save(url)              # 仅 HTML，存 cache/html_fulltext/<slug>.md

CLI::

    python -m skills_src.literature_research.tools.browser_fetch all <url> [--out-dir DIR] [--headed]   # 全套
    python -m skills_src.literature_research.tools.browser_fetch pdf <url> [--out-dir DIR]              # 仅正文 PDF
    python -m skills_src.literature_research.tools.browser_fetch fetch <url> [--output out.md] [--headed]
    python -m skills_src.literature_research.tools.browser_fetch batch <url1> <url2> ... [--out-dir DIR]
    python -m skills_src.literature_research.tools.browser_fetch batch --from-file urls.txt
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

from .config import settings
from .cache_manager import bump_mtime

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
    fulltext_selectors=("#fulltext-content", "section.fulltext div.content", "div.article-fulltext"),
    wait_selector="#fulltext-content",
)
GENERIC_ADAPTER = PublisherAdapter(
    name="generic",
    hosts=(),
    fulltext_selectors=("article", "main", "[role=main]", "#content", "body"),
    wait_selector=None,
)
ADAPTERS: tuple[PublisherAdapter, ...] = (APS_ADAPTER,)


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
        "无法启动浏览器。已尝试:\n  " + "\n  ".join(tried)
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


def _yaml_str(s: str) -> str:
    s = (s or "").replace("\\", "\\\\").replace('"', '\\"')
    return f'"{s}"'


def _slug(res: FetchResult) -> str:
    base = _first(res.meta, "citation_doi") or res.title or res.url
    base = re.sub(r"https?://", "", base)
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("_").lower()
    return (s[:80] or "article")


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
            ctx = browser.new_context(viewport={"width": 1400, "height": 2200}, locale="en-US")
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
            res.cloudflare = any(m in html_low for m in CF_MARKERS) or "just a moment" in title_low
            res.institutional_access = (
                "access provided by" in html_low
                or "tongji" in html_low
                or 'data-auth-required="false"' in html_low
            )

            res.meta = page.evaluate(_JS_META) or {}
            res.title = _pick_title(res.meta)
            res.text = page.evaluate(_JS_EXTRACT, list(ad.fulltext_selectors)) or ""
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
        print("[browser_fetch] headless 触发 Cloudflare，改用 headed 重试…", file=sys.stderr)
        return fetch_html(
            url, headless=False, channel=channel, timeout_ms=timeout_ms,
            adapter=ad, retry_headed_on_cloudflare=False,
        )
    return res


# ---------------------------------------------------------------------------
# 序列化为 Markdown（带 YAML frontmatter）
# ---------------------------------------------------------------------------
def to_markdown(res: FetchResult) -> str:
    lines = ["---", f"source_url: {res.url}", f"title: {_yaml_str(res.title)}"]
    for key, label in (
        ("citation_doi", "doi"),
        ("citation_journal_title", "journal"),
        ("citation_publication_date", "publication_date"),
        ("citation_volume", "volume"),
        ("citation_firstpage", "firstpage"),
    ):
        v = _first(res.meta, key)
        lines.append(f"{label}: {_yaml_str(v) if v else 'null'}")
    authors = res.meta.get("citation_author") or []
    lines.append("authors: [" + ", ".join(_yaml_str(a) for a in authors) + "]")
    lines.append(f"fetched_at: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"fetcher: browser_fetch (playwright, adapter={res.adapter}, browser={res.browser})")
    lines.append(f"institutional_access: {str(bool(res.institutional_access)).lower()}")
    lines.append(f"cloudflare_passed: {str(not res.cloudflare).lower()}")
    lines.append(f"char_count: {len(res.text)}")
    lines.append(
        "equations_note: "
        + _yaml_str("公式可能以图片/SVG 呈现而未被提取；需公式请走 PDF + 云端识别或 arXiv LaTeX")
    )
    lines += ["---", "", f"# {res.title}", "", res.text.strip()]
    if res.math_latex:
        lines += ["", f"<!-- 另检测到 {len(res.math_latex)} 条 MathML alttext(LaTeX) 公式 -->", ""]
        lines += [f"$$ {m} $$" for m in res.math_latex]
    return "\n".join(lines) + "\n"


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


def _looks_like_pdf(data: bytes | None) -> bool:
    return bool(data) and data[:5] == b"%PDF-"


def _download_bytes(ctx: Any, href: str, timeout_ms: int) -> bytes | None:
    """用浏览器上下文的 request（共享 cookie / CF 通行）下载二进制。"""
    try:
        resp = ctx.request.get(href, timeout=timeout_ms)
        if resp.ok:
            return resp.body()
    except Exception:  # noqa: BLE001
        pass
    return None


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
    """APS：把 /abstract/ 换成 /pdf/。"""
    return url.replace("/abstract/", "/pdf/") if "/abstract/" in url else None


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
    if max_age_days is not None and max_age_days > 0 and (time.time() - ts) > max_age_days * 86400:
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
) -> BundleResult:
    """一次浏览器会话拿全套：HTML 正文 + 正文 PDF + 补充材料（若有）。

    全部产物存到 ``<out_dir>/<slug>/``：``<slug>.md``（HTML 全文，带 frontmatter）、
    ``<slug>.pdf``（出版商正文 PDF）、补充材料按原文件名保存。

    缓存（Tier A 持久层）：``use_cache=True``（默认）时先查 ``<out_dir>/.by_url/<url-hash>.json``
    清单，若同一 URL 的 bundle 仍在且满足 want_* 要求，则**不启动浏览器**直接复用；
    ``max_age_days=None`` 表示永不过期。``use_cache=False``（对应 research read 的 ``--force``）强制重抓。
    """
    base_out = Path(out_dir) if out_dir else DEFAULT_OUT_DIR

    if use_cache:
        cached = _read_bundle_cache(
            url, base_out, max_age_days=max_age_days, want_html=want_html, want_pdf=want_pdf
        )
        if cached is not None:
            print(f"[browser_fetch] 命中缓存 bundle：{cached.slug}（未启动浏览器）")
            return cached

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
            page.wait_for_timeout(1000)

            html_low = page.content().lower()
            res.cloudflare = any(m in html_low for m in CF_MARKERS) or "just a moment" in (
                page.title() or ""
            ).lower()
            res.institutional_access = (
                "access provided by" in html_low
                or "tongji" in html_low
                or 'data-auth-required="false"' in html_low
            )
            if res.cloudflare:
                res.notes.append("Cloudflare 拦截，未获取；可加 --headed 重试")
                return res

            meta = page.evaluate(_JS_META) or {}
            res.title = _pick_title(meta)
            fr = FetchResult(
                url=url, ok=True, title=res.title, meta=meta, adapter=ad.name, browser=used
            )
            fr.institutional_access = res.institutional_access
            res.slug = _slug(fr)
            art_dir = base_out / res.slug
            art_dir.mkdir(parents=True, exist_ok=True)
            res.out_dir = art_dir

            if want_html:
                fr.text = page.evaluate(_JS_EXTRACT, list(ad.fulltext_selectors)) or ""
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
                    data = _download_bytes(ctx, pdf_href, timeout_ms)
                    if not _looks_like_pdf(data):
                        data = _download_via_click(page, pdf_href)
                    if _looks_like_pdf(data):
                        pdf_path = art_dir / f"{res.slug}.pdf"
                        pdf_path.write_bytes(data)
                        res.pdf_path = pdf_path
                    else:
                        res.notes.append(f"PDF 下载失败或非 PDF 内容：{pdf_href}")
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
                    data = _download_bytes(ctx, href, timeout_ms) or _download_via_click(page, href)
                    if data:
                        fname = _filename_from_url(href, lk.get("text")) or f"{res.slug}_supp{i}.bin"
                        sp = art_dir / _safe_filename(fname, f"{res.slug}_supp{i}")
                        sp.write_bytes(data)
                        res.supp_paths.append(sp)
                    else:
                        res.notes.append(f"补充材料下载失败：{href}")

            res.ok = (res.html_path is not None) or (res.pdf_path is not None)
        finally:
            browser.close()

    res.seconds = round(time.time() - t0, 1)
    if use_cache:
        _write_bundle_cache(res, url, base_out)
    return res


def fetch_all(url: str, *, out_dir: Path | str | None = None, **kw: Any) -> BundleResult:
    """一次拿全套（HTML 正文 + 正文 PDF + 补充材料）。"""
    return fetch_bundle(
        url, want_html=True, want_pdf=True, want_supplements=True, out_dir=out_dir, **kw
    )


def fetch_pdf(url: str, *, out_dir: Path | str | None = None, **kw: Any) -> Path | None:
    """仅下载正文 PDF，返回保存路径。"""
    r = fetch_bundle(
        url, want_html=False, want_pdf=True, want_supplements=False, out_dir=out_dir, **kw
    )
    return r.pdf_path


def fetch_supplements(
    url: str, *, out_dir: Path | str | None = None, **kw: Any
) -> list[Path]:
    """仅下载补充材料，返回保存路径列表（无则空列表）。"""
    r = fetch_bundle(
        url, want_html=False, want_pdf=False, want_supplements=True, out_dir=out_dir, **kw
    )
    return r.supp_paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="付费墙论文 HTML 全文抓取（Playwright + 本机浏览器）"
    )
    sub = parser.add_subparsers(dest="cmd")

    pf = sub.add_parser("fetch", help="抓取单个 URL")
    pf.add_argument("url")
    pf.add_argument("--output", type=Path, default=None, help="输出 .md；默认打印到 stdout")
    pf.add_argument("--headed", action="store_true", help="有头模式（Cloudflare 更稳，但弹窗）")
    pf.add_argument("--channel", default="auto", help="auto|chrome|msedge|chromium")

    pb = sub.add_parser("batch", help="批量抓取并保存到 out-dir")
    pb.add_argument("urls", nargs="*")
    pb.add_argument("--from-file", type=Path, default=None, help="每行一个 URL，# 开头忽略")
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
            print(f"[browser_fetch] written {args.output} ({len(r.text)} chars, ok={r.ok})")
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
            args.url, out_dir=args.out_dir, headless=not args.headed, channel=args.channel
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
            args.url, out_dir=args.out_dir, headless=not args.headed, channel=args.channel
        )
        print(f"[browser_fetch] pdf -> {path}")
    else:
        parser.print_help()
