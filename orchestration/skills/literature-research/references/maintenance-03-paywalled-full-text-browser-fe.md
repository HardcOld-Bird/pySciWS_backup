# maintenance 分册 3/11

> 含小节：4. Paywalled full text (`browser_fetch.py`) — the most fragile part
> 原 `maintenance.md` 超 8KB 预算拆分（2026-10-10，backlog 20261010-refs-split-8kb）；分册索引见同目录 `maintenance.md`，按需只读所需分册。

## 4. Paywalled full text (`browser_fetch.py`) — the most fragile part

`browser_fetch` drives a real browser via Playwright to get HTML full text, the article PDF, and
supplementary files. It auto-selects a browser channel (chrome → msedge → bundled chromium) and,
on a Cloudflare challenge, retries in headed mode. Results come back as a `BundleResult`
(`ok, title, slug, html_path, pdf_path, supp_paths, cloudflare, institutional_access, seconds,
adapter, browser, char_count, notes`).

### Publisher adapters (why it depends on page structure)

Each publisher's full text sits in a different DOM container. An adapter maps a host to CSS
selectors:

```python
@dataclass
class PublisherAdapter:
    name: str
    hosts: tuple[str, ...]              # matched against the URL hostname (subdomains included)
    fulltext_selectors: tuple[str, ...] # tried in order; first hit wins
    wait_selector: str | None = None    # element to wait for before extracting

APS_ADAPTER = PublisherAdapter(
    name="aps", hosts=("journals.aps.org",),
    fulltext_selectors=("#fulltext-content", "section.fulltext div.content", "div.article-fulltext"),
    wait_selector="#fulltext-content",
)
GENERIC_ADAPTER = PublisherAdapter(      # fallback for any unmatched host
    name="generic", hosts=(),
    fulltext_selectors=("article", "main", "[role=main]", "#content", "body"),
)
ADAPTERS: tuple[PublisherAdapter, ...] = (APS_ADAPTER,)   # ← the registry

def detect_adapter(url):  # hostname match, else GENERIC_ADAPTER
```

Body extraction runs through **`trafilatura`**: an in-page JS routine (`_JS_EXTRACT_HTML`) pulls the
`outerHTML` of the first matching selector, `_wrap_document()` re-roots it as a full `<html>` doc,
and `_trafilatura_body()` converts it to structured Markdown (`favor_precision`; tables/images/links
kept). `_extract_body()` falls back to trafilatura-on-full-page, then to the old innerText routine
(`_JS_EXTRACT`) if trafilatura is unavailable or yields <200 chars — so the adapter's
`fulltext_selectors` still drive scoping. Same-host PDF / supplement links are collected by separate
JS routines (`_JS_PDF_LINK` / `_JS_SUPP_LINKS`).

### Add support for a new publisher

1. Open one of its article pages in a browser; DevTools → find the element that wraps the article
   body (ideally excluding nav/ads). Note a stable CSS selector (id or class).
2. In `browser_fetch.py`, add an adapter and register it:
   ```python
   NATURE_ADAPTER = PublisherAdapter(
       name="nature",
       hosts=("www.nature.com", "nature.com"),
       fulltext_selectors=("article", "div.c-article-body", "main"),
       wait_selector="article",
   )
   ADAPTERS = (APS_ADAPTER, NATURE_ADAPTER)
   ```
3. Test: `research read <a-doi-from-that-publisher> --headed` and check the printed `char_count` /
   the extracted full-text file looks like the article, not the navigation.

### Fix a publisher that stopped working

Symptom: `read` returns a page but the full text is empty/truncated, or it's all nav/boilerplate —
the publisher changed their markup. Re-inspect the DOM (step 1 above) and update that adapter's
`fulltext_selectors` (put the most specific selector first, keep generic ones as fallbacks). If a
login/cookie wall appeared, `institutional_access` in the result will be false and you'll need
`--headed` to sign in interactively.

---
