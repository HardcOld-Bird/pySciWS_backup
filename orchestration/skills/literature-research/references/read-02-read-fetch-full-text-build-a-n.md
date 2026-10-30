# read 分册 2/6

> 含小节：`read` — fetch full text + build a note（续）· When no PDF can be had: the HTML fallback；`read` — fetch full text + build a note（续）· Flags & choices；`read` — fetch full text + build a
> 原 `read.md` 超 8KB 预算拆分（2026-10-10，backlog 20261010-refs-split-8kb）；分册索引见同目录 `read.md`，按需只读所需分册。

### When no PDF can be had: the HTML fallback

If neither route yields a PDF — or extraction produced nothing from one — but the browser did get an
article page, `read` falls back to the **web text** and records it in a *different* field:

- The content is `cache/html_fulltext/<slug>/<slug>.md`: markdown extracted by `trafilatura`, **not**
  raw HTML, carrying `browser_fetch`'s own provenance head.
- It is recorded as **`extracted_html_path`**, and `extracted_md_path` is deliberately left empty.
- It therefore stays **out of the RAG corpus** — `rag` indexes only `cache/extracted/**/*.md`. Mixing
  in web text of a different provenance and quality, where equations are usually gone (they are
  images or SVG), would quietly degrade every later retrieval. The file is *not* copied into
  `cache/extracted/` either: one copy, one location, provenance intact.
- `read` says all of this on stderr, and the file is still protected from
  `cache prune --keep-referenced` (the field is one of `cache_manager.REFERENCED_FIELDS`; the full
  protection set has a second, note-independent source — see `maintenance.md` §5a). Note that
  `prune` treats a whole `html_fulltext/<slug>/` **directory** as one unit, so keeping the article
  text also keeps the supplementary material fetched alongside it.

Treat it as a last resort for reading prose, **not** as an extraction. If you need the equations, get
the PDF and re-run with `--refresh`, or come in through the arXiv id so the LaTeX-source path applies.

**Exit code:** `0` when anything at all was produced (markdown, an HTML fallback, or a note), `1`
when nothing was, `2` when an OpenAlex id could not be resolved to a fetchable URL.

### Flags & choices

- `--headed` — run the browser with a visible window. Use when a page is Cloudflare-gated or needs
  an institutional login; the headed browser can pass challenges the headless one cannot.
- `--no-note` — only fetch + extract full text; do not create/modify a note.
- `--overwrite` — rebuild an existing note **from the template**, resetting the body. Without it an
  existing note is *merged*, never clobbered — see the merge contract below.
- `--backend pymupdf4llm` — fast local extraction when you don't need equations, or when MinerU
  quota is exhausted.
- `--refresh` — ignore the cached full text, cached OA PDF and cached HTML bundle, and re-fetch +
  re-extract from scratch. Use when the previous extraction was truncated/garbled, or the publisher
  page has been fixed. Without it, a cached `<slug>_fulltext.md` short-circuits the whole pipeline.
  (`--force` still works as a **deprecated alias** and prints a migration hint on stderr. It was
  renamed because one word, `--force`, had four unrelated meanings across four subcommands.)

### 下载失败诊断与 OA 回退（2026-10-18）

拿不到全文时 `read` 会逐项打印 **HTTP 状态 / 失败归类 / 可重试性**，例如
`[read] 失败归类：cloudflare · HTTP 403 Cloudflare 拦截 · 可重试=是`。归类与处置：

| 归类 | 含义 | 可重试 |
|---|---|---|
| `cloudflare` / `transient`（429/5xx） | 拦截或瞬时故障 | **是**：可加 `--headed`，或稍后重试 |
| `paywall`（401/403+付费墙标记） | 无机构访问/非 OA | 否：需机构网络或 OA 副本 |
| `not_found`（404/410） | DOI/链接失效 | 否 |
| `pdf_blocked` / `no_fulltext` | 主页可读但 PDF 被拒/无链接 | 否：走 OA 副本 |

随后自动经 **Semantic Scholar**（按 DOI）找 OA 副本并重试：**arXiv 预印本 PDF →
Europe PMC 全文 XML → S2 `openAccessPdf` 直链**；成功则继续正常抽取。S2 限流（429）
或无记录时只留一行说明、不阻塞退出；`--headed` 仍用于人工过 Cloudflare。

### 机构 IP 下的出版商 PDF 直链（2026-10-18）

浏览器之前 `read` 先试**出版商 PDF 直链**（普通 GET，不启动 Playwright）：

- APS：`https://journals.aps.org/<abbr>/pdf/<DOI>`（`PhysRevLett`→`prl`、`PhysRevX`→`prx`、
  `PhysRevApplied`→`prapplied` …）；
- Science：`https://www.science.org/doi/pdf/<DOI>`。

机构 IP 下这些直链可像 `curl` 一样直接拿到 PDF（实测 200）；走 DOI 落地页则可能被
Cloudflare 拦成 `Cloudflare=True 机构访问=False` 的假失败。直链失败才起浏览器/OA 回退。

**统一缓存查找**：`pysci-research cache find <DOI>` 同时扫 `cache/pdfs/`（含 theme 子目录）
与 `cache/html_fulltext/<目录>/*.pdf`（目录名按 DOI 归一匹配），批处理直接复用它，不再手抄
`html_fulltext/...pdf` 路径（20261018-221735-lit）。

### DOI vs arXiv input

- **DOI / OpenAlex id** → richest metadata (citation counts, normalized citations, JIF estimate,
  volume/issue/pages, OA link). Prefer this for published papers. One `pages` caveat is handled for
  you: electronic-only journals (all of APS, the Nature family) use an **article number**, and
  OpenAlex puts the same number in *both* `first_page` and `last_page`. `_format_pages` therefore
  collapses `first == last` to a single value — emitting `124501-124501` would be worse than leaving
  it blank, because the value travels: `add` writes it into the Zotero item's `pages`
  (`zotero_cli.create_item_from_metadata`), and document_writing's `refs_bridge.item_to_bibtex` maps
  *that* field into BibTeX `pages` — so a fake range ends up printed in your reference list.
- **arXiv id** → guaranteed open PDF, but frontmatter has **no citation count** and uses the
  preprint title. The venue is now parsed out of `journal_ref` (`journal: Phys. Rev. Lett.` rather
  than the whole citation string), and when a DOI is present the journal metrics are looked up
  separately — so a published paper reached through its arXiv id is no longer left with `jif: null`.
  Still, for a published paper, `read` the DOI if you have it.

---
