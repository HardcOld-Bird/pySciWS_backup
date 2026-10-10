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
