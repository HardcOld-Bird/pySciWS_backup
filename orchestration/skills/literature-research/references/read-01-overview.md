# read 分册 1/6

> 含小节：概览；`read` — fetch full text + build a note
> 原 `read.md` 超 8KB 预算拆分（2026-10-10，backlog 20261010-refs-split-8kb）；分册索引见同目录 `read.md`，按需只读所需分册。

# read / add / citecheck / index / review — detailed reference + evaluation rubric

All commands are invoked as `research <cmd> …` (see SKILL.md for the full invocation and the
PowerShell single-quote rule).

---

## `read` — fetch full text + build a note

```
research read <doi｜url｜arxiv-id｜openalex-id>
              [--backend mineru-cloud｜pymupdf4llm｜auto]
              [--headed] [--no-note] [--overwrite] [--refresh]
```

What it does, in order:

1. **Classify the input** — a DOI (`10.xxxx/…`), a URL, an arXiv id (`2301.12345`), or an
   OpenAlex id (`W…`). Anything matching none of those patterns is treated as a DOI, so a malformed
   id surfaces as "not found" rather than as a crash.
2. **Fetch + enrich metadata** (unless a bare URL). OpenAlex / arXiv supply the record; the WoS
   Starter API contributes its **accession number** (`wos_id`) when the DOI is indexed there. The
   impact figure is OpenAlex's `2yr_mean_citedness` **estimate**, *not* an official WoS JIF, and venue
   quality is then filled in from `journal_tier` / `scimago_quartile` where derivable — see the
   [metrics glossary](search.md#metrics-glossary).
3. **Fetch the full text, cheapest route first:**
   - *arXiv id* → direct PDF download (most reliable); if that fails, the browser over
     `https://arxiv.org/abs/<id>`.
   - *anything else* → when OpenAlex marked the work open access, **try its `oa_url` with a plain
     HTTP GET first**. No browser is launched. The response must begin with the `%PDF-` magic bytes
     (`Content-Type` is not trusted: repositories often answer `application/octet-stream`, and
     publisher "you need to log in" pages often answer `200 + text/html`), so a landing page is
     **rejected rather than saved**. The download goes to a `.part` file and is renamed atomically,
     because an interrupted write would otherwise leave a truncated `.pdf` that the next run would
     happily reuse as a cache hit. The filename derives from the DOI / OpenAlex id, not the URL, so
     the same paper reached through a different mirror reuses the same file.
   - *only if that fails* → a headless browser grabs the publisher page, the article PDF, and any
     supplementary material.
4. **Extract to Markdown** — MinerU cloud by default (equations → LaTeX, tables → HTML);
   `pymupdf4llm` is the local fallback (equations lost). Override with `--backend`. When the paper has
   an arXiv id — whether you passed one or the metadata carries `arxiv_id` — extraction goes through
   the **arXiv LaTeX source** instead of the PDF, so equations come out as real LaTeX rather than as
   fragments recognized off the rendered page.
   - That path (`pdf_extract._tex_to_markdown`) **strips `%` comment lines first**, then cuts
     everything outside `\begin{document}…\end{document}`, then drops the REVTeX front-matter macros
     that carry no reading value (`\affiliation`/`\email`/`\maketitle`/`\preprint`/…) while promoting
     `\title` to an `# H1` and `\author` to one `**Authors:**` line. Comments were previously kept
     verbatim and the front matter reached the RAG corpus as prose. Verified end-to-end on
     `1803.04110`: 717 lines out, **0** comment lines, **0**
     `\documentclass`/`\usepackage`/`\begin{document}` leaks, all eight body sections intact. See
     `maintenance.md` §3 for the exact regexes and why the comment strip must run *before* the
     preamble cut.
   - **The bibliography is dropped at the source** (`_TEX_DROPPED_ENVS = ("comment",
     "thebibliography")`, 2026-10-04). It used to survive as *raw REVTeX*
     (`\bibitem`/`\citenamefont`/`\bibinfo`/`\BibitemShut`): for that same `1803.04110` paper, lines
     136–709 of 717 — **~80 % of that file's lines** (35,633 chars, 58.8 % of it), pure macro residue
     with zero retrieval value. The drop only fires when the `\begin`/`\end` counts **match**, so an
     unbalanced environment is left intact rather than half-eaten. Bibliographic data has better
     sources anyway (`citecheck` / Zotero / OpenAlex). Artifacts extracted *before* the fix still
     carry it — `rag` scrubs its embedded **copies** at index time (`corpus_clean`, see
     `maintenance.md` §5d), so neither route reaches the embedder.
   - **Known limits — read the output critically.** In-text citations stay as BibTeX *keys*
     (`systems[EP2]`) because key → number mapping is not implemented, and `\ref` becomes `§label`.
     Accents are not converted (`Aubry-Andr{\'e}-Harper`). A `\newcommand` written *inside* the body
     also survives (only the preamble cut removes those, and that cut often fails on multi-file arXiv
     projects). None of this touches the body prose or the equations.
5. **Write outputs** — the extracted full text to `cache/extracted/<slug>_fulltext.md` (the single
   canonical extracted artifact — `read` extracts with `write_cache=False` so no duplicate is
   left behind), and a structured note skeleton to `papers/{year}_{author}_{slug}.md`. The full
   text's path is recorded in the note's `extracted_md_path` field and the PDF's in `local_pdf_path`.

> **Cache reuse (default):** if `cache/extracted/<slug>_fulltext.md` already exists, `read` loads it
> and **skips the fetch + extraction entirely** (printing `[read] 命中缓存全文，跳过抓取/抽取`). A
> previously downloaded OA PDF is likewise reused (and its mtime bumped, which is what `cache prune`
> uses for LRU ordering). The browser-fetch layer reuses a cached bundle by URL. Pass `--refresh` to
> re-fetch and re-extract. Expensive artifacts are kept permanently; see
> `references/maintenance.md` §5 for the two-tier cache and `research cache stats｜clean｜prune`.

It prints the paths. **Read the `全文 MD` file to actually read the paper**, then fill in the
note's evaluation sections (rubric below).
