# maintenance 分册 2/11

> 含小节：3. PDF → Markdown (`pdf_extract.py`)
> 原 `maintenance.md` 超 8KB 预算拆分（2026-10-10，backlog 20261010-refs-split-8kb）；分册索引见同目录 `maintenance.md`，按需只读所需分册。

## 3. PDF → Markdown (`pdf_extract.py`)

- `available_backends()` returns what the environment can actually run: `mineru-cloud` when
  `MINERU_TOKEN` is set, `pymupdf4llm` when that module is importable.
- `extract_pdf(path, backend=None)` with `backend=None`/`auto` picks the best available
  (MinerU cloud first). It caches results in `cache/extracted/` (`use_cache`/`write_cache`).
- **MinerU cloud** is the primary: a VLM pipeline with OCR that renders equations as LaTeX and
  tables as HTML. It goes through the community **`mineru-open-sdk`** (`from mineru import MinerU`),
  which owns auth / upload / polling / result unpacking (only needs `httpx` + `MINERU_TOKEN`);
  `_mineru_extract_pages()` just calls `client.extract(path, model="vlm", ocr=True, formula=True,
  table=True, language="en", extra_formats=["latex"], timeout=…)` and reads `result.markdown`.
  Typical cost ~1–3s/page.
- **MinerU Precision Extract has a 600-page single-file limit** (exceeding it raises the SDK's
  `PageLimitError`). `_extract_with_mineru_cloud()` handles this transparently: it reads the page
  count via fitz and, if > `MINERU_MAX_PAGES` (600), converts in `MINERU_CHUNK_PAGES` (600) page
  segments via the SDK's `pages="a-b"` range (no more fitz temp-file splitting), then concatenates
  the parts (each tagged `<!-- MinerU pages a-b (i/N) -->`). So `read`/`ingest` on a big textbook
  "just works" — expect `MinerU 分段 i/N: 页 a-b` log lines. Quota: MinerU's ~1000-page allowance
  is a *fast-track* quota, not a daily hard cap — beyond it jobs still run, just slower (normal
  queue); the daily file cap is 5000.
- **pymupdf4llm** is the local fallback: fast, CPU-only, but **equations are lost**. Use it only
  when equations don't matter or MinerU is down.
- **arXiv LaTeX source path** (`_tex_to_markdown`, tried *before* any PDF backend once an arXiv id is
  known): unpacks the e-print tarball, picks the main `.tex`, converts it. Step 1 is
  `_strip_tex_comments` — it removes `%` comments (honouring `\%` escapes), passes `verbatim`-class
  environments through untouched, and drops `comment` environments whole. Step 2 cuts everything
  outside `\begin{document}…\end{document}`. Only *then* do the two targeted regexes run:
  `_TEX_FRONT_MATTER_DROP_RE` deletes the REVTeX front-matter macros that carry no reading value
  (`\affiliation`, `\altaffiliation`, `\email`, `\onlinemail`, `\homepage`, `\thanks`, `\date`,
  `\address`, `\maketitle`, `\tableofcontents`, `\keywords`, `\preprint`, `\fax`), while
  `_TEX_TITLE_RE` / `_TEX_AUTHOR_RE` promote `\title` to an `# H1` and merge `\author` into one
  `**Authors:**` line; `_TEX_GRAPHICS_DROP_RE` removes `\includegraphics` and the figure wrappers but
  keeps `\caption` (already turned into `_Figure/Table caption: …_`). Verified end-to-end on
  `1803.04110`: 717 lines out, **0** comment lines, **0**
  `\documentclass`/`\usepackage`/`\begin{document}` leaks, all eight body sections (`Introduction` …
  `Conclusion`) intact.
  **Order matters and is load-bearing**: the comment strip must come *first*, because the
  `\begin{document}` cut routinely fails on multi-file arXiv projects (`\input{sec1.tex}`), so it
  cannot be relied on to swallow commented-out lines. The same reason means a `\newcommand`/`\def`
  written *inside* the body survives — those are only removed when the preamble cut succeeds.
  **The `thebibliography` block is now dropped at the source** (`_TEX_DROPPED_ENVS`, 2026-10-04).
  For that same 717-line file it was lines 136–~709 — **~80 % of that file** — and pure REVTeX macro
  residue (`\bibitem`/`\citenamefont`/`\bibinfo`/`\BibitemShut`) as far as `rag`'s embedder is
  concerned. The drop is guarded on the `\begin`/`\end` counts matching, so an *unbalanced*
  environment is left alone rather than swallowing the rest of the paper; the already-extracted
  artifacts and the unterminated case are covered at read time by `corpus_clean` (§5d).
  Bibliographic data has authoritative sources elsewhere (`citation_verify` / Zotero / OpenAlex), so
  nothing worth keeping is lost. Note the ~80 % figure belongs to *that one file*: measured across
  the whole corpus (74 `.md`, 19.5 M chars) `thebibliography` is only **0.2 %** and appears in
  **1 of 74** files — it is the arXiv-LaTeX path's noise, not the corpus's.
  **Remaining known limits** (pre-existing; none touch body prose or equations): in-text citations
  stay as BibTeX *keys* (`systems[EP2]`), key → number mapping not being implemented; `\ref` becomes
  `§label`; accents are not converted (`Aubry-Andr{\'e}-Harper`). Pinned by
  `test_tex_to_markdown.py`. One further limit is *not* pinned, because it is not recoverable from
  syntax at all: a heading the authors set by hand as `\emph{Introduction.--}` (common in PRL, where
  `\section` is skipped to save space) stays *italic prose* instead of becoming `##`, so such a file
  can end up with **zero** Markdown headings. Verified on 1803.04110 — all eight sections do survive,
  but as `*X.--*` / `*X. --*` lines, and `^#` matches only the H1. Nothing in the source marks an
  italic phrase as a heading, so no amount of regex fixes this.
  **That last one costs far less than it looks like, and the reason is worth un-learning.** This file
  used to record that heading-based chunking in `rag` would then see the whole paper as one block.
  It does not: paperqa `2026.8.12`'s `readers.read_doc` has **no `.md` branch**, so a `.md` falls
  through to `parse_text(split_lines=True)` + `chunk_code_text` — "based on line numbers (for
  code)" — which accumulates lines and hard-cuts at `chunk_chars` (default 5000, overlap 250)
  regardless of headings *or* sentence boundaries. Markdown structure has never influenced chunking;
  only **character count** does. That is precisely why §5d's `corpus_clean` deletes noise instead of
  repairing heading structure — the latter would buy nothing. (Should paperqa ever grow a real
  Markdown parser, the run-in heuristic becomes worth implementing: promote a line that is *only* an
  italic phrase ending in `.--` or `. --`, the APS run-in convention.)

---
