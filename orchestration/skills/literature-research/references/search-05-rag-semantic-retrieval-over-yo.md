# search 分册 5/5

> 含小节：`rag` — semantic retrieval over your own extracted corpus (PaperQA2)
> 原 `search.md` 超 8KB 预算拆分（2026-10-10，backlog 20261010-refs-split-8kb）；分册索引见同目录 `search.md`，按需只读所需分册。

## `rag` — semantic retrieval over your own extracted corpus (PaperQA2)

Where `search` discovers **external** papers, `rag` queries the corpus you have **already** extracted
to `cache/extracted/**/*.md` (via `read`/`ingest`). It builds a local embedding index (SiliconFlow
`bge-m3`, free) and retrieves **by meaning**. Needs `SILICONFLOW_API_KEY` (+ `SILICONFLOW_BASE_URL`).

```
research rag index  [--path P …] [--rebuild] [--no-clean]
research rag search '<query>' [-k N] [--chars N] [--json]
research rag ask    '<query>' [-k N] [--chars N] [--json]
research rag status [--json]
```

| Subcommand | Effect |
|---|---|
| `index` | Build/refresh the embedding index over `cache/extracted/**/*.md` (or explicit `--path` files/dirs). Each file is first **normalized into a cleaned copy** under `cache/rag/corpus/` and that copy is what gets embedded — the ledger's `source_path` still points at the original, which is never modified. **Incremental**: unchanged files skipped, changed files flagged stale, new files embedded; `--rebuild` forces from-scratch (and wipes `corpus/`, so no orphaned copies survive). Persisted to `cache/rag/index.pkl` + `index_meta.json`. Embedding-only (no LLM, free). |
| `search` | **The agent's main tool.** Pure embedding retrieval — top-`k` (default 8) most relevant chunks, each with **rank, text, source citation, file path**. No LLM, no cost, never blocks. Read the chunks and synthesize across papers yourself. |
| `ask` | *Optional* one-shot summary. PaperQA2 answer over the retrieved evidence: **free** `Qwen2.5-7B` first → smooth fallback to paid `Qwen2.5-32B` if the free tier is unavailable → if every LLM path fails, **silently degrades** to the `search` results (never raises). Prefer `search` + your own synthesis. |
| `status` | Backend readiness (`SILICONFLOW_API_KEY`) + local index state (exists, doc/chunk counts, embedding model, built-at, the corpus-cleaner rule version actually used, paperqa version). Does not import the backend or touch the network. |

Flags: `-k/--top-k` (chunks to retrieve, default 8), `--chars` (per-chunk truncation in the human
render, default 400), `--json` (machine-readable result), `--path` (index only these files/dirs instead
of the whole `cache/extracted/`), `--rebuild` (ignore the existing index), `--no-clean` (embed the
raw `cache/extracted` artifacts and build no `corpus/` copies).

### Reading `search` output

```
[1] Xia et al. (2025)  —  cpa_ep__paper_prl  (…/cache/extracted/cpa_ep/paper_prl.md)
    …chunk text (truncated to --chars)…
```
Each hit = `rank`, `citation` (derived from the MinerU `.md` head: first author + year, else title),
`docname` (the `__`-joined relative-path stem), and `source_path`. Retrieval is MMR-ranked (diversity),
so hits come back in relevance order without a numeric score. `--json` emits
`{query, chunks:[{rank,text,docname,citation,source_path,chunk_name}], n_docs, n_chunks, index_path,
embedding_model}`.

### Notes

- **Corpus normalization is deletion-only and never touches the artifacts.** Three rules —
  MinerU's empty-alt image placeholder lines, blocks of ≥3 consecutive numbered references, and
  raw REVTeX `thebibliography` environments left by the arXiv-LaTeX path. Figure *captions* are
  **kept**. Across the whole corpus that is 3.0 % of characters; **per physics paper it is
  25 %–40 %**, which is the number that matters (the average is diluted by the 12.4 M-char COMSOL
  manuals). It matters because paperqa `2026.8.12` chunks a `.md` with `chunk_code_text` — by line
  count to a character budget, blind to headings and sentence boundaries — so noise characters spend
  real chunk budget and push body prose out of the top-`k`.
- **A cleaner-rule version change forces a full rebuild** (`cleaner_version` in `index_meta.json` vs
  `corpus_clean.CLEANER_VERSION`), and so does flipping `--no-clean` either way. Without that guard
  the incremental path would skip files by mtime and the new rules would never take effect. `rag index`
  printing “语料规范化版本变更” is therefore expected self-healing, not an error.
- **Metadata is derived offline** from each `.md` head (`# title`, author line, `published/received`
  year, `DOI:`) — no Semantic Scholar / Crossref enrichment (`use_doc_details=False`), so indexing pays
  no per-doc network cost and stays S2-free.
- **Index lifetime**: the pickle is version-stamped; a paperqa-version mismatch or a corrupt file is
  treated as "no index" (rebuild). `cache/rag/` — including `corpus/` — is git-ignored and fully
  rebuildable with `rag index`.
- **When `search` reports "no index"**, run `research rag index` first. **When `ask` degrades**, it
  prints the reason and returns `search` results instead — by design (never block, never noise).
- **A hit that looks like it is missing its figures or reference list is behaving correctly** — it is
  rendering the cleaned copy while `source_path` points at the original. Read the original file if you
  need what was scrubbed.
