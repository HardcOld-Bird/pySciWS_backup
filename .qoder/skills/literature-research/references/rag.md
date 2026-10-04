# The semantic RAG layer (search your own extracted corpus)

How `research rag index|search|ask|status` build and query a local embedding index over the
Markdown that `read` / `ingest` extracted: corpus cleaning, incremental indexing, cost and
blocking behaviour, and the required `.env` keys. `SKILL.md` keeps only the four subcommand
names and the rule that `rag search` is the agent's main tool here.

Everything `read`/`ingest` extracts lands in `cache/extracted/**/*.md`. The `rag` commands build a
**local embedding index** over that corpus (PaperQA2 + SiliconFlow `bge-m3`) so you can retrieve
across *your own* library **by meaning** — not just discover external papers via `search`.

- **`research rag index`** — build/refresh the index. Each artifact is first **normalized into a
  cleaned copy** under `cache/rag/corpus/` (empty-alt image placeholder lines, blocks of numbered
  references, raw REVTeX bibliographies removed — 25 %–40 % of a typical physics paper) and that copy
  is what gets embedded; `cache/extracted/` is **never modified** and `source_path` still points at the
  original. Incremental (unchanged files skipped, changed files flagged stale, new files embedded);
  persisted as a pickle + JSON meta under `cache/rag/`. **Embedding-only, free** (no LLM). Re-run it
  after `read`/`ingest` adds new full text — a cleaner-rule version change forces a full rebuild by
  itself. `--no-clean` embeds the raw artifacts instead.
- **`research rag search '<query>'`** — the agent's **main** tool here: pure embedding retrieval that
  returns the top-k most relevant chunks **with their source citation + file path**. No LLM, no cost,
  no blocking. Read the returned chunks and synthesize across papers yourself.
- **`research rag ask '<query>'`** — *optional* one-shot summary convenience. Tries the **free** LLM
  (`Qwen2.5-7B`) first, falls back smoothly to a paid model (`Qwen2.5-32B`) if the free tier is
  unavailable, and **never blocks**: if every LLM path fails it silently degrades to returning the
  `rag search` results. Since you are already a strong LLM, prefer `rag search` + your own synthesis;
  use `ask` only when a quick canned answer is wanted.
- **`research rag status`** — backend readiness (`SILICONFLOW_API_KEY`) + whether a local index
  exists (doc/chunk counts, embedding model, built-at).

Requires `SILICONFLOW_API_KEY` (+ `SILICONFLOW_BASE_URL`); model/home overrides are the optional
`PQA_*` vars. Title/year/DOI/first-author → citation is derived from each MinerU `.md` head, so
retrieved chunks carry a human-readable source. Indexing bypasses Semantic Scholar / Crossref
enrichment (`use_doc_details=False`) — it stays free-embedding + offline-metadata only. See
[search.md](search.md) for flags and [maintenance.md](maintenance.md) §5d for internals.
