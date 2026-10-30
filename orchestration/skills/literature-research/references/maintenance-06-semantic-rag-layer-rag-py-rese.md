# maintenance 分册 6/11

> 含小节：5d. Semantic RAG layer (`rag.py` / `research rag`)
> 原 `maintenance.md` 超 8KB 预算拆分（2026-10-10，backlog 20261010-refs-split-8kb）；分册索引见同目录 `maintenance.md`，按需只读所需分册。

## 5d. Semantic RAG layer (`rag.py` / `research rag`)

A local, **embedding-first** retrieval layer over the MinerU-extracted corpus (`cache/extracted/**/*.md`),
built on **PaperQA2** (`paper-qa`, a core dependency — no torch) + **SiliconFlow** (OpenAI-compatible).
Design intent: the Agent's own infrastructure for reading *across* the local library — `search` is the
primary path (pure embedding, free, no LLM); `ask` is an optional convenience that must never block.

- **Backend import is lazy + guarded.** `_import_backend(models=…)` first checks `settings.pqa_ready`
  (= `bool(SILICONFLOW_API_KEY)`) and raises a clear `RuntimeError` naming the missing key if not; it sets
  `LITELLM_LOCAL_MODEL_COST_MAP=True` **before** importing litellm (avoids a remote cost-map fetch that can
  hang), silences litellm's debug banner + a known harmless `async_success_handler` RuntimeWarning, then
  `_register_litellm_models()` declares each model (with **and** without the `openai/` prefix) incl.
  `max_input_tokens` — omitting it makes embedding calls `KeyError`. `paperqa`/`litellm` are imported only
  here, so `import rag` / `rag status` / `index_status()` never pull the heavy stack or touch the network.
- **S2-free by construction.** `_build_pqa_settings()` sets `parsing={"use_doc_details": False,
  "multimodal": False}`, so PaperQA2 does **not** call Semantic Scholar / Crossref for per-doc metadata.
  Citation/title/year/DOI/first-author are derived **offline** from the `.md` head by `_derive_meta_from_md`
  (first `#` H1 → title; a four-digit year only next to `published`/`accepted`/`received`/`©`; the first
  `DOI:`; the first plausible author line — skipping URL/affiliation lines and requiring ≥2 name tokens).
  A journal paper yields `Xia et al. (2025)`; a manual/no-author doc degrades to its title. `_map_mailto_env`
  still forwards `OPENALEX_EMAIL` → `CROSSREF_MAILTO`/`OPENALEX_MAILTO` for politeness if any source is hit.
- **Corpus cleaning before embedding (`corpus_clean.py`, 2026-10-04).** Because paperqa chunks a
  `.md` by *character count* (§3), every noise character is a character of body text pushed out of a
  chunk. `build_index(clean=True)` — the default — therefore passes each artifact through
  `corpus_clean.clean_markdown` first and hands **the copy** to `Docs.aadd`. Three deletion rules,
  each independently unit-tested and *deletion-only* (no rewriting, no reordering, no guessing at
  semantic structure): a LaTeX/REVTeX `thebibliography` environment (unterminated → swallow to EOF);
  MinerU's standalone empty-alt image placeholder lines (`![](images/<64-hex>.jpg)` — figure
  *captions*, which MinerU emits as separate `FIG. 1. (a) …` text lines, are **kept**);
  and numbered reference tables (≥ `MIN_REF_BLOCK` = 3 consecutive `[n] …` lines, blank lines
  between entries allowed — fewer than 3 is kept, because a short list can't be told from prose
  enumeration). Runs of ≥3 blank lines left behind are collapsed to one, since blank lines also cost
  chunk budget.
  Measured on the real corpus: image placeholders 272,685 chars (1.40 %), numbered reference blocks
  319,995 chars (1.64 %, 95 blocks) → **3.0 % overall**, but that average is diluted by the 12.4 M-char
  COMSOL manual (63 % of the corpus, and not what anyone asks questions of). **Per physics paper the
  same noise is 25 %–40 %** — which is the number that actually matters.
  Two invariants worth protecting: the **original artifacts are never modified** (they are either
  MinerU-quota-expensive or hand-curated into topic subdirs like `cpa_ep/` that don't match
  `pdf_extract`'s `{stem}_{key}.{backend}.md` cache naming, i.e. not reproducible), and the copies
  are written with `newline="\n"` so they are byte-identical across platforms.
- **Persistence + incremental index.** `build_index(paths=None, rebuild=False, clean=True)` embeds each candidate `.md`
  (docname = `__`-joined path-relative-to-`cache/extracted` stem) via `Docs.aadd(citation=…, title=…, doi=…)`,
  then pickles the `Docs` to `cache/rag/index.pkl` + writes `index_meta.json` (`files{docname:{path,mtime,
  title,year,doi,first_author,citation}}`, `n_docs`, `n_chunks`, `embedding_model`, `paperqa_version`,
  `cleaner_version`, `built_at`). Re-running is incremental: same mtime → **skip**, changed mtime → **stale** (not silently
  overwritten), new → **add**; `--rebuild` starts fresh *and* wipes `cache/rag/corpus/` so a deleted
  source can't leave an orphaned copy behind (`reset_corpus_dir` refuses to delete anything not named
  `corpus`, so a mistaken `pqa_home` makes it a no-op rather than an accident). `_load_docs` returns `None` on a version mismatch or
  corrupt pickle (→ treated as "no index"). `PQA_HOME` overrides the index dir (default `cache/rag/`).
  **The ledger's `path` stays the *source* file even though the *copy* is what got embedded** — that is
  what keeps `search`'s `source_path` pointing at a file a human can open, rather than at a derivative
  that vanishes on the next `--rebuild`.
- **`cleaner_version` is a rebuild trigger, and it has to be.** The incremental path skips on mtime,
  and a source file's mtime does not change when the *cleaning rules* change — so without this guard
  "copies cleaned by the old rules + code implementing the new ones" would coexist forever and the new
  rule would never take effect. Bump `corpus_clean.CLEANER_VERSION` whenever you touch a deletion rule.
  The same mechanism makes flipping `--no-clean` (stored as version `0`) force a full rebuild.
  `IndexReport.n_chars_raw` / `n_chars_clean` carry the measured cut; `clean=False` leaves both at 0
  because `_prepare_text` returns `None` rather than a zero-valued `CleanStats` — so the verbose
  summary can say "disabled" instead of printing a fake "removed 0 %".
- **`search` (primary).** `Docs.retrieve_texts(query, k)` → `list[Text]` (MMR-ranked, embedding-only, no LLM).
  Each `Text` maps to a `RagChunk(rank, text, docname, citation, source_path, chunk_name)`; the source
  attribution chain is `chunk.text` + `chunk.doc.docname` + `chunk.doc.citation`, with `source_path` recovered
  from `index_meta.json`'s docname→path map. No index → a clear "run `research rag index`" error.
- **`ask` (optional, never blocks).** Tries the free `PQA_LLM` (`Qwen2.5-7B`, cost 0) → on any failure the paid
  `PQA_LLM_FALLBACK` (`Qwen2.5-32B`) → if both fail (or the backend/index is unavailable), `_degrade` sets
  `backend="none"`, `degraded=True`, and fills `result.search` with plain `search()` output instead of raising.
  `render_ask` labels the degraded case. This "free → paid → degrade, no noise" ladder is a hard product
  requirement (the Agent is itself a strong LLM; `ask` is a convenience, never a dependency).
- **Data structures** `RagChunk` / `RagSearchResult` / `IndexReport` / `RagAskResult` are dataclasses with
  `to_dict()` (JSON-serializable, nested dataclasses folded) for `--json`. `research.py`'s `cmd_rag` dispatches
  `index/search/ask/status`; `main()` does **not** attach the Tier-B autoclean hook to `rag` (it writes no
  `api_responses`). Config: `settings.pqa_embedding`/`pqa_llm`/`pqa_llm_fallback`/`pqa_home` (+ `pqa_ready`).

---
