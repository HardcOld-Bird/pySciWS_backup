# Data sources, expectations, and debug backdoors

What the `research` CLI queries directly, what each source can and cannot tell you (including
where every journal-quality number really comes from), and the per-module `python -m …` debug
backdoors that are the one caveat to treating the backend as a black box.

These are the sources the `research` CLI queries **directly** (the fallback + differentiated path).
The `paper-search-mcp` MCP aggregates ~22 sources on the primary path — see `SKILL.md` § *Who does what*.

- **OpenAlex** — primary source, rich metadata + citation counts. Since 2026-02-13 an API key
  (`OPENALEX_API_KEY`) is required for full quota; without it, requests are capped at ~100
  credits/day (testing only). In that key-less state a **zero-result** `search` prints an explicit
  warning instead of staying quiet: an empty list is indistinguishable from an exhausted quota, so
  re-check with `--source arxiv` or the search MCP before concluding a topic has no literature.
- **arXiv** — preprints, no key. Use for preprint-only or latest work.
- **Crossref** — free REST (no key; `mailto` polite pool), queried by the **citation integrity gate**
  (`citecheck` / `add`), not by `search`. It is the second independent source (alongside OpenAlex +
  arXiv) that cross-checks title / DOI / year before a citation is written. See
  [citecheck.md](citecheck.md).
- **Web of Science** (Starter API) — *optional* enrichment: adds the WoS accession no. (`wos_id`)
  + authoritative Times Cited, and a JCR URL. It carries **no official JIF / JCR quartile / ESI**
  (the Starter API doesn't expose those), so those frontmatter fields stay OpenAlex-derived. Degrades
  silently when unavailable; a one-line notice is printed, no action needed.
  *Upgrade path (applied for, not yet granted):* the **WoS Journals API**
  (`https://api.clarivate.com/apis/wos-journals/v1`, same `X-ApiKey` auth) is the programmatic source
  for official JIF / JIF quartile / JCI / ESI — **not** WoS API *Expanded*, which adds author,
  institution, identifier and funder data and still has no JIF. Official client:
  `clarivate/wosjournals-python-client`. Recorded in `tools/wos_client.py`'s module docstring so it
  needn't be re-researched.
- **Journal quality (free layers, no key)** — until the Journals API lands, venue standing comes from
  three independent and **non-official** layers, which coexist because they measure different things:
  OpenAlex `listed_in` → `journal_tier` (JUFO / Norway / KI-JL **expert-panel** tiers, zero extra
  requests; for a low-citation-density field like physical acoustics this tracks domain consensus
  better than any citation metric — JASA scores 0.82 on 2-year mean citedness yet sits in the top
  tier alongside Nature and PRL); **SCImago SJR** → `scimago_quartile` (true Q1–Q4 by exact ISSN,
  from a one-time `journal build-scimago --csv <manually downloaded CSV>`); and OpenAlex
  `2yr_mean_citedness` → `jif` (an **estimate** — fair mid-range, can understate top journals 2–3×).
  `jcr_quartile` stays empty and `esi_*` stays `null` (**unknown**, never `false`). Weigh a venue
  with `research journal lookup <issn>`, which shows all of them side by side.
- **MinerU cloud** (via `mineru-open-sdk`) — primary PDF → Markdown backend (equations → LaTeX,
  tables → HTML). Needs `MINERU_TOKEN`. `pymupdf4llm` is the local fallback (fast, but equations
  are lost). Paywalled HTML full text is body-extracted with `trafilatura`.
- **SiliconFlow** (OpenAI-compatible) — powers the local **RAG layer** (`rag index`/`search`/`ask`):
  `bge-m3` embeddings (free) for the semantic index + retrieval, plus an *optional* Qwen chat model
  for `rag ask`. Needs `SILICONFLOW_API_KEY`. Embedding-only retrieval is free; the `ask` LLM tries a
  free tier first, falls back to paid, and never blocks (degrades to `search`).


## Debug backdoors — the one caveat to "black box"

One honest caveat to “black box”: every backend module also carries a `python -m …` **debug
backdoor** (an `__main__` block) for isolating a single layer, e.g.
`uv run python -m pysci.skills.literature_research.tools.wos_client check`. Each prints a banner to
**stderr** naming the equivalent `research` subcommand, so nothing is hidden — but they are for
troubleshooting only, and normal work always goes through `research`. Three actions have *no* CLI
equivalent and exist only there: `browser_fetch batch|pdf`, `wos_client journal` (WoS journal record
+ JCR URL), `zotero_cli bibtex`. Banners stay on stderr precisely so the JSON some backdoors print on
stdout remains pipe-clean.
