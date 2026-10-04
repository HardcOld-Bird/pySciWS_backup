# Citation integrity gate (OpenAlex + Crossref + arXiv)

How `research citecheck` — and the automatic gate inside `research add` — decide whether a
citation may be written to the library: the three independent sources, the severity model,
the standalone audit modes, and what the gate *cannot* check (plus the anchoring convention
that covers the rest). `SKILL.md` keeps only the rule that `add` verifies before writing.

Before anything is written to the library, every citation is cross-checked against **three
independent open scholarly sources** — OpenAlex, **Crossref** (free REST, no key), and arXiv —
deliberately bypassing Semantic Scholar. This borrows the *idea* of a citation-audit step (no external
skill body is vendored in); the engine lives in `citation_verify.py`.

- **Auto-gate on `add`.** `research add` verifies the fused frontmatter before creating the Zotero
  item / writing the note. A **FAIL** (≥2 reachable sources hard-conflict on title / DOI / year)
  blocks the write; `--allow-fail` overrides, `--no-verify` skips. A verification error degrades
  gracefully (never blocks the main flow).
- **Standalone `citecheck`.** Verify bare DOIs / arXiv ids / OpenAlex ids / titles, audit existing
  notes (`--note <path>`, `--all` over `papers/`), or audit a **reference list** (`--bib refs.bib`,
  `--bib draft.md`, or `--review` for every survey under `reviews/`). Each reference costs up to 3
  network calls, so `--bib`/`--review` cap at `--limit 50` by default (`--limit 0` = no cap). The
  exit code is CI-friendly: **1 if any FAIL**, else 0.
- **Severity model.** Only a **hard** conflict (title mismatch, DOI mismatch, year off by ≥2) is a
  FAIL. Soft signals — author-surname mismatch, year off by 1 (online-first vs issue year),
  journal-name-only difference — are a **WARN** and never block. All three sources missing the record
  is **NOT_FOUND** (suspicious, flagged for human review, not blocked). An unreachable source only
  downgrades; it is never treated as a conflict.

**What the gate can and cannot check.** `citecheck` establishes that a reference *exists* and that its
title / DOI / year / venue agree across three independent sources — it catches fabricated and
mis-paired citations. It **cannot** establish that what your survey *says about* a paper is what that
paper actually says; no tool can. That gap is covered by a convention instead: give every factual
claim in a survey an **anchor** to the extracted full text it came from
(`cache/extracted/<slug>_fulltext.md` — the `全文 MD` path `read` prints, or the `source_path` that
`rag search` already returns, so the pointer costs nothing). An anchored claim can be re-checked by
reading that file; an unanchored one cannot be audited at all.
