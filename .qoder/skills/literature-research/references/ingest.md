# Bulk-ingesting a local PDF repository

How `research ingest` archives a folder of local PDFs that have no DOIs to resolve: the
git-tracked `ingest/manifest.json` ledger, priority batching against the MinerU daily page
quota, resumability, and where copies and extracted Markdown land. `SKILL.md` keeps only the
rule that a local PDF folder goes through `ingest`, not `read` / `add`.

When the user already has a folder of PDFs (no DOIs to resolve), use `ingest` instead of
`read`/`add`. It is driven by a curated, **git-tracked** ledger `ingest/manifest.json` that records
each file's `theme / slug / type / priority / pages / source / status`:

```
research ingest status                     # progress summary (done / pending / failed, by priority & theme)
research ingest --priority 1 --dry-run     # preview a batch, no extraction
research ingest --priority 1               # run it: copy PDF → MinerU extract → cache/extracted/<theme>/<slug>.md
research ingest --priority 1 --limit-pages 800   # cap pages this run (respect MinerU daily quota)
```

The `action` positional is optional and defaults to `run`, so `research ingest --priority 1` and
`research ingest run --priority 1` are the same command; `ingest status` and the legacy
`ingest --status` are likewise equivalent.

Behaviour: PDFs are **copied** (originals untouched) to `cache/pdfs/<theme>/<slug>.pdf`; Markdown is
extracted to `cache/extracted/<theme>/<slug>.md`. It is **resumable** — the manifest is rewritten
after every file, `done` entries are skipped on re-run, `failed` entries can be retried. Use
`priority` to batch by cost (1 = short papers, 2 = reviews/theses, 3 = big textbooks) so you stay
within MinerU's daily page quota and can continue on a later day. Author/edit `manifest.json` by
hand to (re)classify; the `source` paths must match the real files.
