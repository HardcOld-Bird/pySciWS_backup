# maintenance 分册 8/11

> 含小节：5f. Journal quality metrics (`journal_metrics.py`)
> 原 `maintenance.md` 超 8KB 预算拆分（2026-10-10，backlog 20261010-refs-split-8kb）；分册索引见同目录 `maintenance.md`，按需只读所需分册。

## 5f. Journal quality metrics (`journal_metrics.py`)

The free substitute layer for official JIF / JCR quartile / JCI / ESI, which only the **WoS
Journals API** can supply (application still pending — `wos_client.py`'s module docstring records
the upgrade path so it needn't be researched again). Three fields coexist and mean different
things:

| Field | Source | Cost |
|---|---|---|
| `journal_tier` (+ `journal_tier_basis`, `listed_in`) | OpenAlex `listed_in` → `notes.derive_journal_tier` (§5e) | zero extra requests |
| `scimago_quartile` | local SCImago SJR index, exact-ISSN match | one manual CSV download per year |
| `jcr_quartile`, `esi_highly_cited`, `esi_hot_paper` | WoS Journals API | not wired — stays `""` / `null` |

**The module never touches the network.** It answers from a compact JSON index at
`settings.scimago_index_path` = `data/skills/literature_research/data/scimago_index.json`
(git-tracked; `settings.scimago_ready` is just `.exists()`). It is a *data asset*, not a cache
artifact, which is why it lives under `data/` rather than `cache/` and so escapes the `cache/*`
ignore rule.

- **`build_scimago_index(csv_path, *, year=None, out_path=None)`** turns the official CSV
  (scimagojr.com → *Journal rank*, ~11 MB; the 2025 export measured 10.73 MB) into
  `{"_meta": {…}, "by_issn": {"00319007": ["Q1", 2.79, 750], …}}` — only four columns survive
  (`Issn`, `SJR`, `SJR Best Quartile`, `H index`), written with `separators=(",", ":")` and no
  indent, ≈1.4 MB (2025 build: 53,404 ISSNs, 1,427,727 B). A single trailing newline is appended
  on purpose: the index is a *tracked* file, and a tracked file without one gets rewritten by
  `end-of-file-fixer`, which aborts the commit — i.e. every yearly refresh would waste a failed
  `git commit`. Each detail below covers a real failure mode:
  - `_detect_delimiter` picks `;` or `,` by whichever appears more often in the header. The
    official export is semicolon-delimited (European style) but comma mirrors exist; hardcoding
    `;` turns every row into a single column and **silently produces an empty index**.
  - `_to_float` accepts the European decimal comma — the 2025 export writes PRL's SJR as `2,790`,
    which must land in the index as `2.79` and not `2790`.
  - `_extract_issns` splits multi-value cells, and its separator set deliberately **excludes** `-`
    (else `0031-9007` splits in half). The 2025 export *quotes* multi-value cells and separates
    them with `, ` unhyphenated (`"10797114, 00319007"`), which the csv reader handles as one
    field — hence 32,194 CSV lines → 53,404 index keys. Should a future export drop the quotes,
    `;` is both the field delimiter and a plausible ISSN separator, so that row arrives split
    across fields; the builder compensates for the resulting column shift and rejoins them. That
    compensation is **defensive** (`extra` measured 0 on the 2025 file), not the normal path.
  - `_at()` reads columns out-of-bounds-safely — SCImago rows occasionally lack trailing columns,
    which must not cost the whole row.
  - `_infer_year` looks for a parenthesised year in the header (`Total Docs. (2025)`), else in the
    filename, else returns `None`. It never invents one; pass `--year` when it can't tell.
  - Raises `FileNotFoundError` / `ValueError` (empty CSV, missing required column, not one ISSN
    parsed). **This is the one path in the module that must not degrade silently**: a failed build
    that quietly wrote an empty index would blank `scimago_quartile` in every note with no visible
    cause — far harder to diagnose than an error. `cmd_journal` maps it to exit 1 with the reason
    on stderr, and a missing `--csv` to exit 2.
- **Query paths degrade silently**, per the module-wide invariant. `lookup(issn)` accepts a single
  ISSN *or* a candidate list (OpenAlex's `issn` array) and returns the first hit as
  `{issn, quartile, sjr, h_index, sjr_year}`, else `None` — index missing, unreadable, corrupt, or
  no match. A corrupt index is treated as no index: that is safer than half-working.
  `quartile_for(issn)` returns `""` rather than `None` because `scimago_quartile` is a string
  field, and only `""` reads as blank to `merge_frontmatter` (§5e). `_load_index` caches a single
  entry keyed on `(path, mtime, size)`, so a rebuilt index is picked up without a restart and tests
  can redirect the path freely.
- **Matching is exact-ISSN only, by design.** No fuzzy title matching: a title map would add
  ~1.5 MB and introduce mis-matches, while an ISSN is always available from OpenAlex / Crossref /
  WoS metadata. `research journal lookup <issn>` distinguishes "index not built" from "this ISSN
  isn't in the index" — the fixes are completely different, and collapsing them into one empty
  result is exactly what makes venue metrics look broken when they aren't.
