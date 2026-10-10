# maintenance 分册 9/11

> 含小节：5f. Journal quality metrics (`journal_metrics.py`)（续）· Refreshing the SCImago index (yearly)
> 原 `maintenance.md` 超 8KB 预算拆分（2026-10-10，backlog 20261010-refs-split-8kb）；分册索引见同目录 `maintenance.md`，按需只读所需分册。

### Refreshing the SCImago index (yearly)

1. Download the CSV **by hand** from `journal_metrics.DOWNLOAD_URL`
   (<https://www.scimagojr.com/journalrank.php>). The site returns 403 to programmatic fetches and
   needs form/JS interaction, so no auto-download is implemented or planned. `*.csv` is git-ignored
   under `data/skills/literature_research/` to keep the ~11 MB original out of the repo.
2. `research journal build-scimago --csv <path> [--year 2025]`, then check the reported entry count
   is in the tens of thousands. A count near zero means delimiter or column detection missed; the
   command refuses to write that (exit 1), but verify the number anyway.
3. Update `data/skills/literature_research/data/SOURCE.md` — download URL, SJR year, download date,
   attribution (`SCImago Journal & Country Rank, data based on Scopus (Elsevier B.V.)`). The index
   JSON *is* tracked; the CSV is not.
4. Confirm with `research journal status` and the 【期刊质量指标】 section of `research doctor` that
   `sjr_year` moved.

Existing notes are **not** rewritten by a refresh. A note whose `scimago_quartile` is blank gets it
filled on the next `read` / `add` / `get`; a note that already has a value keeps the old quartile,
because `merge_frontmatter` never overwrites (§5e).

Re-running `read <id>` to refresh one field is still the wrong tool, but as of 2026-10-04 it is no
longer *dangerous*. The reason it used to be: `_merge_note` computes its target path from the
*freshly fetched* frontmatter (`notes.note_filename`, whose slug comes from `short_title`), and
`openalex_client._make_short_title` drops stopwords (`of` / `and` / `in` / …) before taking the first
six content words. Whenever that derivation drifts from the value stored in the note — upstream
retitled the work, or the note was named by hand — the computed path missed the existing file,
`_merge_note` saw `existed=False` and **created a second note**. Nothing complained: the output looked
exactly like a normal merge. `2018_zhu` was in that state (stored `short_title` "Simultaneous
Observation **of** Topological Edge State" vs fresh "Simultaneous Observation Topological Edge State
Exceptional" → `2018_zhu_simultaneous-observation-topological-edg.md`), and so was `2023_fang` before
it was renamed with `git mv`.

**The write path now guards on identity, so that whole class of duplicate is gone.** When the derived
path does not exist, `_find_note_by_identity` scans `papers/` for a note carrying the same `doi`,
`openalex_id` or `arxiv_id` and merges into *that* file, printing both filenames so you can `git mv`
if you want the canonical name. The comparison is normalized, because all three identifiers have real
spelling variance: DOI is case-folded (OpenAlex returns lowercase, publishers and humans do not),
`openalex_id` is upper-cased, and an arXiv version suffix is stripped (`1803.04110` and
`1803.04110v2` are the same paper). Three limits are deliberate:

- **Several matches → refuse to write** (`blocked`; no file is created or touched). Guessing would
  fold two different papers' prose into one file, which is unrecoverable; a leftover duplicate is not.
- **`--overwrite` is downgraded to a merge** on that path. The redirect target is a file the caller
  never named, and rebuilding it from the template would silently erase hand-written TLDR / Key
  Claims / Novelty / Rigor prose. Rename or delete it first if you truly want a rebuild.
- **No identifier at all → old behaviour** (create). A note with none of the three keys still cannot
  be recognized — that is the one remaining route to a duplicate, and it is pinned by
  `test_merge_note_without_identity_keys_creates_as_before`.

`title` / `short_title` / `year` / `first_author_last_name` are deliberately **not** identity keys:
they are precisely the inputs to `note_filename`, so using them to decide "same paper?" is circular.
Keep all four out of any batch write-back whitelist (§5e).

Still prefer the §5e whitelist write-back for a single field: it costs no network call, triggers no
re-extraction, and cannot touch anything but the one key.

**No command *reports* the drift — the guard absorbs it at write time.** `index --check` only diffs
`INDEX.md` against `papers/`. `index --fix` *does* warn when a filename doesn't match
`{year}_{last}_{slug}.md` (the warning comes from `_normalize_note_file`, and it never renames) — but
it derives the expected name from the note's **own stored** frontmatter, which by construction agrees
with the file it lives in. Verified 2026-10-04: `index --fix --dry-run` reports 已是规范形态 for all
three notes, including `2018_zhu`, which demonstrably drifts (`…-observation-of-topological.md`
stored vs `…-observation-topological-edg.md` fresh). To see the drift yourself, compare a fresh
derivation against the stored value — `research get <id>` prints the `short_title` a new fetch would
produce.

---
