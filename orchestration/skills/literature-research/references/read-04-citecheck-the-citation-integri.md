# read 分册 4/6

> 含小节：`citecheck` — the citation integrity gate；`index` — rebuild the library index
> 原 `read.md` 超 8KB 预算拆分（2026-10-10，backlog 20261010-refs-split-8kb）；分册索引见同目录 `read.md`，按需只读所需分册。

## `citecheck` — the citation integrity gate

```
research citecheck [<doi｜arxiv-id｜openalex-id｜title> …]
                   [--note <path>] [--all]
                   [--bib <refs.bib｜draft.md> …] [--review] [--limit N]
                   [--json] [--no-color]
```

Cross-checks a citation against **three independent open sources — OpenAlex + Crossref + arXiv**
(bypassing Semantic Scholar) to catch fabricated or mis-paired references: a DOI that doesn't match
its title, a wrong year, a venue that never published it. Feed it three ways:

- **Bare identifiers / titles** as positional `targets` (any number), e.g.
  `citecheck 10.1103/PhysRevLett.121.124501` or `citecheck 'some paper title'`. Each is classified
  (DOI → arXiv id → OpenAlex id → free-text title) then verified.
- **Notes** — `--note <file-or-dir>` (repeatable) verifies the frontmatter of specific `papers/*.md`;
  `--all` sweeps every note in `papers/`.
- **Bibliographies** — `--bib <path>` (repeatable) parses a reference list and verifies each entry.
  A `.bib` file goes through a BibTeX parser: `{}`- and `""`-quoted values, LaTeX escapes (`\"o`,
  `{\'e}`) stripped, `@string` / `@comment` ignored, and `title`/`author`/`year`/`journal`/`doi`/
  `eprint` mapped onto the same shape the other inputs produce. Anything else — typically a draft
  `.md` — goes through a **References-section** parser that reads the lines under a `参考文献` /
  `References` / `Bibliography` heading and picks up DOIs, `arXiv:NNNN.NNNNN` ids, years and titles.
  That parser is explicitly **best-effort**: a line with neither an identifier nor a recognizable
  title is skipped rather than guessed at.
- **`--review`** — shorthand for `--bib` over every `reviews/*.md`, i.e. "audit the survey I just
  wrote". Run it before calling a survey done.

**`--limit N`** (default **50**, `0` = unlimited) caps how many *parsed* references get verified, and
the number actually checked is printed. It exists because each reference costs up to **3 network
requests** (one per source), so a 200-entry bibliography is 600 calls — easy to start by accident and
slow to stop.

**Verdict** per citation: `✓PASS` (sources agree) / `△WARN` (only soft differences) / `✗FAIL` (a hard
conflict) / `?NOT_FOUND` (no source has it). Severity model:

| Signal | Severity | Verdict | Blocks `add`? |
|---|---|---|---|
| Title mismatch / DOI mismatch / year off by ≥2 | hard | **FAIL** | **yes** |
| Author-surname mismatch / year off by 1 / journal-name-only diff | soft | WARN | no |
| All three sources miss the record | — | NOT_FOUND | no (flagged for review) |
| A source is unreachable (network) | — | downgraded | no (never a false FAIL) |

**Exit code:** `1` if any citation is FAIL, else `0` — usable as a CI / pre-write gate. `--json`
emits a machine-readable verdict array (label / kind / status / passed / reasons / per-source records
/ conflicts). `add` calls this same engine internally before writing.

> The gate is deliberately **conservative**: it hard-blocks only on signals that are unambiguous
> across independent sources, so author-name transliteration noise (e.g. `Büttner` → `buttner` vs a
> lossy upstream `bttner`) or an online-first-vs-issue-year off-by-one never fails a real citation.

> **What the gate can and cannot check.** `citecheck` proves a reference *exists* and that three
> independent sources agree on its metadata. It **cannot** prove that what your survey says about a
> paper is what that paper actually says — and nothing can check that mechanically. The convention
> covering it is an **anchor**: every factual claim in a survey carries a pointer to the extracted
> full text it came from (`cache/extracted/<slug>_fulltext.md`, with a section or figure where it
> matters). The pointer is free — `read` prints that path, and `rag search` returns `source_path` with
> every chunk — so there is no excuse for a claim that cannot be traced to a file a reader can open.

---

## `index` — rebuild the library index

```
research index [--check] [--fix [--dry-run]] [--force]
```
Scans `papers/*.md`, reads each note's frontmatter, and rewrites `INDEX.md` as a table sorted by year
(`# / Year / First Author / Title / Journal / Tier / Status / Rating / Note`). `Tier` gets its **own**
column rather than being folded into `Journal`: the journal name is a bibliographic fact and the tier
is a derived evaluation, and concatenating them makes the column neither sortable nor filterable.
Run it after adding or editing notes.

- `--check` — report drift without writing; exit **1** if stale. The build-date line is excluded from
  the comparison, so `--check` fails only when `papers/` actually changed, not merely on a new day.
- `--fix` — normalize every note's frontmatter first: add the keys `templates/paper_note.md` defines
  but the note lacks, unify field order, and expand flow-style lists (`authors: [A, B]`) into block
  style. **All existing values are preserved and every body stays byte-identical.** `INDEX.md` is then
  rebuilt in the same pass, so one run suffices. It also *reports* notes whose filename doesn't match
  `{year}_{last}_{slug}.md` but **never renames them** — a rename would break the `[[wiki-links]]` in
  `reviews/*.md` and any external reference you have made. Be clear about what that report can see: the
  expected name is derived from the note's **own stored** frontmatter, so it catches a hand-renamed file
  or a `short_title` edited after creation, but **not** the gap between a stored and a *freshly fetched*
  `short_title` — `index --fix --dry-run` reports 已是规范形态 for all three current notes even though
  `2018_zhu` demonstrably drifts. That one is now absorbed at write time rather than reported
  (`maintenance.md` §5f).
- `--dry-run` — with `--fix` only: report what would change and write **nothing at all**, `INDEX.md`
  included. (Given without `--fix` it is ignored, with a notice.) On a real library always run this
  first, then confirm `git diff data/skills/literature_research/papers/` shows frontmatter only.
- `--force` — rebuild `INDEX.md` even when `papers/` is empty; by default an existing index is kept,
  since an empty `papers/` far more often means a wrong working directory than an empty library. This
  is the one `--force` that kept its name, because here it really does mean "force the write".

---
