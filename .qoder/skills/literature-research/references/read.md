# read / add / citecheck / index / review — detailed reference + evaluation rubric

All commands are invoked as `research <cmd> …` (see SKILL.md for the full invocation and the
PowerShell single-quote rule).

---

## `read` — fetch full text + build a note

```
research read <doi｜url｜arxiv-id｜openalex-id>
              [--backend mineru-cloud｜pymupdf4llm｜auto]
              [--headed] [--no-note] [--overwrite] [--refresh]
```

What it does, in order:

1. **Classify the input** — a DOI (`10.xxxx/…`), a URL, an arXiv id (`2301.12345`), or an
   OpenAlex id (`W…`). Anything matching none of those patterns is treated as a DOI, so a malformed
   id surfaces as "not found" rather than as a crash.
2. **Fetch + enrich metadata** (unless a bare URL). OpenAlex / arXiv supply the record; the WoS
   Starter API contributes its **accession number** (`wos_id`) when the DOI is indexed there. The
   impact figure is OpenAlex's `2yr_mean_citedness` **estimate**, *not* an official WoS JIF, and venue
   quality is then filled in from `journal_tier` / `scimago_quartile` where derivable — see the
   [metrics glossary](search.md#metrics-glossary).
3. **Fetch the full text, cheapest route first:**
   - *arXiv id* → direct PDF download (most reliable); if that fails, the browser over
     `https://arxiv.org/abs/<id>`.
   - *anything else* → when OpenAlex marked the work open access, **try its `oa_url` with a plain
     HTTP GET first**. No browser is launched. The response must begin with the `%PDF-` magic bytes
     (`Content-Type` is not trusted: repositories often answer `application/octet-stream`, and
     publisher "you need to log in" pages often answer `200 + text/html`), so a landing page is
     **rejected rather than saved**. The download goes to a `.part` file and is renamed atomically,
     because an interrupted write would otherwise leave a truncated `.pdf` that the next run would
     happily reuse as a cache hit. The filename derives from the DOI / OpenAlex id, not the URL, so
     the same paper reached through a different mirror reuses the same file.
   - *only if that fails* → a headless browser grabs the publisher page, the article PDF, and any
     supplementary material.
4. **Extract to Markdown** — MinerU cloud by default (equations → LaTeX, tables → HTML);
   `pymupdf4llm` is the local fallback (equations lost). Override with `--backend`. When the paper has
   an arXiv id — whether you passed one or the metadata carries `arxiv_id` — extraction goes through
   the **arXiv LaTeX source** instead of the PDF, so equations come out as real LaTeX rather than as
   fragments recognized off the rendered page.
5. **Write outputs** — the extracted full text to `cache/extracted/<slug>_fulltext.md` (the single
   canonical extracted artifact — `read` extracts with `write_cache=False` so no duplicate is
   left behind), and a structured note skeleton to `papers/{year}_{author}_{slug}.md`. The full
   text's path is recorded in the note's `extracted_md_path` field and the PDF's in `local_pdf_path`.

> **Cache reuse (default):** if `cache/extracted/<slug>_fulltext.md` already exists, `read` loads it
> and **skips the fetch + extraction entirely** (printing `[read] 命中缓存全文，跳过抓取/抽取`). A
> previously downloaded OA PDF is likewise reused (and its mtime bumped, which is what `cache prune`
> uses for LRU ordering). The browser-fetch layer reuses a cached bundle by URL. Pass `--refresh` to
> re-fetch and re-extract. Expensive artifacts are kept permanently; see
> `references/maintenance.md` §5 for the two-tier cache and `research cache stats｜clean｜prune`.

It prints the paths. **Read the `全文 MD` file to actually read the paper**, then fill in the
note's evaluation sections (rubric below).

### When no PDF can be had: the HTML fallback

If neither route yields a PDF — or extraction produced nothing from one — but the browser did get an
article page, `read` falls back to the **web text** and records it in a *different* field:

- The content is `cache/html_fulltext/<slug>/<slug>.md`: markdown extracted by `trafilatura`, **not**
  raw HTML, carrying `browser_fetch`'s own provenance head.
- It is recorded as **`extracted_html_path`**, and `extracted_md_path` is deliberately left empty.
- It therefore stays **out of the RAG corpus** — `rag` indexes only `cache/extracted/**/*.md`. Mixing
  in web text of a different provenance and quality, where equations are usually gone (they are
  images or SVG), would quietly degrade every later retrieval. The file is *not* copied into
  `cache/extracted/` either: one copy, one location, provenance intact.
- `read` says all of this on stderr, and the file is still protected from
  `cache prune --keep-referenced` (the field is in `cache_manager.REFERENCED_FIELDS`). Note that
  `prune` treats a whole `html_fulltext/<slug>/` **directory** as one unit, so keeping the article
  text also keeps the supplementary material fetched alongside it.

Treat it as a last resort for reading prose, **not** as an extraction. If you need the equations, get
the PDF and re-run with `--refresh`, or come in through the arXiv id so the LaTeX-source path applies.

**Exit code:** `0` when anything at all was produced (markdown, an HTML fallback, or a note), `1`
when nothing was, `2` when an OpenAlex id could not be resolved to a fetchable URL.

### Flags & choices

- `--headed` — run the browser with a visible window. Use when a page is Cloudflare-gated or needs
  an institutional login; the headed browser can pass challenges the headless one cannot.
- `--no-note` — only fetch + extract full text; do not create/modify a note.
- `--overwrite` — rebuild an existing note **from the template**, resetting the body. Without it an
  existing note is *merged*, never clobbered — see the merge contract below.
- `--backend pymupdf4llm` — fast local extraction when you don't need equations, or when MinerU
  quota is exhausted.
- `--refresh` — ignore the cached full text, cached OA PDF and cached HTML bundle, and re-fetch +
  re-extract from scratch. Use when the previous extraction was truncated/garbled, or the publisher
  page has been fixed. Without it, a cached `<slug>_fulltext.md` short-circuits the whole pipeline.
  (`--force` still works as a **deprecated alias** and prints a migration hint on stderr. It was
  renamed because one word, `--force`, had four unrelated meanings across four subcommands.)

### DOI vs arXiv input

- **DOI / OpenAlex id** → richest metadata (citation counts, normalized citations, JIF estimate,
  volume/issue/pages, OA link). Prefer this for published papers.
- **arXiv id** → guaranteed open PDF, but frontmatter has **no citation count** and uses the
  preprint title. The venue is now parsed out of `journal_ref` (`journal: Phys. Rev. Lett.` rather
  than the whole citation string), and when a DOI is present the journal metrics are looked up
  separately — so a published paper reached through its arXiv id is no longer left with `jif: null`.
  Still, for a published paper, `read` the DOI if you have it.

---

## How notes are written: the merge contract

`read` and `add` both write through the same `_merge_note()`, which reports one of four actions. The
distinction matters: the previous writer was all-or-nothing, so on a note that already existed it
**returned without writing anything** — which is how `local_pdf_path` and `extracted_md_path` went
missing from every note built by the normal `add`-then-`read` workflow, and with them the
`cache prune --keep-referenced` protection that reads those two fields.

| Action | When | What happens to the file |
|---|---|---|
| `created` | no note yet | written from `templates/paper_note.md` |
| `merged` | note exists and the new data fills at least one **empty** key | frontmatter re-rendered; **body preserved byte for byte**; one audit line appended under `## Changelog` |
| `unchanged` | note exists, nothing empty left to fill | **not touched at all** — mtime preserved, so `cache prune`'s LRU ordering isn't disturbed |
| `overwritten` | `--overwrite` | rebuilt from the template; the body is reset |

The rules that make this safe to run repeatedly:

- **Only empty keys are filled.** Missing, `null`, `""` and `[]` all count as empty. Your `my_rating`,
  `status`, `related_to_my_work` and every prose section are never replaced by a machine value, so
  re-running cannot downgrade a note you have already evaluated.
- **The body is byte-identical** after a `merged` write. The only body edit is the appended Changelog
  line (`- 2026-09-30: 补齐字段 local_pdf_path, extracted_md_path`), which makes the change auditable
  rather than silent.
- **Unparseable frontmatter is never rewritten.** A hand-written or corrupted note is reported and
  left alone (`unchanged`); pass `--overwrite` explicitly if you truly want the template back.
- The report separates `merged` (naming the keys it filled) from `unchanged` ("nothing to fill, file
  untouched"), so neither can be misread as a failed write.

> **One thing merging cannot do: correct a wrong value.** It fills blanks; it never overwrites. Notes
> written before a field's semantics were fixed therefore keep the old value — re-running `read` will
> not repair it, and `index --fix` normalizes *shape* (field order, block style, missing keys) while
> likewise preserving existing values. Fix such a value by hand.

---

## `add` — put a paper in the library

```
research add <doi｜arxiv-id｜openalex-id> [--tags t1,t2] [--overwrite]
             [--verify｜--no-verify] [--allow-fail]
```
Resolves + enriches metadata, **creates a Zotero item** when `zotero-cli` (community zotero-mcp) is
installed — it normalizes the frontmatter to BibTeX and delegates the write to `zotero-cli add
bibtex` — then writes the returned `zotero_key` + `zotero_uri` into a new `papers/` note skeleton,
and does **not** fetch full text. If `zotero-cli` is unavailable, `add` degrades to a skeleton-only
note (no library write). Use `add` to build the library quickly; use `read` when you intend to
read the paper. Check `library search` first to avoid duplicating an existing Zotero entry. The note
is written under the same [merge contract](#how-notes-are-written-the-merge-contract), so `add`-ing a
paper you later `read` fills in the paths instead of discarding them.

Before writing, `add` runs the **citation integrity gate** (`citecheck` below): a three-source
**FAIL** blocks the Zotero write + note. `--no-verify` skips the check entirely; `--allow-fail`
writes even on FAIL.

> `--allow-fail` was formerly spelled `--force`, which still works as a deprecated alias. The rename
> *is* the point: this flag opens a **safety gate**, and sharing a name with an ordinary cache-refresh
> switch made it far too easy to reach for without realizing what it did.

---

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
  `reviews/*.md` and any external reference you have made.
- `--dry-run` — with `--fix` only: report what would change and write **nothing at all**, `INDEX.md`
  included. (Given without `--fix` it is ignored, with a notice.) On a real library always run this
  first, then confirm `git diff data/skills/literature_research/papers/` shows frontmatter only.
- `--force` — rebuild `INDEX.md` even when `papers/` is empty; by default an existing index is kept,
  since an empty `papers/` far more often means a wrong working directory than an empty library. This
  is the one `--force` that kept its name, because here it really does mean "force the write".

---

## `review` — survey scaffolding, link integrity, count sync

```
research review new <topic> [--purpose '<goal>'] [--time-window '<window>']
                    [--from-shortlist <f1> <f2> …]
research review status [<file>]
research review sync   [<file>] [--dry-run]
```

**There is deliberately no survey generator.** §1–§5 of `templates/review_note.md` (field overview,
key threads, timeline narrative, contested claims, interface with your own work) are the LLM's core
judgment; a template-filler would turn live judgment into dead boilerplate. `review` does only the
three *mechanical* jobs around it.

### `new`

Instantiates `templates/review_note.md` into `reviews/{YYYY-MM}_{topic_slug}_survey.md`, filling the
frontmatter's `topic` / `topic_slug` / `created_date` / `last_updated` / `author` / `status: draft` /
`time_window`, and substituting `--purpose` into the body's 综述目标 line — **only** that line, and if
the template has been edited so the line cannot be found it says so instead of silently dropping your
purpose. An existing file is never overwritten (exit **2**).

`--from-shortlist` aggregates `sources_used` and `query_strings` by **reading the snapshots'
frontmatter** instead of asking you to retype the search strings: a retyped query always drifts, while
the snapshot holds the one that was actually submitted. It accepts full paths or a filename fragment
matching uniquely under `shortlists/`; a snapshot that cannot be found or parsed is reported and
skipped, and the scaffold is still produced.

`inclusion_criteria` and `exclusion_criteria` are left **empty on purpose** — they are a
methodological judgment, not metadata. `new` then stops and prints the next step: write §1–§5, list
the notes you cover as `[[file-name]]` in `papers_reviewed`, and run `review sync`.

### `status`

Read-only audit of one file, or of every `reviews/*.md` when no target is given:

- every `[[wiki-link]]` in `papers_reviewed` must resolve to an existing `papers/*.md`. Aliases
  (`[[x|Zhang 2026]]`) and anchors (`[[x#§2]]`) are handled, and a bare stem with no brackets is
  accepted because hand-written surveys use it;
- `papers_total_count` / `papers_read_count` are compared against reality — the number of resolvable
  links, and of those how many have `status != "unread"` — with any drift printed as
  `papers_read_count 3 → 应为 7`.

**Exit 1** on a broken link, and also on a file that *could not be audited* (unparseable frontmatter):
a survey that was never actually checked passing a gate silently is more dangerous than an error.
Exit **2** if the named file doesn't exist. An empty `reviews/` is a normal state — one line, exit 0.
This is the same CI-friendly contract as `index --check` and `citecheck`.

### `sync`

Recomputes the canonical `papers_reviewed` list, both counts and `last_updated`, then rewrites **only
the frontmatter** — the body stays byte-identical. It appends **no** Changelog line: that log records
a survey's intellectual progress (drafting, revising, adding evidence), and a mechanical count refresh
that can happen many times a day would drown the entries worth reading.

`unchanged` leaves the file untouched (mtime preserved), exactly like the note merge contract.
`--dry-run` reports which keys *would* be rewritten and writes nothing. Broken links are reported on
stderr but **kept** — `sync` fixes counts, it does not guess at your intent. A file that cannot be
audited is skipped with exit **1** rather than rewritten.

---

## The paper note: schema + evaluation rubric

Each `papers/*.md` = YAML frontmatter (machine-filled) + a body (AI/user-filled) from
`templates/paper_note.md`. Frontmatter is auto-populated by `read`/`add`; you fill the body.

### Frontmatter groups (auto-filled — verify, don't hand-edit unless wrong)
- **Identity**: title, short_title, authors, first_author_last_name, corresponding_author, year,
  publication_date, journal, publisher, volume/issue/pages, doi, arxiv_id, openalex_id, wos_id.
- **Links**: zotero_key, zotero_uri, local_pdf_path, extracted_md_path, extracted_html_path, oa_url,
  oa_status. The last two path fields are mutually exclusive in practice: `extracted_md_path` is a
  real PDF extraction and **is** in the RAG corpus, `extracted_html_path` is the web-text fallback and
  is **not** (see the HTML fallback above).
- **Quality/impact**: cited_by_count, cited_by_count_normalized, jif, jif_5yr, jcr_quartile,
  scimago_quartile, citescore, esi_highly_cited, esi_hot_paper, journal_h_index, journal_tier,
  journal_tier_basis, listed_in. Per-field semantics and sources:
  [search.md § Metrics glossary](search.md#metrics-glossary). Two of them are *structural*, not gaps:
  `jcr_quartile` stays `""` until the WoS Journals API lands, and `esi_*` stay **`null`** meaning
  *unknown* — never `false`, which would assert something the data cannot know.
- **Classification (you fill)**: topics, methods, systems, related_to_my_work (high/medium/low/none),
  related_to_my_work_reason, keywords_auto.
- **Status (you fill)**: status (unread/reading/read/archived/rejected), my_rating (1–5),
  added_date, last_reviewed, review_count.

### Body sections — how a physicist should fill them

Read the extracted full text first, then complete:

- **一句话定位 / TLDR** — ≤150 words: what the paper does and at what level it does it well or
  poorly. Write from abstract + last intro paragraph + conclusion.
- **Key Claims** — the 2–4 load-bearing assertions. Phrase them so each is *falsifiable*.
- **Method Summary** — model/theoretical framework; key assumptions; approximations/simplifications;
  numerical or experimental means; **reusability** (can I lift this method onto my own system?).
- **Main Results** — core result; key figures (number + one-line takeaway); the 1–3 governing
  equations in the ```` ```latex ```` block.
- **Novelty Assessment** — *What's new*; *compared to prior work* (name the 1–3 closest papers and
  the delta); *field context* (leading / following / gap-filling?); **Novelty score 1–5 + reason**.
- **Rigor Assessment** — *assumptions validity* (do they hold under the stated experimental /
  numerical conditions?); *potential weaknesses* (the holes you'd raise in review);
  *reproducibility* (enough detail? supplementary complete? code/data public?); **Rigor score 1–5**.
- **Journal-tier Justification** — given the venue's JIF estimate / quartile / `journal_tier`, mark
  **Over-claimed / Matched / Under-placed** and say why. This calibrates "is the venue commensurate
  with the substance?". Where `journal_tier` and `jif` disagree — common in low-citation-density
  fields, where the expert-panel grade is the better signal — say which one you weighted and why.
- **Relevance to My Research** — *direct borrow* (method/model/formula/setup I can reuse);
  *indirect inspiration* (transferable physical picture); *comparison needed* (must my future work
  distinguish itself from this?); *citation intent* (which section I'd cite it in).
- **Related Papers** — internal links `[[note-name]]` to other `papers/` notes.
- **Quoted Excerpts** — verbatim quotes with `(Sec. X, p. Y)` / `(Fig. Z caption)`.
- **Follow-up Questions** — open items to check later.

### Field-specific judgment cues (non-Hermitian / acoustic metamaterials)

When scoring novelty and rigor in this domain, watch for: whether gain/loss is treated rigorously
(a genuine non-Hermitian Hamiltonian vs a lossy Hermitian one in disguise); whether an claimed
exceptional point / degeneracy is verified by **both** eigenvalue coalescence **and** eigenvector
collapse (not just a transmission dip); whether topological claims state the symmetry class and the
invariant; whether active (gain) experiments report the lasing/instability threshold and
signal-to-noise; whether coupled-mode / transfer-matrix models are validated against FEM or
experiment rather than asserted; and whether the metric improvements are benchmarked against the
correct prior art rather than a strawman.

### Scoring anchors (use consistently)

- **5** — field-defining; a result/method others will build on for years.
- **4** — strong, clearly advances the subfield; minor gaps.
- **3** — solid, incremental; correct but not surprising.
- **2** — weak; over-claimed, thin validation, or niche.
- **1** — flawed or trivial; do not cite except as a cautionary contrast.

Set `status` (unread → reading → read), `my_rating`, and `last_reviewed` as you go; then run
`research index`.
