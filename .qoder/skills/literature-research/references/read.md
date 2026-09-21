# read / add / index — detailed reference + evaluation rubric

All commands are invoked as `research <cmd> …` (see SKILL.md for the full invocation and the
PowerShell single-quote rule).

---

## `read` — fetch full text + build a note

```
research read <doi｜url｜arxiv-id｜openalex-id>
              [--backend mineru-cloud｜pymupdf4llm｜auto]
              [--headed] [--no-note] [--overwrite] [--force]
```

What it does, in order:

1. **Classify the input** — a DOI (`10.xxxx/…`), a URL, an arXiv id (`2301.12345`), or an
   OpenAlex id (`W…`). A DOI is resolved through `https://doi.org/<doi>` (follows the redirect to
   the publisher); an arXiv id downloads the PDF directly (most reliable).
2. **Fetch metadata** (unless a bare URL) and enrich it (WoS JIF/JCR; S2 TLDR if a key exists).
3. **Fetch the full text** — arXiv: direct PDF download; otherwise a headless browser grabs the
   publisher page, the article PDF, and any supplementary material.
4. **Extract to Markdown** — MinerU cloud by default (equations → LaTeX, tables → HTML);
   `pymupdf4llm` is the local fallback (equations lost). Override with `--backend`.
5. **Write outputs** — the extracted full text to `cache/extracted/<slug>_fulltext.md` (the single
   canonical extracted artifact — `read` extracts with `write_cache=False` so no duplicate is
   left behind), and a structured note skeleton to `papers/{year}_{author}_{slug}.md`. Its path is
   recorded in the note's `extracted_md_path` frontmatter field.

> **Cache reuse (default):** if `cache/extracted/<slug>_fulltext.md` already exists, `read` loads it
> and **skips the fetch + extraction entirely** (printing `[read] 命中缓存全文，跳过抓取/抽取`). The
> browser-fetch layer also reuses a cached HTML/PDF bundle by URL. Pass `--force` to re-fetch and
> re-extract. Expensive artifacts are kept permanently; see `references/maintenance.md` §5 for the
> two-tier cache and `research cache stats｜clean｜prune`.

It prints the paths. **Read the `全文 MD` file to actually read the paper**, then fill in the
note's evaluation sections (rubric below).

### Flags & choices

- `--headed` — run the browser with a visible window. Use when a page is Cloudflare-gated or needs
  an institutional login; the headed browser can pass challenges the headless one cannot.
- `--no-note` — only fetch + extract full text; do not create/modify a note.
- `--overwrite` — regenerate an existing note. **By default an existing note is never clobbered**
  (your filled-in evaluation is preserved). Use `--overwrite` only to rebuild the skeleton.
- `--backend pymupdf4llm` — fast local extraction when you don't need equations, or when MinerU
  quota is exhausted.
- `--force` (alias `--refresh`) — ignore the cached full text and cached HTML bundle, and re-fetch +
  re-extract from scratch. Use when the previous extraction was truncated/garbled, or the publisher
  page has been fixed. Without it, a cached `<slug>_fulltext.md` short-circuits the whole pipeline.

### DOI vs arXiv input

- **DOI / OpenAlex id** → richest metadata (citation counts, normalized citations, JIF estimate,
  volume/issue/pages, OA link). Prefer this for published papers.
- **arXiv id** → guaranteed open PDF, but frontmatter has **no citation count / JIF** and uses the
  preprint title/journal-ref. Fine for preprints; for a published paper, `read` the DOI instead.

---

## `add` — put a paper in the library

```
research add <doi｜arxiv-id｜openalex-id> [--tags t1,t2] [--overwrite]
```
Resolves + enriches metadata, **creates a Zotero item** (if `ZOTERO_USER_ID`/`ZOTERO_API_KEY` are
set), writes the returned `zotero_key` + `zotero_uri` into a new `papers/` note skeleton, and does
**not** fetch full text. Use `add` to build the library quickly; use `read` when you intend to
read the paper. Check `library search` first to avoid duplicating an existing Zotero entry.

---

## `index` — rebuild the library index

```
research index [--check] [--force]
```
Scans `papers/*.md`, reads each note's frontmatter, and rewrites `INDEX.md` as a table sorted by
year (title, first author, journal, status, rating, link). `--check` reports drift without writing
(exit 1 if stale). Run it after adding or editing notes. If `papers/` is empty it will not clobber
an existing `INDEX.md` unless `--force` is given.

---

## The paper note: schema + evaluation rubric

Each `papers/*.md` = YAML frontmatter (machine-filled) + a body (AI/user-filled) from
`templates/paper_note.md`. Frontmatter is auto-populated by `read`/`add`; you fill the body.

### Frontmatter groups (auto-filled — verify, don't hand-edit unless wrong)
- **Identity**: title, short_title, authors, first_author_last_name, corresponding_author, year,
  publication_date, journal, publisher, volume/issue/pages, doi, arxiv_id, openalex_id, wos_id.
- **Links**: zotero_key, zotero_uri, local_pdf_path, extracted_md_path, oa_url, oa_status.
- **Quality/impact**: cited_by_count, cited_by_count_normalized, jif, jif_5yr, jcr_quartile,
  scimago_quartile, citescore, esi_highly_cited, esi_hot_paper, journal_h_index.
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
- **Journal-tier Justification** — given the journal's JIF/quartile, mark **Over-claimed / Matched /
  Under-placed** and say why. This calibrates "is the venue commensurate with the substance?".
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
