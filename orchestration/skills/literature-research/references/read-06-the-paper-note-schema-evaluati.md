# read 分册 6/6

> 含小节：The paper note: schema + evaluation rubric
> 原 `read.md` 超 8KB 预算拆分（2026-10-10，backlog 20261010-refs-split-8kb）；分册索引见同目录 `read.md`，按需只读所需分册。

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
