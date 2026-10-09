# digest interpretation workflow — fill the skeleton into image-text-linked, LLM-readable md

Applies to: any translation workspace that has already run
`compose slides digest <pptx> --out <dir> --render` (**Phase A** done: `renders/` complete,
`--lint` broken=0 and pending=0). This guide describes **Phase B** (per-figure interpretation) as a
**reusable** process — independent of *which* deck, *what* subject, or *how many* pages. Any
image-heavy slide deck can be translated by following it.

Goal: for each **figure slide**, Read its **whole-page composite render** and fill in a `解读`
(interpretation) for every image, so the image↔text link is never severed; then finish with a
**narrative review** long-form md. The core values are **reviewable + resumable + as automated as
possible**.

> Terminology: **Phase A** = the `slides digest` command auto-emits the skeleton (pure code, no
> vision); **Phase B** = you / the LLM fill each `解读`. These two phases are the digest tool's
> inherent split — see SKILL.md and [read.md](read.md).

---

## 0. Read this first when resuming (after context compression / a new session)

1. If you maintain a **status ledger** for the workspace (see §6, usually a `STATUS.md` in it),
   **read it first** — it is the "memory anchor" for resuming: progress table, glossary, next up,
   decision notes.
2. Read §2–§5 of this guide (the loop and the quality bar).
3. **Progress ground truth = the number of remaining `_(待填)_` occurrences across all `part_*.md`**
   (see §5); do not rely on fragile intermediate state.
4. Continue from the ledger's *next up*; do not start over.
5. If the project has background docs (paper / report / spec), you may register one in the ledger as
   a "global context reference" (usually **read only its opening** to get the big picture; the body is
   mostly implementation detail); use it to calibrate terms and interpretations, but **do not copy it
   verbatim**.
6. Edits may be in a **pending-confirmation state** (in the harness your edits are committed to disk
   manually by the user at end of turn): after editing this turn, a Read may not show your change and
   SearchReplace may occasionally fail to match — both are expected. Trust **this turn's own edit
   ledger**; **do not re-write because a read-back is missing**; the on-disk placeholder count can lag
   by one turn.

## 1. Pre-checks

- Every figure slide has a `renders/slide-NNN.png` (count == the "figure slides" count in index.md).
- `compose slides digest <pptx> --out <dir> --lint` → **broken=0** (pending should also be 0).
- If renders are missing: `... --render` (reuse path — only tops up renders, never touches filled
  interpretations).

## 2. The per-figure-slide loop (core action, repeat)

For a figure slide `Slide NNN`:

- **a. Look at the whole page**: Read `renders/slide-NNN.png`. The point is to see the **true relative
  layout** — which image sits beside which text, and the order/grouping of images.
- **b. Get the skeleton**: Read that page's section in `part_*.md` (`### Slide NNN` up to the next
  `---`), giving each `图 K` (figure K) block's: index/position, image path, nearby text, and the
  page-head `要点/正文` (bullets/body).
- **c. Write the interpretation**: for each `图 K`, combining "what the render shows + nearby text +
  this page's bullets + whole-deck context", write the `解读` covering:
  1. **What** the image is (chart / curve / geometry sketch / photo / simulation field / circuit /
     flowchart …);
  2. **Axes, legend, key annotations, units, order of magnitude** (write them whenever present);
  3. The **role** it plays in this page's argument (evidence / illustration / result / comparison /
     apparatus photo);
  4. Links to **theory or other pages** (only when you can tell; do not fabricate).
  - Bar: **readable without the original ppt**. 2–6 sentences — specific, checkable, not vague.
- **d. Animations (gif)**: the whole-page render only holds the first frame; to understand the motion,
  also Read the extracted frames `images/previews/<stem>_f*of*_frame*.png` (first/mid/last).
- **e. Vector images (wmf/emf)**: Read `images/previews/<stem>.png` (already converted by LibreOffice).

## 3. Writing to disk (edit-anchor technique — must follow)

- Use SearchReplace to change `  - 解读：_(待填)_` into `  - 解读：<your interpretation>`.
- ⚠️ **`_(待填)_` repeats hundreds of times in the file** — never use it alone as original_text. You
  must include a **unique anchor**: start from the image's `  ![](images/sNNN_pK_hash.ext)` line
  (**globally unique**), through the intervening `邻近文字` (nearby-text) lines, all the way to
  `  - 解读：_(待填)_`; take that **whole span** as original_text, and let new_text change only the last
  (interpretation) line.
- ⚠️ **Never include trailing structural lines** (`\n\n---`, `\n\n### Slide …`, etc.) in original_text
  unless new_text reproduces them verbatim — otherwise you swallow the separator/heading and corrupt
  the file.
- **Multiple images on one page**: put several replacements in a single SearchReplace call (one per
  image).
- **Multiple pages**: batch within one turn — Read several renders → one SearchReplace with several
  replacements, to cut round-trips.
- For a gif block, anchor on its `[原文件](images/....gif)` line or the first preview-frame line
  (equally unique).

## 4. Duplicate figures (dedup table in index.md → "高频重复图")

- **First occurrence**: write the full interpretation.
- **Later occurrences**: write `同 Slide X 图 Y（<one line on why it is reused here>）`; do not repeat
  the long version.
- **Anchor collision**: several reuse pages may share the identical `重复图：…` line → anchoring on it
  alone matches multiple places. Extend the anchor to that page's `image` line + `邻近文字` line +
  `重复图` line + `解读` line as one block; if those blocks are **byte-identical** (a page-by-page
  reprint), you may use the `重复图`+`解读` two-line short anchor with `replace_all=true` to fill them
  all at once.
- Effect: each unique figure is interpreted once and every repeat only points back — saving context
  while keeping consistency.

## 5. Progress signal (robust, not JSON-based)

- **Remaining** = total `_(待填)_` occurrences across all `part_*.md` (count with Grep or a small
  script).
- ⚠️ **Counting pitfall**: if each `part_*.md` header carries one explanatory `_(待填)_` string (not a
  real placeholder), then **real placeholders = measured total − number of part files**. Prefer Grep
  (UTF-8) for counting CJK; PowerShell `Select-String` under-counts CJK, and counting commands should
  use ASCII variable names/labels (CJK label + colon interpolation errors out).
- Reaching **0** → Phase B core (per-figure interpretation) is done.
- After each batch: update the status ledger (progress table + next up + new terms); `progress.json`'s
  `done_slides` may optionally be updated (non-critical — lint ignores it).

## 6. Status ledger & batch cadence (the key to resumability)

- It is **recommended** to maintain a status-ledger md in the workspace (e.g. `STATUS.md`; name it
  yourself, just don't collide with tool outputs), recording:
  - **Progress table**: per part "interpreted / total figure slides" + a global *next up* (the next
    target Slide);
  - **Glossary / recurring motifs**: a one-line authoritative interpretation + first-occurrence Slide,
    to keep wording consistent across pages and turns;
  - **Decisions & notes**: anchor methodology, pitfalls and fixes, batch cadence — for fast recovery
    after a session switch.
  This makes resumption **lossless** after context compression or a new session — more robust than
  relying only on `progress.json`'s `done_slides`.
- **Batching**: during the validation phase, 3–5 figure slides/turn (confirm interpretation quality,
  anchors, and term consistency all look right); once stable, run as many as possible per turn (limited
  by context / vision tokens, empirically ~**8–12 figure slides/turn**). **Pure-text slides** (no
  images) need no vision and can be batched many at a time at near-zero cost — they only carry
  `要点/正文` prose and have **no `_(待填)_` slot**, so usually keep the faithful extraction as-is; do
  not rewrite the author's original prose.
- **Not self-triggering**: each turn still needs one user message to start; "automation" = do more
  within a turn + commit the ledger, so resuming is frictionless.
- Always commit the status ledger before ending a turn, guaranteeing lossless resumption after a
  session switch / compression.

## 7. Realistic assessment of sub-agent parallelism (don't fantasize)

- Generic sub-agents (e.g. Browser / CodeReview) **cannot read a local render PNG and write a
  structured md interpretation**, so the core "look-at-figure → interpret" step **cannot** be
  outsourced to parallel sub-agents.
- The only real "parallelism" is **batching within one turn**: Read several renders at once + one
  SearchReplace with several replacements (see §3).

## 8. Finish (after all interpretations are filled and `--lint` broken=0)

- Append a **narrative review** md (e.g. `NARRATIVE.md`): condense the whole deck into a through-line
  long-form (problem → method/theory → platform/implementation → key results → conclusion/outlook),
  linking back to each part and key figure.
- Back-fill index.md's "图像语义清单" (image semantic index): tag high-frequency/key images with a
  semantic label + occurrence pages, for future retrieval and reuse.
- Run `--lint` once more to confirm broken=0, and verify real remaining placeholders == 0.
