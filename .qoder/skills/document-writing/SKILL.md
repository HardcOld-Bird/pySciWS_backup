---
name: document-writing
description: Create, edit, compile, and proofread LaTeX manuscripts (revtex4-2 / article / Beamer) to journal-submission-grade PDF; read, create, and edit PPTX slides and DOCX documents; extract PPTX/DOCX/PDF to Markdown; digest a huge image-heavy .pptx into image-text-linked Markdown for LLM reading; convert Office files to PDF via LibreOffice. Drives a unified `compose` CLI that scaffolds LaTeX projects from templates, compiles via latexmk with file:line error parsing, lints with chktex, refreshes refs.bib from Zotero, renders PDF pages to PNG for visual proofreading, structurally reads/writes .pptx (slides, tables, speaker notes) and .docx (headings, bullets, tables), and builds decks/docs from Markdown outlines. Use when the user asks to write / draft / revise / compile a paper or report in LaTeX, produce or proofread a PDF, read / summarize / create / edit / translate a .pptx or .docx, extract slides or speaker notes, digest a large deck, build slides from an outline, or refresh a bibliography from Zotero.
---

# Document Writing

A single CLI drives the whole writing workflow: **scaffold → write → compile → proofread → references**.

The backend lives in `src/pysci/skills/document_writing/` (a set of modules + a `compose` facade).
**You do not need to read the backend code** — treat it as a black box and drive everything
through the `compose` CLI below. Only open the code when maintaining it (see
[references/maintenance.md](references/maintenance.md)).

## Invocation

Run from the **project root**. Define this once per session:

```
.venv\Scripts\python.exe -m pysci.skills.document_writing.tools.compose <command> [options]
```

Below, `compose …` is shorthand for that full invocation. (`uv run python -m …` also works.)

> **PowerShell rule (critical):** wrap multi-word arguments in **single quotes**, e.g.
> `compose tex new my_paper --title 'Gain-induced Exceptional Points'`. Double quotes get
> split by the shell and break the command. Use `;` (never `&&`) to chain commands.

## Capability boundary (read this first)

**Available now:**
- LaTeX manuscript authoring → compiled PDF (revtex4-2 / generic article / **Beamer slides**).
- Compile with parsed `file:line` errors; chktex lint; PDF→PNG visual proofreading.
- **Read/extract** `.pptx` (per-slide text, bullets, tables, images, speaker notes), `.docx`
  (headings/paragraphs/bullets/tables), `.pdf`, and other Office formats → Markdown.
- **Create/edit** `.pptx` (new deck, append slides, Markdown→pptx) and `.docx` (new doc,
  append blocks, Markdown→docx).
- Refresh `refs.bib` from the user's Zotero library.
- `convert` (pptx/docx ↔ pdf/…) via LibreOffice headless — **only when LibreOffice is
  installed**; otherwise it degrades gracefully with install guidance. Custom install path
  (e.g., a non-standard D-drive location)? Set `DOCWRITING_SOFFICE` in `.env` to the absolute
  path of `soffice.exe`; it takes precedence over auto-detection.

If `convert` reports LibreOffice missing, fall back to the always-available routes
(read/extract, or LaTeX→PDF) instead of failing the task.

## Commands at a glance

| Command | Use when | Key output |
|---|---|---|
| `doctor` | Session start, or anything seems broken | TeX / LibreOffice / Python-lib self-check |
| `tex new <slug>` | Starting a manuscript | Scaffolded project in `projects/<slug>/` |
| `tex build <target>` | Compiling | PDF + parsed errors (`file:line`), warnings, boxes |
| `tex lint <target>` | Static check before submitting | chktex issues |
| `tex refs <target>` | Refreshing the bibliography | `refs.bib` regenerated from Zotero |
| `read <file>` | Quick extract of any doc (pptx/docx/pdf/xlsx/…) | Markdown in `cache/extracted/` |
| `slides extract <pptx>` | Deep-reading a slide deck | Per-slide Markdown + notes/tables/images |
| `slides new / add / from-markdown` | Creating or extending a deck | New / updated `.pptx` |
| `slides digest <pptx>` | Translating a huge, image-heavy deck you must *understand* | Image-text-linked `index.md` + `part_*.md` + deduped `images/` |
| `docx read / from-markdown / add` | Reading or producing Word docs | Markdown / `.docx` |
| `convert <file> --to pdf` | Delivering a PDF of an Office file | Converted file (needs LibreOffice) |
| `verify <pdf>` | Proofreading the compiled layout | PNG pages for you to Read |

Run `compose <command> -h` (and `compose tex <sub> -h`) for the full option list.

## Quick start: zero → submission-grade PDF

```
- [ ] 1. compose doctor                                   # confirm TeX is installed
- [ ] 2. compose tex new my_paper --template revtex --title '<Title>'
- [ ] 3. Edit projects/my_paper/main.tex                  # write the manuscript
- [ ] 4. compose tex refs my_paper --query '<topic>'      # refresh refs.bib from Zotero
- [ ] 5. compose tex build my_paper                       # compile; fix reported file:line errors
- [ ] 6. compose verify projects/my_paper/build/main.pdf  # render PNG, Read them to check layout
- [ ] 7. compose tex lint my_paper                        # final static check
```

Loop steps 3–6 until the PDF looks right. **Always `verify` + Read the PNGs before declaring
done** — LaTeX compiles cleanly even when the layout is wrong (overfull boxes, floats adrift).

If `doctor` reports TeX missing, follow its printed TUNA install guide; reading/extracting
(`read`, `slides extract`, `verify`) works without TeX.

## Reading a slide deck or document

**Structured slide read** (preferred for `.pptx` — keeps notes/tables/image list):
```
compose slides extract 'path/to/deck.pptx' --with-images
```
Then **Read** the printed Markdown path. Speaker notes appear as `> **演讲者备注：** …`;
tables become Markdown tables; `--with-images` dumps embedded images so you can Read them too.

**Fast generic read** (any Office/PDF file → Markdown):
```
compose read 'path/to/file.docx' --preview
```
Re-running reuses the cache; add `--force` to re-extract.

## Translating a huge, image-heavy deck

For a big deck (dozens–hundreds of slides, more images than text) that you must *understand* —
not just extract — use `slides digest`. It builds an **image-text-linked Markdown workspace** so
each picture stays bound to its surrounding text (never split into a separate folder), and every
image gets a `解读` slot you fill in **once** and reuse forever. This is the fix for the classic
failure where bulk-extracting images severs the image↔text link.

```
compose slides digest 'path/to/huge.pptx' --out 'path/to/translated' --render
```

Two phases:
- **Phase A (this command; no LLM vision needed):** dedupes (sha1) + exports every image to
  `images/`, detects logical sections, packs them into `part_*.md` chunks, infers each image's
  nearby caption text, draws an ASCII layout map per slide, and leaves a `解读：_(待填)_` slot
  under each image plus a whole-page render link. Unreadable formats are pre-handled: **gif**
  animations get first/mid/last frames extracted to `images/previews/`; **wmf/emf** vectors get a
  PNG preview via LibreOffice. Writes `progress.json` (a resumable ledger) + `index.md` (nav).
- **Phase B (you, in small batches):** for each figure slide, Read its whole-page composite render
  `renders/slide-NNN.png` **once** (sees all images + text in place), then fill each image's `解读`.
  Default batch suggestion: **5 figure slides / 15 text slides**. Duplicate images just point back
  to their first occurrence — interpret once. Re-running **without** `--force` resumes and never
  overwrites filled interpretations. Health-check links anytime with `--lint` (missing renders /
  vector previews are reported as *pending*, not broken).

`--render` needs LibreOffice (see Capability boundary); without it Phase A still completes and
records renders as pending — add `--render` later. If every slide shares one layout (auto-sectioning
finds nothing), define sections by hand: `--section-at 12,40,77,…`. Finish with a narrative review
md that condenses the whole deck into prose, linking back to each part and key figure.

For the full interpretation workflow — the per-figure loop, SearchReplace anchor technique,
duplicate-figure handling, a resumable status-ledger pattern, batch cadence, and the narrative
finish — see [references/digest.md](references/digest.md).

## Creating & editing slides / Word docs

**Build a deck from a Markdown outline** (the round-trip partner of `slides extract`):
```
compose slides from-markdown 'outline.md' --out 'deck.pptx'
```
Or incrementally: `compose slides new 'deck.pptx' --title '…'` then
`compose slides add 'deck.pptx' --title '…' --bullet '…' --notes '…'`.

**Produce a Word doc** the same way:
```
compose docx from-markdown 'draft.md' --out 'report.docx'
compose docx read 'report.docx' --preview     # structured read-back
```
Markdown conventions (`#`/`##` headings, `-` bullets, `>` notes, pipe tables) are in
[references/read.md](references/read.md).

## Output locations

| Path | Contents |
|---|---|
| `data/skills/document_writing/projects/<slug>/` | One writing project (`main.tex`, `refs.bib`, `figures/`, `build/`) |
| `data/skills/document_writing/templates/latex/` | Scaffold templates (`revtex/`, `article/`) |
| `data/skills/document_writing/cache/extracted/` | Extracted Markdown from `read` / `slides extract` (git-ignored) |
| `data/skills/document_writing/cache/digests/<deck>/` | Default `slides digest` workspace (override with `--out`; `images/` + `renders/` git-ignored) |
| `data/skills/document_writing/cache/renders/` | PNG pages from `verify` (git-ignored) |

## When something breaks

1. Run `compose doctor` — it reports TeX, engines, LibreOffice, and every Python lib.
2. Compile errors are already parsed to `file:line`; open that line in `main.tex`.
3. For backend / template / config issues, see [references/maintenance.md](references/maintenance.md).

## Reference files

- [references/latex.md](references/latex.md) — the LaTeX workflow in depth: templates
  (revtex / article / **beamer**), engines, the compile→verify loop, bibliography handling,
  and manuscript-quality guidance.
- [references/read.md](references/read.md) — reading **and writing** pptx / docx / pdf:
  backends, caches, what each extractor preserves, and the Markdown outline conventions.
- [references/digest.md](references/digest.md) — the `slides digest` interpretation workflow in
  depth: the per-figure loop, SearchReplace anchors, duplicate-figure reuse, the resumable
  status-ledger pattern, batch cadence, and the narrative-review finish.
- [references/maintenance.md](references/maintenance.md) — how the backend works and how to fix
  or extend it (config, log parsing, adding templates, LibreOffice conversion).
