---
name: document-writing
description: Create, edit, compile, and proofread LaTeX manuscripts (revtex4-2 / article / Beamer) to journal-submission-grade PDF; read, create, and edit PPTX slides and DOCX documents; extract PPTX/DOCX/PDF to Markdown; digest a huge image-heavy .pptx into image-text-linked Markdown for LLM reading; convert Office files to PDF via LibreOffice. Drives a unified `compose` CLI that scaffolds LaTeX projects from templates, compiles via latexmk with file:line error parsing, lints with chktex, refreshes refs.bib from Zotero, renders PDF pages to PNG for visual proofreading, structurally reads/writes .pptx and .docx, and builds decks/docs from Markdown outlines. Use when the user asks to write / draft / revise / compile a paper or report in LaTeX, produce or proofread a PDF, read / summarize / create / edit / translate a .pptx or .docx, extract slides or speaker notes, digest a large deck, build slides from an outline, or refresh a bibliography from Zotero.
---

# Document Writing

A single CLI drives the whole writing workflow: **scaffold → write → compile → proofread → references**.

The backend in `src/pysci/skills/document_writing/` is a **black box** — drive it through the `compose`
CLI, open code only when maintaining it ([maintenance.md](references/maintenance.md)).

## Invocation

From the **project root**: `uv run pysci-compose <command> [options]`, below `compose …` (fallback:
`uv run python -m pysci.skills.document_writing.tools.compose …`). This CLI prints Chinese and takes
multi-word titles — apply `.qoder/rules/basic.md` §3 or you read mojibake and broken commands.

## Capability boundary

- LaTeX manuscript authoring → compiled PDF (revtex4-2 / generic article / **Beamer slides**); compile
  with parsed `file:line` errors; chktex lint; PDF→PNG visual proofreading.
- **Read/extract** `.pptx` (per-slide text, bullets, tables, images, speaker notes), `.docx`
  (headings / paragraphs / bullets / tables), `.pdf` and other Office formats → Markdown.
- **Create/edit** `.pptx` and `.docx` via python-pptx / python-docx; **Markdown→pptx/docx** via
  **Pandoc** (`--reference-doc` for house style).
- Refresh `refs.bib` from the user's Zotero library.
- `convert` (pptx/docx ↔ pdf/…) via LibreOffice headless — **only when LibreOffice is installed**.
  Non-standard install path? Set `DOCWRITING_SOFFICE` in `.env` to the absolute path of `soffice.exe`;
  it takes precedence over auto-detection. If `convert` reports LibreOffice missing, fall back to the
  always-available routes (read/extract, or LaTeX→PDF) instead of failing the task.

## Commands

| Group | Commands | For |
|---|---|---|
| health | `doctor` | TeX / LibreOffice / Pandoc / Python-lib self-check |
| LaTeX | `tex new <slug>` · `tex build <target>` · `tex lint <target>` · `tex refs <target>` | scaffold `projects/<slug>/`; compile → PDF + parsed `file:line`; chktex; regenerate `refs.bib` from Zotero |
| read | `read <file>` · `slides extract <pptx>` · `docx read <file>` | fast flat extract of any doc → Markdown; deep per-slide read (notes / tables / images); structured Word read |
| write | `slides new｜add` · `slides from-markdown` · `docx add` · `docx from-markdown` | incremental edits (python-pptx / docx); outline → deck / doc (Pandoc) |
| digest | `slides digest <pptx> [--out --render --lint]` | huge image-heavy deck → image-text-linked workspace |
| deliver | `convert <file> --to pdf` · `verify <pdf>` | Office → PDF (needs LibreOffice); PNG pages for you to Read |

Full flags: `compose <command> -h` (and `compose tex <sub> -h`), or
[latex.md](references/latex.md) / [read.md](references/read.md).

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

Loop steps 3–6 until the PDF looks right. **Always `verify` + Read the PNGs before declaring done** —
LaTeX compiles cleanly even when the layout is wrong (overfull boxes, floats adrift). If `doctor`
reports TeX missing, follow its printed TUNA install guide; reading and extracting (`read`,
`slides extract`, `verify`) works without TeX.

## Office documents: read, digest, write

Three routes; commands, flags and conventions in [read.md](references/read.md).

- **Read a deck or doc.** `slides extract 'deck.pptx' --with-images` for structure (notes, tables,
  bullet hierarchy, embedded images); `read 'file.docx' --preview` for a fast flat dump of anything —
  re-runs reuse the cache, `--force` re-extracts. Then **Read** the printed Markdown path.
- **Digest a huge, image-heavy deck you must *understand*.** `slides digest 'huge.pptx' --out
  'translated/' --render` builds an **image-text-linked** workspace: each image stays inline beside its
  page's text with a `解读：_(待填)_` slot you fill **once** and reuse (sha1-deduped) — the fix for
  bulk extraction severing the image↔text link. Phase A is pure code (skeletons, `index.md`,
  `progress.json`); Phase B is you, batch by batch, Reading each figure slide's whole-page render
  **once**. Resumable: re-running without `--force` never overwrites filled interpretations. The
  interpretation loop, edit anchors and narrative finish: [digest.md](references/digest.md).
- **Write.** `slides|docx from-markdown` (Pandoc — headings, nested lists, pipe tables, math and images
  map faithfully; `--slide-level` picks the slide heading, `--reference-doc` applies house style,
  speaker notes go in a fenced `::: notes` div) for a **fresh** file; `slides new|add` / `docx add`
  (python-pptx / docx) to append to an existing one — **Pandoc cannot append**.

## Output locations

| Path | Contents |
|---|---|
| `data/skills/document_writing/projects/<slug>/` | one writing project (`main.tex`, `refs.bib`, `figures/`, `build/`) |
| `data/skills/document_writing/templates/latex/` | scaffold templates (`revtex/`, `article/`) |
| `data/skills/document_writing/cache/extracted/` | Markdown from `read` / `slides extract` (git-ignored) |
| `data/skills/document_writing/cache/digests/<deck>/` | default `slides digest` workspace (override with `--out`; `images/` + `renders/` git-ignored) |
| `data/skills/document_writing/cache/renders/` | PNG pages from `verify` (git-ignored) |

## When something breaks

`compose doctor` first — it reports TeX, engines, LibreOffice, Pandoc and every Python lib. Compile
errors are already parsed to `file:line`; open that line in `main.tex`. Backend / template / config:
[maintenance.md](references/maintenance.md).

## Reference files

- [latex.md](references/latex.md) — the LaTeX workflow in depth: templates (revtex / article / **beamer**), engines, the compile→verify loop, bibliography handling, manuscript-quality guidance, lint.
- [read.md](references/read.md) — reading **and writing** pptx / docx / pdf: backends, caches, what each extractor preserves, Markdown outline conventions.
- [digest.md](references/digest.md) — the `slides digest` interpretation workflow: per-figure loop, SearchReplace anchors, duplicate-figure reuse, resumable status ledger, batch cadence, narrative-review finish.
- [maintenance.md](references/maintenance.md) — how the backend works and how to fix or extend it (config, log parsing, adding templates, LibreOffice conversion).
