# Reading & extracting documents

Two entry points, both write Markdown you then **Read**:

| Command | Best for | Backend |
|---|---|---|
| `compose slides extract <pptx>` | Slide decks (structured) | `python-pptx` |
| `compose slides digest <pptx>` | Huge, image-heavy decks you must *understand* | `python-pptx` + Pillow (+ LibreOffice for renders) |
| `compose read <file>` | Any Office/PDF/web file (fast) | `markitdown` / `pymupdf4llm` |
| `compose slides new/add` | Incrementally creating / extending decks | `python-pptx` |
| `compose slides from-markdown` | Markdown outline → deck | Pandoc |
| `compose docx read/add` | Word docs (structured read & append) | `python-docx` |
| `compose docx from-markdown` | Markdown → Word doc | Pandoc |
| `compose convert <file> --to pdf` | Office → PDF delivery | LibreOffice headless |

## `slides extract` — structured PPTX read (preferred for decks)

```
compose slides extract 'deck.pptx' --with-images [--no-notes] [--out path.md] [--preview]
```

Per slide it preserves:
- **Title** and **layout name** (`## 第 N 页：<title>` + `_（版式：…）_`).
- **Bullet hierarchy** — nested levels are indented with `•` markers, so outline depth survives.
- **Tables** — converted to Markdown tables (`**表格 k：**` + pipe table).
- **Images** — listed with alt text; `--with-images` (or `--export-images <dir>`) dumps the
  embedded blobs so you can Read them as pictures.
- **Speaker notes** — as `> **演讲者备注：** …` (skip with `--no-notes`).
- Grouped shapes are recursed, so text inside groups is not lost.

Output: `cache/extracted/<stem>__slides.md` (or `--out`). Legacy binary `.ppt` is rejected with
a clear error (convert to `.pptx` first).

This is the tool to mine the group's historical report decks: the notes and outline carry the
reasoning that never made it onto the slides.

## `slides digest` — huge deck → image-text-linked Markdown workspace

When a deck is too big / image-heavy to merely *extract* (you must actually understand it, and
bulk-dumping images would sever each picture from its surrounding text), digest it instead:

```
compose slides digest 'huge.pptx' --out 'translated/' [--render] [--gif-frames 3]
    [--chunk-by section|fixed] [--max-chunk-slides 40] [--section-at 12,40,…]
    [--batch-figure 5] [--batch-text 15] [--force] [--lint]
```

**Why it beats `slides extract` for big decks:** every image is written *inline* into the Markdown
right beside its page's text, position (nine-grid label), inferred caption (nearby text), an ASCII
layout map, and a `解读：_(待填)_` slot — so the image↔text binding survives. Images are sha1-deduped
into `images/`; repeats point back to their first occurrence (interpret once, reuse everywhere).

**Two phases**:
- **Phase A** = this command (pure code, no vision). Emits `index.md` (nav + section list +
  high-frequency duplicate figures + batch policy), `part_*.md` skeletons, `sidecar/slides.jsonl`
  (machine-readable per-slide record), `images_manifest.json`, `progress.json` (resumable ledger).
- **Phase B** = you, batch by batch: Read each figure slide's whole-page render
  `renders/slide-NNN.png` **once**, then fill the `解读` slots.

**Unreadable formats** (your Read tool only sees jpeg/png/webp):
- **gif** animations → first/mid/last frames auto-extracted to `images/previews/*_fJofK_frameN.png`
  (`--gif-frames`, default 3), embedded inline with a “动图（共 N 帧）” note.
- **wmf/emf** vectors → a deterministic `images/previews/<stem>.png` link is written up front and
  filled by LibreOffice at `--render`; until then it is *pending*, not broken.

**Idempotent / resumable:** re-run **without** `--force` and it only tops up missing renders — the
skeletons (and any `解读` you filled) are never touched. `--force` rebuilds from scratch and
**purges stale `part_*.md`** (filenames track section titles, so old ones would otherwise linger
and double-count in `--lint`). `--lint` classifies every image link as ok / broken / pending.

Output defaults to `cache/digests/<stem>/`; use `--out` to place it beside the source deck.
`images/` and `renders/` are git-ignored (regenerable from the pptx); the `.md` / sidecar /
manifest / progress ledger are meant to be committed.

If every slide shares one layout, auto-sectioning finds nothing — define sections by hand with
`--section-at 12,40,77,…`. `--render` needs LibreOffice; without it Phase A still completes and
records renders as *pending*, so add `--render` on a later re-run.

## `read` — fast generic extraction

```
compose read 'file.docx' [--preview] [--force] [--backend auto|markitdown|pymupdf4llm|pptx_io]
```

`--backend auto` picks by extension:
- `.pdf` → `pymupdf4llm` (fast local text; equations are plain text, not LaTeX).
- `.pptx` → `markitdown`, falling back to `pptx_io` if markitdown fails.
- `.docx` / `.xlsx` / `.html` / … → `markitdown` (needs the matching markitdown extra).

Output: `cache/extracted/<stem>__<backend>.md`. Re-running **reuses the cache**; `--force`
re-extracts. `--preview` prints the first 1500 chars inline.

Use `read` when you only need the prose (a docx data source, a PDF report). Use
`slides extract` when structure (notes/tables/outline) matters.

## `docx read` — structured Word read

```
compose docx read 'report.docx' [--preview] [--out path.md]
```
Preserves heading levels (`#`… by Word style), paragraphs, bullet levels, and tables as
Markdown tables — better fidelity than `read` (markitdown) when structure matters. Use
`read` only for a quick flat dump.

## Writing: Markdown → pptx / docx (Pandoc)

`from-markdown` is powered by **Pandoc** (the community document-converter standard), not a
hand-rolled parser — headings, nested lists, pipe tables, math, images, and footnotes all map
faithfully, and you can apply a house style via `--reference-doc`.

```
compose slides from-markdown 'outline.md' --out 'deck.pptx' [--slide-level 2] [--reference-doc master.pptx]
compose docx   from-markdown 'outline.md' --out 'report.docx' [--reference-doc style.docx]
```

Pandoc outline conventions:
- **Slides:** `--slide-level N` (default 2) decides which heading starts a new slide. With the
  default, `#` = title/section slide, `##` = content slide; deeper headings become sub-points.
- **Speaker notes (pptx):** put them in a fenced `notes` div — **not** a `> ` blockquote:

      ## Slide title
      - a bullet

      ::: notes
      These words land in the speaker-notes pane.
      :::

- **Word styles:** `#`..`######` → Heading 1..6; `-`/`*` → list styles; pipe tables → Word tables;
  `--reference-doc style.docx` applies your fonts/colors/heading styles.
- **Encoding:** Pandoc reads the file itself (UTF-8, BOM tolerated) — no preprocessing needed.

`--reference-doc` templates: generate a default with
`pandoc -o ref.docx --print-default-data-file reference.docx` (or `.pptx`), restyle it, then pass it.

Incremental edits still go through python-pptx / python-docx — Pandoc can only generate a **fresh**
file; it cannot append to an existing .pptx/.docx, and it cannot read .pptx at all (see
[maintenance.md](maintenance.md)):

```
compose slides new 'deck.pptx' --title '…'
compose slides add 'deck.pptx' --title '…' --bullet '…' --notes '…'
compose docx add 'report.docx' --heading '…' --level 2
compose docx read 'report.docx' --preview     # structured read-back
```

## Choosing for PDFs

- `read x.pdf` → quick text dump (searchable prose, references).
- `verify x.pdf` → PNG pages to *see* layout (use for compiled manuscripts, scanned figures).
They answer different questions; a compiled-paper proofread needs `verify`, not `read`.

## Caches

All extracted Markdown and rendered PNGs live under `data/skills/document_writing/cache/`
(git-ignored). They are cheap to regenerate; delete freely if stale. There is no LRU prune in
Phase 1 — just remove files/dirs under `cache/` when disk pressure appears.
