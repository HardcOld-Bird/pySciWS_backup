# Reading & extracting documents

Two entry points, both write Markdown you then **Read**:

| Command | Best for | Backend |
|---|---|---|
| `compose slides extract <pptx>` | Slide decks (structured) | `python-pptx` |
| `compose slides digest <pptx>` | Huge, image-heavy decks you must *understand* | `python-pptx` + Pillow (+ LibreOffice for renders) |
| `compose read <file>` | Any Office/PDF/web file (fast) | `markitdown` / `pymupdf4llm` |
| `compose slides new/add/from-markdown` | Creating / extending decks | `python-pptx` |
| `compose docx read/from-markdown/add` | Word docs (structured read & write) | `python-docx` |
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

**Two phases** (overview in SKILL.md → “Translating a huge, image-heavy deck”):
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

## Writing: Markdown → pptx / docx

Both writers share one outline convention (round-trips with the extractors):
- `# X` — deck/doc title (first) or section-divider slide; `## X`… — slide title / doc
  heading (depth = heading level).
- `- ` / `* ` / `• ` with 2-space indent steps — bullets (indent = level).
- `> …` — speaker notes (pptx) / italic quote paragraph (docx).
- Pipe tables — a table on the current slide / doc.
- Anything else — a plain paragraph/bullet.

```
compose slides from-markdown 'outline.md' --out 'deck.pptx'
compose docx   from-markdown 'outline.md' --out 'report.docx'
```
Files are read as `utf-8-sig`, so a Windows BOM never corrupts the first `# `.
Incremental edits: `slides new` / `slides add` and `docx add` (see SKILL.md).

## Choosing for PDFs

- `read x.pdf` → quick text dump (searchable prose, references).
- `verify x.pdf` → PNG pages to *see* layout (use for compiled manuscripts, scanned figures).
They answer different questions; a compiled-paper proofread needs `verify`, not `read`.

## Caches

All extracted Markdown and rendered PNGs live under `data/skills/document_writing/cache/`
(git-ignored). They are cheap to regenerate; delete freely if stale. There is no LRU prune in
Phase 1 — just remove files/dirs under `cache/` when disk pressure appears.
