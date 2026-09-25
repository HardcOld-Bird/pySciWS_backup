# Backend maintenance

Backend package: `src/pysci/skills/document_writing/`. Data root: `data/skills/document_writing/`
(exported as `DOCWRITING_ROOT` from `pysci.paths`). Skill docs: `.qoder/skills/document-writing/`.

## Module map

| Module | Responsibility |
|---|---|
| `config.py` | `Settings` (all paths), toolchain discovery (`which` + fallback bin-dir scan), `tex_ready`, `python_libs`, `summary()` |
| `latex_build.py` | latexmk driver, `.log` → `file:line` diagnostics (`parse_log`), chktex `lint`, `TeXNotInstalled` |
| `pdf_render.py` | PDF → PNG via pymupdf (`render_pdf_pages`, `page_count`) |
| `pptx_io.py` | Structured PPTX read (`read_pptx`, `slides_to_markdown`) **and write** (`build_pptx`, `add_slide_to`, `markdown_to_pptx`) |
| `docx_io.py` | Structured DOCX read (`read_docx`, `docx_to_markdown`) and write (`build_docx`, `add_block_to`, `markdown_to_docx`) |
| `office_convert.py` | LibreOffice headless conversion (`convert`, `LibreOfficeNotInstalled`) |
| `extract.py` | Unified `to_markdown` (markitdown / pymupdf4llm / pptx_io) + cache |
| `deck_digest.py` | Huge-deck digestion: `.pptx` → image-text-linked Markdown workspace (`digest_pptx`, `verify_links`); sha1 dedup, section/chunk detection, gif-frame & vector previews, resumable `progress.json` |
| `refs_bridge.py` | Zotero → BibTeX (`export_bib`), reuses `literature_research.tools.zotero_bridge` |
| `compose.py` | Thin argparse facade; imports heavy modules lazily so `doctor` works without `[writing]` |

`compose.py` must stay thin: orchestration only. Put real logic in the modules above.

## Config & discovery

- Paths come from `config.settings` (built on `DOCWRITING_ROOT`): `templates_dir`,
  `projects_dir`, `assets_dir`, `cache_dir`, `cache_extracted`, `cache_renders`.
- `which(name)` first checks `PATH`, then scans known install roots
  (`C:/texlive/*/bin/windows`, TinyTeX, MiKTeX, LibreOffice `program/`). This makes a freshly
  installed TeX detectable **without** reopening the shell. If TeX is installed but `doctor`
  still says missing, add its `bin/windows` to that scan list.
- `TEX_TOOLS` / `ENGINES` enumerate what `doctor` probes. `tex_ready` = latexmk + one engine.
- LibreOffice discovery: `find_libreoffice()` respects `DOCWRITING_SOFFICE` in `.env` **first**
  (absolute path to `soffice.exe`, for custom / D-drive installs), then `PATH`, then the bin-dir scan.
- Optional deps live in the `[writing]` extra (`pyproject.toml`): `markitdown[pptx]`,
  `python-pptx`. Install with `uv sync --extra writing`. `pymupdf` is a base dep.

## Log parsing (`latex_build.parse_log`)

Pure function `(log_text) -> (errors, warnings)`; unit-tested in
`tests/skills/document_writing/test_latex_build.py`. Handles:
- file-line-error form `./main.tex:42: msg`
- classic form `! msg` + following `l.42`
- `Overfull/Underfull \hbox … at lines a--b`
- generic `LaTeX/Package/Class Warning: … on input line N`

When touching the regexes, keep `_RE_WARNING` greedy-to-end-of-line and pull the line number
separately via `_RE_INPUT_LINE` (a non-greedy message group collapses to one char).

## Adding a template

Create `data/skills/document_writing/templates/latex/<name>/` containing at least `main.tex`
(`compose` lists a dir as a template only if it has `main.tex`). Add `latexmkrc` (set
`$pdf_mode` and `$out_dir='build'`) and `refs.bib` as needed. No code change required —
`_list_templates()` discovers it automatically.

## Deck digestion (`deck_digest.py`)

Two layers, so the expensive *visual* work is done **once** by the LLM and reused forever:
- **Raw layer (`_scan`, pure code, no vision):** per slide → geometry (nine-grid `_pos_label`),
  text, `_layout_map` ASCII, sha1-dedup image export, `_near_texts` caption inference. Unreadable
  formats: gif → Pillow frame previews (`_make_previews` + `_even_indices`), wmf/emf → deterministic
  preview path filled later by LibreOffice (`_make_vector_previews`).
- **Sectioning / chunking:** `_detect_sections` uses “no image + has title + **no body** = divider”
  (robust when a deck reuses one layout — the common report-deck case); `section_starts` overrides
  auto-detection. `_group_chunks` packs sections into ≤`max_chunk_slides` md files (a section longer
  than the cap is split internally; a chunk never straddles the cap).
- **Writers:** `_write_chunk` / `_write_index` / `_write_sidecar` / `_write_manifest` /
  `_write_progress` / `_write_gitignore`. Each image block binds link + position + size + px +
  near-text + duplicate-pointer + `解读：_(待填)_` so image and text are never separated.
- **Render (`_render_deck`):** pptx → pdf (LibreOffice) → per-page PNG (`pdf_render`); the huge
  intermediate PDF is deleted. Optional — missing renders/vector previews are `pending`, not
  `broken` (`verify_links` keys off the `renders/` + `images/previews/` prefixes).

Gotchas learned the hard way:
- Full rebuild **must** `unlink` stale `part_*.md` before writing (filenames track section titles;
  leftovers otherwise double-count in `verify_links`). The reuse path must **never** touch them.
- Reuse path (has `progress.json`, not `--force`) only tops up renders / vector previews and rewrites
  `progress.json` — it never rewrites skeletons, protecting filled `解读`.
- `_layout_map` guards `sw/sh <= 0`; `_scan` swallows per-shape errors so one bad shape cannot abort
  a several-hundred-slide run.

## Tests

`uv run pytest tests/skills/document_writing -q` (52 tests; the TeX-degradation test skips when TeX
**is** installed and the LibreOffice-degradation test skips when LibreOffice is, so the pass/skip
split shifts with the environment). They build throwaway
pptx/docx/pdf in tmp dirs; none require TeX. Keep `parse_log` and the extractors covered by pure
unit tests so the suite stays runnable on machines without a TeX install.

## Phase-2 status (built) & open items

Built in Phase 2:
1. **PPTX create/edit** — `pptx_io` write helpers + `compose slides new/add/from-markdown`.
2. **DOCX read/create/edit** — `docx_io.py` + `python-docx` in `[writing]` + `compose docx …`.
3. **Beamer slides** — `templates/latex/beamer/` (second slide route, compiles to playable PDF).
4. **LibreOffice conversion** — `office_convert.py` + `compose convert`; degrades gracefully
   until LibreOffice is installed (probe covers C:/D: Program Files + registry PATH).

Open items (future):
- In-place editing of existing pptx shapes / docx runs (current write path appends/builds).
- Richer docx styling (styles template, images) and pptx themes beyond the default template.
- A `convert` fallback when LibreOffice is absent (e.g. docx→pdf via LaTeX, pptx→pdf via
  per-slide PNG) — not implemented; prefer installing LibreOffice.
