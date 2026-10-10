# maintenance 分册 2/2

> 含小节：Tests；Phase-2 status (built) & open items
> 原 `maintenance.md` 超 8KB 预算拆分（2026-10-10，backlog 20261010-refs-split-8kb）；分册索引见同目录 `maintenance.md`，按需只读所需分册。

## Tests

`uv run pytest tests/skills/document_writing -q` (53 tests; the TeX-degradation test skips when TeX
**is** installed, the LibreOffice-degradation test skips when LibreOffice is, and the
`test_pandoc_convert` roundtrips skip when Pandoc is missing — so the pass/skip split shifts with the
environment). They build throwaway
pptx/docx/pdf in tmp dirs; none require TeX. Keep `parse_log` and the extractors covered by pure
unit tests so the suite stays runnable on machines without a TeX install.

## Phase-2 status (built) & open items

Built in Phase 2:
1. **PPTX create/edit** — `pptx_io` structured write helpers + `compose slides new/add`;
   `slides from-markdown` now routes through Pandoc.
2. **DOCX read/create/edit** — `docx_io.py` + `python-docx` + `compose docx read/add`;
   `docx from-markdown` now routes through Pandoc.
3. **Beamer slides** — `templates/latex/beamer/` (second slide route, compiles to playable PDF).
4. **LibreOffice conversion** — `office_convert.py` + `compose convert`; degrades gracefully
   until LibreOffice is installed (probe covers C:/D: Program Files + registry PATH).

Open items (future):
- In-place editing of existing pptx shapes / docx runs (current write path appends/builds).
- House-style `--reference-doc` templates checked into `templates/` for reproducible docx/pptx
  branding (Pandoc already accepts them; none are bundled yet).
- A `convert` fallback when LibreOffice is absent (e.g. docx→pdf via LaTeX, pptx→pdf via
  per-slide PNG) — not implemented; prefer installing LibreOffice.
