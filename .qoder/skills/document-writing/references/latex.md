# LaTeX workflow in depth

Everything here is driven through `compose tex …`. This file explains the pieces so you can
make good authoring decisions, not just run commands.

## Projects & templates

`compose tex new <slug> --template <name>` copies a template from
`data/skills/document_writing/templates/latex/<name>/` into
`data/skills/document_writing/projects/<slug>/`, adding empty `figures/` and `build/` dirs.

| Template | Class | Engine (latexmkrc) | Bib backend | Use for |
|---|---|---|---|---|
| `revtex` | `revtex4-2` (reprint, superscriptaddress, longbibliography) | pdflatex (`$pdf_mode=1`) | bibtex (`\bibliography{refs}`) | APS journals (PRL/PRA/PRB/PRApplied…) |
| `article` | `article` + optional `ctex` | xelatex (`$pdf_mode=5`) | biber (`biblatex`, `\printbibliography`) | Generic reports, theses, Chinese text |
| `beamer` | `beamer` (Madrid, 16:9) | pdflatex (`$pdf_mode=1`) | bibtex (`\bibliography{refs}`) | Slides / talks → playable PDF deck |

Pick `revtex` when the target is an APS/physics journal; pick `article` for internal reports or
anything needing Chinese (xelatex + ctex handles CJK fonts); pick `beamer` for a slide deck.
`--title '…'` rewrites the first `\title{…}` in the scaffolded `main.tex`.

**Beamer notes:** keep the body ASCII while using pdflatex; for Chinese uncomment
`\usepackage{ctex}` and set `$pdf_mode = 5` (xelatex) in the project `latexmkrc`. Use
`\item<2->` for overlay (progressive) bullets and `\note{…}` for speaker notes (hidden in the
output PDF by default). The compiled PDF is directly playable — an alternative slide route to
pptx; convert to pptx-friendly assets via `verify` PNGs if a colleague needs PowerPoint.

Each project is self-contained: `main.tex`, `refs.bib`, `latexmkrc`, `figures/`, `build/`.
The `latexmkrc` sets `$out_dir='build'` so intermediates never litter the project root.

## Engines

`compose tex build <target> --engine {auto|pdflatex|xelatex|lualatex}`.
- `auto` (default) passes **no** engine flag; the project's `latexmkrc` `$pdf_mode` decides
  (1=pdflatex, 4=lualatex, 5=xelatex). Prefer this — the template already chose correctly.
- Override only when a specific engine is required (e.g. `--engine xelatex` for CJK in a
  project whose latexmkrc says pdflatex).

`--shell-escape` adds `-shell-escape` (needed by minted / some TikZ externalization).
`--no-bib` skips the bibliography pass (fast iteration while drafting).

## The compile → verify loop (the heart of the skill)

```
compose tex build <target>            # latexmk; prints parsed errors/warnings/boxes
compose verify <pdf> [--pages 1,3]    # render PNG; Read them to see the REAL layout
```

`build` parses the `.log` into `file:line` diagnostics:
- **Errors** — `./main.tex:42: Undefined control sequence` (file-line-error form) and the
  classic `! …` + `l.42` form. Fix the cited line.
- **Warnings** — undefined citations/references, missing figures.
- **Boxes** — `Overfull \hbox (12.3pt too wide) at lines 8–12`. These are *layout* defects:
  LaTeX still "succeeds". Hunt them down before submitting.

A clean exit code does **not** mean a good PDF. Always `verify` and Read the PNGs:
check float placement, equation numbering, figure sizing, bibliography rendering, and that no
text runs into the margin. Iterate edit → build → verify until the pages look right.

`verify` options: `--dpi` (default 140; 100 is enough for a layout glance), `--pages 1,3`
(1-based subset), `--max-pages N`, `--out-dir`. PNGs land in `cache/renders/<stem>/page-NNN.png`.

## Bibliography (refs.bib)

`compose tex refs <target> --query '<topic>' | --collection <key> | --tag <t> | --keys a,b`
rebuilds `<project>/refs.bib` from the user's Zotero library (reuses the
`literature_research` Zotero bridge). Citekeys follow Better-BibTeX style
(`zhu2018simultaneous`). Duplicate keys get `a/b/c` suffixes. `--out` overrides the target path.

While drafting you may also hand-edit `refs.bib`; `tex refs` overwrites it, so re-run only when
you want a fresh Zotero snapshot. After changing `refs.bib`, rebuild (`tex build`) so citations
resolve; undefined-citation warnings in the build output tell you a `\cite{key}` has no entry.

## Manuscript-quality guidance

- **Structure**: one `\section{}`/`\subsection{}` per logical unit; keep equations in `equation`
  / `align` with `\label{eq:…}` and reference via `\eqref`. Never hardcode numbers.
- **Floats**: give every figure/table a `\label{fig:…}` and cite with `\ref`. Place floats near
  first mention; let LaTeX float them — do not force `[H]`.
- **Math**: define macros for repeated symbols in the preamble (`\newcommand{\ep}{\mathrm{EP}}`);
  use `\mathrm`/`\mathbf` for operators/vectors, not italic letters.
- **Units & numbers**: non-breaking space between value and unit (`12~kHz`); en-dash for ranges.
- **Citations**: `\cite{key}`; for APS use superscript style provided by the class.
- **Figures**: vector (PDF/EPS) for plots, placed in `figures/`; set width as a fraction of
  `\columnwidth`/`\textwidth`; keep captions self-contained.
- Before declaring done: `tex build` clean **and** no overfull boxes **and** `verify` PNGs read
  correctly **and** `tex lint` quiet.

## Lint

`compose tex lint <target>` runs chktex (`-f "%f:%l:%c:%t:%m\n"`). It flags style/spacing issues
(e.g. intersentence spacing, `~~` vs `~`). Fix the real ones; chktex is advisory, not blocking.
Requires chktex (ships with TeX Live); returns exit 3 with a notice if absent.
