---
name: scientific-plotting
description: Produce publication-grade scientific figures (multi-panel, nested, data-driven or external-media panels) for journal submission. Drives a unified `figures` CLI that scaffolds a per-figure production pipeline under a research's `article/figures/` dir, applies journal style presets (APS/PRL Arial 8pt 85/170mm; Nature Helvetica 7pt 89/183mm) with colorblind-safe palettes, exports EPS/PDF/SVG deliverables plus a PNG preview, audits publication compliance (width/fontsize/font-embedding/panel labels/colorblind), and closes the visual loop by letting the agent Read the PNG preview to iterate. Use when the user asks to draw / create / revise a paper figure, plot simulation or experiment data into a submission figure, restyle a figure for a different journal, export EPS/SVG, or check a figure against journal specs.
---

# Scientific Plotting

One CLI drives the figure workflow: **scaffold → draw → export → audit → visually verify**.
The backend (`src/pysci/skills/scientific_plotting/`) is a **black box** — drive it through
`figures`; open the code only when maintaining it. This skill owns the publication
**contract** (widths, fonts, palettes, export, self-check); each figure's drawing lives in a
small pipeline script so figures iterate in place.

## Invocation

From the project root: `pysci-figures <command> [options]` (`figures …` is shorthand).
Fallback if the entry is missing: `uv run --no-sync python -m
pysci.skills.scientific_plotting.tools.figures …`. Git Bash: quote multi-word args.
PowerShell: Chinese output → basic.md §3.

## Commands at a glance

| Command | Use when | Key output |
|---|---|---|
| `doctor` | Session start / anything broken | config + backends + fonts |
| `styles` | Choosing journal look / palette | presets + palettes |
| `new <research> <slug>` | Starting a figure | scaffolded pipeline dir |
| `build <figdir>` | Producing deliverables | EPS/PDF/SVG/PNG + `_preview.png` |
| `preview <figdir>` | Fast visual iteration | `_preview.png` |
| `audit <figdir>` | Publication + annotation-semantics check | PASS/FAIL report |
| `inspect <png>` | Non-pipeline image health check | size/colour/watermark scan |
| `overlap <img> --white <W>` | Zero-occlusion pixel verdict | ink ∩ data/line counts |
| `raster-panel` / `compose-grid` / `field-panel` | External media / grids / field data | PNG |
| `list <research>` | Pipeline inventory | listing |

Run `figures <command> -h`. **`new`: `<research> <slug>`; `build`/`preview`/`audit`: one `<figdir>`.**

## Quick start

```
- [ ] 1. figures doctor
- [ ] 2. figures new gain_ep fig1_ep_band --style aps --width double --template multi_panel
- [ ] 3. Edit src/pysci/research/<name>/article/figures/<slug>.py  (`build_figure`)
- [ ] 4. figures build 'data/research/<n>_<name>/article/figures/<slug>'
- [ ] 5. Read out/<slug>_preview.png   # VISUAL CHECK — iterate 3–4
- [ ] 6. figures audit '<figdir>' --panels   # final gate
```

**Always Read `_preview.png` before declaring done** — a figure can export cleanly yet look wrong.

## Pipeline contract

- Code: `src/pysci/research/<name>/article/figures/<slug>.py`, exposing
  `build_figure(style=None, research_dir=None, **kw) -> Figure`.
- Data/out: `data/research/<n>_<name>/article/figures/<slug>/` (with `notes.md` and `out/`).
- `build_figure` runs inside `style_context`: figsize/fonts/sizes/palette inherit; declare only
  the kwargs you need. `STYLE.yaml` (root baseline + figdir key-by-key override) sets defaults;
  CLI wins. Details: `references/plotting.md`, `plotting-01-overview.md`.

## Annotation semantics, layout & reference fidelity (mandatory for production figures)

1. **参考图对照表**：任务书给参考图时，**画前**产出逐要素表——每要素：参考图对应物 →
   物理含义 → 产物元素 → 元素间关系 → **处置（照搬/适配/偏离+理由）**。无表＝缺理解痕迹
   （reviewer G2 判 FAIL）。模板见 taskbook-template.md。
2. **标注语义机械声明**：把任务书的可计数语义模型（如「每侧 2 个框、外框>内框、虚线相连」）
   写入图目录 `ANNOTATIONS.yaml`；给每个标注原语打 `gid="annot:<id>[;role=…][;view=…]"`，
   视图轴 `set_gid("view:<name>")`。`figures audit` 据此核验**计数/包含+边距/相对大小/连接线**。
   schema 与示例 → `references/plotting-04-annotation-semantics.md`。
3. **处置分支不得替换**：按任务书「处置分支声明」走指定分支；欲换处置（含以技术判断否决
   用户显式指令）→ 以 `<blocked>`/疑问清单上报裁决，**不得静默替换**。
4. **布局节奏/编号锚定**：
   ①标题与上方外缘间距 > 与下方子图间距；
   ②图例置于数据图下方、不压主体；
   ③子图组与脚注段留清晰间距；
   ④编号锚定**整体 bbox（含图例/colorbar/标注）左上角**。

## Typesetting, zero-occlusion & style

**零遮挡铁律（硬伤）**：①文字/刻度不压图线/彩色数据区 ②图体与标题留间距 ③colorbar
标题↔色标、色标↔子图留边距 ④任何元素重叠=FAIL 无例外 ⑤**近而不交**：不该相交的线/框
贴边/临界间距=缺陷。**判据命令**：`figures overlap <img> --white <删文字版.png>`。
**设计用色克制**：整图设计用色 **1~2 个色系**；数据 colormap **不计入**色系计数，Rainbow
类必选时设计用色取其**同族/中性**色，**勿另起色系**。
**scienceplots**（可选审美层，契约层优先）：`STYLE.yaml scienceplots: true` 或
`style_context(..., scienceplots=True)`。细则与排障 → `references/plotting-03-typesetting-and-style.md`。

## Reference files

（`*`=分册索引，按需读单册）

- `references/plotting.md` * — index；`plotting-01-overview.md` * — widths/fonts/typography/
  palettes/layout/external media/discovery/iteration；`plotting-02-cli-argument-forms-…md` —
  CLI signatures；`plotting-03-typesetting-and-style.md` * — zero-occlusion + scienceplots +
  troubleshooting；`plotting-04-annotation-semantics.md` * — **ANNOTATIONS.yaml schema、gid
  约定与四类判据**（画带语义标注的场图前必读）。
