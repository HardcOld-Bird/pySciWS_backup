"""scientific_plotting 统一 CLI 入口（对齐 compose / research / simulation 的门面模式）。

对 ``tools/`` 下各模块（config/style/palette/layout/export/runner/audit/scaffold）做薄编排，
让我（Agent）与用户都能用一条命令驱动出版级插图的闭环::

    uv run pysci-figures doctor
    ... figures styles
    ... figures new gain_ep fig1_ep_band --style aps --width double
    ... figures build data/research/1_gain_ep/article/figures/fig1_ep_band
    ... figures preview <figdir> --style nature
    ... figures audit <figdir> --panels

子命令分组：doctor / styles / new / build / preview / audit / list。
build/preview 会在图目录的 out/ 下产出交付件与 _preview.png，供 Agent Read 视觉校验。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# CLI 始终无头运行：在导入任何 pyplot 相关模块前锁定 Agg 后端，避免弹窗/阻塞。
import matplotlib

matplotlib.use("Agg")

from . import audit as _audit  # noqa: E402
from . import palette as _palette  # noqa: E402
from . import runner as _runner  # noqa: E402
from . import scaffold as _scaffold  # noqa: E402
from . import style as _style  # noqa: E402
from .config import settings  # noqa: E402


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------
def cmd_doctor(args: argparse.Namespace) -> int:
    print(settings.summary())
    print()

    # 后端库自检
    print("--- backends ---")
    for mod in ("matplotlib", "numpy", "scipy"):
        try:
            m = __import__(mod)
            print(f"  {mod:<12}: {getattr(m, '__version__', '?')}")
        except Exception as e:  # noqa: BLE001
            print(f"  {mod:<12}: MISSING ({e!r})")
    for mod in ("ultraplot", "scienceplots", "pyvista"):
        try:
            m = __import__(mod)
            print(f"  {mod:<12}: {getattr(m, '__version__', 'ok')} (optional)")
        except Exception:  # noqa: BLE001
            print(f"  {mod:<12}: (not installed, optional)")

    # 字体自检：逐个预设报告首选字体是否可用
    print("\n--- fonts ---")
    for p in _style.list_presets():
        resolved, missing = _style.resolve_font(p.font_chain)
        flag = "OK" if resolved == p.font_chain[0] else "FALLBACK"
        print(f"  {p.name:<8}[{flag}] resolved={resolved}  missing={missing or '[]'}")

    # 无头后端（导出/预览不弹窗）
    import matplotlib

    print(f"\n  matplotlib backend: {matplotlib.get_backend()}")
    return 0


# ---------------------------------------------------------------------------
# styles
# ---------------------------------------------------------------------------
def cmd_styles(args: argparse.Namespace) -> int:
    print("=== journal style presets ===")
    for p in _style.list_presets():
        print(f"\n[{p.name}] {p.label}")
        print(f"  font chain : {' -> '.join(p.font_chain)}")
        print(f"  font sizes : base {p.base_fontsize}pt / tick {p.tick_fontsize}pt")
        print(f"  widths     : single {p.single_width_mm:g}mm / double {p.double_width_mm:g}mm")
        resolved, missing = _style.resolve_font(p.font_chain)
        print(f"  resolved   : {resolved}" + (f"  (missing: {missing})" if missing else ""))
    print("\n=== colorblind-safe palettes ===")
    for name in _palette.list_palettes():
        cols = _palette.get_palette(name)
        print(f"  {name:<12}: {' '.join(cols)}")
    print(f"\ndefault style (config): {settings.default_style}")
    return 0


# ---------------------------------------------------------------------------
# new (scaffold)
# ---------------------------------------------------------------------------
def cmd_new(args: argparse.Namespace) -> int:
    try:
        figdir = _scaffold.scaffold_figure(
            args.research,
            args.slug,
            style=args.style,
            width=args.width,
            template=args.template,
            overwrite=args.force,
        )
    except (KeyError, FileExistsError) as e:
        print(f"[figures] new 失败：{e}", file=sys.stderr)
        return 1
    print(f"已脚手架图管线目录：{figdir}")
    print(f"  管线模块 : {figdir / (args.slug + '.py')}")
    print(f"  模板     : {args.template}  (可选: {', '.join(_scaffold.list_templates())})")
    print("\n下一步：")
    print(f"  1) 编辑 {args.slug}.py 里的 build_figure()（替换示例数据为真实数据）")
    print(f"  2) uv run pysci-figures build '{figdir}'")
    print(f"  3) Read 产出的 out/{args.slug}_preview.png 做视觉校验")
    return 0


# ---------------------------------------------------------------------------
# build / preview
# ---------------------------------------------------------------------------
def _fmt_tuple(raw: str | None) -> tuple[str, ...] | None:
    if not raw:
        return None
    return tuple(p.strip().lower() for p in raw.split(",") if p.strip())


def cmd_build(args: argparse.Namespace) -> int:
    res = _runner.build_figure_dir(
        args.figdir,
        style=args.style,
        width=args.width,
        aspect=args.aspect,
        palette=args.palette,
        formats=_fmt_tuple(args.formats),
        stem=args.stem,
        save_dpi=args.save_dpi,
        preview_dpi=args.preview_dpi,
    )
    print(res.report())
    if res.ok:
        print(f"\n✓ 视觉校验：Read '{res.export.preview}'")
        return 0
    print("\n✗ 构建/导出未完全成功，见上方 ERROR。", file=sys.stderr)
    return 1


def cmd_preview(args: argparse.Namespace) -> int:
    res = _runner.preview_figure_dir(
        args.figdir,
        style=args.style,
        width=args.width,
        aspect=args.aspect,
        palette=args.palette,
        stem=args.stem,
        preview_dpi=args.preview_dpi,
    )
    print(res.report())
    if res.ok:
        print(f"\n✓ 视觉校验：Read '{res.export.preview}'")
        return 0
    print("\n✗ 预览渲染失败，见上方 ERROR。", file=sys.stderr)
    return 1


# ---------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------
def cmd_audit(args: argparse.Namespace) -> int:
    rep = _audit.audit_figure_dir(
        args.figdir,
        style=args.style,
        width=args.width,
        expect_panel_labels=args.panels,
        min_fontsize=args.min_fontsize,
    )
    print(rep.report())
    return 0 if rep.ok else 1


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------
def cmd_list(args: argparse.Namespace) -> int:
    root = _runner.figures_root(args.research)
    if not root.is_dir():
        print(f"[figures] 尚无 figures 根目录：{root}", file=sys.stderr)
        return 0
    print(f"=== figures under {root} ===")
    found = False
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        try:
            pipeline = _runner.discover_pipeline(d)
        except FileNotFoundError:
            continue
        found = True
        out = d / "out"
        n_out = len(list(out.glob("*"))) if out.is_dir() else 0
        print(f"  {d.name:<28} pipeline={pipeline.name}  out_files={n_out}")
    if not found:
        print("  (未发现任何图管线；用 `figures new` 脚手架一个)")
    return 0


# ---------------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pysci-figures",
        description="pySciWS 科研绘图技能：出版级插图的设计/导出/自检/视觉校验闭环。",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # doctor
    s = sub.add_parser("doctor", help="自检：配置/后端库/字体/无头后端")
    s.set_defaults(func=cmd_doctor)

    # styles
    s = sub.add_parser("styles", help="列出期刊风格预设与色盲安全调色板")
    s.set_defaults(func=cmd_styles)

    # new
    s = sub.add_parser("new", help="脚手架一幅图的生产管线目录")
    s.add_argument("research", help="研究线名（不含数字前缀），如 gain_ep")
    s.add_argument("slug", help="图目录名，如 fig1_ep_band")
    s.add_argument("--style", default=settings.default_style, help="期刊预设（aps/nature）")
    s.add_argument("--width", default="double", help="设计宽度 single/double 或毫米数")
    s.add_argument(
        "--template",
        default=_scaffold.DEFAULT_TEMPLATE,
        help=f"模板：{', '.join(_scaffold.list_templates())}",
    )
    s.add_argument("--force", action="store_true", help="目录已存在时覆盖管线文件")
    s.set_defaults(func=cmd_new)

    # build
    s = sub.add_parser("build", help="运行图管线并导出全部格式 + PNG 预览")
    s.add_argument("figdir", type=Path, help="图生产管线目录")
    s.add_argument("--style", default=None, help="覆盖预设（默认取 STYLE.yaml / config）")
    s.add_argument("--width", default=None, help="覆盖设计宽度")
    s.add_argument("--aspect", type=float, default=None, help="高/宽比（默认 0.618）")
    s.add_argument("--palette", default=None, help="调色板名（见 styles）")
    s.add_argument("--formats", default=None, help="逗号分隔交付格式（默认取 config）")
    s.add_argument("--stem", default=None, help="交付件主名（默认图目录名）")
    s.add_argument("--save-dpi", type=int, default=None, dest="save_dpi")
    s.add_argument("--preview-dpi", type=int, default=None, dest="preview_dpi")
    s.set_defaults(func=cmd_build)

    # preview
    s = sub.add_parser("preview", help="只渲染 PNG 预览（快速视觉迭代）")
    s.add_argument("figdir", type=Path)
    s.add_argument("--style", default=None)
    s.add_argument("--width", default=None)
    s.add_argument("--aspect", type=float, default=None)
    s.add_argument("--palette", default=None)
    s.add_argument("--stem", default=None)
    s.add_argument("--preview-dpi", type=int, default=None, dest="preview_dpi")
    s.set_defaults(func=cmd_preview)

    # audit
    s = sub.add_parser("audit", help="出版规范自检（宽度/字号/字体嵌入/角标/色盲）")
    s.add_argument("figdir", type=Path)
    s.add_argument("--style", default=None)
    s.add_argument("--width", default=None)
    s.add_argument("--panels", action="store_true", help="要求多子图带 (a)(b)(c) 角标")
    s.add_argument("--min-fontsize", type=float, default=None, dest="min_fontsize")
    s.set_defaults(func=cmd_audit)

    # list
    s = sub.add_parser("list", help="列出某研究线下已有的图管线")
    s.add_argument("research", help="研究线名（不含数字前缀），如 gain_ep")
    s.set_defaults(func=cmd_list)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        rc = args.func(args)
        return int(rc) if rc else 0
    except KeyboardInterrupt:
        print("\n[figures] 已中断。", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
