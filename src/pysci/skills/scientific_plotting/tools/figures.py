"""scientific_plotting 统一 CLI 入口（对齐 compose / research / simulation 的门面模式）。

对 ``tools/`` 下各模块（config/style/palette/layout/export/runner/audit/scaffold）做薄编排，
让我（Agent）与用户都能用一条命令驱动出版级插图的闭环::

    uv run --no-sync pysci-figures doctor
    ... figures styles
    ... figures new gain_ep fig1_ep_band --style aps --width double
    ... figures build data/research/1_gain_ep/article/figures/fig1_ep_band
    ... figures preview <figdir> --style nature
    ... figures audit <figdir> --panels

子命令分组：doctor / styles / new / build / preview / audit / inspect / overlap / raster-panel / field-panel / compose-grid / list。
build/preview 会在图目录的 out/ 下产出交付件与 _preview.png，供 Agent Read 视觉校验。
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

# CLI 始终无头运行：在导入任何 pyplot 相关模块前锁定 Agg 后端，避免弹窗/阻塞。
import matplotlib

matplotlib.use("Agg")

from . import audit as _audit  # noqa: E402
from . import field as _field  # noqa: E402
from . import imgcheck as _imgcheck  # noqa: E402
from . import occlusion as _occ  # noqa: E402
from . import palette as _palette  # noqa: E402
from . import raster as _raster  # noqa: E402
from . import runner as _runner  # noqa: E402
from . import scaffold as _scaffold  # noqa: E402
from . import style as _style  # noqa: E402
from .config import settings  # noqa: E402

# build/preview/audit 都只收**单一 figdir 位置参数**；<research> <slug> 是 new 的签名。
# 二者混淆会得到 "unrecognized arguments"（backlog 20261009-taskbook-cli-signature），
# 故在 help/epilog 里反复点明正确形态，让误用在 --help 时就暴露。
_FIGDIR_METAVAR = "<figdir>"
_FIGDIR_HELP = (
    "图目录路径（单一位置参数），如 "
    "data/research/1_gain_ep/article/figures/<slug>；"
    "注意不是 <research> <slug>——那是 new 的签名"
)
_FIGDIR_EPILOG = (
    "例：pysci-figures build 'data/research/1_gain_ep/article/figures/fig1_ep_band'\n"
    "build/preview/audit 只收单一 <figdir> 路径；<research> <slug> 是 new 的签名。"
)


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
        print(
            f"  widths     : single {p.single_width_mm:g}mm / double {p.double_width_mm:g}mm"
        )
        resolved, missing = _style.resolve_font(p.font_chain)
        print(
            f"  resolved   : {resolved}"
            + (f"  (missing: {missing})" if missing else "")
        )
    print("\n=== colorblind-safe palettes ===")
    for name in _palette.list_palettes():
        cols = _palette.get_palette(name)
        print(f"  {name:<12}: {' '.join(cols)}")
    print("\n=== scienceplots（可选审美层；契约层优先） ===")
    sp = _style.scienceplots_available()
    if sp:
        print(f"  已安装；可用风格: {', '.join(sp)}")
        print(
            f"  默认叠加: {' + '.join(_style.SCIENCEPLOTS_DEFAULT)}"
            "（STYLE.yaml `scienceplots: true` 启用；契约键覆盖之）"
        )
    else:
        print("  (未安装——可选；`uv add scienceplots`)")
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
    pipeline = _runner.figures_code_root(args.research) / f"{args.slug}.py"
    print("已脚手架图管线（代码与产物分居两侧）：")
    print(f"  管线模块 : {pipeline}")
    print("             ⚠ 该代码路径需列入任务书写权限白名单")
    print(f"  产物目录 : {figdir}")
    print(
        f"  模板     : {args.template}  (可选: {', '.join(_scaffold.list_templates())})"
    )
    print("\n下一步：")
    print(f"  1) 编辑 {pipeline} 里的 build_figure()（替换示例数据为真实数据）")
    print(f"  2) uv run --no-sync pysci-figures build '{figdir}'")
    print(
        f"  3) Read 产出的 {figdir / 'out' / (args.slug + '_preview.png')} 做视觉校验"
    )
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
        prefer="figdir" if args.pipeline_in_figdir else None,
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
        prefer="figdir" if args.pipeline_in_figdir else None,
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
        annotations=args.annotations,
    )
    print(rep.report())
    return 0 if rep.ok else 1


# ---------------------------------------------------------------------------
# inspect（不依赖管线的图像体检；backlog 20261015-204912 §4.4）
# ---------------------------------------------------------------------------
def cmd_inspect(args: argparse.Namespace) -> int:
    """体检单张图：尺寸（像素 + **等效 mm**）/ 色域占用 / 饱和像素占比 / 边缘水印·logo 检测。

    供**非 scaffold 产物**用（那类目录 `audit` 恒 N/A，审查的「步骤 2 工具核验」此前无工具）。
    等效物理尺寸由 PNG dpi 元数据换算；**PNG 不记录渲染时的 `savefig.bbox`**，故本命令只给
    中性说明「bbox 模式未知」——不得拿全局样式断言 tight 已生效（图可在
    ``rc_context(savefig.bbox="standard")`` 下渲染；2026-10-18 backlog 20261018-114448-reviewer）。
    """
    health, rc = _imgcheck.inspect_image(args.image, band=args.band)
    if not health:
        print(f"[!] 图像读不到：{args.image}")
        return rc
    # tight_bbox=None：不可证 → 中性提示（确证模式只有调用方知道，CLI 无从读取）。
    for line in _imgcheck.report_lines(health):
        print(line)
    if rc:
        print("verdict: 有边缘标记（疑似外部渲染器水印/logo 未清除）")
    else:
        print("verdict: 边缘带无标记 ✓")
    return rc


# ---------------------------------------------------------------------------
# overlap（零遮挡像素判据；backlog 20261016-014160）
# ---------------------------------------------------------------------------
def _dpi_of(path: Path) -> float | None:
    """图像的 dpi（元数据缺失 → None）——报告头据此写出判据尺度。"""
    try:
        from PIL import Image

        with Image.open(path) as im:
            dpi = im.info.get("dpi")
        if isinstance(dpi, (tuple, list)) and dpi and dpi[0]:
            return float(dpi[0])
    except Exception:  # noqa: BLE001
        return None
    return None


class _ProgressTicker:
    """每 ``interval`` 秒打印一行「当前阶段 + 已用秒」并 **flush**。

    stdout 重定向时是**块缓冲**，不 flush 就看不到任何进度（进程像挂死，backlog
    20261017-101924：4k 成品单次判据 8–10 分钟，成员据此起了 3 个重复后台任务）。
    """

    def __init__(self, interval: float = 15.0):
        self.interval = interval
        self.stage = "准备"
        self.t0 = time.monotonic()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            print(
                f"[overlap] … {self.stage}（已 {time.monotonic() - self.t0:.0f}s）",
                flush=True,
            )

    def __enter__(self) -> _ProgressTicker:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join(timeout=2)


def cmd_overlap(args: argparse.Namespace) -> int:
    """零遮挡判据：文字墨迹 ∩ 彩色数据区 / 图线真身，含**缓存自防护**。

    每份 ``--white <png>`` 是一层**删文字版**（同版式、把标注/刻度文字置白后重渲染）。派生缓存
    早于受审图即**过期**：给 ``--rebuild`` 则先重生成再复检，否则 rc=2 退出——绝不用过期缓存
    出读数（判法方法学见 ``occlusion`` 模块 docstring）。

    大图（≥4000 px 宽）全分辨率判据可能数分钟——运行期逐阶段打印进度（``flush``，重定向日志
    可 tail 观察）；``--downsample 2|4`` 先粗判（判据尺度与 dpi 写在报告头）。
    """
    img = Path(args.image)
    layers = [(Path(p), Path(p).stem) for p in args.white]
    check = _occ.check_caches(img, [p for p, _ in layers], rebuild=args.rebuild)
    scale = max(1, int(getattr(args, "downsample", 1) or 1))
    print(f"[overlap] 受审图 {img}", flush=True)
    print(f"[overlap] {check.message}", flush=True)
    if not check.ok:
        return 2
    try:
        full = _occ.load_or_raise(img)
    except OSError as e:
        print(f"[overlap] {e}", file=sys.stderr)
        return 2
    h0, w0 = full.shape[:2]
    dpi = _dpi_of(img)
    audited = _occ.downsample_rgb(full, scale)
    hh, ww = audited.shape[:2]
    dpi_note = f"@ {dpi:.0f} dpi" if dpi else "@ dpi 未知"
    if scale == 1:
        print(f"[overlap] 尺度：全分辨率 {w0}×{h0} px {dpi_note}", flush=True)
    else:
        eff = f"（≈{dpi / scale:.0f} dpi）" if dpi else ""
        print(
            f"[overlap] 尺度：原图 {w0}×{h0} px {dpi_note} → downsample=1/{scale}，"
            f"判据尺度 {ww}×{hh} px{eff}（粗判；终判请用 --downsample 1）",
            flush=True,
        )
    box = tuple(args.box) if args.box else None
    mm_per_px = None if args.mm_per_px is None else float(args.mm_per_px) * scale
    overall = True
    with _ProgressTicker() as ticker:
        for path, label in layers:

            def _stage(stage: str, _label: str = label) -> None:
                ticker.stage = f"{_label}：{stage}"
                print(
                    f"[overlap] … {_label}：{stage}"
                    f"（已 {time.monotonic() - ticker.t0:.0f}s）",
                    flush=True,
                )

            try:
                white = _occ.downsample_rgb(_occ.load_or_raise(path), scale)
                rep = _occ.render_ink_verdict(
                    audited,
                    white,
                    ink_tol=args.ink_tol,
                    chroma_min=args.chroma_min,
                    line_lum_max=args.line_lum_max,
                    line_chroma_max=args.line_chroma_max,
                    box=box,
                    progress=_stage,
                )
            except (OSError, ValueError) as e:
                print(f"[overlap] 层 {label}：{e}", file=sys.stderr)
                return 2
            d = rep["ink_line_min_px"]
            dmm = (
                " n/a"
                if (d is None or mm_per_px is None)
                else f" = {float(d) * mm_per_px:.2f}mm"
            )
            print(
                f"[overlap] 层 {label}（{path.name}）：墨迹={rep['ink_px']}px  "
                f"彩色数据区={rep['colored_px']}px  图线真身={rep['line_px']}px",
                flush=True,
            )
            print(
                f"    墨迹∩彩色数据区 = {rep['ink_colored_px']}px -> "
                f"{'PASS' if rep['ink_colored_px'] == 0 else 'FAIL'}",
                flush=True,
            )
            print(
                f"    墨迹∩图线真身   = {rep['ink_line_px']}px -> "
                f"{'PASS' if rep['ink_line_px'] == 0 else 'FAIL'}",
                flush=True,
            )
            print(
                f"    墨迹→图线真身 min = "
                f"{'n/a' if d is None else f'{float(d):.1f}px'}{dmm}"
                f"  @尺度 1/{scale}",
                flush=True,
            )
            overall = overall and bool(rep["pass"])
    print(f"[overlap] 总判：{'PASS' if overall else 'FAIL'}", flush=True)
    return 0 if overall else 1


# ---------------------------------------------------------------------------
# raster-panel（外部渲染栅格 + 数据坐标叠加）
# ---------------------------------------------------------------------------
def cmd_raster_panel(args: argparse.Namespace) -> int:
    overlays = _raster.load_overlays(args.overlay) if args.overlay else None
    try:
        out = _raster.compose_raster_panel(
            args.image,
            args.out,
            extent=tuple(args.extent) if args.extent else None,
            sidecar=args.sidecar,
            crop_box=tuple(args.crop) if args.crop else None,
            overlays=overlays,
            figsize=tuple(args.figsize),
            dpi=args.dpi,
            axis_off=not args.show_axis,
        )
    except ValueError as e:
        print(f"[figures] raster-panel 失败：{e}", file=sys.stderr)
        return 1
    print(f"composed -> {out}")
    print(f"  ✓ 视觉校验：Read '{out}'")
    return 0


# ---------------------------------------------------------------------------
# compose-grid（spec 驱动的多面板组装）
# ---------------------------------------------------------------------------
def cmd_compose_grid(args: argparse.Namespace) -> int:
    try:
        spec = _raster.load_grid_spec(args.spec)
    except (ValueError, OSError) as e:
        print(f"[figures] compose-grid 读取 spec 失败：{e}", file=sys.stderr)
        return 1
    try:
        out = _raster.compose_grid_to_file(spec, args.out, dpi=args.dpi)
    except ValueError as e:
        print(f"[figures] compose-grid 失败：{e}", file=sys.stderr)
        return 1
    print(f"composed -> {out}")
    print(f"  ✓ 视觉校验：Read '{out}'")
    return 0


# ---------------------------------------------------------------------------
# field-panel（数据驱动场渲染，colorbar 与面板严格一致）
# ---------------------------------------------------------------------------
def cmd_field_panel(args: argparse.Namespace) -> int:
    try:
        out = _field.render_field_panel(
            args.source,
            args.out,
            scalars=args.scalars,
            cols=tuple(args.cols) if args.cols else (0, 1, 2),
            cmap=args.cmap,
            vmin=args.vmin,
            vmax=args.vmax,
            shading=args.shading,
            figsize=tuple(args.figsize),
            dpi=args.dpi,
            colorbar=not args.no_colorbar,
            cbar_label=args.cbar_label,
            title=args.title,
            axis_off=args.axis_off,
        )
    except ValueError as e:
        print(f"[figures] field-panel 失败：{e}", file=sys.stderr)
        return 1
    print(f"rendered -> {out}")
    print(f"  ✓ 视觉校验：Read '{out}'")
    return 0


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
            # 批量列举：静默逐目录告警（warn=False），改用 [src]/[figdir] 侧标记呈现选用者。
            pipeline = _runner.discover_pipeline(d, warn=False)
        except FileNotFoundError:
            continue
        found = True
        side = "figdir" if pipeline.resolve().parent == d.resolve() else "src"
        out = d / "out"
        n_out = len(list(out.glob("*"))) if out.is_dir() else 0
        print(f"  {d.name:<28} pipeline={pipeline.name} [{side}]  out_files={n_out}")
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
    s = sub.add_parser(
        "new",
        help="脚手架一幅图的生产管线目录",
        epilog=(
            "new 收 <research> <slug> 两个位置参数；随后的 build/preview/audit 只收\n"
            "脚手架产出的单一 <figdir> 路径（见各子命令 -h）。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    s.add_argument("research", help="研究线名（不含数字前缀），如 gain_ep")
    s.add_argument("slug", help="图目录名，如 fig1_ep_band")
    s.add_argument(
        "--style", default=settings.default_style, help="期刊预设（aps/nature）"
    )
    s.add_argument("--width", default="double", help="设计宽度 single/double 或毫米数")
    s.add_argument(
        "--template",
        default=_scaffold.DEFAULT_TEMPLATE,
        help=f"模板：{', '.join(_scaffold.list_templates())}",
    )
    s.add_argument("--force", action="store_true", help="目录已存在时覆盖管线文件")
    s.set_defaults(func=cmd_new)

    # build
    s = sub.add_parser(
        "build",
        help="运行图管线并导出全部格式 + PNG 预览",
        epilog=_FIGDIR_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    s.add_argument("figdir", type=Path, metavar=_FIGDIR_METAVAR, help=_FIGDIR_HELP)
    s.add_argument(
        "--style", default=None, help="覆盖预设（默认取 STYLE.yaml / config）"
    )
    s.add_argument("--width", default=None, help="覆盖设计宽度")
    s.add_argument("--aspect", type=float, default=None, help="高/宽比（默认 0.618）")
    s.add_argument("--palette", default=None, help="调色板名（见 styles）")
    s.add_argument("--formats", default=None, help="逗号分隔交付格式（默认取 config）")
    s.add_argument("--stem", default=None, help="交付件主名（默认图目录名）")
    s.add_argument("--save-dpi", type=int, default=None, dest="save_dpi")
    s.add_argument("--preview-dpi", type=int, default=None, dest="preview_dpi")
    s.add_argument(
        "--pipeline-in-figdir",
        action="store_true",
        dest="pipeline_in_figdir",
        help="src 与 figdir 双侧同名管线并存时强制选用 figdir 侧（默认 src 优先并告警）",
    )
    s.set_defaults(func=cmd_build)

    # preview
    s = sub.add_parser(
        "preview",
        help="只渲染 PNG 预览（快速视觉迭代）",
        epilog=_FIGDIR_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    s.add_argument("figdir", type=Path, metavar=_FIGDIR_METAVAR, help=_FIGDIR_HELP)
    s.add_argument("--style", default=None)
    s.add_argument("--width", default=None)
    s.add_argument("--aspect", type=float, default=None)
    s.add_argument("--palette", default=None)
    s.add_argument("--stem", default=None)
    s.add_argument("--preview-dpi", type=int, default=None, dest="preview_dpi")
    s.add_argument(
        "--pipeline-in-figdir",
        action="store_true",
        dest="pipeline_in_figdir",
        help="src 与 figdir 双侧同名管线并存时强制选用 figdir 侧（默认 src 优先并告警）",
    )
    s.set_defaults(func=cmd_preview)

    # audit
    s = sub.add_parser(
        "audit",
        help="出版规范 + 标注几何语义自检（宽度/字号/嵌入/角标/色盲/计数/包含/大小/连线）",
        epilog=_FIGDIR_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    s.add_argument("figdir", type=Path, metavar=_FIGDIR_METAVAR, help=_FIGDIR_HELP)
    s.add_argument("--style", default=None)
    s.add_argument("--width", default=None)
    s.add_argument("--panels", action="store_true", help="要求多子图带 (a)(b)(c) 角标")
    s.add_argument("--min-fontsize", type=float, default=None, dest="min_fontsize")
    s.add_argument(
        "--annotations",
        type=Path,
        default=None,
        help="标注语义声明路径（默认自动查找图目录 ANNOTATIONS.yaml|yml|json）",
    )
    s.set_defaults(func=cmd_audit)

    # inspect（不依赖管线：对任意 PNG/JPG 体检；非 scaffold 产物的工具核验面）
    s = sub.add_parser(
        "inspect",
        help="图像体检（不依赖管线）：尺寸/色域占用/饱和像素占比/边缘水印·logo 检测",
        epilog="例：pysci-figures inspect raw/far_full_L.png",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    s.add_argument("image", type=Path, help="单张图片路径（PNG/JPG）")
    s.add_argument(
        "--band",
        type=int,
        default=None,
        help="边缘带宽度（px）；默认 max(4, 短边×2%%)",
    )
    s.set_defaults(func=cmd_inspect)

    # overlap（零遮挡像素判据：墨迹 ∩ 彩色数据区/图线真身；判据唯一实现，bench 调它）
    s = sub.add_parser(
        "overlap",
        help="零遮挡判据：文字墨迹∩彩色数据区/图线真身（含缓存自防护）",
        epilog="例：pysci-figures overlap out/x.png --white bench/white_field.png "
        "--rebuild 'uv run --no-sync python bench/make_white.py'\n"
        "耗时：4k 级成品全分辨率判据约需数分钟（大图先用 --downsample 2 粗判）；"
        "运行期逐阶段打印进度（已 flush，重定向日志可 tail 观察）。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    s.add_argument("image", type=Path, help="受审图（PNG/JPG）")
    s.add_argument(
        "--white",
        action="append",
        required=True,
        metavar="PNG",
        help="一层删文字版（同版式、文字置白后重渲染；可多次，标签取文件名 stem）",
    )
    s.add_argument("--ink-tol", type=int, default=_occ.INK_TOL, dest="ink_tol")
    s.add_argument(
        "--chroma-min", type=float, default=_occ.CHROMA_MIN, dest="chroma_min"
    )
    s.add_argument(
        "--line-lum-max", type=float, default=_occ.LINE_LUM_MAX, dest="line_lum_max"
    )
    s.add_argument(
        "--line-chroma-max",
        type=float,
        default=_occ.LINE_CHROMA_MAX,
        dest="line_chroma_max",
    )
    s.add_argument(
        "--mm-per-px",
        type=float,
        default=None,
        dest="mm_per_px",
        help="像素→毫米换算（给了才打印 mm；如 600dpi 用 0.042333）",
    )
    s.add_argument(
        "--box",
        type=int,
        nargs=4,
        default=None,
        metavar=("X0", "X1", "Y0", "Y1"),
        help="只在该裁切窗内判读（多面板图逐面板量）",
    )
    s.add_argument(
        "--rebuild",
        default=None,
        help="缓存过期/缺失时的重生成命令（shell 串）；不给则过期即 rc=2 报错",
    )
    s.add_argument(
        "--downsample",
        type=int,
        choices=(1, 2, 4),
        default=1,
        help="判据尺度降采样（点采样）：大图先 2/4 粗判，报告头写明尺度与 dpi；终判用 1",
    )
    s.set_defaults(func=cmd_overlap)

    # list
    s = sub.add_parser("list", help="列出某研究线下已有的图管线")
    s.add_argument("research", help="研究线名（不含数字前缀），如 gain_ep")
    s.set_defaults(func=cmd_list)

    # raster-panel
    s = sub.add_parser(
        "raster-panel", help="外部渲染栅格（COMSOL PNG）+ 数据坐标叠加原语 → 单面板 PNG"
    )
    s.add_argument(
        "--image",
        type=Path,
        required=True,
        help="渲染 PNG（如 comsol export image 产物）",
    )
    s.add_argument("--out", type=Path, required=True)
    s.add_argument(
        "--sidecar",
        type=Path,
        default=None,
        help="comsol 导出 sidecar json（提供 extent/crop_box）",
    )
    s.add_argument(
        "--extent",
        type=float,
        nargs=4,
        default=None,
        help="数据窗口 x0 x1 y0 y1（覆盖 sidecar）",
    )
    s.add_argument(
        "--crop",
        type=int,
        nargs=4,
        default=None,
        help="像素裁剪框 x0 y0 x1 y1（覆盖 sidecar）",
    )
    s.add_argument(
        "--overlay", type=Path, default=None, help="叠加原语清单 .json/.yaml"
    )
    s.add_argument("--figsize", type=float, nargs=2, default=(7.0, 7.0))
    s.add_argument("--dpi", type=int, default=150)
    s.add_argument(
        "--show-axis", action="store_true", dest="show_axis", help="保留面板坐标轴"
    )
    s.set_defaults(func=cmd_raster_panel)

    # compose-grid
    s = sub.add_parser(
        "compose-grid",
        help="spec 驱动的多面板组装（N×M 混合 raster/图像/轴 + 共享 colorbar/面板字母/行列标题）",
    )
    s.add_argument(
        "--spec",
        type=Path,
        required=True,
        help="组装 spec .yaml/.json（rows/cols/panels/colorbar…）",
    )
    s.add_argument("--out", type=Path, required=True, help="输出 PNG")
    s.add_argument("--dpi", type=int, default=None, help="覆盖 spec.dpi（默认 150）")
    s.set_defaults(func=cmd_compose_grid)

    # field-panel
    s = sub.add_parser(
        "field-panel",
        help="数据驱动场渲染（VTK/CSV → tripcolor，colorbar 与面板颜色严格一致）",
    )
    s.add_argument(
        "--source", type=Path, required=True, help="场源 .vtk/.vtu/.csv/.txt"
    )
    s.add_argument("--out", type=Path, required=True)
    s.add_argument(
        "--scalars", default=None, help="VTK 标量数组名（默认首个 point_data）"
    )
    s.add_argument(
        "--cols",
        type=int,
        nargs=3,
        default=None,
        metavar=("XI", "YI", "VI"),
        help="CSV 的 x/y/value 列索引（默认 0 1 2）",
    )
    s.add_argument(
        "--cmap",
        default="bwr",
        help="matplotlib colormap（与共享 colorbar 同名即一致）",
    )
    s.add_argument("--vmin", type=float, default=None)
    s.add_argument("--vmax", type=float, default=None)
    s.add_argument("--shading", default="gouraud", choices=["gouraud", "flat"])
    s.add_argument("--figsize", type=float, nargs=2, default=(6.0, 5.0))
    s.add_argument("--dpi", type=int, default=150)
    s.add_argument(
        "--no-colorbar", action="store_true", dest="no_colorbar", help="不画 colorbar"
    )
    s.add_argument("--cbar-label", default=None, dest="cbar_label")
    s.add_argument("--title", default=None)
    s.add_argument(
        "--axis-off", action="store_true", dest="axis_off", help="关闭坐标轴"
    )
    s.set_defaults(func=cmd_field_panel)

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
