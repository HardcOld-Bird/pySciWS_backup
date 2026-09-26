"""图生产管线的发现与运行。

**管线约定**：每幅论文插图的代码与数据分离存放：

代码（Agent 管理）::

    src/pysci/research/<name>/article/figures/<figN_slug>.py   # 管线模块

数据（产物/日志）::

    data/research/<n>_<name>/article/figures/<figN_slug>/
    ├── notes.md         # 迭代日志（可选）
    └── out/             # 导出产物（save_figure 自动创建）

管线模块须暴露::

    def build_figure(**kwargs) -> matplotlib.figure.Figure:
        ...

``build_figure`` 内部用 ``plt.subplots()`` 或 ``layout.grid()`` 建图即可——运行器会在
``style_context`` 里调用它，故 figsize / 字体 / 配色 / 字号已按期刊预设生效。运行器会
自省 ``build_figure`` 的签名，只传入其接受的关键字参数（如 ``style``、``data`` 等），
因此管线可按需声明入参，未声明的会被安全忽略。

**风格绑定**：``figures`` 数据根目录下的 ``STYLE.yaml`` 可固定默认预设/宽度，
命令行 ``--style/--width`` 优先级更高。缺失时用技能默认（config.settings）。

**后向兼容**：若 src/ 下未找到管线，仍会回退到旧模式（在 figdir 内发现 .py）。
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import matplotlib

from pysci.paths import PROJECT_ROOT, research_asset_dir

from . import export as _export
from .config import settings
from .style import StyleInfo, style_context

# 优先作为管线入口的文件名（按顺序）。
_PIPELINE_NAME_PRIORITY = ("fig.py", "build.py", "main.py")


def figures_root(research: str) -> Path:
    """解析某研究线的插图数据根目录：``data/research/<n>_<name>/article/figures``。"""
    return research_asset_dir(research) / "article" / "figures"


def figures_code_root(research: str) -> Path:
    """解析某研究线的插图代码根目录：``src/pysci/research/<name>/article/figures``。"""
    return PROJECT_ROOT / "src" / "pysci" / "research" / research / "article" / "figures"


def _research_from_figdir(figdir: Path) -> tuple[str, str] | None:
    """从 figdir 路径反推 (research_name, slug)。

    figdir 形如 ``data/research/1_gain_ep/article/figures/fig1_ep_band``。
    返回 ``("gain_ep", "fig1_ep_band")`` 或 None（无法解析时）。
    """
    parts = figdir.resolve().parts
    # 找到 "research" 在路径中的位置
    try:
        idx = parts.index("research")
    except ValueError:
        return None
    if idx + 1 >= len(parts):
        return None
    research_dir_name = parts[idx + 1]  # e.g. "1_gain_ep"
    # 去掉数字前缀
    if "_" in research_dir_name:
        research_name = research_dir_name.split("_", 1)[1]
    else:
        research_name = research_dir_name
    slug = figdir.name
    return research_name, slug


def discover_pipeline(figdir: Path | str) -> Path:
    """发现图管线模块（.py）。

    搜索顺序：
    1. src/ 代码目录：``src/pysci/research/<name>/article/figures/<slug>.py``
    2. 旧模式回退：figdir 内的 ``<目录名>.py`` → fig.py/build.py/main.py → 唯一 .py

    Raises:
        FileNotFoundError: 找不到管线模块。
    """
    figdir = Path(figdir)

    # --- 新位置：src/ 代码目录 ---
    parsed = _research_from_figdir(figdir)
    if parsed:
        research_name, slug = parsed
        code_root = figures_code_root(research_name)
        src_pipeline = code_root / f"{slug}.py"
        if src_pipeline.is_file():
            return src_pipeline

    # --- 旧位置回退：figdir 内 ---
    if not figdir.is_dir():
        raise FileNotFoundError(f"图目录不存在：{figdir}")

    named = figdir / f"{figdir.name}.py"
    if named.is_file():
        return named
    for cand in _PIPELINE_NAME_PRIORITY:
        p = figdir / cand
        if p.is_file():
            return p
    pys = sorted(
        p for p in figdir.glob("*.py") if not p.name.startswith("_")
    )
    if len(pys) == 1:
        return pys[0]
    if not pys:
        raise FileNotFoundError(
            f"{figdir} 下没有找到管线 .py 模块，"
            f"且 src/ 代码目录中也未找到对应脚本"
        )
    names = ", ".join(p.name for p in pys)
    raise FileNotFoundError(
        f"{figdir} 下有多个 .py（{names}），无法确定入口；"
        f"请重命名为 {figdir.name}.py 或 fig.py"
    )


def load_style_config(figdir: Path | str) -> dict[str, Any]:
    """读取 STYLE.yaml（图目录优先，其次 figures 根目录）。

    Returns:
        含 ``style`` / ``width`` / ``aspect`` / ``palette`` 等键的 dict；无配置或无 pyyaml 时为空。
    """
    figdir = Path(figdir)
    candidates = [figdir / "STYLE.yaml", figdir.parent / "STYLE.yaml"]
    for path in candidates:
        if path.is_file():
            try:
                import yaml  # type: ignore
            except ImportError:
                print(
                    "[sciplot.runner] WARNING: 找到 STYLE.yaml 但 pyyaml 不可用，已忽略",
                    file=sys.stderr,
                )
                return {}
            with path.open(encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            return data if isinstance(data, dict) else {}
    return {}


def _import_pipeline(module_path: Path) -> Any:
    """按文件路径导入管线模块（把其所在目录加入 sys.path 以便导入同目录辅助模块）。"""
    module_path = module_path.resolve()
    parent = str(module_path.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    mod_name = f"_sciplot_pipeline_{module_path.stem}_{abs(hash(str(module_path)))%10**8}"
    spec = importlib.util.spec_from_file_location(mod_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法为 {module_path} 创建 import spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


def _call_build_figure(module: Any, extra_kwargs: dict[str, Any]) -> matplotlib.figure.Figure:
    """调用管线的 ``build_figure``，只传它签名里接受的关键字参数。"""
    fn = getattr(module, "build_figure", None)
    if fn is None or not callable(fn):
        raise AttributeError(
            "管线模块必须定义 build_figure(**kwargs) -> matplotlib Figure"
        )
    sig = inspect.signature(fn)
    accepts_var_kw = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    if accepts_var_kw:
        kwargs = dict(extra_kwargs)
    else:
        allowed = {
            name
            for name, p in sig.parameters.items()
            if p.kind
            in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        }
        kwargs = {k: v for k, v in extra_kwargs.items() if k in allowed}
    fig = fn(**kwargs)
    if not isinstance(fig, matplotlib.figure.Figure):
        raise TypeError(
            f"build_figure 必须返回 matplotlib Figure，实际返回 {type(fig)!r}"
        )
    return fig


@dataclass
class RunResult:
    """一次"运行管线 + 导出"的完整结果。"""

    figdir: Path
    pipeline: Path
    stem: str
    style_info: StyleInfo | None = None
    export: _export.ExportResult | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.export is not None and self.export.ok

    def report(self) -> str:
        lines = [
            f"figdir   : {self.figdir}",
            f"pipeline : {self.pipeline.name}",
            f"stem     : {self.stem}",
        ]
        if self.style_info is not None:
            lines.append("--- style ---")
            lines.append(self.style_info.report())
        if self.export is not None:
            lines.append("--- export ---")
            lines.append(self.export.report())
        if self.error:
            lines.append(f"ERROR: {self.error}")
        return "\n".join(lines)


def build_figure_dir(
    figdir: Path | str,
    *,
    style: str | None = None,
    width: str | None = None,
    aspect: float | None = None,
    palette: str | None = None,
    formats: tuple[str, ...] | None = None,
    stem: str | None = None,
    save_dpi: int | None = None,
    preview_dpi: int | None = None,
    extra_kwargs: dict[str, Any] | None = None,
) -> RunResult:
    """运行某图目录的管线并在 style_context 下导出全部格式 + 预览。

    优先级：显式入参 > STYLE.yaml > config.settings 默认。

    Args:
        figdir: 图生产管线目录。
        style / width / aspect / palette: 风格覆盖（None 表示交给下层默认）。
        formats: 导出格式（None -> settings.default_formats）。
        stem: 交付件主名（None -> 图目录名）。
        save_dpi / preview_dpi: dpi 覆盖。
        extra_kwargs: 透传给 build_figure 的额外关键字（如数据路径切换）。
    """
    figdir = Path(figdir)
    result = RunResult(figdir=figdir, pipeline=figdir, stem=stem or figdir.name)
    try:
        pipeline = discover_pipeline(figdir)
        result.pipeline = pipeline

        cfg = load_style_config(figdir)
        eff_style = style or cfg.get("style") or settings.default_style
        eff_width = width or cfg.get("width") or "double"
        eff_aspect = aspect if aspect is not None else float(cfg.get("aspect", 0.618))
        eff_palette = palette or cfg.get("palette") or "okabe-ito"
        eff_formats = tuple(formats) if formats else tuple(settings.default_formats)
        eff_stem = stem or cfg.get("stem") or figdir.name
        result.stem = eff_stem

        module = _import_pipeline(pipeline)
        out_dir = figdir / (cfg.get("out_subdir") or "out")
        build_kwargs = dict(extra_kwargs or {})
        build_kwargs.setdefault("research_dir", figures_root_from_figdir(figdir))

        with style_context(
            eff_style, width=eff_width, aspect=eff_aspect, palette=eff_palette
        ) as info:
            result.style_info = info
            build_kwargs["style"] = info
            fig = _call_build_figure(module, build_kwargs)
            result.export = _export.save_figure(
                fig,
                out_dir,
                eff_stem,
                formats=eff_formats,
                save_dpi=save_dpi or settings.save_dpi,
                preview_dpi=preview_dpi or settings.preview_dpi,
                close=True,
            )
    except Exception as e:  # noqa: BLE001 - CLI 需要结构化捕获并报告
        result.error = repr(e)
    return result


def preview_figure_dir(
    figdir: Path | str,
    *,
    style: str | None = None,
    width: str | None = None,
    aspect: float | None = None,
    palette: str | None = None,
    stem: str | None = None,
    preview_dpi: int | None = None,
    extra_kwargs: dict[str, Any] | None = None,
) -> RunResult:
    """只渲染 PNG 预览（不产出交付件），用于快速视觉迭代。"""
    return build_figure_dir(
        figdir,
        style=style,
        width=width,
        aspect=aspect,
        palette=palette,
        formats=(),  # 不导出交付件，save_figure 仍会产出预览
        stem=stem,
        preview_dpi=preview_dpi,
        extra_kwargs=extra_kwargs,
    )


def figures_root_from_figdir(figdir: Path | str) -> Path:
    """由图目录反推其所属 figures 根目录（figdir 的父目录）。"""
    return Path(figdir).parent


@contextmanager
def built_figure(
    figdir: Path | str,
    *,
    style: str | None = None,
    width: str | None = None,
    aspect: float | None = None,
    palette: str | None = None,
    extra_kwargs: dict[str, Any] | None = None,
) -> Iterator[tuple[matplotlib.figure.Figure, StyleInfo]]:
    """在 style_context 下构建图管线，产出 ``(fig, StyleInfo)`` 但不导出、不关闭。

    供 audit 等需要在图对象上做检查的场景复用（build_figure_dir 则在此基础上再导出）。
    退出上下文时关闭图，释放内存。
    """
    figdir = Path(figdir)
    pipeline = discover_pipeline(figdir)
    cfg = load_style_config(figdir)
    eff_style = style or cfg.get("style") or settings.default_style
    eff_width = width or cfg.get("width") or "double"
    eff_aspect = aspect if aspect is not None else float(cfg.get("aspect", 0.618))
    eff_palette = palette or cfg.get("palette") or "okabe-ito"

    module = _import_pipeline(pipeline)
    build_kwargs = dict(extra_kwargs or {})
    build_kwargs.setdefault("research_dir", figures_root_from_figdir(figdir))

    with style_context(
        eff_style, width=eff_width, aspect=eff_aspect, palette=eff_palette
    ) as info:
        build_kwargs["style"] = info
        fig = _call_build_figure(module, build_kwargs)
        try:
            yield fig, info
        finally:
            matplotlib.pyplot.close(fig)
