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

**风格绑定**：``figures`` 数据根目录下的 ``STYLE.yaml`` 可固定默认预设/宽度，图目录级
``STYLE.yaml`` **逐键覆盖**根级（只声明差异键即可，其余继承）；命令行 ``--style/--width``
优先级最高。缺失时用技能默认（config.settings）。

**后向兼容**：若 src/ 下未找到管线，仍会回退到旧模式（在 figdir 内发现 .py）。

**双侧同名冲突（假绿防护）**：src 与 figdir 同时存在同名管线时，默认选用 src 侧，但会
打印 WARNING 指明**实际选用的文件**与被忽略者——避免「渲染了 src 占位管线却静默显示成功」。
可用 CLI ``--pipeline-in-figdir`` 或 STYLE.yaml ``pipeline: src|figdir`` 显式改选。
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

from pysci.paths import PROJECT_ROOT, assert_within_data, research_fig_dir

from . import export as _export
from .config import settings
from .style import StyleInfo, style_context

# 优先作为管线入口的文件名（按顺序）。
_PIPELINE_NAME_PRIORITY = ("fig.py", "build.py", "main.py")


def figures_root(research: str) -> Path:
    """解析某研究线的插图数据根目录：``data/research/<n>_<name>/article/figures``。

    委托 :func:`pysci.paths.research_fig_dir`（规范解析器）。
    """
    return research_fig_dir(research)


def figures_code_root(research: str) -> Path:
    """解析某研究线的插图代码根目录：``src/pysci/research/<name>/article/figures``。"""
    return (
        PROJECT_ROOT / "src" / "pysci" / "research" / research / "article" / "figures"
    )


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


def _rel_to_root(path: Path) -> str:
    """把路径显示为相对 PROJECT_ROOT（在其外时回退绝对路径）——供报告/告警消歧。"""
    try:
        return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def _warn(msg: str) -> None:
    """统一的 runner 告警出口（stderr）。"""
    print(f"[sciplot.runner] WARNING: {msg}", file=sys.stderr)


def _discover_src_pipeline(figdir: Path) -> Path | None:
    """src 代码侧管线：``src/pysci/research/<name>/article/figures/<slug>.py``；无则 None。"""
    parsed = _research_from_figdir(figdir)
    if not parsed:
        return None
    research_name, slug = parsed
    src_pipeline = figures_code_root(research_name) / f"{slug}.py"
    return src_pipeline if src_pipeline.is_file() else None


def _discover_figdir_pipeline(figdir: Path, *, strict: bool = False) -> Path | None:
    """figdir 数据侧管线（旧模式）：``<目录名>.py`` → fig.py/build.py/main.py → 唯一 .py。

    Args:
        figdir: 图数据目录。
        strict: True 时只认**显式入口**（``<目录名>.py`` 与 fig.py/build.py/main.py），
            不含「唯一 .py」宽松回退——用于 src/figdir 双侧冲突判定，避免把辅助 .py
            误判为管线而产生假告警。

    Returns:
        命中的管线路径；找不到返回 None。
    """
    if not figdir.is_dir():
        return None
    named = figdir / f"{figdir.name}.py"
    if named.is_file():
        return named
    for cand in _PIPELINE_NAME_PRIORITY:
        p = figdir / cand
        if p.is_file():
            return p
    if strict:
        return None
    pys = sorted(p for p in figdir.glob("*.py") if not p.name.startswith("_"))
    if len(pys) == 1:
        return pys[0]
    return None


def _resolve_prefer(cli_prefer: str | None, cfg: dict[str, Any]) -> str | None:
    """合并管线侧偏好：CLI（``--pipeline-in-figdir``）> STYLE.yaml ``pipeline:`` > 自动。

    取值归一为 ``"src"`` / ``"figdir"`` / None；未知值告警并忽略（按自动处理）。
    """
    raw = cli_prefer or cfg.get("pipeline")
    if raw is None:
        return None
    val = str(raw).strip().lower()
    if val in ("src", "figdir"):
        return val
    _warn(f"忽略未知的 pipeline 偏好 {raw!r}（应为 src|figdir），按自动处理")
    return None


def discover_pipeline(
    figdir: Path | str, *, prefer: str | None = None, warn: bool = True
) -> Path:
    """发现图管线模块（.py）。

    搜索顺序（src 优先）：
    1. src/ 代码目录：``src/pysci/research/<name>/article/figures/<slug>.py``
    2. 旧模式回退：figdir 内的 ``<目录名>.py`` → fig.py/build.py/main.py → 唯一 .py

    **双侧同名冲突**（src 与 figdir 均有显式入口）：默认选用 src 侧，但打印 WARNING 指明
    实际选用者与被忽略者——消除「渲染了占位管线却静默成功」的假绿。``prefer`` 可显式改选。

    Args:
        figdir: 图数据目录。
        prefer: 冲突时的选侧偏好（``"figdir"`` / ``"src"`` / None=默认 src 优先）；
            仅当偏好侧缺失时回退另一侧并告警。
        warn: 是否打印冲突/回退 WARNING（``list`` 等批量场景可传 False 静默）。

    Raises:
        FileNotFoundError: 找不到管线模块。
    """
    figdir = Path(figdir)
    src_pipe = _discover_src_pipeline(figdir)
    # 冲突判定用 strict（只认显式入口），避免把 figdir 内辅助 .py 误判为管线。
    fig_strict = _discover_figdir_pipeline(figdir, strict=True)

    # --- 双侧同名冲突：显式选侧 + 告警（假绿防护的核心）---
    if src_pipe and fig_strict:
        choice = "figdir" if prefer == "figdir" else "src"
        chosen = fig_strict if choice == "figdir" else src_pipe
        ignored = src_pipe if choice == "figdir" else fig_strict
        if warn:
            _warn(
                f"同名图管线在 src 与 figdir 双侧并存，已选用 {choice} 侧："
                f"{_rel_to_root(chosen)}\n"
                f"  被忽略：{_rel_to_root(ignored)}\n"
                "  改用另一侧：build/preview 加 --pipeline-in-figdir（强制 figdir），"
                "或 STYLE.yaml 设 pipeline: figdir|src"
            )
        return chosen

    # --- 单侧命中 ---
    fig_pipe = fig_strict or _discover_figdir_pipeline(figdir)
    if src_pipe:
        if prefer == "figdir" and warn:
            _warn(
                f"指定了 figdir 侧管线但未找到，回退 src 侧：{_rel_to_root(src_pipe)}"
            )
        return src_pipe
    if fig_pipe:
        if prefer == "src" and warn:
            _warn(
                f"指定了 src 侧管线但未找到，回退 figdir 侧：{_rel_to_root(fig_pipe)}"
            )
        return fig_pipe

    # --- 两侧皆无：保留原有区分性错误信息 ---
    if not figdir.is_dir():
        raise FileNotFoundError(f"图目录不存在：{figdir}")
    pys = sorted(p for p in figdir.glob("*.py") if not p.name.startswith("_"))
    if pys:
        names = ", ".join(p.name for p in pys)
        raise FileNotFoundError(
            f"{figdir} 下有多个 .py（{names}），无法确定入口；"
            f"请重命名为 {figdir.name}.py 或 fig.py"
        )
    raise FileNotFoundError(
        f"{figdir} 下没有找到管线 .py 模块，且 src/ 代码目录中也未找到对应脚本"
    )


def _read_style_yaml(path: Path) -> dict[str, Any]:
    """读取单个 STYLE.yaml 为 dict（不存在 / 无 pyyaml / 非 dict → 空 dict）。"""
    if not path.is_file():
        return {}
    try:
        import yaml  # type: ignore
    except ImportError:
        _warn(f"找到 {path.name} 但 pyyaml 不可用，已忽略")
        return {}
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data if isinstance(data, dict) else {}


def load_style_config(figdir: Path | str) -> dict[str, Any]:
    """读取 STYLE.yaml，**逐键合并**：figures 根目录为基线，图目录覆盖同名键。

    此前实现取「第一个存在的文件」整体返回——图目录级 STYLE.yaml 一旦存在就丢弃根级
    其余键；反之图目录无文件时，根级 ``width: double`` 会整体掩盖单栏图的宽度判定
    （假绿，见 backlog 20261009-audit-width-masked）。逐键合并后，图目录只需声明差异键
    （如 ``width: single``），其余仍继承根级默认；无图目录文件时行为与旧版一致。

    Returns:
        含 ``style`` / ``width`` / ``aspect`` / ``palette`` 等键的 dict；无配置或无 pyyaml 时为空。
    """
    figdir = Path(figdir)
    cfg = _read_style_yaml(figdir.parent / "STYLE.yaml")  # 根级基线
    cfg.update(_read_style_yaml(figdir / "STYLE.yaml"))  # 图目录覆盖同名键
    return cfg


def _import_pipeline(module_path: Path) -> Any:
    """按文件路径导入管线模块（把其所在目录加入 sys.path 以便导入同目录辅助模块）。"""
    module_path = module_path.resolve()
    parent = str(module_path.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    mod_name = (
        f"_sciplot_pipeline_{module_path.stem}_{abs(hash(str(module_path))) % 10**8}"
    )
    spec = importlib.util.spec_from_file_location(mod_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法为 {module_path} 创建 import spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


def _call_build_figure(
    module: Any, extra_kwargs: dict[str, Any]
) -> matplotlib.figure.Figure:
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
            f"pipeline : {_rel_to_root(self.pipeline)}",
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
    prefer: str | None = None,
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
        prefer: src/figdir 双侧同名管线并存时的选侧偏好（CLI ``--pipeline-in-figdir``
            传 ``"figdir"``）；None 时读 STYLE.yaml ``pipeline:`` 键，仍无则 src 优先。
    """
    figdir = Path(figdir)
    result = RunResult(figdir=figdir, pipeline=figdir, stem=stem or figdir.name)
    try:
        # 产物路径护栏：图目录必须落在 data/ 根内，stray（scripts/、仓库外）在写入前即报错。
        assert_within_data(figdir, what="图目录")
        # cfg 先于发现读取：STYLE.yaml 的 pipeline: 键参与选侧（CLI prefer 优先）。
        cfg = load_style_config(figdir)
        pipeline = discover_pipeline(figdir, prefer=_resolve_prefer(prefer, cfg))
        result.pipeline = pipeline

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
    prefer: str | None = None,
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
        prefer=prefer,
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
    prefer: str | None = None,
) -> Iterator[tuple[matplotlib.figure.Figure, StyleInfo]]:
    """在 style_context 下构建图管线，产出 ``(fig, StyleInfo)`` 但不导出、不关闭。

    供 audit 等需要在图对象上做检查的场景复用（build_figure_dir 则在此基础上再导出）。
    退出上下文时关闭图，释放内存。``prefer`` 语义同 build_figure_dir（audit 借此与 build
    选用同一管线，避免审查对象与产出对象不一致）。
    """
    figdir = Path(figdir)
    cfg = load_style_config(figdir)
    pipeline = discover_pipeline(figdir, prefer=_resolve_prefer(prefer, cfg))
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
