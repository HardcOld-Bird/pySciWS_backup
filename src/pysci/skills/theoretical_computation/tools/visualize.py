"""探索性可视化：matplotlib 快速图 + pyvista 3D 离屏渲染 + PNG 导出。

吸收旧 ``pysci.common.plotting``（中文字体配置）和 ``pysci.common.matrix``/``space_curve``
中的绑图逻辑，提供面向理论探索阶段的轻量级可视化（不追求出版规范）。

设计原则：
- 所有函数默认在 Agg/off_screen 模式下渲染（无头，不弹窗）。
- 所有函数返回 fig/plotter 对象供进一步定制。
- 若指定 out_path，自动保存 PNG 并打印路径（供 Agent Read 视觉反馈）。
- 需要出版级插图时，交接给 scientific_plotting 技能。

用法::

    from pysci.skills.theoretical_computation.tools import visualize

    fig = visualize.quick_plot_2d(x, y, labels=("kappa", "Re(omega)"))
    visualize.export_exploration(fig, session_dir / "plots" / "band.png")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from .config import settings

# 确保无头后端
matplotlib.use(settings.plot_backend)


# ---------------------------------------------------------------------------
# 风格设置
# ---------------------------------------------------------------------------
def setup_style(
    chinese_fonts: bool = True,
    grid: bool = True,
    figsize: tuple[float, float] = (10, 7),
) -> None:
    """配置探索图风格（清晰易读，不追求出版规范）。

    Args:
        chinese_fonts: 是否启用中文字体支持。
        grid: 是否默认显示网格。
        figsize: 默认图像尺寸。
    """
    if chinese_fonts:
        plt.rcParams["font.sans-serif"] = [
            "Microsoft JhengHei",
            "Microsoft YaHei",
            "SimHei",
            "SimSun",
            "DejaVu Sans",
        ]
        plt.rcParams["axes.unicode_minus"] = False

    plt.rcParams["figure.figsize"] = figsize
    plt.rcParams["axes.grid"] = grid
    plt.rcParams["grid.alpha"] = 0.3
    plt.rcParams["lines.linewidth"] = 1.8
    plt.rcParams["font.size"] = 11


# 模块加载时自动设置风格
setup_style()


# ---------------------------------------------------------------------------
# 导出
# ---------------------------------------------------------------------------
def export_exploration(
    fig_or_plotter: Any,
    out_path: Path | str,
    dpi: int | None = None,
    close: bool = True,
) -> Path:
    """统一导出探索图为 PNG（Agent 视觉反馈用）。

    支持 matplotlib Figure 和 pyvista Plotter。

    Args:
        fig_or_plotter: matplotlib Figure 或 pyvista Plotter 对象。
        out_path: 输出 PNG 路径。
        dpi: 分辨率，默认使用配置值。
        close: 导出后是否关闭 figure/plotter。

    Returns:
        保存的文件路径。
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if dpi is None:
        dpi = settings.default_plot_dpi

    if isinstance(fig_or_plotter, Figure):
        fig_or_plotter.savefig(
            str(out_path), dpi=dpi, bbox_inches="tight", facecolor="white"
        )
        if close:
            plt.close(fig_or_plotter)
    else:
        # 假设是 pyvista Plotter
        fig_or_plotter.screenshot(str(out_path))
        if close:
            fig_or_plotter.close()

    print(f"[visualize] 探索图已导出: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# 2D 快速图
# ---------------------------------------------------------------------------
def quick_plot_2d(
    x: NDArray,
    y: NDArray | list[NDArray],
    labels: tuple[str, str] = ("x", "y"),
    title: str = "",
    series_labels: list[str] | None = None,
    figsize: tuple[float, float] = (10, 6),
    out_path: Path | None = None,
) -> Figure:
    """快速 2D 线图/散点图。

    Args:
        x: x 轴数据。
        y: y 轴数据（单条线或多条线的列表）。
        labels: (x_label, y_label)。
        title: 图标题。
        series_labels: 各条线的图例标签。
        figsize: 图像尺寸。
        out_path: 若指定，自动导出 PNG。

    Returns:
        matplotlib Figure。
    """
    fig, ax = plt.subplots(figsize=figsize)

    if isinstance(y, np.ndarray) and y.ndim == 1:
        y = [y]

    for i, yi in enumerate(y):
        label = series_labels[i] if series_labels and i < len(series_labels) else None
        ax.plot(x, np.real(yi), linewidth=2, label=label)

    ax.set_xlabel(labels[0], fontsize=12)
    ax.set_ylabel(labels[1], fontsize=12)
    if title:
        ax.set_title(title, fontsize=13)
    if series_labels:
        ax.legend(fontsize=10)

    fig.tight_layout()

    if out_path:
        export_exploration(fig, out_path, close=False)

    return fig


def quick_plot_complex(
    x: NDArray,
    y: NDArray[np.complexfloating] | list[NDArray[np.complexfloating]],
    labels: tuple[str, str] = ("x", "value"),
    title: str = "",
    series_labels: list[str] | None = None,
    figsize: tuple[float, float] = (12, 5),
    out_path: Path | None = None,
) -> Figure:
    """快速复数值图（实部 + 虚部双子图）。

    Args:
        x: 参数数组。
        y: 复数值数组（单条或多条）。
        labels: (x_label, y_label_base)。
        title: 图标题。
        series_labels: 各条线的图例标签。
        figsize: 图像尺寸。
        out_path: 若指定，自动导出 PNG。

    Returns:
        matplotlib Figure。
    """
    fig, (ax_re, ax_im) = plt.subplots(1, 2, figsize=figsize)

    if isinstance(y, np.ndarray) and y.ndim == 1:
        y = [y]

    for i, yi in enumerate(y):
        label = series_labels[i] if series_labels and i < len(series_labels) else None
        ax_re.plot(x, np.real(yi), linewidth=2, label=label)
        ax_im.plot(x, np.imag(yi), linewidth=2, label=label)

    ax_re.set_xlabel(labels[0], fontsize=12)
    ax_re.set_ylabel(f"Re({labels[1]})", fontsize=12)
    ax_re.set_title("实部", fontsize=12)

    ax_im.set_xlabel(labels[0], fontsize=12)
    ax_im.set_ylabel(f"Im({labels[1]})", fontsize=12)
    ax_im.set_title("虚部", fontsize=12)

    if title:
        fig.suptitle(title, fontsize=14, y=0.98)
    if series_labels:
        ax_re.legend(fontsize=10)
        ax_im.legend(fontsize=10)

    fig.tight_layout()

    if out_path:
        export_exploration(fig, out_path, close=False)

    return fig


def quick_plot_complex_plane(
    eigenvalues: NDArray[np.complexfloating],
    param_values: NDArray[np.floating] | None = None,
    title: str = "复平面本征值轨迹",
    labels: tuple[str, str] = ("Re(ω)", "Im(ω)"),
    figsize: tuple[float, float] = (8, 8),
    out_path: Path | None = None,
    mark_ep: bool = False,
) -> Figure:
    """复平面本征值轨迹图。

    Args:
        eigenvalues: 形状 (n_params, n_eigenvalues) 的复数本征值数组。
        param_values: 参数值（用于颜色映射）。
        title: 图标题。
        labels: (x_label, y_label)。
        figsize: 图像尺寸。
        out_path: 若指定，自动导出 PNG。
        mark_ep: 是否标记 EP 点（本征值最近处）。

    Returns:
        matplotlib Figure。
    """
    fig, ax = plt.subplots(figsize=figsize)

    n_eig = eigenvalues.shape[1] if eigenvalues.ndim > 1 else 1
    colors = plt.cm.tab10(np.linspace(0, 1, n_eig))

    for j in range(n_eig):
        vals = eigenvalues[:, j] if eigenvalues.ndim > 1 else eigenvalues
        if param_values is not None:
            sc = ax.scatter(
                np.real(vals), np.imag(vals),
                c=param_values, cmap="viridis", s=8, zorder=3,
            )
            plt.colorbar(sc, ax=ax, label="参数值")
        else:
            ax.plot(np.real(vals), np.imag(vals), "-", color=colors[j],
                    linewidth=1.5, label=f"λ_{j+1}")
            ax.scatter(np.real(vals), np.imag(vals), s=3, color=colors[j])

    # 标记 EP（本征值最接近的点）
    if mark_ep and eigenvalues.ndim > 1 and eigenvalues.shape[1] >= 2:
        dists = np.abs(eigenvalues[:, 0] - eigenvalues[:, 1])
        ep_idx = int(np.argmin(dists))
        ep_val = eigenvalues[ep_idx, 0]
        ax.scatter(
            [np.real(ep_val)], [np.imag(ep_val)],
            color="red", s=200, marker="*", zorder=5,
            label=f"EP (d={dists[ep_idx]:.2e})",
        )

    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.set_xlabel(labels[0], fontsize=12)
    ax.set_ylabel(labels[1], fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.set_aspect("equal")
    if param_values is None:
        ax.legend(fontsize=10)

    fig.tight_layout()

    if out_path:
        export_exploration(fig, out_path, close=False)

    return fig


# ---------------------------------------------------------------------------
# 数据文件快速可视化
# ---------------------------------------------------------------------------
def quick_plot_npz(
    data: dict[str, NDArray],
    title: str = "",
    max_plots: int = 4,
    figsize: tuple[float, float] = (12, 8),
) -> Figure:
    """快速可视化 .npz 文件中的数组数据。

    自动检测 1D/2D 数组并选择合适的图类型。

    Args:
        data: 键值对（从 np.load 得到）。
        title: 图标题。
        max_plots: 最多绘制几个数组。
        figsize: 图像尺寸。

    Returns:
        matplotlib Figure。
    """
    # 筛选可绘制的数组
    plottable = {
        k: v for k, v in data.items()
        if isinstance(v, np.ndarray) and v.ndim in (1, 2) and v.size > 1
    }
    keys = list(plottable.keys())[:max_plots]
    n = len(keys)

    if n == 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "无可视化数据", ha="center", va="center", fontsize=14)
        return fig

    cols = min(n, 2)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)

    for idx, key in enumerate(keys):
        ax = axes[idx // cols][idx % cols]
        arr = plottable[key]

        if arr.ndim == 1:
            if np.iscomplexobj(arr):
                ax.plot(np.real(arr), label="Re", linewidth=1.5)
                ax.plot(np.imag(arr), label="Im", linewidth=1.5, linestyle="--")
                ax.legend(fontsize=9)
            else:
                ax.plot(arr, linewidth=1.5)
            ax.set_title(key, fontsize=11)
        else:  # 2D
            if np.iscomplexobj(arr):
                im = ax.imshow(np.abs(arr), aspect="auto", origin="lower")
            else:
                im = ax.imshow(arr, aspect="auto", origin="lower")
            plt.colorbar(im, ax=ax, fraction=0.046)
            ax.set_title(key, fontsize=11)

    # 隐藏多余的子图
    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=13, y=0.98)
    fig.tight_layout()
    return fig


def quick_plot_dataframe(
    df: Any,
    title: str = "",
    max_cols: int = 6,
    figsize: tuple[float, float] = (12, 6),
) -> Figure:
    """快速可视化 DataFrame 的数值列。

    Args:
        df: pandas DataFrame。
        title: 图标题。
        max_cols: 最多绘制几列。
        figsize: 图像尺寸。

    Returns:
        matplotlib Figure。
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns[:max_cols]
    fig, ax = plt.subplots(figsize=figsize)

    for col in numeric_cols:
        ax.plot(df.index, df[col], linewidth=1.5, label=str(col))

    ax.set_xlabel("Index", fontsize=11)
    ax.legend(fontsize=9, ncol=min(len(numeric_cols), 3))
    if title:
        ax.set_title(title, fontsize=13)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 3D 可视化（pyvista）
# ---------------------------------------------------------------------------
def quick_plot_3d_surface(
    X: NDArray,
    Y: NDArray,
    Z: NDArray,
    title: str = "",
    cmap: str = "viridis",
    out_path: Path | None = None,
    window_size: tuple[int, int] = (1024, 768),
) -> Any:
    """3D 曲面图（pyvista 离屏渲染 → PNG）。

    Args:
        X, Y, Z: 网格坐标数组（2D meshgrid 或 1D +  broadcasting）。
        title: 图标题。
        cmap: 颜色映射。
        out_path: 若指定，自动导出 PNG。
        window_size: 渲染窗口尺寸。

    Returns:
        pyvista Plotter 对象（若未 close）。
    """
    import pyvista as pv

    pv.OFF_SCREEN = settings.pyvista_off_screen

    # 构建 StructuredGrid
    if X.ndim == 2:
        # 已经是 meshgrid
        grid = pv.StructuredGrid(X, Y, Z)
    else:
        # 1D 数组，需要 meshgrid
        Xm, Ym = np.meshgrid(X, Y, indexing="ij")
        grid = pv.StructuredGrid(Xm, Ym, Z)

    plotter = pv.Plotter(off_screen=settings.pyvista_off_screen, window_size=window_size)
    plotter.add_mesh(grid, cmap=cmap, show_scalar_bar=True, smooth_shading=True)
    plotter.background_color = "white"

    if title:
        plotter.add_text(title, position="upper_left", font_size=12, color="black")

    if out_path:
        export_exploration(plotter, out_path, close=False)

    return plotter


def quick_plot_zero_set_3d(
    curve: Any,
    real_face: Any = None,
    imag_face: Any = None,
    param_labels: tuple[str, str, str] = ("x", "y", "z"),
    title: str = "复变函数零点集",
    out_path: Path | None = None,
    window_size: tuple[int, int] = (1024, 768),
) -> Any:
    """3D 零点集空间曲线可视化（pyvista 离屏渲染）。

    Args:
        curve: 零点集曲线（pyvista PolyData）。
        real_face: 实部等值面（可选，半透明显示）。
        imag_face: 虚部等值面（可选，半透明显示）。
        param_labels: 三个轴的标签。
        title: 图标题。
        out_path: 若指定，自动导出 PNG。
        window_size: 渲染窗口尺寸。

    Returns:
        pyvista Plotter 对象。
    """
    import pyvista as pv

    pv.OFF_SCREEN = settings.pyvista_off_screen

    plotter = pv.Plotter(off_screen=settings.pyvista_off_screen, window_size=window_size)

    # 添加等值面（半透明）
    if real_face is not None:
        plotter.add_mesh(real_face, color="blue", opacity=0.15, label="Re=0")
    if imag_face is not None:
        plotter.add_mesh(imag_face, color="green", opacity=0.15, label="Im=0")

    # 添加零点集曲线
    if curve is not None and curve.n_points > 0:
        plotter.add_mesh(curve, color="red", line_width=4, label="零点集")

    plotter.background_color = "white"
    plotter.add_legend(face="triangle", size=(0.2, 0.1))
    plotter.show_bounds(
        grid="back", location="outer", ticks="both",
        xtitle=param_labels[0], ytitle=param_labels[1], ztitle=param_labels[2],
        font_size=10, color="black",
    )

    if title:
        plotter.add_text(title, position="upper_left", font_size=12, color="black")

    if out_path:
        export_exploration(plotter, out_path, close=False)

    return plotter


def quick_plot_band_structure_3d(
    K_mesh: NDArray,
    C_mesh: NDArray,
    eigenvalue_meshes: list[NDArray],
    labels: list[str] | None = None,
    colors: list[str] | None = None,
    axis_titles: tuple[str, str, str] = ("k", "C", "ω"),
    title: str = "能带结构",
    out_path: Path | None = None,
    window_size: tuple[int, int] = (1200, 900),
) -> Any:
    """3D 能带结构曲面图（pyvista 离屏渲染）。

    Args:
        K_mesh: 第一参数网格 (nk, nC)。
        C_mesh: 第二参数网格 (nk, nC)。
        eigenvalue_meshes: 各本征值的网格数组列表。
        labels: 各曲面标签。
        colors: 各曲面颜色。
        axis_titles: 三轴标题。
        title: 图标题。
        out_path: 若指定，自动导出 PNG。
        window_size: 渲染窗口尺寸。

    Returns:
        pyvista Plotter 对象。
    """
    import pyvista as pv

    pv.OFF_SCREEN = settings.pyvista_off_screen

    if colors is None:
        colors = ["blue", "cyan", "red", "orange", "green", "magenta"]
    if labels is None:
        labels = [f"λ_{i+1}" for i in range(len(eigenvalue_meshes))]

    plotter = pv.Plotter(off_screen=settings.pyvista_off_screen, window_size=window_size)

    for i, Z in enumerate(eigenvalue_meshes):
        grid = pv.StructuredGrid(K_mesh, C_mesh, np.real(Z))
        plotter.add_mesh(
            grid,
            color=colors[i % len(colors)],
            opacity=0.75,
            label=labels[i],
            smooth_shading=True,
        )

    plotter.background_color = "white"
    plotter.add_legend(size=(0.2, 0.12), face="rectangle", loc="upper right")
    plotter.show_bounds(
        grid="back", location="outer", ticks="both",
        xtitle=axis_titles[0], ytitle=axis_titles[1], ztitle=axis_titles[2],
        font_size=10, color="black",
    )
    plotter.camera_position = "iso"

    if title:
        plotter.add_text(title, position="upper_left", font_size=12, color="black")

    if out_path:
        export_exploration(plotter, out_path, close=False)

    return plotter
