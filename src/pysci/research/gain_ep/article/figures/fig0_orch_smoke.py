"""fig0_orch_smoke — 单面板示例色散曲线（避免交叉），编排体系 Phase 1 验收图。

数据：``<figdir>/data/dispersion.csv``（``#`` 注释行记录模型与参数，数据 7 列）
      k, ω(裸分支 1), ω(裸分支 2), Re ω₊, Re ω₋, Im ω₊, Im ω₋
模型：ω± = ω̄ ± √(δ² + κ²)，裸分支 ω₁ = ω₀ + c(k−k₀)、ω₂ = ω₀ + s₂(k−k₀)；
      两裸分支在 k = k₀ 相交，耦合 κ 把交点劈成间隙 Δω = 2κ。
"""

from pathlib import Path

import numpy as np
from matplotlib import transforms as mtransforms

from pysci import paths
from pysci.skills.scientific_plotting.tools import layout, palette

_SLUG = "fig0_orch_smoke"


def _load(research_dir=None) -> np.ndarray:
    """读取本图的合成色散数据（跳过 ``#`` 注释行）。

    Args:
        research_dir: 管线注入的 figures 根目录（``data/research/<n>_<线>/article/figures``）；
            None 时按研究线名经 :func:`pysci.paths.research_fig_dir` 解析同一位置。
    """
    figdir = (
        Path(research_dir) / _SLUG
        if research_dir
        else paths.research_fig_dir("gain_ep", slug=_SLUG)
    )
    return np.loadtxt(figdir / "data" / "dispersion.csv", delimiter=",")


def build_figure(style=None, research_dir=None, **kwargs):
    """构建单面板色散图并返回 Figure（由管线在 style_context 内调用）。"""
    k, ω1, ω2, ωp, ωm = _load(research_dir)[:, :5].T

    imin = int(np.argmin(ωp - ωm))  # 两支最贴近处 = 交叉点
    k0, Δω = float(k[imin]), float(ωp[imin] - ωm[imin])
    ω0, ωp0, ωm0 = float(0.5 * (ωp[imin] + ωm[imin])), float(ωp[imin]), float(ωm[imin])

    fig, ax = layout.grid(1, 1)

    ax.plot(k, ω1, ls="--", lw=0.8, color="0.6", label="bare branch")
    ax.plot(k, ω2, ls="--", lw=0.8, color="0.6")
    ax.plot(
        k, ωp, ls="-", lw=1.4, color=palette.color(1), label=r"$\mathrm{Re}\,\omega_+$"
    )
    ax.plot(
        k, ωm, ls="-", lw=1.4, color=palette.color(2), label=r"$\mathrm{Re}\,\omega_-$"
    )

    ax.axvline(k0, ls=":", lw=0.8, color="0.7")
    ax.annotate(
        "",
        xy=(k0, ωp0),
        xytext=(k0, ωm0),
        arrowprops=dict(arrowstyle="<->", lw=0.8, color="0.35"),
    )
    ax.text(
        1.14,
        ω0,
        f"$\\Delta\\omega = 2\\kappa = {Δω:.2f}$",
        ha="left",
        va="center",
        color="0.25",
    )

    # k₀ 标记：x 用数据坐标、y 贴轴底（避免与曲线争位置）
    blended = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(
        k0 - 0.03,
        0.05,
        "$k_0$",
        transform=blended,
        rotation=90,
        ha="right",
        va="bottom",
        color="0.4",
    )

    ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"$\mathrm{Re}\,\omega$")
    ax.legend(frameon=False, loc="lower right", handlelength=1.8)
    return fig
