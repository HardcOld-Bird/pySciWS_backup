"""补充材料图：增益工作点 ci 扫描下的主瓣演化与近阈值谐振敏感性（twin 裁决 C）。

数据：`raw/ci_scan/far_nostruct_<inc>_ci<tag>.csv`（对照③ gain-no-groove，1° 网格 181 点，
由 `export_ci_scan.py` 在原域 R = 4λ 上重算导出）——扫描档 ci ∈ [−0.03, −0.0730324]。

版式（178 mm × 100 mm，四面板）：
  (a) 主瓣角随 ci 的演化（L/R 两条，线性标度）；
  (b) 主瓣幅值随 ci 的演化（L/R 两条，线性标度）；
  (c)(d) ci = −0.05 与 ci = −0.0730324 的远场极坐标快照并置（半极坐标、线性极径）。
图注一句话点明「近阈值谐振敏感性」。

用法::

    uv run --no-sync python scripts/research/gain_ep/simulation/plot_ci_scan_supplement.py
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from pysci.skills.scientific_plotting.tools.style import (  # noqa: E402
    mm_to_inch,
    style_context,
)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
FIG_DIR = (
    ROOT / "data" / "research" / "1_gain_ep" / "article" / "figures" / "beam_scatter"
)
SCAN_DIR = FIG_DIR / "raw" / "ci_scan"
OUT_PNG = FIG_DIR / "beam_scatter_ci_scan_near_threshold.png"

CI_EP = -0.0730324
CI_TAGS = (
    ("m0.03", -0.03),
    ("m0.04", -0.04),
    ("m0.05", -0.05),
    ("m0.06", -0.06),
    ("m0.07", -0.07),
    ("m0.0730324", CI_EP),
)
INCS = (("L", "L-incidence"), ("R", "R-incidence"))
SNAP = ("m0.05", "m0.0730324")  # 两幅快照档位
C_L, C_R = "#1b3f9e", "#d81e2a"
GRID, INK, GRAY = "#b9b9b9", "#222222", "#666666"


def _load(inc: str, tag: str) -> tuple[np.ndarray, np.ndarray]:
    a = np.loadtxt(
        SCAN_DIR / f"far_nostruct_{inc}_ci{tag}.csv", delimiter=",", comments="%"
    )
    return a[:, 2], a[:, 3]


def _curves() -> tuple:
    """(a) 主瓣角随 ci；（b) 主瓣幅值随 ci。"""
    ci = np.array([c for _, c in CI_TAGS])
    peaks = {inc: [] for inc, _ in INCS}
    amps = {inc: [] for inc, _ in INCS}
    for inc, _ in INCS:
        for tag, _ in CI_TAGS:
            ang, amp = _load(inc, tag)
            peaks[inc].append(float(ang[int(np.argmax(amp))]))
            amps[inc].append(float(amp.max()))
    return ci, peaks, amps


def draw() -> None:
    ci, peaks, amps = _curves()
    w_mm, h_mm = 178.0, 100.0
    with style_context("aps", width=str(int(w_mm)), scienceplots=True):
        fig = plt.figure(figsize=(mm_to_inch(w_mm), mm_to_inch(h_mm)))
        ax_a = fig.add_axes([0.085, 0.600, 0.360, 0.300])
        ax_b = fig.add_axes([0.085, 0.180, 0.360, 0.300])
        ax_c = fig.add_axes([0.520, 0.600, 0.455, 0.290])
        ax_d = fig.add_axes([0.520, 0.180, 0.455, 0.290])
        # --- (a)(b) 演化曲线 ---
        for inc, lab in INCS:
            col = C_L if inc == "L" else C_R
            ax_a.plot(ci, peaks[inc], "o-", color=col, ms=2.6, lw=1.0, label=lab)
            ax_b.plot(ci, amps[inc], "o-", color=col, ms=2.6, lw=1.0, label=lab)
        ax_a.set_ylabel("main-lobe angle (deg)")
        ax_b.set_ylabel("main-lobe $|p_\\mathrm{ext}|$ (Pa)")
        for ax in (ax_a, ax_b):
            ax.set_xlim(-0.078, -0.026)
            ax.grid(True, color=GRID, lw=0.4, alpha=0.8)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
        ax_a.set_xticklabels([])
        ax_b.set_xlabel("gain parameter $c_i$")
        ax_a.legend(frameon=False, fontsize=6, loc="center right", handlelength=1.4)
        ax_a.text(
            -0.06, 1.02, "a", transform=ax_a.transAxes, fontweight="bold", color=INK
        )
        ax_b.text(
            -0.06, 1.02, "b", transform=ax_b.transAxes, fontweight="bold", color=INK
        )
        # 阈值漂移区标注（−0.06 起主瓣离开 45°/135° 族）
        for ax in (ax_a,):
            ax.axvspan(-0.078, -0.058, color="#f0f0f0", zorder=0)
            ax.text(
                -0.068,
                0.06,
                "near-threshold\nresonance",
                fontsize=6,
                color=GRAY,
                ha="center",
                va="bottom",
                transform=ax.get_xaxis_transform(),
            )
        # --- (c)(d) 极坐标快照 ---
        for ax, tag in ((ax_c, SNAP[0]), (ax_d, SNAP[1])):
            rmax = max(float(_load(inc, tag)[1].max()) for inc, _ in INCS)
            rmax = math.ceil(rmax * 10.0) / 10.0 * 1.15
            th = np.linspace(0.0, math.pi, 241)
            for frac in (1 / 3, 2 / 3, 1.0):
                ax.plot(
                    rmax * frac * np.cos(th),
                    rmax * frac * np.sin(th),
                    color=GRID,
                    lw=0.4,
                )
            for ang in (0, 45, 90, 135, 180):
                a = math.radians(ang)
                ax.plot(
                    [0, rmax * math.cos(a)], [0, rmax * math.sin(a)], color=GRID, lw=0.4
                )
                ax.text(
                    rmax * 1.08 * math.cos(a),
                    rmax * 1.08 * math.sin(a),
                    f"{ang}$^\\circ$",
                    color=GRAY,
                    fontsize=5,
                    ha="left"
                    if ang == 0
                    else ("right" if ang in (135, 180) else "center"),
                    va="center"
                    if ang in (0, 180)
                    else ("bottom" if ang in (45, 90) else "top"),
                )
            for inc, lab in INCS:
                ang, amp = _load(inc, tag)
                a = np.radians(ang)
                ax.plot(
                    amp * np.cos(a),
                    amp * np.sin(a),
                    color=C_L if inc == "L" else C_R,
                    lw=1.0,
                )
            ax.set_xlim(-rmax * 1.14, rmax * 1.14)
            ax.set_ylim(-0.16 * rmax, rmax * 1.22)
            ax.set_aspect("equal", adjustable="box")
            ax.set_axis_off()
        fig.text(0.747, 0.895, "$c_i$ = $-0.05$", fontsize=7, color=INK, ha="center")
        fig.text(0.747, 0.475, "$c_i$ = $-0.073$", fontsize=7, color=INK, ha="center")
        fig.text(0.487, 0.475, "d", fontsize=8, fontweight="bold", color=INK)
        fig.text(0.487, 0.895, "c", fontsize=8, fontweight="bold", color=INK)
        fig.text(
            0.085,
            0.020,
            "Near-threshold resonance: as $|c_i|$ approaches "
            f"{abs(CI_EP):.3f} the gain-excited resonant mode takes over, the main lobe leaves the "
            "45$^\\circ$/135$^\\circ$ family\nand the amplitude grows by a factor of 3.4 "
            "(circles = computed far-field spectra, 1$^\\circ$ grid; simulation domain "
            "$R$ = 40 cm, unchanged).",
            fontsize=6,
            color=GRAY,
            ha="left",
            va="bottom",
            linespacing=1.6,
        )
        with plt.rc_context({"savefig.bbox": "standard"}):
            fig.savefig(OUT_PNG, dpi=600, facecolor="white")
        plt.close(fig)
    print("supplement ->", OUT_PNG)


if __name__ == "__main__":
    draw()
