"""半圆场图原型合成（v2，数据坐标）：COMSOL 场 PNG 底图 + 实验占位版式叠加。

版式依据参考图 4ref2/4ref3 的 (c)(d)(e) 上半部分：
  - 半圆 "Sim." 场图内部有两个沿 ±45° 波束方向旋转的矩形框（= 该入射下 S 矩阵的两个输出通道）；
  - 每个通道框以虚线连到半圆外顶部角落的一个旋转小方形 "Exp." 面板（放实验结果，现留空占位）。

与 v1（image-fraction 目视标定）的区别——本版走 scientific-plotting 的 raster 原语 + **公式化 extent**：
  - COMSOL Java API 对 2D 绘图组/视图**不暴露**轴限属性（已用 inspect node 确证），export --extent 不生效；
  - 但 COMSOL 2D auto-zoom 规则可精确建模：等纵横比、以几何 bbox 为中心、按轴框像素框
    （sidecar.crop_box_px）的长宽比在宽/高受限方向展开。几何 bbox 由建模参数精确算出，
    故 extent = f(几何 bbox, crop_box_px)，无需目视迭代。
  - 叠加原语（rotbox/panel/dashed）全部用**数据坐标**（米），由 raster.apply_overlays 施加。

用法：python compose_beam_scatter_proto.py [tag ...]（默认全部 6 case）
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from pysci.skills.comsol_simulation.tools.postprocess import (
    comsol_auto_window,  # noqa: E402
)
from pysci.skills.scientific_plotting.tools import raster  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
FIG_BASE = (
    ROOT / "data" / "research" / "1_gain_ep" / "article" / "figures" / "beam_scatter"
)
FIG_DIR = FIG_BASE / "raw"  # 输入：COMSOL 原生渲染 + sidecar
OUT_DIR = FIG_BASE / "panels"  # 输出：逐 case 半圆合成图（含实验占位框）

# --- 与 build_beam_scatter_mph.PARAMS 一致的几何常量（单位 m）---
LAMBDA = 343.0 / 3430.0  # c0/f = 0.1
D = LAMBDA / math.sqrt(2.0)  # 周期
X0 = 8 * D / 2.0  # 半圆中心 x
R = 4 * LAMBDA  # 半圆半径
H1 = 0.569 * LAMBDA  # 管槽1 深（几何下探）
G13 = {"full": 0.0, "nogain": 0.0, "nostruct": 0.05 * LAMBDA}  # 各 cfg 的抹平间隙

TAGS = [f"{c}_{i}" for c in ("full", "nogain", "nostruct") for i in ("L", "R")]


def geom_bbox(g13: float) -> tuple[float, float, float, float]:
    """几何包围盒 (x0,x1,y0,y1)：半圆 + 下探管槽。"""
    return (X0 - R, X0 + R, -(H1 + g13), R)


def polar(phi_deg: float, r: float) -> tuple[float, float]:
    """以半圆中心 (X0,0) 为原点的极坐标 → 直角坐标。"""
    a = math.radians(phi_deg)
    return (X0 + r * math.cos(a), r * math.sin(a))


def overlays_for(tag: str) -> list[dict]:
    """数据坐标叠加版式：两通道框 + 两 Exp 空面板 + 虚线连接。"""
    box_l, box_r = polar(135, 0.26), polar(45, 0.26)  # 半圆内 ±45° 通道框中心
    exp_l, exp_r = polar(115, 0.55), polar(65, 0.55)  # 半圆外顶部角落 Exp 面板中心
    return [
        {
            "type": "rotbox",
            "cx": box_l[0],
            "cy": box_l[1],
            "w": 0.18,
            "h": 0.07,
            "angle": 135,
        },
        {
            "type": "rotbox",
            "cx": box_r[0],
            "cy": box_r[1],
            "w": 0.18,
            "h": 0.07,
            "angle": 45,
        },
        {
            "type": "panel",
            "cx": exp_l[0],
            "cy": exp_l[1],
            "w": 0.13,
            "h": 0.13,
            "angle": 45,
            "text": "Exp.\n(TBD)",
        },
        {
            "type": "panel",
            "cx": exp_r[0],
            "cy": exp_r[1],
            "w": 0.13,
            "h": 0.13,
            "angle": 45,
            "text": "Exp.\n(TBD)",
        },
        {
            "type": "dashed",
            "x0": box_l[0],
            "y0": box_l[1],
            "x1": exp_l[0],
            "y1": exp_l[1],
        },
        {
            "type": "dashed",
            "x0": box_r[0],
            "y0": box_r[1],
            "x1": exp_r[0],
            "y1": exp_r[1],
        },
    ]


def compose(tag: str) -> Path:
    src = FIG_DIR / f"field_{tag}.png"
    sc = raster.load_sidecar(src.with_suffix(".sidecar.json"))
    crop = tuple(sc["crop_box_px"])
    cfg = tag.rsplit("_", 1)[0]
    extent = comsol_auto_window(geom_bbox(G13[cfg]), crop)
    img = raster.read_raster(src, crop)

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img, extent=list(extent), origin="upper", aspect="equal")
    ax.set_axis_off()
    raster.apply_overlays(ax, overlays_for(tag))

    out = OUT_DIR / f"proto_field_{tag}.png"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return out


def main() -> None:
    for tag in sys.argv[1:] or TAGS:
        print(f"composed {tag} -> {compose(tag)}")


if __name__ == "__main__":
    main()
