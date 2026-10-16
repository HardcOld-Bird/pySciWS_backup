"""PPT 页6 主体：本项目主线框图 与 Agent Loop（ReAct）框图 的同构上下对照。

- 上行：``用户输入 → pySci → 生产结果``；pySci 上方 / 下方各一枚**环形自指箭头**回到自身
  （上＝执行受阻后的自我修复与维护，下＝结果不合预期时的自我改进）。
- 下行：``思考 → 行动 → 观察``，观察经**回灌弧**回到思考（ReAct / Agent Loop 闭环）。
- 两行**同轴**（三列框心完全同位）、同字号分级、同线宽；「本项目＝把 Agent loop 的自我改进
  升到工程组织层面」这层意思只靠图形相似传达，不另加解释文字。

气质对齐 `data/research/1_gain_ep/汇报PPT/PPT模板.pptx`（黑白简约、细线冷灰、无彩色）。
导出：``pysci_vs_react.png``（300 dpi → 2880×1440 px，供 PPT 贴入）与 ``pysci_vs_react.pdf``
（矢量，字体以 TrueType 嵌入 type42，文字可编辑）。

用法（项目根运行，可重跑）::

    uv run --no-sync python data/research/1_gain_ep/article/figures/ppt_flow/_build_pysci_vs_react.py

零遮挡自检层（供 ``pysci-figures overlap`` 用）由 ``--occlusion-dir DIR`` 追加导出：同一次
渲染先写受审图、再写删文字版（版面完全一致，仅省略文字图元）。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.patches import Arc, FancyArrowPatch, Rectangle  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402

from pysci.paths import assert_within_data, research_fig_dir  # noqa: E402

# %% 版式常量（数据单位：画布 16×8，1 单位 = 0.6 in）

CANVAS_W, CANVAS_H = 16.0, 8.0
FIG_W_IN, FIG_H_IN = 9.6, 4.8  # 16:8 等比 → 数据坐标轴等比例、图形不走形
DEF_DPI = 300  # → 2880×1440 px（≥2000 px 要求）

ROW_TOP_Y, ROW_BOT_Y = 5.30, 1.95  # 两行框心 y（上下同轴：列心共用）
COL_X = (3.15, 8.50, 13.85)  # 三列框心 x（两行严格同位 → 同轴对照）
BOX_W = (2.30, 2.70, 2.30)  # 中列（pySci）略宽，标出主线核心
BOX_H = 1.05

LOOP_R = 0.62  # 自指环半径（真圆）
LOOP_STEM = 0.30  # 环开口到框边的短颈长度
LOOP_GAP_DEG = 40.0  # 环身开口角（开口正对框，短颈由此穿入）
ARC_DEPTH = 1.00  # 下行回灌弧半高（半椭圆，两端切于框边、切线竖直）

C_LINE = (
    "#33383E"  # 框线 / 箭头（细线冷灰；亮度 56 < 60 → 可被 overlap 线真身蒙版捕获）
)
C_TEXT = "#242A31"  # 框内主标签
C_NOTE = "#5D666E"  # 箭头注 / 行标（次级冷灰，底色对比度 ≈ 5.3:1）
LW_BOX, LW_ARROW, LW_LOOP = 1.1, 1.4, 1.2  # 同线宽：两行共用同一组
HEAD_SCALE = 15.0

FS_MAIN, FS_BOX, FS_NOTE, FS_CAP = (
    18.0,
    14.0,
    10.5,
    9.5,
)  # 字号分级：主框 > 其余框 > 注 > 行标

FONT_STACK = ("DengXian", "Microsoft YaHei", "SimHei")  # 模板正文字族（等线）优先

TOP_BOXES = ("用户输入", "pySci", "生产结果")
BOT_BOXES = ("思考", "行动", "观察")
NOTE_REPAIR = "执行受阻\n自我修复与维护"  # 上自指环注（AI 执行遇阻 → 自修复维护）
NOTE_IMPROVE = "结果不合预期\n自我改进"  # 下自指环注（用户对结果不满 → 自改进）
NOTE_LOOP = "观察回灌"  # 回灌弧注（Observation → Thought）
CAP_TOP, CAP_BOT = "本项目", "Agent Loop"  # 行标（左页边，次级灰）
CAP_X = 0.30  # 行标左端 x（框列自 2.00 起，留白 0.7）


# %% 绘制


def _setup_rcparams() -> None:
    """锁定字族 / 矢量字体嵌入 / 白底——与 PPT 模板气质一致的全局样式。"""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": list(FONT_STACK),
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def _new_figure() -> tuple[Figure, Axes]:
    """建画布：单轴铺满、等比、无框无刻度。"""
    fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN), dpi=DEF_DPI)
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.set_xlim(0.0, CANVAS_W)
    ax.set_ylim(0.0, CANVAS_H)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    return fig, ax


def _box(ax: Axes, cx: float, cy: float, w: float, h: float) -> Rectangle:
    """一枚直角框（白底、细冷灰边；不填充 → 与透明/白底均不冲突）。"""
    patch = Rectangle(
        (cx - w / 2, cy - h / 2),
        w,
        h,
        facecolor="none",
        edgecolor=C_LINE,
        linewidth=LW_BOX,
        joinstyle="miter",
        zorder=2,
    )
    ax.add_patch(patch)
    return patch


def _label(ax: Axes, cx: float, cy: float, s: str, fontsize: float) -> None:
    """框内标签（水平垂直居中）。"""
    ax.text(
        cx,
        cy,
        s,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=C_TEXT,
        zorder=4,
    )


def _harrow(ax: Axes, x0: float, x1: float, y: float) -> None:
    """同高直箭头（框间主线）。"""
    ax.add_patch(
        FancyArrowPatch(
            (x0, y),
            (x1, y),
            arrowstyle="-|>",
            mutation_scale=HEAD_SCALE,
            color=C_LINE,
            linewidth=LW_ARROW,
            shrinkA=0.0,
            shrinkB=0.0,
            joinstyle="miter",
            zorder=3,
        )
    )


def _self_loop(ax: Axes, cx: float, edge_y: float, *, side: int) -> None:
    """环形自指箭头：真圆环身 + 开口正对框、短颈带箭头扎回框边。

    Args:
        cx: 环所在列心。
        edge_y: 所贴框的水平边 y（``side=+1`` 取框上边、``-1`` 取框下边）。
        side: ``+1`` 环在框上方（开口朝下），``-1`` 环在框下方（开口朝上）。
    """
    tip = (cx, edge_y)
    root = (cx, edge_y + side * LOOP_STEM)
    ax.add_patch(
        FancyArrowPatch(
            root,
            tip,
            arrowstyle="-|>",
            mutation_scale=HEAD_SCALE,
            color=C_LINE,
            linewidth=LW_LOOP,
            shrinkA=0.0,
            shrinkB=0.0,
            joinstyle="miter",
            zorder=3,
        )
    )
    center = (cx, edge_y + side * (LOOP_STEM + LOOP_R))
    gap_center = (
        270.0 if side > 0 else 90.0
    )  # 开口正对框：下方环开口朝下、上方环开口朝上
    half = LOOP_GAP_DEG / 2.0
    ax.add_patch(
        Arc(
            center,
            2.0 * LOOP_R,
            2.0 * LOOP_R,
            theta1=gap_center + half,
            theta2=gap_center - half + 360.0,
            color=C_LINE,
            linewidth=LW_LOOP,
            zorder=3,
        )
    )


def _return_arc(ax: Axes, x_from: float, x_to: float, y: float, depth: float) -> None:
    """半椭圆回灌弧：自 ``x_from`` 框边下行、绕弧回到 ``x_to`` 框边（末端箭头竖直扎入）。"""
    cx = (x_from + x_to) / 2.0
    a = abs(x_from - x_to) / 2.0
    t = np.linspace(0.0, -np.pi, 241)
    pts = np.column_stack([cx + a * np.cos(t), y + depth * np.sin(t)])
    ax.add_patch(
        FancyArrowPatch(
            path=MplPath(pts),
            arrowstyle="-|>",
            mutation_scale=HEAD_SCALE,
            color=C_LINE,
            linewidth=LW_LOOP,
            shrinkA=0.0,
            shrinkB=0.0,
            joinstyle="miter",
            zorder=3,
        )
    )


def _note(
    ax: Axes, x: float, y: float, s: str, *, ha: str = "left", fontsize: float = FS_NOTE
) -> None:
    """注记文字（次级冷灰；多行用 ``\\n``，行距 1.45 防叠）。"""
    ax.text(
        x,
        y,
        s,
        ha=ha,
        va="center",
        fontsize=fontsize,
        color=C_NOTE,
        linespacing=1.45,
        zorder=4,
    )


def draw(*, texts: bool = True) -> Figure:
    """绘制整图。

    Args:
        texts: 是否绘制全部文字图元；``False`` 产出**删文字版**（版面不变），
            供 ``pysci-figures overlap`` 的零遮挡像素判据做真值面。

    Returns:
        已绘制完成的 Figure（未保存）。
    """
    _setup_rcparams()
    fig, ax = _new_figure()

    for row_y, boxes, main_size in (
        (ROW_TOP_Y, TOP_BOXES, FS_MAIN),
        (ROW_BOT_Y, BOT_BOXES, FS_BOX),
    ):
        for i, (cx, w) in enumerate(zip(COL_X, BOX_W, strict=True)):
            _box(ax, cx, row_y, w, BOX_H)
            if texts:
                size = main_size if i == 1 else FS_BOX
                _label(ax, cx, row_y, boxes[i], size)
        left_edge = COL_X[0] + BOX_W[0] / 2.0
        mid_l_edge = COL_X[1] - BOX_W[1] / 2.0
        mid_r_edge = COL_X[1] + BOX_W[1] / 2.0
        right_edge = COL_X[2] - BOX_W[2] / 2.0
        _harrow(ax, left_edge, mid_l_edge, row_y)
        _harrow(ax, mid_r_edge, right_edge, row_y)

    # 上行：pySci 上下各一环形自指箭头
    top_edge = ROW_TOP_Y + BOX_H / 2.0
    bot_edge = ROW_TOP_Y - BOX_H / 2.0
    _self_loop(ax, COL_X[1], top_edge, side=1)
    _self_loop(ax, COL_X[1], bot_edge, side=-1)

    # 下行：观察 → 思考 回灌弧
    _return_arc(ax, COL_X[2], COL_X[0], ROW_BOT_Y - BOX_H / 2.0, ARC_DEPTH)

    if texts:
        ring_y = top_edge + LOOP_STEM + LOOP_R
        _note(ax, COL_X[1] + LOOP_R + 0.30, ring_y, NOTE_REPAIR)
        _note(ax, COL_X[1] + LOOP_R + 0.30, bot_edge - LOOP_STEM - LOOP_R, NOTE_IMPROVE)
        _note(
            ax,
            COL_X[1],
            ROW_BOT_Y - BOX_H / 2.0 - ARC_DEPTH / 2.0,
            NOTE_LOOP,
            ha="center",
        )
        _note(ax, CAP_X, ROW_TOP_Y, CAP_TOP, fontsize=FS_CAP)
        _note(ax, CAP_X, ROW_BOT_Y, CAP_BOT, fontsize=FS_CAP)

    return fig


# %% 落盘


def build(
    *, outdir: Path, dpi: int = DEF_DPI, occlusion_dir: Path | None = None
) -> list[Path]:
    """排版 → 导出 PNG（栅格）+ PDF（矢量）；可选追加零遮挡自检的两层同版式 PNG。

    Args:
        outdir: 交付目录（须落在 ``data/`` 内）。
        dpi: PNG 分辨率（300 → 2880 px 宽）。
        occlusion_dir: 非空时在其下写 ``*_text.png`` / ``*_notext.png`` 两层
            （先写受审图、后写删文字版，满足 overlap 的缓存新鲜度纪律）。

    Returns:
        写出的文件路径列表（按写入顺序）。
    """
    stem = "pysci_vs_react"
    assert_within_data(outdir, what="框图交付目录")
    outdir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    fig = draw(texts=True)
    png = outdir / f"{stem}.png"
    fig.savefig(png, dpi=dpi, facecolor="white")
    written.append(png)
    pdf = outdir / f"{stem}.pdf"
    fig.savefig(pdf, facecolor="white")
    written.append(pdf)
    if occlusion_dir is not None:
        occlusion_dir.mkdir(parents=True, exist_ok=True)
        for tag, with_texts in (("text", True), ("notext", False)):
            f = draw(texts=with_texts) if not with_texts else fig
            path = occlusion_dir / f"{stem}_{tag}.png"
            f.savefig(path, dpi=dpi, facecolor="white")
            written.append(path)
            if not with_texts:
                plt.close(f)
    plt.close(fig)
    return written


def main() -> int:
    """命令行入口：默认产出到研究线的 ``article/figures/ppt_flow/``。"""
    ap = argparse.ArgumentParser(
        description="PPT 页6 双流程框图（本项目 vs Agent Loop）"
    )
    ap.add_argument(
        "--outdir",
        type=Path,
        default=research_fig_dir("gain_ep", slug="ppt_flow"),
        help="交付目录（默认研究线图目录下 ppt_flow/）",
    )
    ap.add_argument("--dpi", type=int, default=DEF_DPI, help="PNG 分辨率")
    ap.add_argument(
        "--occlusion-dir",
        type=Path,
        default=None,
        help="可选：追加零遮挡自检两层（受审图 + 删文字版）",
    )
    args = ap.parse_args()
    for p in build(outdir=args.outdir, dpi=args.dpi, occlusion_dir=args.occlusion_dir):
        report_size(p)
    return 0


def report_size(p: Path) -> None:
    """打印产物尺寸（只读核对，便于重跑时现场确认；PDF 不适用）。"""
    if p.suffix.lower() != ".png":
        print(f"[written] {p}")
        return
    from PIL import Image

    with Image.open(p) as im:
        print(f"[written] {p}  {im.size[0]}x{im.size[1]} px  {im.mode}")


if __name__ == "__main__":
    raise SystemExit(main())
