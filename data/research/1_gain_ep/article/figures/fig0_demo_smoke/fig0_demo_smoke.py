\
"""fig0_demo_smoke — 2×2 多子图管线（pysci-figures 脚手架生成，按需修改）。"""

import matplotlib.pyplot as plt
import numpy as np

from pysci.skills.scientific_plotting.tools import layout, palette


def build_figure(style=None, research_dir=None, **kwargs):
    """构建 2×2 复合图并返回 Figure。"""
    fig, axes = layout.grid(2, 2)

    x = np.linspace(0.0, 2.0 * np.pi, 200)
    for i, ax in enumerate(layout.flatten_axes(axes)):
        ax.plot(x, np.sin(x + i * 0.5), color=palette.color(i + 1))
        ax.set_xlabel(r"$x$ / $\mathrm{rad}$")
        ax.set_ylabel(rf"$y_{{{i + 1}}}$")

    layout.label_panels(axes)  # 自动补 (a)(b)(c)(d)
    return fig
