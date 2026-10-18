"""beam_scatter 综合图复跑入口（返工第 3 轮，b2_sim 数据）。

绘图源码在 ``data/research/1_gain_ep/article/figures/beam_scatter/beam_scatter.py``
（图目录管线，供 ``pysci-figures build/audit`` 复用）；本脚本在同一 style_context 下
重建该图并**原地**写 ``beam_scatter_combined.png``（170 mm 宽，非 tight 裁切），
可选导出删文字层供 ``pysci-figures overlap`` 的零遮挡判据使用。

用法::

    uv run --no-sync python scripts/research/gain_ep/simulation/finalize_beam_scatter_figure.py
    uv run --no-sync python ... --occlusion-dir=orchestration/pods/figure/bench/overlap_r3
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.text import Text  # noqa: E402

from pysci.paths import research_fig_dir  # noqa: E402
from pysci.skills.scientific_plotting.tools.runner import built_figure  # noqa: E402

FIGDIR = research_fig_dir("gain_ep", slug="beam_scatter")
OUT_PNG = FIGDIR / "beam_scatter_combined.png"


def main() -> None:
    occ = None
    for arg in sys.argv[1:]:
        if arg.startswith("--occlusion-dir="):
            occ = Path(arg.split("=", 1)[1])
    with built_figure(FIGDIR) as (fig, info):
        with plt.rc_context({"savefig.bbox": "standard", "savefig.pad_inches": 0.0}):
            # 先写受审件、后写删文字层（overlap 的 mtime 新鲜度纪律）
            fig.savefig(OUT_PNG, dpi=600, facecolor="white")
            if occ is not None:
                for t in list(fig.findobj(Text)):
                    try:
                        t.remove()
                    except (NotImplementedError, ValueError):
                        t.set_visible(False)
                occ.mkdir(parents=True, exist_ok=True)
                fig.savefig(
                    occ / f"{OUT_PNG.stem}_notext.png", dpi=600, facecolor="white"
                )
        print(info.report())
    print("combined ->", OUT_PNG)


if __name__ == "__main__":
    main()
