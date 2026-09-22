"""pysci —— 物理声学研究的共享工具、研究代码与 LLM skill 基础设施。

三层结构：
- ``pysci.common``：可复用的理论储备/基础工具（矩阵、数值化、可视化、绘图配置）。
- ``pysci.research.<name>``：各条研究线，内含 ``theory``（理论/仿真）与 ``experiment``（实验平台）。
- ``pysci.skills``：LLM skill 后端（literature_research）。

代码以 editable 方式安装进项目 ``.venv``，任何位置均可 ``import pysci``，无需 sys.path hack。
"""

from pysci.paths import (
    ASSET_ROOT,
    LITERATURE_ROOT,
    PROJECT_ROOT,
    research_asset_dir,
)

__all__ = [
    "ASSET_ROOT",
    "LITERATURE_ROOT",
    "PROJECT_ROOT",
    "research_asset_dir",
]
