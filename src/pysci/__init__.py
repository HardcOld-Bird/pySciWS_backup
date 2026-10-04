"""pysci —— 物理声学研究的科研代码与 LLM skill 基础设施。

两层结构：
- ``pysci.research.<name>``：各条研究线（当前仅 ``gain_ep``），内含 ``theory``（理论/仿真）、
  ``experiment``（实验平台）与 ``article``（论文插图生产模块）。
- ``pysci.skills``：LLM skill 后端，每个技能一个子包（代码在其 ``tools/`` 下）——
  literature_research / comsol_simulation / document_writing / scientific_plotting /
  theoretical_computation / ai_drawing。

``pysci.paths`` 是路径锚点与产物落盘规范的**权威索引**（各 ``*_ROOT`` 常量与
``research_*_dir()`` 的注释逐条写明了目录用途）。共享绘图配置在研究线根（如
``pysci.research.gain_ep.plotting``），**不**设跨研究线的 ``pysci.common`` 层。

代码以 editable 方式安装进项目 ``.venv``，任何位置均可 ``import pysci``，无需 sys.path hack。
"""

from pysci.paths import (
    ASSET_ROOT,
    DOCWRITING_ROOT,
    LITERATURE_ROOT,
    PROJECT_ROOT,
    research_asset_dir,
)

__all__ = [
    "ASSET_ROOT",
    "DOCWRITING_ROOT",
    "LITERATURE_ROOT",
    "PROJECT_ROOT",
    "research_asset_dir",
]
