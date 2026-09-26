"""项目路径锚点。

以稳健方式定位项目根与各类资产目录，替代散落在各模块中脆弱的 ``Path(__file__).parents[N]``
硬编码层级数学——目录结构一旦调整，后者会静默失效。

用法::

    from pysci.paths import PROJECT_ROOT, research_asset_dir

    storage = research_asset_dir("gain_ep") / "storage"
"""

from __future__ import annotations

from pathlib import Path

# 项目根标记：同时包含 pyproject.toml 与 .python-version 的目录即为项目根。
_ROOT_MARKERS = ("pyproject.toml", ".python-version")


def _find_project_root(start: Path) -> Path:
    """从 ``start`` 向上查找项目根目录。

    Args:
        start: 起始目录（通常为本文件所在包目录）。

    Returns:
        项目根路径。若未找到标记（例如被以非 editable 方式安装进 site-packages），
        回退到 ``<site-packages>/pysci/paths.py`` 的上两级，尽力而为。
    """
    for candidate in (start, *start.parents):
        if all((candidate / marker).exists() for marker in _ROOT_MARKERS):
            return candidate
    # 回退：本文件位于 <root>/src/pysci/paths.py，向上两级到 <root>/src 的父级不可靠，
    # 故直接返回 parents[2]（= 包所在 source 根的父目录）作为最后猜测。
    return Path(__file__).resolve().parents[2]


# 本文件位于 <root>/src/pysci/paths.py
_THIS_FILE = Path(__file__).resolve()

#: 项目根目录（含 pyproject.toml / .venv / src 等）。
PROJECT_ROOT: Path = _find_project_root(_THIS_FILE.parent)

#: 研究资产根：各研究线的非代码资产（笔记/PDF/.mph/storage/参考资料）按研究存放于此。
#: 与代码树 ``src/pysci/research/<name>/`` 镜像：``data/research/<n>_<name>/``。
ASSET_ROOT: Path = PROJECT_ROOT / "data" / "research"

#: 文献知识库数据区（papers/shortlists/reviews/cache/templates/INDEX.md）。
#: 与代码树 ``src/pysci/skills/literature_research/`` 镜像：``data/skills/literature_research/``。
LITERATURE_ROOT: Path = PROJECT_ROOT / "data" / "skills" / "literature_research"

#: 文档写作数据区（templates/projects/assets/cache）。
#: 与代码树 ``src/pysci/skills/document_writing/`` 镜像：``data/skills/document_writing/``。
DOCWRITING_ROOT: Path = PROJECT_ROOT / "data" / "skills" / "document_writing"

#: COMSOL 仿真数据区（docs/cache/recipes/templates/knowledge/runs）。
#: 与代码树 ``src/pysci/skills/comsol_simulation/`` 镜像：``data/skills/comsol_simulation/``。
COMSOL_ROOT: Path = PROJECT_ROOT / "data" / "skills" / "comsol_simulation"

#: 科研绘图数据区（templates/cache/recipes）。
#: 与代码树 ``src/pysci/skills/scientific_plotting/`` 镜像：``data/skills/scientific_plotting/``。
#: 注意：本目录只放技能级资产（脚手架模板、预览缓存、可复用配方画廊）；
#: 具体论文插图的产物落在各研究资产目录 ``data/research/<n>_<name>/article/figures/``。
PLOTTING_ROOT: Path = PROJECT_ROOT / "data" / "skills" / "scientific_plotting"


def research_asset_dir(name: str) -> Path:
    """解析某条研究线的资产目录。

    资产目录以数字序号前缀命名（如 ``data/research/1_gain_ep``），而包路径不能以数字开头
    （``pysci.research.gain_ep``）。本函数按 ``*_<name>`` 通配匹配对应资产目录。

    Args:
        name: 研究线名称（不含数字前缀），如 ``"gain_ep"``。

    Returns:
        匹配到的资产目录路径。若不存在多个/零个匹配，返回按约定推导的路径
        （``ASSET_ROOT / name``），由调用方决定是否创建。

    Raises:
        ValueError: 当匹配到多个不同序号的同名研究资产目录时。
    """
    matches = sorted(ASSET_ROOT.glob(f"*_{name}"))
    matches = [m for m in matches if m.is_dir()]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        found = ", ".join(str(m.name) for m in matches)
        raise ValueError(f"研究资产目录 '{name}' 匹配到多个：{found}")
    # 无匹配：退回无序号命名，交由调用方 mkdir。
    return ASSET_ROOT / name
