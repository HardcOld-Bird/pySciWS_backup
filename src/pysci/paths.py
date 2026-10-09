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

#: 理论计算数据区（templates/cache/recipes）。
#: 与代码树 ``src/pysci/skills/theoretical_computation/`` 镜像：``data/skills/theoretical_computation/``。
#: 注意：本目录只放技能级资产（脚手架模板、计算缓存、可复用配方）；
#: 具体研究线的计算产物落在各研究资产目录 ``data/research/<n>_<name>/theory/<slug>/``。
THEORY_ROOT: Path = PROJECT_ROOT / "data" / "skills" / "theoretical_computation"

#: AI 绘图数据区（assets/gallery/workflows/prompts/cache/runs + LEDGER.md）。
#: 与代码树 ``src/pysci/skills/ai_drawing/`` 镜像：``data/skills/ai_drawing/``。
#: 注意：本目录只放技能级资产（生成图入库、审美范本画廊、ComfyUI 工作流配方、prompt 配方、
#: 服务器状态与缓存）；具体研究线的效果图（封面 / graphical abstract / 示意图）落在
#: 各研究资产目录 ``data/research/<n>_<name>/article/artwork/``。
AI_DRAWING_ROOT: Path = PROJECT_ROOT / "data" / "skills" / "ai_drawing"

#: 3D 建模数据区（templates/cache/recipes）。
#: 与代码树 ``src/pysci/skills/modeling3d/`` 镜像：``data/skills/modeling3d/``。
#: 注意：本目录只放技能级资产（脚手架模板、渲染配方、缓存）；具体研究线的模型产物
#: （STL/STEP/渲染图）落在各研究资产目录 ``data/research/<n>_<name>/models/<slug>/``。
MODELING3D_ROOT: Path = PROJECT_ROOT / "data" / "skills" / "modeling3d"

#: 编排系统根：多 Agent 编排的全部资产（顶层设计 README、技能唯一真本 skills/、
#: 机器状态 state/、hook 守卫 guards/、组员工作区 pods/）。权威定义见
#: ``orchestration/README.md``；由 ``pysci-orch`` CLI（pysci.skills.orchestration）读写。
ORCHESTRATION_ROOT: Path = PROJECT_ROOT / "orchestration"

#: 编排机器状态区：registry.json（成员注册表与会话池索引）、plans/、backlog.json、
#: replies/、ledger/。全部为可读 JSON/MD/YAML（orch fail-open 红线），仅 orch 与
#: devops 可写；组长经 orch 间接读写。
ORCH_STATE_ROOT: Path = ORCHESTRATION_ROOT / "state"

#: 组员 pod 工作区根：每 pod 是一个完整 Qoder 项目（独立 .qoder/ + AGENTS.md +
#: bench/inbox/outbox），以 pod 为 cwd 的 headless 会话在此运行。
PODS_ROOT: Path = ORCHESTRATION_ROOT / "pods"


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


#: 项目数据根：所有技能/研究的非代码产物都应落在此目录内（护栏基准）。
DATA_ROOT: Path = PROJECT_ROOT / "data"


def research_fig_dir(name: str, *, slug: str | None = None) -> Path:
    """解析某研究线论文插图目录。

    规范位置：``data/research/<n>_<name>/article/figures[/<slug>]``。技能级模板/缓存
    在 :data:`PLOTTING_ROOT`，具体论文插图产物一律落在本目录。

    Args:
        name: 研究线名称（不含数字前缀），如 ``"gain_ep"``。
        slug: 可选的图 slug（如 ``"fig1_ep_band"``）；None → 返回 figures 根。
    """
    base = research_asset_dir(name) / "article" / "figures"
    return base / slug if slug else base


def research_theory_dir(name: str, *, slug: str | None = None) -> Path:
    """解析某研究线理论计算产物目录。

    规范位置：``data/research/<n>_<name>/theory[/<slug>]``。技能级缓存在
    :data:`THEORY_ROOT`，具体研究线的计算产物一律落在本目录。

    Args:
        name: 研究线名称（不含数字前缀）。
        slug: 可选的计算 slug；None → 返回 theory 根。
    """
    base = research_asset_dir(name) / "theory"
    return base / slug if slug else base


def research_artwork_dir(name: str, *, slug: str | None = None) -> Path:
    """解析某研究线效果图（AI 绘图）产物目录。

    规范位置：``data/research/<n>_<name>/article/artwork[/<slug>]``。技能级资产
    （生成图入库 / 审美范本 / 工作流配方）在 :data:`AI_DRAWING_ROOT`，具体研究线的
    封面图 / graphical abstract / 示意图等效果图一律落在本目录。

    Args:
        name: 研究线名称（不含数字前缀），如 ``"gain_ep"``。
        slug: 可选的效果图 slug；None → 返回 artwork 根。
    """
    base = research_asset_dir(name) / "article" / "artwork"
    return base / slug if slug else base


def research_model_dir(name: str, *, slug: str | None = None) -> Path:
    """解析某研究线 3D 模型产物目录。

    规范位置：``data/research/<n>_<name>/models[/<slug>]``。技能级模板/配方在
    :data:`MODELING3D_ROOT`，具体研究线的模型会话产物（CAD 脚本导出的 STL/STEP、
    Blender 渲染图、notes.md）一律落在本目录。

    Args:
        name: 研究线名称（不含数字前缀），如 ``"gain_ep"``。
        slug: 可选的模型会话 slug；None → 返回 models 根。
    """
    base = research_asset_dir(name) / "models"
    return base / slug if slug else base


def assert_within_data(path: str | Path, *, what: str = "产物") -> Path:
    """断言产物路径落在项目 ``data/`` 根内（防 stray 护栏）。

    在**写入前**调用：静默 stray（如误写到 ``scripts/``、仓库外、临时目录）会在落盘前
    即抛 :class:`ValueError`，避免产物散落到非规范目录后难以追溯。

    Args:
        path: 待校验的产物路径（文件或目录）。
        what: 报错信息里的产物类别描述。

    Returns:
        解析后的绝对路径（便于链式使用）。

    Raises:
        ValueError: 当路径不在 ``data/`` 根内时。
    """
    p = Path(path).resolve()
    root = DATA_ROOT.resolve()
    if p != root and root not in p.parents:
        raise ValueError(
            f"{what}路径不在 data/ 根内（防 stray 护栏）：{p}\n  data 根 = {root}"
        )
    return p
