"""plot-group 自动重绑：求解后把全部绘图组绑回数据集（空白 PNG 防复发关键）。

坑（java-mph-pitfalls §3）：solve 前落盘 / reload 后重跑 geom/mesh/study，绘图组会失去
（或从未建立）dataset 绑定 → 导出 PNG 只剩轴框。**重求解后必须**
``pg.property("data", <dset tag>)``——本模块把这步自动化，是 solve-then-export 跨调用
安全的关键（≈50 行差异化胶水）。

mph API 口径（1.4.0）：``model/'plots'`` / ``model/'datasets'`` 是分组节点（可迭代出
子节点）；绘图组数据集绑定属性是 ``data``，值为**数据集 tag**（"dset1" 这类，非显示名）。
"""

from __future__ import annotations


def plan_rebind(plot_group_names: list[str], dataset_tag: str) -> list[tuple[str, str]]:
    """重绑计划（纯函数）：``[(绘图组名, dataset_tag), …]``。

    Args:
        plot_group_names: 绘图组名字（``model.plots()`` 口径）。
        dataset_tag: 目标数据集 tag。

    Returns:
        计划元组列表（顺序稳定）。

    Raises:
        ValueError: 数据集 tag 为空。
    """
    if not dataset_tag:
        raise ValueError("dataset_tag 不能为空")
    return [(name, dataset_tag) for name in plot_group_names]


def first_dataset_tag(model: object) -> str | None:
    """取模型首个数据集 tag（驱动；``model/'datasets'`` 子节点 ``tag()``）。

    Returns:
        数据集 tag；无数据集返回 None。
    """
    datasets = list(model / "datasets")
    return _tag_of(datasets[0]) if datasets else None


def plot_group_nodes(model: object) -> list[object]:
    """列模型全部绘图组节点（驱动；``model/'plots'`` 子节点）。"""
    return list(model / "plots")


def plot_group_names(model: object) -> list[str]:
    """列模型全部绘图组名字（``model.plots()`` 同口径）。"""
    return [str(_name_of(n)) for n in plot_group_nodes(model)]


def _name_of(node: object) -> object:
    """取节点显示名（mph ``name()``），鸭子类型通用。"""
    v = getattr(node, "name", None)
    return v() if callable(v) else v


def _tag_of(node: object) -> str:
    """取节点 tag（mph ``tag()``），鸭子类型通用。"""
    v = getattr(node, "tag", None)
    return str(v() if callable(v) else v)


def rebind_plot_groups(model: object, dataset_tag: str | None = None) -> list[str]:
    """把全部绘图组重绑到数据集（驱动；默认首个数据集）。

    Args:
        model: mph ``Model``（``model/'plots'``、``model/'datasets'`` 可导航）。
        dataset_tag: 目标数据集 tag；None → :func:`first_dataset_tag`。

    Returns:
        实际重绑的绘图组名字列表。

    Raises:
        ValueError: 无数据集可绑。
    """
    dset = dataset_tag or first_dataset_tag(model)
    if not dset:
        raise ValueError("模型无数据集可绑——先求解（solve）再重绑")
    rebound: list[str] = []
    for node in plot_group_nodes(model):
        node.property("data", dset)
        rebound.append(str(_name_of(node)))
    return rebound
