"""plot-group 自动重绑机械断言（mock mph）：重绑计划 + 驱动判定。"""

from __future__ import annotations

import pytest

from pysci.skills.comsol_simulation import plotgroup


class _FakeNode:
    """鸭子类型节点（mph Node 口径：name()/tag()/property()）。"""

    def __init__(self, name: str, tag: str | None = None):
        self._name = name
        self._tag = tag or name
        self.store: dict[str, object] = {}

    def name(self):
        return self._name

    def tag(self):
        return self._tag

    def property(self, key: str, value: object):
        self.store[key] = value


class _FakeGroup:
    """鸭子类型分组节点（``model/'plots'`` 口径：可迭代出子节点）。"""

    def __init__(self, children):
        self._children = list(children)

    def __iter__(self):
        return iter(self._children)


class _FakeModel:
    def __init__(self, pg_names=("pg1",), dataset_tags=("dset1",)):
        self.groups = {
            "plots": _FakeGroup(_FakeNode(n) for n in pg_names),
            "datasets": _FakeGroup(_FakeNode(t, t) for t in dataset_tags),
        }

    def __truediv__(self, name: str) -> _FakeGroup:
        return self.groups[name]


def test_plan_rebind_pairs_tags_with_dataset():
    assert plotgroup.plan_rebind(["pg1", "pg2"], "dset2") == [
        ("pg1", "dset2"), ("pg2", "dset2")
    ]


def test_plan_rebind_rejects_empty_dataset():
    with pytest.raises(ValueError, match="不能为空"):
        plotgroup.plan_rebind(["pg1"], "")


def test_first_dataset_tag_takes_first():
    model = _FakeModel(dataset_tags=("d2", "d1"))
    assert plotgroup.first_dataset_tag(model) == "d2"


def test_first_dataset_tag_none_without_datasets():
    assert plotgroup.first_dataset_tag(_FakeModel(dataset_tags=())) is None


def test_plot_group_names():
    assert plotgroup.plot_group_names(_FakeModel(pg_names=("pgA", "pgB"))) == ["pgA", "pgB"]


def test_rebind_plot_groups_sets_data_property():
    model = _FakeModel(pg_names=("pg1", "pg2"), dataset_tags=("dset1",))
    rebound = plotgroup.rebind_plot_groups(model)
    assert rebound == ["pg1", "pg2"]
    for node in model.groups["plots"]:
        assert node.store == {"data": "dset1"}


def test_rebind_plot_groups_explicit_dataset():
    model = _FakeModel(pg_names=("pg1",), dataset_tags=("dset1",))
    plotgroup.rebind_plot_groups(model, "dset9")
    for node in model.groups["plots"]:
        assert node.store == {"data": "dset9"}


def test_rebind_plot_groups_without_dataset_raises():
    model = _FakeModel(dataset_tags=())
    with pytest.raises(ValueError, match="无数据集"):
        plotgroup.rebind_plot_groups(model)
