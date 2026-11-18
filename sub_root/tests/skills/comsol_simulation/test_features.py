"""feature/node-set 机械断言（mock mph）：路径解析 / 通配 / 试建探测 / 属性写入。"""

from __future__ import annotations

import pytest

from pysci.skills.comsol_simulation import features

# ---------------------------------------------------------------------------
# parse_node_path / match_path（纯函数）
# ---------------------------------------------------------------------------


def test_parse_node_path_typed_segments():
    segs = features.parse_node_path("component(comp1).physics(acpr).feature(bpf1)")
    assert segs == [("component", "comp1"), ("physics", "acpr"), ("feature", "bpf1")]


def test_parse_node_path_bare_segment():
    assert features.parse_node_path("result") == [(None, "result")]


def test_parse_node_path_rejects_empty():
    with pytest.raises(ValueError, match="为空"):
        features.parse_node_path("  ")


def test_parse_node_path_rejects_garbage_segment():
    with pytest.raises(ValueError, match="无法解析"):
        features.parse_node_path("component(comp1)..physics(acpr)")


def test_match_path_wildcard_single_segment():
    assert features.match_path("result(pg1).feature(surf1)", "result(pg*).feature(*)")
    assert not features.match_path("result(pg1)", "result(pg*).feature(*)")


def test_match_path_exact():
    assert features.match_path("result(pg1)", "result(pg1)")
    assert not features.match_path("result(pg2)", "result(pg1)")


# ---------------------------------------------------------------------------
# probe_feature_types（mock 节点：合法类型白名单）
# ---------------------------------------------------------------------------


class _ProbeHandle:
    """试建探针句柄（mph Node.remove() 口径）。"""

    def __init__(self, parent: _FakeParent, name: str):
        self._parent = parent
        self._name = name

    def remove(self):
        self._parent.on_remove(self._name)


class _FakeParent:
    """鸭子类型父节点：create 仅接受 valid_types（mph ``create(*type, name=…)``）。"""

    def __init__(self, valid_types=("Good",), no_remove: bool = False,
                 can_create: bool = True):
        self.valid_types = set(valid_types)
        self.no_remove = no_remove
        self.created: list[str] = []
        self.removed: list[str] = []
        self.children = lambda: []
        if can_create:
            self.create = self._create

    def _create(self, typ: str, name: str | None = None) -> _ProbeHandle:
        if typ not in self.valid_types:
            raise ValueError(f"unknown feature type: {typ}")
        self.created.append(name or typ)
        return _ProbeHandle(self, name or typ)

    def on_remove(self, name: str):
        if self.no_remove:
            raise RuntimeError("stubborn")
        self.removed.append(name)


def test_probe_feature_types_splits_valid_invalid():
    parent = _FakeParent(valid_types=("Good", "AlsoGood"))
    rep = features.probe_feature_types(parent, ["Good", "Bad", "AlsoGood"])
    assert rep.valid == ["Good", "AlsoGood"]
    assert rep.invalid == ["Bad"]
    assert rep.probes_left == 0
    assert parent.created == ["_ftprobe0", "_ftprobe2"]
    assert parent.removed == ["_ftprobe0", "_ftprobe2"]  # 探针即建即删


def test_probe_feature_types_counts_leftover_probes():
    parent = _FakeParent(valid_types=("Good",), no_remove=True)
    rep = features.probe_feature_types(parent, ["Good"])
    assert rep.probes_left == 1


def test_probe_feature_types_without_create_is_unknown():
    parent = _FakeParent(can_create=False)
    rep = features.probe_feature_types(parent, ["X"])
    assert rep.unknown == ["X"] and rep.valid == [] and rep.invalid == []


def test_format_feature_types_mentions_leftovers():
    rep = features.FeatureTypesReport(probes_left=2)
    assert "勿直接保存" in features.format_feature_types(rep, path="p")


# ---------------------------------------------------------------------------
# set_node_props（k=v 写入；int 消歧；单条失败汇总）
# ---------------------------------------------------------------------------


class _SetNode:
    def __init__(self, fail_keys=()):
        self.store: dict[str, object] = {}
        self.fail_keys = set(fail_keys)

    def property(self, key: str, value: object):
        if key in self.fail_keys:
            raise ValueError("bad property")
        self.store[key] = value


def test_coerce_value_kinds():
    assert features.coerce_value("3") == (3, "int")
    assert features.coerce_value("2.5") == (2.5, "float")
    assert features.coerce_value("on") == ("on", "str")
    assert features.coerce_value("true") == (True, "bool")


def test_set_node_props_applies_all():
    node = _SetNode()
    res = features.set_node_props(node, ["n=3", "x=2.5", "flag=on"])
    assert res.all_ok
    assert node.store == {"n": 3, "x": 2.5, "flag": "on"}


def test_set_node_props_single_failure_summarized():
    node = _SetNode(fail_keys={"bad"})
    res = features.set_node_props(node, ["bad=1", "good=2"])
    assert not res.all_ok
    assert res.applied == ["good=2"] and res.failed == ["bad=1"]
    assert node.store == {"good": 2}


def test_set_node_props_rejects_pair_without_equals():
    with pytest.raises(ValueError, match="k=v"):
        features.set_node_props(_SetNode(), ["noequals"])


# ---------------------------------------------------------------------------
# collect_nodes（树遍历 + 通配）
# ---------------------------------------------------------------------------


class _TreeNode:
    def __init__(self, tag: str, children: tuple = ()):
        self.tag = tag
        self._children = list(children)

    def children(self):
        return self._children


def test_collect_nodes_matches_pattern():
    model = _TreeNode("root", (
        _TreeNode("pg1", (_TreeNode("surf1"),)),
        _TreeNode("pg2", (_TreeNode("surf2"),)),
    ))
    # 路径以裸 tag 逐段拼接；模式段名（feature）只作语义标注、不参与匹配
    hits = features.collect_nodes(model, "pg*.feature(*)")
    assert sorted(hits) == ["pg1.surf1", "pg2.surf2"]


def test_collect_nodes_no_match():
    model = _TreeNode("root", (_TreeNode("pg1"),))
    assert features.collect_nodes(model, "geom*") == []
