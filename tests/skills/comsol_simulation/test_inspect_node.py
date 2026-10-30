"""inspect.resolve_path / dump_node / format_node_dump 纯 Python 单测（假 Java 对象，不启 COMSOL）。"""

from __future__ import annotations

import pytest

from pysci.skills.comsol_simulation.tools import inspect as ins


class _FakeSel:
    def entities(self, dim):
        return [1, 2] if dim == 1 else None  # 未用维返回 None（模拟 Java 侧不可读）


class _FakeFeature:
    def getType(self):
        return "BackgroundPressureField"

    def label(self):
        return "Background L (+45)"

    def properties(self):
        return ["pamp", "PressureFieldType"]

    def getString(self, prop):
        return {"pamp": "pampL", "PressureFieldType": "PlaneWave"}[prop]

    def getAllowedPropertyValues(self, prop):
        if prop == "PressureFieldType":
            return ["PlaneWave", "SphericalWave", "UserDefined"]
        raise RuntimeError("non-enum")

    def selection(self):
        return _FakeSel()


class _FakePhysics:
    def feature(self, tag):
        if tag == "bpf1":
            return _FakeFeature()
        raise KeyError(tag)


class _FakeComp:
    def physics(self, tag):
        if tag == "acpr":
            return _FakePhysics()
        raise KeyError(tag)


class _FakeModel:
    """冒充 Java model：resolve_path 经 _jmodel 的 getattr(model,'java',model) 落到本对象。"""

    def component(self, tag):
        if tag == "comp1":
            return _FakeComp()
        raise KeyError(tag)


PATH = "component(comp1).physics(acpr).feature(bpf1)"


class _FakeTableNode:
    """表格/数据集节点的占位：只求能定位到，不求属性。"""

    kind = "table-or-dataset"


class _FakeResultTree:
    """冒充结果树：``result(tag)`` 只认 plot group；表格/数据集在容器层（COMSOL 实况）。"""

    def __call__(self, tag):
        raise RuntimeError(f"未知结果特征。标记: {tag}")

    def table(self, tag):
        if tag == "tblSL":
            return _FakeTableNode()
        raise KeyError(tag)

    def dataset(self, tag):
        if tag == "dset1":
            return _FakeTableNode()
        raise KeyError(tag)


class _FakeModelWithResult(_FakeModel):
    """与 COMSOL 同形：``result()`` 返回结果树节点；``result(tag)`` 只查 plot group。"""

    def result(self, tag=None):
        tree = _FakeResultTree()
        return tree if tag is None else tree(tag)


def test_resolve_path_walks_segments():
    node = ins.resolve_path(_FakeModel(), PATH)
    assert isinstance(node, _FakeFeature)


def test_resolve_path_falls_back_to_table_container():
    """``result(tblSL)`` 落空时回退 ``result.table(tblSL)``（backlog 20261028-195852-sim）。"""
    node = ins.resolve_path(_FakeModelWithResult(), "result(tblSL)")
    assert isinstance(node, _FakeTableNode)


def test_resolve_path_falls_back_to_dataset_container():
    node = ins.resolve_path(_FakeModelWithResult(), "result(dset1)")
    assert isinstance(node, _FakeTableNode)


def test_resolve_path_still_raises_when_no_container_has_tag():
    """两容器都无此 tag → 仍抛首次异常（原错误语义与文案不变）。"""
    with pytest.raises(RuntimeError, match="未知结果特征"):
        ins.resolve_path(_FakeModelWithResult(), "result(nope)")


def test_resolve_path_plain_segments_unchanged():
    """非结果族路径的回退探测不得改变原语义：无容器层时首次异常照抛。"""
    with pytest.raises(KeyError):
        ins.resolve_path(_FakeModel(), "component(comp9)")


def test_resolve_path_bad_syntax_raises():
    with pytest.raises(ValueError):
        ins.resolve_path(_FakeModel(), "component(comp1).physics(acpr).feature(bpf1")


def test_resolve_path_missing_attr_raises():
    with pytest.raises(AttributeError):
        ins.resolve_path(_FakeModel(), "component(comp1).nope(x)")


def test_dump_node_state():
    info = ins.dump_node(ins.resolve_path(_FakeModel(), PATH))
    assert info["type"] == "BackgroundPressureField"
    assert info["label"].startswith("Background L")
    by_name = {row["name"]: row for row in info["properties"]}
    assert by_name["pamp"]["value"] == "pampL"
    assert by_name["PressureFieldType"]["allowed"] == [
        "PlaneWave",
        "SphericalWave",
        "UserDefined",
    ]
    assert "allowed" not in by_name["pamp"]  # 非枚举属性不附 allowed
    assert info["selection"] == {"dim1": 2}


def test_format_node_dump_readable():
    info = ins.dump_node(ins.resolve_path(_FakeModel(), PATH))
    text = ins.format_node_dump(info, path=PATH)
    assert PATH in text
    assert "BackgroundPressureField" in text
    assert "pamp = 'pampL'" in text
    assert "allowed: PlaneWave|SphericalWave|UserDefined" in text
    assert "selection: dim1=2" in text


# ---------------------------------------------------------------------------
# 几何节点：selection() 是选择特征列表（无 entities，曾抛 AttributeError）
# ---------------------------------------------------------------------------
class _FakeSelFeatureList:
    def __init__(self, tags):
        self._tags = list(tags)

    def tags(self):
        return self._tags

    def size(self):
        return len(self._tags)


class _FakeChildFeature:
    def __init__(self, typ, label):
        self._typ, self._label = typ, label

    def getType(self):
        return self._typ

    def label(self):
        return self._label


class _FakeFeatureList:
    def __init__(self, feats):
        self._feats = feats

    def tags(self):
        return list(self._feats)

    def get(self, tag):
        return self._feats[tag]


class _FakeGeomNode:
    """冒充 component(comp1).geom(geom1)：选择是特征列表，无 entities。"""

    def getType(self):
        return "GeomSequence"

    def label(self):
        return "几何 1"

    def feature(self):
        return _FakeFeatureList(
            {
                "r1": _FakeChildFeature("Rectangle", "矩形 1"),
                "fin": _FakeChildFeature("FormUnion", "形成联合体"),
            }
        )

    def selection(self):
        return _FakeSelFeatureList(["sel1"])

    def getNVertices(self):
        return 19

    def getNEdges(self):
        return 24

    def getNDomains(self):
        return 6

    def getNFaces(self):
        raise RuntimeError("unknown property")

    def objectNames(self):
        return ["r1", "fin"]


def test_dump_node_geometry_node_does_not_crash():
    info = ins.dump_node(_FakeGeomNode())
    assert info["selection"] == {"tags": ["sel1"]}
    assert [f["tag"] for f in info["features"]] == ["r1", "fin"]
    assert info["features"][0]["type"] == "Rectangle"
    assert info["geometry"]["vertices"] == 19
    assert "faces" not in info["geometry"]
    assert info["objects"] == ["r1", "fin"]
    text = ins.format_node_dump(info, path="component(comp1).geom(geom1)")
    assert "features: (2)" in text
    assert "r1 :: Rectangle  [矩形 1]" in text
    assert "domains=6" in text
    assert "objects: ['r1', 'fin']" in text


def test_empty_selection_feature_list_reports_count():
    node = _FakeGeomNode()
    node.selection = lambda: _FakeSelFeatureList([])
    info = ins.dump_node(node)
    assert info["selection"] == {"feature_count": 0}
    assert "selection: feature_count=0" in ins.format_node_dump(info)


# ---------------------------------------------------------------------------
# 命名选择输入：Selection 型属性 + getExtraSelectionNames()
# ---------------------------------------------------------------------------
class _FakeInputSel:
    def __init__(self, dims=None, objects=None):
        self._dims = dims or {}
        self._objects = objects

    def entities(self, dim):
        return self._dims.get(dim)

    def objects(self):
        return self._objects


class _FakePartitionFeature:
    def getType(self):
        return "PartitionDomains"

    def label(self):
        return "分割域 1"

    def properties(self):
        return ["partitionwith", "domain", "edge"]

    def getString(self, prop):
        if self.getValueType(prop) == "Selection":
            raise RuntimeError("未知属性")  # COMSOL 对 Selection 型属性抛 FlException
        return "object"

    def getValueType(self, prop):
        return "Selection" if prop in ("domain", "edge") else "String"

    def getExtraSelectionNames(self):
        return ["destinationDomains"]

    def selection(self, name):
        if name == "domain":
            return _FakeInputSel(objects=["pare2"])
        if name == "edge":
            return _FakeInputSel(objects=[])
        if name == "destinationDomains":
            return _FakeInputSel(dims={1: 3})
        raise KeyError(name)


def test_selection_inputs_named_and_extra():
    rows = ins.selection_inputs(_FakePartitionFeature())
    assert [r["name"] for r in rows] == ["domain", "edge", "destinationDomains"]
    assert rows[0]["objects"] == ["pare2"]
    assert rows[2]["dim1"] == 3


def test_dump_node_renders_selection_inputs():
    info = ins.dump_node(_FakePartitionFeature())
    text = ins.format_node_dump(
        info, path="component(comp1).geom(geom1).feature(pard1)"
    )
    assert "selection inputs: (3)" in text
    assert "domain: objects=['pare2']" in text
    assert "destinationDomains: dim1=3" in text
