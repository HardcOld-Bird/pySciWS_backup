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


def test_resolve_path_walks_segments():
    node = ins.resolve_path(_FakeModel(), PATH)
    assert isinstance(node, _FakeFeature)


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
    assert by_name["PressureFieldType"]["allowed"] == ["PlaneWave", "SphericalWave", "UserDefined"]
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
