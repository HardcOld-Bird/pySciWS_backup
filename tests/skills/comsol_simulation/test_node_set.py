"""inspect.set_node_props / _coerce_value 纯 Python 单测（假 Java 对象，不启 COMSOL）。

覆盖 P0-1「通用节点 setter」：逐条 set、单条失败汇总不中断、整数 JInt 包裹、参数校验。
"""

from __future__ import annotations

import pytest

from pysci.skills.comsol_simulation.tools import inspect as ins


class _FakeNode:
    """可 set 的假节点；fail_on 中的属性 set 时抛异常（模拟坏属性名/非法值）。"""

    def __init__(self, fail_on=()):
        self.store: dict[str, object] = {}
        self._fail = set(fail_on)

    def set(self, key, value):
        if key in self._fail:
            raise RuntimeError(f"bad property {key}")
        self.store[key] = value

    def getClass(self):
        return self

    def getName(self):
        return "com.fake.Node"


class _FakeModel:
    """resolve_path 经 _jmodel 落到本对象；result(tag) 返回预置节点。"""

    def __init__(self, node):
        self._node = node

    def result(self, tag):
        return self._node


PATH = "result(pg_field)"


def test_coerce_value_passthrough_non_int():
    assert ins._coerce_value("on") == "on"
    assert ins._coerce_value("3.5") == "3.5"
    assert ins._coerce_value("-160.0") == "-160.0"


def test_coerce_value_int_wrapped():
    # 有 JPype → JInt；无 → 退回原字符串。两种情况 str() 都应可解析回整数。
    val = ins._coerce_value("5")
    assert int(str(val)) == 5


def test_set_node_props_all_ok():
    node = _FakeNode()
    res = ins.set_node_props(
        _FakeModel(node), PATH, ["rangecoloractive=on", "rangecolormin=-160"]
    )
    assert res.all_ok
    assert res.ok == ["rangecoloractive=on", "rangecolormin=-160"]
    assert node.store["rangecoloractive"] == "on"


def test_set_node_props_partial_failure_summarized():
    node = _FakeNode(fail_on=("bogus",))
    res = ins.set_node_props(
        _FakeModel(node), PATH, ["good=1", "bogus=2", "also_good=3"]
    )
    assert not res.all_ok
    assert res.ok == ["good=1", "also_good=3"]  # 坏属性不中断其余
    assert len(res.failed) == 1 and res.failed[0].startswith("bogus=2")
    assert "bogus" not in node.store


def test_set_node_props_records_class():
    res = ins.set_node_props(_FakeModel(_FakeNode()), PATH, ["a=1"])
    assert res.node_class == "com.fake.Node"


def test_set_node_props_report_readable():
    res = ins.set_node_props(
        _FakeModel(_FakeNode(fail_on=("x",))), PATH, ["a=1", "x=2"]
    )
    text = res.report()
    assert PATH in text and "ok   : a=1" in text and "FAIL : x=2" in text


def test_set_node_props_empty_raises():
    with pytest.raises(ValueError):
        ins.set_node_props(_FakeModel(_FakeNode()), PATH, [])


def test_set_node_props_missing_eq_raises():
    with pytest.raises(ValueError):
        ins.set_node_props(_FakeModel(_FakeNode()), PATH, ["no_equals_here"])


def test_set_node_props_bad_path_propagates():
    with pytest.raises(AttributeError):
        ins.set_node_props(_FakeModel(_FakeNode()), "nope(x)", ["a=1"])
