"""export._apply_color_range / _apply_polar_rmax 纯 Python 单测（假 Java 对象）。

覆盖 P0-3「统一标度」旗标的内核：Surface 色标只作用于 Surface 类特征、
PolarGroup 极径设 axislimits/rmin/rmax；返回实际生效项供 sidecar 记录。
"""

from __future__ import annotations

from pysci.skills.comsol_simulation.tools import export as _export


class _FakeFeature:
    def __init__(self, ftype):
        self.ftype = ftype
        self.store: dict[str, object] = {}

    def getType(self):
        return self.ftype

    def set(self, key, value):
        self.store[key] = value


class _FakeFeatureSeq:
    def __init__(self, feats):
        self._feats = feats

    def tags(self):
        return list(self._feats)


class _FakePG:
    def __init__(self, feats=None):
        self._feats = feats or {}
        self.store: dict[str, object] = {}

    def feature(self, tag=None):
        if tag is None:  # 无参调用 → 特征序列（供 _tags 枚举）
            return _FakeFeatureSeq(self._feats)
        return self._feats[tag]

    def set(self, key, value):
        self.store[key] = value


def test_color_range_only_surface_features():
    surf = _FakeFeature("Surface")
    line = _FakeFeature("Line")
    pg = _FakePG({"surf1": surf, "line1": line})
    applied = _export._apply_color_range(pg, (-160.0, 160.0))
    assert applied == ["surf1.rangecolormin=-160.0", "surf1.rangecolormax=160.0"]
    assert surf.store["rangecoloractive"] == "on"
    assert surf.store["rangecolormin"] == "-160.0"
    assert surf.store["rangecolormax"] == "160.0"
    assert line.store == {}  # 非 Surface 特征不受影响


def test_color_range_no_surface_returns_empty():
    pg = _FakePG({"line1": _FakeFeature("Line")})
    assert _export._apply_color_range(pg, (-1.0, 1.0)) == []


def test_polar_rmax_sets_axis_limits():
    pg = _FakePG()
    applied = _export._apply_polar_rmax(pg, (0.0, 60.0))
    assert applied == ["axislimits=on", "rmin=0.0", "rmax=60.0"]
    assert pg.store == {"axislimits": "on", "rmin": "0.0", "rmax": "60.0"}
