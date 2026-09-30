"""export._apply_color_range / _apply_polar_rmax 纯 Python 单测（假 Java 对象）。

覆盖 P0-3「统一标度」旗标的内核：Surface 色标只作用于 Surface 类特征、
PolarGroup 极径设 axislimits/rmin/rmax；返回实际生效项供 sidecar 记录。
"""

from __future__ import annotations

import json

import numpy as np

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


def test_free_export_tag_skips_existing_nodes(monkeypatch):
    """回归：加载已含 img1/img2/img3 的模型时，自动生成须跳到 img4。

    （原模块级裸计数器从 1 递增，第二次导出会生成 img2 与模型自带节点碰撞。）
    """
    monkeypatch.setattr(_export, "_existing_export_tags", lambda model: {"img1", "img2", "img3"})
    assert _export._free_export_tag(object(), "img") == "img4"


def test_free_export_tag_first_free_on_empty_model(monkeypatch):
    monkeypatch.setattr(_export, "_existing_export_tags", lambda model: set())
    assert _export._free_export_tag(object(), "img") == "img1"


def test_json_default_serializes_numpy_scalars_and_arrays():
    """回归：sidecar payload 含 numpy 标量/数组时 json.dumps 不再抛 TypeError。"""
    payload = {
        "crop_box_px": [np.int64(3), np.int64(4), np.int64(200), np.int64(300)],
        "interior": {"std": np.float64(12.5), "blank": np.bool_(False), "unique_q": np.int64(7)},
        "arr": np.array([1.0, 2.0]),
    }
    round_tripped = json.loads(json.dumps(payload, default=_export._json_default))
    assert round_tripped == {
        "crop_box_px": [3, 4, 200, 300],
        "interior": {"std": 12.5, "blank": False, "unique_q": 7},
        "arr": [1.0, 2.0],
    }


class _JStringLike:
    """模拟 JPype ``java.lang.String`` 代理：非 Python ``str`` 子类，但有 ``__str__``。

    E2 实机复现：``plotgroup`` 若直接取自 ``result().tags()``（绕过 inspect._tags 的
    ``str()`` 归一），进 sidecar payload 后 json 无法序列化，旧 ``_json_default`` 抛
    ``TypeError`` 致整份 sidecar 丢失。
    """

    def __init__(self, v: str) -> None:
        self._v = v

    def __str__(self) -> str:
        return self._v


def test_json_default_falls_back_to_str_for_unknown_types():
    """回归：非 str/非 numpy 对象兜底转 str，而非抛 TypeError（sidecar 绝不因元数据崩溃）。"""
    assert _export._json_default(_JStringLike("pg1")) == "pg1"


def test_sidecar_payload_with_jstring_like_plotgroup_serializes():
    """回归：plotgroup 为 JPype 代理时，sidecar payload 仍能整体序列化落盘。"""
    payload = {
        "plotgroup": _JStringLike("pg1"),
        "crop_box_px": [np.int64(1), 2, 3, 4],
        "interior": {"std": np.float64(9.0), "blank": np.bool_(False), "unique_q": np.int64(5)},
    }
    out = json.loads(json.dumps(payload, default=_export._json_default))
    assert out["plotgroup"] == "pg1"
    assert out["crop_box_px"] == [1, 2, 3, 4]
