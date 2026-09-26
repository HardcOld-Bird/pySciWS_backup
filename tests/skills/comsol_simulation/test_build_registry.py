"""build recipe 注册表纯 Python 单测：装饰器注册、列举、按名取、默认参数合并。

不启动 COMSOL —— 用一个记录型 fake build 函数验证 run_recipe 的
「recipe 默认参数被 override 覆盖后交给纯函数」契约。真实建模原语（几何/物理场/网格）
需要活体模型，由 hardware 冒烟覆盖。
"""

from __future__ import annotations

import pytest

from pysci.skills.comsol_simulation.tools import build

_TMP_NAME = "_tmp_test_recipe"


@pytest.fixture()
def temp_recipe():
    """注册一个临时 recipe，测试后从全局注册表移除（避免污染其它用例）。"""
    calls: dict = {}

    @build.recipe(_TMP_NAME, description="temp", params={"a": 1, "b": 2}, tags=("test",))
    def _fn(model, p):
        calls["model"] = model
        calls["params"] = dict(p)

    yield calls
    build._REGISTRY.pop(_TMP_NAME, None)


def test_builtin_recipe_registered():
    names = [r.name for r in build.list_recipes()]
    assert "waveguide2d_acpr" in names
    r = build.get_recipe("waveguide2d_acpr")
    assert isinstance(r, build.Recipe)
    assert r.params.get("L") is not None
    assert "acoustics" in r.tags
    assert callable(r.fn)
    assert r.description


def test_get_recipe_missing_raises():
    with pytest.raises(KeyError):
        build.get_recipe("__no_such_recipe__")


def test_run_recipe_merges_params(temp_recipe):
    sentinel = object()
    build.run_recipe(sentinel, _TMP_NAME, {"b": 99, "c": 3})
    assert temp_recipe["model"] is sentinel
    # 默认 a 保留，b 被覆盖，c 新增
    assert temp_recipe["params"] == {"a": 1, "b": 99, "c": 3}


def test_run_recipe_defaults_when_no_override(temp_recipe):
    build.run_recipe(object(), _TMP_NAME)
    assert temp_recipe["params"] == {"a": 1, "b": 2}


def test_recipe_registered_in_registry(temp_recipe):
    assert _TMP_NAME in build._REGISTRY
    assert build.get_recipe(_TMP_NAME).tags == ("test",)
