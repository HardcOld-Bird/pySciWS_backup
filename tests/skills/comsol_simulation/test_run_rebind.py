"""run._rebind_datasets 纯 Python 单测（假 Java 对象，不启 COMSOL）。

覆盖 P0-4「run solve 自动重绑数据集」：求解后把全部绘图组绑到首个 dataset，
无 dataset / 无 result 时静默降级为空列表。
"""

from __future__ import annotations

from pysci.skills.comsol_simulation.tools import run as _run


class _FakePG:
    def __init__(self, tag):
        self.tag = tag
        self.data: str | None = None

    def set(self, key, value):
        if key == "data":
            self.data = value  # 返回 None（模拟 Java set）

    def getString(self, key):
        return self.data if key == "data" else ""


class _FakeDatasetSeq:
    def __init__(self, tags):
        self._tags = tags

    def tags(self):
        return list(self._tags)


class _FakeResult:
    """Results 序列节点：只有 dataset()/tags()，**不可**按 tag 调用。

    忠实反映真实 COMSOL：取具体绘图组必须走 ``jm.result(tag)``（模型级），
    而非 ``jm.result()(tag)``。故本类故意不提供 ``__call__``——若实现回退到
    ``result(pg)`` 会抛 TypeError，单测即失败。
    """

    def __init__(self, pg_tags=("pg_field", "pg_far"), dsets=("dset1",)):
        self._pgs = {t: _FakePG(t) for t in pg_tags}
        self._dsets = list(dsets)

    def dataset(self):
        return _FakeDatasetSeq(self._dsets)

    def tags(self):
        return list(self._pgs)

    def pg(self, tag):
        """供测试断言取回绘图组（非 COMSOL API，仅测试便利）。"""
        return self._pgs[tag]


class _FakeJModel:
    """无 .java 属性 → _jmodel 返回自身。

    ``result()`` 返回 Results 序列；``result(tag)`` 返回具体绘图组（与真实 COMSOL 一致）。
    """

    def __init__(self, result):
        self._result = result

    def result(self, tag=None):
        if tag is None:
            return self._result
        return self._result._pgs[tag]


def test_rebind_binds_all_plotgroups():
    res = _FakeResult()
    rebound = _run._rebind_datasets(_FakeJModel(res))
    assert rebound == ["pg_field", "pg_far"]
    assert res.pg("pg_field").data == "dset1"
    assert res.pg("pg_far").data == "dset1"


def test_rebind_no_dataset_is_noop():
    rebound = _run._rebind_datasets(_FakeJModel(_FakeResult(dsets=[])))
    assert rebound == []


def test_rebind_no_result_node_is_noop():
    class _NoResult:
        pass

    assert _run._rebind_datasets(_NoResult()) == []
