"""export.export_polar_csv 纯 Python 单测（假 Java 对象）。

覆盖 G1：沿**角度序列**（1° 步长）把外场量（如 ``acpr.efc1.pext``）导出为数值 CSV 的工具面。
断言的是**结构契约**——在底数据集上建 ``ParCurve2D`` 参数圆弧、置 ``global=on``（网格外求值）、
角度以**度**为参数、点数 = range/step+1，随后 Data 导出节点以「角度 + 场值」两列落盘。

（活体 COMSOL 上的数值正确性核准属 p1 sim 环节——本仓单 license，且本环节不动任何 .mph。）
"""

from __future__ import annotations

from pathlib import Path

from pysci.skills.comsol_simulation.tools import export as _export


class _FakeNode:
    def __init__(self, tag, ntype):
        self.tag = tag
        self.ntype = ntype
        self.store: dict[str, object] = {}

    def set(self, key, value):
        self.store[key] = value


class _FakeDatasetSeq:
    def __init__(self, tags):
        self._nodes = {t: _FakeNode(t, "Solution") for t in tags}
        self.created: list[_FakeNode] = []

    def tags(self):
        return list(self._nodes)

    def create(self, tag, ntype):
        n = _FakeNode(tag, ntype)
        self._nodes[tag] = n
        self.created.append(n)
        return n

    def get(self, tag):
        return self._nodes[tag]


class _FakeExportNode(_FakeNode):
    def run(self):
        p = Path(self.store["filename"])
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("alpha,acpr.efc1.pext\n0,1.0\n1,0.9\n", encoding="utf-8")


class _FakeExportSeq:
    def __init__(self):
        self._nodes: dict[str, _FakeExportNode] = {}
        self.created: list[_FakeExportNode] = []

    def tags(self):
        return list(self._nodes)

    def create(self, tag, kind):
        n = _FakeExportNode(tag, kind)
        self._nodes[tag] = n
        self.created.append(n)
        return n

    def get(self, tag):
        return self._nodes.get(tag)


class _FakeResult:
    def __init__(self, dsets, exports):
        self._dsets = dsets
        self._exports = exports

    def dataset(self):
        return self._dsets

    def export(self):
        return self._exports


class _FakeModel:
    """最小假 model：仅实现 export/data 路径所需面（result()/dataset()/export()）。"""

    def __init__(self, dataset_tags=("dset1",)):
        self.dsets = _FakeDatasetSeq(dataset_tags)
        self.exports = _FakeExportSeq()
        self._result = _FakeResult(self.dsets, self.exports)

    def result(self, tag=None):
        if tag is None:
            return self._result
        raise KeyError(tag)


def test_polar_creates_parcurve2d_arc_with_degree_parameter(tmp_path):
    m = _FakeModel()
    out = tmp_path / "far_L.csv"

    r = _export.export_polar_csv(
        m, out, expr="acpr.efc1.pext", radius=0.4, center=(0.283, 0.0)
    )

    assert r.ok is True
    assert r.kind == "polar"
    assert out.exists()
    assert len(m.dsets.created) == 1
    node = m.dsets.created[0]
    assert node.ntype == "ParCurve2D"
    assert node.store["data"] == "dset1"  # 默认底数据集
    assert node.store["par1"] == "alpha"
    assert node.store["parmin1"] == 0.0
    assert node.store["parmax1"] == 180.0
    assert node.store["res"] == 181  # 0..180 步 1 → 181 点
    assert node.store["global"] is True  # 网格外（远场）求值
    # 参数以「度」进入表达式，圆心/半径注入正确
    assert "cos(alpha*pi/180)" in node.store["exprx"]
    assert "0.283" in node.store["exprx"]
    assert "sin(alpha*pi/180)" in node.store["expry"]
    assert "0.4" in node.store["expry"]


def test_polar_export_node_has_angle_column_and_field_expr(tmp_path):
    m = _FakeModel()
    out = tmp_path / "far_L.csv"

    _export.export_polar_csv(m, out, expr="acpr.efc1.pext", radius=0.4)

    assert len(m.exports.created) == 1
    enode = m.exports.created[0]
    assert enode.ntype == "Data"
    assert enode.store["data"] == m.dsets.created[0].tag  # 指向临时 ParCurve2D
    assert enode.store["expr"] == ["alpha", "acpr.efc1.pext"]
    assert enode.store["filename"] == str(out)


def test_polar_respects_step_and_range(tmp_path):
    m = _FakeModel()
    _export.export_polar_csv(
        m,
        tmp_path / "o.csv",
        expr="acpr.efc1.pext",
        radius=0.4,
        angle_min=10.0,
        angle_max=20.0,
        angle_step=2.0,
    )
    n = m.dsets.created[0]
    assert n.store["parmin1"] == 10.0
    assert n.store["parmax1"] == 20.0
    assert n.store["res"] == 6  # (20-10)/2 + 1


def test_polar_explicit_source_selects_dataset(tmp_path):
    m = _FakeModel(dataset_tags=("dset1", "dset2"))
    _export.export_polar_csv(
        m, tmp_path / "o.csv", expr="acpr.efc1.pext", radius=0.4, source="dset2"
    )
    assert m.dsets.created[0].store["data"] == "dset2"


def test_polar_dataset_tag_avoids_collision(tmp_path):
    m = _FakeModel(dataset_tags=("dset1", "pc2d1"))
    _export.export_polar_csv(m, tmp_path / "o.csv", expr="acpr.efc1.pext", radius=0.4)
    assert m.dsets.created[0].tag == "pc2d2"  # 跳过已占用的 pc2d1


def test_polar_zero_step_fails_cleanly(tmp_path):
    m = _FakeModel()
    r = _export.export_polar_csv(
        m, tmp_path / "o.csv", expr="acpr.efc1.pext", radius=0.4, angle_step=0.0
    )
    assert r.ok is False
    assert "角度序列点数不足" in (r.error or "")
    assert m.dsets.created == []  # 未建数据集


def test_polar_multiple_exprs_kept(tmp_path):
    m = _FakeModel()
    _export.export_polar_csv(
        m,
        tmp_path / "o.csv",
        expr=["acpr.efc1.pext", "abs(acpr.efc1.pext)"],
        radius=0.4,
    )
    assert m.exports.created[0].store["expr"] == [
        "alpha",
        "acpr.efc1.pext",
        "abs(acpr.efc1.pext)",
    ]
