"""export_data 几何选择/过滤 + 变量未定义提示（backlog 20261026-044953-sim）。

契约：``--selection`` 落到 Data 导出节点的 ``selection`` 属性、``--filter-indicator``
落到 ``filterindicator``；``run()`` 报「变量未定义」类错误时错误文本追加**原始自由度
变量**提示（去场/分量前缀候选名——实锤场景 acpr1.* 整体失效而裸 p 全域可用）。
活体 COMSOL 核准属 sim 环节；本组只钉结构契约（假 Java 对象，离线确定）。
"""

from __future__ import annotations

from pathlib import Path

from pysci.skills.comsol_simulation.tools import export as _export
from pysci.skills.comsol_simulation.tools import simulation as _sim


class _FakeNode:
    def __init__(self, tag, ntype=""):
        self.tag, self.ntype = tag, ntype
        self.store: dict[str, object] = {}

    def set(self, key, value):
        self.store[key] = value


class _FakeExportNode(_FakeNode):
    def __init__(self, tag, ntype="", error: str | None = None):
        super().__init__(tag, ntype)
        self.error = error

    def run(self):
        if self.error:
            raise RuntimeError(self.error)
        p = Path(self.store["filename"])
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x,y,p\n0,0,1\n", encoding="utf-8")


class _FakeExportSeq:
    def __init__(self, error: str | None = None):
        self._nodes: dict[str, _FakeExportNode] = {}
        self.created: list[_FakeExportNode] = []
        self.error = error

    def tags(self):
        return list(self._nodes)

    def create(self, tag, kind):
        n = _FakeExportNode(tag, kind, self.error)
        self._nodes[tag] = n
        self.created.append(n)
        return n

    def get(self, tag):
        return self._nodes.get(tag)


class _FakeDatasetSeq:
    def tags(self):
        return ["dset1"]


class _FakeResult:
    def __init__(self, dsets, exports):
        self._dsets, self._exports = dsets, exports

    def dataset(self):
        return self._dsets

    def export(self):
        return self._exports


class _FakeModel:
    """最小假 model：仅实现 export/data 路径所需面（result()/dataset()/export()）。"""

    def __init__(self, error: str | None = None):
        self.dsets = _FakeDatasetSeq()
        self.exports = _FakeExportSeq(error)
        self._result = _FakeResult(self.dsets, self.exports)

    def result(self, tag=None):
        if tag is None:
            return self._result
        raise KeyError(tag)


def test_export_data_sets_selection_and_filter(tmp_path):
    """--selection/--filter-indicator 落到 Data 节点的 selection/filterindicator。"""
    m = _FakeModel()
    out = tmp_path / "d.csv"

    r = _export.export_data(
        m, "dset1", out, expr="p", selection="geom1_dom1", filter_indicator="x>0"
    )

    assert r.ok is True and out.exists()
    node = m.exports.created[0]
    assert node.ntype == "Data"
    assert node.store["data"] == "dset1"
    assert node.store["expr"] == "p"
    assert node.store["selection"] == "geom1_dom1"
    assert node.store["filterindicator"] == "x>0"


def test_export_data_without_selection_keeps_defaults(tmp_path):
    """不传两旗标：节点不设 selection/filterindicator（行为与现状一致）。"""
    m = _FakeModel()
    assert _export.export_data(m, "dset1", tmp_path / "d.csv", expr="p").ok is True
    node = m.exports.created[0]
    assert "selection" not in node.store
    assert "filterindicator" not in node.store


def test_undefined_variable_error_hints_raw_dof(tmp_path):
    """「变量未定义」：错误文本追加原始自由度提示（含去前缀候选名）。"""
    m = _FakeModel(error="变量未定义。 - 变量: comp1.acpr1.p_t")
    r = _export.export_data(m, "dset1", tmp_path / "d.csv", expr="acpr1.p_t")
    assert r.ok is False
    err = r.error or ""
    assert "变量未定义" in err
    assert "原始自由度变量" in err
    assert "`p_t`" in err


def test_other_errors_do_not_get_hint(tmp_path):
    """非变量未定义类错误不加提示（不制造噪声）。"""
    m = _FakeModel(error="磁盘已满")
    r = _export.export_data(m, "dset1", tmp_path / "d.csv", expr="p")
    assert r.ok is False
    assert "原始自由度变量" not in (r.error or "")


def test_cli_parses_selection_flags():
    """CLI：export data 暴露 --selection 与 --filter-indicator 并分发到 cmd_export_data。"""
    args = _sim.build_parser().parse_args(
        [
            "export",
            "data",
            "--mph",
            "m.mph",
            "--source",
            "dset1",
            "--out",
            "o.csv",
            "--expr",
            "p",
            "--selection",
            "geom1_dom1",
            "--filter-indicator",
            "x>0",
        ]
    )
    assert args.selection == "geom1_dom1"
    assert args.filter_indicator == "x>0"
    assert args.func is _sim.cmd_export_data
