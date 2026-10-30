"""``inspect feature-types`` 的单测（假 Java 对象，不启 COMSOL；backlog 20261017-090108-sim）。

COMSOL 没有「枚举接口允许的子特征类型」的 API——命令把组员手工的「试建猜」机械化：
列出已有子特征类型；``--candidates`` 逐个 trial-create（建完即删）给出**本模型实测**读数。
"""

from __future__ import annotations

from pysci.skills.comsol_simulation.tools import inspect as ins


class _FakeFeature:
    def __init__(self, typ: str):
        self._t = typ

    def getType(self) -> str:
        return self._t


class _FakeFeatureList:
    """冒充 COMSOL ``feature()`` 的 FeatureList：``tags()`` / ``fl(tag)`` / ``create`` / ``remove``。"""

    KNOWN = ("PerfectlyMatchedBoundary", "PlaneWaveRadiation")

    def __init__(self, *, remove_fails: bool = False):
        self.items: dict[str, str] = {"bpf1": "BackgroundPressureField"}
        self.remove_fails = remove_fails
        self.removed: list[str] = []

    def tags(self):
        return list(self.items)

    def __call__(self, tag: str):
        return _FakeFeature(self.items[tag])

    def create(self, tag: str, typ: str):
        if typ not in self.KNOWN:
            raise RuntimeError(f"未知特征 ID: {typ}")
        self.items[tag] = typ
        return _FakeFeature(typ)

    def remove(self, tag: str):
        if self.remove_fails:
            raise RuntimeError("remove not supported")
        self.removed.append(tag)
        self.items.pop(tag, None)


class _FakePhysics:
    def __init__(self, *, remove_fails: bool = False):
        self._fl = _FakeFeatureList(remove_fails=remove_fails)

    def feature(self, tag: str | None = None):
        return self._fl if tag is None else _FakeFeature(self._fl.items[tag])


def test_lists_existing_feature_types():
    node = _FakePhysics()
    rep = ins.probe_feature_types(node, [])
    assert rep.existing == [("bpf1", "BackgroundPressureField")]
    out = ins.format_feature_types(rep, path="component(comp1).physics(acpr)")
    assert "bpf1 :: BackgroundPressureField" in out


def test_probe_marks_creatable_and_unknown():
    node = _FakePhysics()
    rep = ins.probe_feature_types(
        node,
        [
            "PerfectlyMatchedBoundary",
            "PerfectlyMatchedBoundaryCondition",
            "PMLBoundary",
        ],
    )
    assert rep.creatable == ["PerfectlyMatchedBoundary"]
    assert rep.unknown == ["PerfectlyMatchedBoundaryCondition", "PMLBoundary"]
    out = ins.format_feature_types(rep, path="p")
    assert "[√] PerfectlyMatchedBoundary" in out
    assert "[✗] PMLBoundary（未知特征 ID）" in out


def test_probe_cleans_up_created_nodes():
    node = _FakePhysics()
    ins.probe_feature_types(node, ["PerfectlyMatchedBoundary", "PlaneWaveRadiation"])
    assert node._fl.removed, "建出来的探针必须删除"
    assert not [t for t in node._fl.items if t.startswith("_ftprobe")], node._fl.items


def test_probe_warns_when_cleanup_fails():
    node = _FakePhysics(remove_fails=True)
    rep = ins.probe_feature_types(node, ["PerfectlyMatchedBoundary"])
    assert rep.probes_left == 1
    assert "勿直接保存" in ins.format_feature_types(rep, path="p")


def test_format_hints_when_no_candidates():
    out = ins.format_feature_types(
        ins.probe_feature_types(_FakePhysics(), []), path="p"
    )
    assert "COMSOL 无" in out and "--candidates" in out


def test_cli_wires_feature_types():
    from pysci.skills.comsol_simulation.tools import simulation as sim

    a = sim.build_parser().parse_args(
        [
            "inspect",
            "feature-types",
            "--mph",
            "m.mph",
            "--path",
            "component(comp1).physics(acpr)",
            "--candidates",
            "A",
            "B",
        ]
    )
    assert a.func is sim.cmd_inspect_feature_types
    assert a.candidates == ["A", "B"]


def test_command_and_root_cause_discipline_are_documented():
    from pysci.paths import ORCHESTRATION_ROOT

    cli = (
        ORCHESTRATION_ROOT
        / "skills"
        / "comsol-simulation"
        / "references"
        / "cli-01-overview.md"
    ).read_text(encoding="utf-8")
    assert "inspect feature-types" in cli and "--candidates" in cli
    gen = (
        ORCHESTRATION_ROOT / "pods" / "reviewer" / "rubrics" / "generic.md"
    ).read_text(encoding="utf-8")
    assert "根因" in gen and "核验" in gen, "reviewer 量表缺仿真设置类根因核验项"
    acc = (
        ORCHESTRATION_ROOT
        / "skills"
        / "orchestration"
        / "references"
        / "acceptance-discipline.md"
    ).read_text(encoding="utf-8")
    assert "根因" in acc and "设置依据表" in acc, "验收纪律缺根因核验前置"
