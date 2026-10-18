"""标注几何语义机械检查（ANNOTATIONS.yaml ↔ 实际 Artist）的回归测试。

覆盖任务书验收：声明 2 框实画 3 框必须报错、合规样例零报；四类判据（计数/包含+边距/
相对大小/连接线）各带阳性与阴性用例；原语通用化（box/polygon/line）；audit 端到端接入。
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402
from matplotlib.patches import Polygon, Rectangle  # noqa: E402

from pysci.skills.scientific_plotting.tools import audit, geometry, runner  # noqa: E402

_SPEC = {
    "counts": [{"view": "L", "kind": "box", "expected": 2}],
    "contain": [{"inner": "mark", "outer": "field", "margin_ratio": 0.02}],
    "sizes": [{"outer": "scan", "inner": "mark", "min_ratio": 1.05}],
    "connectors": [{"source": "mark", "target": "scan", "tolerance_ratio": 0.05}],
    "require": ["field", "mark", "scan"],
}


def _figure(
    *,
    n_boxes: int = 2,
    mark_rect: tuple[float, float, float, float] = (2, 2, 2, 2),
    scan_rect: tuple[float, float, float, float] = (8, 2, 8, 8),
    connector: bool = True,
    inner_polygon: bool = False,
):
    fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_gid("view:L")
    field = Polygon([[1, 1], [9, 1], [9, 9], [1, 9]], closed=True, fill=False)
    field.set_gid("annot:field;role=field")
    ax.add_patch(field)
    if inner_polygon:
        x, y, w, h = mark_rect
        mark = Polygon(
            [[x, y], [x + w, y], [x + w, y + h], [x, y + h]], closed=True, fill=False
        )
    else:
        x, y, w, h = mark_rect
        mark = Rectangle((x, y), w, h, fill=False)
    mark.set_gid(
        "annot:mark;role=mark;kind=box" if inner_polygon else "annot:mark;role=mark"
    )
    ax.add_patch(mark)
    sx, sy, sw, sh = scan_rect
    scan = Rectangle((sx, sy), sw, sh, fill=False)
    scan.set_gid("annot:scan;role=scan")
    ax.add_patch(scan)
    if n_boxes >= 3:
        third = Rectangle((2, 5), 3, 3, fill=False)
        third.set_gid("annot:third;role=mark")
        ax.add_patch(third)
    if connector:
        ax.plot([8, 2], [2, 2], ls="--")[0].set_gid("annot:conn")
    return fig


def _codes(rep: geometry.GeometryReport) -> list[str]:
    return [i.code for i in rep.issues]


# ---------------------------------------------------------------------------
# 四类判据：阳性 + 阴性
# ---------------------------------------------------------------------------
def test_positive_two_boxes_is_compliant():
    fig = _figure()
    try:
        rep = geometry.audit_annotations(fig, _SPEC)
    finally:
        plt.close(fig)
    assert rep.ok, [i.message for i in rep.issues]
    assert rep.metrics["annot_count[L/box]"] == 2
    assert rep.metrics["annot_primitives"] == 4  # field + mark + scan + connector


def test_positive_polygon_primitives_generalized():
    """内框用 Polygon（非 Rectangle）同样通过——不硬编码矩形。"""
    fig = _figure(inner_polygon=True)
    try:
        rep = geometry.audit_annotations(fig, _SPEC)
    finally:
        plt.close(fig)
    assert rep.ok, [i.message for i in rep.issues]


def test_declared_two_but_drew_three_fails_count():
    fig = _figure(n_boxes=3)
    try:
        rep = geometry.audit_annotations(fig, _SPEC)
    finally:
        plt.close(fig)
    assert not rep.ok
    count_issues = [i for i in rep.issues if i.code == "annot-count"]
    assert count_issues, [i.message for i in rep.issues]
    assert "实际 3 个" in count_issues[0].message
    assert "third" in count_issues[0].message


def test_box_out_of_field_fails_contain():
    fig = _figure(mark_rect=(8.5, 2, 2, 2))
    try:
        rep = geometry.audit_annotations(fig, _SPEC)
    finally:
        plt.close(fig)
    assert "annot-contain" in _codes(rep)


def test_touching_field_edge_fails_contain():
    fig = _figure(mark_rect=(1, 2, 2, 2))
    try:
        rep = geometry.audit_annotations(fig, _SPEC)
    finally:
        plt.close(fig)
    assert "annot-contain" in _codes(rep)


def test_near_miss_margin_fails_contain():
    """完整包含但边距不足（< margin_ratio）也算缺陷。"""
    fig = _figure(mark_rect=(1.1, 1.1, 2, 2))
    try:
        rep = geometry.audit_annotations(fig, _SPEC)
    finally:
        plt.close(fig)
    assert "annot-contain" in _codes(rep)


def test_size_inverted_fails():
    """外框画得比内框小（大小倒置）→ annot-size。"""
    fig = _figure(scan_rect=(8, 2, 1, 1))
    try:
        rep = geometry.audit_annotations(fig, _SPEC)
    finally:
        plt.close(fig)
    assert "annot-size" in _codes(rep)


def test_missing_connector_fails():
    fig = _figure(connector=False)
    try:
        rep = geometry.audit_annotations(fig, _SPEC)
    finally:
        plt.close(fig)
    assert "annot-connect" in _codes(rep)


def test_missing_required_primitive_fails():
    spec = dict(_SPEC, require=["field", "mark", "scan", "not_drawn"])
    fig = _figure()
    try:
        rep = geometry.audit_annotations(fig, spec)
    finally:
        plt.close(fig)
    assert "annot-missing" in _codes(rep)


# ---------------------------------------------------------------------------
# 声明缺失/空声明行为
# ---------------------------------------------------------------------------
def test_tagged_but_no_declaration_warns_only():
    fig = _figure()
    try:
        rep = geometry.audit_annotations(fig, None)
    finally:
        plt.close(fig)
    assert rep.ok  # WARN 不阻断
    assert any(i.level == geometry.WARN and i.code == "annotations" for i in rep.issues)


def test_untagged_and_no_declaration_is_silent():
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    try:
        rep = geometry.audit_annotations(fig, None)
    finally:
        plt.close(fig)
    assert not rep.issues


def test_declaration_without_primitives_warns():
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    try:
        rep = geometry.audit_annotations(fig, _SPEC)
    finally:
        plt.close(fig)
    assert any(i.level == geometry.WARN and i.code == "annot-empty" for i in rep.issues)


# ---------------------------------------------------------------------------
# 声明文件查找与解析
# ---------------------------------------------------------------------------
def test_find_sidecar_and_load_yaml_json(tmp_path):
    assert geometry.find_sidecar(tmp_path) is None
    y = tmp_path / "ANNOTATIONS.yaml"
    y.write_text("counts:\n  - view: L\n    expected: 2\n", encoding="utf-8")
    assert geometry.find_sidecar(tmp_path) == y
    assert geometry.load_spec(y)["counts"][0]["expected"] == 2

    j = tmp_path / "annotations.json"
    j.write_text(json.dumps({"counts": []}), encoding="utf-8")
    (tmp_path / "ANNOTATIONS.yaml").unlink()
    assert geometry.find_sidecar(tmp_path) == j
    assert geometry.load_spec(j) == {"counts": []}


def test_load_spec_rejects_non_mapping(tmp_path):
    p = tmp_path / "ANNOTATIONS.json"
    p.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ValueError):
        geometry.load_spec(p)


# ---------------------------------------------------------------------------
# audit 端到端：声明 2 框实画 3 框 → FAIL；合规 → PASS
# ---------------------------------------------------------------------------
_PIPELINE = """\
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle


def build_figure(style=None, **kwargs):
    fig, ax = plt.subplots()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_gid("view:L")
    field = Polygon([[1, 1], [9, 1], [9, 9], [1, 9]], closed=True, fill=False)
    field.set_gid("annot:field;role=field")
    ax.add_patch(field)
    mark = Rectangle((2, 2), 2, 2, fill=False)
    mark.set_gid("annot:mark;role=mark")
    ax.add_patch(mark)
    scan = Rectangle((8, 2), 8, 8, fill=False)
    scan.set_gid("annot:scan;role=scan")
    ax.add_patch(scan)
    {extra}
    ax.plot([8, 2], [2, 2], ls="--")[0].set_gid("annot:conn")
    return fig
"""

_THIRD_BOX = (
    "third = Rectangle((2, 5), 3, 3, fill=False)\n"
    '    third.set_gid("annot:third;role=mark")\n'
    "    ax.add_patch(third)"
)

_SPEC_YAML = """\
counts:
  - view: L
    kind: box
    expected: 2
contain:
  - inner: mark
    outer: field
    margin_ratio: 0.02
sizes:
  - outer: scan
    inner: mark
    min_ratio: 1.05
connectors:
  - source: mark
    target: scan
    tolerance_ratio: 0.05
require: [field, mark, scan]
"""


@pytest.fixture
def fig_roots(tmp_path, monkeypatch):
    data_figures = tmp_path / "data" / "research" / "1_gain_ep" / "article" / "figures"
    src_figures = (
        tmp_path / "src" / "pysci" / "research" / "gain_ep" / "article" / "figures"
    )
    data_figures.mkdir(parents=True)
    src_figures.mkdir(parents=True)
    monkeypatch.setattr(runner, "figures_code_root", lambda research: src_figures)
    return {"data": data_figures, "src": src_figures}


def _make_figure_dir(roots, slug: str, *, extra: str = "") -> object:
    fd = roots["data"] / slug
    (fd / "out").mkdir(parents=True, exist_ok=True)
    (roots["src"] / f"{slug}.py").write_text(
        _PIPELINE.format(extra=extra), encoding="utf-8"
    )
    (fd / "ANNOTATIONS.yaml").write_text(_SPEC_YAML, encoding="utf-8")
    return fd


def test_audit_e2e_compliant_two_boxes_zero_errors(fig_roots):
    fd = _make_figure_dir(fig_roots, "fig_ok")
    rep = audit.audit_figure_dir(fd)
    assert rep.ok, [i.message for i in rep.issues]
    assert rep.metrics["annot_count[L/box]"] == 2
    assert rep.metrics["annotations_file"] == "ANNOTATIONS.yaml"


def test_audit_e2e_three_boxes_reports_error(fig_roots):
    fd = _make_figure_dir(fig_roots, "fig_bad", extra=_THIRD_BOX)
    rep = audit.audit_figure_dir(fd)
    assert not rep.ok
    assert "annot-count" in [i.code for i in rep.issues]


def test_cli_audit_reports_annotation_count(fig_roots, capsys):
    from pysci.skills.scientific_plotting.tools import figures

    fd = _make_figure_dir(fig_roots, "fig_cli", extra=_THIRD_BOX)
    rc = figures.main(["audit", str(fd)])
    out = capsys.readouterr().out
    assert rc == 1
    assert "annot-count" in out
