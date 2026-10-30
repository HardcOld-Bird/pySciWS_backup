"""colorbar 端标/场图 vmax 的 **v2 阶梯**向上取整（`field.ceil_label`）行为钉子。

用户裁决 2026-10-27（v2）：端标合法值 = ``d1.d2 × 10^k``——``d1∈1..9``；``d2∈{0,5}``；
``d1=1`` 时另含 ``d2∈{2,4,6,8}``（并集 ``{0,2,4,5,6,8}``，1.0 保留合法）；第三位起为 0；
仍为向上取整（``>= x`` 的最小合法值）。旧名 `ceil_one_sig` 保留为等价包装（既有调用点
无需改动），径向刻度 rmax 的 `_nice_ceil` 口径**不在**本规则内。
"""

from __future__ import annotations

import numpy as np
import pytest

from pysci.skills.scientific_plotting.tools import field

#: 任务书九组钉值。第 9 组按规则本体取 0.16：0.15 < 0.151 且规则明写「向上取整（≥实测 max）」，
#: 取 0.15 会裁峰；任务书原示例 0.151→0.15 与该条自相矛盾（交付说明已注明）。
CASES = (
    (65.48, 70),
    (1.4655, 1.5),
    (1.8277, 2.0),
    (2.11, 2.5),
    (1.1968, 1.2),
    (0.6436, 0.65),
    (0.98, 1.0),
    (9.7, 10),
    (0.151, 0.16),
)

#: 阶梯覆盖面（d1=1 特例 d2∈{2,4,6,8}、d1>1 的 .5 档、跨十进位）。
EXTRA_CASES = (
    (5.1, 5.5),
    (7.6, 8.0),
    (0.11, 0.12),
    (0.13, 0.14),
    (0.17, 0.18),
    (3.6, 4.0),
    (0.0048, 0.005),
    (99.9, 100),
)


@pytest.mark.parametrize(("raw", "want"), CASES)
def test_ceil_label_pinned_values(raw, want):
    assert field.ceil_label(raw) == want
    assert field.ceil_label(float(raw)) == float(want)


@pytest.mark.parametrize(("raw", "want"), EXTRA_CASES)
def test_ceil_label_extra_ladder_coverage(raw, want):
    assert field.ceil_label(raw) == want


@pytest.mark.parametrize(("raw", "want"), CASES + EXTRA_CASES)
def test_ceil_label_is_ceiling_and_idempotent(raw, want):
    got = field.ceil_label(raw)
    assert got >= raw, "向上取整：结果不得低于实测值（否则色标裁峰）"
    assert field.ceil_label(got) == got, "合法形态（v2 阶梯）应原样返回"


def test_ceil_label_boundary_exact_legal_values_not_bumped():
    """恰为合法值不升档：70→70、1.5→1.5（以及 1.0/2.5 等阶梯点）。"""
    for legal in (70, 1.5, 1.0, 2.5, 0.65, 9.5, 100):
        assert field.ceil_label(legal) == legal


def test_ceil_label_rejects_non_positive_and_non_finite():
    for bad in (0, -1.0, -0.0048, float("inf"), float("-inf"), float("nan")):
        with pytest.raises(ValueError):
            field.ceil_label(bad)


def test_ceil_one_sig_legacy_name_delegates_to_v2():
    for raw in (2.11, 65.48, 0.6436, 1.4655):
        assert field.ceil_one_sig(raw) == field.ceil_label(raw)
    assert field.ceil_one_sig(2.11) == 2.5


# ---------------------------------------------------------------------------
# 接线：场图 vmax 与 colorbar 端标默认走 v2 阶梯（显式量程原样放行）
# ---------------------------------------------------------------------------
def test_independent_norms_default_ends_rounded():
    norms = field.independent_norms({"a": np.array([-2.11, 0.6436])})
    assert norms["a"].vmax == 0.65
    assert norms["a"].vmin == -2.5

    sym = field.independent_norms({"a": np.array([-66.13, 3.87])}, symmetric=True)
    assert sym["a"].vmax == 70 and sym["a"].vmin == -70


def test_independent_norms_explicit_limits_not_rounded():
    norms = field.independent_norms(
        {"a": np.array([0.0, 1.0])}, vmin={"a": -1.5}, vmax={"a": 2.11}
    )
    assert norms["a"].vmax == 2.11 and norms["a"].vmin == -1.5


def test_render_field_panel_default_ends_rounded(tmp_path, monkeypatch):
    src = tmp_path / "f.csv"
    src.write_text("0, 0, 66.13\n0.1, 0.1, -2.11\n0.2, 0, 1.0\n", encoding="utf-8")
    captured: dict = {}
    orig = field.add_field_panel

    def spy(ax, xy, values, **kw):
        captured.update(kw)
        return orig(ax, xy, values, **kw)

    monkeypatch.setattr(field, "add_field_panel", spy)
    out = field.render_field_panel(src, tmp_path / "p.png", cmap="bwr", dpi=60)
    assert out.exists()
    assert captured["vmax"] == 70 and captured["vmin"] == -2.5


def test_render_field_panel_explicit_limits_not_rounded(tmp_path, monkeypatch):
    src = tmp_path / "f.csv"
    src.write_text("0, 0, 66.13\n0.1, 0.1, -2.11\n0.2, 0, 1.0\n", encoding="utf-8")
    captured: dict = {}
    orig = field.add_field_panel

    def spy(ax, xy, values, **kw):
        captured.update(kw)
        return orig(ax, xy, values, **kw)

    monkeypatch.setattr(field, "add_field_panel", spy)
    field.render_field_panel(
        src, tmp_path / "p.png", cmap="bwr", vmin=-1.5, vmax=2.11, dpi=60
    )
    assert captured["vmax"] == 2.11 and captured["vmin"] == -1.5
