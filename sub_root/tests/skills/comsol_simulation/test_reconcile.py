"""理论↔仿真 S 矩阵对账契约机械断言（纯函数，固定样例恒跑）。

覆盖（cmt-absorption.md 步骤 4/5 的测试件）：
- compare_s_matrix：完全一致 / 冻结判据贴边 / 超差 FAIL / 三指标数值；
- 同入同出契约：形状/频率/元数据/摘要不匹配拦截；
- 合成样例确定性（固定种子）与合成 S 矩阵冒烟报告（产物落
  ``tests/artifacts/comsol_simulation/``，测试产物区，不入库）。

冻结判据边界注记（浮点事实，勿"简化"）：``DOMINANT_ABS_TOL = 0.02`` 的双精度值
T = 5764607523034235·2^-58 **不是** 2^-55 的倍数；而主导元（|S_th| > 0.2）幅度差
``||th|-|sim||`` 在双精度下必为 2^-55 的倍数（两侧幅度同处 [0.125, 0.5) binade，
减法由 Sterbenz 引理精确），故 |Δ| 恰为 0.02 **不可达**。贴边语义用双向夹逼钉死：
下侧最近可表达差 D_lo = T − 3·2^-58（≤ 阈值）必须 PASS，上侧最近可表达差
D_hi = T + 5·2^-58（> 阈值）必须 FAIL；比较方向"含等号"（``<=``）另用注入指标
在恰值处直测。
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict

import pytest

from pysci.paths import TESTS_ROOT
from pysci.skills.comsol_simulation import reconcile
from pysci.skills.comsol_simulation.validate import CheckResult

_ARTIFACTS = TESTS_ROOT / "artifacts" / "comsol_simulation"

# 主导元量级（|S| ∈ (0.2, 0.25)）的双精度间距；贴边差值只能取其整数倍。
_Q = 2.0**-55
# ≤ 0.02 的最大可表达幅度差 = T − 3·2^-58（实测 float 构造见边界用例）。
_D_BELOW = (reconcile.DOMINANT_ABS_TOL // _Q) * _Q
# > 0.02 的最小可表达幅度差 = T + 5·2^-58。
_D_ABOVE = _D_BELOW + _Q
# 贴边用例的基准幅度：2^-55 倍数、> DOMINANT_FLOOR、base+Δ < 0.25。
_BASE = 0.21875


def _case(
    s,
    *,
    frequency_hz=10e9,
    labels=("P1", "P2"),
    digest="digest-0",
    source="theory",
):
    return reconcile.SMatrixCase(
        s=s,
        frequency_hz=frequency_hz,
        port_labels=labels,
        params_digest=digest,
        source=source,
    )


# ---------- 冻结常量守卫（改动 = 违反冻结纪律，须用户裁决） ----------


def test_frozen_constants():
    assert reconcile.DOMINANT_ABS_TOL == 0.02
    assert reconcile.REL_FLOOR == 1e-3
    assert reconcile.DOMINANT_FLOOR == 0.2


# ---------- SMatrixCase 构造/校验 ----------


def test_smatrix_case_normalizes_input():
    c = _case([[1, 0], [0, 1 + 2j]])
    assert c.s == ((1 + 0j, 0j), (0j, 1 + 2j))
    assert all(isinstance(v, complex) for row in c.s for v in row)


def test_smatrix_case_rejects_empty():
    with pytest.raises(ValueError, match="方阵"):
        _case(())


def test_smatrix_case_rejects_non_square():
    with pytest.raises(ValueError, match="方阵"):
        _case(((1, 2, 3), (4, 5, 6)))
    with pytest.raises(ValueError, match="方阵"):
        _case(((1, 2), (3,)))


def test_smatrix_case_rejects_non_finite():
    with pytest.raises(ValueError, match="非有限"):
        _case(((complex("nan"), 0), (0, 0)))
    with pytest.raises(ValueError, match="非有限"):
        _case(((complex(0, float("inf")), 0), (0, 0)))


@pytest.mark.parametrize("bad_freq", [0.0, -1.0, float("nan"), float("inf")])
def test_smatrix_case_rejects_bad_frequency(bad_freq):
    with pytest.raises(ValueError, match="frequency_hz"):
        _case(((1, 0), (0, 1)), frequency_hz=bad_freq)


def test_smatrix_case_rejects_label_count_mismatch():
    with pytest.raises(ValueError, match="端口标签"):
        _case(((1, 0), (0, 1)), labels=("P1",))
    with pytest.raises(ValueError, match="端口标签"):
        _case(((1, 0), (0, 1)), labels=())


def test_smatrix_case_rejects_empty_digest():
    with pytest.raises(ValueError, match="params_digest"):
        _case(((1, 0), (0, 1)), digest="")


# ---------- digest_of ----------


def test_digest_of_is_stable_16_hex():
    d = reconcile.digest_of("f0=10e9;J=2")
    assert d == hashlib.sha256(b"f0=10e9;J=2").hexdigest()[:16]
    assert len(d) == 16
    assert reconcile.digest_of("f0=10e9;J=2") == d


def test_digest_of_distinguishes_inputs():
    assert reconcile.digest_of("a") != reconcile.digest_of("b")


# ---------- check_same_in_same_out（同入同出契约） ----------


def test_contract_passes_on_identical_semantics():
    th = _case(((0.5, 0.1), (0.1, 0.5)), digest="same")
    sim = _case(((0.5, 0.1), (0.1, 0.5)), digest="same", source="comsol")
    r = reconcile.check_same_in_same_out(th, sim)
    assert isinstance(r, CheckResult)
    assert r.ok and r.value is None
    assert r.message.startswith("PASS")


def test_contract_blocks_digest_mismatch():
    th = _case(((1, 0), (0, 1)), digest="a")
    sim = _case(((1, 0), (0, 1)), digest="b")
    r = reconcile.check_same_in_same_out(th, sim)
    assert not r.ok and "同入违反" in r.message


def test_contract_blocks_shape_mismatch():
    th = _case(((1, 0), (0, 1)))
    # 3×3 须配 3 标签（SMatrixCase 构造约束）；标签差异也会同报，此处只钉形状拦截。
    sim = _case(((1, 0, 0), (0, 1, 0), (0, 0, 1)), labels=("P1", "P2", "P3"))
    r = reconcile.check_same_in_same_out(th, sim)
    assert not r.ok and "形状违反" in r.message


def test_contract_blocks_frequency_mismatch():
    th = _case(((1, 0), (0, 1)), frequency_hz=10e9)
    sim = _case(((1, 0), (0, 1)), frequency_hz=11e9)
    r = reconcile.check_same_in_same_out(th, sim)
    assert not r.ok and "频率违反" in r.message


def test_contract_blocks_port_metadata_mismatch_and_is_order_sensitive():
    th = _case(((1, 0), (0, 1)), labels=("P1", "P2"))
    sim = _case(((1, 0), (0, 1)), labels=("P2", "P1"))  # 仅顺序不同也拦
    r = reconcile.check_same_in_same_out(th, sim)
    assert not r.ok and "端口元数据违反" in r.message


def test_contract_lists_all_violations():
    th = _case(((1, 0), (0, 1)), frequency_hz=10e9, digest="a")
    sim = _case(((1, 0), (0, 1)), frequency_hz=11e9, digest="b")
    r = reconcile.check_same_in_same_out(th, sim)
    assert not r.ok
    assert "同入违反" in r.message and "频率违反" in r.message


def test_contract_custom_name():
    th = sim = _case(((1, 0), (0, 1)))
    r = reconcile.check_same_in_same_out(th, sim, name="cmt_case")
    assert r.name == "cmt_case"


# ---------- amplitude_metrics（三指标数值 + 掩膜语义） ----------


def test_amplitude_metrics_exact_values():
    th = ((0.5, 0.25), (0.125, 0.75))
    sim = ((0.5, 0.25), (0.125, 0.625))
    m = reconcile.amplitude_metrics(th, sim)
    assert m.max_abs_diff == 0.125
    assert m.max_rel_diff == pytest.approx(0.125 / 0.75)  # 唯一非零相对差
    assert m.rms_diff == 0.0625  # sqrt((0.125²)/4) 恰为 2^-4
    assert m.dominant_max_abs_diff == 0.125
    assert m.n_elements == 4
    assert m.n_dominant == 3  # |th|>0.2：0.5 / 0.25 / 0.75（0.125 不计）


def test_amplitude_metrics_masks_rel_floor_and_dominant_floor():
    # 2^-13 ≈ 1.22e-4 < REL_FLOOR：近零元相对误差不入账；
    # 0.1875 < DOMINANT_FLOOR：其 0.3125 大差不进冻结判据读数。
    th = ((0.5, 2.0**-13), (0.1875, 0.25))
    sim = ((0.5, 0.5), (0.5, 0.25))
    m = reconcile.amplitude_metrics(th, sim)
    assert m.max_abs_diff == 0.5 - 2.0**-13
    assert m.max_rel_diff == pytest.approx(0.3125 / 0.1875)
    assert m.max_rel_diff < 10.0  # 若 2^-13 入账会爆到 ~4000
    assert m.dominant_max_abs_diff == 0.0
    assert m.n_dominant == 2
    assert m.rms_diff == pytest.approx(
        math.sqrt(((0.5 - 2.0**-13) ** 2 + 0.3125**2) / 4)
    )


def test_amplitude_metrics_floor_boundaries_are_strict():
    # |th| 恰为阈值：REL_FLOOR(1e-3) 与 DOMINANT_FLOOR(0.2) 都是严格大于才入账。
    th = ((0.2, 1e-3),)
    sim = ((0.4, 0.5),)
    m = reconcile.amplitude_metrics(th, sim)
    assert m.n_dominant == 0
    assert m.dominant_max_abs_diff == 0.0
    assert m.max_rel_diff == 1.0  # 只剩 |th|=0.2 元素：0.2/0.2（1e-3 元素被掩）


def test_amplitude_metrics_no_dominant_reads_zero():
    th = ((0.1, 0.05), (0.05, 0.1))
    sim = ((0.2, 0.2), (0.2, 0.2))
    m = reconcile.amplitude_metrics(th, sim)
    assert m.n_dominant == 0
    assert m.dominant_max_abs_diff == 0.0
    assert m.max_abs_diff == pytest.approx(0.15)


def test_amplitude_metrics_rejects_empty():
    with pytest.raises(ValueError, match="不得为空"):
        reconcile.amplitude_metrics([], [])


def test_amplitude_metrics_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="形状不一致"):
        reconcile.amplitude_metrics([[1, 2], [3, 4]], [[1, 2], [3]])
    with pytest.raises(ValueError, match="形状不一致"):
        reconcile.amplitude_metrics([[1, 2]], [[1, 2], [3, 4]])


# ---------- compare_s_matrix（冻结判据） ----------


def test_compare_identical_passes_zero():
    th, sim = reconcile.synthetic_case_pair(7, perturbation=0.0)
    r = reconcile.compare_s_matrix(th, sim)
    assert isinstance(r, CheckResult)
    assert r.ok and r.value == 0.0
    assert r.name == "compare_s_matrix"
    assert r.message.startswith("PASS")


def test_compare_over_tol_fails():
    th, sim = reconcile.synthetic_case_pair(7, perturbation=0.05)
    r = reconcile.compare_s_matrix(th, sim)
    assert not r.ok
    assert r.value > reconcile.DOMINANT_ABS_TOL
    assert "FAIL" in r.message and "冻结判据" in r.message


def test_compare_message_carries_three_metrics():
    th = _case(((0.5, 0.25), (0.25, 0.5)))
    sim = _case(((0.6, 0.25), (0.25, 0.5)))
    r = reconcile.compare_s_matrix(th, sim)
    d = abs(0.5 - 0.6)  # 唯一非零幅度差
    expected = (
        f"max_abs={d:.3g}, max_rel={d / 0.5:.3g}, rms={math.sqrt(d * d / 4):.3g}"
    )
    assert expected in r.message
    assert r.value == d  # value = 冻结判据读数（主导元最大绝对幅度差）
    assert not r.ok  # d ≈ 0.1 > 0.02


def _boundary_pair(diff: float):
    """构造主导元幅度差恰为 diff 的对（base 与 base+diff 均为 2^-55 倍数）。"""
    th = _case((((_BASE + diff), 0.25), (0.25, 0.25)))
    sim = _case(((_BASE, 0.25), (0.25, 0.25)))
    return th, sim


def test_frozen_boundary_pass_below():
    """贴边下侧：≤0.02 的最大可表达差 → PASS（判据含等号方向的实数域实证）。"""
    assert _D_BELOW < reconcile.DOMINANT_ABS_TOL < _D_ABOVE
    th, sim = _boundary_pair(_D_BELOW)
    r = reconcile.compare_s_matrix(th, sim)
    assert r.ok
    assert r.value == _D_BELOW  # 减法 Sterbenz 精确，恰为构造值


def test_frozen_boundary_fail_above():
    """贴边上侧：>0.02 的最小可表达差 → FAIL。"""
    th, sim = _boundary_pair(_D_ABOVE)
    r = reconcile.compare_s_matrix(th, sim)
    assert not r.ok
    assert r.value == _D_ABOVE


def test_frozen_criterion_inclusive_at_exact_tol(monkeypatch):
    """比较方向直测：恰为阈值的主导元差 PASS ⇒ 实现为 ``<=``（含等号）。

    真实矩阵的幅度差在双精度下不可达恰值 0.02（见模块 docstring），故恰值
    行为用注入指标钉死比较方向；邻域行为由上下贴边用例双向钉死。
    """
    th, sim = _boundary_pair(_D_BELOW)
    exact = reconcile.AmplitudeMetrics(
        max_abs_diff=reconcile.DOMINANT_ABS_TOL,
        max_rel_diff=0.1,
        rms_diff=0.05,
        dominant_max_abs_diff=reconcile.DOMINANT_ABS_TOL,
        n_elements=4,
        n_dominant=4,
    )
    monkeypatch.setattr(reconcile, "amplitude_metrics", lambda a, b: exact)
    r = reconcile.compare_s_matrix(th, sim)
    assert r.ok
    assert r.value == reconcile.DOMINANT_ABS_TOL


def test_compare_blocks_on_contract_violation():
    """同入同出契约不成立 ⇒ compare_s_matrix 直接 ValueError 拦截，不进对账。"""
    th = _case(((0.5, 0.1), (0.1, 0.5)), frequency_hz=10e9)
    sim = _case(((0.5, 0.1), (0.1, 0.5)), frequency_hz=11e9)
    with pytest.raises(ValueError, match="同入同出"):
        reconcile.compare_s_matrix(th, sim)


def test_compare_rejects_no_dominant():
    th = _case(((0.1, 0.05), (0.05, 0.1)))
    sim = _case(((0.1, 0.05), (0.05, 0.1)))
    with pytest.raises(ValueError, match="无主导元"):
        reconcile.compare_s_matrix(th, sim)


def test_compare_custom_name():
    th, sim = reconcile.synthetic_case_pair(3, perturbation=0.0)
    r = reconcile.compare_s_matrix(th, sim, name="cmt_case")
    assert r.name == "cmt_case"


# ---------- synthetic_case_pair（合成样例确定性） ----------


def test_synthetic_pair_deterministic_same_seed():
    t1, s1 = reconcile.synthetic_case_pair(42, perturbation=0.01)
    t2, s2 = reconcile.synthetic_case_pair(42, perturbation=0.01)
    assert t1 == t2 and s1 == s2  # 逐比特可复现


def test_synthetic_pair_varies_with_seed():
    t1, _ = reconcile.synthetic_case_pair(1)
    t2, _ = reconcile.synthetic_case_pair(2)
    assert t1.s != t2.s


def test_synthetic_pair_reciprocal_lossless():
    th, _ = reconcile.synthetic_case_pair(42)
    r, t = th.s[0][0], th.s[0][1]
    assert th.s[0][1] == th.s[1][0]  # 对称 = 互易（逐元素恒等）
    assert abs(r) ** 2 + abs(t) ** 2 == pytest.approx(1.0, rel=1e-12)


def test_synthetic_pair_all_elements_dominant():
    th, _ = reconcile.synthetic_case_pair(42)
    m = reconcile.amplitude_metrics(th.s, th.s)
    assert m.n_dominant == 4  # |r|,|t| ∈ (0.38, 0.93) 全部 > DOMINANT_FLOOR


def test_synthetic_pair_zero_perturbation_identical():
    th, sim = reconcile.synthetic_case_pair(42, perturbation=0.0)
    assert th.s == sim.s


@pytest.mark.parametrize("bad_p", [-0.1, float("nan"), float("inf")])
def test_synthetic_pair_rejects_bad_perturbation(bad_p):
    with pytest.raises(ValueError, match="perturbation"):
        reconcile.synthetic_case_pair(1, perturbation=bad_p)


# ---------- 合成 S 矩阵冒烟（产物：tests/artifacts/，不入库） ----------


def test_synthetic_smoke_report():
    """冒烟全链：同入同出 → compare_s_matrix 全指标，PASS/FAIL 两例实证落盘。"""
    results = {}
    for verdict, p in (("PASS", 0.01), ("FAIL", 0.05)):
        th, sim = reconcile.synthetic_case_pair(20261117, perturbation=p)
        contract = reconcile.check_same_in_same_out(th, sim)
        assert contract.ok  # 同入同出恒成立（同种子同参数）
        compare = reconcile.compare_s_matrix(th, sim)
        metrics = asdict(reconcile.amplitude_metrics(th.s, sim.s))
        assert compare.ok is (verdict == "PASS")
        assert compare.ok == (compare.value <= reconcile.DOMINANT_ABS_TOL)
        results[verdict] = {
            "perturbation": p,
            "contract_message": contract.message,
            "compare_message": compare.message,
            "metrics": metrics,
            "criterion": {
                "name": "dominant_abs_tol",
                "operator": "<=",
                "tol": reconcile.DOMINANT_ABS_TOL,
            },
        }

    _ARTIFACTS.mkdir(parents=True, exist_ok=True)
    out = _ARTIFACTS / "reconcile_smoke_report.json"
    out.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    back = json.loads(out.read_text(encoding="utf-8"))
    assert set(back) == {"PASS", "FAIL"}
    assert set(back["PASS"]["metrics"]) == {
        "max_abs_diff",
        "max_rel_diff",
        "rms_diff",
        "dominant_max_abs_diff",
        "n_elements",
        "n_dominant",
    }
    assert back["PASS"]["metrics"]["n_elements"] == 4
