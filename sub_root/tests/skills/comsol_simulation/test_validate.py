"""验证函数族机械断言（纯函数）：参考比对 / 能量守恒 / 收敛阶。"""

from __future__ import annotations

import pytest

from pysci.skills.comsol_simulation import validate


def test_relative_error_basic():
    assert validate.relative_error(1.05, 1.0) == pytest.approx(0.05)


def test_relative_error_zero_reference_is_absolute():
    assert validate.relative_error(0.3, 0.0) == pytest.approx(0.3)


def test_relative_error_rejects_nonfinite():
    with pytest.raises(ValueError):
        validate.relative_error(float("nan"), 1.0)


def test_compare_to_reference_passes_within_tolerance():
    r = validate.compare_to_reference([1.0, 2.0], [1.0, 2.01], rtol=0.01)
    assert r.ok
    # 最大相对误差 = |2.0-2.01|/2.01（参考值为分母）
    assert r.value == pytest.approx(abs(2.0 - 2.01) / 2.01)


def test_compare_to_reference_fails_on_outlier():
    r = validate.compare_to_reference([1.0, 3.0], [1.0, 2.0], rtol=0.01)
    assert not r.ok
    assert "第 1 点越界" in r.message


def test_compare_to_reference_length_mismatch():
    with pytest.raises(ValueError, match="等长"):
        validate.compare_to_reference([1.0], [1.0, 2.0])


def test_energy_residual_conserved_is_zero():
    assert validate.energy_residual(10.0, 8.0, 2.0) == pytest.approx(0.0)


def test_energy_residual_nonzero():
    assert validate.energy_residual(10.0, 8.5, 2.0) == pytest.approx(0.05)


def test_energy_residual_zero_incoming_raises():
    with pytest.raises(ValueError, match="无定义"):
        validate.energy_residual(0.0, 1.0)


def test_check_energy_conservation_pass_and_fail():
    ok = validate.check_energy_conservation(10.0, 9.95, 0.0, tol=0.01)
    bad = validate.check_energy_conservation(10.0, 8.0, 0.0, tol=0.01)
    assert ok.ok and not bad.ok
    assert bad.value == pytest.approx(0.2)


def test_convergence_orders_quadratic():
    # 误差 ∝ h²：h=4,2,1 → e=4,1,0.25 → 两段阶都=2
    orders = validate.convergence_orders([4.0, 1.0, 0.25], [4.0, 2.0, 1.0])
    assert orders == pytest.approx([2.0, 2.0])


def test_convergence_orders_rejects_non_decreasing_h():
    with pytest.raises(ValueError, match="严格递减"):
        validate.convergence_orders([1.0, 1.0], [1.0, 2.0])


def test_convergence_orders_rejects_length_mismatch():
    with pytest.raises(ValueError):
        validate.convergence_orders([1.0], [1.0, 2.0])


def test_check_convergence_pass_on_expected_order():
    r = validate.check_convergence([4.0, 1.0, 0.25], [4.0, 2.0, 1.0], expected_order=2.0)
    assert r.ok
    assert r.value == pytest.approx(2.0)


def test_check_convergence_fail_on_wrong_order():
    r = validate.check_convergence([4.0, 2.0, 1.0], [4.0, 2.0, 1.0], expected_order=2.0)
    assert not r.ok  # 实测阶 1.0 ≉ 期望 2.0
