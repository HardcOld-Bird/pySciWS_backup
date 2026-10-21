"""CMT 反射型管槽超表面 S 矩阵通用化的单元测试。

覆盖任务锚点：
- 文献 EP/DP 名义参数回归（与通用化前逐元一致）；
- 互易性自通道恒等式 r_0(θ) ≡ r_0(−θ)（2×2 特例即 S11≡S22，任意非对称几何、
  机器精度）及其多端口推广 S_{+θ}[p,p] = S_{−θ}[n(p),n(p)]；
- 周期 D、入射角 θ_i、频率 f0 解耦为自由输入，端口维度随 (f0, θ_i, D) 变化；
- 任意 J 与逐槽独立 c_s/w_s/h_s/x_s；
- ``__post_init__`` 全缺省才用默认、部分缺省报错；
- 几何/物理参数校验；
- 单频与频率批量接口一致。
"""

import dataclasses
import time

import numpy as np
import pytest

from pysci.research.gain_ep.theory import cmt_reflection_s_matrix as cmt
from pysci.research.gain_ep.theory.cmt_reflection_s_matrix import (
    CMTForwardKernel,
    MetagratingParameterError,
    MetagratingParams,
    clear_forward_cache,
    compute_s_matrix,
    propagating_orders,
    solve_reflection_cmt,
    solve_s_matrix,
    solve_s_matrix_batch,
    sweep_medium,
    sweep_s_matrix,
)

LAM = 343.0 / 3430.0  # 名义波长 0.1 m
C0 = 343.0

# 通用化前（git HEAD）实现给出的名义参数 S 矩阵，作为回归基准。
EP_REF = np.array(
    [
        [
            1.4230572678471607e-02 + 1.7044319768739782e-03j,
            5.4966221775683766e-01 + 8.3514968402301737e-01j,
        ],
        [
            2.5500394355901392e-04 - 3.7661663456826098e-04j,
            1.4230572678471420e-02 + 1.7044319768747007e-03j,
        ],
    ]
)
DP_REF = np.array(
    [
        [
            -3.9524933906268176e-04 + 0.0076196627188723j,
            3.8066254521618345e-05 + 0.00019518268255586j,
        ],
        [
            1.6758421401052530e-03 + 0.00177081105358301j,
            -3.9524933906234317e-04 + 0.0076196627188725j,
        ],
    ]
)


def _asym(**overrides) -> MetagratingParams:
    """构造一组非对称几何（逐槽尺寸/声速均不同）的多端口测试参数。"""
    period = overrides.pop("period", 2.0 * LAM)
    geom_period = 2.0 * LAM if period is None else period
    kw = dict(
        f0=C0 / LAM,
        c0=C0,
        theta_i=20.0,
        period=period,
        J=3,
        h=np.array([0.30, 0.20, 0.25]) * LAM,
        w=np.array([0.18, 0.12, 0.16]) * geom_period,
        d=np.array([0.02, 0.06, 0.04]) * geom_period,
        c_complex=np.array([C0, C0 * (1.02 + 0.04j), C0 * 0.99], dtype=complex),
    )
    kw.update(overrides)
    return MetagratingParams(**kw)


def _negation_map(src: cmt.ScatteringMatrix, dst: cmt.ScatteringMatrix) -> list[int]:
    """dst 中 k_x 为 src 各端口 k_x 相反数的端口下标。"""
    dst_kx = np.array([port.kx for port in dst.ports])
    return [int(np.argmin(np.abs(dst_kx + port.kx))) for port in src.ports]


# ===========================================================================
# 回归与 2×2 兼容性
# ===========================================================================


def test_ep_nominal_regression() -> None:
    S = compute_s_matrix(cmt.make_ep_params())
    assert S.shape == (2, 2)
    np.testing.assert_allclose(S, EP_REF, rtol=1e-11, atol=1e-13)


def test_dp_nominal_regression() -> None:
    S = compute_s_matrix(cmt.make_dp_params())
    assert S.shape == (2, 2)
    np.testing.assert_allclose(S, DP_REF, rtol=1e-11, atol=1e-13)


def test_default_layout_matches_legacy_2x2() -> None:
    """θ_i=45°、D=λ/√2：S = [[r0_L, r+1_R], [r-1_L, r0_R]]。"""
    p = cmt.make_ep_params()
    sm = solve_s_matrix(p)
    assert sm.n_ports == 2
    assert sm.port_orders == (0, -1)
    np.testing.assert_allclose(
        [port.angle_deg for port in sm.ports], [45.0, -45.0], atol=1e-12
    )
    np.testing.assert_array_equal(sm.S, compute_s_matrix(p))

    # 端口角度经 arcsin/degrees 往返，与直接按 θ_i 求解差 ~1e-16，按机器精度比对。
    r_left = solve_reflection_cmt(p, "left")
    r_right = solve_reflection_cmt(p, "right")
    np.testing.assert_allclose(sm.S[0, 0], r_left[p.n_orders], atol=1e-14)  # r_0^L
    np.testing.assert_allclose(sm.S[1, 0], r_left[p.n_orders - 1], atol=1e-14)
    np.testing.assert_allclose(sm.S[0, 1], r_right[p.n_orders + 1], atol=1e-14)
    np.testing.assert_allclose(sm.S[1, 1], r_right[p.n_orders], atol=1e-14)


# ===========================================================================
# 互易性
# ===========================================================================


def test_reciprocity_2port_machine_precision() -> None:
    """任意非对称几何：S11≡S22 到机器精度，S12 与 S21 无约束。"""
    p = _asym(theta_i=45.0, period=LAM / np.sqrt(2.0))
    S = compute_s_matrix(p)
    assert abs(S[0, 0] - S[1, 1]) < 1e-13
    assert abs(S[0, 1] - S[1, 0]) > 1e-2


def test_reciprocity_general_dimension() -> None:
    """多端口推广：S_{+θ}[p,p] = S_{−θ}[n(p),n(p)]（端口 k_x 取负）。"""
    p_plus = _asym(theta_i=20.0, period=2.0 * LAM, J=3)
    p_minus = dataclasses.replace(p_plus, theta_i=-20.0)
    plus = solve_s_matrix(p_plus)
    minus = solve_s_matrix(p_minus)

    assert plus.n_ports == minus.n_ports == 4
    n_map = _negation_map(plus, minus)
    kx_minus = np.array([port.kx for port in minus.ports])
    np.testing.assert_allclose(
        kx_minus[n_map], -np.array([port.kx for port in plus.ports]), atol=1e-10
    )
    for p in range(plus.n_ports):
        assert abs(plus.S[p, p] - minus.S[n_map[p], n_map[p]]) < 1e-12
    # 非平凡性：确有可观的非对角散射
    assert np.max(np.abs(plus.S)) > 0.1


def test_reciprocity_arbitrary_J5() -> None:
    """任意 J 下自通道互易性同样成立。"""
    D = 2.0 * LAM
    p = MetagratingParams(
        f0=C0 / LAM,
        c0=C0,
        theta_i=20.0,
        period=D,
        J=5,
        h=np.linspace(0.20, 0.40, 5) * LAM,
        w=np.array([0.06, 0.07, 0.05, 0.08, 0.06]) * D,
        d=np.array([0.01, 0.04, 0.03, 0.05, 0.02]) * D,
        c_complex=C0 * (1.0 + 0.01j * np.arange(5)),
    )
    plus = solve_s_matrix(p)
    minus = solve_s_matrix(dataclasses.replace(p, theta_i=-20.0))
    assert plus.n_ports == 4
    n_map = _negation_map(plus, minus)
    for i in range(plus.n_ports):
        assert abs(plus.S[i, i] - minus.S[n_map[i], n_map[i]]) < 1e-12


# ===========================================================================
# 端口维度随 (f0, θ_i, D) 变化
# ===========================================================================


@pytest.mark.parametrize(
    ("theta_i", "period_ratio", "expected_ports"),
    [(45.0, 1.0 / np.sqrt(2.0), 2), (20.0, 2.0, 4), (30.0, 1.5, 3)],
)
def test_channel_count_varies(
    theta_i: float, period_ratio: float, expected_ports: int
) -> None:
    p = _asym(theta_i=theta_i, period=period_ratio * LAM)
    assert propagating_orders(p).size == expected_ports
    sm = solve_s_matrix(p)
    assert sm.S.shape == (expected_ports, expected_ports)
    assert len(set(sm.port_orders)) == expected_ports


def test_period_is_free_input() -> None:
    """同一 θ_i 下 D 是自由参数：端口数与所在相图区域随 D 变化。"""
    dense = _asym(theta_i=20.0, period=2.0 * LAM)
    sparse = _asym(theta_i=20.0, period=LAM / np.sqrt(2.0))
    assert dense.period == pytest.approx(2.0 * LAM)
    assert sparse.period == pytest.approx(LAM / np.sqrt(2.0))
    assert solve_s_matrix(dense).n_ports == 4
    assert solve_s_matrix(sparse).n_ports == 1


def test_period_default_from_grating_equation() -> None:
    p = MetagratingParams(theta_i=45.0)
    assert p.period == pytest.approx(LAM / np.sqrt(2.0), rel=1e-12)


def test_frequency_sweep_changes_dimension() -> None:
    p = cmt.make_ep_params()
    freqs = [p.f0, 2.0 * p.f0, 3.0 * p.f0]
    sweep = sweep_s_matrix(p, freqs)
    assert [sm.n_ports for sm in sweep] == [2, 3, 4]
    for sm, f in zip(sweep, freqs):
        np.testing.assert_array_equal(
            sm.S, compute_s_matrix(dataclasses.replace(p, f0=f))
        )


def test_replace_preserves_period() -> None:
    """频率扫描时几何 D 固定，不随 f0 由光栅方程重算。"""
    p = _asym(theta_i=20.0, period=2.0 * LAM)
    q = dataclasses.replace(p, f0=2.0 * p.f0)
    assert q.period == p.period
    assert q.n_orders == p.n_orders


# ===========================================================================
# 参数语义与端口标注
# ===========================================================================


def test_s_matrix_element_semantics() -> None:
    p = _asym()
    sm = solve_s_matrix(p)
    for q, port_q in enumerate(sm.ports):
        r = solve_reflection_cmt(p, port_q.angle_deg)
        for out, port_out in enumerate(sm.ports):
            assert sm.S[out, q] == r[(port_out.order - port_q.order) + p.n_orders]
    label = sm.element_label(0, 1)
    assert "出射" in label and "入射" in label and "衍射阶次" in label


def test_groove_positions_derivation_and_explicit_x() -> None:
    p = _asym()
    x = p.groove_positions
    expected = np.array(
        [p.d[0], p.d[0] + p.w[0] + p.d[1], p.d[0] + p.w[0] + p.d[1] + p.w[1] + p.d[2]]
    )
    np.testing.assert_allclose(x, expected)
    assert p.x is None

    q = _asym(d=None, x=x)
    assert q.d is None
    np.testing.assert_array_equal(q.groove_positions, x)
    np.testing.assert_array_equal(compute_s_matrix(q), compute_s_matrix(p))
    # replace（频率扫描走此路径）在 x-only 参数上同样成立
    assert np.all(np.isfinite(compute_s_matrix(dataclasses.replace(q, f0=2 * q.f0))))


def test_incidence_forms() -> None:
    p = cmt.make_ep_params()
    np.testing.assert_array_equal(
        solve_reflection_cmt(p, "left"), solve_reflection_cmt(p, 45.0)
    )
    np.testing.assert_array_equal(
        solve_reflection_cmt(p, "right"), solve_reflection_cmt(p, -45.0)
    )
    with pytest.raises(MetagratingParameterError, match="incidence"):
        solve_reflection_cmt(p, "up")
    with pytest.raises(MetagratingParameterError, match="入射角"):
        solve_reflection_cmt(p, 90.0)


def test_auto_n_orders_covers_all_relative_orders() -> None:
    p = dataclasses.replace(_asym(), n_orders=0)
    assert p.n_orders >= 4  # 端口跨度 3（−2…1），再留一阶倏逝缓冲
    sm = solve_s_matrix(p)
    assert sm.S.shape == (4, 4)
    assert np.all(np.isfinite(sm.S))


def test_real_c_complex_supported() -> None:
    """全实 c_complex 数组走复声速分支，不产生 NaN（β 的复数开方）。"""
    p_real = _asym()
    p_real.c_complex = np.array([C0, 343.2, 342.9])
    S = compute_s_matrix(p_real)
    assert np.all(np.isfinite(S))

    p_cplx = _asym()
    p_cplx.c_complex = np.array([C0, 343.2, 342.9], dtype=complex)
    np.testing.assert_array_equal(S, compute_s_matrix(p_cplx))


# ===========================================================================
# __post_init__ 与参数校验
# ===========================================================================


def test_defaults_only_when_all_missing() -> None:
    p = MetagratingParams()
    assert p.J == 3
    assert p.h.shape == p.w.shape == p.d.shape == p.c_complex.shape == (3,)
    assert p.x is None
    assert p.groove_positions.shape == (3,)
    assert p.c_complex[1] == pytest.approx(C0 * (1.0081 + 0.0745j))
    assert p.period == pytest.approx(LAM / np.sqrt(2.0), rel=1e-12)
    assert np.all(np.isfinite(compute_s_matrix(p)))


def test_partial_arrays_raise() -> None:
    with pytest.raises(MetagratingParameterError, match="缺省"):
        MetagratingParams(h=np.array([0.03, 0.02, 0.025]))
    with pytest.raises(MetagratingParameterError, match="缺省"):
        MetagratingParams(
            h=np.array([0.03, 0.02, 0.025]),
            w=np.array([0.01, 0.01, 0.01]),
            c_complex=np.full(3, C0, dtype=complex),
        )


def test_all_missing_with_non_default_J_raises() -> None:
    with pytest.raises(MetagratingParameterError, match="J=5"):
        MetagratingParams(J=5)


_INVALID_CASES = [
    pytest.param(
        lambda: _asym(w=np.array([0.18, 0.0, 0.16]) * 2.0 * LAM), id="zero-width"
    ),
    pytest.param(
        lambda: _asym(h=np.array([0.30, -0.20, 0.25]) * LAM), id="negative-depth"
    ),
    pytest.param(
        lambda: _asym(d=np.array([0.02, -0.06, 0.04]) * 2.0 * LAM), id="negative-gap"
    ),
    pytest.param(
        lambda: _asym(c_complex=np.array([C0, 0.0, C0], dtype=complex)),
        id="zero-sound-speed",
    ),
    pytest.param(
        lambda: _asym(w=np.array([0.5, 0.4, 0.3]) * 2.0 * LAM), id="total-width"
    ),
    pytest.param(
        lambda: _asym(d=None, x=np.array([0.0, 0.01, 0.3]) * 2.0 * LAM),
        id="overlapping-grooves",
    ),
    pytest.param(
        lambda: _asym(d=None, x=np.array([0.0, 0.5, 0.9]) * 2.0 * LAM),
        id="outside-period",
    ),
    pytest.param(
        lambda: _asym(x=np.array([0.0, 0.06, 0.13]) * 2.0 * LAM), id="d-x-mismatch"
    ),
    pytest.param(lambda: _asym(theta_i=90.0), id="grazing-incidence"),
    pytest.param(lambda: _asym(theta_i=0.0, period=None), id="zero-angle-no-period"),
    pytest.param(lambda: _asym(f0=0.0), id="zero-frequency"),
    pytest.param(lambda: _asym(c0=-1.0), id="negative-c0"),
    pytest.param(lambda: _asym(period=-1.0), id="negative-period"),
    pytest.param(lambda: _asym(J=2), id="J-shape-mismatch"),
    pytest.param(lambda: _asym(n_orders=-1), id="negative-n-orders"),
    pytest.param(lambda: _asym(k_modes=0), id="zero-k-modes"),
    pytest.param(
        lambda: _asym(h=np.array([0.30, np.nan, 0.25]) * LAM), id="non-finite-depth"
    ),
    pytest.param(
        lambda: _asym(c_complex=np.array([C0, np.inf, C0], dtype=complex)),
        id="non-finite-sound-speed",
    ),
]


@pytest.mark.parametrize("factory", _INVALID_CASES)
def test_invalid_parameters_raise(factory) -> None:
    with pytest.raises(MetagratingParameterError):
        factory()


def test_error_is_value_error() -> None:
    assert issubclass(MetagratingParameterError, ValueError)


def test_sweep_rejects_invalid_frequencies() -> None:
    p = cmt.make_ep_params()
    for bad in (0.0, -1.0, np.nan, np.inf):
        with pytest.raises(MetagratingParameterError, match="频率"):
            sweep_s_matrix(p, [p.f0, bad])
    with pytest.raises(MetagratingParameterError, match="一维"):
        sweep_s_matrix(p, np.ones((2, 2)))


# ===========================================================================
# 快核（G2）：对拍基准、批量接口、缓存与性能硬门槛
# ===========================================================================

# 以下为 G2 改造前（G1 稠密装配路径）输出的参考 S 矩阵，rtol=1e-10 对拍。
ASYM4_REF = np.array(
    [
        [
            -0.2531162892424722 + 0.4164814805655352j,
            0.5796863416118992 - 0.6113715370346681j,
            -0.5314647469117403 + 0.1673969022237203j,
            0.4384306880503372 - 0.09261667055479798j,
        ],
        [
            0.09393836932235403 + 0.5504333506775076j,
            0.1273392383320899 + 0.13831659330498294j,
            0.19372302730589344 - 0.3837900480337932j,
            -0.3311833345423187 + 0.18369465343297303j,
        ],
        [
            0.14426150112088562 + 0.18945606289789896j,
            0.17142554500444274 + 0.5370819290967879j,
            0.06268432494781966 + 0.06544500690617289j,
            0.3713522095522452 - 0.45220958019354773j,
        ],
        [
            0.15879953397860888 + 0.08370548250278471j,
            0.27587007169007 + 0.29783410246096326j,
            0.17296953543379315 + 0.8428543858243012j,
            -0.06401555263937529 + 0.4179296589148636j,
        ],
    ]
)
J5_REF = np.array(
    [
        [
            -0.07022420555804455 + 0.36237443337382746j,
            -0.5847198620549173 - 0.8921421995905984j,
            0.2158436604806233 - 0.3624486952727654j,
            0.00517707868519285 - 0.1700739289770644j,
        ],
        [
            -0.223073950909631 + 0.25439203603838706j,
            0.3089893413764558 + 0.1662668147202682j,
            -0.4407099661970206 - 0.5835969230904012j,
            0.15228122484798964 - 0.2548032959429008j,
        ],
        [
            -0.2907751537561736 + 0.04228118811849949j,
            -0.12252686712069051 + 0.25673886767952186j,
            0.26274368305350304 + 0.18437832659566503j,
            -0.4122652486191858 - 0.5304337154372756j,
        ],
        [
            -0.5026395914992787 + 0.30337379833455125j,
            -0.2833681982054776 + 0.15672991858345478j,
            -0.26554143326110224 + 0.48573459204885744j,
            0.19218565940881646 + 0.38650722050829034j,
        ],
    ]
)
U_MINUS1_REF = np.array(
    [
        [
            -0.27987481001470327 + 0.43357584016924805j,
            0.5910930028542841 - 0.6502185666656403j,
            -0.49147226601604854 + 0.17448497443263788j,
            0.44092927773068774 - 0.04579791562993847j,
        ],
        [
            0.11096764661085101 + 0.5792048879454386j,
            0.0792432625674846 + 0.12179840555606246j,
            0.2197981859590984 - 0.419035109917131j,
            -0.3068442846933702 + 0.19082894938296288j,
        ],
        [
            0.1654433160398708 + 0.15734485937302398j,
            0.18121270401281747 + 0.5967289602763594j,
            0.00639073156920046 + 0.05284686249242428j,
            0.37674257669473915 - 0.4838339959480112j,
        ],
        [
            0.12261547847423153 + 0.09050627003069481j,
            0.3082740770821144 + 0.251687130680978j,
            0.20658975206404023 + 0.8891812983162392j,
            -0.0940449927357645 + 0.44086649180837856j,
        ],
    ]
)


def test_fast_kernel_matches_g1_multiport_reference() -> None:
    """G2 快核与 G1 稠密装配逐点对拍（多端口、任意 J、U→−1 共振）。"""
    np.testing.assert_allclose(
        compute_s_matrix(_asym()), ASYM4_REF, rtol=1e-10, atol=1e-12
    )

    D = 2.0 * LAM
    p5 = MetagratingParams(
        f0=C0 / LAM,
        c0=C0,
        theta_i=20.0,
        period=D,
        J=5,
        h=np.linspace(0.20, 0.40, 5) * LAM,
        w=np.array([0.06, 0.07, 0.05, 0.08, 0.06]) * D,
        d=np.array([0.01, 0.04, 0.03, 0.05, 0.02]) * D,
        c_complex=C0 * (1.0 + 0.01j * np.arange(5)),
    )
    np.testing.assert_allclose(compute_s_matrix(p5), J5_REF, rtol=1e-10, atol=1e-12)

    p_lossless = _asym(c_complex=np.ones(3, dtype=complex) * C0)
    np.testing.assert_allclose(
        compute_s_matrix(p_lossless), U_MINUS1_REF, rtol=1e-10, atol=1e-12
    )


def test_kernel_ports_and_single_solve_match_functional_api() -> None:
    p = _asym()
    kernel = CMTForwardKernel(p)
    sm = solve_s_matrix(p)
    assert kernel.ports == sm.ports
    np.testing.assert_array_equal(kernel.s_matrix(), sm.S)
    np.testing.assert_array_equal(kernel.solve().S, sm.S)


def test_sweep_medium_matches_individual() -> None:
    p = cmt.make_ep_params()
    c_batch = np.array(
        [
            p.c_complex,
            [C0, C0 * (1.0 + 0.03j), C0],
            [C0 * 0.98, C0 * (1.02 - 0.02j), C0 * 1.01],
        ],
        dtype=complex,
    )
    batch = sweep_medium(p, c_batch)
    assert batch.shape == (3, 2, 2)
    for S, c in zip(batch, c_batch):
        np.testing.assert_array_equal(
            S, compute_s_matrix(dataclasses.replace(p, c_complex=c))
        )


def test_solve_s_matrix_batch_matches_individual() -> None:
    params = [cmt.make_ep_params(), _asym(), _asym(theta_i=30.0, period=1.5 * LAM)]
    out = solve_s_matrix_batch(params)
    assert len(out) == len(params)
    for sm, p in zip(out, params):
        np.testing.assert_array_equal(sm.S, compute_s_matrix(p))
        assert sm.ports == solve_s_matrix(p).ports


def test_medium_batch_validation() -> None:
    p = cmt.make_ep_params()
    with pytest.raises(MetagratingParameterError, match="批量形状"):
        sweep_medium(p, np.ones((2, 2), dtype=complex))
    with pytest.raises(MetagratingParameterError, match="有限非零"):
        sweep_medium(p, np.zeros(3, dtype=complex))
    with pytest.raises(MetagratingParameterError, match="有限非零"):
        sweep_medium(p, np.array([[C0, np.inf, C0]], dtype=complex))


def test_cache_key_tracks_geometry_mutation() -> None:
    """缓存键包含 x/w：构造后修改几何必须触发重新装配。"""
    p = cmt.make_ep_params()
    clear_forward_cache()
    S_before = compute_s_matrix(p)
    p.w = p.w * 1.02
    S_after = compute_s_matrix(p)
    assert not np.allclose(S_before, S_after)
    fresh = dataclasses.replace(cmt.make_ep_params(), w=cmt.make_ep_params().w * 1.02)
    np.testing.assert_array_equal(S_after, compute_s_matrix(fresh))


def test_performance_hard_gate() -> None:
    """验收硬门槛：J=3、n_orders=10、k_modes=31 单次 S 求值 < 5 ms。"""
    p = cmt.make_ep_params()
    n_cold, n_warm = 5, 50
    t = time.perf_counter()
    for _ in range(n_cold):
        clear_forward_cache()
        compute_s_matrix(p)
    cold = (time.perf_counter() - t) / n_cold
    t = time.perf_counter()
    for _ in range(n_warm):
        compute_s_matrix(p)
    warm = (time.perf_counter() - t) / n_warm
    assert cold < 5e-3, f"cold {cold * 1e3:.3f} ms ≥ 5 ms"
    assert warm < 5e-3, f"warm {warm * 1e3:.3f} ms ≥ 5 ms"
