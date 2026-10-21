"""validate_s_matrix COMSOL 验证测试（无 COMSOL Server 时整组自动 skip）。

对照口径为幅度（COMSOL 相位参考与 CMT 差固定相位，见模块 docstring）：
- 无损锚点沿用 t4 的 4 位有效数字口径；
- J=4 闭合 t4「J>3 未对照」缺口；
- 含损复声速与 3 端口维度/语义用例。
阈值按 t4/t5 实测的 FEM-CMT 离散化偏差设定并留余量。
"""

import numpy as np
import pytest

from pysci.research.gain_ep.theory.cmt_comsol_validation import (
    compare_s_matrix,
    validate_s_matrix,
    validate_s_matrix_batch,
)
from pysci.research.gain_ep.theory.cmt_reflection_s_matrix import (
    MetagratingParameterError,
    MetagratingParams,
    compute_s_matrix,
    solve_s_matrix,
)

LAM = 343.0 / 3430.0
C0 = 343.0
D0 = LAM / np.sqrt(2.0)


@pytest.fixture(scope="module")
def comsol_client():
    mph = pytest.importorskip("mph")
    try:
        client = mph.start(cores=2)
    except Exception as exc:  # noqa: BLE001 - 无 license/Server 即 skip
        pytest.skip(f"COMSOL 不可用：{type(exc).__name__}: {exc}")
    yield client
    client.disconnect()


def _params(J=3, theta=45.0, D=D0, c=None) -> MetagratingParams:
    h = (np.array([0.569, 0.195, 0.232, 0.30])[:J]) * LAM
    w = (np.array([0.227, 0.115, 0.153, 0.08])[:J]) * D
    x = (np.array([0.0, 0.503, 0.688, 0.86])[:J]) * D
    if c is None:
        c = np.full(J, C0, dtype=complex)
    return MetagratingParams(
        f0=C0 / LAM,
        c0=C0,
        theta_i=theta,
        period=D,
        J=J,
        h=h,
        w=w,
        x=x,
        c_complex=np.asarray(c, dtype=complex),
    )


def _assert_ports(params: MetagratingParams, S: np.ndarray) -> None:
    ports = solve_s_matrix(params).ports
    assert S.shape == (len(ports), len(ports))


def test_lossless_anchor_matches_cmt(comsol_client) -> None:
    """无损锚点：主系数 4 位，小量约 2–4%（t4 口径）。"""
    p = _params()
    S_cmt = compute_s_matrix(p)
    S_fem = validate_s_matrix(p, client=comsol_client)
    _assert_ports(p, S_fem)
    cmp = compare_s_matrix(S_cmt, S_fem)
    assert cmp["max_abs_diff"] < 2e-3
    a, b = np.abs(S_cmt), np.abs(S_fem)
    main = a > 0.9
    assert np.max(np.abs(a[main] - b[main]) / a[main]) < 1e-3
    small = a < 0.1
    assert np.max(np.abs(a[small] - b[small])) < 2e-3


def test_J4_matches_cmt(comsol_client) -> None:
    """J=4（前 3 槽同锚点 + 追加第 4 槽）：闭合 t4「J>3 未对照」缺口。"""
    p = _params(J=4)
    S_cmt = compute_s_matrix(p)
    S_fem = validate_s_matrix(p, client=comsol_client)
    _assert_ports(p, S_fem)
    cmp = compare_s_matrix(S_cmt, S_fem)
    assert cmp["max_abs_diff"] < 2e-2
    a, b = np.abs(S_cmt), np.abs(S_fem)
    assert np.max(np.abs(a - b) / np.maximum(a, 1e-2)) < 8e-2


def test_lossy_complex_sound_speed(comsol_client) -> None:
    """含损（cr=1.005, ci=0.05）非平凡复声速；差异如实申报。"""
    p = _params(c=[C0, C0 * (1.005 + 0.05j), C0])
    S_cmt = compute_s_matrix(p)
    S_fem = validate_s_matrix(p, client=comsol_client)
    cmp = compare_s_matrix(S_cmt, S_fem)
    assert cmp["max_abs_diff"] < 3e-2
    assert cmp["rms_diff"] < 1e-2


def test_three_port_dimension_and_semantics(comsol_client) -> None:
    """D=1.5λ/θ=30°：3 端口 [0,−1,−2]，S 为 3×3 且幅度整体吻合。"""
    p = _params(theta=30.0, D=1.5 * LAM)
    ports = solve_s_matrix(p).ports
    assert tuple(port.order for port in ports) == (0, -1, -2)
    S_cmt = compute_s_matrix(p)
    S_fem = validate_s_matrix(p, client=comsol_client)
    assert S_fem.shape == (3, 3)
    cmp = compare_s_matrix(S_cmt, S_fem)
    assert cmp["max_abs_diff"] < 2e-2
    assert cmp["rms_diff"] < 1e-2


def test_batch_single_session(comsol_client) -> None:
    """批量接口：一次会话返回等长列表，逐项与 CMT 幅度吻合。"""
    params = [_params()]
    out = validate_s_matrix_batch(params, client=comsol_client)
    assert len(out) == 1
    for p, S_fem in zip(params, out):
        S_cmt = compute_s_matrix(p)
        _assert_ports(p, S_fem)
        assert compare_s_matrix(S_cmt, S_fem)["max_abs_diff"] < 2e-3


def test_routing_rejects_J_outside_template_set() -> None:
    # 7 个前导有效槽（J=7 超出模板集）——无需 COMSOL 即报错。
    p7 = MetagratingParams(
        f0=C0 / LAM,
        c0=C0,
        theta_i=45.0,
        period=D0,
        J=7,
        h=np.full(7, 0.20 * LAM),
        w=np.full(7, 0.05 * D0),
        x=np.arange(7) * 0.1 * D0,
        c_complex=np.full(7, C0, dtype=complex),
    )
    with pytest.raises(MetagratingParameterError, match="J∈"):
        validate_s_matrix(p7)


def test_routing_rejects_non_343_background() -> None:
    p = _params()
    p.c0 = 344.0  # 构造后修改，路由校验应拒绝
    with pytest.raises(MetagratingParameterError, match="343"):
        validate_s_matrix(p)


def test_routing_rejects_interleaved_sliver() -> None:
    p = _params()
    p.h = np.array([0.03, 1e-7, 0.02])
    with pytest.raises(MetagratingParameterError, match="前导槽"):
        validate_s_matrix(p)
