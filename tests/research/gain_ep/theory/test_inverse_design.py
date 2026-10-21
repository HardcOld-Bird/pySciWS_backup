"""CMT 逆向设计管线测试：目标项、约束修复、回收、并行、续算、吞吐。

回收用例从 ±1% 扰动初值反演 G1 名义 EP/DP 构型，要求 S 最大逐元误差 < 1e-2。
"""

import json
import time

import numpy as np
import pytest

from pysci.research.gain_ep.theory.cmt_reflection_s_matrix import (
    MetagratingParameterError,
    compute_s_matrix,
    make_dp_params,
    make_ep_params,
    solve_s_matrix,
)
from pysci.research.gain_ep.theory.inverse_design import (
    CombinedObjective,
    DesignSpace,
    Evaluator,
    JordanBlock,
    MatrixMatch,
    ZeroElements,
    inverse_design,
    jordan_structure,
)

LAM = 343.0 / 3430.0
RECOVERY_TOL = 1e-2


def _ep_setup():
    truth = make_ep_params()
    space = DesignSpace(J=3, period=truth.period, theta_i=truth.theta_i)
    target = compute_s_matrix(truth)
    ports = solve_s_matrix(truth).ports
    return truth, space, target, ports


# ===========================================================================
# 设计空间与约束修复
# ===========================================================================


def test_design_space_encode_decode_roundtrip() -> None:
    truth, space, target, _ = _ep_setup()
    assert space.dim == 14  # J=3: h3+w3+x2+cr3+ci3
    genome = space.encode(truth)
    assert genome.shape == (space.dim,)
    assert np.all((genome >= 0.0) & (genome <= 1.0))
    params = space.decode(genome)
    np.testing.assert_allclose(compute_s_matrix(params), target, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(params.h, truth.h, atol=1e-14)
    np.testing.assert_allclose(params.w, truth.w, atol=1e-14)
    np.testing.assert_allclose(
        params.groove_positions, truth.groove_positions, atol=1e-14
    )
    np.testing.assert_allclose(params.c_complex, truth.c_complex, atol=1e-12)


@pytest.mark.parametrize("genome_kind", ["random", "zeros", "ones"])
def test_decode_repairs_geometric_constraints(genome_kind: str) -> None:
    _, space, _, _ = _ep_setup()
    rng = np.random.default_rng(0)
    genomes = {
        "random": rng.random((200, space.dim)),
        "zeros": np.zeros((1, space.dim)),
        "ones": np.ones((1, space.dim)),
    }[genome_kind]
    tol = 1e-9 * space.period
    for genome in genomes:
        params = space.decode(genome)  # 走完整 MetagratingParams 校验
        w = np.asarray(params.w)
        x = np.asarray(params.groove_positions)
        assert np.all(w >= space.w_bounds[0] - tol)
        assert np.all(w <= space.w_bounds[1] + tol)
        assert np.sum(w) + space.J * space.min_feature <= space.period + tol
        assert abs(x[0] - space.x0) <= tol
        assert np.all(np.diff(x) >= w[:-1] + space.min_feature - tol)
        assert x[-1] + w[-1] + space.min_feature <= space.period + tol
        assert np.all(params.h >= space.h_bounds[0] - tol)
        assert np.all(params.h <= space.h_bounds[1] + tol)


def test_decode_rejects_wrong_shape() -> None:
    _, space, _, _ = _ep_setup()
    with pytest.raises(MetagratingParameterError, match="基因型形状"):
        space.decode(np.zeros(space.dim - 1))


def test_infeasible_design_space_raises() -> None:
    D = LAM / np.sqrt(2.0)
    with pytest.raises(MetagratingParameterError, match="不可行"):
        DesignSpace(J=3, period=D, w_bounds=(0.4 * D, 0.45 * D))


def test_encode_out_of_bounds_raises() -> None:
    truth, space, _, _ = _ep_setup()
    bad = make_ep_params()
    bad.h = truth.h * 10.0
    with pytest.raises(MetagratingParameterError, match="超出设计空间"):
        space.encode(bad)


# ===========================================================================
# 目标项
# ===========================================================================


def test_matrix_match_zero_at_target_and_positive_else() -> None:
    _, _, target, ports = _ep_setup()
    obj = MatrixMatch(target)
    assert obj.evaluate(target, ports) == pytest.approx(0.0, abs=1e-30)
    assert obj.evaluate(target + 0.01, ports) > 1e-4
    with pytest.raises(MetagratingParameterError, match="形状"):
        MatrixMatch(target).evaluate(np.zeros((1, 1), complex), ports)


def test_matrix_match_amplitude_and_phase_weights() -> None:
    _, _, target, ports = _ep_setup()
    amp_only = MatrixMatch(target, complex_weight=0.0, amplitude_weight=1.0)
    phase_only = MatrixMatch(target, complex_weight=0.0, phase_weight=1.0)
    perturbed = target.copy()
    perturbed[0, 1] *= 1.1 * np.exp(0.2j)  # 幅值与相位同时改变
    assert amp_only.evaluate(perturbed, ports) > 0
    assert phase_only.evaluate(perturbed, ports) > 0
    # 纯相位旋转：幅值项为 0，相位项 > 0
    rotated = target * np.exp(1j * 0.2)
    assert amp_only.evaluate(rotated, ports) == pytest.approx(0.0, abs=1e-20)
    assert phase_only.evaluate(rotated, ports) > 0


def test_zero_elements_term() -> None:
    _, _, target, ports = _ep_setup()
    term = ZeroElements(((1, 0),))
    S = target.copy()
    S[1, 0] = 0.0
    assert term.evaluate(S, ports) == 0.0
    assert term.evaluate(target, ports) > 0
    with pytest.raises(MetagratingParameterError, match="下标"):
        ZeroElements(((5, 5),)).evaluate(target, ports)


def test_jordan_block_term_and_structure() -> None:
    jordan = np.array([[1.0, 1.0], [0.0, 1.0]], dtype=complex)
    identity = np.eye(2, dtype=complex)
    generic = np.array([[1.0, 0.3], [0.0, -0.5]], dtype=complex)
    ports: tuple = ()
    assert JordanBlock(mode="gap").evaluate(jordan, ports) == pytest.approx(0.0)
    assert JordanBlock(mode="trdet").evaluate(jordan, ports) == pytest.approx(0.0)
    assert JordanBlock(mode="gap").evaluate(generic, ports) > 0.0
    assert JordanBlock(mode="gap").evaluate(identity, ports) == pytest.approx(0.0)

    s_j = jordan_structure(jordan)
    assert s_j.defective and s_j.algebraic == (2,) and s_j.geometric == (1,)
    s_i = jordan_structure(identity)
    assert not s_i.defective and s_i.algebraic == (2,) and s_i.geometric == (2,)
    s_g = jordan_structure(generic)
    assert not s_g.defective and s_g.min_gap > 0

    _, _, target, _ = _ep_setup()
    assert JordanBlock(mode="gap").evaluate(target, ports) < 1e-2


def test_combined_objective_sums_weighted_terms() -> None:
    _, _, target, ports = _ep_setup()
    a = MatrixMatch(target)
    b = ZeroElements(((1, 0),), weight=2.0)
    combined = CombinedObjective((a, b))
    expected = a.evaluate(target, ports) + b.evaluate(target, ports)
    assert combined.evaluate(target, ports) == pytest.approx(expected)
    assert combined(target, ports) == pytest.approx(expected)


# ===========================================================================
# 回收测试（EP/DP）
# ===========================================================================


def test_recover_ep_from_perturbed_init() -> None:
    truth, space, target, _ = _ep_setup()
    pop = space.perturbed_population(truth, scale=0.01, popsize=20, seed=0)
    res = inverse_design(
        space,
        CombinedObjective((MatrixMatch(target),)),
        popsize=20,
        max_generations=120,
        seed=0,
        initial_population=pop,
        patience=30,
        tol=1e-10,
    )
    assert res.objective_value < 1e-3
    assert np.max(np.abs(res.S - target)) < RECOVERY_TOL
    assert res.evaluations >= 20


def test_recover_dp_from_perturbed_init() -> None:
    truth = make_dp_params()
    space = DesignSpace(J=3, period=truth.period, theta_i=truth.theta_i)
    target = compute_s_matrix(truth)
    pop = space.perturbed_population(truth, scale=0.01, popsize=20, seed=1)
    res = inverse_design(
        space,
        CombinedObjective((MatrixMatch(target),)),
        popsize=20,
        max_generations=120,
        seed=1,
        initial_population=pop,
        patience=30,
        tol=1e-10,
    )
    assert res.objective_value < 1e-3
    assert np.max(np.abs(res.S - target)) < RECOVERY_TOL


# ===========================================================================
# 优化器外壳：并行、续算、导出、吞吐
# ===========================================================================


def test_parallel_matches_sequential() -> None:
    truth, space, target, _ = _ep_setup()
    pop = space.perturbed_population(truth, scale=0.01, popsize=8, seed=7)
    kwargs = dict(
        popsize=8,
        max_generations=2,
        seed=7,
        initial_population=pop,
        tol=1e-12,
    )
    seq = inverse_design(space, CombinedObjective((MatrixMatch(target),)), **kwargs)
    par = inverse_design(
        space,
        CombinedObjective((MatrixMatch(target),)),
        workers=2,
        **kwargs,
    )
    assert par.objective_value == seq.objective_value
    np.testing.assert_array_equal(par.genome, seq.genome)
    np.testing.assert_array_equal(par.history, seq.history)


def test_checkpoint_resume_matches_direct(tmp_path) -> None:
    truth, space, target, _ = _ep_setup()
    pop = space.perturbed_population(truth, scale=0.01, popsize=8, seed=5)
    ckpt = tmp_path / "ckpt.npz"
    obj = CombinedObjective((MatrixMatch(target),))
    first = inverse_design(
        space,
        obj,
        popsize=8,
        max_generations=2,
        seed=5,
        initial_population=pop,
        checkpoint_path=ckpt,
        tol=1e-12,
    )
    assert ckpt.exists() and first.generations == 2

    resumed = inverse_design(
        space,
        obj,
        popsize=8,
        max_generations=4,
        seed=5,
        resume=True,
        checkpoint_path=ckpt,
        tol=1e-12,
    )
    direct = inverse_design(
        space,
        obj,
        popsize=8,
        max_generations=4,
        seed=5,
        initial_population=pop,
        tol=1e-12,
    )
    assert resumed.generations == 4
    assert resumed.objective_value == direct.objective_value
    np.testing.assert_array_equal(resumed.genome, direct.genome)
    np.testing.assert_array_equal(resumed.history, direct.history)


def test_resume_requires_checkpoint(tmp_path) -> None:
    _, space, target, _ = _ep_setup()
    with pytest.raises(FileNotFoundError):
        inverse_design(
            space,
            CombinedObjective((MatrixMatch(target),)),
            popsize=4,
            max_generations=1,
            resume=True,
            checkpoint_path=tmp_path / "missing.npz",
        )


def test_result_export(tmp_path) -> None:
    truth, space, target, _ = _ep_setup()
    pop = space.perturbed_population(truth, scale=0.01, popsize=8, seed=2)
    out = tmp_path / "run"
    res = inverse_design(
        space,
        CombinedObjective((MatrixMatch(target),)),
        popsize=8,
        max_generations=2,
        seed=2,
        initial_population=pop,
        output_dir=out,
        tol=1e-12,
    )
    assert res.output_dir == out
    assert (out / "result.npz").exists() and (out / "summary.json").exists()
    with np.load(out / "result.npz") as data:
        for key in ("genome", "h", "w", "x", "c_complex", "S", "history"):
            assert key in data
        np.testing.assert_array_equal(data["S"], res.S)
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["objective_value"] == pytest.approx(res.objective_value)
    assert summary["J"] == 3


def test_evaluator_throughput_at_least_2000_per_s() -> None:
    """验收：单核评估 ≥2000 次/s（t2 快核缓存路径，0.5 ms/次）。"""
    truth, space, target, ports = _ep_setup()
    ev = Evaluator(space, CombinedObjective((MatrixMatch(target),)), ports)
    genomes = space.perturbed_population(truth, scale=0.01, popsize=200, seed=1)
    best = np.inf
    for _ in range(3):
        t = time.perf_counter()
        for genome in genomes:
            ev.score(genome)
        best = min(best, (time.perf_counter() - t) / len(genomes))
    assert best < 0.5e-3, f"单核评估 {best * 1e3:.3f} ms ≥ 0.5 ms"
