"""CMT 逆向设计高维可行性演示（14 维，EP 构型回收，多进程 + 断点续算）。

用法（项目根）：
    uv run --no-sync python scripts/research/gain_ep/theory/inverse_design_demo.py

流程：
1. 以 G1 名义 EP 构型为真值，构造 14 维设计空间
   （h×3 + w×3 + x×2 + cr×3 + ci×3）；
2. 在真值 ±1% 扰动种群上跑 DE，种群级 8 进程并行，每代写检查点；
3. 从检查点续算若干代，验证定点续算；
4. 结果（参数/目标值/S 矩阵/收敛史）导出到
   ``data/research/1_gain_ep/theory/inverse_design/g2a_t3_ep/``；
5. 报告单核评估吞吐（验收 ≥2000 次/s/核）。
"""

from __future__ import annotations

import os
import time

import numpy as np

from pysci.research.gain_ep.theory.cmt_reflection_s_matrix import (
    compute_s_matrix,
    make_ep_params,
    solve_s_matrix,
)
from pysci.research.gain_ep.theory.inverse_design import (
    CombinedObjective,
    DesignSpace,
    Evaluator,
    MatrixMatch,
    default_output_dir,
    inverse_design,
    jordan_structure,
)

POPSIZE = 48
GENERATIONS = 250
RESUME_GENERATIONS = 20
SEED = 0


def main() -> None:
    truth = make_ep_params()
    space = DesignSpace(J=3, period=truth.period, theta_i=truth.theta_i)
    target = compute_s_matrix(truth)
    objective = CombinedObjective((MatrixMatch(target),))
    ports = solve_s_matrix(truth).ports
    workers = min(8, os.cpu_count() or 1)

    print("=" * 72)
    print("CMT 逆向设计高维演示：EP 构型回收")
    print("=" * 72)
    print(
        f"维度 dim={space.dim}（h×3+w×3+x×2+cr×3+ci×3），"
        f"种群 {POPSIZE}，代数 {GENERATIONS}，workers={workers}"
    )
    print(f"端口数 P={len(ports)}，S 形状 {target.shape}")

    # 单核评估吞吐（验收口径）
    bench = space.perturbed_population(truth, scale=0.01, popsize=300, seed=1)
    evaluator = Evaluator(space, objective, ports)
    per_eval = float("inf")
    for _ in range(3):
        t = time.perf_counter()
        for genome in bench:
            evaluator.score(genome)
        per_eval = min(per_eval, (time.perf_counter() - t) / len(bench))
    print(
        f"单核评估（3 次取优）：{per_eval * 1e3:.4f} ms/次 → "
        f"{1.0 / per_eval:.0f} 次/s/核（验收门槛 2000）"
    )

    out_dir = default_output_dir("g2a_t3_ep")
    ckpt = out_dir / "checkpoint.npz"
    pop = space.perturbed_population(truth, scale=0.01, popsize=POPSIZE, seed=SEED)

    t = time.perf_counter()
    res = inverse_design(
        space,
        objective,
        popsize=POPSIZE,
        max_generations=GENERATIONS,
        seed=SEED,
        workers=workers,
        initial_population=pop,
        patience=100,
        tol=1e-12,
        checkpoint_path=ckpt,
        output_dir=out_dir,
    )
    wall = time.perf_counter() - t
    err = float(np.max(np.abs(res.S - target)))
    eval_rate = res.evaluations / max(res.evaluation_time, 1e-12)
    print(
        f"\n[主优化] 代数={res.generations} 评估={res.evaluations} "
        f"墙钟={wall:.2f} s；评估阶段 {res.evaluation_time:.2f} s → "
        f"{eval_rate:.0f} 次/s（{workers} 进程；含每代进程调度开销，"
        f"单核评估吞吐见上）"
    )
    print(f"目标值={res.objective_value:.6e}  max|S−S_target|={err:.3e}")
    structure = jordan_structure(res.S, tol=1e-6)
    print(
        f"谱诊断：min_gap={structure.min_gap:.4e}，"
        f"代数/几何重数={structure.algebraic}/{structure.geometric}，"
        f"defective={structure.defective}"
    )

    # 断点续算演示
    t = time.perf_counter()
    resumed = inverse_design(
        space,
        objective,
        popsize=POPSIZE,
        max_generations=res.generations + RESUME_GENERATIONS,
        seed=SEED,
        workers=workers,
        resume=True,
        patience=60,
        tol=1e-12,
        checkpoint_path=ckpt,
        output_dir=out_dir,
    )
    wall2 = time.perf_counter() - t
    print(
        f"[续算 +{RESUME_GENERATIONS} 代] 代数={resumed.generations} "
        f"目标值={resumed.objective_value:.6e} "
        f"max|ΔS|={np.max(np.abs(resumed.S - target)):.3e} 墙钟={wall2:.2f} s"
    )

    p = resumed.params
    print("\n回收参数（归一化：h/λ, w/D, x/D, c/c0）")
    print(f"  h/λ  = {np.round(p.h / (p.c0 / p.f0), 4)}")
    print(f"  w/D  = {np.round(p.w / p.period, 4)}")
    print(f"  x/D  = {np.round(p.groove_positions / p.period, 4)}")
    print(f"  c/c0 = {np.round(p.c_complex / p.c0, 4)}")
    print("真值：")
    print(f"  h/λ  = {np.round(truth.h / (truth.c0 / truth.f0), 4)}")
    print(f"  w/D  = {np.round(truth.w / truth.period, 4)}")
    print(f"  x/D  = {np.round(truth.groove_positions / truth.period, 4)}")
    print(f"  c/c0 = {np.round(truth.c_complex / truth.c0, 4)}")
    print(f"\n结果已导出：{out_dir}")
    print("  result.npz / summary.json / checkpoint.npz")


if __name__ == "__main__":
    main()
