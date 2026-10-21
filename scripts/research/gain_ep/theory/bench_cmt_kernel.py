"""CMT 前向快核基准（可复跑）。

用法（项目根）：
    uv run --no-sync python scripts/research/gain_ep/theory/bench_cmt_kernel.py

报告单核实测：
- J=3、n_orders=10、k_modes=31 的 2 端口 S：冷启动（每次清缓存）与缓存命中；
- 4 端口、J=5 的缓存命中；
- 固定几何扫介质 ``sweep_medium`` 与多参数集 ``solve_s_matrix_batch`` 的每核吞吐。
验收硬门槛：单次 S 求值 < 5 ms（冷启动口径更严）。
"""

from __future__ import annotations

import time

import numpy as np

from pysci.research.gain_ep.theory import cmt_reflection_s_matrix as cmt

C0 = 343.0
LAM = 343.0 / 3430.0


def _bench(fn, n: int, warmup: int = 1) -> float:
    """返回单次调用平均耗时 (s)。"""
    for _ in range(warmup):
        fn()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t) / n


def _report(label: str, dt: float) -> None:
    print(f"{label:<34s} {dt * 1e3:9.4f} ms   {1.0 / dt:10.0f} 次/s")


def _asym4() -> cmt.MetagratingParams:
    D = 2.0 * LAM
    return cmt.MetagratingParams(
        f0=3430.0,
        c0=C0,
        theta_i=20.0,
        period=D,
        J=3,
        h=np.array([0.30, 0.20, 0.25]) * LAM,
        w=np.array([0.18, 0.12, 0.16]) * D,
        d=np.array([0.02, 0.06, 0.04]) * D,
        c_complex=np.array([C0, C0 * (1.02 + 0.04j), C0 * 0.99]),
    )


def _j5() -> cmt.MetagratingParams:
    D = 2.0 * LAM
    return cmt.MetagratingParams(
        f0=3430.0,
        c0=C0,
        theta_i=20.0,
        period=D,
        J=5,
        h=np.linspace(0.20, 0.40, 5) * LAM,
        w=np.array([0.06, 0.07, 0.05, 0.08, 0.06]) * D,
        d=np.array([0.01, 0.04, 0.03, 0.05, 0.02]) * D,
        c_complex=C0 * (1.0 + 0.01j * np.arange(5)),
    )


def main() -> None:
    print("==" * 34)
    print("CMT 前向快核基准（单核，Python 进程内计时）")
    print("==" * 34)

    p2 = cmt.make_ep_params()
    n_cold = 40
    t = time.perf_counter()
    for _ in range(n_cold):
        cmt.clear_forward_cache()
        cmt.compute_s_matrix(p2)
    cold = (time.perf_counter() - t) / n_cold
    _report("2 端口 EP（冷启动，清缓存）", cold)
    _report("2 端口 EP（缓存命中）", _bench(lambda: cmt.compute_s_matrix(p2), 2000))

    p4 = _asym4()
    _report("4 端口（缓存命中）", _bench(lambda: cmt.compute_s_matrix(p4), 2000))

    p5 = _j5()
    _report("4 端口 J=5（缓存命中）", _bench(lambda: cmt.compute_s_matrix(p5), 2000))

    n_med = 2000
    cs = np.tile(p2.c_complex, (n_med, 1))
    cs[:, 1] = C0 * (1.0 + 0.0745j + 0.03j * np.sin(np.linspace(0.0, 6.28, n_med)))
    t = time.perf_counter()
    batch = cmt.sweep_medium(p2, cs)
    _report("sweep_medium（2000 个介质）", (time.perf_counter() - t) / n_med)
    assert batch.shape == (n_med, 2, 2)

    params = [
        cmt.make_ep_params(cr=1.0 + 0.001 * i, ci=0.05 + 0.0005 * i) for i in range(200)
    ]
    t = time.perf_counter()
    out = cmt.solve_s_matrix_batch(params)
    _report(
        "solve_s_matrix_batch（200 个参数集）", (time.perf_counter() - t) / len(out)
    )

    print("-" * 68)
    gate = 5e-3
    status = "PASS" if cold < gate else "FAIL"
    print(
        f"硬门槛（单次 S < {gate * 1e3:.0f} ms）：{status}（cold={cold * 1e3:.4f} ms）"
    )


if __name__ == "__main__":
    main()
