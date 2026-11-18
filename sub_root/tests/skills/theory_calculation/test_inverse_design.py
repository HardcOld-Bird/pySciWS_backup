"""inverse_design 薄胶水机械断言：目标/约束协议、checkpoint 落盘/恢复往返、
scipy DE 固定种子收敛、非法输入拦截（固定样例恒跑，不依赖研究数据）。"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pysci.paths import research_asset_dir
from pysci.skills.theory_calculation import inverse_design as idesign

TEST_NAME = "zz_theorycalc_inv_selftest"
TARGET = np.array([0.5, -0.3])


def sphere(x: np.ndarray) -> float:
    """固定测试目标：‖x − (0.5, −0.3)‖²（无约束光滑凸函数）。"""
    return float(np.sum((x - TARGET) ** 2))


@pytest.fixture()
def ckpt_dir():
    """data/ 内的测试目录；结束后整目录清理（不污染 data/）。"""
    base = research_asset_dir(TEST_NAME)
    (base / "results").mkdir(parents=True, exist_ok=True)
    yield base
    shutil.rmtree(base, ignore_errors=True)


def _npz(base) -> Path:
    return base / "results" / "de.npz"


# ---------------------------------------------------------------------------
# 目标协议校验
# ---------------------------------------------------------------------------

def test_objective_protocol_vector_in_scalar_out(ckpt_dir):
    seen: list[tuple] = []

    def objective(x):
        seen.append(np.asarray(x).shape)
        return sphere(x)

    res = idesign.run_inverse_design(
        objective, [(-1.0, 1.0), (-1.0, 1.0)], seed=42, maxiter=3
    )
    assert res.x.shape == (2,)
    assert np.isscalar(float(res.fun))
    assert seen and all(s == (2,) for s in seen)  # 恒收一维参数向量


def test_constraint_protocol_leq_zero_semantics(ckpt_dir):
    # 约束 x0 − 0.5 ≤ 0（即 x0 ≤ 0.5），目标推 x0 → 1：最优贴约束边界
    res = idesign.run_inverse_design(
        lambda x: float((x[0] - 1.0) ** 2 + x[1] ** 2),
        [(-2.0, 2.0), (-2.0, 2.0)],
        constraints=[lambda x: float(x[0] - 0.5)],
        seed=0,
        maxiter=120,
        tol=1e-10,
    )
    assert res.x[0] <= 0.5 + 1e-6  # 可行性（≤0 判据）
    assert res.x[0] == pytest.approx(0.5, abs=0.05)  # 贴边界
    assert res.fun <= 0.26


# ---------------------------------------------------------------------------
# checkpoint 落盘 / 恢复往返
# ---------------------------------------------------------------------------

def test_checkpoint_write_load_roundtrip(ckpt_dir):
    path = _npz(ckpt_dir)
    res = SimpleNamespace(
        x=np.array([0.1, -0.2]), fun=0.05,
        population=np.array([[0.1, -0.2], [0.3, 0.4]]),
        population_energies=np.array([0.05, 0.25]),
        nit=7, nfev=100, convergence=0.5,
    )
    cb = idesign.CheckpointCallback(path, generation_offset=10)
    assert cb(res) is False  # 返回 False 表示继续优化

    assert path.is_file()
    json_side = path.with_suffix(".json")
    assert json_side.is_file()
    summary = json.loads(json_side.read_text(encoding="utf-8"))
    assert summary["generation"] == 17  # 累计代数 = nit + offset
    assert summary["best_fun"] == 0.05
    assert summary["population_shape"] == [2, 2]

    ckpt = idesign.load_checkpoint(path)
    assert ckpt.generation == 17
    assert ckpt.best_fun == 0.05
    assert ckpt.nfev == 100
    assert ckpt.convergence == 0.5
    assert np.array_equal(ckpt.best_x, [0.1, -0.2])
    assert np.array_equal(ckpt.population, res.population)
    assert np.array_equal(ckpt.population_energies, res.population_energies)


def test_checkpoint_every_generation_and_resume(ckpt_dir):
    path = _npz(ckpt_dir)
    res = idesign.run_inverse_design(
        sphere, [(-1.0, 1.0), (-1.0, 1.0)], seed=0, maxiter=4,
        checkpoint_path=path,
    )
    assert res.nit == 4
    ckpt = idesign.load_checkpoint(path)
    assert ckpt.generation == 4  # 每代落盘，末代即最终代

    # 恢复往返：maxiter=0 时 DE 只重估 init 种群，返回值=checkpoint 状态
    res2 = idesign.run_inverse_design(
        sphere, [(-1.0, 1.0), (-1.0, 1.0)], seed=99, maxiter=0,
        resume_from=path,
    )
    assert res2.fun == pytest.approx(ckpt.best_fun, rel=0, abs=1e-12)
    assert np.allclose(res2.population, ckpt.population, rtol=0, atol=1e-12)

    # 续跑：代数累计（4 + 2 = 6）
    path2 = ckpt_dir / "results" / "de2.npz"
    idesign.run_inverse_design(
        sphere, [(-1.0, 1.0), (-1.0, 1.0)], seed=0, maxiter=2,
        checkpoint_path=path2, resume_from=path,
    )
    assert idesign.load_checkpoint(path2).generation == 6


# ---------------------------------------------------------------------------
# scipy DE 对简单函数收敛（种子固定可断言）
# ---------------------------------------------------------------------------

def test_de_converges_sphere_fixed_seed(ckpt_dir):
    kwargs = {"bounds": [(-1.0, 1.0), (-1.0, 1.0)], "seed": 42,
              "maxiter": 80, "tol": 1e-10, "polish": False}
    res = idesign.run_inverse_design(sphere, **kwargs)
    assert res.fun < 1e-8
    assert np.allclose(res.x, TARGET, atol=1e-4)
    # 同种子逐位可复现
    again = idesign.run_inverse_design(sphere, **kwargs)
    assert again.fun == res.fun
    assert np.array_equal(again.x, res.x)


# ---------------------------------------------------------------------------
# 非法输入拦截
# ---------------------------------------------------------------------------

def test_illegal_bounds_rejected(ckpt_dir):
    with pytest.raises(ValueError):
        idesign.run_inverse_design(sphere, [(1.0, -1.0), (-1.0, 1.0)])  # lo ≥ hi
    with pytest.raises(ValueError):
        idesign.run_inverse_design(sphere, [(0.0, 0.0), (-1.0, 1.0)])  # lo == hi
    with pytest.raises(ValueError):
        idesign.run_inverse_design(sphere, [(-1.0, np.inf), (-1.0, 1.0)])  # 非有限
    with pytest.raises(ValueError):
        idesign.run_inverse_design(sphere, [(-1.0, 1.0, 0.0)])  # 形状错误


def test_illegal_run_parameters_rejected(ckpt_dir):
    with pytest.raises(TypeError):
        idesign.run_inverse_design("not callable", [(-1.0, 1.0)])
    with pytest.raises(ValueError):
        idesign.run_inverse_design(sphere, [(-1.0, 1.0), (-1.0, 1.0)], popsize=0)
    with pytest.raises(ValueError):
        idesign.run_inverse_design(sphere, [(-1.0, 1.0), (-1.0, 1.0)], maxiter=-1)
    with pytest.raises(TypeError):
        idesign.run_inverse_design(
            sphere, [(-1.0, 1.0)], constraints=["not callable"]
        )


def test_nan_objective_is_intercepted(ckpt_dir):
    with pytest.raises(ValueError, match="NaN"):
        idesign.run_inverse_design(
            lambda x: float("nan"), [(-1.0, 1.0), (-1.0, 1.0)], seed=0, maxiter=1
        )


def test_checkpoint_path_guards(ckpt_dir, tmp_path):
    with pytest.raises(ValueError):  # 非 .npz 后缀
        idesign.CheckpointCallback(ckpt_dir / "results" / "de.txt")
    with pytest.raises(ValueError):  # data/ 外写入
        idesign.save_checkpoint(
            tmp_path / "stray.npz",
            idesign.Checkpoint(
                generation=0, best_x=np.zeros(2), best_fun=0.0,
                population=np.zeros((1, 2)), population_energies=np.zeros(1),
                nfev=0, convergence=0.0,
            ),
        )


def test_resume_guards(ckpt_dir):
    with pytest.raises(FileNotFoundError):
        idesign.run_inverse_design(
            sphere, [(-1.0, 1.0)], resume_from=ckpt_dir / "results" / "missing.npz"
        )
    with pytest.raises(FileNotFoundError):
        idesign.load_checkpoint(ckpt_dir / "results" / "missing.npz")

    # 缺字段的 npz
    bad = ckpt_dir / "results" / "bad.npz"
    np.savez(bad, generation=np.array(0))
    with pytest.raises(ValueError, match="字段缺失"):
        idesign.load_checkpoint(bad)

    # 续跑种群维度与 bounds 不符
    path = _npz(ckpt_dir)
    idesign.run_inverse_design(
        sphere, [(-1.0, 1.0), (-1.0, 1.0)], seed=0, maxiter=1, checkpoint_path=path
    )
    with pytest.raises(ValueError, match="维度"):
        idesign.run_inverse_design(
            sphere, [(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)],
            seed=0, maxiter=1, resume_from=path,
        )
