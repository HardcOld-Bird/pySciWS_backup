# EP 判据 / 分支追踪 / 环绕数 / Morse 分类（代码模式配方）

> 用户裁决（2026-11-17）：本组能力**只做 recipe 文档、不做代码**——它们是物理脚本模式，
> 通用封装不减维护面。以下代码块按新 API（SymPy/NumPy 直调）书写，直接抄进研究线脚本。

## 1. EP 判据（例外点检测）

2×2 非厄米哈密顿量 `H` 的 EP 判据 = **判别式过零 + 本征向量合并**：

```python
import numpy as np
import sympy as sp

def ep_condition_symbolic(H):
    """2x2 EP 符号条件：返回判别式 Δ（Δ=0 即 EP/简并边界）。"""
    lam = sp.symbols("lambda")
    a2, a1, a0 = sp.Poly(H.charpoly(lam).as_expr(), lam).all_coeffs()
    return sp.expand(a1**2 - 4 * a2 * a0)

def detect_ep_numeric(H_of_t, t_values):
    """数值 EP 扫描：判别式近零处再看本征向量夹角（合并=EP，交叉=简并）。"""
    hits = []
    for i in range(len(t_values) - 1):
        h0, h1 = H_of_t(t_values[i]), H_of_t(t_values[i + 1])
        # 判别式（对 2x2：(tr)^2 - 4 det）过零检测
        d0 = np.trace(h0) ** 2 - 4 * np.linalg.det(h0)
        d1 = np.trace(h1) ** 2 - 4 * np.linalg.det(h1)
        if np.real(d0 * d1) < 0 or abs(d1) < 1e-12:
            _, v0 = np.linalg.eig(h0)
            _, v1 = np.linalg.eig(h1)
            cos_min = np.min(np.abs(v0.conj().T @ v1) /
                             (np.linalg.norm(v0, axis=0)[:, None] *
                              np.linalg.norm(v1, axis=0)[None, :]))
            hits.append((t_values[i + 1], float(cos_min)))  # cos→1：向量合并=EP
    return hits
```

要点：判别式过零只是候选；**本征向量夹角趋于 0（cos→1）** 才是 EP（两个本征值+两个
本征向量同时合并），否则只是简并交叉。数值阈值按问题尺度调，勿写死 1e-12。

## 2. 分支追踪（防交叉伪影）

对参数序列上的本征值按**相位连续性**贪心配对（勿对每点独立排序）：

```python
def track_branches(eig_seq):
    """eig_seq: (N, M) 复本征值（N 个参数点、每点 M 支）→ 按连续性重排。

    从第 0 点起，每点把本征值分配给上一点各支的最近邻（|Δ| 贪心）。
    """
    order = [eig_seq[0]]
    for row in eig_seq[1:]:
        prev = order[-1]
        cost = np.abs(row[None, :] - prev[:, None])  # (M, M)
        idx = np.argmin(cost, axis=1)
        order.append(row[idx])
        # 注：两支交换/接近时贪心可能误配，严格情形用 scipy.optimize.linear_sum_assignment
    return np.array(order)
```

适用：复平面轨迹、色散支、能带交叉邻域。实二支退化情形可直接 `np.minimum/np.maximum`
（见 recipe-tbic-band.md）。通用排序器（如按实部排）**不保留**——排序语义属于物理脚本。

## 3. 环绕数（拓扑不变量）

闭回路上复值函数相位的回路积分（分支点计数）：

```python
def winding_number(func, center, radius, n=2048):
    """func: 复平面→复数；沿 |z-center|=radius 逆时针取样，返回环绕数（整数）。

    W = (1/2πi) ∮ dlog f = Δarg(f)/2π（离散相位差分求和，免积分器）。
    """
    theta = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    z = center + radius * np.exp(1j * theta)
    w = func(z)
    if np.any(np.abs(w) < 1e-300):
        raise ValueError("回路上存在零点/极点，回路退化——缩小半径或改道")
    dlog = np.diff(np.log(w))                      # 复对数主值差分
    dlog += -2j * np.pi * np.round(dlog.imag / (2 * np.pi))  # 解缠绕
    return int(np.round(dlog.sum().imag / (2 * np.pi)))
```

要点：**围道检查先行**——回路必须避开零/极点（否则报错或结果无意义）；
采样密度 `n` 需覆盖最快相位变化（相邻相位差 < π）。配套的"收敛半径/围道检查"模式见
[divergence-safety.md](divergence-safety.md)。

## 4. Morse 分类（临界点指标）

数值 Hessian + 惯性指标（负特征值个数 = Morse 指标）：

```python
def numerical_hessian(f, x0, h=1e-5):
    """中心差分 Hessian（x0: (d,)，f: R^d→R）。"""
    x0 = np.asarray(x0, dtype=float)
    d = x0.size
    H = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            ei, ej = np.zeros(d), np.zeros(d)
            ei[i] = ej[j] = h
            H[i, j] = (f(x0 + ei + ej) - f(x0 + ei - ej)
                       - f(x0 - ei + ej) + f(x0 - ei - ej)) / (4 * h * h)
    return H

def morse_index(f, x0):
    """Morse 指标 = Hessian 负特征值个数；=0 局部极小，=d 局部极大。"""
    return int(np.sum(np.linalg.eigvalsh(numerical_hessian(f, x0)) < 0))
```

要点：步长 `h` 按函数尺度取（通常 `cbrt(eps)·|x0|` 量级）；Hessian 近奇异（特征值≈0）
说明退化临界点，Morse 分类不适用。2D 标量场的零点集/等值线探索用
`matplotlib.contour`（3D 等值面渲染归③/⑥域）。
