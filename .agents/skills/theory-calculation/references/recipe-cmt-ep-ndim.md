# Recipe: EP n×n 谱判别式（判别式过零 ∧ defective，排除 diabolic）

> ④理论计算域物理配方（新 API 形态：NumPy/SciPy 直调 + 本域薄胶水）。
> 本篇是 [recipe-ep-bic.md](recipe-ep-bic.md) / [ep-topology-recipes.md](ep-topology-recipes.md) §1
> **2×2 判别式判据的 n×n 推广**：对象从 2×2 哈密顿量推广到任意 n×n 复方阵（散射矩阵
> S、哈密顿量 H、传递矩阵皆可），判据从"判别式过零"升级为
> "**谱判别式过零 ∧ 矩阵亏损（defective）**"，以剔除可对角化的 diabolic 简并点。
> 语义提取自旧 pySciWS 研究线 `pysci.research.gain_ep.theory`（`ep_manifold.py`
> 判别式口径、`inverse_design.py` 的 `jordan_structure` 诊断；⭐研究线不入仓），
> 公式与代码按新 API 重写，未复制旧实现。

## 1. 数学判据

### 1.1 谱判别式（配对归一版）

n×n 特征多项式的判别式为所有本征值两两差的平方积：

```
disc(S) = ∏_{i<j} (λᵢ − λⱼ)²        （2×2 时 = β² − 4αγ，α=1, β=−tr S, γ=det S）
```

实际判据用**配对归一版**（无量纲、尺度不变、O(1)）：

```
F̃(λᵢ, λⱼ) = (λᵢ − λⱼ)² / (1 + |λᵢ + λⱼ|² + |λᵢ λⱼ|²)
```

推导要点（2×2 恒等式，n×n 时按"最可疑的一对"取用）：
`(α,β,γ) ∝ (1, −tr S, det S)` 三系数同乘任何复因子 F̂ = (β²−4αγ)/(|α|²+|β|²+|γ|²)
不变，展开即得上式（λ₁λ₂ = det、λ₁+λ₂ = tr）。归一化分母使 F̃ 对参数扰动一阶稳定，
且对 S 的整体幅值缩放不变。n×n 判别取

```
F̃_min(S) = min_{i<j} |F̃(λᵢ, λⱼ)|        （最接近合并的一对）
```

### 1.2 排假：defective 复核（剔除 diabolic）

**F̃≈0 只是候选**——本征值合并 ≠ EP。厄米/正规矩阵的简并（diabolic point，简并但
可对角化）同样使 F̃=0。真 EP 要求矩阵**亏损**：某本征值簇代数重数 > 几何重数
（本征向量也合并，Jordan 块非对角）。故

```
EP ⟺ F̃_min(S) < τ_F   ∧   jordan_structure(S).defective
```

`jordan_structure` 判定语义（本征值簇的重数诊断）：
1. 归一化尺度 `scale = max(‖S‖_F/√n, 1e-12)`（RMS 幅度）；
2. 本征值聚类：|λ_a − λ_b| ≤ tol·scale 归入同簇；
3. 每簇：**代数重数** = 簇大小；**几何重数** = n − rank(S − λ̄I)（λ̄ 为簇均值，
   矩阵秩用 SVD，数值容差同取 tol·scale）；
4. `defective = ∃簇 代数重数 > 几何重数`。

### 1.3 对拍（独立来源）

大矩阵**带边行列式铅笔**可作独立对拍（旧 `bordered_coefficients` 方法论）：
`q(λ) = det([[M̃, −R̃], [L̃, −λI]])`，用 `slogdet` 缩放防溢出，取 λ = 0, ±λ₀ 三点
插值得 (α̂, β̂, γ̂)（同乘复因子不影响 F̂），再算 F̂ 与 1.1 的谱版对拍；
良态点典型相对差 ~1e-14（不一致 ⇒ 病态/近极点，见 §3）。

## 2. 关键代码模式（新 API，NumPy 直调）

```python
import numpy as np

def spectral_discriminant_pair(lam_a: complex, lam_b: complex) -> complex:
    """配对归一谱判别式 F̃ = (λa−λb)² / (1+|λa+λb|²+|λa·λb|²)。"""
    return (lam_a - lam_b) ** 2 / (1.0 + abs(lam_a + lam_b) ** 2 + abs(lam_a * lam_b) ** 2)

def jordan_structure(S: np.ndarray, tol: float = 1e-6) -> dict:
    """n×n 约当结构诊断：本征值簇的代数/几何重数（defective 判定）。"""
    S = np.asarray(S, dtype=complex)
    if S.ndim != 2 or S.shape[0] != S.shape[1]:
        raise ValueError(f"S 须为方阵，实际 {S.shape}")
    n = S.shape[0]
    scale = max(float(np.linalg.norm(S) / np.sqrt(n)), 1e-12)  # RMS 幅度
    eig = np.linalg.eigvals(S)
    clusters: list[list[complex]] = []          # 单链接聚类（容差 tol·scale）
    for lam in eig:
        for cluster in clusters:
            if abs(lam - cluster[0]) <= tol * scale:
                cluster.append(lam)
                break
        else:
            clusters.append([lam])
    algebraic, geometric = [], []
    for cluster in clusters:
        lam_c = complex(np.mean(cluster))
        algebraic.append(len(cluster))
        rank = np.linalg.matrix_rank(S - lam_c * np.eye(n), tol=tol * scale)
        geometric.append(n - int(rank))
    defective = any(a > g for a, g in zip(algebraic, geometric, strict=True))
    return {"eigenvalues": eig, "algebraic": tuple(algebraic),
            "geometric": tuple(geometric), "defective": defective}

def ep_condition_ndim(S: np.ndarray, *, tol: float = 1e-6, tau_f: float | None = None):
    """n×n EP 判据：F̃ 近零 ∧ defective（diabolic 点被 defective 复核剔除）。"""
    diag = jordan_structure(S, tol=tol)
    eig = diag["eigenvalues"]
    pairs = [spectral_discriminant_pair(eig[i], eig[j])
             for i in range(len(eig)) for j in range(i + 1, len(eig))]
    f_min = min(abs(f) for f in pairs) if pairs else float("nan")
    tau = (tol * 10.0) ** 2 if tau_f is None else tau_f   # 本征值误差 ~tol ⇒ F̃ 误差 ~tol²
    return {"f_min": f_min, "is_ep": bool(f_min < tau and diag["defective"]), **diag}
```

要点：2×2 时 `f_min = |F̃(λ₁,λ₂)|` 与 recipe-ep-bic 的判别式过零完全同判；
n×n 时只检查"最接近的一对"已覆盖单个 2×2 合并型 EP（高阶 EP/多对同时合并用
disc 全量式 `∏(λᵢ−λⱼ)²` 复核）。

## 3. 浮点鲁棒性注记（必读）

1. **容差必须相对化**：`tol` 是相对容差，实际阈值 = `tol·scale`，`scale` 取矩阵
   RMS 幅度（`‖S‖_F/√n`）。**勿写死绝对 1e-12**——矩阵幅值换一个单位判据即失效。
2. **jordan_structure 输入前归一化**：最稳做法是先把 `S / scale` 归一到 O(1) 再诊断
   （固定 `tol` 语义与幅值无关）；若依赖内部 scale 计算，则**聚类容差与矩阵秩容差必须
   同取 `tol·scale`**，两处容差不一致会出现"簇分了但秩没并"的假 defective。
3. **defective 判定是秩诊断**：`S − λ̄I` 的秩用 SVD（`np.linalg.matrix_rank`），
   其阈值受非正规矩阵的本征向量条件数 κ(V) 放大；κ(V) 大时结果可疑，用
   `mpmath.workdps(50)` 高精度复核（见 divergence-safety.md 的复核纪律）。
4. **聚类边界双档复核**：单链接聚类有链式效应（λ_a~λ_b、λ_b~λ_c 但 λ_a≁λ_c）；
   用 `tol` 与 `tol/10` 各跑一遍，结论不一致 ⇒ 处在判据边界，报"未定"勿硬判。
5. **τ_F 阈值量级**：`np.linalg.eigvals` 误差 ~ eps·‖S‖·κ(V)，F̃ ≈ 间隙²，故
   |F̃| 的可信下界 ~ (tol)² 量级（代码默认 `(10·tol)²`）；**勿只看 F̃ 绝对值**——
   必须与 defective 复核成双条件，否则 diabolic 点全数误判为 EP。
6. **近极点/病态**：对拍（§1.3）相对差远超 1e-14 ⇒ 矩阵病态或近极点，先缩放/换
   精度再判；判据口径（tol、τ_F、聚类方式）一旦冻结写进脚本 docstring 勿中途改。

## 4. 扩展钩子（点到为止，研究线自管）

- **EP 强度**：Puiseux 系数（λ ∼ λ_EP + c·(p−p_EP)^{1/2} 的 c）；
- **射影间隙**：ℂP¹ 弦距 |λ₁−λ₂|/√((1+|λ₁|²)(1+|λ₂|²))（∞ 处合并也归零）；
- **响应指标**：log max‖S‖ 替代 EP 处发散的 Petermann K 因子；
- 9D 参数流形/延拓/环绕数等研究形态见 ep-topology-recipes.md，⭐研究线不入仓。

## 来源

语义提取自旧 pySciWS `src/pysci/research/gain_ep/theory/`（`ep_manifold.py` 模块
docstring 判别式口径 F̂=(λ₁−λ₂)²/(1+|λ₁+λ₂|²+|λ₁λ₂|²) 与铅笔对拍方法论、
`inverse_design.py` 的 `jordan_structure` 簇重数诊断语义；2026-11-17 解剖
`others/community_research/2026-11-17/legacy-02-cmt-gain-ep.md`）。公式提取后按
NumPy 新 API 重写为本篇代码模式，未复制旧代码。
