"""CMT 反射型管槽超表面散射矩阵计算（周期 / 入射角 / 端口数通用化）。

物理模型（权威文档：``data/research/1_gain_ep/theory/参考-透反射管槽结构的S矩阵理论计算/
CMT计算与仿真参考.md`` 附录 B.2；B.1 为平面波模特例）：
    一维周期性超表面，每个周期 D 内由 J 个单向封闭矩形管槽（硬壁、硬底）构成。
    第 s 个槽的宽度 w_s、深度 h_s、起点 x_s、以及填充介质的等效复声速 c_s 相互独立；
    c_s 取复数即模拟损耗/增益（c_s = c0(n_s)，虚部对应衰减/放大）。
    周期 D、入射角 θ_i、频率 f0 是三个相互独立的自由输入（不再由光栅方程反推）。

    求解沿用文档 B.2.5–B.2.11 的模态匹配线性方程组
        [(-P2, P3); (-V2, V3)] · (A^-; A) = (P1; V1) · A0,
    其中 A^- 为 2M+1 阶 Floquet 反射系数，A 为各槽内 K 个波导模的幅值。
    数值约定与既有实现一致（e^{-jωt}；法向坐标记作 y，文档 B.2 记作 z）：
      * Floquet 横向波数 k_x^m = k sinθ_i + mG，G = 2π/D；
      * 槽内波导模 α_ks = kπ/w_s，β_ks = sqrt(k_s'^2 - α_ks^2)，k_s' = 2πf0/c_s；
      * 分支取 Im(k_y) ≤ 0、Im(β) ≤ 0（倏逝波沿法向/槽深衰减，往返因子
        U = exp(-2jβh) 满足 |U| ≤ 1）。

端口与 S 矩阵（文档附录 A 衍射模式分布图）：
    对给定 (f0, θ_i, D)，参考栅格上的第 m 阶在声锥内（|k_x^m| < k，等价于 WA 线
    k_x + m(2π/D) = ±n0k 之间、n0 = 1）时才是传播通道，其余为倏逝模。传播阶次集合
    M = {m : |k_x^m| < k} 即端口集合，端口按 k_x 降序编号；S 矩阵维度为 |M|×|M|，
    不恒为 2×2。矩阵元语义：
        S[p, q] = 以端口 q（入射角 θ_q = arcsin(k_x^q/k)）入射时，
                  端口 p 的复反射振幅；对应衍射阶次 m = p.order − q.order。
    因此 ±45°、D = λ/√2 的 2×2 特例与旧接口逐元一致：
        S = [[r_0^L, r_{+1}^R], [r_{-1}^L, r_0^R]]。

互易性（Lorentz 互易，含损耗/增益仍成立，不含复共轭）：
    自通道恒等式 r_0(θ) ≡ r_0(−θ)（任意非对称几何、机器精度），在端口基下写作
        S_{+θ}[p, p] = S_{−θ}[n(p), n(p)],
    其中 n(p) 为 k_x 取负的端口。参考栅格关于 0 对称时（如 θ_i = 45°、D = λ/√2），
    两端口集合重合、S_{−θ} = S_{+θ}，恒等式化为 S = J S J（J 为反序置换），
    2×2 特例即 S11 ≡ S22，而 S12、S21 不受此恒等式约束。
    非对角元涉及不同通道对，反射问题的端口基并非标准多端口波导 S 矩阵，故无转置对称。

坐标与符号：
    x 沿超表面、y 为法向；G = 2π/D 为倒格矢；n_orders 为 Floquet 截断（最低值，
    实际值见 ``MetagratingParams.n_orders``），k_modes 为槽内波导模截断数。

性能实现（G2 快核）：
    槽内振幅 A 通过 P3 的块对角结构做 Schur 消元（A = P3^{-1}(P1+P2 A^-)），
    主系统由 (N+J·K) 降到 N；U→−1 的共振模保留为小型补充块，与稠密解等价。
    与介质无关的装配量（P1/P2/V1/V2、V3 积分核）按 (f0, 入射角, 几何) 做 LRU
    缓存；介质变化只重算 β/U 与一次 N×N 求解。批量接口：
    :meth:`CMTForwardKernel.s_matrix_many` / :func:`sweep_medium`（固定几何扫介质）、
    :func:`solve_s_matrix_batch`（种群/多参数集）、:func:`sweep_s_matrix`（多频）。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from pysci.skills.theoretical_computation.tools import visualize
from pysci.skills.theoretical_computation.tools.session import ensure_session

__all__ = [
    "CMTForwardKernel",
    "MetagratingParameterError",
    "MetagratingParams",
    "Port",
    "ScatteringMatrix",
    "clear_forward_cache",
    "compute_s_matrix",
    "make_dp_params",
    "make_ep_params",
    "propagating_orders",
    "solve_reflection_cmt",
    "solve_s_matrix",
    "solve_s_matrix_batch",
    "sweep_medium",
    "sweep_s_matrix",
]

_ComplexArray = NDArray[np.complex128]
_FloatArray = NDArray[np.float64]
_IntArray = NDArray[np.int64]

# 声锥判据的相对余量：|k_x| < k(1-_CONE_TOL) 才算传播（排除掠射 k_z = 0 的边界模，
# 该模使 V2 对角元为零、方程组奇异）。
_CONE_TOL = 1e-12
# 几何参数校验的相对容差（以周期 D 为尺度）。
_GEOM_RTOL = 1e-9
# Schur 消元的奇异判据：|1+U| ≤ _SCHUR_TOL 的槽内模（U→−1 共振）不参与消元，
# 保留在补充块中，避免 1/(1+U) 数值爆炸。
_SCHUR_TOL = 1e-10


class MetagratingParameterError(ValueError):
    """超表面几何或物理参数不合法。"""


# ===========================================================================
# 积分核（对应 MATLAB 的 N1jifen / fenbujifen / M22jifen）
# ===========================================================================


def _fenbu_integral(t1: complex, s1: float, a1: float, x1: float, x2: float) -> complex:
    """∫_{x1}^{x2} e^{t1·x}·cos(s1·(x−a1)) dx 的解析结果（标量版）。

    保留旧私有签名，供 ``scripts/research/gain_ep/theory/增益半平面谱奇点探测.py``
    等既有调用方使用；求解器内部改用谐波闭式 :func:`_overlap_harmonic`。

    三种情况：
      1. t1=0, s1=0 → x2−x1；t1=0, s1≠0 → 标准正弦积分；
      2. t1²+s1²=0（t1≠0）→ 共振情况；
      3. 一般情况 → 标准解析式。
    """
    denom = t1**2 + s1**2

    if np.isclose(t1, 0) and np.isclose(s1, 0):
        return x2 - x1
    elif np.isclose(denom, 0) and not np.isclose(t1, 0):
        return (
            np.exp(t1 * a1) / 2 * (x2 - x1)
            + (np.exp(t1 * x2 + t1 * (x2 - a1)) - np.exp(t1 * x1 + t1 * (x1 - a1)))
            / 4
            / t1
        )
    else:
        val2 = np.exp(t1 * x2) * (
            t1 * np.cos(s1 * (x2 - a1)) + s1 * np.sin(s1 * (x2 - a1))
        )
        val1 = np.exp(t1 * x1) * (
            t1 * np.cos(s1 * (x1 - a1)) + s1 * np.sin(s1 * (x1 - a1))
        )
        return (val2 - val1) / denom


def _m22_integral(a1: float, a2: float, x1: float, x2: float) -> float:
    """∫_{x1}^{x2} cos(a1·(x−x1))·cos(a2·(x−x1)) dx（波导模正交性，标量版）。

    保留旧私有签名供既有脚本使用；求解器内部利用其严格对角性直接构造 P3。
    """
    width = x2 - x1
    if np.isclose(a1, 0) and np.isclose(a2, 0):
        return width
    elif np.isclose(a1, a2):
        return width / 2
    else:
        return 0.0


def _block_diag(blocks: list[NDArray]) -> NDArray:
    """构建块对角矩阵（保留旧私有签名，供既有脚本使用）。"""
    from scipy.linalg import block_diag

    return block_diag(*blocks)


def _overlap_harmonic(
    t: _ComplexArray,
    s: _FloatArray,
    sign: _FloatArray,
    w: _FloatArray,
    e_tw: _ComplexArray,
    e_tx: _ComplexArray,
) -> _ComplexArray:
    """逐槽波导模重叠积分的向量化闭式（所有槽一次算完）。

    对第 j 槽第 k 模，α = kπ/w 使 ``s·w = kπ``，故 cos(s w) = (−1)^k、sin(s w) = 0，
    一般式退化为 ``t·(e^{tw}(−1)^k − 1)/(t²+s²)``（无需三角函数）。t≈0 时
    k=0 → w、k>0 → 0；t²+s²≈0（t≠0 共振）用标准共振式。列按 j*K+k 排列。
    e^{t·w}、e^{t·x_s} 按槽预计算后重复到各模（指数只算 N·J 次而非 N·J·K 次）。

    Args:
        t: 指数因子（平面波横向波数的 ±j 倍），形状 (N,1)。
        s: 各 (槽,模) 的横向波数 α = kπ/w，形状 (J·K,)。
        sign: 各模的 (−1)^k，形状 (J·K,)。
        w: 各 (槽,模) 对应槽宽，形状 (J·K,)。
        e_tw: e^{t·w}，形状 (N, J·K)。
        e_tx: e^{t·x_s}，形状 (N, J·K)。

    Returns:
        形状 (N, J·K) 的复数组，列序 j*K+k。
    """
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        den = t * t + s * s
        general = t * (e_tw * sign - 1.0) / den
        t_zero = np.abs(t) <= 1e-8
        if np.any(t_zero):
            case_t0 = np.where(s == 0, w, 0.0)
            out = np.where(t_zero, case_t0, general)
        else:
            out = general
        res_mask = (np.abs(den) <= 1e-8) & ~t_zero
        if np.any(res_mask):
            t_safe = np.where(t_zero, 1.0, t)
            resonance = 0.5 * w + (e_tw * e_tw - 1.0) / (4.0 * t_safe)
            out = np.where(res_mask, resonance, out)
    return e_tx * out


# ===========================================================================
# 端口与散射矩阵
# ===========================================================================


@dataclass(frozen=True)
class Port:
    """一个传播通道（衍射端口）。

    Attributes:
        order: 相对参考入射 θ_i 的衍射阶次 m（栅格 k_x = k sinθ_i + mG）。
        kx: 该通道的横向波数 (rad/m)。
        angle_deg: 该通道对应的平面波入射/出射角 arcsin(kx/k)（度）。
        kz: 法向波数 sqrt(k² − kx²) (rad/m)，传播通道为正实数。
    """

    order: int
    kx: float
    angle_deg: float
    kz: float


@dataclass(frozen=True)
class ScatteringMatrix:
    """多端口反射散射矩阵及其通道语义。

    ``S[p, q]`` 为以端口 ``ports[q]`` 入射、在端口 ``ports[p]`` 出射的复反射振幅，
    对应衍射阶次 ``ports[p].order − ports[q].order``。端口按 k_x 降序编号，
    故 θ_i = 45°、D = λ/√2 时 S 就是旧版 2×2 布局
    ``[[r_0^L, r_{+1}^R], [r_{-1}^L, r_0^R]]``。

    Attributes:
        S: 形状 (P, P) 的复散射矩阵，P = 传播通道数。
        ports: 端口元数据，按 k_x 降序。
        params: 求解所用的参数（含参考 θ_i、f0、D）。
    """

    S: _ComplexArray
    ports: tuple[Port, ...]
    params: MetagratingParams

    @property
    def n_ports(self) -> int:
        """端口数（S 矩阵维度）。"""
        return len(self.ports)

    @property
    def port_orders(self) -> tuple[int, ...]:
        """各端口的参考衍射阶次（按 S 的行列顺序）。"""
        return tuple(port.order for port in self.ports)

    def element_label(self, p_out: int, p_in: int) -> str:
        """返回矩阵元的通道语义标注（出射通道 ← 入射通道、衍射阶次）。"""
        po = self.ports[p_out]
        pi = self.ports[p_in]
        return (
            f"S[出射 阶{po.order:+d} (θ={po.angle_deg:+.2f}°) ← "
            f"入射 阶{pi.order:+d} (θ={pi.angle_deg:+.2f}°); "
            f"衍射阶次 m={po.order - pi.order:+d}]"
        )

    def describe(self) -> str:
        """逐元打印 S 的通道语义与数值（便于日志核对）。"""
        rows = [
            f"{self.n_ports} 端口 S 矩阵 (f0={self.params.f0:g} Hz, "
            f"θ_i={self.params.theta_i:g}°, D={self.params.period:g} m)"
        ]
        for p in range(self.n_ports):
            for q in range(self.n_ports):
                rows.append(f"  {self.element_label(p, q)} = {self.S[p, q]:+.6g}")
        return "\n".join(rows)


# ===========================================================================
# 几何与物理参数
# ===========================================================================


@dataclass
class MetagratingParams:
    """反射型管槽超表面几何与物理参数。

    四种几何数组要么全部省略（使用文献 EP 名义参数，仅 J=3），要么全部给出；
    只给一部分会抛 :class:`MetagratingParameterError`，不再静默回填默认值。

    槽位语义：``x`` 为各槽起点（优先），未给时由 ``d`` 递推
    ``x[0]=d[0], x[s]=x[s-1]+w[s-1]+d[s]``，即 ``d[s]``（s≥1）为第 s-1 与第 s 槽
    之间的硬壁宽度、``d[0]`` 为周期起点到首槽的距离；槽后剩余部分为硬壁。
    注意：文档 B.1 的 ``d`` 是硬壁宽度（与本字段一致），而文档 B.2 的 ``d_s`` 是槽深
    （对应本字段 ``h``）——两处文献记号冲突，本模块以 B.1 为准，勿混用。

    Args:
        f0: 工作频率 (Hz)。
        c0: 背景介质（空气）声速实部 (m/s)。
        theta_i: 参考入射角 (度)，须在 (−90°, 90°) 内。
        J: 每个周期的槽数；省略时由数组长度推断，全缺省时取 3。
        h: 各槽深度 (m)，形状 (J,)，须为正。
        w: 各槽宽度 (m)，形状 (J,)，须为正。
        d: 各槽前的硬壁间距/首槽偏移 (m)，形状 (J,)，须非负；与 ``x`` 二选一。
        c_complex: 各槽等效复声速 (m/s)，形状 (J,)，须非零；虚部为损耗/增益。
        n_orders: Floquet 截断阶数下限（求解 −n_orders…+n_orders 阶）。构造时自动
            提升到覆盖全部传播阶次并留至少一阶倏逝缓冲后的值。
        k_modes: 槽内波导模截断数（0…k_modes−1）。
        period: 光栅周期 D (m)。省略时按光栅方程由 θ_i、f0 反推
            ``D = λ/(2|sin θ_i|)``（即旧版 ±θ 对称逆反射特例）；θ_i≈0 时必须显式给定。
        x: 各槽起点 (m)，形状 (J,)，须严格递增且槽间不重叠；给出时优先于 ``d``。

    Attributes:
        J: 构造后恒为整数。
        h, w, c_complex: 构造后恒为形状 (J,) 的数组。
        d: 硬壁间距；与 x 二选一，未给时为 None。
        x: 显式槽位；未给时为 None。实际槽位见 ``groove_positions``。
        n_orders: 构造后为提升后的实际截断阶数。
        period: 构造后为实际使用的周期 D (m)。
    """

    f0: float = 3430.0
    c0: float = 343.0
    theta_i: float = 45.0  # degrees
    J: int | None = None
    h: _FloatArray | None = None
    w: _FloatArray | None = None
    d: _FloatArray | None = None
    c_complex: _ComplexArray | None = None
    n_orders: int = 10
    k_modes: int = 31
    period: float | None = None
    x: _FloatArray | None = None

    def __post_init__(self) -> None:
        self._validate_scalars()
        self.period = self._resolve_period()
        self._normalise_grooves()
        self._validate_grooves()
        self._normalise_orders()

    # -- 校验与归一化 ------------------------------------------------------

    def _validate_scalars(self) -> None:
        if not np.isfinite(self.f0) or self.f0 <= 0:
            raise MetagratingParameterError(f"f0 必须为有限正数，得到 {self.f0!r}")
        if not np.isfinite(self.c0) or self.c0 <= 0:
            raise MetagratingParameterError(f"c0 必须为有限正数，得到 {self.c0!r}")
        if not np.isfinite(self.theta_i) or abs(self.theta_i) >= 90:
            raise MetagratingParameterError(
                f"theta_i 必须在 (−90°, 90°) 内（掠入射奇异），得到 {self.theta_i!r}"
            )
        if not isinstance(self.n_orders, (int, np.integer)) or self.n_orders < 0:
            raise MetagratingParameterError(
                f"n_orders 必须为非负整数，得到 {self.n_orders!r}"
            )
        if not isinstance(self.k_modes, (int, np.integer)) or self.k_modes < 1:
            raise MetagratingParameterError(
                f"k_modes 必须为≥1 的整数，得到 {self.k_modes!r}"
            )

    def _resolve_period(self) -> float:
        if self.period is not None:
            D = float(self.period)
            if not np.isfinite(D) or D <= 0:
                raise MetagratingParameterError(f"period 必须为有限正数，得到 {D!r}")
            return D
        sin_i = abs(np.sin(np.radians(self.theta_i)))
        if sin_i < 1e-12:
            raise MetagratingParameterError(
                "theta_i≈0 时周期无法由光栅方程反推，请显式给定 period"
            )
        return float(self.c0 / self.f0 / (2.0 * sin_i))

    def _normalise_grooves(self) -> None:
        provided = [
            name
            for name in ("h", "w", "d", "c_complex", "x")
            if getattr(self, name) is not None
        ]
        if not provided:
            if self.J is None:
                self.J = 3
            if self.J != 3:
                raise MetagratingParameterError(
                    f"默认 EP 几何仅有 J=3 一套，J={self.J} 时须显式给出 h/w/d(或x)/c_complex"
                )
            lam = self.c0 / self.f0
            D = self.period
            self.h = np.array([0.569, 0.195, 0.232]) * lam
            self.w = np.array([0.227, 0.115, 0.153]) * D
            self.d = np.array([0.0, 0.276, 0.070]) * D
            self.c_complex = self.c0 * np.array([1.0, 1.0081 + 0.0745j, 1.0])
            self.x = None
            return

        missing = [
            name for name in ("h", "w", "c_complex") if getattr(self, name) is None
        ]
        if self.d is None and self.x is None:
            missing.append("d 或 x")
        if missing:
            raise MetagratingParameterError(
                "几何参数部分缺省：缺少 "
                + "、".join(missing)
                + "；默认参数仅在 h/w/d(或x)/c_complex 全缺省时启用"
            )

        h = np.asarray(self.h, dtype=np.float64)
        w = np.asarray(self.w, dtype=np.float64)
        c = np.asarray(self.c_complex, dtype=np.complex128)
        if self.J is None:
            self.J = int(h.size)
        if not isinstance(self.J, (int, np.integer)) or self.J < 1:
            raise MetagratingParameterError(f"J 必须为正整数，得到 {self.J!r}")
        self.J = int(self.J)
        for name, arr in (("h", h), ("w", w), ("c_complex", c)):
            if arr.shape != (self.J,):
                raise MetagratingParameterError(
                    f"{name} 形状应为 ({self.J},)，实际为 {arr.shape}"
                )
        self.h = h
        self.w = w
        self.c_complex = c

        tol = _GEOM_RTOL * self.period
        if self.d is not None:
            d = np.asarray(self.d, dtype=np.float64)
            if d.shape != (self.J,):
                raise MetagratingParameterError(
                    f"d 形状应为 ({self.J},)，实际为 {d.shape}"
                )
            self.d = d
        if self.x is not None:
            x = np.asarray(self.x, dtype=np.float64)
            if x.shape != (self.J,):
                raise MetagratingParameterError(
                    f"x 形状应为 ({self.J},)，实际为 {x.shape}"
                )
            self.x = x
        if (
            self.x is not None
            and self.d is not None
            and np.any(np.abs(self.x - self._x_from_d()) > tol)
        ):
            raise MetagratingParameterError(
                "d 与 x 不一致：两者只能择一，或给出彼此一致的槽位"
            )

    def _x_from_d(self) -> _FloatArray:
        x = np.empty(self.J, dtype=np.float64)
        x[0] = self.d[0]
        for s in range(1, self.J):
            x[s] = x[s - 1] + self.w[s - 1] + self.d[s]
        return x

    @property
    def groove_positions(self) -> _FloatArray:
        """各槽起点 x_s (m)：显式 ``x`` 优先，否则由 ``d`` 递推（随数组改动实时生效）。"""
        return self._x_from_d() if self.x is None else np.asarray(self.x)

    def _validate_grooves(self) -> None:
        x = self.groove_positions
        arrays = [("h", self.h), ("w", self.w), ("x", x)]
        if self.d is not None:
            arrays.append(("d", self.d))
        for name, arr in arrays:
            if not np.all(np.isfinite(arr)):
                raise MetagratingParameterError(f"{name} 含非有限值")
        if not np.all(np.isfinite(self.c_complex)):
            raise MetagratingParameterError("c_complex 含非有限值")
        if np.any(self.w <= 0):
            raise MetagratingParameterError(f"w 必须逐槽为正，得到 {self.w}")
        if np.any(self.h <= 0):
            raise MetagratingParameterError(f"h 必须逐槽为正，得到 {self.h}")
        if self.d is not None and np.any(self.d < 0):
            raise MetagratingParameterError(f"d 必须非负，得到 {self.d}")
        if np.any(np.abs(self.c_complex) == 0):
            raise MetagratingParameterError(
                f"c_complex 必须逐槽非零，得到 {self.c_complex}"
            )
        if np.any(x < 0):
            raise MetagratingParameterError(f"x 必须非负，得到 {x}")

        D = self.period
        tol = _GEOM_RTOL * D
        if np.sum(self.w) > D + tol:
            raise MetagratingParameterError(
                f"管槽总宽 {np.sum(self.w):.6g} m 超出周期 D={D:.6g} m"
            )
        if np.any(np.diff(x) < self.w[:-1] - tol):
            raise MetagratingParameterError(
                "x 必须严格递增且相邻槽不重叠（要求 x[s] ≥ x[s-1] + w[s-1]）"
            )
        end = x[-1] + self.w[-1]
        if end > D + tol:
            raise MetagratingParameterError(
                f"最后一个管槽末端 {end:.6g} m 超出周期 D={D:.6g} m"
            )

    def _normalise_orders(self) -> None:
        prop = _propagating_orders(self.k0, self.G, np.radians(self.theta_i))
        span = int(prop[-1] - prop[0])
        n_floor = int(np.max(np.abs(prop))) + 1
        self.n_orders = max(int(self.n_orders), n_floor, span + 1)

    # -- 派生量 ------------------------------------------------------------

    @property
    def k0(self) -> float:
        """背景介质波数 (rad/m)。"""
        return 2.0 * np.pi * self.f0 / self.c0

    @property
    def wavelength(self) -> float:
        """背景介质波长 (m)。"""
        return self.c0 / self.f0

    @property
    def G(self) -> float:
        """倒格矢 2π/D (rad/m)。"""
        return 2.0 * np.pi / self.period

    @property
    def N(self) -> int:
        """Floquet 阶数总数 (2·n_orders + 1)。"""
        return 2 * self.n_orders + 1

    @property
    def K(self) -> int:
        """槽内波导模数。"""
        return self.k_modes


# ===========================================================================
# 端口集合
# ===========================================================================


def _propagating_orders(k0: float, G: float, theta_rad: float) -> _IntArray:
    """返回声锥内的衍射阶次 m（升序）。

    判据 ``|k0·sinθ + mG| < k0(1 − _CONE_TOL)``。m=0 恒满足（|sinθ| < 1）。
    """
    kx0 = k0 * np.sin(theta_rad)
    m_lo = int(np.floor((-k0 - kx0) / G)) - 1
    m_hi = int(np.ceil((k0 - kx0) / G)) + 1
    ms = np.arange(m_lo, m_hi + 1, dtype=np.int64)
    kx = kx0 + ms * G
    return ms[np.abs(kx) < k0 * (1.0 - _CONE_TOL)]


def propagating_orders(params: MetagratingParams) -> _IntArray:
    """参考入射 θ_i 下的传播阶次集合 M（附录 A 相图）。

    WA 线 ``k_x + m(2π/D) = ±n0·k`` 之间、声锥 ``k = (1/n0)k_x`` 内的阶次为传播模；
    本模块取 n0 = 1（空气背景）。掠射边界 ``|k_x| = k`` 不计入（k_z = 0，方程组奇异）。

    Args:
        params: 超表面参数。

    Returns:
        升序整数数组，例如 θ_i=45°、D=λ/√2 时为 ``[−1, 0]``。
    """
    return _propagating_orders(params.k0, params.G, np.radians(params.theta_i))


# ===========================================================================
# CMT 核心求解器
# ===========================================================================


def _resolve_incidence(params: MetagratingParams, incidence: str | float) -> float:
    """把 ``"left"``/``"right"``/角度（度）解析为入射角（度）。"""
    if isinstance(incidence, str):
        if incidence == "left":
            return params.theta_i
        if incidence == "right":
            return -params.theta_i
        raise MetagratingParameterError(
            f"incidence 须为 'left'、'right' 或角度（度），得到 {incidence!r}"
        )
    theta_deg = float(incidence)
    if not np.isfinite(theta_deg) or abs(theta_deg) >= 90:
        raise MetagratingParameterError(
            f"入射角须在 (−90°, 90°) 内，得到 {theta_deg!r}"
        )
    return theta_deg


@dataclass(frozen=True, eq=False)
class _Assembly:
    """固定 (f0, c0, D, 入射角, 几何) 的介质无关装配量。

    β、U 只依赖介质 c_complex 与槽深 h，其余矩阵块（P1/P2/V1/V2 与 V3 的
    积分核）都与介质无关；因此同几何、同频率、同入射角的批量求值可复用本对象。
    """

    ky: _ComplexArray
    alpha: _FloatArray
    P2: _ComplexArray
    I3: _ComplexArray
    P1: _ComplexArray
    V1: _ComplexArray
    m22: _FloatArray
    D: float
    n_orders: int
    N: int
    K: int
    theta_deg: float


def _build_assembly(
    f0: float,
    c0: float,
    D: float,
    theta_deg: float,
    n_orders: int,
    k_modes: int,
    x: _FloatArray,
    w: _FloatArray,
) -> _Assembly:
    """装配 B.2.6–B.2.11 中与介质无关的矩阵块（含一次积分核求值）。"""
    k0 = 2.0 * np.pi * f0 / c0
    G = 2.0 * np.pi / D
    N = 2 * n_orders + 1
    J = x.size
    K = k_modes
    theta = np.radians(theta_deg)

    n = np.arange(-n_orders, n_orders + 1)
    k_x = k0 * np.sin(theta) + n * G
    k_y = np.sqrt(k0**2 - k_x.astype(complex) ** 2)
    k_y = np.where(k_y.imag > 0, -k_y, k_y)  # Im(k_y) ≤ 0：倏逝波衰减

    k_idx = np.arange(K)
    alpha = (k_idx * np.pi)[np.newaxis, :] / w[:, np.newaxis]  # α_ks (J,K)

    # P_2 (B.2.7)：(J*K, N) 反射压力耦合。所有槽一次向量化；
    # V_3 (B.2.11) 的积分核是它的复共轭（k_x、α、x、w 均为实数）。
    s_modes = alpha.ravel()
    sign_modes = np.tile(np.where(k_idx % 2 == 0, 1.0, -1.0), J)
    w_modes = np.repeat(w, K)
    t_ref = (-1j * k_x)[:, np.newaxis]
    e_tw = np.repeat(np.exp(t_ref * w[np.newaxis, :]), K, axis=1)
    e_tx = np.repeat(np.exp(t_ref * x[np.newaxis, :]), K, axis=1)
    I2 = _overlap_harmonic(t_ref, s_modes, sign_modes, w_modes, e_tw, e_tx)
    P2 = (I2 / w_modes[np.newaxis, :]).T
    I3 = np.conj(I2)

    # P_1 (B.2.6) 就是 P_2 在 m=0 列（入射阶本身就是参考栅格的 m=0）。
    P1 = P2[:, n_orders].copy()

    # V_1 (B.2.9)：入射法向速度投影。
    V1 = np.zeros(N, dtype=complex)
    V1[n_orders] = -k0 * np.cos(theta)

    # P_3 (B.2.8) 的对角系数（M22 正交性在 k1≠k2 时严格为 0）。
    m22_k = np.full(K, 0.5)
    m22_k[0] = 1.0

    return _Assembly(
        ky=k_y,
        alpha=alpha,
        P2=P2,
        I3=I3,
        P1=P1,
        V1=V1,
        m22=np.tile(m22_k, J),
        D=D,
        n_orders=n_orders,
        N=N,
        K=K,
        theta_deg=theta_deg,
    )


@lru_cache(maxsize=256)
def _cached_assembly(
    f0: float,
    c0: float,
    D: float,
    theta_deg: float,
    n_orders: int,
    k_modes: int,
    x_key: tuple[float, ...],
    w_key: tuple[float, ...],
) -> _Assembly:
    return _build_assembly(
        f0,
        c0,
        D,
        theta_deg,
        n_orders,
        k_modes,
        np.asarray(x_key, dtype=float),
        np.asarray(w_key, dtype=float),
    )


def _get_assembly(params: MetagratingParams, theta_deg: float) -> _Assembly:
    """按 (f0, c0, D, 入射角, 几何) 取装配量；介质与槽深不入键。"""
    return _cached_assembly(
        float(params.f0),
        float(params.c0),
        float(params.period),
        float(theta_deg),
        int(params.n_orders),
        int(params.k_modes),
        tuple(float(v) for v in params.groove_positions),
        tuple(float(v) for v in params.w),
    )


@dataclass(frozen=True, eq=False)
class _MediumTerms:
    """介质/槽深相关项（与入射角无关，同一几何的各端口共享）。"""

    c3: _ComplexArray
    g: _ComplexArray
    one_plus: _ComplexArray
    elim: NDArray[np.bool_]


def _medium_terms(
    asm: _Assembly, f0: float, c_complex: _ComplexArray, h: _FloatArray
) -> _MediumTerms:
    """计算 β/U 及其在 Schur 消元中的系数（每介质只算一次）。"""
    kc = 2.0 * np.pi * f0 / np.asarray(c_complex, dtype=complex)
    beta = np.sqrt(kc[:, np.newaxis] ** 2 - asm.alpha**2)
    beta = np.where(beta.imag > 0, -beta, beta)  # Im(β) ≤ 0、|U| ≤ 1
    U = np.exp(-2j * beta * h[:, np.newaxis])  # 往返因子 (J,K)

    one_plus = (1.0 + U).ravel()
    c3 = -(beta.ravel() / asm.D) * (1.0 - U).ravel()  # V3 = I3 · c3（逐列）
    elim = np.abs(one_plus) > _SCHUR_TOL
    g = c3[elim] / asm.m22[elim] / one_plus[elim]
    return _MediumTerms(c3=c3, g=g, one_plus=one_plus, elim=elim)


def _solve_assembly(asm: _Assembly, terms: _MediumTerms) -> _ComplexArray:
    """对装配量做块对角 Schur 消元并求解，返回 r_m（索引 m + n_orders）。

    原文方程组 [(-P2,P3);(-V2,V3)]·(A^-;A) = (P1;V1) 中 P3 为块对角（每槽对角），
    由第一行 A = P3^{-1}(P1 + P2 A^-) 代入第二行，主系统由 (N+J·K) 降到 N：
        (V3 P3^{-1} P2 − V2) A^- = V1 − V3 P3^{-1} P1。
    对 |1+U| ≤ _SCHUR_TOL 的共振模（U→−1）不做除法，保留为补充未知量
    （把 P3 的微小对角元一并保留），与完整稠密解等价。
    """
    N = asm.N
    elim = terms.elim
    g = terms.g
    V3W = asm.I3[:, elim] * g[np.newaxis, :]
    if elim.all():
        M = V3W @ asm.P2
        M[np.arange(N), np.arange(N)] -= asm.ky
        return np.linalg.solve(M, asm.V1 - V3W @ asm.P1)

    n_s = int((~elim).sum())
    M = np.zeros((N + n_s, N + n_s), dtype=complex)
    rhs = np.zeros(N + n_s, dtype=complex)
    M[:N, :N] = V3W @ asm.P2[elim, :]
    M[np.arange(N), np.arange(N)] -= asm.ky
    M[:N, N:] = asm.I3[:, ~elim] * terms.c3[~elim][np.newaxis, :]
    M[N:, :N] = -asm.P2[~elim, :]
    idx = N + np.arange(n_s)
    M[idx, idx] = asm.m22[~elim] * terms.one_plus[~elim]
    rhs[:N] = asm.V1 - V3W @ asm.P1[elim]
    rhs[N:] = asm.P1[~elim]
    return np.linalg.solve(M, rhs)[:N]


def _solve_at_angle(params: MetagratingParams, theta_deg: float) -> _ComplexArray:
    asm = _get_assembly(params, theta_deg)
    return _solve_assembly(
        asm, _medium_terms(asm, params.f0, params.c_complex, params.h)
    )


def _solve_reflection_orders(
    params: MetagratingParams, theta_rad: float
) -> _ComplexArray:
    """在任意入射角（弧度）下求解 Floquet 反射系数 r_m。"""
    return _solve_at_angle(params, float(np.degrees(theta_rad)))


def _ports_for(k0: float, G: float, theta_rad: float) -> tuple[Port, ...]:
    """参考入射下按 k_x 降序排列的传播端口。"""
    ms = _propagating_orders(k0, G, theta_rad)
    kxs = k0 * np.sin(theta_rad) + ms * G
    order = np.argsort(-kxs, kind="stable")
    ms = ms[order]
    kxs = kxs[order]
    return tuple(
        Port(
            order=int(m),
            kx=float(kx),
            angle_deg=float(np.degrees(np.arcsin(kx / k0))),
            kz=float(np.sqrt(k0**2 - kx**2)),
        )
        for m, kx in zip(ms, kxs)
    )


@dataclass(frozen=True, eq=False)
class _Kernel:
    """固定参考 (f0, θ_i, 几何) 的端口与各端口装配量。"""

    ports: tuple[Port, ...]
    assemblies: tuple[_Assembly, ...]
    n_orders: int


def _kernel_key(params: MetagratingParams) -> tuple:
    """参考核的哈希键（介质与槽深不入键）。"""
    return (
        float(params.f0),
        float(params.c0),
        float(params.period),
        float(params.theta_i),
        int(params.n_orders),
        int(params.k_modes),
        tuple(float(v) for v in params.groove_positions),
        tuple(float(v) for v in params.w),
    )


@lru_cache(maxsize=64)
def _mirror_assembly(asm: _Assembly) -> _Assembly:
    """由 +θ 装配量导出 −θ（栅格反序 + 复共轭；几何全实时精确）。

    k_x^n(−θ) = −k_x^{−n}(+θ)，故 I₂(−θ) = conj(I₂(+θ))[反序, :]，
    k_y(−θ) = k_y(+θ)[反序]，V₁ 关于 θ 为偶函数。用于 ±θ 成对端口的快核。
    """
    return _Assembly(
        ky=np.ascontiguousarray(asm.ky[::-1]),
        alpha=asm.alpha,
        P2=np.ascontiguousarray(np.conj(asm.P2)[:, ::-1]),
        I3=np.ascontiguousarray(np.conj(asm.I3)[::-1, :]),
        P1=np.ascontiguousarray(np.conj(asm.P1)),
        V1=asm.V1,
        m22=asm.m22,
        D=asm.D,
        n_orders=asm.n_orders,
        N=asm.N,
        K=asm.K,
        theta_deg=-asm.theta_deg,
    )


@lru_cache(maxsize=64)
def _cached_kernel(
    f0: float,
    c0: float,
    D: float,
    theta_i: float,
    n_orders: int,
    k_modes: int,
    x_key: tuple[float, ...],
    w_key: tuple[float, ...],
) -> _Kernel:
    ports = _ports_for(2.0 * np.pi * f0 / c0, 2.0 * np.pi / D, np.radians(theta_i))
    built: list[tuple[float, _Assembly]] = []
    assemblies: list[_Assembly] = []
    for port in ports:
        mirrored = None
        for angle, asm in built:
            if (
                abs(angle + port.angle_deg) <= 1e-9
                and abs(angle - port.angle_deg) > 1e-9
            ):
                mirrored = _mirror_assembly(asm)
                break
        if mirrored is None:
            mirrored = _cached_assembly(
                f0, c0, D, port.angle_deg, n_orders, k_modes, x_key, w_key
            )
            built.append((port.angle_deg, mirrored))
        assemblies.append(mirrored)
    return _Kernel(ports=ports, assemblies=tuple(assemblies), n_orders=n_orders)


def _get_kernel(params: MetagratingParams) -> _Kernel:
    return _cached_kernel(*_kernel_key(params))


def _s_matrix_array(
    kernel: _Kernel, params: MetagratingParams, c_complex: _ComplexArray
) -> _ComplexArray:
    """由核与介质计算 S 矩阵（端口语义见 :func:`solve_s_matrix`）。"""
    n_orders = kernel.n_orders
    ports = kernel.ports
    # 同一核的各端口共享 α/m22/D，介质项（β/U/g）只需算一次。
    terms = _medium_terms(kernel.assemblies[0], params.f0, c_complex, params.h)
    S = np.empty((len(ports), len(ports)), dtype=complex)
    for q, (port_q, asm) in enumerate(zip(ports, kernel.assemblies)):
        r = _solve_assembly(asm, terms)
        for p, port_p in enumerate(ports):
            S[p, q] = r[(port_p.order - port_q.order) + n_orders]
    return S


def _as_medium(c_complex, J: int) -> _ComplexArray:
    c = np.asarray(c_complex, dtype=complex)
    if c.shape != (J,):
        raise MetagratingParameterError(f"c_complex 形状应为 ({J},)，实际为 {c.shape}")
    if not np.all(np.isfinite(c)) or np.any(c == 0):
        raise MetagratingParameterError("c_complex 必须为有限非零复数")
    return c


def _as_medium_batch(c_complex, J: int) -> _ComplexArray:
    c = np.asarray(c_complex, dtype=complex)
    if c.ndim == 1:
        c = c[np.newaxis, :]
    if c.ndim != 2 or c.shape[1] != J:
        raise MetagratingParameterError(
            f"c_complex 批量形状应为 (B, {J})，实际为 {c.shape}"
        )
    if not np.all(np.isfinite(c)) or np.any(c == 0):
        raise MetagratingParameterError("c_complex 必须为有限非零复数")
    return c


class CMTForwardKernel:
    """固定 (f0, θ_i, 几何) 的快速前向核；介质 c_complex 可逐次/批量替换。

    装配量（P1/P2/V1/V2 与 V3 的积分核）按几何与入射角缓存，介质变化只重算
    β/U 与一次 N×N 求解。端口集合与语义与 :func:`solve_s_matrix` 完全一致；
    同一核内端口数固定，故批量结果可堆叠为 (B, P, P)。核在构造时固定几何，
    构造后修改 ``params`` 的几何字段不会刷新核（请重新构造）。

    Args:
        params: 超表面参数（提供 f0、θ_i、几何与初始 c_complex）。

    Attributes:
        params: 基准参数。
        ports: 端口元数据（按 k_x 降序）。
        n_ports: 端口数 P。
    """

    def __init__(self, params: MetagratingParams) -> None:
        self.params = params
        self._kernel = _get_kernel(params)

    @property
    def ports(self) -> tuple[Port, ...]:
        """端口元数据（与介质无关）。"""
        return self._kernel.ports

    @property
    def n_ports(self) -> int:
        """端口数 P。"""
        return len(self._kernel.ports)

    def s_matrix(self, c_complex=None) -> _ComplexArray:
        """计算 (P, P) S 矩阵；``c_complex`` 省略时用基准参数的介质。"""
        c = (
            self.params.c_complex
            if c_complex is None
            else _as_medium(c_complex, self.params.J)
        )
        return _s_matrix_array(self._kernel, self.params, c)

    def solve(self, c_complex=None) -> ScatteringMatrix:
        """返回带端口元数据的 :class:`ScatteringMatrix`（params 反映实际介质）。"""
        if c_complex is None:
            c = self.params.c_complex
            params = self.params
        else:
            c = _as_medium(c_complex, self.params.J)
            params = replace(self.params, c_complex=c)
        return ScatteringMatrix(
            S=_s_matrix_array(self._kernel, params, c),
            ports=self._kernel.ports,
            params=params,
        )

    def s_matrix_many(self, c_complex_batch) -> _ComplexArray:
        """批量介质求值，返回 (B, P, P) 堆叠 S 矩阵。

        Args:
            c_complex_batch: 形状 (J,) 或 (B, J) 的复声速数组。

        Returns:
            形状 (B, P, P) 的复 S 矩阵；第 i 个对应 c_complex_batch[i]。
        """
        cs = _as_medium_batch(c_complex_batch, self.params.J)
        out = np.empty((cs.shape[0], self.n_ports, self.n_ports), dtype=complex)
        for i, c in enumerate(cs):
            out[i] = _s_matrix_array(self._kernel, self.params, c)
        return out


def solve_s_matrix_batch(
    param_sets: list[MetagratingParams],
) -> list[ScatteringMatrix]:
    """多参数集批量求解（种群 / 多频 / 多几何一次调用）。

    同一 (f0, θ_i, 几何) 的参数复用同一核；介质或频率不同的参数各自求解。
    需要固定几何、只扫介质时优先用 :func:`sweep_medium`（返回堆叠数组，开销更低）。

    Args:
        param_sets: 超表面参数序列。

    Returns:
        与输入等长的 :class:`ScatteringMatrix` 列表。
    """
    kernels: dict[tuple, _Kernel] = {}
    out: list[ScatteringMatrix] = []
    for params in param_sets:
        key = _kernel_key(params)
        kernel = kernels.get(key)
        if kernel is None:
            kernel = _get_kernel(params)
            kernels[key] = kernel
        out.append(
            ScatteringMatrix(
                S=_s_matrix_array(kernel, params, params.c_complex),
                ports=kernel.ports,
                params=params,
            )
        )
    return out


def sweep_medium(params: MetagratingParams, c_complex_batch) -> _ComplexArray:
    """固定几何与参考角、批量替换介质 c_complex 的 S 矩阵。

    这是参数空间探索／逆向设计的热路径：装配量只算一次，每个介质仅重算
    β/U 与 N×N 求解。

    Args:
        params: 基准超表面参数。
        c_complex_batch: 形状 (J,) 或 (B, J) 的复声速数组。

    Returns:
        形状 (B, P, P) 的复 S 矩阵；端口语义见 ``CMTForwardKernel(params).ports``。
    """
    return CMTForwardKernel(params).s_matrix_many(c_complex_batch)


def clear_forward_cache() -> None:
    """清空装配/核的 LRU 缓存（基准脚本与测试用；正常使用无需调用）。"""
    _cached_assembly.cache_clear()
    _cached_kernel.cache_clear()


def solve_reflection_cmt(
    params: MetagratingParams, incidence: str | float = "left"
) -> _ComplexArray:
    """求解反射型超表面的 CMT 线性方程组（文档 B.2.5–B.2.11）。

    未知量前 N 个为 Floquet 反射振幅 A^-（索引 n_orders 对应 m=0），后 J·K 个为
    各槽波导模振幅。分支取 Im(k_y)≤0、Im(β)≤0（倏逝波衰减），往返因子
    U = exp(−2jβh)（与 MATLAB 参考代码的 exp(+2jβh) 不同，后者对有损槽会溢出）。

    Args:
        params: 超表面参数。
        incidence: ``"left"``（θ_i）、``"right"``（−θ_i）或任意入射角（度）。

    Returns:
        反射振幅向量 r_m，形状 (2·n_orders+1,)，索引 ``m + n_orders`` 对应第 m 阶。
    """
    theta_deg = _resolve_incidence(params, incidence)
    return _solve_at_angle(params, theta_deg)


def solve_s_matrix(params: MetagratingParams) -> ScatteringMatrix:
    """求解多端口反射 S 矩阵及其通道语义。

    端口集合由附录 A 相图决定：``M = {m : |k sinθ_i + mG| < k}``，按 k_x 降序编号，
    S 为 |M|×|M|。对每个入射端口 q 以角度 ``ports[q].angle_deg`` 求解一次 CMT，
    再按出射阶次 ``p.order − q.order`` 提取矩阵元。端口数随 (f0, θ_i, D) 变化，
    不恒为 2×2。

    Args:
        params: 超表面参数。

    Returns:
        :class:`ScatteringMatrix`，含 S、端口元数据与参数引用。
    """
    kernel = _get_kernel(params)
    return ScatteringMatrix(
        S=_s_matrix_array(kernel, params, params.c_complex),
        ports=kernel.ports,
        params=params,
    )


def compute_s_matrix(params: MetagratingParams) -> _ComplexArray:
    """计算多端口反射 S 矩阵（``solve_s_matrix(params).S`` 的便捷入口）。

    θ_i = 45°、D = λ/√2（默认 EP/DP 参数）时返回旧版 2×2 布局
    ``[[r_0^L, r_{+1}^R], [r_{-1}^L, r_0^R]]``；其他 (f0, θ_i, D) 下维度由传播
    阶次集合决定。通道语义见 :func:`solve_s_matrix`。

    Args:
        params: 超表面参数。

    Returns:
        (P, P) 复散射矩阵，P 为传播通道数。
    """
    return solve_s_matrix(params).S


def sweep_s_matrix(
    params: MetagratingParams, freqs: float | _FloatArray
) -> list[ScatteringMatrix]:
    """频率批量接口：固定几何、逐频求解 S 矩阵。

    取舍：各频率的 Floquet 基（k_x、k_y）与槽内模态（α 与 w 无关、β、U）都随频率
    变化，几乎没有可复用的装配中间量；且端口数会随频率改变，S 矩阵维度不同，无法
    堆叠成单个 ndarray。因此按频率逐次装配：每次调用复用几何数组，返回带各自端口
    元数据的 :class:`ScatteringMatrix` 列表。频率扫描时 ``period`` 固定（不再由光栅
    方程随 f0 重算），``c_complex`` 按频率无关的等效声速处理。

    Args:
        params: 基准超表面参数（几何与声速固定，f0 仅作首个参考）。
        freqs: 频率 (Hz) 标量或一维数组，须为有限正数。

    Returns:
        与 ``freqs`` 等长的列表，第 i 项为 f = freqs[i] 的 S 矩阵。
    """
    freqs_arr = np.atleast_1d(np.asarray(freqs, dtype=np.float64))
    if freqs_arr.ndim != 1:
        raise MetagratingParameterError("freqs 必须为一维数组")
    if np.any(~np.isfinite(freqs_arr)) or np.any(freqs_arr <= 0):
        raise MetagratingParameterError(f"频率必须为有限正数，得到 {freqs_arr}")
    return [solve_s_matrix(replace(params, f0=float(f))) for f in freqs_arr]


# ===========================================================================
# 参数化便捷接口
# ===========================================================================


def make_ep_params(
    cr: float = 1.0081,
    ci: float = 0.0745,
    n_orders: int = 10,
    k_modes: int = 31,
) -> MetagratingParams:
    """创建文献损耗 EP 构型的超表面参数（θ_i=45°，D=λ/√2，J=3）。

    Args:
        cr: 第 2 槽归一化复声速实部。
        ci: 第 2 槽归一化复声速虚部（正为损耗）。
        n_orders: Floquet 截断阶数下限。
        k_modes: 槽内波导模截断数。

    Returns:
        超表面参数。
    """
    lam = 343.0 / 3430.0  # 0.1 m
    a = lam / np.sqrt(2)  # period
    return MetagratingParams(
        f0=3430.0,
        c0=343.0,
        theta_i=45.0,
        J=3,
        h=np.array([0.569, 0.195, 0.232]) * lam,
        w=np.array([0.227, 0.115, 0.153]) * a,
        d=np.array([0.0, 0.276, 0.070]) * a,
        c_complex=np.array([343.0, 343.0 * (cr + ci * 1j), 343.0]),
        n_orders=n_orders,
        k_modes=k_modes,
    )


def make_dp_params(
    cr1: float = 1.0,
    ci1: float = 0.240,
    cr3: float = 1.0,
    ci3: float = 0.251,
    n_orders: int = 10,
    k_modes: int = 31,
) -> MetagratingParams:
    """创建文献 DP 构型的超表面参数（θ_i=45°，D=λ/√2，J=3）。

    DP 构型中第 1、3 槽有损耗，第 2 槽无损耗。

    Args:
        cr1: 第 1 槽归一化复声速实部。
        ci1: 第 1 槽归一化复声速虚部。
        cr3: 第 3 槽归一化复声速实部。
        ci3: 第 3 槽归一化复声速虚部。
        n_orders: Floquet 截断阶数下限。
        k_modes: 槽内波导模截断数。

    Returns:
        超表面参数。
    """
    lam = 343.0 / 3430.0
    a = lam / np.sqrt(2)
    return MetagratingParams(
        f0=3430.0,
        c0=343.0,
        theta_i=45.0,
        J=3,
        h=np.array([0.211, 0.491, 0.210]) * lam,
        w=np.array([0.137, 0.187, 0.145]) * a,
        d=np.array([0.0, 0.096, 0.081]) * a,
        c_complex=np.array([343.0 * (cr1 + ci1 * 1j), 343.0, 343.0 * (cr3 + ci3 * 1j)]),
        n_orders=n_orders,
        k_modes=k_modes,
    )


# ===========================================================================
# 主计算入口
# ===========================================================================


def main(session_dir: Path | None = None) -> None:
    """计算入口：验证收敛 → 定位 EP → 参数空间扫描 → 黎曼面可视化。"""
    if session_dir is None:
        session_dir = ensure_session("gain_ep", "cmt_reflection_s_matrix")
    plots_dir = session_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "results").mkdir(parents=True, exist_ok=True)
    print(f"[session] {session_dir}")

    # === 0. 收敛性检查：不同截断阶数对比 ===
    print("\n" + "=" * 60)
    print("Step 0: 收敛性检查 (EP 名义参数, 不同截断)")
    print("=" * 60)
    for n_ord, k_mod in [(4, 11), (6, 21), (8, 31), (10, 31), (12, 41)]:
        p_test = make_ep_params(cr=1.0081, ci=0.0745, n_orders=n_ord, k_modes=k_mod)
        S_test = compute_s_matrix(p_test)
        eigs_test = np.linalg.eigvals(S_test)
        print(
            f"  n_orders={n_ord:2d}, k_modes={k_mod:2d} -> "
            f"|S11|={abs(S_test[0, 0]):.4f}, |S12|={abs(S_test[0, 1]):.4f}, "
            f"|S21|={abs(S_test[1, 0]):.4f}, |S22|={abs(S_test[1, 1]):.4f}, "
            f"|λ1-λ2|={abs(eigs_test[0] - eigs_test[1]):.4e}"
        )

    # === 1. 在 (cr, ci) 参数空间中定位 EP：|r_-1^L|=|S21| → 0 ===
    print("\n" + "=" * 60)
    print("Step 1: 定位 EP（|S21|=|r_-1^L| 最小处, n_orders=8, k_modes=21）")
    print("=" * 60)
    cr_scan = np.arange(0.99, 1.05, 0.005)
    ci_scan = np.arange(0.03, 0.16, 0.005)
    best = (cr_scan[0], ci_scan[0], np.inf, None)
    for cr in cr_scan:
        for ci in ci_scan:
            p = make_ep_params(cr=cr, ci=ci, n_orders=8, k_modes=21)
            S = compute_s_matrix(p)
            s21 = abs(S[1, 0])
            if s21 < best[2]:
                best = (cr, ci, s21, S)
    cr_ep, ci_ep, s21_ep, S_ep = best
    print(f"  EP 候选: cr={cr_ep:.4f}, ci={ci_ep:.4f}, |S21|={s21_ep:.4e}")
    print(f"  S = [[{S_ep[0, 0]:+.5f}, {S_ep[0, 1]:+.5f}],")
    print(f"       [{S_ep[1, 0]:+.5f}, {S_ep[1, 1]:+.5f}]]")
    eigenvalues, eigenvectors = np.linalg.eig(S_ep)
    print(f"  本征值: λ₁={eigenvalues[0]:+.6f}, λ₂={eigenvalues[1]:+.6f}")
    print(f"  |λ₁-λ₂| = {abs(eigenvalues[0] - eigenvalues[1]):.4e}（EP 处应→0）")
    print(
        f"  Jordan 判据: S11≈S22? {np.isclose(S_ep[0, 0], S_ep[1, 1], atol=1e-3)}, "
        f"|S21|≈0? {s21_ep < 0.05}, |S12|≠0? {abs(S_ep[0, 1]) > 0.3}"
    )

    # === 2. 参数空间扫描（黎曼面），以定位到的 EP 为中心 ===
    print("\n" + "=" * 60)
    print("Step 2: (cr, ci) 参数空间扫描 — 黎曼面")
    print("=" * 60)

    n_scan_orders = 6
    k_scan_modes = 21

    cr_range = np.linspace(cr_ep - 0.02, cr_ep + 0.02, 25)
    ci_range = np.linspace(ci_ep - 0.04, ci_ep + 0.04, 25)

    n_cr, n_ci = len(cr_range), len(ci_range)
    eig1_real = np.zeros((n_cr, n_ci))
    eig1_imag = np.zeros((n_cr, n_ci))
    eig2_real = np.zeros((n_cr, n_ci))
    eig2_imag = np.zeros((n_cr, n_ci))
    s11_abs = np.zeros((n_cr, n_ci))
    s12_abs = np.zeros((n_cr, n_ci))
    s21_abs = np.zeros((n_cr, n_ci))
    s22_abs = np.zeros((n_cr, n_ci))

    total = n_cr * n_ci
    count = 0
    for i, cr in enumerate(cr_range):
        for j, ci in enumerate(ci_range):
            count += 1
            if count % 100 == 0:
                print(f"  进度: {count}/{total}")
            p = make_ep_params(
                cr=cr, ci=ci, n_orders=n_scan_orders, k_modes=k_scan_modes
            )
            S = compute_s_matrix(p)
            eigs = np.linalg.eigvals(S)
            eigs = eigs[np.argsort(eigs.real)]
            eig1_real[i, j] = eigs[0].real
            eig1_imag[i, j] = eigs[0].imag
            eig2_real[i, j] = eigs[1].real
            eig2_imag[i, j] = eigs[1].imag
            s11_abs[i, j] = abs(S[0, 0])
            s12_abs[i, j] = abs(S[0, 1])
            s21_abs[i, j] = abs(S[1, 0])
            s22_abs[i, j] = abs(S[1, 1])

    print(f"  扫描完成: {total} 点")

    # === 3. 可视化 ===
    print("\n" + "=" * 60)
    print("Step 3: 可视化")
    print("=" * 60)

    import matplotlib.pyplot as plt

    visualize.setup_style(chinese_fonts=True)

    CR, CI = np.meshgrid(cr_range, ci_range, indexing="ij")

    # --- 图 1: 黎曼面 (3D) ---
    fig = plt.figure(figsize=(14, 6))

    ax1 = fig.add_subplot(121, projection="3d")
    ax1.plot_surface(CI, CR, eig1_real, alpha=0.8, cmap="coolwarm")
    ax1.plot_surface(CI, CR, eig2_real, alpha=0.8, cmap="coolwarm")
    ax1.set_xlabel(r"Im($c_2$)/$c_0$")
    ax1.set_ylabel(r"Re($c_2$)/$c_0$")
    ax1.set_zlabel(r"Re($\lambda$)")
    ax1.set_title("黎曼面: 本征值实部")

    ax2 = fig.add_subplot(122, projection="3d")
    ax2.plot_surface(
        CI,
        CR,
        eig1_real,
        facecolors=plt.cm.viridis(
            (eig1_imag - eig1_imag.min()) / (eig1_imag.max() - eig1_imag.min() + 1e-15)
        ),
        alpha=0.9,
    )
    ax2.plot_surface(
        CI,
        CR,
        eig2_real,
        facecolors=plt.cm.viridis(
            (eig2_imag - eig2_imag.min()) / (eig2_imag.max() - eig2_imag.min() + 1e-15)
        ),
        alpha=0.9,
    )
    ax2.set_xlabel(r"Im($c_2$)/$c_0$")
    ax2.set_ylabel(r"Re($c_2$)/$c_0$")
    ax2.set_zlabel(r"Re($\lambda$)")
    ax2.set_title("黎曼面: 颜色=Im(λ)")

    fig.tight_layout()
    visualize.export_exploration(fig, plots_dir / "riemann_surface_3d.png", close=False)

    # --- 图 2: 本征值劈裂 + S 矩阵元 ---
    fig2, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig2.suptitle(
        rf"CMT 反射型超表面: EP 参数空间探索 (EP$\approx$cr={cr_ep:.3f}, ci={ci_ep:.3f})",
        fontsize=14,
        fontweight="bold",
    )

    im = axes[0, 0].pcolormesh(
        CI, CR, np.abs(eig1_real - eig2_real), cmap="hot_r", shading="auto"
    )
    axes[0, 0].set_xlabel(r"Im($c_2$)/$c_0$")
    axes[0, 0].set_ylabel(r"Re($c_2$)/$c_0$")
    axes[0, 0].set_title(r"|Re($\lambda_1$) - Re($\lambda_2$)|")
    plt.colorbar(im, ax=axes[0, 0])

    im = axes[0, 1].pcolormesh(
        CI, CR, np.abs(eig1_imag - eig2_imag), cmap="hot_r", shading="auto"
    )
    axes[0, 1].set_xlabel(r"Im($c_2$)/$c_0$")
    axes[0, 1].set_ylabel(r"Re($c_2$)/$c_0$")
    axes[0, 1].set_title(r"|Im($\lambda_1$) - Im($\lambda_2$)|")
    plt.colorbar(im, ax=axes[0, 1])

    eig_gap = np.sqrt((eig1_real - eig2_real) ** 2 + (eig1_imag - eig2_imag) ** 2)
    im = axes[0, 2].pcolormesh(CI, CR, eig_gap, cmap="magma_r", shading="auto")
    axes[0, 2].set_xlabel(r"Im($c_2$)/$c_0$")
    axes[0, 2].set_ylabel(r"Re($c_2$)/$c_0$")
    axes[0, 2].set_title(r"|$\lambda_1 - \lambda_2$| (本征值间距)")
    plt.colorbar(im, ax=axes[0, 2])

    im = axes[1, 0].pcolormesh(CI, CR, s11_abs, cmap="viridis", shading="auto")
    axes[1, 0].set_xlabel(r"Im($c_2$)/$c_0$")
    axes[1, 0].set_ylabel(r"Re($c_2$)/$c_0$")
    axes[1, 0].set_title(r"$|S_{11}| = |r_0^L|$")
    plt.colorbar(im, ax=axes[1, 0])

    im = axes[1, 1].pcolormesh(CI, CR, s12_abs, cmap="viridis", shading="auto")
    axes[1, 1].set_xlabel(r"Im($c_2$)/$c_0$")
    axes[1, 1].set_ylabel(r"Re($c_2$)/$c_0$")
    axes[1, 1].set_title(r"$|S_{12}| = |r_{+1}^R|$")
    plt.colorbar(im, ax=axes[1, 1])

    im = axes[1, 2].pcolormesh(CI, CR, s21_abs, cmap="viridis", shading="auto")
    axes[1, 2].set_xlabel(r"Im($c_2$)/$c_0$")
    axes[1, 2].set_ylabel(r"Re($c_2$)/$c_0$")
    axes[1, 2].set_title(r"$|S_{21}| = |r_{-1}^L|$")
    plt.colorbar(im, ax=axes[1, 2])

    fig2.tight_layout()
    visualize.export_exploration(fig2, plots_dir / "ep_parameter_space.png")

    # --- 图 3: 沿 ci 方向的 1D 切片（穿过 EP 点），验证 √δ 劈裂 ---
    fig3, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12, 5))
    fig3.suptitle(f"穿过 EP 点的 1D 切片 (cr = {cr_ep:.4f})", fontsize=13)

    cr_idx = np.argmin(np.abs(cr_range - cr_ep))
    ax_a.plot(
        ci_range, eig1_real[cr_idx, :], "b-o", markersize=3, label=r"Re($\lambda_1$)"
    )
    ax_a.plot(
        ci_range, eig2_real[cr_idx, :], "r-s", markersize=3, label=r"Re($\lambda_2$)"
    )
    ax_a.set_xlabel(r"Im($c_2$)/$c_0$")
    ax_a.set_ylabel(r"Re($\lambda$)")
    ax_a.set_title("本征值实部")
    ax_a.legend()
    ax_a.axvline(ci_ep, color="gray", linestyle="--", alpha=0.5, label="EP")

    ax_b.plot(
        ci_range, eig1_imag[cr_idx, :], "b-o", markersize=3, label=r"Im($\lambda_1$)"
    )
    ax_b.plot(
        ci_range, eig2_imag[cr_idx, :], "r-s", markersize=3, label=r"Im($\lambda_2$)"
    )
    ax_b.set_xlabel(r"Im($c_2$)/$c_0$")
    ax_b.set_ylabel(r"Im($\lambda$)")
    ax_b.set_title("本征值虚部")
    ax_b.legend()
    ax_b.axvline(ci_ep, color="gray", linestyle="--", alpha=0.5)

    fig3.tight_layout()
    visualize.export_exploration(fig3, plots_dir / "ep_slice_1d.png")

    # === 4. 保存数值结果 ===
    np.savez(
        session_dir / "results" / "riemann_scan.npz",
        cr_range=cr_range,
        ci_range=ci_range,
        eig1_real=eig1_real,
        eig1_imag=eig1_imag,
        eig2_real=eig2_real,
        eig2_imag=eig2_imag,
        s11_abs=s11_abs,
        s12_abs=s12_abs,
        s21_abs=s21_abs,
        s22_abs=s22_abs,
        cr_ep=cr_ep,
        ci_ep=ci_ep,
    )
    print(f"\n[done] 数值结果已保存至 {session_dir / 'results' / 'riemann_scan.npz'}")
    print(f"[done] 图表已保存至 {plots_dir}/")


if __name__ == "__main__":
    main()
