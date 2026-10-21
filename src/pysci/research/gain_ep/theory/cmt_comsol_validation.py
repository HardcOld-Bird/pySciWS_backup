"""CMT 反射 S 矩阵的 COMSOL 独立验证封装（三表层 API 之三）。

**何时用哪个表层 API**：
- :func:`~pysci.research.gain_ep.theory.cmt_reflection_s_matrix.compute_s_matrix`：
  日常快速求值（毫秒级），适合参数扫描/论文绘图；
- :func:`~pysci.research.gain_ep.theory.inverse_design.inverse_design`：
  CMT 目标函数上的逆向设计（不需 FEM）；
- :func:`validate_s_matrix`：高价值结果的 COMSOL 独立复核（分钟级、需 license）。

输入与 :func:`compute_s_matrix` 完全相同（同一个 ``MetagratingParams``）；输出为
同一语义的 |M|×|M| 复 S 矩阵：端口按 k_x 降序、``S[p, q]`` 为端口 q 入射时端口 p 的
反射系数，端口集合与 :func:`solve_s_matrix` 的 ``Port`` 元数据一致。内部按有效槽数
J 智能路由 t4 模板（J=2..6），单 JVM 会话内完成参数注入、几何/网格重建、逐端口求解
与全传播阶次 Fourier 提取；不暴露 mph 细节。

坐标/约定（t4 标定）：模板 bpf1/bpf2 分别为 +|θ|/−|θ| 入射，反射扫描线 L=scan1、
R=scan2；散射场 x 相位为 ``e^{-i k_x x}``，故按**参考端口栅格**用 ``e^{+i k_x^m x}``
做 Fourier 积分可直接得到参考端口阶次 m 的系数（不再使用入射栅格，避免 t4 的
阶次翻转）。COMSOL 相位参考与 CMT 相差一个与阶次/扫描面有关的固定相位；**验收以
幅度为准**（无损锚点主系数 4 位、小量约 3 位，沿用 t4 口径）。

限制：模板仅覆盖有效槽数 J∈{2,3,4,5,6}；J 超出抛 ``MetagratingParameterError``
（扩模板策略：按 ``build_cmt_J2toJ6_templates.py`` 的 sliver 切换方式新增并接线）。
背景声速须为 343 m/s（模板材料 ``cs(T)``/常数 343 的标定口径）。
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from pysci.paths import research_asset_dir
from pysci.research.gain_ep.theory.cmt_reflection_s_matrix import (
    MetagratingParameterError,
    MetagratingParams,
    solve_s_matrix,
)

__all__ = [
    "compare_s_matrix",
    "validate_s_matrix",
    "validate_s_matrix_batch",
]

_SLIVER = 1e-6  # t4 模板的 sliver 槽判据/尺寸 (m)
_TEMPLATE_SET = (2, 3, 4, 5, 6)
_C0_TEMPLATE = 343.0


def _template_path(J: int, template_dir: Path | None) -> Path:
    base = (
        Path(template_dir)
        if template_dir is not None
        else research_asset_dir("gain_ep") / "simulation"
    )
    return base / f"12 反射管槽_J{J}模板.mph"


def _active_grooves(params: MetagratingParams) -> int:
    """按 t4 路由判据返回有效槽数：w_j>1µm 且 h_j>1µm 的前导槽数。"""
    active = (np.asarray(params.w) > _SLIVER) & (np.asarray(params.h) > _SLIVER)
    count = int(active.sum())
    if count == 0 or not bool(np.all(active[:count])):
        raise MetagratingParameterError(
            "有效槽必须为前导槽（t4 模板拓扑）：存在 w/h≤1µm 的槽位于有效槽之前"
        )
    if count not in _TEMPLATE_SET:
        raise MetagratingParameterError(
            f"COMSOL 模板集仅覆盖有效槽数 J∈{_TEMPLATE_SET}，得到 J={count}；"
            "扩模板策略见模块 docstring"
        )
    if abs(params.c0 - _C0_TEMPLATE) > 1e-9:
        raise MetagratingParameterError(
            f"COMSOL 模板按背景声速 343 m/s 标定，params.c0={params.c0} 不受支持"
        )
    return count


class _Runner:
    """单个 JVM 会话内的模板缓存与逐参数验证。"""

    def __init__(self, client, template_dir: Path | None = None) -> None:
        self.client = client
        self.template_dir = template_dir
        self.models: dict[int, object] = {}
        self.tmp = Path(tempfile.mkdtemp(prefix="cmt_validate_"))

    def model(self, J: int):
        if J not in self.models:
            self.models[J] = self.client.load(str(_template_path(J, self.template_dir)))
        return self.models[J]

    def _set_slots(self, jm, params: MetagratingParams, J: int, n_slots: int) -> None:
        w = np.asarray(params.w, dtype=float)
        h = np.asarray(params.h, dtype=float)
        x = np.asarray(params.groove_positions, dtype=float)
        c = np.asarray(params.c_complex, dtype=complex)
        for j in range(n_slots):
            idx = j + 1
            if j < J:
                jm.param().set(f"w{idx}", f"{w[j]:.9g}[m]")
                jm.param().set(f"h{idx}", f"{h[j]:.9g}[m]")
                jm.param().set(f"x{idx}", f"{x[j]:.9g}[m]")
                jm.param().set(f"cr{idx}", f"{c[j].real / params.c0:.9g}")
                jm.param().set(f"ci{idx}", f"{c[j].imag / params.c0:.9g}")
            else:
                jm.param().set(f"w{idx}", f"{_SLIVER:.9g}[m]")
                jm.param().set(f"h{idx}", f"{_SLIVER:.9g}[m]")
                jm.param().set(f"cr{idx}", "1")
                jm.param().set(f"ci{idx}", "0")

    def _export_scan(self, model, line: str, tag: str) -> NDArray:
        from pysci.skills.comsol_simulation.tools.export import export_data

        cols = []
        for expr, suf in (("real(acpr.p_s)", "re"), ("imag(acpr.p_s)", "im")):
            path = self.tmp / f"{tag}_{suf}.csv"
            if not export_data(model, line, path, expr=expr).ok:
                raise RuntimeError(f"COMSOL 扫描线导出失败：{line}/{expr}")
            cols.append(np.loadtxt(path, delimiter=",", comments="%"))
        x = cols[0][:, 0]
        ps = cols[0][:, -1] + 1j * cols[1][:, -1]
        return np.asarray(x), np.asarray(ps)

    def validate(self, params: MetagratingParams) -> NDArray:
        J = _active_grooves(params)
        sm = solve_s_matrix(params)  # CMT 端口语义（端口集合/顺序与 S 形状）
        ports = sm.ports
        model = self.model(J)
        jm = model.java
        comp = jm.component("comp1")
        ph = comp.physics("acpr")

        jm.param().set("D", f"{params.period:.9g}[m]")
        jm.param().set("f", f"{params.f0:.9g}[1/s]")
        self._set_slots(jm, params, J, 3 if J == 3 else 6)
        comp.geom("geom1").run()
        comp.mesh("mesh1").run()

        kxs = np.array([port.kx for port in ports], dtype=float)
        S = np.empty((len(ports), len(ports)), dtype=complex)
        for q, port_q in enumerate(ports):
            side = "L" if port_q.angle_deg > 0 else "R"
            jm.param().set("theta", f"{abs(port_q.angle_deg):.9g}*pi/180")
            ph.feature("bpf1").set("pamp", "1" if side == "L" else "0")
            ph.feature("bpf2").set("pamp", "0" if side == "L" else "1")
            jm.study("std1").run()
            line = "scan1" if side == "L" else "scan2"
            x, ps = self._export_scan(model, line, f"{side}_{q}")
            for p, kx in enumerate(kxs):
                # 散射场 x 相位 e^{-i kx x}：用 +i 参考端口栅格直接提取端口阶次。
                S[p, q] = np.trapezoid(ps * np.exp(1j * kx * x), x) / (
                    2.0 * params.period
                )
        return S


def _start_client(cores: int):
    import mph

    return mph.start(cores=cores)


def validate_s_matrix(
    params: MetagratingParams,
    *,
    client=None,
    cores: int = 4,
    template_dir: str | Path | None = None,
) -> NDArray:
    """用 COMSOL 专用 mph 模板独立计算与 ``compute_s_matrix`` 同语义的 S 矩阵。

    输入/输出与 :func:`compute_s_matrix` 对齐：同一 ``MetagratingParams``、同一
    |M|×|M| 端口顺序（k_x 降序）与 S[p,q] 语义。内部按有效槽数 J 路由
    ``12 反射管槽_J{J}模板.mph``（J=2..6），在单 JVM 会话内完成参数注入、
    几何/网格重建、逐端口（θ_q 入射，L/R 扫描线）求解与全传播阶次 Fourier 提取。

    Args:
        params: 超表面参数（与 ``compute_s_matrix`` 相同；要求 c0=343 m/s，
            有效槽为前导槽且 J∈{2,3,4,5,6}）。
        client: 可选的已启动 mph client（批扫/测试复用一次 JVM）；省略时本函数
            自行启动并在结束时断开。
        cores: 自行启动 COMSOL 时的核数。
        template_dir: 模板目录覆盖（默认 ``data/research/1_gain_ep/simulation``）。

    Returns:
        与 ``compute_s_matrix`` 同形状/端口语义的复 S 矩阵。幅度为验收口径；
        COMSOL 相位参考与 CMT 相差与阶次/扫描面有关的固定相位（见模块 docstring）。
    """
    _active_grooves(params)  # 先做路由/参数校验，无 COMSOL 时也能给出明确错误
    started = client is None
    if started:
        client = _start_client(cores)
    try:
        return _Runner(client, Path(template_dir) if template_dir else None).validate(
            params
        )
    finally:
        if started:
            client.disconnect()


def validate_s_matrix_batch(
    params_list: list[MetagratingParams],
    *,
    client=None,
    cores: int = 4,
    template_dir: str | Path | None = None,
) -> list[NDArray]:
    """批量 COMSOL 验证（一次 JVM 会话、按 J 复用已加载模板）。

    Args:
        params_list: 超表面参数序列（每项语义同 :func:`validate_s_matrix`）。
        client: 可选的已启动 mph client。
        cores: 自行启动 COMSOL 时的核数。
        template_dir: 模板目录覆盖。

    Returns:
        与输入等长的复 S 矩阵列表；每项端口数可不同。
    """
    for params in params_list:
        _active_grooves(params)  # 先校验全部参数再启动 license
    started = client is None
    if started:
        client = _start_client(cores)
    try:
        runner = _Runner(client, Path(template_dir) if template_dir else None)
        return [runner.validate(params) for params in params_list]
    finally:
        if started:
            client.disconnect()


def compare_s_matrix(cmt_S: NDArray, comsol_S: NDArray) -> dict:
    """按幅度对照 CMT 与 COMSOL 的 S 矩阵（验收口径）。

    Args:
        cmt_S: CMT 的 |M|×|M| 复 S。
        comsol_S: COMSOL 验证的同形状复 S。

    Returns:
        含 ``max_abs_diff``、``max_rel_diff``（逐元，|CMT|>1e-3 的最小相对误差）、
        ``rms_diff`` 与两矩阵幅度列表的字典。
    """
    cmt = np.asarray(cmt_S, dtype=complex)
    com = np.asarray(comsol_S, dtype=complex)
    if cmt.shape != com.shape:
        raise MetagratingParameterError(
            f"S 形状不一致：CMT {cmt.shape} vs COMSOL {com.shape}"
        )
    a, b = np.abs(cmt), np.abs(com)
    diff = np.abs(a - b)
    mask = a > 1e-3
    rel = float(np.max(diff[mask] / a[mask])) if np.any(mask) else 0.0
    return {
        "max_abs_diff": float(np.max(diff)),
        "max_rel_diff": rel,
        "rms_diff": float(np.sqrt(np.mean(diff**2))),
        "cmt_abs": a.tolist(),
        "comsol_abs": b.tolist(),
    }
