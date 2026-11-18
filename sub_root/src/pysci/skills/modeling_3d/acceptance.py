"""验收门红线——交付前的物理/工程正确性判定（⑥域）。

**验收门是唯一交付路径**：打印件 STL 不过门不得切片，渲染件/展示件不过门不得入图。
打印件与渲染件**分开判据**（红线之一，细则见
``.agents/skills/modeling-3d/references/acceptance-gate.md``）：

============  ===================  ==========  ==========  ==========
判据          打印件（print）       渲染件（render）         说明
============  ===================  ==========  ==========  ==========
尺寸          必判（expect 断言）   仅 sanity 警告           mm，排序后逐维
流形/闭合     必判（红线）          不判（开面合法）         watertight
法线一致性    必判（红线）          不判（渲染可双面）       winding
面数预算      判（切片负担）        必判（渲染性能）         max_faces
退化面片      判（红线）            仅警告（渲染瑕疵）       零面积三角
============  ===================  ==========  ==========  ==========

网格度量 :func:`stl_metrics` 为**纯 Python** STL 解析（二进制 + ASCII）——
用户裁决 B（纯 Blender）下 trimesh 不引入的显式豁免自研面（模块 ADR 见包
``__init__``）。GLB/OBJ 渲染件的面数由导出脚本的 ``[model3d]`` 标记行申报
（:func:`metrics_from_markers`），不重复实现网格解析。
"""

from __future__ import annotations

import struct
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class MeshMetrics:
    """一次网格度量（打印件实测 / 渲染件可由标记行申报）。

    ``watertight`` / ``winding_consistent`` / ``bbox_mm`` 为 ``None`` 表示"未度量"：
    打印判据把缺失当失败（红线不许空过），渲染判据不使用这些字段。
    """

    faces: int
    verts: int = 0
    bbox_mm: tuple[float, float, float] | None = None
    volume_mm3: float | None = None
    watertight: bool | None = None
    winding_consistent: bool | None = None
    degenerate_faces: int = 0


@dataclass
class GateResult:
    """验收门判定结果。``criteria`` 为 ``"print"`` 或 ``"render"``。"""

    criteria: str
    ok: bool
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metrics: MeshMetrics | None = None

    def format(self) -> str:
        """人类可读验收报告（中文，逐项 PASS/FAIL/WARN）。"""
        m = self.metrics
        lines = [f"=== 验收门（{self.criteria} 判据）==="]
        if m is not None:
            lines.append(f"  faces            : {m.faces}")
            lines.append(f"  verts            : {m.verts}")
            lines.append(f"  watertight       : {m.watertight}")
            lines.append(f"  winding_consist. : {m.winding_consistent}")
            lines.append(f"  degenerate_faces : {m.degenerate_faces}")
            if m.bbox_mm is not None:
                b = m.bbox_mm
                lines.append(f"  bbox_mm          : ({b[0]:.4f}, {b[1]:.4f}, {b[2]:.4f})")
            if m.volume_mm3 is not None:
                lines.append(f"  volume_mm3       : {m.volume_mm3:.4f}")
        lines.append("  RESULT           : PASS ✅" if self.ok else "  RESULT           : FAIL ❌")
        for f_ in self.failures:
            lines.append(f"    - {f_}")
        for w in self.warnings:
            lines.append(f"    ~ {w}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# STL 度量（纯 Python：二进制 + ASCII）
# ---------------------------------------------------------------------------


def stl_metrics(path: str | Path) -> MeshMetrics:
    """实测 STL 网格度量：面数/顶点数/包围盒/体积/流形/法线/退化面。

    流形判据：每条无向边恰被 2 个三角面片共享（watertight）；法线判据：每条有向边
    恰出现 1 次（winding consistent）。顶点按坐标精确焊接（Blender 导出的 STL
    共享顶点坐标逐位相同）。

    Args:
        path: ``.stl`` 文件（二进制或 ASCII）。

    Returns:
        :class:`MeshMetrics`。

    Raises:
        ValueError: 文件为空、不含三角面片或不是合法 STL。
        OSError: 文件不可读。
    """
    raw = Path(path).read_bytes()
    return _metrics_from_triangles(_read_stl(raw, Path(path)))


def _read_stl(raw: bytes, name: Path) -> list[tuple[float, ...]]:
    """解析 STL 为三角面片列表（每项 9 个 float）。"""
    if len(raw) >= 84:
        (count,) = struct.unpack_from("<I", raw, 80)
        if len(raw) == 84 + 50 * count:
            tris = []
            off = 84
            for _ in range(count):
                vals = struct.unpack_from("<12f", raw, off)  # 法线 3 + 顶点 9
                tris.append(vals[3:12])
                off += 50
            return tris
    if raw.lstrip()[:5].lower() == b"solid":
        return _read_ascii_stl(raw.decode("utf-8", "replace"))
    raise ValueError(f"无法解析 STL（既非二进制也非 ASCII）: {name}")


def _read_ascii_stl(text: str) -> list[tuple[float, ...]]:
    """解析 ASCII STL 的 ``vertex`` 行（每 3 个顶点成一面）。"""
    tris: list[tuple[float, ...]] = []
    verts: list[float] = []
    for line in text.splitlines():
        parts = line.split()
        if parts[:1] == ["vertex"] and len(parts) == 4:
            verts.extend(float(v) for v in parts[1:4])
            if len(verts) == 9:
                tris.append(tuple(verts))
                verts = []
    return tris


def _metrics_from_triangles(tris: list[tuple[float, ...]]) -> MeshMetrics:
    """由三角面片列表计算度量（纯函数，无 I/O）。"""
    if not tris:
        raise ValueError("STL 未包含任何三角面片")
    weld: dict[tuple[float, float, float], int] = {}
    undirected: Counter[tuple[int, int]] = Counter()
    directed: Counter[tuple[int, int]] = Counter()
    volume = 0.0
    degenerate = 0
    for tri in tris:
        vs = [tuple(tri[i : i + 3]) for i in (0, 3, 6)]
        ids = []
        for v in vs:
            ids.append(weld.setdefault(v, len(weld)))
        # 退化面片：重复顶点或零面积（两次叉积判）
        ux = (vs[1][0] - vs[0][0], vs[1][1] - vs[0][1], vs[1][2] - vs[0][2])
        wx = (vs[2][0] - vs[0][0], vs[2][1] - vs[0][1], vs[2][2] - vs[0][2])
        cx = (
            ux[1] * wx[2] - ux[2] * wx[1],
            ux[2] * wx[0] - ux[0] * wx[2],
            ux[0] * wx[1] - ux[1] * wx[0],
        )
        if len(set(ids)) < 3 or cx[0] ** 2 + cx[1] ** 2 + cx[2] ** 2 < 1e-20:
            degenerate += 1
        v0, v1, v2 = vs
        volume += (
            v0[0] * (v1[1] * v2[2] - v1[2] * v2[1])
            - v0[1] * (v1[0] * v2[2] - v1[2] * v2[0])
            + v0[2] * (v1[0] * v2[1] - v1[1] * v2[0])
        ) / 6.0
        for a, b in ((ids[0], ids[1]), (ids[1], ids[2]), (ids[2], ids[0])):
            directed[(a, b)] += 1
            undirected[(min(a, b), max(a, b))] += 1
    xs = [v[0] for v in weld]
    ys = [v[1] for v in weld]
    zs = [v[2] for v in weld]
    bbox = (max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs))
    return MeshMetrics(
        faces=len(tris),
        verts=len(weld),
        bbox_mm=bbox,
        volume_mm3=volume,
        watertight=all(c == 2 for c in undirected.values()),
        winding_consistent=all(c == 1 for c in directed.values()),
        degenerate_faces=degenerate,
    )


def metrics_from_markers(markers: Mapping[str, str]) -> MeshMetrics:
    """由 ``[model3d]`` 标记行申报值构造渲染件度量（GLB/OBJ 不重复解析）。

    仅 ``faces`` / ``volume_mm3`` 有实值；流形/法线字段留 ``None``（渲染判据不用）。
    """
    return MeshMetrics(
        faces=int(float(markers.get("faces", "0"))),
        volume_mm3=float(markers.get("volume_mm3", "0") or 0),
    )


# ---------------------------------------------------------------------------
# 判定（打印件 / 渲染件分开）
# ---------------------------------------------------------------------------


def judge_print(
    metrics: MeshMetrics,
    *,
    expect_bbox_mm: tuple[float, float, float] | None = None,
    bbox_atol_mm: float = 0.1,
    expect_volume_mm3: float | None = None,
    volume_rtol: float = 0.01,
    max_faces: int = 400_000,
) -> GateResult:
    """打印件判据（严格）：流形/法线/尺寸/面数预算，一红即拒。

    Args:
        metrics: :func:`stl_metrics` 实测结果（渲染件申报值在此不适用）。
        expect_bbox_mm: 期望包围盒尺寸 (Lx, Ly, Lz)，mm，**排序后**逐维比较
            （STL 无方向语义）。None → 尺寸断言缺失，记警告（打印件建议必给）。
        bbox_atol_mm: 包围盒绝对容差（mm）。
        expect_volume_mm3: 期望体积（mm³）；None → 跳过。
        volume_rtol: 体积相对容差（三角化会略低于设计值，默认 1%）。
        max_faces: 面数预算（切片负担/打印机分辨率之上无意义）。

    Returns:
        :class:`GateResult`（``criteria="print"``）。
    """
    failures: list[str] = []
    warnings: list[str] = []
    if metrics.watertight is None or metrics.winding_consistent is None:
        failures.append("流形/法线度量缺失：打印件必须以 stl_metrics 实测，禁止空过验收门")
    else:
        if not metrics.watertight:
            failures.append(
                "网格非流形闭合（watertight=False）：切片器无法正确求交；"
                "查布尔残留/自交，在 bpy 脚本里修几何后重导出（勿手改网格）"
            )
        if not metrics.winding_consistent:
            failures.append("法线缠绕不一致（winding_consistent=False）：存在翻转面片，内外会误判")
    if metrics.degenerate_faces:
        failures.append(f"退化面片 ×{metrics.degenerate_faces}（零面积/重复顶点）：打印几何非法")
    if metrics.volume_mm3 is None:
        failures.append("体积度量缺失")
    elif metrics.volume_mm3 <= 0:
        failures.append(f"体积非正（{metrics.volume_mm3:.4f} mm³）：网格内外翻转或退化")
    if expect_bbox_mm is None:
        warnings.append("未给 expect_bbox_mm：尺寸断言缺失（打印件建议必给，防静默出图）")
    elif metrics.bbox_mm is None:
        failures.append("包围盒度量缺失")
    else:
        got = sorted(metrics.bbox_mm)
        want = sorted(expect_bbox_mm)
        if any(abs(g - w) > bbox_atol_mm for g, w in zip(got, want, strict=True)):
            failures.append(
                f"包围盒 ({got[0]:.3f}, {got[1]:.3f}, {got[2]:.3f}) 与期望 "
                f"({want[0]:.3f}, {want[1]:.3f}, {want[2]:.3f}) 偏差超过 {bbox_atol_mm} mm：尺寸不符"
            )
    if (
        expect_volume_mm3 is not None
        and metrics.volume_mm3 is not None
        and abs(metrics.volume_mm3 - expect_volume_mm3)
        > abs(expect_volume_mm3) * volume_rtol
    ):
        failures.append(
            f"体积 {metrics.volume_mm3:.4f} mm³ 与期望 {expect_volume_mm3:.4f} mm³ "
            f"偏差超过 {volume_rtol:.1%}：几何与设计不符"
        )
    if metrics.faces > max_faces:
        failures.append(f"面数 {metrics.faces} 超预算 {max_faces}：切片负担过重，降网格精度重导出")
    return GateResult("print", not failures, failures, warnings, metrics)


def judge_render(
    metrics: MeshMetrics,
    *,
    max_faces: int = 200_000,
    expect_bbox_mm: tuple[float, float, float] | None = None,
    bbox_atol_mm: float = 0.5,
) -> GateResult:
    """渲染件/展示件判据（宽）：开面合法、不判流形/法线/体积；红线=面数预算。

    Args:
        metrics: 度量（:func:`stl_metrics` 或 :func:`metrics_from_markers`）。
        max_faces: 渲染面数预算（Cycles CPU 内存 ~5GB 约束下的性能红线）。
        expect_bbox_mm: 可选尺寸 sanity（渲染件常为任意尺度，仅警告不判负）。
        bbox_atol_mm: sanity 绝对容差（mm；渲染件按导入尺度解释）。

    Returns:
        :class:`GateResult`（``criteria="render"``）。
    """
    failures: list[str] = []
    warnings: list[str] = []
    if metrics.faces > max_faces:
        failures.append(f"面数 {metrics.faces} 超渲染预算 {max_faces}：Cycles CPU 内存/耗时不可控，先减面")
    if metrics.degenerate_faces:
        warnings.append(f"退化面片 ×{metrics.degenerate_faces}：可能渲染出黑斑/漏光")
    if metrics.watertight is False:
        warnings.append("网格非闭合：渲染合法（开面），但不可直接切片打印（见 print-vs-render 红线）")
    if expect_bbox_mm is not None and metrics.bbox_mm is not None:
        got = sorted(metrics.bbox_mm)
        want = sorted(expect_bbox_mm)
        if any(abs(g - w) > bbox_atol_mm for g, w in zip(got, want, strict=True)):
            warnings.append(
                f"包围盒 ({got[0]:.3f}, {got[1]:.3f}, {got[2]:.3f}) 与参考 "
                f"({want[0]:.3f}, {want[1]:.3f}, {want[2]:.3f}) 偏差超过 {bbox_atol_mm} mm（仅提示）"
            )
    return GateResult("render", not failures, failures, warnings, metrics)
