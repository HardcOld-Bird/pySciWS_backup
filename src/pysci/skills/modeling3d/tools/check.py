"""网格验收门：3D 打印交付前的物理正确性检查（trimesh）。

打印件 STL 在交付（切片）前必须通过本模块的检查——**这是打印管线的质量闸门**：

- watertight（流形闭合）：非闭合网格切片必失败或产出废件；
- winding consistent（法向一致）：内外翻转会导致切片器误判实体/空腔；
- 体积为正且与期望值一致（可选断言，相对容差）；
- 包围盒与期望尺寸一致（可选断言，绝对容差，毫米）；
- 连通分量数（多壳体报告：打印多件装配体时合法，单件时提示检查布尔运算残留）。

用法::

    from pysci.skills.modeling3d.tools.check import check_mesh

    report = check_mesh("part.stl", expect_bbox=(40.0, 20.0, 8.0))
    if not report.ok:
        ...
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import trimesh


@dataclass
class CheckReport:
    """一次网格验收的结果。``ok`` 为 False 时 ``failures`` 列出未通过项。"""

    path: Path
    ok: bool
    failures: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    def format(self) -> str:
        """人类可读的验收报告（中文，逐项 PASS/FAIL）。"""
        m = self.metrics
        lines = [
            f"=== mesh check: {self.path.name} ===",
            f"  faces            : {m.get('faces')}",
            f"  watertight       : {_mark('watertight', self)}",
            f"  winding_consist. : {_mark('winding', self)}",
            f"  volume_positive  : {_mark('volume_positive', self)}",
            f"  volume_mm3       : {m.get('volume'):.4f}"
            if m.get("volume") is not None
            else "  volume_mm3       : N/A",
            f"  bbox_mm          : ({m['bbox'][0]:.4f}, {m['bbox'][1]:.4f}, "
            f"{m['bbox'][2]:.4f})"
            if m.get("bbox")
            else "  bbox_mm          : N/A",
            f"  bodies           : {m.get('bodies')}",
            f"  euler_number     : {m.get('euler')}",
        ]
        if self.ok:
            lines.append("  RESULT           : PASS ✅（可交付切片）")
        else:
            lines.append("  RESULT           : FAIL ❌")
            for f_ in self.failures:
                lines.append(f"    - {f_}")
        return "\n".join(lines)


def _mark(key: str, report: CheckReport) -> str:
    v = report.metrics.get(key)
    if v is None:
        return "N/A"
    return "True  [PASS]" if v else "False [FAIL]"


def check_mesh(
    path: str | Path,
    *,
    expect_volume: float | None = None,
    volume_rtol: float = 0.01,
    expect_bbox: tuple[float, float, float] | None = None,
    bbox_atol: float = 0.1,
    require_watertight: bool = True,
) -> CheckReport:
    """加载网格并执行打印验收检查。

    Args:
        path: 网格文件（.stl / .glb / .obj / .ply 等 trimesh 可读格式）。
        expect_volume: 期望体积（mm³，STL 三角化会略低于 B-rep 解析体积，
            故默认给 1% 相对容差）。None → 跳过断言。
        volume_rtol: 体积相对容差。
        expect_bbox: 期望包围盒尺寸 (Lx, Ly, Lz)，毫米，按**排序后**逐维比较
            （STL 无方向语义，不假定轴向对应）。None → 跳过断言。
        bbox_atol: 包围盒绝对容差（mm）。
        require_watertight: 渲染用途网格可设 False（仅报告不判负）。

    Returns:
        CheckReport；``ok=False`` 时 CLI 以非零码退出。
    """
    p = Path(path)
    if not p.exists():
        return CheckReport(p, ok=False, failures=[f"文件不存在: {p}"])

    mesh = trimesh.load(str(p), force="mesh")
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        return CheckReport(p, ok=False, failures=["未加载到任何三角面片"])

    failures: list[str] = []
    vol = float(mesh.volume)
    bbox = tuple(float(x) for x in mesh.extents)

    if require_watertight and not mesh.is_watertight:
        failures.append(
            "网格非流形闭合（watertight=False）：切片器无法正确求交，"
            "检查布尔运算残留/自交，或在 CAD 脚本中重新导出"
        )
    if not mesh.is_winding_consistent:
        failures.append("法向缠绕不一致（winding_consistent=False）：存在翻转面片")
    if vol <= 0:
        failures.append(f"体积非正（{vol:.4f} mm³）：网格内外翻转或退化")

    if expect_volume is not None:
        if abs(vol - expect_volume) > abs(expect_volume) * volume_rtol:
            failures.append(
                f"体积 {vol:.4f} mm³ 与期望 {expect_volume:.4f} mm³ 偏差超过 "
                f"{volume_rtol:.1%}：几何与设计不符"
            )
    if expect_bbox is not None:
        got = sorted(bbox)
        want = sorted(expect_bbox)
        if any(abs(g - w) > bbox_atol for g, w in zip(got, want, strict=True)):
            failures.append(
                f"包围盒 ({got[0]:.3f}, {got[1]:.3f}, {got[2]:.3f}) 与期望 "
                f"({want[0]:.3f}, {want[1]:.3f}, {want[2]:.3f}) 偏差超过 "
                f"{bbox_atol} mm：尺寸不符"
            )

    metrics: dict[str, Any] = {
        "faces": len(mesh.faces),
        "watertight": bool(mesh.is_watertight),
        "winding": bool(mesh.is_winding_consistent),
        "volume_positive": vol > 0,
        "volume": vol,
        "bbox": bbox,
        "bodies": int(mesh.body_count),
        "euler": int(mesh.euler_number),
    }
    return CheckReport(p, ok=not failures, failures=failures, metrics=metrics)
