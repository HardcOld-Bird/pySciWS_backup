"""⑥3D 建模能力域薄胶水（用户裁决 B：纯 Blender 全链）。

bpy 无头工作流的四块自研面（其余一律社区/官方件，见技能文档
``.agents/skills/modeling-3d/SKILL.md``）：

- :mod:`.scaffold` — bpy 脚手架模板生成（脚本即模型：场景/单位/相机/灯光骨架）；
- :mod:`.acceptance` — 验收门红线（尺寸/流形/法线/面数预算，打印件与渲染件分开判据）；
- :mod:`.ledger` — 交付物冻结回脚本（产物↔脚本指纹对账，防"手改模型丢失复现"）；
- :mod:`.doctor` — Blender 路径 / Cycles CPU / 内存预算 / 单实例自检。

ADR（2026-11-17，用户裁决 B）：build123d / trimesh 一律不引入（STEP 需求出现再立项），
故验收门的网格度量（STL 流形/法线/体积/包围盒）以纯 Python 自实现（:mod:`.acceptance`
≈百行，唯一自研几何代码）——这是"社区优先"在裁决 B 约束下的显式豁免面，升级需求出现时
优先改判 trimesh。命名注记：代码包 ``modeling_3d``，数据区仍为 ``data/skills/modeling3d``
（:data:`pysci.paths.MODELING3D_ROOT`），勿互换。
"""

from __future__ import annotations

from pysci.skills.modeling_3d.acceptance import (
    GateResult,
    MeshMetrics,
    judge_print,
    judge_render,
    stl_metrics,
)
from pysci.skills.modeling_3d.doctor import find_blender
from pysci.skills.modeling_3d.ledger import freeze, parse_markers, reconcile
from pysci.skills.modeling_3d.scaffold import ensure_model_session, render_scaffold

__all__ = [
    "GateResult",
    "MeshMetrics",
    "ensure_model_session",
    "find_blender",
    "freeze",
    "judge_print",
    "judge_render",
    "parse_markers",
    "reconcile",
    "render_scaffold",
    "stl_metrics",
]
