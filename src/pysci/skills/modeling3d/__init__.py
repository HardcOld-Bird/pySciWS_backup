"""3D 建模技能（modeling3d）：双管线设计——打印件走参数化 CAD，概念图走 Blender。

定位
----
本技能覆盖两类互不混用的管线：

1. **3D 打印管线**（几何精确优先）：build123d 参数化脚本（毫米单位，B-rep 内核，
   流形闭合由构造保证）→ STL/STEP 导出 → trimesh 验收门（watertight/体积/包围盒）。
   **完全不经过 Blender**——网格建模器无法提供尺寸约束体系。
2. **概念渲染管线**（美观优先）：模型来源三选一（打印管线 STL / Tripo 图生 3D /
   Blender 脚本建模）→ Blender 场景（材质/灯光/相机）→ headless 渲染出图。

设计要点
--------
- **交付物必须出自版本化脚本**：``blender -b -P`` headless 执行（与 comsol_simulation
  同构，可复现）；blender-mcp（addon + MCP server，社区维护）仅作交互式探索/预览通道，
  探索成果须固化为脚本方可产出交付物——两通道因此不构成能力重叠。
- **自研只有胶水**：几何内核（build123d/OCCT）、网格校验（trimesh）、渲染（Blender）、
  图生 3D（Tripo 云 API）全部委托社区/商业工具；本包只做脚手架、验收门与 CLI 门面。
- CLI 入口 ``pysci-model3d``：doctor / new / build / check / render / tripo / list。
- 产物落盘：研究线会话在 ``data/research/<n>_<name>/models/<slug>/``
  （见 :func:`pysci.paths.research_model_dir`）；技能级模板/配方/缓存在
  ``data/skills/modeling3d/``（见 :data:`pysci.paths.MODELING3D_ROOT`）。
"""
