# bpy 无头配方与坑点（⑥域）

脚本运行于 **Blender 自带 Python**（`blender -b -P <script>.py`），不是项目 .venv——
**不能 import pysci**，脚本必须自包含。**脚本即模型**：几何真源是版本化脚本文件，
交互探索（blender-mcp/GUI）只做试验，成果冻结回脚本重跑无头才算交付。

## 脚本形态：四段式（part）/ 影棚骨架（scene）

`scaffold.render_scaffold("part" | "scene", ...)` 生成的模板即规范形态：

1. **参数区**（mm）：每个尺寸一行、逐项注来源（图纸/文献/理论值）；④域理论值改动→
   参数+断言同步改。
2. **建模区**：bpy 算子 / bmesh 程序化建模；布尔、阵列、圆角都在这里。
3. **几何断言区**：`part.dimensions` 对照参数逐维 `assert`——参数一改必须复核，
   防"尺寸不符"静默出图。
4. **导出区**：导出 + `[model3d]` 标记行（见下）。scene 模板对应为：清场导入→
   归一化取景→材质/布光/相机→Cycles CPU 渲染。

## 单位契约（毫米）

```python
scene.unit_settings.system = "METRIC"
scene.unit_settings.scale_length = 0.001   # 1 BU = 1 mm
scene.unit_settings.length_unit = "MILLIMETERS"
```

打印件建模、尺寸断言、STL 导出全部按 mm 语义。导入别人的 GLB/STL 先核对尺度
（云生成件是任意尺度，见 print-vs-render 红线）。

## `[model3d]` 标记行协议（勿改格式）

脚本向 stdout 打 `print(f"[model3d] <key>: <value>")`，胶水 `ledger.parse_markers`
收集。约定键：`stl:` `obj:` `glb:` `render:`（产物路径）、`faces:`（面数申报）、
`volume_mm3:`（体积申报）。产物↔脚本指纹对账（`ledger.freeze`）以标记行收集的产物
清单为输入。

## 建模配方速查

```python
# 长方体（mm）：primitive_cube_add 的 size 是边长；非立方用 scale 后 apply
bpy.ops.mesh.primitive_cube_add(size=1.0)
part.scale = (L, W, H)
bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

# 圆柱 / 环 / 球
bpy.ops.mesh.primitive_cylinder_add(radius=r, depth=h, vertices=64)
bpy.ops.mesh.primitive_torus_add(major_radius=R, minor_radius=r)
bpy.ops.mesh.primitive_uv_sphere_add(radius=r)

# 布尔（差集：在 part 上减去 cutter）
mod = part.modifiers.new("cut", "BOOLEAN")
mod.operation = "DIFFERENCE"
mod.object = cutter
bpy.context.view_layer.objects.active = part
bpy.ops.object.modifier_apply(modifier=mod.name)

# 阵列（沿 X 复制 n 份，间距 d）
mod = part.modifiers.new("arr", "ARRAY")
mod.count = n
mod.use_relative_offset = False
mod.use_constant_offset = True
mod.constant_offset_displace = (d, 0, 0)

# 圆角 / 倒角（先选边）
mod = part.modifiers.new("bev", "BEVEL")
mod.width = 1.0
mod.segments = 3

# 程序化点面建模
import bmesh
bm = bmesh.new()
bmesh.ops.create_cube(bm, size=1.0)
bm.to_mesh(part.data)
bm.free()
```

打印件几何红线速记：布尔后必查残留（验收门 `watertight` 判）、壁厚留 ≥ 喷嘴/工艺
下限、法线朝外（`winding_consistent` 判）。退化面（零面积三角）常见于布尔残留与
共面焊接——门会拦。

## 导入/导出算子改名坑（版本间漂移，胶水模板已带回退链）

| 用途 | Blender ≥ 4.x | 旧版（≤ 3.6） |
|---|---|---|
| STL 导出 | `bpy.ops.wm.stl_export(filepath=...)` | `bpy.ops.export_mesh.stl(...)` |
| OBJ 导出 | `bpy.ops.wm.obj_export(...)` | `bpy.ops.export_scene.obj(...)` |
| STL 导入 | `bpy.ops.wm.stl_import(...)` | `bpy.ops.import_mesh.stl(...)` |
| OBJ 导入 | `bpy.ops.wm.obj_import(...)` | `bpy.ops.import_scene.obj(...)` |
| GLB 导入/导出 | `bpy.ops.import_scene.gltf` / `bpy.ops.export_scene.gltf(export_format="GLB")` | 同左（稳定） |

写法：`try: 新算子 except (AttributeError, TypeError): 旧算子`。`AttributeError` 在
访问不存在的算子名时就抛，两个包在同一个 try 里逐条试。

## 其他坑点

- **平滑着色改名**：`bpy.ops.object.shade_auto_smooth(angle=...)`（4.1+）/
  `shade_smooth_by_angle`（4.0）/ `shade_smooth`（旧）——全平滑会让平面"融蜡"，
  优先按角度平滑。三名回退链。
- **world.use_nodes**：5.x 默认已启用；直接赋值 `world.use_nodes = True` 在 6.0 触发
  弃用警告——先 `if not world.use_nodes:` 再赋值。
- **transform_apply 前提**：目标对象须是 active + selected；primitive_* 算子新建后
  自动满足，别在 select_all(DESELECT) 之后忘了选。
- **维度断言与 scale**：`obj.scale` 未 apply 时 `obj.dimensions` 已含缩放，但导出的
  网格顶点是 scale 后值——**断言前 apply**，让脚本内度量与导出件一致。
- **bmesh 体积**：`bm.calc_volume(signed=False)` 出 mm³（本单位契约下），用于
  `volume_mm3` 标记与体积断言。
- **随机性/时间戳**：脚本内禁用未固定种子的随机；渲染时间戳不进几何，保证同脚本
  重跑产物逐位可复现（对账 sha256 才有意义）。

## blender-mcp（交互探索通道）使用规则

blender-mcp 是 agent↔Blender 的**唯一**交互通道（豁免 pin，ADR 见选型调研）：
只用于探索/预览/试材质布光；**一切值得交付的成果冻结回 scene 脚本**，`blender -b -P`
重跑无头出正式产物——MCP 与无头不是重复实现，是"试验态 vs 交付态"两态。
