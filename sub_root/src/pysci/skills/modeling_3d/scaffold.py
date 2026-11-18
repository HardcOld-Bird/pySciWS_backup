"""bpy 脚手架模板生成——"脚本即模型"的起点（⑥域）。

生成**自包含**的 Blender 无头脚本文本（运行于 Blender 自带 Python，不 import pysci）：

- ``kind="part"``：3D 打印件四段式（①设计参数含来源注记 → ②建模 → ③几何断言 →
  ④导出 + ``[model3d]`` 标记行协议）；
- ``kind="scene"``：概念渲染场景（清场/导入 → 归一化取景 → 材质/三点布光/相机骨架 →
  **Cycles CPU** 低采样单帧渲染）。

单位契约：毫米（``scale_length=0.001``，1 BU = 1 mm）。模板以 ``%TOKEN%`` 占位符替换
生成——生成代码里的 f-string 花括号不受影响。

ADR（2026-11-17）：模板是**生成的起点**而非库——建模几何由用户/agent 在脚本内改写，
脚本文件本身就是模型真源；本模块不试图参数化一切。会话目录规范沿用旧域形态
（``stl/`` ``renders/`` ``notes.md``），路径走 :mod:`pysci.paths`（无 ``parents[N]``）。
"""

from __future__ import annotations

from pathlib import Path

from pysci.paths import assert_within_data, research_model_dir

#: 脚手架支持的脚本种类。
SCAFFOLD_KINDS = ("part", "scene")

_PART_TEMPLATE = '''\
"""%TITLE% — 3D 打印件参数化建模脚本（bpy 无头；脚本即模型）。

单位：毫米（mm）。所有尺寸必须与设计依据一致；改参数后同步复核 §3 几何断言。
执行：blender -b -P 本脚本（Blender 自带 Python，不能 import pysci）。
交付：stl/%SLUG%.stl 过 acceptance 验收门（流形/法线/尺寸/面数预算）后方可切片打印。
"""

from pathlib import Path

import bmesh
import bpy

SESSION_DIR = Path(r"%SESSION_DIR%")
SLUG = "%SLUG%"
STL_OUT = SESSION_DIR / "stl" / (SLUG + ".stl")
OBJ_OUT = SESSION_DIR / "stl" / (SLUG + ".obj")
GLB_OUT = SESSION_DIR / "renders" / (SLUG + ".glb")

# %% === 1. 设计参数（mm；来源注记逐项写明） ===
L = %L%  # 长（X）— 来源：<设计依据/图纸/文献>
W = %W%  # 宽（Y）— 来源：<设计依据/图纸/文献>
H = %H%  # 高（Z）— 来源：<设计依据/图纸/文献>

# %% === 2. 场景/单位 + 建模（交付物必须能由本脚本重跑复现） ===
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.unit_settings.system = "METRIC"
scene.unit_settings.scale_length = 0.001  # 1 BU = 1 mm
scene.unit_settings.length_unit = "MILLIMETERS"

bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, 0.0))
part = bpy.context.active_object
part.name = SLUG
part.scale = (L, W, H)  # TODO: 替换为实际几何（布尔/阵列/圆角配方见 references/bpy-recipes.md）
bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

# %% === 3. 几何断言（物理正确性守门；参数一改这里必须跟着复核） ===
dims = part.dimensions
assert abs(dims.x - L) < 1e-4, f"X 尺寸不符: {dims.x}"
assert abs(dims.y - W) < 1e-4, f"Y 尺寸不符: {dims.y}"
assert abs(dims.z - H) < 1e-4, f"Z 尺寸不符: {dims.z}"
# 期望包围盒（验收门 judge_print(expect_bbox_mm=...) 同源）：%EXPECT_BBOX%

# %% === 4. 导出 + 标记行协议（[model3d] 协议勿改格式） ===
STL_OUT.parent.mkdir(parents=True, exist_ok=True)
GLB_OUT.parent.mkdir(parents=True, exist_ok=True)
try:
    bpy.ops.wm.stl_export(filepath=str(STL_OUT))
except (AttributeError, TypeError):  # Blender < 4.0 旧算子名
    bpy.ops.export_mesh.stl(filepath=str(STL_OUT))
try:
    bpy.ops.wm.obj_export(filepath=str(OBJ_OUT))
except (AttributeError, TypeError):
    bpy.ops.export_scene.obj(filepath=str(OBJ_OUT))
bpy.ops.export_scene.gltf(filepath=str(GLB_OUT), export_format="GLB")

bm = bmesh.new()
bm.from_mesh(part.data)
volume_mm3 = bm.calc_volume(signed=False)
bm.free()
print(f"[model3d] stl: {STL_OUT}")
print(f"[model3d] obj: {OBJ_OUT}")
print(f"[model3d] glb: {GLB_OUT}")
print(f"[model3d] faces: {len(part.data.polygons)}")
print(f"[model3d] volume_mm3: {volume_mm3:.4f}")
'''

_SCENE_TEMPLATE = '''\
"""%TITLE% — Blender 概念渲染场景脚本（bpy 无头；脚本即模型）。

运行于 Blender 自带 Python（不能 import pysci）。交互探索（blender-mcp）的成果
必须冻结回本脚本重跑无头——交付物只认版本化脚本产物。
执行：blender -b -P 本脚本
渲染纪律：Cycles CPU 单帧低采样（EEVEE 需 GPU 本机不可用；内存 ~5GB 约束禁并行）。
"""

from pathlib import Path
import math

import bpy
from mathutils import Vector

SESSION_DIR = Path(r"%SESSION_DIR%")
MESH_IN = %MESH%  # 模型来源：打印管线 STL / Tripo GLB / 本脚本建模
OUT = Path(r"%RENDER_OUT%")

# %% === 1. 清场 + 导入 ===
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.resolution_x = 1280  # 先定画幅：取景按其纵横比计算
scene.render.resolution_y = 960
if MESH_IN is not None and MESH_IN.exists():
    ext = MESH_IN.suffix.lower()
    if ext == ".stl":
        try:
            bpy.ops.wm.stl_import(filepath=str(MESH_IN))
        except (AttributeError, TypeError):  # Blender < 4.2 旧算子名
            bpy.ops.import_mesh.stl(filepath=str(MESH_IN))
    elif ext in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=str(MESH_IN))
    else:
        try:
            bpy.ops.wm.obj_import(filepath=str(MESH_IN))
        except (AttributeError, TypeError):
            bpy.ops.import_scene.obj(filepath=str(MESH_IN))
meshes = [o for o in scene.objects if o.type == "MESH"]
# TODO: 多物体/程序化建模在此扩展（primitive_* / mesh.from_pydata）

# %% === 2. 归一化取景（最长边 → 2 单位；布光/相机参数与输入尺度解耦） ===
def world_bounds(objs):
    pts = [o.matrix_world @ Vector(c) for o in objs for c in o.bound_box]
    lo = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
    hi = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
    return lo, hi

lo, hi = world_bounds(meshes)
max_dim = max((hi - lo).x, (hi - lo).y, (hi - lo).z)
scale = 2.0 / max_dim if max_dim > 0 else 1.0
for o in meshes:
    o.scale *= scale
scene.view_layers[0].update()
lo, hi = world_bounds(meshes)
for o in meshes:  # 落地：物体底部放 z=0
    o.location.z += -lo.z
scene.view_layers[0].update()
lo, hi = world_bounds(meshes)
center = (lo + hi) / 2
size = hi - lo
max_dim = max(size.x, size.y, size.z)

# %% === 3. 材质 / 地面 / 三点布光 / 相机（骨架；按需改写） ===
for o in meshes:
    if not o.data.materials:
        mat = bpy.data.materials.new("StudioGray")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes["Principled BSDF"]
        bsdf.inputs["Base Color"].default_value = (0.72, 0.70, 0.68, 1.0)
        bsdf.inputs["Metallic"].default_value = 0.08
        bsdf.inputs["Roughness"].default_value = 0.42
        o.data.materials.append(mat)

bpy.ops.mesh.primitive_plane_add(
    size=max_dim * 6, location=(center.x, center.y, 0)
)
ground = bpy.context.active_object
gmat = bpy.data.materials.new("Ground")
gmat.use_nodes = True
gbsdf = gmat.node_tree.nodes["Principled BSDF"]
gbsdf.inputs["Base Color"].default_value = (0.10, 0.10, 0.11, 1.0)
gbsdf.inputs["Roughness"].default_value = 0.85
ground.data.materials.append(gmat)

def add_area_light(name, energy, size_, location, target):
    ld = bpy.data.lights.new(name, type="AREA")
    ld.energy = energy
    ld.size = size_
    lo_ = bpy.data.objects.new(name, ld)
    scene.collection.objects.link(lo_)
    lo_.location = location
    direction = Vector(target) - lo_.location
    lo_.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()

d = max_dim
add_area_light("Key", 400, d * 1.4,
               (center.x + 1.6 * d, center.y - 1.8 * d, 1.8 * d), center)
add_area_light("Fill", 120, d * 1.8,
               (center.x - 1.8 * d, center.y - 1.2 * d, 0.9 * d), center)
add_area_light("Rim", 260, d * 0.9,
               (center.x - 0.6 * d, center.y + 2.0 * d, 1.5 * d), center)

cam_data = bpy.data.cameras.new("Camera")
cam_data.lens = 50.0
cam = bpy.data.objects.new("Camera", cam_data)
scene.collection.objects.link(cam)
scene.camera = cam
half_diag = (size.x ** 2 + size.y ** 2 + size.z ** 2) ** 0.5 / 2
aspect = scene.render.resolution_y / scene.render.resolution_x
half_fov = math.atan(cam_data.sensor_width * aspect / (2 * cam_data.lens))
dist = half_diag / math.sin(half_fov) * 1.15  # 包围球适配，留 ~15% 边距
cam_dir = Vector((1.0, -1.35, 0.62)).normalized()
cam.location = Vector((center.x, center.y, max_dim * 0.45)) + cam_dir * dist
target = bpy.data.objects.new("CamTarget", None)
scene.collection.objects.link(target)
target.location = center
constraint = cam.constraints.new(type="TRACK_TO")
constraint.target = target
constraint.track_axis = "TRACK_NEGATIVE_Z"
constraint.up_axis = "UP_Y"

world = bpy.data.worlds.new("World")
scene.world = world
if not world.use_nodes:
    world.use_nodes = True
bg_node = world.node_tree.nodes["Background"]
bg_node.inputs["Color"].default_value = (0.045, 0.047, 0.052, 1.0)
bg_node.inputs["Strength"].default_value = 1.0

# %% === 4. Cycles CPU 渲染（低采样单帧；勿开 EEVEE/勿并行多实例） ===
scene.render.engine = "CYCLES"  # 引擎 ID 坑点见 references/cycles-cpu.md
scene.cycles.device = "CPU"
scene.cycles.samples = 32
scene.cycles.use_denoising = True
scene.render.image_settings.file_format = "PNG"
OUT.parent.mkdir(parents=True, exist_ok=True)
scene.render.filepath = str(OUT)
bpy.ops.render.render(write_still=True)
print(f"[model3d] render: {OUT}")
'''


def render_scaffold(
    kind: str,
    slug: str,
    session_dir: Path | str,
    *,
    title: str | None = None,
    size_mm: tuple[float, float, float] = (40.0, 20.0, 8.0),
    mesh: Path | str | None = None,
) -> str:
    """生成 bpy 脚手架脚本文本（``%TOKEN%`` 占位替换，生成代码可直接 ``compile``）。

    Args:
        kind: ``"part"``（打印件四段式）或 ``"scene"``（概念渲染场景）。
        slug: 脚本/产物文件名前缀，如 ``"sample_holder"``。
        session_dir: 会话目录（产物落点；会写入脚本文本，运行时按此导出）。
        title: 脚本标题；缺省由 slug 派生。
        size_mm: 仅 ``part``：占位几何 (L, W, H)（mm），生成后必按实际设计改写。
        mesh: 仅 ``scene``：要导入渲染的模型路径（STL/GLB/OBJ）；None → 纯建模骨架。

    Returns:
        脚本文本（UTF-8）。用 :func:`write_scaffold` 落盘。

    Raises:
        ValueError: kind 不在 :data:`SCAFFOLD_KINDS` 内。
    """
    if kind not in SCAFFOLD_KINDS:
        raise ValueError(f"未知脚手架种类 {kind!r}，可选：{SCAFFOLD_KINDS}")
    title = title or f"{slug}（{kind}）"
    session = Path(session_dir)
    if kind == "part":
        dim_l, dim_w, dim_h = size_mm
        tokens = {
            "%TITLE%": title,
            "%SLUG%": slug,
            "%SESSION_DIR%": str(session),
            "%L%": repr(dim_l),
            "%W%": repr(dim_w),
            "%H%": repr(dim_h),
            "%EXPECT_BBOX%": repr((dim_l, dim_w, dim_h)),
        }
        text = _PART_TEMPLATE
    else:
        mesh_expr = f"Path({str(Path(mesh))!r})" if mesh is not None else "None"
        render_out = session / "renders" / f"{slug}.png"
        tokens = {
            "%TITLE%": title,
            "%SLUG%": slug,
            "%SESSION_DIR%": str(session),
            "%MESH%": mesh_expr,
            "%RENDER_OUT%": str(render_out),
        }
        text = _SCENE_TEMPLATE
    for key, value in tokens.items():
        text = text.replace(key, value)
    return text


def write_scaffold(text: str, path: Path | str, *, guard: bool = True) -> Path:
    """把脚手架文本落盘（含父目录创建），默认带 ``data/`` 防散落护栏。

    Args:
        text: :func:`render_scaffold` 的输出。
        path: 目标 ``.py`` 路径。
        guard: True（默认）时路径必须在项目 ``data/`` 根内（脚本是会话运行产物）；
            测试沙箱/联调场景传 False（豁免面，勿在交付路径上使用）。

    Returns:
        落盘路径（解析后的绝对路径）。

    Raises:
        ValueError: ``guard=True`` 且路径在 ``data/`` 根外
            （:func:`pysci.paths.assert_within_data`）。
    """
    out = Path(path)
    if guard:
        out = assert_within_data(out, what="bpy 脚手架脚本")
    out = out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    return out


def ensure_model_session(name: str, slug: str) -> Path:
    """建立（或复用）一次 3D 建模会话的目录骨架（旧域形态保留）。

    骨架：``stl/``（打印件导出）、``renders/``（渲染图/GLB）、``scripts/``（bpy 脚本）、
    ``notes.md``（建模日志，首次写入表头）。规范位置
    ``data/research/<n>_<name>/models/<slug>/``（:func:`pysci.paths.research_model_dir`）。

    Args:
        name: 研究线名称（不含数字前缀），如 ``"example"``。
        slug: 模型会话 slug，如 ``"sample_holder"``。

    Returns:
        会话目录绝对路径。
    """
    session = assert_within_data(
        research_model_dir(name, slug=slug), what="3D 建模会话"
    )
    (session / "stl").mkdir(parents=True, exist_ok=True)
    (session / "renders").mkdir(parents=True, exist_ok=True)
    (session / "scripts").mkdir(parents=True, exist_ok=True)
    notes = session / "notes.md"
    if not notes.exists():
        notes.write_text("# 3D 建模日志（时间线）\n\n", encoding="utf-8")
    return session
