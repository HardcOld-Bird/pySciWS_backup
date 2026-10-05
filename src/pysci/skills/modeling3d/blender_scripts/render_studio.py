"""render_studio.py — 在 Blender 内执行的功能性渲染脚本（studio 布光快速出图）。

由 ``pysci-model3d render --mesh <file>`` 通过 ``blender -b -P render_studio.py -- <args>``
驱动，**不是项目 .venv 里的模块**——运行于 Blender 自带 Python，不能 import pysci。

职责：清空场景 → 导入网格（按扩展名选 importer）→ 归一化尺度（物体最长边 = 2 场景单位，
从而布光/取景参数与输入单位无关）→ 无材质网格补 studio 灰材质 → 地面 + 三点布光 →
50mm 相机自动取景 → 引擎 ID 回退链设置 → 渲染 PNG。

``--`` 之后的参数（全部必传由 CLI 保证）：
    --mesh <path>      网格文件（.stl/.glb/.gltf/.obj/.ply）
    --out <path>       输出 PNG
    --engine <name>    EEVEE | CYCLES（大小写不敏感；实际 ID 走回退链）
    --samples <int>    渲染采样
    --width <int>      出图宽 px
    --height <int>     出图高 px
    --bg <spec>        可选：studio（默认，深灰背景）| transparent | 0xRRGGBB
"""

import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector

# ---------------------------------------------------------------------------
# 参数解析（-- 之后）
# ---------------------------------------------------------------------------
argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
opts = {}
i = 0
while i < len(argv):
    key = argv[i].lstrip("-")
    opts[key] = argv[i + 1]
    i += 2

MESH = Path(opts["mesh"])
OUT = Path(opts["out"])
ENGINE = opts.get("engine", "EEVEE").upper()
SAMPLES = int(opts.get("samples", "128"))
WIDTH = int(opts.get("width", "1920"))
HEIGHT = int(opts.get("height", "1440"))
BG = opts.get("bg", "studio")

# ---------------------------------------------------------------------------
# 清空场景
# ---------------------------------------------------------------------------
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene

# ---------------------------------------------------------------------------
# 导入网格（按扩展名选 importer；均为 Blender 4.x/5.x 内置算子）
# ---------------------------------------------------------------------------
ext = MESH.suffix.lower()
if ext == ".stl":
    bpy.ops.wm.stl_import(filepath=str(MESH))
elif ext in (".glb", ".gltf"):
    bpy.ops.import_scene.gltf(filepath=str(MESH))
elif ext == ".obj":
    bpy.ops.wm.obj_import(filepath=str(MESH))
elif ext == ".ply":
    bpy.ops.wm.ply_import(filepath=str(MESH))
else:
    raise SystemExit(f"[render_studio] 不支持的网格格式: {ext}")

meshes = [o for o in scene.objects if o.type == "MESH"]
if not meshes:
    raise SystemExit("[render_studio] 导入后场景中没有网格对象")


# ---------------------------------------------------------------------------
# 归一化尺度：物体世界包围盒最长边 → 2.0 场景单位（布光参数与输入单位解耦）
# ---------------------------------------------------------------------------
def world_bounds(objs):
    pts = [obj.matrix_world @ Vector(c) for obj in objs for c in obj.bound_box]
    lo = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
    hi = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
    return lo, hi


lo, hi = world_bounds(meshes)
max_dim = max((hi - lo).x, (hi - lo).y, (hi - lo).z)
scale = 2.0 / max_dim if max_dim > 0 else 1.0
for obj in meshes:
    obj.scale *= scale
scene.view_layers[0].update()
lo, hi = world_bounds(meshes)
center = (lo + hi) / 2
size = hi - lo
max_dim = max(size.x, size.y, size.z)

# 落地：把物体底部放到 z=0
dz = -lo.z
for obj in meshes:
    obj.location.z += dz
scene.view_layers[0].update()
lo, hi = world_bounds(meshes)
center = (lo + hi) / 2

# ---------------------------------------------------------------------------
# 材质：无材质的网格（典型 STL）补一个中性 studio 灰
# ---------------------------------------------------------------------------
for obj in meshes:
    if not obj.data.materials:
        mat = bpy.data.materials.new("StudioGray")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes["Principled BSDF"]
        bsdf.inputs["Base Color"].default_value = (0.72, 0.70, 0.68, 1.0)
        bsdf.inputs["Metallic"].default_value = 0.08
        bsdf.inputs["Roughness"].default_value = 0.42
        obj.data.materials.append(mat)
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    # 全网格 shade_smooth 会让平面呈"融蜡"感；按角度自动平滑（圆柱/球面光滑、平面保持锐利）
    try:
        bpy.ops.object.shade_auto_smooth(angle=math.radians(30))
    except AttributeError:
        try:
            bpy.ops.object.shade_smooth_by_angle(angle=math.radians(30))
        except AttributeError:
            bpy.ops.object.shade_smooth()

# ---------------------------------------------------------------------------
# 地面（承接阴影，尺度随物体）
# ---------------------------------------------------------------------------
ground_size = max_dim * 6
bpy.ops.mesh.primitive_plane_add(size=ground_size, location=(center.x, center.y, 0))
ground = bpy.context.active_object
gmat = bpy.data.materials.new("Ground")
gmat.use_nodes = True
gbsdf = gmat.node_tree.nodes["Principled BSDF"]
gbsdf.inputs["Base Color"].default_value = (0.10, 0.10, 0.11, 1.0)
gbsdf.inputs["Roughness"].default_value = 0.85
ground.data.materials.append(gmat)


# ---------------------------------------------------------------------------
# 三点布光（距离/能量按归一化尺度取定值）
# ---------------------------------------------------------------------------
def add_area_light(name, energy, size, location, target):
    light_data = bpy.data.lights.new(name, type="AREA")
    light_data.energy = energy
    light_data.size = size
    light_obj = bpy.data.objects.new(name, light_data)
    scene.collection.objects.link(light_obj)
    light_obj.location = location
    direction = Vector(target) - light_obj.location
    light_obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    return light_obj


d = max_dim
add_area_light(
    "Key", 400, d * 1.4, (center.x + 1.6 * d, center.y - 1.8 * d, 1.8 * d), center
)
add_area_light(
    "Fill", 120, d * 1.8, (center.x - 1.8 * d, center.y - 1.2 * d, 0.9 * d), center
)
add_area_light(
    "Rim", 260, d * 0.9, (center.x - 0.6 * d, center.y + 2.0 * d, 1.5 * d), center
)

# ---------------------------------------------------------------------------
# 世界背景
# ---------------------------------------------------------------------------
world = bpy.data.worlds.new("World")
scene.world = world
if not world.use_nodes:  # 5.x 默认已启用；直接赋值会触发 6.0 弃用警告
    world.use_nodes = True
bg_node = world.node_tree.nodes["Background"]
if BG == "transparent":
    scene.render.film_transparent = True
    bg_node.inputs["Color"].default_value = (0, 0, 0, 0)
elif BG.startswith("0x") and len(BG) == 8:
    r = int(BG[2:4], 16) / 255
    g = int(BG[4:6], 16) / 255
    b = int(BG[6:8], 16) / 255
    bg_node.inputs["Color"].default_value = (r, g, b, 1)
else:  # studio：深灰影棚背景
    bg_node.inputs["Color"].default_value = (0.045, 0.047, 0.052, 1)
    bg_node.inputs["Strength"].default_value = 1.0

# ---------------------------------------------------------------------------
# 相机：50mm，自动取景（水平/垂直视场角取更保守者，留 15% 边距）
# ---------------------------------------------------------------------------
cam_data = bpy.data.cameras.new("Camera")
cam_data.lens = 50.0
cam = bpy.data.objects.new("Camera", cam_data)
scene.collection.objects.link(cam)
scene.camera = cam

sensor_w = cam_data.sensor_width  # 36 mm
sensor_h = sensor_w * HEIGHT / WIDTH  # 有效竖直画幅（按出图比例）
lens = cam_data.lens
half_diag = math.sqrt(size.x**2 + size.y**2 + size.z**2) / 2
# 距离 = 半extent / tan(半视场角)，tan(θ) = sensor/(2·lens)；水平/垂直/对角取最保守者，留 15% 边距
dist = (
    max(
        (max_dim / 2) * (2 * lens / sensor_w),
        (size.z / 2 + max_dim * 0.1) * (2 * lens / sensor_h),
        half_diag,
    )
    * 1.15
)
cam_dir = Vector((1.0, -1.35, 0.62)).normalized()
cam.location = Vector((center.x, center.y, max_dim * 0.45)) + cam_dir * dist
constraint = cam.constraints.new(type="TRACK_TO")
constraint.target = bpy.data.objects.new("CamTarget", None)
scene.collection.objects.link(constraint.target)
constraint.target.location = center
constraint.track_axis = "TRACK_NEGATIVE_Z"
constraint.up_axis = "UP_Y"

# ---------------------------------------------------------------------------
# 渲染设置（引擎 ID 回退链：请求名 → 各版本真实 ID → CYCLES 兜底）
# ---------------------------------------------------------------------------
# 引擎 ID 随版本浮动（实测 Blender 5.1：EEVEE 的 ID 为 BLENDER_EEVEE；4.2–4.5 为
# BLENDER_EEVEE_NEXT），按回退链逐个试设，CYCLES 兜底
candidates = (
    ["CYCLES"]
    if ENGINE == "CYCLES"
    else ["BLENDER_EEVEE", "BLENDER_EEVEE_NEXT", "EEVEE_NEXT", "EEVEE"]
) + ["CYCLES"]
for cand in candidates:
    try:
        scene.render.engine = cand
        break
    except TypeError:
        continue
print(f"[render_studio] engine: {scene.render.engine}")

if scene.render.engine == "CYCLES":
    scene.cycles.samples = SAMPLES
    scene.cycles.use_denoising = True
else:
    try:
        scene.eevee.taa_render_samples = max(32, SAMPLES)
    except AttributeError:
        pass

scene.render.resolution_x = WIDTH
scene.render.resolution_y = HEIGHT
scene.render.image_settings.file_format = "PNG"
scene.render.filepath = str(OUT)

bpy.ops.render.render(write_still=True)
print(f"[render_studio] render: {OUT}")
