"""smoke_model3d/test_part — 端到端冒烟测试件（build123d）。

单位：毫米（mm）。几何：40×20×5 底板 + 双 φ3 通孔 + 中央 φ10×3 凸台（φ4 通孔）+ 竖边 0.8 倒角。
执行：uv run pysci-model3d build 本文件 --strict-volume --expect-bbox 40,20,8
"""

from pathlib import Path

from build123d import *

SESSION_DIR = Path(
    r"D:/XXXIIIGGG/projects/pySci/pySciWS/data/research/smoke_model3d/models/test_part"
)
STL_OUT = SESSION_DIR / "stl" / "test_part.stl"
STEP_OUT = SESSION_DIR / "step" / "test_part.step"

# %% === 1. 设计参数（mm） ===
L_base = 40.0  # 底板长
W_base = 20.0  # 底板宽
H_base = 5.0  # 底板厚
d_hole = 3.0  # 安装孔径（φ3）
x_hole = 12.0  # 孔位 x（±）
r_boss = 5.0  # 凸台半径（φ10）
h_boss = 3.0  # 凸台高
d_boss_hole = 4.0  # 凸台通孔（φ4）
c_edge = 0.8  # 竖边倒角


# %% === 2. 建模 ===
with BuildPart() as part:
    Box(L_base, W_base, H_base)
    with Locations((x_hole, 0, 0), (-x_hole, 0, 0)):
        Hole(d_hole / 2)
    with BuildSketch(part.part.faces().sort_by(Axis.Z)[-1]):
        Circle(r_boss)
    extrude(amount=h_boss)
    with Locations((0, 0, H_base + h_boss)):
        Hole(d_boss_hole / 2)
    chamfer(length=c_edge, objects=part.part.edges().filter_by(Axis.Z))


# %% === 3. 几何断言 ===
bb = part.part.bounding_box()
assert abs(bb.size.X - L_base) < 1e-6, f"X 尺寸不符: {bb.size.X}"
assert abs(bb.size.Y - W_base) < 1e-6, f"Y 尺寸不符: {bb.size.Y}"
assert abs(bb.size.Z - (H_base + h_boss)) < 1e-6, f"Z 尺寸不符: {bb.size.Z}"


# %% === 4. 导出（标记行是与 pysci-model3d build 的协议，勿改动格式） ===
STL_OUT.parent.mkdir(parents=True, exist_ok=True)
STEP_OUT.parent.mkdir(parents=True, exist_ok=True)
export_stl(part.part, str(STL_OUT))
export_step(part.part, str(STEP_OUT))
print(f"[model3d] stl: {STL_OUT}")
print(f"[model3d] step: {STEP_OUT}")
print(f"[model3d] volume_mm3: {part.part.volume:.4f}")
