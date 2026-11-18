"""脚手架机械断言：生成文本可编译、含四段式/红线要素、data/ 护栏、会话骨架。"""

from __future__ import annotations

import shutil

import pytest

from pysci.paths import research_asset_dir
from pysci.skills.modeling_3d import scaffold

TEST_NAME = "zz_modeling3d_selftest"


@pytest.fixture()
def session_dir():
    """在 data/ 建临时建模会话；结束整目录清理。"""
    base = research_asset_dir(TEST_NAME)
    sess = scaffold.ensure_model_session(TEST_NAME, "s1")
    yield sess
    shutil.rmtree(base, ignore_errors=True)


def test_part_scaffold_compiles_and_has_contract():
    text = scaffold.render_scaffold(
        "part", "demo", r"D:\data\demo", size_mm=(40.0, 20.0, 8.0)
    )
    compile(text, "gen_part", "exec")  # 生成代码必须是合法 Python
    # 四段式
    assert "1. 设计参数" in text and "来源：<设计依据" in text
    assert "2. 场景/单位 + 建模" in text
    assert "3. 几何断言" in text
    assert "4. 导出 + 标记行协议" in text
    # 单位契约 + 标记行协议 + 导出回退链
    assert "scale_length = 0.001" in text
    assert '[model3d] stl:' in text and '[model3d] volume_mm3:' in text
    assert "wm.stl_export" in text and "export_mesh.stl" in text
    assert "export_scene.gltf" in text and 'export_format="GLB"' in text
    # 参数替换进脚本
    assert "L = 40.0" in text and "H = 8.0" in text
    assert r"D:\data\demo" in text


def test_scene_scaffold_compiles_and_has_cycles_cpu():
    text = scaffold.render_scaffold(
        "scene", "demo", r"D:\data\demo", mesh=r"D:\data\demo\stl\demo.stl"
    )
    compile(text, "gen_scene", "exec")
    # Cycles CPU 红线（EEVEE 需 GPU 不可用）
    assert 'scene.render.engine = "CYCLES"' in text
    assert 'scene.cycles.device = "CPU"' in text
    assert "use_denoising" in text
    assert "EEVEE" in text  # 注释里注明不可用
    # 影棚骨架 + 取景 + 渲染产物标记
    assert "add_area_light" in text and "TRACK_TO" in text
    assert "half_fov" in text and "sin(half_fov)" in text  # 包围球取景
    assert '[model3d] render:' in text
    assert "demo.stl" in text  # 模型来源内嵌


def test_scene_scaffold_without_mesh():
    text = scaffold.render_scaffold("scene", "demo", r"D:\data\demo")
    compile(text, "gen_scene2", "exec")
    assert "MESH_IN = None" in text


def test_unknown_kind_raises():
    with pytest.raises(ValueError):
        scaffold.render_scaffold("gltf", "demo", r"D:\data\demo")


def test_write_scaffold_guard_rejects_outside_data(tmp_path):
    with pytest.raises(ValueError):
        scaffold.write_scaffold("x = 1\n", tmp_path / "stray.py")


def test_write_scaffold_and_session_layout(session_dir):
    script = scaffold.write_scaffold(
        scaffold.render_scaffold("part", "s1", session_dir),
        session_dir / "scripts" / "s1.py",
    )
    assert script.is_file()
    assert "SESSION_DIR" in script.read_text(encoding="utf-8")
    assert (session_dir / "stl").is_dir()
    assert (session_dir / "renders").is_dir()
    assert (session_dir / "scripts").is_dir()
    notes = session_dir / "notes.md"
    assert notes.is_file() and notes.read_text(encoding="utf-8").startswith("# 3D 建模日志")


def test_ensure_model_session_idempotent(session_dir):
    again = scaffold.ensure_model_session(TEST_NAME, "s1")
    assert again == session_dir
