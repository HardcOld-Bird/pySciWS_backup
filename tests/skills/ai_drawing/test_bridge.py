"""bridge 单测：范本分析（纯函数）/ design_spec 渲染 / 桥接脚手架产物 / 调色板注册。

`bridge()` 会调 scientific_plotting 的 scaffold_figure（写真实项目路径）与 assert_within_data
（要求产物在 data/ 内）。测试把 scaffold/runner 的 figures_root·figures_code_root 重定向到
tmp_path，并把 assert_within_data 置为 no-op，从而完全隔离、不污染 src/pysci/research 与
data/research。调色板注册另用 _PALETTES 副本 + tmp 持久化文件隔离。
"""

from __future__ import annotations

import pytest
from PIL import Image

from pysci.skills.ai_drawing.tools import bridge as br


def _ref(tmp_path, w=60, h=40, name="ref.png"):
    """造一张横构图合成范本（左暖右冷），返回路径。"""
    im = Image.new("RGB", (w, h))
    for x in range(w):
        c = (220, 120, 40) if x < w // 2 else (40, 90, 200)
        for y in range(h):
            im.putpixel((x, y), c)
    p = tmp_path / name
    im.save(p)
    return p


# ---------------------------------------------------------------------------
# analyze_reference（纯函数，无副作用）
# ---------------------------------------------------------------------------
def test_analyze_reference_fields(tmp_path):
    p = _ref(tmp_path, w=60, h=40)
    info = br.analyze_reference(p, n_colors=4)
    assert info["width"] == 60 and info["height"] == 40
    assert info["orientation"] == "landscape"
    assert info["aspect"] == round(60 / 40, 4)
    assert info["aspect_hw"] == round(40 / 60, 4)
    assert 0 <= info["mean_brightness"] <= 255
    assert isinstance(info["palette"], list) and info["palette"]
    assert all(h.startswith("#") for h in info["palette"])


def test_analyze_reference_orientation(tmp_path):
    assert br.analyze_reference(_ref(tmp_path, 30, 50, "p.png"))["orientation"] == "portrait"
    assert br.analyze_reference(_ref(tmp_path, 40, 40, "s.png"))["orientation"] == "square"


def test_analyze_reference_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="审美范本不存在"):
        br.analyze_reference(tmp_path / "nope.png")


# ---------------------------------------------------------------------------
# _render_design_spec（纯函数）
# ---------------------------------------------------------------------------
def test_render_design_spec_content(tmp_path):
    info = {
        "path": str(tmp_path / "ref.png"),
        "width": 60, "height": 40, "aspect": 1.5, "aspect_hw": 0.667,
        "orientation": "landscape", "mean_brightness": 90.0,
        "palette": ["#1b2a4a", "#e0b050", "#cccccc"],
    }
    txt = br._render_design_spec(
        slug="fig1_cover", research="demo", info=info,
        figdir=tmp_path / "fig1_cover", pipeline=tmp_path / "code" / "fig1_cover.py",
        ref_copy=tmp_path / "fig1_cover" / "_reference.png",
        style="nature", width="single", template="multi_panel", palette_name=None,
    )
    assert "# fig1_cover" in txt
    assert "不可信" in txt          # AI 数据不可信警告
    assert "#1b2a4a" in txt        # 主色 hex
    assert "![reference]" in txt   # 内嵌范本
    assert "偏暗背景" in txt        # 亮度提示（90<128）
    assert "横构图" in txt          # 朝向提示


def test_render_design_spec_with_palette_name(tmp_path):
    info = {
        "path": "x", "width": 10, "height": 10, "aspect": 1.0, "aspect_hw": 1.0,
        "orientation": "square", "mean_brightness": 200.0, "palette": ["#aabbcc"],
    }
    txt = br._render_design_spec(
        slug="s", research="demo", info=info, figdir=tmp_path, pipeline=tmp_path / "s.py",
        ref_copy=None, style="aps", width="double", template="multi_panel",
        palette_name="my-ai-pal",
    )
    assert "my-ai-pal" in txt
    assert "色盲" in txt           # 色盲安全告警
    assert "偏亮背景" in txt


# ---------------------------------------------------------------------------
# register_palette（scientific_plotting 侧，持久化）——隔离全局态
# ---------------------------------------------------------------------------
@pytest.fixture
def isolated_palette(tmp_path, monkeypatch):
    from pysci.skills.scientific_plotting.tools import palette

    monkeypatch.setattr(palette, "_PALETTES", dict(palette._PALETTES))
    monkeypatch.setattr(palette, "_USER_PALETTE_FILE", tmp_path / "palettes.json")
    monkeypatch.setattr(palette, "_user_loaded", True)  # 不去读真实持久化文件
    return palette


def test_register_palette_persists(isolated_palette, tmp_path):
    palette = isolated_palette
    norm = palette.register_palette("ai-ref", ["#1b2a4a", "#e0b050"])
    assert norm == ("#1B2A4A", "#E0B050")  # 规范化为大写
    assert palette.get_palette("ai-ref") == norm
    assert palette.color(0, "ai-ref") == "#1B2A4A"
    # 持久化文件已写、含该调色板
    data = (tmp_path / "palettes.json").read_text(encoding="utf-8")
    assert "ai-ref" in data and "#1B2A4A" in data


def test_register_palette_validation(isolated_palette):
    palette = isolated_palette
    with pytest.raises(ValueError, match="name 不能为空"):
        palette.register_palette("", ["#ffffff"])
    with pytest.raises(ValueError, match="colors 不能为空"):
        palette.register_palette("x", [])
    with pytest.raises(ValueError, match="非法 hex"):
        palette.register_palette("x", ["#zzz"])


def test_register_palette_no_persist(isolated_palette, tmp_path):
    palette = isolated_palette
    palette.register_palette("mem-only", ["#123456"], persist=False)
    assert palette.get_palette("mem-only") == ("#123456",)
    assert not (tmp_path / "palettes.json").exists()


# ---------------------------------------------------------------------------
# bridge()（脚手架产物）——重定向绘图路径 + no-op 护栏
# ---------------------------------------------------------------------------
@pytest.fixture
def isolated_plotting(tmp_path, monkeypatch):
    """把 scaffold/runner 的 figures 根重定向到 tmp，并把 assert_within_data 置 no-op。"""
    from pysci.skills.scientific_plotting.tools import runner, scaffold

    data_root = tmp_path / "data" / "figures"
    code_root = tmp_path / "code" / "figures"
    monkeypatch.setattr(scaffold, "figures_root", lambda research: data_root)
    monkeypatch.setattr(scaffold, "figures_code_root", lambda research: code_root)
    monkeypatch.setattr(runner, "figures_code_root", lambda research: code_root)
    monkeypatch.setattr(br, "assert_within_data", lambda p, **kw: p)
    return {"data_root": data_root, "code_root": code_root}


def test_bridge_scaffolds_pipeline(tmp_path, isolated_plotting):
    ref = _ref(tmp_path)
    res = br.bridge(ref, "demo", "fig1_cover", style="nature", width="single")
    figdir = res["figdir"]
    assert figdir == isolated_plotting["data_root"] / "fig1_cover"
    assert figdir.is_dir()
    # design_spec.md 写出且引用了范本
    spec = res["design_spec"]
    assert spec.is_file()
    assert "fig1_cover" in spec.read_text(encoding="utf-8")
    # 范本被复制进图目录
    assert res["ref_copy"] is not None and res["ref_copy"].is_file()
    # 管线代码路径落在重定向的 code_root
    assert res["pipeline"] == isolated_plotting["code_root"] / "fig1_cover.py"
    assert res["pipeline"].is_file()
    assert res["palette_name"] is None  # 未给 palette_name


def test_bridge_no_copy_ref(tmp_path, isolated_plotting):
    ref = _ref(tmp_path)
    res = br.bridge(ref, "demo", "fig_nc", copy_ref=False)
    assert res["ref_copy"] is None


def test_bridge_with_palette_registers_and_writes_style(tmp_path, isolated_plotting, monkeypatch):
    from pysci.skills.scientific_plotting.tools import palette

    monkeypatch.setattr(palette, "_PALETTES", dict(palette._PALETTES))
    monkeypatch.setattr(palette, "_USER_PALETTE_FILE", tmp_path / "palettes.json")
    monkeypatch.setattr(palette, "_user_loaded", True)

    ref = _ref(tmp_path)
    res = br.bridge(ref, "demo", "fig_pal", palette_name="ai-bridge-demo")
    assert res["palette_name"] == "ai-bridge-demo"
    assert palette.get_palette("ai-bridge-demo")
    # figdir 级 STYLE.yaml 绑定了该调色板
    style_yaml = res["figdir"] / "STYLE.yaml"
    assert style_yaml.is_file()
    assert "palette: ai-bridge-demo" in style_yaml.read_text(encoding="utf-8")


def test_bridge_missing_ref_raises(tmp_path, isolated_plotting):
    with pytest.raises(FileNotFoundError):
        br.bridge(tmp_path / "nope.png", "demo", "fig_x")
