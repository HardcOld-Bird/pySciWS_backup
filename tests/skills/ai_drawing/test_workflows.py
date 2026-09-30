"""workflows 单测：离线断言 API 格式图的 class_type / 连线 / 参数（不联网）。

这些用例是"class_type 禁止猜"铁律的护栏：图的节点类型、link 编码、成本护栏夹取都在此固化，
一旦 workflows.py 构造逻辑回归即被捕获。
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from pysci.skills.ai_drawing.tools import workflows as wf


def _patch_settings(monkeypatch, **overrides):
    """替换 workflows 模块引用的 settings（Settings 是 frozen dataclass，不能就地改属性）。"""
    monkeypatch.setattr(wf, "settings", dataclasses.replace(wf.settings, **overrides))


def test_link_encoding():
    assert wf._link("2", 0) == ["2", 0]
    assert wf._link("5") == ["5", 0]


def test_api_client_node():
    node = wf.api_client_node("MyKey")
    assert node["class_type"] == "JimengAPIClient"
    assert node["inputs"]["key_name"] == "MyKey"
    # pysci 绝不在此传原始密钥
    assert node["inputs"]["new_api_key"] == ""


def test_txt2img_minimal_graph():
    g = wf.txt2img_seedream("a journal cover", seed=42)
    # 3 节点：client(1) → gen(2) → save(3)
    assert set(g) == {"1", "2", "3"}
    assert g["1"]["class_type"] == "JimengAPIClient"
    assert g["2"]["class_type"] == "JimengSeedream4"
    assert g["3"]["class_type"] == "SaveImage"
    # gen 连 client 的 slot 0
    assert g["2"]["inputs"]["client"] == ["1", 0]
    # save 连 gen 的 images 输出（slot 0）
    assert g["3"]["inputs"]["images"] == ["2", 0]
    assert g["2"]["inputs"]["prompt"] == "a journal cover"
    assert g["2"]["inputs"]["seed"] == 42
    assert g["2"]["inputs"]["model_version"] == "doubao-seedream-4.0"
    # 纯文生图无 images 输入
    assert "images" not in g["2"]["inputs"]


def test_txt2img_default_single_image():
    """成本护栏：默认 generation_count=1、watermark=False。"""
    g = wf.txt2img_seedream("x")
    assert g["2"]["inputs"]["generation_count"] == 1
    assert g["2"]["inputs"]["watermark"] is False


def test_txt2img_n_capped_by_max_images(monkeypatch):
    """n 受 settings.max_images 夹取（成本护栏）。"""
    _patch_settings(monkeypatch, max_images=2)
    g = wf.txt2img_seedream("x", n=10)
    assert g["2"]["inputs"]["generation_count"] == 2


def test_txt2img_size_normalized():
    g = wf.txt2img_seedream("x", size="2K")
    assert g["2"]["inputs"]["size"] == "2K (adaptive)"


def test_txt2img_with_quota_inserts_node():
    """image_quota>0 → 插入 JimengQuotaSettings，节点 id 顺移。"""
    g = wf.txt2img_seedream("x", image_quota=5)
    # client(1) → quota(2) → gen(3) → save(4)
    assert g["2"]["class_type"] == "JimengQuotaSettings"
    assert g["2"]["inputs"]["image_limit"] == 5
    assert g["2"]["inputs"]["client"] == ["1", 0]
    assert g["3"]["class_type"] == "JimengSeedream4"
    assert g["4"]["class_type"] == "SaveImage"
    assert g["4"]["inputs"]["images"] == ["3", 0]


def test_txt2img_group_and_max_images_clamped():
    g = wf.txt2img_seedream("x", group=True, max_images=99)
    assert g["2"]["inputs"]["enable_group_generation"] is True
    # max_images 夹到 [1,15]
    assert g["2"]["inputs"]["max_images"] == 15


def test_txt2img_model_45():
    g = wf.txt2img_seedream("x", model="doubao-seedream-4.5")
    assert g["2"]["inputs"]["model_version"] == "doubao-seedream-4.5"


def test_img2img_graph_links_reference_images():
    g = wf.img2img_seedream("restyle", ["a.png", "b.png"], seed=7)
    # client(1) → load(2) → load(3) → gen(4) → save(5)
    assert g["2"]["class_type"] == "LoadImage"
    assert g["3"]["class_type"] == "LoadImage"
    assert g["2"]["inputs"]["image"] == "a.png"
    assert g["3"]["inputs"]["image"] == "b.png"
    assert g["4"]["class_type"] == "JimengSeedream4"
    # autogrow images 用嵌套 dict 序列化
    images = g["4"]["inputs"]["images"]
    assert images == {"image_1": ["2", 0], "image_2": ["3", 0]}
    assert g["5"]["inputs"]["images"] == ["4", 0]


def test_img2img_requires_reference():
    with pytest.raises(ValueError, match="至少一张参考图"):
        wf.img2img_seedream("x", [])


def test_img2img_provider_without_i2i_raises():
    with pytest.raises(ValueError, match="不支持图生图"):
        wf.img2img_seedream("x", ["a.png"], provider="jimeng-ark-s3")


def test_apply_args_friendly_keys():
    """友好键自动定位生成节点并覆盖对应输入。"""
    g = wf.txt2img_seedream("old", seed=1)
    wf.apply_args(g, {"prompt": "new", "seed": 99, "n": 2})
    assert g["2"]["inputs"]["prompt"] == "new"
    assert g["2"]["inputs"]["seed"] == 99
    assert g["2"]["inputs"]["generation_count"] == 2


def test_apply_args_dotted_keys():
    """点号键精确定位节点输入。"""
    g = wf.txt2img_seedream("x")
    wf.apply_args(g, {"2.size": "4K (adaptive)", "3.filename_prefix": "Custom/Dir"})
    assert g["2"]["inputs"]["size"] == "4K (adaptive)"
    assert g["3"]["inputs"]["filename_prefix"] == "Custom/Dir"


def test_apply_args_none_is_noop():
    g = wf.txt2img_seedream("x", seed=5)
    before = json.dumps(g, sort_keys=True)
    wf.apply_args(g, None)
    assert json.dumps(g, sort_keys=True) == before


def test_find_gen_node():
    g = wf.txt2img_seedream("x")
    assert wf._find_gen_node(g) == "2"
    g2 = wf.txt2img_seedream("x", image_quota=3)
    assert wf._find_gen_node(g2) == "3"


def test_save_load_roundtrip(tmp_path, monkeypatch):
    """工作流配方保存/加载往返（重定向 workflows_dir 到 tmp）。"""
    _patch_settings(monkeypatch, workflows_dir=tmp_path)
    g = wf.txt2img_seedream("cover art", seed=3)
    dest = wf.save_workflow(g, "cover")
    assert dest == tmp_path / "cover.json"
    assert dest.is_file()
    loaded = wf.load_workflow("cover")
    assert loaded == g
    assert wf.list_workflows() == [dest]


def test_load_workflow_missing_raises(tmp_path, monkeypatch):
    _patch_settings(monkeypatch, workflows_dir=tmp_path)
    with pytest.raises(FileNotFoundError):
        wf.load_workflow("nope")


def test_graph_summary_renders():
    g = wf.txt2img_seedream("x")
    s = wf.graph_summary(g)
    assert "3 节点" in s
    assert "JimengSeedream4" in s
    assert "SaveImage" in s
