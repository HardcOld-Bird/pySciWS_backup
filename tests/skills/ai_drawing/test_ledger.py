"""ledger 单测：记账 / 查询 / 统计 / Markdown 渲染 / PNG 元数据内嵌。

用 dataclasses.replace 把账本目录重定向到 tmp_path（Settings 是 frozen dataclass），
绝不写真实技能数据区。
"""

from __future__ import annotations

import dataclasses

import pytest
from PIL import Image

from pysci.skills.ai_drawing.tools import ledger


@pytest.fixture
def tmp_ledger(tmp_path, monkeypatch):
    """把 ledger.settings.module_dir 指向 tmp_path（隔离账本文件）。"""
    monkeypatch.setattr(
        ledger, "settings", dataclasses.replace(ledger.settings, module_dir=tmp_path)
    )
    return tmp_path


def test_record_appends_manifest(tmp_ledger):
    e = ledger.record(prompt="a cover", seed=42, model="m1", backend="ark", out="x.png")
    assert e.prompt == "a cover"
    assert e.seed == 42
    assert ledger.manifest_path().is_file()
    entries = ledger.load_all()
    assert len(entries) == 1
    assert entries[0].backend == "ark"


def test_record_does_not_render_markdown(tmp_ledger):
    """record() 只追加 JSONL，不重渲染 MD（避免旧实现的 O(n) 开销）。"""
    ledger.record(prompt="p1")
    ledger.record(prompt="p2")
    assert not ledger.ledger_path().exists()
    ledger.render_markdown()               # 渲染由调用方在批量结束时显式做一次
    assert ledger.ledger_path().is_file()


def test_record_multiple_appends(tmp_ledger):
    ledger.record(prompt="p1", backend="ark")
    ledger.record(prompt="p2", backend="imagegen")
    assert len(ledger.load_all()) == 2


def test_record_stores_recipe(tmp_ledger):
    ledger.record(prompt="p", backend="ark", recipe="cover_v1")
    assert ledger.load_all()[0].recipe == "cover_v1"


def test_entry_maps_legacy_workflow_to_recipe():
    """旧账本（重构前）写的是 ``workflow`` 字段 → 读时映射到 ``recipe``，不丢历史。"""
    e = ledger.Entry.from_json({"prompt": "x", "workflow": "txt2img_seedream", "backend": "comfyui"})
    assert e.recipe == "txt2img_seedream"
    assert e.backend == "comfyui"          # 历史值原样保留，不改写


def test_rel_makes_paths_relative(tmp_ledger):
    """out/ref 路径规范化为相对项目根的 POSIX 串。"""
    abs_out = ledger.settings.project_root / "data" / "skills" / "ai_drawing" / "assets" / "z.png"
    e = ledger.record(prompt="p", out=abs_out)
    assert e.out == "data/skills/ai_drawing/assets/z.png"
    assert "\\" not in e.out  # POSIX 分隔


def test_query_filter_backend(tmp_ledger):
    ledger.record(prompt="p1", backend="ark")
    ledger.record(prompt="p2", backend="imagegen")
    hits = ledger.query(backend="ark")
    assert len(hits) == 1
    assert hits[0].prompt == "p1"
    # 大小写不敏感
    assert len(ledger.query(backend="ARK")) == 1


def test_query_filter_model_substring(tmp_ledger):
    ledger.record(prompt="p", model="doubao-seedream-4-0-250828")
    ledger.record(prompt="q", model="other-model")
    assert len(ledger.query(model="seedream")) == 1


def test_query_contains(tmp_ledger):
    ledger.record(prompt="a red nebula cover", notes="draft")
    ledger.record(prompt="blue grid")
    assert len(ledger.query(contains="nebula")) == 1
    assert len(ledger.query(contains="DRAFT")) == 1  # notes 子串、大小写不敏感


def test_query_limit_and_order(tmp_ledger):
    for i in range(5):
        ledger.record(prompt=f"p{i}")
    hits = ledger.query(limit=2)
    assert len(hits) == 2
    # 最新在前
    assert hits[0].prompt == "p4"


def test_stats(tmp_ledger):
    ledger.record(prompt="a", backend="ark", model="m1", seed=1)
    ledger.record(prompt="b", backend="ark", model="m2")
    ledger.record(prompt="c", backend="imagegen", model="m1", seed=3)
    st = ledger.stats()
    assert st["total"] == 3
    assert st["by_backend"] == {"ark": 2, "imagegen": 1}
    assert st["by_model"] == {"m1": 2, "m2": 1}
    assert st["with_seed"] == 2


def test_render_markdown_content(tmp_ledger):
    ledger.record(prompt="cover | art", backend="ark", model="m1", seed=7, out="o.png")
    md = ledger.render_markdown().read_text(encoding="utf-8")
    assert "# AI 绘图生成账本" in md
    assert "cover \\| art" in md  # 竖线转义
    assert "ark" in md
    # 账本头部必须诚实声明它不是复现手段
    assert "不是复现手段" in md


def test_render_markdown_idempotent(tmp_ledger):
    """全量重渲染（非追加）→ 反复调不会让 MD 里出现重复行。"""
    ledger.record(prompt="p1", backend="ark")
    ledger.render_markdown()
    ledger.render_markdown()
    md = ledger.render_markdown().read_text(encoding="utf-8")
    assert md.count("p1") == 1


def test_render_markdown_empty(tmp_ledger):
    md = ledger.render_markdown().read_text(encoding="utf-8")
    assert "账本为空" in md


def test_load_all_skips_bad_lines(tmp_ledger):
    ledger.record(prompt="good")
    # 手动追加一行坏 JSON → 应被跳过而非整本失败
    with ledger.manifest_path().open("a", encoding="utf-8") as f:
        f.write("{not json}\n")
    entries = ledger.load_all()
    assert len(entries) == 1
    assert entries[0].prompt == "good"


def test_entry_from_json_tolerates_extra_keys():
    e = ledger.Entry.from_json({"prompt": "x", "unknown_field": 1, "seed": 5})
    assert e.prompt == "x"
    assert e.seed == 5


def test_format_table_empty(tmp_ledger):
    assert ledger.format_table([]) == "(账本为空)"


def test_format_table_renders(tmp_ledger):
    ledger.record(prompt="hello", backend="ark", model="m1", seed=1, out="a.png")
    txt = ledger.format_table(ledger.query())
    assert "hello" in txt
    assert "ark" in txt


# ---------------------------------------------------------------------------
# PNG tEXt 元数据内嵌（脱离账本也能溯源）
# ---------------------------------------------------------------------------
def test_embed_and_read_metadata_roundtrip(tmp_path):
    p = tmp_path / "x.png"
    Image.new("RGB", (8, 8), (10, 20, 30)).save(p)
    meta = {"prompt": "a red square", "model": "doubao-seedream-4-0-250828", "kind": "gen"}
    assert ledger.embed_metadata(p, meta) is True
    got = ledger.read_metadata(p)
    assert got == meta
    # 像素不能被写坏
    with Image.open(p) as im:
        assert im.size == (8, 8) and im.mode == "RGB"
        assert im.getpixel((0, 0)) == (10, 20, 30)


def test_embed_metadata_prefixes_keys(tmp_path):
    """键带 ``pysci:`` 前缀，与 ComfyUI/A1111 的元数据约定共存不撞名。"""
    p = tmp_path / "x.png"
    Image.new("RGB", (4, 4)).save(p)
    ledger.embed_metadata(p, {"prompt": "hi"})
    with Image.open(p) as im:
        assert f"{ledger.META_PREFIX}:prompt" in im.info


def test_embed_metadata_is_idempotent(tmp_path):
    """重复写入同一键不会累加多份（旧值被新值取代）。"""
    p = tmp_path / "x.png"
    Image.new("RGB", (4, 4)).save(p)
    ledger.embed_metadata(p, {"prompt": "v1"})
    ledger.embed_metadata(p, {"prompt": "v2"})
    assert ledger.read_metadata(p)["prompt"] == "v2"


def test_embed_metadata_skips_non_png(tmp_path):
    """JPEG 不写（重存会二次压缩，不划算）→ 返回 False 而非抛异常。"""
    p = tmp_path / "x.jpg"
    Image.new("RGB", (8, 8)).save(p)
    before = p.read_bytes()
    assert ledger.embed_metadata(p, {"prompt": "hi"}) is False
    assert p.read_bytes() == before
    assert ledger.read_metadata(p) == {}


def test_embed_metadata_missing_file_is_false(tmp_path):
    """元数据是兼保手段，不应因文件不在而让出图流程失败。"""
    assert ledger.embed_metadata(tmp_path / "nope.png", {"prompt": "x"}) is False
    assert ledger.read_metadata(tmp_path / "nope.png") == {}


def test_read_metadata_ignores_foreign_keys(tmp_path):
    """只读回自己前缀的键（方舟/其它工具写的文本块不当成我们的元数据）。"""
    from PIL import PngImagePlugin

    p = tmp_path / "x.png"
    info = PngImagePlugin.PngInfo()
    info.add_text("Software", "ark")
    Image.new("RGB", (4, 4)).save(p, pnginfo=info)
    assert ledger.read_metadata(p) == {}
    ledger.embed_metadata(p, {"prompt": "hi"})
    with Image.open(p) as im:
        assert im.info.get("Software") == "ark"    # 原有文本块被保留
    assert ledger.read_metadata(p) == {"prompt": "hi"}
