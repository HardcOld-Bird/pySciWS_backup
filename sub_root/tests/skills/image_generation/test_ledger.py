"""生成账本的机械断言：manifest 往返 / 过滤 / 统计 / PNG tEXt 溯源往返。"""

from __future__ import annotations

import json

from PIL import Image

from pysci.skills.image_generation import ledger


def _png(tmp_path, name="out.png"):
    p = tmp_path / name
    Image.new("RGB", (8, 8), (200, 100, 50)).save(p, format="PNG")
    return p


def test_record_load_roundtrip(tmp_path):
    """record → load_all 往返：字段保留、追加写、坏行跳过不整本失败。"""
    base = tmp_path
    e1 = ledger.record(
        prompt="a red circle", seed=7, model="doubao-seedream-5-5", provider="ark",
        out=tmp_path / "a.png", notes="first", base_dir=base,
    )
    ledger.record(
        prompt="blue square", model="Qwen/Qwen-Image", provider="siliconflow",
        out=tmp_path / "b.png", base_dir=base,
    )
    # 坏行注入（模拟截断/手改）
    manifest = ledger.manifest_path(base)
    with manifest.open("a", encoding="utf-8") as f:
        f.write("{broken json\n")

    entries = ledger.load_all(base)
    assert len(entries) == 2
    assert entries[0].prompt == "a red circle"
    assert entries[0].seed == 7
    assert entries[0].ts == e1.ts
    assert entries[0].provider == "ark"
    assert entries[1].seed is None


def test_query_and_stats(tmp_path):
    ledger.record(prompt="cat diagram", provider="ark", model="m1", base_dir=tmp_path)
    ledger.record(prompt="dog figure", provider="siliconflow", model="m2", base_dir=tmp_path)
    ledger.record(prompt="cat again", provider="ark", model="m1", base_dir=tmp_path)

    assert len(ledger.query(provider="ark", base_dir=tmp_path)) == 2
    assert len(ledger.query(model="m2", base_dir=tmp_path)) == 1
    assert len(ledger.query(contains="CAT", base_dir=tmp_path)) == 2
    assert len(ledger.query(limit=1, base_dir=tmp_path)) == 1
    # 最新在前
    assert ledger.query(base_dir=tmp_path)[0].prompt == "cat again"

    st = ledger.stats(tmp_path)
    assert st["total"] == 3
    assert st["by_provider"] == {"ark": 2, "siliconflow": 1}
    assert st["by_model"] == {"m1": 2, "m2": 1}


def test_png_text_metadata_roundtrip(tmp_path):
    """PNG tEXt 溯源：prompt/seed/模型/时间写入 → read_metadata 原样读回。"""
    p = _png(tmp_path)
    meta = {
        "prompt": "a red circle",
        "seed": "7",
        "model": "doubao-seedream-5-5",
        "ts": "2026-11-17T12:00:00",
    }
    assert ledger.embed_metadata(p, meta) is True
    back = ledger.read_metadata(p)
    assert back == meta
    # 覆盖更新不重复堆积
    ledger.embed_metadata(p, {"prompt": "v2"})
    assert ledger.read_metadata(p)["prompt"] == "v2"
    assert ledger.read_metadata(p)["seed"] == "7"


def test_embed_metadata_preserves_foreign_text(tmp_path):
    """已有非 pysci 文本块保留（与 ComfyUI/A1111 元数据约定共存）。"""
    p = tmp_path / "foreign.png"
    img = Image.new("RGB", (4, 4))
    from PIL import PngImagePlugin

    info = PngImagePlugin.PngInfo()
    info.add_text("parameters", "A1111-style metadata")
    img.save(p, format="PNG", pnginfo=info)
    assert ledger.embed_metadata(p, {"model": "m"}) is True
    back = ledger.read_metadata(p)
    assert back["model"] == "m"
    with Image.open(p) as im:
        assert im.info["parameters"] == "A1111-style metadata"


def test_embed_metadata_non_png_or_missing(tmp_path):
    assert ledger.embed_metadata(tmp_path / "nope.png", {"a": "1"}) is False
    jpg = tmp_path / "x.jpg"
    Image.new("RGB", (4, 4)).save(jpg, format="JPEG")
    assert ledger.embed_metadata(jpg, {"a": "1"}) is False
    assert ledger.read_metadata(tmp_path / "nope.png") == {}


def test_provenance_for(tmp_path):
    """按产物路径找回账本记录（多条命中取最后）。"""
    out = tmp_path / "art.png"
    ledger.record(prompt="v1", model="m1", provider="ark", out=out, base_dir=tmp_path)
    ledger.record(prompt="v2", model="m2", provider="ark", out=out, base_dir=tmp_path)
    prov = ledger.provenance_for(out, base_dir=tmp_path)
    assert prov["prompt"] == "v2"
    assert prov["model"] == "m2"
    assert ledger.provenance_for(tmp_path / "unknown.png", base_dir=tmp_path) == {}


def test_manifest_is_jsonl_append_only(tmp_path):
    """manifest.jsonl：一行一条 JSON（追加写语义）。"""
    ledger.record(prompt="x", base_dir=tmp_path)
    ledger.record(prompt="y", base_dir=tmp_path)
    lines = ledger.manifest_path(tmp_path).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["prompt"] == "x"
    assert json.loads(lines[1])["prompt"] == "y"
