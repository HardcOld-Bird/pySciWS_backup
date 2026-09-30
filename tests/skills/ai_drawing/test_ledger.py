"""ledger 单测：记账 / 查询 / 统计 / Markdown 渲染。

用 dataclasses.replace 把账本目录重定向到 tmp_path（Settings 是 frozen dataclass），
绝不写真实技能数据区。
"""

from __future__ import annotations

import dataclasses

import pytest

from pysci.skills.ai_drawing.tools import ledger


@pytest.fixture
def tmp_ledger(tmp_path, monkeypatch):
    """把 ledger.settings.module_dir 指向 tmp_path（隔离账本文件）。"""
    monkeypatch.setattr(
        ledger, "settings", dataclasses.replace(ledger.settings, module_dir=tmp_path)
    )
    return tmp_path


def test_record_appends_manifest(tmp_ledger):
    e = ledger.record(prompt="a cover", seed=42, model="m1", backend="comfyui", out="x.png")
    assert e.prompt == "a cover"
    assert e.seed == 42
    assert ledger.manifest_path().is_file()
    assert ledger.ledger_path().is_file()
    entries = ledger.load_all()
    assert len(entries) == 1
    assert entries[0].backend == "comfyui"


def test_record_multiple_appends(tmp_ledger):
    ledger.record(prompt="p1", backend="comfyui")
    ledger.record(prompt="p2", backend="imagegen")
    assert len(ledger.load_all()) == 2


def test_rel_makes_paths_relative(tmp_ledger):
    """out/ref 路径规范化为相对项目根的 POSIX 串。"""
    abs_out = ledger.settings.project_root / "data" / "skills" / "ai_drawing" / "assets" / "z.png"
    e = ledger.record(prompt="p", out=abs_out)
    assert e.out == "data/skills/ai_drawing/assets/z.png"
    assert "\\" not in e.out  # POSIX 分隔


def test_query_filter_backend(tmp_ledger):
    ledger.record(prompt="p1", backend="comfyui")
    ledger.record(prompt="p2", backend="imagegen")
    hits = ledger.query(backend="comfyui")
    assert len(hits) == 1
    assert hits[0].prompt == "p1"
    # 大小写不敏感
    assert len(ledger.query(backend="COMFYUI")) == 1


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
    ledger.record(prompt="a", backend="comfyui", model="m1", seed=1)
    ledger.record(prompt="b", backend="comfyui", model="m2")
    ledger.record(prompt="c", backend="imagegen", model="m1", seed=3)
    st = ledger.stats()
    assert st["total"] == 3
    assert st["by_backend"] == {"comfyui": 2, "imagegen": 1}
    assert st["by_model"] == {"m1": 2, "m2": 1}
    assert st["with_seed"] == 2


def test_render_markdown_content(tmp_ledger):
    ledger.record(prompt="cover | art", backend="comfyui", model="m1", seed=7, out="o.png")
    md = ledger.ledger_path().read_text(encoding="utf-8")
    assert "# AI 绘图生成账本" in md
    assert "cover \\| art" in md  # 竖线转义
    assert "comfyui" in md


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
    ledger.record(prompt="hello", backend="comfyui", model="m1", seed=1, out="a.png")
    txt = ledger.format_table(ledger.query())
    assert "hello" in txt
    assert "comfyui" in txt
