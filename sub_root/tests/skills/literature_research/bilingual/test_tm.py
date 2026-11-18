"""TM 回填回归：决策层（精确优先/模糊/占位拒收）+ 旧工作区提取 + 自研薄 TM 存储层接缝。

决策层是纯函数（注入 lookup），始终受测；薄 TM 存储层（SQLite + RapidFuzz）缺
rapidfuzz 时集成用例跳过，装好后自动激活。旧 --reuse 期望值平移：等值回填/占位
不回填/未匹配保持未译。存储层集成面覆盖：等值回填/模糊 ≥75% 回填/低于阈值不回填/
占位不入库/持久化往返。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pysci.skills.literature_research.bilingual import tm

from .conftest import RAPIDFUZZ_READY


def fake_lookup(pairs: list[tuple[str, str, int]]):
    """由 (source, target, quality) 三元组构造 lookup（quality=模拟相似度百分数）。"""

    def lookup(en: str) -> list[tm.TmMatch]:
        out = []
        for s, t, q in pairs:
            if s == en or tm._norm(s) == tm._norm(en):
                out.append(tm.TmMatch(quality=100, source=s, target=t))
            else:
                out.append(tm.TmMatch(quality=q, source=s, target=t))
        return sorted(out, key=lambda m: -m.quality)

    return lookup


# ---------------------------------------------------------------------------
# 回填决策（纯函数）
# ---------------------------------------------------------------------------
def test_backfill_prefers_exact_over_fuzzy() -> None:
    """等值命中优先于高相似度模糊命中（升级但不改旧语义：等值仍是第一口径）。"""
    lookup = fake_lookup(
        [
            ("Hello world!", "模糊命中", 90),
            ("Hello  world.", "精确命中", 50),
        ]
    )
    res = tm.backfill_units([("S0001", "Hello world.")], lookup)
    assert res.filled == {"S0001": "精确命中"}
    assert res.exact == 1 and res.fuzzy == 0


def test_backfill_fuzzy_above_threshold() -> None:
    """无等值时模糊命中（≥阈值）回填，计入 fuzzy 计数。"""
    lookup = fake_lookup([("Hello there world.", "模糊命中", 80)])
    res = tm.backfill_units([("S0001", "Hello world.")], lookup)
    assert res.filled == {"S0001": "模糊命中"}
    assert res.fuzzy == 1 and res.exact == 0


def test_backfill_below_threshold_stays_untranslated() -> None:
    """模糊低于阈值不回填——未匹配保持未译（宁缺勿滥）。"""
    lookup = fake_lookup([("Totally different text.", "不该回填", 40)])
    res = tm.backfill_units([("S0001", "Hello world.")], lookup)
    assert res.filled == {}
    assert res.unmatched == ["S0001"]


def test_backfill_rejects_placeholder_candidates() -> None:
    """占位假译候选防御性拒收（不入 TM 是第一道，这里是第二道）；同文候选也不回填。"""
    lookup = fake_lookup(
        [
            ("Hello world.", "假译：Hello world.", 100),
            ("Hello world.", "Hello world.", 100),
        ]
    )
    res = tm.backfill_units([("S0001", "Hello world.")], lookup)
    assert res.filled == {}
    assert res.skipped_placeholder == ["S0001"]
    assert res.unmatched == ["S0001"]


def test_backfill_skips_placeholder_but_uses_real_candidate() -> None:
    """同查询下占位被拒、真实译文照常回填。"""
    lookup = fake_lookup(
        [
            ("Hello world.", "假译：x", 100),
            ("Hello world.", "真实译文", 100),
        ]
    )
    res = tm.backfill_units([("S0001", "Hello world.")], lookup)
    assert res.filled == {"S0001": "真实译文"}
    assert res.skipped_placeholder == ["S0001"]


# ---------------------------------------------------------------------------
# 旧工作区提取（--reuse 源：work.md + manifest.json）
# ---------------------------------------------------------------------------
def make_old_workspace(root: Path) -> Path:
    """合成旧工作区：2 行真实译文 + 1 行占位 + 1 行未译。"""
    old = root / "old"
    old.mkdir()
    (old / "manifest.json").write_text(
        '{"units": ['
        '{"id": "H0001", "en": "Title"}, '
        '{"id": "S0001", "en": "First sentence."}, '
        '{"id": "S0002", "en": "Second sentence."}, '
        '{"id": "S0003", "en": "Third sentence."}]}',
        encoding="utf-8",
    )
    (old / "work.md").write_text(
        "# 翻译规则：x\n"
        "H0001\t真译标题\n"
        "S0001\tFirst sentence.\n"
        "S0002\t假译：S0002\n"
        "S0003\t真译第三句\n",
        encoding="utf-8",
    )
    return old


def test_load_reuse_source_extracts_real_pairs_only(tmp_path: Path) -> None:
    """只收真实译文对；占位与未改英文行不入 TM（防夹具扩散）。"""
    src = tm.load_reuse_source(make_old_workspace(tmp_path))
    assert sorted(src.pairs) == [("Third sentence.", "真译第三句"), ("Title", "真译标题")]
    assert src.skipped_placeholder == ["S0002"]


def test_load_reuse_source_missing_work_md(tmp_path: Path) -> None:
    """源 work.md 不存在：FileNotFoundError（prepare 报 --reuse 拒绝）。"""
    with pytest.raises(FileNotFoundError, match="work.md"):
        tm.load_reuse_source(tmp_path / "nope")


# ---------------------------------------------------------------------------
# 自研薄 TM 存储层接缝（SQLite + RapidFuzz；rapidfuzz 缺件跳过）
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not RAPIDFUZZ_READY, reason="rapidfuzz（薄 TM 模糊引擎）待依赖代理安装")
def test_tm_exact_backfill(tmp_path: Path) -> None:
    """薄 TM 等值回填：入库 → 等值命中（quality=100）→ exact 计数。"""
    memory = tm.TranslationMemory(tmp_path / "tm.sqlite3")
    memory.add_pairs([("Hello world.", "你好世界。")])
    res = memory.backfill([("S0001", "Hello world.")])
    assert res.filled == {"S0001": "你好世界。"}
    assert res.exact == 1 and res.fuzzy == 0


@pytest.mark.skipif(not RAPIDFUZZ_READY, reason="rapidfuzz（薄 TM 模糊引擎）待依赖代理安装")
def test_tm_fuzzy_backfill_above_threshold(tmp_path: Path) -> None:
    """模糊 ≥75% 回填："A second example."/"A second example!"（fuzz.ratio≈94）。"""
    memory = tm.TranslationMemory(tmp_path / "tm.sqlite3")
    memory.add_pairs([("A second example.", "第二个例子。")])
    hits = memory.lookup("A second example!")
    assert hits and hits[0].quality >= tm.DEFAULT_MIN_SIMILARITY
    res = memory.backfill([("S0001", "A second example!")])
    assert res.filled == {"S0001": "第二个例子。"}
    assert res.fuzzy == 1 and res.exact == 0


@pytest.mark.skipif(not RAPIDFUZZ_READY, reason="rapidfuzz（薄 TM 模糊引擎）待依赖代理安装")
def test_tm_below_threshold_not_filled(tmp_path: Path) -> None:
    """低于阈值不回填：查询与库内句对无重叠，候选为空 → 保持未译。"""
    memory = tm.TranslationMemory(tmp_path / "tm.sqlite3")
    memory.add_pairs([("Hello world.", "你好世界。")])
    res = memory.backfill([("S0001", "Totally different text with no overlap.")])
    assert res.filled == {}
    assert res.unmatched == ["S0001"]


@pytest.mark.skipif(not RAPIDFUZZ_READY, reason="rapidfuzz（薄 TM 模糊引擎）待依赖代理安装")
def test_tm_placeholder_not_stored(tmp_path: Path) -> None:
    """占位假译不入库：入库侧直接拒收、查询侧不可见（第一道防线）。"""
    memory = tm.TranslationMemory(tmp_path / "tm.sqlite3")
    added = memory.add_pairs([("Hello world.", "假译：Hello world."), ("Real sentence.", "真译。")])
    assert added == 1  # 占位被拒不计
    assert memory.lookup("Hello world.") == []
    assert [m.target for m in memory.lookup("Real sentence.")] == ["真译。"]


@pytest.mark.skipif(not RAPIDFUZZ_READY, reason="rapidfuzz（薄 TM 模糊引擎）待依赖代理安装")
def test_tm_persistence_roundtrip(tmp_path: Path) -> None:
    """持久化往返：跨实例（同库文件）沉淀句对仍等值命中；同源同译幂等去重。"""
    db = tmp_path / "tm.sqlite3"
    first = tm.TranslationMemory(db)
    assert first.add_pairs([("Hello world.", "你好世界。")]) == 1
    assert first.add_pairs([("Hello world.", "你好世界。")]) == 0  # UNIQUE 幂等
    second = tm.TranslationMemory(db)
    res = second.backfill([("S0001", "Hello world.")])
    assert res.filled == {"S0001": "你好世界。"}
    assert res.exact == 1
