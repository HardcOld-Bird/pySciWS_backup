"""bilingual 测试共享垫片：确定性参照切分器、工作区隔离、Fake TM。

设计：切分契约（哨兵/游标/句末合并）与切分器无关，固定样例断言注入**确定性参照
切分器**（句末符 ``. ? !`` 后断句），wtpsplit 真接缝另行 importorskip 验证——
依赖（wtpsplit / rapidfuzz）未装时套件仍全绿，装好后自动加深。
"""

from __future__ import annotations

import importlib.util
import re
from collections.abc import Callable
from difflib import SequenceMatcher
from pathlib import Path

import pytest

from pysci import paths
from pysci.skills.literature_research.bilingual import images, tm, units

#: 参照切分器：对**已加哨兵**文本在句末符（``. ? !`` 后接空白/行尾）处断句。
_BOUNDARY_RE = re.compile(r"[.?!](?=\s|$)")


def reference_fragmenter(protected: str) -> list[str]:
    """确定性切分参照实现（测试注入用；生产默认是 wtpsplit，见 units.default_segmenter）。"""
    frags: list[str] = []
    last = 0
    for m in _BOUNDARY_RE.finditer(protected):
        frags.append(protected[last : m.end()])
        last = m.end()
    if last < len(protected):
        frags.append(protected[last:])
    return frags


#: 真 wtpsplit 接缝（monkeypatch 前原件；wtpsplit 就绪时的接缝验证用）。
REAL_DEFAULT_SEGMENTER: Callable[[str], list[str]] = units.default_segmenter

#: wtpsplit + vendored 模型/tokenizer 是否就绪（缺件时真接缝用例跳过）。
_MODEL_ROOT = units.wtpsplit_model_dir()
WTPSPLIT_READY = importlib.util.find_spec("wtpsplit") is not None and (
    (_MODEL_ROOT / "config.json").is_file()
    or any((p / "config.json").is_file() for p in _MODEL_ROOT.glob("*/"))
) and (_MODEL_ROOT / "xlm-roberta-base-tokenizer").is_dir()

#: 自研薄 TM 存储层的模糊引擎 rapidfuzz 是否就绪（缺件时真接缝用例跳过）。
RAPIDFUZZ_READY = importlib.util.find_spec("rapidfuzz") is not None


@pytest.fixture(autouse=True)
def _ref_split(monkeypatch: pytest.MonkeyPatch) -> None:
    """全部用例默认走参照切分器（与 wtpsplit 解耦；真接缝用例显式用原件）。"""
    monkeypatch.setattr(units, "default_segmenter", reference_fragmenter)


class FakeMemory:
    """tm.TranslationMemory 的离线替身（同一决策层 :func:`tm.backfill_units`）。"""

    def __init__(self, db_path: Path, **_kw: object) -> None:
        self._pairs: list[tuple[str, str]] = []

    def add_from_workspace(self, src: str | Path) -> tm.ReuseSource:
        reuse = tm.load_reuse_source(src)
        self._pairs.extend(reuse.pairs)
        return reuse

    def lookup(self, en: str) -> list[tm.TmMatch]:
        out = [
            tm.TmMatch(
                quality=(
                    100
                    if tm._norm(s) == tm._norm(en)
                    else int(SequenceMatcher(None, s, en).ratio() * 100)
                ),
                source=s,
                target=t,
            )
            for s, t in self._pairs
        ]
        return sorted(out, key=lambda m: -m.quality)

    def backfill(self, units_: object, **kw: object) -> tm.BackfillResult:
        return tm.backfill_units(units_, self.lookup, **kw)


@pytest.fixture
def fake_tm(monkeypatch: pytest.MonkeyPatch) -> None:
    """把 TM 存储层换成离线替身（决策层照常受测）。"""
    monkeypatch.setattr(tm, "TranslationMemory", FakeMemory)


@pytest.fixture
def ws(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """工作区隔离：LITERATURE_ROOT/DATA_ROOT/INBOX_ROOT 全重定向进 tmp_path；回补打桩。

    布局镜像生产（LITERATURE_ROOT 位于 DATA_ROOT 内，assert_within_data 护栏照常
    生效）；INBOX_ROOT 隔离保证测试产物绝不落入用户全权管理区（AGENTS.md 内容
    目录政策：中间产物严禁进 用户的inbox）。
    """
    lit = tmp_path / "data" / "skills" / "literature_research"
    (lit / "papers").mkdir(parents=True)
    (lit / "bilingual").mkdir(parents=True)
    monkeypatch.setattr(paths, "LITERATURE_ROOT", lit)
    monkeypatch.setattr(paths, "DATA_ROOT", tmp_path / "data")
    monkeypatch.setattr(paths, "INBOX_ROOT", tmp_path / "用户的inbox")

    def fake_backfill(
        workdir: Path,
        refs: list[str],
        source_pdf: Path | None,
        *,
        no_images: bool,
        source_md: Path | None = None,
        provider: object = None,
    ) -> None:
        if no_images or not refs:
            return
        img_dir = workdir / "images"
        img_dir.mkdir(parents=True, exist_ok=True)
        for r in refs:
            (img_dir / r.rsplit("/", 1)[-1]).write_bytes(b"fake-img")

    monkeypatch.setattr(images, "backfill_images", fake_backfill)
    return lit


def make_source(lit: Path, md: str, name: str = "mini.md") -> Path:
    """把合成 fixture 落成源 markdown（papers/ 提取件区），返回路径。"""
    src = lit / "papers" / name
    src.write_text(md, encoding="utf-8")
    return src


def fake_translations(workdir: Path) -> dict[str, str]:
    """把 work.md 每个 ID 的英文行换成带 CJK 的假翻译（模拟翻译组员）。"""
    path = workdir / "work.md"
    out: dict[str, str] = {}
    lines = path.read_text(encoding="utf-8").split("\n")
    for k, line in enumerate(lines):
        uid, sep, _tail = line.partition("\t")
        if sep and re.match(r"^[A-Z]\d{4}\t", line):
            out[uid] = f"中译 {uid}"
            lines[k] = f"{uid}\t{out[uid]}"
    path.write_text("\n".join(lines), encoding="utf-8")
    return out
