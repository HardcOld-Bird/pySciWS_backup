"""TM 回填：SQLite 句对库 + RapidFuzz 模糊匹配（升级旧 --reuse 英文原文等值回填）。

ADR（2026-11-17）：旧 ``--reuse`` 只做「归一化空白后英文原文等值」匹配（无模糊）。
吸收实施升级为 **TM（翻译记忆）回填**——译文沉淀进 ``bilingual/tm.sqlite3``，
prepare 时精确优先、模糊（相似度 ≥ :data:`DEFAULT_MIN_SIMILARITY`）次优回填，
占位假译不入 TM。

ADR（2026-11-17，用户裁决方案 B——存储层自研）：原接缝 translate-toolkit
``translate.storage.tmdb`` 自上游 **3.19 移除**（"tmserver and related code was
removed"），守着它只能 pin ≤3.18、锁死依赖升级。用户裁决：translate-toolkit 升级
3.20.6（仅保留其 TMX/TBX **格式层**，将来做 TM 导入导出），TM **存储层自研薄胶水**
——stdlib ``sqlite3`` 句对库 + RapidFuzz 模糊匹配（:class:`TranslationMemory`
~100 行），RapidFuzz 免费（MIT）且直用直声明，维护面小于死守已删模块。依赖缺失报
:class:`TmDependencyError`，不静默回退。

决策层（:func:`backfill_units`）是纯函数（注入 lookup 即可测），存储层
（:class:`TranslationMemory`）是唯一接缝——换存储不动决策。
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .manifest import read_work_translations
from .reconcile import is_placeholder_translation

#: 模糊匹配相似度下限（RapidFuzz ``fuzz.ratio`` 口径，百分数）。
DEFAULT_MIN_SIMILARITY = 75

#: TM 持久库文件名（工作区根下，跨会话沉淀）。
TM_DB_NAME = "tm.sqlite3"


class TmDependencyError(RuntimeError):
    """RapidFuzz（TM 模糊匹配引擎）不可用——由依赖代理安装，勿静默回退。"""


@dataclass(frozen=True)
class TmMatch:
    """一条 TM 命中（存储层候选；quality=相似度百分数，等值命中=100）。"""

    quality: int
    source: str
    target: str


@dataclass
class BackfillResult:
    """回填决策结果（纯函数输出，CLI 只负责打印）。"""

    filled: dict[str, str] = field(default_factory=dict)
    exact: int = 0
    fuzzy: int = 0
    skipped_placeholder: list[str] = field(default_factory=list)
    unmatched: list[str] = field(default_factory=list)


@dataclass
class ReuseSource:
    """旧工作区提取结果：真实译文对 + 被拒的占位假译 ID（告警留痕）。"""

    pairs: list[tuple[str, str]] = field(default_factory=list)
    skipped_placeholder: list[str] = field(default_factory=list)


def _norm(text: str) -> str:
    """归一化空白（等值匹配口径）。"""
    return re.sub(r"\s+", " ", str(text)).strip()


def backfill_units(
    units: Iterable[tuple[str, str]],
    lookup: Callable[[str], list[TmMatch]],
    *,
    min_quality: int = DEFAULT_MIN_SIMILARITY,
) -> BackfillResult:
    """逐单元 TM 回填决策：精确优先、模糊次优；占位假译/同文不回填。

    Args:
        units: ``(unit_id, en)`` 序列。
        lookup: ``en → 候选列表``（相似度降序；source=记忆库英文、target=旧译文）。
        min_quality: 模糊命中的相似度下限（百分数）。

    Returns:
        :class:`BackfillResult`——``filled`` 为 ``{unit_id: 旧译文}``，未匹配 ID 在
        ``unmatched``（保持未译待补）。
    """
    result = BackfillResult()
    for uid, en in units:
        hits = lookup(en)
        # 占位假译候选（TM 里的历史污染）防御性拒收——正常路径下入库时已拦。
        if any(is_placeholder_translation(m.target, m.source) for m in hits):
            result.skipped_placeholder.append(uid)
        candidates = [
            m
            for m in hits
            if not is_placeholder_translation(m.target, m.source)
            and m.target.strip() != en.strip()
        ]
        if not candidates:
            result.unmatched.append(uid)
            continue
        exact = [m for m in candidates if _norm(m.source) == _norm(en)]
        if exact:
            result.filled[uid] = exact[0].target
            result.exact += 1
            continue
        best = max(candidates, key=lambda m: m.quality)
        if best.quality >= min_quality:
            result.filled[uid] = best.target
            result.fuzzy += 1
        else:
            result.unmatched.append(uid)
    return result


def load_reuse_source(src: str | Path) -> ReuseSource:
    """从旧工作区提取 ``(英文原文, 旧译文)`` 对（只收真实译文）。

    源可以是旧 work.md 文件、工作区目录或 stem（工作区根下解析）；英文键取同目录
    ``manifest.json`` 的 ``en``（无 manifest 时行尾自键——此时已译行无键可用）。
    占位假译（假译类）与未改英文行**不入 TM**（防夹具扩散），占位 ID 记入
    ``skipped_placeholder`` 供告警。

    Raises:
        FileNotFoundError: 源 work.md 不存在。
    """
    p = Path(src)
    if p.is_dir():
        work_md = p / "work.md"
    elif p.is_file():
        work_md = p
    elif Path(str(src)).name == str(src):
        from .manifest import workdir_for

        work_md = workdir_for(str(src)) / "work.md"
    else:  # 带路径形态但文件不存在（绝对/相对路径都直接报缺源，勿当 stem 解析）
        raise FileNotFoundError(f"--reuse 源 work.md 不存在：{p}")
    if not work_md.is_file():
        raise FileNotFoundError(f"--reuse 源 work.md 不存在：{work_md}")
    old_rows = read_work_translations(work_md)
    en_of: dict[str, str] = {}
    man = work_md.parent / "manifest.json"
    if man.is_file():
        data = json.loads(man.read_text(encoding="utf-8"))
        en_of = {str(u.get("id")): str(u.get("en", "")) for u in data.get("units", [])}
    out = ReuseSource()
    for uid, tail in old_rows.items():
        en = en_of.get(uid, tail)
        if not tail.strip() or tail.strip() == en.strip():
            continue
        if is_placeholder_translation(tail, en):
            out.skipped_placeholder.append(uid)
            continue
        out.pairs.append((en, tail))
    return out


class TranslationMemory:
    """自研薄 TM 存储层：stdlib SQLite 句对库 + RapidFuzz 模糊匹配（存储接缝）。

    ADR（2026-11-17，用户裁决方案 B）：原接缝 ``translate.storage.tmdb`` 自
    translate-toolkit 3.19 被上游移除，自研替代——理由详见模块 docstring。匹配
    口径与决策层对齐：等值优先（:func:`_norm` 空白归一）、模糊 ``fuzz.ratio``
    ≥ :attr:`min_similarity`、占位假译不入库。**表结构**：单表 ``tm_pairs``
    （``source``/``target`` 原文对 + ``source_norm`` 等值匹配键；
    ``UNIQUE(source_norm, target)`` 幂等去重）。规模假设：个人 TM 万级以内，
    模糊匹配全表扫描 + RapidFuzz C 层足够，增长后再议候选预过滤。

    Args:
        db_path: SQLite 文件路径（不存在则创建）。
        min_similarity: 模糊候选相似度下限（百分数；决策层 ``min_quality`` 是
            同口径第二道阈值）。
        max_candidates: 每次查询返回的候选上限（等值优先，其余按模糊降序）。

    Raises:
        TmDependencyError: RapidFuzz 不可用（由依赖代理安装，勿静默回退）。
    """

    def __init__(
        self,
        db_path: Path,
        *,
        min_similarity: int = DEFAULT_MIN_SIMILARITY,
        max_candidates: int = 3,
    ) -> None:
        try:
            from rapidfuzz import fuzz
        except ImportError as exc:  # pragma: no cover - 依赖代理安装后消失
            raise TmDependencyError(
                "rapidfuzz 不可用（依赖代理负责：uv add rapidfuzz）"
            ) from exc
        self._ratio = fuzz.ratio
        self.min_similarity = min_similarity
        self.max_candidates = max_candidates
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tm_pairs (
                id INTEGER PRIMARY KEY,
                source TEXT NOT NULL,
                target TEXT NOT NULL,
                source_norm TEXT NOT NULL,
                UNIQUE (source_norm, target)
            )
            """
        )
        self._conn.commit()

    def add_pairs(self, pairs: Iterable[tuple[str, str]]) -> int:
        """批量入 TM（``(英文原文, 译文)``）；返回**新入 TM 条数**（幂等去重后）。

        同源同译由 ``UNIQUE(source_norm, target)`` 去重；占位假译与空译文不入库
        （入库侧第一道防线，决策层 :func:`backfill_units` 对历史污染还有第二道）。
        """
        accepted = [
            (en, zh, _norm(en))
            for en, zh in pairs
            if zh.strip() and not is_placeholder_translation(zh, en)
        ]
        if not accepted:
            return 0
        cur = self._conn.executemany(
            "INSERT OR IGNORE INTO tm_pairs (source, target, source_norm)"
            " VALUES (?, ?, ?)",
            accepted,
        )
        self._conn.commit()
        return cur.rowcount

    def add_from_workspace(self, src: str | Path) -> ReuseSource:
        """把旧工作区的真实译文对沉淀进 TM；返回提取结果（含占位跳过清单）。"""
        reuse = load_reuse_source(src)
        self.add_pairs(reuse.pairs)
        return reuse

    def lookup(self, en: str) -> list[TmMatch]:
        """查询候选（等值优先，其余模糊降序；共 :attr:`max_candidates` 条上限）。"""
        qn = _norm(en)
        exact = [
            TmMatch(quality=100, source=s, target=t)
            for s, t in self._conn.execute(
                "SELECT source, target FROM tm_pairs WHERE source_norm = ?", (qn,)
            )
        ]
        fuzzy: list[TmMatch] = []
        for s, t, sn in self._conn.execute(
            "SELECT source, target, source_norm FROM tm_pairs WHERE source_norm != ?",
            (qn,),
        ):
            quality = int(round(self._ratio(qn, sn)))
            if quality >= self.min_similarity:
                fuzzy.append(TmMatch(quality=quality, source=s, target=t))
        fuzzy.sort(key=lambda m: -m.quality)
        return (exact + fuzzy)[: self.max_candidates]

    def backfill(self, units: Iterable[tuple[str, str]], **kw: Any) -> BackfillResult:
        """查询 + 决策一体（:func:`backfill_units` 的绑定方法）。"""
        return backfill_units(units, self.lookup, **kw)
