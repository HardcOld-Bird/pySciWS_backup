"""引用核验门的机械对账：文内引用键 × 文献列表（refs.bib / 笔记条目）。

ADR（自研理由，2026-11-17）：旧项目 citation_verify 的三方元数据核验已由 paper-search-mcp 的
DOI 核验工具承载（调用方式见技能文档「引用核验门」），剩下的"写入前对账"是纯机械比对，
社区无对应薄件（pandoc-citeproc 只管排版渲染、不管核验门）。故自研本纯函数层。

分级合同（沿用旧项目验收口径）：``PASS ✓``（全对齐）/ ``WARN △``（仅孤儿条目）/
``FAIL ✗``（文内引用缺条目，写入前**拦截**）/ ``NOT_FOUND ?``（两列表皆空）。
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

_CITE_RE = re.compile(
    r"\\cite\w*\*?(?:\[[^\]]*\])*?\{([^}]*)\}"  # \cite / \citep / \citet / \citealp ...
    r"|\[@([A-Za-z0-9_:.+/-]+)\]"  # pandoc [@key]
    r"|(?<![\w@\\])@([A-Za-z0-9_][\w:.\-+]*)"  # pandoc bare @key
)
_BIB_KEY_RE = re.compile(r"^@\w+\s*\{\s*([^,\s]+)\s*,", re.MULTILINE)
_GRADE_MARKS = {"PASS": "✓", "WARN": "△", "FAIL": "✗", "NOT_FOUND": "?"}


def _dedup(keys: Iterable[str]) -> tuple[str, ...]:
    """按出现顺序去重并去掉尾随标点（pandoc ``@key.`` 句末点号不算键名）。"""
    out: list[str] = []
    for key in keys:
        key = key.strip().rstrip(".,;").strip()
        if key and key not in out:
            out.append(key)
    return tuple(out)


def extract_citation_keys(text: str) -> tuple[str, ...]:
    """抽取文内引用键（TeX ``\\cite{a,b}`` 与 pandoc ``[@key]`` / ``@key`` 两记法）。"""
    keys: list[str] = []
    for match in _CITE_RE.finditer(text):
        for group in match.groups():
            if group is not None:
                keys.extend(group.split(","))
    return _dedup(keys)


def extract_bib_keys(bib_text: str) -> tuple[str, ...]:
    """抽取 ``refs.bib`` 条目键（``@article{key,`` 形态）。"""
    return _dedup(m.group(1) for m in _BIB_KEY_RE.finditer(bib_text))


@dataclass(frozen=True)
class CiteReport:
    """对账结果；字段均按出现顺序去重。``missing`` 非空即 FAIL（写入拦截）。"""

    cited: tuple[str, ...]
    listed: tuple[str, ...]
    missing: tuple[str, ...]  # 文内引用但列表缺条目 → FAIL
    orphans: tuple[str, ...]  # 列表有条目但正文未引用 → WARN

    @property
    def grade(self) -> str:
        """分级：NOT_FOUND / FAIL / WARN / PASS（判定顺序即优先级）。"""
        if not self.cited and not self.listed:
            return "NOT_FOUND"
        if self.missing:
            return "FAIL"
        if self.orphans:
            return "WARN"
        return "PASS"

    def render(self) -> str:
        """审计输出（等级+标记+差异明细），随产物落盘。"""
        head = f"[{_GRADE_MARKS[self.grade]} {self.grade}] cited={len(self.cited)} listed={len(self.listed)}"
        lines = [head]
        lines += [f"  missing: {key}" for key in self.missing]
        lines += [f"  orphan: {key}" for key in self.orphans]
        return "\n".join(lines)


def reconcile(cited: Iterable[str], listed: Iterable[str]) -> CiteReport:
    """两键集合对账（已可机械判定的最小入口）。"""
    cited_keys, listed_keys = _dedup(cited), _dedup(listed)
    return CiteReport(
        cited=cited_keys,
        listed=listed_keys,
        missing=tuple(k for k in cited_keys if k not in listed_keys),
        orphans=tuple(k for k in listed_keys if k not in cited_keys),
    )


def reconcile_note(body: str, bib_text: str = "") -> CiteReport:
    """笔记级入口：正文抽引用键 × refs.bib 抽条目键 → 对账报告。"""
    return reconcile(extract_citation_keys(body), extract_bib_keys(bib_text))
