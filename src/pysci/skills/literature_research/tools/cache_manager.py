"""cache_manager —— 文献调研模块的缓存治理层。

两层缓存模型
------------
- **Tier A 持久层**（``cache/pdfs``、``cache/extracted``、``cache/html_fulltext``）：
  重获代价高（MinerU 配额 / 浏览器爬取 / 下载），默认**永久保留、命中即复用**；
  仅 ``research cache prune`` 手动按 LRU 淘汰。
- **Tier B 易失层**（``cache/api_responses``）：廉价可重取的 JSON 元数据，TTL 过期即视为
  死文件；由 :func:`maybe_autoclean` 钩子定期清理，或 ``research cache clean`` 手动清理。

设计要点
--------
- Windows 上 atime 不可靠，故统一用 **mtime 近似"最近访问"**：每次缓存命中调用
  :func:`bump_mtime` touch 一下文件，使 prune 的 LRU 排序准确。
- **不引入常驻进程 / 数据库**：自动清理挂在命令入口（见 ``research.main``），靠一个小的
  ``cache/.autoclean_state.json`` 记录上次清理时间；未到期时仅一次小 JSON 读，开销可忽略。
- 所有清理/淘汰都提供 ``dry_run``，先列后删。

本模块只依赖 :mod:`config` 与标准库；各 client 通过 ``from .cache_manager import bump_mtime``
在命中处回touch，不构成循环导入。
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .config import settings

# ---------------------------------------------------------------------------
# 分层定义
# ---------------------------------------------------------------------------
#: Tier A —— 持久层（仅手动 prune）
TIER_A_DIRS: tuple[Path, ...] = (
    settings.cache_pdfs,
    settings.cache_extracted,
    settings.cache_html_fulltext,
)
#: Tier B —— 易失层（自动 + 手动 clean）
TIER_B_DIRS: tuple[Path, ...] = (settings.cache_api_responses,)

_STATE_PATH: Path = settings.cache_dir / ".autoclean_state.json"
_MB: int = 1024 * 1024


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------
def bump_mtime(path: Path | str) -> None:
    """把文件 mtime 更新为当前时间（缓存命中时调用，使 mtime≈最近访问，供 LRU 用）。

    失败静默忽略——touch 只是优化，绝不应影响主流程。
    """
    try:
        os.utime(Path(path), None)
    except OSError:
        pass


def _dir_files(d: Path) -> list[Path]:
    """递归列出目录下所有文件；目录不存在则返回空表。"""
    if not d.exists():
        return []
    return [p for p in d.rglob("*") if p.is_file()]


def _size_and_mtime(p: Path) -> tuple[int, float]:
    try:
        st = p.stat()
        return st.st_size, st.st_mtime
    except OSError:
        return 0, 0.0


def _fmt_size(nbytes: float) -> str:
    n = float(nbytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{int(n)} B" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _fmt_age(ts: float) -> str:
    if not ts:
        return "—"
    secs = time.time() - ts
    if secs < 3600:
        return f"{int(secs // 60)} 分钟前"
    if secs < 86400:
        return f"{int(secs // 3600)} 小时前"
    return f"{int(secs // 86400)} 天前"


# ---------------------------------------------------------------------------
# 被笔记引用的文件（prune --keep-referenced 时保护）
# ---------------------------------------------------------------------------
def _frontmatter_values(text: str, keys: set[str]) -> list[str]:
    """从 markdown 的 YAML frontmatter 尽力提取指定 key 的标量值（不依赖 PyYAML）。"""
    if not text.startswith("---"):
        return []
    end = text.find("\n---", 3)
    fm = text[3:end] if end != -1 else text[3:]
    out: list[str] = []
    for line in fm.splitlines():
        m = re.match(r"\s*([A-Za-z_]\w*)\s*:\s*(.*)$", line)
        if not m:
            continue
        k, v = m.group(1), m.group(2)
        if k in keys:
            v = v.split(" #", 1)[0].strip().strip('"').strip("'")
            if v:
                out.append(v)
    return out


def referenced_paths() -> set[Path]:
    """收集所有 ``papers/*.md`` frontmatter 引用的本地文件绝对路径。

    取 ``local_pdf_path`` 与 ``extracted_md_path`` 两个字段；同时收录其 resolve 形态，
    便于与实际文件比对。prune 的 ``keep_referenced`` 用它避免误删仍在用的产物。
    """
    papers = settings.module_dir / "papers"
    refs: set[Path] = set()
    if not papers.exists():
        return refs
    for note in papers.glob("*.md"):
        try:
            text = note.read_text(encoding="utf-8")
        except OSError:
            continue
        for raw in _frontmatter_values(text, {"local_pdf_path", "extracted_md_path"}):
            p = Path(raw)
            if not p.is_absolute():
                p = settings.project_root / p
            refs.add(p)
            try:
                refs.add(p.resolve())
            except OSError:
                pass
    return refs


def _is_referenced(p: Path, refs: set[Path]) -> bool:
    if not refs:
        return False
    if p in refs:
        return True
    try:
        return p.resolve() in refs
    except OSError:
        return False


# ---------------------------------------------------------------------------
# 统计
# ---------------------------------------------------------------------------
def _tier_summary(dirs: Iterable[Path]) -> dict[str, Any]:
    files: list[Path] = []
    for d in dirs:
        files.extend(_dir_files(d))
    total = 0
    oldest = 0.0
    newest = 0.0
    for p in files:
        sz, mt = _size_and_mtime(p)
        total += sz
        if mt:
            oldest = mt if not oldest or mt < oldest else oldest
            newest = mt if mt > newest else newest
    return {"files": len(files), "bytes": total, "oldest": oldest, "newest": newest}


def _expired_tier_b(max_age_days: int) -> list[Path]:
    cutoff = time.time() - max_age_days * 86400
    out: list[Path] = []
    for d in TIER_B_DIRS:
        for p in _dir_files(d):
            _, mt = _size_and_mtime(p)
            if mt and mt < cutoff:
                out.append(p)
    return out


def _read_state() -> dict[str, Any]:
    try:
        if _STATE_PATH.exists():
            return json.loads(_STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        pass
    return {}


def _write_state(last_autoclean: float) -> None:
    try:
        _STATE_PATH.write_text(
            json.dumps({"last_autoclean": last_autoclean}, ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError:
        pass


def stats() -> dict[str, Any]:
    """汇总两层缓存的占用 / 文件数 / 时效 / 上次自动清理时间，供 ``cache stats`` 展示。"""
    a = _tier_summary(TIER_A_DIRS)
    b = _tier_summary(TIER_B_DIRS)
    per_dir = {d.name: _tier_summary([d]) for d in (*TIER_A_DIRS, *TIER_B_DIRS)}
    state = _read_state()
    total_bytes = a["bytes"] + b["bytes"]
    soft = settings.cache_soft_limit_mb * _MB
    return {
        "root": settings.cache_dir,
        "tier_a": a,
        "tier_b": b,
        "per_dir": per_dir,
        "total_bytes": total_bytes,
        "soft_limit_bytes": soft,
        "over_soft_limit": total_bytes > soft,
        "expired_b": len(_expired_tier_b(settings.cache_b_max_age_days)),
        "last_autoclean": state.get("last_autoclean"),
    }


def format_stats(s: dict[str, Any]) -> str:
    """把 :func:`stats` 的结果渲染为可读文本块。"""
    lines = [
        "=== 缓存概览 (research cache stats) ===",
        f"缓存根 : {s['root']}",
        f"总占用 : {_fmt_size(s['total_bytes'])} / 软上限 {_fmt_size(s['soft_limit_bytes'])}"
        + ("  [已超软上限]" if s["over_soft_limit"] else ""),
        "",
        "[Tier A 持久层 — 仅手动 prune]",
    ]
    for name in ("pdfs", "extracted", "html_fulltext"):
        d = s["per_dir"].get(name)
        if d:
            lines.append(
                f"  {name:<14} {d['files']:>5} 文件   {_fmt_size(d['bytes']):>10}"
                f"   最新 {_fmt_age(d['newest'])}"
            )
    lines += ["", "[Tier B 易失层 — 自动清理超阈值]"]
    d = s["per_dir"].get("api_responses")
    if d:
        lines.append(
            f"  {'api_responses':<14} {d['files']:>5} 文件   {_fmt_size(d['bytes']):>10}"
            f"   其中 {s['expired_b']} 个已过期(>{settings.cache_b_max_age_days}d)"
        )
    last = s.get("last_autoclean")
    lines += [
        "",
        f"上次自动清理 : {(_fmt_age(last) if last else '从未')}"
        f"（间隔 {settings.cache_autoclean_interval_days} 天）",
    ]
    if s["over_soft_limit"]:
        lines.append(
            f"提示：Tier A 超软上限，可运行 research cache prune "
            f"--max-mb {settings.cache_soft_limit_mb}"
        )
    lines.append("=== 概览结束 ===")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tier B：清理（自动 + 手动）
# ---------------------------------------------------------------------------
def clean_tier_b(
    *,
    purge_all: bool = False,
    older_than_days: int | None = None,
    dry_run: bool = False,
) -> tuple[int, int, list[Path]]:
    """清理 Tier B（api_responses）。返回 ``(删除数, 释放字节, 删除清单)``。

    - ``purge_all=True``：删除全部 api_responses 文件；
    - 否则删除 mtime 超过 ``older_than_days``（默认 ``settings.cache_b_max_age_days``）者。
    ``dry_run=True`` 只列不删。非 dry_run 时清理后更新 ``.autoclean_state.json``。
    """
    if purge_all:
        targets = [p for d in TIER_B_DIRS for p in _dir_files(d)]
    else:
        days = (
            older_than_days
            if older_than_days is not None
            else settings.cache_b_max_age_days
        )
        targets = _expired_tier_b(days)

    freed = 0
    removed: list[Path] = []
    for p in targets:
        sz, _ = _size_and_mtime(p)
        if dry_run:
            freed += sz
            removed.append(p)
            continue
        try:
            p.unlink()
            freed += sz
            removed.append(p)
        except OSError:
            pass
    if not dry_run:
        _write_state(time.time())
    return len(removed), freed, removed


def maybe_autoclean() -> None:
    """命令入口钩子：距上次自动清理超过 interval 时，清理过期 Tier B。

    全程 try/except **非致命**（缓存清理绝不应影响主命令）；未到期时仅一次小 JSON 读。
    仅在实际删除了文件时打印一行，避免噪声。
    """
    try:
        state = _read_state()
        last = float(state.get("last_autoclean") or 0.0)
        interval = settings.cache_autoclean_interval_days * 86400
        if interval > 0 and (time.time() - last) < interval:
            return
        n, freed, _ = clean_tier_b()
        if n:
            print(
                f"[cache] 自动清理：删除 {n} 个过期 API 缓存"
                f"（>{settings.cache_b_max_age_days}d），释放 {_fmt_size(freed)}。"
            )
    except Exception:  # noqa: BLE001 —— 钩子必须吞掉一切异常
        pass


# ---------------------------------------------------------------------------
# Tier A：LRU 淘汰（仅手动）
# ---------------------------------------------------------------------------
def _collect_tier_a_units(refs: set[Path]) -> list[dict[str, Any]]:
    """构造 Tier A 淘汰单元列表。

    - ``pdfs/``、``extracted/``：以**单文件**为单元；
    - ``html_fulltext/<slug>/``：以**整个 bundle 目录**为单元（``.`` 开头的清单目录不淘汰）。
    每个单元记录 ``bytes``、``mtime``（LRU 键）、``paths``、``ref``（是否被笔记引用）。
    """
    units: list[dict[str, Any]] = []
    for d in (settings.cache_pdfs, settings.cache_extracted):
        for p in _dir_files(d):
            sz, mt = _size_and_mtime(p)
            units.append(
                {
                    "bytes": sz,
                    "mtime": mt,
                    "paths": [p],
                    "is_dir": False,
                    "ref": _is_referenced(p, refs),
                }
            )
    hdir = settings.cache_html_fulltext
    if hdir.exists():
        for sub in hdir.iterdir():
            if not (sub.is_dir() and not sub.name.startswith(".")):
                continue
            files = [p for p in sub.rglob("*") if p.is_file()]
            if not files:
                continue
            sz = 0
            mt = 0.0
            ref = False
            for p in files:
                s, m = _size_and_mtime(p)
                sz += s
                mt = max(mt, m)
                ref = ref or _is_referenced(p, refs)
            units.append(
                {"bytes": sz, "mtime": mt, "paths": [sub], "is_dir": True, "ref": ref}
            )
    return units


def _delete_unit(u: dict[str, Any]) -> bool:
    ok = True
    for p in u["paths"]:
        try:
            if u.get("is_dir") or p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
            else:
                p.unlink()
        except OSError:
            ok = False
    return ok


def prune_tier_a(
    *,
    max_mb: int | None = None,
    dry_run: bool = False,
    keep_referenced: bool = True,
) -> tuple[int, int, list[Path], int]:
    """按 LRU（mtime 升序）淘汰 Tier A，直到总占用 ≤ ``max_mb``（默认软上限）。

    ``keep_referenced=True`` 时跳过被 ``papers/`` 笔记引用的文件。返回
    ``(淘汰单元数, 释放字节, 淘汰清单, 淘汰后剩余总字节)``。``dry_run`` 只列不删。
    """
    target_bytes = (
        max_mb if max_mb is not None else settings.cache_soft_limit_mb
    ) * _MB
    refs = referenced_paths() if keep_referenced else set()
    units = _collect_tier_a_units(refs)

    total = sum(u["bytes"] for u in units)
    if total <= target_bytes:
        return 0, 0, [], total

    # LRU：mtime 升序（最久未用在前）；被引用的排到最后（尽量不删）
    units.sort(key=lambda u: (u["ref"], u["mtime"]))
    removed: list[Path] = []
    freed = 0
    n = 0
    for u in units:
        if total - freed <= target_bytes:
            break
        if u["ref"] and keep_referenced:
            continue
        if dry_run:
            freed += u["bytes"]
            removed.extend(u["paths"])
            n += 1
            continue
        if _delete_unit(u):
            freed += u["bytes"]
            removed.extend(u["paths"])
            n += 1
    return n, freed, removed, total - freed


# ---------------------------------------------------------------------------
# CLI（便于独立调试）
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(format_stats(stats()))
