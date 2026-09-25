"""local_ingest —— 把「本地已有 PDF 文献仓库」批量入库到文献库。

场景
----
用户手动维护了一批本地 PDF（如 data/research/1_gain_ep/relevant_literature/），
希望像"从网上下载"一样把它们纳入 skill 的文献库：复制归档 + MinerU 抽取为 Markdown，
但**不逐篇精读写报告**。research 的 read/add 子命令是围绕 DOI/URL 下载设计的，
不适配"本地已有 PDF"，故本模块提供一条独立的入库通道，并由 `research ingest` 门面暴露。

与 skill 现有基础设施的关系（关键：单一位置，不另开平行目录）
--------------------------------------------------------------
- PDF 归档到 ``settings.cache_pdfs/<theme>/<slug>.pdf``（与下载的 PDF 同处 cache/pdfs）。
- 抽取的 Markdown 写到 ``settings.cache_extracted/<theme>/<slug>.md``——即 skill 原生的
  转换结果目录 cache/extracted/。该目录已被 .gitignore 单独纳入 Git 跟踪（cache/* +
  !cache/extracted/），因为它是 MinerU 配额换来的、重获代价高的产物。
- 抽取复用 ``pdf_extract.extract_pdf(..., write_cache=False)``，与 research read 一致，
  只留这一份规范全文，消除双副本。

清单（manifest）是唯一真源
--------------------------
``ingest/manifest.json``（受 Git 跟踪）逐条登记每篇文献的 theme/slug/type/priority/pages/
source/status/paths，既是我的分类账本，也是**跨会话断点续传**的进度记录：每处理完一条
立即回写，中断后重跑自动跳过 done、重试 failed。

用法（门面）::

    research ingest --status                    # 只看进度汇总
    research ingest --dry-run --priority 1      # 预演第 1 批，不实际抽取
    research ingest --priority 1                # 跑第 1 批（普通论文）
    research ingest --priority 1 --limit-pages 800   # 带页数预算
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from . import pdf_extract
from .config import settings

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
DEFAULT_MANIFEST: Path = settings.module_dir / "ingest" / "manifest.json"

# 状态取值
ST_PENDING = "pending"
ST_DONE = "done"
ST_FAILED = "failed"
ST_SKIPPED = "skipped"


# ---------------------------------------------------------------------------
# 清单读写
# ---------------------------------------------------------------------------
def load_manifest(path: Path | str | None = None) -> dict[str, Any]:
    """读取 manifest.json；缺失时报错并给出提示。"""
    p = Path(path) if path else DEFAULT_MANIFEST
    if not p.exists():
        raise FileNotFoundError(
            f"未找到入库清单：{p}\n"
            f"  请先创建 manifest.json（逐条登记 theme/slug/type/priority/pages/source/status）。"
        )
    return json.loads(p.read_text(encoding="utf-8"))


def save_manifest(data: dict[str, Any], path: Path | str | None = None) -> None:
    """把 manifest 回写为可读的 UTF-8 JSON（indent=2，保留中文）。"""
    p = Path(path) if path else DEFAULT_MANIFEST
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _source_root(data: dict[str, Any]) -> Path:
    """解析清单里的 source_root（相对值按项目根解析）。"""
    raw = data.get("source_root") or ""
    p = Path(raw)
    if not p.is_absolute():
        p = settings.project_root / p
    return p


def _resolve_source(src: Path) -> Path:
    """返回可用于文件系统操作的源路径（处理 Windows 超长路径）。

    Windows 上超过 MAX_PATH(260 字符) 的路径，``exists()``/``open()`` 会失败。此类路径
    回退到 ``\\\\?\\`` 扩展长度前缀（要求绝对路径 + 反斜杠，故先用 os.path.abspath 规范化）。
    先试原路径（绝大多数短路径直接命中），仅当不存在时才尝试扩展前缀，最小化行为改变。
    """
    if src.exists():
        return src
    if os.name == "nt":
        ext = Path("\\\\?\\" + os.path.abspath(str(src)))
        if ext.exists():
            return ext
    return src


# ---------------------------------------------------------------------------
# 状态汇总
# ---------------------------------------------------------------------------
def summarize(data: dict[str, Any]) -> str:
    """生成 manifest 的人类可读进度汇总（按 status × priority 统计篇数/页数）。"""
    entries = data.get("entries", [])
    lines = [
        "=== 本地入库清单进度汇总 (ingest manifest) ===",
        f"清单来源根 : {_source_root(data)}",
        f"PDF 条目数 : {len(entries)}",
    ]
    by_status: dict[str, int] = {}
    pages_by_status: dict[str, int] = {}
    for e in entries:
        st = e.get("status") or ST_PENDING
        by_status[st] = by_status.get(st, 0) + 1
        pages_by_status[st] = pages_by_status.get(st, 0) + int(e.get("pages") or 0)
    lines.append("")
    lines.append("【按状态】")
    for st in (ST_DONE, ST_PENDING, ST_FAILED, ST_SKIPPED):
        if st in by_status:
            lines.append(f"  {st:<8}: {by_status[st]:>3} 篇  {pages_by_status[st]:>5} 页")
    # 分优先级的待办
    lines.append("")
    lines.append("【待办 (pending) 按优先级】")
    prio_pages: dict[int, list[int]] = {}
    for e in entries:
        if (e.get("status") or ST_PENDING) != ST_PENDING:
            continue
        pr = int(e.get("priority") or 9)
        prio_pages.setdefault(pr, [0, 0])
        prio_pages[pr][0] += 1
        prio_pages[pr][1] += int(e.get("pages") or 0)
    for pr in sorted(prio_pages):
        cnt, pgs = prio_pages[pr]
        lines.append(f"  P{pr}: {cnt:>3} 篇  {pgs:>5} 页")
    # 分主题
    lines.append("")
    lines.append("【按主题 (theme)】")
    theme_cnt: dict[str, int] = {}
    for e in entries:
        th = e.get("theme") or "(none)"
        theme_cnt[th] = theme_cnt.get(th, 0) + 1
    for th in sorted(theme_cnt):
        lines.append(f"  {th:<32}: {theme_cnt[th]:>3} 篇")
    non_pdf = data.get("non_pdf_assets", [])
    if non_pdf:
        lines.append("")
        lines.append(f"【非 PDF 资产（登记但不转换）】: {len(non_pdf)} 项")
    lines.append("============================================")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 单条入库
# ---------------------------------------------------------------------------
def _select(
    entries: list[dict[str, Any]],
    *,
    priority: int | None,
    theme: str | None,
    force: bool,
) -> list[dict[str, Any]]:
    """按过滤条件挑出待处理条目，并按 (priority, pages) 升序（先做小而快的）。"""
    sel = []
    for e in entries:
        st = e.get("status") or ST_PENDING
        if st == ST_SKIPPED:
            continue
        if st == ST_DONE and not force:
            continue
        if priority is not None and int(e.get("priority") or 9) != priority:
            continue
        if theme is not None and e.get("theme") != theme:
            continue
        sel.append(e)
    sel.sort(key=lambda e: (int(e.get("priority") or 9), int(e.get("pages") or 0)))
    return sel


def ingest_entry(
    e: dict[str, Any],
    *,
    source_root: Path,
    backend: str | None,
    dry_run: bool,
    force: bool,
) -> str:
    """入库单条：复制 PDF → MinerU 抽取 → 写 cache/extracted/<theme>/<slug>.md → 回写字段。

    返回处理结果状态字符串（done/failed/pending[dry-run]）。异常不外抛，写入 e['note']。
    """
    theme = e.get("theme") or "misc"
    slug = e.get("slug") or "untitled"
    src = _resolve_source(source_root / e["source"])
    dest_pdf = settings.cache_pdfs / theme / f"{slug}.pdf"
    dest_md = settings.cache_extracted / theme / f"{slug}.md"

    if dry_run:
        verb = "重抽" if (dest_md.exists() and force) else "抽取"
        print(f"  [dry-run] {theme}/{slug}.pdf  ({e.get('pages')} 页) → {verb}")
        return ST_PENDING

    # 1) 源文件存在性
    if not src.exists():
        e["status"] = ST_FAILED
        e["note"] = f"源文件不存在：{src}"
        print(f"  [失败] {theme}/{slug}: 源文件不存在", file=sys.stderr)
        return ST_FAILED

    try:
        # 2) 复制归档 PDF（幂等：大小一致则跳过）
        dest_pdf.parent.mkdir(parents=True, exist_ok=True)
        if not (dest_pdf.exists() and dest_pdf.stat().st_size == src.stat().st_size):
            shutil.copy2(src, dest_pdf)
        e["pdf_path"] = str(dest_pdf)

        # 2b) 补齐缺失页数（源路径曾超长时，生成清单阶段可能未测得 pages）
        if not e.get("pages"):
            try:
                import fitz  # type: ignore

                with fitz.open(str(dest_pdf)) as _doc:
                    e["pages"] = int(_doc.page_count)
            except Exception:
                pass

        # 3) 抽取（命中已有 md 且非 force → 复用，跳过 MinerU）
        dest_md.parent.mkdir(parents=True, exist_ok=True)
        if dest_md.exists() and not force:
            e["md_path"] = str(dest_md)
            e["status"] = ST_DONE
            e["note"] = "复用已存在的转换全文"
            print(f"  [复用] {theme}/{slug}.md 已存在，跳过抽取")
            return ST_DONE

        t0 = time.time()
        md = pdf_extract.extract_pdf(dest_pdf, backend=backend, write_cache=False)
        dest_md.write_text(md, encoding="utf-8")
        e["md_path"] = str(dest_md)
        e["status"] = ST_DONE
        e["note"] = ""
        e["extracted_chars"] = len(md)
        print(
            f"  [完成] {theme}/{slug}.md  ({e.get('pages')} 页, {len(md)} 字符, "
            f"{time.time() - t0:.0f}s)"
        )
        return ST_DONE
    except Exception as ex:  # 单条失败不阻塞其余
        e["status"] = ST_FAILED
        e["note"] = f"{type(ex).__name__}: {ex}"[:300]
        print(
            f"  [失败] {theme}/{slug}: {type(ex).__name__}: {ex}",
            file=sys.stderr,
        )
        return ST_FAILED


# ---------------------------------------------------------------------------
# 编排入口
# ---------------------------------------------------------------------------
def run_ingest(
    manifest_path: Path | str | None = None,
    *,
    priority: int | None = None,
    theme: str | None = None,
    backend: str | None = None,
    limit_pages: int | None = None,
    limit_files: int | None = None,
    dry_run: bool = False,
    force: bool = False,
) -> int:
    """按清单批量入库。每处理完一条立即回写 manifest（断点续传）。

    返回退出码：0 正常（含全部已完成）；1 有失败项。
    """
    mpath = Path(manifest_path) if manifest_path else DEFAULT_MANIFEST
    data = load_manifest(mpath)
    source_root = _source_root(data)
    entries = data.get("entries", [])
    selected = _select(entries, priority=priority, theme=theme, force=force)

    scope = []
    if priority is not None:
        scope.append(f"priority={priority}")
    if theme is not None:
        scope.append(f"theme={theme}")
    scope_s = ", ".join(scope) or "全部"
    print(
        f"[ingest] 清单：{mpath}\n"
        f"[ingest] 范围：{scope_s}；命中 {len(selected)} 条待处理"
        + ("（dry-run 预演）" if dry_run else "")
    )
    if not selected:
        print("[ingest] 无待处理条目（可能已全部完成，或过滤条件过窄）。")
        return 0

    pages_done = 0
    files_done = 0
    n_done = n_failed = 0
    for i, e in enumerate(selected, 1):
        # 页数/文件数预算：在开始下一条前判断（已转换页数达预算即停）
        if limit_pages is not None and pages_done >= limit_pages:
            print(f"[ingest] 已达页数预算 {limit_pages} 页，停止（剩余留待下次续跑）。")
            break
        if limit_files is not None and files_done >= limit_files:
            print(f"[ingest] 已达文件数预算 {limit_files}，停止。")
            break
        print(f"[ingest] ({i}/{len(selected)}) {e.get('theme')}/{e.get('slug')} ...")
        st = ingest_entry(
            e,
            source_root=source_root,
            backend=backend,
            dry_run=dry_run,
            force=force,
        )
        if not dry_run:
            files_done += 1
            pages_done += int(e.get("pages") or 0)
            if st == ST_DONE:
                n_done += 1
            elif st == ST_FAILED:
                n_failed += 1
            # 逐条落盘：中断也不丢进度
            save_manifest(data, mpath)

    if dry_run:
        print("[ingest] dry-run 结束（未复制/未抽取/未改状态）。")
        return 0

    print(
        f"\n[ingest] 本批完成：成功 {n_done} 篇，失败 {n_failed} 篇，"
        f"累计抽取约 {pages_done} 页。清单已更新：{mpath}"
    )
    return 1 if n_failed else 0
