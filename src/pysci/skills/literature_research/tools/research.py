"""research —— 文献调研统一 CLI 门面（literature-research skill 的后端）。

一个入口收敛「检索 / 阅读 / 元数据 / 入库 / 库查询 / 索引」的全部能力，
供 skill 文档与 LLM 直接调用，无需了解底层 8 个客户端模块的实现细节。

子命令
------
    doctor   环境与能力自检（检索源 / PDF 后端 / Playwright / Zotero 就绪状态）
    search   多源融合检索（OpenAlex 主 + arXiv；--enrich 追加 WoS 官方 JIF/JCR + S2 TLDR）
    read     给 DOI/URL/arXiv id：抓全文 → 抽取 Markdown → 生成 papers/ 笔记骨架
    get      给 DOI/OpenAlex/arXiv id：输出融合后的完整元数据（--json 输出机器可读）
    add      给 DOI：入库 Zotero + 生成 papers/ 笔记骨架（不抓全文）
    library  查询 Zotero 库（ping / list / search / get）
    index    扫描 papers/ 重建 INDEX.md（--check 仅校验不写）
    cache    缓存治理（stats 概览 / clean 清 Tier B / prune 手动 LRU 淘汰 Tier A）

用法（在项目根目录）::

    python -m pysci.skills.literature_research.tools.research <cmd> [options]

设计原则
--------
1. 只做编排，不重复造轮子——所有能力复用 tools/ 下已验证的客户端模块。
2. S2（Semantic Scholar）为末位可选源：无 API key 时**完全跳过**；即便配置了 key，
   其所有输出与异常也被静默吞掉（校园网通常不可达，失败是预期行为，不需为它排障）。
3. WoS（Web of Science）为可选增强源：未配置或调用失败时静默降级到 OpenAlex 估算值。
4. 所有产物路径基于 settings.module_dir（= literature/）。
5. 不依赖 PyYAML：frontmatter 由内置 _dump_yaml 生成，避免额外依赖。
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
import sys
from datetime import date
from pathlib import Path
from typing import Any

from . import (
    arxiv_client,
    browser_fetch,
    cache_manager,
    openalex_client,
    pdf_extract,
    semantic_scholar_client,
    wos_client,
    zotero_bridge,
)
from .config import settings

# ---------------------------------------------------------------------------
# 路径常量（全部基于模块根 settings.module_dir = literature/）
# ---------------------------------------------------------------------------
MODULE_DIR: Path = settings.module_dir
PAPERS_DIR: Path = MODULE_DIR / "papers"
SHORTLISTS_DIR: Path = MODULE_DIR / "shortlists"
REVIEWS_DIR: Path = MODULE_DIR / "reviews"
TEMPLATES_DIR: Path = MODULE_DIR / "templates"
INDEX_PATH: Path = MODULE_DIR / "INDEX.md"

# search --sort 的取值到各源排序键的映射
SORT_MAP_OPENALEX = {
    "relevance": "relevance_score:desc",
    "date": "publication_date:desc",
    "citations": "cited_by_count:desc",
}
SORT_MAP_ARXIV = {
    "relevance": "relevance",
    "date": "submittedDate",
    "citations": "relevance",  # arXiv 无被引排序，退化为相关性
}


# ===========================================================================
# 通用小工具
# ===========================================================================
def _today() -> str:
    return date.today().isoformat()


def _module_available(name: str) -> bool:
    """检查某第三方库是否可导入（不实际导入）。"""
    import importlib.util

    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def _slugify(text: str, max_len: int = 48) -> str:
    """把任意标题转成文件名安全的 slug（保留中文，标点/空白折叠为 -）。"""
    if not text:
        return "untitled"
    text = text.strip().lower()
    text = re.sub(r"[^\w\u4e00-\u9fff]+", "-", text)
    text = re.sub(r"-{2,}", "-", text).strip("-")
    return (text[:max_len].rstrip("-")) or "untitled"


def _yaml_scalar(v: Any) -> str:
    """把标量转为 YAML 字面量（不依赖 PyYAML）。"""
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    s = str(v)
    if s == "":
        return '""'
    if "\n" in s:
        esc = s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
        return f'"{esc}"'
    needs_quote = (
        s[0] in "!&*-?|>%@`\"'#,[]{}"
        or ": " in s
        or s.startswith(" ")
        or s.endswith(" ")
        or s.lower() in {"true", "false", "null", "yes", "no", "on", "off", "~"}
        or bool(re.match(r"^[-+]?\d", s))
    )
    if needs_quote:
        esc = s.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{esc}"'
    return s


def _dump_yaml(data: dict[str, Any], indent: int = 0) -> str:
    """把（可嵌套 dict / 标量列表的）dict 转为 YAML 文本。

    仅覆盖本项目 frontmatter 需要的类型：标量、字符串列表、嵌套 dict。
    """
    pad = "  " * indent
    lines: list[str] = []
    for k, v in data.items():
        if isinstance(v, dict):
            if not v:
                lines.append(f"{pad}{k}: {{}}")
            else:
                lines.append(f"{pad}{k}:")
                lines.append(_dump_yaml(v, indent + 1).rstrip("\n"))
        elif isinstance(v, (list, tuple)):
            if not v:
                lines.append(f"{pad}{k}: []")
            else:
                lines.append(f"{pad}{k}:")
                for item in v:
                    lines.append(f"{pad}  - {_yaml_scalar(item)}")
        else:
            lines.append(f"{pad}{k}: {_yaml_scalar(v)}")
    return "\n".join(lines) + "\n"


def _load_template(name: str) -> str:
    """读取 templates/<name>（如 'paper_note.md'）；缺失返回空串。"""
    p = TEMPLATES_DIR / name
    if not p.exists():
        return ""
    return p.read_text(encoding="utf-8")


def _split_template(tpl: str) -> tuple[str, str]:
    """把模板拆成 (frontmatter骨架, 正文)。以开头的 --- ... --- 为界。"""
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", tpl, re.DOTALL)
    if not m:
        return "", tpl
    return m.group(1), m.group(2)


def _fill_placeholders(body: str, values: dict[str, Any]) -> str:
    """替换正文中的 {{key}} 占位符；未知键原样保留（供 AI 后续填写）。"""

    def repl(m: re.Match) -> str:
        key = m.group(1).strip()
        if key in values:
            v = values[key]
            if v is None or v == "":
                return "—"
            if isinstance(v, (list, tuple)):
                return ", ".join(str(x) for x in v) or "—"
            return str(v)
        return m.group(0)

    return re.sub(r"\{\{\s*([\w.]+)\s*\}\}", repl, body)


def _note_filename(fm: dict[str, Any]) -> str:
    """papers/ 命名规范：{year}_{firstauthor_lastname}_{slug}.md"""
    year = fm.get("year") or "nd"
    last = _slugify(str(fm.get("first_author_last_name") or "unknown"), 24)
    slug = _slugify(str(fm.get("short_title") or fm.get("title") or "paper"), 40)
    return f"{year}_{last}_{slug}.md"


def build_note_markdown(fm: dict[str, Any]) -> str:
    """由 frontmatter dict 生成完整 paper note（套用 templates/paper_note.md 正文骨架）。"""
    tpl = _load_template("paper_note.md")
    _skel, body = _split_template(tpl)
    if not body.strip():
        body = "\n# {{short_title}} ({{first_author_last_name}} {{year}})\n\n> 一句话定位：\n"
    body = _fill_placeholders(body, fm)
    fm = dict(fm)
    fm.setdefault("added_date", _today())
    return f"---\n{_dump_yaml(fm)}---\n{body}"


def _write_note(fm: dict[str, Any], *, overwrite: bool = False) -> Path:
    """把笔记骨架写入 papers/，返回路径。已存在且非 overwrite 时不覆盖。"""
    PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    path = PAPERS_DIR / _note_filename(fm)
    if path.exists() and not overwrite:
        return path
    path.write_text(build_note_markdown(fm), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# ID 解析与元数据获取
# ---------------------------------------------------------------------------
def _classify_id(raw: str) -> tuple[str, str]:
    """判断输入类型，返回 (kind, value)，kind ∈ {url, doi, arxiv, openalex}。"""
    s = (raw or "").strip()
    if s.startswith("http://") or s.startswith("https://"):
        if "doi.org/" in s:
            return "doi", s.split("doi.org/", 1)[1].strip("/")
        m = re.search(r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5})", s)
        if m:
            return "arxiv", m.group(1)
        return "url", s
    if re.match(r"^\d{4}\.\d{4,5}(v\d+)?$", s):
        return "arxiv", s
    if re.match(r"^[Ww]\d{6,}$", s):
        return "openalex", s.upper()
    if re.match(r"^10\.\d{4,9}/", s):
        return "doi", s
    return "doi", s  # 兜底按 DOI 处理


def _resolve_work(raw: str) -> tuple[str | None, dict[str, Any] | None]:
    """把 doi/openalex/arxiv id 解析为 (source, work_dict)；url 或查不到返回 (None, None)。"""
    kind, val = _classify_id(raw)
    try:
        if kind == "arxiv":
            w = arxiv_client.get_paper(val)
            return ("arxiv", w) if w else (None, None)
        if kind == "openalex":
            w = openalex_client.get_work(openalex_id=val)
            return ("openalex", w) if w else (None, None)
        if kind == "doi":
            w = openalex_client.get_work(doi=val)
            if w:
                return ("openalex", w)
            # DOI 未收录于 OpenAlex：若形如 arXiv DOI 再试 arXiv
            return (None, None)
    except Exception as e:  # 网络/解析异常不致命
        print(
            f"[research] 元数据解析失败（{kind}={val}）：{type(e).__name__}: {e}",
            file=sys.stderr,
        )
    return (None, None)


def _frontmatter_from_work(source: str, work: dict[str, Any]) -> dict[str, Any]:
    """按来源选择正确的转换器，产出统一的 paper_note frontmatter dict。"""
    if source == "arxiv":
        return arxiv_client.arxiv_to_note_frontmatter(work)
    return openalex_client.work_to_note_frontmatter(work)


def _enrich_work(
    work: dict[str, Any], *, collect: list[str] | None = None
) -> dict[str, Any]:
    """用 WoS（官方 JIF/JCR/ESI）+ S2（TLDR，若有 key）增强单个 work dict。

    两源均静默降级：全程重定向 stdout/stderr，任何异常都吞掉并返回原 dict。
    S2 无 key 时完全不调用（符合"失败是预期、不引导排障"的策略）。
    被抑制的提示信息收集进 collect（若提供），供上层汇总一行简报。
    """
    out, err = io.StringIO(), io.StringIO()
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            if settings.wos_ready:
                try:
                    work = wos_client.enrich_openalex_work(work)
                except Exception:
                    pass
            if settings.semantic_scholar_api_key:
                try:
                    work = semantic_scholar_client.enrich_from_s2(work)
                except Exception:
                    pass
    finally:
        if collect is not None:
            txt = (out.getvalue() + err.getvalue()).strip()
            if txt:
                collect.append(txt)
    return work


def _overlay_enrichment(fm: dict[str, Any], work: dict[str, Any]) -> dict[str, Any]:
    """把 _enrich_work 写进 work 的独家字段叠加到 frontmatter（覆盖 OpenAlex 估算值）。"""
    for k in (
        "wos_id",
        "jif",
        "jif_5yr",
        "jcr_quartile",
        "esi_highly_cited",
        "esi_hot_paper",
    ):
        v = work.get(k)
        if v not in (None, "", False):
            fm[k] = v
    if work.get("tldr"):
        fm["tldr"] = work["tldr"]
    if work.get("influential_citation_count"):
        fm["influential_citation_count"] = work["influential_citation_count"]
    return fm


# ---------------------------------------------------------------------------
# 检索结果归一化与展示
# ---------------------------------------------------------------------------
def _row(src: str, w: dict[str, Any]) -> dict[str, Any]:
    """把任一源的 work dict 投影为统一展示行（防御式取值，兼容各源字段差异）。"""
    year = (
        w.get("publication_year")
        or w.get("year")
        or w.get("pub_year")
        or (str(w.get("published") or "")[:4])
        or None
    )
    title = w.get("title") or w.get("display_name") or ""
    return {
        "title": title,
        "first_author_last_name": w.get("first_author_last_name") or "",
        "year": year,
        "journal": w.get("journal")
        or w.get("journal_ref")
        or ("arXiv" if src == "arxiv" else ""),
        "doi": w.get("doi") or "",
        "arxiv_id": w.get("arxiv_id") or "",
        "openalex_id": w.get("openalex_id") or "",
        "cited_by_count": w.get("cited_by_count"),
        "oa_status": w.get("oa_status") or ("green" if src == "arxiv" else ""),
        "jif": w.get("jif"),
        "jcr_quartile": w.get("jcr_quartile") or "",
        "source": src,
    }


def _dedupe(rows: list[tuple[str, dict]]) -> list[tuple[str, dict]]:
    """按 DOI → arXiv id → 标题 slug 优先级去重，保留首次出现（主源优先）。"""
    seen: set[str] = set()
    out: list[tuple[str, dict]] = []
    for src, w in rows:
        r = _row(src, w)
        key = (
            (r["doi"] or "").lower()
            or (r["arxiv_id"] or "").lower()
            or _slugify(r["title"], 60)
        )
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        out.append((src, w))
    return out


def _print_rows(rows: list[tuple[str, dict]]) -> None:
    if not rows:
        print("（无结果）")
        return
    for i, (src, w) in enumerate(rows, 1):
        r = _row(src, w)
        cite = r["cited_by_count"]
        cite_s = str(cite) if cite is not None else "—"
        jif_s = ""
        if r["jif"]:
            jif_s += f"JIF {r['jif']}"
        if r["jcr_quartile"]:
            jif_s += f" {r['jcr_quartile']}"
        print(
            f"{i:>2}. [{r['year'] or '—'}] 被引 {cite_s:>4}  "
            f"{(r['oa_status'] or '—'):<7} {r['journal'] or '—'}  ({src})"
        )
        print(f"     {r['first_author_last_name']}—— {r['title']}")
        ids = []
        if r["doi"]:
            ids.append(f"DOI:{r['doi']}")
        if r["arxiv_id"]:
            ids.append(f"arXiv:{r['arxiv_id']}")
        if r["openalex_id"]:
            ids.append(r["openalex_id"])
        tail = " | ".join(ids)
        if jif_s:
            tail = f"{tail}  [{jif_s}]" if tail else f"[{jif_s}]"
        if tail:
            print(f"     {tail}")
        print()


def _render_candidates(rows: list[tuple[str, dict]]) -> str:
    """把检索结果渲染为 shortlist 候选清单（供 AI 后续初筛分级）。"""
    out: list[str] = []
    for i, (src, w) in enumerate(rows, 1):
        r = _row(src, w)
        ids = []
        if r["doi"]:
            ids.append(f"DOI: {r['doi']}")
        if r["arxiv_id"]:
            ids.append(f"arXiv: {r['arxiv_id']}")
        jif = f", JIF {r['jif']}" if r["jif"] else ""
        quart = f" {r['jcr_quartile']}" if r["jcr_quartile"] else ""
        cite = r["cited_by_count"] if r["cited_by_count"] is not None else "—"
        out.append(
            f"#### {i}. {r['title']}\n"
            f"- **Authors**: {r['first_author_last_name']} et al.\n"
            f"- **Journal / Year**: {r['journal'] or '—'} ({r['year'] or '—'}){jif}{quart}\n"
            f"- **Cited by**: {cite}\n"
            f"- **OA**: {r['oa_status'] or '—'}\n"
            f"- **IDs**: {' | '.join(ids) or '—'}\n"
            f"- **Source**: {src}\n"
            f"- **Action**: [ ] 入库 Zotero  [ ] 生成 paper note  [ ] 精读\n"
        )
    return "\n".join(out)


def _parse_year(spec: str | None) -> tuple[int | None, int | None]:
    """解析 --year：支持 '2023-2026' / '2023-' / '2023'。"""
    if not spec:
        return None, None
    spec = spec.strip()
    m = re.match(r"^(\d{4})\s*-\s*(\d{4})$", spec)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.match(r"^(\d{4})\s*-$", spec)
    if m:
        return int(m.group(1)), None
    m = re.match(r"^(\d{4})$", spec)
    if m:
        return int(m.group(1)), int(m.group(1))
    return None, None


def _write_shortlist(
    query: str, args: argparse.Namespace, rows: list[tuple[str, dict]]
) -> Path:
    """套用 templates/shortlist.md 生成检索快照，写入 shortlists/。"""
    SHORTLISTS_DIR.mkdir(parents=True, exist_ok=True)
    today = _today()
    qslug = _slugify(query, 40)
    tpl = _load_template("shortlist.md")
    _skel, body = _split_template(tpl)
    sources_used = _sources_used(args)
    fm = {
        "query_date": today,
        "query_slug": qslug,
        "queried_by": "AI Agent (research.py)",
        "purpose": args.purpose or query,
        "sources": sources_used,
        "query_string": query,
        "filters": {
            "year_range": args.year or "",
            "min_citations": args.min_citations,
            "venues": [],
            "concepts": [],
            "oa_only": bool(getattr(args, "oa_only", False)),
        },
        "sort_by": SORT_MAP_OPENALEX.get(args.sort, args.sort),
        "results_returned": len(rows),
        "results_after_screening": 0,
        "screening_status": "pending",
    }
    values = {"query_slug": qslug, "query_date": today, "purpose": fm["purpose"]}
    body = _fill_placeholders(body, values)
    body += (
        "\n\n---\n\n## 机器检索结果（research.py 自动生成，待 AI 初筛分级）\n\n"
        + _render_candidates(rows)
    )
    out = SHORTLISTS_DIR / f"{today}_{qslug}.md"
    out.write_text(f"---\n{_dump_yaml(fm)}---\n{body}", encoding="utf-8")
    return out


def _sources_used(args: argparse.Namespace) -> list[str]:
    src = args.source
    if src == "auto":
        used = ["openalex", "arxiv"]
        if getattr(args, "enrich", False) and settings.wos_ready:
            used.append("wos")
        if settings.semantic_scholar_api_key:
            used.append("semantic_scholar")
        return used
    return [src]


# ===========================================================================
# 子命令：doctor
# ===========================================================================
def cmd_doctor(args: argparse.Namespace) -> int:
    """环境与能力自检。"""
    print("=== pySciWS literature_research — 能力自检 (doctor) ===\n")
    print(f"模块根        : {settings.module_dir}")
    print(f"缓存目录      : {settings.cache_dir}")
    print(f"项目根        : {settings.project_root}")
    print()

    print("【检索源】")
    oa = (
        "就绪"
        if settings.openalex_email
        else "可用（未设 OPENALEX_EMAIL，仍能用；建议设置以进入 polite pool 提速）"
    )
    print(f"  OpenAlex         : {oa}（主源，无需 key）")
    print("  arXiv            : 就绪（无需 key）")
    print(
        f"  Web of Science   : {'就绪（官方 JIF/JCR/ESI 增强）' if settings.wos_ready else '未配置（可选增强源）'}"
    )
    s2 = (
        "已配置 key"
        if settings.semantic_scholar_api_key
        else "未配置（可选末位源；校园网通常不可达，属预期，无需处理）"
    )
    print(f"  Semantic Scholar : {s2}")
    print(
        f"  Elsevier/Scopus  : {'已配置 key' if settings.elsevier_api_key else '未配置（预留）'}"
    )
    print()

    print("【PDF → Markdown 后端】")
    backends = pdf_extract.available_backends()
    print(f"  可用后端         : {', '.join(backends) or '（无——公式抽取将不可用）'}")
    print(f"  当前默认         : {settings.pdf_extract_backend}")
    print(
        f"  MinerU 云端      : {'就绪（公式→LaTeX 首选）' if settings.mineru_token else '未配置 MINERU_TOKEN'}"
    )
    print(
        f"  pymupdf4llm 本地 : {'可用（兜底，公式会丢失）' if _module_available('pymupdf4llm') or _module_available('fitz') else '未安装'}"
    )
    print()

    print("【付费墙抓取 Playwright】")
    print(
        f"  playwright 库    : {'已安装' if _module_available('playwright') else '未安装（read 抓取付费墙 PDF 会失败）'}"
    )
    print("  浏览器           : 运行时自动探测 chrome → msedge → bundled chromium")
    print()

    print("【Zotero 文献库】")
    print(
        f"  Web API 凭据     : {'就绪' if settings.zotero_web_ready else '未配置（ZOTERO_USER_ID / ZOTERO_API_KEY）'}"
    )
    try:
        zb = zotero_bridge.ZoteroBridge()
        info = zb.ping()
        print(f"  连通性           : 后端={zb.backend or '—'}；ping={info}")
    except Exception as e:
        print(
            f"  连通性           : 不可达（{type(e).__name__}）——本地 API 需 Zotero 桌面版开启 Settings→Advanced→Allow other applications"
        )
    print()
    print("=== 自检结束 ===")
    return 0


# ===========================================================================
# 子命令：search
# ===========================================================================
def cmd_search(args: argparse.Namespace) -> int:
    """多源融合检索。"""
    query = args.query
    year_from, year_to = _parse_year(args.year)
    limit = max(1, args.limit)
    source = args.source
    collected: list[str] = []
    rows: list[tuple[str, dict]] = []

    if source in ("auto", "openalex"):
        res = openalex_client.search_works(
            query,
            year_from=year_from,
            year_to=year_to,
            min_citations=args.min_citations,
            oa_only=bool(getattr(args, "oa_only", False)),
            sort=SORT_MAP_OPENALEX.get(args.sort, "relevance_score:desc"),
            per_page=limit,
        )
        results = res.get("results", [])
        rows.extend(("openalex", w) for w in results)
        print(
            f"[search] OpenAlex: 取回 {len(results)} 条（库中匹配 {res.get('meta', {}).get('count')}）"
        )

    if source in ("auto", "arxiv"):
        res = arxiv_client.search_arxiv(
            query,
            max_results=limit,
            sort_by=SORT_MAP_ARXIV.get(args.sort, "relevance"),
        )
        entries = res.get("entries", [])
        rows.extend(("arxiv", e) for e in entries)
        print(
            f"[search] arXiv: 取回 {len(entries)} 条（总匹配 {res.get('total_results')}）"
        )

    if source == "wos":
        if not settings.wos_ready:
            print(
                "[search] WoS 未配置（缺 WOS_API_KEY），无法作为主源。", file=sys.stderr
            )
            return 2
        client = wos_client.WOSClient()
        res = client.search(query, limit=limit)
        rows.extend(("wos", h) for h in res.get("hits", []))
        print(
            f"[search] WoS: 取回 {len(res.get('hits', []))} 条（总匹配 {res.get('total')}）"
        )

    if source == "s2":
        if not settings.semantic_scholar_api_key:
            print(
                "[search] S2 未配置 key——可选末位源，已跳过（校园网通常不可达，属预期）。"
            )
            return 0
        with (
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            try:
                res = semantic_scholar_client.search_papers(
                    query, limit=limit, year_range=args.year
                )
            except Exception:
                res = {"data": []}
        rows.extend(("s2", p) for p in res.get("data", []))
        print(f"[search] S2: 取回 {len(res.get('data', []))} 条")

    rows = _dedupe(rows)

    # 批量增强（--enrich）：对 OpenAlex 条目补 WoS 官方 JIF/JCR + S2 TLDR
    if getattr(args, "enrich", False) and source == "auto":
        if settings.wos_ready or settings.semantic_scholar_api_key:
            enriched: list[tuple[str, dict]] = []
            for src, w in rows:
                if src == "openalex" and w.get("doi"):
                    w = _enrich_work(w, collect=collected)
                enriched.append((src, w))
            rows = enriched
        else:
            print("[search] --enrich 已忽略：WoS/S2 均未配置。")

    # auto 融合后可能超过 limit（两源相加），截断
    if source == "auto":
        rows = rows[:limit]

    print()
    _print_rows(rows)

    if collected:
        print(
            f"[search] 注：{len(collected)} 处 WoS/S2 增强不可用，相关条目已回退到 OpenAlex 估算值。"
        )

    if args.save:
        path = _write_shortlist(query, args, rows)
        print(f"[search] 检索快照已保存：{path}")
    return 0


# ===========================================================================
# 子命令：read
# ===========================================================================
def _minimal_fm(bundle: Any, raw: str) -> dict[str, Any]:
    """URL-only 抓取（无元数据）时的最小 frontmatter。"""
    title = (getattr(bundle, "title", "") or "") if bundle else ""
    words = [w for w in re.split(r"\W+", title) if w][:6]
    return {
        "title": title,
        "short_title": " ".join(words) or "paper",
        "authors": [],
        "first_author_last_name": "",
        "year": None,
        "journal": "",
        "doi": "",
        "arxiv_id": "",
        "openalex_id": "",
        "oa_url": (getattr(bundle, "url", "") if bundle else raw) or raw,
        "oa_status": "",
        "status": "unread",
        "added_date": _today(),
    }


def cmd_read(args: argparse.Namespace) -> int:
    """抓全文 → 抽取 Markdown → 生成 papers/ 笔记骨架（不碰 Zotero/INDEX）。"""
    raw = args.target
    kind, val = _classify_id(raw)
    out_dir = settings.cache_dir / "html_fulltext"
    force = getattr(args, "force", False)

    # 1) 元数据（尽力；url 无法查元数据）
    src: str | None = None
    work: dict[str, Any] | None = None
    if kind != "url":
        src, work = _resolve_work(raw)
        if work:
            work = _enrich_work(work)

    # 1.5) 尽早组装 frontmatter，并据标题算出规范全文路径（供命中复用 / prune 保护）
    fm: dict[str, Any] | None = None
    if work:
        fm = _overlay_enrichment(_frontmatter_from_work(src or "openalex", work), work)
    fulltext_path: Path | None = None
    if fm is not None:
        stem = _slugify(str(fm.get("short_title") or fm.get("title") or "paper"), 40)
        fulltext_path = settings.cache_extracted / f"{stem}_fulltext.md"

    # 命中复用：已有规范全文且非 --force → 直接读缓存，跳过抓取与抽取
    md = ""
    if fulltext_path is not None and fulltext_path.exists() and not force:
        md = fulltext_path.read_text(encoding="utf-8")
        cache_manager.bump_mtime(fulltext_path)
        print(
            f"[read] 命中缓存全文，跳过抓取/抽取（--force 强制重取）：{fulltext_path}"
        )

    # 2) 抓取 PDF / 全套（命中缓存则整体跳过）
    bundle = None
    pdf_path: Path | None = None
    if not md:
        if kind == "arxiv":
            try:
                got = arxiv_client.download_pdf(val, dest_dir=settings.cache_pdfs)
                if got:
                    cand = Path(got)
                    if cand.exists():
                        pdf_path = cand
            except Exception as e:
                print(
                    f"[read] arXiv PDF 下载失败：{type(e).__name__}: {e}",
                    file=sys.stderr,
                )
            if not pdf_path:
                url = f"https://arxiv.org/abs/{val}"
                print(f"[read] 回退浏览器抓取：{url}")
                bundle = _safe_fetch(url, out_dir, args)
                pdf_path = bundle.pdf_path if bundle else None
        else:
            if kind == "url":
                url = val
            elif kind == "doi":
                url = f"https://doi.org/{val}"
            else:  # openalex
                url = ""
                if work:
                    url = work.get("oa_url") or (
                        f"https://doi.org/{work['doi']}" if work.get("doi") else ""
                    )
                if not url:
                    print(
                        "[read] 无法从 OpenAlex id 解析出可抓取 URL。", file=sys.stderr
                    )
                    return 2
            print(f"[read] 抓取：{url}")
            bundle = _safe_fetch(url, out_dir, args)
            pdf_path = bundle.pdf_path if bundle else None

        # 3) 抽取全文 → Markdown（write_cache=False：只留规范的 {stem}_fulltext.md，消除双副本）
        if pdf_path and Path(pdf_path).exists():
            print(f"[read] 抽取 PDF（backend={args.backend or 'auto'}）...")
            try:
                md = pdf_extract.extract_pdf(
                    pdf_path, backend=args.backend or None, write_cache=False
                )
            except Exception as e:
                print(f"[read] PDF 抽取失败：{type(e).__name__}: {e}", file=sys.stderr)
        if not md and bundle and bundle.html_path and Path(bundle.html_path).exists():
            md = Path(bundle.html_path).read_text(encoding="utf-8")
            print("[read] 无 PDF 或未抽出，回退到 HTML 全文。")

    # 4) 组装 frontmatter（URL-only 无元数据时回退最小骨架）
    if fm is None:
        fm = _minimal_fm(bundle, raw)
    if pdf_path:
        fm["local_pdf_path"] = str(pdf_path)
    fm["added_date"] = _today()
    fm.setdefault("status", "unread")

    # 5) 写全文副本 + 笔记骨架
    if md:
        settings.cache_extracted.mkdir(parents=True, exist_ok=True)
        if fulltext_path is None:
            stem = _slugify(
                str(fm.get("short_title") or fm.get("title") or "paper"), 40
            )
            fulltext_path = settings.cache_extracted / f"{stem}_fulltext.md"
        if force or not fulltext_path.exists():
            fulltext_path.write_text(md, encoding="utf-8")
        fm["extracted_md_path"] = str(fulltext_path)
    note_path: Path | None = None
    if args.note:
        existed = (PAPERS_DIR / _note_filename(fm)).exists()
        note_path = _write_note(fm, overwrite=args.overwrite)
        if existed and not args.overwrite:
            print(f"[read] 笔记已存在，未覆盖（加 --overwrite 可重建）：{note_path}")

    # 6) 报告
    print()
    if bundle is not None:
        print(
            f"[read] 抓取：ok={bundle.ok} 适配器={bundle.adapter or '—'} 浏览器={bundle.browser or '—'} "
            f"耗时={bundle.seconds:.1f}s Cloudflare={bundle.cloudflare} 机构访问={bundle.institutional_access}"
        )
        if bundle.supp_paths:
            print(
                f"[read] 补充材料 {len(bundle.supp_paths)} 份：{', '.join(str(p.name) for p in bundle.supp_paths)}"
            )
    if pdf_path:
        print(f"[read] PDF      : {pdf_path}")
    if fulltext_path:
        print(
            f"[read] 全文 MD  : {fulltext_path}（{len(md)} 字符）——精读请 Read 此文件"
        )
    else:
        print("[read] 警告：未获得全文（PDF 抓取与抽取均失败）。", file=sys.stderr)
    if note_path:
        print(f"[read] 笔记骨架 : {note_path}")
    return 0 if (md or note_path) else 1


def _safe_fetch(url: str, out_dir: Path, args: argparse.Namespace) -> Any:
    """调用 browser_fetch.fetch_all，异常时返回 None（不致命）。"""
    try:
        return browser_fetch.fetch_all(
            url,
            out_dir=out_dir,
            headless=not args.headed,
            use_cache=not getattr(args, "force", False),
        )
    except Exception as e:
        print(f"[read] 浏览器抓取失败：{type(e).__name__}: {e}", file=sys.stderr)
        print(
            "[read] 提示：付费墙/Cloudflare 可加 --headed 用有头浏览器重试。",
            file=sys.stderr,
        )
        return None


# ===========================================================================
# 子命令：get
# ===========================================================================
def cmd_get(args: argparse.Namespace) -> int:
    """输出融合元数据（OpenAlex + WoS 官方 JIF/JCR + S2 TLDR，若可用）。"""
    src, work = _resolve_work(args.identifier)
    if not work:
        print(f"[get] 未找到：{args.identifier}", file=sys.stderr)
        return 1
    collected: list[str] = []
    work = _enrich_work(work, collect=collected)
    fm = _overlay_enrichment(_frontmatter_from_work(src or "openalex", work), work)
    if args.json:
        print(json.dumps(fm, ensure_ascii=False, indent=2))
    else:
        print(f"---\n{_dump_yaml(fm)}---")
    if collected:
        print(
            f"[get] 注：{len(collected)} 处 WoS/S2 增强不可用，已用 OpenAlex 估算值。",
            file=sys.stderr,
        )
    return 0


# ===========================================================================
# 子命令：add
# ===========================================================================
def _extract_zotero_key(resp: Any) -> str:
    """从 Zotero 创建响应里尽力提取新条目 key（兼容 pyzotero / 原生 requests）。"""
    if not isinstance(resp, dict):
        return ""
    succ = resp.get("success") or resp.get("successful")
    if isinstance(succ, dict) and succ:
        first = next(iter(succ.values()))
        if isinstance(first, dict):
            return first.get("key") or (first.get("data") or {}).get("key") or ""
        return str(first)
    if resp.get("key"):
        return str(resp["key"])
    data = resp.get("data")
    if isinstance(data, dict) and data.get("key"):
        return str(data["key"])
    return ""


def cmd_add(args: argparse.Namespace) -> int:
    """入库 Zotero + 生成 papers/ 笔记骨架（不抓全文）。"""
    src, work = _resolve_work(args.doi)
    if not work:
        print(f"[add] 未能从 {args.doi} 解析元数据，无法入库。", file=sys.stderr)
        return 1
    work = _enrich_work(work)
    fm = _overlay_enrichment(_frontmatter_from_work(src or "openalex", work), work)
    tags = [t.strip() for t in args.tags.split(",")] if args.tags else []

    if not settings.zotero_web_ready:
        print(
            "[add] Zotero Web API 未配置（缺 ZOTERO_USER_ID/API_KEY），跳过入库，仅生成笔记骨架。",
            file=sys.stderr,
        )
    else:
        try:
            zb = zotero_bridge.ZoteroBridge()
            resp = zb.create_item_from_metadata(fm, tags=tags)
            key = _extract_zotero_key(resp)
            if key:
                fm["zotero_key"] = key
                fm["zotero_uri"] = (
                    f"https://zotero.org/users/{settings.zotero_user_id}/items/{key}"
                )
                print(f"[add] 已入库 Zotero：{key}")
            else:
                print(f"[add] Zotero 响应未含 key：{resp}", file=sys.stderr)
        except Exception as e:
            print(f"[add] Zotero 入库失败：{type(e).__name__}: {e}", file=sys.stderr)

    fm["added_date"] = _today()
    note_path = PAPERS_DIR / _note_filename(fm)
    existed = note_path.exists() and not args.overwrite
    note_path = _write_note(fm, overwrite=args.overwrite)
    suffix = "（已存在，未覆盖；加 --overwrite 重建）" if existed else ""
    print(f"[add] 笔记骨架：{note_path}{suffix}")
    return 0


# ===========================================================================
# 子命令：library
# ===========================================================================
def _print_zotero_items(items: list[dict[str, Any]]) -> None:
    if not items:
        print("（无条目）")
        return
    for i, it in enumerate(items, 1):
        data = it.get("data", it) if isinstance(it, dict) else {}
        key = it.get("key", "") if isinstance(it, dict) else ""
        title = data.get("title", "")
        itype = data.get("itemType", "")
        creators = data.get("creators", []) or []
        first = ""
        if creators and isinstance(creators[0], dict):
            first = creators[0].get("lastName") or creators[0].get("name") or ""
        year = str(data.get("date") or "")[:4]
        print(f"{i:>2}. [{key}] ({itype}) {first} {year} — {title}")


def cmd_library(args: argparse.Namespace) -> int:
    """查询 Zotero 库：ping / list / search / get。"""
    try:
        zb = zotero_bridge.ZoteroBridge()
    except Exception as e:
        print(f"[library] Zotero 初始化失败：{type(e).__name__}: {e}", file=sys.stderr)
        return 1
    action = args.action
    if action == "ping":
        try:
            print(json.dumps(zb.ping(), ensure_ascii=False, indent=2))
        except Exception as e:
            print(f"[library] ping 失败：{type(e).__name__}: {e}", file=sys.stderr)
            return 1
        return 0
    if action == "list":
        _print_zotero_items(zb.list_items(limit=args.limit, item_type=args.type))
        return 0
    if action == "search":
        if not args.query:
            print("[library] search 需要 --query。", file=sys.stderr)
            return 2
        _print_zotero_items(zb.search_items(args.query, limit=args.limit))
        return 0
    if action == "get":
        if not args.key:
            print("[library] get 需要 --key。", file=sys.stderr)
            return 2
        item = zb.get_item(args.key)
        print(json.dumps(item, ensure_ascii=False, indent=2) if item else "（未找到）")
        return 0
    return 2


# ===========================================================================
# 子命令：index
# ===========================================================================
def _parse_scalar(v: str) -> Any:
    if v in ("null", "~", ""):
        return None
    if v in ("true", "True"):
        return True
    if v in ("false", "False"):
        return False
    if v.startswith('"') and v.endswith('"') and len(v) >= 2:
        return v[1:-1].replace('\\"', '"').replace("\\n", "\n").replace("\\\\", "\\")
    if re.match(r"^[-+]?\d+$", v):
        return int(v)
    if re.match(r"^[-+]?\d*\.\d+$", v):
        return float(v)
    return v


def _parse_frontmatter(text: str) -> dict[str, Any]:
    """轻量解析本项目生成的 frontmatter（仅取标量；跳过列表/嵌套/注释）。"""
    m = re.match(r"^---\s*\n(.*?)\n---", text, re.DOTALL)
    fm: dict[str, Any] = {}
    if not m:
        return fm
    for line in m.group(1).splitlines():
        line = line.rstrip()
        if (
            not line
            or line.lstrip().startswith("#")
            or line.startswith((" ", "\t", "-"))
        ):
            continue
        if ":" not in line:
            continue
        k, _, v = line.partition(":")
        v = v.strip()
        if " #" in v:
            v = v.split(" #", 1)[0].strip()
        fm[k.strip()] = _parse_scalar(v)
    return fm


def _render_index(entries: list[tuple[str, dict[str, Any]]]) -> str:
    def yr(fm: dict[str, Any]) -> int:
        y = fm.get("year")
        if isinstance(y, int):
            return y
        s = str(y or "")
        return int(s) if s.isdigit() else 0

    lines = [
        "# Literature Index",
        "",
        f"> 由 `research index` 自动重建于 {_today()}；共 {len(entries)} 篇。",
        "> 请勿手动编辑本文件——改 `papers/*.md` 的 frontmatter 后重新运行 `research index`。",
        "",
        "| # | Year | First Author | Title | Journal | Status | Rating | Note |",
        "|---|------|--------------|-------|---------|--------|--------|------|",
    ]
    ordered = sorted(entries, key=lambda kv: yr(kv[1]), reverse=True)
    for i, (stem, fm) in enumerate(ordered, 1):
        title = str(fm.get("title") or stem).replace("|", "\\|")
        fa = fm.get("first_author_last_name") or ""
        journal = str(fm.get("journal") or "").replace("|", "\\|")
        status = fm.get("status") or ""
        rating = fm.get("my_rating")
        rating_s = f"{rating}\u2605" if rating not in (None, "") else "—"
        lines.append(
            f"| {i} | {fm.get('year') or '—'} | {fa} | {title} | {journal} | "
            f"{status} | {rating_s} | [{stem}](papers/{stem}.md) |"
        )
    lines.append("")
    return "\n".join(lines)


def cmd_index(args: argparse.Namespace) -> int:
    """扫描 papers/*.md 重建 INDEX.md（--check 仅校验）。"""
    PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    notes = sorted(PAPERS_DIR.glob("*.md"))
    entries: list[tuple[str, dict[str, Any]]] = []
    for p in notes:
        try:
            fm = _parse_frontmatter(p.read_text(encoding="utf-8"))
        except Exception as e:
            print(
                f"[index] 解析 {p.name} 失败：{type(e).__name__}: {e}", file=sys.stderr
            )
            fm = {}
        entries.append((p.stem, fm))

    if not entries and INDEX_PATH.exists() and not args.force:
        print("[index] papers/ 下无笔记；保留现有 INDEX.md（如需清空重建加 --force）。")
        return 0

    content = _render_index(entries)
    if args.check:
        existing = INDEX_PATH.read_text(encoding="utf-8") if INDEX_PATH.exists() else ""
        if existing.strip() == content.strip():
            print("[index] INDEX.md 已是最新。")
            return 0
        print("[index] INDEX.md 与 papers/ 不一致，运行 `research index` 重建。")
        return 1
    INDEX_PATH.write_text(content, encoding="utf-8")
    print(f"[index] 已重建 INDEX.md（{len(entries)} 篇）：{INDEX_PATH}")
    return 0


# ===========================================================================
# 子命令：cache（缓存治理门面；实现在 cache_manager）
# ===========================================================================
def _fmt_bytes(nbytes: int) -> str:
    """把字节数格式化为可读字符串（KB/MB/GB）。"""
    n = float(nbytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{int(n)} B" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _print_removed(removed: list[Path], dry_run: bool) -> None:
    """逐条打印（将）删除项，最多 40 条，避免刷屏。"""
    verb = "将删除" if dry_run else "已删除"
    for p in removed[:40]:
        print(f"  {verb}：{p}")
    if len(removed) > 40:
        print(f"  ...（其余 {len(removed) - 40} 项略）")


def cmd_cache(args: argparse.Namespace) -> int:
    """缓存治理：stats 概览 / clean 清 Tier B / prune 手动 LRU 淘汰 Tier A。"""
    action = args.action
    if action == "stats":
        print(cache_manager.format_stats(cache_manager.stats()))
        return 0

    if action == "clean":
        n, freed, removed = cache_manager.clean_tier_b(
            purge_all=args.all,
            older_than_days=args.older_than,
            dry_run=args.dry_run,
        )
        days = (
            args.older_than
            if args.older_than is not None
            else settings.cache_b_max_age_days
        )
        scope = "全部" if args.all else f"> {days} 天"
        verb = "将清理" if args.dry_run else "已清理"
        _print_removed(removed, args.dry_run)
        print(
            f"[cache] {verb} Tier B（api_responses，{scope}）：{n} 个文件，释放 {_fmt_bytes(freed)}。"
        )
        return 0

    # prune
    n, freed, removed, remain = cache_manager.prune_tier_a(
        max_mb=args.max_mb,
        dry_run=args.dry_run,
        keep_referenced=args.keep_referenced,
    )
    verb = "将淘汰" if args.dry_run else "已淘汰"
    target = args.max_mb if args.max_mb is not None else settings.cache_soft_limit_mb
    _print_removed(removed, args.dry_run)
    if n == 0:
        print(f"[cache] Tier A 占用未超目标（{target} MB），无需淘汰。")
    else:
        keep = "（保留被笔记引用者）" if args.keep_referenced else ""
        print(
            f"[cache] {verb} Tier A {n} 个单元{keep}，释放 {_fmt_bytes(freed)}，剩余 {_fmt_bytes(remain)}。"
        )
    return 0


# ===========================================================================
# 参数解析与入口
# ===========================================================================
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="research",
        description="pySciWS 文献调研统一 CLI（检索 / 阅读 / 元数据 / 入库 / 库查询 / 索引）",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("doctor", help="环境与能力自检")
    sp.set_defaults(func=cmd_doctor)

    sp = sub.add_parser("search", help="多源融合检索（OpenAlex + arXiv）")
    sp.add_argument("query", help="检索式（多词请用单引号包裹）")
    sp.add_argument(
        "--source", default="auto", choices=["auto", "openalex", "arxiv", "wos", "s2"]
    )
    sp.add_argument("--year", default=None, help="如 2023-2026 / 2023- / 2023")
    sp.add_argument("--limit", type=int, default=15)
    sp.add_argument(
        "--sort", default="relevance", choices=["relevance", "date", "citations"]
    )
    sp.add_argument("--min-citations", type=int, default=None, dest="min_citations")
    sp.add_argument("--oa-only", action="store_true", dest="oa_only", help="仅开放获取")
    sp.add_argument(
        "--enrich",
        action="store_true",
        help="补 WoS 官方 JIF/JCR + S2 TLDR（逐条调用，较慢）",
    )
    sp.add_argument("--save", action="store_true", help="保存检索快照到 shortlists/")
    sp.add_argument("--purpose", default=None, help="本次检索目的（写入快照）")
    sp.set_defaults(func=cmd_search)

    sp = sub.add_parser("read", help="抓全文 + 抽取 + 建 papers/ 笔记骨架")
    sp.add_argument("target", help="DOI / URL / arXiv id / OpenAlex id")
    sp.add_argument(
        "--backend",
        default=None,
        help="PDF 抽取后端：mineru-cloud / pymupdf4llm / auto",
    )
    sp.add_argument(
        "--headed", action="store_true", help="有头浏览器（应对 Cloudflare 等）"
    )
    sp.add_argument(
        "--no-note", dest="note", action="store_false", help="只抓全文，不建笔记骨架"
    )
    sp.add_argument("--overwrite", action="store_true", help="覆盖已存在的同名笔记")
    sp.add_argument(
        "--force",
        "--refresh",
        action="store_true",
        dest="force",
        help="忽略缓存全文，强制重新抓取/抽取",
    )
    sp.set_defaults(func=cmd_read, note=True)

    sp = sub.add_parser("get", help="输出融合元数据")
    sp.add_argument("identifier", help="DOI / arXiv id / OpenAlex id")
    sp.add_argument("--json", action="store_true", help="输出 JSON（机器可读）")
    sp.set_defaults(func=cmd_get)

    sp = sub.add_parser("add", help="入库 Zotero + 建 papers/ 笔记骨架（不抓全文）")
    sp.add_argument("doi", help="DOI（或 arXiv / OpenAlex id）")
    sp.add_argument("--tags", default=None, help="逗号分隔的标签")
    sp.add_argument("--overwrite", action="store_true")
    sp.set_defaults(func=cmd_add)

    sp = sub.add_parser("library", help="查询 Zotero 库")
    sp.add_argument("action", choices=["ping", "list", "search", "get"])
    sp.add_argument("--query", default=None, help="search 的关键词")
    sp.add_argument("--key", default=None, help="get 的条目 key")
    sp.add_argument("--type", default=None, dest="type", help="list 时按 itemType 过滤")
    sp.add_argument("--limit", type=int, default=25)
    sp.set_defaults(func=cmd_library)

    sp = sub.add_parser("index", help="从 papers/ 重建 INDEX.md")
    sp.add_argument("--check", action="store_true", help="仅校验一致性，不写文件")
    sp.add_argument("--force", action="store_true", help="papers/ 为空时也重建")
    sp.set_defaults(func=cmd_index)

    sp = sub.add_parser(
        "cache", help="缓存治理（stats 概览 / clean 清 Tier B / prune 淘汰 Tier A）"
    )
    sp.add_argument("action", choices=["stats", "clean", "prune"])
    sp.add_argument(
        "--all", action="store_true", help="clean：删除全部 Tier B（不限过期）"
    )
    sp.add_argument(
        "--older-than",
        type=int,
        default=None,
        dest="older_than",
        help="clean：清理超过 N 天的 Tier B",
    )
    sp.add_argument(
        "--max-mb",
        type=int,
        default=None,
        dest="max_mb",
        help="prune：淘汰到总占用 ≤ N MB（默认软上限）",
    )
    sp.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help="只列将删除项，不实际删除",
    )
    sp.add_argument(
        "--keep-referenced",
        action=argparse.BooleanOptionalAction,
        default=True,
        dest="keep_referenced",
        help="prune：跳过被 papers/ 笔记引用的文件（默认开）",
    )
    sp.set_defaults(func=cmd_cache)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "cmd", None) in {"search", "read", "get", "add"}:
        cache_manager.maybe_autoclean()
    try:
        rc = args.func(args)
        return int(rc) if rc else 0
    except KeyboardInterrupt:
        print("\n[research] 已中断。", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
