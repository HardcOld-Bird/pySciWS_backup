"""research —— 文献调研统一 CLI 门面（literature-research skill 的后端）。

一个入口收敛「检索 / 阅读 / 元数据 / 入库 / 库查询 / 索引」的全部能力，
供 skill 文档与 LLM 直接调用，无需了解底层 13 个后端模块的实现细节。

子命令
------
    doctor   环境与能力自检（检索源 / PDF 后端 / Playwright / Zotero / 期刊指标 就绪状态）
    search   多源融合检索（OpenAlex 主 + arXiv + WoS；--enrich 追加 WoS 收录号）
    read     给 DOI/URL/arXiv id：抓全文 → 抽取 Markdown → 生成/合并 papers/ 笔记
    get      给 DOI/OpenAlex/arXiv id：输出融合后的完整元数据（--json 输出机器可读）
    add      给 DOI：入库 Zotero + 生成 papers/ 笔记骨架（不抓全文；写入前三源引用核验）
    citecheck 引用完整性门：DOI/arXiv id/标题、papers/ 笔记或参考文献文件（--bib / --review）
             走 OpenAlex+Crossref+arXiv 三源交叉核验
    citegraph 引文图谱滚雪球（backward = 它引了谁 / forward = 谁引了它；走 OpenAlex）
    library  查询 Zotero 库（ping / list / search / get）
    journal  期刊质量指标（lookup / build-scimago / status）
    review   综述笔记脚手架（new / status / sync）；**不生成综述正文**，那是 AI 的活
    rag      PaperQA2 语义检索本地文献库（index 建索引 / search 纯 embedding 检索 / ask 可选 LLM 综述 / status）
    index    扫描 papers/ 重建 INDEX.md（--check 仅校验；--fix 规范化既有笔记的 frontmatter）
    cache    缓存治理（stats 概览 / clean 清 Tier B / prune 手动 LRU 淘汰 Tier A）
    ingest   把本地已有的 PDF/Markdown 纳入本模块管理（run / status）

两种参数形态（详见 ``research -h`` 的 epilog）：动词 + 位置目标（search / read / get /
add / citecheck / citegraph / index / doctor），或名词 + action 位置参数（library /
journal / review / rag / cache / ingest）。

用法（在项目根目录）::

    python -m pysci.skills.literature_research.tools.research <cmd> [options]

设计原则
--------
1. 只做编排，不重复造轮子——所有能力复用 tools/ 下已验证的后端模块。
2. **OpenAlex 为主源**（检索 + 元数据 + JIF 估算），arXiv 为预印本源；WoS（Starter API）是
   **可选增强**而非主力：它只提供 Times Cited 与收录号（wos_id），**不含**官方 JIF / JCR
   分区 / ESI（那些需 WoS Journals API，见 :mod:`.wos_client` 模块 docstring 的升级路径）。
   未配置或调用失败时静默跳过，绝不阻塞主流程。
3. 所有产物路径基于 settings.module_dir（= data/skills/literature_research/）。
4. frontmatter 的解析与序列化统一走 :mod:`.notes` 叶子模块（基于 PyYAML），
   本模块不再自带手写 YAML 读写器。
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
    citation_verify,
    journal_metrics,
    local_ingest,
    notes,
    openalex_client,
    pdf_extract,
    rag,
    wos_client,
    zotero_cli,
)
from .config import http_session, settings

# ---------------------------------------------------------------------------
# 路径常量（全部基于模块根 settings.module_dir = data/skills/literature_research/）
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


def build_note_markdown(fm: dict[str, Any]) -> str:
    """由 frontmatter dict 生成完整 paper note（套用 templates/paper_note.md 正文骨架）。"""
    tpl = _load_template("paper_note.md")
    _skel, body = _split_template(tpl)
    if not body.strip():
        body = "\n# {{short_title}} ({{first_author_last_name}} {{year}})\n\n> 一句话定位：\n"
    body = _fill_placeholders(body, fm)
    fm = dict(fm)
    fm.setdefault("added_date", _today())
    return notes.render_note(fm, body)


# ---------------------------------------------------------------------------
# 笔记写入：合并语义（而非全有或全无）
# ---------------------------------------------------------------------------
#: :func:`_merge_note` 的四种 action 取值。
NOTE_CREATED = "created"  # 文件不存在 → 新建（正文即模板骨架）
NOTE_MERGED = "merged"  # 已存在 → 只补空字段，正文保留 + Changelog 追加一行
NOTE_UNCHANGED = "unchanged"  # 已存在且无空字段可补 → 完全不触碰文件（保留 mtime）
NOTE_OVERWRITTEN = "overwritten"  # --overwrite → 全量重建（正文重置为模板）


def _read_note_text(path: Path) -> tuple[str, str]:
    """读笔记，返回 ``(LF 归一化文本, 原文件的换行风格)``。

    ``newline=""`` 关闭 Python 的通用换行翻译，以便**探测**文件本来的换行风格；
    :func:`_write_note_text` 再按同一风格写回。若图省事直接用默认的
    ``read_text``/``write_text``，Windows 上会把一个 LF-only 笔记的正文整体翻译成
    CRLF——那是对「正文逐字节不变」这条不变量的破坏，而且在 git diff 里看不出原因。
    """
    raw = path.read_text(encoding="utf-8", newline="")
    newline = "\r\n" if "\r\n" in raw else "\n"
    return raw.replace("\r\n", "\n").replace("\r", "\n"), newline


def _write_note_text(path: Path, text: str, newline: str) -> None:
    """按 :func:`_read_note_text` 探测到的换行风格写回笔记。"""
    path.write_text(text, encoding="utf-8", newline=newline)


def _merge_note(
    fm: dict[str, Any], *, overwrite: bool = False
) -> tuple[Path, str, list[str]]:
    """把 frontmatter 合并进 ``papers/`` 下的笔记，返回 ``(path, action, changed)``。

    取代原先全有或全无的 ``_write_note``。旧实现的问题是：文件已存在且非
    ``overwrite`` 时**直接 return、什么都不写**，于是 ``cmd_read`` 辛苦算出的
    ``local_pdf_path`` 与 ``extracted_md_path`` 在标准工作流（先 ``add`` 建骨架、
    后 ``read`` 抓全文）下永远写不进笔记；连带 ``cache_manager.referenced_paths()``
    取不到这两个字段，``cache prune --keep-referenced`` 的保护对现存笔记完全失效。

    action 的四种取值见 :data:`NOTE_CREATED` / :data:`NOTE_MERGED` /
    :data:`NOTE_UNCHANGED` / :data:`NOTE_OVERWRITTEN`。关键约束：

    * 只有 :func:`notes.merge_frontmatter` 认定的**空键**会被填，用户手填的
      ``my_rating`` / ``status`` / ``related_to_my_work`` 永不被机器值顶掉；
    * ``merged`` 时正文逐字节保留，只在 ``## Changelog`` 段末追加一行审计记录；
    * ``unchanged`` 时**完全不触碰文件**（保留 mtime，免得无谓地刷新
      ``cache prune`` 赖以为 LRU 依据的时间戳）；
    * frontmatter 无法解析的既有文件（人工笔记 / 损坏）按 ``unchanged`` 处理并
      打一行提示——**绝不**自动改写它。

    Returns:
        ``changed`` 仅在 ``action == "merged"`` 时非空，为本次补齐的键列表；
        调用方据此打印准确信息（见 :func:`_report_note_action`）。
    """
    PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    path = PAPERS_DIR / notes.note_filename(fm)
    existed = path.exists()
    if not existed or overwrite:
        path.write_text(build_note_markdown(fm), encoding="utf-8")
        return path, (NOTE_OVERWRITTEN if existed else NOTE_CREATED), []
    try:
        text, newline = _read_note_text(path)
    except OSError as e:
        print(
            f"[note] 读取失败，未改动：{path}（{type(e).__name__}: {e}）",
            file=sys.stderr,
        )
        return path, NOTE_UNCHANGED, []
    old_fm, body = notes.split_note(text)
    if not old_fm:
        print(
            f"[note] {path.name} 无可解析的 frontmatter，未改动"
            "（人工笔记请手工维护；如需按模板重建请加 --overwrite）。",
            file=sys.stderr,
        )
        return path, NOTE_UNCHANGED, []
    merged, changed = notes.merge_frontmatter(old_fm, fm)
    if not changed:
        return path, NOTE_UNCHANGED, []
    new_body = notes.append_changelog(
        body, f"- {_today()}: 补齐字段 {', '.join(changed)}"
    )
    _write_note_text(path, notes.render_note(merged, new_body), newline)
    return path, NOTE_MERGED, changed


def _report_note_action(tag: str, path: Path, action: str, changed: list[str]) -> None:
    """把 :func:`_merge_note` 的四种 action 打印成准确的一两行信息。

    旧实现只有一句「笔记已存在，未覆盖（加 --overwrite 可重建）」，把「已合并补齐了
    字段」与「本来就无需改动」压成同一句话，读到的 LLM 容易误判成写入失败而反复重试。
    """
    if action == NOTE_CREATED:
        print(f"[{tag}] 笔记骨架 : {path}（新建）")
    elif action == NOTE_OVERWRITTEN:
        print(f"[{tag}] 笔记骨架 : {path}（--overwrite 重建，正文已重置为模板）")
    elif action == NOTE_MERGED:
        print(f"[{tag}] 笔记合并 : {path}")
        print(f"[{tag}]   补齐 {len(changed)} 个空字段：{', '.join(changed)}")
        print(f"[{tag}]   正文逐字节未改动；本次变更已记入笔记的 ## Changelog 段。")
    else:
        print(f"[{tag}] 笔记未变 : {path}（无空字段可补，文件未被触碰）")


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
    """用 WoS Starter API 增强单个 work dict（补收录号 wos_id）。

    静默降级：全程重定向 stdout/stderr，任何异常都吞掉并返回原 dict。
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
    finally:
        if collect is not None:
            txt = (out.getvalue() + err.getvalue()).strip()
            if txt:
                collect.append(txt)
    return work


def _overlay_enrichment(fm: dict[str, Any], work: dict[str, Any]) -> dict[str, Any]:
    """把 _enrich_work 写进 work 的 WoS 独家字段叠加到 frontmatter。

    WoS Starter API 只贡献 wos_id（收录号）；jif/jcr_quartile/esi_* 并非 Starter
    API 能力，仍由 OpenAlex 估算值填充，不在此覆盖。
    """
    for k in ("wos_id",):
        v = work.get(k)
        if v not in (None, "", False):
            fm[k] = v
    return fm


# ---------------------------------------------------------------------------
# 期刊质量指标补齐
# ---------------------------------------------------------------------------
def _lookup_source(fm: dict[str, Any]) -> dict[str, Any] | None:
    """按「DOI 优先、刊名兜底」的顺序取 OpenAlex 的 source（期刊）记录。

    两级查找的精度差很多，所以先试精确的：

    1. ``fm["doi"]`` → ``get_work`` → ``journal_openalex_id`` → ``get_source``：走 OpenAlex
       自己的论文→期刊关联，**唯一确定**；
    2. ``get_source(name=fm["journal"])``：走 ``sources?search=``，模糊匹配取首个命中。
       缩写刊名（``Phys. Rev. Lett.``）通常能命中全名记录，但不保证——这是兜底路径。

    两级都拿不到时返回 ``None``，**不抛异常**。
    """
    doi = str(fm.get("doi") or "").strip()
    if doi:
        w = openalex_client.get_work(doi=doi)
        sid = str((w or {}).get("journal_openalex_id") or "").strip()
        if sid:
            src = openalex_client.get_source(openalex_id=sid)
            if src:
                return src
    name = str(fm.get("journal") or "").strip()
    return openalex_client.get_source(name=name) if name else None


def _attach_journal_metrics(fm: dict[str, Any]) -> dict[str, Any]:
    """为已生成的 frontmatter 补齐期刊质量指标，**就地修改并返回** ``fm``。

    存在的理由是一个实测缺陷：arXiv 来源的笔记 ``jif`` / ``journal_h_index`` 恒为
    ``null``——不是因为该刊无数据（PRL 的 OpenAlex ``2yr_mean_citedness`` = 8.97），
    而是 ``openalex_client.work_to_note_frontmatter`` 只在 ``journal_openalex_id`` 存在时
    才查 source，而 arXiv entry 根本没有这个字段。本函数按 DOI / 刊名重走一遍。

    触发条件（三条同时满足，避免无谓的网络请求）：

    1. ``fm["jif"]`` 为空——已有值说明上游已经查过；
    2. ``fm["journal"]`` 非空；
    3. ``fm["journal"]`` 不是 :data:`~.arxiv_client.ARXIV_PREPRINT_JOURNAL`（那是「尚未
       发表」的占位，不是刊名，拿它去搜期刊会得到完全无关的命中）。

    全程静默降级（不变量 1）：任何异常都吞掉，拿不到指标时只向 stderr 留一行提示。
    额外请求走 Tier B ``api_responses`` 缓存，重复调用不产生新的网络开销。
    """
    journal = str(fm.get("journal") or "").strip()
    # 不用 ``not in (None, "")``：``jif`` 可以是合法的 ``0``（零引用密度刊），而
    # ``0 not in (None, "")`` 为真，会把已有值误当成「未查过」而重走一遍网络请求。
    if fm.get("jif") is not None and fm.get("jif") != "":
        return fm
    if not journal or journal == arxiv_client.ARXIV_PREPRINT_JOURNAL:
        return fm

    src: dict[str, Any] | None = None
    out, err = io.StringIO(), io.StringIO()
    try:
        # 客户端内部失败时会直接 print；这里全部拦下，改由本函数输出一行统一提示。
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            src = _lookup_source(fm)
    except Exception:  # noqa: BLE001 —— 降级路径不得让 read/add/get 整体失败
        src = None

    if not src:
        print(
            f"[journal] {journal}：未取到期刊指标（OpenAlex 无匹配或网络不可达），"
            "jif / journal_h_index 保持为空。",
            file=sys.stderr,
        )
        return fm

    filled: list[str] = []
    jif = src.get("2yr_mean_citedness")
    if isinstance(jif, (int, float)):
        fm["jif"] = round(float(jif), 2)
        filled.append("jif")
    if fm.get("journal_h_index") in (None, "") and src.get("h_index") is not None:
        fm["journal_h_index"] = src["h_index"]
        filled.append("journal_h_index")

    # E1：listed_in 是**零额外请求**就能拿到的专家评议分级（JUFO / Norway / KI-JL）。
    # 对物理声学这类低引用密度领域，它比 JIF 更贴近领域共识：JASA 的
    # 2yr_mean_citedness 只有 0.82，但 JUFO 把它判为最高档，与 Nature / PRL 同级。
    listed_in = src.get("listed_in") or []
    if listed_in and not fm.get("listed_in"):
        fm["listed_in"] = listed_in
        filled.append("listed_in")
    if listed_in and not fm.get("journal_tier"):
        tier, basis = notes.derive_journal_tier(listed_in)
        if tier:
            fm["journal_tier"] = tier
            fm["journal_tier_basis"] = basis
            filled.append("journal_tier")

    # E2：SCImago 分区按 ISSN 精确匹配本地索引；索引未建时静默留空（不变量 1）。
    if not fm.get("scimago_quartile"):
        quartile = journal_metrics.quartile_for(src.get("issn") or [])
        if quartile:
            fm["scimago_quartile"] = quartile
            filled.append("scimago_quartile")

    # 注：此处**不**把 ``fm["journal"]`` 改写成 OpenAlex 的规范全名。能走到这里说明刊名已
    # 非空且不是占位串，改写就等于覆盖数据源/人工给出的值；缩写与全名的差异交给展示层。

    if filled:
        print(
            f"[journal] {journal}：补齐 {', '.join(filled)}"
            "（JIF 为 OpenAlex 2yr_mean_citedness 估算值，非官方 JCR；"
            "journal_tier 为 JUFO/Norway/KI-JL 专家评议派生值）。"
        )
    return fm


def _build_frontmatter(src: str | None, work: dict[str, Any]) -> dict[str, Any]:
    """生成 frontmatter 的**唯一漏斗**：转换 → WoS 叠加 → 期刊指标补齐。

    ``cmd_read`` / ``cmd_get`` / ``cmd_add`` 三处原本各自拼一遍同样的三步。收成一个函数
    的理由不只是 DRY：期刊指标补齐是容易忘的一步，放在漏斗里就不会有第四个调用点遗漏。
    """
    fm = _overlay_enrichment(_frontmatter_from_work(src or "openalex", work), work)
    return _attach_journal_metrics(fm)


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
    # arXiv 的 work dict 只有自由文本 ``journal_ref``（一整串引文）。直接当刊名展示会
    # 把 Journal 列变成 ``Phys. Rev. Lett. 121, 124501 (2018)``，与 frontmatter 里刚修好的
    # ``journal`` 字段不一致，所以这里走同一个解析器。
    journal = str(w.get("journal") or "")
    if not journal and w.get("journal_ref"):
        journal = arxiv_client.parse_journal_ref(w["journal_ref"])["journal"]
    if not journal and src == "arxiv":
        journal = "arXiv"
    # journal_tier 对 OpenAlex 行是**直接就有**的：``_extract_work_summary`` 从 work 响应
    # 内联的 ``primary_location.source.listed_in`` 零额外请求地派生了它（实测 Nature
    # Physics 的内联 source 给 ``cwts-core,jufo-3,ki-jl-2,norway-2``），于是 search /
    # citegraph / get 三条路径的展示行都带档次。
    # 下面那个 ``listed_in`` 回退不是给它们的，而是给**只**拿到原始 source 数据的
    # 调用方（例如把 :func:`openalex_client._extract_source` 的结果叠上去）：它只比主
    # 路径多一次纯函数调用，却使展示层永不比 frontmatter 少一维信息。
    # arXiv 行两者皆无（Atom 响应里没有期刊元数据，无从派生），tier 为空属正常。
    tier = str(w.get("journal_tier") or "")
    if not tier and w.get("listed_in"):
        tier = notes.derive_journal_tier(w["listed_in"])[0]
    return {
        "title": title,
        "first_author_last_name": w.get("first_author_last_name") or "",
        "year": year,
        "journal": journal,
        "doi": w.get("doi") or "",
        "arxiv_id": w.get("arxiv_id") or "",
        "openalex_id": w.get("openalex_id") or "",
        "cited_by_count": w.get("cited_by_count"),
        "oa_status": w.get("oa_status") or ("green" if src == "arxiv" else ""),
        "jif": w.get("jif"),
        "jcr_quartile": w.get("jcr_quartile") or "",
        "journal_tier": tier,
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
            or notes.slugify(r["title"], 60)
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
        bits: list[str] = []
        if r["jif"]:
            bits.append(f"JIF {r['jif']}")
        # jcr_quartile 需 WoS Journals API（未接入），故过去这个分支**恒为假**、等于死
        # 代码。回退到 journal_tier（OpenAlex listed_in 的专家评议派生值）后尾注才有用。
        if r["jcr_quartile"]:
            bits.append(r["jcr_quartile"])
        elif r["journal_tier"]:
            bits.append(f"tier:{r['journal_tier']}")
        jif_s = " ".join(bits)
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
        # 同 _print_rows：jcr_quartile 恒为空，回退到 journal_tier。
        quart = ""
        if r["jcr_quartile"]:
            quart = f" {r['jcr_quartile']}"
        elif r["journal_tier"]:
            quart = f" tier:{r['journal_tier']}"
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
    """套用 templates/shortlist.md 生成检索快照，写入 shortlists/。

    ``args`` 上除 ``query`` 外的字段全部用 ``getattr`` 取默认值：本函数不只服务
    ``search``，``citegraph --save`` 也走这里（滚雪球结果同样需要可复现快照），
    而后者没有 ``--min-citations`` / ``--source`` 这些参数。
    """
    SHORTLISTS_DIR.mkdir(parents=True, exist_ok=True)
    today = _today()
    qslug = notes.slugify(query, 40)
    tpl = _load_template("shortlist.md")
    _skel, body = _split_template(tpl)
    sources_used = _sources_used(args)
    fm = {
        "query_date": today,
        "query_slug": qslug,
        "queried_by": "AI Agent (research.py)",
        "purpose": getattr(args, "purpose", None) or query,
        "sources": sources_used,
        "query_string": query,
        "filters": {
            "year_range": getattr(args, "year", None) or "",
            "min_citations": getattr(args, "min_citations", None),
            "venues": [],
            "concepts": [],
            "oa_only": bool(getattr(args, "oa_only", False)),
        },
        "sort_by": SORT_MAP_OPENALEX.get(
            getattr(args, "sort", "relevance"), getattr(args, "sort", "relevance")
        ),
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
    out.write_text(notes.render_note(fm, body), encoding="utf-8")
    return out


def _sources_used(args: argparse.Namespace) -> list[str]:
    src = getattr(args, "source", "auto")
    if src == "auto":
        used = ["openalex", "arxiv"]
        if getattr(args, "enrich", False) and settings.wos_ready:
            used.append("wos")
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
    if settings.openalex_api_key:
        oa = "就绪（已配置 OPENALEX_API_KEY）"
    elif settings.openalex_email:
        oa = "降级（仅设 mailto；OpenAlex 自 2026-02-13 起需 API key，否则限 100 credits/天）"
    else:
        oa = "降级（未配置 OPENALEX_API_KEY，限 100 credits/天测试配额；建议申请免费 key）"
    print(f"  OpenAlex         : {oa}（主源）")
    print("  arXiv            : 就绪（无需 key）")
    print("  Crossref         : 就绪（引用完整性门三源之一，无需 key）")
    print(
        f"  Web of Science   : {'就绪（补收录号 wos_id + Times Cited）' if settings.wos_ready else '未配置（可选增强源）'}"
    )
    print(
        f"  Elsevier/Scopus  : {'已配置 key' if settings.elsevier_api_key else '未配置（预留）'}"
    )
    print()

    print("【期刊质量指标】")
    print(
        "  OpenAlex listed_in : 就绪（无需凭据；派生 journal_tier——"
        "JUFO/Norway/KI-JL 专家评议）"
    )
    _sjr = journal_metrics.status()
    if not _sjr["exists"]:
        sjr = (
            "未建（运行 `research journal build-scimago --csv <官方 CSV>`；"
            "scimago_quartile 将留空）"
        )
    elif not _sjr["readable"]:
        sjr = f"文件存在但无法解析（{_sjr['path']}）——建议重建"
    else:
        sjr = f"已建（SJR {_sjr['sjr_year'] or '?'} 版，{_sjr['n_entries']} 条 ISSN）"
    print(f"  SCImago SJR 索引   : {sjr}")
    print(
        "  WoS Journals API   : 未接入（官方 JIF / JCR 分区 / JCI / ESI 的唯一"
        "程序化来源；申请中）"
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
        f"  zotero-cli       : {'已安装（社区 zotero-mcp）' if zotero_cli.available() else '未安装——运行 scripts/zotero_mcp/setup_zotero_mcp.ps1'}"
    )
    print(
        f"  Web API 凭据     : {'就绪' if settings.zotero_web_ready else '未配置（本地模式无需；Web 模式需 ZOTERO_USER_ID / ZOTERO_API_KEY）'}"
    )
    if zotero_cli.available():
        try:
            zb = zotero_cli.ZoteroCli()
            info = zb.ping()
            print(
                f"  连通性           : 后端={zb.backend}；ping ok={info.get('ok')}"
                + (f"；{info.get('error')}" if not info.get("ok") else "")
            )
        except Exception as e:
            print(f"  连通性           : 探测失败（{type(e).__name__}）")
    print()

    print("【PaperQA2 RAG 语义检索】")
    _st = rag.index_status()
    print(
        f"  后端就绪         : {'是（硅基流动 embedding）' if _st['ready'] else '否——缺 SILICONFLOW_API_KEY'}"
    )
    print(f"  embedding 模型   : {_st['embedding_model']}")
    if _st["exists"]:
        print(
            f"  本地索引         : docs={_st['n_docs']} chunks={_st['n_chunks']}（built {_st['built_at'] or '?'}）"
        )
    else:
        print("  本地索引         : 未建（运行 `research rag index`）")
    print(
        f"  ask LLM          : {settings.pqa_llm}"
        + (f"（回退 {settings.pqa_llm_fallback}）" if settings.pqa_llm_fallback else "")
    )
    print()
    print("=== 自检结束 ===")
    return 0


# ===========================================================================
# 子命令：search
# ===========================================================================
def cmd_search(args: argparse.Namespace) -> int:
    """多源融合检索。

    ``--json`` 时 stdout 只输出 :func:`_row` 的 dict 数组（LLM 做程序化筛选时唯一可靠的
    形态——人类可读的对齐文本只能靠猜列宽解析），各源的取回条数等进度提示一律改走
    stderr，否则 ``search --json | ConvertFrom-Json`` 在第一行就失败（同
    :func:`cmd_citegraph` 的约定）。
    """
    as_json = bool(getattr(args, "json", False))

    def _say(msg: str) -> None:
        """进度提示：JSON 模式下给 stdout 上的纯 JSON 让路。"""
        print(msg, file=sys.stderr if as_json else sys.stdout)

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
        _say(
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
        _say(
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
        _say(
            f"[search] WoS: 取回 {len(res.get('hits', []))} 条（总匹配 {res.get('total')}）"
        )

    rows = _dedupe(rows)

    # 批量增强（--enrich）：对 OpenAlex 条目补 WoS 收录号（wos_id）
    if getattr(args, "enrich", False) and source == "auto":
        if settings.wos_ready:
            enriched: list[tuple[str, dict]] = []
            for src, w in rows:
                if src == "openalex" and w.get("doi"):
                    w = _enrich_work(w, collect=collected)
                enriched.append((src, w))
            rows = enriched
        else:
            _say("[search] --enrich 已忽略：WoS 未配置。")

    # auto 融合后可能超过 limit（两源相加），截断
    if source == "auto":
        rows = rows[:limit]

    # 降级态下的空结果**不可判读**：可能是 100 credits/天 的配额已耗尽，也可能是该主题
    # 确实无文献。静默返回空集会让调用方（尤其是 LLM）直接得出后一个结论，并据此
    # 写进综述——这是最坏的失败形态，因为它看起来完全正常。故此处破例不静默。
    # 只在真走过 OpenAlex 那条分支时告警：``--source arxiv`` 压根没查 OpenAlex，
    # 对它喊配额耗尽是误导。走 stderr：它是告警而非结果，且 JSON 模式下 stdout 必须纯净。
    if not rows and source in ("auto", "openalex") and not settings.openalex_api_key:
        print(
            "[search] 注意：OpenAlex 处于无 key 降级态（限 ~100 credits/天）。\n"
            "         空结果可能是配额耗尽，而非该主题确实无文献——"
            "建议改用 arxiv-mcp-server 或 --source arxiv 复核。",
            file=sys.stderr,
        )

    if as_json:
        # 空结果也是合法的 JSON（``[]``），不能退化成一句「（无结果）」的人类文本：
        # 调用方分不出「这个方向真没文献」与「我的解析器坏了」。
        print(json.dumps([_row(s, w) for s, w in rows], ensure_ascii=False, indent=2))
    else:
        print()
        _print_rows(rows)

    if collected:
        _say(
            f"[search] 注：{len(collected)} 处 WoS 增强不可用，相关条目已回退到 OpenAlex 估算值。"
        )

    if args.save:
        path = _write_shortlist(query, args, rows)
        _say(f"[search] 检索快照已保存：{path}")
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


#: OA 直链下载时覆盖 ``http_session`` 默认的 ``Accept: application/json``——那个头是给
#: 元数据 API 用的，拿去要 PDF 会被部分出版商的内容协商直接拒掉。
_PDF_ACCEPT = "application/pdf,application/octet-stream;q=0.9,*/*;q=0.8"

#: PDF 魔数长度。只验前 5 字节：它是事实，而 ``Content-Type`` 不是（机构库常返
#: ``application/octet-stream``，出版商的「无权访问」页面常返 200 + ``text/html``）。
_PDF_MAGIC = b"%PDF-"


def _oa_pdf_filename(work: dict[str, Any] | None, oa_url: str) -> str:
    """为 OA 直链下载的 PDF 取一个**稳定**的文件名。

    优先用 DOI / OpenAlex id 派生而不是 URL：同一篇论文换个镜像（机构库 vs 出版商
    vs PMC）时仍复用同一个文件。按 URL 命名（或其哈希）会把「同一篇」拆成「多个
    不同文件」：既攒出多份副本，也让 ``cache prune --keep-referenced`` 认不出笔记
    引用的那一份。
    """
    w = work or {}
    stem = str(w.get("doi") or w.get("openalex_id") or oa_url)
    return notes.slugify(stem, 60) + ".pdf"


def _try_oa_pdf(url: str, dest: Path, *, force: bool = False) -> Path | None:
    """一次普通 HTTP GET 直接取回 OA PDF；任何不成立的情形都返回 ``None``。

    这是 :func:`cmd_read` 抓取顺序里的**第一档**（方案 H5）：``oa_url`` 是 OpenAlex
    明确标为开放获取的直链，普通 GET 就能拿到，不必为它启动一次 Playwright。

    判据只看魔数 :data:`_PDF_MAGIC` 而**不看** ``Content-Type``（理由见该常量）。
    先写 ``.part`` 再原子改名：直接写 ``dest`` 的话，一次中断的下载会留下一个
    截断的 ``.pdf``，而下次进来会被当成「已在缓存」直接复用——那比下载失败难查得多。

    全路径降级静默（不变量 1）：非 200、非 PDF、网络异常一律返回 ``None``，
    由调用方回落到浏览器抓取。
    """
    if dest.exists() and not force:
        cache_manager.bump_mtime(dest)  # 命中 touch，使 mtime≈最近访问（供 prune LRU）
        print(f"[read] OA PDF 已在缓存：{dest}")
        return dest
    tmp = dest.with_name(dest.name + ".part")
    try:
        with http_session() as s:
            r = s.get(
                url,
                headers={"Accept": _PDF_ACCEPT},
                timeout=settings.http_timeout * 2,
                stream=True,
                allow_redirects=True,
            )
            if r.status_code != 200:
                return None
            chunks = r.iter_content(chunk_size=65536)
            # 累到足以验魔数为止，而不是假定第一个 chunk 就有 5 字节：
            # ``iter_content`` 只保证**至多** chunk_size，不保证至少。
            head = b""
            for chunk in chunks:
                head += chunk or b""
                if len(head) >= len(_PDF_MAGIC):
                    break
            if not head.startswith(_PDF_MAGIC):
                # 200 + 非 PDF：几乎总是出版商的落地页或「需登录」提示。
                # 不是错误，只是这条路走不通——不刷错误日志。
                return None
            dest.parent.mkdir(parents=True, exist_ok=True)
            with tmp.open("wb") as f:
                f.write(head)
                for chunk in chunks:
                    f.write(chunk or b"")
        tmp.replace(dest)
    except Exception as e:  # noqa: BLE001 —— 降级路径，网络不可达不该变成 traceback
        print(
            f"[read] OA 直链下载失败（{type(e).__name__}），改用浏览器抓取。",
            file=sys.stderr,
        )
        return None
    finally:
        # 成功时 tmp 已被 replace 掉，missing_ok 使其成为空操作；失败时清掉半截文件。
        tmp.unlink(missing_ok=True)
    return dest


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
        fm = _build_frontmatter(src, work)
    fulltext_path: Path | None = None
    if fm is not None:
        stem = notes.slugify(
            str(fm.get("short_title") or fm.get("title") or "paper"), 40
        )
        fulltext_path = settings.cache_extracted / f"{stem}_fulltext.md"

    # 命中复用：已有规范全文且非 --refresh → 直接读缓存，跳过抓取与抽取
    md = ""
    if fulltext_path is not None and fulltext_path.exists() and not force:
        md = fulltext_path.read_text(encoding="utf-8")
        cache_manager.bump_mtime(fulltext_path)
        print(
            f"[read] 命中缓存全文，跳过抓取/抽取（--refresh 强制重取）：{fulltext_path}"
        )

    # 2) 抓取 PDF / 全套（命中缓存则整体跳过）
    bundle = None
    pdf_path: Path | None = None
    html_path: Path | None = None
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
            # 抓取顺序（方案 H5）：**先试 OA 直链的普通 HTTP GET**（免费、不启动浏览器），
            # 失败才起 Playwright。原实现对 DOI 一律拼 ``https://doi.org/{doi}`` 直接进浏览器，
            # 而 ``oa_url`` 只在 openalex 分支用到——对 OpenAlex 已标明 OA 的论文，
            # 这等于每次都为一份能直接 GET 到的 PDF 白付一次浏览器启动 + 页面渲染。
            oa_url = str((work or {}).get("oa_url") or "")
            direct = oa_url or (val if kind == "url" else "")
            if direct:
                print(f"[read] 先试 OA 直链（免浏览器）：{direct}")
                pdf_path = _try_oa_pdf(
                    direct,
                    settings.cache_pdfs / _oa_pdf_filename(work, direct),
                    force=force,
                )
            if not pdf_path:
                if kind == "url":
                    url = val
                elif kind == "doi":
                    url = f"https://doi.org/{val}"
                else:  # openalex
                    url = oa_url or (
                        f"https://doi.org/{work['doi']}"
                        if work and work.get("doi")
                        else ""
                    )
                    if not url:
                        print(
                            "[read] 无法从 OpenAlex id 解析出可抓取 URL。",
                            file=sys.stderr,
                        )
                        return 2
                print(f"[read] 抓取：{url}")
                bundle = _safe_fetch(url, out_dir, args)
                pdf_path = bundle.pdf_path if bundle else None

        # 3) 抽取全文 → Markdown（write_cache=False：只留规范的 {stem}_fulltext.md，消除双副本）
        if pdf_path and Path(pdf_path).exists():
            # arXiv LaTeX 源码优先：公式保留为原始 LaTeX，而不是 PDF 解析出的碎片，
            # 对物理论文是实打实的抽取质量提升。id 优先取用户直接给的（kind == "arxiv"），
            # 否则取元数据里的 arxiv_id——从 DOI / OpenAlex 入口进来的论文若有 arXiv
            # 预印本，同样能走这条更准的路径。取不到就是 None，extract_pdf 直接抽 PDF。
            #
            # 这里 fm 可能仍是 None（url 形态、或元数据解析失败，第 4 步才兜底成
            # _minimal_fm），故用 (fm or {}) 而不是 fm.get。
            aid = val if kind == "arxiv" else str((fm or {}).get("arxiv_id") or "")
            print(
                f"[read] 抽取 PDF（backend={args.backend or 'auto'}"
                + (f"，优先 arXiv LaTeX 源码 {aid}" if aid else "")
                + "）..."
            )
            try:
                md = pdf_extract.extract_pdf(
                    pdf_path,
                    backend=args.backend or None,
                    write_cache=False,
                    prefer_latex_source=aid or None,
                )
            except Exception as e:
                print(f"[read] PDF 抽取失败：{type(e).__name__}: {e}", file=sys.stderr)
        if not md and bundle and bundle.html_path and Path(bundle.html_path).exists():
            # HTML 兜底（方案 H5）：**不**把它当成 PDF 抽取的等价物。
            #
            # ``bundle.html_path`` 是 browser_fetch 已经写好的
            # ``cache/html_fulltext/<slug>/<slug>.md``——内容是 trafilatura 抽出的结构化
            # markdown（**不是**原始 HTML），头部还带 browser_fetch 自己的 frontmatter。
            # 旧实现把它整篇读进 ``md``，于是被写进 ``cache/extracted/<stem>_fulltext.md``
            # 并记为 ``extracted_md_path``，后果有两个：RAG 语料里混进一段来源与质量都
            # 不同的网页正文（公式通常丢失，见该文件自己的 equations_note），以及同一份
            # 内容在两个 Tier A 目录里各存一遍。
            #
            # 现在只把**已存在的**路径记进 ``extracted_html_path``：不复制、不设
            # ``extracted_md_path``。``rag`` 只索引 ``cache/extracted/**/*.md``，语料因此
            # 保持纯粹；``cache_manager.REFERENCED_FIELDS`` 收录了新字段，故这份网页全文
            # 照样受 ``prune --keep-referenced`` 保护。
            html_path = Path(bundle.html_path)
            print("[read] 无 PDF 或未抽出，回退到网页正文（不进 RAG 语料）。")

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
            stem = notes.slugify(
                str(fm.get("short_title") or fm.get("title") or "paper"), 40
            )
            fulltext_path = settings.cache_extracted / f"{stem}_fulltext.md"
        if force or not fulltext_path.exists():
            fulltext_path.write_text(md, encoding="utf-8")
        fm["extracted_md_path"] = str(fulltext_path)
    elif html_path is not None:
        # 只记路径，不复制文件（它已在 cache/html_fulltext/ 里）。
        fm["extracted_html_path"] = str(html_path)
    note_path: Path | None = None
    note_action = NOTE_UNCHANGED
    note_changed: list[str] = []
    if args.note:
        note_path, note_action, note_changed = _merge_note(fm, overwrite=args.overwrite)

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
    # ``fulltext_path`` 在第 1.5 步就按标题算出来了，与「是否真拿到全文」无关。
    # 旧报告只判 ``if fulltext_path``，于是抽取失败时也会打一行
    # 「全文 MD : <路径>（0 字符）——精读请 Read 此文件」，指向一个**不存在**的
    # 文件。那是一句看起来完全正常的谎话，故改为以 ``md`` 为判据。
    if md and fulltext_path:
        print(
            f"[read] 全文 MD  : {fulltext_path}（{len(md)} 字符）——精读请 Read 此文件"
        )
    elif html_path is not None:
        print(f"[read] 全文 HTML: {html_path}——精读请 Read 此文件")
        print(
            "[read] 注意：这是浏览器抓取的网页正文（trafilatura 抽取），不是 PDF 抽取：\n"
            "         公式常以图片/SVG 呈现而丢失。需公式请取得 PDF 后加 --refresh 重抽，\n"
            "         或改走 arXiv 入口（它会自动试 LaTeX 源码）。\n"
            "         它刻意未写入 cache/extracted/，因此不在 rag 语料里。",
            file=sys.stderr,
        )
    else:
        print("[read] 警告：未获得全文（PDF 抓取与抽取均失败）。", file=sys.stderr)
    if note_path:
        _report_note_action("read", note_path, note_action, note_changed)
    return 0 if (md or html_path is not None or note_path) else 1


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
    """输出融合元数据（OpenAlex + WoS 收录号，若可用）。"""
    src, work = _resolve_work(args.identifier)
    if not work:
        print(f"[get] 未找到：{args.identifier}", file=sys.stderr)
        return 1
    collected: list[str] = []
    work = _enrich_work(work, collect=collected)
    fm = _build_frontmatter(src, work)
    if args.json:
        print(json.dumps(fm, ensure_ascii=False, indent=2))
    else:
        print(f"---\n{notes.dump_frontmatter(fm)}---")
    if collected:
        print(
            f"[get] 注：{len(collected)} 处 WoS 增强不可用，已用 OpenAlex 估算值。",
            file=sys.stderr,
        )
    return 0


# ===========================================================================
# 引用完整性门（阶段6）：三源核验 + citecheck 子命令
# ===========================================================================
def _citation_gate(
    fm: dict[str, Any], *, force: bool = False, tag: str = "add"
) -> bool:
    """写入前引用核验门。

    - 三源（OpenAlex+Crossref+arXiv）实质冲突 → FAIL：非 ``force`` 时返回 False（阻止写入）；
    - PASS / WARN / NOT_FOUND → 放行（True）；
    - 核验本身抛异常（网络全断等）→ 优雅降级放行（True），绝不因核验故障卡住主流程。

    ``force`` 形参对应的 CLI 开关是 ``add --allow-fail``（旧名 ``--force`` 仍可用，见
    :data:`_FORCE_RENAMED`）。形参名保持不变：它是内部 API，与 CLI 拆名无关。
    """
    try:
        v = citation_verify.verify_frontmatter(fm)
    except Exception as e:
        print(f"[{tag}] 引用核验跳过（{type(e).__name__}: {e}）。", file=sys.stderr)
        return True
    print(citation_verify.render_verdict(v, color=sys.stdout.isatty()))
    if v.status == citation_verify.FAIL:
        if force:
            print(
                f"[{tag}] ✗ 核验 FAIL，但 --allow-fail 已指定，继续写入。",
                file=sys.stderr,
            )
            return True
        print(
            f"[{tag}] ✗ 引用核验未通过（三源实质冲突），已阻止写入。"
            "修正引用后重试；或 --allow-fail 越过、--no-verify 跳过。",
            file=sys.stderr,
        )
        return False
    return True


def _cite_from_identifier(raw: str) -> dict[str, Any]:
    """把裸标识（DOI / arXiv id / OpenAlex id / 标题）转为 verify_citation 的 cite dict。

    ``_classify_id`` 对未知形式兜底归为 ``doi``；此处额外要求值确实匹配 DOI 形态
    （``10.<前缀>/``），否则归为自由文本标题——避免把论文标题误当非法 DOI。
    """
    kind, val = _classify_id(raw)
    if kind == "doi" and re.match(r"^10\.\d{4,9}/", val):
        return {"doi": val}
    if kind == "arxiv":
        return {"arxiv_id": val}
    if kind == "openalex":
        return {"openalex_id": val}
    return {"title": raw}  # 自由文本标题 / url / 非法 DOI → 尽力当标题核验


def _verdict_to_json(v: Any) -> dict[str, Any]:
    """把 CitationVerdict 投影为机器可读 dict（--json 输出）。"""
    return {
        "label": v.label,
        "kind": v.kind,
        "status": v.status,
        "passed": v.passed(),
        "reasons": v.reasons,
        # 来自参考文献文件的引用带上原文定位（parse_markdown_references 填的），
        # 一份 200 行的参考文献段里光看 DOI 不好定位。笔记/裸标识没这两个键。
        **(
            {"line": v.input["_line"], "raw": v.input.get("_raw", "")}
            if "_line" in v.input
            else {}
        ),
        "sources": {
            name: {
                "reachable": r.reachable,
                "found": r.found,
                "title": r.title,
                "first_author_last_name": r.first_author_last_name,
                "year": r.year,
                "journal": r.journal,
                "doi": r.doi,
                "error": r.error,
            }
            for name, r in v.records.items()
        },
        "conflicts": [
            {
                "field": c.field,
                "worst": c.worst,
                "values": c.values,
                "pairs": [f"{a}!={b}" for a, b, _ in c.conflicts],
            }
            for c in v.checks
            if c.conflicts
        ],
    }


def _collect_bib_citations(
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[str]]:
    """从 ``--bib`` / ``--review`` 指定的文件里解析出待核引用。

    返回 ``(cites, notices)``。``notices`` 是给用户的**一行简报**（每个文件解析出几条、
    被 ``--limit`` 截断了多少）：解析器对坏行是静默跳过的（best-effort），不报出来
    用户会把「这份 .bib 里 200 条全是非标准格式」误读成「这份 .bib 是空的」。

    文件格式按后缀判定：``.bib`` 走 :func:`citation_verify.parse_bibtex`，其余（包括
    ``.md``）走 :func:`citation_verify.parse_markdown_references`。
    """
    notices: list[str] = []
    paths: list[Path] = [Path(b) for b in (getattr(args, "bib", None) or [])]
    if getattr(args, "review", False):
        if not REVIEWS_DIR.is_dir():
            notices.append("[citecheck] --review：reviews/ 目录不存在，跳过。")
        else:
            found = sorted(REVIEWS_DIR.glob("*.md"))
            if not found:
                notices.append("[citecheck] --review：reviews/ 下没有 .md，跳过。")
            paths.extend(found)

    cites: list[dict[str, Any]] = []
    for p in paths:
        if not p.exists():
            notices.append(f"[citecheck] 跳过 {p}：文件不存在。")
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            notices.append(f"[citecheck] 跳过 {p.name}：{type(e).__name__}: {e}")
            continue
        if p.suffix.lower() == ".bib":
            got = [
                citation_verify.citation_from_bibtex(e)
                for e in citation_verify.parse_bibtex(text)
            ]
            kind = "BibTeX 条目"
        else:
            got = citation_verify.parse_markdown_references(text)
            kind = "参考文献行"
        got = [g for g in got if g]  # 只有 @misc{} 壳子的空条目不参与核验
        notices.append(f"[citecheck] {p.name}：解析出 {len(got)} 条 {kind}。")
        cites.extend(got)
    return cites, notices


def cmd_citecheck(args: argparse.Namespace) -> int:
    """引用完整性门：对 DOI/arXiv id/标题、papers/ 笔记或参考文献文件做三源交叉核验。

    退出码：存在 FAIL（三源实质冲突）→ 1；无可核引用 → 2；其余 0（可作 CI/写入前门）。

    能力边界（SKILL.md 已写明，此处再钉一次）：本命令核的是「该引用**存在**且三源
    元数据一致」，核不了「综述对这篇论文说的话是否真的是这篇论文说的」。后者靠约定：
    综述里每条事实性论断带一个指向 ``cache/extracted/*.md`` 的锚点。
    """
    verdicts: list[Any] = []
    color = sys.stdout.isatty() and not getattr(args, "no_color", False)

    # 1) 笔记文件：--all 扫 papers/；--note 指定文件/目录
    note_paths: list[Path] = []
    if getattr(args, "all", False):
        note_paths.extend(sorted(PAPERS_DIR.glob("*.md")))
    for np in getattr(args, "note", None) or []:
        p = Path(np)
        if p.is_dir():
            note_paths.extend(sorted(p.glob("*.md")))
        else:
            note_paths.append(p)
    for p in note_paths:
        try:
            verdicts.append(citation_verify.verify_note_file(p))
        except Exception as e:
            print(
                f"[citecheck] 跳过 {p.name}：{type(e).__name__}: {e}", file=sys.stderr
            )

    # 2) 参考文献文件：--bib <path>（.bib / .md）与 --review（扫 reviews/*.md）
    bib_cites, notices = _collect_bib_citations(args)
    limit = int(getattr(args, "limit", 50) or 0)
    if limit and len(bib_cites) > limit:
        notices.append(
            f"[citecheck] 已按 --limit {limit} 截断：共解析出 {len(bib_cites)} 条，"
            f"只核验前 {limit} 条（每条最多 3 次网络请求）；--limit 0 取消上限。"
        )
        bib_cites = bib_cites[:limit]
    for n in notices:
        print(n, file=sys.stderr)
    for c in bib_cites:
        verdicts.append(citation_verify.verify_citation(c))

    # 3) 裸标识（DOI / arXiv id / OpenAlex id / 标题）——用户亲手敲的，**不受 --limit 限制**
    for t in getattr(args, "targets", None) or []:
        verdicts.append(citation_verify.verify_citation(_cite_from_identifier(t)))

    if not verdicts:
        print(
            "[citecheck] 无引用可核验（给 DOI/arXiv id/标题、--note <path>、--bib <path> 或 --all）。",
            file=sys.stderr,
        )
        return 2

    if getattr(args, "json", False):
        print(
            json.dumps(
                [_verdict_to_json(v) for v in verdicts], ensure_ascii=False, indent=2
            )
        )
    else:
        print(citation_verify.render_report(verdicts, color=color))

    n_fail = sum(1 for v in verdicts if v.status == citation_verify.FAIL)
    return 1 if n_fail else 0


# ===========================================================================
# 子命令：citegraph（引文图谱 / 滚雪球）
# ===========================================================================
def _resolve_openalex_work(raw: str) -> tuple[str, dict[str, Any] | None]:
    """把任意标识解析为 **OpenAlex** work_summary（引文图谱专用）。

    与 :func:`_resolve_work` 的区别：后者对 arXiv id 会走 arXiv 自己的 API，而 arXiv 的
    Atom 响应里**没有**引文网络（``referenced_works`` 与 ``cites:`` 过滤都只有 OpenAlex
    有）。因此这里一律走 OpenAlex：arXiv id 用 OpenAlex 为预印本自动登记的 DOI 形态
    ``10.48550/arXiv.<id>`` 去查。

    返回 ``(kind, work)``，``kind`` 只用于错误信息；查不到时 ``work is None``。
    """
    kind, val = _classify_id(raw)
    if kind == "url":
        return kind, None
    try:
        if kind == "openalex":
            return kind, openalex_client.get_work(openalex_id=val)
        if kind == "arxiv":
            return kind, openalex_client.get_work(doi=f"10.48550/arXiv.{val}")
        return kind, openalex_client.get_work(doi=val)
    except Exception as e:  # 网络/解析异常不致命，但要让用户知道为何空手而归
        print(
            f"[citegraph] OpenAlex 解析失败（{kind}={val}）：{type(e).__name__}: {e}",
            file=sys.stderr,
        )
    return kind, None


def _sort_rows(rows: list[tuple[str, dict]], key: str) -> list[tuple[str, dict]]:
    """按 ``--sort`` 给展示行排序（客户端侧）。

    forward 方向已由 OpenAlex 服务端排好；backward 方向拿到的是一次批量查询的结果，
    顺序不保证，必须在客户端排——否则 ``--sort citations --limit 20`` 会给出「随便 20 条」
    而不是「被引最高的 20 条」。
    """
    if key not in ("citations", "date"):
        return rows

    def _yr(w: dict[str, Any]) -> int:
        try:
            return int(w.get("publication_year") or w.get("year") or 0)
        except (TypeError, ValueError):
            return 0

    if key == "citations":
        return sorted(rows, key=lambda r: -(r[1].get("cited_by_count") or 0))
    return sorted(rows, key=lambda r: _yr(r[1]), reverse=True)


def _citegraph_json(
    raw: str, work: dict[str, Any], back: list, fwd: list, fwd_total: int | None
) -> dict[str, Any]:
    """``citegraph --json`` 的输出形状（两个方向分开，而不是一个大数组）。

    分开的理由：``both`` 时把两个方向混成一个数组会丢掉最有用的那一维信息——一篇
    文献是「它引的」还是「引它的」，对滚雪球的下一步决策完全不同。
    """
    return {
        "id": raw,
        "openalex_id": work.get("openalex_id") or "",
        "title": work.get("title") or "",
        "year": work.get("publication_year"),
        "cited_by_count": work.get("cited_by_count"),
        "backward": {
            "referenced_works_count": work.get("referenced_works_count"),
            "resolved_count": len(back),
            "rows": [_row("openalex", w) for _, w in back],
        },
        "forward": {
            "total_citing": fwd_total,
            "returned_count": len(fwd),
            "rows": [_row("openalex", w) for _, w in fwd],
        },
    }


def cmd_citegraph(args: argparse.Namespace) -> int:
    """引文图谱（滚雪球）：backward = 它引了谁，forward = 谁引了它。

    与 ``arxiv-mcp-server`` 的 ``citation_graph`` 的分工：后者只覆盖 arXiv 论文，
    本命令覆盖 OpenAlex 收录的**全部**文献（期刊论文 / 会议 / 预印本）。

    退出码：解析不出 OpenAlex 记录 → 2（forward 方向硬需 OpenAlex id）；其余 0。
    引文图谱本身的网络失败不报错，只给空列表 + 一行提示（不变量 1）。
    """
    raw = str(args.id or "").strip()
    if not raw:
        print(
            "[citegraph] 需要一个标识（DOI / arXiv id / OpenAlex id）。",
            file=sys.stderr,
        )
        return 2

    kind, work = _resolve_openalex_work(raw)
    if not work:
        print(
            f"[citegraph] 无法从 OpenAlex 解析 {kind}={raw}。引文图谱只有 OpenAlex 有；"
            "请改用 DOI / OpenAlex id，或先跑 `research get <id>` 确认该记录是否被收录。",
            file=sys.stderr,
        )
        return 2

    oid = work.get("openalex_id") or ""
    direction = args.direction
    year_from, year_to = _parse_year(args.year)
    limit = max(0, int(getattr(args, "limit", 25) or 0))
    oa_sort = SORT_MAP_OPENALEX.get(args.sort, "cited_by_count:desc")

    back: list[tuple[str, dict]] = []
    fwd: list[tuple[str, dict]] = []
    fwd_total: int | None = None

    if direction in ("backward", "both"):
        refs = work.get("referenced_works") or []
        total = work.get("referenced_works_count") or len(refs)
        if not refs:
            print(
                f"[citegraph] {oid or raw}：OpenAlex 未记录它的参考文献（backward 为空）。",
                file=sys.stderr,
            )
        else:
            # 先取全量再排序截断：works_by_ids 每批 50 个 id，一篇论文的参考文献通常 1-2 批
            # 就完了；反过来「先截断再取」会让 --sort citations 变成「随便 50 条里挑最高的」。
            got = openalex_client.works_by_ids(refs)
            back = _sort_rows([("openalex", w) for w in got], args.sort)
            miss = total - len(got)
            print(
                f"[citegraph] backward：参考文献 {total} 条，OpenAlex 取回 {len(got)} 条"
                + (f"（缺 {miss} 条，多为已合并/未收录记录）" if miss > 0 else ""),
                file=sys.stderr,
            )
            if limit:
                back = back[:limit]

    if direction in ("forward", "both"):
        if not oid:
            print(
                f"[citegraph] forward 方向需要 OpenAlex id，而 {kind}={raw} 没解析出来。",
                file=sys.stderr,
            )
            if direction == "forward":
                return 2
        else:
            # 服务端直接按排序取 top-N，避免拉回 200 条只为了显示 25 条。
            per_page = limit if limit else 200
            res = openalex_client.works_citing(
                oid,
                per_page=per_page,
                year_from=year_from,
                year_to=year_to,
                sort=oa_sort,
            )
            fwd_total = (res.get("meta") or {}).get("count")
            fwd = [("openalex", w) for w in (res.get("results") or [])]
            more = ""
            if fwd_total is not None and fwd_total > len(fwd):
                more = f"（还有 {fwd_total - len(fwd)} 条未取，调高 --limit）"
            print(
                f"[citegraph] forward：共 {fwd_total if fwd_total is not None else '?'} 篇引用它，"
                f"本次取回 {len(fwd)} 篇{more}",
                file=sys.stderr,
            )

    if direction == "both":
        # 两个方向理论上不重叠（一篇不可能既引它又被它引），但 OpenAlex 的引文数据有
        # 少量双向脏记录，得去掉重叠项——否则同一篇会在两个分区里各出现一次。
        # 去重键与 :func:`_dedupe` 一致（DOI → arXiv id → 标题 slug）；backward 先入，
        # 重叠时保留 backward 那一条。
        def _key(t: tuple[str, dict]) -> str:
            r = _row(t[0], t[1])
            return (
                (r["doi"] or "").lower()
                or (r["arxiv_id"] or "").lower()
                or notes.slugify(r["title"], 60)
            )

        back_keys = {_key(t) for t in back}
        fwd = [t for t in fwd if _key(t) not in back_keys]

    combined = _dedupe(back + fwd)
    if not combined:
        print("[citegraph] 两个方向都无结果。", file=sys.stderr)
        return 0

    if getattr(args, "json", False):
        print(
            json.dumps(
                _citegraph_json(raw, work, back, fwd, fwd_total),
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        if direction == "both":
            print(f"=== {work.get('title') or raw} ===\n")
            print(f"── backward（它引用的，展示 {len(back)} 条）──")
            _print_rows(back)
            print(f"── forward（引用它的，展示 {len(fwd)} 条）──")
            _print_rows(fwd)
        else:
            _print_rows(combined)

    if getattr(args, "save", False):
        # 快照标题用 openalex_id 而不是原始标识：后者可能是长 DOI，而
        # :func:`notes.slugify` 截到 40 字——两篇同期刊同卷的论文会撞出同一个快照名。
        out = _write_shortlist(f"citegraph-{direction}-{oid or raw}", args, combined)
        print(f"[citegraph] 已保存快照：{out}")
    return 0


# ===========================================================================
# 子命令：add
# ===========================================================================
def _extract_zotero_key(resp: Any) -> str:
    """从建条目响应里尽力提取新条目 key。

    兼容两种形态：zotero_cli.create_item_from_metadata 返回的 {"key": ..., "raw": ...}，
    以及旧式 Zotero POST 响应（success/successful/data.key）。
    """
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


def _zotero_uri(key: str) -> str:
    """构造条目 URI：Web 模式用 zotero.org/users/<id>；本地模式用 zotero://select。"""
    if settings.zotero_user_id:
        return f"https://zotero.org/users/{settings.zotero_user_id}/items/{key}"
    return f"zotero://select/library/items/{key}"


def cmd_add(args: argparse.Namespace) -> int:
    """入库 Zotero + 生成 papers/ 笔记骨架（不抓全文）。"""
    src, work = _resolve_work(args.doi)
    if not work:
        print(f"[add] 未能从 {args.doi} 解析元数据，无法入库。", file=sys.stderr)
        return 1
    work = _enrich_work(work)
    fm = _build_frontmatter(src, work)
    tags = [t.strip() for t in args.tags.split(",")] if args.tags else []

    # 引用完整性门（阶段6）：默认三源核验；FAIL 且非 --allow-fail → 阻止入库/写笔记。
    if getattr(args, "verify", True) and not _citation_gate(
        fm, force=getattr(args, "force", False), tag="add"
    ):
        return 1

    if not zotero_cli.available():
        print(
            "[add] 未检测到 zotero-cli（zotero-mcp 未安装），跳过入库，仅生成笔记骨架。\n"
            "      安装：运行 scripts/zotero_mcp/setup_zotero_mcp.ps1。",
            file=sys.stderr,
        )
    else:
        try:
            zb = zotero_cli.ZoteroCli()
            resp = zb.create_item_from_metadata(fm, tags=tags)
            key = _extract_zotero_key(resp)
            if key:
                fm["zotero_key"] = key
                fm["zotero_uri"] = _zotero_uri(key)
                print(f"[add] 已入库 Zotero：{key}")
            else:
                print(f"[add] Zotero 响应未含 key：{resp}", file=sys.stderr)
        except Exception as e:
            print(f"[add] Zotero 入库失败：{type(e).__name__}: {e}", file=sys.stderr)

    fm["added_date"] = _today()
    note_path, action, changed = _merge_note(fm, overwrite=args.overwrite)
    _report_note_action("add", note_path, action, changed)
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
    """查询 Zotero 库：ping / list / search / get（委托 zotero-cli）。

    四个 action 的失败一律降级为「一行 stderr 提示 + 退出码 1」（不变量 1）：除 ``ping``
    自己把错误折进返回值外，:mod:`.zotero_cli` 的读操作都抛 ``ZoteroCliError``，而
    Zotero 没启动 / 本地 API 未授权是常态而非异常，不该变成未处理 traceback。

    ``get`` 还要多分一层：「桥断了」与「这个 key 不在库里」的处置完全不同（前者去开
    Zotero，后者去核对 key），故失败时用 ``ping`` 作连通性裁判再决定怎么说。
    """
    if not zotero_cli.available():
        print(
            "[library] 未检测到 zotero-cli（zotero-mcp 未安装）。运行 "
            "scripts/zotero_mcp/setup_zotero_mcp.ps1 安装后重试。",
            file=sys.stderr,
        )
        return 1
    try:
        zb = zotero_cli.ZoteroCli()
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
        # 诚实说明（方案 H4）：这里的 ``list`` 不是真枚举。zotero-cli 没有「列出全部
        # 条目」的命令，:meth:`zotero_cli.ZoteroCli.list_items` 在无 collection/tag 时
        # 退化成一次**空关键词** search，且 limit 在上游被夹到 ≤ 100。不说的话，
        # 用户会把「只回了 25 条」读成「库里就 25 条」。
        print(
            "[library] list 是尽力而为的枚举（上游无「列全部」命令，退化为空关键词检索，"
            "单次上限 100 条）；结果不全属正常。按关键词找请用 "
            "`research library search --query '<kw>'`。",
            file=sys.stderr,
        )
        try:
            items = zb.list_items(limit=args.limit, item_type=args.type)
        except zotero_cli.ZoteroCliError as e:
            # zotero_cli 的读操作（除 ping 自己把错误折进返回值）都会往上抛。
            # 不接住就是一个未处理 traceback——而 Zotero 没开 / 本地 API 没授权是常态，
            # 属降级路径，该一行提示 + 非零退出码（不变量 1）。
            print(f"[library] list 失败：{e}", file=sys.stderr)
            return 1
        if not items:
            # 实测（本机 Zotero 确实有藏书）：``library list`` 返回 0 条，而
            # ``library search --query acoustic`` 能命中。此时沿用 ``_print_zotero_items``
            # 的「（无条目）」会被读成「你的库是空的」——而那是错的。必须点破。
            # （``search`` 的空结果是真没匹配上，不需这层注解，故只挂在 list 上。）
            # 否定词用「并不」而非 markdown 的 ``**不**``：这串走终端 stderr，``**``
            # 不渲染，只会显示成字面星号并把句子割裂（也会让子串断言失配）。
            print(
                "[library] 返回 0 条。这并不代表库是空的：空关键词检索在上游不保证命中，"
                "请改用 `research library search --query '<kw>'` 复核。",
                file=sys.stderr,
            )
            return 0
        _print_zotero_items(items)
        return 0
    if action == "search":
        if not args.query:
            print("[library] search 需要 --query。", file=sys.stderr)
            return 2
        try:
            items = zb.search_items(args.query, limit=args.limit)
        except zotero_cli.ZoteroCliError as e:
            print(f"[library] search 失败：{e}", file=sys.stderr)
            return 1
        _print_zotero_items(items)
        return 0
    if action == "get":
        if not args.key:
            print("[library] get 需要 --key。", file=sys.stderr)
            return 2
        try:
            item = zb.get_item(args.key)
        except zotero_cli.ZoteroCliError as e:
            # zotero-cli 对「连不上」与「没这条」都回 ok:false，错误文本本身不足以区分
            # 两者。用 ping 作裁判——「桥能不能通」正是它存在的理由（参见
            # ZoteroCli.library_info 的 docstring）。一律报「（未找到）」会把「Zotero
            # 桌面没开」伪装成「你的库里没这篇」，而这两种误判的下一步行动相反。
            # ping 只在失败路径上调，不给正常路径加子进程开销。
            if zb.ping().get("ok"):
                print(
                    f"[library] 取不到 key {args.key}（桥连通，故多半是该 key 不在当前库）：{e}",
                    file=sys.stderr,
                )
            else:
                print(
                    f"[library] get 失败（Zotero 不可达，这不是「未找到」）：{e}",
                    file=sys.stderr,
                )
            return 1
        if not item:
            # 桥回了 ok:true 但 data 为空 / 非 dict：查询本身成功了，答案就是「没有」。
            # 这是唯一能说「未找到」的分支，故退出码 0。
            print("（未找到）")
            return 0
        print(json.dumps(item, ensure_ascii=False, indent=2))
        return 0
    return 2


# ===========================================================================
# 子命令：journal（期刊质量指标）
# ===========================================================================
#: ``journal lookup`` 从 OpenAlex source 里取回并展示的字段（其余字段对「这本刊什么
#: 档次」这个问题无贡献，列出来只会淡掉重点）。
_JOURNAL_OPENALEX_FIELDS = (
    "openalex_id",
    "display_name",
    "issn",
    "issn_l",
    "publisher",
    "country_code",
    "h_index",
    "works_count",
    "2yr_mean_citedness",
    "listed_in",
)


def _journal_lookup_payload(issn: str) -> dict[str, Any]:
    """并排收集一个 ISSN 在各免费指标层的结果；任何一层拿不到都静默留空。

    与 :func:`_attach_journal_metrics` 的区别：后者是笔记生成链路上的一步，要尽量不打扰
    用户；本函数是用户**显式查一本刊**，因此即使什么都没查到也要把「哪一层为什么空」
    说清楚（索引未建 vs 该 ISSN 不在索引里，处置方式完全不同）。
    """
    out: dict[str, Any] = {
        "issn": issn,
        "issn_normalized": journal_metrics.normalize_issn(issn),
        "scimago": journal_metrics.lookup(issn),
        "openalex": None,
        "journal": "",
        "jif": None,
        "journal_h_index": None,
        "listed_in": [],
        "journal_tier": "",
        "journal_tier_basis": [],
    }
    src: dict[str, Any] | None = None
    buf_out, buf_err = io.StringIO(), io.StringIO()
    try:
        # 客户端内部失败时会直接 print；查询命令不该把这些噪声当成正文输出。
        with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
            src = openalex_client.get_source(issn=issn)
    except Exception:  # 网络不可达不该变成 traceback（不变量 1：降级静默）
        src = None

    if src:
        out["openalex"] = {k: src.get(k) for k in _JOURNAL_OPENALEX_FIELDS}
        out["journal"] = src.get("display_name") or ""
        out["jif"] = src.get("2yr_mean_citedness")
        out["journal_h_index"] = src.get("h_index")
        out["listed_in"] = src.get("listed_in") or []
        out["journal_tier"], out["journal_tier_basis"] = notes.derive_journal_tier(
            out["listed_in"]
        )
    return out


def _print_journal_lookup(d: dict[str, Any]) -> None:
    """把 :func:`_journal_lookup_payload` 的结果渲染成四段并列的人类可读输出。"""

    def _v(x: Any) -> str:
        return "—" if x is None or x == "" else str(x)

    def _num(x: Any) -> str:
        """数值按笔记里的精度显示（浮点两位小数），其余走 :func:`_v`。"""
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            return _v(x)
        return f"{x:.2f}" if isinstance(x, float) else str(x)

    print(f"=== 期刊指标：{d['issn']} ===")
    print(f"  刊名              : {_v(d['journal'])}")
    print()
    print("  [1] SCImago SJR（本地索引，按 ISSN 精确匹配）")
    s = d["scimago"]
    if s:
        print(
            f"      分区 {_v(s['quartile']):<4}  SJR {_v(s['sjr']):<8} "
            f"H index {_v(s['h_index']):<6}  版本年 {_v(s['sjr_year'])}"
        )
    else:
        st = journal_metrics.status()
        reason = "索引未建" if not st["exists"] else "该 ISSN 不在索引里"
        print(f"      （无数据——{reason}）")
    print()
    print("  [2] OpenAlex listed_in → journal_tier（专家评议分级，零额外请求）")
    if d["listed_in"]:
        print(f"      档次            : {_v(d['journal_tier'])}")
        print(f"      依据            : {_v(', '.join(d['journal_tier_basis']))}")
        print(f"      全部收录名单    : {', '.join(d['listed_in'])}")
    else:
        print("      （无数据——OpenAlex 未收录该 ISSN，或网络不可达）")
    print()
    print("  [3] OpenAlex 引用类指标")
    # 原始值是十几位小数（实测 PRL = 8.966303149123881），而笔记 frontmatter 的 ``jif`` 是
    # ``round(x, 2)`` = 8.97。人类可读输出必须与笔记一致，否则同一本刊出现两个数、无从
    # 判断哪个对。``--json`` 保留原始精度——那是给程序用的，不该丢信息。
    print(f"      2yr_mean_citedness（≈JIF 估算）: {_num(d['jif'])}")
    print(f"      期刊 h_index                  : {_num(d['journal_h_index'])}")
    print()
    print("  [4] 官方 JCR（jcr_quartile / 官方 JIF / JCI / ESI）")
    print("      未接入——需 Web of Science Journals API（申请中）")


def cmd_journal(args: argparse.Namespace) -> int:
    """期刊质量指标：lookup / build-scimago / status。

    这一层存在的理由：官方 JIF / JCR 分区 / JCI / ESI 只能由 **WoS Journals API** 给出
    （WoS Starter API 不含这些数据，Elsevier/Scopus 需付费凭据），而该 API 的申请尚未
    落地。在此之前有两个**免费**替代，三者语义不同、并存不冲突：

    - ``journal_tier`` —— OpenAlex ``listed_in`` 里 JUFO / Norway / KI-JL 的**专家评议**
      分级，零额外请求；对低引用密度领域比 JIF 更贴近共识；
    - ``scimago_quartile`` —— SCImago SJR 的 Q1-Q4，需用户手工下载官方 CSV 后构建本地
      索引（scimagojr.com 对程序化抓取返回 403，故不实现自动下载）；
    - ``jcr_quartile`` —— 在 WoS Journals API 接入前恒为空。
    """
    action = args.action

    if action == "status":
        st = journal_metrics.status()
        if args.json:
            print(json.dumps(st, ensure_ascii=False, indent=2))
            return 0
        print("【SCImago SJR 本地索引】")
        if not st["exists"]:
            print(f"  状态     : 未建（{st['path']}）")
            print(f"  构建方式 : 从 {journal_metrics.DOWNLOAD_URL} 手工下载 CSV 后运行")
            print(
                "             research journal build-scimago --csv <路径> [--year 2024]"
            )
            print("  影响     : scimago_quartile 一律留空；journal_tier / jif 不受影响")
            return 0
        if not st["readable"]:
            print(f"  状态     : 文件存在但无法解析（{st['path']}）——建议重建")
            return 1
        print(f"  状态     : 已建（{st['path']}）")
        print(f"  SJR 版本 : {st['sjr_year'] or '（未知）'}")
        print(f"  条目数   : {st['n_entries']}")
        print(f"  构建于   : {st['built_at'] or '（未知）'}")
        print(f"  源 CSV   : {st['source_csv'] or '（未知）'}")
        print(f"  归属     : {st['attribution']}")
        return 0

    if action == "build-scimago":
        if not args.csv:
            print(
                "[journal] build-scimago 需要 --csv <官方 CSV 路径>。",
                file=sys.stderr,
            )
            return 2
        try:
            payload = journal_metrics.build_scimago_index(args.csv, year=args.year)
        except (OSError, ValueError) as e:
            # 构建失败**必须**报错（与查询路径的静默降级相反）：静默产出空索引会让
            # 所有笔记的 scimago_quartile 悄悄留空，比构建失败难查得多。
            print(f"[journal] 构建失败：{type(e).__name__}: {e}", file=sys.stderr)
            return 1
        meta = payload["_meta"]
        print(
            f"[journal] SCImago 索引已建：{meta['n_entries']} 条 ISSN → "
            f"{journal_metrics.index_path()}"
        )
        print(
            f"[journal] SJR 版本年：{meta['sjr_year'] or '（未推断出，可用 --year 指定）'}"
            f"；源文件：{meta['source_csv']}"
        )
        print(f"[journal] 归属：{meta['attribution']}")
        print(
            "[journal] 提示：把下载 URL / 版本年 / 下载日期补进同目录的 SOURCE.md，"
            "以便年度刷新时核对。"
        )
        return 0

    if action == "lookup":
        issn = (args.issn or "").strip()
        if not issn:
            print("[journal] lookup 需要一个 ISSN（如 0031-9007）。", file=sys.stderr)
            return 2
        payload = _journal_lookup_payload(issn)
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            _print_journal_lookup(payload)
        return 0

    return 2


# ===========================================================================
# 子命令：review（综述脚手架 + 链接完整性 + 计数同步）
# ===========================================================================
#: ``papers_reviewed`` 里的 wiki 链接。截到 ``|``（别名）/ ``#``（锚点）/ ``]`` 为止，
#: 使 ``[[2026_zhang_x|Zhang 2026]]`` 与 ``[[2026_zhang_x#§2]]`` 都能取出同一个目标。
_WIKI_LINK_RE = re.compile(r"\[\[([^\[\]|#]+)")

#: ``templates/review_note.md`` 正文里的「综述目标」提示行。``--purpose`` 只替掉
#: 那个括号提示，不动其余任何正文。
_REVIEW_PURPOSE_RE = re.compile(r"(>[ \t]*\*\*综述目标\*\*：)（[^）]*）")

#: ``review new`` 产出的文件名形状：``{YYYY-MM}_{topic_slug}_survey.md``。
_REVIEW_NAME_FMT = "{month}_{slug}_survey.md"


def _wiki_targets(value: Any) -> list[str]:
    """从 ``papers_reviewed`` 抽出 wiki 链接目标（去 ``[[``/``]]``、别名与锚点）。

    容忍裸 stem（``2026_zhang_x``，不带括号）——手写综述里很常见，而拒绝它会让
    ``status`` 把一条完全正常的链接报成断链。

    以 ``[[`` 开头但抽不出内容的（``[[]]``）**整个跳过**：它没有任何目标，回落成裸
    字符串会让 ``status`` 报出一条名为 ``[[[[]]]]`` 的断链——把一个空壳渲染成看起来
    像真问题的东西，比忽略它更糟。
    """
    items = value if isinstance(value, (list, tuple)) else [value]
    out: list[str] = []
    for item in items:
        s = str(item or "").strip()
        if not s:
            continue
        if s.startswith("[["):
            m = _WIKI_LINK_RE.search(s)
            if m is None:
                continue
            target = m.group(1).strip()
        else:
            target = s
        if target.endswith(".md"):
            target = target[:-3].strip()
        target = target.strip("/")
        if target:
            out.append(target)
    return out


def _resolve_paper_link(target: str) -> Path | None:
    """把一个 wiki 链接目标解析为 ``papers/`` 下实存的笔记；解不出返回 ``None``。

    只按文件名匹配（不按 frontmatter 的 title 模糊匹配）：综述里的链接写的就是笔记
    文件名，而模糊匹配会把「断链」这个真问题掊成「大概能对上」。
    """
    t = str(target or "").strip()
    if not t:
        return None
    direct = PAPERS_DIR / f"{t}.md"
    if direct.is_file():
        return direct
    base = Path(t).name  # 容忍 [[papers/2026_zhang_x]] 这种带路径前缀的写法
    if base and base != t:
        alt = PAPERS_DIR / f"{base}.md"
        if alt.is_file():
            return alt
    return None


def _resolve_in_dir(directory: Path, token: str) -> Path | None:
    """把一个命令行参数解析为 ``directory`` 下唯一确定的 ``.md`` 文件。

    依次试：原样路径（绝对或相对 cwd）→ ``<dir>/<token>`` → ``<dir>/<token>.md`` →
    ``*<token>*.md`` 的**唯一**匹配。多个候选时返回 ``None`` 而不猜——猜错了会让
    ``sync`` 改错文件，那比报错严重得多。
    """
    t = str(token or "").strip()
    if not t:
        return None
    p = Path(t)
    if p.is_file():
        return p
    for cand in (directory / t, directory / f"{t}.md"):
        if cand.is_file():
            return cand
    if not directory.exists():
        return None
    hits = sorted(directory.glob(f"*{t}*.md"))
    return hits[0] if len(hits) == 1 else None


def _note_status(path: Path) -> str:
    """取一篇论文笔记的 ``status``（小写）；读不出 / 解析失败时返回 ``""`` 并留一行提示。"""
    try:
        fm = notes.load_frontmatter(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(
            f"[review] 解析 {path.name} 失败（计作未读）：{type(e).__name__}: {e}",
            file=sys.stderr,
        )
        return ""
    return str(fm.get("status") or "").strip().lower()


def _recount(fm: dict[str, Any]) -> tuple[list[str], int, int, list[str]]:
    """由 ``papers_reviewed`` 重算规范链接列表、两个计数与断链清单。

    ``papers_read_count`` 的判据是 ``status != "unread"``：即 reading / read / archived /
    rejected 都算「已处理」。这个口径偏宽，但它是**可复算**的——任何更细的判据
    （比如只算 ``read``）都得先约定 ``archived`` 到底算不算读过，而那个约定不在数据里。

    断链**计入** ``papers_reviewed`` 而**不计入** ``papers_total_count``：前者是人写的
    意图，后者是「实际链接到的笔记数」。``sync`` 不会静默删掉断链（那会把需要修的
    问题藏起来），只把计数算到能解析的那部分上。

    Returns:
        ``(links, total, read, broken)``。``links`` 已去重并统一为 ``[[target]]`` 形式。
    """
    seen: set[str] = set()
    links: list[str] = []
    broken: list[str] = []
    n_read = 0
    for raw in _wiki_targets(fm.get("papers_reviewed")):
        key = raw.lower()
        if key in seen:
            continue
        seen.add(key)
        links.append(f"[[{raw}]]")
        p = _resolve_paper_link(raw)
        if p is None:
            broken.append(raw)
            continue
        st = _note_status(p)
        if st and st != "unread":
            n_read += 1
    return links, len(links) - len(broken), n_read, broken


def _read_review(path: Path) -> tuple[dict[str, Any], str, str, str | None]:
    """读一份综述，返回 ``(frontmatter, body, 原换行风格, 错误信息)``。

    解析不出 frontmatter 时 ``fm`` 为 ``{}`` 且带错误信息——调用方据此**跳过而不改写**
    该文件（人工手写的综述可能根本没有 frontmatter，自动改写会破坏它）。
    """
    try:
        text, newline = _read_note_text(path)
    except OSError as e:
        return {}, "", "\n", f"读取失败（{type(e).__name__}: {e}）"
    fm, body = notes.split_note(text)
    if not fm:
        return {}, body, newline, "无可解析的 frontmatter（人工综述？）——不改写"
    return fm, body, newline, None


def _review_audit(path: Path) -> dict[str, Any]:
    """审计一份综述：链接是否断、两个计数是否对得上实况。

    只读，不改任何文件。返回的 dict 同时供 :func:`_print_review_status` 渲染与
    :func:`cmd_review` 定退出码。
    """
    fm, _body, _nl, err = _read_review(path)
    links, total, n_read, broken = _recount(fm)
    return {
        "path": path,
        "name": path.name,
        "error": err,
        "topic": fm.get("topic") or "",
        "status": fm.get("status") or "",
        "last_updated": fm.get("last_updated") or "",
        "links": links,
        "broken": broken,
        "total_actual": total,
        "read_actual": n_read,
        "total_declared": fm.get("papers_total_count"),
        "read_declared": fm.get("papers_read_count"),
    }


def _count_drift(aud: dict[str, Any]) -> list[str]:
    """两个计数与实况的偏差，渲染成人话；无偏差返回 ``[]``。"""

    def _v(x: Any) -> str:
        return "—" if x is None or x == "" else str(x)

    drift: list[str] = []
    if aud["total_declared"] != aud["total_actual"]:
        drift.append(
            f"papers_total_count {_v(aud['total_declared'])} → 应为 {aud['total_actual']}"
        )
    if aud["read_declared"] != aud["read_actual"]:
        drift.append(
            f"papers_read_count {_v(aud['read_declared'])} → 应为 {aud['read_actual']}"
        )
    return drift


def _print_review_status(aud: dict[str, Any]) -> None:
    """渲染一份综述的审计结果（断链逐条列出，最多 20 条）。"""

    def _v(x: Any) -> str:
        return "—" if x is None or x == "" else str(x)

    print(f"【综述】{aud['name']}")
    if aud["error"]:
        print(f"  无法审计 : {aud['error']}")
        return
    print(f"  主题     : {_v(aud['topic'])}")
    print(f"  状态     : {_v(aud['status'])}")
    print(f"  最后更新 : {_v(aud['last_updated'])}")
    print(
        f"  论文链接 : {len(aud['links'])} 条"
        f"（解析成功 {aud['total_actual']}，断链 {len(aud['broken'])}）"
    )
    print(
        f"  已处理   : {aud['read_actual']} / {aud['total_actual']}"
        "（判据：status != unread）"
    )
    drift = _count_drift(aud)
    if drift:
        print("  计数偏差 : " + "；".join(drift))
        print("             运行 `research review sync` 可修正（只改 frontmatter）")
    else:
        print("  计数偏差 : 无")
    if aud["broken"]:
        print("  断链     :")
        for b in aud["broken"][:20]:
            print(f"    [[{b}]] —— papers/ 下无同名笔记")
        if len(aud["broken"]) > 20:
            print(f"    ...（其余 {len(aud['broken']) - 20} 条略）")


def _review_sync(path: Path, *, dry_run: bool = False) -> tuple[str, list[str]]:
    """重算 ``papers_reviewed`` 规范形态与两个计数，**只重写 frontmatter**。

    body 逐字节保留（不变量 2），而且**不追加 Changelog**：Changelog 记的是综述的
    认知进展（起草 / 审校 / 补证），而计数刷新是机械的，可能频繁发生，写进去会把
    真正值得看的条目浹没。

    Returns:
        ``(action, changed_keys)``，``action`` ∈ ``{"synced", "unchanged",
        "would-sync", "skipped"}``。``unchanged`` 时完全不触碰文件（保留 mtime），
        与 :func:`_merge_note` 的语义一致。
    """
    fm, body, newline, err = _read_review(path)
    if err:
        return "skipped", [err]
    links, total, n_read, _broken = _recount(fm)
    new_fm = dict(fm)
    changed: list[str] = []
    for key, want in (
        ("papers_reviewed", links),
        ("papers_total_count", total),
        ("papers_read_count", n_read),
        ("last_updated", _today()),
    ):
        if new_fm.get(key) != want:
            new_fm[key] = want
            changed.append(key)
    if not changed:
        return "unchanged", []
    if dry_run:
        return "would-sync", changed
    try:
        _write_note_text(path, notes.render_note(new_fm, body), newline)
    except OSError as e:
        return "skipped", changed + [f"写入失败（{type(e).__name__}: {e}）"]
    return "synced", changed


def _aggregate_shortlists(tokens: Any) -> tuple[list[str], list[str], list[str]]:
    """从若干检索快照聚合 ``sources_used`` 与 ``query_strings``。

    直接读快照的 frontmatter（复用 :func:`notes.load_frontmatter`）而不是让用户把检索式
    再手拄一遍：手拄必然漂移，而快照里本就存着**实际提交**的那一条。

    解析不出的快照只留一行提示（不变量 1）：综述骨架照样生成，``query_strings``
    少一条总比整个命令失败好。

    Returns:
        ``(sources_used, query_strings, notices)``，两个列表均按首次出现序去重。
    """
    sources: list[str] = []
    queries: list[str] = []
    notices: list[str] = []
    for tok in tokens or []:
        p = _resolve_in_dir(SHORTLISTS_DIR, tok)
        if p is None:
            notices.append(
                f"[review] 找不到检索快照 {tok}（已跳过）——需完整路径，"
                "或 shortlists/ 下能唯一匹配的文件名片段"
            )
            continue
        try:
            fm = notes.load_frontmatter(p.read_text(encoding="utf-8"))
        except Exception as e:
            notices.append(
                f"[review] 解析 {p.name} 失败（已跳过）：{type(e).__name__}: {e}"
            )
            continue
        for s in fm.get("sources") or []:
            s = str(s).strip().lower()
            if s and s not in sources:
                sources.append(s)
        q = str(fm.get("query_string") or "").strip()
        if q:
            if q not in queries:
                queries.append(q)
        else:
            notices.append(f"[review] {p.name} 的 query_string 为空——未贡献检索式")
    return sources, queries, notices


def _review_targets(args: argparse.Namespace, label: str) -> tuple[list[Path], int]:
    """``status`` / ``sync`` 共用的目标文件解析。

    Returns:
        ``(files, rc)``。``rc`` 非 0 表示参数本身就错（指定的文件找不到），调用方应
        直接返回；``rc`` 为 0 而 ``files`` 为空则是「目录下本来就没综述」，属于正常
        降级（一行提示 + 退出码 0）。
    """
    token = str(getattr(args, "target", "") or "").strip()
    if token:
        p = _resolve_in_dir(REVIEWS_DIR, token)
        if p is None:
            print(f"[review] 找不到综述：{token}", file=sys.stderr)
            return [], 2
        return [p], 0
    if not REVIEWS_DIR.exists():
        print(f"[review] reviews/ 目录不存在——还没有综述。{label}")
        return [], 0
    files = sorted(REVIEWS_DIR.glob("*.md"))
    if not files:
        print(f"[review] reviews/ 下没有综述笔记。{label}")
    return files, 0


def cmd_review(args: argparse.Namespace) -> int:
    """综述笔记：new / status / sync。

    **故意不做综述生成器**：``review_note.md`` 的 §1-§5（领域概览、关键脉络、时间线
    叙事、争议判断、与我研究的接口）是 LLM 的核心判断工作，做成模板填充器只会把活的
    判断变成僵的套话。本命令只提供三件**机械**活：脚手架、链接完整性校验、计数同步。

    退出码：``status`` 发现断链或无法审计 → 1；参数错 / 无法执行 → 2；其余 0。
    """
    action = getattr(args, "action", "")

    if action == "new":
        topic = str(getattr(args, "target", "") or "").strip()
        if not topic:
            print("[review] new 需要一个主题。", file=sys.stderr)
            return 2
        slug = notes.slugify(topic, 48)
        tpl = _load_template("review_note.md")
        _skel, body = _split_template(tpl)
        if not body.strip():
            print(
                "[review] templates/review_note.md 缺失或无正文，无法生成综述骨架。",
                file=sys.stderr,
            )
            return 2
        REVIEWS_DIR.mkdir(parents=True, exist_ok=True)
        out = REVIEWS_DIR / _REVIEW_NAME_FMT.format(month=_today()[:7], slug=slug)
        if out.exists():
            print(f"[review] 已存在，不覆盖：{out}", file=sys.stderr)
            print("[review] 如需重建请先手工删除，或换个主题措辞。", file=sys.stderr)
            return 2

        sources, queries, notices = _aggregate_shortlists(
            getattr(args, "from_shortlist", None)
        )
        for n in notices:
            print(n, file=sys.stderr)

        purpose = str(getattr(args, "purpose", "") or "").strip()
        if purpose:
            body, n_sub = _REVIEW_PURPOSE_RE.subn(
                lambda m: m.group(1) + purpose, body, count=1
            )
            if not n_sub:
                # 模板改了就会走到这里。宁可报一行也不静默丢掉用户给的目标。
                print(
                    f"[review] 模板里找不到「综述目标」提示行，--purpose 未写入：{purpose}",
                    file=sys.stderr,
                )
        body = _fill_placeholders(body, {"topic": topic})

        today = _today()
        fm = {
            "topic": topic,
            "topic_slug": slug,
            "created_date": today,
            "last_updated": today,
            "author": "AI Agent + User",
            "status": "draft",
            "time_window": str(getattr(args, "time_window", "") or ""),
            "sources_used": sources,
            "query_strings": queries,
            # 下面两项故意留空：纳入/排除标准是综述的**方法学判断**，只能由人/AI 填。
            "inclusion_criteria": "",
            "exclusion_criteria": "",
            "papers_reviewed": [],
            "papers_total_count": 0,
            "papers_read_count": 0,
        }
        out.write_text(notes.render_note(fm, body), encoding="utf-8")
        print(f"[review] 已创建综述骨架：{out}")
        print(
            f"[review] 检索快照聚合：sources_used={sources or '[]'}，"
            f"query_strings {len(queries)} 条"
        )
        print(
            "[review] 下一步：由 AI 填写 §1-§5 与纳入/排除标准，"
            "把涉及的笔记以 [[文件名]] 写进 papers_reviewed，然后跑 review sync。"
        )
        return 0

    if action == "status":
        files, rc = _review_targets(args, "用 `research review new <topic>` 创建。")
        if rc:
            return rc
        exit_code = 0
        for p in files:
            aud = _review_audit(p)
            _print_review_status(aud)
            # 断链 → 1（方案 F1 的约定，与 index --check / citecheck 的 CI 友好一致）。
            # 无法审计同样计 1：一份读不出 frontmatter 的综述根本没能被检查，让它默默
            # 通过一道门比报错更危险。
            if aud["broken"] or aud["error"]:
                exit_code = 1
        return exit_code

    if action == "sync":
        files, rc = _review_targets(args, "无可同步对象。")
        if rc:
            return rc
        dry_run = bool(getattr(args, "dry_run", False))
        exit_code = 0
        for p in files:
            act, changed = _review_sync(p, dry_run=dry_run)
            if act == "unchanged":
                print(f"[review] {p.name}：计数与链接已是最新，未改动。")
            elif act == "would-sync":
                print(f"[review] {p.name}：将重写 frontmatter（{', '.join(changed)}）")
            elif act == "synced":
                print(f"[review] {p.name}：已重写 frontmatter（{', '.join(changed)}）")
            else:
                print(f"[review] {p.name}：跳过——{'; '.join(changed)}", file=sys.stderr)
                exit_code = 1
                continue
            for b in _review_audit(p)["broken"]:
                print(
                    f"[review] {p.name}：断链 [[{b}]]（已保留，请修正链接）",
                    file=sys.stderr,
                )
        if dry_run:
            print("[review] --dry-run：未写入任何文件。去掉 --dry-run 即落盘。")
        return exit_code

    print(f"[review] 未知 action：{action!r}", file=sys.stderr)
    return 2


# ===========================================================================
# 子命令：index
# ===========================================================================
#: 匹配 INDEX.md 那行「自动重建于 YYYY-MM-DD」，供 :func:`_index_comparable` 归一化。
_INDEX_DATE_RE = re.compile(r"(自动重建于 )\d{4}-\d{2}-\d{2}")


def _index_comparable(text: str) -> str:
    """抹掉 INDEX.md 里的构建日期后返回可比对文本。

    那行日期是**渲染时刻**烘进文件的。若让它参与比对，``index --check`` 在任何不同于
    上次构建日的日期上都会报「不一致」——与它作为 CI / 写入前门的用途直接冲突
    （``--check`` 应当只在 ``papers/`` 的实际内容变化时才失败）。
    """
    return _INDEX_DATE_RE.sub(r"\1<DATE>", text or "").strip()


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
        "| # | Year | First Author | Title | Journal | Tier | Status | Rating | Note |",
        "|---|------|--------------|-------|---------|------|--------|--------|------|",
    ]
    ordered = sorted(entries, key=lambda kv: yr(kv[1]), reverse=True)
    for i, (stem, fm) in enumerate(ordered, 1):
        title = str(fm.get("title") or stem).replace("|", "\\|")
        fa = fm.get("first_author_last_name") or ""
        journal = str(fm.get("journal") or "").replace("|", "\\|")
        # Tier 独立成列而不拼进 Journal 列：后者是书目事实，前者是**派生评价**，混在一
        # 起会让用户无法单独按刊名排序/过滤（也是 D1 那类污染的成因）。
        tier = str(fm.get("journal_tier") or "") or "—"
        status = fm.get("status") or ""
        rating = fm.get("my_rating")
        rating_s = f"{rating}\u2605" if rating not in (None, "") else "—"
        lines.append(
            f"| {i} | {fm.get('year') or '—'} | {fa} | {title} | {journal} | {tier} | "
            f"{status} | {rating_s} | [{stem}](papers/{stem}.md) |"
        )
    lines.append("")
    return "\n".join(lines)


def _read_note_frontmatter(path: Path) -> dict[str, Any]:
    """只读地取一篇笔记的 frontmatter；读不出 / 解析失败时返回 ``{}`` 并打一行提示。"""
    try:
        return notes.load_frontmatter(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(
            f"[index] 解析 {path.name} 失败：{type(e).__name__}: {e}", file=sys.stderr
        )
        return {}


def _describe_fix(added: list[str]) -> str:
    """把 ``--fix`` 对单篇笔记的变更渲染成一句人话。"""
    if added:
        return f"补齐 {len(added)} 个缺失字段（{', '.join(added)}）并规范化 frontmatter"
    return "规范化 frontmatter（统一字段序与 block style）"


def _normalize_note_file(
    path: Path, *, dry_run: bool
) -> tuple[dict[str, Any], list[str], str, list[str]]:
    """``index --fix`` 对单篇笔记的处理：规范化 frontmatter，**正文逐字节保留**。

    做三件事：

    1. 按 :func:`notes.normalize_frontmatter` 补齐 ``templates/paper_note.md`` 定义了
       而本篇缺失的键（所有既有值一律不动）；
    2. 统一字段序与 block style（flow-style 的 ``authors: [A, B]`` 展开为缩进列表）；
    3. 报告文件名与 :func:`notes.note_filename` 不一致者——**只报告不重命名**，
       因为改名会破坏 ``reviews/*.md`` 里的 ``[[wiki-link]]`` 与用户的外部引用。

    Returns:
        ``(fm, added_keys, status, warnings)``。``status`` ∈
        ``{"unchanged", "would-rewrite", "rewritten", "skipped"}``；``fm`` 是规范化后的
        frontmatter（``skipped`` 时为 ``{}``），直接喂给 :func:`_render_index`，使本次
        重建的 INDEX.md 就反映规范化结果而无需跑第二遍。
    """
    try:
        text, newline = _read_note_text(path)
    except OSError as e:
        return {}, [], "skipped", [f"读取失败（{type(e).__name__}: {e}），未改动"]
    old_fm, body = notes.split_note(text)
    if not old_fm:
        return (
            {},
            [],
            "skipped",
            ["无可解析的 frontmatter（人工笔记或文件损坏？）——不自动改写，请手工处理"],
        )
    new_fm, added = notes.normalize_frontmatter(old_fm)
    rendered = notes.render_note(new_fm, body)
    warns: list[str] = []
    want = notes.note_filename(new_fm)
    if want != path.name:
        warns.append(
            f"文件名与命名规范不符：现为 {path.name}，按 frontmatter 应为 {want}"
            "——不自动重命名（会破坏 reviews/ 的 [[wiki-link]] 与外部引用），"
            "如需改名请手工处理"
        )
    if rendered == text:
        return new_fm, added, "unchanged", warns
    if dry_run:
        return new_fm, added, "would-rewrite", warns
    try:
        _write_note_text(path, rendered, newline)
    except OSError as e:
        return (
            new_fm,
            added,
            "skipped",
            warns + [f"写入失败（{type(e).__name__}: {e}）"],
        )
    return new_fm, added, "rewritten", warns


def cmd_index(args: argparse.Namespace) -> int:
    """扫描 papers/*.md 重建 INDEX.md。

    ``--check``    只校验 INDEX.md 与 papers/ 是否一致（构建日期不参与比对）；不一致退出码 1
    ``--fix``      先规范化每篇笔记的 frontmatter（补键 + 统一字段序 + flow→block），
                   **正文逐字节保留**，再重建 INDEX.md
    ``--dry-run``  与 ``--fix`` 同用：只报告将发生的变更，任何文件都不写（含 INDEX.md）
    ``--force``    papers/ 为空时也重建 INDEX.md（默认保留现有文件）
    """
    PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    note_files = sorted(PAPERS_DIR.glob("*.md"))
    fix = bool(getattr(args, "fix", False))
    dry_run = bool(getattr(args, "dry_run", False))
    if dry_run and not fix:
        print("[index] --dry-run 需与 --fix 同用，已忽略。", file=sys.stderr)
        dry_run = False

    entries: list[tuple[str, dict[str, Any]]] = []
    n_changed = 0
    for p in note_files:
        if not fix:
            entries.append((p.stem, _read_note_frontmatter(p)))
            continue
        fm, added, status, warns = _normalize_note_file(p, dry_run=dry_run)
        if status == "unchanged":
            print(f"[index] {p.name}：已是规范形态，未改动。")
        elif status in {"would-rewrite", "rewritten"}:
            n_changed += 1
            verb = "将" if status == "would-rewrite" else "已"
            print(f"[index] {p.name}：{verb}{_describe_fix(added)}")
        else:
            print(f"[index] {p.name}：跳过，未改动。", file=sys.stderr)
        for w in warns:
            print(f"[index] {p.name}：{w}", file=sys.stderr)
        entries.append((p.stem, fm))

    if fix:
        verb = "将规范化" if dry_run else "已规范化"
        print(f"[index] --fix：{len(note_files)} 篇中 {verb} {n_changed} 篇。")
        if dry_run:
            print(
                "[index] --dry-run：未写入任何文件（含 INDEX.md）。去掉 --dry-run 即落盘。"
            )
            return 0

    if not entries and INDEX_PATH.exists() and not args.force:
        print("[index] papers/ 下无笔记；保留现有 INDEX.md（如需清空重建加 --force）。")
        return 0

    content = _render_index(entries)
    if args.check:
        existing = INDEX_PATH.read_text(encoding="utf-8") if INDEX_PATH.exists() else ""
        if _index_comparable(existing) == _index_comparable(content):
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
            f"[cache] {verb} Tier B（api_responses，{scope}）：{n} 个文件，释放 {notes.fmt_size(freed)}。"
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
        keep = "（保留笔记与 ingest 清单引用者）" if args.keep_referenced else ""
        print(
            f"[cache] {verb} Tier A {n} 个单元{keep}，释放 {notes.fmt_size(freed)}，剩余 {notes.fmt_size(remain)}。"
        )
    return 0


# ===========================================================================
# 子命令：ingest（本地 PDF 文献仓库批量入库；实现在 local_ingest）
# ===========================================================================
def cmd_ingest(args: argparse.Namespace) -> int:
    """把本地已有 PDF 仓库入库：复制归档 → MinerU 抽取 → 回写 manifest 状态。

    ``status`` 有两种写法（``ingest status`` 与 ``ingest --status``），因为本子命令后来
    才补上 action 位置参数以对齐形态 B，而旧的布尔写法已经进了文档与肌肉记忆，
    故不做破坏性变更。两者任一为 status 即只汇总。
    """
    manifest = args.manifest or str(local_ingest.DEFAULT_MANIFEST)
    if getattr(args, "action", "run") == "status" or args.status:
        data = local_ingest.load_manifest(manifest)
        print(local_ingest.summarize(data))
        return 0
    return local_ingest.run_ingest(
        manifest,
        priority=args.priority,
        theme=args.theme,
        backend=args.backend,
        limit_pages=args.limit_pages,
        limit_files=args.limit_files,
        dry_run=args.dry_run,
        force=args.force,
    )


# ===========================================================================
# 子命令：rag（PaperQA2 语义检索本地文献库）
# ===========================================================================
def _print_rag_status(st: dict[str, Any]) -> None:
    ready = (
        "是（硅基流动 embedding）" if st.get("ready") else "否——缺 SILICONFLOW_API_KEY"
    )
    print(f"  后端就绪   : {ready}")
    print(f"  embedding  : {st.get('embedding_model')}")
    if st.get("exists"):
        print(
            f"  本地索引   : docs={st.get('n_docs')} chunks={st.get('n_chunks')}"
            f"（built {st.get('built_at') or '?'}）"
        )
    else:
        print("  本地索引   : 未建（运行 `research rag index`）")
    print(f"  索引位置   : {st.get('index_path')}")


def cmd_rag(args: argparse.Namespace) -> int:
    """PaperQA2 RAG：本地文献库语义检索基础设施（index / search / ask / status）。

    search 为 agent 主力（纯 embedding、零 LLM、零成本）；ask 为可选 LLM 综述，
    免费档不可用时回退付费档，全失败则静默降级为 search 结果，永不阻塞。
    """
    action = args.action
    as_json = getattr(args, "json", False)

    if action == "status":
        st = rag.index_status()
        if as_json:
            print(json.dumps(st, ensure_ascii=False, indent=2))
        else:
            print("【PaperQA2 RAG 检索状态】")
            _print_rag_status(st)
        return 0

    if not settings.pqa_ready:
        print(
            "[rag] 未就绪：缺 SILICONFLOW_API_KEY（.env）。索引/检索依赖硅基流动的 "
            "OpenAI 兼容 embedding。",
            file=sys.stderr,
        )
        return 1

    if action == "index":
        try:
            rep = rag.build_index(
                getattr(args, "paths", None) or None,
                rebuild=getattr(args, "rebuild", False),
                verbose=not as_json,
            )
        except Exception as e:
            print(f"[rag] 索引失败：{type(e).__name__}: {e}", file=sys.stderr)
            return 1
        if as_json:
            print(json.dumps(rep.to_dict(), ensure_ascii=False, indent=2))
        for err in rep.errors:
            print(f"[rag] ! {err}", file=sys.stderr)
        return 1 if (rep.errors and rep.n_docs == 0) else 0

    if not args.query:
        print(f"[rag] {action} 需要查询词。", file=sys.stderr)
        return 2

    if action == "search":
        try:
            res = rag.search(args.query, k=getattr(args, "top_k", 8))
        except Exception as e:
            print(f"[rag] 检索失败：{type(e).__name__}: {e}", file=sys.stderr)
            return 1
        if as_json:
            print(json.dumps(res.to_dict(), ensure_ascii=False, indent=2))
        else:
            print(rag.render_search(res, chars=getattr(args, "chars", 400)))
        return 0

    if action == "ask":
        try:
            ans = rag.ask(args.query, k=getattr(args, "top_k", rag.ASK_EVIDENCE_K))
        except Exception as e:
            print(f"[rag] ask 失败：{type(e).__name__}: {e}", file=sys.stderr)
            return 1
        if as_json:
            print(json.dumps(ans.to_dict(), ensure_ascii=False, indent=2))
        else:
            print(rag.render_ask(ans, chars=getattr(args, "chars", 400)))
        return 0

    return 2


# ===========================================================================
# 参数解析与入口
# ===========================================================================
#: ``pysci-research -h`` 的 epilog。
#:
#: 写在 CLI 里而不只写在 SKILL.md 里的理由：``-h`` 是调用方（尤其是 LLM）在不知道
#: 文档存在时唯一能自助取得的接口，而 14 个子命令曾有**五种**参数形态并存（动词+目标、
#: 名词+action、纯 flags、布尔充当 action、action 可省），不给出判据就只能逐个背。
#: 收敛到两种后，形态本身就可推了。
_CLI_SHAPE_EPILOG = """\
子命令只有两种参数形态（14 个全部落在这两类里）：

  A  动词 + 位置目标（目标是被处理的对象：检索式 / DOI / arXiv id / OpenAlex id）
       search <query>    read <id>    get <id>    add <doi>
       citegraph <id>    citecheck [target ...]    index    doctor

  B  名词 + action 位置参数（action 是本命令自己的一个子动作，-h 里列出全部取值）
       library <ping|list|search|get>      rag <index|search|ask|status>
       cache <stats|clean|prune>           journal <lookup|build-scimago|status>
       review <new|status|sync>            ingest [run|status]   ← action 可省，默认 run

判据：要在命令行上写出「处理谁」→ A；要写出「做哪件事」→ B。

跨子命令的通用约定：
  --json      机器可读输出（search / get / citecheck / citegraph / rag / journal）。
              给了它，stdout 就**只有** JSON，进度提示与告警一律转 stderr，
              因此可以直接管道给解析器，不必先剥离噪声行。
  退出码      0 = 成功；1 = 检出问题（citecheck 有 FAIL、index --check 不一致、
              review status 有断链）或所需能力未就绪（如 rag 缺 SILICONFLOW_API_KEY）；
              2 = 用法错误或前置条件不满足（缺查询词、标识解析不出来、数据源未配置）。
  降级        缺凭据 / 缺数据文件 / 网络不可达时静默跳过并留一行提示，不报错、
              不阻塞——唯一例外是 search 在 OpenAlex 降级态下返回空集时会告警，
              因为那种空结果分不清「配额耗尽」与「真没文献」。
"""

#: ``--force`` 一词曾同时承载四种语义，故按语义拆出主名（旧名保留为隐藏弃用别名）：
#:
#: - ``read`` / ``ingest``：绕过缓存重抓重抽 → ``--refresh``
#: - ``add``：**越过引用完整性安全门** → ``--allow-fail``
#: - ``index``：papers/ 为空时也强制重建 → **保留 ``--force``**（此处语义确实是「强制写」，
#:   最贴近原义，故不在本表内）
_FORCE_RENAMED: dict[str, str] = {
    "read": "--refresh",
    "ingest": "--refresh",
    "add": "--allow-fail",
}


def _add_force_alias(sp: argparse.ArgumentParser) -> None:
    """给子命令挂一个隐藏的 ``--force`` 弃用别名。

    刻意**不**与新主名共用 dest（那会吃掉发迁移提示的机会），而是落到独立的
    ``force_deprecated``，由 :func:`_migrate_force_alias` 归一。配 ``argparse.SUPPRESS``
    使其不出现在 ``-h`` 里——不诱导新用法，但照常工作，不做破坏性变更。
    """
    sp.add_argument(
        "--force", action="store_true", dest="force_deprecated", help=argparse.SUPPRESS
    )


def _migrate_force_alias(args: argparse.Namespace) -> None:
    """把弃用的 ``--force`` 归一到新主名的 dest，并向 stderr 打一行迁移提示。

    拆名的理由：``add --force`` 尤其危险——它把一个**安全门**的开关与普通的缓存刷新
    混为同一个词，调用方（尤其是 LLM）极易误用；拆名后 ``--allow-fail`` 的意图无法被
    误解。别名长期保留、不设移除期限（本项目无 CI/CD、单人使用，破坏性变更的收益
    低于其风险）。

    只在 :func:`main` 里调用；直接构造 ``Namespace`` 调 ``cmd_*`` 的路径（含既有测试）
    不经过此处，故一律用 ``getattr`` 探测，缺 ``force_deprecated`` 键也不炸。
    """
    if not getattr(args, "force_deprecated", False):
        return
    cmd = getattr(args, "cmd", "") or ""
    new_name = _FORCE_RENAMED.get(cmd)
    if not new_name:  # index 的 --force 仍是主名，走不到这里；万一走到就什么都不做
        return
    args.force = True
    why = (
        "它开的是引用核验安全门，不是缓存刷新。"
        if new_name == "--allow-fail"
        else "它绕过的是缓存，不是安全门。"
    )
    print(
        f"[research] `{cmd} --force` 已更名为 `{cmd} {new_name}`"
        f"（行为不变，旧名长期保留）。建议改用新名：{why}",
        file=sys.stderr,
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pysci-research",
        description="pySciWS 文献调研统一 CLI（检索 / 阅读 / 元数据 / 入库 / 库查询 / 索引）",
        epilog=_CLI_SHAPE_EPILOG,
        # epilog 里的对齐是靠空格手排的，默认 formatter 会把它重新折行成一团。
        # 只影响顶层 -h 的 description/epilog：子解析器不继承 formatter_class，
        # 它们的选项 help 仍正常按终端宽度折行。
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("doctor", help="环境与能力自检")
    sp.set_defaults(func=cmd_doctor)

    sp = sub.add_parser("search", help="多源融合检索（OpenAlex + arXiv）")
    sp.add_argument("query", help="检索式（多词请用单引号包裹）")
    sp.add_argument(
        "--source", default="auto", choices=["auto", "openalex", "arxiv", "wos"]
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
        help="补 WoS 收录号 wos_id（逐条调用，较慢）",
    )
    sp.add_argument("--save", action="store_true", help="保存检索快照到 shortlists/")
    sp.add_argument("--purpose", default=None, help="本次检索目的（写入快照）")
    sp.add_argument(
        "--json",
        action="store_true",
        help="输出 JSON（_row() dict 数组，机器可读；进度提示转 stderr）",
    )
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
        "--refresh",
        action="store_true",
        dest="force",
        help="忽略缓存全文，强制重新抓取/抽取",
    )
    _add_force_alias(sp)
    sp.set_defaults(func=cmd_read, note=True)

    sp = sub.add_parser("get", help="输出融合元数据")
    sp.add_argument("identifier", help="DOI / arXiv id / OpenAlex id")
    sp.add_argument("--json", action="store_true", help="输出 JSON（机器可读）")
    sp.set_defaults(func=cmd_get)

    sp = sub.add_parser("add", help="入库 Zotero + 建 papers/ 笔记骨架（不抓全文）")
    sp.add_argument("doi", help="DOI（或 arXiv / OpenAlex id）")
    sp.add_argument("--tags", default=None, help="逗号分隔的标签")
    sp.add_argument("--overwrite", action="store_true")
    sp.add_argument(
        "--verify",
        action=argparse.BooleanOptionalAction,
        default=True,
        dest="verify",
        help="写入前三源引用核验（默认开；--no-verify 跳过）",
    )
    sp.add_argument(
        "--allow-fail",
        action="store_true",
        dest="force",
        help="越过引用核验门（FAIL 也写入）——这是安全门开关，与缓存刷新无关",
    )
    _add_force_alias(sp)
    sp.set_defaults(func=cmd_add)

    sp = sub.add_parser(
        "citecheck", help="引用完整性门（OpenAlex+Crossref+arXiv 三源交叉核验）"
    )
    sp.add_argument(
        "targets", nargs="*", help="DOI / arXiv id / OpenAlex id / 论文标题（可多个）"
    )
    sp.add_argument(
        "--note",
        action="append",
        default=None,
        help="核验指定笔记 .md 文件/目录（可重复）",
    )
    sp.add_argument("--all", action="store_true", help="扫 papers/ 全部笔记核验")
    sp.add_argument(
        "--bib",
        action="append",
        default=None,
        help="核验参考文献文件（可重复）：.bib 走 BibTeX 解析，其余（如 .md）走参考文献段解析",
    )
    sp.add_argument(
        "--review",
        action="store_true",
        help="--bib 的简写：扫 reviews/*.md 全部综述的参考文献段",
    )
    sp.add_argument(
        "--limit",
        type=int,
        default=50,
        help="--bib/--review 解析出的引用的核验上限（每条最多 3 次网络请求）；0 = 不限",
    )
    sp.add_argument("--json", action="store_true", help="输出 JSON（机器可读）")
    sp.add_argument(
        "--no-color", action="store_true", dest="no_color", help="禁用 ANSI 颜色"
    )
    sp.set_defaults(func=cmd_citecheck)

    sp = sub.add_parser(
        "citegraph",
        help="引文图谱滚雪球（backward = 它引了谁，forward = 谁引了它；走 OpenAlex）",
    )
    sp.add_argument("id", help="DOI / arXiv id / OpenAlex id")
    sp.add_argument(
        "--direction",
        default="both",
        choices=["backward", "forward", "both"],
        help="backward 取参考文献，forward 取被引（默认 both）",
    )
    sp.add_argument(
        "--limit", type=int, default=25, help="每个方向展示的条数；0 = 不限"
    )
    sp.add_argument("--year", default=None, help="forward：如 2020-2026 / 2020- / 2020")
    sp.add_argument(
        "--sort",
        default="citations",
        choices=["relevance", "date", "citations"],
        help="默认按被引降序（滚雪球先看最有影响力的那几篇）",
    )
    sp.add_argument("--save", action="store_true", help="保存结果快照到 shortlists/")
    sp.add_argument("--purpose", default=None, help="本次滚雪球目的（写入快照）")
    sp.add_argument("--json", action="store_true", help="输出 JSON（两个方向分开）")
    sp.set_defaults(func=cmd_citegraph)

    sp = sub.add_parser("library", help="查询 Zotero 库")
    sp.add_argument("action", choices=["ping", "list", "search", "get"])
    sp.add_argument("--query", default=None, help="search 的关键词")
    sp.add_argument("--key", default=None, help="get 的条目 key")
    sp.add_argument("--type", default=None, dest="type", help="list 时按 itemType 过滤")
    sp.add_argument(
        "--limit", type=int, default=25, help="上限（list 被上游夹到 ≤100）"
    )
    sp.set_defaults(func=cmd_library)

    sp = sub.add_parser(
        "journal",
        help="期刊质量指标（SCImago 分区 / OpenAlex listed_in 专家评议分级）",
    )
    sp.add_argument("action", choices=["lookup", "build-scimago", "status"])
    sp.add_argument(
        "issn", nargs="?", default=None, help="lookup：期刊 ISSN（如 0031-9007）"
    )
    sp.add_argument(
        "--csv",
        default=None,
        help="build-scimago：官方 SCImago CSV 路径（须手工从 scimagojr.com 下载）",
    )
    sp.add_argument(
        "--year",
        type=int,
        default=None,
        help="build-scimago：SJR 版本年（默认从 CSV 表头/文件名推断）",
    )
    sp.add_argument("--json", action="store_true", help="输出 JSON（机器可读）")
    sp.set_defaults(func=cmd_journal)

    sp = sub.add_parser(
        "review",
        help="综述笔记脚手架（new / status / sync）；不生成综述正文，那是 AI 的活",
    )
    sp.add_argument("action", choices=["new", "status", "sync"])
    # 只用**一个**可选位置参数（与 ``rag`` 的 action + query 同形）。拆成 ``topic`` 与
    # ``path`` 两个 nargs="?" 位置参数是行不通的：argparse 按声明顺序贪婪分配，
    # ``review status <文件>`` 会把文件当成 topic，而 ``path`` 永远是 None。
    sp.add_argument(
        "target",
        nargs="?",
        default=None,
        help="new：综述主题（多词请用单引号包裹）；status/sync：综述文件（缺省扫 reviews/*.md）",
    )
    sp.add_argument(
        "--purpose", default=None, help="new：综述目标（填入正文的「综述目标」行）"
    )
    sp.add_argument(
        "--from-shortlist",
        nargs="+",
        default=None,
        dest="from_shortlist",
        help="new：从检索快照聚合 sources_used 与 query_strings（可多个）",
    )
    sp.add_argument(
        "--time-window",
        default=None,
        dest="time_window",
        help="new：检索时间窗（如 '2024-01 至 2026-09'）",
    )
    sp.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help="sync：只报告将重写的字段，不写文件",
    )
    sp.set_defaults(func=cmd_review)

    sp = sub.add_parser(
        "rag",
        help="PaperQA2 语义检索本地文献库（index/search/ask/status）",
    )
    sp.add_argument("action", choices=["index", "search", "ask", "status"])
    sp.add_argument("query", nargs="?", default=None, help="search/ask 的查询词")
    sp.add_argument(
        "--path",
        action="append",
        default=None,
        dest="paths",
        help="index：显式指定 .md 文件/目录（可重复；默认扫 cache/extracted）",
    )
    sp.add_argument("--rebuild", action="store_true", help="index：清空全量重建")
    sp.add_argument(
        "-k", "--top-k", type=int, default=8, dest="top_k", help="检索/证据块数"
    )
    sp.add_argument("--chars", type=int, default=400, help="渲染时每块正文截断长度")
    sp.add_argument("--json", action="store_true", help="输出 JSON（机器可读）")
    sp.set_defaults(func=cmd_rag)

    sp = sub.add_parser("index", help="从 papers/ 重建 INDEX.md")
    sp.add_argument(
        "--check",
        action="store_true",
        help="仅校验一致性，不写文件（构建日期不参与比对）；不一致时退出码 1",
    )
    sp.add_argument(
        "--fix",
        action="store_true",
        help="规范化 papers/*.md 的 frontmatter（补齐模板字段、统一字段序与 block style）；正文逐字节保留，不自动重命名文件",
    )
    sp.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help="与 --fix 同用：只报告将发生的变更，任何文件都不写（含 INDEX.md）",
    )
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
        help=(
            "prune：跳过仍被引用的文件——两个来源：papers/ 笔记的 frontmatter 与 "
            "ingest/manifest.json（ingest 产物不挂在笔记上，只认前者会漏掉整个入库语料）。默认开"
        ),
    )
    sp.set_defaults(func=cmd_cache)

    sp = sub.add_parser(
        "ingest",
        help="本地 PDF 文献仓库批量入库（复制→抽取→登记 manifest，断点续跑）",
    )
    sp.add_argument(
        "action",
        nargs="?",
        default="run",
        choices=["run", "status"],
        help="run = 跑入库（默认，可省）；status = 只打印清单进度汇总（等同 --status）",
    )
    sp.add_argument(
        "--manifest",
        default=None,
        help="入库清单 JSON 路径（默认 ingest/manifest.json）",
    )
    sp.add_argument(
        "--status",
        action="store_true",
        help="等同 `ingest status`（旧写法，保留）：只打印清单进度汇总，不做任何转换",
    )
    sp.add_argument(
        "--priority",
        type=int,
        default=None,
        help="只处理该优先级（1 论文/2 综述长文/3 教材）",
    )
    sp.add_argument("--theme", default=None, help="只处理该主题（如 cpa_ep）")
    sp.add_argument(
        "--backend",
        default=None,
        help="PDF 抽取后端：mineru-cloud / pymupdf4llm / auto（默认 auto=云端优先）",
    )
    sp.add_argument(
        "--limit-pages",
        type=int,
        default=None,
        dest="limit_pages",
        help="本次最多抽取多少页（页数预算，防超 MinerU 日限额）",
    )
    sp.add_argument(
        "--limit-files",
        type=int,
        default=None,
        dest="limit_files",
        help="本次最多处理多少篇",
    )
    sp.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help="只预演将处理哪些条目，不复制/不抽取/不改状态",
    )
    sp.add_argument(
        "--refresh",
        action="store_true",
        dest="force",
        help="已 done 的条目也重新抽取（绕过缓存/状态，非安全门）",
    )
    _add_force_alias(sp)
    sp.set_defaults(func=cmd_ingest)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _migrate_force_alias(args)
    if getattr(args, "cmd", None) in {
        "search",
        "read",
        "get",
        "add",
        "citecheck",
        "citegraph",
    }:
        cache_manager.maybe_autoclean()
    try:
        rc = args.func(args)
        return int(rc) if rc else 0
    except KeyboardInterrupt:
        print("\n[research] 已中断。", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
