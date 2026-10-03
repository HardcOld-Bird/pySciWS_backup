"""rag —— PaperQA2 语义检索基础设施（**服务于 agent，检索为主**）。

阶段 5：把 `paper-qa`（PaperQA2）封装成本项目文献库的语义检索层。设计理念源自
用户的澄清——RAG 引擎是 **agent（我）** 的基础设施：核心是用 embedding 对全文做
语义索引 + 检索，让 agent 直接读到「相关 chunk + 出处引文」，由 agent 自己跨文献
综合，而无需逐一阅读正文。PaperQA2 自带的 LLM 综述/回答对此场景是冗余的，仅保留为
**可选**的便利功能（``ask``），且严格遵循「免费优先 → 付费丝滑回退 → 永不阻塞 →
不制造噪音」：全部失败时静默降级为返回检索结果。

三条路径（成本/依赖递增）
------------------------
- :func:`build_index` —— 对 ``cache/extracted/**/*.md``（MinerU 全文）建 embedding
  索引，pickle 持久化到 ``pqa_home``。**仅用免费 embedding，无 LLM**。增量：只加新文件，
  ``--rebuild`` 全量重建。
- :func:`search` —— 纯 embedding 语义检索 top-k chunk + 出处（rank 序）。**agent 主力**，
  零 LLM、零成本、无模型可用性顾虑。
- :func:`ask` —— 可选：用 LLM 对检索证据做一句话综述。免费档 LLM 优先，不可用则回退
  付费档，再不可用则**降级为 search 结果**（``degraded=True``），绝不抛异常/阻塞。

关键工程约束（均由 spike 实证）
------------------------------
1. **绕过 Semantic Scholar**：``parsing.use_doc_details=False`` + 自带 citation 元数据，
   索引/检索/ask 全程不联网查元数据源（S2 已删，天然 S2-free）。
2. **注册自定义模型的 token 上限**：硅基流动的 ``openai/...`` 兼容模型名不在 LiteLLM 内置
   cost map 里，``lmi.embeddings._truncate_if_large`` 会取 ``model_cost[name]['max_input_tokens']``
   而抛 ``KeyError``。故任何 embedding/LLM 调用前必须 :func:`_register_litellm_models`。
3. **本地 cost map**：``LITELLM_LOCAL_MODEL_COST_MAP=True`` 必须在导入 litellm 前设置，
   否则启动会去拉 raw.githubusercontent.com（校园网超时 ~40s）。
4. **懒导入**：``paperqa`` / ``litellm`` 只在真正调用时导入，避免拖慢 research CLI 启动，
   也便于离线单测 monkeypatch。

用法::

    from pysci.skills.literature_research.tools import rag
    rag.build_index()                              # 建/更新索引（免费 embedding）
    res = rag.search("exceptional point absorption", k=8)
    for c in res.chunks:
        print(c.rank, c.citation, c.text[:120])
    ans = rag.ask("How is CPA EP realized?")       # 可选综述；失败自动降级
"""

from __future__ import annotations

import asyncio
import json
import os
import pickle
import re
import warnings
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import settings

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
# 自定义模型的 token 上限（LiteLLM cost map 缺失时的注册值）。embedding 取 bge-m3 真实
# 上限 8192；chat 取宽松 32768（截断仅在超长输入触发，我们的 chunk 有界，够用）。
EMBED_MAX_TOKENS = 8192
CHAT_MAX_TOKENS = 32768

INDEX_FILENAME = "index.pkl"  # pickle 持久化的 Docs 对象
META_FILENAME = (
    "index_meta.json"  # 索引台账：版本/模型/文件清单（docname→path/mtime/引文）
)

# ask 综述降本参数
ASK_EVIDENCE_K = 4
ASK_MAX_SOURCES = 3


# ---------------------------------------------------------------------------
# 后端懒加载 + LiteLLM 模型注册
# ---------------------------------------------------------------------------
def _register_litellm_models(pairs: list[tuple[str, str]]) -> None:
    """让 LiteLLM 认识自定义 ``openai/`` 兼容模型的 token 上限（否则 embedding 抛 KeyError）。

    ``pairs`` 为 ``[(model_name, mode), ...]``，``mode`` ∈ {"embedding", "chat"}。带/不带
    ``openai/`` 前缀都注册，以覆盖 lmi 可能使用的两种 ``self.name`` 形式。
    """
    # 静音 LiteLLM 的调试噪声（如「Provider List: ...」）：用户要求不制造分心/噪音。
    # 仅压制 INFO/DEBUG，ERROR 仍保留，不掩盖真错误。
    import logging

    import litellm

    litellm.suppress_debug_info = True
    for _name in ("LiteLLM", "litellm"):
        logging.getLogger(_name).setLevel(logging.ERROR)

    entries: dict[str, dict[str, Any]] = {}
    for name, mode in pairs:
        max_tok = EMBED_MAX_TOKENS if mode == "embedding" else CHAT_MAX_TOKENS
        for variant in {name, name.removeprefix("openai/")}:
            entries[variant] = {
                "max_input_tokens": max_tok,
                "max_tokens": max_tok,
                "input_cost_per_token": 0.0,
                "output_cost_per_token": 0.0,
                "litellm_provider": "openai",
                "mode": mode,
            }
    litellm.register_model(entries)


def _import_backend(*, models: list[tuple[str, str]]) -> Any:
    """设置本地 cost map → 导入 paperqa → 注册自定义模型。返回 paperqa 模块。"""
    if not settings.pqa_ready:
        raise RuntimeError(
            "PaperQA2 RAG 未就绪：缺少 SILICONFLOW_API_KEY（.env）。"
            "检索/索引依赖硅基流动的 OpenAI 兼容 embedding，请先填写密钥。"
        )
    # 必须在导入 litellm（paperqa 内部依赖）之前设置，避免拉远程 cost map 超时
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    # litellm 的异步成功日志回调在 asyncio.run 收尾时可能未被 await，触发无害的
    # RuntimeWarning；用户要求「不制造噪音」，精确过滤这一条（不掩盖其它 RuntimeWarning）。
    warnings.filterwarnings(
        "ignore",
        message=r"coroutine 'Logging\.async_success_handler' was never awaited",
        category=RuntimeWarning,
    )
    import paperqa  # noqa: PLC0415

    _register_litellm_models(models)
    return paperqa


def _router_cfg(model: str) -> dict[str, Any]:
    """PaperQA2 的 ``*_config``：LiteLLM Router 的 ``model_list`` 形状（走硅基流动）。"""
    return {
        "model_list": [
            {
                "model_name": model,
                "litellm_params": {
                    "model": model,
                    "api_base": settings.siliconflow_base_url.rstrip("/"),
                    "api_key": settings.siliconflow_api_key,
                },
            }
        ]
    }


def _build_pqa_settings(*, llm_model: str | None = None) -> Any:
    """构造 paperqa.Settings。

    - 只给 embedding（``llm_model=None``）→ 索引/纯检索路径，**不含任何 LLM 配置**。
    - 给 ``llm_model`` → 额外配 llm/summary_llm（ask 综述路径），并收紧证据数降本。
    两者都 ``use_doc_details=False``（绕 S2/Crossref，元数据自带）+ ``multimodal=False``。
    """
    from paperqa import Settings  # noqa: PLC0415

    kwargs: dict[str, Any] = {
        "embedding": settings.pqa_embedding,
        "embedding_config": _router_cfg(settings.pqa_embedding),
        "parsing": {"use_doc_details": False, "multimodal": False},
    }
    if llm_model:
        kwargs.update(
            llm=llm_model,
            llm_config=_router_cfg(llm_model),
            summary_llm=llm_model,
            summary_llm_config=_router_cfg(llm_model),
        )
    s = Settings(**kwargs)
    if llm_model:
        s.answer.evidence_k = ASK_EVIDENCE_K
        s.answer.answer_max_sources = ASK_MAX_SOURCES
    return s


def _map_mailto_env() -> None:
    """把 OpenAlex 邮箱映射到 PaperQA2 读的 ``CROSSREF_MAILTO``/``OPENALEX_MAILTO``。

    主路径 ``use_doc_details=False`` 不查元数据源，此映射只在极端情况（内部偶发查询）
    让我们进 polite pool；无邮箱则无副作用。
    """
    mailto = (settings.openalex_email or "").strip()
    if mailto:
        os.environ.setdefault("CROSSREF_MAILTO", mailto)
        os.environ.setdefault("OPENALEX_MAILTO", mailto)


def _run(coro: Any) -> Any:
    """在同步 CLI 里驱动 async 协程。"""
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------
@dataclass
class RagChunk:
    """检索命中的单个文本块（rank 序，1 起）。"""

    rank: int
    text: str
    docname: str
    citation: str
    source_path: str | None = None
    chunk_name: str = ""


@dataclass
class RagSearchResult:
    """一次纯 embedding 检索的结果。"""

    query: str
    chunks: list[RagChunk] = field(default_factory=list)
    n_docs: int = 0
    n_chunks: int = 0
    index_path: str | None = None
    embedding_model: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class IndexReport:
    """一次索引构建的报告。"""

    n_added: int = 0
    n_skipped: int = 0
    n_stale: int = 0  # mtime 变了但 docname 已在索引（建议 --rebuild）
    n_docs: int = 0
    n_chunks: int = 0
    index_path: str = ""
    embedding_model: str = ""
    rebuilt: bool = False
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RagAskResult:
    """一次 ask（可选 LLM 综述）的结果。``degraded=True`` 表示已降级为检索结果。"""

    query: str
    answer: str | None = None
    backend: str = "none"  # free | fallback | none
    model: str | None = None
    cost: float = 0.0
    citations: list[str] = field(default_factory=list)
    degraded: bool = False
    search: RagSearchResult | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


# ---------------------------------------------------------------------------
# 从 MinerU .md 头部派生轻量元数据（无 frontmatter，故解析正文）
# ---------------------------------------------------------------------------
_DOI_RE = re.compile(r"\b(10\.\d{4,9}/[^\s,;)\]}]+)", re.IGNORECASE)


def _clean_inline(s: str) -> str:
    """去掉 LaTeX 上标/脚注标记与多余空白，用于作者行解析。"""
    s = re.sub(r"\$[^$]*\$", " ", s)  # $^{1,*}$ 之类
    s = re.sub(r"[*†‡§∥¶]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _derive_meta_from_md(path: Path) -> dict[str, Any]:
    """从 .md 头部（前 ~4000 字）尽力派生 title/year/doi/首作者/引文。全部 best-effort。

    MinerU 产物形如：``# 标题`` → 作者行 → 单位行 → ``(Received …; published 6 August 2025)``
    → 摘要 → ``DOI: 10.xxxx/xxxx`` → 正文。任一字段缺失都有兜底，绝不抛异常。
    """
    try:
        head = path.read_text(encoding="utf-8", errors="replace")[:4000]
    except OSError:
        head = ""
    lines = [ln.strip() for ln in head.splitlines()]

    # 标题：首个 Markdown 标题行；兜底用文件名 stem
    title = ""
    for ln in lines:
        if ln.startswith("#"):
            title = ln.lstrip("#").strip()
            break
    if not title:
        title = path.stem.replace("_", " ")

    # 年份：只认 published/accepted/received/© 旁的四位年（不把正文里任意数字当年份，
    # 否则手册/综述会误抓到无关年份）
    year: int | None = None
    for pat in (
        r"published[^)\n]*?((?:19|20)\d{2})",
        r"accepted[^)\n]*?((?:19|20)\d{2})",
        r"received[^)\n]*?((?:19|20)\d{2})",
        r"[©Ⓒ]\s*((?:19|20)\d{2})",
    ):
        m = re.search(pat, head, re.IGNORECASE)
        if m:
            year = int(m.group(1))
            break

    # DOI
    m = _DOI_RE.search(head)
    doi = m.group(1).rstrip(".") if m else ""

    # 首作者姓：标题后第一条「像人名行」的逗号/and 前末词。含 URL、单位关键词、
    # 或不足「名 姓」两段的行一律跳过；宁可无姓（退化为标题引文）也不要抓错。
    affil = (
        "university",
        "institute",
        "laborator",
        "department",
        "academy",
        "school",
        "college",
        "centre",
        "center",
        "faculty",
    )
    url_tok = ("www", "http", ".com", ".se", ".org", ".cn", ".edu", ".gov")
    surname = ""
    for ln in lines[1:14]:
        cleaned = _clean_inline(ln)
        low = cleaned.lower()
        if not cleaned or cleaned.startswith(("(", "doi", "http")):
            continue
        if any(tok in low for tok in url_tok) or any(tok in low for tok in affil):
            continue
        if "," not in cleaned and " and " not in low:
            continue
        first = re.split(r",|\s+and\s+", cleaned)[0].strip()
        words = [w for w in re.split(r"\s+", first) if w and re.match(r"^[A-Za-z]", w)]
        if len(words) >= 2:  # 至少「名 姓」两段才认为确是人名
            cand = re.sub(r"[^A-Za-z]", "", words[-1])
            if len(cand) >= 2:
                surname = cand
        break

    if surname and year:
        citation = f"{surname} et al. ({year})"
    elif year:
        citation = f"{title[:60]} ({year})"
    else:
        citation = title[:80] or path.stem

    return {
        "title": title,
        "year": year,
        "doi": doi,
        "first_author": surname,
        "citation": citation,
    }


# ---------------------------------------------------------------------------
# 索引持久化（pqa_home 下 pickle + 台账 json）
# ---------------------------------------------------------------------------
def _index_path() -> Path:
    return settings.pqa_home / INDEX_FILENAME


def _meta_path() -> Path:
    return settings.pqa_home / META_FILENAME


def _docname_for(path: Path) -> str:
    """文件路径 → 稳定唯一 docname（相对数据区，分隔符折成 ``__``，去 .md）。"""
    try:
        rel = path.resolve().relative_to(settings.cache_extracted.resolve())
    except (ValueError, OSError):
        try:
            rel = path.resolve().relative_to(settings.project_root.resolve())
        except (ValueError, OSError):
            rel = Path(path.name)
    parts = list(rel.with_suffix("").parts)
    return "__".join(parts)


def _candidate_md_files(paths: list[str | Path] | None) -> list[Path]:
    """确定要索引的 .md 列表：显式 paths 优先，否则默认扫 cache/extracted/**/*.md。"""
    out: list[Path] = []
    if paths:
        for p in paths:
            pp = Path(p)
            if pp.is_dir():
                out.extend(sorted(pp.rglob("*.md")))
            elif pp.is_file() and pp.suffix.lower() == ".md":
                out.append(pp)
    else:
        out.extend(sorted(settings.cache_extracted.rglob("*.md")))
    # 去重（保持顺序）
    seen: set[str] = set()
    uniq: list[Path] = []
    for pp in out:
        key = str(pp.resolve())
        if key not in seen:
            seen.add(key)
            uniq.append(pp)
    return uniq


def _load_meta() -> dict[str, Any]:
    p = _meta_path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _save_meta(meta: dict[str, Any]) -> None:
    _meta_path().write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _load_docs(paperqa: Any) -> Any | None:
    """载入已持久化的 Docs；版本不符/损坏则返回 None（触发重建）。"""
    p = _index_path()
    meta = _load_meta()
    if not p.exists():
        return None
    if meta.get("paperqa_version") != getattr(paperqa, "__version__", None):
        return None  # 版本漂移，pickle 可能不兼容 → 让上层重建
    try:
        with p.open("rb") as f:
            return pickle.load(f)  # noqa: S301
    except Exception:  # noqa: BLE001
        return None


def _save_docs(docs: Any, meta: dict[str, Any]) -> None:
    with _index_path().open("wb") as f:
        pickle.dump(docs, f)
    _save_meta(meta)


def index_status() -> dict[str, Any]:
    """索引状态（供 doctor / __main__ 用，不触发任何导入或网络）。"""
    meta = _load_meta()
    return {
        "ready": settings.pqa_ready,
        "exists": _index_path().exists(),
        "index_path": str(_index_path()),
        "n_docs": meta.get("n_docs", 0),
        "n_chunks": meta.get("n_chunks", 0),
        "embedding_model": meta.get("embedding_model") or settings.pqa_embedding,
        "built_at": meta.get("built_at"),
        "paperqa_version": meta.get("paperqa_version"),
    }


# ---------------------------------------------------------------------------
# 公开 API：build_index / search / ask
# ---------------------------------------------------------------------------
def build_index(
    paths: list[str | Path] | None = None,
    *,
    rebuild: bool = False,
    verbose: bool = True,
) -> IndexReport:
    """对文献库全文 .md 建/更新 embedding 索引（**仅免费 embedding，无 LLM**）。

    默认扫 ``cache/extracted/**/*.md``；``paths`` 给定则只索引这些。增量：已在台账且
    未变更的文件跳过；``rebuild=True`` 清空重建。索引 pickle 到 ``pqa_home/index.pkl``。
    """
    paperqa = _import_backend(models=[(settings.pqa_embedding, "embedding")])
    _map_mailto_env()

    files = _candidate_md_files(paths)
    report = IndexReport(
        embedding_model=settings.pqa_embedding,
        index_path=str(_index_path()),
        rebuilt=rebuild,
    )
    if not files:
        report.errors.append(
            f"未找到可索引的 .md（默认目录 {settings.cache_extracted}）。"
            "先用 `research read`/`ingest` 抽取全文，或显式传路径。"
        )
        return report

    meta = {} if rebuild else _load_meta()
    docs = None if rebuild else _load_docs(paperqa)
    if docs is None:
        docs = paperqa.Docs()
        meta = {
            "files": {},
            "embedding_model": settings.pqa_embedding,
            "paperqa_version": getattr(paperqa, "__version__", None),
        }
        rebuild = True
        report.rebuilt = True
    file_meta: dict[str, Any] = meta.setdefault("files", {})

    pqa_settings = _build_pqa_settings()  # embedding-only

    async def _add_all() -> None:
        for path in files:
            docname = _docname_for(path)
            try:
                mtime = path.stat().st_mtime
            except OSError as e:
                report.errors.append(f"{path.name}: 无法 stat（{e}）")
                continue
            prev = file_meta.get(docname)
            if prev is not None:
                if abs(float(prev.get("mtime", 0)) - mtime) < 1e-6:
                    report.n_skipped += 1
                    continue
                # 内容变了：增量不覆盖同名 doc，标记 stale 提示重建
                report.n_stale += 1
                if verbose:
                    print(f"[rag] 变更（需 --rebuild 才更新）：{docname}")
                continue
            derived = _derive_meta_from_md(path)
            try:
                await docs.aadd(
                    str(path),
                    docname=docname,
                    citation=derived["citation"],
                    title=derived["title"],
                    doi=derived["doi"] or None,
                    settings=pqa_settings,
                )
            except Exception as e:  # noqa: BLE001
                report.errors.append(f"{path.name}: aadd 失败 {type(e).__name__}: {e}")
                continue
            file_meta[docname] = {
                "path": str(path.resolve()),
                "mtime": mtime,
                **derived,
            }
            report.n_added += 1
            if verbose:
                print(f"[rag] + {docname}  «{derived['citation']}»")

    _run(_add_all())

    report.n_docs = len(getattr(docs, "docs", {}))
    report.n_chunks = len(getattr(docs, "texts", []))
    meta.update(
        n_docs=report.n_docs,
        n_chunks=report.n_chunks,
        embedding_model=settings.pqa_embedding,
        paperqa_version=getattr(paperqa, "__version__", None),
        built_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )
    _save_docs(docs, meta)
    if verbose:
        print(
            f"[rag] 索引就绪：docs={report.n_docs} chunks={report.n_chunks} "
            f"(+{report.n_added} 新增, {report.n_skipped} 跳过, {report.n_stale} 待重建) → {_index_path()}"
        )
    return report


def _load_index_or_raise(paperqa: Any) -> tuple[Any, dict[str, Any]]:
    docs = _load_docs(paperqa)
    if docs is None:
        raise RuntimeError(
            "尚无可用索引（或 paper-qa 版本已变需重建）。请先运行 `research rag index`。"
        )
    return docs, _load_meta()


def search(query: str, k: int = 8) -> RagSearchResult:
    """纯 embedding 语义检索 top-k chunk + 出处（**agent 主力，无 LLM、零成本**）。"""
    paperqa = _import_backend(models=[(settings.pqa_embedding, "embedding")])
    docs, meta = _load_index_or_raise(paperqa)
    file_meta: dict[str, Any] = meta.get("files", {})
    pqa_settings = _build_pqa_settings()  # embedding-only

    async def _retrieve() -> list[Any]:
        return await docs.retrieve_texts(query, k, settings=pqa_settings)

    texts = _run(_retrieve())
    chunks: list[RagChunk] = []
    for i, t in enumerate(texts, 1):
        doc = getattr(t, "doc", None)
        docname = getattr(doc, "docname", None) or getattr(t, "name", "") or ""
        citation = getattr(doc, "citation", "") or ""
        src = file_meta.get(docname, {}).get("path")
        chunks.append(
            RagChunk(
                rank=i,
                text=getattr(t, "text", "") or "",
                docname=docname,
                citation=citation,
                source_path=src,
                chunk_name=getattr(t, "name", "") or "",
            )
        )
    return RagSearchResult(
        query=query,
        chunks=chunks,
        n_docs=len(getattr(docs, "docs", {})),
        n_chunks=len(getattr(docs, "texts", [])),
        index_path=str(_index_path()),
        embedding_model=settings.pqa_embedding,
    )


def _extract_answer(ans: Any) -> tuple[str, float, list[str]]:
    """从 PQASession 提取 (formatted_answer, cost, citations)，全部 best-effort。"""
    text = getattr(ans, "formatted_answer", None) or getattr(ans, "answer", None) or ""
    cost = float(getattr(ans, "cost", 0.0) or 0.0)
    cites: list[str] = []
    ctx = getattr(ans, "context", None) or []
    for c in ctx:
        cit = getattr(c, "citation", None)
        if cit:
            cites.append(str(cit))
    return str(text), cost, cites


def _ask_once(docs: Any, query: str, llm_model: str) -> tuple[str, float, list[str]]:
    """用指定 LLM 跑一次 aquery；失败抛异常由上层回退。"""
    pqa_settings = _build_pqa_settings(llm_model=llm_model)

    async def _q() -> Any:
        return await docs.aquery(query, settings=pqa_settings)

    return _extract_answer(_run(_q()))


def ask(query: str, k: int = ASK_EVIDENCE_K) -> RagAskResult:
    """可选：LLM 对检索证据做一句话综述。免费档优先 → 付费回退 → 降级 search，**永不阻塞**。"""
    result = RagAskResult(query=query)
    try:
        paperqa = _import_backend(
            models=[
                (settings.pqa_embedding, "embedding"),
                (settings.pqa_llm, "chat"),
                *(
                    [(settings.pqa_llm_fallback, "chat")]
                    if settings.pqa_llm_fallback
                    else []
                ),
            ]
        )
        docs, _meta = _load_index_or_raise(paperqa)
    except Exception as e:  # noqa: BLE001
        return _degrade(result, k, f"后端/索引不可用：{type(e).__name__}: {e}")

    # 1) 免费档
    try:
        ans, cost, cites = _ask_once(docs, query, settings.pqa_llm)
        if ans.strip():
            result.answer, result.cost, result.citations = ans, cost, cites
            result.backend, result.model = "free", settings.pqa_llm
            return result
    except Exception as e:  # noqa: BLE001
        result.error = f"free({settings.pqa_llm}) 失败：{type(e).__name__}: {e}"

    # 2) 付费回退
    if settings.pqa_llm_fallback:
        try:
            ans, cost, cites = _ask_once(docs, query, settings.pqa_llm_fallback)
            if ans.strip():
                result.answer, result.cost, result.citations = ans, cost, cites
                result.backend, result.model = "fallback", settings.pqa_llm_fallback
                return result
        except Exception as e:  # noqa: BLE001
            result.error = (
                (result.error or "")
                + f" | fallback({settings.pqa_llm_fallback}) 失败：{type(e).__name__}: {e}"
            )

    # 3) 全部失败 → 降级为检索结果
    return _degrade(result, k, result.error or "LLM 综述不可用")


def _degrade(result: RagAskResult, k: int, reason: str) -> RagAskResult:
    """把 ask 降级为 search 结果（静默、不抛异常）。"""
    result.backend, result.model, result.degraded = "none", None, True
    result.error = reason
    try:
        result.search = search(result.query, k=k)
    except Exception as e:  # noqa: BLE001
        result.error = f"{reason} | 检索亦失败：{type(e).__name__}: {e}"
    return result


# ---------------------------------------------------------------------------
# 渲染（供 research CLI 人类可读输出）
# ---------------------------------------------------------------------------
def render_search(res: RagSearchResult, *, chars: int = 400) -> str:
    lines = [
        f"检索：{res.query!r}  （embedding={res.embedding_model}，索引 docs={res.n_docs}/chunks={res.n_chunks}）",
    ]
    if not res.chunks:
        lines.append("（无命中）")
        return "\n".join(lines)
    for c in res.chunks:
        src = f"  ← {c.source_path}" if c.source_path else ""
        body = c.text.strip().replace("\n", " ")
        if len(body) > chars:
            body = body[:chars] + "…"
        lines.append(f"\n[{c.rank}] {c.citation}  ({c.docname}){src}\n    {body}")
    return "\n".join(lines)


def render_ask(res: RagAskResult, *, chars: int = 400) -> str:
    if res.degraded:
        head = f"（LLM 综述不可用，已降级为检索结果：{res.error}）"
        body = (
            render_search(res.search, chars=chars) if res.search else "（检索也无结果）"
        )
        return head + "\n" + body
    cites = "\n".join(f"  - {c}" for c in res.citations)
    return (
        f"综述（backend={res.backend}, model={res.model}, cost={res.cost}）：\n"
        f"{res.answer}\n" + (f"出处：\n{cites}" if cites else "")
    )


# ---------------------------------------------------------------------------
# CLI: 打印索引状态
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    print(
        "[rag] 调试后门——本入口只打印 index_status()，能力是 `research rag` 的真子集："
        "看状态用 `research rag status`（加 `--json` 得到与下方同形的输出），"
        "建索引 / 检索 / 问答分别是 `rag index` / `rag search` / `rag ask`。",
        file=sys.stderr,
    )
    print(json.dumps(index_status(), ensure_ascii=False, indent=2))
