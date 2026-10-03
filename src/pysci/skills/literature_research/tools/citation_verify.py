"""citation_verify —— 引用完整性门（OpenAlex + Crossref + arXiv 三源交叉核验）。

阶段 6：借鉴 academic-research-skills 的「引用核验」思路（**只借鉴思路，不引入其
CC-BY-NC 本体**），对笔记/综述里的每条 DOI / arXiv id / 参考文献，用三个相互独立的
开放学术源交叉核对，检测 **标题 / 首作者 / 年份 / 期刊 / DOI** 的实质性冲突——
冲突则标红（``FAIL``），供上层「阻止写入」。三源均绕开 Semantic Scholar（S2 已删）。

核心设计
--------
- 三源各自 ``fetch_*`` → 归一化为 :class:`SourceRecord`（统一字段形状）；
- 逐字段成对比对（含「笔记声称值 claim」这一伪源），比对器返回严重度：
  ``hard``（标题/DOI/首作者/年份相差≥2）→ ``FAIL``；``soft``（期刊/年份相差 1）→ ``WARN``；
- **优雅降级**：某源网络不可达只记 ``reachable=False``，绝不当作冲突；
  只有「两个及以上可达源就同一字段给出实质冲突」才判 ``FAIL``；
  三源全部查无此条 → ``NOT_FOUND``（可疑但不阻断，供人工复核）；
- 纯归一化函数（``normalize_*`` / ``title_similarity``）与比对器均可离线单测；
  三源 HTTP 全走 :func:`config.http_session`，Crossref 响应复用两层缓存。

Crossref REST（免费、无需 key，``mailto`` 进 polite pool）::

    GET https://api.crossref.org/works/{doi}          # 按 DOI 精确取
    GET https://api.crossref.org/works?query.bibliographic=<title>&rows=1  # 仅标题兜底

用法::

    from pysci.skills.literature_research.tools.citation_verify import (
        verify_citation, verify_frontmatter, render_verdict,
    )
    v = verify_citation({"doi": "10.1103/PhysRevLett.121.124501", "title": "..."})
    print(render_verdict(v))
    if not v.passed():        # 仅 FAIL 阻断；WARN / NOT_FOUND 放行
        ...
"""

from __future__ import annotations

import difflib
import hashlib
import json
import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote

from .config import http_session, settings

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
CROSSREF_BASE = "https://api.crossref.org"

# 标题相似度阈值：低于此值视为「不是同一篇」（归一化后 difflib 比率）
TITLE_SIM_THRESHOLD = 0.82
# 期刊名相似度阈值：期刊全称/缩写差异大，阈值放低，且冲突只记 soft（WARN）
JOURNAL_SIM_THRESHOLD = 0.55
# 年份相差达此值视为 hard 冲突（相差 1 常见于 online-first vs 见刊年，仅 soft）
YEAR_HARD_GAP = 2

# 三源标识（fetch 顺序即此顺序；claim 为「笔记声称值」伪源，比对时并入）
SOURCES = ("openalex", "crossref", "arxiv")

# 状态常量
PASS, WARN, FAIL, NOT_FOUND = "PASS", "WARN", "FAIL", "NOT_FOUND"


# ---------------------------------------------------------------------------
# 归一化纯函数（离线可测）
# ---------------------------------------------------------------------------
def _strip_accents(s: str) -> str:
    """去声调（NFKD 分解后剔除组合记号），便于跨源作者名比对。"""
    return "".join(
        c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)
    )


def normalize_title(t: str | None) -> str:
    """标题归一化：去声调 → 小写 → 剔除标点 → 折叠空白。用于相似度比对。"""
    if not t:
        return ""
    s = _strip_accents(str(t)).lower()
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def title_similarity(a: str | None, b: str | None) -> float:
    """两标题（归一化后）的相似度 ∈ [0,1]；任一为空返回 0.0。"""
    na, nb = normalize_title(a), normalize_title(b)
    if not na or not nb:
        return 0.0
    return difflib.SequenceMatcher(None, na, nb).ratio()


def normalize_last_name(name: str | None) -> str:
    """从各种作者名形态提取「姓」并归一化（去声调/小写/去标点）。

    兼容 ``'Zhu, Zheng'``（逗号前为姓）、``'Zheng Zhu'``（末词为姓）、
    ``'Zheng'``（单词名）三种形态；空值返回 ``''``。
    """
    if not name:
        return ""
    s = _strip_accents(str(name)).strip()
    if not s:
        return ""
    if "," in s:  # 'Last, First' → 取逗号前
        s = s.split(",", 1)[0]
    else:  # 'First M. Last' → 取末词
        parts = [p for p in re.split(r"\s+", s) if p]
        s = parts[-1] if parts else s
    s = re.sub(r"[^a-zA-Z]", "", s).lower()
    return s


def normalize_doi(d: str | None) -> str:
    """DOI 归一化：去 ``https://doi.org/`` / ``doi:`` 前缀 → 小写 → 去空白。"""
    if not d:
        return ""
    s = str(d).strip().lower()
    s = re.sub(r"^https?://(dx\.)?doi\.org/", "", s)
    s = re.sub(r"^doi:\s*", "", s)
    return s.strip()


def extract_year(v: Any) -> int | None:
    """从 int / '2018' / '2018-09-21' / [[2018,9,21]] 等形态提取四位年份。"""
    if v is None or v == "":
        return None
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v if 1000 <= v <= 2999 else None
    if isinstance(v, (list, tuple)):  # Crossref date-parts: [[2018, 9, 21]]
        for x in v:
            y = extract_year(x)
            if y:
                return y
        return None
    m = re.search(r"(1[5-9]\d{2}|20\d{2})", str(v))
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------
@dataclass
class SourceRecord:
    """单一源对某条引用的归一化记录。

    ``reachable=False`` 表示该源网络/接口异常（无法佐证，**不计入冲突**）；
    ``reachable=True, found=False`` 表示该源明确查无此条（弱信号）。
    """

    source: str
    reachable: bool = False
    found: bool = False
    title: str = ""
    authors: list[str] = field(default_factory=list)
    first_author_last_name: str = ""
    year: int | None = None
    journal: str = ""
    doi: str = ""
    arxiv_id: str = ""
    error: str = ""
    raw: dict[str, Any] | None = None


@dataclass
class FieldCheck:
    """单个字段的成对比对结果。``conflicts`` 为 [(srcA, srcB, severity), ...]。"""

    field: str
    values: dict[str, Any] = field(default_factory=dict)
    conflicts: list[tuple[str, str, str]] = field(default_factory=list)

    @property
    def worst(self) -> str | None:
        """该字段最严重的冲突等级（hard > soft > None）。"""
        sevs = [c[2] for c in self.conflicts]
        if "hard" in sevs:
            return "hard"
        if "soft" in sevs:
            return "soft"
        return None


@dataclass
class CitationVerdict:
    """一条引用的三源核验裁决。"""

    input: dict[str, Any]
    kind: str  # 主标识类型：doi | arxiv | openalex | title
    records: dict[str, SourceRecord] = field(default_factory=dict)
    checks: list[FieldCheck] = field(default_factory=list)
    status: str = NOT_FOUND
    reasons: list[str] = field(default_factory=list)

    def passed(self) -> bool:
        """是否放行写入：仅 ``FAIL`` 阻断；``PASS`` / ``WARN`` / ``NOT_FOUND`` 均放行。"""
        return self.status != FAIL

    @property
    def label(self) -> str:
        """供 CLI 打印的稳定标识（DOI 优先，其次 arXiv id / 标题）。"""
        return (
            self.input.get("doi")
            or self.input.get("arxiv_id")
            or self.input.get("openalex_id")
            or (self.input.get("title") or "")[:60]
            or "(空引用)"
        )


# ---------------------------------------------------------------------------
# Crossref REST（新源；免费无 key，mailto 进 polite pool，响应复用两层缓存）
# ---------------------------------------------------------------------------
def _crossref_cache_path(url: str, params: dict[str, Any]) -> Any:
    raw = url + "|" + json.dumps(params or {}, sort_keys=True, ensure_ascii=False)
    h = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", url.split("?")[0])[-60:]
    return settings.cache_api_responses / f"crossref_{slug}_{h}.json"


def _crossref_message(
    path: str, params: dict[str, Any] | None = None, *, use_cache: bool = True
) -> dict[str, Any] | None:
    """GET Crossref，返回 ``message`` 对象；404/未命中返回 None。

    网络/HTTP 异常向上抛，由 :func:`fetch_crossref` 归类为 ``reachable=False``。
    """
    from .openalex_client import _read_cache, _write_cache

    url = f"{CROSSREF_BASE}/{path}"
    params = dict(params or {})
    if settings.openalex_email and "mailto" not in params:
        params["mailto"] = settings.openalex_email

    cache_path = _crossref_cache_path(url, params)
    if use_cache:
        cached = _read_cache(cache_path)
        if cached is not None:
            return cached.get("message") if isinstance(cached, dict) else None

    with http_session() as s:
        r = s.get(url, params=params, timeout=settings.http_timeout)
        if r.status_code == 404:
            return None  # Crossref 明确查无此条（reachable, not found），区别于网络异常
        r.raise_for_status()
        data = r.json()

    _write_cache(cache_path, data)
    return data.get("message") if isinstance(data, dict) else None


def _record_from_crossref(m: dict[str, Any]) -> SourceRecord:
    """Crossref ``message`` → SourceRecord。"""
    rec = SourceRecord(source="crossref", reachable=True, found=True, raw=m)
    titles = m.get("title") or []
    rec.title = titles[0] if titles else ""
    container = m.get("container-title") or []
    rec.journal = container[0] if container else ""
    rec.doi = normalize_doi(m.get("DOI", ""))
    rec.year = extract_year((m.get("issued") or {}).get("date-parts"))
    authors: list[str] = []
    for a in m.get("author") or []:
        if not isinstance(a, dict):
            continue
        fam = a.get("family") or ""
        giv = a.get("given") or ""
        nm = a.get("name") or f"{giv} {fam}".strip()
        authors.append(nm)
    rec.authors = authors
    if m.get("author") and isinstance(m["author"][0], dict):
        rec.first_author_last_name = normalize_last_name(
            m["author"][0].get("family") or m["author"][0].get("name") or ""
        )
    return rec


def fetch_crossref(
    *,
    doi: str | None = None,
    title: str | None = None,
    first_author: str | None = None,
    use_cache: bool = True,
) -> SourceRecord:
    """从 Crossref 取记录：优先按 DOI 精确取；仅有标题时走 bibliographic 查询兜底。"""
    try:
        if doi:
            msg = _crossref_message(
                f"works/{quote(normalize_doi(doi), safe='')}", use_cache=use_cache
            )
            if msg:
                return _record_from_crossref(msg)
            return SourceRecord(source="crossref", reachable=True, found=False)
        if title:
            q = title if not first_author else f"{title} {first_author}"
            msg = _crossref_message(
                "works",
                params={"query.bibliographic": q, "rows": 1},
                use_cache=use_cache,
            )
            items = (msg or {}).get("items") if isinstance(msg, dict) else None
            if items:
                return _record_from_crossref(items[0])
            return SourceRecord(source="crossref", reachable=True, found=False)
        return SourceRecord(source="crossref", reachable=True, found=False)
    except Exception as e:  # 网络/HTTP/解析异常 → 不可达（不计冲突）
        return SourceRecord(
            source="crossref", reachable=False, error=f"{type(e).__name__}: {e}"
        )


# ---------------------------------------------------------------------------
# OpenAlex 源（复用既有 openalex_client）
# ---------------------------------------------------------------------------
def _record_from_openalex(w: dict[str, Any]) -> SourceRecord:
    authors = [
        a.get("name", "") for a in (w.get("authors") or []) if isinstance(a, dict)
    ]
    # 首作者姓优先从完整作者名派生（各源用同一 normalize_last_name 转写声调），
    # 而非直接用 openalex 预处理的 first_author_last_name（它删声调而非转写，会与其他源不一致）。
    first = (
        normalize_last_name(authors[0])
        if authors
        else normalize_last_name(w.get("first_author_last_name", ""))
    )
    return SourceRecord(
        source="openalex",
        reachable=True,
        found=True,
        title=w.get("title", "") or "",
        authors=authors,
        first_author_last_name=first,
        year=extract_year(w.get("publication_year")),
        journal=w.get("journal", "") or "",
        doi=normalize_doi(w.get("doi", "")),
        arxiv_id=w.get("arxiv_id", "") or "",
        raw=w,
    )


def fetch_openalex(
    *,
    doi: str | None = None,
    openalex_id: str | None = None,
    arxiv_id: str | None = None,
    title: str | None = None,
    use_cache: bool = True,
) -> SourceRecord:
    """从 OpenAlex 取记录：DOI/OpenAlex id 精确取；仅标题时 search 取最佳匹配。"""
    from . import openalex_client

    try:
        w = None
        if doi:
            w = openalex_client.get_work(doi=doi, use_cache=use_cache)
        elif openalex_id:
            w = openalex_client.get_work(openalex_id=openalex_id, use_cache=use_cache)
        elif title:
            res = openalex_client.search_works(
                query=title, per_page=3, use_cache=use_cache
            )
            w = _best_title_match(res.get("results") or [], title, key="title")
        if w:
            return _record_from_openalex(w)
        return SourceRecord(source="openalex", reachable=True, found=False)
    except Exception as e:
        return SourceRecord(
            source="openalex", reachable=False, error=f"{type(e).__name__}: {e}"
        )


# ---------------------------------------------------------------------------
# arXiv 源（复用既有 arxiv_client）
# ---------------------------------------------------------------------------
def _record_from_arxiv(e: dict[str, Any]) -> SourceRecord:
    authors = [
        a.get("name", "") for a in (e.get("authors") or []) if isinstance(a, dict)
    ]
    first = (
        normalize_last_name(authors[0])
        if authors
        else normalize_last_name(e.get("first_author_last_name", ""))
    )
    return SourceRecord(
        source="arxiv",
        reachable=True,
        found=True,
        title=e.get("title", "") or "",
        authors=authors,
        first_author_last_name=first,
        year=extract_year(e.get("published")),
        journal=e.get("journal_ref", "") or "",
        doi=normalize_doi(e.get("doi", "")),
        arxiv_id=e.get("arxiv_id", "") or "",
        raw=e,
    )


def fetch_arxiv(
    *,
    arxiv_id: str | None = None,
    title: str | None = None,
    use_cache: bool = True,
) -> SourceRecord:
    """从 arXiv 取记录：有 arXiv id 精确取；仅标题时 ti: 检索取最佳匹配。"""
    from . import arxiv_client

    try:
        entry = None
        if arxiv_id:
            entry = arxiv_client.get_paper(arxiv_id, use_cache=use_cache)
        elif title:
            res = arxiv_client.search_arxiv(
                f'ti:"{title}"', max_results=3, use_cache=use_cache
            )
            entry = _best_title_match(res.get("entries") or [], title, key="title")
        if entry:
            return _record_from_arxiv(entry)
        return SourceRecord(source="arxiv", reachable=True, found=False)
    except Exception as e:
        return SourceRecord(
            source="arxiv", reachable=False, error=f"{type(e).__name__}: {e}"
        )


def _best_title_match(
    items: list[dict[str, Any]], title: str, *, key: str
) -> dict[str, Any] | None:
    """从若干候选里挑标题相似度最高者（低于阈值也返回最佳，交由比对判定）。"""
    if not items:
        return None
    return max(items, key=lambda it: title_similarity(it.get(key, ""), title))


# ---------------------------------------------------------------------------
# 比对器（返回 'hard' / 'soft' / None）
# ---------------------------------------------------------------------------
def _cmp_title(a: str, b: str) -> str | None:
    if not a or not b:
        return None
    return "hard" if title_similarity(a, b) < TITLE_SIM_THRESHOLD else None


def _cmp_exact(a: Any, b: Any) -> str | None:
    """DOI / 首作者姓：两者皆非空且不等 → hard。"""
    if not a or not b:
        return None
    return "hard" if a != b else None


def _cmp_year(a: int | None, b: int | None) -> str | None:
    if not a or not b:
        return None
    gap = abs(int(a) - int(b))
    if gap == 0:
        return None
    return "hard" if gap >= YEAR_HARD_GAP else "soft"


def _cmp_name(a: str, b: str) -> str | None:
    """首作者姓：跨库转写/缩写/排序差异大，冲突只记 soft（WARN），不硬阻断。"""
    if not a or not b:
        return None
    return "soft" if a != b else None


def _cmp_journal(a: str, b: str) -> str | None:
    """期刊名差异大（全称/缩写），冲突只记 soft（WARN），且容忍包含关系。"""
    if not a or not b:
        return None
    if a in b or b in a:
        return None
    return "soft" if title_similarity(a, b) < JOURNAL_SIM_THRESHOLD else None


# 字段规格：(字段名, 从 SourceRecord 取值, 比对器)。claim 伪源用同名字典键取值。
_FIELD_SPECS: list[tuple[str, Any, Any]] = [
    ("title", lambda r: normalize_title(r.title), _cmp_title),
    (
        "first_author",
        lambda r: normalize_last_name(
            r.first_author_last_name or (r.authors[0] if r.authors else "")
        ),
        _cmp_name,
    ),
    ("year", lambda r: r.year, _cmp_year),
    ("doi", lambda r: normalize_doi(r.doi), _cmp_exact),
    ("journal", lambda r: normalize_title(r.journal), _cmp_journal),
]


def _claim_record(cite: dict[str, Any]) -> SourceRecord:
    """把「笔记声称值」包成一个 source='claim' 的伪 SourceRecord，并入成对比对。

    首作者姓优先从完整作者名派生（与各源一致），避免用笔记里可能已被上游
    删声调预处理的 ``first_author_last_name`` 字段而造成假冲突。
    """
    authors = cite.get("authors") or []
    if authors:
        a0 = authors[0]
        a0 = (
            a0
            if isinstance(a0, str)
            else (a0.get("name", "") if isinstance(a0, dict) else "")
        )
        first_last = normalize_last_name(a0)
    else:
        first_last = normalize_last_name(cite.get("first_author_last_name") or "")
    return SourceRecord(
        source="claim",
        reachable=True,
        found=True,
        title=cite.get("title") or "",
        authors=list(authors),
        first_author_last_name=first_last,
        year=extract_year(cite.get("year")),
        journal=cite.get("journal") or "",
        doi=normalize_doi(cite.get("doi") or ""),
        arxiv_id=cite.get("arxiv_id") or "",
    )


# ---------------------------------------------------------------------------
# 主入口：三源交叉核验
# ---------------------------------------------------------------------------
def _classify_citation(cite: dict[str, Any]) -> str:
    """判定引用主标识类型（决定各源如何解析）。"""
    if normalize_doi(cite.get("doi")):
        return "doi"
    if cite.get("arxiv_id"):
        return "arxiv"
    if cite.get("openalex_id"):
        return "openalex"
    if cite.get("title"):
        return "title"
    return "empty"


def verify_citation(
    cite: dict[str, Any],
    *,
    sources: Iterable[str] = SOURCES,
    use_cache: bool = True,
) -> CitationVerdict:
    """对单条引用做三源交叉核验，返回 :class:`CitationVerdict`。

    ``cite`` 支持键：doi / arxiv_id / openalex_id / title / first_author_last_name /
    authors / year / journal（均可缺省；至少需一个标识或标题）。
    """
    cite = dict(cite or {})
    kind = _classify_citation(cite)
    verdict = CitationVerdict(input=cite, kind=kind)
    if kind == "empty":
        verdict.status = NOT_FOUND
        verdict.reasons.append("引用为空（无 DOI / arXiv id / 标题），无从核验。")
        return verdict

    doi = normalize_doi(cite.get("doi"))
    arxiv_id = cite.get("arxiv_id") or ""
    openalex_id = cite.get("openalex_id") or ""
    title = cite.get("title") or ""
    first_author = cite.get("first_author_last_name") or ""

    want = set(sources)
    records: dict[str, SourceRecord] = {}
    if "openalex" in want:
        records["openalex"] = fetch_openalex(
            doi=doi or None,
            openalex_id=openalex_id or None,
            title=title or None,
            use_cache=use_cache,
        )
    if "crossref" in want:
        records["crossref"] = fetch_crossref(
            doi=doi or None,
            title=title or None,
            first_author=first_author or None,
            use_cache=use_cache,
        )
    if "arxiv" in want:
        # arXiv 仅在确有 arXiv id 或有标题可检索时才查（否则直接判未命中，省一次请求）
        if arxiv_id or title:
            records["arxiv"] = fetch_arxiv(
                arxiv_id=arxiv_id or None, title=title or None, use_cache=use_cache
            )
        else:
            records["arxiv"] = SourceRecord(source="arxiv", reachable=True, found=False)
    verdict.records = records

    found = {k: r for k, r in records.items() if r.found}
    unreachable = [k for k, r in records.items() if not r.reachable]

    # 三源全部查无此条 → NOT_FOUND（可疑但不阻断）
    if not found:
        verdict.status = NOT_FOUND
        if unreachable:
            verdict.reasons.append(
                f"三源均未确认该引用；其中不可达：{', '.join(unreachable)}（网络异常，非判定为错）。"
            )
        else:
            verdict.reasons.append(
                "三源（OpenAlex/Crossref/arXiv）均查无此条——疑似虚构或标识有误，请人工复核。"
            )
        return verdict

    # 成对比对（并入 claim 伪源）
    pool: dict[str, SourceRecord] = {"claim": _claim_record(cite)}
    pool.update(found)
    checks: list[FieldCheck] = []
    severities: list[str] = []
    for fname, getter, cmp in _FIELD_SPECS:
        values: dict[str, Any] = {}
        for sname, rec in pool.items():
            try:
                values[sname] = getter(rec)
            except Exception:
                values[sname] = None
        chk = FieldCheck(field=fname, values=values)
        names = list(values)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                sa, sb = names[i], names[j]
                sev = cmp(values[sa], values[sb])
                if sev:
                    chk.conflicts.append((sa, sb, sev))
                    severities.append(sev)
        checks.append(chk)
    verdict.checks = checks

    # 裁决
    hard = [c for c in checks if c.worst == "hard"]
    soft = [c for c in checks if c.worst == "soft"]
    if hard:
        verdict.status = FAIL
        for c in hard:
            verdict.reasons.append(_describe_conflict(c))
    elif soft:
        verdict.status = WARN
        for c in soft:
            verdict.reasons.append(_describe_conflict(c))
    else:
        verdict.status = PASS
        verdict.reasons.append(
            f"三源一致（{', '.join(sorted(found))} 相互印证，无字段冲突）。"
        )

    if unreachable:
        verdict.reasons.append(
            f"注：{', '.join(unreachable)} 不可达，已跳过（不影响判定）。"
        )
    return verdict


def _describe_conflict(chk: FieldCheck) -> str:
    """把字段冲突渲染成一句人类可读的诊断。"""
    pairs = []
    for sa, sb, sev in chk.conflicts:
        pairs.append(f"{sa}≠{sb}")
    vals = chk.values
    sample = " | ".join(f"{k}={vals[k]!r}" for k in list(vals)[:4])
    tag = "严重" if chk.worst == "hard" else "轻微"
    return f"{chk.field}（{tag}）：{'/'.join(pairs)} —— {sample}"


# ---------------------------------------------------------------------------
# 便捷封装：frontmatter / 笔记文件
# ---------------------------------------------------------------------------
def citation_from_frontmatter(fm: dict[str, Any]) -> dict[str, Any]:
    """从 paper-note frontmatter 抽出核验所需的引用字段。"""
    return {
        "doi": fm.get("doi") or "",
        "arxiv_id": fm.get("arxiv_id") or "",
        "openalex_id": fm.get("openalex_id") or "",
        "title": fm.get("title") or "",
        "first_author_last_name": fm.get("first_author_last_name") or "",
        "authors": fm.get("authors") or [],
        "year": fm.get("year"),
        "journal": fm.get("journal") or "",
    }


def verify_frontmatter(
    fm: dict[str, Any], *, sources: Iterable[str] = SOURCES, use_cache: bool = True
) -> CitationVerdict:
    """核验一份 paper-note frontmatter（供 research.py 写入前 gate 调用）。"""
    return verify_citation(
        citation_from_frontmatter(fm), sources=sources, use_cache=use_cache
    )


def _parse_frontmatter_scalars(text: str) -> dict[str, Any]:
    """极简 frontmatter 解析（仅标量；够核验用）。避免与 research.py 循环依赖。"""
    m = re.match(r"^---\s*\n(.*?)\n---", text or "", re.DOTALL)
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
        if v.startswith('"') and v.endswith('"') and len(v) >= 2:
            v = v[1:-1]
        elif v in ("null", "~"):
            v = None  # type: ignore[assignment]
        elif re.match(r"^[-+]?\d+$", v):
            v = int(v)  # type: ignore[assignment]
        fm[k.strip()] = v
    return fm


def verify_note_file(
    path: Any, *, sources: Iterable[str] = SOURCES, use_cache: bool = True
) -> CitationVerdict:
    """核验一个 papers/*.md 笔记文件的 frontmatter 引用。"""
    from pathlib import Path

    text = Path(path).read_text(encoding="utf-8")
    return verify_frontmatter(
        _parse_frontmatter_scalars(text), sources=sources, use_cache=use_cache
    )


# ---------------------------------------------------------------------------
# 渲染：人类可读报告（FAIL 标红）
# ---------------------------------------------------------------------------
_STATUS_MARK = {PASS: "✓", WARN: "△", FAIL: "✗", NOT_FOUND: "?"}
_RED, _YELLOW, _RESET = "\033[31m", "\033[33m", "\033[0m"


def render_verdict(v: CitationVerdict, *, color: bool = False) -> str:
    """渲染单条裁决为多行报告。``color=True`` 时 FAIL 用 ANSI 红、WARN 用黄。"""
    mark = _STATUS_MARK.get(v.status, "?")
    head = f"[{mark} {v.status}] {v.label}"
    if color and v.status == FAIL:
        head = f"{_RED}{head}{_RESET}"
    elif color and v.status == WARN:
        head = f"{_YELLOW}{head}{_RESET}"

    lines = [head]
    # 各源命中概览
    src_bits = []
    for name in SOURCES:
        r = v.records.get(name)
        if r is None:
            continue
        if not r.reachable:
            src_bits.append(f"{name}=不可达")
        elif r.found:
            src_bits.append(f"{name}=命中")
        else:
            src_bits.append(f"{name}=无")
    if src_bits:
        lines.append("    源: " + "  ".join(src_bits))
    for reason in v.reasons:
        lines.append(f"    - {reason}")
    return "\n".join(lines)


def render_report(verdicts: list[CitationVerdict], *, color: bool = False) -> str:
    """渲染一批裁决 + 末尾汇总（PASS/WARN/FAIL/NOT_FOUND 计数）。"""
    if not verdicts:
        return "（无引用可核验）"
    lines = [render_verdict(v, color=color) for v in verdicts]
    tally = {PASS: 0, WARN: 0, FAIL: 0, NOT_FOUND: 0}
    for v in verdicts:
        tally[v.status] = tally.get(v.status, 0) + 1
    lines.append("")
    lines.append(
        f"汇总：共 {len(verdicts)} 条 —— ✓PASS {tally[PASS]}  △WARN {tally[WARN]}  "
        f"✗FAIL {tally[FAIL]}  ?NOT_FOUND {tally[NOT_FOUND]}"
    )
    if tally[FAIL]:
        lines.append("存在 FAIL（三源实质冲突）：应修正引用后再写入。")
    return "\n".join(lines)
