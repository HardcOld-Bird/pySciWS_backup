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
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote

from .cache_manager import cache_key, read_cache, write_cache
from .config import http_session, settings
from .notes import load_frontmatter, normalize_last_name, strip_accents

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
#: 声调**转写**的唯一权威实现在 :mod:`.notes`（``Büttner`` → ``Buttner``，而非删除声调
#: 得到 ``Bttner``）；此处保留私有别名以免改动本模块内的既有调用方。
_strip_accents = strip_accents


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


# ``normalize_last_name`` 已从 :mod:`.notes` re-export（见文件顶部 import）。
# 保留本模块作为对外入口，使 ``citation_verify.normalize_last_name`` 这个历史路径
# 继续成立；全项目现在只有 notes.py 里那一份实现，不会再出现同一位作者
# 在不同源派生出不同姓氏的情况。


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
    """Crossref 响应的缓存路径（一行委托 :func:`cache_manager.cache_key`）。

    本函数过去是 ``openalex_client._cache_key`` 的**逐行副本**，只差文件名前缀：四行
    哈希/slug 逻辑写了两遍，任何一边改了另一边就默默漂移。保留这个具名函数而不直接
    内联调用，是因为「crossref 的缓存路径」是个有用的概念名（且现有测试就盯着它）。
    """
    return cache_key(url, params, prefix="crossref")


def _crossref_message(
    path: str, params: dict[str, Any] | None = None, *, use_cache: bool = True
) -> dict[str, Any] | None:
    """GET Crossref，返回 ``message`` 对象；404/未命中返回 None。

    网络/HTTP 异常向上抛，由 :func:`fetch_crossref` 归类为 ``reachable=False``。
    """
    url = f"{CROSSREF_BASE}/{path}"
    params = dict(params or {})
    if settings.openalex_email and "mailto" not in params:
        params["mailto"] = settings.openalex_email

    cache_path = _crossref_cache_path(url, params)
    if use_cache:
        cached = read_cache(cache_path)
        if cached is not None:
            return cached.get("message") if isinstance(cached, dict) else None

    with http_session() as s:
        r = s.get(url, params=params, timeout=settings.http_timeout)
        if r.status_code == 404:
            return None  # Crossref 明确查无此条（reachable, not found），区别于网络异常
        r.raise_for_status()
        data = r.json()

    write_cache(cache_path, data)
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


#: 笔记 frontmatter 的解析统一到 :mod:`.notes`（基于 ``yaml.safe_load``）。
#:
#: 旧的手写「仅标量」解析器会跳过所有以 ``-`` 开头的行，导致 block-style 的
#: ``authors`` / ``topics`` 全部丢失。对核验而言影响有限（``first_author_last_name``
#: 仍在），但它是同一个缺陷在 ``cache_manager.referenced_paths()`` 里造成
#: ``prune --keep-referenced`` 保护失效的根源，因此一并消除。
#: 保留旧名作为别名，既有调用方与测试无需改动。
_parse_frontmatter_scalars = load_frontmatter


def verify_note_file(
    path: Any, *, sources: Iterable[str] = SOURCES, use_cache: bool = True
) -> CitationVerdict:
    """核验一个 papers/*.md 笔记文件的 frontmatter 引用。"""
    from pathlib import Path

    text = Path(path).read_text(encoding="utf-8")
    return verify_frontmatter(
        load_frontmatter(text), sources=sources, use_cache=use_cache
    )


# ---------------------------------------------------------------------------
# 参考文献解析：BibTeX 与 markdown 参考文献段（``citecheck --bib``）
# ---------------------------------------------------------------------------
#: BibTeX 条目头 ``@article{key,``。``@string`` / ``@comment`` / ``@preamble`` 由
#: :data:`_BIB_SKIP_TYPES` 排除——它们定义宏而非文献，混进核验列表只会产出一堆
#: NOT_FOUND 噪声。
_BIB_ENTRY_HEAD_RE = re.compile(r"@(\w+)\s*\{\s*([^,\s]*)\s*,")
_BIB_SKIP_TYPES = frozenset({"string", "comment", "preamble", "set"})

#: DOI 与 arXiv 编号在**自由文本**里的形态（参考文献段不像 BibTeX 有字段名可依靠）。
_DOI_IN_TEXT_RE = re.compile(r"\b10\.\d{4,9}/[^\s\]\)>,;\"'\u3001\uff0c\u3002\uff1b]+")
_ARXIV_IN_TEXT_RE = re.compile(
    r"(?:arxiv\s*[:\uff1a]?\s*)?(\d{4}\.\d{4,5})(v\d+)?", re.I
)
#: 显式带 ``arXiv:`` 前缀的编号（带 DOI 的行里只认这一种）。
_ARXIV_PREFIXED_RE = re.compile(r"arxiv\s*[:\uff1a#]\s*(\d{4}\.\d{4,5})(?:v\d+)?", re.I)
#: 裸编号。前后都加了边界断言，避免从 ``1234.56789`` 这样的长数字串里切出假编号。
_ARXIV_BARE_RE = re.compile(r"(?<![\d.])(\d{4}\.\d{4,5})(?:v\d+)?(?!\d)")

#: 参考文献段的标题（中英、带不带编号都认）。
_REFS_HEADING_RE = re.compile(
    r"^(?P<hashes>#{1,6})[ \t]*"
    r"(?:\d+(?:[\.\u3001\)])?[ \t]*)?"
    r"(?:\u53c2\u8003\u6587\u732e|\u53c2\u8003\u8d44\u6599|\u5f15\u6587|References?|Bibliography)"
    r"[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)

#: TeX 重音原语（``\"`` ``\'`` ``\^`` ``\~`` ``\=`` ``\.`` ``\u`` ``\v`` ``\H`` ``\t``
#: ``\c`` ``\d`` ``\b``）。它们后面紧跟的**那一个字母**就是基字母，重音本身丢弃。
#:
#: 三种写法都得认，而**顺序**（长形态在前）是关键：``{\"u}`` 是 Zotero/BibTeX 导出的
#: 主流形态，``\"{u}`` 与 ``\"u`` 也出现。若只写一条「命令 + 可选花括号 + 字母 + 可选
#: 花括号」的松散正则，``B{\"u}ttner`` 会匹配到 ``\"u}`` ——即把**闭合花括号**当成重音
#: 自己的可选右括号吃掉，留下一个孤立的 ``{`` 在后续步骤里变成空格，得到 ``B uttner``。
#: 那个空格随后会被姓氏归一化折叠掉，于是 BibTeX 的 ``Buttner`` 与 OpenAlex 的
#: ``Büttner`` 对不上，每条含分音符作者的引用都报一个假 WARN。
#:
#: 裸字母形态只给**符号型**重音（``\` \' \^ \" \~ \= \.``），字母型（``\u \v \H \t \c \d \b``）
#: 只认带花括号的形态：``\bibitem`` / ``\usepackage`` 的前两个字符会被误当成 ``\b i`` /
#: ``\u s`` 而把命令名劈开，而 ``\c{c}`` ``\u{a}`` 正是字母型重音的常见写法。
#: 两个裸形态都**不允许空白**（``\"`` 与 ``h`` 之间不得有空格）：TeX 里控制符号后的
#: 空白是真实排版空格而非终止符，所以 ``\" loudly`` 里的 ``\"`` 只是个转义引号（第 4 步
#: 会把它变回 ``"``）。允许空白会把它误读成「l 上的分音符」，连同空格一起吃掉，
#: 得到 ``helloloudly`` 这种把两个词糊在一起的结果。
#: 三个分支合在一条正则里而不是拆成两个常量：两份重叠的重音正则很容易只改其中
#: 一份，而那就是先前 ``B{\"u}ttner`` → ``B uttner`` 那个 bug 的形状。
_LATEX_ACCENT_RE = re.compile(
    r"\{\s*\\[`'^\"~=.uvHtcdb]\s*([a-zA-Z])\s*\}"  # {\"u} —— 花括号在外
    r"|\\[`'^\"~=.uvHtcdb]\s*\{\s*([a-zA-Z])\s*\}"  # \"{u} —— 花括号在内
    r"|\\[`'^\"~=.]([a-zA-Z])"  # \"u   —— 裸字母（仅符号型重音）
)


def _accent_repl(m: re.Match[str]) -> str:
    """三个分支各有一个捕获组，命中哪个就取哪个（都未命中说明基字母缺失）。"""
    return m.group(1) or m.group(2) or m.group(3) or ""


def _strip_latex(s: str) -> str:
    """剥离 BibTeX 值里的 LaTeX 标记，得到可与三源比对的纯文本。

    重音命令按**转写**处理（``\\"o`` → ``o``、``{\\'e}`` → ``e``），与
    :func:`notes.normalize_last_name` 的语义一致——于是 BibTeX 里的 ``B{\\"u}ttner``
    与 OpenAlex 里的 ``Büttner`` 归一化后能对上，不会因排版差异产生假冲突。

    best-effort：本项目只需处理 ``compose refs.bib`` 从 Zotero 产出的规整格式
    （方案假设 1：不引入 ``bibtexparser``），不追求覆盖任意 TeX 宏包。
    """
    s = str(s or "")
    # 1) TeX 重音原语 → 基字母。**必须排在命名命令解包之前**：字母型重音
    #    （``\c`` ``\u`` ``\v`` ``\b`` ``\d`` ``\t`` ``\H``）同时也是合法的「命令名 + 分组」，
    #    先跑解包会把 ``\c{c}ervenka`` 变成 `` c ernenka``——凭空多出的那个空格使它与
    #    OpenAlex 的 ``Červenka``（归一化后 ``cervenka``）对不上。先转写重音则得到
    #    ``cervenka``；而 ``\textbf`` / ``\title`` 这类真命名命令不会被重音正则误伤
    #    （它们的第二个字符不是 ``{``）。
    s = _LATEX_ACCENT_RE.sub(_accent_repl, s)
    # 2) 命名命令的分组解包：``\textbf{X}`` → ``X``（留内容、丢命令）
    s = re.sub(r"\\[a-zA-Z]+\s*\{([^{}]*)\}", r" \1 ", s)
    # 3) 剩下的单/双字母命令（``\oe`` ``\ss`` ``\i`` ``\l`` ``\o`` ``\aa``）→ 去反斜杠。
    #    连同其后的空白一起吃掉：TeX 里控制词的结尾空白是**终止符而非排版空格**，
    #    ``\oe uvre`` 排版出来就是 ``oeuvre``。留着那个空格会让它永远对不上
    #    OpenAlex 的 ``œuvre``（归一化后 ``oeuvre``），产生一条假 WARN。
    s = re.sub(r"\\([a-zA-Z]{1,2})\s*", r"\1", s)
    # 4) 转义的标点：``\&`` ``\$`` ``\%`` ``\_`` ``\#`` → 字面量
    s = re.sub(r"\\([^a-zA-Z\s])", r"\1", s)
    # 5) 分组花括号与数学模式定界符
    s = s.replace("{", " ").replace("}", " ").replace("$", " ")
    # 6) BibTeX 的不换行空格与页码连字符习惯
    s = s.replace("~", " ").replace("--", "-")
    return re.sub(r"\s+", " ", s).strip()


def _bibtex_tokens(
    text: str, start: int = 0, depth: int = 0
) -> Iterator[tuple[int, str, str, int]]:
    """逐字符扫描 BibTeX 文本，产出 ``(索引, 字符, 类型, 更新后深度)``。

    类型 ∈ ``open`` / ``close`` / ``comma`` / ``quote`` / ``text``；只有前四种是
    **结构性**的，``text`` 一律当字面量看待。:func:`_scan_bibtex_entry` 与
    :func:`_split_bibtex_fields` 共用本分词器，因为它们需要的是**同一套**「哪个字符
    算数」的判据——两份各自手写的状态机正是本模块先前那个 bug 的根源。

    三条规则：

    1. **反斜杠转义下一个字符**。``author = {B{\\"u}ttner, Ralph}`` 里那个 ``"`` 是重音
       命令的参数，不是字符串定界符。不跳过它会让状态机进入 in_str，把随后的闭合
       ``}`` 当普通字符吞掉，深度计数从此崩坏——该条目一路吞到文件末尾，把后面
       所有条目全都吃进一个 ``author`` 字段里。而 ``\\"u`` 是 Zotero 导出里最常见的
       分音符编码，也就是说：只要 .bib 里有一位带分音符的作者，它**之后**的全部内容
       都会解析崩坏。
    2. **``"`` 只在深度 0 处才是定界符**。花括号包起来的值里，``"`` 只是字面量
       （``title = {The "Best" Paper}``）；当成定界符会让奇数个引号的值把后续 ``}``
       全部吞掉。
    3. ``{`` / ``}`` 只在字符串外才改变深度（引号值里的花括号不参与字段切分）。
    """
    i, n = start, len(text)
    in_str = False
    while i < n:
        ch = text[i]
        if ch == "\\" and i + 1 < n:
            yield i, ch, "text", depth
            yield i + 1, text[i + 1], "text", depth
            i += 2
            continue
        if in_str:
            if ch == '"':
                in_str = False
                yield i, ch, "quote", depth
            else:
                yield i, ch, "text", depth
            i += 1
            continue
        if ch == '"' and depth == 0:
            in_str = True
            yield i, ch, "quote", depth
        elif ch == "{":
            depth += 1
            yield i, ch, "open", depth
        elif ch == "}":
            depth -= 1
            yield i, ch, "close", depth
        elif ch == "," and depth == 0:
            yield i, ch, "comma", depth
        else:
            yield i, ch, "text", depth
        i += 1


def _scan_bibtex_entry(text: str, start: int) -> tuple[str, int]:
    """从 ``@type{key,`` 之后扫到配对的 ``}``，返回（字段区文本, 结束位置）。

    用**深度计数**而不是「找下一个 ``}``」：BibTeX 标题字段里常含成对花括号
    （``title = {The {C--H} bond}``），按第一个 ``}`` 截断会把字段区切坏。未闭合时
    尽力而为取到文末（残缺的 .bib 仍应能核对其前面的条目）。

    ``start`` 已在条目的 ``{`` **之后**，因此条目自己的闭合 ``}`` 会把相对深度推到
    ``-1``；值字段的内层 ``}`` 只回到 ``0``，据此区分两者。
    """
    for i, _ch, kind, depth in _bibtex_tokens(text, start):
        if kind == "close" and depth < 0:
            return text[start:i], i + 1
    return text[start:], len(text)


def _split_bibtex_fields(body: str) -> list[str]:
    """按**顶层**逗号切分字段区（花括号内与引号内的逗号不算分隔符）。

    作者字段里满是逗号（``B{\\"u}ttner, Ralph and ...``），直接 ``split(",")`` 会把一个
    字段拆成好几块。
    """
    parts: list[str] = []
    buf: list[str] = []
    for _i, ch, kind, _depth in _bibtex_tokens(body):
        if kind == "comma":
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    if "".join(buf).strip():
        parts.append("".join(buf))
    return [p for p in parts if p.strip()]


def _strip_bibtex_value(raw: str) -> str:
    """去掉值外层的 ``{}`` / ``""``，再剥离 LaTeX 标记。"""
    v = str(raw or "").strip()
    if len(v) >= 2 and v[0] == "{" and v[-1] == "}":
        v = v[1:-1]
    elif len(v) >= 2 and v[0] == '"' and v[-1] == '"':
        v = v[1:-1]
    return _strip_latex(v)


def _split_bibtex_authors(value: str) -> list[str]:
    """按 BibTeX 的 `` and `` 分隔符切分作者列表。"""
    return [
        a.strip()
        for a in re.split(r"\s+and\s+", str(value or ""), flags=re.I)
        if a.strip()
    ]


def parse_bibtex(text: str) -> list[dict[str, Any]]:
    """解析 ``.bib`` 文本，返回每条记录一个 dict。

    输出形状：``{"key": <引用键>, "entry_type": <article|inproceedings|...>, <字段名小写>: <值>}``。
    值已经去引号并剥离 LaTeX 标记（见 :func:`_strip_latex`），因此可直接拿去与三源比对。

    不用 ``bibtexparser`` 是方案的明确取舍（假设 1：不引入新依赖）；代价是只支持
    ``@type{key, ...}`` 形态，**不支持** ``@type(key, ...)`` 的旧式括号。Zotero 与
    本项目 ``compose refs.bib`` 产出的都是前者。

    畸形输入（未闭合、缺 ``=``）一律跳过该字段/条目而不抛异常：一份 .bib 里有一条坏记录
    不应该让其余几百条都核不了。
    """
    text = str(text or "")
    entries: list[dict[str, Any]] = []
    pos = 0
    while True:
        m = _BIB_ENTRY_HEAD_RE.search(text, pos)
        if m is None:
            break
        etype = m.group(1).lower()
        key = m.group(2).strip()
        body, pos = _scan_bibtex_entry(text, m.end())
        if etype in _BIB_SKIP_TYPES:
            continue
        entry: dict[str, Any] = {"key": key, "entry_type": etype}
        for part in _split_bibtex_fields(body):
            if "=" not in part:
                continue
            name, _, raw = part.partition("=")
            name = name.strip().lower()
            if not name:
                continue
            entry[name] = _strip_bibtex_value(raw)
        entries.append(entry)
    return entries


def citation_from_bibtex(entry: dict[str, Any]) -> dict[str, Any]:
    """把 :func:`parse_bibtex` 的一条记录映射为 :func:`verify_citation` 的 ``cite`` dict。

    只写**确实有值**的键：所有比对器（``_cmp_*``）对空值一律返回 ``None``（不计冲突），
    因此「BibTeX 没写这个字段」与「BibTeX 声称它为空」在行为上无区别——但前者语义上
    更诚实，也避免 JSON 日志里充满无信息量的空字符串。

    期刊字段兼容三种命名：``journal``（传统 BibTeX）、``journaltitle``（biblatex）、
    ``booktitle``（会议论文的 venue）。不兼容 ``booktitle`` 会让所有 ``@inproceedings``
    的期刊比对直接缺失。
    """
    entry = dict(entry or {})
    cite: dict[str, Any] = {}

    title = str(entry.get("title") or "").strip()
    if title:
        cite["title"] = title

    authors = _split_bibtex_authors(entry.get("author") or "")
    if authors:
        cite["authors"] = authors
        last = normalize_last_name(authors[0])
        if last:
            cite["first_author_last_name"] = last

    year = extract_year(entry.get("year") or entry.get("date"))
    if year:
        cite["year"] = year

    journal = ""
    for k in ("journal", "journaltitle", "booktitle"):
        if str(entry.get(k) or "").strip():
            journal = str(entry[k]).strip()
            break
    if journal:
        cite["journal"] = journal

    doi = normalize_doi(entry.get("doi"))
    if doi:
        cite["doi"] = doi

    # ``eprint`` 只在确实指向 arXiv 时才当 arXiv id 用：它也可能是 Zenodo / HAL 编号，
    # 误当 arXiv id 会让 arXiv 源报一个假 NOT_FOUND。
    eprint = str(entry.get("eprint") or "").strip()
    prefix = str(entry.get("archiveprefix") or entry.get("eprinttype") or "").lower()
    if eprint and ("arxiv" in prefix or _ARXIV_IN_TEXT_RE.fullmatch(eprint)):
        cite["arxiv_id"] = eprint
    return cite


def _references_line_range(lines: list[str]) -> tuple[int, int]:
    """定位参考文献段，返回半开区间 ``[start, end)`` 的 0-based 行号；找不到返回 ``(-1, -1)``。

    段落边界：从匹配 :data:`_REFS_HEADING_RE` 的标题起，到**同级或更高级**的下一个标题
    （或文末）止——``## References`` 不会吃掉后面 ``## Appendix`` 的内容。

    按行号而不是按子串切分，是因为调用方需要给每条引用报准确的原文行号（``_line``），
    而「先切子串再 ``text.find`` 反推偏移」在段落内容重复出现时会算错，且是 O(n²)。
    """
    start, level = -1, 0
    for i, raw in enumerate(lines):
        line = raw.rstrip()
        m = _REFS_HEADING_RE.match(line)
        if m:
            start, level = i + 1, len(m.group("hashes"))
            continue
        if start >= 0 and re.match(rf"^#{{1,{level}}}[ \t]+\S", line):
            return start, i
    return (start, len(lines)) if start >= 0 else (-1, -1)


def _clean_doi(raw: str) -> str:
    """去掉 DOI 末尾误吞的句读（``...124501.`` 的那个句点不属于 DOI）。"""
    return str(raw or "").strip().rstrip(".,;:)]}>\"'\u3002\uff0c\uff1b\u3001")


def _title_from_reference_line(line: str) -> str:
    """从一行参考文献里尽力提取标题（候选中取最长的一段）。

    best-effort：参考文献的排版千差万别（APS / IEEE / Nature 各一套）且无可靠语法。
    策略是**剔除**已知噪声（序号标记、URL、DOI、arXiv 编号、括号年份）后，在剩下的
    片段里取最长的那段——标题几乎总是最长的一段。候选还需过两道门：≥ 3 个词且
    字母占比 ≥ 60%，否则返回空串——这能挡下 ``Phys. Rev. Lett. 121, 124501 (2018).``
    这种**无标题**格式里的期刊碎片，避免把碎片当标题去核（那会产生一堆假 NOT_FOUND）。
    """
    s = str(line or "")
    s = re.sub(r"^\s*(?:[-*\u2022]|\[\d+\]|\d+[.)\u3001]|\(\d+\))\s*", "", s)
    s = re.sub(r"https?://\S+", " ", s)
    s = _DOI_IN_TEXT_RE.sub(" ", s)
    s = re.sub(r"arxiv\s*[:\uff1a]?\s*\d{4}\.\d{4,5}(?:v\d+)?", " ", s, flags=re.I)
    s = re.sub(r"[\(\uff08\[]\s*(?:19|20)\d{2}[a-z]?\s*[\)\uff09\]]", " ", s)
    s = re.sub(r"^\s*(?:[A-Z][a-zA-Z'\-.]+(?:,?\s+[A-Z]\.?)+)\s*[,;.]\s*", "", s)

    chunks = [
        c.strip(" \t\"'“”‘’")
        for c in re.split(r"[.,;:|\u3002\uff0c\uff1b\uff1a\u3001]+", s)
    ]
    best, best_len = "", 0
    for c in chunks:
        if len(c) < 20 or len(c.split()) < 3:
            continue
        letters = sum(ch.isalpha() or ord(ch) > 0x2E80 for ch in c)
        if letters / max(1, len(c)) < 0.6:
            continue
        if len(c) > best_len:
            best, best_len = c, len(c)
    return best


def parse_markdown_references(text: str) -> list[dict[str, Any]]:
    """从 markdown 文本里提取参考文献，返回 :func:`verify_citation` 可直接吃的 ``cite`` 列表。

    两档强度（故意不对称）：

    - **参考文献段内**（标题匹配 ``参考文献`` / ``References`` / ``Bibliography``）：
      接受带 DOI / arXiv 编号的行，**也**接受只带可辨识标题的行；
    - **段外**：只接受带硬标识（DOI / 显式 ``arXiv:`` 前缀）的行。否则正文里一句提到
      某篇论文的话会被当成一条待核引用，而那句散文根本提不出可靠标题。

    arXiv 编号同样分两档：行里已有 DOI 时只认**显式前缀**的 ``arXiv:NNNN.NNNNN``。裸的
    四位点五位数字在带 DOI 的行里更可能是 DOI 尾巴或页码，误当成 arXiv id 会让 arXiv
    源报一个假 NOT_FOUND，甚至把本该 PASS 的引用拖成 WARN。

    无标识且无标题的行一律跳过（方案明确要求的 best-effort 语义）。每条额外携带
    ``_line``（原文行号，从 1 起）与 ``_raw``（原行），供报告里定位「哪一行错了」。
    这两个键以 ``_`` 开头：``verify_citation`` 只按名取键，多余键不影响核验。
    """
    lines = str(text or "").splitlines()
    start, end = _references_line_range(lines)
    out: list[dict[str, Any]] = []
    for idx, line in enumerate(lines):
        if not line.strip():
            continue
        in_refs = start >= 0 and start <= idx < end

        dois: list[str] = []
        for m in _DOI_IN_TEXT_RE.finditer(line):
            d = normalize_doi(_clean_doi(m.group(0)))
            if d and d not in dois:
                dois.append(d)
        m_arx = _ARXIV_PREFIXED_RE.search(line)
        if m_arx is None and not dois:
            m_arx = _ARXIV_BARE_RE.search(line)
        arxiv_id = m_arx.group(1) if m_arx else ""
        title = _title_from_reference_line(line) if in_refs else ""
        if not dois and not arxiv_id and not title:
            continue

        cite: dict[str, Any] = {}
        if dois:
            cite["doi"] = dois[0]
        if arxiv_id:
            cite["arxiv_id"] = arxiv_id
        if title:
            cite["title"] = title
        m_year = re.search(r"\b(?:19|20)\d{2}\b", line)
        if m_year:
            cite["year"] = int(m_year.group(0))
        cite["_line"] = idx + 1
        cite["_raw"] = line.strip()
        out.append(cite)
    return out


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
