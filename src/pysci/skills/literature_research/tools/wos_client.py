"""Web of Science (Clarivate) Starter API 客户端。

状态：**骨架，待 Clarivate 批准后启用**。

申请进度：
- Application ID: pysciws-wos-lit-review
- Application Name: pySciWS Physics Literature Research Assistant
- Client Type: Confidential
- OAuth2 flows: Client Credentials only

WoS Starter API 提供 OpenAlex 无法覆盖的独家能力：
1. **官方 JIF**（Journal Impact Factor）与 **JCR 分区**（Q1-Q4）
2. **ESI Highly Cited Papers / Hot Papers** 标签（前 1% / 前 0.1%）
3. **严格的期刊收录过滤**（自动排除掠夺性期刊）
4. **官方 Citation Report**（h-index、总被引等）

官方文档：
- 门户: https://developer.clarivate.com/
- Starter API: https://developer.clarivate.com/apis/wos-starter
- OAuth2: https://developer.clarivate.com/docs/oauth

WoS Starter API 支持两种鉴权方式：
- **API Key**：直接在 header 中传 `X-ApiKey: <key>`（简单，适用于个人研究工具）
- **OAuth2 Client Credentials**：先用 key+secret 换 access_token（适用于生产环境）

本模块默认走 API Key 模式；若未来需要 OAuth2，:class:`WOSClient` 会自动升级。

配额（Starter API）：
- 25,000 records/year（学术申请）
- 每秒 1 请求（软限速）
"""

from __future__ import annotations

import json
import time
from typing import Any

from .config import http_session, settings

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
WOS_BASE = settings.wos_api_base_url.rstrip("/")

# 常用端点
EP_SEARCH = "/documents"                # 检索文献
EP_CITATION_REPORT = "/citation-report" # 引用报告
EP_JIF = "/journals"                    # 期刊指标（若 Starter API 提供）
EP_ESI = "/esi-highly-cited"            # ESI 高被引（若 Starter API 提供）

# 令牌缓存
_TOKEN_CACHE: dict[str, Any] = {"access_token": None, "expires_at": 0.0}


# ---------------------------------------------------------------------------
# 异常
# ---------------------------------------------------------------------------
class WOSNotConfigured(RuntimeError):
    """WoS 凭据未配置时抛出，附带申请指引。"""

    def __init__(self) -> None:
        super().__init__(
            "WoS Starter API credentials are not configured.\n"
            "Please:\n"
            "  1. Wait for Clarivate approval (Application ID: pysciws-wos-lit-review)\n"
            "  2. Copy WOS_API_KEY into project-root .env\n"
            "     (Starter API only requires the key; WOS_API_SECRET is optional)\n"
            "  3. Retry the request"
        )


class WOSAPIError(RuntimeError):
    """API 返回错误状态码时抛出。"""


# ---------------------------------------------------------------------------
# OAuth2 Client Credentials
# ---------------------------------------------------------------------------
def _get_access_token() -> str:
    """获取（或复用）OAuth2 access_token。

    Clarivate OAuth2 token 端点：https://gateway.webofknowledge.com/gateway/Gateway.cgi?GWVersion=2&SrcApp=...
    或 https://gateway.clarivate.com/uaa/token  （较新）

    本实现按较新的 gateway.clarivate.com/uaa/token 端点编写；若批准邮件指定不同 URL，
    请在 .env 中新增 WOS_TOKEN_URL 并修改本函数。
    """
    if not settings.wos_ready:
        raise WOSNotConfigured()

    now = time.time()
    if _TOKEN_CACHE["access_token"] and _TOKEN_CACHE["expires_at"] > now + 30:
        return _TOKEN_CACHE["access_token"]

    token_url = "https://gateway.clarivate.com/uaa/token"
    with http_session() as s:
        r = s.post(
            token_url,
            data={"grant_type": "client_credentials"},
            auth=(settings.wos_api_key, settings.wos_api_secret),
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=settings.http_timeout,
        )
        if r.status_code != 200:
            raise WOSAPIError(f"OAuth2 token request failed ({r.status_code}): {r.text[:500]}")
        data = r.json()

    token = data.get("access_token")
    expires_in = int(data.get("expires_in", 3600))
    if not token:
        raise WOSAPIError(f"OAuth2 token response missing access_token: {data}")

    _TOKEN_CACHE["access_token"] = token
    _TOKEN_CACHE["expires_at"] = now + expires_in
    return token


# ---------------------------------------------------------------------------
# 主客户端
# ---------------------------------------------------------------------------
class WOSClient:
    """WoS Starter API 客户端。

    用法::

        client = WOSClient()
        results = client.search("exceptional point acoustic", limit=10)
        for w in results["hits"]:
            print(w["title"], w["ut"])

        jif = client.get_journal_metrics("0031-9007")   # PRL
    """

    def __init__(self, mode: str = "apikey") -> None:
        """
        mode:
            - 'apikey': 直接用 X-ApiKey header（**Starter API 推荐**，只需 key）
            - 'oauth2': 走 Client Credentials flow（需要 key + secret，完整版 API 才需要）
        """
        if not settings.wos_api_key:
            raise WOSNotConfigured()
        self.mode = mode.lower()
        if self.mode == "oauth2" and not settings.wos_api_secret:
            raise WOSNotConfigured()
        if self.mode not in ("apikey", "oauth2"):
            raise ValueError(f"Unknown mode: {mode}")

    def _headers(self) -> dict[str, str]:
        if self.mode == "apikey":
            return {"X-ApiKey": settings.wos_api_key or ""}
        token = _get_access_token()
        return {"Authorization": f"Bearer {token}"}

    def _get(self, endpoint: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        url = f"{WOS_BASE}{endpoint}"
        with http_session() as s:
            r = s.get(url, params=params or {}, headers=self._headers(), timeout=settings.http_timeout)
            if r.status_code == 401:
                # 令牌过期或无效，重试一次
                _TOKEN_CACHE["access_token"] = None
                r = s.get(url, params=params or {}, headers=self._headers(), timeout=settings.http_timeout)
            if r.status_code != 200:
                raise WOSAPIError(f"WoS API {endpoint} failed ({r.status_code}): {r.text[:500]}")
            return r.json()

    # -----------------------------------------------------------------------
    # 检索
    # -----------------------------------------------------------------------
    def search(
        self,
        query: str,
        *,
        limit: int = 25,
        offset: int = 0,
        year_from: int | None = None,
        year_to: int | None = None,
        doc_type: str = "Article",
        sort: str = "relevance",
    ) -> dict[str, Any]:
        """按 query 检索 WoS Core Collection。

        query 语法示例（WoS 高级检索语法）::

            "exceptional point" AND acoustic
            TS=(non-hermitian AND metasurface) AND PY=(2024-2026)
            AU=Zhang AND TI=phononic

        字段代码：TS=主题, TI=标题, AU=作者, SO=来源出版物, PY=出版年, DO=DOI
        """
        # 追加年份过滤
        q = query
        if year_from or year_to:
            yr_parts = []
            if year_from and year_to:
                yr_parts.append(f"PY={year_from}-{year_to}")
            elif year_from:
                yr_parts.append(f"PY>={year_from}")
            elif year_to:
                yr_parts.append(f"PY<={year_to}")
            if yr_parts and "PY=" not in q:
                q = f"({q}) AND ({' AND '.join(yr_parts)})"

        params = {
            "q": q,
            "limit": min(limit, 100),
            "offset": offset,
            "sort": sort,
        }
        if doc_type:
            params["docType"] = doc_type
        data = self._get(EP_SEARCH, params=params)
        return {
            "total": data.get("totalRecords", data.get("total", 0)),
            "hits": [_normalize_hit(h) for h in (data.get("hits") or data.get("records") or [])],
            "_raw": data,
        }

    def get_document(self, ut: str | None = None, doi: str | None = None) -> dict[str, Any] | None:
        """按 UT (Web of Science accession number) 或 DOI 获取单篇详情。"""
        if ut:
            return _normalize_hit(self._get(f"/documents/{ut}"))
        if doi:
            r = self.search(f"DO={doi}", limit=1)
            return r["hits"][0] if r["hits"] else None
        raise ValueError("Must provide ut or doi")

    # -----------------------------------------------------------------------
    # 期刊指标（JIF / JCR）
    # -----------------------------------------------------------------------
    def get_journal_metrics(self, issn: str | None = None, name: str | None = None, year: int | None = None) -> dict[str, Any] | None:
        """获取期刊的官方 JIF、JCR 分区、5-year JIF 等。

        Starter API 的期刊端点在不同账户下开放程度不同；本函数按公开文档编写，
        若批准后端点路径不同，请修改 EP_JIF 常量。
        """
        params: dict[str, Any] = {}
        if issn:
            params["issn"] = issn
        elif name:
            params["journalName"] = name
        else:
            raise ValueError("Must provide issn or name")
        if year:
            params["year"] = year

        try:
            data = self._get(EP_JIF, params=params)
        except WOSAPIError as e:
            print(f"[wos] get_journal_metrics failed: {e}")
            return None

        return {
            "issn": data.get("issn", issn),
            "journal_name": data.get("journalName") or data.get("title") or name,
            "jif": data.get("impactFactor") or data.get("jif"),
            "jif_5yr": data.get("fiveYearImpactFactor") or data.get("jif5yr"),
            "jcr_quartile": data.get("quartile") or data.get("jcrQuartile"),
            "jcr_category": data.get("category"),
            "eigenfactor": data.get("eigenfactor"),
            "normalized_eigenfactor": data.get("normalizedEigenfactor"),
            "total_cites": data.get("totalCites"),
            "year": data.get("year"),
            "_raw": data,
        }

    # -----------------------------------------------------------------------
    # ESI 高被引 / 热点论文
    # -----------------------------------------------------------------------
    def is_esi_highly_cited(self, ut: str) -> bool:
        """检查一篇论文是否为 ESI 高被引论文（前 1%）。

        Starter API 通常不直接提供该字段；此处按启发式实现：
        通过 ESI 端点查询，若不可用则退化为 False。
        """
        try:
            data = self._get(f"{EP_ESI}/{ut}")
            return bool(data.get("isHighlyCited", False))
        except WOSAPIError:
            return False

    def is_esi_hot_paper(self, ut: str) -> bool:
        """检查是否为 ESI 热点论文（前 0.1%，近两年）。"""
        try:
            data = self._get(f"{EP_ESI}/{ut}")
            return bool(data.get("isHotPaper", False))
        except WOSAPIError:
            return False


# ---------------------------------------------------------------------------
# 数据规范化
# ---------------------------------------------------------------------------
def _normalize_hit(h: dict[str, Any]) -> dict[str, Any]:
    """把 WoS 原始记录规范化为本项目统一格式。"""
    authors = h.get("authors") or h.get("authorNames") or []
    if isinstance(authors, str):
        authors = [a.strip() for a in authors.split(";")]
    elif authors and isinstance(authors[0], dict):
        authors = [a.get("displayName") or a.get("name") or "" for a in authors]

    return {
        "ut": h.get("ut") or h.get("accessionNumber") or "",
        "doi": h.get("doi", ""),
        "title": h.get("title", ""),
        "authors": authors,
        "source": h.get("source") or h.get("sourceTitle") or "",
        "published_year": h.get("publishedYear") or h.get("py"),
        "volume": h.get("volume", ""),
        "issue": h.get("issue", ""),
        "pages": h.get("pages", ""),
        "cited_by_count": h.get("timesCited") or h.get("citedByCount") or 0,
        "doc_type": h.get("docType", ""),
        "issn": h.get("issn", ""),
        "eissn": h.get("eissn", ""),
        "abstract": h.get("abstract", ""),
        "keywords": h.get("keywords") or [],
        "esi_highly_cited": h.get("isHighlyCited", False),
        "esi_hot_paper": h.get("isHotPaper", False),
        "jif": h.get("impactFactor"),
        "jcr_quartile": h.get("quartile"),
        "_raw": h,
    }


# ---------------------------------------------------------------------------
# 与 openalex_client 的对接：为 OpenAlex 结果补充 WoS 独家字段
# ---------------------------------------------------------------------------
def enrich_openalex_work(w: dict[str, Any]) -> dict[str, Any]:
    """对 openalex_client.get_work() 返回的 dict 补充 WoS 独家字段。

    补充字段：wos_id, jif (官方), jif_5yr, jcr_quartile, esi_highly_cited, esi_hot_paper

    若 WoS 未配置或查询失败，静默返回原 dict（不阻塞主流程）。
    """
    if not settings.wos_ready:
        return w
    try:
        client = WOSClient()
        doc = None
        if w.get("doi"):
            doc = client.get_document(doi=w["doi"])
        if not doc:
            return w
        enriched = dict(w)
        enriched["wos_id"] = doc["ut"]
        if doc.get("jif") is not None:
            enriched["jif"] = doc["jif"]
        if doc.get("jcr_quartile"):
            enriched["jcr_quartile"] = doc["jcr_quartile"]
        enriched["esi_highly_cited"] = bool(doc.get("esi_highly_cited"))
        enriched["esi_hot_paper"] = bool(doc.get("esi_hot_paper"))
        return enriched
    except (WOSNotConfigured, WOSAPIError) as e:
        print(f"[wos] enrichment skipped: {e}")
        return w


# ---------------------------------------------------------------------------
# CLI: 凭据检查
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="WoS Starter API CLI")
    sub = parser.add_subparsers(dest="cmd")

    p_check = sub.add_parser("check", help="检查凭据是否配置正确")
    p_search = sub.add_parser("search", help="检索测试")
    p_search.add_argument("query", nargs="?", default='"exceptional point" AND acoustic')
    p_search.add_argument("--limit", type=int, default=5)

    args = parser.parse_args()

    if args.cmd == "check" or args.cmd is None:
        print(f"wos_ready       : {settings.wos_ready}")
        print(f"wos_api_key     : {'(set)' if settings.wos_api_key else '(unset)'}")
        print(f"wos_api_secret  : {'(set)' if settings.wos_api_secret else '(unset — Starter API 不需要)'}")
        print(f"wos_api_base_url: {settings.wos_api_base_url}")
        if settings.wos_ready:
            print("\n[OK] API Key present. Try: python -m pysci.skills.literature_research.tools.wos_client search")
        else:
            print("\n[!] API Key missing. Please fill WOS_API_KEY in .env")

    elif args.cmd == "search":
        try:
            client = WOSClient()
            r = client.search(args.query, limit=args.limit)
            print(f"Total hits: {r['total']}")
            for i, h in enumerate(r["hits"], 1):
                print(f"\n#{i} [{h['published_year']}] {h['title']}")
                print(f"   UT      : {h['ut']}")
                print(f"   Source  : {h['source']}   JIF={h['jif']}  Q={h['jcr_quartile']}")
                print(f"   Cited by: {h['cited_by_count']}")
                print(f"   ESI     : highly_cited={h['esi_highly_cited']}  hot={h['esi_hot_paper']}")
        except WOSNotConfigured as e:
            print(e)
