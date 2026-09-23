"""Zotero 桥接模块：读写用户的 Zotero 文献库。

支持两种后端：
1. **pyzotero**（首选）：成熟的 Python 库，自动处理 Web API 与本地 API 差异
2. **原生 requests**（回退）：直接调用 Zotero HTTP API

支持的操作：
- :meth:`ZoteroBridge.ping`          : 检查 Zotero 是否可达
- :meth:`ZoteroBridge.list_items`    : 列出库中条目（支持增量同步）
- :meth:`ZoteroBridge.get_item`      : 按 key 获取
- :meth:`ZoteroBridge.search_items`  : 关键词/标签检索
- :meth:`ZoteroBridge.create_item_from_metadata` : 由 OpenAlex/arXiv dict 创建新条目
- :meth:`ZoteroBridge.add_note`      : 添加子笔记（可写入 markdown 评价）
- :meth:`ZoteroBridge.add_tag`       : 打标签
- :meth:`ZoteroBridge.get_attachment_path` : 获取附件 PDF 的本地路径
- :meth:`ZoteroBridge.list_collections` : 列出分类
- :meth:`ZoteroBridge.add_to_collection` : 加入分类

Zotero 配置前提：
- Zotero 桌面版运行中（用于本地 API 与附件路径解析）
- Settings → Advanced → 勾选 "Allow other applications to communicate with Zotero"
- 已在 https://www.zotero.org/settings/keys 创建 private key（含 library + notes + write 权限）
- .env 中已填 ZOTERO_USER_ID 与 ZOTERO_API_KEY
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .config import http_session, settings

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
ZOTERO_ITEM_TYPES = {
    "journal-article",
    "preprint",
    "conference-paper",
    "book",
    "book-section",
    "thesis",
    "report",
    "manuscript",
    "patent",
    "webpage",
    "note",
    "attachment",
}


# ---------------------------------------------------------------------------
# 异常
# ---------------------------------------------------------------------------
class ZoteroNotConfigured(RuntimeError):
    def __init__(self, reason: str) -> None:
        super().__init__(
            f"Zotero not configured: {reason}\n"
            "Please ensure:\n"
            "  1. Zotero desktop is running\n"
            "  2. Settings → Advanced → 'Allow other applications' is enabled\n"
            "  3. ZOTERO_USER_ID and ZOTERO_API_KEY are set in project-root .env\n"
            "     (create key at https://www.zotero.org/settings/keys)"
        )


class ZoteroAPIError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# 主类
# ---------------------------------------------------------------------------
class ZoteroBridge:
    """Zotero 桥接。优先使用 pyzotero；不可用时回退到原生 requests。"""

    def __init__(self, prefer: str = "auto") -> None:
        """
        prefer:
            - 'auto'  : 优先 pyzotero，回退到 requests
            - 'pyzotero': 强制使用 pyzotero（不可用则抛错）
            - 'requests': 强制使用原生 requests
        """
        self.backend: str = ""
        self._zot = None  # pyzotero.Zotero 实例
        self._library_type = "user"
        self._library_id: str | int | None = None
        self._server_id: str | None = (
            None  # 本地 API 写操作需要的 Zotero-Server-ID（由 GET 响应捕获）
        )

        if prefer in ("auto", "pyzotero"):
            try:
                from pyzotero import zotero as pyzotero_mod  # type: ignore

                # 优先本地 API（无需网络、附件路径可解析、读到的是桌面端最新数据）
                if self._can_reach_local():
                    try:
                        # pyzotero >= 1.15：原生本地 API 支持（local=True）
                        self._zot = pyzotero_mod.Zotero(
                            library_id="0",
                            library_type="user",
                            local=True,
                        )
                    except TypeError:
                        # 旧版 pyzotero：用 base_url 指向本地端口
                        self._zot = pyzotero_mod.Zotero(
                            library_id="0",
                            library_type="user",
                            api_key="",
                            base_url=settings.zotero_local_api_base.rstrip("/"),
                        )
                    self._library_id = 0
                    self.backend = "pyzotero-local"
                elif settings.zotero_web_ready:
                    self._zot = pyzotero_mod.Zotero(
                        library_id=int(settings.zotero_user_id or 0),
                        library_type="user",
                        api_key=settings.zotero_api_key or "",
                    )
                    self._library_id = int(settings.zotero_user_id or 0)
                    self.backend = "pyzotero-web"
                elif prefer == "pyzotero":
                    raise ZoteroNotConfigured(
                        "Neither local Zotero nor Web API credentials available"
                    )
            except ImportError:
                if prefer == "pyzotero":
                    raise ZoteroNotConfigured(
                        "pyzotero not installed. Run: pip install pyzotero"
                    ) from None
            except Exception as e:
                if prefer == "pyzotero":
                    raise ZoteroNotConfigured(str(e)) from e

        if not self.backend:
            # 回退到原生 requests
            if self._can_reach_local():
                self.backend = "requests-local"
                self._library_id = 0
            elif settings.zotero_web_ready:
                self.backend = "requests-web"
                self._library_id = int(settings.zotero_user_id or 0)
            else:
                raise ZoteroNotConfigured("No local Zotero, no Web API credentials")

        print(f"[zotero] Backend: {self.backend}")

    # ------------------------------------------------------------------
    # 内部：URL 与请求
    # ------------------------------------------------------------------
    def _can_reach_local(self) -> bool:
        """检查 Zotero 桌面版的本地 API 是否可达。"""
        try:
            with http_session(retries=0) as s:
                r = s.get(
                    f"{settings.zotero_local_api_base.rstrip('/')}/users/0/items",
                    params={"limit": 1, "format": "keys"},
                    timeout=2,
                )
                return r.status_code in (
                    200,
                    403,
                )  # 403 也说明 Zotero 在跑，只是权限问题
        except Exception:
            return False

    def _base_url(self, for_write: bool = False) -> str:
        """返回 API base URL。

        写操作策略：本地 API 的写操作在 Zotero 7+ 需要交互式授权
        （authorize_local() 弹窗 + 单次/永久 local key），不适合自动化流程；
        因此写操作统一回退到 Web API（用 user_id + api_key，非交互、可靠）。
        读操作仍优先本地 API（离线、快、能解析附件路径）。
        """
        if for_write and self.backend.endswith("-local"):
            # 写操作走 Web API（避免本地 API 的交互式授权）
            if settings.zotero_web_ready:
                return settings.zotero_web_api_base.rstrip("/")
        if self.backend.endswith("-local"):
            return settings.zotero_local_api_base.rstrip("/")
        return settings.zotero_web_api_base.rstrip("/")

    def _library_id_for(self, for_write: bool = False) -> str | int:
        """写操作回退到 Web API 时，需要用真实 user_id 而非 0。"""
        if for_write and self.backend.endswith("-local") and settings.zotero_web_ready:
            return int(settings.zotero_user_id or 0)
        return self._library_id or 0

    def _url(self, path: str, for_write: bool = False) -> str:
        base = self._base_url(for_write=for_write)
        lib_id = self._library_id_for(for_write=for_write)
        if path.startswith("/users/"):
            return f"{base}{path}"
        return f"{base}/users/{lib_id}{path}"

    def _headers(self, for_write: bool = False) -> dict[str, str]:
        h = {"Accept": "application/json", "Content-Type": "application/json"}
        # Web API 需要 API key
        use_web = self.backend.endswith("-web") or (
            for_write and self.backend.endswith("-local") and settings.zotero_web_ready
        )
        if use_web and settings.zotero_api_key:
            h["Zotero-API-Key"] = settings.zotero_api_key
            h["Zotero-API-Version"] = "3"
        # 本地 API 写操作需要 Server-ID
        if not use_web and self.backend.endswith("-local") and for_write:
            if self._server_id:
                h["Zotero-Server-ID"] = self._server_id
        return h

    def _raw_get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        url = self._url(path)
        with http_session() as s:
            r = s.get(
                url,
                params=params or {},
                headers=self._headers(),
                timeout=settings.http_timeout,
            )
            # 捕获 Zotero-Server-ID（本地 API 写操作需要）
            sid = r.headers.get("Zotero-Server-ID") or r.headers.get("zotero-server-id")
            if sid:
                self._server_id = sid
            if r.status_code != 200:
                raise ZoteroAPIError(
                    f"GET {url} failed ({r.status_code}): {r.text[:300]}"
                )
            return r.json()

    def _raw_post(self, path: str, payload: Any) -> Any:
        url = self._url(path, for_write=True)
        with http_session() as s:
            r = s.post(
                url,
                json=payload,
                headers=self._headers(for_write=True),
                timeout=settings.http_timeout,
            )
            if r.status_code not in (200, 201, 204):
                raise ZoteroAPIError(
                    f"POST {url} failed ({r.status_code}): {r.text[:300]}"
                )
            return r.json() if r.text else {}

    def _raw_patch(self, path: str, payload: Any) -> Any:
        url = self._url(path, for_write=True)
        with http_session() as s:
            r = s.patch(
                url,
                json=payload,
                headers=self._headers(for_write=True),
                timeout=settings.http_timeout,
            )
            if r.status_code not in (200, 204):
                raise ZoteroAPIError(
                    f"PATCH {url} failed ({r.status_code}): {r.text[:300]}"
                )
            return r.json() if r.text else {}

    # ------------------------------------------------------------------
    # 公共：连接性
    # ------------------------------------------------------------------
    def ping(self) -> dict[str, Any]:
        """检查 Zotero 是否可达，返回基本信息。"""
        try:
            # 优先用 pyzotero 的内置方法
            if self._zot is not None:
                items = self._zot.items(limit=1)
                count = len(items) if isinstance(items, list) else 0
                return {"ok": True, "backend": self.backend, "items_sample": count}
            # 回退到 raw GET（用 json format 避免解析错误）
            url = self._url("/items")
            with http_session() as s:
                r = s.get(
                    url,
                    params={"limit": 1, "format": "json"},
                    headers=self._headers(),
                    timeout=10,
                )
                if r.status_code == 200:
                    data = r.json()
                    return {
                        "ok": True,
                        "backend": self.backend,
                        "items_sample": len(data) if isinstance(data, list) else 0,
                    }
                return {
                    "ok": False,
                    "backend": self.backend,
                    "error": f"HTTP {r.status_code}: {r.text[:200]}",
                }
        except Exception as e:
            return {"ok": False, "backend": self.backend, "error": str(e)}

    # ------------------------------------------------------------------
    # 读操作
    # ------------------------------------------------------------------
    def list_items(
        self,
        *,
        since: int | None = None,
        limit: int = 100,
        item_type: str | None = None,
        tag: str | None = None,
        collection: str | None = None,
    ) -> list[dict[str, Any]]:
        """列出条目。

        since: Zotero 库版本号，用于增量同步
        item_type: 过滤条目类型（journalArticle, preprint, ...）
        tag: 过滤标签（可用多个，逗号分隔）
        collection: 分类 key
        """
        params: dict[str, Any] = {"limit": min(limit, 100), "format": "json"}
        if since is not None:
            params["since"] = since
        if item_type:
            params["itemType"] = item_type
        if tag:
            params["tag"] = tag
        if collection:
            path = f"/collections/{collection}/items"
        else:
            path = "/items"
        return self._dispatch_list(path, params)

    def _dispatch_list(self, path: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        """统一 list 操作，兼容 pyzotero 与 requests 后端。

        pyzotero 后端：根据 path 选择正确的方法（不能总是用 .items()，
        否则 /items/KEY/children 会错误地返回整个库）：
          - /items/{key}/children   -> .children(key)
          - /collections/{key}/items-> .collection_items(key)
          - /collections            -> .collections()
          - /items （默认）          -> .items()
        任一环节失败则回退到 raw GET（本地 API 同样能正确解析该 path）。
        """
        if self._zot is not None:
            kwargs = _to_pyzotero_kwargs(params)
            try:
                m = re.match(r"^/items/([^/]+)/children$", path)
                if m:
                    return list(self._zot.children(m.group(1), **kwargs))
                m = re.match(r"^/collections/([^/]+)/items$", path)
                if m:
                    return list(
                        self._zot.everything(
                            self._zot.collection_items(m.group(1), **kwargs)
                        )
                    )
                if path == "/collections":
                    return list(self._zot.everything(self._zot.collections(**kwargs)))
                # 默认：/items
                return list(self._zot.everything(self._zot.items(**kwargs)))
            except Exception:
                # 回退到 raw
                pass
        return self._raw_get(path, params=params)

    def get_item(self, key: str) -> dict[str, Any] | None:
        """按 item key 获取单条。"""
        try:
            if self._zot is not None:
                return self._zot.item(key)
            return self._raw_get(f"/items/{key}")
        except ZoteroAPIError:
            return None
        except Exception as e:
            print(f"[zotero] get_item({key}) failed: {e}")
            return None

    def search_items(self, query: str, *, limit: int = 25) -> list[dict[str, Any]]:
        """按关键词搜索（命中标题/作者/标签等元数据，不含全文）。"""
        params = {"q": query, "limit": limit, "format": "json"}
        return self._dispatch_list("/items", params)

    def fulltext_search(self, query: str, *, limit: int = 25) -> list[dict[str, Any]]:
        """全文搜索（需 Zotero 已建立全文索引，仅本地 API 可用）。"""
        if not self.backend.endswith("-local"):
            print("[zotero] fulltext_search only available on local API")
            return []
        params = {"q": query, "limit": limit, "format": "json"}
        try:
            return self._raw_get("/fulltext", params=params)
        except ZoteroAPIError:
            return []

    def list_collections(self) -> list[dict[str, Any]]:
        return self._dispatch_list("/collections", {"limit": 100, "format": "json"})

    def get_attachment_path(self, parent_item_key: str) -> Path | None:
        """获取父条目下第一个 PDF 附件的本地路径。

        仅本地 API + Zotero 桌面版运行时可用。
        """
        try:
            children = self._dispatch_list(
                f"/items/{parent_item_key}/children",
                {"format": "json", "limit": 20},
            )
        except ZoteroAPIError:
            return None

        for c in children:
            data = c.get("data") or c
            if data.get("itemType") != "attachment":
                continue
            content_type = data.get("contentType", "")
            if "pdf" not in content_type.lower():
                continue
            # 附件文件名与 key
            filename = data.get("filename") or ""
            att_key = c.get("key") or data.get("key") or ""
            if not filename or not att_key:
                continue
            # Zotero 存储结构：<data_dir>/storage/<ATTACHMENT_KEY>/<filename>
            if settings.zotero_data_dir:
                p = settings.zotero_data_dir / "storage" / att_key / filename
                if p.exists():
                    return p
            # 或者 linked attachment（绝对路径）
            link_mode = data.get("linkMode", "")
            if link_mode == "linked_file":
                abs_path = data.get("path", "")
                # Zotero 用 attachments: 前缀表示相对 data_dir
                if abs_path.startswith("attachments:"):
                    rel = abs_path.replace("attachments:", "", 1).lstrip("/\\")
                    if settings.zotero_data_dir:
                        p = settings.zotero_data_dir / rel
                        if p.exists():
                            return p
                else:
                    p = Path(abs_path)
                    if p.exists():
                        return p
        return None

    # ------------------------------------------------------------------
    # 写操作
    # ------------------------------------------------------------------
    def create_item_from_metadata(
        self,
        meta: dict[str, Any],
        *,
        collections: Iterable[str] = (),
        tags: Iterable[str] = (),
    ) -> dict[str, Any]:
        """由 openalex_client / arxiv_client 生成的元数据 dict 创建 Zotero 条目。

        meta 应包含至少 title；其他字段按需填入。
        返回 Zotero 创建响应（含新条目 key）。
        """
        item_type = _infer_zotero_item_type(meta)
        creators = _build_creators(meta.get("authors") or [])

        data: dict[str, Any] = {
            "itemType": item_type,
            "title": meta.get("title", ""),
            "creators": creators,
            "abstractNote": meta.get("abstract", "") or "",
            "date": _normalize_date(meta.get("publication_date") or meta.get("year")),
            "DOI": meta.get("doi", "") or "",
            "url": meta.get("oa_url") or "",
            "tags": [{"tag": t} for t in tags] if tags else [],
            "collections": list(collections),
        }

        if item_type == "journalArticle":
            data.update(
                {
                    "publicationTitle": meta.get("journal", ""),
                    "volume": meta.get("volume", ""),
                    "issue": meta.get("issue", ""),
                    "pages": meta.get("pages", ""),
                    "ISSN": "",
                    "publisher": meta.get("publisher", ""),
                }
            )
        elif item_type == "preprint":
            data.update(
                {
                    "repository": "arXiv",
                    "archiveID": meta.get("arxiv_id", ""),
                    "place": "",
                }
            )
            if meta.get("arxiv_id"):
                data["url"] = f"https://arxiv.org/abs/{meta['arxiv_id']}"

        # 存入 extra 字段：所有本项目自定义元数据（openalex_id, jif, quartile 等）
        extra_lines = []
        for k in (
            "openalex_id",
            "wos_id",
            "arxiv_id",
            "jif",
            "jcr_quartile",
            "cited_by_count",
            "oa_status",
        ):
            v = meta.get(k)
            if v not in (None, "", 0):
                extra_lines.append(f"{k}: {v}")
        if extra_lines:
            data["extra"] = "\n".join(extra_lines)

        payload = [data]  # Zotero POST /items 需要数组
        return self._raw_post("/items", payload)

    def add_note(
        self, parent_item_key: str, note_markdown: str, *, tags: Iterable[str] = ()
    ) -> dict[str, Any]:
        """为某条目添加子笔记。

        note_markdown 会被简单转换为 HTML（Zotero 笔记存 HTML）。
        复杂 markdown 建议先经 markdown → html 库转换。
        """
        html = _md_to_zotero_html(note_markdown)
        data = {
            "itemType": "note",
            "parentItem": parent_item_key,
            "note": html,
            "tags": [{"tag": t} for t in tags] if tags else [],
        }
        return self._raw_post("/items", [data])

    def add_tag(self, item_key: str, tags: Iterable[str]) -> Any:
        """给条目追加标签（保留原有标签）。"""
        item = self.get_item(item_key)
        if not item:
            return None
        existing = {t["tag"] for t in ((item.get("data") or item).get("tags") or [])}
        merged = sorted(existing | set(tags))
        version = (item.get("data") or item).get("version") or item.get("version")
        payload = {"tags": [{"tag": t} for t in merged], "version": version}
        return self._raw_patch(f"/items/{item_key}", payload)

    def add_to_collection(self, item_key: str, collection_key: str) -> Any:
        item = self.get_item(item_key)
        if not item:
            return None
        data = item.get("data") or item
        cols = list(data.get("collections") or [])
        if collection_key not in cols:
            cols.append(collection_key)
        version = data.get("version") or item.get("version")
        return self._raw_patch(
            f"/items/{item_key}", {"collections": cols, "version": version}
        )

    # ------------------------------------------------------------------
    # 便捷：库信息
    # ------------------------------------------------------------------
    def library_version(self) -> int | None:
        """返回当前库版本号（用于增量同步）。"""
        try:
            url = self._url("/items")
            with http_session() as s:
                r = s.head(url, params={"limit": 1}, headers=self._headers(), timeout=5)
                v = r.headers.get("Last-Modified-Version") or r.headers.get(
                    "Zotero-Schema-Version"
                )
                return int(v) if v else None
        except Exception:
            return None


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------
def _to_pyzotero_kwargs(params: dict[str, Any]) -> dict[str, Any]:
    """把通用 params 转换为 pyzotero 接受的 kwargs。"""
    mapping = {
        "limit": "limit",
        "format": None,  # pyzotero 不接受 format
        "since": "since",
        "itemType": "itemType",
        "tag": "tag",
        "q": "q",
    }
    out = {}
    for k, v in params.items():
        mapped = mapping.get(k, k)
        if mapped and v is not None:
            out[mapped] = v
    return out


def _infer_zotero_item_type(meta: dict[str, Any]) -> str:
    """根据元数据推断 Zotero 条目类型。"""
    journal = meta.get("journal", "") or ""
    if "arxiv" in journal.lower() or (meta.get("arxiv_id") and not journal):
        return "preprint"
    if journal:
        return "journalArticle"
    if meta.get("type") == "preprint":
        return "preprint"
    return "journalArticle"


def _build_creators(authors: Iterable[Any]) -> list[dict[str, str]]:
    """把 authors（字符串或 dict 列表）转换为 Zotero creators 结构。"""
    creators = []
    for a in authors:
        if isinstance(a, str):
            name = a
        elif isinstance(a, dict):
            name = a.get("name", "")
        else:
            continue
        if not name:
            continue
        # Zotero 期望 "Last, First" 或 "First Last"（firstName + lastName）
        parts = name.split()
        if len(parts) >= 2:
            creators.append(
                {
                    "creatorType": "author",
                    "firstName": " ".join(parts[:-1]),
                    "lastName": parts[-1],
                }
            )
        else:
            creators.append({"creatorType": "author", "name": name})
    return creators


def _normalize_date(raw: Any) -> str:
    """把 year / date 混合字段规范化为 Zotero 可接受的字符串。"""
    if raw is None or raw == "":
        return ""
    if isinstance(raw, int):
        return str(raw)
    s = str(raw).strip()
    if re.match(r"^\d{4}$", s):
        return s
    if re.match(r"^\d{4}-\d{2}-\d{2}$", s):
        return s
    if re.match(r"^\d{4}-\d{2}$", s):
        return s
    return s


def _md_to_zotero_html(md: str) -> str:
    """把 markdown 简单转为 HTML（Zotero 笔记存 HTML）。

    仅做基础转换：标题、粗体、斜体、列表、代码块、链接。
    复杂 markdown 建议装 `markdown` 库后走 :func:`markdown.markdown`。
    """
    try:
        import markdown  # type: ignore

        return markdown.markdown(md, extensions=["extra", "sane_lists"])
    except ImportError:
        pass

    # 兜底：极简转换
    lines = md.split("\n")
    out: list[str] = []
    in_code = False
    in_list = False
    for line in lines:
        if line.strip().startswith("```"):
            if in_code:
                out.append("</pre>")
                in_code = False
            else:
                out.append("<pre>")
                in_code = True
            continue
        if in_code:
            out.append(line.replace("<", "&lt;").replace(">", "&gt;"))
            continue

        # 标题
        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            level = len(m.group(1))
            out.append(f"<h{level}>{_inline_md(m.group(2))}</h{level}>")
            continue

        # 列表
        m = re.match(r"^\s*[-*+]\s+(.*)$", line)
        if m:
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{_inline_md(m.group(1))}</li>")
            continue
        elif in_list:
            out.append("</ul>")
            in_list = False

        # 空行
        if not line.strip():
            out.append("")
            continue

        out.append(f"<p>{_inline_md(line)}</p>")

    if in_list:
        out.append("</ul>")
    if in_code:
        out.append("</pre>")
    return "\n".join(out)


def _inline_md(s: str) -> str:
    """处理行内 markdown：粗体、斜体、代码、链接。"""
    s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"__(.+?)__", r"<strong>\1</strong>", s)
    s = re.sub(r"\*(.+?)\*", r"<em>\1</em>", s)
    s = re.sub(r"`(.+?)`", r"<code>\1</code>", s)
    s = re.sub(r"\[(.+?)\]\((.+?)\)", r'<a href="\2">\1</a>', s)
    return s


# ---------------------------------------------------------------------------
# CLI: 连通性测试
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Zotero bridge CLI")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("ping", help="检查连通性")

    p_list = sub.add_parser("list", help="列出条目")
    p_list.add_argument("--limit", type=int, default=10)
    p_list.add_argument("--type", default=None)

    p_search = sub.add_parser("search", help="搜索")
    p_search.add_argument("query")
    p_search.add_argument("--limit", type=int, default=10)

    p_get = sub.add_parser("get", help="获取单条")
    p_get.add_argument("key")
    p_get.add_argument(
        "--attachment", action="store_true", help="同时打印附件 PDF 路径"
    )

    args = parser.parse_args()

    try:
        bridge = ZoteroBridge()
    except ZoteroNotConfigured as e:
        print(e)
        raise SystemExit(1) from e

    if args.cmd == "ping" or args.cmd is None:
        print(json.dumps(bridge.ping(), indent=2, ensure_ascii=False))

    elif args.cmd == "list":
        items = bridge.list_items(limit=args.limit, item_type=args.type)
        for it in items:
            d = it.get("data") or it
            print(
                f"[{d.get('key', '?')}] {d.get('itemType', '?')}: {d.get('title', '?')[:80]}"
            )

    elif args.cmd == "search":
        items = bridge.search_items(args.query, limit=args.limit)
        for it in items:
            d = it.get("data") or it
            print(
                f"[{d.get('key', '?')}] {d.get('itemType', '?')}: {d.get('title', '?')[:80]}"
            )

    elif args.cmd == "get":
        it = bridge.get_item(args.key)
        if it:
            print(json.dumps(it, indent=2, ensure_ascii=False)[:2000])
            if args.attachment:
                p = bridge.get_attachment_path(args.key)
                print(f"\nAttachment PDF: {p}")
        else:
            print(f"Item {args.key} not found")
