"""zotero_cli —— 通过社区 zotero-mcp 的 `zotero-cli` 读写 Zotero 文献库。

本模块替代原手写 :mod:`zotero_bridge`（requests + pyzotero 双后端），把所有 Zotero
交互委托给 zotero-mcp-server 包提供的 ``zotero-cli --json``，解析其**稳定 JSON 契约**
（见上游 docs/cli.md）::

    成功 {"ok": true,  "command": "...", "schema": 1, "data": {...}}
    失败 {"ok": false, "command": "...", "schema": 1, "error": {"message": ..., "code": ...}}

stdout 只含 JSON；``[INFO]`` / ``[WARN]`` 诊断走 stderr。

设计要点
--------
1. **未安装即优雅降级**：:func:`available` 返回 False 时，上层（``research add``）跳过
   入库、仅生成笔记骨架；``research library`` 打印安装指引后返回非零。
2. **条目形状透传**：zotero-mcp 内部依赖 pyzotero，``search`` / ``get metadata`` 的
   ``data`` 多为 Zotero API 原形（``{key, version, data:{itemType,title,creators,...}}``）。
   本模块原样透传条目 dict；下游 :mod:`research` 与 document_writing.refs_bridge 均用
   ``it.get("data", it)`` 兼容「嵌套 / 扁平」两种形态，无需改动字段访问。
3. **建条目走 `add bibtex`**：frontmatter → BibTeX 的**规范化契约保留在本 CLI**（项目
   差异化层），自定义增强字段（openalex_id / wos_id / jif / jcr_quartile / cited_by_count /
   oa_status）写入 BibTeX ``note`` 字段（Zotero 导入时映射为条目的 Extra）。``add`` 默认
   幂等（``--if-exists file``）：命中同 DOI/arXiv/ISBN 的既有条目则复用而非重复创建。
4. **方法名沿用旧接口**：:class:`ZoteroCli` 暴露 ping / list_items / get_item /
   search_items / create_item_from_metadata / get_bibtex / add_note，签名与 ZoteroBridge
   兼容，令 research.py 与 refs_bridge.py 的调用点几乎零改动。

Zotero 配置前提（由 zotero-cli 自身解析，本模块只做变量名翻译）：
- **本地模式**：Zotero 桌面版运行中，Settings → Advanced → 勾选 "Allow other
  applications..."；Zotero 10+ 写入需一次性 ``zotero-mcp authorize-local``（选 Always Allow）。
- **Web 模式**：``ZOTERO_API_KEY`` + ``ZOTERO_LIBRARY_ID``（本项目 .env 的
  ZOTERO_USER_ID 会在子进程环境中翻译为 ZOTERO_LIBRARY_ID）。
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from collections.abc import Iterable
from typing import Any

from .config import settings

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
# zotero-mcp-server 安装后提供的可执行名（console script）。
CLI_BIN_CANDIDATES = ("zotero-cli",)

# frontmatter 自定义增强字段 → 折进 BibTeX note（Zotero 导入映射为 Extra）。
# 顺序即写入 note 的行序，与旧 create_item_from_metadata 的 extra 保持一致。
_EXTRA_FIELDS = (
    "openalex_id",
    "wos_id",
    "arxiv_id",
    "jif",
    "jcr_quartile",
    "cited_by_count",
    "oa_status",
)

# BibTeX note → Zotero Extra 的映射键。若首次真机验证发现 zotero-mcp 把 note 落成
# 子笔记而非 Extra，改此常量为 "annote" 即可（一行切换，见 UPSTREAM.lock pending_live_checks）。
_BIBTEX_EXTRA_FIELD = "note"

# citekey 首词跳过词（与 document_writing.refs_bridge.make_citekey 保持同风格）。
_CITEKEY_STOPWORDS = {
    "the",
    "a",
    "an",
    "on",
    "of",
    "in",
    "for",
    "and",
    "to",
    "with",
    "via",
    "from",
    "by",
    "at",
}


# ---------------------------------------------------------------------------
# 异常
# ---------------------------------------------------------------------------
class ZoteroCliError(RuntimeError):
    """zotero-cli 调用失败（返回 ok:false / 无 JSON 输出 / 超时 / 无法解析）。"""


class ZoteroNotConfigured(ZoteroCliError):
    """zotero-cli 未安装或不可执行（保留旧 zotero_bridge 的异常名以兼容捕获）。"""


# ---------------------------------------------------------------------------
# 探测与环境
# ---------------------------------------------------------------------------
def zotero_cli_bin() -> str | None:
    """返回 ``zotero-cli`` 可执行文件路径；未安装返回 None。"""
    for name in CLI_BIN_CANDIDATES:
        found = shutil.which(name)
        if found:
            return found
    return None


def available() -> bool:
    """zotero-cli 是否在 PATH 中（即 zotero-mcp-server 是否已安装）。"""
    return zotero_cli_bin() is not None


def _cli_env() -> dict[str, str]:
    """构造子进程环境：继承 os.environ，并把本项目 .env 的 Zotero 凭据翻译成
    zotero-cli 期望的变量名。

    - ZOTERO_API_KEY：直接沿用（本项目与 zotero-cli 同名）。
    - ZOTERO_USER_ID → ZOTERO_LIBRARY_ID（zotero-cli/Web API 用 LIBRARY_ID 命名）。
    - 本地模式（ZOTERO_LOCAL）由 zotero-cli 自身 config 决定，此处不强制覆盖。
    已存在于环境中的同名变量优先（尊重用户在 shell / zotero-mcp config 的显式设置）。
    """
    env = dict(os.environ)
    if settings.zotero_api_key and not env.get("ZOTERO_API_KEY"):
        env["ZOTERO_API_KEY"] = settings.zotero_api_key
    if settings.zotero_user_id and not env.get("ZOTERO_LIBRARY_ID"):
        env["ZOTERO_LIBRARY_ID"] = settings.zotero_user_id
        env.setdefault("ZOTERO_LIBRARY_TYPE", "user")
    return env


# ---------------------------------------------------------------------------
# 子进程调用与信封解析
# ---------------------------------------------------------------------------
def _run_json(args: list[str], *, timeout: int | None = None) -> Any:
    """运行 ``zotero-cli --json <args>``，校验信封 ``ok`` 后返回 ``data``。

    失败（未安装 / ok:false / 非 JSON / 超时）统一抛 :class:`ZoteroCliError` 家族。
    """
    bin_ = zotero_cli_bin()
    if not bin_:
        raise ZoteroNotConfigured(
            "zotero-cli 未安装（zotero-mcp-server）。请运行 "
            "scripts/zotero_mcp/setup_zotero_mcp.ps1（内部执行 "
            "uv tool install 'zotero-mcp-server[pdf,scite]'）。"
        )
    cmd = [bin_, "--json", *args]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=timeout or max(60, settings.http_timeout * 2),
            env=_cli_env(),
            check=False,
        )
    except FileNotFoundError as e:
        raise ZoteroNotConfigured(f"zotero-cli 不可执行：{e}") from e
    except subprocess.TimeoutExpired as e:
        raise ZoteroCliError(f"zotero-cli 超时：{' '.join(args)}") from e

    out = (proc.stdout or "").strip()
    if not out:
        raise ZoteroCliError(
            f"zotero-cli 无 JSON 输出（exit={proc.returncode}）："
            f"{(proc.stderr or '')[:300]}"
        )
    try:
        envelope = json.loads(out)
    except json.JSONDecodeError as e:
        raise ZoteroCliError(f"zotero-cli 输出非 JSON：{out[:300]}") from e
    if not isinstance(envelope, dict):
        raise ZoteroCliError(f"zotero-cli 输出结构异常：{out[:300]}")
    if not envelope.get("ok", False):
        err = envelope.get("error") or {}
        msg = err.get("message") if isinstance(err, dict) else str(err)
        code = err.get("code") if isinstance(err, dict) else ""
        raise ZoteroCliError(
            f"zotero-cli {envelope.get('command', '?')} 失败"
            f"{f'（{code}）' if code else ''}：{msg or out[:300]}"
        )
    return envelope.get("data")


def _items_from_data(data: Any) -> list[dict[str, Any]]:
    """从 zotero-cli 的 ``data`` 里尽力取出条目列表（兼容多种字段名 / 直接是 list）。"""
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        for k in ("items", "results", "entries", "data"):
            v = data.get(k)
            if isinstance(v, list):
                return [x for x in v if isinstance(x, dict)]
    return []


def _text_from_data(data: Any) -> str:
    """从 ``data`` 里取文本载荷（bibtex / 全文等；兼容 str 与多种键名）。"""
    if isinstance(data, str):
        return data
    if isinstance(data, dict):
        for k in ("bibtex", "text", "output", "content", "fulltext"):
            v = data.get(k)
            if isinstance(v, str) and v:
                return v
    return ""


def _extract_key_from_add(data: Any) -> str:
    """从 ``add`` 命令的 data 里尽力提取新条目（或复用的既有条目）key。"""
    if isinstance(data, dict):
        for k in ("key", "item_key", "itemKey"):
            if data.get(k):
                return str(data[k])
        item = data.get("item")
        if isinstance(item, dict):
            ik = item.get("key") or (item.get("data") or {}).get("key")
            if ik:
                return str(ik)
        items = data.get("items")
        if isinstance(items, list) and items:
            first = items[0]
            if isinstance(first, dict):
                return str(
                    first.get("key") or (first.get("data") or {}).get("key") or ""
                )
            return str(first)
        # 兜底：写命令常把结果放在 data.text，从里面抓一个 8 位 Zotero key
        txt = str(data.get("text") or "")
        m = re.search(r"\b([A-Z0-9]{8})\b", txt)
        if m:
            return m.group(1)
    return ""


# ---------------------------------------------------------------------------
# frontmatter → BibTeX（规范化契约，保留在 CLI）
# ---------------------------------------------------------------------------
def _bib_escape(s: str) -> str:
    """转义 BibTeX 中易出问题的字符（保守处理，不动数学符号 $ 与中文）。"""
    return s.replace("&", r"\&").replace("%", r"\%").replace("#", r"\#")


def _bib_author_name(name: str) -> str:
    """把 'First Last' / 'First von Last' 规范为 BibTeX 的 'Last, First'。

    已含逗号（视为 'Last, First'）则原样返回；单词名原样返回。
    """
    name = (name or "").strip()
    if not name:
        return ""
    if "," in name:
        return name
    parts = name.split()
    if len(parts) >= 2:
        return f"{parts[-1]}, {' '.join(parts[:-1])}"
    return name


def _bib_authors(authors: Iterable[Any]) -> str:
    """把 frontmatter 的 authors（字符串名列表，或 {name}/{firstName,lastName} dict）
    转为 BibTeX author 字段（'Last, First and Last, First'）。"""
    out: list[str] = []
    for a in authors or []:
        if isinstance(a, dict):
            nm = a.get("name") or ""
            if not nm and (a.get("lastName") or a.get("firstName")):
                nm = f"{a.get('firstName', '')} {a.get('lastName', '')}".strip()
        else:
            nm = str(a)
        bib = _bib_author_name(nm)
        if bib:
            out.append(bib)
    return " and ".join(out)


def _bib_year(fm: dict[str, Any]) -> str:
    """从 year / publication_date 里取 4 位年份。"""
    for k in ("year", "publication_date"):
        v = fm.get(k)
        if v in (None, ""):
            continue
        m = re.search(r"(\d{4})", str(v))
        if m:
            return m.group(1)
    return ""


def _is_preprint(fm: dict[str, Any]) -> bool:
    """判断是否预印本（arXiv）——决定 BibTeX 条目类型走 @misc 还是 @article。"""
    journal = str(fm.get("journal") or "").lower()
    if "arxiv" in journal:
        return True
    if str(fm.get("publisher") or "").lower() == "arxiv":
        return True
    if fm.get("arxiv_id") and not str(fm.get("journal") or "").strip():
        return True
    return False


def make_citekey(fm: dict[str, Any]) -> str:
    """Better BibTeX 风格 citekey：首作者姓 + 年 + 标题首个实词（全小写）。

    与 document_writing.refs_bridge.make_citekey 同风格，令调研入库与写作导出的
    citekey 尽量一致（减少 .bib 刷新后 \\cite 漂移）。
    """
    last = re.sub(r"[^\w]", "", str(fm.get("first_author_last_name") or "")).lower()
    if not last:
        authors = fm.get("authors") or []
        if authors:
            a0 = authors[0]
            nm = (
                a0
                if isinstance(a0, str)
                else (a0.get("name", "") if isinstance(a0, dict) else "")
            )
            toks = str(nm).split()
            last = re.sub(r"[^\w]", "", toks[-1] if toks else "").lower()
    last = last or "anon"
    year = _bib_year(fm) or "nd"
    word = ""
    for w in re.split(r"[^\w]+", str(fm.get("title") or "").lower()):
        if w and w not in _CITEKEY_STOPWORDS:
            word = w
            break
    return f"{last}{year}{word or 'untitled'}"


def frontmatter_to_bibtex(
    fm: dict[str, Any],
    *,
    citekey: str | None = None,
    tags: Iterable[str] = (),
) -> str:
    """把本项目 paper-note frontmatter 规范化为一条 BibTeX（供 ``zotero-cli add bibtex``）。

    - 期刊文章 → ``@article``；arXiv 预印本 → ``@misc`` + eprint/archivePrefix。
    - 自定义增强字段（openalex_id/wos_id/arxiv_id/jif/jcr_quartile/cited_by_count/
      oa_status）折进 ``note``（多行 ``key: value``），Zotero 导入后落到条目 Extra，
      与旧 create_item_from_metadata 的 extra 行为一致，refs_bridge 可按行读回。
    - tags → BibTeX ``keywords``（Zotero 导入映射为标签），避免依赖不确定的 add --tags 旗标。
    """
    key = citekey or make_citekey(fm)
    preprint = _is_preprint(fm)
    bibtype = "misc" if preprint else "article"

    fields: dict[str, str] = {}
    author = _bib_authors(fm.get("authors"))
    if author:
        fields["author"] = author
    if fm.get("title"):
        fields["title"] = _bib_escape(str(fm["title"]))
    year = _bib_year(fm)
    if year:
        fields["year"] = year

    if preprint:
        arxiv = str(fm.get("arxiv_id") or "").replace("arXiv:", "").strip()
        if arxiv:
            fields["eprint"] = arxiv
            fields["archivePrefix"] = "arXiv"
        if fm.get("journal"):
            fields["howpublished"] = str(fm["journal"])
    else:
        if fm.get("journal"):
            fields["journal"] = str(fm["journal"])
        if fm.get("volume"):
            fields["volume"] = str(fm["volume"])
        if fm.get("issue"):
            fields["number"] = str(fm["issue"])
        if fm.get("pages"):
            fields["pages"] = str(fm["pages"])
        if fm.get("publisher"):
            fields["publisher"] = str(fm["publisher"])

    if fm.get("doi"):
        fields["doi"] = str(fm["doi"])
    if fm.get("oa_url"):
        fields["url"] = str(fm["oa_url"])
    if fm.get("abstract"):
        abstract = str(fm["abstract"]).replace("\n", " ").strip()
        if len(abstract) > 1200:
            abstract = abstract[:1200].rstrip() + "..."
        fields["abstract"] = _bib_escape(abstract)

    extra_lines = []
    for k in _EXTRA_FIELDS:
        v = fm.get(k)
        if v not in (None, "", 0, False):
            extra_lines.append(f"{k}: {v}")
    if extra_lines:
        fields[_BIBTEX_EXTRA_FIELD] = "\n".join(extra_lines)

    tag_list = [str(t).strip() for t in (tags or []) if str(t).strip()]
    if tag_list:
        fields["keywords"] = ", ".join(tag_list)

    fields = {k: v for k, v in fields.items() if v not in (None, "")}
    body = ",\n".join(f"  {k} = {{{v}}}" for k, v in fields.items())
    return f"@{bibtype}{{{key},\n{body}\n}}"


# ---------------------------------------------------------------------------
# 主类
# ---------------------------------------------------------------------------
class ZoteroCli:
    """Zotero 库读写（委托 zotero-mcp 的 ``zotero-cli --json``）。

    替代原 :class:`ZoteroBridge`（手写 requests + pyzotero 双后端）。构造不抛异常——
    未安装时 :attr:`backend` 标记为不可用，各方法在被调用时抛 :class:`ZoteroCliError`，
    便于 doctor / library 优雅报告而非崩溃。
    """

    def __init__(self) -> None:
        self.backend: str = "zotero-cli" if available() else "zotero-cli(unavailable)"

    # ------------------------------------------------------------------
    # 连通性
    # ------------------------------------------------------------------
    def ping(self) -> dict[str, Any]:
        """检查 zotero-cli 是否可达并回显其解析后的配置。"""
        if not available():
            return {
                "ok": False,
                "backend": self.backend,
                "error": "zotero-cli 未安装（zotero-mcp-server）",
            }
        try:
            data = _run_json(["config"])
            return {"ok": True, "backend": self.backend, "config": data}
        except ZoteroCliError as e:
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

        - collection：``get collection-items <key>``
        - tag：``search --mode tag <tag>``
        - 否则：``search <item_type|''>``（zotero-cli 无专用 list-all，尽力而为；
          见 scripts/zotero_mcp/UPSTREAM.lock.json 的 pending_live_checks）
        ``since`` 由 zotero-cli 侧的库版本语义处理，此处不透传（本地/ Web 差异大）。
        """
        n = str(min(max(limit, 1), 100))
        if collection:
            data = _run_json(["get", "collection-items", collection, "--limit", n])
            return _items_from_data(data)
        if tag:
            data = _run_json(["search", "--mode", "tag", tag, "--limit", n])
            return _items_from_data(data)
        q = item_type or ""
        data = _run_json(["search", q, "--limit", n])
        return _items_from_data(data)

    def get_item(self, key: str) -> dict[str, Any] | None:
        """按 item key 获取单条元数据（返回 zotero-cli 的 data，形状透传）。"""
        try:
            data = _run_json(["get", "metadata", key])
        except ZoteroCliError as e:
            print(f"[zotero-cli] get_item({key}) 失败：{e}")
            return None
        return data if isinstance(data, dict) else None

    def search_items(
        self, query: str, *, limit: int = 25, mode: str = "items"
    ) -> list[dict[str, Any]]:
        """按关键词 / 语义 / 标签等检索（mode ∈ items|semantic|tag|advanced|citekey|notes）。"""
        args = ["search"]
        if mode and mode != "items":
            args += ["--mode", mode]
        args += [query, "--limit", str(limit)]
        data = _run_json(args)
        return _items_from_data(data)

    def get_bibtex(self, key: str) -> str:
        """状态：预留，``research`` CLI 未暴露；仅本模块 ``__main__`` 调试后门在用。

        导出单条 BibTeX（``get metadata <key> --format bibtex``）。
        """
        data = _run_json(["get", "metadata", key, "--format", "bibtex"])
        return _text_from_data(data)

    def get_fulltext(self, key: str) -> str:
        """状态：预留，当前无调用方（仅测试覆盖）。

        导出条目全文（``get fulltext <key>``）。

        与 :func:`pdf_extract.extract_pdf` 的职责重叠：后者抽 PDF 得到带 LaTeX 公式的
        markdown，质量高于 Zotero 存的纯文本全文，所以 ``read`` 走的是后者。
        """
        data = _run_json(["get", "fulltext", key])
        return _text_from_data(data)

    def library_info(self) -> dict[str, Any]:
        """状态：预留，当前无调用方（仅测试覆盖）。

        库概况（``library info``）。``research library`` 的四个 action
        （ping / list / search / get）都不用它：「桥能不能通」由 :meth:`ping` 回答，
        而条目计数对文献评估没有贡献。
        """
        data = _run_json(["library", "info"])
        return data if isinstance(data, dict) else {"raw": data}

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
        """由 paper-note frontmatter 建 Zotero 条目（委托 ``add bibtex``）。

        规范化 frontmatter → BibTeX（含自定义增强字段折进 note）后交 zotero-cli 导入。
        返回 ``{"key": <新条目或复用的既有条目 key>, "raw": <zotero-cli data>}``，
        供 research.py 的 :func:`_extract_zotero_key` 提取并回写 frontmatter。
        """
        bib = frontmatter_to_bibtex(meta, tags=tags)
        args = ["add", "bibtex", "--bibtex", bib]
        for c in collections:
            if str(c).strip():
                args += ["-c", str(c)]
        data = _run_json(args)
        return {"key": _extract_key_from_add(data), "raw": data}

    def add_doi(self, doi: str, *, collections: Iterable[str] = ()) -> dict[str, Any]:
        """状态：预留，当前无调用方（仅测试覆盖）。

        按 DOI 建条目（``add doi``，自动抓元数据 + OA PDF）。返回同 create_item_*。

        ``research add <doi>`` 走的是 :meth:`create_item_from_metadata`：它先自己把
        OpenAlex/Crossref 元数据拉齐并过引用核验门，再交给 Zotero；而 ``add doi`` 把
        元数据抓取交给 zotero-cli 的 translator，跳过了那道门。
        """
        args = ["add", "doi", doi]
        for c in collections:
            if str(c).strip():
                args += ["-c", str(c)]
        data = _run_json(args)
        return {"key": _extract_key_from_add(data), "raw": data}

    def add_note(
        self, parent_item_key: str, note_text: str, *, tags: Iterable[str] = ()
    ) -> dict[str, Any]:
        """状态：预留，当前无调用方（仅测试覆盖）。

        为某条目追加子笔记（``notes create --item-key KEY --text ...``）。

        zotero-cli 负责文本 → Zotero 笔记（HTML）转换；本模块不再自带 md→html。
        设计意图是把 ``papers/*.md`` 的评估结论同步成 Zotero 子笔记（在 Zotero 里就能
        读到判断，不用切回仓库），但 ``research`` CLI 当前没有任何把笔记正文写回
        Zotero 的动作：``add`` 只建 / 更新条目元数据（走 :meth:`create_item_from_metadata`）。
        """
        args = ["notes", "create", "--item-key", parent_item_key, "--text", note_text]
        tag_list = [str(t).strip() for t in (tags or []) if str(t).strip()]
        if tag_list:
            args += ["--tags", ",".join(tag_list)]
        data = _run_json(args)
        return data if isinstance(data, dict) else {"raw": data}


# ---------------------------------------------------------------------------
# CLI 自测：连通性
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import sys

    print(
        "[zotero_cli] 调试后门——ping / search / get 三个动作等价于 "
        "`research library ping|search|get`（另有 `library list`，是尽力而为的枚举）。"
        "只有 bibtex 动作在 research CLI 里没有入口：要导出 BibTeX 请用 "
        "`pysci-compose tex refs`（从 Zotero 整库产 refs.bib）"
        "或 zotero MCP 的 zotero_export_bibliography。",
        file=sys.stderr,
    )

    parser = argparse.ArgumentParser(description="zotero-cli 桥接自测")
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("ping", help="检查 zotero-cli 可达性与配置")
    p_search = sub.add_parser("search", help="检索库")
    p_search.add_argument("query")
    p_search.add_argument("--limit", type=int, default=10)
    p_get = sub.add_parser("get", help="获取单条元数据")
    p_get.add_argument("key")
    p_bib = sub.add_parser("bibtex", help="导出单条 BibTeX")
    p_bib.add_argument("key")
    args = parser.parse_args()

    bridge = ZoteroCli()
    if not available():
        print(
            "[zotero-cli] 未安装。运行 scripts/zotero_mcp/setup_zotero_mcp.ps1 "
            "安装 zotero-mcp-server[pdf,scite]。"
        )
        raise SystemExit(1)

    if args.cmd in (None, "ping"):
        print(json.dumps(bridge.ping(), indent=2, ensure_ascii=False))
    elif args.cmd == "search":
        items = bridge.search_items(args.query, limit=args.limit)
        for it in items:
            d = it.get("data", it)
            print(
                f"[{it.get('key', d.get('key', '?'))}] {d.get('itemType', '?')}: "
                f"{str(d.get('title', '?'))[:80]}"
            )
    elif args.cmd == "get":
        print(
            json.dumps(bridge.get_item(args.key), indent=2, ensure_ascii=False)[:2000]
        )
    elif args.cmd == "bibtex":
        print(bridge.get_bibtex(args.key))
