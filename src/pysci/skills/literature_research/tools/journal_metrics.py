"""journal_metrics —— 期刊质量指标的免费替代层（SCImago SJR 本地索引）。

背景：官方 JIF / JCR 分区 / JCI / ESI 只能由 **Web of Science Journals API** 给出
（WoS Starter API 不含这些数据；Elsevier/Scopus 需付费凭据，本项目搁置）。在申请落地
之前，期刊质量指标由两个**免费**来源承担：

1. ``scimago_quartile`` —— SCImago Journal & Country Rank 的 SJR 分区（Q1-Q4）。
   源 CSV 由用户手工下载（scimagojr.com 对程序化抓取返回 403），构建成本地紧凑索引后
   按 **ISSN 精确匹配**查询。**本模块负责这一半。**
2. ``journal_tier`` —— 由 OpenAlex ``sources.listed_in`` 的专家评议名单派生，零额外
   请求。实现在 :func:`~.notes.derive_journal_tier`（纯函数，不涉及本地数据文件），
   不在本模块。

**依赖约束**：只用 stdlib（``csv`` / ``json`` / ``re``）+ :mod:`.config`。绝不 import
同层客户端——本模块会被 ``openalex_client`` / ``arxiv_client`` / ``research`` 反向依赖，
import 它们即成环。

**不变量（静默降级）**：查询路径上，索引不存在 / ISSN 无命中 / 索引损坏，一律返回空值
而不抛异常、不阻塞笔记生成。缺数据是常态而非错误；调用方据此把字段留空，绝不编造。
**构建路径相反**：CSV 缺列或解析不出任何条目必须抛异常——静默产出空索引会让所有笔记的
``scimago_quartile`` 悄悄留空，比构建失败难查得多。

**设计取舍**：只按 ISSN 精确匹配，**不做**标题模糊匹配。标题映射会额外增加约 1.5 MB
并引入误匹配风险，而 OpenAlex / Crossref / WoS 的元数据里 ISSN 总是可得。无 ISSN 时
静默返回 ``""``。
"""

from __future__ import annotations

import csv
import io
import json
import re
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import settings

# ---------------------------------------------------------------------------
# 数据来源与归属（写进索引 _meta，也写进 data/SOURCE.md）
# ---------------------------------------------------------------------------
#: SCImago 官方下载页。数据须由用户**手工**下载后把路径传给 ``build_scimago_index``
#: ——该站对程序化抓取返回 403，且下载入口需表单/JS 交互，故本项目不实现自动抓取。
DOWNLOAD_URL = "https://www.scimagojr.com/journalrank.php"

#: 归属声明。SCImago 的数据基于 Scopus（Elsevier B.V.），二次分发须保留此声明。
ATTRIBUTION = (
    "SCImago Journal & Country Rank (https://www.scimagojr.com/journalrank.php), "
    "data based on Scopus (Elsevier B.V.)."
)

#: 索引文件名。真正的路径由 :attr:`~.config.Settings.scimago_index_path` 统一锚定
#: （``settings.module_dir / "data" / scimago_index.json``），此处只作导出用的文档常量。
INDEX_FILENAME = "scimago_index.json"

# ---------------------------------------------------------------------------
# CSV 列名（大小写/空白不敏感；首项为官方导出的拼写）
# ---------------------------------------------------------------------------
_COL_ISSN = ("issn", "e-issn", "eissn")
_COL_SJR = ("sjr", "sjr value")
_COL_QUARTILE = ("sjr best quartile", "best quartile", "quartile")
_COL_HINDEX = ("h index", "h-index", "hindex")

#: 合法分区值。SCImago 对部分新刊/无数据刊给 ``"-"`` 之类，一律按「无分区」处理
#: （但仍保留其 SJR 与 H index，``journal lookup`` 里这两个数依然有参考价值）。
QUARTILES = frozenset({"Q1", "Q2", "Q3", "Q4"})

#: 单条 ISSN 的形态。不带连字符时（``00319007``）与 ``Sourceid`` / ``Total Refs.`` 等
#: 纯数字列无法区分，故本模块**只**在确认取到的是 Issn 列时才接受无连字符形态。
_ISSN_TOKEN_RE = re.compile(r"^\d{4}-?\d{3}[\dXx]$")

#: 从表头的 ``Total Docs. (2024)`` 一类列名里取 SJR 版本年。
_HEADER_YEAR_RE = re.compile(r"\((\d{4})\)")

#: 进程内缓存：``(path, mtime, size) -> payload``。索引约 1.4 MB，逐篇笔记重读一遍
#: JSON 的开销远大于查询本身；缓存键含 mtime/size，故 ``build_scimago_index`` 重写
#: 文件后会自动失效，无需显式清缓存。
_CACHE: dict[tuple[str, float, int], dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# ISSN 归一化
# ---------------------------------------------------------------------------
def normalize_issn(issn: Any) -> str:
    """把 ISSN 归一化为索引 key：只留数字与校验符 ``X``，大写、去连字符与空白。

    ``"0031-9007"`` → ``"00319007"``；``"00319007"`` → ``"00319007"``；
    ``"1079-711x"`` → ``"1079711X"``；``""`` / ``None`` → ``""``。

    不猜测、不补零：非 ISSN 形态的串归一化后自然查不到，静默 miss。
    """
    return re.sub(r"[^0-9Xx]", "", str(issn or "")).upper()


def _issn_candidates(issn: Any) -> list[str]:
    """把「单个 ISSN 或 ISSN 候选列表」摊平为归一化 key 列表（去空、保序去重）。

    ``str`` 本身也是 ``Iterable[str]``，故必须先判类型——否则 ``"0031-9007"`` 会被拆成
    9 个单字符逐个查表，全部 miss 且白白遍历一遍。
    """
    if issn is None:
        return []
    items: Iterable[Any] = (issn,) if isinstance(issn, str) else issn
    try:
        raw = list(items)
    except TypeError:  # 既非字符串也不可迭代（如 int）
        raw = [issn]
    out: list[str] = []
    for it in raw:
        key = normalize_issn(it)
        if key and key not in out:
            out.append(key)
    return out


# ---------------------------------------------------------------------------
# 索引路径与加载
# ---------------------------------------------------------------------------
def index_path() -> Path:
    """SCImago 索引 JSON 的路径（跟随 ``settings.module_dir``，测试可整体重定向）。"""
    return settings.scimago_index_path


def _load_index() -> dict[str, Any] | None:
    """读取索引；文件不存在、不可读或结构损坏时返回 ``None``（静默降级，不抛异常）。"""
    path = index_path()
    try:
        st = path.stat()
    except OSError:
        return None
    key = (str(path), st.st_mtime, st.st_size)
    hit = _CACHE.get(key)
    if hit is not None:
        return hit
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get("by_issn"), dict):
        # 损坏的索引比没有索引更糟：静默当作没有，让 scimago_quartile 留空。
        return None
    _CACHE.clear()  # 只留当前这一份，避免测试反复换路径时累积
    _CACHE[key] = payload
    return payload


# ---------------------------------------------------------------------------
# 查询
# ---------------------------------------------------------------------------
def lookup(issn: Any) -> dict[str, Any] | None:
    """按 ISSN 查 SCImago 记录。

    Args:
        issn: 单个 ISSN 字符串，或候选列表（如 OpenAlex source 的 ``issn`` 数组）；
            传列表时取**首个命中**。

    Returns:
        ``{"issn", "quartile", "sjr", "h_index", "sjr_year"}``；索引未建 / 无命中 /
        输入非法时返回 ``None``。``quartile`` 可能是 ``""``（该刊无分区数据）。
    """
    idx = _load_index()
    if not idx:
        return None
    by_issn: dict[str, Any] = idx["by_issn"]
    sjr_year = (idx.get("_meta") or {}).get("sjr_year")
    for key in _issn_candidates(issn):
        rec = by_issn.get(key)
        if not rec:
            continue
        rec = [*rec, *[None] * max(0, 3 - len(rec))]
        return {
            "issn": key,
            "quartile": rec[0] or "",
            "sjr": rec[1],
            "h_index": rec[2],
            "sjr_year": sjr_year,
        }
    return None


def quartile_for(issn: Any) -> str:
    """按 ISSN 取 SJR 分区（``"Q1"``..``"Q4"``）。

    任何拿不到结果的情形（索引未建、ISSN 缺失、该刊无分区）都返回 ``""`` 而非 ``None``
    ——``scimago_quartile`` 在 frontmatter 与模板里都是字符串字段，空串才能被
    :func:`~.notes.merge_frontmatter` 当作「待填」，``None`` 会与模板默认值不一致。
    """
    return (lookup(issn) or {}).get("quartile") or ""


def status() -> dict[str, Any]:
    """索引状态摘要，供 ``research journal status`` 与 ``research doctor`` 使用。

    Returns:
        ``{"exists", "path", "readable", "n_entries", "sjr_year", "built_at",
        "source_csv", "attribution", "download_url"}``。``exists`` 为假时计数字段为
        ``0``/``None``，调用方据此打印「未建」而不是报错；``exists`` 为真而 ``readable``
        为假表示索引**损坏**，与「未建」区分开，便于用户知道该重建。
    """
    path = index_path()
    out: dict[str, Any] = {
        "exists": path.exists(),
        "path": str(path),
        "readable": False,
        "n_entries": 0,
        "sjr_year": None,
        "built_at": None,
        "source_csv": None,
        "attribution": ATTRIBUTION,
        "download_url": DOWNLOAD_URL,
    }
    if not out["exists"]:
        return out
    idx = _load_index()
    if not idx:
        return out
    meta = idx.get("_meta") or {}
    out.update(
        readable=True,
        n_entries=len(idx.get("by_issn") or {}),
        sjr_year=meta.get("sjr_year"),
        built_at=meta.get("built_at"),
        source_csv=meta.get("source_csv"),
    )
    return out


# ---------------------------------------------------------------------------
# 构建索引
# ---------------------------------------------------------------------------
def _detect_delimiter(first_line: str) -> str:
    """判定 CSV 分隔符：官方导出是**分号**分隔的欧洲风格，也存在逗号分隔的镜像。

    靠表头里出现次数多者判定，不能写死——写死分号会让逗号版整行变成单列，静默产出一个
    空索引（比报错更难查）。
    """
    return ";" if first_line.count(";") >= first_line.count(",") else ","


def _col_index(header: list[str], names: tuple[str, ...]) -> int:
    """按候选名（大小写/空白/引号不敏感）找列下标；找不到返回 ``-1``。"""
    norm = {h.strip().strip('"').lower(): i for i, h in enumerate(header)}
    for n in names:
        if n in norm:
            return norm[n]
    return -1


def _at(row: list[str], i: int) -> str:
    """越界安全地取列。SCImago 导出偶有尾列缺失，不该因此把整行丢掉。"""
    if i < 0 or i >= len(row):
        return ""
    return row[i]


def _to_float(raw: str) -> float | None:
    """把 ``"2.845"`` / ``"2,845"``（欧洲小数逗号）转 float；空串或非法值返回 ``None``。"""
    s = (raw or "").strip().strip('"')
    if not s:
        return None
    if "," in s and "." not in s:
        s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def _to_int(raw: str) -> int | None:
    s = re.sub(r"\D", "", raw or "")
    return int(s) if s else None


def _rehyphen(tok: str) -> str:
    """``"00319007"`` → ``"0031-9007"``，仅为复用同一个校验正则；非 8 位原样返回。"""
    return f"{tok[:4]}-{tok[4:]}" if len(tok) == 8 else tok


def _extract_issns(cell: str) -> list[str]:
    """从 Issn 列的原始文本里抽出全部归一化 ISSN（保序去重）。

    官方 CSV 的多值 ISSN 用 ``;`` 分隔——而 ``;`` 恰好也是字段分隔符，因此当某行的
    Issn 未被引号包住时，多值会被 csv reader 拆成**多个字段**（见
    :func:`build_scimago_index` 里的错位补偿，它会把它们重新拼回来）。分隔符集合刻意
    **不含** ``-``，否则 ``0031-9007`` 会被切成两半。
    """
    keys: list[str] = []
    for tok in re.split(r"[^0-9Xx-]+", cell or ""):
        if not (_ISSN_TOKEN_RE.match(tok) or _ISSN_TOKEN_RE.match(_rehyphen(tok))):
            continue
        key = normalize_issn(tok)
        if key and key not in keys:
            keys.append(key)
    return keys


def _infer_year(header: list[str], csv_path: Path) -> int | None:
    """推断 SJR 版本年：先找表头里 ``Total Docs. (2024)`` 一类括号年，再退回文件名。

    推断不出返回 ``None``（**不编造**），调用方可用 ``--year`` 显式指定。
    """
    years = [int(m.group(1)) for h in header if (m := _HEADER_YEAR_RE.search(h))]
    if years:
        return max(years)
    found = re.findall(r"(?:19|20)\d{2}", csv_path.name)
    return int(found[-1]) if found else None


def build_scimago_index(
    csv_path: str | Path,
    *,
    year: int | None = None,
    out_path: str | Path | None = None,
) -> dict[str, Any]:
    """把官方 SCImago CSV 构建成紧凑的本地索引 JSON 并写盘。

    只保留 ``Issn`` / ``SJR`` / ``SJR Best Quartile`` / ``H index`` 四列（其余列对分区
    判断无用；丢掉可把约 15 MB 的 CSV 压到约 1.4 MB 的 JSON，git 可接受）。``Issn``
    字段可能多值，每个值都建一个 key、指向同一条记录。序列化用 ``separators=(",", ":")``
    且不缩进。

    Args:
        csv_path: 用户手工下载的官方 CSV 路径（见 :data:`DOWNLOAD_URL`）。
        year: SJR 版本年；``None`` 时从表头/文件名推断（见 :func:`_infer_year`）。
        out_path: 索引输出路径；``None`` 时用 :func:`index_path`。测试传 ``tmp_path`` 下
            的路径即可，不必 monkeypatch ``settings``。

    Returns:
        写入的 payload（含 ``_meta`` 与 ``by_issn``）。

    Raises:
        FileNotFoundError: ``csv_path`` 不存在。
        ValueError: CSV 为空、缺必需的列，或一行都没解析出 ISSN。这三种都**必须**抛，
            理由见模块 docstring 的「构建路径相反」。
    """
    src = Path(csv_path)
    if not src.exists():
        raise FileNotFoundError(f"SCImago CSV 不存在：{src}")

    text = src.read_text(encoding="utf-8-sig", newline="")
    delim = _detect_delimiter(text.split("\n", 1)[0])
    rows = [r for r in csv.reader(io.StringIO(text), delimiter=delim) if r]
    if not rows:
        raise ValueError(f"CSV 为空：{src}")
    header, body = rows[0], rows[1:]

    i_issn = _col_index(header, _COL_ISSN)
    i_quart = _col_index(header, _COL_QUARTILE)
    if i_issn < 0 or i_quart < 0:
        raise ValueError(
            f"CSV 缺少必需列（需 Issn 与 SJR Best Quartile）；实际表头：{header[:8]}"
        )
    i_sjr = _col_index(header, _COL_SJR)
    i_h = _col_index(header, _COL_HINDEX)
    width = len(header)

    by_issn: dict[str, list[Any]] = {}
    for row in body:
        # 错位补偿：多值 ISSN 用 `;` 分隔而 `;` 又是字段分隔符，未被引号包住时该行会比
        # 表头多出 k 个字段，且多出的都紧跟在 Issn 列之后。把它们并回 Issn 列，同时把它
        # 右侧的列整体右移 k 位——否则 SJR / Quartile / H index 会全部读错列。
        extra = max(0, len(row) - width)
        if extra:
            issn_cell = ";".join(row[i_issn : i_issn + extra + 1])
        else:
            issn_cell = _at(row, i_issn)
        quartile = _at(row, i_quart + extra if i_quart > i_issn else i_quart)
        quartile = quartile.strip().strip('"').upper()
        if quartile not in QUARTILES:
            quartile = ""
        sjr = _to_float(_at(row, i_sjr + extra if i_sjr > i_issn else i_sjr))
        h_index = _to_int(_at(row, i_h + extra if i_h > i_issn else i_h))

        for key in _extract_issns(issn_cell):
            # 同一本刊可能因历年导出重复出现；保留首个（CSV 按 Rank 升序，首个即最优记录）。
            by_issn.setdefault(key, [quartile, sjr, h_index])

    if not by_issn:
        raise ValueError(f"CSV 里没解析出任何 ISSN（{len(body)} 行数据）：{src}")

    payload = {
        "_meta": {
            "source_csv": src.name,
            "sjr_year": year if year is not None else _infer_year(header, src),
            "built_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "n_entries": len(by_issn),
            "download_url": DOWNLOAD_URL,
            "attribution": ATTRIBUTION,
        },
        "by_issn": by_issn,
    }

    dest = Path(out_path) if out_path else index_path()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    _CACHE.clear()  # 旧路径/旧内容的缓存立即作废
    return payload


# ---------------------------------------------------------------------------
# CLI: 打印索引状态（调试后门；日常请走 `research journal status`）
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    # banner 走 stderr：下方输出的是一段 JSON，一行提示混进 stdout 会让
    # `python -m ... | ConvertFrom-Json` 直接解析失败（其余 9 个后门同此约定）。
    print(
        "[journal_metrics] 调试后门——等价能力请用 `research journal status`"
        "（加 `--json` 得到与下方同形的输出）与 "
        "`research journal build-scimago --csv <path>`；"
        "按 ISSN 并排看 SCImago 分区与 OpenAlex listed_in 分级则是 "
        "`research journal lookup <issn>`。",
        file=sys.stderr,
    )
    print(json.dumps(status(), ensure_ascii=False, indent=2))
