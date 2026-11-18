"""笔记 frontmatter 归一化与 INDEX.md 生成（笔记结构化的机械契约）。

ADR（自研理由，2026-11-17）：frontmatter 契约是本技能自定的笔记口径（字段表+status 机，
见 ``.agents/skills/literature-research/references/note-rubric.md``），社区 YAML 库只解决
解析不解决口径；为零运行时依赖（pyyaml 仅 dev 组）自研本平面 ``key: value`` 子集
解析/序列化。子集合同：值为标量或行内列表 ``[a, b]``，不支持嵌套。

笔记布局：``data/skills/literature_research/papers/<key>.note.md``；INDEX.md 由
:func:`build_index` 生成于同目录，勿手改。路径锚点走 :data:`pysci.paths.LITERATURE_ROOT`。
"""

from __future__ import annotations

from pathlib import Path

from pysci.paths import LITERATURE_ROOT

#: 必填字段：缺任一写入前拦截（与 citecheck 门同地位）。
REQUIRED_FIELDS = ("title", "authors", "year")

#: status 状态机取值（unread → skimmed → read → cited，只进不退）。
STATUS_VALUES = ("unread", "skimmed", "read", "cited")

_KEY_ORDER = ("title", "authors", "year", "venue", "doi", "arxiv_id", "status", "tags", "zotero_key", "source")
_DEFAULTS = {"status": "unread", "tags": []}
_FENCE = "---"


def _parse_value(raw: str) -> str | list[str]:
    """标量或行内列表 ``[a, b]`` → str / list[str]。"""
    if raw.startswith("[") and raw.endswith("]"):
        return [item.strip() for item in raw[1:-1].split(",") if item.strip()]
    return raw


def parse_frontmatter(text: str) -> tuple[dict[str, str | list[str]], str]:
    """解析平面 frontmatter 子集。

    Returns:
        ``(meta, body)``。无 frontmatter 或 fence 未闭合时 meta 为空、body 为原文。
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != _FENCE:
        return {}, text
    meta: dict[str, str | list[str]] = {}
    for i, line in enumerate(lines[1:], start=1):
        if line.strip() == _FENCE:
            return meta, "\n".join(lines[i + 1 :]).lstrip("\n")
        if line.strip() and ":" in line:
            key, _, value = line.partition(":")
            meta[key.strip()] = _parse_value(value.strip())
    return {}, text


def normalize_meta(meta: dict[str, object]) -> dict[str, object]:
    """归一化：补默认值、year 取整、tags 小写去重排序、status 校验、键序规范。

    Raises:
        ValueError: 缺必填字段 / year 非数字 / status 越界。
    """
    missing = [k for k in REQUIRED_FIELDS if not str(meta.get(k, "")).strip()]
    if missing:
        raise ValueError(f"笔记 frontmatter 缺必填字段：{missing}")
    out = {**_DEFAULTS, **meta}
    year = str(out["year"]).strip()
    if not year.isdigit():
        raise ValueError(f"year 必须是数字：{year!r}")
    out["year"] = int(year)
    tags = out.get("tags") or []
    if isinstance(tags, str):
        tags = tags.split(",")
    out["tags"] = sorted({t.strip().lower() for t in tags if str(t).strip()})
    status = str(out.get("status", "unread")).strip().lower()
    if status not in STATUS_VALUES:
        raise ValueError(f"status 不在 {STATUS_VALUES} 内：{status!r}")
    out["status"] = status
    out["title"] = str(out["title"]).strip()
    out["authors"] = str(out["authors"]).strip()
    order = {key: i for i, key in enumerate(_KEY_ORDER)}
    return {k: out[k] for k in sorted(out, key=lambda k: (order.get(k, len(order)), k))}


def dump_frontmatter(meta: dict[str, object], body: str = "") -> str:
    """序列化回规范 frontmatter 文本（键序即规范序；tags 回写行内列表）。"""
    lines = [_FENCE]
    for key, value in meta.items():
        rendered = "[" + ", ".join(str(v) for v in value) + "]" if isinstance(value, list) else str(value)
        lines.append(f"{key}: {rendered}")
    lines.append(_FENCE)
    head = "\n".join(lines)
    return f"{head}\n\n{body.strip()}\n" if body.strip() else f"{head}\n"


def normalize_note(text: str) -> str:
    """笔记全文归一化：解析 → normalize_meta → 序列化（正文保留在 frontmatter 之后）。"""
    meta, body = parse_frontmatter(text)
    return dump_frontmatter(normalize_meta(meta), body)


def build_index(notes_dir: Path | None = None) -> str:
    """扫描 ``*.md`` 笔记生成 INDEX.md 内容（key | title | year | status | tags，年份倒序）。

    归一化失败的笔记以 ``INVALID`` 行标注并继续汇总（事后修复，不中断索引）。
    """
    root = notes_dir if notes_dir is not None else LITERATURE_ROOT / "papers"
    rows: list[tuple[int, str, str, str, str, str]] = []
    for path in sorted(root.glob("*.md")):
        if path.name == "INDEX.md":
            continue
        text = path.read_text(encoding="utf-8")
        meta, _ = parse_frontmatter(text)
        if not meta and not text.lstrip().startswith(_FENCE):
            continue  # 提取稿/非笔记（无 frontmatter，如 <key>.md）：不入索引、不算坏笔记
        try:
            norm = normalize_meta(meta)
        except ValueError as exc:
            rows.append((0, path.stem, f"INVALID: {exc}", "", "", ""))
            continue
        rows.append(
            (int(norm["year"]), path.stem, str(norm["title"]), str(norm["status"]), " ".join(norm["tags"]), "")
        )
    rows.sort(key=lambda r: (-r[0], r[1]))
    table = [
        "# INDEX — 文献笔记索引",
        "",
        "> 由 `frontmatter.build_index` 生成，勿手改。",
        "",
        "| key | title | year | status | tags |",
        "|---|---|---|---|---|",
    ]
    for _year, key, title, status, tags, _ in rows:
        table.append(f"| {key} | {title} | {_year or ''} | {status} | {tags} |")
    return "\n".join(table) + "\n"
