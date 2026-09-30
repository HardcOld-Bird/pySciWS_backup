"""生成账本：每次出图的可复现记录。

AI 出图是**非确定性**的（同 prompt 换 seed/模型/后端结果迥异），且云端生成**要花钱**
（即梦约 0.2 元/张）。故每次生成都必须留痕，才能：

1. **复现**：记下 prompt / seed / model / backend / workflow，日后能原样重跑；
2. **溯源**：一张入库图（assets/gallery）对应哪次生成、参考了哪张图（i2i）；
3. **审计成本**：累计出了多少张、花了多少额度。

双写两份（互为补充）：

- ``manifest.jsonl`` —— 机器可读，一行一条 JSON（追加写，永不重写；供程序过滤/统计）；
- ``LEDGER.md``      —— 人类可读表格（由 manifest 全量重渲染；供 Agent / 用户 Read 速览）。

字段（``Entry``）：``ts / prompt / seed / model / backend / workflow / ref / out / notes``。
其中 ``out`` 是产物图路径，``ref`` 是图生图的源图（可空）。

用法::

    from pysci.skills.ai_drawing.tools.ledger import record, query

    record(prompt="...", seed=42, model="doubao-seedream-4-0-250828",
           backend="comfyui", out="data/skills/ai_drawing/assets/x.png")
    hits = query(backend="comfyui", limit=10)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import settings

#: 机器可读账本（JSON Lines，追加写）。
MANIFEST_NAME = "manifest.jsonl"

#: 人类可读账本（Markdown 表格，由 manifest 全量重渲染）。
LEDGER_NAME = "LEDGER.md"


def _now_iso() -> str:
    """本地时间 ISO8601（秒精度，无时区后缀，便于表格展示）。"""
    return datetime.now().isoformat(timespec="seconds")


@dataclass
class Entry:
    """一条生成记录。

    Attributes:
        ts: 生成时间（ISO8601）。
        prompt: 文生图/图生图用的 prompt（可含正/负向）。
        seed: 随机种子（None 表示未固定/后端未回报）。
        model: 图像模型 ID（如 ``doubao-seedream-4-0-250828``）。
        backend: 出图后端（``comfyui`` | ``imagegen`` | ``manual``）。
        workflow: ComfyUI API 格式工作流配方名/路径（imagegen/manual 可空）。
        ref: 图生图源图路径（文生图为空）。
        out: 产物图路径。
        notes: 自由备注（迭代意图、评估结论等）。
        extra: 其它可复现参数（size/strength/negative…），原样存 JSON。
    """

    prompt: str = ""
    seed: int | None = None
    model: str = ""
    backend: str = ""
    workflow: str = ""
    ref: str = ""
    out: str = ""
    notes: str = ""
    ts: str = field(default_factory=_now_iso)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        """转为可 JSON 序列化的 dict。"""
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> Entry:
        """从 dict 重建（容忍多余/缺失键，向后兼容账本演进）。"""
        known = {f for f in cls.__dataclass_fields__}  # noqa: SLF001
        kwargs = {k: v for k, v in d.items() if k in known}
        return cls(**kwargs)


def manifest_path() -> Path:
    """账本 manifest.jsonl 路径（技能数据区根）。"""
    return settings.module_dir / MANIFEST_NAME


def ledger_path() -> Path:
    """账本 LEDGER.md 路径（技能数据区根）。"""
    return settings.module_dir / LEDGER_NAME


def record(
    *,
    prompt: str = "",
    seed: int | None = None,
    model: str = "",
    backend: str = "",
    workflow: str = "",
    ref: str | Path = "",
    out: str | Path = "",
    notes: str = "",
    ts: str | None = None,
    extra: dict[str, Any] | None = None,
) -> Entry:
    """追加一条生成记录到 manifest.jsonl，并同步重渲染 LEDGER.md。

    所有路径参数会被规范化为**相对项目根**的 POSIX 串（账本可跨机器阅读，不写死绝对路径）。

    Returns:
        写入的 :class:`Entry`。
    """
    entry = Entry(
        prompt=prompt,
        seed=seed,
        model=model,
        backend=backend,
        workflow=str(workflow or ""),
        ref=_rel(ref),
        out=_rel(out),
        notes=notes,
        ts=ts or _now_iso(),
        extra=dict(extra or {}),
    )
    mp = manifest_path()
    mp.parent.mkdir(parents=True, exist_ok=True)
    with mp.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry.to_json(), ensure_ascii=False) + "\n")
    render_markdown()
    return entry


def _rel(p: str | Path) -> str:
    """把路径规范化为相对项目根的 POSIX 串；非路径/空值原样返回空串。"""
    if not p:
        return ""
    path = Path(p)
    try:
        return path.resolve().relative_to(settings.project_root.resolve()).as_posix()
    except (ValueError, OSError):
        return path.as_posix()


def load_all() -> list[Entry]:
    """读取 manifest.jsonl 全部记录（坏行跳过，不整本失败）。"""
    mp = manifest_path()
    if not mp.exists():
        return []
    entries: list[Entry] = []
    with mp.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(Entry.from_json(json.loads(line)))
            except (json.JSONDecodeError, TypeError):
                continue
    return entries


def query(
    *,
    backend: str | None = None,
    model: str | None = None,
    contains: str | None = None,
    limit: int | None = None,
) -> list[Entry]:
    """按条件过滤账本（最新在前）。

    Args:
        backend: 精确匹配 backend（大小写不敏感）。
        model: 子串匹配 model。
        contains: 在 prompt/notes/out 中做子串匹配（大小写不敏感）。
        limit: 最多返回条数（None 为全部）。
    """
    entries = load_all()
    entries.reverse()  # 最新在前
    if backend:
        b = backend.lower()
        entries = [e for e in entries if (e.backend or "").lower() == b]
    if model:
        m = model.lower()
        entries = [e for e in entries if m in (e.model or "").lower()]
    if contains:
        c = contains.lower()
        entries = [
            e
            for e in entries
            if c in (e.prompt or "").lower()
            or c in (e.notes or "").lower()
            or c in (e.out or "").lower()
        ]
    if limit is not None:
        entries = entries[: max(0, int(limit))]
    return entries


def stats() -> dict[str, Any]:
    """账本汇总统计：总数、按 backend/model 计数、有 seed 的条数。"""
    entries = load_all()
    by_backend: dict[str, int] = {}
    by_model: dict[str, int] = {}
    for e in entries:
        by_backend[e.backend or "(unknown)"] = by_backend.get(e.backend or "(unknown)", 0) + 1
        by_model[e.model or "(unknown)"] = by_model.get(e.model or "(unknown)", 0) + 1
    return {
        "total": len(entries),
        "by_backend": by_backend,
        "by_model": by_model,
        "with_seed": sum(1 for e in entries if e.seed is not None),
    }


def _md_escape(s: str) -> str:
    """转义 Markdown 表格单元里的竖线与换行。"""
    return (s or "").replace("|", "\\|").replace("\n", " ").strip()


def render_markdown() -> Path:
    """由 manifest 全量重渲染人类可读的 LEDGER.md（最新在前）。

    Returns:
        写出的 LEDGER.md 路径。
    """
    entries = list(reversed(load_all()))  # 最新在前
    st = stats()
    lines: list[str] = [
        "# AI 绘图生成账本（LEDGER）",
        "",
        f"> 共 {st['total']} 条记录；由 `manifest.jsonl` 自动渲染，请勿手改本文件。",
        "> 追加记录用 `pysci-imagine ledger` 查看、由 `ingest`/`gen`/`i2i` 自动写入。",
        "",
    ]
    if st["by_backend"]:
        backend_str = ", ".join(f"{k}={v}" for k, v in sorted(st["by_backend"].items()))
        lines.append(f"- **按后端**：{backend_str}")
    if st["by_model"]:
        model_str = ", ".join(f"{k}={v}" for k, v in sorted(st["by_model"].items()))
        lines.append(f"- **按模型**：{model_str}")
    lines += ["", "| # | 时间 | 后端 | 模型 | seed | 产物 | prompt / 备注 |",
              "|---|------|------|------|------|------|----------------|"]
    for i, e in enumerate(entries, 1):
        seed = "" if e.seed is None else str(e.seed)
        note = _md_escape(e.prompt)
        if e.notes:
            note = f"{note} — {_md_escape(e.notes)}" if note else _md_escape(e.notes)
        lines.append(
            f"| {i} | {e.ts} | {_md_escape(e.backend)} | {_md_escape(e.model)} "
            f"| {seed} | `{_md_escape(e.out)}` | {note} |"
        )
    if not entries:
        lines.append("| - | - | - | - | - | - | (账本为空) |")
    lp = ledger_path()
    lp.parent.mkdir(parents=True, exist_ok=True)
    lp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return lp


def format_table(entries: list[Entry]) -> str:
    """把记录列表格式化为对齐的纯文本表（供 CLI 打印）。"""
    if not entries:
        return "(账本为空)"
    rows = [
        (
            e.ts,
            e.backend or "-",
            (e.model or "-")[:28],
            "" if e.seed is None else str(e.seed),
            Path(e.out).name if e.out else "-",
            (e.prompt or e.notes or "")[:40],
        )
        for e in entries
    ]
    header = ("时间", "后端", "模型", "seed", "产物", "prompt/备注")
    widths = [max(len(str(r[i])) for r in [header, *rows]) for i in range(6)]
    out = ["  ".join(str(header[i]).ljust(widths[i]) for i in range(6))]
    out.append("  ".join("-" * widths[i] for i in range(6)))
    for r in rows:
        out.append("  ".join(str(r[i]).ljust(widths[i]) for i in range(6)))
    return "\n".join(out)


if __name__ == "__main__":
    print(format_table(query(limit=20)))
