"""生成账本：manifest.jsonl 记录 + PNG tEXt 溯源（脱离账本也能自带出处）。

**账本不是复现手段**——云端生成是非确定的（方舟官方声明固定 seed 也仅"类似结果"）。
账本只做三件事：

1. **溯源**：一张入库图是哪次生成、什么 prompt/模型/参数/参考图；
2. **审计成本**：累计出了多少张、哪个通道/模型最烧钱；
3. **避免重花钱**：好结果立即归档，不指望重跑拿回同一张。

两层溯源（互为兜底）：

- ``manifest.jsonl`` —— 机器可读账本（**追加写，永不重写**；坏行跳过不整本失败）；
- **PNG tEXt** —— ``pysci:*`` 键把 prompt/seed/模型/时间内嵌进产物 PNG
  （ComfyUI/A1111 的通行元数据约定，前缀不撞名）。图被拷走、账本丢了，它仍自带出处。

默认账本在技能数据区 ``data/skills/image_generation/``（:data:`pysci.paths.IMAGEGEN_ROOT`）；
所有函数接受 ``base_dir`` 覆盖（测试/分账本场景）。路径一律记**相对项目根的 POSIX 串**
（账本可跳机器阅读）。

用法::

    from pysci.skills.image_generation import ledger

    ledger.record(prompt="...", model="doubao-seedream-5-5", provider="ark",
                  out="data/research/1_demo/article/artwork/cover.png", seed=None)
    ledger.embed_metadata(".../cover.png",
                          {"prompt": "...", "model": "...", "ts": "..."})
    ledger.query(provider="ark", limit=10)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from pysci.paths import IMAGEGEN_ROOT, PROJECT_ROOT

#: 机器可读账本文件名（JSON Lines，追加写）。
MANIFEST_NAME: str = "manifest.jsonl"

#: PNG tEXt 键前缀（与 ComfyUI/A1111 元数据约定共存不撞名）。
META_PREFIX: str = "pysci"


def _now_iso() -> str:
    """本地时间 ISO8601（秒精度，表格友好）。"""
    return datetime.now().isoformat(timespec="seconds")


@dataclass
class Entry:
    """一条生成记录。

    Attributes:
        ts: 生成时间（ISO8601）。
        prompt: 文生图/图生图用的 prompt（图层/编辑的意图也写这里）。
        seed: 随机种子（None=未固定；方舟档位**不可作复现依据**）。
        model: 模型 ID（如 ``doubao-seedream-5-5``）。
        provider: 出图通道（``ark`` | ``siliconflow`` | ``manual``）。
        ref: 图生图源图（文生图为空）。
        out: 产物图路径（相对项目根 POSIX 串）。
        notes: 自由备注（迭代意图、评估结论）。
        extra: 其它参数（size/usage/url_ttl/snapshot…），原样存 JSON。
    """

    prompt: str = ""
    seed: int | None = None
    model: str = ""
    provider: str = ""
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
        """从 dict 重建（容忍多余/缺失键，账本可演进）。"""
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})


def manifest_path(base_dir: str | Path | None = None) -> Path:
    """账本 manifest.jsonl 路径（``base_dir`` 为账本目录，默认技能数据区根）。"""
    return (Path(base_dir) if base_dir else IMAGEGEN_ROOT) / MANIFEST_NAME


def _rel(p: str | Path) -> str:
    """路径规范化为相对项目根的 POSIX 串；空值/不可相对时尽力而为。"""
    if not p:
        return ""
    path = Path(p)
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except (ValueError, OSError):
        return path.as_posix()


def record(
    *,
    prompt: str = "",
    seed: int | None = None,
    model: str = "",
    provider: str = "",
    ref: str | Path = "",
    out: str | Path = "",
    notes: str = "",
    ts: str | None = None,
    extra: dict[str, Any] | None = None,
    base_dir: str | Path | None = None,
) -> Entry:
    """追加一条生成记录到 manifest.jsonl（批量出图逐张记，写完即生效）。"""
    entry = Entry(
        prompt=prompt,
        seed=seed,
        model=model,
        provider=provider,
        ref=_rel(ref),
        out=_rel(out),
        notes=notes,
        ts=ts or _now_iso(),
        extra=dict(extra or {}),
    )
    mp = manifest_path(base_dir)
    mp.parent.mkdir(parents=True, exist_ok=True)
    with mp.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry.to_json(), ensure_ascii=False) + "\n")
    return entry


def load_all(base_dir: str | Path | None = None) -> list[Entry]:
    """读取账本全部记录（坏行跳过，不整本失败）。"""
    mp = manifest_path(base_dir)
    if not mp.is_file():
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
    provider: str | None = None,
    model: str | None = None,
    contains: str | None = None,
    limit: int | None = None,
    base_dir: str | Path | None = None,
) -> list[Entry]:
    """按条件过滤账本（最新在前）：通道精确匹配、模型/文本子串匹配。"""
    entries = list(reversed(load_all(base_dir)))
    if provider:
        p = provider.lower()
        entries = [e for e in entries if (e.provider or "").lower() == p]
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
    return entries[: max(0, int(limit))] if limit is not None else entries


def stats(base_dir: str | Path | None = None) -> dict[str, Any]:
    """账本汇总：总数、按通道/模型计数、固定 seed 的条数（成本审计速览）。"""
    entries = load_all(base_dir)
    by_provider: dict[str, int] = {}
    by_model: dict[str, int] = {}
    for e in entries:
        by_provider[e.provider or "(unknown)"] = by_provider.get(e.provider or "(unknown)", 0) + 1
        by_model[e.model or "(unknown)"] = by_model.get(e.model or "(unknown)", 0) + 1
    return {
        "total": len(entries),
        "by_provider": by_provider,
        "by_model": by_model,
        "with_seed": sum(1 for e in entries if e.seed is not None),
    }


# ---------------------------------------------------------------------------
# PNG tEXt 元数据内嵌（脱离账本的溯源兜底）
# ---------------------------------------------------------------------------
def embed_metadata(
    path: str | Path, meta: dict[str, Any], *, prefix: str = META_PREFIX
) -> bool:
    """把生成参数写进 PNG 的 tEXt 块，使图片**自带出处**（prompt/seed/模型/时间）。

    仅对 ``.png`` 生效（jpeg 无 tEXt，重存还二次压缩）。已有文本块保留、``pysci:*``
    键**合并覆盖**（旧值留、新值赢）。失败返回 False 不抛——元数据是兜底，
    不该让出图流程失败。
    """
    p = Path(path).expanduser()
    if p.suffix.lower() != ".png" or not p.is_file():
        return False
    try:
        from PIL import Image, PngImagePlugin

        with Image.open(p) as im:
            im.load()
            mode, size, data = im.mode, im.size, im.tobytes()
            pnginfo = PngImagePlugin.PngInfo()
            merged: dict[str, str] = {
                k: v for k, v in (im.info or {}).items() if isinstance(v, str)
            }
        for k, v in (meta or {}).items():
            merged[f"{prefix}:{k}"] = str(v)
        for k, v in merged.items():
            pnginfo.add_text(k, v)
        img = Image.frombytes(mode, size, data)
        img.save(p, format="PNG", pnginfo=pnginfo)
        return True
    except Exception:  # noqa: BLE001
        return False


def read_metadata(path: str | Path, *, prefix: str = META_PREFIX) -> dict[str, str]:
    """读回 :func:`embed_metadata` 写入的元数据（无则空 dict）。"""
    p = Path(path).expanduser()
    if not p.is_file():
        return {}
    try:
        from PIL import Image

        with Image.open(p) as im:
            info = dict(im.info or {})
        head = f"{prefix}:"
        return {
            k[len(head) :]: str(v)
            for k, v in info.items()
            if k.startswith(head) and isinstance(v, str)
        }
    except Exception:  # noqa: BLE001
        return {}


def provenance_for(path: str | Path, *, base_dir: str | Path | None = None) -> dict[str, str]:
    """按产物路径从账本找回 provenance（prompt/model/provider/seed 有值即回）。

    路径按 :func:`record` 同样规范化后匹配 ``out``；命中多条取最后一条（后写更新）。
    查不到返回 ``{}``——兜底手段不该因缺项让调用方失败。
    """
    rel = _rel(path)
    if not rel:
        return {}
    hit: Entry | None = None
    for e in load_all(base_dir):
        if e.out == rel:
            hit = e
    if hit is None:
        return {}
    meta = {
        "prompt": hit.prompt,
        "model": hit.model,
        "provider": hit.provider,
        "seed": "" if hit.seed is None else str(hit.seed),
    }
    return {k: v for k, v in meta.items() if v}
