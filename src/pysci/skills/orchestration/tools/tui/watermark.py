"""主菜单更新水位的落盘面（README §12）——真源 ``state/mirror-seen.json``。

形状：``{"version": 1, "modules": {<module key>: <cursor>}}``。``cursor`` 由各模块自定
（记录流=条数、状态读数=内容指纹）；读侧 :func:`load_seen` **fail-open**（缺失/坏 JSON/
形状不符 → 空表，不阻塞 TUI 启动），写侧 :func:`save_seen` **格式稳定**（键排序 + 缩进 2 +
末尾换行），便于人读与 git diff。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pysci.paths import ORCH_STATE_ROOT

#: 水位文件（运行态；``orchestration/state/`` 下，与其余 state 同族）。
SEEN_PATH: Path = ORCH_STATE_ROOT / "mirror-seen.json"


def load_seen(path: Path | None = None) -> dict[str, str]:
    """读 ``{module: cursor}``；缺失/坏 JSON/形状不符 → ``{}``（fail-open，绝不抛）。"""
    target = path if path is not None else SEEN_PATH
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    mods = data.get("modules") if isinstance(data, dict) else None
    if not isinstance(mods, dict):
        return {}
    return {str(k): str(v) for k, v in mods.items()}


def save_seen(seen: dict[str, str], path: Path | None = None) -> None:
    """写水位：键排序 + 缩进 2 + 末尾换行（格式稳定，人读/diff 友好）。"""
    target = path if path is not None else SEEN_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": 1, "modules": {k: seen[k] for k in sorted(seen)}}
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def fingerprint(lines: list[str]) -> str:
    """无记录流模块的「更新」判据：读数内容**指纹**（变了即视为有更新）。"""
    return hashlib.sha1("\n".join(lines).encode("utf-8")).hexdigest()[:12]
