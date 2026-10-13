"""用户任务书：存储 + 状态机（REFACTOR.md §3.4，M3）。

落点 ``data/taskbooks/``（**入 git**，用户裁决 ①：用户可见可编辑）。每件两个文件：

- ``<id>.taskbook.md``：元数据头（ts / title / status / source）＋ 用户**纯文本正文**；
- ``<id>.state.json``：**状态机真源**（status / plan_id / rejections / 时间戳）——任何界面
  （TUI / orch）不得各存一份状态（纪律：状态机真源 = state.json）。

id 形如 ``tb-YYYYMMDD-HHMMSS-<slug>``；写 ``data/`` 前过 :func:`pysci.paths.assert_within_data`。

状态机：``new → planning → active → delivered → accepted | rejected``；**rejected 带用户批语
回 new**（返工重拆解，批语持久留存于 ``state.json`` 的 ``rejections``）。

本模块是 TUI 与 orch 共用的**数据层**（§2.5：读数进纯函数层，界面只调用）。
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from pysci.paths import DATA_ROOT, assert_within_data

#: 任务书目录（**入 git**；用户裁决 ① 的落点）。
TASKBOOKS_DIR: Path = DATA_ROOT / "taskbooks"

STATUS_NEW = "new"
STATUS_PLANNING = "planning"
STATUS_ACTIVE = "active"
STATUS_DELIVERED = "delivered"
STATUS_ACCEPTED = "accepted"
STATUS_REJECTED = "rejected"

#: 状态机（唯一真源；越界迁移一律拒绝，不静默改）。
ALLOWED: dict[str, set[str]] = {
    STATUS_NEW: {STATUS_PLANNING, STATUS_ACTIVE, STATUS_REJECTED},
    STATUS_PLANNING: {STATUS_ACTIVE, STATUS_DELIVERED, STATUS_REJECTED, STATUS_NEW},
    STATUS_ACTIVE: {STATUS_DELIVERED, STATUS_REJECTED, STATUS_NEW},
    STATUS_DELIVERED: {STATUS_ACCEPTED, STATUS_REJECTED},
    STATUS_ACCEPTED: set(),
    STATUS_REJECTED: {STATUS_NEW},  # 返工重拆解
}

#: 列表三段（§3.4 / §2.4 #1）。
OPEN_STATUSES = (STATUS_NEW, STATUS_PLANNING, STATUS_ACTIVE)
REVIEW_STATUSES = (STATUS_DELIVERED,)
HISTORY_STATUSES = (STATUS_ACCEPTED,)


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _slug(title: str) -> str:
    """标题 → 文件名安全 slug（非 ASCII 折叠为 ``-``，空则 ``task``）。"""
    s = re.sub(r"[^0-9A-Za-z]+", "-", str(title)).strip("-").lower()
    return (s or "task")[:40]


def new_id(title: str, *, now: datetime | None = None) -> str:
    """``tb-YYYYMMDD-HHMMSS-<slug>``。"""
    stamp = (now or datetime.now().astimezone()).strftime("%Y%m%d-%H%M%S")
    return f"tb-{stamp}-{_slug(title)}"


def _md_path(tb_id: str) -> Path:
    return TASKBOOKS_DIR / f"{tb_id}.taskbook.md"


def _state_path(tb_id: str) -> Path:
    return TASKBOOKS_DIR / f"{tb_id}.state.json"


def _write(path: Path, text: str) -> None:
    """写 ``data/`` 下的文件——先过散落护栏（basic.md §5）。"""
    assert_within_data(path, what="任务书")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _unique_id(base: str) -> str:
    """同秒同 slug 撞名 → 顺延 ``-2``/``-3``（秒级 id 不足以区分同刻多件）。"""
    if not _state_path(base).exists() and not _md_path(base).exists():
        return base
    n = 2
    while _state_path(f"{base}-{n}").exists() or _md_path(f"{base}-{n}").exists():
        n += 1
    return f"{base}-{n}"


def create(
    title: str,
    body: str,
    *,
    source: str = "mirror-tui",
    tb_id: str | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """落盘一件任务书（``.taskbook.md`` + ``.state.json``），返回其记录。

    ``source``：``mirror-tui``（用户 TUI 投递）或 ``llm``（组长程序化投递）。
    """
    ts = (now or datetime.now().astimezone()).isoformat(timespec="seconds")
    ident = tb_id or _unique_id(new_id(title, now=now))
    state: dict[str, Any] = {
        "id": ident,
        "ts": ts,
        "title": title,
        "status": STATUS_NEW,
        "source": source,
        "plan_id": "",
        "rejections": [],
        "updated": ts,
    }
    md = (
        "---\n"
        f"id: {ident}\n"
        f"ts: {ts}\n"
        f"title: {title}\n"
        f"status: {STATUS_NEW}\n"
        f"source: {source}\n"
        "---\n\n"
        f"{body.rstrip()}\n"
    )
    _write(_md_path(ident), md)
    _write(_state_path(ident), json.dumps(state, ensure_ascii=False, indent=2) + "\n")
    return state


def load_state(tb_id: str) -> dict[str, Any] | None:
    """读状态（真源）；缺失/坏 JSON → ``None``（fail-soft，不抛）。"""
    p = _state_path(tb_id)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _save_state(tb_id: str, state: dict[str, Any]) -> dict[str, Any]:
    state["updated"] = _now()
    _write(_state_path(tb_id), json.dumps(state, ensure_ascii=False, indent=2) + "\n")
    return state


def read_body(tb_id: str) -> str:
    """读用户纯文本正文（剥掉 frontmatter 头）。"""
    p = _md_path(tb_id)
    try:
        text = p.read_text(encoding="utf-8")
    except OSError:
        return ""
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            return parts[2].strip()
    return text.strip()


def list_all() -> list[dict[str, Any]]:
    """全部任务书状态（按 ts 升序）。坏文件跳过（fail-soft）。"""
    if not TASKBOOKS_DIR.exists():
        return []
    out: list[dict[str, Any]] = []
    for p in sorted(TASKBOOKS_DIR.glob("*.state.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict) and data.get("id"):
            out.append(data)
    out.sort(key=lambda d: (str(d.get("ts", "")), str(d.get("id", ""))))
    return out


def sections() -> dict[str, list[dict[str, Any]]]:
    """列表三段（§3.4 / §2.4 #1）：现有大任务 / 待批复 / 历史（带批语）。"""
    items = list_all()
    return {
        "open": [d for d in items if d.get("status") in OPEN_STATUSES],
        "review": [d for d in items if d.get("status") in REVIEW_STATUSES],
        "history": [
            d
            for d in items
            if d.get("status") in HISTORY_STATUSES or d.get("rejections")
        ],
    }


def transition(
    tb_id: str, status: str, *, note: str = "", plan_id: str | None = None
) -> dict[str, Any]:
    """受控状态迁移（真源 ``state.json``）；非法目标/非法跃迁 → ``ValueError``。

    ``note`` 追加进 ``rejections``（驳回批语持久留存）；``plan_id`` 非空则一并记链接。
    """
    st = load_state(tb_id)
    if st is None:
        raise ValueError(f"任务书不存在：{tb_id}")
    if status not in ALLOWED:
        raise ValueError(f"未知状态：{status}")
    cur = str(st.get("status") or STATUS_NEW)
    if status != cur and status not in ALLOWED.get(cur, set()):
        raise ValueError(f"非法状态迁移：{cur} → {status}")
    if note:
        st.setdefault("rejections", []).append({"ts": _now(), "note": note})
    if plan_id is not None:
        st["plan_id"] = plan_id
    st["status"] = status
    return _save_state(tb_id, st)


def link_plan(tb_id: str, plan_id: str) -> dict[str, Any]:
    """组长链接 plan → 记 ``plan_id`` 并置 ``planning``（§3.4 最小状态推进）。"""
    return transition(tb_id, STATUS_PLANNING, plan_id=plan_id)


def accept(tb_id: str) -> dict[str, Any]:
    """用户验收（**只经 TUI**，§3.4）：``delivered → accepted``。"""
    return transition(tb_id, STATUS_ACCEPTED)


def reject(tb_id: str, note: str) -> dict[str, Any]:
    """用户驳回（**只经 TUI**）：记批语（持久）→ **回 ``new``** 返工重拆解（§3.4）。

    一步完成「rejected 带用户批语回 new」：先把「被驳回」事实落成 ``rejected``，随即迁回
    ``new``；批语留在 ``state.json`` 的 ``rejections`` 里（历史段可见）。
    """
    transition(tb_id, STATUS_REJECTED, note=note)
    return transition(tb_id, STATUS_NEW)


def sync_delivered() -> list[str]:
    """幂等对账：``planning/active`` 且所链 plan **已完成** → 置 ``delivered``。

    Returns:
        本次推进的任务书 id 列表（空=无变化）。plan 完成态读 orch ``plans``（延迟导入，免环）。
    """
    from . import plans as _plans  # 延迟导入：plans 与 taskbooks 无依赖，但保持叶子化

    done: list[str] = []
    for st in list_all():
        if st.get("status") not in (STATUS_PLANNING, STATUS_ACTIVE):
            continue
        pid = str(st.get("plan_id") or "")
        if not pid:
            continue
        try:
            pstate = _plans.load_state(pid)
        except Exception:  # noqa: BLE001 - 对账绝不阻塞（观察面）
            continue
        if isinstance(pstate, dict) and pstate.get("completed_at"):
            try:
                transition(str(st["id"]), STATUS_DELIVERED)
                done.append(str(st["id"]))
            except ValueError:
                continue
    return done
