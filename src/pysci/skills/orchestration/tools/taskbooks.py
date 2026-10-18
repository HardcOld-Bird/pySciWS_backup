"""用户任务书：存储 + 状态机（README §12，M3）。

落点 ``data/taskbooks/``（**入 git**，用户裁决 ①：用户可见可编辑）。每件两个文件：

- ``<id>.taskbook.md``：元数据头（ts / title / status / source）＋ 用户**纯文本正文**；
- ``<id>.state.json``：**状态机真源**（status / plan_id / rejections / 时间戳）——任何界面
  （TUI / orch）不得各存一份状态（纪律：状态机真源 = state.json）。

id 形如 ``tb-YYYYMMDD-HHMMSS-<slug>``；写 ``data/`` 前过 :func:`pysci.paths.assert_within_data`。

状态机：``new → planning → active → delivered → accepted | rejected``；``rejected``＝驳回留档
终态（返工走 :func:`create_rework` **另起新任务书**，批语持久留存于 ``state.json`` 的
``rejections``，带 ``round`` 轮次归属）。

本模块是 TUI 与 orch 共用的**数据层**（§2.5：读数进纯函数层，界面只调用）。
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from pysci.paths import DATA_ROOT, assert_within_data

from . import events, statecommit

#: 任务书目录（**入 git**；用户裁决 ① 的落点）。新建文件由 :mod:`statecommit` 落盘即提交
#: （backlog 20261015-085823-devops：此前只声明不提交，远端/新克隆取不到任务书）。
TASKBOOKS_DIR: Path = DATA_ROOT / "taskbooks"

#: 新任务书 / 新报告的落盘即提交信息（:mod:`statecommit`）。
TASKBOOK_COMMIT_MSG: str = "chore(state): 任务书入库（新建即提交）"
REPORT_COMMIT_MSG: str = "chore(state): 任务书全流程报告入库（新报告即提交）"

STATUS_NEW = "new"
STATUS_PLANNING = "planning"
STATUS_ACTIVE = "active"
STATUS_DELIVERED = "delivered"
STATUS_ACCEPTED = "accepted"
STATUS_REJECTED = "rejected"
#: 执行失败终态（PSEP_3 §4.3 #3：两次送审不过 → 中止）。
STATUS_FAILED = "failed"

#: 状态机（唯一真源；越界迁移一律拒绝，不静默改）。
ALLOWED: dict[str, set[str]] = {
    STATUS_NEW: {STATUS_PLANNING, STATUS_ACTIVE, STATUS_REJECTED, STATUS_FAILED},
    STATUS_PLANNING: {
        STATUS_ACTIVE,
        STATUS_DELIVERED,
        STATUS_REJECTED,
        STATUS_NEW,
        STATUS_FAILED,
    },
    STATUS_ACTIVE: {STATUS_DELIVERED, STATUS_REJECTED, STATUS_NEW, STATUS_FAILED},
    STATUS_DELIVERED: {STATUS_ACCEPTED, STATUS_REJECTED},
    STATUS_ACCEPTED: set(),
    STATUS_REJECTED: set(),  # 驳回留档终态（返工另起新任务书 create_rework，不回 new）
    STATUS_FAILED: set(),  # 失败终态（待用户补充指引后另起新任务书）
}

#: 终态（报告快照须随之刷新，见 :func:`transition`）。
_TERMINAL_STATUSES: frozenset[str] = frozenset(
    {STATUS_DELIVERED, STATUS_ACCEPTED, STATUS_REJECTED, STATUS_FAILED}
)

#: 优先级（PSEP_3 §3.1 / 用户裁决⑤⑦）：仅影响**引导顺序**，不抢占在途。
PRIORITY_HIGH = "high"
PRIORITY_NORMAL = "normal"
PRIORITY_LOW = "low"
PRIORITIES: tuple[str, ...] = (PRIORITY_HIGH, PRIORITY_NORMAL, PRIORITY_LOW)
DEFAULT_PRIORITY = PRIORITY_NORMAL

#: 返工三档模式（PSEP_3 §4.1）：完整 / 无监察改进 / 仅返工。
#: **兼容字段**——真源是 :data:`IMPROVE_KEYS` 三标志（用户 2026-10-17 三复选改造），
#: 本串只作旧读取面的派生显示（:func:`legacy_mode_of` / :func:`improve_flags`）。
MODE_FULL = "full"
MODE_NO_INSPECTOR = "no_inspector"
MODE_REWORK_ONLY = "rework_only"
MODES: tuple[str, ...] = (MODE_FULL, MODE_NO_INSPECTOR, MODE_REWORK_ONLY)
DEFAULT_MODE = MODE_FULL

#: 改进项三标志（用户 2026-10-17 裁决：reject 三选一改三复选，8 种组合皆合法）：
#: ``infra``＝A 综合项目改进（含基础设施改进实施）、``leader``＝L 组长改进环节、
#: ``inspector``＝C 监察改进；返工执行 B 恒在。真源字段 ``state["improve"]``。
IMPROVE_INFRA = "infra"
IMPROVE_LEADER = "leader"
IMPROVE_INSPECTOR = "inspector"
IMPROVE_KEYS: tuple[str, ...] = (IMPROVE_INFRA, IMPROVE_LEADER, IMPROVE_INSPECTOR)


def improve_flags(state: dict[str, Any] | None) -> dict[str, bool]:
    """改进项三标志读数：真源 ``improve`` 优先；缺席按旧 ``mode`` 三档映射回落。

    legacy 映射（三复选改造前）：``full``={infra,inspector}、``no_inspector``={infra}、
    ``rework_only``={}（空）；``leader`` 恒 False——旧档没有组长改进。
    """
    out = {k: False for k in IMPROVE_KEYS}
    if not isinstance(state, dict):
        return out
    raw = state.get("improve")
    if isinstance(raw, dict):
        for k in IMPROVE_KEYS:
            out[k] = bool(raw.get(k))
        return out
    mode = str(state.get("mode") or "")
    if mode == MODE_FULL:
        out[IMPROVE_INFRA] = True
        out[IMPROVE_INSPECTOR] = True
    elif mode == MODE_NO_INSPECTOR:
        out[IMPROVE_INFRA] = True
    return out


#: 改进项 → 中文标签（卡片/详情/报告共用）。
IMPROVE_LABELS: dict[str, str] = {
    IMPROVE_INFRA: "基础设施",
    IMPROVE_LEADER: "组长",
    IMPROVE_INSPECTOR: "监察",
}


def improve_label(state: dict[str, Any] | None) -> str:
    """改进项中文标签：「改进：基础设施+监察」；全无 → 「改进：无」（替代旧三档名）。"""
    flags = improve_flags(state)
    names = [IMPROVE_LABELS[k] for k in IMPROVE_KEYS if flags[k]]
    return "改进：" + ("+".join(names) if names else "无")


def legacy_mode_of(flags: dict[str, bool]) -> str:
    """三标志 → 旧三档**派生**（兼容字段 ``mode`` 的写侧）。

    ``leader`` 不参与派生（旧三档没有组长改进的表达位）；仅 inspector 无 infra 的组合
    无对应旧档，派生为 ``rework_only``——真源仍是 ``improve``，此处只保旧显示面不裂。
    """
    infra = bool(flags.get(IMPROVE_INFRA))
    inspector = bool(flags.get(IMPROVE_INSPECTOR))
    if infra and inspector:
        return MODE_FULL
    if infra:
        return MODE_NO_INSPECTOR
    return MODE_REWORK_ONLY


#: 列表三段（§3.4 / §2.4 #1）。
OPEN_STATUSES = (STATUS_NEW, STATUS_PLANNING, STATUS_ACTIVE, STATUS_FAILED)
REVIEW_STATUSES = (STATUS_DELIVERED,)
HISTORY_STATUSES = (STATUS_ACCEPTED,)

#: **状态徽标**（TUI 列表与 CLI 共用真源，用户 2026-10-15「看不出完成与否」）：
#: 每态一个可辨字形 + 状态名——字形让"完成/在做/被拒"一眼可扫，状态名保留精确定义。
STATUS_BADGES: dict[str, str] = {
    STATUS_NEW: "○ new",
    STATUS_PLANNING: "◔ planning",
    STATUS_ACTIVE: "◑ active",
    STATUS_DELIVERED: "◆ delivered",
    STATUS_ACCEPTED: "✔ accepted",
    STATUS_REJECTED: "✗ rejected",
    STATUS_FAILED: "⚠ failed",
}


def status_badge(status: str) -> str:
    """状态徽标串（TUI 列表/详情行首用）。

    未知状态**原样回显**而非吞成占位：状态机新增一态而徽标表未跟随时，界面必须看得见
    真值（否则用户把「未识别的态」读成「没有态」）。
    """
    s = str(status or "")
    return STATUS_BADGES.get(s, s or "?")


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


def md_path(tb_id: str) -> Path:
    """公开 ``<id>.taskbook.md`` 路径（TUI「用 VSCode 打开」的目标）。"""
    return _md_path(tb_id)


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
    priority: str = DEFAULT_PRIORITY,
    improve: dict[str, bool] | None = None,
    mode: str = "",
    rework_of: str = "",
    rework_round: int = 0,
) -> dict[str, Any]:
    """落盘一件任务书（``.taskbook.md`` + ``.state.json``），返回其记录。

    ``source``：``mirror-tui``（用户 TUI 投递）或 ``llm``（组长程序化投递）。
    字段面（PSEP_3 §3.1/§4.1 + 用户 2026-10-17 三标志改造）：``priority``（默认 normal）／
    ``improve``（改进项三标志**真源**）／``mode``（旧三档，仅兼容入参与派生显示）／
    ``rework_of``（原 tb-id）／``rework_round``（轮次）。

    ``improve`` 与 ``mode`` **互斥**（都给 → ValueError）：``improve`` 是真源，``mode``
    只为旧调用面留门（按 :func:`improve_flags` 的 legacy 映射折算）。非返工件 ``mode``
    恒留空（「仅返工」等旧档名挂到新任务书上是误导显示）。

    Raises:
        ValueError: 优先级/模式/标志键非法，或 ``improve`` 与 ``mode`` 同时给出。
    """
    if priority not in PRIORITIES:
        raise ValueError(f"未知优先级：{priority}（合法：{'、'.join(PRIORITIES)}）")
    if improve is not None and mode:
        raise ValueError("improve 与 mode 互斥（improve 是真源，mode 仅为旧接口兼容）")
    if improve is not None:
        unknown = set(improve) - set(IMPROVE_KEYS)
        if unknown:
            raise ValueError(
                f"未知改进标志：{'、'.join(sorted(unknown))}（合法：{'、'.join(IMPROVE_KEYS)}）"
            )
        flags = {k: bool(improve.get(k)) for k in IMPROVE_KEYS}
    elif mode:
        if mode not in MODES:
            raise ValueError(f"未知返工模式：{mode}（合法：{'、'.join(MODES)}）")
        flags = improve_flags({"mode": mode})
    else:
        flags = {k: False for k in IMPROVE_KEYS}
    mode_out = legacy_mode_of(flags) if rework_of else ""
    ts = (now or datetime.now().astimezone()).isoformat(timespec="seconds")
    ident = tb_id or _unique_id(new_id(title, now=now))
    state: dict[str, Any] = {
        "id": ident,
        "ts": ts,
        "title": title,
        "status": STATUS_NEW,
        "source": source,
        "priority": priority,
        "improve": flags,
        "mode": mode_out,
        "rework_of": rework_of,
        "rework_round": int(rework_round),
        "plan_id": "",
        "rejections": [],
        "updated": ts,
    }
    front = [
        "---",
        f"id: {ident}",
        f"ts: {ts}",
        f"title: {title}",
        f"status: {STATUS_NEW}",
        f"source: {source}",
        f"priority: {priority}",
    ]
    if any(flags.values()):
        front.append(f"improve: {','.join(k for k in IMPROVE_KEYS if flags[k])}")
    if mode_out:
        front.append(f"mode: {mode_out}")
    if rework_of:
        front.append(f"rework_of: {rework_of}")
        front.append(f"rework_round: {int(rework_round)}")
    md = "\n".join(front) + "\n---\n\n" + f"{body.rstrip()}\n"
    _write(_md_path(ident), md)
    _write(_state_path(ident), json.dumps(state, ensure_ascii=False, indent=2) + "\n")
    # 「入 git」是声明不是希望：新任务书落盘即提交（两文件一笔），否则远端/新克隆取不到。
    statecommit.commit_face(TASKBOOKS_DIR, TASKBOOK_COMMIT_MSG)
    # 落盘即发 TASKBOOK-NEW（§3.4：用户投递任务书立即唤醒组长）。写失败 fail-open。
    # **返工任务书走专属引导**（用户 2026-10-15 实锤：返工时只给「taskbooks link」通用
    # 引导，组长得自己 grep 才找到 `plan rework`——引导缺口。返工的下一步是**生成返工
    # 骨架**，不是 `plan new`）。
    if rework_of:
        detail = (
            f"返工任务书（第 {int(rework_round)} 轮）已生成：{title}；"
            f"原任务书 {rework_of} 已驳回留档"
        )
        # 唤醒详情**全量附总览**（用户 2026-10-17：从最初要求到历轮批语一次性可见）。
        try:
            digest = requirements_digest(ident)
        except Exception:  # noqa: BLE001 - 事件面绝不阻塞落盘（§0.6）
            digest = ""
        if digest:
            detail += "\n\n" + digest
        action = (
            f"orch plan rework {ident} 生成返工骨架（改进项按任务书标志）"
            "→ 送审 gate 过后 plan advance 推进"
        )
    else:
        detail = f"用户投递了新任务书：{title}（source={source}）"
        action = (
            "orch taskbooks list 查看 → taskbooks link <id> --plan <pid> 拆解推进"
            "（用户 accept/reject 只经 TUI）"
        )
    # serial 排队语义（用户 2026-10-17 裁决）：非队首不推待办——组长侧零规程零决策，
    # 「是否把下一个大任务的待办推到面前」由 orch 内部编排（nextroute 过滤 + 本文案）。
    if execution_mode() == "serial":
        rows = queue_rows()
        pos = next((i for i, r in enumerate(rows) if str(r.get("id")) == ident), None)
        if pos is not None and pos > 0:
            action = (
                f'已排队第 {pos + 1} 位（[taskbooks] execution_mode="serial"）：'
                f"前面还有 {pos} 件未交付；无需行动，"
                "解锁后路由卡片会自动推送拆解待办"
            )
    try:
        events.emit(
            events.TASKBOOK_NEW,
            subject=ident,
            detail=detail,
            action=action,
        )
    except Exception:  # noqa: BLE001 - 事件面绝不阻塞落盘（§0.6）
        pass
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
    """落状态真源，并**即写即提交**该内容面（backlog 20261015-153541-tier-a）。

    为什么改写也要提交（此前只在新文件诞生时提交，见 :mod:`statecommit` docstring）：
    `data/taskbooks/` 是**声明入 git** 的面，而它的 ``state.json`` 由运行期（TUI 推进/批复/
    对账）反复改写 → 忠实停在 main 工作区改动态。后果不是「看着脏」而是**合并被拦**：
    别的成员合并时 `pysci-dev worktree merge` 的交集闸报 rc=2（Tier A 立案的 3 次
    `merge_dirty_face` 里，2026-10-15T13:34 那次正是 `data/taskbooks/tb-…state.json` 相交）。
    ``commit_face`` 只提交**本面**（``data/taskbooks/``），不碰别人的在飞编辑——故这是
    「机械收尾提交」在**内容面**上的安全落点（组长 2026-10-15 的取舍指向此路）。
    """
    state["updated"] = _now()
    _write(_state_path(tb_id), json.dumps(state, ensure_ascii=False, indent=2) + "\n")
    statecommit.commit_face(TASKBOOKS_DIR, TASKBOOK_COMMIT_MSG)
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


def find_by_plan(plan_id: str) -> dict[str, Any] | None:
    """plan → 归属任务书状态（``state.plan_id`` 正查）；无归属（adhoc plan）→ ``None``。"""
    pid = str(plan_id or "")
    if not pid:
        return None
    for st in list_all():
        if str(st.get("plan_id") or "") == pid:
            return st
    return None


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


# ---------------------------------------------------------------------------
# 串行执行模式（用户 2026-10-17 裁决）：队列谓词 + 硬闸文案
# ---------------------------------------------------------------------------
#: 「在途」态：任务书离开这些态即视为**整体完成并交付**（delivered/failed/accepted/
#: rejected 均解锁下一件——用户批复与下一任务并行）。
SERIAL_BLOCKING: frozenset[str] = frozenset(
    {STATUS_NEW, STATUS_PLANNING, STATUS_ACTIVE}
)


def execution_mode() -> str:
    """当前大任务执行模式（``parallel``/``serial``）。

    每次现读 settings（无缓存）＝**热切换**：手改 toml 保存即生效于组长下一回合首个
    命令；谓词无状态，不存在在途锁定。fail-open ``parallel``（配置写坏不锁队列）。
    """
    from . import settings as _settings

    mode, _warnings = _settings.taskbook_execution_mode()
    return mode


def queue_rows() -> list[dict[str, Any]]:
    """serial 队列读数：在途（:data:`SERIAL_BLOCKING`）任务书按**产生时序**编号。

    返回 ``list_all()`` 记录的浅拷贝 + ``queue_no``（1=队首）/``queue_total``。
    返工任务书**不继承家族位置**：``ts`` 是新生时刻，天然落队尾（用户裁决）。
    """
    rows = [
        dict(d) for d in list_all() if str(d.get("status") or "") in SERIAL_BLOCKING
    ]
    total = len(rows)
    for i, row in enumerate(rows):
        row["queue_no"] = i + 1
        row["queue_total"] = total
    return rows


def serial_head() -> dict[str, Any] | None:
    """队首（产生最早的在途任务书）；队列空 → ``None``。"""
    rows = queue_rows()
    return rows[0] if rows else None


def is_unlocked(tb_id: str) -> bool:
    """serial 放行判据：**全部更早**（ts,id 升序）任务书 ∉ :data:`SERIAL_BLOCKING`。

    任务书自身已交付/终态 → 恒放行（它不再占队）；不存在的 id → 放行（fail-soft，
    「不存在」报错交给后续命令自己出）。
    """
    target = str(tb_id)
    for d in list_all():
        did = str(d.get("id") or "")
        if did == target:
            return True
        if str(d.get("status") or "") in SERIAL_BLOCKING:
            return False
    return True


def queue_gate(tb_id: str) -> str:
    """serial 硬闸文案（用户裁决「才允许进行下一个大任务」的机械执行面）。

    Returns:
        空串＝放行；非空＝完整拒绝文案（调用方原样打印后 rc=2，且不执行任何变更）。
        ``parallel`` 模式恒放行——闸只在 serial 下存在。
    """
    if execution_mode() != "serial":
        return ""
    if is_unlocked(tb_id):
        return ""
    rows = queue_rows()
    head = rows[0] if rows else None
    pos = next((r for r in rows if str(r.get("id")) == str(tb_id)), None)
    no = pos.get("queue_no") if pos else "?"
    head_txt = f"{head.get('id')}（status={head.get('status')}）" if head else "（无）"
    return (
        f'[!] 串行执行模式（[taskbooks] execution_mode="serial"）：'
        f"任务书 {tb_id} 暂不可推进（队列第 {no} 位）\n"
        f"    队首={head_txt}；解锁条件：全部更早任务书离开 new/planning/active\n"
        "    （到达 delivered/failed/accepted/rejected 即解锁，用户批复与下一任务并行）\n"
        "    （本命令未执行任何变更）"
    )


def _round_of(state: dict[str, Any]) -> int:
    """返工轮次（坏值按 0——观察面不因一条脏数据塌）。"""
    try:
        return int(state.get("rework_round") or 0)
    except (TypeError, ValueError):
        return 0


def family_label(state: dict[str, Any]) -> str:
    """返工家族成员标签：``rework_round >= 1`` → 「返工第 n 轮」；否则「本体」。"""
    rnd = _round_of(state)
    return f"返工第 {rnd} 轮" if rnd >= 1 else "本体"


def families() -> list[dict[str, Any]]:
    """返工家族归组读数（TUI 任务书列表按家族分组；沿 ``rework_of`` 链）。

    规则：``rework_of`` 为空、或指向不存在 id 的项 = 家族根（**孤儿自成一家**）；每家
    成员 = 沿 ``rework_of`` 的传递闭包（按 parent map 收集，防多子分叉/病态环）；家内
    排序 ``rework_round`` 降序、同轮 ``ts`` 降序 → ``members`` = [最新…本体]（元素为
    state 记录 dict，:func:`family_label` 直接可用）。返回 ``[{root, no, latest, members,
    size}]``（``root``/``latest`` 为 id；``no`` = 家族编号，按 **root 创建序**升序 1..N，
    与展示序解耦故稳定——tui-polish-c 反馈 #9），家族列表按 ``latest`` 的 ts 降序（最新在上）。
    """
    items = list_all()
    by_id = {str(d.get("id")): d for d in items}
    children: dict[str, list[dict[str, Any]]] = {}
    roots: list[dict[str, Any]] = []
    for d in items:
        parent = str(d.get("rework_of") or "")
        if parent and parent in by_id:
            children.setdefault(parent, []).append(d)
        else:
            roots.append(d)
    # 家族编号（tui-polish-c 反馈 #9）：按**家族 root 创建时间**升序 1..N——稳定编号，
    # 与展示序（latest ts 降序）**解耦**：新家族只会递增新号，既有编号不因列表排序/新家族
    # 加入而漂移。本体与返工共享同一号（号属于家族，不属于单件）。
    no_of = {
        str(r.get("id") or ""): i + 1
        for i, r in enumerate(sorted(roots, key=lambda d: str(d.get("ts") or "")))
    }
    out: list[dict[str, Any]] = []
    for root_rec in roots:
        members: list[dict[str, Any]] = []
        seen: set[str] = set()
        stack = [root_rec]
        while stack:
            rec = stack.pop()
            rid = str(rec.get("id") or "")
            if rid in seen:
                continue
            seen.add(rid)
            members.append(rec)
            stack.extend(children.get(rid, []))
        members.sort(key=lambda d: (_round_of(d), str(d.get("ts") or "")), reverse=True)
        latest = members[0]
        out.append(
            {
                "root": str(root_rec.get("id") or ""),
                "no": no_of.get(str(root_rec.get("id") or ""), 0),
                "latest": str(latest.get("id") or ""),
                "members": members,
                "size": len(members),
            }
        )
    out.sort(
        key=lambda fam: str((by_id.get(fam["latest"]) or {}).get("ts") or ""),
        reverse=True,
    )
    return out


def transition(
    tb_id: str, status: str, *, note: str = "", plan_id: str | None = None
) -> dict[str, Any]:
    """受控状态迁移（真源 ``state.json``）；非法目标/非法跃迁 → ``ValueError``。

    ``note`` 追加进 ``rejections``（驳回批语持久留存，带 ``round``＝该批语驱动的返工
    轮次＝被驳回件自身轮次+1——展示层据此给逐条批语标注轮次归属）；``plan_id`` 非空则一并记链接。
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
        st.setdefault("rejections", []).append(
            {"ts": _now(), "note": note, "round": _round_of(st) + 1}
        )
    if plan_id is not None:
        st["plan_id"] = plan_id
    st["status"] = status
    saved = _save_state(tb_id, st)
    # 终态变更 ⇒ **刷新报告快照**（backlog 20261015-230248 ②）：`<id>.report.md` 是
    # delivered 的**交付物**（§4.3 #5），若它还留着旧状态（实测：sync 前生成的快照永远写着
    # `状态：active`），组长批复时读到的就是过期口径。只在快照**已存在**时重建（不因此凭空
    # 造文件）；写失败 fail-soft——留档面绝不阻塞状态迁移。
    if status in _TERMINAL_STATUSES and report_path(tb_id).exists():
        try:
            build_report(tb_id)
        except Exception:  # noqa: BLE001 - 报告面 fail-soft（状态迁移已完成）
            pass
    return saved


def link_plan(tb_id: str, plan_id: str) -> dict[str, Any]:
    """组长链接 plan → 记 ``plan_id`` 并置 ``planning``（§3.4 最小状态推进）。

    任务书勾了「包含组长改进」（improve.leader）→ **自动追加 L 环节**（用户 2026-10-17：
    工具组装、组长零手写；与串行无关）。追加失败不阻塞链接（引导面 fail-soft）。
    """
    saved = transition(tb_id, STATUS_PLANNING, plan_id=plan_id)
    if improve_flags(saved).get(IMPROVE_LEADER):
        try:
            from . import plans as _plans

            _plans.ensure_leader_step(plan_id)
        except Exception:  # noqa: BLE001 - 追加面 fail-soft（链接已完成）
            pass
    return saved


def accept(tb_id: str) -> dict[str, Any]:
    """用户验收（**只经 TUI**，§3.4）：``delivered → accepted``。"""
    return transition(tb_id, STATUS_ACCEPTED)


#: 返工标题后缀（:func:`rework_title` 增补）：重入时先剥旧的再补新的，防逐轮叠加成
#: 「（返工第 1 轮）（返工第 2 轮）」（返工件本身又当原件的链式返工是常规路径）。
_REWORK_SUFFIX_RE = re.compile(r"（返工第 \d+ 轮）\s*$")


def rework_title(base: str, rnd: int) -> str:
    """返工任务书标题：``<原名>（返工第 n 轮）``——同名返工件在列表/详情里可辨。

    用户 2026-10-15：返工任务与原任务同名，列表上一串同名项分不出哪件是哪轮。
    """
    return f"{_REWORK_SUFFIX_RE.sub('', str(base)).rstrip()}（返工第 {int(rnd)} 轮）"


def create_rework(
    orig_id: str,
    *,
    improve: dict[str, bool] | None = None,
    mode: str | None = None,
    priority: str = DEFAULT_PRIORITY,
    note: str = "",
    now: datetime | None = None,
) -> dict[str, Any]:
    """TUI 驳回交付 → 生成**返工大任务书**（PSEP_3 §4.1 + 2026-10-17 三标志改造）。

    ① 原任务书置 ``rejected`` **留档**（批语入 ``rejections``，终态不再回 ``new``）；
    ② 新建任务书继承原**名称/简介**，字段 ``improve``（真源）/``rework_of``/
    ``rework_round(=原轮次+1)``/``priority`` 齐全，入大任务待办（``new``＋TASKBOOK-NEW
    唤醒）。返回**新**任务书状态。

    ``improve``（三复选真源）与 ``mode``（旧三档兼容入口，按 :func:`improve_flags`
    legacy 映射折算）**互斥**；两者都缺 → ``ValueError``（调用面必须显式表态）。

    标题**自动增补「（返工第 n 轮）」**（用户 2026-10-15：返工件与原件同名，列表里分不清
    哪件是哪轮）——只在**数据面**改 title，列表/详情/TUI/卡片均直读该字段，不需各视图另写。

    Raises:
        ValueError: 原任务书不存在 / improve 与 mode 同给或都缺 / 标志或优先级非法。
    """
    orig = load_state(orig_id)
    if orig is None:
        raise ValueError(f"任务书不存在：{orig_id}")
    if improve is not None and mode is not None:
        raise ValueError("improve 与 mode 互斥（improve 是真源，mode 仅为旧接口兼容）")
    if improve is None and mode is None:
        raise ValueError("必须给 improve 或 mode（返工改进项须显式表态）")
    if priority not in PRIORITIES:
        raise ValueError(f"未知优先级：{priority}（合法：{'、'.join(PRIORITIES)}）")
    transition(orig_id, STATUS_REJECTED, note=note)  # 原任务书终态留档
    # 原件已终态 → 报告缺则补建：③历轮收获以各轮报告为食，报告缺位会退化成正文回落。
    if not report_path(orig_id).exists():
        try:
            build_report(orig_id)
        except Exception:  # noqa: BLE001 - 报告面 fail-soft（驳回主流程已完成）
            pass
    rnd = int(orig.get("rework_round") or 0) + 1
    title = rework_title(str(orig.get("title") or "返工大任务"), rnd)
    flags = (
        {k: bool(improve.get(k)) for k in IMPROVE_KEYS}
        if improve is not None
        else improve_flags({"mode": mode})
    )
    label = "、".join(k for k in IMPROVE_KEYS if flags[k]) or "无"
    # 正文＝**要求与历史总览**（用户 2026-10-17：从最初要求到历轮批语集中展示）。
    # 批语不再另段堆叠——它已在总览 ②（带轮次归属）；正文前缀仅一行元数据，不逐代嵌套。
    new_body = (
        f"（返工第 {rnd} 轮，原任务书 {orig_id}；改进项 {label}）\n\n"
        + requirements_digest(orig_id, include_self_harvest=True)
    )
    return create(
        title,
        new_body,
        source="mirror-tui",
        now=now,
        priority=priority,
        improve=flags,
        rework_of=orig_id,
        rework_round=rnd,
    )


def report_path(tb_id: str) -> Path:
    """全流程报告落点（``data/taskbooks/<id>.report.md``；**delivered 交付物即本报告**，§4.3 #5）。"""
    return TASKBOOKS_DIR / f"{tb_id}.report.md"


def _evidence_text(path_str: str) -> str:
    """读 evidence 文件正文（绝对路径原样；相对路径按项目根解析；缺失/坏读返回空串）。"""
    if not path_str:
        return ""
    p = Path(path_str)
    if not p.is_absolute():
        from pysci.paths import PROJECT_ROOT

        p = PROJECT_ROOT / p
    try:
        return p.read_text(encoding="utf-8").strip() if p.is_file() else ""
    except OSError:
        return ""


def _role_texts(pid: str) -> tuple[dict[str, str], list[tuple[str, str]]]:
    """从所链 plan 聚合「叶子 role → evidence 正文」与「叶子 (路径, 状态)」清单。"""
    if not pid:
        return {}, []
    from . import plans as _plans  # 延迟导入：plans 与 taskbooks 无依赖，保持叶子化

    plan = _plans._load_yaml(pid)
    pstate = _plans._load_state(pid)
    rows = {str(s.get("id")): s for s in pstate.get("steps", [])}
    roles: dict[str, str] = {}
    leaves: list[tuple[str, str]] = []
    for leaf in _plans._tree(plan).leaves:
        row = rows.get(leaf.path) or {}
        role = str(leaf.yaml.get("role") or "")
        if role:
            roles[role] = _evidence_text(str(row.get("evidence") or ""))
        leaves.append((leaf.path, str(row.get("status") or "pending")))
    return roles, leaves


# ---------------------------------------------------------------------------
# 要求与历史总览（用户 2026-10-17 裁决：返工上下文「集中展示、相邻、一次性可见」）
# ---------------------------------------------------------------------------
#: ③ 历轮收获摘要的关键词段（＝「那一轮做了什么」的留痕位置；改这里＝改摘要取数面）。
#: 含旧摘要器全键（任务概述/批语/反馈/改进/裁决/结论/总结）+「历轮摘要」（旧键恰好
#: 漏掉它——那正是批语/改进汇总处，2026-10-17 实锤的淹没源之一）。
_HARVEST_SECTIONS: tuple[str, ...] = (
    "任务概述",
    "批语",
    "反馈",
    "历轮摘要",
    "成因分析",
    "解决方案",
    "裁决",
    "改进",
    "结论",
    "总结",
    "返工产物",
    "环节完成清单",
)
#: 「历轮摘要」段**强制整段纳入**（不按段截断，只受总预算约束）。
_HARVEST_FULL_SECTIONS: tuple[str, ...] = ("历轮摘要",)
#: 每段/每轮截断预算（220 字/段的旧预算会把原始要求与批语切掉——用户实锤的淹没源之一）。
_HARVEST_PER_SECTION: int = 500
_HARVEST_PER_ROUND: int = 2500


def family_chain(tb_id: str) -> list[dict[str, Any]]:
    """家族链按轮次升序 ``[本体, 返工1, …, 本件]``（沿 ``rework_of`` 回溯到根再正序）。

    防御：已见 id 不再展开（防病态环）；缺件即止（fail-soft，坏链不塌观察面）。
    """
    st = load_state(tb_id)
    if st is None:
        return []
    chain = [st]
    seen = {str(st.get("id") or tb_id)}
    prev = str(st.get("rework_of") or "")
    while prev and prev not in seen:
        pst = load_state(prev)
        if pst is None:
            break
        seen.add(prev)
        chain.append(pst)
        prev = str(pst.get("rework_of") or "")
    chain.reverse()
    return chain


def digest_has_history(tb_id: str) -> bool:
    """总览是否含历史（家族 >1 件或任一成员带批语）——无历史时展示面可省略总览块。"""
    chain = family_chain(tb_id)
    return len(chain) > 1 or any(m.get("rejections") for m in chain)


def _split_sections(text: str) -> list[tuple[str, str]]:
    """按 ``## `` 标题切段 → ``[(标题, 正文)]``；无标题正文归入标题 ``""`` 的段。"""
    out: list[tuple[str, str]] = []
    title: str = ""
    buf: list[str] = []
    for line in text.splitlines():
        if line.startswith("## "):
            out.append((title, "\n".join(buf).strip()))
            title, buf = line[3:].strip(), []
        else:
            buf.append(line)
    out.append((title, "\n".join(buf).strip()))
    return out


def _harvest_digest(text: str) -> str:
    """一轮的「收获」摘要：关键词段各截前 N 字（无标题整段兜底），总预算截断。"""
    if not text.strip():
        return ""
    picked: list[str] = []
    total = 0
    for title, body in _split_sections(text):
        if not body:
            continue
        if title:
            if not any(k in title for k in _HARVEST_SECTIONS):
                continue
        elif body.lstrip().startswith("#"):
            continue  # 纯标题行（如报告 H1）不入摘要
        full = any(k in title for k in _HARVEST_FULL_SECTIONS)
        label = f"〔{title}〕" if title else ""
        if full:
            chunk = label + body
        else:
            chunk = label + body[:_HARVEST_PER_SECTION]
            if len(body) > _HARVEST_PER_SECTION:
                chunk += "…"
        picked.append(chunk)
        total += len(chunk)
        if total >= _HARVEST_PER_ROUND:
            break
    digest = "\n".join(picked)
    return digest[:_HARVEST_PER_ROUND] + (
        "…" if len(digest) > _HARVEST_PER_ROUND else ""
    )


def _family_notes(
    chain: list[dict[str, Any]],
) -> list[tuple[int, str, str, str]]:
    """全家族批语 ``[(round, ts, member_id, note)]``（按 (round, ts) 升序，逐条带轮次归属）。

    旧数据 ``rejections[]`` 无 ``round`` 字段 → 按「被驳回件自身轮次+1」推算。
    """
    out: list[tuple[int, str, str, str]] = []
    for m in chain:
        mid = str(m.get("id") or "")
        for r in m.get("rejections") or []:
            try:
                round_no = int(r.get("round"))
            except (TypeError, ValueError):
                round_no = _round_of(m) + 1
            out.append(
                (round_no, str(r.get("ts") or ""), mid, str(r.get("note") or ""))
            )
    out.sort(key=lambda t: (t[0], t[1]))
    return out


def _round_label(member: dict[str, Any]) -> str:
    """轮次名：本体=「初版」，返工件=「第 n 轮」。"""
    rnd = _round_of(member)
    return "初版" if rnd <= 0 else f"第 {rnd} 轮"


def requirements_digest(tb_id: str, *, include_self_harvest: bool = False) -> str:
    """要求与历史总览（**单一真源**，从家族链机械重建）。

    ① 用户原始要求（家族根正文全文）→ ② 历轮驳回批语（逐条带轮次归属）→
    ③ 历轮收获（各轮报告的改进/交付摘要，防「已解决问题复发」）。

    **不信任累积正文**：旧返工正文逐代堆叠（重复「## 本轮驳回批语」、前缀嵌套），
    原始要求被淹没——故从 ``rejections[]`` 与各轮报告重建，存量脏数据同样受益。

    Args:
        tb_id: 定位家族的任务书 id（任意成员皆可）。
        include_self_harvest: ③ 是否含**本件自己**那一轮的收获（返工件正文生成时要含
            刚被驳回那一轮的产出摘要；常规展示面不含——流程报告另有全文段）。

    Returns:
        总览 Markdown 块；家族读不到 → 空串。
    """
    chain = family_chain(tb_id)
    if not chain:
        return ""
    root = chain[0]
    parts = ["## 要求与历史总览（自动附送）", "", "### ① 用户原始要求", ""]
    body = read_body(str(root.get("id") or "")) or "（无正文）"
    parts += [body, ""]
    notes = _family_notes(chain)
    if notes:
        parts += ["### ② 历轮驳回批语", ""]
        for round_no, ts, mid, note in notes:
            parts += [
                f"#### 第 {round_no} 轮返工批语（{ts}，驳回 {mid}）",
                "",
                note,
                "",
            ]
    harvest_members = chain if include_self_harvest else chain[:-1]
    harvest_rows: list[str] = []
    for m in harvest_members:
        mid = str(m.get("id") or "")
        rp = report_path(mid)
        try:
            text = rp.read_text(encoding="utf-8") if rp.exists() else ""
        except OSError:
            text = ""
        if not text.strip():
            text = read_body(mid)
        digest = _harvest_digest(text)
        if digest:
            harvest_rows += [f"#### {_round_label(m)}（{mid}）", "", digest, ""]
    if harvest_rows:
        parts += ["### ③ 历轮收获（改进与交付摘要——防复发）", "", *harvest_rows]
    return "\n".join(parts).rstrip() + "\n"


# ---------------------------------------------------------------------------
# 流程报告素材：环节完成清单（TUI 详情页与全流程报告**同一组装器**，§2.5 不留两份）
# ---------------------------------------------------------------------------
#: 交付留档正文在页内摘要的字符上限：全文始终可由留档路径取，摘要够长以致「不必翻文件」
#: （用户 2026-10-15：TUI 里就要能看到交付报告）。
DELIVERY_EXCERPT_CHARS: int = 2000

#: 环节状态字形（与 ``plans.plan_tree_lines`` 同族语义）。值自带**尾随空格**——
#: 图标与文字之间必隔一个空格（终端宽字符重叠修复）；格式化处不再另加。
_STEP_ICONS: dict[str, str] = {
    "done": "✔ ",
    "queued": "▶ ",
    "self_awaiting": "✎ ",
    "blocked": "✗ ",
    "failed": "✗ ",
    "run_failed": "✗ ",
}


def _deliveries_dir() -> Path:
    """交付全文留档目录：**现读** ``queueworker.DELIVERIES_DIR``（测试 monkeypatch 可隔离）。"""
    from . import queueworker

    return Path(queueworker.DELIVERIES_DIR)


def _rel_show(p: Path) -> str:
    """相对项目根的显示串（越界原样返回）。"""
    from .registry import orchestration_rel

    return orchestration_rel(p)


def _delivery_body(text: str) -> str:
    """从留档文件里剥出「交付正文」段（失败交付的「错误正文」另段，不混入）。"""
    _head, sep, rest = text.partition("## 交付正文")
    body = (rest if sep else text).strip()
    return body.split("\n## ", 1)[0].strip()


def _delivery_text(p: Path) -> str:
    """读留档全文（缺读返回空串——读数面 fail-soft）。"""
    try:
        return p.read_text(encoding="utf-8")
    except OSError:
        return ""


def _delivery_excerpt(text: str, name: str, limit: int = DELIVERY_EXCERPT_CHARS) -> str:
    """留档「交付正文」的页内摘要（空正文 → 空串）。"""
    body = _delivery_body(text)
    if not body:
        return ""
    if len(body) > limit:
        return body[:limit].rstrip() + f"…（已截断，全文见 {name}）"
    return body


def _delivery_duration(text: str) -> int:
    """留档头部的 ``- duration_ms: N``（无则 0）。

    为何需要它：台账 ``plan`` 字段在 2026-10-13～2026-10-15 的遮蔽缺陷窗口内落的是席表
    对象（见 :func:`_durations`），那段历史的耗时只能从留档头读——留档头由
    ``queueworker.archive_delivery`` 逐 job 写，与岗位归属无关。
    """
    for ln in text.splitlines():
        if ln.startswith("- duration_ms:"):
            try:
                return int(ln.split(":", 1)[1].strip())
            except ValueError:
                return 0
    return 0


def _delivery_file(plan_id: str, step: str, job_id: str) -> Path | None:
    """环节的交付全文留档：优先 ``<job_id>.md``，否则按 ``*-<plan>-<step>.md`` 后缀反查。"""
    d = _deliveries_dir()
    if job_id:
        p = d / f"{job_id}.md"
        if p.is_file():
            return p
    try:
        hits = sorted(d.glob(f"*-{plan_id}-{step}.md"))
    except OSError:
        return None
    return hits[-1] if hits else None


def _durations(plan_id: str) -> dict[str, int]:
    """环节**累计耗时**（ms）：台账按 ``plan``+``step`` 聚合（重派多跳即累加）。

    读不到即留空——台账 ``plan`` 在 2026-10-13～2026-10-15 的遮蔽缺陷窗口内落的是席表
    对象（非 id），那段历史由 :func:`_delivery_duration` 从留档头补位。
    """
    from .ledger import read_all

    out: dict[str, int] = {}
    for e in read_all():
        if str(getattr(e, "plan", "") or "") != plan_id or not getattr(e, "step", ""):
            continue
        key = str(e.step)
        out[key] = out.get(key, 0) + int(e.duration_ms or 0)
    return out


def flow_steps(plan_id: str) -> list[dict[str, Any]]:
    """所链 plan 的**环节完成清单**（每环节：状态/成员/粗描述/耗时/交付留档路径与正文摘要）。

    交付正文的定位与读取全走 :mod:`..queueworker` 的留档落点（``<job_id>.md``，2026-10-15
    落地的「交付正文全文入 git」）；plan 缺席/坏读 → 空表（观察面 fail-soft，不塌界面）。
    耗时优先取台账（可累积重派），台账无归属时回落**留档头**的 ``duration_ms``。
    """
    if not plan_id:
        return []
    from . import plans as _plans  # 延迟导入：plans 与 taskbooks 无依赖，保持叶子化

    try:
        plan = _plans._load_yaml(plan_id)
        pstate = _plans._load_state(plan_id)
    except Exception:  # noqa: BLE001 - 读数失败不塌详情（§0.6）
        return []
    rows = {str(s.get("id")): s for s in (pstate.get("steps") or [])}
    dur = _durations(plan_id)
    out: list[dict[str, Any]] = []
    for leaf in _plans._tree(plan).leaves:
        row = rows.get(leaf.path) or {}
        sy = leaf.yaml
        f = _delivery_file(plan_id, leaf.path, str(row.get("job_id") or ""))
        text = _delivery_text(f) if f else ""
        out.append(
            {
                "id": leaf.path,
                "status": str(row.get("status") or "pending"),
                "member": str(
                    sy.get("member")
                    or ("self（组长）" if leaf.kind == _plans.KIND_SELF else "?")
                ),
                "brief": str(sy.get("brief") or ""),
                "duration_ms": dur.get(leaf.path) or _delivery_duration(text),
                "evidence": str(row.get("evidence") or ""),
                "delivery_file": str(f) if f else "",
                "delivery_path": _rel_show(f) if f else "",
                "delivery_excerpt": _delivery_excerpt(text, f.name) if f else "",
            }
        )
    return out


def _flow_body(steps: list[dict[str, Any]]) -> str:
    """环节完成清单的 Markdown 正文（报告段与 TUI 页内块共用）。"""
    if not steps:
        return ""
    out: list[str] = []
    for s in steps:
        icon = _STEP_ICONS.get(s["status"], "· ")
        dur = f"{s['duration_ms'] / 1000:.0f}s" if s["duration_ms"] else "—"
        out.append(f"- {icon}{s['id']} [{s['status']}] {s['member']}｜耗时 {dur}")
        if s["brief"]:
            out.append(f"  - 粗描述：{s['brief']}")
        if s["delivery_path"]:
            out.append(
                f"  - 交付正文留档：{s['delivery_path']}"
                f"（页内摘要 {len(s['delivery_excerpt'])} 字）"
            )
            if s["delivery_excerpt"]:
                out.append(
                    "\n".join(f"    {ln}" for ln in s["delivery_excerpt"].splitlines())
                )
        if s["evidence"]:
            out.append(f"  - evidence：{s['evidence']}")
    return "\n".join(out)


def report_sections(tb_id: str) -> list[str]:
    """组装**全流程报告正文**（**纯组装、不落盘**）→ 行文本；``build_report`` 与 TUI 详情页共用。

    通用段（任何任务书都有）：任务概述 / **环节完成清单**（所链 plan 的逐环节状态·成员·耗时·
    交付正文留档路径与页内摘要）。
    返工专属段（仅带 ``rework_of``/``mode`` 的任务书）：历轮摘要 / 本轮成因分析 / 解决方案 /
    裁决书 / 监察判断 / 改进清单 / 返工产物报告——普通任务书省略（省得满屏「（缺：无留痕）」）。

    Raises:
        ValueError: 任务书不存在。
    """
    from . import jury as _jury

    st = load_state(tb_id)
    if st is None:
        raise ValueError(f"任务书不存在：{tb_id}")
    pid = str(st.get("plan_id") or "")
    try:
        roles, leaves = _role_texts(pid)
    except Exception:  # noqa: BLE001 - 计划侧读数坏不塌报告（观察面）
        roles, leaves = {}, []
    steps = flow_steps(pid)
    rework = bool(st.get("rework_of") or st.get("mode"))

    def sec(title: str, body: object) -> str:
        return f"## {title}\n\n{str(body or '').strip() or '（缺：无留痕）'}\n"

    parts: list[str] = [f"# 全流程报告：{st.get('title')}（{tb_id}）", ""]
    overview = (
        f"- 状态：{st.get('status')}；轮次：第 {int(st.get('rework_round') or 1)} 轮；"
        f"模式：{st.get('mode') or '（非返工）'}；优先级：{st.get('priority')}"
    )
    if pid:
        done = sum(1 for s in steps if s["status"] in ("done", "skipped"))
        overview += f"\n- 计划：{pid}（{done}/{len(steps)} 环节完成）"
    body_txt = read_body(tb_id)
    parts.append(sec("任务概述", overview + ("\n\n" + body_txt if body_txt else "")))
    parts.append(sec("环节完成清单", _flow_body(steps)))

    # 「组长改进」段（用户 2026-10-17）：通用段——非返工件含 L 时也出（放在提前 return 前）。
    lead = "\n\n".join(
        t
        for t in (roles.get("leader_reflection"), roles.get("leader_improvement_plan"))
        if t
    )
    if lead:
        parts.append(sec("组长改进", lead))

    if not rework:
        return parts

    hist: list[str] = []
    for m in family_chain(tb_id)[:-1]:
        mid = str(m.get("id") or "")
        notes = "；".join(
            f"第 {round_no} 轮批语：{note}"
            for round_no, _ts, nid, note in _family_notes([m])
            if nid == mid and note
        )
        line = (
            f"- {_round_label(m)} {mid}「{m.get('title')}」状态={m.get('status')}"
            + (f"；{notes}" if notes else "")
        )
        hist.append(line)
    parts.append(sec("历轮摘要", "\n".join(hist) if hist else "（首轮，无历史）"))

    parts.append(sec("本轮成因分析", roles.get("cause", "")))
    parts.append(sec("解决方案", roles.get("solution", "")))

    rulings: list[str] = []
    for cid in (f"{tb_id}-jury", f"{tb_id}-discoverability"):
        rp = _jury.record_path(cid)
        try:
            rec = json.loads(rp.read_text(encoding="utf-8")) if rp.exists() else {}
        except (OSError, ValueError):
            rec = {}
        text = str((rec.get("verdict_summary") or {}).get("text") or "")
        if text:
            rulings.append(f"- {cid}：{text}")
    parts.append(sec("裁决书", "\n".join(rulings) if rulings else "（无送审裁决留痕）"))

    parts.append(sec("监察判断", roles.get("discoverability", "")))
    if roles.get("improvement_plan"):
        parts.append(sec("改进清单", roles["improvement_plan"]))
    elif leaves:
        parts.append(
            sec(
                "改进清单",
                "计划环节（未填派单正文）：\n"
                + "\n".join(f"- {p} [{s}]" for p, s in leaves),
            )
        )
    else:
        parts.append(sec("改进清单", ""))
    parts.append(sec("返工产物报告", roles.get("final_report", "")))
    return parts


def build_report(tb_id: str) -> str:
    """程序化拼装**全流程报告**（PSEP_3 §4.3 #5）→ ``<id>.report.md``；返回正文。

    段落由 :func:`report_sections` 组装（单一实现）；本函数只负责**落盘与回写状态**。

    Raises:
        ValueError: 任务书不存在。
    """
    text = "\n".join(report_sections(tb_id)).rstrip() + "\n"
    _write(report_path(tb_id), text)
    statecommit.commit_face(TASKBOOKS_DIR, REPORT_COMMIT_MSG)
    st = load_state(tb_id) or {}
    st["report"] = str(report_path(tb_id))
    _save_state(tb_id, st)
    return text


def _detail_blocks(tb_id: str) -> list[dict[str, Any]]:
    """详情页段落组装**私件**（:func:`detail_lines` 与 :func:`detail_markdown` 共享，
    改一处两面生效——不留两份实现）。纯读数、无副作用。

    段序（保持原 ``detail_lines`` 语义）：标题/状态徽标 → （delivered 态）**批复参考＝
    流程报告置顶** → 任务书正文 → 批语历史 → 流程报告（非 delivered）→ 返工卡片。
    流程报告走 :func:`report_sections`（**不落盘**——详情页只读，不因看一眼就写出
    ``.report.md``）。每块 ``{"role", ...}``；``header`` 块带结构化字段，其余带 ``lines``。
    """
    st = load_state(tb_id) or {}
    status = str(st.get("status") or "")
    pid = str(st.get("plan_id") or "")
    delivered = status == STATUS_DELIVERED
    header: dict[str, Any] = {
        "role": "header",
        "tb_id": tb_id,
        "status": status,
        "title": str(st.get("title") or ""),
        "plan": "",
    }
    if pid:
        steps = flow_steps(pid)
        done = sum(1 for s in steps if s["status"] in ("done", "skipped"))
        header["plan"] = f"计划：{pid}（{done}/{len(steps)} 环节完成）"
    blocks: list[dict[str, Any]] = [header]
    # 要求与历史总览**置顶**（用户 2026-10-17：原始要求+历轮批语集中展示、正文之前）。
    if digest_has_history(tb_id):
        blocks.append(
            {"role": "digest", "lines": requirements_digest(tb_id).splitlines()}
        )
    try:
        report = report_sections(tb_id)
    except Exception as exc:  # noqa: BLE001 - 报告坏一段不塌详情（观察面 fail-soft）
        report = [f"（流程报告读数失败：{type(exc).__name__}: {exc}）"]
    if delivered:
        blocks.append({"role": "star", "lines": report})
    blocks.append({"role": "body", "lines": [read_body(tb_id) or "（空）"]})
    # 批语历史＝**全家族**逐条（带轮次归属）——只读本件会让进行中返工件显示「（无）」，
    # 与「历史」二字名实不符（用户 2026-10-17 实锤的淹没源之一）。
    notes = _family_notes(family_chain(tb_id))
    blocks.append(
        {
            "role": "rejections",
            "count": len(notes),
            "lines": [
                f"  · 第 {round_no} 轮批语（{ts}，驳回 {mid}）：{note}"
                for round_no, ts, mid, note in notes
            ]
            or ["  （无）"],
        }
    )
    if not delivered:
        blocks.append({"role": "report", "lines": report})
    card = rework_card(tb_id)
    if card:
        blocks.append({"role": "rework", "lines": card})
    return blocks


#: 各段在**文本通道**（:func:`detail_lines`）的标题行；Markdown 通道有自己的呈现。
_DETAIL_HEADS: dict[str, str] = {
    "digest": "—— 要求与历史总览（自动附送）——",
    "star": "★ 批复参考 —— 流程报告（本大任务的流程交付；看毕再批复）",
    "body": "—— 任务书正文 ——",
    "report": "—— 流程报告（所链计划的环节完成清单）——",
    "rework": "—— 返工卡片 ——",
}


def detail_lines(tb_id: str) -> list[str]:
    """任务书**详情页**内容（TUI Enter；纯读数、无副作用；静态通道/测试用）。

    段落组装走 :func:`_detail_blocks`（与 :func:`detail_markdown` 共享单一实现）。
    「批复入口」提示行已按用户明令从此数据面清除（操作提示属界面层，不进读数）。
    """
    out: list[str] = []
    for b in _detail_blocks(tb_id):
        role = b["role"]
        if role == "header":
            out = [
                f"任务书 {b['tb_id']}　[{status_badge(b['status'])}]",
                f"标题：{b['title']}",
            ]
            if b["plan"]:
                out.append(b["plan"])
            continue
        out.append("")
        if role == "rejections":
            out.append(f"—— 批语历史（{b['count']} 条）——")
        else:
            out.append(_DETAIL_HEADS[role])
        out += b["lines"]
    return out


def detail_markdown(tb_id: str) -> str:
    """任务书详情页的 **Markdown 文档**（详情页 Markdown 渲染的单一实现）。

    段落与顺序同 :func:`detail_lines`（共用 :func:`_detail_blocks`）；delivered 态
    「★ 批复参考」置顶。流程报告内嵌为原文（其自带 ``#``/``##`` 标题，渲染器容忍
    嵌套层级）。
    """
    out: list[str] = []
    for b in _detail_blocks(tb_id):
        role = b["role"]
        if role == "header":
            out.append(f"# 任务书 {b['tb_id']}")
            out += ["", f"- 状态：{status_badge(b['status'])}", f"- 标题：{b['title']}"]
            if b["plan"]:
                out.append(f"- {b['plan']}")
        elif role == "digest":
            out += ["", "---", "", *b["lines"]]
        elif role == "star":
            out += ["", "---", "", f"**{_DETAIL_HEADS['star']}**", "", *b["lines"]]
        elif role == "body":
            out += ["", "---", "", "## 任务书正文", "", *b["lines"]]
        elif role == "rejections":
            out += ["", "## 批语历史", "", *b["lines"]]
        elif role == "report":
            out += ["", "---", "", "## 流程报告", "", *b["lines"]]
        elif role == "rework":
            out += ["", "---", "", "## 返工卡片", "", *b["lines"]]
    return "\n".join(out).strip() + "\n"


def rework_card(tb_id: str) -> list[str]:
    """TUI **返工卡片**（§4.3 #8）：轮次/模式/优先级 + 各轮裁决书摘要 + 改进清单 + superstep 进度树。

    仅对带返工字段（``rework_of`` 或 ``mode``）的任务书返回内容；普通任务书返回空表（面板照旧）。
    """
    st = load_state(tb_id)
    if st is None or not (st.get("rework_of") or st.get("mode")):
        return []
    from . import jury as _jury
    from . import plans as _plans

    rnd = int(st.get("rework_round") or 1)
    out = [
        f"轮次：第 {rnd} 轮返工｜{improve_label(st)}"
        f"｜优先级：{st.get('priority') or 'normal'}"
    ]
    if st.get("rework_of"):
        out.append(f"原任务书：{st['rework_of']}")
    out.append("—— 各轮裁决书摘要 ——")
    any_ruling = False
    for m in family_chain(tb_id):
        mid = str(m.get("id") or "")
        for cid in (f"{mid}-jury", f"{mid}-discoverability"):
            rp = _jury.record_path(cid)
            try:
                rec = json.loads(rp.read_text(encoding="utf-8")) if rp.exists() else {}
            except (OSError, ValueError):
                rec = {}
            text = str((rec.get("verdict_summary") or {}).get("text") or "")
            if text:
                out.append(f"  · {cid}：{text}")
                any_ruling = True
    if not any_ruling:
        out.append("  （无送审裁决留痕）")
    pid = str(st.get("plan_id") or "")
    if pid:
        roles, _leaves = _role_texts(pid)
        out.append("—— 改进清单 ——")
        out.append(roles.get("improvement_plan") or "  （无派单正文；见进度树）")
        out.append("—— superstep 进度树 ——")
        out.extend("  " + ln for ln in _plans.plan_tree_lines(pid))
    return out


def sync_active(plan_id: str) -> list[str]:
    """幂等对账：所链 plan **已开始执行**（有环节入队/自办引导）→ 任务书 ``planning → active``。

    ``active`` 的语义＝「计划执行中」，此前**无人触发**（ALLOWED 里有 ``planning → active``
    但全仓无调用点，是死状态）。触发点＝``plans`` 推进首环节的那一刻（入队或 ``kind:self``
    引导），实现只此一处（组长 2026-10-15：单点不复制）。

    Returns:
        本次推进的任务书 id 列表（空=无变化；已是 active 的不重复计入）。
    """
    started: list[str] = []
    for st in list_all():
        if str(st.get("status") or "") != STATUS_PLANNING:
            continue
        if str(st.get("plan_id") or "") != str(plan_id):
            continue
        try:
            transition(str(st["id"]), STATUS_ACTIVE)
        except ValueError:  # 单件坏读数不塌对账（观察面）
            continue
        started.append(str(st["id"]))
    return started


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
            # 交付即补全流程报告（缺则建）：报告是 delivered 的**交付物**，也是下一轮
            # 「③ 历轮收获」的取数源——留待组长手动 report 会让历史摘要退化（2026-10-17）。
            if not report_path(str(st["id"])).exists():
                try:
                    build_report(str(st["id"]))
                except Exception:  # noqa: BLE001 - 报告面 fail-soft（对账已完成）
                    pass
    return done
