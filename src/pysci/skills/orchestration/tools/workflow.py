"""改进循环与统计命令：approve/reject/suggestions/consult/stats（README §4.4/§6）。

审批链：dispatch 把 infra_suggestion 持久化到 ``state/suggestions/<id>.md``（pending）
→ 组长 approve（入 backlog + 写审批回复）/ reject（**须 --override 留痕**，写回复 + 落决策日志）
→ 回复在下次派发该成员时自动附送（闭环告知）。护栏级建议不得自行 approve（须转呈用户）——由组长规程软性约束。
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from pysci.paths import ORCH_STATE_ROOT, PROJECT_ROOT, WORKTREES_ROOT

from . import backpressure as _bp
from . import journal, verdict
from .delivery import parse_suggestion_fields
from .dispatch import REPLIES_DIR, conflict_notice, criteria_conflicts, do_dispatch
from .failure_tags import (
    TAGS_PATH,
    failure_counts,
    load_infra_kinds,
    load_tags,
    within_window,
)
from .ledger import (
    LEDGER_PATH,
    format_by_effort,
    format_by_failure,
    format_credits,
    format_est_tokens,
    read_all,
    summarize,
)
from .registry import Registry, orchestration_rel
from .runner import normalize_effort

SUGGESTIONS_DIR = ORCH_STATE_ROOT / "suggestions"
BACKLOG_PATH = ORCH_STATE_ROOT / "backlog.json"

#: 建议文件头部的机器标记行（dispatch 落盘）：读「人读的第一行」时必须整类跳过。
_HEAD_MARKER_RE = re.compile(r"^(kind|delivery|source)\s*[:：]", re.IGNORECASE)


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def pending_suggestions() -> list[Path]:
    """待审批建议文件列表（suggestions/ 顶层 = pending；子目录为已处理归档）。"""
    if not SUGGESTIONS_DIR.exists():
        return []
    return sorted(SUGGESTIONS_DIR.glob("*.md"))


def _find_suggestion(id_prefix: str) -> Path | None:
    hits = [p for p in pending_suggestions() if p.stem.startswith(id_prefix)]
    return hits[0] if len(hits) == 1 else None


def _member_of(sugg: Path) -> str:
    # 文件名约定：<ts>-<member>.md
    parts = sugg.stem.split("-", 2)
    return parts[2] if len(parts) >= 3 else "unknown"


def _archive(sugg: Path, dest_dir: Path) -> Path:
    """把建议移入归档目录，同名已存在时加 ``-2``/``-3``… 序号。

    判决留痕不得互相覆盖：重名覆盖会让上一笔判决的原文消失，而审计要的正是原文。
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / sugg.name
    n = 2
    while target.exists():
        target = dest_dir / f"{sugg.stem}-{n}{sugg.suffix}"
        n += 1
    sugg.rename(target)
    return target


def _backlog_save(data: dict) -> None:
    """把整本 backlog 写回磁盘（与 :func:`_backlog_append` 同一格式，读-改-写用）。"""
    BACKLOG_PATH.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _backlog_append(entry: dict) -> int:
    data = {"version": 1, "items": []}
    if BACKLOG_PATH.exists():
        data = json.loads(BACKLOG_PATH.read_text(encoding="utf-8"))
    items = data.setdefault("items", [])
    items.append(entry)
    _backlog_save(data)
    return len(items)


def backlog_pending() -> list[dict]:
    """backlog 中 status=pending 的条目（FIFO 顺序）。"""
    if not BACKLOG_PATH.exists():
        return []
    data = json.loads(BACKLOG_PATH.read_text(encoding="utf-8"))
    return [i for i in data.get("items", []) if i.get("status") == "pending"]


def backlog_by_status(status: str) -> list[dict]:
    """backlog 中指定 status 的条目（FIFO 顺序）。"""
    if not BACKLOG_PATH.exists():
        return []
    data = json.loads(BACKLOG_PATH.read_text(encoding="utf-8"))
    return [i for i in data.get("items", []) if i.get("status") == status]


def backlog_find(item_id: str) -> dict | None:
    """按 id 取 backlog 条目（交付后核验回路用；找不到返回 None）。"""
    if not BACKLOG_PATH.exists():
        return None
    data = json.loads(BACKLOG_PATH.read_text(encoding="utf-8"))
    return next(
        (i for i in data.get("items", []) if str(i.get("id")) == str(item_id)), None
    )


def backlog_take_first() -> dict | None:
    """取队首 pending 条目标记 in_progress 并返回（devops 派发用）。

    同时记入 ``worktree`` 字段（= 条目 id，或孤儿回收判定旧面不干净后的世代名
    ``<id>-r<gen>``，见 :func:`worktree_name`）：devops 按任务书约定用此名开 worktree，
    worker 中途死亡后 drain 孤儿回收据此定位并清理残留 worktree/branch（见
    :func:`drain.cleanup_orphan_worktrees`）。
    """
    if not BACKLOG_PATH.exists():
        return None
    data = json.loads(BACKLOG_PATH.read_text(encoding="utf-8"))
    for item in data.get("items", []):
        if item.get("status") == "pending":
            item["status"] = "in_progress"
            item["started_at"] = _now()
            item["worktree"] = worktree_name(item)
            BACKLOG_PATH.write_text(
                json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            return item
    return None


def _effort_key(item: dict) -> str:
    """条目 effort 的**组包比较键**：规范档位名，空标注 → ""（=跟随用户级默认）。

    非法标注不抛（fail-soft）：原样作键，让它自己当种子派发时由 do_dispatch 硬失败并
    报出原因——组包阶段抛异常会拖垮整个 drain 循环。
    """
    try:
        return normalize_effort(item.get("effort")) or ""
    except ValueError:
        return str(item.get("effort") or "").strip().lower()


def backlog_take_batch(
    max_batch: int = 4, *, now: float | None = None
) -> tuple[dict | None, list[dict]]:
    """取队首 pending 为**种子**（标记 in_progress），收集同 member（提请者）的 pending
    为**组包**（批量消化，backlog 20261010-batch-digest）。

    启发式：同提请者 ≈ 同模块/同视角（如 figure 的多项常同落 scientific_plotting），
    合并实施省重复上下文；leader 等较杂的组由 devops 的选取权兜底（可只吃种子）。

    Returns:
        ``(seed, group)``：无 pending 时 seed 为 None、group 空。group 为与种子同提请者
        的其余 pending 条目，**保持 pending**（不标 in_progress）——唯种子是本轮承诺单元，
        组包中未被 devops 选取的项留待下轮，worker 死亡也只种子成孤儿。
        ``len(group) <= max_batch - 1``（护栏：单批 ≤ max_batch 项，含种子）。
        member 为空的条目不组包（按单项处理），兼容旧数据与手动入队条目。
        effort 标注不同的条目也不组包（一批只有一跳，跳级参数不能一档多标）。
        ``solo: true`` 的条目**不并入组包也不作组包种子**（batch_size 恒 1）：大单并批
        必触该跳 max_turns 腰斩，尺寸盲区的聚类就是 E4+E5a 连坏的形状（backlog
        20261012-170301-leader）。
    """
    if not BACKLOG_PATH.exists():
        return None, []
    data = json.loads(BACKLOG_PATH.read_text(encoding="utf-8"))
    items = data.get("items", [])
    # **跳过未到期的延期条目**（带宽类失败的主处置，backlog 20261013-193500-leader）：
    # 带宽不足时正确动作是「等」而非换席/烧跳——条目回队并挂 defer_until，
    # 到期前不再被取。不跳就会立刻重取同一条 → 空转。
    now_s = time.time() if now is None else float(now)
    seed = next(
        (
            it
            for it in items
            if it.get("status") == "pending" and not _is_deferred(it, now_s)
        ),
        None,
    )
    if seed is None:
        return None, []
    seed["status"] = "in_progress"
    seed["started_at"] = _now()
    seed["worktree"] = worktree_name(seed)
    member = seed.get("member")
    seed_effort = _effort_key(seed)
    group: list[dict] = []
    if member and max_batch > 1 and not seed.get("salvage") and not seed.get("solo"):
        for it in items:
            if it is seed or it.get("status") != "pending":
                continue
            if it.get("salvage"):
                # 续作单各带自己的抢救面（一跳一树），不组包：合并实施会把两份
                # 未收尾进度塞进同一棵树，指令里的「原地续完」就自相矛盾了
                continue
            if it.get("solo"):
                # 大单不并批（backlog 20261012-170301-leader）：并批必触该跳 max_turns
                # 腰斩，且「没做」伪装成「做了」比 blocked 危险（README §14）
                continue
            # 组包只吃**同档位**：一批只有一跳，effort 是跳级参数——混档会让被标的项
            # 按种子档位跑完，静默污染台账 A/B 数据（backlog 20261010-orch-effort-per-item）
            if _is_deferred(it, now_s):
                continue  # 延期条目到期前不进组包（同上：不许拿去烧跳）
            if it.get("member") == member and _effort_key(it) == seed_effort:
                group.append(it)
                if len(group) >= max_batch - 1:
                    break
    BACKLOG_PATH.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return seed, group


def _is_deferred(item: dict, now_s: float) -> bool:
    """该条目是否仍处在延期待重试窗口内（读侧判据单一源在 :mod:`backpressure`）。"""
    return _bp.item_deferred(item, now_s)


def _backlog_items() -> list[dict]:
    """backlog.json 的 items（缺失/损坏 → 空表；fail-open，观察面不堵派发链）。"""
    if not BACKLOG_PATH.exists():
        return []
    try:
        data = json.loads(BACKLOG_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    items = data.get("items") if isinstance(data, dict) else data
    return items if isinstance(items, list) else []


def next_defer_s(*, now: float | None = None) -> float:
    """到**最近一个**延期条目到期的秒数；没有延期条目 → ``0.0``。

    worker 在「全队皆未到期」时靠它睡到最近到期时刻再查（别空转烧 CPU，也别直接退出
    让队列干等）。判据与成员队列共用 :func:`backpressure.next_defer_s`（§3.3 复用不重写）。
    """
    now_s = time.time() if now is None else float(now)
    return _bp.next_defer_s(_backlog_items(), now=now_s)


def deferred_count(*, now: float | None = None) -> int:
    """仍处在延期待重试窗口内的 pending 条目数（mirror 观察面用）。"""
    now_s = time.time() if now is None else float(now)
    return _bp.deferred_count(_backlog_items(), now=now_s)


def backlog_defer(
    item_id: str,
    *,
    until: float,
    count: int,
    elapsed_s: float,
    note: str = "",
) -> None:
    """把 in_progress 条目复位 pending 并挂上延期（带宽类失败的主处置）。

    与 :func:`backlog_requeue` 的区别：那个是「渠道整体坏了，原地等恢复」；这个是「这一项
    撞上带宽上限，退避一会儿再来」——条目**带账单回队**（``defer_count`` / ``defer_elapsed_s``
    / ``defer_until``），调度侧据此跳过与睡眠，超预算时由调用方改走停止/上报。
    """
    if not BACKLOG_PATH.exists():
        return
    data = json.loads(BACKLOG_PATH.read_text(encoding="utf-8"))
    for item in data.get("items", []):
        if item.get("id") == item_id:
            item["status"] = "pending"
            item.pop("started_at", None)
            item["defer_until"] = float(until)
            item["defer_count"] = int(count)
            item["defer_elapsed_s"] = float(elapsed_s)
            if note:
                item["note"] = note
    BACKLOG_PATH.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def worktree_name(item: dict) -> str:
    """条目对应的隔离 worktree 名：默认=条目 id，孤儿回收后带世代后缀 ``<id>-r<gen>``。

    ``worktree_gen`` 由 :func:`bump_worktree_generation` 写入（缺省=第 1 代）。世代不是
    装饰：上一代的隔离面在被回收时若仍有活着的会话进程写着它（或目录摘不掉），复用同名
    就是让两个写入者落到同一个目录上（backlog 20261012-184000-leader 的并发双写）。
    """
    entry_id = str(item.get("id", ""))
    gen = int(item.get("worktree_gen") or 1)
    return f"{entry_id}-r{gen}" if gen > 1 else entry_id


def bump_worktree_generation(ids: list[str]) -> dict[str, str]:
    """给这些条目开新一代隔离面（``<id>-r<gen>``），返回 ``{id: 新名}``。

    只在「旧会话进程杀不掉」或「旧 worktree 目录摘不掉」时调用——回收者无法让旧写者
    离场，就换一个它不可能已经占着的名字，把并发双写关在门外。世代号从**磁盘上已有的
    最大世代**推进（不是条目计数器自增）：条目可能早被重置，而残留目录还挂在那儿，
    读磁盘事实才不会撞回同一个坑。
    """
    wanted = {str(i) for i in ids if i}
    if not wanted or not BACKLOG_PATH.exists():
        return {}
    data = json.loads(BACKLOG_PATH.read_text(encoding="utf-8"))
    renamed: dict[str, str] = {}
    root = WORKTREES_ROOT
    for item in data.get("items", []):
        entry_id = str(item.get("id", ""))
        if entry_id not in wanted:
            continue
        gen = _next_generation(root, entry_id)
        name = f"{entry_id}-r{gen}"
        item["worktree_gen"] = gen
        item["worktree"] = name
        note = str(item.get("note") or "")
        item["note"] = (
            f"{note}｜隔离重建：第 {gen} 代用 {name}"
            "（上一代仍被活进程或残骸占了同名面，禁止复用旧名）"
        ).strip("｜")
        renamed[entry_id] = name
    if renamed:
        BACKLOG_PATH.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return renamed


def _next_generation(root: Path, entry_id: str) -> int:
    """下一个世代号：裸名（第 1 代）与已有 ``<id>-r*`` 目录里的最大世代 + 1。

    fail-open：目录不可读/名字不合式 → 视作只有第 1 代，返回 2。
    """
    gens = {1}
    try:
        names = {p.name for p in root.glob(f"{entry_id}-r*") if p.is_dir()}
    except OSError:
        names = set()
    for name in names:
        suffix = name[len(entry_id) + 2 :]
        if suffix.isdigit():
            gens.add(int(suffix))
    return max(gens) + 1


def backlog_reclaim_orphans() -> list[str]:
    """把所有 in_progress 条目重置为 pending（孤儿回收），返回回收的 id 列表。

    in_progress 仅由 :func:`backlog_take_first` 标记，而 take_first 只在持锁的 drain
    循环内调用。因此**新 drain 成功获锁之时**，任何 in_progress 条目都必然是上一个
    worker 在 take_first 与 backlog_complete 之间被杀留下的孤儿（锁已判死/超龄才被
    接管）——不回收则永久卡死（take_first 只找 pending）。
    """
    if not BACKLOG_PATH.exists():
        return []
    data = json.loads(BACKLOG_PATH.read_text(encoding="utf-8"))
    reclaimed = []
    for item in data.get("items", []):
        if item.get("status") == "in_progress":
            item["status"] = "pending"
            item["note"] = (
                f"孤儿条目回收（原 started_at={item.pop('started_at', '?')}）"
            )
            item.pop(
                "worktree", None
            )  # 复位为干净 pending（残留 worktree 由 drain 清理）
            reclaimed.append(str(item.get("id", "")))
    if reclaimed:
        BACKLOG_PATH.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return reclaimed


def _backlog_set_status(
    item_id: str,
    status: str,
    note: str = "",
    *,
    stamp: str = "",
    extra: dict | None = None,
) -> None:
    """把 backlog 条目置为指定 status（可选写时间戳字段、附注与额外字段）。"""
    if not BACKLOG_PATH.exists():
        return
    data = json.loads(BACKLOG_PATH.read_text(encoding="utf-8"))
    for item in data.get("items", []):
        if item.get("id") == item_id:
            item["status"] = status
            if stamp:
                item[stamp] = _now()
            if note:
                item["note"] = note
            if extra:
                item.update(extra)
    BACKLOG_PATH.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


#: 「无产物任务」标记（销账锚点三选一的第三项）：条目字段或 note/summary/text 里出现即认。
NO_ARTIFACT_MARKER = "no_artifact"


def delivery_anchor(item: dict) -> tuple[bool, str]:
    """销账锚点校验：**三选一**（backlog 20261015-145953-f4b）。

    实案：组长裁量销账 TUI-UX 单，「账上 done 实际未交付」静默 3 小时（成果卡 worktree）。

    ① **合并提交在 main**：条目 ``commit`` 非空且 ``git merge-base --is-ancestor <sha> main``
       成立（可 git 查——这是「真交付」最强的锚）；
    ② **产物/留档在盘**：``orchestration/state/deliveries/`` 下有该条目的留档（``<item_id>.md``
       或派发留档 ``*-<item_id>.md``——批次派发落的是 ``<member>-<ts>-<item_id>.md``），
       或条目声明的 ``evidence_file`` 文件在盘；
    ③ **显式无产物标记**：条目 ``no_artifact`` 为真，或 ``note``/``summary``/``text`` 里出现
       :data:`NO_ARTIFACT_MARKER`（纯咨询/探针一类本就不产物的任务）。

    **生效边界（爆炸半径处置）**：只作用于**销账这一刻**（:func:`backlog_complete` /
    :func:`backlog_resolve`），**不回溯改判**已 done 的历史条目——历史维持原状，新销账才受校验。

    Returns:
        ``(True, 命中说明)`` 或 ``(False, 缺什么)``——后者用于点名提示与 needs_leader note。
    """
    iid = str(item.get("id") or "")
    # ③ 显式无产物标记（先查它：纯咨询/探针不必去碰 git 与磁盘）
    if item.get("no_artifact"):
        return True, "显式无产物标记（no_artifact）"
    blob = " ".join(str(item.get(k) or "") for k in ("note", "summary", "text"))
    if NO_ARTIFACT_MARKER in blob:
        return True, f"显式无产物标记（{NO_ARTIFACT_MARKER}）"
    # ① 合并提交在 main
    sha = str(item.get("commit") or "").strip()
    if sha:
        try:
            from . import drain as _drain

            rc, _out = _drain._run_git("merge-base", "--is-ancestor", sha, "main")
            if rc == 0:
                return True, f"合并提交在 main（{sha[:12]}）"
        except Exception:  # noqa: BLE001 - git 不可用不该让销账卡死（退到②③判）
            pass
    # ② 产物/留档在盘（**现读** queueworker.DELIVERIES_DIR：测试可 monkeypatch 隔离该面，
    #    与 taskbooks._deliveries_dir 同口径；模块级 ORCH_STATE_ROOT 会被隔离面绕过）
    try:
        from . import queueworker as _qw

        root = Path(_qw.DELIVERIES_DIR)
    except Exception:  # noqa: BLE001 - 读不到落点就不做这半支（退到①③判）
        root = ORCH_STATE_ROOT / "deliveries"
    if iid:
        cands = [root / f"{iid}.md"]
        try:
            cands += sorted(root.glob(f"*-{iid}.md"))
        except OSError:
            pass
        for p in cands:
            if p.is_file():
                return True, f"交付留档在盘（{p.name}）"
    ev = str(item.get("evidence_file") or "").strip()
    if ev:
        p = Path(ev)
        p = p if p.is_absolute() else PROJECT_ROOT / p
        if p.is_file():
            return True, f"声明的证据在盘（{ev}）"
    if sha:
        return False, (
            f"commit {sha[:12]} 不在 main（已提交未合并 ≠ 已交付），且无留档/证据在盘"
        )
    return False, (
        "无锚点：既无「合并提交在 main」的 commit，也无 deliveries/ 留档或声明的证据在盘；"
        f"若本条目本就无产物，请在 note/summary 里写 {NO_ARTIFACT_MARKER} 标记"
    )


def backlog_complete(
    item_id: str,
    note: str = "",
    *,
    batch_size: int | None = None,
    batch_seed: str | None = None,
) -> bool:
    """标记 backlog 条目完成（devops 交付后由组长/orch 调用）。

    **先过销账锚点校验**（backlog 20261015-145953-f4b，:func:`delivery_anchor`）：无锚 →
    **拒绝 done** 并转 ``needs_leader`` 出声（点名缺什么），**不静默**——「账上 done 实际未
    交付」曾静默 3 小时。返回 ``True``＝已销账，``False``＝被锚点校验拦下（转 needs_leader）。

    batch_size/batch_seed：批量消化时记录本条目所属批次（同批一起 done 的条目数与
    种子 id），供 orch stats 批量大小分布与审计；单条消化时留空。

    E5b（README §5.3 核验闭环）：done 之后自动为带 M 通道（``--bench``）的条排出 A/B 核验项——
    核验方式在审批时一次定死，交付后不再有人裁决，所以这一步必须机械发生。核验入队
    失败不能让销账本身失败（条目已交付是事实），故 fail-soft 打印并继续。
    """
    ok, why = delivery_anchor(backlog_find(item_id) or {})
    if not ok:
        backlog_needs_leader(
            item_id, note=f"[销账锚点校验] {why}——未销账", extra={"anchor_missing": why}
        )
        print(f"[!] 拒绝销账 {item_id}：{why}（已转 needs_leader，补齐锚点后再销）")
        return False
    extra: dict = {}
    if batch_size is not None:
        extra["batch_size"] = batch_size
    if batch_seed is not None:
        extra["batch_seed"] = batch_seed
    _backlog_set_status(item_id, "done", note, stamp="done_at", extra=extra or None)
    entry = backlog_find(item_id)
    if not entry:
        return True
    try:
        item = verdict.post_verify_enqueue(entry)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"[!] 交付后核验入队失败（{item_id}）：{type(exc).__name__}: {exc}")
        return True
    if item:
        print(
            f"    已排交付后核验：{item['id']}（label={item['label']}，"
            f"a={item['ref_a'][:8]} → b=HEAD）——判决由 bencher 消化后自动落地"
        )
    return True


def backlog_needs_leader(
    item_id: str, note: str = "", *, extra: dict | None = None
) -> None:
    """标记 backlog 条目需组长介入（drain 单项失败跳过时用；不再自动重试）。

    extra：失败成因的结构化字段（如交集类 blocked 的 ``merge_intersect`` 待清文件清单），
    让「该谁动手清什么」在 backlog 面上可机械读，不必从 note 全文里再解析一遍。
    """
    _backlog_set_status(
        item_id, "needs_leader", note, stamp="needs_leader_at", extra=extra or None
    )


def backlog_requeue(item_id: str, note: str = "") -> None:
    """把 in_progress 条目复位为 pending（清除 started_at）。

    用于**系统性故障**（如额度耗尽）导致的全局停止：条目本身没问题，应回队等待
    渠道恢复后续消化，而非标记 needs_leader 污染 backlog 语义。
    """
    if not BACKLOG_PATH.exists():
        return
    data = json.loads(BACKLOG_PATH.read_text(encoding="utf-8"))
    for item in data.get("items", []):
        if item.get("id") == item_id and item.get("status") == "in_progress":
            item["status"] = "pending"
            item.pop("started_at", None)
            if note:
                item["note"] = note
    BACKLOG_PATH.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _backlog_transition(
    item_id: str,
    *,
    allowed_from: set[str],
    to_status: str,
    note: str = "",
    stamp: str = "",
) -> dict | None:
    """受控状态迁移：**仅当**条目当前 status ∈ ``allowed_from`` 时改 ``to_status``。

    与 :func:`backlog_requeue`（只看 in_progress）不同，本函数给组长面的
    `orch requeue`/`orch resolve` 共用——那两个动作都必须**核对起点状态**（别把 pending 的
    条目 "销账"、别对已 done 的条目 "回排"），故起点不在允许集就**不动**、返回 None。
    ``note`` 非空才覆盖（留空=保留原附注）；``stamp`` 非空记一个时间戳字段。

    Returns:
        迁移后的条目（磁盘副本）；未找到 / 起点状态不符 → None（未写盘）。
    """
    if not BACKLOG_PATH.exists():
        return None
    data = json.loads(BACKLOG_PATH.read_text(encoding="utf-8"))
    target: dict | None = None
    for item in data.get("items", []):
        if str(item.get("id")) != str(item_id):
            continue
        if str(item.get("status")) not in allowed_from:
            return None
        item["status"] = to_status
        if stamp:
            item[stamp] = _now()
        if note:
            item["note"] = note
        target = item
        break
    if target is None:
        return None
    BACKLOG_PATH.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return target


def backlog_requeue_needs_leader(item_id: str, *, note: str = "") -> dict | None:
    """``orch requeue``：needs_leader → pending，供组长回排重派（backlog 20261013-215909-leader）。

    与 :func:`backlog_requeue`（系统性故障的 in_progress→pending 复位）分工：本函数专治
    组长对 **needs_leader** 条目的回排。``--note`` 覆盖条目附注（补授权/重排说明）——带
    ``awaiting_decision`` 的条目，note 即重排任务书里「本次补齐情况」的来源
    （:func:`_awaiting_decision_instruction`）；不传则保留原附注（drain 的失败全文）。
    原 needs_leader 事实不抹（``needs_leader_at`` 等字段原样留着，git 历史另有记录）。

    Returns:
        迁移后的条目；未找到 / 状态不是 needs_leader → None。
    """
    return _backlog_transition(
        item_id, allowed_from={"needs_leader"}, to_status="pending", note=note
    )


def backlog_resolve(item_id: str, *, note: str = "") -> dict | None:
    """``orch resolve``：组长确认 needs_leader 条目**已达标** → done（backlog 20261013-215909-leader）。

    「原 blocked 事实」不删：只改 status 并追加销账 note（stamp ``done_at``），
    ``needs_leader_at``/``blocked_report`` 等字段原样留着。这是「组长手动把已消解条目
    收尾」的正门，取代直改 ``state/backlog.json`` 的野路子（drain 在跑时有并发覆盖风险）。

    **销账锚点校验**（backlog 20261015-145953-f4b）：组长裁量销账同样过
    :func:`delivery_anchor`——无锚即**拒绝**（返回 ``None``，条目留在 needs_leader）并点名
    缺什么，**不静默**（实案：组长裁量销账 TUI-UX 单，「账上 done 实际未交付」静默 3 小时）。

    Returns:
        迁移后的条目；未找到 / 状态不是 needs_leader / 无锚点 → ``None``。
    """
    item = backlog_find(item_id) or {}
    ok, why = delivery_anchor(item)
    if not ok:
        print(
            f"[!] 拒绝销账 {item_id}：{why}（条目留在 needs_leader，补齐锚点后再 resolve）"
        )
        return None
    return _backlog_transition(
        item_id,
        allowed_from={"needs_leader"},
        to_status="done",
        note=note,
        stamp="done_at",
    )


# ---------------------------------------------------------------------------
# 抢救续作面（backlog 20261012-234000-leader：崩溃/腰斩留下的未收尾进度自动接手续完）
# ---------------------------------------------------------------------------
#: 抢救链深度上限：同一份工作最多自动续作几跳。到顶就不再自动立单、只在 note 留话——
#: 一个反复修不好的单若每跳都能再生一单，队列就成了自我复制的永动机。
SALVAGE_MAX_CHAIN: int = 3

#: 续作条目的来源标记（与 Tier A 自动立案的 ``source: tier-a`` 同字段同意图：复盘时一眼
#: 看出这条不是组员提的建议，是机器接的手）。
SALVAGE_SOURCE = "salvage"

#: 抢救记录的键（drain 侧构造，任务书 / status / watch 三处读取；缺键只降级显示不抛）。
SALVAGE_FIELDS: tuple[str, ...] = ("worktree", "branch", "commit", "path", "reason")


def _salvage_note(record: dict) -> str:
    """抢救事实的单行注记（进条目 note——有人手读 backlog 时不必再挖 JSON 字段）。"""
    return (
        f"｜抢救续作：面={record.get('worktree', '?')} 分支={record.get('branch', '?')} "
        f"提交={record.get('commit') or '?'} 成因={record.get('reason', '?')}"
        + ("（未提交部分未能自动取回）" if record.get("uncommitted") else "")
    )


def _salvage_entry_id(items: list[dict]) -> str:
    """续作条目 id：``<ts>-salvage``，同秒撞名顺延 ``.<n>``（Tier A 立案同法）。"""
    taken = {str(i.get("id", "")) for i in items}
    base = f"{_filed_ts()}-salvage"
    if base not in taken:
        return base
    n = 2
    while f"{base}.{n}" in taken:
        n += 1
    return f"{base}.{n}"


def backlog_attach_salvage(name: str, record: dict) -> dict | None:
    """把抢救事实挂到**仍在队列里**的条目上（崩溃回收后会被自动重派的那条）。

    匹配判据：status ∈ {pending, in_progress}，且 ``name`` 等于条目 id 或它的世代名
    ``<id>-r<gen>``（回收先把孤儿复位 pending，再由本函数注记，所以这里看到的 pending
    就是那条待续完的原始单）。needs_leader **不匹配也不改状态**——「须组长处理」是已经
    做出的裁决，代码无权撤销；那种收场走 :func:`backlog_append_salvage` 另立续作单。

    Args:
        name: 抢救时的隔离面名（= 条目 id 或 ``<id>-r<gen>``）。
        record: :func:`drain.salvage_worktree` 的读数（含 commit/branch/path）。

    Returns:
        注记成功的条目（磁盘副本）；没匹配到、或同一提交已挂过（幂等）→ None。
    """
    if not name or not record or not BACKLOG_PATH.exists():
        return None
    data = json.loads(BACKLOG_PATH.read_text(encoding="utf-8"))
    hit = None
    for item in data.get("items", []):
        entry_id = str(item.get("id", ""))
        if not entry_id:
            continue
        if name != entry_id and not name.startswith(f"{entry_id}-r"):
            continue
        if item.get("status") not in ("pending", "in_progress"):
            continue
        hit = item
        break
    if hit is None:
        return None
    prev = hit.get("salvage") or {}
    if str(prev.get("commit", "")) == str(record.get("commit", "")):
        return None  # 同一份进度已挂过：重跑回收不重复注记
    merged = {
        **record,
        "from": str(hit.get("id")),
        "gen": int(prev.get("gen") or 0) + 1,
    }
    hit["salvage"] = merged
    hit["note"] = f"{str(hit.get('note') or '')}{_salvage_note(merged)}".strip("｜")
    _backlog_save(data)
    return dict(hit)


def backlog_append_salvage(item: dict, record: dict) -> dict | None:
    """为「已标 needs_leader 但面上还有未收尾进度」的单自动注册**续作单**。

    与 :func:`backlog_attach_salvage` 的分工看条目状态：还在队列里（pending/
    in_progress）的会被 drain 自动重派，挂注记即可原地续完；已经落到 needs_leader 的
    不再会自动重派，就必须**另立一条 pending**——否则「进度保住了」仍然只是躺在分支上
    等人想起来（本单要消灭的手动动作）。原条目保持 needs_leader 并留下续作单 id，
    组长仍看得见「这一跳失败了」这一事实。

    插在**第一个 pending 之前**：未合并提交随 main 前进而贬值，续作不是新提案，
    不该再排一次队。

    Args:
        item: 失败收场的原条目（drain 手里的种子字典）。
        record: :func:`drain.salvage_worktree` 的读数。

    Returns:
        新注册的续作条目；同一提交已在队 / 链深超 :data:`SALVAGE_MAX_CHAIN` → None
        （后者只在原条目 note 里留「须组长处置」）。
    """
    orig_id = str(item.get("id", ""))
    commit = str(record.get("commit") or "")
    if not orig_id or not BACKLOG_PATH.exists():
        return None
    data = json.loads(BACKLOG_PATH.read_text(encoding="utf-8"))
    items = data.get("items", [])
    orig = next((i for i in items if str(i.get("id", "")) == orig_id), None)
    gen = int((item.get("salvage") or {}).get("gen") or 0) + 1
    if commit and commit in {
        str((i.get("salvage") or {}).get("commit", ""))
        for i in items
        if i.get("salvage") and i.get("status") in ("pending", "in_progress")
    }:
        return None  # 同一份 WIP 已有单在队，不重复立
    if gen > SALVAGE_MAX_CHAIN:
        if orig is not None:
            orig["note"] = (
                f"{str(orig.get('note') or '')}｜抢救链已达上限 {SALVAGE_MAX_CHAIN}，"
                f"不再自动续作{_salvage_note(record)}——须组长处置"
            ).strip("｜")
            _backlog_save(data)
        return None
    entry = {
        "id": _salvage_entry_id(items),
        "member": item.get("member") or "leader",
        "summary": f"（续作 {orig_id}）{str(item.get('summary', ''))[:150]}",
        "kind": item.get("kind") or "defect",
        "success_criteria": item.get("success_criteria")
        or (
            f"抢救提交 {commit or '?'} 的内容合并进 main，且续作面合并后 "
            "``git rev-list --count main..wt-<隔离面名>`` 为空"
        ),
        "evidence": item.get("evidence") or item.get("evidence_file") or "",
        "source": SALVAGE_SOURCE,
        "salvage": {**record, "from": orig_id, "gen": gen},
        "status": "pending",
        "proposed": _now(),
        "note": f"自动续作单（上次尝试 {orig_id} 未收尾，进度已抢救）",
    }
    if orig is not None:
        orig["note"] = (
            f"{str(orig.get('note') or '')}｜已自动立续作单 {entry['id']}".strip("｜")
        )
    idx = next(
        (k for k, i in enumerate(items) if i.get("status") == "pending"), len(items)
    )
    items.insert(idx, entry)
    _backlog_save(data)
    return entry


def backlog_salvage() -> list[dict]:
    """队列里带抢救记录的条目（pending / in_progress）——watch 事件与 status 段的读数源。"""
    if not BACKLOG_PATH.exists():
        return []
    data = json.loads(BACKLOG_PATH.read_text(encoding="utf-8"))
    return [
        i
        for i in data.get("items", [])
        if i.get("salvage") and i.get("status") in ("pending", "in_progress")
    ]


def _salvage_instruction(sal: dict, wt: str) -> str:
    """任务书里的续作指令：三条路按「面还在不在册、名字换没换」分。

    ``pysci-dev worktree add`` 只会从当前 HEAD 新建分支，所以三种现场各有各的正确命令——
    把命令写死在任务书里，续作的一跳才不必自己猜（猜错就是把进度丢在旧分支上）。
    """
    commit = str(sal.get("commit") or "")
    branch = str(sal.get("branch") or "")
    path = str(sal.get("path") or "")
    reason = str(sal.get("reason") or "上次尝试未收尾")
    sha = commit or "<抢救提交>"
    if str(sal.get("worktree") or "") != wt:
        how = (
            f"上次尝试（成因：{reason}）的进度已抢救为提交 ``{sha}``"
            f"（分支 ``{branch}``，旧面 ``{path}``）。旧面没能确认干净，**禁止复用旧名**："
            f"先 ``pysci-dev worktree add {wt}``，再在新面里 "
            f"``git cherry-pick --no-commit {sha}`` 取回进度续完。"
        )
    elif sal.get("registered"):
        how = (
            f"上次尝试（成因：{reason}）的隔离面**还在原地且登记在册**：``{path}``"
            f"（分支 ``{branch}``，抢救提交 ``{sha}``）。"
            f"``pysci-dev worktree add {wt}`` 会报「已登记在册」——那不是故障，"
            "是让你直接进那棵树续完：**不要** reset/另起，也别从 main 重开"
            "（未合并的进度只在这条分支上）。"
        )
    else:
        how = (
            f"上次尝试（成因：{reason}）的工作树已不在，但分支 ``{branch}`` 上还留着"
            f"未合并的抢救提交 ``{sha}``。名字没换所以 **别用** "
            f"``pysci-dev worktree add {wt}``（它从 main 建分支，撞已存在的名会 128 失败），"
            f"改把原分支请回来：``git worktree add "
            f"{(WORKTREES_ROOT / wt).as_posix()} {branch}``——进度就在脚下那棵树里。"
        )
    tail = (
        "\n\n旧面当时还有**未提交**改动没能自动取回（抢救提交只含未合并进度里的已提交部分）："
        f"开工先 ``git -C {path} status --short`` 判读，该留的手工移植。"
        if sal.get("uncommitted")
        else ""
    )
    return f"**续作指令（自动抢救）**：{how}{tail}\n\n"


def _shape_sources(item: dict) -> tuple[str, bool, str]:
    """``(形状, 是否已有判决记录, 两源不一致的告警行)``——**findings 行为准**。

    `remedy_shape` 曾有两个真源，而执法只读 backlog 条目那个（backlog
    20261013-184040-devops，实测 `insp-20261013-045910-mechanical`）：backlog 条目上的值是
    **晋升瞬间从 findings 行拷过来的**，组长事后在 findings 行上裁定 `restore` 不会回流，
    于是任务书照旧注入「change 缺判决 → 交付 blocked」，组员若照做就白烧一跳；反向
    （findings 已是 change 而 backlog 还是 restore）更危险——那是**静默放行**一段本该过
    判决的实施，直接绕过免判决通道的边界。

    故：inspector 来源的条目**以 findings 行为准**（可按 id 反查），backlog 条目那份降为
    兜底（findings 行不在/无该字段时用）。两源不一致时出声，而不是默默按某一侧走。
    """
    shape = str(item.get("remedy_shape") or "")
    judged = bool(str(item.get("judgment") or "").strip()) or (
        str(item.get("channel") or "mechanical") != "mechanical"
    )
    if str(item.get("source") or "") not in ("inspector", "inspectd"):
        return shape, judged, ""
    from . import inspectd as _insp  # 延迟导入：inspectd 顶层引整条 worker 链，避免成环

    row = _insp.finding_by_id(str(item.get("id") or ""))
    if not isinstance(row, dict):
        return shape, judged, ""
    fresh = str(row.get("remedy_shape") or "") or shape
    judged = judged or bool(str(row.get("judgment") or "").strip())
    note = ""
    if shape and fresh and shape != fresh:
        note = (
            f"\n[!] 修复形状两源不一致：findings 行 = `{fresh}`，backlog 条目 = `{shape}`"
            f"——**以 findings 行为准**（组长在那一侧裁定）；建议顺手把 backlog 条目改正。\n"
        )
    return fresh, judged, note


def _remedy_shape_instruction(item: dict) -> str:
    """修复形状分流约束注入任务书（用户 2026-10-13 审查裁决，backlog 20261013-060000-leader）。

    只管 inspector 来源的条目：``restore``=复原既有声明行为，免判决通道照旧；``change``
    =改声明行为/策略/权衡面——**无判决记录**时 devops 必须交付 blocked 上报而非实施。
    判决记录的判据＝findings 行的 ``judgment`` 字段有内容（组长裁定留痕），或该条目经
    非免判决通道（channel≠``mechanical``，如 judgment 类经 J 晋升）进来。

    形状与判决记录**都按 :func:`_shape_sources` 以 findings 行为准**（backlog
    20261013-184040-devops）；backlog 条目上的那份只是兜底。缺 remedy_shape 的旧条目不注入
    （分流只执法于新形状体系，不追溯）。
    """
    if str(item.get("source") or "") not in ("inspector", "inspectd"):
        return ""
    shape, judged, note = _shape_sources(item)
    if not shape:
        return note
    if shape == "change" and not judged:
        return note + (
            "\n**修复形状约束（remedy_shape=change，免判决通道——先读这一段）**：\n"
            "本条经 inspector mechanical 通道入账，无 J/M/twin 判决记录，而其修复形状="
            "「改变声明行为/策略/权衡面」。按修复形状分流规则，实施前必须先过判决包"
            "（`pysci-orch jury` 或 consult-twin）。**未经判决不得实施**：若本任务书未附"
            "判决记录（案卷 id/计票），直接交付 <blocked>（正文注明「remedy_shape=change "
            "缺判决记录」），由组长补判决后重派。\n"
        )
    if shape == "change":
        return note + (
            "\n修复形状=change（改声明行为/策略/权衡面），判决记录已在案"
            "（findings 行的 judgment，或经非免判决通道晋升）——照常实施，"
            "交付仍须过 success_criteria 机械验收。\n"
        )
    return note + (
        "\n修复形状=restore（复原既有声明行为：恢复被删行、修断正则、回滚误删配置之类）"
        "——免判决通道照常实施，仍须过 success_criteria 机械验收。\n"
    )


#: 「因缺授权/缺裁决 blocked」的条目标记（backlog 20261013-193058-devops）：drain 在
#: 交付 `<blocked reason="needs_decision">` 时置上（同 extra 落 ``blocked_report`` 存原点名
#: 原文）。任务书据此注入「原点名事项 ↔ 本次补齐情况」对照，让「执行端点名要授权 → 组长
#: 补授权 → 重排」这条闭环在机械面上可见，而不只靠人记得写附注。
AWAITING_DECISION_KEY = "awaiting_decision"


def _awaiting_decision_instruction(item: dict) -> str:
    """「补授权/补裁决」重排提示：**原点名事项 ↔ 本次补齐情况**的对照行。

    为什么值得单独一行：被 `<blocked reason="needs_decision">` 挡下的条目，重排时若授权仍
    缺，worker 只会照同一成因再 blocked 一次——白烧一跳。把「上轮点名了什么、这次补没补」
    直接摆进任务书，worker 与组长在**同一行**上对账。本次补齐情况取条目当前 ``note``
    （组长补授权应写在此；空=未补，出声提示）。
    """
    named = (
        str(item.get("blocked_report") or "").strip()
        or "（见条目 note 的上轮 blocked 全文）"
    )
    note = str(item.get("note") or "").strip()
    filled = note or "（空——授权/裁决尚未补；重排会再次白烧，请先写入 note 再派）"
    return (
        "**本条为「补授权/补裁决」重排**：上一轮交付 "
        '`<blocked reason="needs_decision">` 而停链——此时重派改变不了「缺授权」，'
        "须组长补授权/判决后才有意义。\n"
        f"· 原点名事项（上轮 blocked 原文）：{named}\n"
        f"· 本次补齐情况（条目当前附注）：{filled}\n"
        "先确认授权/裁决已齐再实施；若仍缺，按同一成因交付 "
        '`<blocked reason="needs_decision">` 再上报，不要重复实施。\n\n'
    )


def backlog_task_text(item: dict) -> str:
    """由 backlog 条目生成 devops 派发任务书正文。

    ``dispatch --from-backlog``（手动）与 ``_drain-devops``（自动）共用此模板，保证
    两条路径派发内容一致。末尾附**合并安全**提示：devops 合并 worktree 前须核对
    main 工作区，避免覆盖组长未提交改动（README §6 自动唤醒下的并发防护）。

    隔离面名走 :func:`worktree_name`——孤儿回收判定「旧面还不干净」时条目带
    ``worktree_gen``，任务书就让 worker 用新一代名字，不复用可能仍有活写者的旧面。
    """
    wt = worktree_name(item)
    gen = int(item.get("worktree_gen") or 1)
    head = (
        f"基础设施改进任务（backlog FIFO 队首，id={item['id']}）：\n\n"
        f"摘要：{item.get('summary', '')}\n\n"
        f"证据：{item.get('evidence') or item.get('evidence_file', '（见建议归档）')}\n\n"
        f"组长附注：{item.get('note', '（无）')}\n\n"
    )
    if gen > 1:
        head += (
            f"**本条已进入第 {gen} 代隔离面**：上一代回收时没能确认干净（旧会话进程或"
            "残骸仍在原地写着），严禁改用回旧名。\n\n"
        )
    sal = item.get("salvage") or {}
    if sal:
        # 抢救事实进任务书：worker 不必猜「上次到哪了」，指令里直接给它续作面与提交
        head += _salvage_instruction(sal, wt)
    if item.get(AWAITING_DECISION_KEY):
        head += _awaiting_decision_instruction(item)
    head += _remedy_shape_instruction(item)
    return head + (
        "要求：以 worktree 隔离实施，**worktree 名用这里给出的隔离面名**"
        "（默认=本条目 id，孤儿回收升代后带 ``-r<代>`` 后缀）——"
        f"`pysci-dev worktree add {wt}`（分支 wt-{wt}），以便 worker "
        "中途死亡时 drain 孤儿回收能据此定位并清理残留 worktree；完整测试后合并、"
        "commit（commit 后按 charter 的 push 规程执行：限时 best-effort、失败忽略）；"
        "交付中报告改动清单与验证证据，"
        f"并注明 backlog id={item['id']} 以便销账。\n\n"
        "grep 判据约定：任务书要求的 grep 核对，**禁词一律运行期拼接**"
        '（如 `"m3" + " 补"`，防 Python 常量折叠把字面量折进 `.pyc` 令含二进制的 '
        "grep 仍命中、判据把自己判红），且判据限 `*.py` 或排除 `__pycache__`。\n\n"
        "合并前 git status 检查——若 main 工作区存在会被本次合并触碰的未提交改动，"
        "交付 <blocked> 说明，等待组长清理；无关的未提交改动可照常合并。\n"
        '**交集类 blocked 请按 <blocked tags="merge_dirty_face"> 交付，并把 '
        "`pysci-dev worktree merge` 打印的「[✗] 交集 N 项」及其后的文件清单原样贴进正文**"
        "——成因是 main 的脏面，重派改变不了它；标注后 drain 会把进度抢救成提交并"
        "**停链**（不立续作单、不烧第二跳），清单挂回原条目等组长清面后手动合并。"
    )


def backlog_batch_text(seed: dict, group: list[dict], *, max_batch: int = 4) -> str:
    """批量消化任务书：种子（backlog_task_text）+ 同提请者组包菜单 + 选取/销账指令。

    group 为空时退化为 :func:`backlog_task_text`（单项，向后兼容）。devops 自主选取
    ≥1 项（**必含种子**），可合并为同一 worktree 实施，交付以 ``backlog id=<id1>,<id2>,…``
    列出**本次完成的全部 id**，orch 逐 id 销账；未选项留 pending 下轮；单项失败按种子
    blocked 交付即可（组内其余不连坐）。
    """
    base = backlog_task_text(seed)
    if not group:
        return base
    member = seed.get("member", "?")
    seed_id = seed.get("id", "")
    menu = "\n".join(
        f"- `{g.get('id', '')}`：{str(g.get('summary', ''))[:120]}" for g in group
    )
    return (
        f"{base}\n\n---\n\n"
        f"## 批量菜单（同提请者 `{member}`，自主选题合并实施）\n\n"
        f"上面是**种子任务**（必做）。以下是同提请者的其余待办，你**可自主选取 ≥0 项**"
        f"与种子合并实施（含种子单批 ≤ {max_batch} 项）；同提请者≈同模块/同视角，"
        "合并可省重复上下文。可按需把选中项重写为合并方案/任务书，在**同一 worktree** 实施。\n\n"
        f"{menu}\n\n"
        "**选取与销账**：只吃种子也可以（组内其余自动留 pending 下轮）。"
        "**菜单项与种子同批合入时，每个已实施的 id 都必须列**——未列即视为未销账、"
        "会被重派，而重派的任务书**读起来像从头做**（实际可能只缺测试/文档面），"
        "下一位组员得先花一跳才搞清真实缺口。交付时在 "
        f"<result> 中以 `backlog id=<id1>,<id2>,…` 列出**本次完成的全部 id**（必含种子 "
        f"`{seed_id}`），orch 据此逐 id 销账；未完成/未选的**不要**列入。若种子本身受阻，"
        "照常交付 <blocked>（组内其余不连坐，留 pending）。"
    )


#: 空壳建议判据（呈现面标注，backlog 20261012-120001-leader）：剥掉标题与结构化标记行
#: 后，剩余实质正文的**非空白字符数**低于此值即标注。阈值取 40（约一行中文短句）——
#: 再短就承载不了「证据 + 修复面」，approve 面也入不了队。实测样例：非法 kind 的占位
#: 提案正文只有 ``placeholder`` 一词（11 字）。
SHELL_MIN_CHARS = 40

#: 结构化标记行前缀：这些行是协议字段，不算「实质正文」。``success_criteria`` 行只剥前缀，
#: 其后的判据内容仍计入（那是人写的真话）。
_MARKER_LINE_RE = re.compile(r"^\s*(kind|source|fingerprint)\s*[:：]", re.IGNORECASE)
_SC_PREFIX_RE = re.compile(
    r"^\s*(?:[-*+•]\s*)?(?:\*\*|__)?success_criteria\s*[:：]?\s*", re.IGNORECASE
)


def _substantive_chars(body: str) -> int:
    """建议正文里「实质内容」的字符数（去空白、去标题行、去协议标记行前缀）。"""
    total = 0
    for i, ln in enumerate(body.splitlines()):
        s = ln.strip()
        if not s or (i == 0 and s.startswith("#")):
            continue
        if _MARKER_LINE_RE.match(s):
            continue
        total += len(re.sub(r"\s", "", _SC_PREFIX_RE.sub("", s, count=1)))
    return total


def suggestion_marks(body: str) -> list[str]:
    """呈现面质检标记：``[!] 空壳建议``/``[!] kind 非法``/``[!] 缺 kind``/
    ``[!] 判据矛盾``（合规时 ``[]``）。

    词表校验在 gate（组员侧）与 approve（入队侧）都已有，这里是**第三面**：提案来源不
    止交付一条路（组长代补、自动立案、手写文件都会直接落进 ``suggestions/``），枚举时
    不标出来，组长就得逐条打开才知道是空壳。

    ``判据矛盾``（backlog 20261013-071759-devops）同理要在**列建议时**就点名：批准单是
    追加式文本，``**补充（用户指令 …）**`` 段与 ``success_criteria`` 段就同一配置键给出
    相反取值时，机械验收只读后者，执行者只能自己猜谁优先——趁还在组长手里改判据最便宜。
    """
    out: list[str] = []
    kind, _ = parse_suggestion_fields(body)
    if not kind:
        out.append("[!] 缺 kind")
    elif kind not in load_infra_kinds():
        out.append(f"[!] kind 非法（{kind}）")
    if _substantive_chars(body) < SHELL_MIN_CHARS:
        out.append("[!] 空壳建议")
    conflicts = criteria_conflicts(body)
    if conflicts:
        out.append(
            "[!] 判据矛盾（" + "、".join(str(c["key"]) for c in conflicts) + "）"
        )
    return out


def cmd_suggestions(args) -> int:
    """列出待审批改进建议，并对空壳/非法分类给出机械标注。"""
    del args
    items = pending_suggestions()
    if not items:
        print("（无待审批建议）")
        return 0
    flagged = 0
    for p in items:
        body = p.read_text(encoding="utf-8")
        first = next((ln.strip() for ln in body.splitlines()[2:] if ln.strip()), "")
        marks = suggestion_marks(body)
        flagged += 1 if marks else 0
        suffix = f"  {' '.join(marks)}" if marks else ""
        print(f"  {p.stem}  [{_member_of(p)}]  {first[:70]}{suffix}")
    if flagged:
        print(
            f"[!] {flagged} 条待审批建议存在空壳/分类非法标注：空壳=正文不足 "
            f"{SHELL_MIN_CHARS} 实质字，reject 即可（--override 记一句依据），无需细读。"
        )
    print()
    print(
        "[NEXT] 审批须带判决包（§5.1 四态其一）：approve <id> --jury（仲裁）/ "
        '--bench <案id>（实测）/ --jury --bench <案id>（双）/ --override "<理由>"（自裁留痕）；'
        "缺陷类且该跳交付含 blocked 走例外通道（裸 approve 即可，建议连带 --bench 复跑失败案）。"
        '否决：uv run --no-sync pysci-orch reject <id> --override "<判决依据>"（必填，留痕入决策日志）。'
    )
    return 0


def cmd_approve(args) -> int:
    """采纳建议：入 backlog FIFO + 写审批回复（下次派发自动附送）。"""
    # Tier A 阈值扫描（README §5.4）：每次审批入口机械重算，把越限的失败标签立成案。
    tier_a_sweep()
    sugg = _find_suggestion(args.suggestion_id)
    if sugg is None:
        print(f"[!] 未找到待审批建议 '{args.suggestion_id}'（orch suggestions 查看）")
        return 2
    member = _member_of(sugg)
    body = sugg.read_text(encoding="utf-8")
    # E0-B（README §6 判据纪律）：backlog 条目须带结构化 kind 与可证伪 success_criteria。
    # 从建议正文解析（约定标记行），success_criteria 可由 --criteria 补录；缺任一 → 拒绝入队。
    kind, criteria = parse_suggestion_fields(body)
    cli_criteria = (getattr(args, "criteria", "") or "").strip()
    if cli_criteria:
        criteria = cli_criteria
    # 判据矛盾（backlog 20261013-071759-devops）：批准单里 success_criteria 段与其后
    # 「**补充（用户指令 …）**」段就同一配置键给出相反取值。**告警不阻断**——入队照做，
    # 但把矛盾摆在组长眼前：此时改写判据文本只需一笔，派发后再发现就要烧一整跳。
    for c in criteria_conflicts(body):
        print(f"[!] {conflict_notice(c)}")
    if cli_criteria:
        print(
            "    （已用 --criteria 补录判据：若补录内容就是上文矛盾的正解，视为已消解）"
        )

    # 摘要行须跳过头部所有标记行（kind/delivery），否则 delivery 会被当成建议内容。
    first = sugg.stem
    for ln in body.splitlines()[2:]:
        s = ln.strip()
        if not s or _HEAD_MARKER_RE.match(s):
            continue
        first = s
        break
    if not kind:
        print(
            f"[!] 建议 '{sugg.stem}' 缺 kind 分类，拒绝入队。请在建议正文补一行"
            "  `kind: defect`（缺陷，附复现证据）或 `kind: usability`（难用/体验类）。\n"
            '    （交付侧本应带 <infra_suggestion kind="...">；此处缺失说明来源早于 E0-B。）'
        )
        return 2
    if kind and kind not in load_infra_kinds():
        # 入队面的词表硬闸（backlog 20261012-120001-leader）：非法分类进 backlog 后，
        # E5 判决包分流与 stats 的 kind 分布都会被一个自造词带偏，而那时再发现就要
        # 回滚整条队列。缺失走上面那条提示，这里只管「标了但标错」。
        vocab = "|".join(load_infra_kinds())
        print(
            f"[!] 建议 '{sugg.stem}' 的 kind={kind!r} 不在受控词表（{vocab}），拒绝入队。\n"
            f"    词表：{orchestration_rel(TAGS_PATH)} 的 infra_kinds 段"
            "（增补须附定义并走 devops 交付，不是本地偏好）。"
        )
        return 2
    if not criteria:
        print(
            f"[!] 建议 '{sugg.stem}'（kind={kind}）缺可证伪成功判据，拒绝入队。\n"
            "    **先对盘**：判据可能只是写成了解析器读不到的形态。接受的书写形态（任一）：\n"
            "      · `success_criteria: …`（行首）\n"
            "      · `- success_criteria: …`（列表项）\n"
            "      · `**success_criteria**: …`（粗体）\n"
            "      且判据正文＝标记行**之后的连续非空行**（最小示例：\n"
            "      `- success_criteria:` 换行后 `  - <可机械复核的判据>`）。\n"
            "    补写其一：① 在建议正文按上述任一形态补判据段；"
            '② 或本次 approve 追加 --criteria "<做完后如何机械验证算成功>"。'
            "（README §6 判据纪律：无判据的提案不可证伪，不进队。）"
        )
        return 2
    # ---- E5b 判决包闸（README §5.3）：基建判断须与组长职责事实分离 ----
    delivery = verdict.delivery_of(body)
    channel, gate_err = verdict.validate(args, kind=kind, delivery=delivery)
    if gate_err:
        print(gate_err)
        return 2
    if channel == verdict.CHANNEL_EXCEPTION:
        print(
            "    判决通道=缺陷例外（kind=defect 且提请那一跳交付为 blocked，"
            "in-vivo 实测即依据）。\n"
            "    规程注记：建议连带 --bench 复跑原失败案，把「当时失败」变成可比的 A/B 读数。"
        )
    if channel in ("jury", "jury+bench"):
        case_path = verdict.assemble_case(
            sugg_id=sugg.stem, summary=first, kind=kind, criteria=criteria, body=body
        )
        try:
            run = verdict.run_jury(case_path)
        except (OSError, ValueError) as exc:
            print(
                f"[!] 案卷不可用，未开庭：{exc}\n    案卷 {orchestration_rel(case_path)}"
            )
            return 2
        call = (run.record or {}).get("verdict")
        if run.status != "recorded":
            print(
                f"[{'!' if run.status == 'queued' else '✗'}] 仲裁未出判决（{run.status}）："
                f"{run.message}\n"
                "    提案保持待审批、未入队。修好渠道后重跑本命令，"
                '或改走 --override "<理由>"（计入 override 率）。'
            )
            return 1 if run.status == "queued" else 2
        if call == "B":
            tally = (run.record or {}).get("tally", {})
            _finalize_rejection(
                sugg,
                member=member,
                situation=first,
                reason=f"仲裁裁决 B（不采纳）A:{tally.get('A')} B:{tally.get('B')}",
                note=args.note or "",
                source="jury",
                channel=channel,
            )
            print(
                f"[√] 仲裁判为不采纳，已自动驳回 {sugg.stem}"
                f"（A:{tally.get('A')} B:{tally.get('B')}；此判决不计 override 率）。"
            )
            return 0
        if not call:
            journal.append(
                {
                    "ts": _now(),
                    "situation": first[:200],
                    "options": "",
                    "choice": "pending",
                    "reason": run.message,
                    "source": "jury",
                    "kind": "ruling",
                    "channel": channel,
                    "suggestion_id": sugg.stem,
                    "member": member,
                }
            )
            print(
                f"[!] 仲裁模棱两可 → escalate=twin 且 twin 不可达/未表态 → 保守回退挂起："
                f"提案保持待审批，未入队。\n"
                f"    {run.message}\n"
                "    人工终裁走 orch consult-twin；眼下要定案："
                '--override "<理由>"（计入 override 率）。'
            )
            return 1
    verify: dict | None = None
    if channel in ("bench", "jury+bench"):
        try:
            verify = verdict.capture_baseline(
                verdict.parse_bench_cases(getattr(args, "bench", "")),
                item_id=sugg.stem,
            )
        except RuntimeError as exc:
            print(
                f"[!] M 通道需要可锚定的基线 commit：{exc}\n"
                "    未入队。请确认在 git 仓库内运行，或改走 --jury / --override。"
            )
            return 2
    entry = {
        "id": sugg.stem,
        "member": member,
        "summary": first[:200],
        "kind": kind,
        "success_criteria": criteria,
        "evidence_file": f"orchestration/state/suggestions/approved/{sugg.name}",
        "status": "pending",
        "proposed": _now(),
        "note": args.note or "",
    }
    # 提案来源标记（Tier A 自动立案写 `source: tier-a`）随条目入队，让 drain 与人工
    # 复盘都能看出这条是「统计机制立的案」还是「组员自己提的建议」。
    # 同族前缀盲区（backlog adhoc-criteria-parse）：`- source: tier-a` 这类列表项写法
    # 旧口径 `startswith("source:")` 读不到 → 元字段静默丢失；去掉行首列表标记再判。
    source = next(
        (
            ln.split(":", 1)[1].strip()
            for ln in body.splitlines()
            if re.sub(r"^[-*+•]\s*", "", ln.strip()).lower().startswith("source:")
        ),
        "",
    )
    if source:
        entry["source"] = source
    # effort 是**入队时**的组长判断（推理密集项标 high，机械项不标=默认中）；drain 消化
    # 时透传给该跳（见 drain._drain_batch）。getattr 同 no_wake：程序化调用方常建 partial 参数。
    level = normalize_effort(getattr(args, "effort", None))
    if level:
        entry["effort"] = level
    # solo 是**入队时**的尺寸判断（pod 级/多子面大单）：drain 组包不并入此项、此项也不
    # 作组包种子（batch_size 恒 1）——大单并批必触该跳 max_turns 腰斩（backlog 20261012-170301-leader）。
    if getattr(args, "solo", False):
        entry["solo"] = True
    if verify:
        entry["verify"] = verify
    n = _backlog_append(entry)
    # 每条判决都进决策日志（含通道），override 率与用量由此回读，不另设第二本账。
    journal.append(
        {
            "ts": _now(),
            "situation": first[:200],
            "options": "",
            "choice": "approve",
            "reason": (args.note or ""),
            "source": channel,
            "kind": "ruling",
            "channel": channel,
            "suggestion_id": sugg.stem,
            "member": member,
        }
    )
    # 审批回复的收件人：自动立案没有「提请成员」可回话（member=tier-a 不是真成员，
    # 回复永远不会被附送），而基建修复的承接方按角色边界就是 devops——归它，回执才落地。
    reply_member = "devops" if entry.get("source") == "tier-a" else member
    REPLIES_DIR.joinpath(reply_member).mkdir(parents=True, exist_ok=True)
    tail = ""
    if verify:
        tail = (
            f"\n判决通道={channel}（README §5.3）：基线 commit "
            f"{verify['baseline_commit']}，已排基线捕获 {verify['baseline_item']}；"
            "本条合并销账时自动入 A/B 核验队列，判决直接落地（改善→销账、"
            "未改善→自动立案纠正提案、不可判→挂起呈组长）。\n"
        )
    (REPLIES_DIR / reply_member / f"{sugg.stem}.md").write_text(
        f"# 组长审批：采纳（{_now()}）\n\n{args.note or '（无附注）'}\n\n"
        f"已入 backlog（第 {n} 位），devops 将按 FIFO 处理；完成后你会在后续任务中获知。\n"
        + tail,
        encoding="utf-8",
    )
    target = _archive(sugg, SUGGESTIONS_DIR / "approved")
    print(
        f"[√] 已采纳 {sugg.stem}（kind={kind}，判决通道={channel}）→ backlog 第 {n} 位；"
        f"审批回复将在下次派发 {reply_member} 时附送。"
        + (f"归档重名，另存为 {target.name}。" if target.stem != sugg.stem else "")
    )
    # 自动唤醒 devops 后台 FIFO 消化 backlog（README §6）；--no-wake 跳过。
    if not getattr(args, "no_wake", False):
        from .drain import (
            wake_devops,  # 延迟导入：workflow↔drain 环（drain 依赖 workflow）
        )

        _, msg = wake_devops()
        print(f"    {msg}")
    # 巡检内嵌拉起之三（README §6）：审批是组长「把一批工作交出去」的时刻，交完之后的
    # 空闲正是 inspector 的窗口。让路/迟滞由引擎判定，这里只负责「想起来要试」。
    from .inspectd import (
        ensure_inspector,  # 延迟导入：inspectd↔workflow 成环（晋升管道要写 backlog）
    )

    ensure_inspector()
    return 0


def _finalize_rejection(
    sugg: Path,
    *,
    member: str,
    situation: str,
    reason: str,
    note: str = "",
    source: str = "override",
    channel: str = "override",
) -> None:
    """驳回落盘三件套：决策日志（先）→ 审批回复 → 移入 ``rejected/``。

    顺序刻意：journal 写失败即中止，不留下「已否决却无留痕」——那比多留一份 pending
    文件坏得多（override 审计面从此残缺）。

    Args:
        sugg: 待审批建议文件（驳回后改名移入 ``rejected/``）。
        member: 提案成员（回复收件人）。
        situation: 建议摘要，进决策日志 ``situation``。
        reason: 判决依据（override 理由或仲裁裁决理由）。
        note: 附送给成员的额外说明（可选）。
        source: 判决来源——``override``（组长自裁）或 ``jury``/``bench``（经判决通道）。
        channel: 通道读数，进 ``channel`` 字段供 override 率统计（§5.1：经 J 或 M 判决
            返回的 reject **不算** override，两者必须在同一份账里可分辨）。
    """
    journal.append(
        {
            "ts": _now(),
            "situation": situation[:200],
            "options": "",
            "choice": "reject",
            "reason": reason,
            "source": source,
            "kind": "override" if source == "override" else "ruling",
            "channel": channel,
            "suggestion_id": sugg.stem,
            "member": member,
        }
    )
    head = "否决依据（override）" if source == "override" else "仲裁判决：不采纳。依据"
    reply = f"# 组长审批：暂不采纳（{_now()}）\n\n{head}：{reason}\n"
    if note:
        reply += f"\n附注：{note}\n"
    REPLIES_DIR.joinpath(member).mkdir(parents=True, exist_ok=True)
    (REPLIES_DIR / member / f"{sugg.stem}.md").write_text(reply, encoding="utf-8")
    _archive(sugg, SUGGESTIONS_DIR / "rejected")


def cmd_reject(args) -> int:
    """否决建议：**必须附 ``--override`` 判决依据**（留痕入决策日志），否则非零退出。

    README §5.3/E0——组长对组员基建提案的否决 = override（有判决依据的 reject
    才算 override；无依据的否决不合规）。当前 approve 侧判决包（J/M/J+M）E5 才实装，
    故 E0 阶段一切 reject 均属 override，强制留痕。缺失 → 非零退出 + 可读提示；
    提供 → 正常否决（写回复 + 移入 rejected/）并自动追加 journal 行
    （``source=override``、``situation``=建议摘要、``reason``=override 理由）。
    """
    override = (getattr(args, "override", "") or "").strip()
    if not override:
        print(
            "[!] 否决必须附判决依据（override 留痕）：orch reject <id> --override "
            '"<详细理由>"\n'
            "    无 --override 的否决不合规（README §5.3/E0：reject=override，须计入 "
            "override 率并落决策日志供审计回放）。"
        )
        return 2
    sugg = _find_suggestion(args.suggestion_id)
    if sugg is None:
        print(f"[!] 未找到待审批建议 '{args.suggestion_id}'")
        return 2
    member = _member_of(sugg)
    body = sugg.read_text(encoding="utf-8")
    first = next(
        (
            ln.strip()
            for ln in body.splitlines()[2:]
            if ln.strip() and not _HEAD_MARKER_RE.match(ln.strip())
        ),
        sugg.stem,
    )
    note = (args.note or "").strip()
    _finalize_rejection(
        sugg,
        member=member,
        situation=first,
        reason=override,
        note=note,
        source="override",
        channel="override",
    )
    print(
        f"[√] 已否决 {sugg.stem}（override 留痕入决策日志）；"
        f"理由将在下次派发 {member} 时附送。"
    )
    return 0


def cmd_requeue(args) -> int:
    """``orch requeue <id> [--note]``：needs_leader → pending **并唤醒 drain**。

    组长面缺的那条正门（backlog 20261013-215909-leader）：把一个被 block 的条目回排重派
    ——此前只能直调 drain 内部函数 ``wake_devops``（或直改 backlog.json），本命令把
    「迁移状态 + 唤醒」收成一条可测动作。回排只对**本就 needs_leader** 的条目有效（起点
    状态受控，见 :func:`backlog_requeue_needs_leader`）；``--note`` 写回排说明，
    带 ``awaiting_decision`` 的条目即「本次补齐情况」。
    """
    item = backlog_requeue_needs_leader(args.item_id, note=(args.note or ""))
    if item is None:
        print(
            f"[!] 未找到 status=needs_leader 的条目 '{args.item_id}'——"
            "回排只对 needs_leader 有效（orch status 看清单）"
        )
        return 2
    print(
        f"[√] {item['id']} 已回排 pending（needs_leader → pending），"
        "devops FIFO 将重新取到它"
    )
    from .drain import (
        wake_devops,  # 延迟导入：workflow↔drain 环（drain 依赖 workflow）
    )

    _, msg = wake_devops()
    print(f"    {msg}")
    return 0


def cmd_resolve(args) -> int:
    """``orch resolve <id> --note "<为何销账>"``：组长确认**已达标** → done。

    把一条已被 drain 挂成 needs_leader、但组长判定已消解的条目收尾（backlog
    20261013-215909-leader）——此前只能直改 ``state/backlog.json``（drain 在跑时有并发
    覆盖风险）。原 blocked/needs_leader 事实**不删**：保留在 note 与 git 历史。只对
    needs_leader 起点有效。
    """
    item = backlog_resolve(args.item_id, note=(args.note or "").strip())
    if item is None:
        print(
            f"[!] 未找到 status=needs_leader 的条目 '{args.item_id}'——"
            "销账只对 needs_leader 有效（orch status 看清单）"
        )
        return 2
    print(
        f"[√] {item['id']} 已销账 done（组长确认达标）；"
        "原 needs_leader 事实保留于 note 与 git 历史"
    )
    return 0


def cmd_consult(args) -> int:
    """快捷咨询副组长（平级协作条款：异议义务提醒注入任务文本）。"""
    text = (
        f"组长咨询（顾问模式，轻量回答即可）：\n\n{args.question}\n\n"
        "提醒（charter 平级协作条款）：你有责任补充不同视野——如认为组长的判断或前提"
        "有误，请直接指出并给出依据；鼓励以实测佐证。若你认为该咨询背后的任务派发本身"
        "不合理，你有权拒绝并说明理由。"
    )
    outcome = do_dispatch(
        "deputy",
        text=text,
        slug=f"consult-{datetime.now().strftime('%H%M%S')}",
        session=args.session or "latest",
    )
    if outcome.error == "pod 不存在":
        print("    （deputy pod 属 Phase 3；当前可先向对应专职组员或组长自行判断。）")
    return outcome.code


def _fmt_dur(seconds: float) -> str:
    """秒 → 人类可读时长（h/m/s）。"""
    if seconds >= 3600:
        return f"{seconds / 3600:.1f}h"
    if seconds >= 60:
        return f"{seconds / 60:.0f}m"
    return f"{seconds:.0f}s"


def backlog_stats() -> dict:
    """backlog 侧统计：done 条目**等待时长**（proposed→done_at）与**批量大小分布**。

    数据源为 backlog.json（非台账）——proposed/done_at/batch_size 均为 backlog 条目字段
    （batch_size 由批量消化销账时写入，缺省按 1）。等待时长反映建议从被采纳到落地的时延，
    批量分布反映 drain 组包消化的实际批大小（backlog 20261010-batch-digest）。
    """
    empty = {"done": 0, "pending": 0, "waits_s": [], "batch_dist": {}}
    if not BACKLOG_PATH.exists():
        return empty
    try:
        data = json.loads(BACKLOG_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return empty
    waits: list[float] = []
    batch_dist: dict[int, int] = {}
    done = pending = 0
    for it in data.get("items", []):
        st = it.get("status")
        if st == "pending":
            pending += 1
        if st != "done":
            continue
        done += 1
        proposed, done_at = it.get("proposed"), it.get("done_at")
        if proposed and done_at:
            try:
                dt = (
                    datetime.fromisoformat(str(done_at))
                    - datetime.fromisoformat(str(proposed))
                ).total_seconds()
            except (ValueError, TypeError):
                dt = -1
            if dt >= 0:
                waits.append(dt)
        try:
            bs = max(1, int(it.get("batch_size", 1)))
        except (ValueError, TypeError):
            bs = 1
        batch_dist[bs] = batch_dist.get(bs, 0) + 1
    return {
        "done": done,
        "pending": pending,
        "waits_s": waits,
        "batch_dist": batch_dist,
    }


def _print_backlog_stats() -> None:
    """打印 backlog 等待时长 + 批量大小分布（cmd_stats 用；独立于台账数据）。"""
    bs = backlog_stats()
    if not bs["done"]:
        return
    print("== backlog 消化 ==")
    waits = sorted(bs["waits_s"])
    if waits:
        n = len(waits)
        median = waits[n // 2] if n % 2 else (waits[n // 2 - 1] + waits[n // 2]) / 2
        avg = sum(waits) / n
        print(
            f"  等待时长（proposed→done）：中位 {_fmt_dur(median)} / 平均 {_fmt_dur(avg)}"
            f" / 最长 {_fmt_dur(waits[-1])}（n={n}）"
        )
    dist = dict(sorted(bs["batch_dist"].items()))
    print(f"  批量大小分布（done 条目）：{dist}  当前待办={bs['pending']}")


#: 立案指纹台账（运行态，gitignore）：``{"version":1,"filed":{fp: 立案时刻}}``。
TIER_A_FILED_PATH = ORCH_STATE_ROOT / "tier-a-filed.json"
#: 同一立案指纹的重复抑制窗（README §5.4「同指纹 24h 内不重复立案」）。
TIER_A_DEDUP_HOURS = 24


def _load_fingerprints(path: Path) -> dict[str, str]:
    """读立案指纹台账 ``{fp: 立案时刻 ISO}``；缺失/损坏 → 空（fail-open）。

    读不到只会失去一次去重（可能重复立案一条提案），不会丢数据——提案本体在
    ``suggestions/`` 里，台账只是「最近立过哪些」的记忆。
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    filed = data.get("filed") if isinstance(data, dict) else None
    if not isinstance(filed, dict):
        return {}
    return {str(k): str(v) for k, v in filed.items() if isinstance(v, str)}


def _save_fingerprints(path: Path, fps: dict[str, str]) -> None:
    """写立案指纹台账（带 ``{fp: ts}`` 原始映射，供人工核对何时立过案）。

    只保留最近 500 个指纹（插入序淘汰）：指纹键含窗口桶，桶随时间单调推进，旧键永不再
    命中，留着只是让文件无界增长。
    """
    keep = dict(list(fps.items())[-500:])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"version": 1, "filed": keep}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _recent_cutoff() -> datetime:
    """重复抑制窗的起点（现在 − :data:`TIER_A_DEDUP_HOURS` 小时）。"""
    return datetime.now().astimezone() - timedelta(hours=TIER_A_DEDUP_HOURS)


def tier_a_sweep(
    *,
    entries: list | None = None,
    tier: dict[str, int] | None = None,
    filed_path: Path | None = None,
) -> list[str]:
    """§2.4 阈值自动立案：同一 ``(标签, 成员)`` 在滑窗内达阈值 → 登记基建提案。

    在 ``orch status`` / ``orch approve`` 入口机械调用（README §5.4 硬性依赖：统计必须
    被流程硬性依赖，不靠「组长有空去看」）。产出的是**待审批提案**
    ``suggestions/<ts>-tier-a.md``（``kind: defect`` + 正文 ``kind:``/``success_criteria``
    标记行齐备，故直接走现行 approve 面），而不是直接写 backlog——§5.1 把「是否接受
    这份判决」设为强制选择点，代码负责把案立到人眼前，人负责点头或 override。

    Args:
        entries: 台账行（缺省现场 ``read_all()``；测试注入用）。
        tier: 窗口与阈值参数（缺省读 registry ``tier_a``）。
        filed_path: 立案指纹台账路径（测试注入用）。

    Returns:
        本次新立案的提案 id（文件名 stem）列表，按标签名排序；无新案 → ``[]``。
    """
    rows = read_all() if entries is None else entries
    params = tier if tier is not None else Registry.load().tier_a
    win = within_window(rows, days=params["window_days"], hops=params["window_hops"])
    counts = failure_counts(win)
    path = filed_path or TIER_A_FILED_PATH
    fps = _load_fingerprints(path)
    cutoff = _recent_cutoff().isoformat(timespec="seconds")
    vocab = load_tags()
    open_keys = _open_tier_a_keys()
    filed: list[str] = []
    for tag in sorted(counts):
        for member in sorted(counts[tag]):
            total = counts[tag][member]
            if total < params["threshold"]:
                continue
            # 窗口桶 = 窗内该 (标签, 成员) 最新一条台账行的日期：同一天内重跑必落同一桶
            # （幂等），跨天才可能换桶。
            hit = [r for r in win if _row_has_tag(r, tag, member)]
            newest = max((_row_ts(r) for r in hit), default="")
            fp = f"{tag}|{member}|{newest[:10]}"
            if fps.get(fp, "") >= cutoff:
                continue  # 同指纹 24h 内立过（§2.4 的去重窗）
            if f"{tag}|{member}" in open_keys:
                # 旧案仍在待审批面（既没 approve 也没 reject）→ 不重立。缺这一道，
                # 窗口桶每天前进就会把「同一个没修好的问题」每天再立一次。
                continue
            SUGGESTIONS_DIR.mkdir(parents=True, exist_ok=True)
            sid = f"{_filed_ts()}-tier-a"
            n = 0
            while (SUGGESTIONS_DIR / f"{sid}.md").exists():
                # 秒级 id 会同名（E4 引出的同类缺陷在 backlog 20261011-233248-devops）：
                # 一次扫描里立多个案子必须各占一文件，否则后写的把先写的静默覆盖。
                n += 1
                sid = f"{_filed_ts()}.{n}-tier-a"
            body = _tier_a_proposal(
                tag=tag,
                member=member,
                total=total,
                params=params,
                rows=hit,
                definition=vocab.get(tag, ""),
                fingerprint=fp,
            )
            (SUGGESTIONS_DIR / f"{sid}.md").write_text(body, encoding="utf-8")
            open_keys.add(f"{tag}|{member}")
            fps[fp] = _now()
            filed.append(sid)
    if filed:
        _save_fingerprints(path, fps)
    return filed


def _open_tier_a_keys() -> set[str]:
    """仍在待审批面的自动立案键 ``{"tag|member"}``（读提案正文的 ``fingerprint:`` 标记）。

    靠文件内容而非文件名识别来源：提案文件名里只有一个信息位（成员），留给了
    ``tier-a`` 这个来源标记，具体标签与被立案成员在正文里。
    """
    out: set[str] = set()
    for p in pending_suggestions():
        try:
            lines = p.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        if not any(ln.strip().lower() == "source: tier-a" for ln in lines):
            continue
        for ln in lines:
            s = ln.strip()
            if s.startswith("fingerprint:"):
                parts = s.split(":", 1)[1].strip().split("|")
                if len(parts) >= 2:
                    out.add(f"{parts[0]}|{parts[1]}")
    return out


def _filed_ts() -> str:
    """提案文件名时间戳（与 dispatch 的 ``_ts()`` 同法：``YYYYMMDD-HHMMSS``）。"""
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _row_has_tag(row: Any, tag: str, member: str) -> bool:
    """台账行是否属该成员且带该标签（sweep 取证用，dataclass 与 dict 两种形态都认）。"""
    if _cell(row, "member", "") != member:
        return False
    return tag in (_cell(row, "tags", []) or [])


def _row_ts(row: Any) -> str:
    """台账行的 ``ts``（缺失 → 空串，字典序排最前，不抛）。"""
    return str(_cell(row, "ts", ""))


def _cell(row: Any, name: str, default: Any) -> Any:
    """属性或字典键两种形态都认（与 :func:`failure_tags._field` 同法，本地不引私有函数）。"""
    if isinstance(row, dict):
        return row.get(name, default)
    return getattr(row, name, default)


def _tier_a_proposal(
    *,
    tag: str,
    member: str,
    total: int,
    params: dict[str, int],
    rows: list,
    definition: str,
    fingerprint: str,
) -> str:
    """自动生成提案正文（approve 面要求的标记行齐备：``kind:`` / ``success_criteria``）。

    行序有讲究：``cmd_approve`` 取「标题+空行+``kind:`` 行之后第一个非空行」当 backlog
    摘要，故摘要行必须紧跟在 ``kind:`` 行之后、其余 ``source:``/``fingerprint:`` 标记之前。

    证据取台账行引用（最多 8 行，附台账文件相对路径）——立案不靠模型回忆，
    靠可复核的时间戳与任务书。
    """
    refs = "\n".join(
        f"- {_cell(r, 'ts', '')} {_cell(r, 'kind', '')} "
        f"tags={','.join(_cell(r, 'tags', []) or [])} "
        f"任务书={_cell(r, 'task_file', '')}"
        for r in rows[-8:]
    )
    tail = "" if len(rows) <= 8 else f"\n- …（窗内共 {len(rows)} 行，仅列最近 8 行）"
    return (
        f"# Tier A 自动立案（{member} × {tag}，待审批）\n\n"
        f"kind: defect\n"
        f"**{member} 的 `{tag}` 类失败在 {params['window_days']} 天 / "
        f"{params['window_hops']} 跳滑窗内累计 {total} 次**"
        f"（阈值 {params['threshold']}，README §5.4 自动立案）\n\n"
        f"source: tier-a\n"
        f"fingerprint: {fingerprint}\n\n"
        f"标签定义（词表 {orchestration_rel(TAGS_PATH)}）："
        f"{definition or '（词表无定义）'}\n\n"
        f"## 证据（台账行引用，台账 {orchestration_rel(LEDGER_PATH)}）\n\n"
        f"{refs}{tail}\n\n"
        f"## 待判定\n\n"
        f"- 设施缺陷（补能力/修工具）、流程缺口（改任务书写法或增补词表），"
        f"还是成员用法问题（回派纠正）？\n"
        f"- 复核：``orch ledger --member {member} --days {params['window_days']}``\n\n"
        f"success_criteria:\n"
        f"- 同一 `(标签={tag}, 成员={member})` 在下一个 {params['window_days']} 天 / "
        f"{params['window_hops']} 跳滑窗内的计数低于阈值 {params['threshold']}"
        f"（``orch stats`` 的 by_failure 段可复核）。\n"
    )


def format_failure_section(entries: list, *, tier: dict[str, int] | None = None) -> str:
    """Tier A 失败标签段（README §5.4）：滑窗计数 + 阈值标注，渲染成可打印多行文本。

    窗口与阈值都取自 registry ``tier_a``（:meth:`Registry.tier_a`），与 :func:`tier_a_sweep`
    的立案判定同一把尺——统计上看着达标的标签，立案逻辑必须同样够得着，两侧不许各算各的。

    Args:
        entries: 台账行（已按 --plan/--member/--days 过滤过的那批）。
        tier: 覆盖窗口参数（测试注入用）；缺省现场读 registry。

    Returns:
        以标题行开头的多行文本（无末行换行）。
    """
    params = tier if tier is not None else Registry.load().tier_a
    win = within_window(entries, days=params["window_days"], hops=params["window_hops"])
    counts = failure_counts(win)
    head = (
        f"  失败标签 by_failure（窗口 {params['window_days']}天 或 最近"
        f"{params['window_hops']}跳，阈值 {params['threshold']}，! = 达阈值将自动立案）："
    )
    return "\n".join(
        [
            head,
            format_by_failure(counts, threshold=params["threshold"], vocab=load_tags()),
        ]
    )


def cmd_stats(args) -> int:
    """三级统计：计划级（--plan）/ 成员级（--member/--days）/ 全体历史 + backlog 消化。"""
    entries = read_all()
    if args.plan:
        sub = [e for e in entries if e.plan == args.plan]
        title = f"计划 {args.plan}"
    elif args.member:
        sub = [e for e in entries if e.member == args.member]
        title = f"成员 {args.member}"
    else:
        sub = entries
        title = "全体历史"
    if args.days:
        cutoff = (datetime.now().astimezone() - timedelta(days=args.days)).isoformat()
        sub = [e for e in sub if e.ts >= cutoff]
        title += f"（近 {args.days} 天）"
    s = summarize(sub)
    if not s["hops"]:
        print(f"{title}：暂无台账数据")
    else:
        print(f"== {title} ==")
        print(
            f"  跳数={s['hops']}  总耗时={s['duration_ms'] / 1000:.0f}s  "
            f"{format_est_tokens(s)}  {format_credits(s)}"
        )
        for mid, row in sorted(s["by_member"].items()):
            print(
                f"  {mid:<10} 跳数={row['hops']:<4} 耗时占比={row['duration_pct']}%  "
                f"est占比={row['est_tokens_pct']}%  credits占比={row['credits_pct']}%"
            )
        if len(s["by_effort"]) > 1:
            # 单一档时 by_effort ≡ 全体，无 A/B 信号；>=2 档才打（backlog 20261010-153041-devops）
            print(
                "  按推理强度档位（轮均=num_turns 均值，返工率=kind≠result 跳占比）："
            )
            print(format_by_effort(s["by_effort"]))
        kinds: dict[str, int] = {}
        for e in sub:
            kinds[e.kind] = kinds.get(e.kind, 0) + 1
        print(f"  交付分布：{kinds}")
        print(format_failure_section(sub))
    _print_backlog_stats()
    return 0


# ---------------------------------------------------------------------------
# review（README §5.2：reviewer 派发 + VERDICT 解析 + FAIL 自动回派 + 复审一次）
# ---------------------------------------------------------------------------
import re  # noqa: E402

REVIEWS_DIR = ORCH_STATE_ROOT / "reviews"

_VERDICT_RE = re.compile(r"<verdict>\s*(PASS|FAIL)\s*</verdict>", re.IGNORECASE)


def parse_verdict(body: str) -> str | None:
    """从 reviewer 交付正文解析 VERDICT（PASS/FAIL/None）。"""
    m = _VERDICT_RE.search(body or "")
    return m.group(1).upper() if m else None


def _save_review(record: dict) -> Path:
    REVIEWS_DIR.mkdir(parents=True, exist_ok=True)
    p = REVIEWS_DIR / f"{record['id']}.json"
    p.write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return p


def _extract_goal_excerpt(path: Path, limit: int = 1500) -> str:
    """从原生产任务书摘取目标陈述正文（供 review 任务书 G2 独立核验）。

    剥离 :func:`dispatch.write_taskbook` 自动加的头（``# 任务书 …`` 与 ``（派发时间：…）``）
    与前导空行，保留组长撰写的任务正文，限长 ``limit`` 字符（超出截断并标注）。

    Args:
        path: 原生产任务书路径（绝对或相对项目根）。
        limit: 摘录正文的最大字符数。

    Returns:
        目标陈述正文；文件不可读/为空时返回 ``""``（调用方据此告警，不中断审查）。
    """
    p = path if path.is_absolute() else PROJECT_ROOT / path
    try:
        raw = p.read_text(encoding="utf-8")
    except OSError:
        return ""
    body: list[str] = []
    for ln in raw.splitlines():
        s = ln.strip()
        # 跳过自动头与前导空行，直到遇到第一行真实正文
        if not body and (
            s.startswith("# 任务书") or s.startswith("（派发时间") or not s
        ):
            continue
        body.append(ln)
    text = "\n".join(body).strip()
    if len(text) > limit:
        text = text[:limit].rstrip() + " …（截断）"
    return text


def cmd_review(args) -> int:
    """审查产物：reviewer 出具 VERDICT；FAIL → 自动回派 origin 返工 → 复审一次；
    二次 FAIL → 升级组长仲裁（README §5.2）。长链路，建议后台 Bash 运行。"""
    from pysci.paths import PODS_ROOT

    review_id = f"rev-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    rubric_rel = f"rubrics/{args.rubric}.md"
    rubric_abs = PODS_ROOT / "reviewer" / rubric_rel
    if not rubric_abs.exists():
        print(f"[!] rubric 不存在：{rubric_abs}")
        return 2
    # 目标陈述（G2 独立核验）：组长显式提供，避免 reviewer 依赖生产者 notes.md 自述。
    # --goal（直接文本）优先于 --goal-from（从原生产任务书摘录）；二者皆缺则不嵌入并告警。
    goal_text = ""
    goal_source = ""
    if getattr(args, "goal", None):
        goal_text = str(args.goal).strip()
        goal_source = "literal(--goal)"
    elif getattr(args, "goal_from", None):
        gp = Path(args.goal_from)
        goal_text = _extract_goal_excerpt(gp)
        if goal_text:
            goal_source = orchestration_rel(
                gp if gp.is_absolute() else PROJECT_ROOT / gp
            )
        else:
            print(
                f"[!] --goal-from 任务书不可读或为空：{gp}"
                "（本次审查 G2 将缺独立目标陈述）"
            )
    if not goal_text:
        print(
            "[!] 未提供目标陈述（--goal/--goal-from）——reviewer 的 G2「与任务书目标一致」"
            "将无独立依据，建议补派时带上原生产任务书。"
        )
    record: dict = {
        "id": review_id,
        "artifact": args.artifact,
        "origin": args.origin,
        "rubric": args.rubric,
        "goal_source": goal_source,
        "goal_excerpt": goal_text,
        "started": _now(),
        "rounds": [],
    }
    goal_block = (
        (
            f"- 目标陈述（来自原生产任务书：{goal_source}；G2「与任务书目标一致」"
            "**据此独立核验**，勿以生产者 notes.md/自述替代）：\n"
            f"{goal_text}\n"
        )
        if goal_text
        else ""
    )
    max_rounds = 2  # 初审 + 复审一次（用户裁决）
    for round_no in range(1, max_rounds + 1):
        task = (
            f"审查任务（第 {round_no} 轮，review id={review_id}）：\n\n"
            f"- 产物：{args.artifact}\n"
            f"- rubric（必读，按它逐项检查）：{rubric_abs}\n"
            f"- 生产组员：{args.origin}\n"
            + goal_block
            + (
                "- 上一轮 FAIL 证据与返工说明见你的会话历史/任务书附件。\n"
                if round_no > 1
                else ""
            )
            + "\n要求：按 charter 的 VERDICT 交付格式出具 <result>，内含 "
            "<verdict>PASS|FAIL</verdict>、<scores>、<evidence>（FAIL 时附可执行返工指引）。"
        )
        outcome = do_dispatch(
            "reviewer", text=task, slug=f"{review_id}-r{round_no}", session="new"
        )
        verdict = parse_verdict(outcome.body) if outcome.kind == "result" else None
        record["rounds"].append(
            {
                "round": round_no,
                "reviewer_session": outcome.sid,
                "kind": outcome.kind,
                "verdict": verdict,
                "body_excerpt": outcome.body[:1200],
            }
        )
        if outcome.kind != "result" or verdict is None:
            _save_review(record)
            print(
                f"[?] 审查未完成（kind={outcome.kind}, verdict={verdict}）——见上方 reviewer 交付。"
            )
            print()
            print(
                "[NEXT] 人工判读 reviewer 输出：补派审查（dispatch reviewer --session "
                f"{outcome.sid[:8]}）或放弃本次审查。审查记录：{_rel(record)}"
            )
            return 2
        if verdict == "PASS":
            record["completed"] = _now()
            record["final"] = "PASS"
            p = _save_review(record)
            print(f"[√] 审查 PASS（第 {round_no} 轮）。记录：{p.name}")
            print()
            print("[NEXT] 该环节工作宣告完成；若有后续环节按计划推进，否则向用户交付。")
            return 0
        # FAIL
        print(f"[✗] 审查 FAIL（第 {round_no} 轮）。")
        if round_no >= max_rounds:
            record["completed"] = _now()
            record["final"] = "FAIL_ESCALATED"
            p = _save_review(record)
            print()
            print("[NEXT] 复审仍 FAIL → 组长仲裁（用户裁决的升级路径）：")
            print(
                '  1) 改派副组长攻坚：orch plan adhoc deputy --text "..."（附审查记录）'
            )
            print("  2) 修改方案/rubric 适用性后重审")
            print("  3) 呈报用户裁定。审查记录：" + p.name)
            return 2
        record_round = record["rounds"][-1]
        rework_text = (
            f"返工任务（审查 FAIL 自动回派，review id={review_id}）：\n\n"
            f"reviewer 对你的产物 {args.artifact} 出具 FAIL。审查证据与返工指引：\n\n"
            f"{outcome.body[:1500]}\n\n"
            "请针对证据逐条修正后重新交付（机械验收照旧声明）。"
        )
        rework = do_dispatch(
            args.origin, text=rework_text, slug=f"{review_id}-rework", session="latest"
        )
        record_round["rework_session"] = rework.sid
        record_round["rework_kind"] = rework.kind
        if rework.code != 0:
            _save_review(record)
            print(f"[!] 返工未成功（kind={rework.kind}）——中止复审，升级组长决策。")
            print()
            print(
                "[NEXT] 可选：补充信息再回派 / 咨询副组长 / 呈报用户（附审查记录 "
                f"{review_id}.json）"
            )
            return 2
        print("[√] 返工交付成功，进入复审…")
    return 2


def _rel(record: dict) -> str:
    return f"orchestration/state/reviews/{record['id']}.json"
