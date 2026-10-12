"""inspector 闲时巡检引擎（REFACTOR v1.0 §7/§5.2；E5a，backlog 20261012-120003-leader）。

**这个模块管的是「什么时候允许 inspector 花资源」，不是「巡检怎么做」**——后者是 inspector
组员自己的会话工作（见 ``orchestration/pods/inspector/`` charter 与技能）。§7 的成本控制
全部是硬代码：全局开关、幂等拉起、单次激活预算到顶自停、让路、双队列迟滞节流。设计纪律
③「硬代码优先」的教训来自 ``consult`` 软选项从不被使用——把节流写成文档提醒等于没写。

四类闸门（都在 :func:`activation_gate` 一处收口，任何入口都绕不过）：

1. **全局开关** ``state/inspector.enabled``（:func:`enabled`）：**文件缺失=开启**（§7
   「默认开启」），``orch inspector-off`` 写 ``false`` 后所有拉起逻辑 no-op——关比开容易
   生效是刻意的（成本失控时用户要一键止住，而不是去猜默认值）。
2. **激活预算**（registry ``inspector.max_hops`` / ``wallclock_s``）：单次激活派给
   inspector 本人的跳数与墙钟到顶即自停（``.done`` 记 ``stopped=budget_*``）。
3. **冷却**（``cooldown_s``）：读最近一次 ``.done`` 的时间。没有这一条，内嵌拉起会让
   组长每敲一次 ``orch status`` 就重新激活一轮，成本按敲击次数放大。
4. **让路 + 迟滞**（:func:`busy_surfaces` / :func:`throttle_state`）：生产侧在派跳
   （devops drain 忙 / bencher worker 忙 / 计划 state 新鲜）时**不派 LLM 跳、不晋升**，
   只跑零成本的读侧机械检查；backlog 或 bench 队列超 ``pause_*`` 阈值 → 暂停产出，
   回落到 ``resume_*``（严格低于 pause，迟滞防振荡）才恢复。

**内嵌拉起**（:func:`ensure_inspector`）挂在 ``orch status`` / ``orch approve`` /
``orch dispatch`` 三个入口——「等组长记得开」不是机制（§0 设计纪律 1）。

发现管道（§5.2，:func:`append_finding` / :func:`promote_finding` / :func:`submit_judgment`）
是本模块的第二职责：**机械证据类自动晋升 backlog**（附证据、无需判决），**判断类必须经 J
仲裁**（复用 :mod:`.verdict`/ :mod:`.jury` 共享库）。inspector **没有 override 面**——
:func:`submit_judgment` 的参数表里根本没有这个选项，而不是「有但劝你别用」。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from pysci.paths import ORCH_STATE_ROOT

from . import workq
from .drain import (
    DRY_ENV as _DRY_ENV,
)
from .drain import (
    WorkerPaths,
    _consume_stop_request,
    _log,
    _now,
    _spawn_detached,
    acquire_lock,
    cancel_status,
    latest_done,
    lock_status,
    new_run_log,
    release_lock,
    request_stop,
)
from .registry import Registry, orchestration_rel

#: inspector 的运行态面（findings/throttle/evidence 都在这，整面 gitignored）。
STATE_DIR: Path = ORCH_STATE_ROOT / "inspector"

#: 发现台账（§7③：巡检产出先落这里，再按类别分流晋升）。
FINDINGS_PATH: Path = STATE_DIR / "findings.jsonl"

#: 迟滞闸的当前态（``produce`` ∈ on|paused）。**必须落盘**：「暂停/恢复」是记忆性判断，
#: 每轮重算会把迟滞变成抖动（上一轮暂停、这一轮看不到自己曾暂停 → 立即恢复）。
THROTTLE_PATH: Path = STATE_DIR / "throttle.json"

#: 全局开关文件（REFACTOR §7「默认开启」⇒ **缺席=on**，显式写 ``false`` 才关）。
ENABLE_PATH: Path = ORCH_STATE_ROOT / "inspector.enabled"

#: 巡检工作队列（§7①「queue-first」：组长指派核查与 Tier A 案文撰写排在这里）。
QUEUE_PATH: Path = STATE_DIR / "queue.json"

#: 晋升证据正文落盘处（backlog 条目 ``evidence_file`` 指向它，组长/devops 复核看的是这份）。
EVIDENCE_DIR: Path = STATE_DIR / "evidence"

#: 生命周期落点——锁语义与 devops drain / bencher worker 共用一套原语（§0.9）。
INSPECT_PATHS = WorkerPaths(
    lock=ORCH_STATE_ROOT / "inspector.lock",
    runs=ORCH_STATE_ROOT / "inspector-runs",
    cancel=ORCH_STATE_ROOT / "inspector.cancel",
)

#: 队列项类型：巡检菜单项 / 组长指派核查 / Tier A 案文撰写。
QUEUE_KINDS: tuple[str, ...] = ("patrol", "assignment", "tier-a-draft")

#: 发现类别（§5.2 分流）：机械证据=自动晋升；判断=必须经 J。
FINDING_CLASSES: tuple[str, ...] = ("mechanical", "judgment")

#: 发现状态机。``queued``=仲裁渠道不可达（§4 降级），不是失败，下轮重投。
FINDING_STATUSES: tuple[str, ...] = (
    "pending",
    "promoted",
    "rejected",
    "queued",
    "escalated",
)

_ID_PREFIX = "insp"

#: 巡检菜单（§7②「随机抽取」）。每条是一段任务书素材——菜单在**引擎侧**固定，
#: 组员不能自选做什么，否则「闲时巡检」会变成「组员挑感兴趣的做」。
#: ``self_cover`` 标出 §0.2 的自覆盖项（改进管道自身），``rubric`` 标出设计审查三维。
PATROL_MENU: tuple[dict[str, Any], ...] = (
    {
        "id": "doctor-sweep",
        "label": "doctor 全域巡检",
        "brief": (
            "跑 `pysci-dev doctor` 全域，逐条核对报告里的 [!] 与 [✗]：能复现的按机械证据类"
            "登记 finding（附命令+实际输出+期望输出），判不准的按判断类登记。"
        ),
        "self_cover": False,
    },
    {
        "id": "transcript-audit",
        "label": "随机 pod transcript 抽查（诚实条款审计端）",
        "brief": (
            "随机挑一个成员 pod 的会话 jsonl，抽查 pod-guard 拦截记录与后续绕行迹象、"
            "交付正文与台账读数是否自相矛盾。"
        ),
        "self_cover": False,
    },
    {
        "id": "self-maint-audit",
        "label": "随机 pod 自维护层抽查",
        "brief": (
            "随机挑一个成员 pod 的 AGENTS.md 与自建 rules/skills：核对是否仍在讲已经不做"
            "的事（文档与行为漂移），体量是否触预算上限，条件式规则是否真条件式。"
        ),
        "self_cover": False,
    },
    {
        "id": "code-walk",
        "label": "随机代码走查",
        "brief": (
            "在 src/pysci/ 随机选一个模块走查：未被要求的兼容垫片、静默 except、"
            "fail-open 吞掉自身 bug 的写法、以及与 basic.md 约定冲突处。"
        ),
        "self_cover": False,
    },
    {
        "id": "pipeline-self",
        "label": "改进管道自身组件走查（§0.2 自覆盖）",
        "brief": (
            "检查 orch/drain/benchd/inspectd/verdict/jury/guards 这些**改进管道自身**的"
            "组件：判决是否真被闸门强制、有无「文档承诺但代码没实现」的条款。"
            "注意豁口：`pysci-inspect` 自身不在巡检范围（§0.2）。"
        ),
        "self_cover": True,
    },
    {
        "id": "intent-drift",
        "label": "设计意图一致性核对",
        "brief": (
            "挑 README/charter/技能真本里的一段**宣称**，用可执行命令核验实际行为是否一致；"
            "不一致即是缺陷（文档承诺 > 代码实现 = 最坏的一类漂移）。"
        ),
        "self_cover": False,
    },
    {
        "id": "design-rubric",
        "label": "设计审查 rubric 三维",
        "brief": (
            "照 `rubrics/design-review.md` 的三维（社区优先/反冗余/LLM 直觉友好）审一个"
            "选定模块，产出可证伪的改进提案（判断类，须经 J）。"
        ),
        "rubric": True,
        "self_cover": False,
    },
    {
        "id": "bench-probe",
        "label": "抽跑 pysci-bench 队列提案",
        "brief": (
            "读 `orch bench-queue` 面与 evals 台账，挑一个值得复测/重取基线的读数缺口，"
            "经 bench 队列提案（inspector 不自己跑压测）。"
        ),
        "self_cover": False,
    },
)


#: 菜单项按 id 索引（队列项只存 id，任务书正文现取——避免菜单文案在队列里双份漂移）。
PATROL_BY_ID: dict[str, dict[str, Any]] = {e["id"]: e for e in PATROL_MENU}


# ---------------------------------------------------------------------------
# 全局开关与参数
# ---------------------------------------------------------------------------
def enabled() -> bool:
    """巡检引擎是否允许被拉起（§7 全局开关：文件缺席=开启）。"""
    if not ENABLE_PATH.exists():
        return True
    try:
        raw = ENABLE_PATH.read_text(encoding="utf-8").strip().lower()
    except OSError:
        return True
    if not raw:
        return True
    return raw not in ("false", "0", "off", "no")


def set_enabled(on: bool) -> str:
    """写开关（``orch inspector-on/off``），返回落盘值。"""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    ENABLE_PATH.write_text("true\n" if on else "false\n", encoding="utf-8")
    return "true" if on else "false"


def params() -> dict[str, Any]:
    """生效的 inspector 参数（registry 逐键覆盖，非法回落；见 :data:`DEFAULT_INSPECTOR`）。"""
    return Registry.load().inspector


# ---------------------------------------------------------------------------
# 让路（§7）与双队列迟滞（§5.2）
# ---------------------------------------------------------------------------
def _plan_run_active(window_s: int) -> bool:
    """计划侧是否正在派跳：任一未完成计划的 ``.state.json`` mtime 距今小于窗口即算活跃。

    计划本身没有「正在运行」标记（``plan run`` 是组长前台阻塞执行），故用 state 文件的
    新鲜度做判据——写得动它就说明组长正在推进环节。窗口取 0 表示永不判活跃（关掉让路）。
    """
    from .plans import (  # 延迟导入：plans 顶层引 workflow/registry，避免成环
        PLANS_DIR,
    )

    if window_s <= 0 or not PLANS_DIR.exists():
        return False
    now = time.time()
    for p in sorted(PLANS_DIR.glob("*.state.json")):
        try:
            if now - p.stat().st_mtime > window_s:
                continue
            st = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if st.get("completed_at") or st.get("abandoned_at"):
            continue
        return True
    return False


def busy_surfaces(*, p: dict[str, Any] | None = None) -> list[str]:
    """生产侧哪些面正忙（§7 让路判据）。**空表=全闲**，才允许 inspector 派跳与晋升。"""
    from . import benchd  # 延迟导入：benchd 顶层引 eval/replica，非队列常驻路径

    par = p or params()
    out: list[str] = []
    if lock_status()["state"] == "running":
        out.append("devops-drain")
    if lock_status(benchd.BENCH_PATHS)["state"] == "running":
        out.append("bencher-worker")
    if _plan_run_active(int(par["yield_window_s"])):
        out.append("plan-run")
    return out


def depths(*, p: dict[str, Any] | None = None) -> dict[str, int]:
    """双队列积压读数（迟滞闸的输入）：devops backlog pending 与 bencher 队列 pending。"""
    from . import (  # 延迟导入：workflow↔inspectd 成环（approve 面内嵌拉起）
        benchd,
        workflow,
    )

    return {
        "backlog_pending": len(workflow.backlog_pending()),
        "bench_pending": workq.pending_count(benchd.QUEUE_PATH),
    }


def _read_throttle() -> dict[str, Any]:
    if not THROTTLE_PATH.exists():
        return {"produce": "on", "since": _now(), "reason": "初始态"}
    try:
        data = json.loads(THROTTLE_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"produce": "on", "since": _now(), "reason": "台账不可读，按 on 处置"}
    return data if isinstance(data, dict) else {}


def throttle_state(
    *,
    refresh: bool = True,
    p: dict[str, Any] | None = None,
    d: dict[str, int] | None = None,
) -> dict[str, Any]:
    """双队列迟滞闸（§5.2 用户裁决 v0.5）：返回当前产出态，``refresh`` 时推进状态机。

    迟滞的形状：**超过 ``pause_*`` 才暂停，回落到 ``resume_*`` 及以下才恢复**。两个阈值
    相等就是每轮翻转的开关，不叫节流——:meth:`Registry.inspector` 已把非法配置回落出厂值，
    这里再读一次 ``hysteresis_valid`` 供 status 面标注「你配的值被忽略了」。

    Returns:
        ``{"produce": "on"|"paused", "since", "reason", "depths", "thresholds",
        "hysteresis_valid"}``。
    """
    par = p or params()
    st = _read_throttle()
    produce = str(st.get("produce", "on"))
    d = depths(p=par) if d is None else d
    pause_hit = [
        name
        for name, key in (
            ("backlog", "pause_backlog"),
            ("bench", "pause_bench"),
        )
        if d.get(_depth_key(key), 0) > int(par[key])
    ]
    resume_ok = d["backlog_pending"] <= int(par["resume_backlog"]) and d[
        "bench_pending"
    ] <= int(par["resume_bench"])
    if produce == "on" and pause_hit:
        produce, reason = "paused", f"积压超限：{'/'.join(pause_hit)} > pause 阈值"
        since = _now()
    elif produce == "paused" and resume_ok:
        produce, reason = "on", "双队列回落至 resume 阈值及以下，恢复产出"
        since = _now()
    else:
        reason = str(st.get("reason", "维持现态（迟滞窗内）"))
        since = str(st.get("since", _now()))
    if refresh:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        THROTTLE_PATH.write_text(
            json.dumps(
                {
                    "produce": produce,
                    "since": since,
                    "reason": reason,
                    "depths": d,
                    "thresholds": {
                        k: par[k]
                        for k in (
                            "pause_backlog",
                            "pause_bench",
                            "resume_backlog",
                            "resume_bench",
                        )
                    },
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    return {
        "produce": produce,
        "since": since,
        "reason": reason,
        "depths": d,
        "thresholds": {
            k: par[k]
            for k in ("pause_backlog", "pause_bench", "resume_backlog", "resume_bench")
        },
        "hysteresis_valid": bool(par.get("hysteresis_valid", True)),
    }


def _depth_key(threshold_key: str) -> str:
    """``pause_backlog`` → ``backlog_pending``（读数与阈值同名录在，别让人猜映射）。"""
    return threshold_key.removeprefix("pause_") + "_pending"


def activation_gate(*, p: dict[str, Any] | None = None) -> dict[str, Any]:
    """一次激活的闸门读数（**唯一收口**：拉起、循环、晋升都读它，不各算各的）。

    Returns:
        ``{"allow_dispatch": bool, "busy": [...], "throttle": {...}, "reason": str}``——
        ``allow_dispatch`` 为假时只允许零成本读侧检查（巡检不退出，但什么都不派发）。
    """
    par = p or params()
    busy = busy_surfaces(p=par)
    th = throttle_state(refresh=True, p=par)
    reasons = []
    if busy:
        reasons.append(f"让路：{', '.join(busy)} 正忙")
    if th["produce"] == "paused":
        reasons.append(f"迟滞暂停：{th['reason']}")
    return {
        "allow_dispatch": not busy and th["produce"] == "on",
        "busy": busy,
        "throttle": th,
        "reason": "；".join(reasons) or "全闲，允许派跳与晋升",
    }


# ---------------------------------------------------------------------------
# 巡检工作队列（§7① queue-first）
# ---------------------------------------------------------------------------
def enqueue(
    kind: str,
    *,
    brief: str = "",
    requested_by: str = "leader",
    summary: str = "",
    stamp: str | None = None,
    target: str = "",
) -> dict[str, Any]:
    """向 inspector 工作队列追加一项（``patrol``/``assignment``/``tier-a-draft``）。

    Raises:
        ValueError: kind 不在词表。
    """
    if kind not in QUEUE_KINDS:
        raise ValueError(f"未知队列项类型 {kind!r}；可用：{', '.join(QUEUE_KINDS)}")
    item: dict[str, Any] = {
        "id": workq.new_item_id(QUEUE_PATH, _ID_PREFIX, kind, stamp=stamp),
        "kind": kind,
        "brief": brief,
        "target": target,
        "requested_by": requested_by,
        "summary": summary or f"{kind}: {brief[:60] or '（按菜单素材执行）'}",
        "status": workq.STATUS_PENDING,
        "enqueued_at": _now(),
    }
    return workq.append(QUEUE_PATH, item)


def pick_patrol(*, rng: random.Random | None = None) -> dict[str, Any]:
    """从巡检菜单随机抽一项（§7②「随机抽取」）；``rng`` 是测试注入口。"""
    return (rng or random).choice(list(PATROL_MENU))


def patrol_brief(entry: dict[str, Any]) -> str:
    """把菜单项拼成任务书正文（自包含：组员看不到引擎的会话上下文）。"""
    return (
        f"闲时巡检（inspector 自主巡逻，类别={entry['id']}）：{entry['label']}\n\n"
        f"要做的事：{entry['brief']}\n\n"
        "收尾纪律（§5.2）：\n"
        "- 每条发现先落台账：`pysci-inspect finding add --class mechanical|judgment "
        '--title "<一句话现象>" --symptom "<现象>" --repro "<复现命令>" '
        '--evidence "<实际输出>" --expect "<期望输出>"`；\n'
        "- 机械证据类会自动晋升 backlog；判断类必须 "
        "`pysci-inspect submit <finding-id> --jury`（**没有 override 面**，"
        "凭印象自裁在这条通道上不存在）；\n"
        "- 两类都要 `--criteria` 写可证伪判据，缺判据按 §6 总闸不晋升；\n"
        "- 全项目只读：不改任何代码/文档/配置，产出只落 `orchestration/state/inspector/`"
        "（判决案卷落 `orchestration/state/evals/jury-cases/`）。\n"
    )


# ---------------------------------------------------------------------------
# 发现台账与晋升管道（§5.2）
# ---------------------------------------------------------------------------
def findings(*, status: str | None = None) -> list[dict[str, Any]]:
    """读发现台账（jsonl）；可按状态过滤。坏行跳过（同 :func:`workq.load` 的口径）。"""
    if not FINDINGS_PATH.exists():
        return []
    out: list[dict[str, Any]] = []
    for ln in FINDINGS_PATH.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        try:
            rec = json.loads(ln)
        except ValueError:
            continue
        if (
            isinstance(rec, dict)
            and rec.get("id")
            and (status is None or rec.get("status") == status)
        ):
            out.append(rec)
    return out


def finding_by_id(fid: str) -> dict[str, Any] | None:
    """按 id 或 id 前缀取一条发现（找不到返回 None，不抛——CLI 侧据此给可读提示）。"""
    rows = findings()
    exact = [r for r in rows if str(r.get("id")) == fid]
    if exact:
        return exact[-1]
    pref = [r for r in rows if str(r.get("id", "")).startswith(fid)]
    return pref[-1] if len(pref) == 1 else None


def _mutate_finding(fid: str, **fields: Any) -> dict[str, Any] | None:
    """整表重写式改写一条发现（jsonl 无原地更新；与队列同理，靠原子 replace）。"""
    rows = findings()
    hit: dict[str, Any] | None = None
    for i, r in enumerate(rows):
        if str(r.get("id")) == fid:
            r.update(fields)
            rows[i] = r
            hit = r
    if hit is None:
        return None
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = FINDINGS_PATH.with_name(FINDINGS_PATH.name + ".tmp")
    tmp.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8",
    )
    os.replace(str(tmp), str(FINDINGS_PATH))
    return hit


def append_finding(
    *,
    cls: str,
    title: str,
    symptom: str = "",
    repro: str = "",
    evidence: str = "",
    expect: str = "",
    tag: str = "",
    kind: str = "usability",
    criteria: str = "",
    source: str = "inspector",
    stamp: str | None = None,
) -> dict[str, Any]:
    """登记一条发现（§7③：先落 ``findings.jsonl``，再按类别分流）。

    Args:
        cls: ``mechanical``（有复现命令与实际输出）或 ``judgment``（须经判决）。
        kind: 提案分类，进 backlog 条目的 ``kind`` 字段（受控词表 defect/usability）。
        criteria: 可证伪成功判据（§6 总闸：无判据不晋升）。
        stamp: id 时间戳覆盖（测试注入）。

    Raises:
        ValueError: 类别/分类不在词表，或缺 title。
    """
    if cls not in FINDING_CLASSES:
        raise ValueError(f"未知发现类别 {cls!r}；可用：{', '.join(FINDING_CLASSES)}")
    if kind not in ("defect", "usability"):
        raise ValueError(f"未知提案分类 kind={kind!r}（只接受 defect/usability）")
    if not title.strip():
        raise ValueError("发现必须有 title（一句话现象，进 backlog 摘要）")
    if cls == "mechanical" and not (repro.strip() and evidence.strip()):
        raise ValueError(
            "机械证据类必须附复现命令与实际输出（--repro/--evidence）——"
            "「机械证据」的含义就是别人能重跑（basic.md §2 上报纪律）。"
        )
    fid = f"{_ID_PREFIX}-{stamp or datetime.now().strftime('%Y%m%d-%H%M%S')}-{cls}"
    rows = {str(r.get("id")) for r in findings()}
    n = 2
    while fid in rows:
        fid = f"{_ID_PREFIX}-{stamp or datetime.now().strftime('%Y%m%d-%H%M%S')}-{cls}-{n:03d}"
        n += 1
    rec: dict[str, Any] = {
        "id": fid,
        "ts": _now(),
        "class": cls,
        "status": "pending",
        "title": title.strip(),
        "symptom": symptom.strip(),
        "repro": repro.strip(),
        "evidence": evidence.strip(),
        "expect": expect.strip(),
        "tag": tag.strip(),
        "kind": kind,
        "success_criteria": criteria.strip(),
        "source": source,
    }
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with FINDINGS_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


def _evidence_file(rec: dict[str, Any]) -> Path:
    """把发现写成可复核的证据正文（backlog 条目的 ``evidence_file`` 指向它）。"""
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    path = EVIDENCE_DIR / f"{rec['id']}.md"
    path.write_text(
        "\n".join(
            (
                f"# inspector 发现 {rec['id']}（{rec['class']}）",
                "",
                f"- 时间：{rec['ts']}｜来源：{rec.get('source', 'inspector')}",
                f"- 现象：{rec.get('title', '')}",
                f"- 症状：{rec.get('symptom', '') or '（无）'}",
                f"- 复现：`{rec.get('repro', '') or '（判断类无机械复现）'}`",
                f"- 实际输出：{rec.get('evidence', '') or '（无）'}",
                f"- 期望输出：{rec.get('expect', '') or '（无）'}",
                f"- 失败标签：{rec.get('tag', '') or '（未标注）'}",
                "",
                "## 成功判据（提案者写，判决案卷审定，devops 只执行；§6）",
                "",
                rec.get("success_criteria", "") or "（未填写）",
                "",
            )
        ),
        encoding="utf-8",
    )
    return path


def promote_finding(
    rec: dict[str, Any],
    *,
    channel: str,
    ruling: str = "",
    wake: bool = True,
    p: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """晋升发现 → backlog 条目（source=inspector，附证据）+ 决策日志 + 唤醒 devops。

    Args:
        channel: 入账通道（``mechanical``=§5.1 免判决的机械证据类，``jury``=经仲裁）。
            两者都**不计 override 率**——inspector 没有自裁面（§5.2）。
        ruling: 判决理由（仲裁裁决摘要），进 journal ``reason``。

    Returns:
        ``{"action": "promoted", "backlog_id": ..., "position": n}``；闸门不放行时返回
        ``{"action": "held", "reason": ...}``（条目留在 ``pending``，下轮全闲时再投）。
    """
    from . import journal, workflow  # 延迟导入：workflow 顶层引 verdict，避免成环

    gate = activation_gate(p=p)
    if not gate["allow_dispatch"]:
        return {"action": "held", "reason": gate["reason"]}
    bid = str(rec["id"])
    if workflow.backlog_find(bid):
        return {"action": "duplicate", "backlog_id": bid}
    if not (rec.get("success_criteria") or "").strip():
        # 判成 rejected 而非留 pending：worker 每轮都会重扫 pending 的机械类，留着就是
        # 每跳重跑一次同样的拒绝——缺判据的发现要靠补判据复活，不靠重试。
        _mutate_finding(
            bid, status="rejected", note="缺可证伪成功判据，未晋升（§6 总闸）"
        )
        return {
            "action": "rejected_no_criteria",
            "reason": "缺 success_criteria：无判据的提案不可证伪，不进队（§6）",
        }
    ev = _evidence_file(rec)
    entry = {
        "id": bid,
        "member": "inspector",
        "actor": "inspector",
        "source": "inspector",
        "summary": str(rec.get("title", ""))[:200],
        "kind": rec.get("kind", "usability"),
        "success_criteria": rec["success_criteria"],
        "evidence_file": orchestration_rel(ev),
        "class": rec.get("class"),
        "channel": channel,
        "status": "pending",
        "proposed": _now(),
        "note": ruling or "inspector 自主发现（§5.2 不经组长）",
    }
    pos = workflow._backlog_append(entry)
    journal.append(
        {
            "ts": _now(),
            "situation": str(rec.get("title", ""))[:200],
            "options": "",
            "choice": "approve",
            "reason": ruling or "机械证据类：复现命令与实际输出即依据（§5.2）",
            "source": "inspector",
            "kind": "ruling",
            "channel": channel,
            "finding_id": bid,
            "member": "inspector",
        }
    )
    _mutate_finding(bid, status="promoted", backlog_id=bid, promoted_at=_now())
    out = {
        "action": "promoted",
        "backlog_id": bid,
        "position": pos,
        "evidence_file": entry["evidence_file"],
    }
    if wake:
        from .drain import (  # 延迟导入：drain↔workflow 已有环，这里只用出口函数
            wake_devops,
        )

        _, msg = wake_devops()
        out["wake"] = msg
    return out


def submit_judgment(
    rec: dict[str, Any], *, dry: bool = False, timeout: int | None = None
) -> dict[str, Any]:
    """判断类发现经 J 仲裁后晋升（§5.2 强制选择点；inspector **无 override 面**）。

    复用 :func:`verdict.assemble_case` / :func:`verdict.run_jury`：案卷由代码组，
    选项 A=采纳、B=维持现状，与组长侧审批同一个裁决面（§0.9）。

    Returns:
        ``{"outcome": "promoted"|"rejected"|"queued"|"escalated"|"dry", ...读数}``。
    """
    from . import journal, verdict  # 延迟导入：verdict 顶层引 registry/jury

    if rec.get("class") != "judgment":
        return {
            "outcome": "wrong_class",
            "reason": "机械证据类免判决（走 promote_finding），判断类才经 J",
        }
    case = verdict.assemble_case(
        sugg_id=str(rec["id"]),
        summary=str(rec.get("title", "")),
        kind=str(rec.get("kind", "usability")),
        criteria=str(rec.get("success_criteria", "") or "（未填写）"),
        body=_evidence_file(rec).read_text(encoding="utf-8"),
    )
    run = verdict.run_jury(case, dry=dry, timeout=timeout)
    if dry:
        return {"outcome": "dry", "status": run.status, "case": orchestration_rel(case)}
    call = (run.record or {}).get("verdict")
    tally = (run.record or {}).get("tally", {})
    if run.status == "queued":
        _mutate_finding(
            str(rec["id"]), status="queued", note=f"仲裁渠道不可达：{run.message}"
        )
        return {"outcome": "queued", "reason": run.message}
    if run.status != "recorded":
        _mutate_finding(
            str(rec["id"]), status="pending", note=f"开庭失败：{run.message}"
        )
        return {"outcome": "failed", "reason": run.message}
    if call == "A":
        promoted = promote_finding(
            rec,
            channel="jury",
            ruling=f"仲裁裁决 A（采纳）A:{tally.get('A')} B:{tally.get('B')}",
        )
        return {"outcome": promoted.get("action", "?"), "tally": tally, **promoted}
    if call == "B":
        journal.append(
            {
                "ts": _now(),
                "situation": str(rec.get("title", ""))[:200],
                "options": "A=采纳 B=维持现状",
                "choice": "reject",
                "reason": f"仲裁裁决 B A:{tally.get('A')} B:{tally.get('B')}",
                "source": "inspector",
                "kind": "ruling",
                "channel": "jury",
                "finding_id": str(rec["id"]),
                "member": "inspector",
            }
        )
        _mutate_finding(
            str(rec["id"]),
            status="rejected",
            note="仲裁判 B（维持现状），不计 override（§5.1）",
        )
        return {"outcome": "rejected", "tally": tally}
    _mutate_finding(
        str(rec["id"]),
        status="escalated",
        note=f"仲裁模棱两可 → escalate=twin：{run.message}",
    )
    return {"outcome": "escalated", "reason": run.message}


# ---------------------------------------------------------------------------
# 读侧机械检查（引擎自己做，零模型成本；让路时仍允许跑）
# ---------------------------------------------------------------------------
def check_cost_curve(*, factor: float = 3.0, min_rows: int = 8) -> list[dict[str, Any]]:
    """台账成本曲线异常：某跳 ``est_tokens`` 超过同成员均值 ``factor`` 倍即成案（§7②）。

    判据用**相对排名**而非绝对值（``registry.est_tokens_per_char`` 是启发式尺子，两渠道
    同法才可比）。行数不足时不判——样本太小的均值没有解释力。
    """
    from .ledger import (  # 延迟导入：台账读全量，非队列常驻路径
        read_all,
    )

    rows = [r for r in read_all() if getattr(r, "est_tokens", 0)]
    by_member: dict[str, list[int]] = {}
    for r in rows:
        by_member.setdefault(str(getattr(r, "member", "?")), []).append(
            int(getattr(r, "est_tokens", 0))
        )
    out: list[dict[str, Any]] = []
    for member, vals in sorted(by_member.items()):
        if len(vals) < min_rows:
            continue
        mean = sum(vals) / len(vals)
        if mean <= 0:
            continue
        tail = [v for v in vals[-3:] if v > mean * factor]
        if not tail:
            continue
        out.append(
            {
                "cls": "mechanical",
                "kind": "usability",
                "title": f"{member} 最近三跳的上下文用量超均值 {factor:.0f} 倍",
                "symptom": "疑似上下文膨胀（长文档反复注入/会话未归档）",
                "repro": f"orch ledger --member {member} --days 14",
                "evidence": (f"均值={mean:.0f}，尾部读数={tail}，样本={len(vals)} 跳"),
                "expect": "尾部读数回落到均值 3 倍以内，或该会话归档后重新起会话",
                "tag": "",
                "criteria": (
                    f"- {member} 最近 3 跳的 est_tokens 均低于其滑窗均值的 {factor:.0f} 倍"
                    "（`orch stats` 成本列可复核）"
                ),
                "source": "inspectd",
            }
        )
    return out


def check_throttle_config() -> list[dict[str, Any]]:
    """迟滞配置健全性：``resume_*`` 必须严格低于 ``pause_*``（非法值已被回落并标注）。"""
    par = params()
    if par.get("hysteresis_valid", True):
        return []
    return [
        {
            "cls": "mechanical",
            "kind": "defect",
            "title": "registry.inspector 迟滞阈值配置非法，已回落出厂值",
            "symptom": "resume_* 未严格低于 pause_*，闸会变成每轮翻转的开关",
            "repro": "orch inspector-status",
            "evidence": f"生效值：{par}",
            "expect": "hysteresis_valid=true 且生效阈值等于所配阈值",
            "tag": "",
            "criteria": "- `orch inspector-status` 不再报「配置被忽略」",
            "source": "inspectd",
        }
    ]


def run_readonly_checks(*, dry: bool = False) -> list[str]:
    """跑引擎侧机械检查并登记发现，返回新登记条目的 id 列表（不晋升，晋升看闸门）。

    **按 (title, evidence) 去重**：读侧检查是确定性的（同一份台账每轮算出同一条异常），而
    发现 id 带秒级时间戳——不去重就是每次激活给同一件事立案一次，backlog 被自己人淹掉，
    而 §2.4 的指纹去重只管失败标签、管不到这里。取「现象+读数」做指纹而不是只取现象：
    同一成员**换了量级**的再次膨胀是新事实，要再立案（只按 title 去重会把复发永久吞掉）。
    """
    seen = {(str(r.get("title", "")), str(r.get("evidence", ""))) for r in findings()}
    found: list[str] = []
    for cand in [*check_cost_curve(), *check_throttle_config()]:
        if (str(cand.get("title", "")), str(cand.get("evidence", ""))) in seen:
            continue
        rec = append_finding(**cand)
        found.append(str(rec["id"]))
        if dry:
            break
    return found


# ---------------------------------------------------------------------------
# 条目执行与 worker 循环
# ---------------------------------------------------------------------------
def _run_patrol(item: dict[str, Any], run_log: Path, *, dry: bool) -> dict[str, Any]:
    """派一跳给 inspector 组员（§7 的 LLM 侧工作都在这里，受预算与让路双重钳制）。"""
    from .dispatch import do_dispatch  # 延迟导入：与 drain/benchd 同样的环规避

    entry = dict(PATROL_BY_ID.get(str(item.get("target"))) or {})
    brief = str(item.get("brief") or "").strip()
    if not brief:
        if not entry:
            raise ValueError(
                f"巡检项缺 brief 且 target 不在菜单：{item.get('target')!r}"
            )
        brief = patrol_brief(entry)
        item = workq.mutate(QUEUE_PATH, str(item["id"]), target=entry["id"]) or item
    if dry:
        _log(
            run_log,
            f"[dry] 巡检 {item.get('id')}（{entry.get('label', '指派项')}）演练",
        )
        return {"kind": "dry", "hop": True}
    out = do_dispatch(
        "inspector",
        text=brief,
        slug=str(item["id"]),
        session="latest",
        quiet=True,
    )
    return {
        "hop": True,
        "delivery": out.kind,
        "code": out.code,
        "body": (out.body or "")[:2000],
        "quota_exhausted": out.kind == "quota_exhausted",
    }


_RUNNERS = {
    "patrol": _run_patrol,
    "assignment": _run_patrol,
    "tier-a-draft": _run_patrol,
}


def run_item(
    item: dict[str, Any], run_log: Path, *, dry: bool = False
) -> dict[str, Any]:
    """执行一个队列项并写终态。异常不外抛：后台无人接盘，一项坏不许拖垮整轮。"""
    t0 = time.time()
    runner = _RUNNERS.get(str(item.get("kind")))
    if runner is None:
        result = {"error": f"未知类型 {item.get('kind')!r}", "status": "failed"}
    else:
        try:
            result = runner(item, run_log, dry=dry)
            result["status"] = "failed" if result.get("error") else "done"
        except Exception as exc:  # noqa: BLE001 — 后台无人值守：异常即终态
            result = {"error": f"{type(exc).__name__}: {exc}", "status": "failed"}
            _log(run_log, f"[!] {item.get('id')} 失败：{result['error']}")
    result["elapsed_s"] = round(time.time() - t0, 1)
    workq.finish(QUEUE_PATH, str(item["id"]), result["status"], result)
    return result


def inspect_drain(
    run_log: Path, *, dry: bool = False, rng: random.Random | None = None
) -> dict[str, Any]:
    """一次激活：持锁 → 读侧检查 → 闸门放行才派跳/晋升 → 预算到顶自停（§7）。

    与 bencher worker 的差别是**队列空不自毁**：§7 的工作循环是 queue-first，队列空
    则从巡检菜单抽一项继续做；退出条件只有四个——协作停止、开关关闭、预算到顶、让路。

    Returns:
        ``.done`` 摘要（``started/completed/count/hops/elapsed_s/items/run_log`` +
        ``stopped`` ∈ cooperative|switch_off|budget_hops|budget_wallclock|yield|throttled）。
    """
    par = params()
    done_path = run_log.with_suffix(".done")
    lock = acquire_lock(orchestration_rel(run_log), INSPECT_PATHS)
    if lock is None:
        _log(
            run_log, "[i] 另一 inspector worker 持锁运行中，本进程退出（不重复激活）。"
        )
        return {"skipped": "lock_busy", "run_log": orchestration_rel(run_log)}

    t0 = time.time()
    summary: dict[str, Any] = {
        "started": _now(),
        "run_log": orchestration_rel(run_log),
        "dry": dry,
        "items": [],
        "hops": 0,
    }
    _log(
        run_log,
        f"[▶] inspector worker 启动（pid={os.getpid()}, dry={dry}, "
        f"预算 max_hops={par['max_hops']} wallclock={par['wallclock_s']}s）",
    )
    orphans = workq.reset_orphans(QUEUE_PATH)
    if orphans:
        summary["reclaimed"] = orphans
        _log(run_log, f"[i] 回收孤儿在途条目：{', '.join(orphans)}")
    try:
        while True:
            if _consume_stop_request(INSPECT_PATHS):
                summary["stopped"] = "cooperative"
                _log(run_log, "[s] 收到协作停止信号，干净退出（条目间隙）")
                break
            if not enabled():
                summary["stopped"] = "switch_off"
                _log(
                    run_log, "[o] 全局开关已关闭（state/inspector.enabled=false），自停"
                )
                break
            if time.time() - t0 > int(par["wallclock_s"]):
                summary["stopped"] = "budget_wallclock"
                _log(run_log, f"[$] 墙钟预算 {par['wallclock_s']}s 到顶，自停")
                break
            if summary["hops"] >= int(par["max_hops"]):
                summary["stopped"] = "budget_hops"
                _log(run_log, f"[$] 跳数预算 {par['max_hops']} 到顶，自停")
                break
            gate = activation_gate(p=par)
            if not dry:
                # 读侧检查零模型成本，让路/暂停时照跑（§7「巡检读侧可继续」）。
                new_ids = run_readonly_checks()
                if new_ids:
                    summary.setdefault("findings", []).extend(new_ids)
                    _log(run_log, f"[f] 引擎侧机械检查新登记：{', '.join(new_ids)}")
            if not gate["allow_dispatch"]:
                summary["stopped"] = "yield" if gate["busy"] else "throttled"
                _log(run_log, f"[y] {gate['reason']} → 不派跳、不晋升，本轮收束")
                break
            for rec in findings(status="pending"):
                if rec.get("class") != "mechanical":
                    continue
                got = promote_finding(rec, channel="mechanical", p=par)
                if got.get("action") == "promoted":
                    _log(
                        run_log,
                        f"[↑] 机械证据类自动晋升 {rec['id']} → backlog 第 "
                        f"{got.get('position')} 位",
                    )
            item = workq.take_next(QUEUE_PATH)
            if item is None:
                entry = pick_patrol(rng=rng)
                item = enqueue("patrol", target=entry["id"], brief="")
                _log(run_log, f"[→] 队列空，抽巡检项 {entry['id']}（{entry['label']}）")
            result = run_item(item, run_log, dry=dry)
            if result.get("hop"):
                summary["hops"] = int(summary["hops"]) + 1
            summary["items"].append(
                {
                    "id": item["id"],
                    "kind": item.get("kind"),
                    "status": result.get("status"),
                    "elapsed_s": result.get("elapsed_s"),
                }
            )
            if result.get("quota_exhausted"):
                summary["stopped"] = "quota_exhausted"
                summary["error"] = "系统性故障：模型渠道额度耗尽（quota_exhausted）"
                _log(run_log, "[Q] 全局停止：额度耗尽，释放锁")
                break
    except Exception as exc:  # noqa: BLE001 — 记录后仍写 .done（不留黑洞）
        summary["error"] = f"{type(exc).__name__}: {exc}"
        _log(run_log, f"[!] inspector worker 异常中断：{summary['error']}")
    finally:
        summary["count"] = len(summary["items"])
        summary["completed"] = _now()
        summary["elapsed_s"] = round(time.time() - t0, 1)
        release_lock(run_log, INSPECT_PATHS)
        try:
            done_path.parent.mkdir(parents=True, exist_ok=True)
            done_path.write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        except OSError as exc:
            _log(run_log, f"[!] .done 写入失败：{exc}")
    _log(
        run_log,
        f"[√] inspector worker 结束：派跳 {summary['hops']} 次，"
        f"处理 {summary['count']} 项，耗时 {summary['elapsed_s']}s",
    )
    return summary


# ---------------------------------------------------------------------------
# 拉起（幂等）与内嵌入口
# ---------------------------------------------------------------------------
def _worker_argv(run_log: Path, *, dry: bool) -> list[str]:
    argv = [
        sys.executable,
        "-m",
        "pysci.skills.orchestration.tools.orch",
        "_inspect-worker",
        "--run-log",
        str(run_log),
    ]
    if dry:
        argv.append("--dry")
    return argv


def wake_inspector(
    *, force_dry: bool | None = None, ignore_cooldown: bool = False
) -> tuple[bool, str]:
    """幂等拉起 inspector worker（§7「生命周期：代码保证」）。

    已在跑 / 开关关 / 冷却窗内（``ignore_cooldown`` 可越）→ 不派生。
    返回 ``(spawned, message)``；消息里必带**为什么没拉起**，静默 no-op 会让人以为
    设施坏了。
    """
    if not enabled():
        return (
            False,
            "inspector 全局开关已关闭（state/inspector.enabled=false），不拉起。",
        )
    st = lock_status(INSPECT_PATHS)
    if st["state"] == "running":
        return False, "inspector worker 正在运行（不重复激活）。"
    par = params()
    done = latest_done(INSPECT_PATHS)
    if done and not ignore_cooldown:
        try:
            ended = datetime.fromisoformat(str(done.get("completed") or ""))
            age = (datetime.now().astimezone() - ended).total_seconds()
        except (ValueError, TypeError):
            age = None
        if age is not None and age < int(par["cooldown_s"]):
            return (
                False,
                f"冷却窗内（上轮 {done.get('name')} 结束于 {int(age)}s 前，"
                f"cooldown={par['cooldown_s']}s），本轮不激活。",
            )
    INSPECT_PATHS.runs.mkdir(parents=True, exist_ok=True)
    run_log = new_run_log(INSPECT_PATHS)
    dry = bool(os.environ.get(_DRY_ENV)) if force_dry is None else bool(force_dry)
    env = {**os.environ, "PYTHONUTF8": "1"}
    with open(run_log, "ab") as logf:
        _spawn_detached(_worker_argv(run_log, dry=dry), env=env, stdout=logf)
    return True, (
        f"inspector worker 已后台拉起（日志 {orchestration_rel(run_log)}；"
        f"预算 hops={par['max_hops']} wallclock={par['wallclock_s']}s）"
    )


def ensure_inspector(*, quiet: bool = True) -> tuple[bool, str]:
    """内嵌拉起：``orch status``/``approve``/``dispatch`` 入口调用（§7「代码保证」）。

    与 :func:`wake_inspector` 的差别只在**容错**：这是组长命令的副作用，任何异常都
    不该让 ``orch status`` 失败——巡检设施抖动时组长的正事必须照跑（fail-soft），
    但要把抖动留在消息里而不是吞干净。
    """
    try:
        spawned, msg = wake_inspector()
    except Exception as exc:  # noqa: BLE001 — 内嵌副作用不许打断组长命令
        return False, f"inspector 拉起异常（不影响本命令）：{type(exc).__name__}: {exc}"
    if not quiet:
        print(f"    [i] {msg}")
    return spawned, msg


def inspector_status() -> dict[str, Any]:
    """status 面读数：锁态/开关/闸门/队列/发现分布。"""
    par = params()
    return {
        "enabled": enabled(),
        "lock": lock_status(INSPECT_PATHS),
        "cancel": cancel_status(INSPECT_PATHS),
        "params": par,
        "throttle": throttle_state(refresh=True, p=par),
        "busy": busy_surfaces(p=par),
        "queue": workq.counts(QUEUE_PATH),
        "findings": _finding_counts(),
        "last_done": latest_done(INSPECT_PATHS),
    }


def _finding_counts() -> dict[str, int]:
    out: dict[str, int] = {}
    for rec in findings():
        st = str(rec.get("status", "?"))
        out[st] = out.get(st, 0) + 1
    return out


def status_lines() -> list[str]:
    """``orch status`` 的 inspector 段（开关关闭也要在场——缺席会被读成「没配」）。"""
    s = inspector_status()
    if not s["enabled"]:
        return [
            "    [o] inspector 全局开关=关闭（orch inspector-on 开启；拉起逻辑 no-op）"
        ]
    lock = s["lock"]
    state = lock["state"]
    head = f"    inspector worker={state}"
    if state == "running":
        head += f"（pid {lock.get('pid')}，启动 {lock.get('started')}）"
    th = s["throttle"]
    gate = "可派跳" if (not s["busy"] and th["produce"] == "on") else "只读"
    line2 = (
        f"    闸门={gate}：让路={', '.join(s['busy']) or '无'}｜"
        f"产出={th['produce']}（backlog={th['depths']['backlog_pending']}/"
        f"bench={th['depths']['bench_pending']}，"
        f"暂停>{th['thresholds']['pause_backlog']}/{th['thresholds']['pause_bench']} "
        f"恢复≤{th['thresholds']['resume_backlog']}/{th['thresholds']['resume_bench']}）"
    )
    f = s["findings"]
    line3 = (
        f"    发现：待处理={f.get('pending', 0)} 已晋升={f.get('promoted', 0)} "
        f"已驳回={f.get('rejected', 0)} 仲裁排队={f.get('queued', 0)} "
        f"挂起终裁={f.get('escalated', 0)}｜队列 pending={s['queue'].get('pending', 0)}"
    )
    lines = [head, line2, line3]
    if not th["hysteresis_valid"]:
        lines.append(
            "    [!] registry.inspector 阈值非法（resume 须严格低于 pause），已回落出厂值"
        )
    if th["produce"] == "paused":
        lines.append(f"    [!] 迟滞暂停：{th['reason']}")
    for rec in findings(status="escalated")[:3]:
        lines.append(
            f"    [!] 待 twin 终裁：{rec['id']} {str(rec.get('title', ''))[:40]}"
        )
    return lines


# ---------------------------------------------------------------------------
# CLI 面（orch inspector-*）
# ---------------------------------------------------------------------------
def cmd_inspector_up(args: argparse.Namespace) -> int:
    """``orch inspector-up``：幂等拉起（``--dry`` 演练，``--force`` 越冷却窗）。

    ``--force`` 只越**冷却**，不越**开关**，也不越**锁**：开关是用户的成本控制裁决
    （要开就明确 ``orch inspector-on``），锁是幂等性的定义本身。临时翻开关再改回来还会
    和刚派生的 worker 抢同一个文件（它一读见 false 就 self-stop），不是「强制」是竞态。
    """
    spawned, msg = wake_inspector(
        force_dry=True if args.dry else None, ignore_cooldown=bool(args.force)
    )
    mark = "▶" if spawned else "i"
    extra = "（--force：已越冷却窗）" if args.force and spawned else ""
    print(f"[{mark}] {msg}{extra}")
    if not spawned and args.force:
        return 2
    return 0


def _cmd_switch(on: bool) -> int:
    val = set_enabled(on)
    label = "开启" if on else "关闭"
    tail = (
        "后续 orch 命令的自动拉起恢复生效。"
        if on
        else "已在跑的 worker 会在下一个条目间隙自停（stopped=switch_off）。"
    )
    print(f"[√] inspector 全局开关={val}（{label}）；{tail}")
    return 0


def cmd_inspector_on(args: argparse.Namespace) -> int:
    """``orch inspector-on``：打开全局开关（默认即开启，本命令用于关掉后再开）。"""
    del args
    return _cmd_switch(True)


def cmd_inspector_off(args: argparse.Namespace) -> int:
    """``orch inspector-off``：关闭全局开关（成本控制，§7 用户裁决）。"""
    del args
    rc = _cmd_switch(False)
    # 开关只管「不再激活」；已在跑的 worker 靠协作停止信号在条目间隙干净退出，
    # 否则关闭要等到下一轮预算耗尽才见效。
    if (
        cancel_status(INSPECT_PATHS) is None
        and lock_status(INSPECT_PATHS)["state"] == "running"
    ):
        ok, msg = request_stop(INSPECT_PATHS)
        print(f"    {'[▶]' if ok else '[i]'} 顺带请求在跑 worker 协作停止：{msg}")
    return rc


def cmd_inspector_status(args: argparse.Namespace) -> int:
    """``orch inspector-status``：引擎面细读（闸门/阈值/队列/发现）。"""
    del args
    for line in status_lines():
        print(line)
    items = workq.load(QUEUE_PATH)
    for it in items[-10:]:
        print(
            f"    · {it['id']} [{it.get('status')}] {str(it.get('summary', ''))[:60]}"
        )
    return 0


def _enqueue_and_wake(kind: str, **kw: Any) -> int:
    """入队 + 按 ``--start`` 拉起（两个 CLI 面共用，失败口径一致）。"""
    start = bool(kw.pop("start", False))
    try:
        item = enqueue(kind, requested_by="leader", **kw)
    except ValueError as exc:
        print(f"[✗] 入队被拒：{exc}")
        return 2
    print(f"[√] 已入队 {item['id']}（{kind}）：{item['summary']}")
    if start:
        spawned, msg = wake_inspector()
        print(f"[{'▶' if spawned else 'i'}] {msg}")
    else:
        print("    [NEXT] `orch inspector-up` 拉起 worker（或 --start 入队即拉起）。")
    return 0


def cmd_inspector_patrol(args: argparse.Namespace) -> int:
    """``orch inspector-patrol``：手动排一项巡检（不给 ``--target`` 就随机抽一项）。"""
    target = str(args.target or "")
    if target and target not in PATROL_BY_ID:
        print(f"[✗] 菜单项 {target!r} 不存在；可用：{', '.join(sorted(PATROL_BY_ID))}")
        return 2
    if not target:
        # 空 target 的队列项是死信：_run_patrol 既无 brief 也无菜单可查，只能失败。
        target = pick_patrol()["id"]
    return _enqueue_and_wake(
        "patrol", target=target, summary=str(args.summary or ""), start=bool(args.start)
    )


def cmd_inspector_assign(args: argparse.Namespace) -> int:
    """``orch inspector-assign``：组长指派一项核查（kind=assignment，带任务书正文）。"""
    brief = str(args.brief or "").strip()
    if not brief:
        print("[✗] 指派项需要 --brief（核查任务书正文；空指派等于让组员猜）")
        return 2
    return _enqueue_and_wake(
        "assignment",
        brief=brief,
        summary=str(args.summary or ""),
        start=bool(args.start),
    )


def cmd_inspect_worker(args: argparse.Namespace) -> int:
    """``_inspect-worker``（内部）：一次激活的巡检循环。

    **不向 stdout 打印任何东西**：派生时 stdout 就重定向进了本 run 的 ``.log``
    （照 drain/benchd 的写法），一次 ``print`` 会从偏移 0 覆写整段运行日志——实测
    2026-10-12：run 日志只剩摘要 JSON，巡检过程（让路/闸门/派跳）全灭。摘要已落
    ``.done``，读它，不读 stdout。
    """
    run_log = Path(args.run_log) if args.run_log else new_run_log(INSPECT_PATHS)
    summary = inspect_drain(run_log, dry=bool(args.dry))
    return 0 if summary.get("skipped") != "lock_busy" else 1
