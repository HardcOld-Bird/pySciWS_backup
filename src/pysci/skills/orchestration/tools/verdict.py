"""组长判决包（README §5.3）——审批通道的机械闸 + 交付后核验闭环。

§5.1 分流：缺陷类且当条交付含 ``blocked``（in-vivo 实测即依据）→ 保留「直接 approve」
的现行为；其余提案必须落入四态之一——**J**（``--jury`` 阻塞仲裁，复用 :mod:`.jury`
共享库）/ **M**（``--bench <案卷id>`` 实测）/ **J+M**（双，实测覆盖仲裁）/ **都不选
= override**（``--override "理由"`` 留痕）。闸在代码里，不靠组长记住规程。

§5.6 交付后核验：核验方式在**审批时**一次定死，交付后无人再裁决——带 M 的条目入队时
捕获基线 commit，devops 合并销账时自动排一条 ``ab`` 队列项，判决落地三态：改善→销账、
未改善→自动生成纠正提案（``source=post-verify``，新案卷进正常判决流）、inconclusive→
挂起呈组长（``escalate=twin`` 标记，twin 终裁接线属 E6 全量）。

通道用量与 override 率都从决策日志回读（经 J 或 M 判决返回的 reject/纠正**不算
override**，§5.1），所以每条判决都在 journal 落 ``channel`` 字段，读数无需第二本账。

组员侧同名能力（inspector 咨询 twin）经本模块复用，不另起 CLI（§0.9 一成员一技能一 CLI）。
"""

from __future__ import annotations

import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from pysci.paths import ORCH_STATE_ROOT, PROJECT_ROOT

from . import channels, journal, jury
from .registry import Registry, orchestration_rel

#: 审批时自动组的案卷落盘处（``evals/`` 整面 gitignored——案卷是运行态，
#: 长期资产是 :data:`.jury.JURY_DIR` 里的裁决留痕）。
JURY_CASE_DIR: Path = ORCH_STATE_ROOT / "evals" / "jury-cases"

#: 判决通道（§5.1 四态 + 缺陷例外通道）；``channel`` 字段取值即此集合。
CHANNELS: tuple[str, ...] = ("jury", "bench", "jury+bench", "override")
#: 缺陷类且交付含 blocked 的例外通道（现行为保留，不计入 override 率分子）。
CHANNEL_EXCEPTION = "exception"

#: 交付后核验阶段（写在 backlog 条目的 ``verify`` 字段里，防重复入队）。
STAGE_APPROVED = "approved"
STAGE_VERIFYING = "verifying"
STAGE_CLEARED = "cleared"
STAGE_CORRECTED = "corrected"
STAGE_ESCALATED = "escalated"

_DELIVERY_RE = re.compile(r"^\s*delivery\s*:\s*(?P<val>\S+)\s*$", re.IGNORECASE)


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# 分流判据：例外通道与通道归一
# ---------------------------------------------------------------------------
def delivery_of(body: str) -> str:
    """读建议头部的 ``delivery:`` 标记行，返回提请该建议的那一跳交付类型（``result``/``blocked``）。

    标记行由 :func:`.dispatch._persist_suggestion` 落盘（E5b 起）。缺失返回空串——
    来自旧版本或人工撰写的建议本就没有当条交付，**不满足**例外通道（例外通道的前提是
    「这条提案背后有一次真实失败」）。
    """
    for ln in body.splitlines()[:8]:
        m = _DELIVERY_RE.match(ln)
        if m:
            return m.group("val").lower()
    return ""


def is_exception_channel(kind: str, delivery: str) -> bool:
    """§5.1 例外通道：缺陷类**且**当条交付含 blocked（in-vivo 实测即依据）。"""
    return kind == "defect" and delivery == "blocked"


def parse_bench_cases(spec: str) -> list[str]:
    """``--bench`` 取值 → 案 id 列表（逗号/空白分隔；空列表=全套案卷）。"""
    return [t for t in re.split(r"[,\s]+", (spec or "").strip()) if t]


def channel_of(*, jury_on: bool, bench_on: bool, override_on: bool) -> str:
    """三开关归一成通道名；override 与 J/M 互斥时返回 ``"conflict"``。"""
    if override_on and (jury_on or bench_on):
        return "conflict"
    if jury_on and bench_on:
        return "jury+bench"
    if jury_on:
        return "jury"
    if bench_on:
        return "bench"
    if override_on:
        return "override"
    return ""


def validate(args: Any, *, kind: str, delivery: str) -> tuple[str, str]:
    """审批判决包闸：返回 ``(通道, 错误提示)``——通道非空即放行，提示非空即拒绝。

    Args:
        args: ``orch approve`` 的参数对象（读 ``jury``/``bench``/``override``；
            程序化调用方可缺这些属性，按未提供处理）。
        kind: 建议分类（``defect``/``usability``，已过词表闸）。
        delivery: 提请建议那一跳的交付类型（:func:`delivery_of`）。

    Returns:
        放行时 ``(通道, "")``，通道 ∈ :data:`CHANNELS` 加
        :data:`CHANNEL_EXCEPTION`；拒绝时 ``("", 可读提示)``，调用方据此非零退出。
    """
    jury_on = bool(getattr(args, "jury", False))
    bench_spec = str(getattr(args, "bench", "") or "").strip()
    bench_on = bool(bench_spec)
    override_on = bool(str(getattr(args, "override", "") or "").strip())
    ch = channel_of(jury_on=jury_on, bench_on=bench_on, override_on=override_on)
    if ch == "conflict":
        return "", (
            "[!] 判决包参数冲突：--override 是「不选 J/M」的分支（README §5.3），"
            "不能与 --jury/--bench 同时给。\n"
            "    要用仲裁或实测就只给 --jury / --bench；要凭自己的判断直接定案就只给 "
            '--override "<理由>"。'
        )
    if ch:
        return ch, ""
    if is_exception_channel(kind, delivery):
        return CHANNEL_EXCEPTION, ""
    return "", (
        f"[!] 缺判决包，拒绝入队（README §5.3：基建判断须与组长职责事实分离）。\n"
        f"    建议 '{getattr(args, 'suggestion_id', '?')}' 的分类是 kind={kind}、"
        f"提请交付 delivery={delivery or '（无标记）'}，不在例外通道内"
        "（例外通道=缺陷类且当条交付含 blocked，那是 in-vivo 实测即依据）。\n"
        "    四态必居其一：\n"
        "      J   approve <id> --jury                 阻塞仲裁（组卷→双序两局→留痕）\n"
        "      M   approve <id> --bench <案id[,案id]>   实测（捕获基线 + 合并后自动 A/B）\n"
        "      J+M approve <id> --jury --bench <案id>   双轨，实测覆盖仲裁\n"
        '      —   approve <id> --override "<理由>"     组长自裁（留痕，计入 override 率）'
    )


# ---------------------------------------------------------------------------
# J 路：自动组卷 → 仲裁
# ---------------------------------------------------------------------------
def assemble_case(
    *, sugg_id: str, summary: str, kind: str, criteria: str, body: str
) -> Path:
    """把待审批建议自动摆成 :mod:`.jury` 案卷（选项 A=采纳实施，B=维持现状）。

    ``## 证据`` 只放可证伪的成功判据——建议正文已整体进 ``## 选项 A``，重复附证据会让
    裁判把同一份文字算两次分量。
    """
    JURY_CASE_DIR.mkdir(parents=True, exist_ok=True)
    path = JURY_CASE_DIR / f"{sugg_id}.md"
    text = "\n".join(
        (
            f"# 案卷 {sugg_id}（approve --jury 自动组卷）",
            "",
            "## 问题",
            "",
            f"是否采纳这条基建改进提案并派 devops 实施？提案摘要：{summary}",
            "",
            "## 选项 A",
            "",
            f"采纳（kind={kind}）。提案全文：",
            "",
            body.strip(),
            "",
            "## 选项 B",
            "",
            "暂不采纳：维持现状——这套摩擦继续由组长和组员的记忆与文档吸收，"
            "不占 backlog、不派 devops、不增加设施复杂度。",
            "",
            "## 证据",
            "",
            "成功判据（做完后可机械验证的条款）：",
            criteria.strip(),
            "",
        )
    )
    path.write_text(text, encoding="utf-8")
    return path


def run_jury(path: Path, *, dry: bool = False, timeout: int | None = None) -> jury.Run:
    """对自动组好的案卷开庭（阻塞）；渠道不可用时按 :mod:`.jury` 语义降级入队。"""
    return jury.adjudicate(jury.parse_case(path), dry=dry, deadline_s=timeout)


# ---------------------------------------------------------------------------
# M 路：审批时定死核验方式（基线 commit + 交付后自动 A/B）
# ---------------------------------------------------------------------------
def head_commit(*, short: bool = False) -> str:
    """当前 HEAD 的 commit hash（基线锚点；取不到返回空串，调用方据此拒绝标 M）。"""
    args = ["rev-parse", "--short" if short else "--verify", "HEAD"]
    try:
        out = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), *args],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout.strip() if out.returncode == 0 else ""


def capture_baseline(cases: list[str], *, item_id: str) -> dict[str, Any]:
    """审批时入队基线捕获（``capture`` 项），返回 ``verify`` 字段载体。

    基线锚点是**审批时刻的 HEAD commit**——它与 ``capture`` 何时真跑无关：副本 A/B 按
    ref 检出，故即使 bencher 排到几小时后才消化，``ref_a`` 仍指向提案实施前的状态。
    """
    commit = head_commit()
    if not commit:
        raise RuntimeError("取不到 HEAD commit（git 不可用或不在仓库内），无法锚定基线")
    from . import benchd  # 延迟导入：benchd→drain→workflow→本模块，顶层导入成环

    item = benchd.enqueue(
        "capture",
        cases=list(cases or []),
        ref=commit,
        label=f"baseline-{item_id}",
        requested_by="leader",
        summary=f"审批基线：{item_id}",
    )
    return {
        "stage": STAGE_APPROVED,
        "cases": list(cases or []),
        "baseline_commit": commit,
        "baseline_item": str(item.get("id", "")),
        "label": f"post-verify-{item_id}",
    }


def post_verify_enqueue(entry: dict[str, Any]) -> dict[str, Any] | None:
    """销账时自动排 A/B 核验项（§5.6：交付后无人再裁决，只跑审批时定死的核验）。

    只认 ``verify.stage == "approved"`` 的条目，故重复调用（批量消化、孤儿重跑）不会
    排出第二项；无 ``--bench`` 标记的条目返回 ``None``。
    """
    from . import benchd  # 延迟导入：同上（benchd→drain→workflow）
    from .workflow import (  # 延迟导入：workflow 顶层依赖本模块
        _backlog_set_status,
    )

    verify = dict(entry.get("verify") or {})
    if verify.get("stage") != STAGE_APPROVED:
        return None
    item_id = str(entry.get("id", ""))
    item = benchd.enqueue(
        "ab",
        cases=list(verify.get("cases") or []),
        ref_a=str(verify.get("baseline_commit") or ""),
        ref_b="HEAD",
        label=str(verify.get("label") or f"post-verify-{item_id}"),
        requested_by="leader",
        summary=f"交付后核验：{item_id}",
        backlog_id=item_id,
    )
    verify.update(stage=STAGE_VERIFYING, verify_item=str(item.get("id", "")))
    _backlog_set_status(
        item_id, str(entry.get("status", "done")), extra={"verify": verify}
    )
    return item


# ---------------------------------------------------------------------------
# 核验判决落地（benchd 的 ab 项消化后回调）
# ---------------------------------------------------------------------------
def apply_ab_verdict(item: dict[str, Any], ab_doc: dict[str, Any]) -> dict[str, Any]:
    """把 A/B 判决落到提案账上：改善→销账、未改善→自动立案、inconclusive→挂起。

    Args:
        item: bench 队列里的 ``ab`` 条目（须带 ``backlog_id``，否则视为普通测量）。
        ab_doc: :func:`.replica.run_ab` 的返回值（``verdict`` ∈ 改善/未改善/inconclusive）。

    Returns:
        ``{"action": ..., ...读数}``；``action`` ∈ cleared / correction / escalated /
        skipped（非核验项）。落盘副作用：backlog 条目 ``verify`` 字段、决策日志行，
        未改善时另立一份待审批建议。
    """
    bid = str(item.get("backlog_id") or "")
    if not bid:
        return {"action": "skipped"}
    from .workflow import (  # 延迟导入：见 post_verify_enqueue 同注
        _backlog_set_status,
        backlog_find,
    )

    entry = backlog_find(bid) or {}
    verify = dict(entry.get("verify") or {})
    verdict = str(ab_doc.get("verdict") or "")
    label = str(ab_doc.get("label") or item.get("label") or "")
    readout = f"A/B[{label}] verdict={verdict or '空'}"
    out: dict[str, Any] = {"action": "", "backlog_id": bid, "verdict": verdict}
    if verdict == "改善":
        verify.update(stage=STAGE_CLEARED, result=readout)
        _backlog_set_status(
            bid,
            "done",
            note=f"核验通过：{readout}（§5.6 销账）",
            stamp="verified_at",
            extra={"verify": verify},
        )
        out["action"] = "cleared"
    elif verdict == "未改善":
        sid = file_correction_suggestion(bid, entry, ab_doc)
        verify.update(stage=STAGE_CORRECTED, result=readout, correction=sid)
        _backlog_set_status(
            bid,
            str(entry.get("status", "done")),
            note=f"核验未通过：{readout} → 已自动立案 {sid}",
            extra={"verify": verify},
        )
        out.update(action="correction", suggestion_id=sid)
    else:
        verify.update(stage=STAGE_ESCALATED, result=readout, escalate="twin")
        _backlog_set_status(
            bid,
            str(entry.get("status", "done")),
            note=f"核验不可判：{readout} → 挂起呈组长（escalate=twin）",
            extra={"verify": verify},
        )
        out["action"] = "escalated"
    journal.append(
        {
            "ts": _now(),
            "situation": f"交付后核验 {bid}：{readout}",
            "options": "",
            "choice": out["action"],
            "reason": str(ab_doc.get("error") or ab_doc.get("compared") or ""),
            "source": "post-verify",
            "kind": "verdict",
            "channel": "bench",
            "backlog_id": bid,
        }
    )
    return out


def file_correction_suggestion(
    bid: str, entry: dict[str, Any], ab_doc: dict[str, Any]
) -> str:
    """未改善 → 自动生成纠正提案（新证据=A/B 数据，进正常判决流，§5.6）。

    分类固定 ``defect``：机制宣称要改善某读数而实测没改善，行为与声明不符即缺陷。
    交付类型不写 ``delivery:`` 标记（本案卷背后没有失败的一跳），因此它进不了例外
    通道——纠正提案同样要过判决包，这正是「无平局」要求的下一环。
    """
    from .dispatch import (  # 延迟导入：dispatch 顶层不依赖本模块，但避免成环
        _persist_suggestion,
    )

    member = str(entry.get("member") or "leader")
    criteria = str(entry.get("success_criteria") or "（原条目未记成功判据）")
    compared = ab_doc.get("compared") or ""
    if isinstance(compared, (list, tuple)):
        compared = ", ".join(str(case) for case in compared)
    body = "\n".join(
        (
            f"交付后核验未通过：条目 {bid} 实施后 A/B 实测判为「未改善」。",
            "",
            f"- 原摘要：{entry.get('summary', '')}",
            f"- 核验标签：{ab_doc.get('label', '')}"
            f"｜a={ab_doc.get('commit_a') or ab_doc.get('ref_a', '')}"
            f" → b={ab_doc.get('commit_b') or ab_doc.get('ref_b', '')}",
            "",
            "source: post-verify",
            "success_criteria",
            f"- 未改善案卷须复测出改善：{compared or '（A/B 未记案卷）'}",
            "原判据（审批时定死，不得改口）：",
            criteria,
            f"- 复跑本核验：orch bench-add --kind ab --ref-a {ab_doc.get('ref_a', '')} "
            f"--ref-b HEAD --label {ab_doc.get('label', '')}-rerun",
        )
    )
    return _persist_suggestion(member, body, "defect")


# ---------------------------------------------------------------------------
# 读数面：判决包健康（orch status 段）
# ---------------------------------------------------------------------------
def row_channel(record: dict[str, Any]) -> str:
    """决策日志行 → 判决通道（兼容 E5b 之前只有 ``source=override`` 的行）。"""
    ch = str(record.get("channel") or "")
    if ch:
        return ch
    return "override" if record.get("source") == "override" else ""


def health() -> dict[str, Any]:
    """各判决通道用量、override 率、escalate 积压（§5.1/§5.6 的账要能一眼看见）。"""
    channels = {c: 0 for c in (*CHANNELS, CHANNEL_EXCEPTION)}
    for r in journal.read_records():
        ch = row_channel(r)
        if ch in channels:
            channels[ch] += 1
    decided = sum(channels.values())
    escalate: list[str] = []
    pending: list[str] = []
    for it in _backlog_items():
        v = it.get("verify") or {}
        stage = str(v.get("stage") or "")
        if stage == STAGE_ESCALATED:
            escalate.append(str(it.get("id", "")))
        elif stage in (STAGE_APPROVED, STAGE_VERIFYING):
            pending.append(str(it.get("id", "")))
    return {
        "channels": channels,
        "decided": decided,
        "override_rate": (channels["override"] / decided) if decided else 0.0,
        "escalate": escalate,
        "pending": pending,
    }


def _backlog_items() -> list[dict[str, Any]]:
    """backlog 全量条目（读不到文件返回空表，不阻塞 status）。"""
    from .workflow import (  # 延迟导入：workflow 顶层依赖本模块
        BACKLOG_PATH,
    )

    if not BACKLOG_PATH.exists():
        return []
    try:
        data = json.loads(BACKLOG_PATH.read_text(encoding="utf-8"))
    except ValueError:
        return []
    return list(data.get("items", []))


def seat_alert_lines() -> list[str]:
    """仲裁席位告警行（§4「触发回退必须显式告警」在 status 面的落点）。

    只读配置与 Key 就位性，不开庭、不发请求；主席有缺位就点名（无论回退席是否上场）——
    「J 判决还能跑」不等于「三席直调还健康」。registry 读不通也只报读数失败，不拦 status。
    """
    try:
        plan = jury.seat_readiness()
    except (OSError, ValueError) as exc:
        return [f"    仲裁席位读数失败：{type(exc).__name__}: {exc}"]
    if not plan["missing_primary"]:
        return []
    return [f"    仲裁 {jury.readiness_summary(plan)}"]


def channel_fallback_lines() -> list[str]:
    """渠道回退事件行（用户裁决 2026-10-12 21:1x 规格 4b 的组长侧落点）。

    与 mirror 概览项共读 :func:`channels.summary_line`——同一句话两处呈现，不各写各的。
    观察面失败只报读数失败，绝不拦 ``orch status``。
    """
    try:
        body = channels.summary_line()
    except (OSError, ValueError) as exc:
        return [f"    渠道回退读数失败：{type(exc).__name__}: {exc}"]
    return [f"    {line}" for line in body.splitlines()]


def health_lines() -> list[str]:
    """``orch status`` 的「判决包健康」段（无判决记录时给一行引导，不静默缺席）。"""
    h = health()
    if not h["decided"] and not h["pending"] and not h["escalate"]:
        return [
            "    （尚无判决包记录：approve 须带 --jury / --bench / --override，"
            "§5.1 四态之一）",
            *seat_alert_lines(),
            *channel_fallback_lines(),
        ]
    c = h["channels"]
    line = (
        f"    通道用量：J={c['jury']} M={c['bench']} J+M={c['jury+bench']} "
        f"override={c['override']} 缺陷例外={c[CHANNEL_EXCEPTION]}"
        f"｜override 率={h['override_rate'] * 100:.0f}%"
    )
    lines = [line]
    if h["pending"]:
        lines.append(
            f"    核验在途={len(h['pending'])}（待交付后排 A/B）："
            f"{', '.join(h['pending'][:5])}"
        )
    if h["escalate"]:
        lines.append(
            f"    [!] 核验挂起（escalate=twin）={len(h['escalate'])}："
            f"{', '.join(h['escalate'][:5])} → 需组长人工终裁"
        )
    lines.extend(seat_alert_lines())
    lines.extend(channel_fallback_lines())
    return lines


# ---------------------------------------------------------------------------
# 咨询环节（§5.1 可选咨询；twin 面等 E6 上线，接口先就位）
# ---------------------------------------------------------------------------
def taste_enabled() -> bool:
    """twin 偏好面是否已由用户开启（registry 顶层开关，组长无权代按）。"""
    try:
        return bool(Registry.load().taste_enabled)
    except (OSError, ValueError):
        return False


def cmd_consult_twin(args) -> int:
    """``orch consult-twin <案卷>``——咨询 twin 组员（高风险裁决的第二意见）。

    ``taste_enabled=false`` 时明确拒绝并提示待上线，不静默降级成 deputy 咨询：把「问
    了个不存在的声音」当成答案是更坏的失败。
    """
    from .dispatch import do_dispatch  # 延迟导入：与 consult 同面

    if not taste_enabled():
        print(
            "[!] twin 偏好面未启用（registry.json 顶层 taste_enabled=false 或缺席），"
            "拒绝咨询。\n"
            "    该通道待 E6 上线：用户确认 PREFERENCE.md 草案后由用户本人置 true，"
            "组长无权代按（§5.5/§10）。\n"
            '    眼下要第二意见：orch consult "<问题>"（副组长，平级协作条款）。'
        )
        return 2
    case = Path(str(args.case_file))
    if not case.is_absolute():
        case = PROJECT_ROOT / case
    if not case.exists():
        print(f"[!] 案卷不存在：{orchestration_rel(case)}")
        return 2
    text = (
        "组长咨询（twin 偏好面，顾问模式）：请就该案卷给出与组长判断独立的意见，"
        "并指出你认为组长前提有误之处。\n\n"
        f"案卷 {case.name}：\n\n{case.read_text(encoding='utf-8')}\n"
    )
    outcome = do_dispatch(
        "twin",
        text=text,
        slug=f"consult-twin-{datetime.now().strftime('%H%M%S')}",
        session=args.session or "latest",
    )
    if outcome.error == "pod 不存在":
        print("    （twin pod 未就位：E6 上线前该通道不可用。）")
    return outcome.code
