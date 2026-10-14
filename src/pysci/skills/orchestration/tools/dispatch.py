"""派发核心：任务书落 inbox → headless 执行 → 交付解析 → 机械验收 → 台账/建议持久化。

从门面（orch.py）抽出，供 ``dispatch`` 命令与 plan-runner（plans.py）复用。
输出协议：``quiet=False`` 时打印完整组员报告与 [NEXT]；plan-runner 以 quiet=True
调用后自行打印交接报告。
"""

from __future__ import annotations

import json
import re
import shutil
import sys
import time
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from pysci.paths import ORCH_STATE_ROOT, PROJECT_ROOT

from . import channels as _channels
from . import failure_classes, presets, procs, routing
from . import sessions as _sessions
from . import sync as _sync
from .checks import run_checks
from .delivery import parse_delivery
from .failure_tags import load_infra_kinds
from .ledger import LedgerEntry, append
from .registry import Member, Registry, note_hop_pid, orchestration_rel
from .runner import (
    PROTOCOL_REMINDER,
    Envelope,
    build_command,
    build_env,
    new_session_id,
    normalize_effort,
    run_headless,
)

SUGGESTIONS_DIR = ORCH_STATE_ROOT / "suggestions"
REPLIES_DIR = ORCH_STATE_ROOT / "replies"

#: 任务书轻量 lint 规则表（JSON: {rules: [{id?, re, hint}]}）；devops 可增补。
#: 组长手写任务书易把错误 CLI 范式传染给照抄的组员，落盘前扫一遍作机械护栏。
LINT_RULES_PATH: Path = ORCH_STATE_ROOT / "taskbook-lint.json"

#: **失败词表不是本模块的快照**（backlog 20261013-183521-devops）：quota / channel 的判据
#: 一律**按调用时**读 ``state/failure-classes.json``。原先这里是两个启动时编译的模块级常量
#: （``QUOTA_ERROR_RE`` / ``CHANNEL_ERROR_RE``），于是规格承诺的「devops 可扩、改后全链生效」
#: 只对**新起进程**成立——正在跑的 drain worker 与已 import 的 orch 进程仍用旧词表，改一条
#: pattern 却像「没生效」（与 §3.4「解析器早于协议」同族的启动快照陷阱）。
#: 现在两条判据路径（:func:`_detect_quota_exhaustion` 与 :func:`_detect_channel_exhaustion`
#: 的文本支/形状支）都经 :mod:`failure_classes` 的函数按调用取表，**改词表下一跳即生效**。

#: 系统性故障 kind 族——「不是任务书坏了，是渠道坏了」。三处消费面（drain / benchd /
#: inspectd 的全局停止、watch 的事件、orch status 的提醒行）按这一份清单判，别各写各的。
SYSTEMIC_KINDS: tuple[str, ...] = (
    "quota_exhausted",
    "channel_error",
    # 带宽兜底（用户裁决 2026-10-13，backlog 20261013-193500-leader）：带宽类延期
    # **触顶预算仍不愈** → 回升为全局停止。它防的是「分类器误判/文案漂移导致逐项
    # 烧跳」（2026-10-12 连烧四单形态）：延期是让步，不是放弃熔断。
    "channel_saturated",
)

#: kind → （报告面短标题, 长说明）。文案单一源：drain/bench/inspect 的 ``.done`` error、
#: orch status、dispatch 报告面共读，避免「同一故障三副面孔」。
SYSTEMIC_TEXT: dict[str, tuple[str, str]] = {
    "quota_exhausted": ("额度类失败", "模型渠道额度耗尽（quota_exhausted）"),
    "channel_error": ("渠道级失败", "模型渠道级故障（channel_error）"),
    "channel_saturated": (
        "带宽饱和",
        "带宽类延期已触顶预算仍未愈（channel_saturated）——回升为全局停止",
    ),
}


def _sid_inflight(m: Member, sid: str) -> dict[str, Any] | None:
    """目标 sid 的**上一跳进程仍活着**的证据（在途闸判据，184000 修复面④）。

    运行态会话池的 ``pid``/``pid_task`` 由 :func:`registry.note_hop_pid` 在派生瞬间写下、
    正常收尾时被常规写覆盖——**看得到非零 pid 就等于那跳没走完**。但 pid 会被系统复用，
    故与第四面回收同口径过两道闸：``pid_alive`` 判活 + ``pid_is_session`` 核对 argv 里的
    sid，两闸都过才算在途。任一不过 → None（fail-open 放行，不冤枉正常续跑）。

    Returns:
        ``{"pid": int, "task": str}``（占用证据，供拒绝信息点名）；不在途 → None。
    """
    entry = m.find_session(sid)
    if not entry:
        return None
    try:
        pid = int(entry.get("pid") or 0)
    except (TypeError, ValueError):
        return None
    task = str(entry.get("pid_task") or "")
    if pid <= 0 or not task:
        return None
    if not procs.pid_alive(pid) or not procs.pid_is_session(pid, sid):
        return None
    return {"pid": pid, "task": task}


def _audit_blocked_delivery(pod: Path, raw: str) -> None:
    """blocked/失败交付的**全文**审计副本（184000 修复面⑤）。

    ``needs_leader`` note 曾截 200 字——两案（183545/192119）的关键证据都变残片，违背
    §0.4 留痕原则。落 ``pods/<m>/outbox/delivery-<ts>.xml``（gitignore 面）供组长/审计
    复原。fail-open：审计副本写失败绝不能反噬交付本身。
    """
    try:
        outbox = pod / "outbox"
        outbox.mkdir(parents=True, exist_ok=True)
        (outbox / f"delivery-{_ts()}.xml").write_text(raw, encoding="utf-8")
    except OSError as exc:  # pragma: no cover - 只防反噬，不值得为它造故障面
        print(f"[!] 审计副本落盘失败（不影响交付）：{exc}", file=sys.stderr)


def systemic_error_text(kind: str) -> str:
    """全局停止的说明句（``.done`` 的 ``error`` 字段，三个 worker 面同文）。"""
    label = SYSTEMIC_TEXT.get(kind, (kind, kind))[1]
    return f"系统性故障：{label}——全局停止，全部条目保持 pending 待渠道恢复"


def _sleep(seconds: float) -> None:
    """真实等待（退避 / 边界避让共用一个落点，测试可整只替换）。

    单独抽成函数不是洁癖：退避 45s、避让最多 300s 的等待**不能**在测试里真发生，
    而两处等待的语义又都要被断言（等了多久、等了几次）。替换本函数即同时拿到两者。
    """
    time.sleep(seconds)


def _clock() -> float:
    """单调时钟读数（重试预算闸与避让判据的计时源，测试可替换）。"""
    return time.monotonic()


#: 入台账的 provider 报错原文上限（字符）。为什么截：原文可能夹整段会话上下文，
#: 而分类器只需签名——2000 字足够覆盖任何单条 provider 报错，且台账行不会爆。
ERROR_TEXT_MAX: int = 2000


def _provider_error_text(env: Envelope, out: str, err: str) -> str:
    """失败跳的 **provider 报错原文**（入台账 ``error_text``，backlog 20261013-180000-leader）。

    取值优先级：``envelope.result``（渠道故障时 CLI 合成的那条回合正文就在这）→ stderr
    → 原始 stdout。全空则退回 ``stop_reason``——短码不是原文，但总比空行强（读者至少
    知道这一跳是怎么收的尾）。
    """
    for chunk in (env.result, err, out):
        if chunk and chunk.strip():
            return chunk.strip()[:ERROR_TEXT_MAX]
    return (env.stop_reason or "").strip()


def _detect_quota_exhaustion(env: Envelope, out: str, err: str) -> bool:
    """识别额度类失败：文案可能出现在 envelope.result / stderr / stdout 全文。

    **仅在运行已失败时调用**——成功交付的正文若只是提及该文案（如复盘任务）不应
    重分类。

    走共享分类器（backlog 20261013-180000-leader）：判据即「分类器判成 quota」，
    与词表同源（同一份 config），不再各持一份关键词表；**按调用取表**，改词表即生效。
    """
    return any(
        failure_classes.classify_failure(s, is_error=True) == "quota"
        for s in (env.result, err, out)
        if s
    )


def _detect_channel_exhaustion(
    env: Envelope, out: str, err: str, *, member: Member, sid: str
) -> bool:
    """识别渠道级系统性故障：固定文案 ∧ 会话档末轮合成故障形状，**两支合取**。

    为什么不能只看文案（2026-10-10 宽匹配教训的延续）：那句话也会出现在组员正文里
    （复盘本故障的任务书就是活样本），而把它误读成系统性故障的代价是**整条队列停下**。
    形状支读 :func:`sessions.channel_failure_turn`——真实模型回合的 ``model`` 是渠道名，
    故障回合才是 ``<synthetic>``＋自称停在 stop_sequence 却给出空序列。

    会话档读不出（缺失/半写/窗口无 assistant 记录）时采信文案单特征：漏判的代价是队列
    再逐单空烧一轮，正是本单要修的缺陷（backlog 20261012-194500-leader 证据段）。

    文本支走共享分类器（backlog 20261013-180000-leader）：判成 ``bandwidth``/``outage``
    即算 provider 报错签名（形状支的 regex 同样按调用取）。分类器 precedence 让
    ``quota`` 优先——与 :func:`_detect_systemic`「额度面先判」一致，两处不会对同一跳
    给出不同归类。
    """
    text_hit = any(
        failure_classes.classify_failure(s, is_error=True) in ("bandwidth", "outage")
        for s in (env.result, err, out)
        if s
    )
    shape = _sessions.channel_failure_turn(
        member, sid, text_re=failure_classes.provider_error_re()
    )
    return bool(text_hit) if shape is None else shape


def _detect_systemic(
    env: Envelope, out: str, err: str, *, member: Member, sid: str
) -> str:
    """失败跳的系统性故障归类，非系统性故障返回空串。

    额度面先判：credit 文案更特异且是「必须充值」的硬事实；渠道面需要读会话档，留给
    额度未命中的场合（同一跳两者同现时，取额度=更保守的停止理由，处置本就相同）。
    """
    if _detect_quota_exhaustion(env, out, err):
        return "quota_exhausted"
    if _detect_channel_exhaustion(env, out, err, member=member, sid=sid):
        return "channel_error"
    return ""


def _seat_startup_failure(env: Envelope, rc: int) -> bool:
    """席位本身**没起跑**（型号 UUID 失效、exe 直接拒绝该 ``-m``）的判据。

    为什么要把这一类单独判出来（backlog 20261012-211500-leader 规格 3）：跑了一半才断
    （瞬时中断、工具报错）同渠道重试有意义；而**席位本身不存在**时同渠道重试就是白烧
    一跳——链里的下一席才是解。判据取「零回合 + envelope 不可解析」而非文案，因为这条
    路径的 stdout 常常根本不是 envelope（exe 在参数校验阶段就退出）。
    """
    return rc != 0 and (
        env.stop_reason == "envelope_parse_failed" or env.num_turns == 0
    )


def _seat_broken(env: Envelope, rc: int, systemic: str) -> str:
    """本跳是否该换席：返回故障成因串（``systemic`` 的 kind 或 ``startup_failure``），不该换则空。"""
    return systemic or ("startup_failure" if _seat_startup_failure(env, rc) else "")


def _seat_reason_text(reason: str) -> str:
    """换席原因的文案（两类系统性故障共读 :data:`SYSTEMIC_TEXT`，启动即失败自有一句）。"""
    if reason == "startup_failure":
        return "席位启动即失败（型号不可用）"
    return SYSTEMIC_TEXT.get(reason, (reason, reason))[1]


def _boundary_hold(
    reg: Registry, *, no_hold: bool, quiet: bool = False
) -> tuple[str, int]:
    """距峰谷切换点不足阈值时睡到过点再派（边界避让，用户裁决 2026-10-13）。

    为什么值得等（backlog 20261013-144500-leader）：席链随时间走（路由 v3 谷时隐性
    DeepSeek），卡在切换点前几秒出发=整跳落在**将要过期**的窗口里，成本错配；而等待期
    零 LLM 成本。必须在解析角色/席链**之前**调用——等过界后再读，读到的才是新窗口的链。

    Args:
        reg: 注册表（``routing.boundary_hold_s`` 旧位，仅作**兼容回退**）。
        no_hold: CLI ``--no-hold``（草稿/调试场景跳过等待）。
        quiet: 静默派发（drain/批次）——迁移提示是给人看的信息面，**不扰**静默通路。

    Returns:
        ``(一行读数, 实际等待秒)``；不需避让/判不出 → ``("", 0)``。

    单一真源：生效值取 ``data/pysci_settings.toml`` ``[model] boundary_hold_s``
    （2026-10-14 用户实测反馈迁 toml）；仅当 toml 未配置时回退 registry 旧位**并出声**提示迁移。
    """
    if no_hold:
        return "", 0
    fv = routing.face()
    hold_s = fv.boundary_hold_s
    if hold_s is None:
        # 旧位兼容读一次 + 出声（沿 [model]/[routing] leadtime 同款语义）
        hold_s = int(reg.routing_hold_s)
        if hold_s > 0 and not quiet:
            print(
                "[!] boundary_hold_s 仍在 registry 旧位（routing.boundary_hold_s）——"
                "本次已兼容读取；请迁进 data/pysci_settings.toml 的 [model] 节生效键"
            )
    if hold_s <= 0:
        return "", 0
    if not fv.enabled:
        return "", 0
    now = routing.now_cn()
    when, mode, _warn = routing.next_switch(now, fv)
    if when is None:
        return "", 0
    wait = (when - now).total_seconds()
    if not 0 < wait < hold_s:
        return "", 0
    _sleep(wait)
    return f"[i] 边界避让：等待 {wait:.0f}s 对齐 {mode}（{when:%H:%M}）", int(wait)


def _ts() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


@dataclass
class DispatchOutcome:
    """一次派发（一跳）的结构化结果。"""

    code: int  # 0=成功交付；2=blocked/parse_error/run_failed/系统性故障/验收FAIL
    #: result | blocked | parse_error | run_failed | SYSTEMIC_KINDS 之一
    #: （quota_exhausted / channel_error——渠道级系统性故障，drain 据此全局停止）
    kind: str
    member: str = ""
    sid: str = ""
    session_name: str = ""
    task_file: str = ""
    effort: str = ""  # 本跳实际所用的推理强度档位（""=跟随用户级默认）
    delta_chars: int = 0  # 本跳会话 jsonl 字符增量（相对用量尺子，见 LedgerEntry）
    est_tokens: int = 0  # = delta_chars × est_tokens_per_char（启发式估算）
    body: str = ""  # 交付正文（result/blocked 块内容；长交付可能带截断标记）
    #: 交付正文**全文**留档路径（``state/queues-runs``→``state/deliveries/<job_id>.md``）；
    #: 空串=无留档（短交付亦落盘；写失败时为空）。
    body_file: str = ""
    checks: list[dict] = field(default_factory=list)
    infra_suggestion: str = ""
    infra_kind: str = ""  # E0-B：交付 XML 上的 kind 属性（defect/usability/""）
    tags: list[str] = field(default_factory=list)  # E1：<blocked tags="a,b"> 失败标签
    #: <blocked reason="…"> 的成因档位（backlog 20261013-191213-devops，单值受控词表）。
    #: 空串=未标注。drain 据它把 blocked 分列进 .done，供 orch status 分列计数。
    reason: str = ""
    #: 解析器版本错位告警（delivery.parse：严格面全缺、宽松面救回时非空）。判读照常，
    #: 但组长/drain 必须看见这一行——交付是在「旧解析器读新协议」的窗口里被兜底救下的
    #: （backlog 20261012-002857-devops）。空串=解析器与协议同版。
    warning: str = ""
    #: 本跳实际发生的**降级回退事件**（首席→…→最终席，含 ``to=""`` 的耗尽事件）。
    #: drain 据此把告警行写进运行日志；空表=没换席。全量账在 ``channels`` 事件流里。
    fallbacks: list[dict[str, Any]] = field(default_factory=list)
    suggestion_id: str = ""
    envelope: Envelope | None = None
    error: str = ""
    #: provider 报错原文（与台账同源；drain 据此判失败类别——带宽类走延期而非停链）。
    error_text: str = ""
    #: sid 在途闸换新会话的原因（``prev_alive``=上一跳进程仍活；""=没换）。台账同名字段
    #: 留痕，drain 据此写 run 日志——人工拒绝与批次换新都必须可见（184000 修复面④）。
    sid_new_reason: str = ""
    #: 本跳的**路由切换**读数（路由 v3 规格 2）：跨过一次峰谷界时是一句人读文案，平常空串。
    #: drain 把它抄进 run 日志——「这一跳烧的是隐性 DeepSeek 席」必须留在批次账上，
    #: 而不能只活在 toml 与 routing-last.json 里（组长读的是 run 日志）。
    route: str = ""
    #: 本跳出发前的**边界避让**读数（用户裁决 2026-10-13，backlog 20261013-144500-leader）：
    #: 等了多少秒、对齐的是哪个窗口。没等时空串——drain 据此抄 run 日志。
    hold: str = ""
    hold_s: int = 0
    #: **同席重试**读数（用户裁决 2026-10-13，backlog 20261013-143000-leader）：channel_error
    #: 是瞬态的，先在原席上 resume 续跑自救，付费席降为后手。0/0 = 没重试。不进
    #: ``fallbacks``（重试不是降级，mirror「渠道回退」只记真实换席），但次数与耗时
    #: 必须留在台账与 run 日志上，否则「渠道在抖 vs 渠道已死」无从分辨。
    same_seat_retries: int = 0
    same_seat_retry_ms: int = 0


def collect_replies(member_id: str) -> tuple[str, list[Path]]:
    """收集待附送的审批回复（approve/reject 写入；派发成功后移入 sent/）。"""
    d = REPLIES_DIR / member_id
    if not d.exists():
        return "", []
    files = sorted(p for p in d.glob("*.md"))
    if not files:
        return "", []
    body = "\n\n".join(
        f"### 组长对你改进建议的回复（{p.stem}）\n{p.read_text(encoding='utf-8')}"
        for p in files
    )
    return body, files


def deployed_skills_for(pod: Path) -> list[str]:
    """从 manifest 解析部署到指定 pod 的技能名（pod-guard 保护部署副本）。"""
    names: list[str] = []
    skills_dir = (pod / ".qoder" / "skills").resolve()
    try:
        for spec in _sync.load_manifest():
            if any(Path(t).resolve() == skills_dir for t in spec.targets):
                names.append(spec.name)
    except OSError:
        pass
    return names


def load_lint_rules() -> list[dict]:
    """加载任务书 lint 规则表；缺失/损坏 → 空列表（fail-open，绝不阻塞派发）。"""
    if not LINT_RULES_PATH.exists():
        return []
    try:
        data = json.loads(LINT_RULES_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    rules = data.get("rules", []) if isinstance(data, dict) else []
    return [r for r in rules if isinstance(r, dict) and r.get("re")]


#: 引述段（「」『』“” 与成对 ASCII 引号）：其中的范式是**被引述**——缺陷样例、grep 判据的
#: 禁词字面量、shell 单引号串——不是正文承诺本身。判据自己的禁词字面量不该被自己判红
#: （20261014-075230-devops 约定「禁词运行期拼接」，本规则是该约定的第一次实战）。
_QUOTE_SPAN_RE = re.compile(
    r"「[^」\n]*」|『[^』\n]*』|“[^”\n]*”|\"[^\"\n]*\"|'[^'\n]*'"
)


def _strip_quoted(body: str) -> str:
    """去掉引述段，只留正文陈述（供环节 id 承诺判据扫描）。"""
    return _QUOTE_SPAN_RE.sub(" ", body)


def _known_plan_ids() -> list[str]:
    """已注册计划 id（``state/plans/<id>.plan.yaml``）；取不到 → 空表（fail-open）。"""
    try:
        from . import plans as _plans  # 局部导入：避开 plans 侧对 dispatch 的再入

        d = _plans.PLANS_DIR
    except Exception:  # noqa: BLE001 - 定位计划失败不阻塞派发
        return []
    if not d.exists():
        return []
    return [p.name[: -len(".plan.yaml")] for p in d.glob("*.plan.yaml")]


def _plan_steps(plan_id: str) -> list[str]:
    """计划 YAML 的**叶子**环节点路径清单（superstep 递归展开；读不到 → 空表）。"""
    try:
        from .planflow import PlanTree
        from .plans import _load_yaml

        plan = _load_yaml(plan_id)
    except Exception:  # noqa: BLE001
        return []
    return PlanTree(plan.get("steps") or []).leaf_paths()


def _plan_from_slug(slug: str) -> str | None:
    """由任务书 slug 剥尾段定位注册计划 id（``tui-fixup-20261014-f3`` → ``tui-fixup-20261014``）。"""
    ids = set(_known_plan_ids())
    parts = slug.split("-")
    for cut in range(len(parts), 0, -1):
        cand = "-".join(parts[:cut])
        if cand in ids:
            return cand
    return None


def _resolve_plan_id(body: str, hint: str | None) -> str | None:
    """「对应计划」：显式 hint 优先；否则正文中出现且真实存在的计划 id（多命中取最长/最具体）。"""
    ids = _known_plan_ids()
    if hint and hint in ids:
        return hint
    hits = [pid for pid in ids if pid in body]
    return max(hits, key=len) if hits else None


#: 叙述性（过去/完成态）痕迹：动词后面紧跟**完成**、或 id 前面紧跟「自/已/曾」——
#: 「自 m4 **建成**后」「m4 已建」是**讲史**，不是「mN 补/充实/建 + 宾语」的未来承诺。
#: （2026-10-14 实测误报：dispatch-unify 任务书正文「`state/queues/` 自 m4 建成后长期空转」
#: 被判成环节承诺——m4 根本不在该 adhoc 计划 steps 里。）
_NARRATIVE_TAIL: tuple[str, ...] = ("成", "了", "完")
_NARRATIVE_HEAD: tuple[str, ...] = ("自", "已", "曾")


def _is_narrative(text: str, m: re.Match[str]) -> bool:
    """该命中是否**叙述性用法**（不是环节承诺）？``text`` 须与构造 ``m`` 的同一份正文。"""
    tail = text[m.end() : m.end() + 4].lstrip("*_ ")
    if tail[:1] in _NARRATIVE_TAIL:
        return True
    prefix = re.sub(r"[*_（(【\[\s]", "", text[max(0, m.start() - 8) : m.start()])
    return bool(prefix) and prefix[-1] in _NARRATIVE_HEAD


def _step_promise_warning(
    rule: dict, pat: re.Pattern[str], body: str, plan_id: str | None
) -> str | None:
    """环节 id 承诺判据：正文（去引述后）命中的 id 必须能在对应计划 steps 里找到。

    定位不到对应计划 → 返回 ``None``（fail-open：无法核对就不误报）。
    """
    shown = _strip_quoted(body)
    found: set[str] = set()
    for m in pat.finditer(shown):
        if not m.groups() or _is_narrative(shown, m):
            continue
        found.add(str(m.group(1)))
    if not found:
        return None
    pid = _resolve_plan_id(body, plan_id)
    if pid is None:
        return None
    steps = _plan_steps(pid)
    missing = sorted(found - set(steps))
    if not missing:
        return None
    tag = f"[{rule['id']}] " if rule.get("id") else ""
    hint = rule.get("hint", "环节 id 承诺不可溯")
    return (
        f"{tag}{hint}（对应计划 {pid} 的 steps=[{', '.join(steps) or '空'}]，"
        f"承诺了不存在的：{'、'.join(missing)}）"
    )


#: 「白名单」段标题与任意标题（用于切出白名单段的边界）。
_WHITELIST_HEAD_RE = re.compile(r"^#{1,6}\s*白名单")
_ANY_HEAD_RE = re.compile(r"^#{1,6}\s")

#: 正文点名的**仓库路径**（顶层目录起；前后不许接路径字符，避免咬进长串）。
_REPO_PATH_RE = re.compile(
    r"(?<![\w/.\-])((?:src|tests|orchestration|data|scripts|\.qoder)/[A-Za-z0-9_./\-*]+)"
)

#: 「改动目标」的动作词：路径须与它在同一行才算「要改的」（排除纯证据/背景引用）。
_EDIT_VERBS: tuple[str, ...] = (
    "改",
    "修改",
    "新增",
    "新建",
    "创建",
    "删除",
    "移除",
    "写入",
    "落地",
    "重写",
    "补",
)

#: 否定前缀字（「**不**手改」「**勿**动」「**禁**改」）——否定语境里的路径是「别改」，
#: 正好相反（m7 任务书：`.qoder/skills/orchestration/` 是部署副本……**不手改**）。
_NEG_WORDS = "不勿禁别毋莫休"


def _is_edit_line(ln: str) -> bool:
    """该行是否在讲「要改」：命中改动动词，且该动词前 3 字内无否定词。"""
    for v in _EDIT_VERBS:
        i = ln.find(v)
        while i != -1:
            if not any(c in _NEG_WORDS for c in ln[max(0, i - 3) : i]):
                return True
            i = ln.find(v, i + 1)
    return False


def _whitelist_entries(lines: list[str]) -> tuple[list[str], set[int]]:
    """解析「白名单」段 → (条目前缀, 段内行号集)。无段 → ``([], set())``。"""
    entries: list[str] = []
    used: set[int] = set()
    on = False
    for i, ln in enumerate(lines):
        if _WHITELIST_HEAD_RE.match(ln):
            on = True
            used.add(i)
            continue
        if on and _ANY_HEAD_RE.match(ln):
            on = False
        if on:
            used.add(i)
            for seg in re.split(r"[、,，;；]", ln):
                raw = re.sub(r"[`（）()【】\[\]]", "", seg).strip().replace("\\", "/")
                # 只取行首的路径样片段：条目常带中文注记（`data/taskbooks/**`（新建））
                m = re.match(r"[A-Za-z0-9_./\-*]+", raw)
                if not m:
                    continue
                tok = m.group(0)
                # 路径样才收（在规整**之前**判：`src/**` 规整成 `src` 后已不含 `/`）
                if "/" not in tok and not tok.endswith(
                    (".md", ".py", ".toml", ".json")
                ):
                    continue
                norm = tok.rstrip("*").rstrip("/")
                if norm:
                    entries.append(norm)
    return entries, used


def _covered_by(path: str, entries: list[str]) -> bool:
    """路径是否落在白名单某条目之下（前缀语义，宽松取静：宁可少报不误报）。"""
    return any(
        path == e or path.startswith(e.rstrip("/") + "/") or path.startswith(e)
        for e in entries
    )


def _whitelist_warning(rule: dict, body: str) -> str | None:
    """白名单覆盖判据：正文点名的**改动目标**路径必须落在其声明的白名单内。

    未声明白名单 → 返回 ``None``（无从判断，fail-open）。命中说明该路径会在派发时被
    pod-guard 拦成 ``whitelist_conflict``（backlog 20261014-113347-tier-a 的机械预防面）。
    """
    lines = body.splitlines()
    entries, used = _whitelist_entries(lines)
    if not entries:
        return None
    missing: list[str] = []
    for i, ln in enumerate(lines):
        if i in used or not _is_edit_line(ln):
            continue
        for m in _REPO_PATH_RE.finditer(ln):
            if not _covered_by(m.group(1), entries):
                missing.append(m.group(1))
    if not missing:
        return None
    tag = f"[{rule['id']}] " if rule.get("id") else ""
    hint = rule.get("hint", "白名单未覆盖正文点名的改动目标")
    return (
        f"{tag}{hint}（白名单：{'、'.join(entries[:8])}；"
        f"未覆盖：{'、'.join(sorted(set(missing))[:8])}）"
    )


def lint_taskbook(body: str, *, plan_id: str | None = None) -> list[str]:
    """扫任务书正文，返回命中的告警（空=干净）。**告警不阻断**。

    单条规则 regex 非法只跳过该条，不影响其余规则与派发。规则可选：

    - ``absent``（第二正则）：只有 ``re`` 命中**且** ``absent`` **不**命中时才告警——用于
      「正文提到 A 却没提 B」的条件判据（如白名单列了运行态目录却漏了 `.gitignore`）；
    - ``steps: true``（环节 id 承诺，backlog 20261014-075230-devops 的落地）：``re`` 第 1
      捕获组是**环节 id**（如 m3/f4）；该 id 必须存在于**对应计划** YAML 的 steps，否则
      告警（附计划 id 与现有 steps 供核对）。对应计划由 ``plan_id``（任务书 slug 派生）或
      正文中最具体的真实计划 id 定位；定位不到则 fail-open 不告警。**引号内的引述**（缺陷
      样例 / grep 判据的禁词字面量）不算承诺；
    - ``whitelist: true``（白名单覆盖，backlog 20261014-113347-tier-a）：正文声明了
      「白名单」段时，正文点名的**改动目标**路径必须落在其内（同行动作词判「改动目标」），
      否则告警——这类路径会在派发时被 pod-guard 拦成 ``whitelist_conflict``。未声明
      白名单 → fail-open 不告警。
    """
    warnings: list[str] = []
    for rule in load_lint_rules():
        try:
            pat = re.compile(rule["re"])
        except re.error:
            continue
        if rule.get("steps"):
            warn = _step_promise_warning(rule, pat, body, plan_id)
            if warn:
                warnings.append(warn)
            continue
        if rule.get("whitelist"):
            warn = _whitelist_warning(rule, body)
            if warn:
                warnings.append(warn)
            continue
        if not pat.search(body):
            continue
        absent = rule.get("absent")
        if absent:
            try:
                if re.search(absent, body):
                    continue
            except re.error:
                continue  # 非法 absent 只跳过本规则
        tag = f"[{rule['id']}] " if rule.get("id") else ""
        warnings.append(f"{tag}{rule.get('hint', '命中错误 CLI 范式 ' + rule['re'])}")
    return warnings


# ── 判据矛盾检测（backlog 20261013-071759-devops）────────────────────────────
#
# 批准单是追加式文本：转 approved 之后组长/用户常往同一份文件里再补一段
# ``**补充（用户指令 …）**``，而机械验收只读 ``success_criteria`` 标记行。两处就同一配置键
# 给出相反取值时（实锤反例 ``20261012-164601-leader``：``taste_enabled 仍 false`` vs
# ``将 registry.taste_enabled 置 true``），执行者只能自行猜「后到者优先」——猜错被打回，
# 猜对也留下一条与判据不符的交付读数。这里把矛盾本身和取舍规则都变成可见的机械事实。

#: 判据标记行（容错半/全角冒号；``success_criteria：`` 单独成行时判据在其后 bullet 段里）。
_CRITERIA_MARK_RE = re.compile(r"^\s*success_criteria\s*[:：]", re.IGNORECASE)

#: 追加段起始行（``**补充（用户指令 …）**：``，也认修订/更正/增补同族措辞）。
_AMEND_HEAD_RE = re.compile(r"^\s*\**\s*(补充|修订|更正|增补)")

#: 关键标识候选：点分配置名（``registry.taste_enabled``）。带下划线是「这是个配置键」
#: 的最强信号；纯单词不取——true/false/enabled 之类会让任何两段文本都被判成同类。
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_.]*[A-Za-z0-9_]", re.ASCII)

#: 子句边界：中英文逗号/顿号/分号/句号/冒号——布尔取值读数的最小语境。**换行不算边界**
#: （markdown 正文按 ~80 列折行，把折行当边界会让「置\ntrue」读不出值——反例正是这个形状）。
_CLAUSE_SPLIT_RE = re.compile(r"[，,。；;、：]")

#: bullet 起始（``- ``/``* ``/``1. ``）：折行规则之外的**真**边界，用它把列表项切开。
_BULLET_RE = re.compile(r"^(?:[-*·]\s+|\d+[.)]\s+)")


def _clauses(text: str) -> list[str]:
    """段落文本 → 子句清单：bullet 起始行是边界，行内换行只是排版折行（用空格接回）。"""
    buf: list[str] = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s:
            continue
        buf.append(("；" if buf else "") + s if _BULLET_RE.match(s) else s)
    parts = _CLAUSE_SPLIT_RE.split(re.sub(r"\*\*|`", "", " ".join(buf)))
    return [p.strip() for p in parts if p.strip()]


#: 布尔字面量（大小写不敏感；同一子句里同时出现 true 与 false 视为「在讲切换」，不判）。
#: 边界只按 ASCII 字母判，不用 ``\b``——中文动词可以直接贴着写（「置true」），而 CJK 在
#: Unicode 模式下算 word char，``\b`` 在那种写法上根本不成立。
_BOOL_RE = re.compile(r"(?<![A-Za-z])(true|false)(?![A-Za-z])", re.IGNORECASE)


#: 单一源登记：追加段里出现 ``superseded`` 且点名该键，即视为组长已裁决「后到者优先」，
#: 矛盾消解（本约定出自 backlog 20261013-071759-devops 的原文建议「或在补充段标注
#: superseded by」——标注了就要被机械认账，否则登记等于没登记）。
_SUPERSEDED_RE = re.compile(r"superseded", re.IGNORECASE)

#: 单条矛盾的取舍规则（同一句话既进组长 stdout 也进任务书正文，两面不得各说各话）。
CONFLICT_GUIDE = (
    "判据须单一源：改写 success_criteria，或在追加段标 superseded by 本段并点名该键"
    "（标了 lint 就认账）；未登记则执行者按**追加段（后到者优先）**实施，"
    "并在交付里点名该取舍"
)


def _clause_bools(clauses: list[str], key: str) -> set[bool]:
    """某键在各子句里被要求取的值（键与布尔字面量必须在**同一子句**，跨句不认）。"""
    vals: set[bool] = set()
    for c in clauses:
        if key not in c.lower():
            continue
        found = {v.lower() for v in _BOOL_RE.findall(c)}
        if found == {"true"}:
            vals.add(True)
        elif found == {"false"}:
            vals.add(False)
    return vals


def _conflict_clauses(clauses: list[str], key: str) -> str:
    """取该键第一条带布尔读数的子句（供告警原文点名，不编造）。"""
    for c in clauses:
        if key in c.lower() and _BOOL_RE.search(c):
            return c.strip()[:120]
    return ""


def criteria_conflicts(body: str) -> list[dict]:
    """批准单/建议正文里「判据面」与「追加段」对同一配置键给出相反布尔取值。

    只判布尔矛盾：这是实锤反例的形状，误报面窄。义务/禁止一类的语义反差不判——中文
    语气词无处不在，误报会把告警刷成噪声，噪声告警和没有告警一样没人读。

    Args:
        body: 批准单或建议全文。

    Returns:
        矛盾清单 ``{key, criteria, criteria_value, amend, amend_value}``；无矛盾 → ``[]``。
    """

    def _paragraphs(head_re: re.Pattern) -> list[str]:
        lines = body.splitlines()
        out: list[str] = []
        for i, ln in enumerate(lines):
            if not head_re.match(ln):
                continue
            para: list[str] = []
            for seg in lines[i:]:
                if not seg.strip():
                    break
                para.append(seg)
            out.append("\n".join(para))
        return out

    def _keys(text: str) -> set[str]:
        out: set[str] = set()
        for tok in _TOKEN_RE.findall(text):
            for seg in tok.split("."):
                if "_" in seg and len(seg) >= 4:
                    out.add(seg.lower())
        return out

    crit = "\n".join(_paragraphs(_CRITERIA_MARK_RE))
    amend = "\n".join(_paragraphs(_AMEND_HEAD_RE))
    if not crit or not amend:
        return []
    cc, ac = _clauses(crit), _clauses(amend)

    out: list[dict] = []
    for key in sorted(_keys(crit) & _keys(amend)):
        if any(key in c.lower() for c in ac if _SUPERSEDED_RE.search(c)):
            continue  # 追加段的 superseded 子句点名了该键 → 单一源已登记，矛盾消解
        cvals, avals = _clause_bools(cc, key), _clause_bools(ac, key)
        if len(cvals) == 1 and len(avals) == 1 and cvals != avals:
            out.append(
                {
                    "key": key,
                    "criteria": _conflict_clauses(cc, key),
                    "criteria_value": "true" if True in cvals else "false",
                    "amend": _conflict_clauses(ac, key),
                    "amend_value": "true" if True in avals else "false",
                }
            )
    return out


def conflict_notice(c: dict) -> str:
    """单条矛盾的告警文本（组长 stdout 用，与任务书注入面同源）。"""
    return (
        f"[判据矛盾] {c['key']}：success_criteria 段要求 {c['criteria_value']}"
        f"（「{c['criteria']}」），追加段要求 {c['amend_value']}"
        f"（「{c['amend']}」）——{CONFLICT_GUIDE}"
    )


def evidence_conflicts(body: str) -> list[dict]:
    """任务书正文 → 其「证据：…approved/<id>.md」指向的批准单 → 矛盾清单（fail-open）。

    证据行缺失、文件读不到、批准单没判据段 → 一律返回空列表：这是观测面，绝不挡派发。
    """
    m = re.search(r"^\s*证据[:：]\s*(\S+\.md)\s*$", body, re.M)
    if not m:
        return []
    path = PROJECT_ROOT / m.group(1)
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    return criteria_conflicts(text)


def taskbook_conflict_block(conflicts: list[dict]) -> str:
    """把矛盾与取舍规则写成任务书正文块——**组员侧可见**才是本单的病根。

    派发 stdout 只有组长读到，执行者仍然只能自己猜后到者优先；所以告警必须落进任务书
    本身，而不只是打在组长控制台。
    """
    lines = [
        "",
        "> ⚠ **判据矛盾（不阻断派发，但别再自行裁量）**——本单证据批准单里判据面与追加段"
        "对同一配置键给出相反取值：",
    ]
    for c in conflicts:
        lines.append(
            f"> - `{c['key']}`：success_criteria 段 `{c['criteria_value']}`（「{c['criteria']}」）"
            f"｜追加段 `{c['amend_value']}`（「{c['amend']}」）"
        )
    lines.append(f"> 处置：{CONFLICT_GUIDE}。")
    return "\n".join(lines) + "\n"


def write_taskbook(
    pod: Path, *, text: str | None, task_file: str | None, slug: str
) -> Path:
    """把任务内容写成 pod inbox 内的任务书，返回路径。"""
    inbox = pod / "inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    dest = inbox / f"task-{_ts()}-{slug}.md"
    header = (
        f"# 任务书 {slug}\n\n"
        f"（派发时间：{datetime.now().astimezone().isoformat(timespec='seconds')}）\n\n"
    )
    if task_file:
        src = Path(task_file)
        if not src.is_absolute():
            src = PROJECT_ROOT / src
        body = src.read_text(encoding="utf-8")
    else:
        body = text or ""
    # 落盘前轻量 lint：命中错误 CLI 范式仅告警（组长确认后可照常派发），不阻断
    for warn in lint_taskbook(body, plan_id=_plan_from_slug(slug)):
        print(f"[!] 任务书 lint：{warn}")
    conflicts = evidence_conflicts(body)
    for c in conflicts:
        print(f"[!] 任务书 lint：{conflict_notice(c)}")
    if conflicts:
        body = body.rstrip("\n") + "\n\n" + taskbook_conflict_block(conflicts)
    # 结尾恰好一个换行：裸文本会触发 pre-commit end-of-file-fixer 拦截任务书提交
    dest.write_text(header + body.rstrip("\n") + "\n", encoding="utf-8")
    return dest


def _persist_suggestion(
    member_id: str, body: str, kind: str = "", delivery: str = ""
) -> str:
    """把 infra_suggestion 持久化为待审批条目，返回建议 id（文件名 stem）；未入库返回空串。

    E0-B（判据 2）：把交付 XML 上解析出的 kind 写进文件头部标记行，使 approve 侧
    ``parse_suggestion_fields`` 无论建议正文是否自带 ``kind:`` 行都能读到权威分类
    （正文若自带则以头部这条为准——它来自被 gate 校验过的属性）。

    E5b（README §5.3 例外通道）：同处理由把**当条交付类型**也写成头部 ``delivery:``
    标记行。「缺陷类且交付含 blocked」这个判据必须在落盘时就固化——组长审批时若只能靠
    台账反查那一跳，反查不到的（归档、清理）就一律走不到例外通道，而那是最该免检的一类。

    词表二次校验（backlog 20261012-120001-leader）：gate 拦在组员的 Stop 上，但那一跳
    仍可能以带非法 kind 的最终回复收场（实测 110101：交付里连发两块，占位块
    ``kind="defusability-placeholder"`` 在前，被解析器取走并落盘，整改后的重交付另立新
    文件，空壳从此留在审批队列）。落盘是入队的**唯一**门，故在此复核词表：非法或缺失
    kind 一律不入库并告警——「退回改写」该是组员的下一步，不是组长的筛除负担。
    """
    vocab = load_infra_kinds()
    if kind not in vocab:
        print(
            f"[!] 改进建议未入库（{member_id}）：kind={kind!r} 不在受控词表"
            f"（{'|'.join(vocab)}）。交付本身照常记账，但队列里不会有它；"
            "确要提案请组员改正 kind 属性后重新交付。"
        )
        return ""
    SUGGESTIONS_DIR.mkdir(parents=True, exist_ok=True)
    sid = f"{_ts()}-{member_id}"
    p = SUGGESTIONS_DIR / f"{sid}.md"
    head = f"# 改进建议（{member_id}，待审批）\n\n"
    if kind:
        head += f"kind: {kind}\n"
    if delivery:
        head += f"delivery: {delivery}\n"
    p.write_text(head + "\n" + body + "\n", encoding="utf-8")
    return sid


def _accumulate_hop(final: Envelope, prev: Envelope) -> Envelope:
    """把重试/换档跳的 envelope 与已跑过的那跳**合并成整任务读数**。

    ``-o json`` 的 envelope 是**本次进程调用**的读数：``resume=True`` 的续跑只报它
    自己那几轮（E2 验收实测 fig-basic 首跳崩、续跑报 turns=2，而会话 transcript 里
    躺着 50+ 条 assistant 消息）。直接拿末跳 envelope 记账/判决有三处失真：

    - ``num_turns``/``duration_ms``/``total_credits`` 少报 → harness 的均值轮数判决、
      ``orch stats`` 的耗时画像都被做低；
    - ``permission_denials`` 少报 → pod-guard 在首跳拦下的拒绝被续跑清零，
      案卷的 ``权限拒绝上限`` 判据会**假通过**；
    - 只有 ``context_usage_ratio`` 是会话态而非增量，取两跳最大值。

    Args:
        final: 最后一跳解析出的 envelope。
        prev: 此前各跳的累计 envelope（可为本函数自己的返回值）。

    Returns:
        合并后的 envelope（不改动入参）。
    """
    return replace(
        final,
        num_turns=final.num_turns + prev.num_turns,
        duration_ms=final.duration_ms + prev.duration_ms,
        total_credits=final.total_credits + prev.total_credits,
        context_usage_ratio=max(final.context_usage_ratio, prev.context_usage_ratio),
        permission_denials=[*prev.permission_denials, *final.permission_denials],
    )


def _shallow_seat(reg: Registry, tier: str, *, current: str) -> str:
    """隐性态下「生效预设」的主力席——**浅回退**的落点；取不到/与当前席相同 → 空串。

    谷时 main 席链被隐性覆盖成 DeepSeek（路由 v3），而带宽打满的恰恰是它；此时先回
    到用户配置的预设主力席试一拍（用户补充口径 2026-10-13，backlog 20261013-193500-leader），
    比直接换到链上下一席更省、也更贴近「本预设本来该用的渠道」。
    """
    from . import presets as _presets

    name = _presets.active().name
    presets_face = reg.data.get("presets") if isinstance(reg.data, dict) else None
    chain = (presets_face or {}).get(name, {}).get(tier) or []
    seat = str(chain[0]) if chain else ""
    return seat if seat and seat != current else ""


def do_dispatch(
    member_id: str,
    *,
    text: str | None = None,
    task_file: str | None = None,
    slug: str | None = None,
    session: str = "latest",
    model_tier: str | None = None,
    effort: str | None = None,
    max_turns: int | None = None,
    timeout: int | None = None,
    dirs: list[str] | None = None,
    plan: str = "",
    step: str = "",
    no_checks: bool = False,
    no_retry: bool = False,
    no_hold: bool = False,
    quiet: bool = False,
    attach_replies: bool = True,
    eval_hop: bool = False,
    auto: bool = False,
    force_new_session: bool = False,
) -> DispatchOutcome:
    """执行一跳派发（完整链路）。参数语义与 ``orch dispatch`` 一致。

    ``auto=True``（drain 批次通路）：目标 sid 的上一跳进程仍在途时**自动换新 sid**并在
    台账标 ``sid_new_reason=prev_alive``；人工派发（默认）同情形**拒绝**（rc=2 点名占用
    PID/任务书），``force_new_session=True``（CLI ``--force-new-session``）放行为新 sid——
    禁止双进程同 sid 并发写会话历史（184000 修复面④/192119）。

    ``attach_replies=False``：不把组长的审批回复附进提示词、也不归档 ``sent/``
    （README §5.5：harness 压测跳必须与「生产跳」隔离，否则回复里的上下文会让
    压测结果不可比——``orch eval run`` 走这一支）。

    ``eval_hop=True``（``orch eval run`` 的压测跳）：不写成员生产台账
    （``state/ledger/deliveries.jsonl``）、不把 <infra_suggestion> 落进建议队列——
    压测的成本/轮数/ctx 只进 ``state/ledger/evals.jsonl``，否则合成案卷的读数会
    被 ``orch stats`` 计到组员头上、每轮跑还会往队列塞假建议。

    推理强度（``effort``）来源优先级：**调用方覆盖**（CLI ``--effort``、plan 环节、
    backlog 条目）> **registry 成员级** ``effort`` > 不传（跟随用户级默认=中）。
    标注非法时**硬失败且不派发**（code=2）——exe 收到非法档位会启动即失败（实测 rc=1），
    在本层先拦住才拿得到可读原因，也不白烧一跳、不写无意义台账行。

    相对用量（est_tokens）：本跳进出各测一次会话 jsonl 字符数，增量与换算值入台账，
    结束水位入 registry 会话条目（双渠道 usage 上报全为 0，绝对精度不可得——用户裁决
    2026-10-10 改做相对排名，backlog 20261010-ledger-est-tokens）。

    ``no_hold=True``（CLI ``--no-hold``）：跳过**边界避让**——距峰谷切换点不足
    ``routing.boundary_hold_s``（缺省 300s，硬钳制 ≤300s）时默认先睡过界再派，因为席链
    随时间走，卡在边界前几秒出发等于整跳落在将过期的窗口里（backlog 20261013-144500-leader）。
    草稿/调试场景才该跳过。

    ``channel_error`` 走**同席 resume 重试**而非直接换席（次数+壁钟预算双闸，
    ``model_fallback.same_seat_retry``，backlog 20261013-143000-leader）：渠道级故障是
    瞬态的，会话逐回合持久化故续跑零工作损失；重试不写回退事件（不是降级），次数/耗时
    入台账 ``same_seat_retries``/``same_seat_retry_ms``。额度耗尽与启动即失败不重试。

    Returns:
        DispatchOutcome（code=0 仅当 result 交付且声明的机械验收全过）。
    """
    reg = Registry.load()
    reg.ensure_member_defaults(member_id)
    m = reg.member(member_id)
    try:
        level = normalize_effort(effort) or normalize_effort(m.effort) or ""
    except ValueError as exc:
        if not quiet:
            print(f"[!] effort 标注无效，未派发：{exc}")
        return DispatchOutcome(
            code=2, kind="run_failed", member=member_id, error=str(exc)
        )
    if not m.pod.exists():
        if not quiet:
            print(f"[!] 成员 pod 不存在：{orchestration_rel(m.pod)}")
            print("    请先建 pod 骨架（见 orchestration/README.md §12 阶段计划）。")
        return DispatchOutcome(
            code=2, kind="run_failed", member=member_id, error="pod 不存在"
        )
    # 无记忆纪律（README §3.1：twin 每次判断必须冷启动）：会话复用在此硬拦。写成
    # 「charter 里劝一句」就必然被忘——纪律要落成 exit（§0.3 硬代码优先）。
    if m.raw.get("cold_start") and session != "new":
        if not quiet:
            print(f"[!] {member_id} 冷启动纪律：拒绝复用会话（session={session!r}）")
        return DispatchOutcome(
            code=2,
            kind="run_failed",
            member=member_id,
            error=f"cold_start 成员只允许 session='new'（收到 {session!r}）",
        )
    # 边界避让（用户裁决 2026-10-13，backlog 20261013-144500-leader）：距峰谷切换点
    # 不足阈值就先睡过界再派——席链随时间走，卡在边界前几秒出发=整跳落在将过期的窗口
    # 里（成本错配），而等待期零 LLM 成本。必须在**解析角色之前**：等过界后再读，
    # 读到的才是新窗口的链。避让本身不派发、不烧 token，故放在一切早退闸之后。
    hold_line, hold_wait_s = _boundary_hold(reg, no_hold=no_hold, quiet=quiet)

    # 路由切换检测（路由 v3 规格 2「切换时机」的记录面）：在**解析角色之前**读一次，
    # 于是这一跳用的席链就是切换后的新链——「只影响新派发、在途单跳不换模型」正是这个
    # 顺序保证的：已经在跑的那跳早已把 model 写进 argv，本行改的是它**下一跳**的读数。
    # 返回值是「跨过一次峰谷界」才有文案，平常空串不进台账。
    route_line = routing.note_switch()
    if route_line and not quiet:
        print(route_line)

    def _out(**kw: Any) -> DispatchOutcome:
        """本跳之后的每个出口都带上路由读数与边界避让读数。

        切换记录只在成功跳可见是不够的：谷时席表为空被激活闸拒掉的批次，恰恰最需要
        看见「这一跳本来要落到 DeepSeek 席」——拒发的出口也得带着它的路由身份。
        避让同理：拒发的那一跳也真的等过（队列被占了那么多秒），账要认。
        """
        return DispatchOutcome(
            route=route_line, hold=hold_line, hold_s=hold_wait_s, **kw
        )

    # 角色解析（档位模式退役，用户裁决 2026-10-12 22:4x）：覆盖（--model-tier / plan YAML
    # 的 model: / 案卷 tier:，旧档名 flash/max 归一）> 预设 pro 名单 > main。非法覆盖在此
    # 硬拦（code=2 不派发）——静默回落会把这一跳烧到用户没点名的档上去。
    try:
        role = presets.resolve_role(member_id, model_tier)
    except ValueError as exc:
        if not quiet:
            print(f"[!] 角色/档位无效，未派发：{exc}")
        return _out(code=2, kind="run_failed", member=member_id, error=str(exc))
    active = presets.active()
    tier = role
    # 席位选择（沿 #80：角色 → **有序回退链**，首席在前）：链里落在降级粘滞窗内的席本跳
    # 不探测（省跳数），但仍要逐席打出来——故障可查性不能换成安静。
    plan = _channels.plan_seats(reg, tier)
    chain = list(plan.usable)
    gate = presets.preset_gate(reg, tier)
    if gate:
        # 本预设该档席表为空 = 没配席：**拒绝激活**而不是悄悄落回上一个预设——静默回落
        # 会让台账烧的是别的渠道，而用户以为切换已经生效。
        if not quiet:
            print(f"[!] {gate}")
            print(
                "    处置：改 data/pysci_settings.toml 的 [model] preset（合法串见该文件注释）。"
            )
        return _out(code=2, kind="run_failed", member=member_id, error=gate)
    if not quiet:
        for warn in active.warnings:  # 非法/缺配置回落出厂预设时必须出声
            print(f"[!] {warn}")
    model = chain[0] if chain else str(reg.models.get(tier, ""))
    if not quiet:
        for item in plan.skipped:
            print(_channels.skip_line(item))
        if plan.forced_probe:
            print(_channels.forced_line(plan))
    slug = slug or (Path(task_file).stem if task_file else f"adhoc-{_ts()}")

    # 会话选择（README §3.2：默认续最近活跃会话）
    if session == "new":
        sid, resume = new_session_id(), False
    elif session == "latest":
        # fe68a40 的 FRESH_CTX_RATIO 水位闸已回退（用户 2026-10-13 审查裁决：策略变更
        # 绕过了全部判决通道，且压缩防溢出使闸的宣称收益无效）。ctx_ratio 仍随 touch_session
        # 落池（观测面），恢复「续最近活跃会话」语义；重启类似机制须经判决包。
        active = m.active_sessions
        sid, resume = (active[0]["sid"], True) if active else (new_session_id(), False)
    else:
        found = m.find_session(session)
        if found is None:
            if not quiet:
                print(f"[!] 会话 '{session}' 未在池中找到。当前池：")
                print("\n".join(_sessions.format_pool(m)))
            return _out(
                code=2,
                kind="run_failed",
                member=member_id,
                error=f"会话 {session} 未找到",
            )
        sid, resume = found["sid"], True
    # sid 在途闸（184000 修复面④/192119）：上一跳进程仍活 = 双进程同 sid 并发写会话历史
    # （#69 实证，两个进程共写 6f0cc8e5…）。人工派发默认拒绝并点名占用证据；drain 批次
    # （auto）自动换新 sid 并在台账标注；--force-new-session 人工放行同法。无在途证据
    # （没记 pid / pid 已死 / argv 对不上 sid）fail-open 照续，不冤枉正常跑。
    sid_new_reason = ""
    if resume:
        occ = _sid_inflight(m, sid)
        if occ and auto:
            old_sid, sid, resume = sid, new_session_id(), False
            sid_new_reason = "prev_alive"
            print(
                f"[i] sid {old_sid[:8]} 上一跳进程仍在途（pid={occ['pid']}，"
                f"任务书 {occ['task']}）→ 自动起新 sid，台账 sid_new_reason=prev_alive"
            )
        elif occ and not force_new_session:
            msg = (
                f"会话 {sid} 的上一跳进程仍在途：pid={occ['pid']}，"
                f"任务书 {occ['task']}（双进程同 sid 会并发写会话历史，已拒绝）。"
                "待其退出后重试，或加 --force-new-session 改起新会话"
            )
            print(f"[!] {msg}")
            return _out(
                code=2,
                kind="run_failed",
                member=member_id,
                sid=sid,
                error=msg,
            )
        elif occ:  # 人工 + --force-new-session：放行 = 改起新 sid（不与在途进程同 sid）
            sid, resume = new_session_id(), False
            sid_new_reason = "prev_alive"
            print(f"[i] --force-new-session：上一跳 pid={occ['pid']} 仍活 → 改起新会话")
    if not resume:
        _sessions.register_session(m, sid, slug)
    name = slug if not resume else (m.find_session(sid) or {}).get("name", slug)

    # 相对用量尺子（backlog 20261010-ledger-est-tokens，用户裁决 2026-10-10）：双渠道
    # usage/credits 上报缺口 → 精确 token 不可得，改测「本跳期间会话 jsonl 长多少」。
    # 增量基准优先取 registry 会话条目里**上一跳记账的水位**：崩溃跳/手动 resume 造成的
    # 记账外增长不会凭空蒸发，而是计入同一会话的下一跳（总量守恒 → 成员间排名不被漏记
    # 扭曲）。水位反而高于实测起点，只可能是文件被重写（压缩/清理），此时以实测起点为准。
    per_char = reg.est_tokens_per_char
    chars_start = _sessions.session_chars(m, sid)
    stored_offset = int((m.find_session(sid) or {}).get("chars_offset", 0) or 0)
    chars_base = min(stored_offset, chars_start) if stored_offset else chars_start

    def measure_usage() -> tuple[int, int, int]:
        """返回 ``(delta_chars, est_tokens, 本跳结束时的字符水位)``。"""
        after = _sessions.session_chars(m, sid)
        delta = max(0, after - chars_base)
        return delta, int(round(delta * per_char)), after

    taskbook = write_taskbook(m.pod, text=text, task_file=task_file, slug=slug)

    def _note_pid(pid: int) -> None:
        """派生瞬间把会话进程 pid 记进运行态（第四面回收的判据，见 note_hop_pid）。"""
        note_hop_pid(member_id, sid, pid=pid, task=taskbook.name)

    replies_body, reply_files = (
        collect_replies(member_id) if attach_replies else ("", [])
    )
    parts = [
        f"你的任务书（必读，位于你的 pod 内）：{taskbook.name}",
        f"绝对路径：{taskbook}",
    ]
    if replies_body:
        parts.append(f"\n{replies_body}")
    parts.append(f"\n{PROTOCOL_REMINDER}")
    prompt = "\n".join(parts)

    extra_dirs = list(m.raw.get("extra_dirs", []))
    all_dirs = list(dirs or []) + extra_dirs
    cmd = build_command(
        m,
        prompt,
        session_id=sid,
        resume=resume,
        model=model,
        max_turns=max_turns,
        effort=level,
    )
    env = build_env(m, all_dirs, deployed_skills=deployed_skills_for(m.pod))

    if not quiet:
        eff = f"，effort {level}" if level else ""
        print(
            f"→ 派发 {member_id}（会话 {'续:' + sid[:8] if resume else '新:' + sid[:8]}"
            f"，模型 {model or '(默认)'}{eff}，任务书 {taskbook.name}）"
        )
    rc, out, err = run_headless(
        cmd,
        cwd=PROJECT_ROOT,
        env=env,
        timeout_s=timeout or m.timeout_s,
        on_spawn=_note_pid,
    )
    env_json = acc_env = Envelope.parse(out)
    failed = rc != 0 or env_json.is_error
    # 席位不可用（额度耗尽 / 渠道级故障 / 启动即失败）**不做普通重试**（上面那段只吃
    # 非系统性失败）；出路是下面的处置：渠道级故障先同席 resume 自救（用户裁决
    # 2026-10-13），额度耗尽/启动即失败直接沿该档有序回退链换席（机制 2026-10-10 flash
    # 回退，链形态与「启动即失败也算席坏」的用户裁决 2026-10-12 21:1x，
    # backlog 20261012-211500-leader）
    systemic = _detect_systemic(env_json, out, err, member=m, sid=sid) if failed else ""
    broken = _seat_broken(env_json, rc, systemic)

    if failed and not no_retry and not broken:
        if not quiet:
            print(
                f"[!] 首跑失败（rc={rc}, stop={env_json.stop_reason}），自动重试一次…"
            )
        retry_prompt = (
            f"上一次运行异常中断（rc={rc}；stderr 摘要：{err.strip()[-400:]}）。"
            f"请继续完成任务书 {taskbook}，按交付协议收尾。\n\n{PROTOCOL_REMINDER}"
        )
        cmd2 = build_command(
            m,
            retry_prompt,
            session_id=sid,
            resume=True,
            model=model,
            max_turns=max_turns,
            effort=level,
        )
        rc, out, err = run_headless(
            cmd2,
            cwd=PROJECT_ROOT,
            env=env,
            timeout_s=timeout or m.timeout_s,
            on_spawn=_note_pid,
        )
        env_json = acc_env = _accumulate_hop(Envelope.parse(out), acc_env)
        failed = rc != 0 or env_json.is_error
        systemic = systemic or (
            _detect_systemic(env_json, out, err, member=m, sid=sid) if failed else ""
        )
        broken = _seat_broken(env_json, rc, systemic)

    # 同席重试（用户裁决 2026-10-13，backlog 20261013-143000-leader）：channel_error 是
    # **瞬态**的（平台侧算力不足，今日 4 起），同会话 resume 续跑大概率即愈且零工作损失
    # （会话逐回合持久化）；而换席会落付费席。故付费席降为后手：先在原席自救，次数
    # （缺省 10）与壁钟预算（缺省 1800s）双闸都过才换席。额度耗尽（确定性）与启动即失败
    # （席位根本不存在）**不进本循环**——表里没有它们的键，同席重试只会白烧。
    # 重试不写 note_fallback（规格 3：重试不是降级，不污染 mirror「渠道回退」事件面），
    # 次数/耗时另走台账与 run 日志。
    same_seat_retries = 0
    same_seat_retry_ms = 0
    policy = reg.same_seat_retry
    retry_limit = int(policy.get("channel_error", 0))
    retry_backoff_s = float(policy.get("backoff_s", 0))
    retry_budget_s = float(policy.get("budget_s", 0))
    retry_started = _clock()
    while failed and broken == "channel_error" and same_seat_retries < retry_limit:
        if retry_budget_s > 0 and _clock() - retry_started >= retry_budget_s:
            # 次数闸挡不住「真下线」：每次失败若耗时数分钟，10 次会拖住串行队列近小时。
            if not quiet:
                print(
                    f"[!] 同席重试预算触顶（{retry_budget_s:.0f}s 已耗）→ 转换席，"
                    "不再在原席上等"
                )
            break
        same_seat_retries += 1
        if retry_backoff_s > 0:
            _sleep(retry_backoff_s)
        if not quiet:
            print(
                f"[!] 渠道瞬态故障，同席 resume 重试 "
                f"{same_seat_retries}/{retry_limit}（席位不动，不烧换席）…"
            )
        rc, out, err = run_headless(
            build_command(
                m,
                (
                    f"上一次运行因渠道瞬态故障中断（rc={rc}）。已原席续跑，"
                    f"请继续完成任务书 {taskbook}，按交付协议收尾。\n\n{PROTOCOL_REMINDER}"
                ),
                session_id=sid,
                resume=True,
                model=model,
                max_turns=max_turns,
                effort=level,
            ),
            cwd=PROJECT_ROOT,
            env=env,
            timeout_s=timeout or m.timeout_s,
            on_spawn=_note_pid,
        )
        env_json = acc_env = _accumulate_hop(Envelope.parse(out), acc_env)
        failed = rc != 0 or env_json.is_error
        # 每轮**重新判**而不是 `or` 累积：重试跑通时必须把首轮的 channel_error 抹掉，
        # 否则成功交付会被首轮的判定拖成系统性故障（kind 错、drain 还会全局停）。
        systemic = (
            _detect_systemic(env_json, out, err, member=m, sid=sid) if failed else ""
        )
        broken = _seat_broken(env_json, rc, systemic)
    same_seat_retry_ms = int(round((_clock() - retry_started) * 1000))
    if same_seat_retries and not quiet and not (failed and broken):
        print(f"[√] 渠道瞬态故障已同席自愈（重试 {same_seat_retries} 次）")

    # **浅回退一拍**（用户补充 2026-10-13，backlog 20261013-193500-leader）：隐性 DeepSeek
    # 态下带宽失败，先回到「生效预设的主力席」试一跳；该席亦爆才走下面的换席链
    # （而 drain 侧会把带宽类判成延期）。不写 note_fallback：这一跳不是「席位坏了」的
    # 降级，而是「隐性覆盖席被带宽打满」时的回退，把它记成降级会污染渠道健康读数。
    if broken == "channel_error" and not no_retry and routing.hidden_valley_preset():
        shallow = _shallow_seat(reg, tier, current=model)
        if shallow:
            if not quiet:
                print(
                    f"[!] 隐性席带宽不足 → 浅回退到本预设主力席 {reg.seat_label(shallow)}"
                )
            model = shallow
            rc, out, err = run_headless(
                build_command(
                    m,
                    (
                        "上一次运行因渠道带宽不足中断（隐性覆盖席）。已浅回退到本预设主力席，"
                        f"请继续完成任务书 {taskbook}，按交付协议收尾。{chr(10)}{chr(10)}"
                        + PROTOCOL_REMINDER
                    ),
                    session_id=sid,
                    resume=True,
                    model=model,
                    max_turns=max_turns,
                    effort=level,
                ),
                cwd=PROJECT_ROOT,
                env=env,
                timeout_s=timeout or m.timeout_s,
                on_spawn=_note_pid,
            )
            env_json = acc_env = _accumulate_hop(Envelope.parse(out), acc_env)
            failed = rc != 0 or env_json.is_error
            systemic = (
                _detect_systemic(env_json, out, err, member=m, sid=sid)
                if failed
                else ""
            )
            broken = _seat_broken(env_json, rc, systemic)

    # 沿链换席：一席一跳，跑通即止（规格 3）。每次降级都过一遍 note_fallback，它同时
    # 落两笔账：故障席进**粘滞窗**（窗内后续派发直接跳过，不再白烧探测跳——BYOK 渠道
    # 余额长期不可用是用户 21:3x 明示的现状）+ 渠道事件流一行（mirror/status 面的读数源）。
    # 告警行是硬要求（规格 4a，双面之一）：谁挂了、换到哪，必须让调用者当场看见。
    fallbacks: list[dict[str, Any]] = []
    if broken and not no_retry:
        for nxt in chain[1:]:
            event = _channels.note_fallback(
                reg,
                tier=tier,
                frm=model,
                to=nxt,
                reason=broken,
                member=member_id,
                sid=sid,
            )
            fallbacks.append(event)
            if not quiet:
                print(_channels.alert_line(event))
            model = nxt  # 台账/后续报告记实际所用席位
            hop_prompt = (
                f"上一次运行因{_seat_reason_text(broken)}而中断（rc={rc}）。"
                f"已切到 {tier} 档下一席，请继续完成任务书 {taskbook}，"
                f"按交付协议收尾。\n\n{PROTOCOL_REMINDER}"
            )
            rc, out, err = run_headless(
                build_command(
                    m,
                    hop_prompt,
                    session_id=sid,
                    resume=True,
                    model=model,
                    max_turns=max_turns,
                    effort=level,
                ),
                cwd=PROJECT_ROOT,
                env=env,
                timeout_s=timeout or m.timeout_s,
                on_spawn=_note_pid,
            )
            env_json = acc_env = _accumulate_hop(Envelope.parse(out), acc_env)
            failed = rc != 0 or env_json.is_error
            # 最后一席的结果即最终结果：仍系统性故障 → 该 kind（drain 全局停止）；
            # 普通失败 → run_failed；成功 → 落入正常交付解析
            systemic = (
                _detect_systemic(env_json, out, err, member=m, sid=sid)
                if failed
                else ""
            )
            broken = _seat_broken(env_json, rc, systemic)
            if not broken:
                break
        if broken and fallbacks:
            # 全链耗尽：这一笔也必须落账，否则「停止」在观测面上是隐形的。
            # 启动即失败耗尽 → 升格为 channel_error：UUID 失效本来就是渠道级故障，
            # 且只有进了 SYSTEMIC_KINDS 才拿得到 #77 的全局停止（规格 3 要的正是
            # 「别逐单空烧」）；单席链（无可换）不升格，免得把一次性抖动说成停摆。
            event = _channels.note_fallback(
                reg,
                tier=tier,
                frm=model,
                to="",
                reason=broken,
                member=member_id,
                sid=sid,
            )
            fallbacks.append(event)
            if not quiet:
                print(_channels.alert_line(event))
            if broken == "startup_failure":
                systemic = "channel_error"

    if not failed:
        # 跑通即销窗：充值/修好后的第一次成功就恢复正常优先级（无需改代码自愈）
        _channels.note_seat_ok(reg, tier=tier, seat=model)

    if failed:
        kind = systemic or "run_failed"
        err_text = (err.strip()[-400:] or env_json.stop_reason).strip()
        if systemic:
            # 台账 error 面须分辨两类系统性故障（backlog 20261012-194500-leader 要求 3）：
            # 两类的原始读数可能完全相同（渠道故障常只报 stop_sequence），不带 kind 就分不开
            err_text = f"{SYSTEMIC_TEXT[systemic][1]}｜{err_text}"
        delta_chars, est_tokens, chars_after = measure_usage()
        if not eval_hop:
            append(
                LedgerEntry(
                    ts="",
                    member=member_id,
                    session_id=sid,
                    session_name=name,
                    task_file=orchestration_rel(taskbook),
                    kind=kind,
                    num_turns=env_json.num_turns,
                    duration_ms=env_json.duration_ms,
                    credits=env_json.total_credits,
                    model=model,
                    effort=level,
                    delta_chars=delta_chars,
                    est_tokens=est_tokens,
                    ctx_ratio=env_json.context_usage_ratio,
                    permission_denials=len(env_json.permission_denials),
                    plan=plan,
                    step=step,
                    sid_new_reason=sid_new_reason,
                    error=err_text,
                    error_text=_provider_error_text(env_json, out, err),
                    same_seat_retries=same_seat_retries,
                    same_seat_retry_ms=same_seat_retry_ms,
                    hold_s=hold_wait_s,
                )
            )
        _sessions.touch_session(
            m, sid, chars_offset=chars_after, ctx_ratio=env_json.context_usage_ratio
        )
        reg.write_sessions(m)
        reg.save()
        if not eval_hop:
            # ⑤ run_failed/系统性同样落审计副本——它们在 drain 也会产 needs_leader note，
            # 截断的残片证据同源。env_json.result 是组员原始输出（含 <blocked>/<result> 全文）。
            _audit_blocked_delivery(m.pod, env_json.result)
        outcome = _out(
            code=2,
            kind=kind,
            member=member_id,
            sid=sid,
            session_name=name,
            task_file=orchestration_rel(taskbook),
            effort=level,
            delta_chars=delta_chars,
            est_tokens=est_tokens,
            body=env_json.result[:800],
            envelope=env_json,
            sid_new_reason=sid_new_reason,
            error=err_text,
            error_text=_provider_error_text(env_json, out, err),
            fallbacks=fallbacks,
            same_seat_retries=same_seat_retries,
            same_seat_retry_ms=same_seat_retry_ms,
        )
        if not quiet:
            if systemic:
                _report_systemic(outcome)
            else:
                _report_failure(outcome)
        return outcome

    delivery = parse_delivery(env_json.result)
    # ⑤ blocked 交付全文落 outbox 审计副本（184000 修复面⑤）：needs_leader note 截 200 字
    # 会让关键证据变残片（两案实证），全文必须留在 `pods/<m>/outbox/delivery-<ts>.xml`
    # 供组长/审计复原。失败通路（run_failed/系统性）同理——它们不走 parse_delivery，
    # 在失败分支另落一份 env_json.result。
    if delivery.kind == "blocked" and not eval_hop:
        _audit_blocked_delivery(m.pod, env_json.result)
    check_results = (
        [] if no_checks else run_checks(delivery.artifacts, reg, member_pod=m.pod)
    )
    suggestion_id = ""
    if delivery.infra_suggestion.strip() and not eval_hop:
        suggestion_id = _persist_suggestion(
            member_id,
            delivery.infra_suggestion,
            delivery.infra_kind,
            delivery.kind,
        )
    delta_chars, est_tokens, chars_after = measure_usage()
    entry = LedgerEntry(
        ts="",
        member=member_id,
        session_id=sid,
        session_name=name,
        task_file=orchestration_rel(taskbook),
        kind=delivery.kind,
        checks=check_results,
        # 口径合一（backlog 20261012-132138-devops）：本字段答的是**「队列里有没有这条
        # 建议」**，不是**「交付文本里有没有那个块」**。此前写的是后者，于是非法 kind 被
        # :func:`_persist_suggestion` 拒之门外的那一跳，台账仍打「建议✓」——组长照读数
        # 去找提案，`orch suggestions` 里却什么都没有（#62 同源：读数面与入库面分裂）。
        # 未入库这件事仍可见：呈现面按 infra_kind 非空而本字段为假打「建议✗未入库」。
        infra_suggestion=bool(suggestion_id),
        infra_kind=delivery.infra_kind,
        tags=delivery.tags,
        reason=delivery.reason,
        num_turns=env_json.num_turns,
        duration_ms=env_json.duration_ms,
        credits=env_json.total_credits,
        model=model,
        effort=level,
        delta_chars=delta_chars,
        est_tokens=est_tokens,
        ctx_ratio=env_json.context_usage_ratio,
        permission_denials=len(env_json.permission_denials),
        plan=plan,
        step=step,
        sid_new_reason=sid_new_reason,
        same_seat_retries=same_seat_retries,
        same_seat_retry_ms=same_seat_retry_ms,
        hold_s=hold_wait_s,
    )
    if not eval_hop:
        append(entry)
    for p in reply_files:  # 审批回复已附送 → 归档 sent/
        sent = p.parent / "sent"
        sent.mkdir(exist_ok=True)
        shutil.move(str(p), str(sent / p.name))
    _sessions.touch_session(
        m, sid, chars_offset=chars_after, ctx_ratio=env_json.context_usage_ratio
    )
    reg.write_sessions(m)
    reg.save()

    failed_checks = [c for c in check_results if c["verdict"] == "fail"]
    code = 0 if (delivery.kind == "result" and not failed_checks) else 2
    outcome = _out(
        code=code,
        kind=delivery.kind,
        member=member_id,
        sid=sid,
        session_name=name,
        task_file=orchestration_rel(taskbook),
        effort=level,
        delta_chars=delta_chars,
        est_tokens=est_tokens,
        body=delivery.body,
        checks=check_results,
        infra_suggestion=delivery.infra_suggestion,
        infra_kind=delivery.infra_kind,
        tags=delivery.tags,
        reason=delivery.reason,
        warning=delivery.warning,
        suggestion_id=suggestion_id,
        envelope=env_json,
        sid_new_reason=sid_new_reason,
        fallbacks=fallbacks,
        same_seat_retries=same_seat_retries,
        same_seat_retry_ms=same_seat_retry_ms,
    )
    if not quiet:
        _report_success(outcome, failed_checks)
    return outcome


# ---------------------------------------------------------------------------
# 报告输出（面向组长；[NEXT] 红线）
# ---------------------------------------------------------------------------
def _print_next(lines: list[str]) -> None:
    print()
    print("[NEXT]")
    for line in lines:
        print(f"  {line}")


def _report_failure(o: DispatchOutcome) -> None:
    print(f"[✗] 运行失败：{o.error}")
    print(f"    envelope.result 原文（前 600 字）：\n{o.body[:600]}")
    _print_next(
        [
            "运行级失败（非交付失败）。可选：",
            f'  1) 再试一跳：uv run --no-sync pysci-orch dispatch {o.member} --session {o.sid[:8]} --text "继续任务书 {o.task_file}"',
            "  2) 查会话原文定位卡点（registry/pod 键目录下 jsonl）",
            "  3) 仍失败 → 上报用户（附本输出全文）",
        ]
    )


def _report_systemic(o: DispatchOutcome) -> None:
    """系统性故障报告面：两类故障的**处置**相同（换档/停止）、**成因**不同（文案分列）。"""
    headline = SYSTEMIC_TEXT.get(o.kind, (o.kind, o.kind))[0]
    print(f"[✗] {headline}（{o.kind}）：{(o.error or o.body)[:160]}")
    print(f"    envelope.result 原文（前 400 字）：\n{o.body[:400]}")
    if o.same_seat_retries:
        # 「先自救过了」必须可见：否则组长看到 channel_error 会以为直接跳到了付费席，
        # 而实际已经把瞬态窗口等满（次数+预算双闸），这是判断「渠道在抖 vs 已死」的读数
        print(
            f"    [i] 已同席 resume 重试 {o.same_seat_retries} 次"
            f"（{o.same_seat_retry_ms / 1000:.0f}s）仍未愈 → 才换席/停链"
        )
    cause = (
        "模型渠道额度耗尽——充值或换渠道才恢复"
        if o.kind == "quota_exhausted"
        else "模型渠道级故障（服务面/限速）——等渠道恢复或换渠道，"
        "同渠道逐单重试只会连排空烧"
    )
    _print_next(
        [
            f"系统性故障（{cause}）——非任务本身问题，勿逐跳重试再烧：",
            "  1) 查该档回退链 registry models（README §3.5：Mimo 主力 → Qwen BYOK 回退 →"
            "内置兜底；档值是**有序链**，首席挂了会自动往后换席）；"
            "链里全挂了才说明整档要动，models 属护栏级、改动须经用户",
            "  2) 或等待额度重置 / 请用户充值切换渠道后重派；故障席进跳过窗（"
            "registry model_fallback.skip_window_s），窗内不重复探测，过期自动恢复",
            "  3) drain 场景已触发全局停止：剩余条目保持 pending，渠道恢复后自动续消化",
            f"  4) 回退事件账：{orchestration_rel(_channels.EVENTS_PATH)}"
            "（mirror status / orch status 健康段已汇总）",
        ]
    )


def _report_success(o: DispatchOutcome, failed_checks: list[dict]) -> None:
    env = o.envelope
    icon = {"result": "√", "blocked": "✗", "parse_error": "?"}.get(o.kind, "?")
    dur = env.duration_ms / 1000 if env else 0
    turns = env.num_turns if env else 0
    ctx = env.context_usage_ratio if env else 0
    est = f"，est≈{o.est_tokens:,} tok" if o.est_tokens else ""
    print(
        f"[{icon}] {o.member} 交付类型：{o.kind}"
        f"（{turns} 轮，{dur:.0f}s，ctx={ctx:.0%}{est}）"
    )
    print(f"    会话：{o.sid[:8]}（{o.session_name}）")
    if o.warning:
        # 协议扩展的滚动升级窗口：本跳交付由宽松兜底面救回，判读可用但不可当正常路径
        print(f"    [!] {o.warning}")
    if o.hold:
        # 边界避让是**等出来的**窗口对齐：不报这一行，组长看台账耗时会以为派发卡了
        print(f"    {o.hold}")
    if o.same_seat_retries:
        print(
            f"    [!] 渠道瞬态故障：同席 resume 重试 {o.same_seat_retries} 次"
            f"（{o.same_seat_retry_ms / 1000:.0f}s）后{'仍未愈，已换席' if o.fallbacks else '自愈'}"
        )
    if o.fallbacks:
        # 规格 4a「交付告警」：降级跑通的跳**也是**告警跳——只报成功不报路径，
        # 组长就会以为主力渠道还好着（事件流的账只有 mirror/status 面才去看）
        print(
            f"    [!] 本跳发生 {len(o.fallbacks)} 起档位回退，最终所用席位见台账 model 字段"
        )
    if o.tags:
        print(f"    失败标签：{', '.join(o.tags)}")
    print(
        f"---- 交付正文（前 800 字）----\n{o.body[:800]}\n----------------------------"
    )
    if o.checks:
        print("机械验收：")
        for c in o.checks:
            mark = {
                "pass": "√",
                "fail": "✗",
                "unrouted": "○ 未注册(放行)",
                "skipped": "– 成员判断不验收",
            }[c["verdict"]]
            line = f"  [{mark}] {c['check']}: {c['path']}"
            if c["verdict"] == "fail":
                line += f"\n      输出摘要：{c.get('output_tail', '')[-300:]}"
            if c.get("reason"):
                line += f"（理由：{c['reason']}）"
            print(line)
    if o.infra_suggestion:
        print(
            f"---- 改进建议（已登记 {o.suggestion_id}，待审批）----\n"
            f"{o.infra_suggestion[:600]}\n------------------"
        )
    if env and env.permission_denials:
        print(
            f"[!] 权限拒绝 {len(env.permission_denials)} 次（配置体检信号）："
            f"{env.permission_denials[:3]}"
        )

    if o.kind == "result" and not failed_checks:
        nxt = ["本环节完成。"]
        if o.suggestion_id:
            nxt.append(
                f'改进建议待审批：uv run --no-sync pysci-orch approve {o.suggestion_id} --note "..."'
            )
            nxt.append(
                f'                 或 uv run --no-sync pysci-orch reject {o.suggestion_id} --note "..."'
            )
        nxt.append(
            "若属多环节工作：推进计划（orch plan run <id>）或书写下一任务书再 dispatch。"
        )
        _print_next(nxt)
        return
    if failed_checks:
        detail = "; ".join(
            f"{c['check']}:{Path(c['path']).name}" for c in failed_checks
        )
        _print_next(
            [
                f"机械验收 FAIL（{detail}）→ 建议回派返工：",
                f'  uv run --no-sync pysci-orch dispatch {o.member} --session {o.sid[:8]} --text "返工：机械验收未过（{detail}）。修正后重新交付。"',
                "同指纹连败 3 次 → 停止回派，升级决策（改任务书/咨询副组长/上报用户）。",
            ]
        )
        return
    if o.kind == "blocked":
        _print_next(
            [
                "组员被阻塞。可选：",
                f'  1) 补充信息重派：uv run --no-sync pysci-orch dispatch {o.member} --session {o.sid[:8]} --text "<补充>"',
                '  2) 咨询副组长：uv run --no-sync pysci-orch consult "<问题>"',
                "  3) 上报用户（附交付正文）",
            ]
        )
        return
    _print_next(
        [
            "交付格式解析失败（parse_error）。原回复见上方正文预览。",
            f'重派补格式：uv run --no-sync pysci-orch dispatch {o.member} --session {o.sid[:8]} --text "上次交付缺 <result>/<blocked> 标签块，请按 charter 重新收尾（工作不必重做）。"',
        ]
    )
