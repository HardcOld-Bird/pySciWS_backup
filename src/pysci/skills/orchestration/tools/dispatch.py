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
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from pysci.paths import ORCH_STATE_ROOT, PROJECT_ROOT

from . import channels as _channels
from . import presets, procs, routing
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

#: 额度类失败文案（2026-10-10 事故原文 "You've reached your credit usage limit…"）。
#: 匹配面故意收窄到 credit 类短语——误判为额度故障会全局停 drain（fail-safe 方向），
#: 但宽匹配（如裸 "usage limit"）会把速率限制等可自愈故障也停掉。
QUOTA_ERROR_RE = re.compile(
    r"credit usage limit|out of credits?|insufficient credits?", re.IGNORECASE
)

#: 渠道级失败的固定句（2026-10-12 实测：BYOK 渠道 token-plan 月额度耗尽 / 传输层故障时，
#: CLI 不写 credit 文案、也不给非零退出码，而是合成一条 assistant 回合说这句话）。
#: 与 quota 文案**分列**：两类故障的恢复动作不同（充值 vs 等待/换渠道），台账读数须可辨。
CHANNEL_ERROR_RE = re.compile(
    r"custom model service error|retry later or switch to another model",
    re.IGNORECASE,
)

#: 系统性故障 kind 族——「不是任务书坏了，是渠道坏了」。三处消费面（drain / benchd /
#: inspectd 的全局停止、watch 的事件、orch status 的提醒行）按这一份清单判，别各写各的。
SYSTEMIC_KINDS: tuple[str, ...] = ("quota_exhausted", "channel_error")

#: kind → （报告面短标题, 长说明）。文案单一源：drain/bench/inspect 的 ``.done`` error、
#: orch status、dispatch 报告面共读，避免「同一故障三副面孔」。
SYSTEMIC_TEXT: dict[str, tuple[str, str]] = {
    "quota_exhausted": ("额度类失败", "模型渠道额度耗尽（quota_exhausted）"),
    "channel_error": ("渠道级失败", "模型渠道级故障（channel_error）"),
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


def _detect_quota_exhaustion(env: Envelope, out: str, err: str) -> bool:
    """识别额度类失败：文案可能出现在 envelope.result / stderr / stdout 全文。

    **仅在运行已失败时调用**——成功交付的正文若只是提及该文案（如复盘任务）不应
    重分类。
    """
    return any(QUOTA_ERROR_RE.search(s) for s in (env.result, err, out) if s)


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
    """
    text_hit = any(CHANNEL_ERROR_RE.search(s) for s in (env.result, err, out) if s)
    shape = _sessions.channel_failure_turn(member, sid, text_re=CHANNEL_ERROR_RE)
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
    body: str = ""  # 交付正文（result/blocked 块内容）
    checks: list[dict] = field(default_factory=list)
    infra_suggestion: str = ""
    infra_kind: str = ""  # E0-B：交付 XML 上的 kind 属性（defect/usability/""）
    tags: list[str] = field(default_factory=list)  # E1：<blocked tags="a,b"> 失败标签
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
    #: sid 在途闸换新会话的原因（``prev_alive``=上一跳进程仍活；""=没换）。台账同名字段
    #: 留痕，drain 据此写 run 日志——人工拒绝与批次换新都必须可见（184000 修复面④）。
    sid_new_reason: str = ""
    #: 本跳的**路由切换**读数（路由 v3 规格 2）：跨过一次峰谷界时是一句人读文案，平常空串。
    #: drain 把它抄进 run 日志——「这一跳烧的是隐性 DeepSeek 席」必须留在批次账上，
    #: 而不能只活在 toml 与 routing-last.json 里（组长读的是 run 日志）。
    route: str = ""


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


def lint_taskbook(body: str) -> list[str]:
    """扫任务书正文，返回命中的错误 CLI 范式告警（空=干净）。

    **告警不阻断**：单条规则 regex 非法只跳过该条，不影响其余规则与派发。
    """
    warnings: list[str] = []
    for rule in load_lint_rules():
        try:
            pat = re.compile(rule["re"])
        except re.error:
            continue
        if pat.search(body):
            tag = f"[{rule['id']}] " if rule.get("id") else ""
            warnings.append(
                f"{tag}{rule.get('hint', '命中错误 CLI 范式 ' + rule['re'])}"
            )
    return warnings


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
    for warn in lint_taskbook(body):
        print(f"[!] 任务书 lint：{warn}")
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
    # 路由切换检测（路由 v3 规格 2「切换时机」的记录面）：在**解析角色之前**读一次，
    # 于是这一跳用的席链就是切换后的新链——「只影响新派发、在途单跳不换模型」正是这个
    # 顺序保证的：已经在跑的那跳早已把 model 写进 argv，本行改的是它**下一跳**的读数。
    # 返回值是「跨过一次峰谷界」才有文案，平常空串不进台账。
    route_line = routing.note_switch()
    if route_line and not quiet:
        print(route_line)

    def _out(**kw: Any) -> DispatchOutcome:
        """本跳之后的每个出口都带上路由读数。

        切换记录只在成功跳可见是不够的：谷时席表为空被激活闸拒掉的批次，恰恰最需要
        看见「这一跳本来要落到 DeepSeek 席」——拒发的出口也得带着它的路由身份。
        """
        return DispatchOutcome(route=route_line, **kw)

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
    # 席位不可用（额度耗尽 / 渠道级故障 / 启动即失败）**不做同席位重试**——同席位再跳
    # 只会再烧一跳；出路是沿该档有序回退链换席（机制 2026-10-10 flash 回退，链形态与
    # 「启动即失败也算席坏」的用户裁决 2026-10-12 21:1x，backlog 20261012-211500-leader）
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
            fallbacks=fallbacks,
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
        warning=delivery.warning,
        suggestion_id=suggestion_id,
        envelope=env_json,
        sid_new_reason=sid_new_reason,
        fallbacks=fallbacks,
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
