"""派发核心：任务书落 inbox → headless 执行 → 交付解析 → 机械验收 → 台账/建议持久化。

从门面（orch.py）抽出，供 ``dispatch`` 命令与 plan-runner（plans.py）复用。
输出协议：``quiet=False`` 时打印完整组员报告与 [NEXT]；plan-runner 以 quiet=True
调用后自行打印交接报告。
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from pysci.paths import ORCH_STATE_ROOT, PROJECT_ROOT

from . import sessions as _sessions
from . import sync as _sync
from .checks import run_checks
from .delivery import parse_delivery
from .ledger import LedgerEntry, append
from .registry import Registry, orchestration_rel
from .runner import (
    PROTOCOL_REMINDER,
    Envelope,
    build_command,
    build_env,
    new_session_id,
    run_headless,
)

SUGGESTIONS_DIR = ORCH_STATE_ROOT / "suggestions"
REPLIES_DIR = ORCH_STATE_ROOT / "replies"

#: 任务书轻量 lint 规则表（JSON: {rules: [{id?, re, hint}]}）；devops 可增补。
#: 组长手写任务书易把错误 CLI 范式传染给照抄的组员，落盘前扫一遍作机械护栏。
LINT_RULES_PATH: Path = ORCH_STATE_ROOT / "taskbook-lint.json"


def _ts() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


@dataclass
class DispatchOutcome:
    """一次派发（一跳）的结构化结果。"""

    code: int  # 0=成功交付；2=blocked/parse_error/run_failed/验收FAIL
    kind: str  # result | blocked | parse_error | run_failed
    member: str = ""
    sid: str = ""
    session_name: str = ""
    task_file: str = ""
    body: str = ""  # 交付正文（result/blocked 块内容）
    checks: list[dict] = field(default_factory=list)
    infra_suggestion: str = ""
    suggestion_id: str = ""
    envelope: Envelope | None = None
    error: str = ""


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


def _persist_suggestion(member_id: str, body: str) -> str:
    """把 infra_suggestion 持久化为待审批条目，返回建议 id（文件名 stem）。"""
    SUGGESTIONS_DIR.mkdir(parents=True, exist_ok=True)
    sid = f"{_ts()}-{member_id}"
    p = SUGGESTIONS_DIR / f"{sid}.md"
    p.write_text(f"# 改进建议（{member_id}，待审批）\n\n{body}\n", encoding="utf-8")
    return sid


def do_dispatch(
    member_id: str,
    *,
    text: str | None = None,
    task_file: str | None = None,
    slug: str | None = None,
    session: str = "latest",
    model_tier: str | None = None,
    max_turns: int | None = None,
    timeout: int | None = None,
    dirs: list[str] | None = None,
    plan: str = "",
    step: str = "",
    no_checks: bool = False,
    no_retry: bool = False,
    quiet: bool = False,
) -> DispatchOutcome:
    """执行一跳派发（完整链路）。参数语义与 ``orch dispatch`` 一致。

    Returns:
        DispatchOutcome（code=0 仅当 result 交付且声明的机械验收全过）。
    """
    reg = Registry.load()
    reg.ensure_member_defaults(member_id)
    m = reg.member(member_id)
    if not m.pod.exists():
        if not quiet:
            print(f"[!] 成员 pod 不存在：{orchestration_rel(m.pod)}")
            print("    请先建 pod 骨架（见 orchestration/README.md §12 阶段计划）。")
        return DispatchOutcome(
            code=2, kind="run_failed", member=member_id, error="pod 不存在"
        )
    model = reg.models.get(model_tier or m.model_tier, "")
    slug = slug or (Path(task_file).stem if task_file else f"adhoc-{_ts()}")

    # 会话选择（README §3.2：默认续最近活跃会话）
    if session == "new":
        sid, resume = new_session_id(), False
    elif session == "latest":
        active = m.active_sessions
        sid, resume = (active[0]["sid"], True) if active else (new_session_id(), False)
    else:
        found = m.find_session(session)
        if found is None:
            if not quiet:
                print(f"[!] 会话 '{session}' 未在池中找到。当前池：")
                print("\n".join(_sessions.format_pool(m)))
            return DispatchOutcome(
                code=2,
                kind="run_failed",
                member=member_id,
                error=f"会话 {session} 未找到",
            )
        sid, resume = found["sid"], True
    if not resume:
        _sessions.register_session(m, sid, slug)
    name = slug if not resume else (m.find_session(sid) or {}).get("name", slug)

    taskbook = write_taskbook(m.pod, text=text, task_file=task_file, slug=slug)
    replies_body, reply_files = collect_replies(member_id)
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
        m, prompt, session_id=sid, resume=resume, model=model, max_turns=max_turns
    )
    env = build_env(m, all_dirs, deployed_skills=deployed_skills_for(m.pod))

    if not quiet:
        print(
            f"→ 派发 {member_id}（会话 {'续:' + sid[:8] if resume else '新:' + sid[:8]}"
            f"，模型 {model or '(默认)'}，任务书 {taskbook.name}）"
        )
    rc, out, err = run_headless(
        cmd, cwd=PROJECT_ROOT, env=env, timeout_s=timeout or m.timeout_s
    )
    env_json = Envelope.parse(out)

    if (rc != 0 or env_json.is_error) and not no_retry:
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
        )
        rc, out, err = run_headless(
            cmd2, cwd=PROJECT_ROOT, env=env, timeout_s=timeout or m.timeout_s
        )
        env_json = Envelope.parse(out)

    if rc != 0 or env_json.is_error:
        append(
            LedgerEntry(
                ts="",
                member=member_id,
                session_id=sid,
                session_name=name,
                task_file=orchestration_rel(taskbook),
                kind="run_failed",
                num_turns=env_json.num_turns,
                duration_ms=env_json.duration_ms,
                credits=env_json.total_credits,
                model=model,
                ctx_ratio=env_json.context_usage_ratio,
                permission_denials=len(env_json.permission_denials),
                plan=plan,
                step=step,
                error=(err.strip()[-400:] or env_json.stop_reason),
            )
        )
        _sessions.touch_session(m, sid)
        reg.write_member(m)
        reg.save()
        outcome = DispatchOutcome(
            code=2,
            kind="run_failed",
            member=member_id,
            sid=sid,
            session_name=name,
            task_file=orchestration_rel(taskbook),
            body=env_json.result[:800],
            envelope=env_json,
            error=err.strip()[-400:] or env_json.stop_reason,
        )
        if not quiet:
            _report_failure(outcome)
        return outcome

    delivery = parse_delivery(env_json.result)
    check_results = (
        [] if no_checks else run_checks(delivery.artifacts, reg, member_pod=m.pod)
    )
    suggestion_id = ""
    if delivery.infra_suggestion.strip():
        suggestion_id = _persist_suggestion(member_id, delivery.infra_suggestion)
    entry = LedgerEntry(
        ts="",
        member=member_id,
        session_id=sid,
        session_name=name,
        task_file=orchestration_rel(taskbook),
        kind=delivery.kind,
        checks=check_results,
        infra_suggestion=bool(delivery.infra_suggestion),
        num_turns=env_json.num_turns,
        duration_ms=env_json.duration_ms,
        credits=env_json.total_credits,
        model=model,
        ctx_ratio=env_json.context_usage_ratio,
        permission_denials=len(env_json.permission_denials),
        plan=plan,
        step=step,
    )
    append(entry)
    for p in reply_files:  # 审批回复已附送 → 归档 sent/
        sent = p.parent / "sent"
        sent.mkdir(exist_ok=True)
        shutil.move(str(p), str(sent / p.name))
    _sessions.touch_session(m, sid)
    reg.write_member(m)
    reg.save()

    failed_checks = [c for c in check_results if c["verdict"] == "fail"]
    code = 0 if (delivery.kind == "result" and not failed_checks) else 2
    outcome = DispatchOutcome(
        code=code,
        kind=delivery.kind,
        member=member_id,
        sid=sid,
        session_name=name,
        task_file=orchestration_rel(taskbook),
        body=delivery.body,
        checks=check_results,
        infra_suggestion=delivery.infra_suggestion,
        suggestion_id=suggestion_id,
        envelope=env_json,
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
            f'  1) 再试一跳：uv run pysci-orch dispatch {o.member} --session {o.sid[:8]} --text "继续任务书 {o.task_file}"',
            "  2) 查会话原文定位卡点（registry/pod 键目录下 jsonl）",
            "  3) 仍失败 → 上报用户（附本输出全文）",
        ]
    )


def _report_success(o: DispatchOutcome, failed_checks: list[dict]) -> None:
    env = o.envelope
    icon = {"result": "√", "blocked": "✗", "parse_error": "?"}.get(o.kind, "?")
    dur = env.duration_ms / 1000 if env else 0
    turns = env.num_turns if env else 0
    ctx = env.context_usage_ratio if env else 0
    print(
        f"[{icon}] {o.member} 交付类型：{o.kind}（{turns} 轮，{dur:.0f}s，ctx={ctx:.0%}）"
    )
    print(f"    会话：{o.sid[:8]}（{o.session_name}）")
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
                f'改进建议待审批：uv run pysci-orch approve {o.suggestion_id} --note "..."'
            )
            nxt.append(
                f'                 或 uv run pysci-orch reject {o.suggestion_id} --note "..."'
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
                f'  uv run pysci-orch dispatch {o.member} --session {o.sid[:8]} --text "返工：机械验收未过（{detail}）。修正后重新交付。"',
                "同指纹连败 3 次 → 停止回派，升级决策（改任务书/咨询副组长/上报用户）。",
            ]
        )
        return
    if o.kind == "blocked":
        _print_next(
            [
                "组员被阻塞。可选：",
                f'  1) 补充信息重派：uv run pysci-orch dispatch {o.member} --session {o.sid[:8]} --text "<补充>"',
                '  2) 咨询副组长：uv run pysci-orch consult "<问题>"',
                "  3) 上报用户（附交付正文）",
            ]
        )
        return
    _print_next(
        [
            "交付格式解析失败（parse_error）。原回复见上方正文预览。",
            f'重派补格式：uv run pysci-orch dispatch {o.member} --session {o.sid[:8]} --text "上次交付缺 <result>/<blocked> 标签块，请按 charter 重新收尾（工作不必重做）。"',
        ]
    )
