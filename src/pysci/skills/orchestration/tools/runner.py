"""headless 组员进程的命令组装、执行与 envelope 解析。

调用契约（README §3.3 / 附录 A）：原生 exe 直调（`.cmd` 包装器剥引号）、pod 为 cwd、
``-o json`` 单行 envelope、会话池经 ``--resume``/``--session-id`` + ``--name``。
权限依赖用户级 yolo 默认（Phase 0 实锤），路径纪律由 pod-guard hook 强制。
"""

from __future__ import annotations

import json
import os
import subprocess
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pysci.paths import PROJECT_ROOT

from .registry import Member, resolve_exe

#: 注入到每次派发的交付协议提醒（charter 有全文，这里只防健忘）。
PROTOCOL_REMINDER = """
--- 交付协议提醒（charter 有全文）---
最终回复必须含且仅含一个 <result>...</result> 或 <blocked>...</blocked> 标签块；
可选 <infra_suggestion>...</infra_suggestion>（缺陷类附证据，难用类描述场景即可）。
产物用 <artifact check="类型|none" reason="...">路径</artifact> 声明机械验收。
交付前把任务书要求的路径全部落盘；Stop hook 会校验格式，不合规会被退回改正。
""".strip()


#: 推理强度档位（原生 exe 接受的**实测**集合，2026-10-10：对每个值跑
#: ``qoderclicn --reasoning-effort <v> --list-models`` 看退出码——auto/none/low/medium/
#: high/xhigh/max/ultracode 与别名 off/disabled 均 rc=0，``min`` 等非法值 rc=1
#: 「Invalid reasoning effort: … Valid values are: auto, none, low, medium, high,
#: xhigh, max, ultracode」；大小写不敏感）。
#: **不传本旗标 = 跟随用户级默认（中）**——用户裁决 2026-10-10：不做全局 high，只由组长
#: 对推理密集项**按项**标 high（旗标/计划环节/backlog 条目/registry 成员四层来源）。
VALID_EFFORTS: tuple[str, ...] = (
    "auto",
    "none",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
    "ultracode",
)
EFFORT_ALIASES: dict[str, str] = {"off": "none", "disabled": "none"}


def normalize_effort(value: object) -> str | None:
    """把 effort 标注规范成 CLI 档位名。

    Args:
        value: 任意标注串（CLI 参数、registry/计划/backlog 字段）；``None``/空白 = 不覆盖。

    Returns:
        规范档位名；空标注返回 ``None``（意为跟随用户级默认，不透传旗标）。

    Raises:
        ValueError: 标注不在 :data:`VALID_EFFORTS` 内。exe 对非法值本身会**启动即失败**
            （rc=1），但在我们这层先拦住才能给出可读的 ``[!]`` 提示、不烧跳、不往台账
            里塞一条无意义记录（registry/条目来源的值可不经 argparse choices 直接到达）。
    """
    text = str(value or "").strip().lower()
    if not text:
        return None
    text = EFFORT_ALIASES.get(text, text)
    if text not in VALID_EFFORTS:
        raise ValueError(
            f"无效 effort '{value}'；可用：{', '.join(VALID_EFFORTS)}"
            "（别名 off/disabled→none；留空=跟随用户级默认）"
        )
    return text


@dataclass
class Envelope:
    """``-o json`` 结果 envelope 的感兴趣字段（Phase 0 实测字段清单）。"""

    result: str = ""
    session_id: str = ""
    num_turns: int = 0
    is_error: bool = False
    stop_reason: str = ""
    duration_ms: int = 0
    # ADR（2026-10-10，backlog 20261009-byok-cost-accounting）：BYOK 下 total_credits 并非恒 0，
    # 而是**依模型上报**——实测 max 档（Qwen3.8-Max）正常计量、flash 档（Qwen3.8-Flash）恒 0；
    # usage.input_tokens/modelUsage.contextWindow 在两档均 0，唯 context_usage_ratio 恒有效。
    # 用户裁决（2026-10-09）完整计费留待未来，故此处只忠实透传 envelope 原值，不做换算/估算；
    # 成本可解释性由 ledger 记录 model + format_credits 标注覆盖率解决（见 ledger.py ADR）。
    total_credits: float = 0.0
    context_usage_ratio: float = 0.0
    permission_denials: list[Any] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def parse(cls, stdout: str) -> Envelope:
        """从 stdout 解析 envelope JSON（容忍前后杂讯：取最后一个平衡 JSON 对象）。

        Args:
            stdout: headless 进程的标准输出全文。

        Returns:
            Envelope 实例；解析失败时 ``is_error=True`` 且 ``result`` 存原始输出。
        """
        text = stdout.strip()
        data: dict[str, Any] | None = None
        # 优先整体解析；失败则从最后一个 '{' 起尝试（envelope 是最后输出物的约定）
        for candidate in (
            text,
            text[text.rfind("\n{") + 1 :] if "\n{" in text else None,
        ):
            if not candidate:
                continue
            try:
                data = json.loads(candidate)
                break
            except json.JSONDecodeError:
                continue
        if data is None:
            return cls(result=text, is_error=True, stop_reason="envelope_parse_failed")
        usage = data.get("usage") or {}
        return cls(
            result=str(data.get("result", "")),
            session_id=str(data.get("session_id", "")),
            num_turns=int(data.get("num_turns", 0)),
            is_error=bool(data.get("is_error", False)),
            stop_reason=str(data.get("stop_reason", "")),
            duration_ms=int(data.get("duration_ms", 0)),
            total_credits=float(data.get("total_credits", 0) or 0),
            context_usage_ratio=float(usage.get("context_usage_ratio", 0) or 0),
            permission_denials=list(data.get("permission_denials", [])),
            raw=data,
        )


def build_command(
    member: Member,
    prompt: str,
    *,
    session_id: str,
    resume: bool,
    model: str,
    max_turns: int | None = None,
    effort: str | None = None,
    timeout_note: bool = True,
    exe: Path | None = None,
) -> list[str]:
    """组装一次 headless 派发的完整 argv。

    Args:
        member: 成员条目（pod/mcp_config 来源）。
        prompt: 注入文本（任务书路径 + 审批回复 + 协议提醒）。
        session_id: 会话 id（resume=True 时为既有 sid，否则为新 sid）。
        resume: True 用 ``--resume``，False 用 ``--session-id``。
        model: 具体模型名（已经过档位路由）。
        max_turns: 覆盖成员默认轮数上限。
        effort: 推理强度档位（**已规范**的档位名，见 :func:`normalize_effort`）；
            空/None 时不透传 ``--reasoning-effort``，跟随用户级默认（中）。
        timeout_note: 占位（保持签名稳定），未使用。
        exe: 覆盖注册表/默认 exe。

    Returns:
        subprocess 可用的 argv 列表。
    """
    del timeout_note  # 保留参数位，暂无用途
    exe = exe or resolve_exe()
    cmd = [str(exe), "--cwd", str(member.pod), "-p", prompt, "-o", "json"]
    cmd += ["--resume", session_id] if resume else ["--session-id", session_id]
    cmd += ["--max-turns", str(max_turns or member.max_turns)]
    if model:
        cmd += ["-m", model]
    if effort:
        cmd += ["--reasoning-effort", effort]
    if member.mcp_config:
        # pod 相对路径：headless 以 pod 为 cwd，相对路径即 pod 内文件（Phase 0 实锤）
        cmd += ["--mcp-config", member.mcp_config, "--strict-mcp-config"]
    return cmd


def build_env(
    member: Member,
    task_dirs: list[str],
    deployed_skills: list[str] | None = None,
    readonly_extra: list[str] | None = None,
) -> dict[str, str]:
    """构造派发进程的环境：guard 白名单注入 + UTF-8 纪律。

    Args:
        member: 成员条目（读取 pod 路径供只读层清单推导）。
        task_dirs: 本任务额外放行的项目内目录（相对项目根或绝对路径）。
        deployed_skills: 部署副本技能名清单（pod-guard 保护其不被成员改写；
            由 orch 门面从 manifest 解析后传入）。
        readonly_extra: 调用方追加的成员级只读条目；registry 成员配置中的同名
            字段（如 reviewer 的 ``rubrics``）也会自动并入。

    Returns:
        完整的子进程环境变量字典。
    """
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    # 组员裸 pysci-X 命令可用性不依赖组长会话环境快照的时效：显式注入 venv Scripts
    # （旧会话快照可能早于 ~/.bashrc 的 PATH 追加；charter 的 uv run fallback 仍保留兜底）
    scripts_dir = str(PROJECT_ROOT / ".venv" / "Scripts")
    env["PATH"] = scripts_dir + os.pathsep + env.get("PATH", "")
    env["PYSCI_TASK_DIRS"] = ";".join(
        task_dirs
    )  # 固定分号：Windows 路径含冒号，不能用 os.pathsep
    # 只读层清单（pod-guard 消费；README §2 矩阵的组员列）
    readonly = [
        ".qoder/rules/charter.md",
        ".qoder/settings.json",
        ".qoder/mcp.json",
        ".qoder/hooks",
        "inbox",
    ]
    readonly += list(readonly_extra or [])
    readonly += [str(r) for r in member.raw.get("readonly_extra", [])]
    env["PYSCI_READONLY"] = ";".join(readonly)
    env["PYSCI_DEPLOYED_SKILLS"] = ";".join(deployed_skills or [])
    env["PYSCI_POD"] = str(member.pod)
    env["PYSCI_PROJECT_ROOT"] = str(PROJECT_ROOT)
    return env


def run_headless(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    timeout_s: int,
) -> tuple[int, str, str]:
    """同步执行 headless 进程并捕获输出。

    orch 自身阻塞等待（进程级等待，零 token）；组长按 README §4.5 以**后台 Bash**
    方式运行 orch，长任务完成时由 Qoder 事件通知唤醒。

    Args:
        cmd: build_command 产出的 argv。
        cwd: 进程工作目录（orch 调用方项目根；成员 cwd 经 --cwd 指定）。
        env: build_env 产出的环境。
        timeout_s: 墙钟超时秒数。

    Returns:
        (returncode, stdout, stderr)；超时时 returncode=-1、stderr 含超时说明。
    """
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
        )
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as exc:
        partial = (
            (exc.stdout or b"").decode("utf-8", errors="replace")
            if isinstance(exc.stdout, bytes)
            else (exc.stdout or "")
        )
        return -1, partial, f"TIMEOUT after {timeout_s}s"


def new_session_id() -> str:
    """生成新会话 uuid（与 Qoder ``--session-id`` 约定一致）。"""
    return str(uuid.uuid4())
