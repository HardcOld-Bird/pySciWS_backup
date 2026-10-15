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
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pysci.paths import PROJECT_ROOT

from . import procs
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
    # 而是**依模型上报**——实测内置渠道（Qwen3.8-Flash，无连字符）恒 0、BYOK 渠道
    # （Qwen-3.8-Max，带连字符）正常计量；计量随**渠道**而非档位名（2026-10-12 flash 档改
    # BYOK 优先，见 backlog 20261012-173500-leader）。usage.input_tokens/modelUsage.contextWindow
    # 在两渠道均 0，唯 context_usage_ratio 恒有效。
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


class SeatShapeError(ValueError):
    """席位/会话/配置路径**不是字符串**——配置形状与在跑代码的读法不匹配时止血。

    事故原型（backlog 20261013-003000-leader，2026-10-13 00:25 实测）：21:14 派生的
    drain 进程内存里是**旧代码**（``Registry.models`` 裸合并跟踪面），00:22 并入 main 的
    新跟踪面是**链形态**（数组）。旧读法把整条链当一个席位取走，list 进 argv 后由
    ``subprocess`` 抛 ``expected str, bytes or os.PathLike object, not list``——这条消息
    不说是哪个旗标、也不说值从哪来，排查只能靠 git blame 猜。长生命周期 worker 撞上配置
    迁移是**一类**竞态（进程数小时不改代码，磁盘随时在换形状），故在派生**之前**拦。
    """


def _argv_value(flag: str, value: object) -> str:
    """进 argv 的取值闸：只放行字符串，其余带着**旗标名与实况值**报错。

    Args:
        flag: 该值所属的命令行旗标（写进消息，定位用）。
        value: 待进 argv 的值。

    Returns:
        原样字符串（不做 ``str()`` 兜底——静默串化会把 list 变成 ``"['a', 'b']"``
        这种「启动即失败」的垃圾席位，比报错更难查）。

    Raises:
        SeatShapeError: 值不是字符串。
    """
    if isinstance(value, str):
        return value
    raise SeatShapeError(
        f"{flag} 收到 {type(value).__name__} 而非席位串：{value!r}"
        "——跟踪面形状与本进程代码读法不匹配（长生命周期 worker 撞上配置迁移即此类）；"
        "处置：重启该 worker 载入新代码，或把该值写成字符串/取链首一名"
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
        subprocess 可用的 argv 列表（每个元素都是字符串，见 :func:`_argv_value`）。

    Raises:
        SeatShapeError: 席位/会话/配置路径不是字符串——在派生**之前**带着旗标名报错。
    """
    del timeout_note  # 保留参数位，暂无用途
    exe = exe or resolve_exe()
    sid = _argv_value("--resume" if resume else "--session-id", session_id)
    cmd = [str(exe), "--cwd", str(member.pod), "-p", prompt, "-o", "json"]
    cmd += ["--resume", sid] if resume else ["--session-id", sid]
    cmd += ["--max-turns", str(max_turns or member.max_turns)]
    if model:
        cmd += ["-m", _argv_value("-m", model)]
    if effort:
        cmd += ["--reasoning-effort", _argv_value("--reasoning-effort", effort)]
    if member.mcp_config:
        # pod 相对路径：headless 以 pod 为 cwd，相对路径即 pod 内文件（Phase 0 实锤）
        cfg = _argv_value("--mcp-config", member.mcp_config)
        cmd += ["--mcp-config", cfg, "--strict-mcp-config"]
    return cmd


def _default_venv_scripts(root: Path) -> Path:
    """某树 venv 的命令目录：Windows ``Scripts``，POSIX ``bin``（都不存在时返 ``Scripts``）。"""
    win = root / ".venv" / "Scripts"
    return win if win.exists() else root / ".venv" / "bin"


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
    # PYSCI_VENV_SCRIPTS 是 E4 副本 A/B 的旋钮：测量副本树**没有** .venv（gitignored），
    # 不覆盖就会让副本里的组员裸命令全灭、退化成 `uv run` 就地新建空 venv（已知坑）。
    # 脚本来自被指定的 venv、代码由 PYTHONPATH 决定——「脚本来自主 venv、代码来自副本」。
    scripts_dir = os.environ.get("PYSCI_VENV_SCRIPTS") or str(
        _default_venv_scripts(PROJECT_ROOT)
    )
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
    on_spawn: Callable[[int], None] | None = None,
) -> tuple[int, str, str]:
    """同步执行 headless 进程并捕获输出。

    orch 自身阻塞等待（进程级等待，零 token）；组长按 README §4.5 以**后台 Bash**
    方式运行 orch，长任务完成时由 Qoder 事件通知唤醒。

    Args:
        cmd: build_command 产出的 argv。
        cwd: 进程工作目录（orch 调用方项目根；成员 cwd 经 --cwd 指定）。
        env: build_env 产出的环境。
        timeout_s: 墙钟超时秒数。
        on_spawn: 进程**一落地就**调用，参数为子进程 OS pid。为什么要有它：worker 若在跳
            中途被杀，台账里那一行永远不会写，可它派生的会话进程还活着继续写 worktree，
            下一次 drain 重派同名 worktree 就并发双写（backlog 20261012-184000-leader）。
            回收者需要的 pid 只能来自派生瞬间的登记，故在此留钩子；回调自身的
            fail-open 由调用方负责（记账面不得影响派发结果）。

    Returns:
        (returncode, stdout, stderr)；超时时 returncode=-1、stderr 含超时说明。
    """
    # Popen 而非 subprocess.run：后者不暴露子进程 pid，而 on_spawn 要的恰是它（超时语义
    # 在此手工复刻——kill + 二次 communicate，与 subprocess.run 的超时路径一致）。
    # 窗口隐藏走唯一口径 ``procs.popen_hidden``（win32 恒 CREATE_NO_WINDOW；本函数常由
    # **分离进程** drain worker 派起组员 headless 会话，无控制台时裸 spawn 会闪空终端）。
    proc = procs.popen_hidden(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if on_spawn is not None:
        on_spawn(proc.pid)
    try:
        out, err = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        # 超时**杀进程树**而非只杀直接子进程（2026-10-15 worker-stall 单）：headless CLI
        # 自己还会派生子进程（工具调用、构建子壳），``proc.kill()`` 只终结它一个，孙进程
        # 继续占着 worktree 写盘 → 下一次回收重派同名的隔离面就并发双写（procs 模块
        # docstring 记的同一族事故）。Windows 走 ``taskkill /T /F``，POSIX 走 SIGTERM
        # 并对好给对手收尾；无论哪条路径随后都补一次 ``proc.kill()`` 兜底（直系子进程
        # 若忽略 SIGTERM，communicate 会一直等到它退出——超时路径不能挂住 worker）。
        procs.terminate_tree(proc.pid)
        try:
            proc.kill()
        except OSError:
            pass
        out, _ = proc.communicate()
        return -1, out or "", f"TIMEOUT after {timeout_s}s（已杀进程树 pid={proc.pid}）"
    return proc.returncode, out or "", err or ""


def new_session_id() -> str:
    """生成新会话 uuid（与 Qoder ``--session-id`` 约定一致）。"""
    return str(uuid.uuid4())
