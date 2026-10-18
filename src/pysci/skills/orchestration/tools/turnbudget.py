"""会话轮次余量告警（F1，tb-20261017-212141-3-3 L.l3）。

背景：生产单 60 轮耗尽即 ``fatal_error`` 断线、零交付（组长翻盘半成品才接力）。
旧通道只在**跳结束后**才看得到 ``num_turns``，那时已来不及告警，故本模块走 CLI 的
``PostToolUse`` hook：每一工具回合结束都会执行，退出码 2 的 stderr 会被 CLI 注入会话
（实测 1.1.73），模型在**跳进行中**即可看到收尾提示。

触发口径（起点可调）：

- 阈值 = ``ceil(budget × ratio)``；``ratio`` 默认 :data:`WARN_RATIO`（0.8），经环境变量
  ``PYSCI_TURN_WARN_RATIO`` 或 hook 参数 ``--ratio`` 覆盖（取值 ``(0, 1]``，非法回退）。
- ``budget`` = 本跳 ``--max-turns``（覆盖值或成员缺省），由 :func:`hook_settings` 写进
  hook 命令。
- ``--since`` = orch 组装命令的 epoch 秒：只数此后 transcript 里的 assistant 回合
  （按 ``message.id`` 去重——同一回合的 thinking 与 tool_use 是两行），与 envelope 的
  ``num_turns``（按次调用计）同口径，resume 会话的历史轮次不会把小预算误判成贴线。
- 每跳只告警一次（标记文件按 session+since 落系统临时目录，``O_EXCL`` 原子创建）：
  80% 之后每个工具回合都退回同一句提示会把预算烧在重复告警上。

文案标定见 :func:`warning_text`：告警必须落到「交付半成品＋接手清单」的可执行动作，
只报数字的提示会让模型继续硬扛。hook 自身任何异常一律 fail-open（exit 0）——门禁故障
不得阻塞生产会话。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

#: 余量告警起点：达本跳轮次上限的 80% 即提示收尾（可经 ``PYSCI_TURN_WARN_RATIO`` 覆盖）。
WARN_RATIO = 0.8

#: 告警正文（任务书 L.l3 F1 定稿措辞；括号内补读数，便于成员判断离上限还有多远）。
WARNING_TEXT = (
    "余量告警：轮次即将耗尽（已用 {turns}/{budget} 轮，告警阈值 {threshold}）。"
    "请立即收尾：交付半成品＋接手清单（产物路径/已完成/待办），勿硬扛。"
)


def warn_ratio() -> float:
    """返回生效的告警起点比例。

    Returns:
        ``(0, 1]`` 内的比例；``PYSCI_TURN_WARN_RATIO`` 缺失或非法时回退 :data:`WARN_RATIO`。
    """
    raw = os.environ.get("PYSCI_TURN_WARN_RATIO", "").strip()
    if not raw:
        return WARN_RATIO
    try:
        value = float(raw)
    except ValueError:
        return WARN_RATIO
    return value if 0 < value <= 1 else WARN_RATIO


def warning_threshold(budget: int, ratio: float | None = None) -> int:
    """计算本轮告警的轮次阈值。

    Args:
        budget: 本跳轮次上限（``--max-turns``）。
        ratio: 告警起点比例；``None`` = 读环境/缺省（:func:`warn_ratio`）。

    Returns:
        ``ceil(budget × ratio)``，至少 1（小预算也保证有告警点）。
    """
    effective = warn_ratio() if ratio is None else ratio
    if not 0 < effective <= 1:
        effective = WARN_RATIO
    return max(1, math.ceil(int(budget) * effective))


def _parse_timestamp(text: str) -> float:
    """把 transcript 的 ISO 时间戳（UTC ``Z`` 后缀）转成 epoch 秒。"""
    stamp = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=UTC)  # 无时区按 UTC（transcript 惯例）
    return stamp.timestamp()


def count_turns(transcript: Path, since: float | None = None) -> int:
    """数 transcript 里的 assistant 回合数（按 ``message.id`` 去重）。

    Args:
        transcript: 会话 jsonl 路径。
        since: 只数此 epoch 秒之后的回合（``None`` = 全量）。resume 场景必须传，否则
            历史轮次会把本跳的小预算误判为贴线（CLI 的 ``--max-turns`` 按次调用计）。

    Returns:
        回合数。单行解析失败跳过（半行/并发写不阻断计数）。
    """
    seen: set[str] = set()
    with transcript.open(encoding="utf-8", errors="replace") as fh:
        for idx, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("type") != "assistant":
                continue
            if record.get("isSidechain"):
                continue  # 子代理回合与 CLI 的 num_turns（主循环）不同口径，别提前误报
            if since is not None:
                stamp = record.get("timestamp")
                if stamp:
                    try:
                        if _parse_timestamp(str(stamp)) < since:
                            continue
                    except ValueError:
                        pass
            message = record.get("message") or {}
            key = message.get("id") or record.get("uuid") or f"line-{idx}"
            seen.add(str(key))
    return len(seen)


def warning_text(turns: int, budget: int, threshold: int) -> str:
    """渲染告警正文（阈值/读数可核对；核心动作句固定不变）。"""
    return WARNING_TEXT.format(turns=turns, budget=budget, threshold=threshold)


def _marker_path(session_id: str, since: float | None) -> Path:
    """本跳告警的一次性标记：session + since 唯一（每次 build_command 一个 since）。"""
    token = f"{session_id}-{(since or 0.0):.3f}".replace(":", "_")
    return Path(tempfile.gettempdir()) / "pysci-turnbudget" / f"{token}.warned"


def _cli(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--ratio", type=float, default=None)
    parser.add_argument("--since", type=float, default=None)
    args = parser.parse_args(list(argv))

    try:
        payload = json.loads(sys.stdin.read() or "{}")
    except json.JSONDecodeError:
        return 0
    session_id = str(payload.get("session_id") or "")
    transcript = payload.get("transcript_path")
    if not session_id or not transcript:
        return 0
    if args.budget <= 0:
        return 0

    threshold = warning_threshold(args.budget, args.ratio)
    try:
        turns = count_turns(Path(str(transcript)), args.since)
    except OSError:
        return 0
    if turns < threshold:
        return 0

    marker = _marker_path(session_id, args.since)
    try:
        marker.parent.mkdir(parents=True, exist_ok=True)
        with marker.open("x", encoding="utf-8"):
            pass
    except FileExistsError:
        return 0  # 本跳已告警过：别在 80% 之后每回合重复烧提示
    except OSError:
        return 0
    print(warning_text(turns, args.budget, threshold), file=sys.stderr)
    return 2


def main(argv: Sequence[str] | None = None) -> int:
    """hook 入口：返回 2 表示把告警注入会话（CLI 语义），其余一律放行。

    Args:
        argv: hook 参数（缺省读 ``sys.argv[1:]``）。

    Returns:
        ``2``（注入告警）或 ``0``（静默）；内部异常一律 fail-open 为 ``0``。
    """
    try:
        return _cli(sys.argv[1:] if argv is None else argv)
    except Exception:  # noqa: BLE001 —— 门禁故障不得阻塞生产会话
        return 0


def _hook_exe() -> Path:
    """hook 解释器路径：win32 优先同目录 ``pythonw.exe``（GUI 子系统，不分配控制台）。

    派发会话的 CLI 自身由 ``CREATE_NO_WINDOW`` 拉起（有自己的**不可见**控制台），但
    ``PostToolUse`` 钩子仍可能在无控制台父进程下执行（GUI/DETACHED 形态）——裸 spawn
    ``python.exe`` 会新开可见控制台。pythonw 下 stdin/stdout/stderr 由 CLI 提供管道，
    exit 2 + stderr 注入语义实测不变；非 win32 原样用 ``sys.executable``。
    """
    exe = Path(sys.executable).resolve()
    if sys.platform == "win32":
        candidate = exe.with_name("pythonw.exe")
        if candidate.is_file():
            return candidate
    return exe


def hook_settings(
    budget: int, *, ratio: float | None = None, since: float | None = None
) -> str:
    """组装注入 CLI 的 ``--settings`` JSON（``PostToolUse`` → 本模块）。

    Args:
        budget: 本跳轮次上限。
        ratio: 告警起点比例；``None`` = 读环境/缺省（:func:`warn_ratio`）。
        since: 本跳起算 epoch 秒；``None`` = 全量计数（仅供手工调试）。

    Returns:
        单行 JSON 字符串（``--settings`` 接受内联 JSON；与 pod 既有 settings 合并而非替换，
        实测 1.1.73：项目级 Stop hook 与内联 PostToolUse 同时生效）。
    """
    exe = _hook_exe().as_posix()
    script = Path(__file__).resolve().as_posix()
    effective = warn_ratio() if ratio is None else ratio
    command = f'"{exe}" "{script}" --budget {int(budget)} --ratio {effective:g}'
    if since is not None:
        command += f" --since {since:.3f}"
    settings = {
        "hooks": {
            "PostToolUse": [
                {
                    "hooks": [
                        {
                            "type": "command",
                            "command": command,
                            "name": "turn-budget",
                            "timeout": 30,
                        }
                    ]
                }
            ]
        }
    }
    return json.dumps(settings, ensure_ascii=False)


if __name__ == "__main__":
    raise SystemExit(main())
