"""夜航模式（PSEP_2）政策注入层：SessionStart / PostCompact 钩子的**判定与载荷单点**。

用户离机时段需无人值守推进（夜航）：总开关是 ``data/pysci_settings.toml`` 的
``[night] enabled``（手动一行、**每回合现读热切换**）；本模块只做**政策注入层**——
开关 ON 时产出 ``hookSpecificOutput.additionalContext`` 载荷（政策全文＝用户签发的
限时授权书），由 ``.qoder/hooks/night-policy.sh`` 在 SessionStart / PostCompact 两个
触发点注出；OFF 零注入零噪声。**判据层**（AskUserQuestion 拦截 / queue_pressure 夜航
判据）是 n2，不在本模块。

为什么挂 SessionStart + PostCompact 而非每回合 Stop（PSEP_2 §3.2）：逐回合注入烧
token；而上下文压缩正是政策**最易丢**的时刻——PostCompact 补一针即覆盖真风险。

**fail-open**：任何异常都零输出、exit 0——本钩子挂在会话初始化/压缩链上，绝不能塌
任何主链路（watch-arm.sh 的 token 竞争 / wake-claim 均不经过本模块）。
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path
from typing import Any

from .settings import night_enabled

#: 政策全文（用户授权书；PSEP_2 §3.2 草案逐字）。四要素齐全：
#: ① 持续推进不等用户；② 一般决策自行执行且事后追溯责任在用户；③ 高风险咨询 twin 且
#: twin 决策权等同用户（leader-only.md §3 限时预授权；不可达降级副组长+自行决断+留痕）；
#: ④ 终态仅「全部完成 / 严重阻碍」+ 禁把 LLM 当定时器 + 权限弹窗属硬阻塞。
NIGHT_POLICY: str = (
    "**【夜航模式生效】** 用户已离机并签发以下授权（本授权随 [night] enabled=true "
    "有效，关闭即失效）：① 尽可能持续推进所有事项/大任务，不得因等待用户回复而停滞或"
    "结束回合观望；② 一般决策你自行执行，事后追溯责任在用户；③ 需用户口头授权的高风险"
    "事项（含护栏级/规则级裁决）**咨询 twin 并依其裁决执行**——twin 的决策权等同用户"
    "（本条为 leader-only.md §3 的限时预授权）；twin 不可达时降级为「副组长咨询 + 自行"
    "决断 + 留痕」，仍不等用户；④ 终态仅两种：**全部完成**或**不得不的严重阻碍**——"
    "达成即写一段收尾总结、将 enabled 改回 false、结束推进；⑤ 禁止把 LLM 当定时器"
    "（空转打回）；⑥ 工具权限弹窗属硬阻塞，用户离机前应自调权限档（见 §3.6）。"
)


def policy_hook_output(
    event_name: str = "SessionStart", *, data: dict[str, Any] | None = None
) -> dict[str, Any] | None:
    """开关 ON → 钩子输出载荷（``hookSpecificOutput.additionalContext`` = 政策全文）；
    OFF → ``None``（零注入）。

    ``hookEventName`` 回填触发事件名（SessionStart / PostCompact）——同一政策两处注入，
    载荷形状一致，便于测试与后续 n2 复用。
    """
    on, _warnings = night_enabled(data)
    if not on:
        return None
    return {
        "hookSpecificOutput": {
            "hookEventName": str(event_name or "SessionStart"),
            "additionalContext": NIGHT_POLICY,
        }
    }


def _read_settings_file(path: Path) -> dict[str, Any]:
    """按显式路径读配置（钩子传 ``--settings``）；读不出返回 ``{}``（fail-open 上游兜）。"""
    try:
        with path.open("rb") as fh:
            data = tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def main(argv: list[str] | None = None) -> int:
    """钩子入口：stdin 载荷取 ``hook_event_name`` → 开关 ON 时打印 JSON；OFF/异常一律 exit 0。

    ``--settings PATH``：显式配置文件路径（钩子按 ``$QODER_PROJECT_DIR`` 传入；
    测试隔离面用同一参数）。载荷解析失败按 SessionStart 名义注入（两触发点政策同一份，
    名义差异无行为影响）；任何异常零输出退出。
    """
    parser = argparse.ArgumentParser(prog="night-policy")
    parser.add_argument(
        "--settings", default="", help="配置文件路径（缺省走 paths 默认）"
    )
    args = parser.parse_args(argv)
    try:
        raw_payload = sys.stdin.read()
    except Exception:  # noqa: BLE001 - stdin 不可读 = 无触发名，按缺省
        raw_payload = ""
    event = "SessionStart"
    try:
        parsed = json.loads(raw_payload)
        name = parsed.get("hook_event_name")
        event = (
            name.strip() if isinstance(name, str) and name.strip() else "SessionStart"
        )
    except (json.JSONDecodeError, TypeError, AttributeError):
        pass
    try:
        data = _read_settings_file(Path(args.settings)) if args.settings else None
        out = policy_hook_output(event, data=data)
    except Exception:  # noqa: BLE001 - fail-open：任何异常零输出 exit 0
        return 0
    if out:
        print(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
