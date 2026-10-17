"""夜航模式（PSEP_2）政策注入层：SessionStart / PostCompact 钩子的**判定与载荷单点**。

用户离机时段需无人值守推进（夜航）：总开关是 ``data/pysci_settings.toml`` 的
``[night] enabled``（手动一行、**每回合现读热切换**）；本模块只做**政策注入层**——
开关 ON 时产出 ``hookSpecificOutput.additionalContext`` 载荷（政策全文＝用户签发的
限时授权书），由 ``.qoder/hooks/night-policy.sh`` 在 SessionStart / PostCompact 两个
触发点注出；OFF 零注入零噪声。**判据层**（AskUserQuestion 拦截 / queue_pressure 夜航
判据）是 n2，不在本模块。

**回合边界差分（热生效单 2026-10-15，用户裁决）**：仅 SessionStart/PostCompact 注入时，
开关的**中途跳变**要等到下一次压缩才被看见——用户关掉夜航后，会话可能仍按授权书行事。
故新增 ``--diff`` 口：watch-arm.sh（Stop 钩子）每回合边界调用本模块，与**本会话上次读数**
（``watch.json`` 的 :data:`NIGHT_SEEN_KEY`）比对，**有跳变才**产出差分文本（关→开＝授权书
全文；开→关＝**零注入**），由该钩子走既有的 rewake 通道（stderr + exit 2）在**下一轮**注入；
无变化零输出、零打扰。开关**只归用户手动**：本模块不实现自拨回、不实现时限到期。

**基线落点**（2026-10-15 缺陷修补）：SessionStart 注入时**顺手落基线**（:func:`record_baseline`）
——只靠首个 Stop 补记的话，「注入后、首个 Stop 前」的拨动会被当基线吞掉。PostCompact
**不**落（只补注入、不动读数）。

为什么挂 SessionStart + PostCompact 而非每回合 Stop（PSEP_2 §3.2）：逐回合注入烧
token；而上下文压缩正是政策**最易丢**的时刻——PostCompact 补一针即覆盖真风险。
（差分口不是「每回合注入」：只在跳变的那一轮出声。）

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
#:
#: ④ 于 2026-10-15（用户裁决）改写：**开关只归用户手动**——LLM 永不改写 `enabled`
#: （不实现自拨回、不实现时限到期），达成终态写收尾总结即结束推进。
NIGHT_POLICY: str = (
    "**【夜航模式生效】** 用户已离机并签发以下授权（本授权随 [night] enabled=true "
    "有效，关闭即失效）：① 尽可能持续推进所有事项/大任务，不得因等待用户回复而停滞或"
    "结束回合观望；② 一般决策你自行执行，事后追溯责任在用户；③ 需用户口头授权的高风险"
    "事项（含护栏级/规则级裁决）**咨询 twin 并依其裁决执行**——`pysci-orch twin consult "
    '<案卷.md>`（twin 终裁）、`pysci-orch deputy consult "<问题>"`（副组长）；twin 的决策权等同'
    "用户（本条为 leader-only.md §3 的限时预授权）；twin 不可达时降级为「副组长咨询 + 自行"
    "决断 + 留痕」，仍不等用户；④ 终态仅两种：**全部完成**或**不得不的严重阻碍**——"
    "达成即写一段收尾总结（进展/遗留/关键决策点）**即结束推进**；开关只归用户手动拨动，"
    "**你永不改写 `enabled`**（不做自拨回、不设时限）；⑤ 禁止把 LLM 当定时器：不用空回合/"
    "轮询充当定时器，等待一律走事件机制（watch 自动挂载，命中即唤醒）；⑥ 工具权限弹窗属"
    "硬阻塞——它不写钩子接管，用户离机前应把会话权限调到宽松档（把常用工具设为允许，"
    "PSEP_2 §3.6）。"
)


#: 拦截（deny）时注入给模型的转 twin 指令（PSEP_2 §3.3a 文案，task-20261014-psep2-n2）。
NIGHT_GATE_DENY: str = (
    "夜航模式：用户已离机（[night] enabled=true），提问工具不可用——该决策转 twin 终裁："
    "`pysci-orch twin consult <案卷.md>`，依其裁决执行；twin 不可达时降级「副组长咨询 + 自行"
    "决断 + 留痕」，仍不等用户。"
)


#: 关→开差分 = 与 SessionStart **同一份**授权书全文（同一政策只有一份真源，不另写）。
#: **开→关差分 = 零注入**（用户裁决 2026-10-17）：关闭状态**不注入任何提示词**——旧的
#: 关闭态文案（曾写「恢复交互/决策权回到用户/该问用户的照常问」等）实测反复暗示「推进前
#: 等用户」，与「本项目始终鼓励自主推进」相悖；而授权回收已自助（NIGHT_POLICY 原文自带
#: 「关闭即失效」）、AskUserQuestion 拦截按开关每回合现读（OFF 自动放行）——行为面已随
#: 开关恢复，无需文字告知。故该方向返回 ``None``（与「无变化」同路）。

#: watch.json 里的**会话级上次读数**表（``{session_id: "on"|"off"}``）。
#: 为什么寄存在 watch 面：① 差分由 watch-arm（Stop 钩子）触发，同一份文件当回合已被读写
#: ——零新增 I/O 与零新增文件；② watch.json 是 gitignored 运行态，不污染跟踪面（本项目
#: 对运行态文件的既有纪律）；③ 它跨 watch.json 的**整体覆写**由 ``watch.CARRY_KEYS`` 兜住。
NIGHT_SEEN_KEY: str = "night_seen"

#: 上表条目上限（按插入序淘汰最旧）：防会话池长期演进后无界增长。
NIGHT_SEEN_MAX: int = 64


def _seen_write(session_id: str, value: str) -> None:
    """记录该会话的上次读数（fail-open：写不动就算了，下次当「首次见」处理）。"""
    try:
        from . import watch as _watch  # 延迟导入：watch 链重，且避免模块级环

        st = _watch.load_state()
        seen = st.get(NIGHT_SEEN_KEY)
        seen = dict(seen) if isinstance(seen, dict) else {}
        seen[session_id] = value
        if len(seen) > NIGHT_SEEN_MAX:
            for k in list(seen)[: len(seen) - NIGHT_SEEN_MAX]:
                seen.pop(k, None)
        st[NIGHT_SEEN_KEY] = seen
        _watch.save_state(st)
    except Exception:  # noqa: BLE001 - 记账面绝不反噬（差分本身照常产出）
        pass


def diff_message(session_id: str, *, data: dict[str, Any] | None = None) -> str | None:
    """回合边界差分：开关与**本会话上次读数**不同 → 返回该注入的差分文本，否则 ``None``。

    - 关→开 → :data:`NIGHT_POLICY`（授权书全文，同 SessionStart 注入文本）；
    - 开→关 → ``None``（**零注入**：用户裁决 2026-10-17，关闭状态不注入任何提示词）；
    - 无变化 → ``None``（**零输出零写入**，调用方据此零打扰）；
    - **首次见该会话** → 记基线但**不注入**：SessionStart 已按当时开关注入过一次，再注
      一遍纯属烧 token；基线记下即可捕获此后每一次真实跳变；
    - 无 ``session_id`` / 读开关异常 → ``None``（fail-open；且**永不跨会话误报**——
      上次读数按 session_id 各存一份）。
    """
    sid = str(session_id or "").strip()
    if not sid:
        return None
    try:
        on, _warnings = night_enabled(data)
    except Exception:  # noqa: BLE001 - 读开关异常 → 零输出
        return None
    now = "on" if on else "off"
    prior: Any = None
    try:
        from . import watch as _watch  # 延迟导入（同上）

        seen = _watch.load_state().get(NIGHT_SEEN_KEY)
        prior = dict(seen).get(sid) if isinstance(seen, dict) else None
    except Exception:  # noqa: BLE001 - 读不出当首次见（不注入、不炸）
        prior = None
    if prior == now:
        return None
    _seen_write(sid, now)
    if prior is None:
        return None
    return NIGHT_POLICY if on else None


def gate_deny_message(data: dict[str, Any] | None = None) -> str | None:
    """PreToolUse（AskUserQuestion）拦截判定：夜航 ON → 返回 deny 指令；OFF → ``None``。

    **fail-open 方向**：读开关异常一律返回 ``None``（放行）——不能因配置损坏就废掉提问
    工具；调用方（:func:`main` 的 ``--gate`` 支与 night-gate.sh）据此 exit 2 / exit 0。
    """
    try:
        on, _warnings = night_enabled(data)
    except Exception:  # noqa: BLE001 - 读开关出错不拦
        return None
    return NIGHT_GATE_DENY if on else None


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


def record_baseline(session_id: str, *, data: dict[str, Any] | None = None) -> str:
    """SessionStart 注入时**顺手落基线**（与 ``--diff`` 同一落点与键，单点）。

    为什么必须在这里落（2026-10-15 真机缺陷）：基线原本只由**首个 Stop 钩子**补记——开关若
    在 SessionStart 之后、首个 Stop 之前被拨动，那次真跳变会被当作基线吞掉（实案：会话
    起始 on、中途拨 off，`night_seen` 被记成 off）。
    SessionStart 是「本会话此刻开关读数」的最早可信落点，落在这里，跳变就再也躲不过。

    **只在 SessionStart 调**：PostCompact 只补注入、不动读数——它若重写基线，就会把
    「SessionStart → 拨动 → 下一次注入」之间的跳变盖掉（本缺陷的同一形态）。

    Returns:
        落下的读数（``"on"``/``"off"``）；无 session_id 或读开关异常 → 空串（fail-open 不落）。
    """
    sid = str(session_id or "").strip()
    if not sid:
        return ""
    try:
        on, _warnings = night_enabled(data)
    except Exception:  # noqa: BLE001 - 读开关异常不落基线（下次现读）
        return ""
    value = "on" if on else "off"
    _seen_write(sid, value)
    return value


def _read_settings_file(path: Path) -> dict[str, Any]:
    """按显式路径读配置（钩子传 ``--settings``）；读不出返回 ``{}``（fail-open 上游兜）。"""
    try:
        with path.open("rb") as fh:
            data = tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def main(argv: list[str] | None = None) -> int:
    """钩子入口，三模式：

    - ``--gate``（PreToolUse 拦截口，PSEP_2 §3.3a）：夜航 ON → **stderr 注出转 twin
      指令 + exit 2**（harness 拦截语义：exit 2 = deny、stderr 注入给模型）；OFF /
      读开关异常 → exit 0 放行（fail-open）。
    - ``--diff``（回合边界差分口，热生效单 2026-10-15）：与 ``--session`` 的上次读数比对，
      有跳变才把差分文本打到 **stdout**（调用方 watch-arm.sh 转 stderr + exit 2 = rewake
      注入下一轮）；无变化零输出。恒 exit 0（是否注入由调用方按输出是否为空决定）。
    - 缺省（政策注入口，PSEP_2 §3.2）：stdin 载荷取 ``hook_event_name`` → 开关 ON 时
      打印 ``hookSpecificOutput.additionalContext`` JSON；OFF/异常一律 exit 0。

    ``--settings PATH``：显式配置文件路径（钩子按 ``$QODER_PROJECT_DIR`` 传入；
    测试隔离面用同一参数）。载荷解析失败按 SessionStart 名义注入（两触发点政策同一份，
    名义差异无行为影响）；任何异常零输出退出。
    """
    parser = argparse.ArgumentParser(prog="night-policy")
    parser.add_argument(
        "--settings", default="", help="配置文件路径（缺省走 paths 默认）"
    )
    parser.add_argument(
        "--gate",
        action="store_true",
        help="PreToolUse 拦截口：夜航 ON → stderr 指令 + exit 2；OFF/异常 → exit 0",
    )
    parser.add_argument(
        "--diff",
        action="store_true",
        help="回合边界差分口：开关跳变 → stdout 差分文本；无变化零输出（恒 exit 0）",
    )
    parser.add_argument(
        "--session", default="", help="--diff 用：会话 id（上次读数按会话各存一份）"
    )
    args = parser.parse_args(argv)
    if args.gate:
        try:
            data = _read_settings_file(Path(args.settings)) if args.settings else None
            deny = gate_deny_message(data)
        except Exception:  # noqa: BLE001 - fail-open：gate 自身异常放行
            return 0
        if deny:
            print(deny, file=sys.stderr)
            return 2
        return 0
    if args.diff:
        try:
            data = _read_settings_file(Path(args.settings)) if args.settings else None
            msg = diff_message(args.session, data=data)
        except Exception:  # noqa: BLE001 - fail-open：差分口异常零输出
            return 0
        if msg:
            print(msg)
        return 0
    try:
        raw_payload = sys.stdin.read()
    except Exception:  # noqa: BLE001 - stdin 不可读 = 无触发名，按缺省
        raw_payload = ""
    event = "SessionStart"
    sid = ""
    try:
        parsed = json.loads(raw_payload)
        name = parsed.get("hook_event_name")
        event = (
            name.strip() if isinstance(name, str) and name.strip() else "SessionStart"
        )
        raw_sid = parsed.get("session_id")
        sid = raw_sid.strip() if isinstance(raw_sid, str) else ""
    except (json.JSONDecodeError, TypeError, AttributeError):
        pass
    try:
        data = _read_settings_file(Path(args.settings)) if args.settings else None
        if event == "SessionStart":
            # SessionStart 顺手落基线（本单缺陷修复）：否则「注入后、首个 Stop 前」的拨动
            # 会被首个 Stop 当基线吞掉。PostCompact 不写（见 record_baseline）。
            record_baseline(sid, data=data)
        out = policy_hook_output(event, data=data)
    except Exception:  # noqa: BLE001 - fail-open：任何异常零输出 exit 0
        return 0
    if out:
        print(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
