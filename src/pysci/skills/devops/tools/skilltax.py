"""平台注入税（技能清单 / agent 清单）实测与按 pod 关停。

设计依据 backlog ``20261010-plugin-tax-probe``；实测结论已登记
``orchestration/README.md`` §1.1。原生 CLI 在每个会话首跳注入两份清单附件
（transcript 中 ``type=attachment``）：

``skill_listing``
    技能名 + 一行描述（每条描述截到 ``skillListingMaxDescChars``，默认 300 字符）。
    pod 自有/部署技能按 cwd 发现，**平台内置技能与插件技能跨 cwd 恒定注入**——
    后者不计入成员 harness 预算，但每一跳都在向上下文交税。
``agent_listing_delta``
    可用 subagent 类型清单（Explore/Plan/general-purpose/…）。

关停键实测（headless 探针 pod，见 :func:`probe_session`）：

===========================  =============================================
settings 键                   实测效果
===========================  =============================================
``skills.disabled``           整条移除，含 ``security-scan``；自有技能不受影响
``skillOverrides``            ``{"<名>": "off"}`` 同样移除；插件技能须用全限定名
``enabledPlugins``            ``{"<plugin>@<marketplace>": false}`` 只动插件技能，
                              动不了内置技能
``skillListingMaxDescChars``  软手段：保名截描述（60 → 税降约 68%）
``skillListingBudgetFraction`` 按优先级丢描述，落到哪条不可预测，不用于治理
===========================  =============================================
"""

from __future__ import annotations

import json
import shutil
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pysci.paths import PODS_ROOT

#: headless 会话 transcript 分储根（按 cwd 项目键分目录，README §1.1）。
SESSIONS_ROOT = Path.home() / ".qoder-cn" / "projects"

#: 探针模型（flash 档足够复现清单注入；省信用）。
PROBE_MODEL = "Qwen3.8-Flash"

#: 平台内置技能（跨 cwd 恒定注入，headless pod 也注入，2026-10-10 实测 11 条）。
PLATFORM_SKILLS: tuple[str, ...] = (
    "agent-creator",
    "deep-research",
    "hook-config",
    "loop",
    "mcp-config",
    "run",
    "sdk",
    "security-scan",
    "skill-creator",
    "verify",
    "workflow-authoring",
)

#: 插件技能（全限定名 ``plugin:skill``；随用户级安装面变化，实测 2026-10-10）。
PLATFORM_PLUGIN_SKILLS: tuple[str, ...] = (
    "qoder-create-plugin:create-plugin",
    "qoder-sites:sites-build-agent-app",
    "qoder-sites:sites-building",
    "qoder-sites:sites-hosting",
    "qoder-sites:sites-management",
)

#: 组员 pod 仍保留的平台技能。默认空集：全部关停。
#: ``security-scan`` 也关——它在 push 前会用 AskUserQuestion 追问扫描模式，而 headless
#: 会话无人应答；组员的写权限与 push 规程本就由 charter + pod-guard + delivery-gate
#: 决定，平台技能只是引导性提示。**这是对安全提示面的主动取舍，须经用户认可**
#: （见 README §1.1 与本条 backlog 交付）。
KEEP_ON_PODS: tuple[str, ...] = ()


def recommended_disabled() -> list[str]:
    """组员 pod 应写进 ``settings.skills.disabled`` 的平台技能名。"""
    keep = set(KEEP_ON_PODS)
    return sorted({*PLATFORM_SKILLS, *PLATFORM_PLUGIN_SKILLS} - keep)


@dataclass(frozen=True)
class SkillLine:
    """清单里的一条技能（含其字节数与来源归类）。"""

    name: str
    desc: str
    bytes: int
    source: str


def parse_listing(content: str) -> list[SkillLine]:
    """把 ``skill_listing`` 附件正文解析成逐条技能，并按落盘 UTF-8 字节计量。

    Args:
        content: 附件 ``content`` 字段全文（``- name: desc`` 行，描述可折行）。

    Returns:
        每条技能的名称/描述/字节/来源（``自有``/``插件``/``平台``，来源由
        :func:`classify` 依名称判定）。
    """
    out: list[SkillLine] = []
    name, desc, raw = "", "", ""
    for ln in content.split("\n"):
        if ln.startswith("- "):
            if name:
                out.append(_line(name, desc, raw))
            head = ln[2:]
            name, _, desc = head.partition(": ")
            raw = ln
        elif name:
            desc += "\n" + ln
            raw += "\n" + ln
    if name:
        out.append(_line(name, desc, raw))
    return out


def _line(name: str, desc: str, raw: str) -> SkillLine:
    return SkillLine(
        name=name, desc=desc, bytes=len(raw.encode("utf-8")) + 1, source=classify(name)
    )


def classify(name: str) -> str:
    """按技能名判定来源：``自有``（成员/部署技能）、``插件``、``平台``（内置）。

    来源归类只用于报告；判定依据是名称形态——带 ``:`` 的是插件技能，其余落在
    :data:`PLATFORM_SKILLS` 里的是平台内置，剩下的即 pod 自有/自建/部署技能。
    """
    if ":" in name:
        return "插件"
    return "平台" if name in PLATFORM_SKILLS else "自有"


def pod_key(cwd: Path) -> str:
    """cwd → transcript 项目键（非字母数字一律换成 ``-``，实测与 CLI 分储一致）。"""
    return "".join(ch if ch.isalnum() else "-" for ch in str(cwd))


def own_skills(pod: Path) -> set[str]:
    """pod 自有/部署技能名（``<pod>/.qoder/skills/`` 下的目录名）。"""
    d = pod / ".qoder" / "skills"
    return {p.name for p in d.iterdir() if p.is_dir()} if d.exists() else set()


def settings_of(pod: Path) -> dict[str, Any]:
    """读 pod 的 ``.qoder/settings.json``（缺失或非法返回空 dict）。"""
    f = pod / ".qoder" / "settings.json"
    if not f.exists():
        return {}
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def disabled_in_settings(pod: Path) -> list[str]:
    """pod settings 里声明的 ``skills.disabled`` 列表。"""
    v = (settings_of(pod).get("skills") or {}).get("disabled")
    return [str(x) for x in v] if isinstance(v, list) else []


def disabled_gap(pod: Path) -> list[str]:
    """应关未关的平台技能名（新 pod 脚手架漏配、或平台新增技能时的漂移面）。"""
    return sorted(set(recommended_disabled()) - set(disabled_in_settings(pod)))


def apply_disabled(pod: Path) -> bool:
    """把应关的平台技能写进该 pod 的 ``skills.disabled``（幂等，其余键原样保留）。

    Args:
        pod: pod 目录。

    Returns:
        是否实际改写了文件（已齐全时返回 ``False``）。缺 settings.json 视为脚手架未完备，
        抛 :class:`FileNotFoundError` 由调用方报可读错误——不静默新建只读层文件。
    """
    f = pod / ".qoder" / "settings.json"
    if not f.exists():
        raise FileNotFoundError(str(f))
    data = settings_of(pod)
    if not data:
        raise ValueError(f"{f} 无法解析（拒绝覆盖非法 JSON）")
    skills = data.get("skills")
    if not isinstance(skills, dict):
        skills = {}
    have = [str(x) for x in skills.get("disabled") or [] if isinstance(x, str)]
    want = sorted(set(have) | set(recommended_disabled()))
    if have == want:
        return False
    skills["disabled"] = want
    data["skills"] = skills
    f.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return True


def listing_attachment(pod: Path, sid: str = "") -> dict[str, Any] | None:
    """取 pod 会话的 ``skill_listing`` 附件（``sid`` 为空则取最近一条会话）。"""
    d = SESSIONS_ROOT / pod_key(pod)
    if not d.is_dir():
        return None
    files = (
        [d / f"{sid}.jsonl"]
        if sid
        else sorted(d.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    )
    for f in files:
        if not f.exists():
            continue
        for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                att = json.loads(line).get("attachment") or {}
            except json.JSONDecodeError:
                continue
            if att.get("type") == "skill_listing":
                return att
    return None


def agent_listing_bytes(pod: Path, sid: str = "") -> int:
    """同会话 ``agent_listing_delta`` 附件的行字节（次要税，仅报告用）。"""
    d = SESSIONS_ROOT / pod_key(pod)
    if not d.is_dir():
        return 0
    files = (
        [d / f"{sid}.jsonl"]
        if sid
        else sorted(d.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    )
    for f in files:
        if not f.exists():
            continue
        for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                att = json.loads(line).get("attachment") or {}
            except json.JSONDecodeError:
                continue
            if att.get("type") == "agent_listing_delta":
                lines = att.get("addedLines") or []
                return sum(len(str(x).encode("utf-8")) + 1 for x in lines)
    return 0


def measure(pod: Path, sid: str = "") -> dict[str, Any]:
    """量一个 pod 会话的平台税：清单字节、按来源拆分、仍可见的平台项、被误伤的自有技能。

    Args:
        pod: pod 目录（决定 transcript 项目键与 ``.qoder/skills`` 面）。
        sid: 指定会话；缺省取该 pod 最近一条会话。

    Returns:
        ``seen=False`` 时只有 ``pod``/``seen`` 两键（该 pod 尚无会话记录）；否则含
        ``skills``（清单里的技能名）、``total_bytes``（清单正文落盘 UTF-8 字节）、
        ``tax_platform_bytes``/``tax_plugin_bytes``/``own_bytes``（按来源拆分的字节）、
        ``agent_bytes``（agent 清单附件字节）、``platform_visible``（仍可见的平台/插件项）、
        ``own_missing``（部署了却没进清单的自有技能 = 关停误伤面）、``gap``（settings
        应关未关项）。
    """
    att = listing_attachment(pod, sid)
    if att is None:
        return {"pod": pod.name, "seen": False}
    lines = parse_listing(att.get("content") or "")
    own_visible = sorted(sk.name for sk in lines if sk.source == "自有")
    return {
        "pod": pod.name,
        "seen": True,
        "skills": [sk.name for sk in lines],
        "total_bytes": len((att.get("content") or "").encode("utf-8")),
        "tax_platform_bytes": sum(sk.bytes for sk in lines if sk.source == "平台"),
        "tax_plugin_bytes": sum(sk.bytes for sk in lines if sk.source == "插件"),
        "own_bytes": sum(sk.bytes for sk in lines if sk.source == "自有"),
        "agent_bytes": agent_listing_bytes(pod, sid),
        "platform_visible": sorted(
            sk.name for sk in lines if sk.source in ("平台", "插件")
        ),
        "own_visible": own_visible,
        "own_missing": sorted(own_skills(pod) - set(own_visible)),
        "gap": disabled_gap(pod),
    }


def short_names(names: list[str], n: int = 5) -> str:
    """技能名列表短显（超出折叠成「+k」，避免十行刷屏）。"""
    if len(names) <= n:
        return " ".join(names)
    return " ".join(names[:n]) + f" +{len(names) - n}"


def pod_report(pod: Path) -> list[str]:
    """单 pod 的税报告行（无历史会话时降级为 settings 声明检查）。"""
    m = measure(pod)
    if not m["seen"]:
        gap = disabled_gap(pod)
        mark = "√" if not gap else "✗"
        return [
            f"  {pod.name:<10} [{mark}] 无会话可量；settings 应关未关={len(gap)}"
            + (f" ({short_names(gap)})" if gap else "")
        ]
    gap = m["gap"]
    left = m["platform_visible"]
    mark = "√" if not gap and not left else "✗"
    return [
        f"  {pod.name:<10} [{mark}] 清单 {m['total_bytes']}B"
        f"（平台 {m['tax_platform_bytes']}B + 插件 {m['tax_plugin_bytes']}B"
        f" + 自有 {m['own_bytes']}B） agent 清单 {m['agent_bytes']}B"
        f"；仍可见平台项={len(left)}"
        + (f" ({short_names(left)})" if left else "")
        + (f"；settings 缺口={len(gap)}" if gap else "")
    ]


# ---------------------------------------------------------------------------
# 探针（复测用：CLI 升级后重跑，确认键义未变）
# ---------------------------------------------------------------------------
#: 探针 pod 的自有技能（验证关停不误伤自有）。
_PROBE_OWN_SKILL = (
    "---\ndescription: 探针自有技能，验证关停平台技能不误伤 pod 自有技能。\n"
    "name: probe-own\n---\n\n被调用时回答 OWN-OK。\n"
)
_PROBE_PROMPT = "只输出 ok，不要使用任何工具。"


def probe_pod(
    name: str,
    overlay: dict[str, Any] | None = None,
    *,
    base: Path | None = None,
) -> Path:
    """造一个 Temp 探针 pod：可选克隆真 pod 的只读层，再叠 ``overlay`` 到 settings。

    Args:
        name: 探针 pod 名（决定 Temp 路径与 transcript 项目键）。
        overlay: 叠加进 ``.qoder/settings.json`` 的键值（关停实验变量）。
        base: 要克隆的真 pod；缺省用最小骨架 + 一个自有技能。

    Returns:
        探针 pod 目录。settings 里自动加 ``disableAllHooks: true``——税与 hooks 无关，
        而 pod 的 hook 命令是相对路径（``../../guards/*.mjs``），在 Temp 下必然失效。
    """
    root = Path(__import__("tempfile").gettempdir()) / "pysci-skilltax" / name
    shutil.rmtree(root, ignore_errors=True)
    if base is not None and base.exists():
        shutil.copytree(base / ".qoder", root / ".qoder")
        if (base / "AGENTS.md").exists():
            shutil.copy2(base / "AGENTS.md", root / "AGENTS.md")
    else:
        (root / ".qoder" / "skills" / "probe-own").mkdir(parents=True, exist_ok=True)
        (root / ".qoder" / "skills" / "probe-own" / "SKILL.md").write_text(
            _PROBE_OWN_SKILL, encoding="utf-8"
        )
        (root / ".qoder" / "mcp.json").write_text(
            '{"mcpServers": {}}', encoding="utf-8"
        )
        (root / "AGENTS.md").write_text(
            "# 探针 pod\n\n仅用于实测。\n", encoding="utf-8"
        )
    settings = settings_of(root)
    settings.update(overlay or {})
    settings["disableAllHooks"] = True
    settings.setdefault("agentsMdExcludes", ["**/leader-only.md"])
    (root / ".qoder").mkdir(parents=True, exist_ok=True)
    (root / ".qoder" / "settings.json").write_text(
        json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return root


def run_session(
    pod: Path,
    *,
    model: str = PROBE_MODEL,
    max_turns: int = 1,
    prompt: str = _PROBE_PROMPT,
    timeout: int = 420,
) -> dict[str, Any]:
    """在 ``pod`` 为 cwd 跑一跳 headless 会话，返回 envelope dict（含 ``_sid``/``_rc``）。"""
    from pysci.skills.orchestration.tools.registry import resolve_exe

    sid = str(uuid.uuid4())
    cmd = [
        str(resolve_exe()),
        "--cwd",
        str(pod),
        "-p",
        prompt,
        "-o",
        "json",
        "--session-id",
        sid,
        "--max-turns",
        str(max_turns),
        "-m",
        model,
        "--mcp-config",
        ".qoder/mcp.json",
        "--strict-mcp-config",
    ]
    p = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        cwd=str(pod),
    )
    from pysci.skills.orchestration.tools.runner import Envelope

    env = Envelope.parse(p.stdout or "").raw
    env["_sid"] = sid
    env["_rc"] = p.returncode
    return env


def probe_session(
    name: str,
    overlay: dict[str, Any] | None = None,
    *,
    base: Path | None = None,
) -> dict[str, Any]:
    """探针一跳并量出该 settings 叠加层下的清单税（A/B 实验原语）。

    Returns:
        :func:`measure` 的字段外加 ``turns``/``rc``/``ratio``。``ratio`` 是 envelope 的
        ``usage.context_usage_ratio``（BYOK 下 token 计数全 0，唯此比值可用，见 runner.py
        ADR）——同一 pod 两个 settings 变体对比它，即首跳注入面（技能清单 + agent 清单）
        的机械增减证据。
    """
    from pysci.skills.orchestration.tools.runner import Envelope

    pod = probe_pod(name, overlay, base=base)
    env = run_session(pod)
    m = measure(pod, env.get("_sid", ""))
    parsed = Envelope.parse(env.get("_stdout", "") or "")
    m["turns"] = parsed.num_turns
    m["rc"] = env.get("_rc")
    m["ratio"] = parsed.context_usage_ratio
    return m


def main(argv: list[str] | None = None) -> int:
    """``pysci-dev skilltax`` 子命令体：量税、查缺口、按需落配置或实跑探针。"""
    import argparse

    ap = argparse.ArgumentParser(
        prog="pysci-dev skilltax", description=__doc__.split("\n")[0]
    )
    ap.add_argument("--pod", help="只处理指定成员")
    ap.add_argument("--probe", action="store_true", help="造探针 pod 实跑一跳复测键义")
    ap.add_argument("--overlay", help="探针 settings 叠加层（JSON 串）")
    ap.add_argument(
        "--apply", action="store_true", help="把应关平台技能写进各 pod settings"
    )
    args = ap.parse_args(argv)

    pods = (
        [PODS_ROOT / args.pod]
        if args.pod
        else sorted(p for p in PODS_ROOT.iterdir() if p.is_dir())
        if PODS_ROOT.exists()
        else []
    )
    if args.probe:
        overlay = json.loads(args.overlay or "{}")
        print(
            json.dumps(
                probe_session(f"cli-{args.pod or 'all'}", overlay),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.apply:
        changed = 0
        for pod in pods:
            try:
                if apply_disabled(pod):
                    changed += 1
                    print(f"  [+] {pod.name}：skills.disabled 已补齐")
            except (FileNotFoundError, ValueError) as exc:
                print(f"  [!] {pod.name}：{exc}")
                return 2
        print(
            f"[√] 落配置完成：改写 {changed} 个 pod（应关 {len(recommended_disabled())} 项）"
        )
        return 0
    print("== 平台注入税（skill_listing / agent_listing_delta）==")
    bad = 0
    for pod in pods:
        for line in pod_report(pod):
            print(line)
            bad += "[✗]" in line
    if not bad:
        print(
            f"  [√] 应关平台技能 {len(recommended_disabled())} 项已全部关停，"
            f"保留 {list(KEEP_ON_PODS)}"
        )
    return 2 if bad else 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
