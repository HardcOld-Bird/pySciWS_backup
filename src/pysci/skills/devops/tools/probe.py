"""agentsMdExcludes 生效性机械探针（backlog 20261010-165408-devops）。

设计依据已批准建议 ``orchestration/state/suggestions/approved/20261010-165408-devops.md``：
`agentsMdExcludes` 是 CLI 侧行为，doctor 的静态巡检只能证明 pod settings 里写了这条
glob，**不能证明 CLI 真的据此把组长专属根规则（``leader-only.md``）挡在组员会话之外**。
若将来 CLI 改了该键名或 glob 语义（例如不再支持 ``**/`` 前缀），十 pod 会**静默退回**
「组员看见组长规程」（越权面），而 doctor 仍报全绿。本模块把一次性口头探针固化成
可复跑、带机械断言的哨兵。

方法（自足 fixture + 因果 A/B，不触碰受治理的根规则文件）：

- 在 Temp 造一个独立 git 仓库（``git init``）——rules 依 §1.1 从仓库根 ``.qoder/rules/**``
  沿 cwd 向上泄漏至 ``.git`` 边界，故 Temp 仓库自有一套 ``trigger: always_on`` 规则；
- 两枚规则各带一枚**不可猜随机 token**（``TOK-<16hex>``）：``keep.md``（永不排除）与
  ``hide.md``（被 pod 的 ``agentsMdExcludes`` glob 命中）；
- 两个探针 pod 跑同一句「列出你上下文里所有 TOK- 标记」的 headless 一跳：
  ``withexclude``（``["**/hide.md"]``）与 ``control``（``[]`` 不排除）；
- **因果断言**：``hide`` token 在 control 出现（证明探针灵敏、模型确会复述被注入的 token）
  且在 withexclude 消失（证明排除生效）。二者同时成立 → ``mechanism_ok``。control 不出现
  token → ``inconclusive``（模型未灵敏，绝不误判为绿）；withexclude 仍出现 → ``exclusion_failed``
  （CLI 不再honoring glob，真回归，硬告警）。

结论写入 git 跟踪台账 ``orchestration/state/agents-md-excludes-probe.json``，键为
``{cli_version, exclude_patterns}`` 的哈希——**不含时间戳**（确定性、跨 worktree 合并零漂移）。
staleness 只由「CLI 升级」或「pod 排除 glob 面变更」触发，与墙钟无关。doctor 读台账：
verdict 假 → 硬告警；哈希与当前面不符 → 提示复测（advisory，因 headless 昂贵不硬失败）。
"""

from __future__ import annotations

import functools
import hashlib
import json
import re
import secrets
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any

from pysci.paths import ORCH_STATE_ROOT, PODS_ROOT

#: 探针模型（flash 档足够复现规则注入/排除；省信用，与 skilltax 同源）。
from pysci.skills.devops.tools.skilltax import PROBE_MODEL

#: token 形态：不可猜随机串，模型只能逐字复述被注入上下文里的原值。
TOKEN_RE = re.compile(r"TOK-[0-9a-f]{8,}")

#: 排除 glob 用与真实 pod 相同的形态（`**/<名>`），专门捕获 `**/` 前缀语义漂移。
GLOB_SHAPE = "**/hide.md"

#: 台账（git 跟踪；确定性内容，无时间戳）。
LEDGER = ORCH_STATE_ROOT / "agents-md-excludes-probe.json"

#: 规则 frontmatter——与根 basic.md/leader-only.md 同为 always_on，才会全文注入。
_FRONT = "---\ntrigger: always_on\nalwaysApply: true\n---\n\n"

_PROMPT = (
    "只输出你的上下文里所有以 TOK- 开头的标记串（原样逐字照抄，多个用空格分隔）；"
    "如果一个都没有就输出 NONE。不要使用任何工具，不要任何解释。"
)


@functools.lru_cache(maxsize=1)
def cli_version() -> str:
    """当前原生 exe 版本号（staleness 键的一部分；CLI 升级即触发复测）。进程内缓存。"""
    from pysci.skills.orchestration.tools.registry import resolve_exe

    p = subprocess.run(
        [str(resolve_exe()), "--version"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )
    return (p.stdout or "").strip() or "unknown"


def pod_exclude_patterns() -> list[str]:
    """全部 pod settings 里声明的 ``agentsMdExcludes`` 归一化模式集合（排序去重）。

    归一化到**文件名段**（与 dev.exclude_problems 同口径），故 ``**/leader-only.md``、
    ``rules/leader-only.md``、``leader-only.md`` 视为同一模式——staleness 只对语义面敏感。
    """
    out: set[str] = set()
    if PODS_ROOT.exists():
        for pod in PODS_ROOT.iterdir():
            f = pod / ".qoder" / "settings.json"
            if not f.exists():
                continue
            try:
                ex = (
                    json.loads(f.read_text(encoding="utf-8")).get("agentsMdExcludes")
                    or []
                )
            except (json.JSONDecodeError, OSError):
                continue
            for e in ex:
                out.add(Path(str(e).replace("\\", "/")).name)
    return sorted(out)


def config_key() -> dict[str, Any]:
    """staleness 键：CLI 版本 + 当前排除模式面（决定探针结论是否仍适用）。"""
    return {"cli_version": cli_version(), "patterns": pod_exclude_patterns()}


def config_hash() -> str:
    """对 :func:`config_key` 取稳定 sha256（键序固定，跨进程/平台一致）。"""
    blob = json.dumps(config_key(), ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _front_rule(token: str, label: str) -> str:
    return _FRONT + f"# {label}（探针规则，勿当真）\n\n标记 {token} 结束\n"


def build_fixture() -> tuple[Path, str, str, Path, Path]:
    """造自足 Temp git fixture，返回 (root, keep_token, hide_token, withexclude_pod, control_pod)。

    两个 pod 都 ``disableAllHooks``（探针与守卫无关，且真实 hook 是相对路径，Temp 下失效）。
    """
    root = Path(tempfile.gettempdir()) / "amdepx-probe"
    shutil.rmtree(root, ignore_errors=True)
    rules = root / ".qoder" / "rules"
    rules.mkdir(parents=True)
    keep = "TOK-" + secrets.token_hex(8)
    hide = "TOK-" + secrets.token_hex(8)
    (rules / "keep.md").write_text(_front_rule(keep, "应可见"), encoding="utf-8")
    (rules / "hide.md").write_text(_front_rule(hide, "应被排除"), encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(root)], check=True)

    def mkpod(name: str, excludes: list[str]) -> Path:
        p = root / name
        q = p / ".qoder"
        q.mkdir(parents=True)
        (q / "settings.json").write_text(
            json.dumps({"agentsMdExcludes": excludes, "disableAllHooks": True}),
            encoding="utf-8",
        )
        return p

    return root, keep, hide, mkpod("withexclude", [GLOB_SHAPE]), mkpod("control", [])


def _run_session(pod_cwd: Path) -> dict[str, Any]:
    """在 ``pod_cwd`` 为 cwd 跑一跳 headless，返回 {tokens, text, ratio, turns, rc}。"""
    from pysci.skills.orchestration.tools.registry import resolve_exe
    from pysci.skills.orchestration.tools.runner import Envelope

    sid = str(uuid.uuid4())
    cmd = [
        str(resolve_exe()),
        "--cwd",
        str(pod_cwd),
        "-p",
        _PROMPT,
        "-o",
        "json",
        "--session-id",
        sid,
        "--max-turns",
        "1",
        "-m",
        PROBE_MODEL,
    ]
    p = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=240,
        cwd=str(pod_cwd),
    )
    env = Envelope.parse(p.stdout or "").raw
    text = str(env.get("result", ""))
    return {
        "tokens": set(TOKEN_RE.findall(text)),
        "text": text.strip()[:200],
        "ratio": (env.get("usage") or {}).get("context_usage_ratio"),
        "turns": env.get("num_turns"),
        "rc": p.returncode,
    }


def probe_mechanism(retries: int = 3) -> dict[str, Any]:
    """跑因果 A/B 得出机制结论。control 灵敏性不成立时重试（绝不把「模型没复述」误判为绿）。

    Returns:
        verdict dict：``status`` ∈ {``ok``, ``exclusion_failed``, ``inconclusive``}、
        ``mechanism_ok`` 布尔、以及两跑观测（token 命中、ratio、turns）供台账与排障。
    """
    root, keep, hide, wpath, cpath = build_fixture()
    ctrl = None
    for _ in range(max(1, retries)):
        ctrl = _run_session(cpath)
        if hide in ctrl["tokens"]:  # 探针灵敏：模型确会复述被注入 token
            break
    wout = _run_session(wpath)
    shutil.rmtree(root, ignore_errors=True)

    sensitive = hide in (ctrl or {"tokens": set()})["tokens"]
    hidden = hide not in wout["tokens"]
    if not sensitive:
        status = "inconclusive"
    elif hidden:
        status = "ok"
    else:
        status = "exclusion_failed"
    return {
        "status": status,
        "mechanism_ok": status == "ok",
        "sensitive": sensitive,
        "keep_injected": keep in wout["tokens"],
        "hide_leaked_in_withexclude": hide in wout["tokens"],
        "model": PROBE_MODEL,
        "control_ratio": (ctrl or {}).get("ratio"),
        "withexclude_ratio": wout["ratio"],
    }


def read_ledger() -> dict[str, Any] | None:
    """读台账（缺失/非法返回 ``None``）。"""
    if not LEDGER.exists():
        return None
    try:
        data = json.loads(LEDGER.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return data if isinstance(data, dict) else None


def write_ledger(verdict: dict[str, Any]) -> dict[str, Any]:
    """把机制结论 + staleness 键写入台账（确定性字段，无时间戳）。"""
    key = config_key()
    rec = {
        "schema": 1,
        "config_hash": hashlib.sha256(
            json.dumps(key, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16],
        "cli_version": key["cli_version"],
        "patterns": key["patterns"],
        "status": verdict["status"],
        "mechanism_ok": verdict["mechanism_ok"],
        "keep_injected": verdict["keep_injected"],
        "model": verdict["model"],
    }
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    LEDGER.write_text(
        json.dumps(rec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return rec


def doctor_status() -> tuple[str, str]:
    """给 doctor 的一行结论：返回 (level, message)，level ∈ {ok, advisory, fail}。

    - fail：台账说机制失效（真越权回归）；
    - advisory：未跑过 / CLI 或排除面已变（哈希不符当前 staleness 键）→ 该复测；
    - ok：台账新鲜且机制成立。
    """
    led = read_ledger()
    if led is None:
        return (
            "advisory",
            "未跑过探针：执行 `pysci-dev probe`（headless 约 2 跳）建立基线",
        )
    if not led.get("mechanism_ok"):
        return (
            "fail",
            f"agentsMdExcludes 机制判为 {led.get('status')!r}——组员可能注入组长专属规程，"
            "须排查 CLI glob 语义变更",
        )
    if led.get("config_hash") != config_hash():
        return (
            "advisory",
            f"台账过期（CLI 或排除面已变：{led.get('cli_version')}→{cli_version()}），"
            "请 `pysci-dev probe` 复测",
        )
    return (
        "ok",
        f"机制经实测确认（CLI {led.get('cli_version')}，排除面 {led.get('patterns')}）",
    )


def main(argv: list[str] | None = None) -> int:
    """``pysci-dev probe`` 子命令体：跑机制探针、写台账、打印结论。"""
    import argparse

    ap = argparse.ArgumentParser(
        prog="pysci-dev probe", description=__doc__.split("\n")[0]
    )
    ap.add_argument(
        "--json", action="store_true", help="只打印 verdict JSON（不写台账判断行）"
    )
    ap.add_argument("--no-ledger", action="store_true", help="只复测不落台账")
    args = ap.parse_args(argv)

    verdict = probe_mechanism()
    if not args.no_ledger:
        rec = write_ledger(verdict)
    else:
        rec = verdict
    if args.json:
        print(json.dumps(verdict, ensure_ascii=False, indent=2))
    else:
        mark = {"ok": "√", "exclusion_failed": "✗", "inconclusive": "!"}[
            verdict["status"]
        ]
        print(f"== agentsMdExcludes 生效性探针 == [{mark}] status={verdict['status']}")
        print(
            f"  灵敏度(control 复述 hide token)={verdict['sensitive']}  "
            f"排除态 hide 泄漏={verdict['hide_leaked_in_withexclude']}  "
            f"keep 注入={verdict['keep_injected']}"
        )
        print(
            f"  首跳上下文 withexclude ratio={verdict['withexclude_ratio']}"
            f"（排除省下的即组长规程注入体量）"
        )
        if not args.no_ledger:
            print(f"  台账已写：{LEDGER.name}  config_hash={rec.get('config_hash')}")
    return 0 if verdict["mechanism_ok"] else 2


if __name__ == "__main__":
    import sys

    sys.exit(main())
