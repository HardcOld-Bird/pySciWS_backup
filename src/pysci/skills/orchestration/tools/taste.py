"""twin 判断通道（``orch taste``）：宪章注入 + 冷启动单跳 + 结构化留痕（REFACTOR §9，E6-prep）。

为什么长这样：

- **开关在代码里，生效权在用户手里**：``registry.taste_enabled`` 默认 False（见
  :data:`registry.DEFAULT_TASTE_ENABLED`）。宪章未确认时本通道**拒绝派发**，而不是
  「提醒一句照样跑」——twin 一开口就以「用户的声音」对外服务（§9.4 代理裁决），误开
  的代价远高于多等一次确认。
- **冷启动是双硬闸**：本模块只以 ``session="new"`` 调 :func:`dispatch.do_dispatch`，
  而 registry 成员字段 ``cold_start`` 让 dispatch 层拒绝任何非 new 会话（§9 无记忆
  纪律；「别 resume」写在 charter 里迟早被忘，§0.3 硬代码优先）。
- **案卷格式复用 jury**（:func:`jury.parse_case`）：同一份案卷既能送客观仲裁又能送
  审美终裁，§5.5 模棱两可回退链才接得上；共享逻辑做库、不做第二套解析器（§0.9）。

留痕形态：``state/evals/taste/<case-id>.json``（``{选择, 理由, 置信度}`` + 会话 sid +
材料哈希），供 ``pysci-mirror review`` 盲审回放与 ``status`` 计数（§10）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from pysci.paths import ORCH_STATE_ROOT, PROJECT_ROOT

from . import presets
from .dispatch import DispatchOutcome, do_dispatch
from .jury import Case, parse_case
from .registry import Registry

#: 判断留痕目录（与 jury 的 ``evals/jury/`` 同族）。twin 判断即以用户名义表态，
#: 属 §0.4「高风险授权操作必留痕」，故入库跟踪而非运行态排除。
TASTE_DIR: Path = ORCH_STATE_ROOT / "evals" / "taste"

#: 宪章文件名（pod 内，registry 的 ``readonly_extra`` 使其对 twin 只读）。
PREFERENCE_NAME = "PREFERENCE.md"

#: 未生效时的拒绝语（一字不差地进 CLI 输出——success_criteria 的判据面）。
DISABLED_HINT = (
    "twin 宪章未生效：PREFERENCE.md 仍是草案 v1，生效确认权归用户"
    "（REFACTOR §9.4），已拒绝派发"
)
NEXT_HINT = (
    "[NEXT] 用户确认宪章草案后，把 registry.json 顶层 taste_enabled 置 true 即对外服务；"
    "先看注入内容可加 --dry（不派发、零成本）"
)

#: 判断输出的 JSON 块（与 jury 同法：取文中最后一个平衡对象，容忍前后杂讯）。
_JSON_BLOCK_RE = re.compile(r"\{[\s\S]*\}")


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _atomic_write_json(path: Path, obj: Any) -> None:
    """原子写 JSON（``*.tmp`` → replace，与 registry/jury 同法：中断不留半文件）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    tmp.replace(path)


def _rel(p: Path) -> str:
    """打印用相对路径；测试把留痕目录 monkeypatch 到 tmp 时原样显示（不炸）。"""
    try:
        return str(p.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(p)


def preference_text(reg: Registry | None = None, member_id: str = "twin") -> str:
    """读该成员 pod 内的宪章正文；缺失/不可读返回空串（由调用方判不可用）。"""
    reg = reg or Registry.load()
    try:
        pod = reg.member(member_id).pod
    except KeyError:
        return ""
    p = pod / PREFERENCE_NAME
    return p.read_text(encoding="utf-8") if p.is_file() else ""


def record_path(case_id: str) -> Path:
    """判断留痕路径（同案重判覆盖＝以最新为准，与 jury 一致）。"""
    return TASTE_DIR / f"{case_id}.json"


def build_prompt(case: Case, preference: str) -> str:
    """组装注入正文：宪章全文 + 案卷 + 输出格式硬要求。

    顺序有意为之：宪章在前（判断依据）、案卷在中（对象）、格式在后（最近指令位对
    单跳会话最稳），格式段要求 ``<result>`` 正文含且仅含一个 JSON 对象。
    """
    options = (
        f"### 选项 A\n{case.option_a.strip()}\n\n### 选项 B\n{case.option_b.strip()}"
    )
    evidence = (
        f"\n## 证据（案卷撰写者所附）\n{case.evidence.strip()}\n"
        if case.evidence
        else ""
    )
    return (
        "# 本次任务：代用户作一次二选一判断（twin）\n\n"
        "## 判断宪章（PREFERENCE.md 全文，你的唯一偏好依据）\n\n"
        f"{preference.strip()}\n\n"
        f"## 问题\n{case.question.strip()}\n\n"
        f"{options}\n{evidence}\n"
        "## 输出要求（硬格式）\n\n"
        "在 `<result>` 正文里给出**且只给出**一个 JSON 对象：\n\n"
        '{"choice": "A", "reason": "援引宪章条款编号（P-xx）与案卷证据的简短理由", '
        '"confidence": 0.62}\n\n'
        "- `choice` 只能是 `A` 或 `B`；**必须表态**，不许第三方案、不许「视情况而定」"
        "（REFACTOR §0.5 终裁不给平局）；\n"
        "- 把握不足用低 `confidence` 表达，而不是回避；\n"
        "- `reason` 点明依据的条款编号；宪章没有对应条款时写明「外推，宪章无既例」。"
    )


def parse_judgement(body: str) -> dict[str, Any]:
    """从交付正文抽 ``{choice, reason, confidence}``；抽不到即 ``parsed=False``。

    钳制规则与 :func:`jury.parse_vote` 同源：choice 只认 A/B（大写化后校验，其余置空），
    confidence 越界钳到 0–1、非数置 None——**不猜**（宁可留空让人看出这条没读出来）。
    """
    out: dict[str, Any] = {
        "choice": "",
        "reason": "",
        "confidence": None,
        "parsed": False,
    }
    m = _JSON_BLOCK_RE.search(body or "")
    if not m:
        return out
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return out
    if not isinstance(data, dict):
        return out
    choice = str(data.get("choice", "")).strip().upper()
    out["choice"] = choice if choice in ("A", "B") else ""
    out["reason"] = str(data.get("reason", "")).strip()
    try:
        out["confidence"] = min(1.0, max(0.0, float(data.get("confidence"))))
    except (TypeError, ValueError):
        out["confidence"] = None
    out["parsed"] = True
    return out


def judge(
    case_file: Path | str,
    *,
    member_id: str = "twin",
    model_tier: str | None = None,
    quiet: bool = False,
) -> tuple[int, dict[str, Any]]:
    """执行一次判断跳并落留痕。

    Returns:
        ``(code, record)``：0=已表态且落盘；1=派发失败或没读出选择（留痕仍写，
        便于 mirror 看到「跑过但没读出来」）；2=未生效/案卷不可用/宪章缺失。
    """
    reg = Registry.load()
    if not reg.taste_enabled:
        print(f"✗ {DISABLED_HINT}")
        print(f"    {NEXT_HINT}")
        return 2, {}
    pref = preference_text(reg, member_id)
    if not pref.strip():
        print(f"✗ 宪章缺失：{member_id} pod 内无 {PREFERENCE_NAME}，无依据不判断")
        return 2, {}
    try:
        case = parse_case(case_file)
    except (OSError, ValueError) as exc:
        print(f"✗ 案卷不可用：{exc}")
        return 2, {}
    prompt = build_prompt(case, pref)
    if not quiet:
        print(
            f"→ taste {case.case_id}（宪章 {len(pref)} B，案卷 {len(prompt) - len(pref)} B 注入）"
        )
    outcome: DispatchOutcome = do_dispatch(
        member_id,
        text=prompt,
        slug=f"taste-{case.case_id}",
        session="new",  # 冷启动硬保证；成员 cold_start 使非 new 会话在 dispatch 层被拒
        model_tier=model_tier,
        no_checks=True,
        attach_replies=False,
        eval_hop=True,  # 判断跳不入生产台账、不塞建议队列（与压测跳同一隔离语义）
        quiet=quiet,
    )
    j = parse_judgement(outcome.body)
    env = outcome.envelope
    record = {
        "ts": _now(),
        "case_id": case.case_id,
        "case_file": str(case.path or case_file),
        "dossier_hash": hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16],
        "member": member_id,
        "model_tier": presets.resolve_role(member_id, model_tier),
        "code": outcome.code,
        "kind": outcome.kind,
        "sid": outcome.sid,
        "turns": env.num_turns if env else 0,
        "duration_ms": env.duration_ms if env else 0,
        "ctx_ratio": round(env.context_usage_ratio, 5) if env else 0,
        "delta_chars": outcome.delta_chars,
        "est_tokens": outcome.est_tokens,
        "choice": j["choice"],
        "confidence": j["confidence"],
        "reason": j["reason"],
        "parsed": j["parsed"],
        "body": (outcome.body or "")[:2000],
    }
    p = write_record(record)
    if not quiet:
        if j["parsed"] and j["choice"]:
            conf = "-" if j["confidence"] is None else f"{j['confidence']:.2f}"
            print(
                f"  判断：{j['choice']}（置信度 {conf}，会话 {str(outcome.sid)[:8]}）"
                f"\n  理由：{j['reason']}"
            )
        else:
            print(f"  [!] 未从交付里读出选择（kind={outcome.kind}），已原样留痕")
        print(f"  留痕：{_rel(p)}")
    return (0 if j["parsed"] and j["choice"] else 1), record


def write_record(record: dict[str, Any]) -> Path:
    """留痕落盘（原子写；目录自动建）。"""
    p = record_path(str(record["case_id"]))
    _atomic_write_json(p, record)
    return p


def cmd_taste(args: argparse.Namespace) -> int:
    """``orch taste <case-file> [--dry] [--model-tier …]``——twin 判断单跳。

    退出码：0=已表态并落盘；1=派发失败或读不出选择；2=未生效/案卷不可用/宪章缺失。
    """
    reg = Registry.load()
    if args.dry:
        pref = preference_text(reg, args.member)
        try:
            case = parse_case(args.case_file)
        except (OSError, ValueError) as exc:
            print(f"✗ 案卷不可用：{exc}")
            return 2
        prompt = build_prompt(case, pref)
        print(prompt)
        print(
            f"\n— 注入 {len(prompt)} B（宪章 {len(pref)} B 或 0=缺失），"
            f"哈希 {hashlib.sha256(prompt.encode('utf-8')).hexdigest()[:16]}"
        )
        print(f"— taste_enabled={reg.taste_enabled}（未生效时正式运行会被拒绝派发）")
        return 0
    code, _ = judge(args.case_file, member_id=args.member, model_tier=args.model_tier)
    return code


def _add_parser(sub: Any) -> None:
    """注册 ``orch taste`` 子命令（由 orch.py 门面调用，保持 CLI 装配单点）。"""
    p = sub.add_parser(
        "taste",
        help="twin 审美孪生判断（REFACTOR §9）：宪章注入 + 冷启动单跳 → {选择,理由,置信度}",
    )
    p.add_argument(
        "case_file", help="案卷文件（## 问题 / ## 选项 A / ## 选项 B [+ ## 证据]）"
    )
    p.add_argument(
        "--dry",
        action="store_true",
        help="只打印注入内容（宪章+案卷+格式）与哈希，不派发",
    )
    p.add_argument("--member", default="twin", help="判断成员 id（默认 twin）")
    p.add_argument(
        "--model-tier",
        type=presets.normalize_role,
        choices=list(presets.ROLE_CHOICES),
        help="覆盖成员档位（main/pro/minimum，旧档名 flash/max 自动归一）",
    )
    p.set_defaults(func=cmd_taste)
