"""twin 判断通道（``orch twin taste``）：宪章注入 + 冷启动单跳 + 结构化留痕（README §1/§3.1，E6-prep）。

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
材料哈希）。盲审回放面留后续单（§12），暂由 ``pysci status`` 计数（§10）。
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
    "（README §1 代理裁决：上线/下线宣告权归用户），已拒绝派发"
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


def case_id_from_path(path: Path | str) -> str:
    """案卷路径 → 留痕 case id（文件名 stem；非法字符折 ``-``；空则 ``case``）。

    consult 案卷是自由形式文件（无 A/B 解析），故 case id 取 stem——与 taste 的
    ``case_id``（jury 案卷解析所得）同作**关联键**：记录 ↔ journal 行 ↔ 复核入口。
    """
    stem = Path(path).stem.strip()
    cleaned = re.sub(r"[^0-9A-Za-z_\u4e00-\u9fff.-]+", "-", stem)
    return cleaned or "case"


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
        "（README §5.3 终裁不给平局）；\n"
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


#: 夜航预授权标记（PSEP_2 §3.2 授权书；写进判断/裁决留痕供用户事后复核）。
NIGHT_AUTHORITY: str = "夜航预授权"


def parse_ruling(body: str) -> dict[str, Any]:
    """从**裁决**交付正文抽 ``{choice, reason, confidence}``——choice 为**自由文本**
    （consult 案卷是自由形式，没有 A/B 选项；盲审二选一仍走 :func:`parse_judgement`）。

    与 :func:`parse_judgement` 同钳制精神：confidence 越界钳 0–1、非数置 None；抽不到即
    ``parsed=False``（不猜——宁可让人看出这条没读出来）。
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
    out["choice"] = str(data.get("choice", "") or "").strip()
    out["reason"] = str(data.get("reason", "") or "").strip()
    try:
        out["confidence"] = min(1.0, max(0.0, float(data.get("confidence"))))
    except (TypeError, ValueError):
        out["confidence"] = None
    out["parsed"] = True
    return out


def _night_marker() -> dict[str, Any]:
    """夜航标记：开关 ON 时记授权依据（用户复核看的就是它）；读开关失败按常规时段。"""
    try:
        from .settings import night_enabled

        on, _warnings = night_enabled()
    except Exception:  # noqa: BLE001 - 读开关失败按非夜航（标记缺席=常规时段，保守）
        on = False
    return {"night": on, "authority": NIGHT_AUTHORITY if on else ""}


def _journal_ruling(record: dict[str, Any], options_text: str) -> None:
    """裁决入**单一账目源** preference-journal（channel=twin + case_id + 夜航标记）。

    「单一账目源」纪律（§纪律「同一能力不留两份实现」）：判决流水只此一处；taste/jury/
    post-verify 各面都在此落行，读数无需第二本账（verdict.py 同口径）。写失败 fail-soft
    ——记录文件（``evals/taste/<case_id>.json``）才是硬留痕，账目面不炸判断。
    """
    try:
        from . import journal

        journal.append(
            {
                "ts": str(record.get("ts") or ""),
                "situation": f"twin 裁决 {record.get('case_id')}（{record.get('case_file') or '?'}）",
                "options": options_text,
                "choice": str(record.get("choice") or ""),
                "reason": str(record.get("reason") or ""),
                "source": "twin",
                "kind": "ruling",
                "channel": "twin",
                "case_id": str(record.get("case_id") or ""),
                "confidence": record.get("confidence"),
                "night": bool(record.get("night")),
            }
        )
    except Exception:  # noqa: BLE001 - 账目写失败不炸判断（记录文件仍是硬留痕）
        pass


def _run_case_judgement(
    case: Case,
    *,
    reg: Registry,
    pref: str,
    member_id: str,
    model_tier: str | None,
    quiet: bool,
    record_extra: dict[str, Any] | None = None,
    journal_options: str = "A/B（盲审二选一）",
) -> tuple[int, dict[str, Any]]:
    """跑一次 twin 判断跳并落留痕（:func:`judge` 与 :func:`ruling_for_case` 共用唯一实现）。"""
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
        "case_file": str(case.path or ""),
        "case_head": (case.question or "").strip()[:120],  # 案卷摘要（审计面要素）
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
        "record_kind": "taste",
        **_night_marker(),
        **(record_extra or {}),
    }
    p = write_record(record)
    _journal_ruling(record, journal_options)
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


def judge(
    case_file: Path | str,
    *,
    member_id: str = "twin",
    model_tier: str | None = None,
    quiet: bool = False,
) -> tuple[int, dict[str, Any]]:
    """执行一次判断跳并落留痕（记录 + **单一账目源** journal；夜航时带预授权标记）。

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
    return _run_case_judgement(
        case,
        reg=reg,
        pref=pref,
        member_id=member_id,
        model_tier=model_tier,
        quiet=quiet,
    )


def ruling_for_case(
    case: Case,
    *,
    member_id: str = "twin",
    model_tier: str | None = None,
    quiet: bool = True,
    origin: str = "jury-escalate",
) -> tuple[int, dict[str, Any]]:
    """把 twin 判断跑在**已解析**案卷上（jury E6 平票终裁复用；同机制、同留痕面）。

    与 :func:`judge` 的唯一差别：入参是 :class:`~.jury.Case`（jury 已解析），且**默认静默**
    ——输出由 jury 统一呈现；留痕（``evals/taste/<case>.json`` + journal）与 judge 同形态。

    Returns:
        ``(code, record)``：0=已表态（record 含 choice/confidence/reason）；1=派发失败或读不出
        选择；2=不可用（taste_enabled=false / 宪章缺失）——此时 record 带 ``available=False``
        与 ``reason``，供 jury 保守回退（不静默放行）。
    """
    reg = Registry.load()
    if not reg.taste_enabled:
        return 2, {"available": False, "reason": DISABLED_HINT}
    pref = preference_text(reg, member_id)
    if not pref.strip():
        return 2, {
            "available": False,
            "reason": f"宪章缺失：{member_id} pod 内无 {PREFERENCE_NAME}",
        }
    return _run_case_judgement(
        case,
        reg=reg,
        pref=pref,
        member_id=member_id,
        model_tier=model_tier,
        quiet=quiet,
        record_extra={"origin": origin},
        journal_options="A/B（jury 平票 → twin 终裁）",
    )


def write_record(record: dict[str, Any]) -> Path:
    """留痕落盘（原子写；目录自动建）。"""
    p = record_path(str(record["case_id"]))
    _atomic_write_json(p, record)
    return p


# ---------------------------------------------------------------------------
# 追溯与复核（PSEP_2 §1 配套要求/§5；task-20261014-psep2-n3）
# ---------------------------------------------------------------------------
def _load_records() -> list[dict[str, Any]]:
    """读全部留痕记录（taste 盲审 + consult 夜航代裁**同读一目录**；坏文件跳过）。"""
    rows: list[dict[str, Any]] = []
    try:
        paths = sorted(TASTE_DIR.glob("*.json"))
    except OSError:
        return rows
    for p in paths:
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(d, dict) and d.get("case_id"):
            rows.append(d)
    return rows


def twin_rulings(limit: int = 10) -> list[dict[str, Any]]:
    """twin 判断/裁决留痕（按时间倒序；``limit<=0`` 全量）。"""
    rows = _load_records()
    rows.sort(key=lambda r: str(r.get("ts") or ""), reverse=True)
    return rows if limit <= 0 else rows[:limit]


def pending_review_count() -> int:
    """未复核（记录无 ``review``）的裁决数——用户复核链路的待办读数。"""
    return sum(1 for r in _load_records() if not r.get("review"))


# ---------------------------------------------------------------------------
# 盲审反转流程（2_REFACTOR §10：隐去 twin 答案逐案盲审 → 偏差报告入强制处理队列）
# ---------------------------------------------------------------------------
def blind_queue() -> list[dict[str, Any]]:
    """待盲审队列：无 ``review`` 字段的记录，按 ``ts`` 升序（TUI 盲审逐案呈现用）。

    每项 ``{case_id, ts, case_file, question, option_a, option_b, options, evidence,
    case_head}``：案情复用 :func:`jury.parse_case`；``options``＝案卷**全部**选项
    ``[(字母, 全文)]``（三选项 consult 案卷全量呈现，2026-10-17 实案 C 选项曾整块
    不可见）。案卷缺失/不可解析（含 consult 自由形式案卷）**分别**出声：缺失给路径、
    不可解析附原文摘录（案卷内容非 twin 答案，盲审可看）——**可审不崩**。
    **绝不含** twin 答案（``choice``/``reason``/``body``/``confidence``/``parsed``）：
    先隐去答案让用户自选，提交时的一致性比对走 :func:`blind_case`。
    """
    rows = [r for r in _load_records() if not r.get("review")]
    rows.sort(key=lambda r: str(r.get("ts") or ""))
    out: list[dict[str, Any]] = []
    for r in rows:
        case_file = str(r.get("case_file") or "")
        question = option_a = option_b = evidence = ""
        options: list[tuple[str, str]] = []
        try:
            case = parse_case(case_file)
            question = case.question.strip()
            option_a = case.option_a.strip()
            option_b = case.option_b.strip()
            evidence = case.evidence.strip()
            options = [(lbl, txt) for lbl, txt in case.options]
        except OSError:
            question = str(r.get("case_head") or "").strip() or "（无案情摘要）"
            evidence = f"（案卷缺失：{case_file or '未记录路径'}）"
        except ValueError:
            question = str(r.get("case_head") or "").strip() or "（无案情摘要）"
            raw = ""
            try:
                raw = Path(case_file).read_text(encoding="utf-8")[:2000]
            except OSError:
                raw = ""
            evidence = f"（案卷存在但不可解析：{case_file}）\n{raw}".rstrip()
        out.append(
            {
                "case_id": str(r.get("case_id") or ""),
                "ts": str(r.get("ts") or ""),
                "case_file": case_file,
                "question": question,
                "option_a": option_a,
                "option_b": option_b,
                "options": options,
                "evidence": evidence,
                "case_head": str(r.get("case_head") or ""),
            }
        )
    return out


def choice_letter(choice: str) -> str:
    """自由文本裁示 → 选项字母（``C（双工作点并列）：…``→``C``、``选项 B``→``B``）。

    盲审一致性比对用：consult 裁示常是整句，与用户 ``A``/``B``/``C`` 选择**逐字**
    比对永不一致（实案 tb-20261017-044604-3-2：裁示以「C（…）」开头）。取不出 → 空串。
    """
    s = str(choice or "").strip()
    if not s:
        return ""
    m = re.match(r"^选项\s*([A-Za-z])", s)
    if m:
        return m.group(1).upper()
    if s[0].isalpha() and (len(s) == 1 or not s[1].isalnum()):
        return s[0].upper()
    return ""


def blind_case(case_id: str) -> dict[str, Any] | None:
    """完整留痕记录（**含 twin 答案**）——盲审提交时的一致性比对面；无此 id → ``None``。"""
    for r in _load_records():
        if str(r.get("case_id")) == str(case_id):
            return r
    return None


def ruling_rows(limit: int = 0) -> list[dict[str, Any]]:
    """已复核（记录含 ``review``）的裁决行，新→旧；``limit<=0`` 全量。

    每行 ``{case_id, ts, choice, confidence, reason, review, night, authority,
    case_head}``——待决/验收页与审计视图的结构化读数。
    """
    rows = [r for r in _load_records() if r.get("review")]
    rows.sort(key=lambda r: str(r.get("ts") or ""), reverse=True)
    out: list[dict[str, Any]] = []
    for r in rows if limit <= 0 else rows[:limit]:
        out.append(
            {
                "case_id": str(r.get("case_id") or ""),
                "ts": str(r.get("ts") or ""),
                "choice": str(r.get("choice") or ""),
                "confidence": r.get("confidence"),
                "reason": str(r.get("reason") or ""),
                "review": r.get("review"),
                "night": bool(r.get("night")),
                "authority": str(r.get("authority") or ""),
                "case_head": str(r.get("case_head") or ""),
            }
        )
    return out


def ruling_lines(limit: int = 8) -> list[str]:
    """审计面逐条呈现（要素：时间/类型/案卷 id/选择/置信度/理由/夜航预授权标记/复核态）。"""
    rows = twin_rulings(limit)
    if not rows:
        return ["（无 twin 判断/裁决留痕）"]
    out: list[str] = []
    for r in rows:
        conf = "-"
        try:
            if r.get("confidence") not in (None, ""):
                conf = f"{float(r['confidence']):.2f}"
        except (TypeError, ValueError):
            conf = "-"
        tags: list[str] = []
        if r.get("night"):
            tags.append(str(r.get("authority") or NIGHT_AUTHORITY))
        rev = r.get("review") or {}
        tags.append(f"已复核:{rev.get('verdict')}" if rev else "待复核")
        out.append(
            f"  {str(r.get('ts'))[:19]} [{r.get('record_kind') or 'taste'}] "
            f"{r.get('case_id')}：{r.get('choice') or '（未读出）'}（置信度 {conf}）"
            f"〔{'；'.join(tags)}〕"
        )
        head = str(r.get("case_head") or "").strip()
        if head:
            out.append(f"      案卷：{head[:100]}")
        reason = str(r.get("reason") or "").strip()
        if reason:
            out.append(f"      理由：{reason[:120]}")
    return out


def apply_review(
    case_id: str,
    *,
    verdict: str,
    note: str = "",
    blind: dict[str, Any] | None = None,
) -> tuple[int, str]:
    """用户复核一次 twin 判断/裁决（``confirmed``/``rejected``）→ 记录留痕 + 单一账目记账。

    ``blind``（盲审反转流程 §10 的批复载荷，可选——旧调用不传照常工作）：
    ``{user_choice, agree: bool, remedy: "restore"|"none"|"other"|None, requirements: str}``
    ——原样并入 ``review`` 落盘；不一致时的偏差报告由调用方（TUI）另经
    :func:`mirror.append_deviation` 入强制处理队列，本函数只管留痕。

    否决另入 journal（``kind="override"``、reason=批语）——后续改进（twin 语料/宪章修订）
    循此追溯；确认入 ``kind="verdict"``。账目写失败 fail-soft（review 已落进记录文件）。
    """
    path = record_path(case_id)
    if not path.exists():
        return 2, f"[!] 无此裁决留痕：{_rel(path)}（orch twin review --list 看现有）"
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return 2, f"[!] 留痕不可读：{exc}"
    if not isinstance(record, dict):
        return 2, "[!] 留痕格式异常（非对象）"
    record["review"] = {
        "ts": _now(),
        "verdict": verdict,
        "note": note,
        "blind": blind,
    }
    _atomic_write_json(path, record)
    try:
        from . import journal

        journal.append(
            {
                "ts": _now(),
                "situation": f"用户复核 twin 裁决 {case_id}",
                "options": "",
                "choice": verdict,
                "reason": note,
                "source": "user",
                "kind": "override" if verdict == "rejected" else "verdict",
                "channel": "twin",
                "case_id": case_id,
                "night": bool(record.get("night")),
            }
        )
    except Exception:  # noqa: BLE001 - 账目写失败不炸复核（记录里 review 已落）
        pass
    label = "确认" if verdict == "confirmed" else "否决"
    tail = f"；批语：{note[:80]}" if note else ""
    return 0, f"[√] 已{label}：{case_id}（{_rel(path)}{tail}）"


def cmd_twin_review(args: argparse.Namespace) -> int:
    """``orch twin review``：复核入口——``--list`` 看最近裁决；``--confirm|--reject`` 留痕。"""
    case_id = str(getattr(args, "case_id", "") or "")
    if getattr(args, "list", False) or not case_id:
        for line in ruling_lines(limit=int(getattr(args, "limit", 8) or 8)):
            print(line)
        print(
            f"  待用户复核 = {pending_review_count()}"
            '（复核：orch twin review <case_id> --confirm|--reject [--note "…"]）'
        )
        return 0
    if not (getattr(args, "confirm", False) or getattr(args, "reject", False)):
        print("[!] 需 --confirm 或 --reject（或 --list 查看）")
        return 2
    rc, msg = apply_review(
        case_id,
        verdict="confirmed" if getattr(args, "confirm", False) else "rejected",
        note=str(getattr(args, "note", "") or ""),
    )
    print(msg)
    return rc


def cmd_taste(args: argparse.Namespace) -> int:
    """``orch twin taste <case-file> [--dry] [--model-tier …]``——twin 判断单跳。

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
    """注册 ``orch twin taste`` 子命令（由 orch.py 门面调用，保持 CLI 装配单点）。"""
    p = sub.add_parser(
        "taste",
        help="twin 审美孪生判断（README §1/§3.1）：宪章注入 + 冷启动单跳 → {选择,理由,置信度}",
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
