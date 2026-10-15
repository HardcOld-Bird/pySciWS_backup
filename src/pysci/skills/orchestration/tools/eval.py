"""Tier B harness 测试套件执行面（README §5.5；E2，backlog 20261011-233002-leader）。

**为什么需要它**：§0.5 要「二元判决」，§6 要「可证伪成功判据」——而组长侧过去的
判据都是散文（「跑通了」「看着更好」）。本模块把套件案变成**可重跑的确定性测量**：
同一案、全新会话、固定 ``minimum`` 档、不附审批回复（与生产跳隔离），跑完用案卷自带的
机械判据出 **通过/失败**，指标行入 ``state/ledger/evals.jsonl``；与
``state/evals/baseline.json`` 对照出 **改善/未改善**。§0.5 硬要求无平局。

设计纪律（§0.9）：**不建独立 CLI**，``orch eval run`` / ``orch eval set-baseline``
是本库的封装面；执行一跳**复用** :func:`dispatch.do_dispatch`，不在这里重复派发链。

案卷位置 ``state/evals/harness/<case-id>.md``（所有权归 bencher，§3；E4 前由 devops
按历史真实任务提炼首批案）。案卷头部是 ``key: value`` 标记行（同建议文件的解析约定，
见 :func:`delivery.parse_suggestion_fields`），``## 任务书`` 之后整段是派发正文。
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from pysci.paths import ORCH_STATE_ROOT, PROJECT_ROOT

from . import procs
from .checks import run_checks
from .delivery import Artifact
from .dispatch import do_dispatch
from .registry import Registry, dirty_tracked_state

#: 套件案目录（bencher 完全写权，§3）。
HARNESS_DIR: Path = ORCH_STATE_ROOT / "evals" / "harness"
#: 压测台账（与生产台账 ``deliveries.jsonl`` 分文件：指标口径不同，且不许污染成员排名）。
EVAL_LEDGER_PATH: Path = ORCH_STATE_ROOT / "ledger" / "evals.jsonl"
#: 基线快照（``orch eval set-baseline`` 写入，``orch eval run`` 读取对照）。
BASELINE_PATH: Path = ORCH_STATE_ROOT / "evals" / "baseline.json"

#: 压测默认档（§1.1：harness 压测走成本最低档；案卷可用 ``tier:`` 行覆盖）。
DEFAULT_TIER: str = "minimum"
#: 台账行的发起方取值（E4 所有权移交的审计面）：组长抽检 vs bencher 专测。
RUNNER_LEADER: str = "leader"
RUNNER_BENCHER: str = "bencher"
#: 旧行无 ``runner`` 字段时的读法——移交前只有组长这一条通道。
RUNNER_DEFAULT: str = RUNNER_LEADER
#: 案卷缺省预算（压测要**便宜且可重跑**：轮次上限小、超时短，跑不起的套件等于没有）。
DEFAULT_MAX_TURNS: int = 12
DEFAULT_TIMEOUT_S: int = 900

_BODY_MARK = "## 任务书"
_KEY_RE = re.compile(r"^(?P<k>[\w\u4e00-\u9fff_]+)\s*:\s*(?P<v>.*?)\s*$", re.MULTILINE)
_ARTIFACT_RE = re.compile(r"^(?P<path>[^\s]+)(?:\s+check=(?P<check>\S+))?$")

#: 已知头部键（其余键一律报错——静默忽略拼错的键，等于判据没执行却没人在看）。
_KNOWN_KEYS = {
    "member",
    "tier",
    "max_turns",
    "timeout_s",
    "dirs",
    "产物",
    "断言",
    "断言于",
    "禁止标签",
    "权限拒绝上限",
    "说明",
}

#: 允许重复出现的键（一份案可声明多个产物）。
_LIST_KEYS = {"产物"}


@dataclass
class Case:
    """一份套件案（案卷 = 任务书 + 白名单 + 机械判据 + 预算，§3）。"""

    case_id: str
    member: str
    brief: str
    tier: str = DEFAULT_TIER
    dirs: list[str] = field(default_factory=list)
    max_turns: int = DEFAULT_MAX_TURNS
    timeout_s: int = DEFAULT_TIMEOUT_S
    artifacts: list[Artifact] = field(default_factory=list)
    expect_re: str = ""
    expect_on: str = ""  # 断言面：空=交付正文；否则为项目根相对的产物文件路径
    forbid_tags: list[str] = field(default_factory=list)
    max_denials: int = 0
    note: str = ""
    path: str = ""


@dataclass
class Row:
    """一次压测的台账行（``evals.jsonl`` 的一行）。"""

    ts: str
    case: str
    member: str
    tier: str
    model: str
    verdict: str  # pass | fail
    reasons: list[str] = field(default_factory=list)
    checks: list[dict[str, Any]] = field(default_factory=list)
    num_turns: int = 0
    duration_ms: int = 0
    est_tokens: int = 0
    delta_chars: int = 0
    ctx_ratio: float = 0.0
    credits: float = 0.0
    sid: str = ""
    task_file: str = ""
    label: str = ""
    #: 案内轮次预算（``预算用满`` 哨兵的对照面：turns 用到 == 此值的 pass 不稳——
    #: E2 验收实测 blocked-recheck 4→10 轮跳，正是贴着上限跑完的）。
    max_turns: int = 0
    #: 本跳污染了组员跟踪面（如自己的 ``AGENTS.md``）并被还原的路径——非空即说明该轮的
    #: 「组员自维护层」被压测跳过，噪声已隔离但要在台账可见。
    reverted: list[str] = field(default_factory=list)
    #: 谁发起的这次测量（E4 所有权移交的审计面）：``leader``=组长抽检 ``orch eval run``，
    #: ``bencher``=bencher 经 ``pysci-bench``。旧行无该字段读作 ``leader``（缺席即移交前
    #: 的历史数据，不必回填——回填等于伪造当时的发起方）。
    runner: str = "leader"


# ---------------------------------------------------------------------------
# 案卷解析
# ---------------------------------------------------------------------------
def _head_and_body(text: str) -> tuple[str, str]:
    """按 ``## 任务书`` 切案卷为（头部标记区, 派发正文）。"""
    if _BODY_MARK not in text:
        raise ValueError(f"案卷缺「{_BODY_MARK}」小节（派发正文无处安放）")
    head, body = text.split(_BODY_MARK, 1)
    return head, body.strip()


def parse_case(path: Path | str) -> Case:
    """读一份案卷，返回 :class:`Case`。

    Args:
        path: 案卷文件（``state/evals/harness/<case-id>.md``）。

    Raises:
        ValueError: 案卷形状不合法——缺 ``member``、缺任务书正文、出现未知键、
            或数值键不是正整数。判据写错必须当场失败，不能变成「静默少测一项」。
    """
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    head, body = _head_and_body(text)
    keys: dict[str, list[str]] = {}
    for m in _KEY_RE.finditer(head):
        k, v = m.group("k"), m.group("v")
        if k in _LIST_KEYS:
            keys.setdefault(k, []).append(v)
        elif k in keys:
            raise ValueError(f"案卷 {p.name} 键 {k}: 重复（已有 {keys[k][0]!r}）")
        else:
            keys[k] = [v]
    unknown = sorted(set(keys) - _KNOWN_KEYS)
    if unknown:
        raise ValueError(f"案卷 {p.name} 出现未知头部键：{unknown}")
    if not keys.get("member", [""])[0]:
        raise ValueError(f"案卷 {p.name} 缺 member:（压测要指明派给哪个 pod）")
    if not body:
        raise ValueError(f"案卷 {p.name} 的「{_BODY_MARK}」正文为空")

    artifacts: list[Artifact] = []
    for raw in keys.get("产物", []):
        am = _ARTIFACT_RE.match(raw)
        if not am:
            raise ValueError(
                f"案卷 {p.name} 的产物行不可解析：{raw!r}（应为 路径 [check=id]）"
            )
        artifacts.append(
            Artifact(path=am.group("path"), check=am.group("check") or "none")
        )

    def _pos(key: str, default: int) -> int:
        val = keys.get(key, [""])[0]
        if not val:
            return default
        try:
            n = int(val)
        except ValueError as exc:
            raise ValueError(f"案卷 {p.name} 的 {key}: 不是整数：{val!r}") from exc
        if n < 0:
            raise ValueError(f"案卷 {p.name} 的 {key}: 不得为负：{n}")
        return n

    expect_re = keys.get("断言", [""])[0]
    if expect_re:
        try:
            re.compile(expect_re)
        except re.error as exc:
            raise ValueError(f"案卷 {p.name} 的 断言: 不是合法正则：{exc}") from exc

    return Case(
        case_id=p.stem,
        member=keys["member"][0],
        brief=body,
        tier=keys.get("tier", [""])[0] or DEFAULT_TIER,
        dirs=[
            d.strip() for d in re.split(r"[,;]", keys.get("dirs", [""])[0]) if d.strip()
        ],
        max_turns=_pos("max_turns", DEFAULT_MAX_TURNS),
        timeout_s=_pos("timeout_s", DEFAULT_TIMEOUT_S),
        artifacts=artifacts,
        expect_re=expect_re,
        expect_on=keys.get("断言于", [""])[0],
        forbid_tags=[
            t
            for t in re.split(r"[,;]", keys.get("禁止标签", [""])[0].replace("，", ","))
            if t.strip()
        ],
        max_denials=_pos("权限拒绝上限", 0),
        note=keys.get("说明", [""])[0],
        path=str(p),
    )


def load_cases(case_id: str = "") -> list[Case]:
    """读套件案；``case_id`` 为空则按文件名序读全部。

    Raises:
        ValueError: 目录不存在/无案可跑/指定案不存在（用法层错误，交给 CLI 打印）。
    """
    if not HARNESS_DIR.exists():
        raise ValueError(f"套件目录不存在：{HARNESS_DIR}（先落 ≥1 份案卷）")
    files = sorted(HARNESS_DIR.glob("*.md"))
    if case_id:
        files = [f for f in files if f.stem == case_id]
        if not files:
            raise ValueError(
                f"案卷 {case_id} 不在 {HARNESS_DIR}（现有：{[f.stem for f in HARNESS_DIR.glob('*.md')]}）"
            )
    if not files:
        raise ValueError(f"套件目录无案卷：{HARNESS_DIR}/*.md")
    return [parse_case(f) for f in files]


# ---------------------------------------------------------------------------
# 执行与判据
# ---------------------------------------------------------------------------
def resolve_artifact(art: Artifact, member_pod: Path) -> Path:
    """案卷里的产物路径 → 绝对路径（先按项目根、再按该成员 pod，同 :mod:`checks` 口径）。"""
    p = Path(art.path)
    if p.is_absolute():
        return p
    cand_root = PROJECT_ROOT / p
    return cand_root if cand_root.exists() else member_pod / p


def judge(
    out: Any, case: Case, reg: Registry, member_pod: Path
) -> tuple[list[str], list[dict[str, Any]]]:
    """对一跳结果执行案内机械判据，返回 ``(未满足项列表, 各判据读数)``（空列表=通过）。

    Args:
        out: :func:`dispatch.do_dispatch` 的 :class:`~dispatch.DispatchOutcome`。
        case: 案卷。
        reg: 注册表（checks 命令模板来源；由调用方 load 一次，别在这里反复读盘）。
        member_pod: 该案成员 pod 的绝对路径（相对产物路径的第二解析基准）。
    """
    reasons: list[str] = []
    checks: list[dict[str, Any]] = []
    if out.kind != "result":
        reasons.append(f"交付={out.kind}（期望 result）")
    bad_tags = [t for t in case.forbid_tags if t in (out.tags or [])]
    if bad_tags:
        reasons.append(f"出现禁止标签 {bad_tags}（本案的失败类=历史卡点）")
    denials = len(out.envelope.permission_denials) if out.envelope else 0
    if denials > case.max_denials:
        reasons.append(f"权限拒绝 {denials} 次 > 上限 {case.max_denials}")
    if case.artifacts:
        results = run_checks(case.artifacts, reg, member_pod=member_pod)
        for art, res in zip(case.artifacts, results, strict=False):
            checks.append(res)
            abs_p = resolve_artifact(art, member_pod)
            if not abs_p.exists():
                reasons.append(f"产物缺失：{art.path}")
            elif res["verdict"] == "fail":
                reasons.append(f"机械判据 FAIL：{art.path} [{art.check}]")
            elif res["verdict"] == "unrouted":
                reasons.append(f"判据未注册：{art.check}（registry.checks 无此 id）")
    if case.expect_re:
        hay = out.body or ""
        where = "交付正文"
        if case.expect_on:
            f = PROJECT_ROOT / case.expect_on
            if not f.exists():
                reasons.append(f"断言面文件不存在：{case.expect_on}")
                return reasons, checks
            hay = f.read_text(encoding="utf-8", errors="replace")
            where = case.expect_on
        if not re.search(case.expect_re, hay, re.IGNORECASE):
            reasons.append(f"{where}不含断言 {case.expect_re!r}")
    return reasons, checks


#: 压测跳的隔离纪律（附在案内任务书之后，不进生产语义）。
#: E2 验收实测：压测跳会把「这一跳学到的东西」写进组员自己的 ``AGENTS.md``
#: （git 跟踪面）——测量噪声变成合并前检查的脏面，正是 #58 那一类事故。
EVAL_HOP_NOTE = (
    "\n\n---\n\n**本跳是 harness 压测，不是生产任务**：只完成任务书描述的动作。"
    "**不要**更新你的 `AGENTS.md`、不要新建/修改自建 rules 或 skills、不要写 pod "
    "与案内白名单之外的任何文件——测量要可比，组员长期记忆由生产任务负责维护。"
)


#: pod 相对路径根（同 ``registry.STATE_REL_PATH`` 口径：git 只认项目根相对路径）。
_PODS_REL: str = "orchestration/pods"


def pod_tracked_dirt(member_id: str) -> set[str]:
    """该组员 pod 下 **git 跟踪且已脏** 的文件集合——压测跳前拍一张快照。

    为什么压测要拍它（E2 验收实测）：压测跳会让组员把「这一跳学到的东西」写进自己的
    ``AGENTS.md``（跟踪面）。噪声一旦留在树上，合并前检查就出现与本单无关的脏面，且下一轮
    测量起点与上一轮不同。判据是「本跳新增的脏面」，故先记下跳前已有的脏面。
    """
    return set(dirty_tracked_state(path=f"{_PODS_REL}/{member_id}"))


def revert_new_dirt(member_id: str, before: set[str]) -> list[str]:
    """把本跳**新弄脏**的跟踪文件还原到 HEAD，返回实际还原的路径。

    只还原 ``after - before``：跳前就脏的文件可能是组长在飞的工作，动它=覆盖他人改动。
    范围再钉一层「必须落在该 pod 前缀下」——这是 ``git checkout --``，作用面写错就是破坏性。
    新增的**未跟踪**文件不删（删文件风险高于留一份说明）。

    fail-open：git 缺失/超时/非零退出一律返回 ``[]``，测量噪声不该掀翻一次压测。
    """
    pod_rel = f"{_PODS_REL}/{member_id}"
    after = set(dirty_tracked_state(path=pod_rel))
    fresh = sorted(
        p for p in after - before if p.replace("\\", "/").startswith(f"{pod_rel}/")
    )
    if not fresh:
        return []
    try:
        proc = procs.run_hidden(
            ["git", "checkout", "--", *fresh],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30,
            check=False,
        )
    except Exception:
        return []
    return fresh if proc.returncode == 0 else []


def run_case(
    case: Case,
    *,
    tier: str = "",
    label: str = "",
    quiet: bool = True,
    runner: str = "leader",
) -> Row:
    """跑一案：全新会话 + 固定档位 + 不附审批回复 + 不进生产台账 → 判据 → 台账行。

    隔离面：任务书后附 :data:`EVAL_HOP_NOTE`（禁止本跳写组员自维护层），并用
    :func:`pod_tracked_dirt` / :func:`revert_new_dirt` 夹住这一跳——组员在本跳新弄脏的
    **跟踪**文件在判据前还原，还原掉的路径记进 ``Row.reverted``。跳前就脏的文件不动。

    Args:
        case: :func:`parse_case` 的产物。
        tier: 覆盖案卷档位（CLI ``--model-tier``）；空则用案卷的 ``tier:``。
        label: 本轮标签（如 ``before``/``after-<commit>``），入台账供分组对照。
        quiet: 抑制 do_dispatch 的面向组长报告（压测要的是行，不是 [NEXT] 提示）。
        runner: 发起方（``leader``/``bencher``），入台账作审计面——E4 后套件运行职责
            归 bencher，组长的 ``orch eval run`` 只是抽检通道；两条通道的读数必须能
            在台账里分开（否则「谁测的」这一事实随移交一起消失）。

    Returns:
        :class:`Row`（同时已 append 进 ``evals.jsonl``）。
    """
    tier = tier or case.tier
    before = pod_tracked_dirt(case.member)
    out = do_dispatch(
        case.member,
        text=case.brief + EVAL_HOP_NOTE,
        slug=f"eval-{case.case_id}",
        session="new",
        model_tier=tier,
        max_turns=case.max_turns,
        timeout=case.timeout_s,
        dirs=list(case.dirs),
        no_checks=True,
        attach_replies=False,
        eval_hop=True,
        quiet=quiet,
    )
    reverted = revert_new_dirt(case.member, before)
    reg = Registry.load()
    reasons, checks = judge(out, case, reg, reg.member(case.member).pod)
    env = out.envelope
    row = Row(
        ts=datetime.now().astimezone().isoformat(timespec="seconds"),
        case=case.case_id,
        member=case.member,
        tier=tier,
        model=str(reg.models.get(tier, "")),
        verdict="fail" if reasons else "pass",
        reasons=reasons,
        checks=checks,
        num_turns=env.num_turns if env else 0,
        duration_ms=env.duration_ms if env else 0,
        est_tokens=out.est_tokens,
        delta_chars=out.delta_chars,
        ctx_ratio=env.context_usage_ratio if env else 0.0,
        credits=env.total_credits if env else 0.0,
        sid=out.sid,
        task_file=out.task_file or "",
        label=label,
        max_turns=case.max_turns,
        reverted=reverted,
        runner=runner,
    )
    append_row(row)
    return row


# ---------------------------------------------------------------------------
# 台账与基线
# ---------------------------------------------------------------------------
def append_row(row: Row) -> None:
    """压测行追加进 ``state/ledger/evals.jsonl``（同 deliveries.jsonl 的 append-only 口径）。"""
    EVAL_LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with EVAL_LEDGER_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")


def read_rows() -> list[dict[str, Any]]:
    """读全部压测行（坏行 fail-open 跳过——台账不许因一次截断而不可读）。"""
    if not EVAL_LEDGER_PATH.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in EVAL_LEDGER_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            out.append(obj)
    return out


def latest_rows(label: str = "", runner: str = "") -> dict[str, dict[str, Any]]:
    """每案取最近一行（可选按 ``label``/``runner`` 过滤）——基线与本轮对照都用这个口径。

    ``runner`` 过滤是 E4 移交后的必要读法：组长抽检与 bencher 专测混在同一台账里，
    对照时把别人的读数算进本轮，判决就成了两条通道的平均数。旧行无 ``runner`` 字段，
    按 :data:`RUNNER_DEFAULT` 读（缺席=移交前由组长侧发起）。
    """
    out: dict[str, dict[str, Any]] = {}
    for r in read_rows():
        if label and r.get("label", "") != label:
            continue
        if runner and str(r.get("runner") or RUNNER_DEFAULT) != runner:
            continue
        if r.get("case"):
            out[str(r["case"])] = r
    return out


def write_baseline(label: str = "", runner: str = "") -> dict[str, Any]:
    """以最近一轮（可按 ``label``/``runner`` 选）回填 ``baseline.json``，返回写入内容。"""
    rows = latest_rows(label, runner)
    if not rows:
        raise ValueError(
            "evals.jsonl 无可回填的压测行——先跑 `orch eval run --all`（或 --case）"
        )
    doc = {
        "version": 1,
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "label": label,
        "cases": {
            cid: {
                "verdict": r.get("verdict", "fail"),
                "num_turns": int(r.get("num_turns") or 0),
                "duration_ms": int(r.get("duration_ms") or 0),
                "est_tokens": int(r.get("est_tokens") or 0),
                "ctx_ratio": float(r.get("ctx_ratio") or 0.0),
                "model": r.get("model", ""),
                "ts": r.get("ts", ""),
            }
            for cid, r in sorted(rows.items())
        },
    }
    BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = BASELINE_PATH.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    tmp.replace(BASELINE_PATH)
    return doc


def read_baseline() -> dict[str, Any]:
    """读基线快照；缺席或损坏返回空 dict（首轮没有对照面，不是错误）。"""
    if not BASELINE_PATH.exists():
        return {}
    try:
        doc = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return doc if isinstance(doc, dict) else {}


def compare(
    rows: dict[str, dict[str, Any]], baseline: dict[str, Any]
) -> dict[str, Any]:
    """本轮 vs 基线的二元判决（§0.5 无平局）。

    规则（首版机械规则，可调）：**改善** 当且仅当
    ① 通过数不减、② 均值 ``num_turns`` 不劣化、③ 至少一项严格更优。
    否则 **未改善**（含完全持平——持平不奖励无变化的改动）。

    Returns:
        ``{verdict, pass_now, pass_base, mean_turns_now, mean_turns_base,
        compared, only_now, diffs}``。无基线时 verdict=``无基线``。
    """
    base_cases: dict[str, Any] = baseline.get("cases", {}) if baseline else {}
    shared = sorted(set(rows) & set(base_cases))
    if not shared:
        return {
            "verdict": "无基线",
            "pass_now": sum(1 for r in rows.values() if r.get("verdict") == "pass"),
            "pass_base": 0,
            "mean_turns_now": _mean_turns(list(rows.values())),
            "mean_turns_base": 0,
            "compared": [],
            "only_now": sorted(set(rows) - set(base_cases)),
            "diffs": [],
        }
    now_pass = sum(1 for c in shared if rows[c].get("verdict") == "pass")
    base_pass = sum(1 for c in shared if base_cases[c].get("verdict") == "pass")
    now_turns = _mean_turns([rows[c] for c in shared])
    base_turns = _mean_turns([base_cases[c] for c in shared])
    no_worse_pass = now_pass >= base_pass
    no_worse_turns = now_turns <= base_turns
    strict = (now_pass > base_pass) or (now_turns and now_turns < base_turns)
    verdict = "改善" if (no_worse_pass and no_worse_turns and strict) else "未改善"
    diffs = [
        {
            "case": c,
            "turns": f"{base_cases[c].get('num_turns', 0)} → {rows[c].get('num_turns', 0)}",
            "verdict": f"{base_cases[c].get('verdict', '-')} → {rows[c].get('verdict', '-')}",
        }
        for c in shared
    ]
    return {
        "verdict": verdict,
        "pass_now": now_pass,
        "pass_base": base_pass,
        "mean_turns_now": round(now_turns, 2),
        "mean_turns_base": round(base_turns, 2),
        "compared": shared,
        "only_now": sorted(set(rows) - set(base_cases)),
        "diffs": diffs,
    }


def _mean_turns(rows: list[dict[str, Any]]) -> float:
    """非零 ``num_turns`` 的均值（全为 0 时返回 0，不参与「更优」判定，防除零式假改善）。"""
    vals = [
        int(r.get("num_turns") or 0) for r in rows if int(r.get("num_turns") or 0) > 0
    ]
    return statistics.fmean(vals) if vals else 0.0


# ---------------------------------------------------------------------------
# CLI 面（orch eval …）
# ---------------------------------------------------------------------------
def _print_row(row: Row) -> None:
    mark = "√" if row.verdict == "pass" else "✗"
    budget = f"/{row.max_turns}" if row.max_turns else ""
    print(
        f"  [{mark}] {row.case:<18} {row.verdict:<4} "
        f"turns={row.num_turns}{budget} dur={row.duration_ms / 1000:.0f}s "
        f"est_tokens={row.est_tokens} ctx={row.ctx_ratio:.4f} "
        f"model={row.model or '(默认)'}"
    )
    for r in row.reasons:
        print(f"        · {r}")
    if row.max_turns and row.num_turns >= row.max_turns:
        print(
            f"        [!] 预算用满（turns {row.num_turns}==max_turns "
            f"{row.max_turns}）：本案判决已被轮次上限截断，"
            + (
                "fail 未必是能力问题、pass 也未必是干净通过——"
                if row.verdict == "fail"
                else "pass 属贴线通过，下一跳多一次自证就会被拦成 fail——"
            )
            + "先放宽案内 max_turns 再取判决"
        )
    if row.reverted:
        print(
            "        [!] 本跳弄脏组员跟踪面，已由测量面还原："
            + "、".join(row.reverted)
            + "（噪声不留在树上，但反复出现=案内纪律或 EVAL_HOP_NOTE 该加强）"
        )


def cmd_eval_run(args: argparse.Namespace) -> int:
    """``orch eval run``：跑案（一案或全部）→ 二元判决 + 指标表 + 基线对照。

    Returns:
        0=全部通过；1=有案失败（判决是测量结果，不是崩溃）；2=用法/案卷错误。
    """
    try:
        cases = load_cases("" if args.all else args.case)
    except ValueError as exc:
        print(f"[✗] 案卷不可用：{exc}")
        print(
            "    [NEXT] 案卷=state/evals/harness/<case-id>.md：头部 member:/tier:/dirs:/"
        )
        print("           产物:/断言: 标记行，`## 任务书` 之后是派发正文。")
        return 2
    label = args.label or ""
    print(
        f"== harness 压测（{len(cases)} 案，档位 {args.model_tier or '案内 tier'}）=="
    )
    ran: dict[str, dict[str, Any]] = {}
    for case in cases:
        row = run_case(case, tier=args.model_tier or "", label=label)
        _print_row(row)
        ran[row.case] = asdict(row)
    cmp = compare(ran, read_baseline())
    print()
    print(f"== 与基线对照（本轮 {len(ran)} 案，{cmp['verdict']}）==")
    print(
        f"  通过数 {cmp['pass_base']} → {cmp['pass_now']}；"
        f"均值 num_turns {cmp['mean_turns_base']} → {cmp['mean_turns_now']}"
    )
    for d in cmp["diffs"]:
        now = ran.get(d["case"], {})
        tight = now.get("max_turns") and now.get("num_turns", 0) >= now["max_turns"]
        print(
            f"  - {d['case']}: {d['verdict']}｜turns {d['turns']}"
            + ("（预算用满，读数不可信）" if tight else "")
        )
    if cmp["only_now"]:
        print(f"  [i] 本轮新增案（基线无记录）：{cmp['only_now']}")
    if cmp["verdict"] == "无基线":
        print(
            "    [NEXT] 首轮无对照：跑完满意的一轮后 `orch eval set-baseline` 固化基线。"
        )
    else:
        print(
            "    [NEXT] 二元判决="
            + cmp["verdict"]
            + "（规则：通过数不减 且 均值轮次不劣化 且 至少一项严格更优）"
        )
    return 0 if ran and all(r["verdict"] == "pass" for r in ran.values()) else 1


def cmd_eval_set_baseline(args: argparse.Namespace) -> int:
    """``orch eval set-baseline``：把最近一轮（可按 ``--label``/``--runner`` 选）回填为基线快照。"""
    try:
        doc = write_baseline(args.label or "", getattr(args, "runner", "") or "")
    except ValueError as exc:
        print(f"[✗] {exc}")
        return 2
    print(f"[√] 基线已回填：{len(doc['cases'])} 案 → {BASELINE_PATH.name}")
    for cid, c in doc["cases"].items():
        print(
            f"  - {cid:<18} {c['verdict']:<4} turns={c['num_turns']} "
            f"est_tokens={c['est_tokens']} ctx={c['ctx_ratio']:.4f}"
        )
    print("    [NEXT] 改动后 `orch eval run --all` 即出 before/after 对照表。")
    return 0
