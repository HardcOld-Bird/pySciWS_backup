"""plan-runner：计划驱动的任务流程状态机（README §4.2）。

计划 = ``state/plans/<id>.plan.yaml``（组长可直接编辑——amend 即编辑此文件，
advance 时按环节 id 对账）；执行状态 = ``state/plans/<id>.state.json``（orch 专属）。
核心原则：**计划即状态机，聊天记录不是**——组长会话被压缩/重开后仍从
断点继续；交接处暂停，精准 recall 下一环节粗描述原文。
"""

from __future__ import annotations

import copy
import hashlib
import json
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from pysci.paths import ORCH_STATE_ROOT

from .dispatch import DispatchOutcome
from .ledger import format_credits, format_est_tokens, read_all, summarize
from .planflow import (
    DONE_STATES,
    KIND_SELF,
    SELF_AWAITING,
    STARTED_STATES,
    PlanTree,
    freeze_violations,
)

PLANS_DIR = ORCH_STATE_ROOT / "plans"

_DONE = DONE_STATES

#: 拆解指导真本（PSEP_3 §2.5 三级方案：必读文档 + plan new 打印 + read: 标记 lint）。
DECOMP_REF = "plan-decomposition.md"

#: ``plan new`` 输出**头部**固定打印的必读指引。
DECOMP_READ_HINT = f"※ 多环节大任务拆解前必读：orchestration/skills/orchestration/references/{DECOMP_REF}"


def _tree(plan: dict) -> PlanTree:
    """plan YAML → 编排树（PSEP_3 §2.1；纯解析无 I/O，每次现建——plan 小）。"""
    return PlanTree(plan.get("steps") or [])


def _leaf_map(plan: dict) -> dict[str, dict]:
    """叶子**点路径** → 叶子 YAML（superstep 嵌套下为全部叶子；扁平 plan 恒等原 id）。"""
    return _tree(plan).leaf_map()


def _status_of(state: dict):
    """状态读数闭包：叶子点路径 → status（缺省 pending）。"""
    st = {
        str(s.get("id")): str(s.get("status") or "pending")
        for s in state.get("steps", [])
    }
    return lambda path: st.get(str(path), "pending")


def _has_read_marker(plan: dict) -> bool:
    """plan YAML 是否含 ``read: plan-decomposition.md`` 标记行（PSEP_3 §2.5）。"""
    v = plan.get("read")
    if isinstance(v, (list, tuple)):
        return any(DECOMP_REF in str(x) for x in v)
    return DECOMP_REF in str(v or "")


def _print_plan_lints(plan: dict) -> None:
    """plan 结构 lint（**不阻断**，PSEP_3 §2.4/§2.5）：软提示 + 拆解必读标记行。"""
    tree = _tree(plan)
    lines = tree.soft_hints()
    n = len(tree.leaves)
    if (tree.has_superstep() or n >= 5) and not _has_read_marker(plan):
        lines.append(
            f"plan 含 superstep 或环节数≥{n}——建议在 YAML 加 `read: {DECOMP_REF}` 标记行"
            f"（拆解前必读 references/{DECOMP_REF}）"
        )
    for ln in lines:
        print(f"[lint] {ln}（软提示，不阻断）")


#: role → 该环节 evidence 的**内容特征段**（backlog 20261015-135000-f2-ledger）。
#: 为什么按 role 判内容：原先 `evidence_sections` 只有 a1/a2/a3 显式写了，`b1`（重拆解）与
#: `b_last`（全流程报告）**没有** → 退回「至少一个标题行」的弱判据，于是 `b_last` 拿一份
#: 别的环节的方案文件也能过（实案：误收 c2 文件）。YAML 里的 `evidence_sections` 仍**优先**
#: （它才是每份计划的可定制面），本表只是**缺省**。
ROLE_EVIDENCE_SECTIONS: dict[str, tuple[str, ...]] = {
    "cause": ("成因分析", "结论"),
    "solution": ("解决方案", "结论"),
    "review": ("送审结论",),
    "improvement_plan": ("拆解",),
    "rework_plan": ("拆解", "结论"),
    # 「全流程报告」= `taskbooks.report_sections` 组装的通用两段（任何任务书都有）
    "final_report": ("任务概述", "环节完成清单"),
    # L 组长改进（用户 2026-10-17）：l2 必须用**新 role**——复用 improvement_plan 会与
    # a4 在 taskbooks._role_texts（按 role 为键）互相覆盖。
    "leader_reflection": ("摩擦反思", "改进清单", "结论"),
    "leader_improvement_plan": ("拆解",),
}


#: L 环节（组长改进）正式要求文本（基于用户 2026-10-17 裁决草稿扩写）：l1 引导时全文打印。
LEADER_IMPROVE_PROMPT: str = """\
※ 组长改进环节（harnessing 摩擦反思）——正式要求：
请回顾这段时间对本系统的使用，积极立单修复你感受到的摩擦：
1. harnessing 指导是否充分？——你的工作执行情形（各环节该做什么、判据是什么）在
   harnessing（rules/charter/技能/orch 引导）里有没有足够指导？相关功能有没有被合理介绍？
2. 信息获取是否方便？——工作过程中拿到所需信息要几步？若 orch 在触发点直接附送这些
   信息（如要求与历史总览、引导块），会不会更好？哪些信息是你靠 grep/翻文件才找到的？
3. 是否存在误导？——harnessing 的表述与实际行为不符、或引导指向了错误命令/路径的，逐条点名。
对每个真实摩擦**立单改进**（下一环节 l2 拆解改进派单，随后 l3 派 devops 实施）；
若确无摩擦，在 l2 写明「本轮无需改进，空清单」，并以
`pysci-orch plan advance <pid> --skip L.l3` 收尾（空改进清单合法）。"""


def _check_evidence(path: str, sy: dict) -> tuple[bool, str]:
    """校验 ``kind: self`` 环节的 evidence（PSEP_3 §2.1）：文件存在 + 段落要求。

    段落要求的**判据来源**（按优先级）：
    1. YAML ``evidence_sections: [...]``（显式声明，逐段匹配 Markdown 标题行）；
    2. 节点的 ``role`` 对应 :data:`ROLE_EVIDENCE_SECTIONS` 的**内容特征段**——照抄一份
       别的环节的方案文件不再算数（实案 b_last 误收 c2）；
    3. 两者皆无才退回「至少一个 ATX 标题行」（有结构的报告，非空）。

    不合规一律**拒收并点名缺哪一段**。
    """
    from pysci.paths import PROJECT_ROOT

    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    if not p.is_file():
        return False, f"evidence 文件不存在：{p}"
    text = p.read_text(encoding="utf-8", errors="replace")
    if not text.strip():
        return False, f"evidence 文件为空：{p}"
    headings = [ln for ln in text.splitlines() if ln.lstrip().startswith("#")]
    role = str(sy.get("role") or "")
    sections = sy.get("evidence_sections") or ROLE_EVIDENCE_SECTIONS.get(role, ())
    if sections:
        missing = [t for t in sections if not any(str(t) in h for h in headings)]
        if missing:
            what = "标题段" if sy.get("evidence_sections") else f"role={role} 特征段"
            return (
                False,
                f"evidence 缺{what}：{'、'.join(str(m) for m in missing)}（{p}）",
            )
        return True, f"evidence 通过（{len(sections)} 个标题段齐全）：{p}"
    if not headings:
        return False, f"evidence 无标题段（需至少一个 Markdown 标题行 `# …`）：{p}"
    return True, f"evidence 通过（含标题段）：{p}"


def _self_target(plan: dict, state: dict) -> dict | None:
    """待收尾的 ``kind:self`` 叶子：优先 ``self_awaiting``（已引导待证据），否则首个可派发 self。"""
    for s in state["steps"]:
        if str(s.get("status")) == SELF_AWAITING:
            return s
    lm = _leaf_map(plan)
    disp = _tree(plan).dispatchable(_status_of(state))
    for p in disp:
        if str((lm.get(p) or {}).get("kind") or "dispatch") == KIND_SELF:
            return next((s for s in state["steps"] if str(s["id"]) == p), None)
    return None


def _apply_evidence(plan_id: str, plan: dict, state: dict, evidence: str) -> int | None:
    """把 ``--evidence`` 落到当前 self 环节。返回 ``None``=已收尾（调用方继续推进）；否则错误码。"""
    tgt = _self_target(plan, state)
    if tgt is None:
        print(f"[!] 计划 {plan_id} 无可收尾的 kind:self 环节——--evidence 无处可施。")
        return 2
    sy = _leaf_map(plan).get(str(tgt["id"]), {})
    ok, msg = _check_evidence(evidence, sy)
    if not ok:
        tgt["status"] = SELF_AWAITING
        _save_state(plan_id, state)
        print(f"[!] 组长环节 {tgt['id']} evidence 校验未过：{msg}")
        return 2
    tgt["status"] = "done"
    tgt["evidence"] = str(evidence)
    _save_state(plan_id, state)
    print(f"[√] 组长环节 {tgt['id']} 收尾：{msg}")
    return None


def _report_after_evidence(plan_id: str, plan: dict, state: dict) -> None:
    """``--evidence`` 收尾 self 环节后报告：全完成则报完成，否则刷工作清单。"""
    if _plan_settled(plan, state):
        if not state.get("completed_at"):
            state["completed_at"] = _now()
            _save_state(plan_id, state)
        _report_plan_complete(plan_id, state)
    else:
        _print_worklist(plan_id, plan, state)


def _print_self_guidance(
    plan_id: str, step_state: dict, sy: dict, *, plan: dict | None = None
) -> None:
    """``kind: self`` 环节的引导块（evidence 规格 + 收尾命令）——plan new / advance 共用。

    ``expand_history`` 叶子（返工 a1，round≥2）在 [NEXT] 后给出历次改进的**摘要＋报告路径**
    （§4.3 #4；不再全文灌入——截断反噬，backlog 20261017-104503）。
    """
    print()
    print("[NEXT]")
    print(f"  环节 {step_state['id']} 为组长自办环节（kind: self）——不入成员队列。")
    print(f"    粗描述原文：{sy.get('brief', '（无）')}")
    secs = sy.get("evidence_sections")
    if not secs:  # role-only leaf：回退 role 缺省特征段，**与 _check_evidence 同源**
        # （backlog 20261015-235101-devops）此前引导面只认 evidence_sections，role 缺省段
        # 不打印 → 组长写 evidence 靠猜关键词（写中「拆解」才过、写「改进派单」被拒）。
        # 把判据**附送到触发点**：打印与校验取自同一 ROLE_EVIDENCE_SECTIONS。
        secs = ROLE_EVIDENCE_SECTIONS.get(str(sy.get("role") or ""), ())
    if secs:
        src = "本环节声明" if sy.get("evidence_sections") else "role 缺省"
        print(f"  交付 evidence 须含标题段（{src}）：{'、'.join(str(s) for s in secs)}")
    else:
        print("  交付 evidence 须为 Markdown 报告（含至少一个 `# …` 标题段）")
    print(
        "  evidence 落点惯例：data/taskbooks/<id>.<leaf>-<名>.md"
        "（骨架＝`# 标题`＋上述标题段，如 `## 拆解`）"
    )
    print(f"  完成后提交证据收尾：pysci-orch plan advance {plan_id} --evidence <路径>")
    if str(sy.get("role") or "") == "leader_reflection":
        print()
        for ln in LEADER_IMPROVE_PROMPT.splitlines():
            print(f"  {ln}")
    if sy.get("expand_history") and plan is not None:
        for ln in _rework_history_lines(plan):
            print(ln)


def _rework_history_lines(plan: dict) -> list[str]:
    """rework_round≥2 的**历次改进强制展示**（§4.3 #4）：**要求与历史总览 ＋ 报告路径**。

    数据源＝:func:`taskbooks.requirements_digest`（用户 2026-10-17：原始要求＋历轮批语
    ＋历轮收获集中展示）。旧关键词摘要有三重漏（跳过「历轮摘要」段、220 字/段切掉原始
    要求、无标题正文不进摘要）——已被总览单源取代；报告路径提示保留（要全文组长自读）。
    """
    from . import taskbooks as _tb

    tb_id = str(plan.get("taskbook") or "")
    st = (_tb.load_state(tb_id) or {}) if tb_id else {}
    rnd = int(st.get("rework_round") or plan.get("rework_round") or 1)
    if rnd < 2 or not tb_id:
        return []
    out = [f"  == 历次改进强制展示（第 {rnd} 轮；PSEP_3 §4.3 #4）=="]
    for ln in _tb.requirements_digest(tb_id).splitlines():
        out.append(f"  {ln}" if ln else "")
    for member in _tb.family_chain(tb_id)[:-1]:
        mid = str(member.get("id") or "")
        rp = _tb.report_path(mid)
        if rp.exists():
            out.append(f"  细节自查该文件：{rp}")
        else:
            out.append(f"  （{mid} 无报告留痕；细节查该轮任务书）")
    return out


def _guide_self_step(plan_id: str, plan: dict, state: dict, cur: dict, sy: dict) -> int:
    """引导一个 ``kind:self`` 组长环节（不入成员队列）：打印 evidence 规格与收尾命令。"""
    if str(cur.get("status")) != SELF_AWAITING:
        cur["status"] = SELF_AWAITING
        _save_state(plan_id, state)
    _sync_taskbook_active(plan_id)  # 首环节开始执行 → 所链任务书「执行中」
    _print_self_guidance(plan_id, cur, sy, plan=plan)
    _print_worklist(plan_id, plan, state)
    return 0


def _sync_taskbook_active(plan_id: str) -> None:
    """计划**开始执行**（首环节入队/自办引导）→ 所链任务书 ``planning → active``。

    回写实现是单点（:func:`taskbooks.sync_active`，幂等），此处只负责在推进的两个执行
    分支上触发。失败不阻塞推进（观察面 fail-soft，§0.6）。
    """
    try:
        from . import taskbooks as _tb

        _tb.sync_active(plan_id)
    except Exception:  # noqa: BLE001 - 对账绝不阻塞推进
        pass


#: 阻塞推进原语的同步等待上限（秒）——与成员 registry ``timeout_s`` 默认（3h）同量级：
#: 直派时代它就一路阻塞在 ``do_dispatch`` 里，改为「入队+等收尾」后阻塞面不变。
PLAN_RUN_TIMEOUT_S: int = 3 * 3600

#: 等收尾的轮询间隔（秒）。等的是**队列条目状态**（worker 落盘），不是进程——进程死了
#: 条目会由孤儿回收/reset_orphans 收敛，本函数到点即退，把环节留 ``queued`` 交 advance 结账。
PLAN_RUN_POLL_S: float = 2.0


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _plan_path(plan_id: str) -> Path:
    return PLANS_DIR / f"{plan_id}.plan.yaml"


def _state_path(plan_id: str) -> Path:
    return PLANS_DIR / f"{plan_id}.state.json"


#: 计划 YAML 的序列化宽度：**禁用折行**。PyYAML ``safe_dump`` 默认 ``width=80`` 会把长
#: ``brief`` 折成多行，于是任何**重新序列化**的计划（plan new/edit/amend/rework/adhoc 都会
#: 走过 ``safe_dump``）在手写长行计划上产生「未编辑环节也变」的折行漂移——工具重序列化
#: 即漂移源头（backlog 20261015-235100-devops）。取足够大值使 ``dump(load(x)) == x``
#: 对**规范形态**（不折行）成立 ⇒ 重序列化幂等、未改动的计划字节不变。
_PLAN_YAML_WIDTH: int = 10**9


def _dump_plan(plan: dict) -> str:
    """计划 dict → YAML 文本（**稳定/幂等**：不折行；重复序列化字节不变）。"""
    return yaml.safe_dump(
        plan, allow_unicode=True, sort_keys=False, width=_PLAN_YAML_WIDTH
    )


class PlanYamlError(yaml.YAMLError):
    """计划 YAML 解析失败：带 ``file:line:col`` 与人话成因提示（CLI 单行报错用）。

    继承 :class:`yaml.YAMLError` 是**兼容性选择**：既有读面（dispatch / nextroute /
    taskbooks）里「接住 ``yaml.YAMLError``」的代码不受影响；而想给单行可读报错的调用方用
    :meth:`render`，定位与常见成因已经备好。

    为什么需要它（backlog 20261015-202556）：裸 ``safe_load`` 的失败形态是 10 余行 traceback，
    且把文件名显示成 ``<unicode string>``——手写 plan 缩进错一格就被糊脸，看不出是哪个文件
    的哪一行、更看不出成因。
    """

    def __init__(self, path: Path | str | None, exc: yaml.YAMLError) -> None:
        mark = getattr(exc, "problem_mark", None) or getattr(exc, "context_mark", None)
        self.path = str(path) if path is not None else ""
        self.line = int(getattr(mark, "line", -1)) + 1
        self.col = int(getattr(mark, "column", -1)) + 1
        self.problem = str(getattr(exc, "problem", None) or exc)
        super().__init__(self.render())

    def render(self) -> str:
        loc = (
            f"{self.path}:{self.line}:{self.col}"
            if self.path
            else f"line {self.line} col {self.col}"
        )
        return (
            f"[!] YAML 解析失败 {loc} {self.problem}"
            "（常见成因：层级缩进错位 / 冒号后缺空格 / 混入 Tab）"
        )


def parse_plan_yaml(src: Path | str) -> dict:
    """读并解析计划 YAML；坏 YAML → :class:`PlanYamlError`（带 file:line:col 与成因提示）。"""
    p = Path(src)
    try:
        return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise PlanYamlError(p, exc) from exc


#: 解析缓存：``解析后路径 → (st_mtime_ns, st_size, payload)``。stat 比对未变即命中、
#: **不解析**（TUI 徽标指纹/概览轮询的性能钉——此前每次全量解析 46 个 plan YAML）。
#: key 含解析后路径 ⇒ 测试 monkeypatch ``PLANS_DIR`` 后自然隔离。
_YAML_CACHE: dict[Path, tuple[int, int, dict]] = {}
_STATE_CACHE: dict[Path, tuple[int, int, dict]] = {}


def _cached_payload(
    cache: dict[Path, tuple[int, int, dict]], path: Path, parse: Callable[[Path], dict]
) -> dict:
    """stat（mtime+size）比对命中即返回缓存**拷贝**；否则现解析并回填。

    返回拷贝是必须的：调用方普遍就地改写 state（``_reconcile`` 的中间态、``_save_state``
    前的推进），回同一对象会让未落盘的改写污染缓存。``_save_state`` 落盘即变 mtime，
    下次 stat 自然未命中。
    """
    key = path.resolve()
    try:
        st = path.stat()
    except OSError:
        cache.pop(key, None)
        payload = parse(
            path
        )  # 缺档走既有语义（state 回 {}；YAML 抛 FileNotFoundError）
        return copy.deepcopy(payload)
    stamp = (st.st_mtime_ns, st.st_size)
    hit = cache.get(key)
    if hit is not None and (hit[0], hit[1]) == stamp:
        return copy.deepcopy(hit[2])
    payload = parse(path)
    cache[key] = (stamp[0], stamp[1], payload)
    return copy.deepcopy(payload)


def _read_state_json(path: Path) -> dict:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {"steps": []}
    data.setdefault("steps", [])
    return data


def _load_yaml(plan_id: str) -> dict:
    return _cached_payload(_YAML_CACHE, _plan_path(plan_id), parse_plan_yaml)


def _load_state(plan_id: str) -> dict:
    """读计划状态（**保 falsy 语义**：文件缺席＝「无此计划」，仍回 ``{}``）。

    只有**文件在册**时才归一 `steps` 键（缺键补空表）——实案：`pysci-orch next --limit 10`
    崩 `KeyError: 'steps'`（盘上有计划 YAML 却无 state 文件的一批计划 → 读面直接下标）。
    缺席路径**不动**：多处用 ``if not state:`` 判「计划是否存在」（如 ``cmd_plan_drop``），
    改成 truthy 会改掉这些调用点的行为。崩溃点本身按空处理，见
    :func:`_inflight` / :func:`_worklist_panes`（backlog 20261015-154311）。
    """
    return _cached_payload(_STATE_CACHE, _state_path(plan_id), _read_state_json)


#: 公开别名（门面 status 命令读取计划状态用）。
load_state = _load_state


def _save_state(plan_id: str, state: dict) -> None:
    PLANS_DIR.mkdir(parents=True, exist_ok=True)
    _state_path(plan_id).write_text(
        json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _started_rows_by_leaf_name(old: dict[str, dict]) -> dict[str, dict]:
    """已 started 行的**叶名**索引（``BC.C.c4`` → ``c4``）；同名多于一行即丢弃（歧义不猜）。

    为什么要它（backlog 20261015-135000-f2-ledger）：superstep 重排会让叶子的**点路径**变
    （``C.c4`` → ``BC.C.c4``），此时按 id 严格对账会把**已派发的行**当成「新叶子」置回
    ``pending``——「已完成的工作显示待派发」（实案 c4：已 queued 却显示待派发，组长只能手工
    账实修正）。唯一同名命中时按「同一环节」带过状态；歧义（同名多行）时保守置 pending。
    """
    buckets: dict[str, list[dict]] = {}
    for sid, row in old.items():
        if str(row.get("status") or "") in STARTED_STATES:
            buckets.setdefault(str(sid).rsplit(".", 1)[-1], []).append(row)
    return {k: v[0] for k, v in buckets.items() if len(v) == 1}


def _reconcile(plan: dict, state: dict) -> dict:
    """按**叶子点路径 id** 对账 YAML（可能被组长 amend 过）与既有状态。

    superstep 只在 YAML 里存在（递归结构），**state.json 只落叶子行**（PSEP_3 §2.1：
    聚合状态由叶子推导不落盘）；存量扁平 plan 的顶层叶子路径 == 原环节 id，行为完全不变
    （无 superstep 等价单元素串行）。

    对账**只做两件事**（账实保护，backlog 20261015-135000-f2-ledger）：
    - 增补 YAML 里有、state 里没有的叶子（新叶子置 ``pending``）；
    - 删掉 YAML 里已不存在的幽灵行。
    凡能认出是「同一环节」的既有行，一律**原样带过**（status + 既有字段），**绝不把
    已 started 的状态打回 pending**。识别按 id；id 对不上时按**唯一叶名**兜底（见
    :func:`_started_rows_by_leaf_name`），歧义则保守置 pending。
    """
    tree = _tree(plan)
    old = {str(s["id"]): s for s in state.get("steps", []) if s.get("id")}
    by_name = _started_rows_by_leaf_name(old)
    steps: list[dict] = []
    for path in tree.leaf_paths():
        prev = old.get(path)
        if prev is None:
            prev = by_name.get(
                path.rsplit(".", 1)[-1]
            )  # id 迁移容错：唯一同名 + 已 started
        if prev is None:
            row: dict = {"id": path, "status": "pending", "review": "pending"}
        else:
            # 从既有行**整体拷贝**再覆写 id：evidence/job_id/note 等字段也是账实的一部分，
            # 原实现只挑五个键重建 → 每次对账都静默丢掉 evidence（自办环节的交付证据）。
            row = dict(prev)
            row["id"] = path
            row.setdefault("status", "pending")
            row.setdefault("review", "pending")
        steps.append(row)
    state.update(
        {
            "plan_id": state.get("plan_id") or plan.get("id") or "",
            "goal": plan.get("goal", ""),
            "steps": steps,
            "completed_at": state.get("completed_at"),
        }
    )
    return state


def ensure_leader_step(plan_id: str) -> bool:
    """任务书勾了「包含组长改进」→ 在 plan 顶层末尾**自动追加 L 环节**（用户 2026-10-17）。

    组长零手写（与返工骨架同风格的「工具组装」）；幂等：任一叶子
    ``role == "leader_reflection"`` 即视为已有。只**追加**（不动任何既有行），已 started
    环节不受影响；对账走 :func:`_reconcile`（amend 同款）。

    Returns:
        True=本次追加；False=已有 L / plan 不存在。
    """
    if not _plan_path(plan_id).exists():
        return False
    plan = _load_yaml(plan_id)
    if any(
        str(leaf.yaml.get("role") or "") == "leader_reflection"
        for leaf in _tree(plan).leaves
    ):
        return False
    plan.setdefault("steps", []).append(_rework_L())
    _plan_path(plan_id).write_text(_dump_plan(plan), encoding="utf-8")
    _save_state(plan_id, _reconcile(plan, _load_state(plan_id)))
    print(f"[√] 已自动追加 L 组长改进环节（任务书勾选「包含组长改进」）：{plan_id}")
    return True


def list_plans() -> list[tuple[str, str, int, int]]:
    """列出全部计划：(id, goal, 完成环节数, 总环节数)。"""
    out = []
    if not PLANS_DIR.exists():
        return out
    for p in sorted(PLANS_DIR.glob("*.plan.yaml")):
        pid = p.name[: -len(".plan.yaml")]
        plan = _load_yaml(pid)
        state = _load_state(pid)
        steps = state.get("steps", [])
        done = len([s for s in steps if s["status"] in _DONE])
        out.append((pid, str(plan.get("goal", ""))[:50], done, len(steps)))
    return out


# ---------------------------------------------------------------------------
# 概览/指纹读数（TUI 改版 s1-data）：**只读 state.json**，徽标指纹不再解析 46 个 YAML
# ---------------------------------------------------------------------------
def _overview_plan_ids() -> list[str]:
    """YAML ∪ state 两侧的计划 id 清单（缺一侧也能入列表）。"""
    if not PLANS_DIR.exists():
        return []
    ids = {p.name[: -len(".plan.yaml")] for p in PLANS_DIR.glob("*.plan.yaml")}
    ids |= {p.name[: -len(".state.json")] for p in PLANS_DIR.glob("*.state.json")}
    return sorted(ids)


def _plan_overview_state(pid: str) -> dict:
    """state-only 读数（缺档/坏文件按空态；**不碰 YAML**——性能钉的边界）。"""
    if not _state_path(pid).exists():
        return {}
    try:
        return _load_state(pid)
    except (OSError, ValueError):
        return {}


def _plan_mark(state: dict) -> str:
    """概览标记（语义同 ``cmd_plan_list``/TUI 计划列表：completed→「✔ 完成」、abandoned→「⊘ 中止」）。

    图标后**必带一个空格**再跟文字（用户 2026-10-16 实测：终端下宽字符与文字会重叠）。
    """
    if state.get("completed_at"):
        return "✔ 完成"
    if state.get("abandoned_at"):
        return "⊘ 中止"
    return "▶ 进行中"


def _done_total(state: dict) -> tuple[int, int]:
    steps = state.get("steps") or []
    return len([s for s in steps if str(s.get("status")) in _DONE]), len(steps)


def overview_rows() -> list[dict]:
    """计划概览行（TUI 计划列表/徽标指纹用；**只读 state.json**，缺 state 回落
    缓存 ``_load_yaml`` 取 goal——回落 fail-soft，YAML 面坏不塌概览）。

    每行 ``{id, goal, mark, done, total, created, completed_at, abandoned_at, current}``；
    ``mark`` ∈ {``▶ 进行中``, ``✔ 完成``, ``⊘ 中止``}；``current`` = 首个非 done/skipped 步
    ``{id, status, session_id, task_file}``（全完成 → ``None``）。
    按 ``created`` 降序（缺 created 者按 id 降序、自然殿后）。
    """
    rows: list[dict] = []
    for pid in _overview_plan_ids():
        state_exists = _state_path(pid).exists()
        state = _plan_overview_state(pid)
        goal = str(state.get("goal") or "")
        if not state_exists:
            try:
                goal = str(_load_yaml(pid).get("goal") or "")
            except Exception:  # noqa: BLE001 - 概览读数 fail-soft（缺 state + YAML 坏）
                goal = ""
        done, total = _done_total(state)
        cur = next(
            (
                s
                for s in (state.get("steps") or [])
                if str(s.get("status")) not in ("done", "skipped")
            ),
            None,
        )
        rows.append(
            {
                "id": pid,
                "goal": goal,
                "mark": _plan_mark(state),
                "done": done,
                "total": total,
                "created": str(state.get("created") or ""),
                "completed_at": str(state.get("completed_at") or ""),
                "abandoned_at": str(state.get("abandoned_at") or ""),
                "current": (
                    {
                        "id": str(cur.get("id") or ""),
                        "status": str(cur.get("status") or "pending"),
                        "session_id": str(cur.get("session_id") or ""),
                        "task_file": str(cur.get("task_file") or ""),
                    }
                    if cur is not None
                    else None
                ),
            }
        )
    rows.sort(key=lambda r: (r["created"], r["id"]), reverse=True)
    return rows


def plan_digest() -> str:
    """廉价计划指纹（TUI 徽标刷新判据）：sha1 前 12 hex。

    输入＝按 id 序的 ``id:mark:done:total:状态向量`` 用 ``|`` 连接。**只含概览可见
    字段**：不读 YAML、不构建 detail——性能钉由 tests 断言（``_load_yaml`` 抛异常
    时本函数与 :func:`overview_rows` 仍正常）。
    """
    parts: list[str] = []
    for pid in sorted(_overview_plan_ids()):
        state = _plan_overview_state(pid)
        done, total = _done_total(state)
        vec = ",".join(
            str(s.get("status") or "pending") for s in (state.get("steps") or [])
        )
        parts.append(f"{pid}:{_plan_mark(state)}:{done}:{total}:{vec}")
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:12]


def cmd_plan_new(args) -> int:
    """登记新计划（复制 YAML 入 state/plans/，初始化状态）。"""
    print(DECOMP_READ_HINT)  # 头部固定打印必读指引（PSEP_3 §2.5 三级方案之二）
    src = Path(args.file)
    if not src.is_absolute():
        from pysci.paths import PROJECT_ROOT

        src = PROJECT_ROOT / src
    if not src.exists():
        print(f"[!] 计划文件不存在：{src}")
        return 2
    try:
        plan = parse_plan_yaml(src)
    except PlanYamlError as exc:
        print(exc.render())
        return 2
    if not plan.get("steps"):
        print("[!] 计划缺少 steps（格式见 README §4.2）")
        return 2
    if (gate := _serial_gate(str(plan.get("taskbook") or ""))) is not None:
        return gate
    plan_id = args.id or src.stem
    if _plan_path(plan_id).exists() and not args.force:
        print(
            f"[!] 计划 {plan_id} 已存在（覆盖用 --force；续跑用 plan advance {plan_id}）"
        )
        return 2
    PLANS_DIR.mkdir(parents=True, exist_ok=True)
    plan["id"] = plan_id
    _plan_path(plan_id).write_text(_dump_plan(plan), encoding="utf-8")
    state = _reconcile(plan, {"plan_id": plan_id, "created": _now()})
    _save_state(plan_id, state)
    tree = _tree(plan)
    print(f"[√] 计划已登记：{plan_id}（{len(state['steps'])} 环节）")
    print(f"    目标：{state['goal']}")
    for leaf in tree.leaves:
        sy = leaf.yaml
        indent = "  " * leaf.path.count(".")
        who = sy.get("member") or ("self（组长）" if leaf.kind == KIND_SELF else "?")
        print(
            f"    - {indent}{leaf.path}: {who} | "
            f"{str(sy.get('brief', ''))[:60]}"
            + ("  [review]" if sy.get("review") else "")
        )
    _print_plan_lints(plan)
    first = state["steps"][0]
    fy = next((n.yaml for n in tree.leaves if n.path == first["id"]), {})
    if str(fy.get("kind") or "dispatch") == KIND_SELF:
        _print_self_guidance(plan_id, first, fy, plan=plan)
    else:
        _print_run_guidance(plan_id, first, fy)
    return 0


def _print_run_guidance(plan_id: str, step_state: dict, step_yaml: dict) -> None:
    print()
    print("[NEXT]")
    print(f"  当前环节 {step_state['id']}（{step_yaml.get('member')}）粗描述原文：")
    print(f"    {step_yaml.get('brief', '（无）')}")
    print(
        f"  白名单建议：{step_yaml.get('dirs', [])}；"
        f"档位：{step_yaml.get('model', '(成员默认)')}；"
        f"effort：{step_yaml.get('effort', '(默认=中)')}"
    )
    print("  → 书写该环节任务书后推进：")
    print(f'    pysci-orch plan advance {plan_id} --text "<任务书全文>"')
    print(f"    或 pysci-orch plan advance {plan_id} --task <任务书文件>")
    print(
        "  （**入队即返回**，完成事件驱动下一次 advance——长任务首选；计划外单件事用"
        " `plan adhoc`（同步阻塞，须后台运行），见 README §4.5）"
    )


def _outcome_from_job(member: str, item: dict) -> DispatchOutcome:
    """从成员队列条目**还原**一次派发的结构化结果（交接报告的输入面）。

    直派退役后阻塞推进原语不再经手 ``DispatchOutcome``——worker 把分类/正文/验收/
    建议/时长落在 ``item.result``（见 ``queueworker._execute``），这里还原成交接报告
    原本就认识的形状，避免两套交接渲染。
    """
    from . import workq
    from .runner import Envelope

    res = item.get("result") or {}
    status = str(item.get("status") or "")
    code = res.get("code")
    code = int(code) if code is not None else (0 if status == workq.STATUS_DONE else 2)
    kind = str(res.get("kind") or "")
    if not kind:
        kind = "result" if status == workq.STATUS_DONE else "run_failed"
    return DispatchOutcome(
        code=code,
        kind=kind,
        member=str(res.get("member") or member),
        sid=str(res.get("sid") or ""),
        session_name=str(res.get("session_name") or ""),
        task_file=str(res.get("task_file") or item.get("task_file") or ""),
        body=str(res.get("body") or ""),
        body_file=str(res.get("body_file") or ""),
        checks=[dict(c) for c in (res.get("checks") or [])],
        suggestion_id=str(res.get("suggestion_id") or ""),
        envelope=Envelope(duration_ms=int(res.get("duration_ms") or 0)),
    )


def _wait_all_inflight(
    plan_id: str,
    plan: dict,
    state: dict,
    *,
    timeout_s: int = PLAN_RUN_TIMEOUT_S,
    sleep: Any = None,
    now: Any = None,
) -> list[tuple[dict, DispatchOutcome]]:
    """等本计划**全部在途 job** 收尾（阻塞推进原语的等待语义，PSEP_3 §2.3）。

    逐轮把已到齐的结果落终态（**乱序即到达序**）；到点仍未收尾 → 留 ``queued`` 交
    advance（与「超时不猜结果」旧语义一致）。无在途时也先收尾一轮（把已终态条目结掉）。
    ``sleep``/``now`` 是**测试注入口**。
    """
    do_sleep = time.sleep if sleep is None else sleep
    do_now = time.time if now is None else now
    deadline = do_now() + float(timeout_s)
    settled: list[tuple[dict, DispatchOutcome]] = []
    while True:
        settled.extend(_settle_terminal_jobs(plan_id, plan, state))
        pend = _inflight(state)
        if not pend:
            break
        # 条目失踪（被裁剪 / 入队未落盘 / 测试桩）：无法观察 → 不猜结果，停止等待留 queued
        # （与旧 `_wait_job`「条目消失 → 不改状态」语义一致；防下游按 3h 上限空转）。
        lm = _leaf_map(plan)
        if any(
            _queue_item_for_step(
                plan_id,
                str(s["id"]),
                str((lm.get(str(s["id"])) or {}).get("member") or ""),
            )
            is None
            for s in pend
        ):
            break
        if do_now() >= deadline:
            break
        do_sleep(PLAN_RUN_POLL_S)
    _save_state(plan_id, state)
    return settled


def cmd_plan_run(args) -> int:
    """**阻塞推进原语**（无 CLI 入口，`plan adhoc` 内部使用）：等全部在途 job 收尾后交接。

    2026-10-16（backlog 20261016-220500）：原 `plan run` 命令已废弃——与 `plan advance`
    同能力两实现；本函数作为**内部阻塞原语**保留给 `plan adhoc`（计划外单件事的同步语义）。
    语义（PSEP_3 §2.3）：① 先把已终态队列条目全部结掉（多 job 并行时乱序到达，逐项交接）；
    ② 若仍有在途 job → 阻塞等它们全部收尾；③ 目标＝blocked/失败叶子（续作优先，--text
    即补充信息）或**首个可派发叶子**（递归就绪谓词）；④ 交接处打印三栏工作清单后暂停。
    """
    plan_id = args.id
    if not _plan_path(plan_id).exists():
        print(f"[!] 计划不存在：{plan_id}（orch plan list 查看）")
        return 2
    plan = _load_yaml(plan_id)
    state = _reconcile(plan, _load_state(plan_id))
    tree = _tree(plan)
    steps_yaml = _leaf_map(plan)
    if (gate := _serial_gate(_plan_taskbook_id(plan_id, plan))) is not None:
        return gate
    _print_digest_head(plan_id, plan)

    if tpl := template_violations(plan):
        print("[!] 推进被拒：rework 模板必留节点缺失/乱序（PSEP_3 §4.3 #1）——")
        for v in tpl:
            print(f"  - {v}")
        return 2

    # ①② 收尾全部已终态条目 + 等在途全部收尾（run 的阻塞语义）
    settled = _wait_all_inflight(plan_id, plan, state)
    for s_done, o_done in settled:
        _report_handoff(plan_id, plan, state, s_done, o_done)

    # --evidence：先把一个 kind:self 组长环节收尾（PSEP_3 §2.1）
    if getattr(args, "evidence", None):
        rc = _apply_evidence(plan_id, plan, state, args.evidence)
        if rc is not None:
            return rc
        _report_after_evidence(plan_id, plan, state)
        return 0

    status_of = _status_of(state)
    # blocked/失败叶子优先（续作语义：--text 即补充信息）；否则**首可派发**（就绪谓词）
    cur = next(
        (
            s
            for s in state["steps"]
            if str(s.get("status")) in ("blocked", "failed", "run_failed")
        ),
        None,
    )
    if cur is None:
        disp = tree.dispatchable(status_of)
        cur = (
            next((s for s in state["steps"] if str(s["id"]) == disp[0]), None)
            if disp
            else None
        )
    if cur is None:
        if _plan_settled(plan, state):
            if not state.get("completed_at"):
                state["completed_at"] = _now()
                _save_state(plan_id, state)
            _report_plan_complete(plan_id, state)
            return 0
        if _inflight(state):
            print(
                f"[i] 计划 {plan_id} 无就绪环节；在途 {len(_inflight(state))} 项"
                "——交 `orch plan advance` 结账。"
            )
        else:
            _print_worklist(plan_id, plan, state)
        return 0

    # blocked 环节续作：需要组长决策后带 --text/--task 重入
    if cur["status"] in ("blocked", "failed", "run_failed") and not (
        args.text or args.task
    ):
        print(f"[!] 环节 {cur['id']} 上次状态为 {cur['status']}，需要决策：")
        _print_blocked_options(plan_id, cur, steps_yaml.get(cur["id"], {}))
        _print_worklist(plan_id, plan, state)
        return 2

    sy = steps_yaml.get(cur["id"], {})
    # 送审 gate（§4.3 #2/#3）：gate_jury 叶子须案卷 verdict=A 才派单
    if (gate := _gate_blocks(plan_id, plan, state, cur, sy)) is not None:
        return gate
    # kind:self 组长环节：不入成员队列，引导书写 + evidence 收尾
    if str(sy.get("kind") or "dispatch") == KIND_SELF:
        return _guide_self_step(plan_id, plan, state, cur, sy)
    if not (args.text or args.task) and not cur.get("task_file"):
        _print_run_guidance(plan_id, cur, sy)
        print("\n[!] 尚未提供本环节任务书（延迟书写：只强制当前环节）。")
        return 2

    slug = args.name or f"{plan_id}-{cur['id']}"
    # 环节曾有会话（blocked/failed 续作）→ 复用之；否则按组长指定或默认 latest
    sel = cur["session_id"][:8] if cur.get("session_id") else (args.session or "latest")
    member = str(sy.get("member") or cur.get("member") or "")

    # ── 直派退役（用户裁定 2026-10-14）：环节执行一律经成员队列 ────────────────
    # 对外语义**不变**（仍是「推进到交接暂停」的同步命令），内部改为：入队 → 唤醒 worker
    # → 等它收尾 → 从条目还原 outcome 再报交接。于是队列面板在途计数 / active-sessions
    # 注册 / STALL 判据对**一切** plan 环节生效（此前 state/queues/ 长期空转、面板全盲）。
    # ``runner.run_headless``/``do_dispatch`` 的直调只剩**队列 worker 一个入口**。
    from . import queues, queueworker

    # 成员闸（全链一致性，task-20261014-devops-gap-d1）：与 plan advance 同款——非队列
    # 成员（打错字）在入口就拒，别入队到无人消费的野生队列（devops 双通道后 QUEUE_MEMBERS 即真集）。
    if member not in queues.QUEUE_MEMBERS:
        print(
            f"[!] 成员 {member} 非队列成员——plan 只覆盖 "
            f"{'、'.join(queues.QUEUE_MEMBERS)}（devops 双通道：plan 环节直派 + backlog 组包）"
        )
        return 2

    _warn_dirty_face_before_enqueue(args, sy, plan_id, cur)
    item = queues.enqueue(
        member,
        text=str(args.text or ""),
        task_file=str(args.task or ""),
        dirs=_split_dirs(args.dirs.split(","))
        if args.dirs
        else _split_dirs(sy.get("dirs")),
        model_tier=args.model_tier or (str(sy["model"]) if sy.get("model") else ""),
        effort=args.effort or (str(sy["effort"]) if sy.get("effort") else ""),
        plan=plan_id,
        step=str(cur["id"]),
        session=sel,
        review=bool(sy.get("review")),
        slug=slug,
        max_turns=args.max_turns,
    )
    job_id = str(item["id"])
    cur["status"] = "queued"
    cur["job_id"] = job_id
    if args.task:
        cur["task_file"] = args.task
    _save_state(plan_id, state)
    _sync_taskbook_active(plan_id)  # 首环节入队即「执行中」

    _, wake_msg = queueworker.wake_member(member)
    outcomes = _wait_all_inflight(plan_id, plan, state)
    if not outcomes:
        # 不猜结果：留 queued 交 plan advance 结账（事件驱动语义优先于「同步要有个结论」）。
        print(
            f"[!] job {job_id} 未在 {member} 队列收尾（超时/条目失踪）——"
            f"环节 {cur['id']} 保持 queued，交 `orch plan advance {plan_id}` 结账（不重跑）。"
        )
        print(f"    {wake_msg}")
        return 2

    for s_done, o_done in outcomes:
        _report_handoff(plan_id, plan, state, s_done, o_done)
    if _plan_settled(plan, state):
        if not state.get("completed_at"):
            state["completed_at"] = _now()
            _save_state(plan_id, state)
        _report_plan_complete(plan_id, state)
    for line in _routing_card():
        print(line)
    return 0 if all(o.code == 0 for _, o in outcomes) else 2


def _split_dirs(dirs) -> list[str]:
    if not dirs:
        return []
    if isinstance(dirs, str):
        return [d.strip() for d in dirs.split(",") if d.strip()]
    return [str(d) for d in dirs]


def _queue_item_for_step(plan_id: str, step_id: str, member: str) -> dict | None:
    """环节在成员队列里的**最新** job 条目（按其 ``plan``/``step`` 反查；无则 ``None``）。

    这是 advance 事件驱动闭环的接线点：``queueworker`` 消化时把结果写回条目，advance 每次
    被唤醒后回读该条目判「在途环节收尾没有」——不必让 worker 直接改 plan 状态（解耦）。
    """
    from . import queues

    if not member:
        return None
    hit = [
        i
        for i in queues.items(member)
        if str(i.get("plan")) == plan_id and str(i.get("step")) == step_id
    ]
    return hit[-1] if hit else None


def _terminal_ready_ids(plan_id: str, plan: dict, state: dict) -> list[str]:
    """已在成员队列里**收尾但尚未结账**的叶子 id（**只读**；结账见 :func:`_settle_terminal_jobs`）。

    为什么要有只读版（backlog 20261015-135000-f2-ledger）：`advance --evidence` 想收尾一个
    self 环节，却先被在途结算挤占——结算动作与收尾动作搅在一条命令里（实案：c2 收尾被挤）。
    有它就能在「本次只做证据收尾」时**如实提示**还有几项待结账，而不动手。
    """
    from . import workq

    lm = _leaf_map(plan)
    out: list[str] = []
    for s in state["steps"]:
        if s.get("status") != "queued":
            continue
        sid = str(s["id"])
        member = str((lm.get(sid) or {}).get("member") or "")
        item = _queue_item_for_step(plan_id, sid, member)
        if item is None:
            continue
        if str(item.get("status") or "") not in (
            workq.STATUS_PENDING,
            workq.STATUS_IN_PROGRESS,
        ):
            out.append(sid)
    return out


def _settle_terminal_jobs(
    plan_id: str, plan: dict, state: dict
) -> list[tuple[dict, DispatchOutcome]]:
    """把**全部**已在成员队列收尾的 ``queued`` 叶子落终态（state 序＝叶子点路径序）。

    多 job 并行时结果**乱序到达**：一次把已到齐的都结掉（仍在途 / 条目失踪的保持
    ``queued`` 不动）；返回 ``[(叶子状态行, outcome)]`` 供调用方**逐项** ``_report_handoff``。
    本函数只改内存，调用方负责 ``_save_state``。
    """
    from . import workq

    lm = _leaf_map(plan)
    out: list[tuple[dict, DispatchOutcome]] = []
    for s in state["steps"]:
        if s.get("status") != "queued":
            continue
        sid = str(s["id"])
        member = str((lm.get(sid) or {}).get("member") or "")
        item = _queue_item_for_step(plan_id, sid, member)
        if item is None:
            continue  # 条目失踪：保守保持 queued（不猜结果）
        st = str(item.get("status") or "")
        if st in (workq.STATUS_PENDING, workq.STATUS_IN_PROGRESS):
            continue
        kind = str((item.get("result") or {}).get("kind") or "")
        if st == workq.STATUS_DONE:
            s["status"] = "done"
            if (lm.get(sid) or {}).get("review"):
                s["review"] = "review_done"  # 合并单元：宿主已交付（review 走附属）
        elif kind == "blocked":
            s["status"] = "blocked"
        else:
            s["status"] = "failed"
        s["job_id"] = str(item.get("id"))
        outcome = _outcome_from_job(member, item)
        # 会话/任务书/交付类别回填（续作复用与台账读数；run/advance 同口径）
        s["task_file"] = outcome.task_file or s.get("task_file")
        s["session_id"] = outcome.sid or s.get("session_id")
        s["kind"] = outcome.kind
        out.append((s, outcome))
    if state.get("steps") and all(x["status"] in _DONE for x in state["steps"]):
        state["completed_at"] = state.get("completed_at") or _now()
    return out


def _inflight(state: dict) -> list[dict]:
    """在途叶子（status=queued）。

    按**空表**容错读 steps（backlog 20261015-154311）：state 文件缺席时 `_load_state` 保
    falsy 语义回 `{}`（多处 `if not state:` 依赖它判「计划是否存在」），故容错落在读点上。
    """
    return [s for s in (state.get("steps") or []) if s.get("status") == "queued"]


def _worklist_panes(plan: dict, state: dict) -> dict[str, list[str]]:
    """三栏工作清单的数据面（可派发/在途/待验收）——[NEXT] 等处共用同一判据。

    「在途」含成员队列 job（``queued``）与组长自办环节（``self_awaiting``）——两者都是
    「已在办、未收尾」，只是驱动者不同（worker vs 组长）。**steps 缺席按空表读**（实案
    `pysci-orch next` 即在此崩 `KeyError: 'steps'`，backlog 20261015-154311）。
    """
    tree = _tree(plan)
    status_of = _status_of(state)
    rows = state.get("steps") or []
    return {
        "dispatchable": tree.dispatchable(status_of),
        "inflight": [
            str(s["id"])
            for s in rows
            if str(s.get("status")) in ("queued", SELF_AWAITING)
        ],
        "need_leader": [
            str(s["id"])
            for s in rows
            if str(s.get("status")) in ("blocked", "failed", "run_failed")
        ],
    }


def _plan_settled(plan: dict, state: dict) -> bool:
    """三栏全空（无可派发/无在途/无待验收）→ 计划可判完成。"""
    panes = _worklist_panes(plan, state)
    return not (panes["dispatchable"] or panes["inflight"] or panes["need_leader"])


def _print_worklist(plan_id: str, plan: dict, state: dict) -> None:
    """[NEXT] **三栏工作清单**（PSEP_3 §2.3）：可派发（按 id 序，一次引导一个）/在途/待验收。

    待验收栏含 blocked/失败决策点（可能多个）；可派发栏给出首个的成员与粗描述原文，
    其余逐一列名（本环节入队后可连续引导下一个）。
    """
    del plan_id
    panes = _worklist_panes(plan, state)
    lm = _leaf_map(plan)
    disp = panes["dispatchable"]
    print("[NEXT] 工作清单（可派发 / 在途 / 待验收）：")
    if disp:
        head = disp[0]
        sy = lm.get(head, {})
        print(
            f"  可派发（按 id 序，一次引导一个）：{head}（{sy.get('member', '?')}）"
            f" | {str(sy.get('brief', '（无）'))[:50]}"
        )
        if len(disp) > 1:
            print(f"    其余可派发：{', '.join(disp[1:])}（入队后可继续引导下一个）")
    else:
        print("  可派发：（无）")
    inflight_disp = [
        f"{sid}(self·待证据)"
        if str((lm.get(sid) or {}).get("kind") or "dispatch") == KIND_SELF
        else sid
        for sid in panes["inflight"]
    ]
    print(f"  在途：{', '.join(inflight_disp) if inflight_disp else '（无）'}")
    need = panes["need_leader"]
    print(
        f"  待验收：{', '.join(need) if need else '（无）'}"
        + ("（blocked/失败决策点见上方交接段；可能多个）" if need else "")
    )


def _routing_card() -> list[str]:
    """机械推送点用的**全局**路由卡片（PSEP_3 §3.3）：fail-soft（推送面绝不阻塞推进）。"""
    try:
        from . import nextroute

        return nextroute.card_lines(None)
    except Exception:  # noqa: BLE001 - 路由读数坏一项不塌推进
        return []


def _warn_dirty_face_before_enqueue(
    args: Any, step_yaml: dict[str, Any], plan_id: str, cur: dict[str, Any]
) -> None:
    """派发前哨（backlog 20261016-022657）：声明面 ∩ main 未提交面 非空即告警（**不阻断**）。

    组员侧无法自救（merge 交集闸在他们那里同样 rc=2、整单被抢救成 blocked），故这条必须在
    **组长看得见的地方**先说——`plan advance`/`plan run` 的主 stdout 就是这个面。
    """
    from .dispatch import warn_dirty_face_clash

    dirs = (
        _split_dirs(args.dirs.split(","))
        if getattr(args, "dirs", None)
        else _split_dirs(step_yaml.get("dirs"))
    )
    task = str(getattr(args, "task", "") or cur.get("task_file") or "")
    warn_dirty_face_clash(
        dirs,
        Path(task) if task else None,
        where=f"派发计划 {plan_id} 环节 {cur.get('id')}",
    )


def _resolve_step(leaves: dict[str, dict], sid: str) -> tuple[str | None, str]:
    """``--step`` 解析：显示名（``B.b2``）与短 id（``b2``）皆接受。

    Returns:
        ``(path, "")`` 唯一命中；``(None, 报错文案)`` 未命中或短 id 歧义。
    """
    sid = str(sid or "")
    if sid in leaves:
        return sid, ""
    cands = [p for p in leaves if p.rsplit(".", 1)[-1] == sid]
    if len(cands) == 1:
        return cands[0], ""
    hint = "、".join(list(leaves)[:8]) or "（无叶子）"
    if cands:
        return None, (
            f"[!] --step {sid} 有歧义（多个同名叶子）：{'、'.join(cands)}"
            f"——请用显示名（如 {cands[0]}）"
        )
    return None, (
        f"[!] 计划无环节 {sid}；可用环节：{hint}"
        "（--step 接受显示名与短 id，如 B.b2 或 b2）"
    )


def _spec_default_step(plan: dict, state: dict) -> str | None:
    """``plan spec`` 缺省环节：首个在途/待证据/待裁决（按状态记录序），否则首个可派发。"""
    leaves = _leaf_map(plan)
    live = ("queued", SELF_AWAITING, "blocked", "failed", "run_failed")
    for s in state.get("steps", []):
        sid = str(s.get("id"))
        if sid in leaves and str(s.get("status")) in live:
            return sid
    disp = _tree(plan).dispatchable(_status_of(state))
    return disp[0] if disp else next(iter(leaves), None)


def cmd_plan_spec(args) -> int:
    """回打环节规格（粗描述 + evidence 标题段 + 收尾命令），**任何状态可查、不改状态**。

    为什么需要（backlog tb-20261017-211704-2-2 F1）：环节已入队/待证据后，advance 不再
    引导（只印工作清单），组长要重看规格只能翻 YAML。标题段与 :func:`_check_evidence`
    同源（``ROLE_EVIDENCE_SECTIONS`` / YAML 声明），不另维护一份。
    """
    plan_id = args.id
    if not _plan_path(plan_id).exists():
        print(f"[!] 计划不存在：{plan_id}（orch plan list 查看）")
        return 2
    plan = _load_yaml(plan_id)
    state = _reconcile(plan, _load_state(plan_id))
    leaves = _leaf_map(plan)
    sid = str(getattr(args, "step", "") or "")
    if sid:
        path, err = _resolve_step(leaves, sid)
        if path is None:
            print(err)
            return 2
        sid = path
    else:
        sid = _spec_default_step(plan, state) or ""
    if not sid:
        print(f"[!] 计划 {plan_id} 无环节可显示")
        return 2
    row = next((s for s in state.get("steps", []) if str(s.get("id")) == sid), None)
    if row is None:
        row = {"id": sid, "status": "pending"}
    sy = leaves.get(sid, {})
    kind = str(sy.get("kind") or "")
    who = "self（组长）" if kind == KIND_SELF else str(sy.get("member") or "?")
    print(f"== 计划 {plan_id} 环节 {sid}（{who}，status={row.get('status')}）==")
    if kind == KIND_SELF:
        _print_self_guidance(plan_id, row, sy, plan=plan)
    else:
        _print_run_guidance(plan_id, row, sy)
    return 0


def _plan_taskbook_id(plan_id: str, plan: dict) -> str:
    """plan → 归属任务书 id（``plan.taskbook`` 优先，回落 state 反查；adhoc → 空串）。"""
    from . import taskbooks as _tb

    tb_id = str(plan.get("taskbook") or "")
    if not tb_id:
        tb_id = str((_tb.find_by_plan(plan_id) or {}).get("id") or "")
    return tb_id


def _serial_gate(tb_id: str) -> int | None:
    """serial 硬闸（用户 2026-10-17「才允许进行下一个大任务」的机械执行面）。

    Returns:
        2=已打印拒绝文案（调用方直接 return）；``None``=放行（parallel / 队首 / 无归属）。
    """
    if not tb_id:
        return None
    from . import taskbooks as _tb

    gate = _tb.queue_gate(tb_id)
    if not gate:
        return None
    print(gate)
    return 2


def _print_digest_head(plan_id: str, plan: dict) -> None:
    """plan 面输出**头部置顶**要求与历史总览（用户 2026-10-17：决策面全量、一次性可见）。

    仅当 plan 有归属任务书**且**总览含历史时打印（无历史任务书正文即原始要求，不重复）。
    附送面 fail-soft：历史读数坏不阻塞推进。
    """
    try:
        from . import taskbooks as _tb

        tb_id = _plan_taskbook_id(plan_id, plan)
        if tb_id and _tb.digest_has_history(tb_id):
            print(_tb.requirements_digest(tb_id).rstrip())
            print()
    except Exception:  # noqa: BLE001 - 附送面 fail-soft（推进主流程不被历史读数阻塞）
        pass


def _apply_skip(plan_id: str, plan: dict, state: dict, leaf_path: str) -> int:
    """``--skip <leaf>``：把 **pending** 叶子置 ``skipped``（空改进清单等「合法不做事」收尾）。

    ``skipped`` 的语义是「从未需要做」（DONE_STATES 一员），不是撤销——已 started
    一律拒绝。收尾同普通 advance：全完成则报完成，否则刷工作清单。
    """
    lm = _leaf_map(plan)
    if leaf_path not in lm:
        hint = "、".join(list(lm)[:8]) or "（无叶子）"
        print(f"[!] 计划 {plan_id} 无环节 {leaf_path}；可用环节：{hint}")
        return 2
    row = next((s for s in state["steps"] if str(s["id"]) == leaf_path), None)
    if row is None:
        state = _reconcile(plan, state)
        row = next((s for s in state["steps"] if str(s["id"]) == leaf_path), None)
    status = str((row or {}).get("status") or "pending")
    if status != "pending":
        print(
            f"[!] 环节 {leaf_path} 状态为 {status}——--skip 只对 pending 环节"
            "（skipped=从未需要做，不是撤销）"
        )
        return 2
    row["status"] = "skipped"
    row["note"] = "skip: 组长判定无需执行（空清单/不适用）"
    _save_state(plan_id, state)
    print(f"[√] 环节 {leaf_path} 置 skipped（无需执行收尾）")
    if _plan_settled(plan, state):
        if not state.get("completed_at"):
            state["completed_at"] = _now()
            _save_state(plan_id, state)
        _report_plan_complete(plan_id, state)
    else:
        _print_worklist(plan_id, plan, state)
    return 0


def cmd_plan_advance(args) -> int:
    """**事件驱动推进**（§3.5，M5）：把下一环节入队 m4 成员队列**即返回**。

    与阻塞推进原语（``plan adhoc`` 内部使用，见 ``cmd_plan_run``）的区别：本命令不等待组员，入队后由 job 完成
    **事件**触发组长再敲一次 advance。任务书延迟书写语义保留——advance 时给**当前环节**
    任务书。多 plan 并行：各 active plan 各自 advance 互不阻塞；同一成员队列 FIFO 串行、
    不同成员即并行（用户模型）。既有语义保留：交接暂停 / 机械验收 FAIL 回派 / blocked
    由组长在 drain 后决策。

    闭环：advance → 入队（环节置 ``queued``）→ worker 消化（发 JOB-DONE/STEP-REVIEW-DONE）
    → 组长 ``orch events drain`` 醒来 → 再 advance：本命令先收尾**全部**已终态条目（多 job
    并行时乱序到达，逐项交接），再按**递归就绪谓词**（PSEP_3 §2.2）入队**首个可派发**叶子，
    并以**三栏工作清单**（可派发/在途/待验收）收尾。

    **一次命令一个动作**（backlog 20261015-135000-f2-ledger）：带 ``--evidence`` 时只做组长
    环节的证据收尾，**不**顺带结算在途条目（只如实提示还有几项待结账）；结算请另跑一次不带
    ``--evidence`` 的 advance。
    """
    from . import queues, queueworker

    plan_id = args.id
    if not _plan_path(plan_id).exists():
        print(f"[!] 计划不存在：{plan_id}（orch plan list 查看）")
        return 2
    plan = _load_yaml(plan_id)
    state = _reconcile(plan, _load_state(plan_id))
    tree = _tree(plan)
    steps_yaml = _leaf_map(plan)
    if (gate := _serial_gate(_plan_taskbook_id(plan_id, plan))) is not None:
        return gate
    _print_digest_head(plan_id, plan)

    if tpl := template_violations(plan):
        print("[!] advance 被拒：rework 模板必留节点缺失/乱序（PSEP_3 §4.3 #1）——")
        for v in tpl:
            print(f"  - {v}")
        return 2

    # --skip：pending 叶子的「合法不做事」收尾（空改进清单路径；一次命令一个动作）
    skip = str(getattr(args, "skip", "") or "")
    if skip:
        return _apply_skip(plan_id, plan, state, skip)

    # ① 收尾**全部**已终态队列条目（多 job 乱序到达：一次结清，逐项交接决策点）
    #    **但带了 --evidence 时不做**（backlog 20261015-135000-f2-ledger）：一次命令一个
    #    动作，收尾证据不该被在途结算挤占（实案：`advance --evidence` 想收尾 self 环节，
    #    先撞在途结算、输出与动作搅在一起 → c2 收尾被挤）。此时只**如实提示**待结账项。
    evidence = getattr(args, "evidence", None)
    if evidence:
        ready = _terminal_ready_ids(plan_id, plan, state)
        print("[本次动作] 只做 evidence 收尾（在途结算留给下一次普通 advance）。")
        if ready:
            print(
                f"[i] 另有 {len(ready)} 项已在队里收尾、尚待结账："
                f"{'、'.join(ready)}——结算请另跑一次 `orch plan advance {plan_id}`。"
            )
    else:
        settled = _settle_terminal_jobs(plan_id, plan, state)
        if settled:
            _save_state(plan_id, state)
        for s_done, o_done in settled:
            _report_handoff(plan_id, plan, state, s_done, o_done)
    inflight = _inflight(state)
    if inflight:
        print(
            f"[i] 在途 {len(inflight)} 项仍在跑：{', '.join(str(s['id']) for s in inflight)}"
            "——等 JOB-DONE 事件驱动下一次 advance。"
        )

    # ② --evidence：把一个 kind:self 组长环节收尾（PSEP_3 §2.1）
    if evidence:
        rc = _apply_evidence(plan_id, plan, state, args.evidence)
        if rc is not None:
            return rc
        _report_after_evidence(plan_id, plan, state)
        return 0

    # ③ 目标环节：blocked/失败叶子（续作优先）；否则**首可派发**（递归就绪谓词）
    cur = next(
        (
            s
            for s in state["steps"]
            if str(s.get("status")) in ("blocked", "failed", "run_failed")
        ),
        None,
    )
    if cur is None:
        disp = tree.dispatchable(_status_of(state))
        cur = (
            next((s for s in state["steps"] if str(s["id"]) == disp[0]), None)
            if disp
            else None
        )
    if cur is None:
        if _plan_settled(plan, state):
            if not state.get("completed_at"):
                state["completed_at"] = _now()
                _save_state(plan_id, state)
            _report_plan_complete(plan_id, state)
            return 0
        _print_worklist(plan_id, plan, state)
        return 0
    if cur["status"] in ("blocked", "failed", "run_failed") and not (
        args.text or args.task
    ):
        print(f"[!] 环节 {cur['id']} 上次状态为 {cur['status']}，需要决策：")
        _print_blocked_options(plan_id, cur, steps_yaml.get(cur["id"], {}))
        _print_worklist(plan_id, plan, state)
        return 2
    sy = steps_yaml.get(cur["id"], {})
    # 送审 gate（§4.3 #2/#3）：gate_jury 叶子须案卷 verdict=A 才派单
    if (gate := _gate_blocks(plan_id, plan, state, cur, sy)) is not None:
        return gate
    # kind:self 组长环节：不入成员队列，引导书写 + evidence 收尾
    if str(sy.get("kind") or "dispatch") == KIND_SELF:
        return _guide_self_step(plan_id, plan, state, cur, sy)
    if not (args.text or args.task) and not cur.get("task_file"):
        _print_run_guidance(plan_id, cur, sy)
        print("\n[!] 尚未提供本环节任务书（延迟书写：只强制当前环节）。")
        _print_worklist(plan_id, plan, state)
        return 2

    member = str(sy.get("member") or "")
    if member not in queues.QUEUE_MEMBERS:
        print(
            f"[!] 成员 {member} 非队列成员——plan advance 只覆盖 "
            f"{'、'.join(queues.QUEUE_MEMBERS)}（devops 双通道：plan 环节直派 + backlog 组包）"
        )
        return 2
    # 续作语义（承接原同步推进子命令，backlog 20261016-220500）：环节曾有会话
    # （blocked/failed 续作）→ 复用其 sid；否则按组长指定或默认 latest。
    sel = cur["session_id"][:8] if cur.get("session_id") else (args.session or "latest")
    _warn_dirty_face_before_enqueue(args, sy, plan_id, cur)
    item = queues.enqueue(
        member,
        text=str(args.text or ""),
        task_file=str(args.task or ""),
        dirs=_split_dirs(args.dirs.split(","))
        if args.dirs
        else _split_dirs(sy.get("dirs")),
        model_tier=args.model_tier or (str(sy["model"]) if sy.get("model") else ""),
        effort=args.effort or (str(sy["effort"]) if sy.get("effort") else ""),
        plan=plan_id,
        step=str(cur["id"]),
        session=sel,
        review=bool(sy.get("review")),
        slug=str(getattr(args, "name", None) or f"{plan_id}-{cur['id']}"),
        max_turns=getattr(args, "max_turns", None),
    )
    cur["status"] = "queued"
    cur["job_id"] = str(item["id"])
    if args.task:
        cur["task_file"] = args.task
    _save_state(plan_id, state)
    _sync_taskbook_active(plan_id)  # 首环节入队即「执行中」

    _, msg = queueworker.wake_member(member, dry=bool(getattr(args, "dry", False)))
    print(
        f"[√] 计划 {plan_id} 环节 {cur['id']}（{member}）已入队 job {item['id']}"
        "——入队即返回，完成事件驱动下一次 advance"
    )
    print(f"    {msg}")
    _print_worklist(plan_id, plan, state)
    print(
        f"  事件驱动：orch events drain → "
        f'orch plan advance {plan_id} --text "<下一环节任务书>"'
    )
    for line in _routing_card():
        print(line)
    return 0


def _last_result(plan_id: str, step: dict, sy: dict) -> dict:
    """该环节上次收尾的队列结果面（``item.result``；读不到 → ``{}``，fail-open）。"""
    try:
        member = str(sy.get("member") or step.get("member") or "")
        item = _queue_item_for_step(plan_id, str(step.get("id") or ""), member)
        return dict((item or {}).get("result") or {})
    except Exception:  # noqa: BLE001 - 读数面 fail-open
        return {}


def _capacity_returned(plan_id: str, step: dict, sy: dict) -> bool:
    """该环节上次交回是否属 **capacity/余量耗尽**（backlog adhoc-guidance-fixes ③）。

    判据取自该环节队列条目的交付面：``tags`` 含 ``capacity``/``model_capability``，
    或 blocked 正文含「余量/capacity/耗尽」类字样。**为什么值得判**：capacity 交回时原会话
    余量已尽，而 [NEXT] 原先写「续作自动复用本环节会话」——照做就派进死会话（实案
    2026-10-15 `adhoc-20261015-154311` 首跳），与 SKILL.md「续作必须 `--session new`」打架。
    """
    result = _last_result(plan_id, step, sy)
    tags = {str(t).lower() for t in (result.get("tags") or [])}
    if tags & {"capacity", "model_capability"}:
        return True
    body = str(result.get("body") or "") + str(result.get("error") or "")
    return any(w in body for w in ("余量", "capacity", "配额", "quota"))


def _print_blocked_options(plan_id: str, step: dict, sy: dict) -> None:
    sid8 = (step.get("session_id") or "")[:8]
    result = _last_result(plan_id, step, sy)
    run_failed = str(result.get("kind") or "") == "run_failed"
    capacity = _capacity_returned(plan_id, step, sy)
    exhausted = capacity or run_failed
    print()
    print("[NEXT] 可选：")
    if exhausted:
        if run_failed and not capacity:
            head = "**上次为 run_failed（基础设施终止）→ 建议 `--session new`**"
        else:
            head = "**本环节会话余量已尽 → 建议 `--session new`**"
        print(
            f"  1) 补充信息续作本环节（{head}）："
            f'pysci-orch plan advance {plan_id} --session new --text "<补充>"'
        )
        print(
            "     原因：run_failed/余量耗尽后原会话不可复用（自动复用大概率再撞墙），"
            "续作进它＝派进死会话（SKILL.md 派发要领「续作必须 --session new」）。"
        )
    else:
        print(
            f'  1) 补充信息续作本环节：pysci-orch plan advance {plan_id} --text "<补充>"'
        )
    print('  2) 咨询副组长：pysci-orch deputy consult "<问题>"')
    print(
        f"  3) 修改计划（编辑 {_plan_path(plan_id).name} 后重新 advance；或跳过本环节：--skip）"
    )
    print("  4) 中止计划并上报用户")
    if sid8 and not exhausted:
        print(f"  （本环节会话 {sid8}，续作自动复用）")
    elif exhausted:
        print("  （本环节会话**不复用**——见 1) 的 `--session new`）")
    if run_failed:
        print(
            "  [run_failed 恢复] 先盘点半成品再决定续作/重做：读产物目录、交付说明、管线脚本的"
            "在盘情况——完整→续作只核对与正式交付、不重做；缺失→整单重做"
            "（见 dispatch-playbook「run_failed 恢复清单」）。"
        )


def _report_handoff(
    plan_id: str, plan: dict, state: dict, cur: dict, outcome: DispatchOutcome
) -> None:
    """交接处暂停报告：本环节摘要 + 下一环节原文 recall + 决策选项。"""
    print(f"== 计划 {plan_id}：环节 {cur['id']} → {cur['status']} ==")
    icon = {"result": "√", "blocked": "✗", "parse_error": "?", "run_failed": "✗"}.get(
        outcome.kind, "?"
    )
    env = outcome.envelope
    dur = env.duration_ms / 1000 if env else 0
    print(f"[{icon}] {outcome.member} 交付={outcome.kind}（{dur:.0f}s）")
    print(
        f"---- 交付正文（前 600 字）----\n{outcome.body[:600]}\n----------------------------"
    )
    if getattr(outcome, "body_file", ""):
        print(f"  交付全文留档：{outcome.body_file}")
    for c in outcome.checks:
        print(f"  验收[{c['verdict']}] {c['check']}: {Path(c['path']).name}")
    if outcome.suggestion_id:
        print(f"  改进建议已登记：{outcome.suggestion_id}（orch approve/reject 处理）")

    if cur["status"] in ("blocked", "failed"):
        sy = {
            str(s.get("id") or f"s{i + 1}"): s
            for i, s in enumerate(plan.get("steps") or [])
        }.get(cur["id"], {})
        _print_blocked_options(plan_id, cur, sy)
        return
    if cur.get("review") == "not_wired":
        print(
            "  [i] 本环节标记 review:true，但 reviewer 尚未接入（Phase 3）——已记 not_wired 跳过。"
        )

    panes = _worklist_panes(plan, state)
    nxt_id = next(iter(panes["dispatchable"]), "")
    if not nxt_id:
        # 完成判定交给调用方**收尾一次**（多 job 乱序同批到达时逐项交接，避免重复播报）。
        _print_worklist(plan_id, plan, state)
        return
    nxt = next((s for s in state["steps"] if str(s["id"]) == nxt_id), None)
    if nxt is None:  # 防御：理论上不可达（可派发项必有状态行）
        _print_worklist(plan_id, plan, state)
        return
    steps_yaml = _leaf_map(plan)
    ny = steps_yaml.get(nxt["id"], {})
    print()
    print(f"== 交接决策点：下一环节 {nxt['id']}（{ny.get('member', '?')}）==")
    print(f"  粗描述原文：{ny.get('brief', '（无）')}")
    print(
        f"  白名单建议：{ny.get('dirs', [])}；档位：{ny.get('model', '(成员默认)')}；"
        f"effort：{ny.get('effort', '(默认=中)')}"
    )
    if outcome.kind == "result":
        print("  交接物：上一环节交付正文中的产物路径（书写任务书时引用）。")
    _print_worklist(plan_id, plan, state)
    print("[NEXT] 可选：")
    print(
        f"  1) 推进下一环节（书写任务书后）：pysci-orch plan advance "
        f'{plan_id} --text "..."'
        "　——入队即返回【长任务首选】"
    )
    print(f"  2) 修改后续计划：编辑 {_plan_path(plan_id)} 后重新推进（按环节 id 对账）")
    print(f"  3) 中止：pysci-orch plan drop {plan_id}")


def _report_plan_complete(plan_id: str, state: dict) -> None:
    print(f"== 计划 {plan_id} 全部环节完成（{state.get('completed_at') or _now()}）==")
    entries = [e for e in read_all() if e.plan == plan_id]
    s = summarize(entries)
    print(
        f"计划统计：{s['hops']} 跳，总耗时 {s['duration_ms'] / 1000:.0f}s，"
        f"{format_est_tokens(s)}，{format_credits(s)}"
    )
    for mid, row in sorted(s["by_member"].items()):
        print(
            f"  {mid:<10} 跳数={row['hops']}  耗时占比={row['duration_pct']}%  "
            f"est占比={row['est_tokens_pct']}%  credits占比={row['credits_pct']}%"
        )
    print()
    print("[NEXT]")
    print(
        "  → 向用户交付汇总：成果路径 + 上述统计 + 未决事项（含待审批建议）+ 你自己的改进建议（可选）"
    )
    # taskbook 联动提示（backlog adhoc-guidance-fixes ②）：plan 被任务书 link 过时，收尾的
    # 两条命令**直接给出**——否则全凭组长记得 `taskbooks sync` + `taskbooks report <id>`
    # （实测漏了得回头补）。未关联则**不加噪声**。
    tb_ids: list[str] = []
    try:
        from . import taskbooks as _tb

        tb_ids = [
            str(t.get("id"))
            for t in _tb.list_all()
            if str(t.get("plan_id") or "") == plan_id
        ]
    except Exception:  # noqa: BLE001 - 读数面 fail-open（提示加不上不影响收尾）
        tb_ids = []
    if tb_ids:
        print(
            f"  → 本 plan 关联任务书 {', '.join(tb_ids)}：跑 "
            "`pysci-orch taskbooks sync`（对账 delivered）→ "
            f"`pysci-orch taskbooks report {tb_ids[0]}`（组装全流程报告，TUI 详情页即读它）"
        )


# ---------------------------------------------------------------------------
# 返工流骨架（PSEP_3 §4.1/§4.2/§4.3）：模板三档 / 必留节点 lint / 送审 gate
# ---------------------------------------------------------------------------
MODE_FULL = "full"
MODE_NO_INSPECTOR = "no_inspector"
MODE_REWORK_ONLY = "rework_only"
REWORK_MODES: tuple[str, ...] = (MODE_FULL, MODE_NO_INSPECTOR, MODE_REWORK_ONLY)


def _rework_A(tb_id: str, rnd: int) -> dict:
    """A 综合项目改进（serial superstep）：a1 成因分析 → … → a5 改进实施。

    ``rnd>=2`` 时 a1 触发**历次改进强制展示**（``expand_history``）且 evidence 须含
    「前次改进复盘」标题段（§4.3 #4）。
    """
    a1_sections = ["成因分析", "结论"]
    a1: dict = {"id": "a1", "kind": "self", "brief": "成因分析报告", "role": "cause"}
    if rnd >= 2:
        a1_sections = ["前次改进复盘", "成因分析", "结论"]
        a1["expand_history"] = True
    a1["evidence_sections"] = a1_sections
    return {
        "id": "A",
        "mode": "serial",
        "brief": "综合项目改进",
        "children": [
            a1,
            {
                "id": "a2",
                "kind": "self",
                "brief": "解决方案报告",
                "role": "solution",
                "evidence_sections": ["解决方案", "结论"],
            },
            {
                "id": "a3",
                "kind": "self",
                "brief": f"送审 jury（改进后状态 vs 现状；案卷 {tb_id}-jury）",
                "role": "review",
                "evidence_sections": ["送审结论"],
            },
            {
                "id": "a4",
                "kind": "self",
                "brief": "拆解改进派单",
                "role": "improvement_plan",
                "gate_jury": f"{tb_id}-jury",
            },
            {
                "id": "a5",
                "member": "devops",
                "brief": "改进实施",
                "member_placeholder": True,
            },
        ],
    }


def _rework_B(node_id: str, tb_id: str) -> dict:
    """B 返工执行（serial superstep）：b1 重拆解 → b2 返工生产 → b_last 全流程报告。"""
    return {
        "id": node_id,
        "mode": "serial",
        "brief": "返工执行",
        "children": [
            {"id": "b1", "kind": "self", "brief": "重拆解返工", "role": "rework_plan"},
            {
                "id": "b2",
                "member": "devops",
                "brief": "返工生产环节",
                "member_placeholder": True,
            },
            {
                "id": "b_last",
                "kind": "self",
                "brief": "全流程报告组装",
                "role": "final_report",
            },
        ],
    }


def _rework_C(tb_id: str) -> dict:
    """C 监察改进（serial superstep）：可发现性送审 → 方案 → 派单 → 实施。"""
    return {
        "id": "C",
        "mode": "serial",
        "brief": "监察改进",
        "children": [
            {
                "id": "c1",
                "kind": "self",
                "brief": f"可发现性案卷送审（案卷 {tb_id}-discoverability）",
                "role": "discoverability",
                "evidence_sections": ["可发现性判断"],
            },
            {"id": "c2", "kind": "self", "brief": "改进方案（若判需改进）"},
            {
                "id": "c3",
                "kind": "self",
                "brief": "拆解派单",
                "gate_jury": f"{tb_id}-discoverability",
            },
            {
                "id": "c4",
                "member": "devops",
                "brief": "改进实施",
                "member_placeholder": True,
            },
        ],
    }


def _rework_L() -> dict:
    """L 组长改进（serial superstep，**恒居顶层末尾**；用户 2026-10-17 裁决）。

    l1 反思 → l2 拆解改进派单 → l3 改进实施（devops 占位，仿 A 组 a4→a5）。l2 用**新
    role** ``leader_improvement_plan``：复用 ``improvement_plan`` 会与 a4 在
    ``taskbooks._role_texts``（按 role 为键）互相覆盖。空改进清单合法——l2 写明后
    ``plan advance <pid> --skip L.l3`` 收尾。
    """
    return {
        "id": "L",
        "mode": "serial",
        "brief": "组长改进",
        "children": [
            {
                "id": "l1",
                "kind": "self",
                "brief": "harnessing 摩擦反思",
                "role": "leader_reflection",
                "evidence_sections": ["摩擦反思", "改进清单", "结论"],
            },
            {
                "id": "l2",
                "kind": "self",
                "brief": "拆解改进派单（空改进清单合法）",
                "role": "leader_improvement_plan",
            },
            {
                "id": "l3",
                "member": "devops",
                "brief": "改进实施",
                "member_placeholder": True,
            },
        ],
    }


def _improve_of(src: dict) -> dict[str, bool]:
    """plan/任务书记录 → 改进三标志（真源 ``improve`` 优先，回退 legacy 档名）。"""
    from . import taskbooks as _tb

    return _tb.improve_flags(
        {
            "improve": src.get("improve"),
            "mode": src.get("rework_mode") or src.get("mode") or "",
        }
    )


def _rework_layout(
    tb_id: str, rnd: int, improve: dict[str, bool]
) -> tuple[list[dict], tuple[str, ...]]:
    """返工骨架**单源**（2026-10-17 三标志泛化）：``(顶层 steps[], 必留叶子 DFS 序)``。

    infra → A 综合项目改进；inspector → BC 并行组（B 在前＝派发级跟随），否则 B 单独；
    leader → L 追加**顶层末尾**——顶层 steps[] 是隐式 serial 组
    （:meth:`planflow.PlanTree.prefix_siblings`），「任务主体全部完成后才进组长改进」
    由就绪谓词机械保证，不靠自觉。
    """
    steps: list[dict] = []
    required: list[str] = []
    if improve.get("infra"):
        steps.append(_rework_A(tb_id, rnd))
        required += [f"A.a{i}" for i in range(1, 6)]
    b = _rework_B("B", tb_id)
    if improve.get("inspector"):
        steps.append(
            {
                "id": "BC",
                "mode": "parallel",
                "brief": "返工与监察（并行）",
                "children": [b, _rework_C(tb_id)],
            }
        )
        required += ["BC.B.b1", "BC.B.b2", "BC.B.b_last"]
        required += [f"BC.C.c{i}" for i in range(1, 5)]
    else:
        steps.append(b)
        required += ["B.b1", "B.b2", "B.b_last"]
    if improve.get("leader"):
        steps.append(_rework_L())
        required += ["L.l1", "L.l2", "L.l3"]
    return steps, tuple(required)


#: 旧三档 → 必留叶子**派生**兼容表（单源 :func:`_rework_layout`；旧读取面/测试用）。
REWORK_REQUIRED: dict[str, tuple[str, ...]] = {
    MODE_FULL: _rework_layout("", 1, {"infra": True, "inspector": True})[1],
    MODE_NO_INSPECTOR: _rework_layout("", 1, {"infra": True})[1],
    MODE_REWORK_ONLY: _rework_layout("", 1, {})[1],
}


def rework_required_leaves(improve: dict[str, bool]) -> tuple[str, ...]:
    """给定改进标志组合的**必留叶子**（DFS 序＝骨架顺序）——单源 :func:`_rework_layout`。"""
    return _rework_layout("", 1, improve)[1]


def template_violations(plan: dict) -> list[str]:
    """template=rework 计划的**必留节点 lint**（缺失/乱序即违规）；非模板计划恒空。"""
    if str(plan.get("template") or "") != "rework":
        return []
    required = list(rework_required_leaves(_improve_of(plan)))
    actual = _tree(plan).leaf_paths()
    out: list[str] = []
    missing = [p for p in required if p not in actual]
    if missing:
        out.append("必留节点缺失：" + "、".join(missing))
    present = [p for p in actual if p in required]
    if present != required:
        out.append(
            "必留节点乱序（应 "
            + " → ".join(required)
            + "；实 "
            + " → ".join(present)
            + "）"
        )
    return out


def build_rework_plan(
    tb_id: str,
    *,
    improve: dict[str, bool] | None = None,
    mode: str | None = None,
    goal: str = "",
    rework_round: int = 1,
) -> dict:
    """生成返工 plan **骨架**（2026-10-17 三标志泛化；:func:`_rework_layout` 单源组装）。

    结构：A、BC 顶层 serial（A 全 done → B 起）；BC 并行组内 B id 靠前、C 靠后（B 派发后
    C 即可推进＝派发级）；B 是串行链（重拆解→返工生产→全流程报告）；**L 恒居顶层末尾**
    （前序全 done 才就绪）。``rework_round>=2`` 时 a1 触发历次改进强制展示 + 复盘段
    lint（§4.3 #4）。

    Args:
        improve: 改进三标志（真源）。
        mode: 旧三档兼容入口（与 ``improve`` 互斥；两者皆缺按 ``full``＝旧签名缺省）。

    Raises:
        ValueError: ``improve`` 与 ``mode`` 同给 / 未知模式。
    """
    from . import taskbooks as _tb

    if improve is not None and mode is not None:
        raise ValueError("improve 与 mode 互斥（improve 是真源，mode 仅为旧接口兼容）")
    if mode is not None and mode not in REWORK_MODES:
        raise ValueError(f"未知返工模式：{mode}（合法：{'、'.join(REWORK_MODES)}）")
    flags = (
        {k: bool(improve.get(k)) for k in _tb.IMPROVE_KEYS}
        if improve is not None
        else _tb.improve_flags({"mode": mode or MODE_FULL})
    )
    rnd = int(rework_round or 1)
    steps, _required = _rework_layout(tb_id, rnd, flags)
    return {
        "goal": goal or f"返工：{tb_id}",
        "template": "rework",
        "improve": flags,
        "rework_mode": _tb.legacy_mode_of(flags),
        "rework_round": rnd,
        "taskbook": tb_id,
        "steps": steps,
    }


def plan_tree_lines(plan_id: str) -> list[str]:
    """plan 的 **superstep 进度树**（文本；组带聚合态、叶子带 ``kind=status``）——TUI 返工卡片用。"""
    if not _plan_path(plan_id).exists():
        return []
    plan = _load_yaml(plan_id)
    status_of = _status_of(_load_state(plan_id))
    tree = _tree(plan)
    # 图标串自带**尾随空格**（图标与文字之间必须隔一个空格——终端宽字符重叠修复，
    # 且使源码字面量本身不含「图标紧贴字符」形态）；格式化处不再另加空格。
    icons = {
        "done": "✔ ",
        "queued": "▶ ",
        "self_awaiting": "✎ ",
        "blocked": "✗ ",
        "failed": "✗ ",
        "run_failed": "✗ ",
    }
    out: list[str] = []

    def walk(items: list, prefix: str, depth: int) -> None:
        for i, raw in enumerate(items):
            if not isinstance(raw, dict):
                continue
            name = str(raw.get("id") or f"s{i + 1}")
            path = f"{prefix}.{name}" if prefix else name
            indent = "  " * depth
            kids = raw.get("children")
            if kids:
                agg = tree.aggregate(path, status_of)
                mark = "✔ " if agg["done"] else ("▶ " if agg["started"] else "· ")
                out.append(f"{indent}{mark}{name} [{raw.get('mode') or 'serial'}]")
                walk(kids, path, depth + 1)
            else:
                st = status_of(path)
                kind = str(raw.get("kind") or "dispatch")
                out.append(f"{indent}{icons.get(st, '· ')}{name} ({kind}={st})")

    walk(plan.get("steps") or [], "", 0)
    return out


def plan_tree_nodes(pid: str) -> list[dict]:
    """superstep 树的**结构化**节点（plan YAML 树 × state 对账一次；TUI 计划详情页用）。

    每节点 ``{id, path, kind, mode, status, member, brief, children}``：
    ``kind`` ∈ {``super``, ``leaf``}；组 ``mode`` ∈ {``serial``, ``parallel``}（叶子空串）；
    叶子 ``status`` = state 原值（缺省 pending）；组 ``status`` = 由叶子推导的聚合
    （``done``=全完成 / ``started``=任一起跑 / ``pending``=全未动，不落盘）。
    ``plan_tree_lines``（文本树，返工卡片仍用）保留不动；plan 缺席 → 空表。
    """
    if not _plan_path(pid).exists():
        return []
    plan = _load_yaml(pid)
    status_of = _status_of(_load_state(pid))
    tree = _tree(plan)

    def walk(items: list, prefix: str) -> list[dict]:
        out: list[dict] = []
        for i, raw in enumerate(items):
            if not isinstance(raw, dict):
                continue
            name = str(raw.get("id") or f"s{i + 1}")
            path = f"{prefix}.{name}" if prefix else name
            kids = raw.get("children")
            is_group = isinstance(kids, list) and bool(kids)
            if is_group:
                agg = tree.aggregate(path, status_of)
                status = (
                    "done"
                    if agg["done"]
                    else ("started" if agg["started"] else "pending")
                )
                mode = (
                    "parallel"
                    if str(raw.get("mode") or "serial") == "parallel"
                    else "serial"
                )
                children = walk(kids, path)
            else:
                status = status_of(path)
                mode = ""
                children = []
            out.append(
                {
                    "id": name,
                    "path": path,
                    "kind": "super" if is_group else "leaf",
                    "mode": mode,
                    "status": status,
                    "member": str(raw.get("member") or ""),
                    "brief": str(raw.get("brief") or ""),
                    "children": children,
                }
            )
        return out

    return walk(plan.get("steps") or [], "")


def _jury_case_ruling(case_id: str) -> tuple[str, str]:
    """读 jury 案卷留痕的**确定裁决**：返回 ``(verdict∈{"A","B",""}, 裁决书文本)``。"""
    from . import jury as _jury

    p = _jury.record_path(case_id)
    try:
        rec = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "", ""
    v = str(rec.get("final_verdict") or rec.get("verdict") or "")
    text = str((rec.get("verdict_summary") or {}).get("text") or "")
    return (v if v in ("A", "B") else ""), text


def _abandon_for_gate(
    plan_id: str, plan: dict, state: dict, leaf: dict, fails: list[dict]
) -> int:
    """两次送审不过 → 中止计划（abandoned）＋任务书 failed ＋附两轮裁决书（§4.3 #3）。"""
    state["abandoned_at"] = _now()
    state["abandon_reason"] = f"两次送审不过（{leaf['id']} gate）"
    state["gate_rulings"] = fails
    _save_state(plan_id, state)
    tb = str(plan.get("taskbook") or state.get("taskbook") or "")
    if tb:
        try:
            from . import taskbooks as _tb

            _tb.transition(
                tb, _tb.STATUS_FAILED, note="两次送审不过，已中止（附两轮裁决书）"
            )
        except Exception:  # noqa: BLE001 - 中止链绝不因台账写失败而回退
            pass
    print(f"== 计划 {plan_id} 已中止（abandoned）：两次送审不过（{leaf['id']}）==")
    for i, f in enumerate(fails, 1):
        print(f"  第 {i} 轮裁决书：{f.get('text') or f.get('case')}")
    if tb:
        print(f"  任务书 {tb} → failed（待用户补充指引；附两轮裁决书于 plan 状态）")
    return 2


def _gate_blocks(
    plan_id: str, plan: dict, state: dict, leaf: dict, sy: dict
) -> int | None:
    """送审 gate（§4.3 #2/#3）：``gate_jury`` 叶子派发前置校验——案卷 ``verdict=A`` 才放行。

    Returns ``None``＝放行；否则返回拒绝码（``2``）——两次 verdict=B 时中止计划并返回。
    """
    case_id = str(sy.get("gate_jury") or "")
    if not case_id:
        return None
    verdict, ruling = _jury_case_ruling(case_id)
    if verdict == "A":
        return None
    sid = str(leaf["id"])
    fails = state.setdefault("gate_fails", {}).setdefault(sid, [])
    if verdict == "B":
        fails.append({"ts": _now(), "case": case_id, "text": ruling})
        if len(fails) >= 2:
            return _abandon_for_gate(plan_id, plan, state, leaf, fails)
        _save_state(plan_id, state)
        print(
            f"[!] 送审 gate 未过（第 {len(fails)} 次 verdict=B，案卷 {case_id}）"
            f"——拒绝派发 {sid}"
        )
        print(f"    裁决书：{ruling or '（未读得裁决书）'}")
        _print_worklist(plan_id, plan, state)
        return 2
    print(
        f"[!] 送审 gate：案卷 {case_id} 未落/未裁决（verdict 非 A）"
        f"——先送审（orch jury）再派单 {sid}"
    )
    _print_worklist(plan_id, plan, state)
    return 2


def cmd_plan_rework(args) -> int:
    """``orch plan rework <tb-id>``：由返工任务书生成骨架并登记（§4.2，三标志泛化）。

    改进项来源优先级：``--improve``（显式，逗号串，空串=全不含）＞ ``--mode``（旧三档
    兼容映射）＞ **任务书标志**（默认——reject 时的复选即真源，无需再报一遍）。
    """
    tb_id = args.id
    if (gate := _serial_gate(tb_id)) is not None:
        return gate
    raw_improve = str(getattr(args, "improve", "") or "")
    mode = getattr(args, "mode", None)
    improve: dict[str, bool] | None = None
    if raw_improve:
        from . import taskbooks as _tb

        keys = [k.strip() for k in raw_improve.split(",") if k.strip()]
        unknown = set(keys) - set(_tb.IMPROVE_KEYS)
        if unknown:
            print(
                f"[!] 未知改进标志：{'、'.join(sorted(unknown))}"
                f"（合法：{'、'.join(_tb.IMPROVE_KEYS)}）"
            )
            return 2
        improve = {k: k in keys for k in _tb.IMPROVE_KEYS}
        source = "显式 --improve"
    elif mode:
        if mode not in REWORK_MODES:
            print(f"[!] 未知返工模式：{mode}（合法：{'、'.join(REWORK_MODES)}）")
            return 2
        source = "兼容 --mode 映射"
    else:
        from . import taskbooks as _tb

        st = _tb.load_state(tb_id)
        if st is not None:
            improve = _tb.improve_flags(st)
            source = "任务书标志"
        else:
            mode = MODE_FULL  # 任务书缺席：回落旧缺省（骨架登记不因此拦）
            source = "任务书缺席回落 full"
    goal = str(getattr(args, "goal", "") or "")
    rnd = 1
    try:
        from . import taskbooks as _tb

        rnd = int((_tb.load_state(tb_id) or {}).get("rework_round") or 1)
    except Exception:  # noqa: BLE001 - 任务书缺席不拦骨架（round 回落 1）
        rnd = 1
    try:
        plan = build_rework_plan(
            tb_id,
            improve=improve,
            mode=(mode if improve is None else None),
            goal=goal,
            rework_round=rnd,
        )
    except ValueError as exc:
        print(f"[!] {exc}")
        return 2
    plan_id = str(getattr(args, "plan_id", "") or f"rework-{tb_id}")
    if _plan_path(plan_id).exists() and not getattr(args, "force", False):
        print(
            f"[!] 计划 {plan_id} 已存在（覆盖用 --force）；续跑用 plan advance {plan_id}"
        )
        return 2
    PLANS_DIR.mkdir(parents=True, exist_ok=True)
    plan["id"] = plan_id
    _plan_path(plan_id).write_text(_dump_plan(plan), encoding="utf-8")
    _save_state(plan_id, _reconcile(plan, {"plan_id": plan_id, "created": _now()}))
    from . import taskbooks as _tb

    flags = _improve_of(plan)
    print(
        f"[√] 返工计划已登记：{plan_id}"
        f"（{_tb.improve_label({'improve': flags})}，模板骨架 {len(plan['steps'])} 顶层节点）"
    )
    print(f"    任务书 {tb_id}；标志来源：{source}")
    print(f"    必留叶子：{'、'.join(rework_required_leaves(flags))}")
    try:
        digest = _tb.requirements_digest(tb_id)
    except Exception:  # noqa: BLE001 - 附送面 fail-soft
        digest = ""
    if digest:
        print()
        print(digest.rstrip())
    tree = _tree(plan)
    for leaf in tree.leaves:
        indent = "  " * leaf.path.count(".")
        who = leaf.yaml.get("member") or (
            "self（组长）" if leaf.kind == KIND_SELF else "?"
        )
        if leaf.yaml.get("member_placeholder"):  # backlog 20261015-235101-devops
            who += "（占位，按 §5 分工于拆解环节用 plan reassign 改派）"
        gate = (
            f"  [gate:{leaf.yaml['gate_jury']}]" if leaf.yaml.get("gate_jury") else ""
        )
        print(
            f"    - {indent}{leaf.path}: {who} | {str(leaf.yaml.get('brief', ''))[:50]}{gate}"
        )
    # [NEXT] 引导（用户 2026-10-15「返工要自行 grep 才找到 plan rework」）：骨架登记完的
    # 下一步是**链接任务书 + 推进首环节**，两条命令原样写出（不再靠 grep 找入口）。
    print("\n[NEXT]")
    print("  1) 链接任务书（记 plan_id 并置 planning）：")
    print(f"     orch taskbooks link {tb_id} --plan {plan_id}")
    print(f"  2) 推进首环节（执行即置任务书 active）：orch plan advance {plan_id}")
    print("     首环节若为组长自办（kind:self），会打印 evidence 规格与收尾命令")
    print(f"  3) 看返工卡片与流程报告：pysci（TUI）→ 任务书 → 选中 {tb_id} → Enter")
    return 0


def cmd_plan_edit(args) -> int:
    """**结构化**改既有环节的 ``brief``/``dirs``，并自动对账（backlog 20261015-222248）。

    为什么需要它：`plan new` 登记时把 YAML 交给 ``yaml.safe_dump`` **重新折行**，人工 Edit 的
    ``old_string``（照自己写入时的折行写）**逐字对不上** → 实测连续两次 `0 occurrences found`，
    每次都得先 `sed` 读回实际折行再改。本命令让人**不必猜折行**：直接给"改什么"。

    与 `amend` 的分工：`amend` 是「你改了 YAML 文件，我来校验对账」；`edit` 是「你说改什么，
    我来改并对账」。护栏同 `amend`（模板必留节点 + 已 started 结构冻结——字段编辑不动结构，
    本应恒过，跑一遍是**保险**：真被拦说明计划已进入不该改的状态）。
    """
    plan_id = args.id
    if not _plan_path(plan_id).exists():
        print(f"[!] 计划不存在：{plan_id}")
        return 2
    plan = _load_yaml(plan_id)
    sid = str(getattr(args, "step", "") or "")
    leaves = _leaf_map(plan)
    if sid not in leaves:
        hint = "、".join(list(leaves)[:8]) or "（无叶子）"
        print(f"[!] 计划 {plan_id} 无环节 {sid}；可用环节：{hint}")
        return 2
    leaf = leaves[sid]
    if args.brief is None and not args.append_brief and args.dirs is None:
        print("[!] 未给任何编辑项（至少一个：--brief / --append-brief / --dirs）")
        return 2

    if args.brief is not None:
        leaf["brief"] = str(args.brief)
    if args.append_brief:
        cur = str(leaf.get("brief") or "").strip()
        leaf["brief"] = f"{cur}；{args.append_brief}" if cur else str(args.append_brief)
    if args.dirs is not None:
        leaf["dirs"] = _split_dirs(args.dirs)

    tpl = template_violations(plan)
    if tpl:
        print("[!] edit 被拒：rework 模板必留节点不完整（PSEP_3 §4.3 #1）——")
        for v in tpl:
            print(f"  - {v}")
        return 2
    old = _load_state(plan_id)
    violations = freeze_violations(
        {str(s["id"]): str(s.get("status") or "pending") for s in old.get("steps", [])},
        _tree(plan).leaf_paths(),
    )
    if violations:
        print("[!] edit 被拒：已 started 的 superstep 结构冻结（PSEP_3 §2.4）——")
        for v in violations:
            print(f"  - {v}")
        return 2

    _plan_path(plan_id).write_text(_dump_plan(plan), encoding="utf-8")
    _save_state(plan_id, _reconcile(plan, old))
    print(f"[√] 环节 {sid} 已更新（计划 {plan_id}）：")
    print(f"    brief = {str(leaf.get('brief') or '')[:80]}")
    print(f"    dirs  = {leaf.get('dirs') or '（未声明）'}")
    print(f"[NEXT] 核对：pysci-orch plan list / plan advance {plan_id}")
    return 0


def _children_items(plan: dict, parent_path: str) -> list | None:
    """按点路径取父节点的子列表（``""`` → 顶层 ``steps``）；路径不可解析回 ``None``。"""
    items: Any = plan.get("steps")
    if not parent_path:
        return items if isinstance(items, list) else None
    for seg in parent_path.split("."):
        if not isinstance(items, list):
            return None
        node = next(
            (it for it in items if isinstance(it, dict) and str(it.get("id")) == seg),
            None,
        )
        if node is None:
            return None
        items = node.get("children")
    return items if isinstance(items, list) else None


def _item_index(items: list, name: str) -> int:
    for i, it in enumerate(items):
        if isinstance(it, dict) and str(it.get("id")) == name:
            return i
    return -1


def _commit_plan_edit(plan_id: str, plan: dict, old: dict, action: str) -> int:
    """reassign/split 共用收尾：模板必留节点 lint → started 冻结 → 写 YAML → 对账落状态。"""
    tpl = template_violations(plan)
    if tpl:
        print(f"[!] {action} 被拒：rework 模板必留节点不完整（PSEP_3 §4.3 #1）——")
        for v in tpl:
            print(f"  - {v}")
        return 2
    violations = freeze_violations(
        {str(s["id"]): str(s.get("status") or "pending") for s in old.get("steps", [])},
        _tree(plan).leaf_paths(),
    )
    if violations:
        print(f"[!] {action} 被拒：已 started 的 superstep 结构冻结（PSEP_3 §2.4）——")
        for v in violations:
            print(f"  - {v}")
        return 2
    _plan_path(plan_id).write_text(_dump_plan(plan), encoding="utf-8")
    _save_state(plan_id, _reconcile(plan, old))
    return 0


def _refuse_if_started(old: dict, sid: str, action: str) -> bool:
    """已 started 叶子拒绝 reassign/split（结构冻结只拦删/挪，改派与插叶须自查）。"""
    status = _status_of(old)(sid)
    if status not in STARTED_STATES:
        return False
    print(
        f"[!] {action} 被拒：环节 {sid} 已 {status}（已派发/已 started 冻结，"
        "PSEP_3 §2.4）"
    )
    return True


def cmd_plan_reassign(args) -> int:
    """把**未派发**叶子改派给另一成员（``member_placeholder`` 的正规出口）。

    为什么需要它（backlog 20261017-104504）：返工模板把 b2 置为
    ``member: devops + member_placeholder: true``（语义是「按 §5 分工于拆解环节改派」），
    但改派此前只能手编 plan YAML 再 advance——缩进/字段名写错要到 advance 才报。改派不动
    结构，故照走 amend 的两道护栏；已 started 的叶子直接拒绝（防账实脱节）。
    """
    plan_id = args.id
    if not _plan_path(plan_id).exists():
        print(f"[!] 计划不存在：{plan_id}")
        return 2
    plan = _load_yaml(plan_id)
    leaves = _leaf_map(plan)
    sid, err = _resolve_step(leaves, str(args.step or ""))
    if sid is None:
        print(err)
        return 2
    member = str(args.member or "").strip()
    if not member:
        print("[!] --member 不能为空")
        return 2
    leaf = leaves[sid]
    if str(leaf.get("kind") or "") == KIND_SELF:
        print(f"[!] 环节 {sid} 是 kind:self（组长自办），无成员可改派")
        return 2
    old = _load_state(plan_id)
    if _refuse_if_started(old, sid, "reassign"):
        return 2
    prev = str(leaf.get("member") or "")
    leaf["member"] = member
    had_placeholder = bool(leaf.pop("member_placeholder", False))
    rc = _commit_plan_edit(plan_id, plan, old, "reassign")
    if rc:
        return rc
    tail = "（占位符已清）" if had_placeholder else ""
    print(f"[√] 环节 {sid} 成员：{prev or '（未声明）'} → {member}{tail}")
    print(f"[NEXT] 核对进度：pysci-orch plan advance {plan_id}")
    return 0


def cmd_plan_split(args) -> int:
    """把**未派发**叶子拆成两叶（新叶插在目标叶之前/之后）——插叶不再手编 YAML。

    backlog 20261017-104504 实案：tb-3-2 把 b2 拆成 b2_pre(sim) + b2(figure)。新 id 缺省
    ``<leaf>_pre``／``<leaf>_post``（``--new-id`` 覆盖）；原叶保持 id/成员/描述不变，故
    模板必留节点与 started 冻结照常执行——必留叶子删除/乱序仍被拒，已 started 原叶直接
    拒绝（避免把新叶插进已推进的串行链中段）。
    """
    plan_id = args.id
    if not _plan_path(plan_id).exists():
        print(f"[!] 计划不存在：{plan_id}")
        return 2
    plan = _load_yaml(plan_id)
    sid = str(args.step or "")
    leaves = _leaf_map(plan)
    if sid not in leaves:
        hint = "、".join(list(leaves)[:8]) or "（无叶子）"
        print(f"[!] 计划 {plan_id} 无环节 {sid}；可用环节：{hint}")
        return 2
    member = str(args.member or "").strip()
    brief = str(args.brief or "").strip()
    if not member or not brief:
        print("[!] split 须给 --member 与 --brief（新叶的成员与描述）")
        return 2
    side = "before" if getattr(args, "before", False) else "after"
    leaf_name = sid.rsplit(".", 1)[-1]
    new_id = str(
        getattr(args, "new_id", "")
        or f"{leaf_name}_{'pre' if side == 'before' else 'post'}"
    )
    if not new_id.strip() or "." in new_id:
        print(f"[!] 非法新叶 id：{new_id!r}（不能为空、不能含「.」）")
        return 2
    parent = sid.rsplit(".", 1)[0] if "." in sid else ""
    items = _children_items(plan, parent)
    if items is None:
        print(f"[!] 未找到环节 {sid} 的父列表（计划结构异常）")
        return 2
    idx = _item_index(items, sid.rsplit(".", 1)[-1])
    if idx < 0:
        print(f"[!] 未找到环节 {sid}（计划结构异常）")
        return 2
    if _item_index(items, new_id) >= 0:
        print(f"[!] 同组已存在环节 {new_id}（用 --new-id 换名）")
        return 2
    old = _load_state(plan_id)
    if _refuse_if_started(old, sid, "split"):
        return 2
    items.insert(
        idx if side == "before" else idx + 1,
        {
            "id": new_id,
            "member": member,
            "brief": brief,
        },
    )
    rc = _commit_plan_edit(plan_id, plan, old, "split")
    if rc:
        return rc
    pos = "之前" if side == "before" else "之后"
    print(f"[√] 环节 {sid} 已拆叶：{new_id}（{member}）插在其{pos}")
    print(f"    新叶描述：{brief[:80]}")
    print(f"[NEXT] 核对进度：pysci-orch plan advance {plan_id}")
    return 0


def cmd_plan_amend(args) -> int:
    """amend = 直接编辑计划 YAML（组长文件工具可写 plans/*.yaml）；本命令做校验与状态对账。

    护栏（PSEP_3 §2.4）：**已 started 的 superstep 结构冻结**——删/挪已派发（或已 started
    子树内）节点即拒绝（rc=2，不改状态）；未派发部分放行。template=rework 计划另加**必留节点**
    lint（模板节点删除/乱序即拒，§4.3 #1）。附带打印结构 lint（软提示，不阻断）。
    """
    plan_id = args.id
    if not _plan_path(plan_id).exists():
        print(f"[!] 计划不存在：{plan_id}")
        return 2
    plan = _load_yaml(plan_id)
    old = _load_state(plan_id)
    tpl = template_violations(plan)
    if tpl:
        print("[!] amend 被拒：rework 模板必留节点不完整（PSEP_3 §4.3 #1）——")
        for v in tpl:
            print(f"  - {v}")
        print("  （返工模板节点不可删/不可乱序）")
        return 2
    violations = freeze_violations(
        {str(s["id"]): str(s.get("status") or "pending") for s in old.get("steps", [])},
        _tree(plan).leaf_paths(),
    )
    if violations:
        print("[!] amend 被拒：已 started 的 superstep 结构冻结（PSEP_3 §2.4）——")
        for v in violations:
            print(f"  - {v}")
        print("  （amend 只能改未派发部分；已派发/已 started 子树内节点不得删/挪）")
        return 2
    state = _reconcile(plan, old)
    _save_state(plan_id, state)
    rel: Any = _plan_path(plan_id)
    try:
        rel = rel.relative_to(ORCH_STATE_ROOT.parent)
    except ValueError:  # PLANS_DIR 被隔离到 tmp（测试面）：原样显示
        pass
    print(f"[√] 计划 {plan_id} 已对账（编辑 {rel} 后运行本命令校验）：")
    st = {str(s["id"]): s for s in state["steps"]}
    for leaf in _tree(plan).leaves:
        ss = st.get(leaf.path, {})
        print(
            f"  - {leaf.path}: {leaf.yaml.get('member', '?')} "
            f"[{ss.get('status', '?')}] {str(leaf.yaml.get('brief', ''))[:50]}"
        )
    _print_plan_lints(plan)
    return 0


def cmd_plan_drop(args) -> int:
    """中止计划（标记 abandoned，保留状态供审计）。"""
    plan_id = args.id
    state = _load_state(plan_id)
    if not state:
        print(f"[!] 计划不存在：{plan_id}")
        return 2
    state["abandoned_at"] = _now()
    _save_state(plan_id, state)
    print(f"[√] 计划 {plan_id} 已标记中止（状态文件保留供审计）。")
    return 0


def cmd_plan_list(args) -> int:
    """列出全部计划与进度。"""
    del args
    plans = list_plans()
    if not plans:
        print("（暂无计划；orch plan new <file> 登记）")
        return 0
    for pid, goal, done, total in plans:
        state = _load_state(pid)
        mark = (
            "✔ 完成"
            if state.get("completed_at")
            else ("⊘ 中止" if state.get("abandoned_at") else "▶ 进行中")
        )
        print(f"  {pid:<28} {mark}  {done}/{total}  {goal}")
    return 0


def cmd_plan_adhoc(args) -> int:
    """单步快捷计划：登记并立即执行（计划外事项的正确姿势，README §4.1）。"""
    plan_id = f"adhoc-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    plan = {
        "id": plan_id,
        "goal": f"adhoc：{str(args.text or args.task)[:60]}",
        "steps": [
            {
                "id": "s1",
                "member": args.member,
                "brief": str(args.text or "见任务书")[:120],
                "dirs": _split_dirs(args.dirs),
                "review": False,
            }
        ],
    }
    PLANS_DIR.mkdir(parents=True, exist_ok=True)
    _plan_path(plan_id).write_text(_dump_plan(plan), encoding="utf-8")
    _save_state(plan_id, _reconcile(plan, {"plan_id": plan_id, "created": _now()}))
    print(f"[√] adhoc 计划已登记：{plan_id}")

    class _Args:  # 复用 cmd_plan_run 的参数面
        pass

    a = _Args()
    a.id, a.text, a.task, a.name, a.session = (
        plan_id,
        args.text,
        args.task,
        None,
        "latest",
    )
    a.model_tier, a.max_turns, a.dirs = args.model_tier, args.max_turns, args.dirs
    a.effort = args.effort
    return cmd_plan_run(a)
