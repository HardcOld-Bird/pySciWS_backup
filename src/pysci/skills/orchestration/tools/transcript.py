"""会话记录审计工具链（inspector v2，用户设计 2026-10-13／backlog 20261013-130000-leader）。

**纯 agent 导向**：CLI 返回值即阅读面，落盘文件只是缓存——本模块不产出面向人类的
阅读形态（用户查聊天用 Qoder CLI 本身）。

三件工具（`pysci-inspect transcript` 子命令的命令面）：

- :func:`list_sessions` —— 扫 ``~/.qoder-cn/projects/*/``，按**优先级**列出可检会话
  （leader 会话 > 组员；新 > 旧），每轮巡检取头部 ≤:data:`ROUND_BUDGET` 个；
- :func:`scan_session` —— 简版：滤注入噪声（``<synthetic>`` 合成回合、attachment 等
  簿记行），思考与工具调用折叠为单行（``[thinking]``/``[tool: 名称, 尺寸]``），只留
  user 消息与 assistant 正式回复；**每回合一行索引** + 行锚；
- :func:`fetch_anchor` / :func:`fetch_grep` —— 从原始 jsonl **精确取段**渲染，有界
  返回（工具结果截断 + 大小标注）——零盲读、字节级定位，findings 的 ``--repro``
  可直接引行锚。

为什么要有它（旧形态的缺陷）：原 `transcript-audit` 菜单项是「随机挑一个会话 grep
pod-guard」——随机盲读，命中率与可复现性都靠运气。工具链把动线改成**摘要先行、锚点
深挖**：先 scan 拿全貌定位疑点，再 fetch 按锚取原文，findings 引锚可复现。

会话档案的原生形状（2026-10-13 实测，devops pod 23663 条 assistant 记录）：jsonl 逐行
JSON，行序即时间序；``type`` 有 user/assistant/active-leaf/attachment/runtime-config/
workspace-directories/worktree-state/file-history-snapshot/last-prompt/ai-title/system。
**只有 user 与 assistant 承载内容**，其余全是簿记——简版把它们整体丢掉。
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from pysci.paths import ORCH_STATE_ROOT

#: 原生会话档案根（Qoder CLI 按 cwd 键分储；键规则见
#: :func:`registry.pod_session_key`——非字母数字逐字符换 ``-``）。
PROJECTS_ROOT: Path = Path.home() / ".qoder-cn" / "projects"

#: 三态游标（gitignored 运行态；每会话一行）。
CURSOR_PATH: Path = ORCH_STATE_ROOT / "inspector" / "transcript-cursor.json"

#: digest/扫描缓存目录（gitignored 运行态，保留最近 :data:`CACHE_KEEP` 期）。
CACHE_DIR: Path = ORCH_STATE_ROOT / "inspector" / "transcript"

#: 首达该回合数即「可检查」（用户确认：5）。
MIN_ROUNDS: int = 5

#: 会话静止时长：有新回合就刷新计时器，到点且期间新增 ≥1 回合 → 无条件重可检。
STATIC_S: int = 2 * 3600

#: 每轮巡检最多检查几个会话（用户确认：3）——按优先级取头部，不贪多。
ROUND_BUDGET: int = 3

#: 缓存保留期数（自动清理旧的）。
CACHE_KEEP: int = 20

#: 单轮索引行里 user/assistant 首段的截断宽度（超长正文截断并标注实际大小）。
SLICE_CHARS: int = 80

#: ``scan --verbose`` 正文里 user/assistant 的截断宽度。比索引宽得多——索引给的是
#: 「一眼扫到哪一轮」，正文才是真读的地方；同宽会让 verbose 变成索引的复读机。
VERBOSE_SLICE_CHARS: int = 400

#: ``fetch`` 单次返回的字符上限（有界返回：读完一段就知道够不够，不够再加锚）。
FETCH_MAX_CHARS: int = 20000

#: 工具结果/正式回复在 fetch 里的截断阈值与保留头。
FETCH_SLICE_CHARS: int = 2000

#: 合成回合的模型名（渠道故障时 CLI 合成的那条 assistant 记录，见
#: :data:`sessions.SYNTHETIC_MODEL`）——简版里属注入噪声，整条滤掉。
SYNTHETIC_MODEL: str = "<synthetic>"

#: 只承载内容的记录类型；其余一律是簿记（简版丢弃）。
CONTENT_TYPES: frozenset[str] = frozenset({"user", "assistant"})


@dataclass(frozen=True)
class Round:
    """一个**回合**：一位 user 记录 + 其后至下一 user 前的 assistant/工具段。"""

    index: int  # 1 起
    start: int  # 起始行号（1 起，含）
    end: int  # 结束行号（1 起，含）
    ts: str  # user 记录时间戳（缺则空串）
    user: str  # user 消息正文（首个真实 user 记录）
    assistant: str  # 该回合 assistant 的正式回复首段
    nbytes: int  # 该回合占用的字节量（行长之和）

    @property
    def size(self) -> int:
        """行锚跨度的行数（``L<a>-L<b>`` 的 b-a+1）。"""
        return self.end - self.start + 1


@dataclass(frozen=True)
class SessionRef:
    """一个候选会话的读数（``list`` 的每一行）。"""

    sid: str
    project: str  # 项目键（``~/.qoder-cn/projects/<键>``）
    kind: str  # leader | member | other
    rounds: int
    last_active: float  # epoch 秒（0=读不出）
    state: str  # checkable | checked | short

    @property
    def rank(self) -> int:
        """优先级序（小者先）：leader > 组员 > 其它。"""
        return {"leader": 0, "member": 1}.get(self.kind, 2)


# ---------------------------------------------------------------------------
# 读盘：会话档案与项目键
# ---------------------------------------------------------------------------
def project_dirs(root: Path | None = None) -> list[Path]:
    """``~/.qoder-cn/projects/`` 下的项目目录（不可读 → 空表，fail-open）。"""
    base = PROJECTS_ROOT if root is None else root
    try:
        return sorted(p for p in base.iterdir() if p.is_dir())
    except OSError:
        return []


def session_files(project: Path) -> list[Path]:
    """某项目目录下的会话档（``<sid>.jsonl``；同名目录是伴生元数据，不算）。"""
    try:
        return sorted(p for p in project.glob("*.jsonl") if p.is_file())
    except OSError:
        return []


def read_records(path: Path) -> list[tuple[int, dict[str, Any]]]:
    """逐行读会话档，返回 ``(行号, 记录)`` 序列（1 起）。

    **fail-open**：读不动 → 空表；坏行（半写/非 JSON）跳过——会话档可达上百 MB，
    为一行坏数据放弃整个读取不值。
    """
    out: list[tuple[int, dict[str, Any]]] = []
    try:
        with path.open(encoding="utf-8", errors="replace") as fh:
            for no, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue
                if isinstance(rec, dict):
                    out.append((no, rec))
    except OSError:
        return []
    return out


def _block_text(content: Any) -> str:
    """从 message.content 提取纯文本（str 直接用；list 只取 text 块）。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(c.get("text", ""))
            for c in content
            if isinstance(c, dict) and c.get("type") == "text"
        )
    return ""


def is_real_user(rec: dict[str, Any]) -> bool:
    """**真实 user 消息**？——tool_result 回灌也走 ``type=user``，但它不是新回合。

    判据：``message.content`` 是含非空 text 的串或列表。工具结果（list 里全是
    ``tool_result`` 块）不算——否则每个工具调用都会被当成新回合，回合数虚高一倍。
    """
    if rec.get("type") != "user":
        return False
    content = (rec.get("message") or {}).get("content")
    if isinstance(content, str):
        return bool(content.strip())
    if isinstance(content, list):
        return any(
            isinstance(c, dict)
            and c.get("type") == "text"
            and str(c.get("text", "")).strip()
            for c in content
        )
    return False


def _is_synthetic(rec: dict[str, Any]) -> bool:
    """合成故障回合（渠道故障时 CLI 写的那条 assistant 记录）——注入噪声。"""
    if rec.get("type") != "assistant":
        return False
    return str((rec.get("message") or {}).get("model", "")) == SYNTHETIC_MODEL


def _assistant_text(rec: dict[str, Any]) -> str:
    """assistant 记录的**正式回复**（text 块；思考与工具调用不计入）。"""
    return _block_text((rec.get("message") or {}).get("content"))


def build_rounds(path: Path) -> list[Round]:
    """把会话档切成回合表（回合定义见 :class:`Round`）。

    回合从**第一个真实 user 记录**起算；其前的簿记行（workspace-directories 等）不
    构成回合。档案末尾若有未闭合的尾巴（user 之后还没有 assistant 回复），也算一回合
    ——inspector 要看的恰恰可能是「这一跳还没收尾」。
    """
    records = read_records(path)
    starts: list[int] = [i for i, (_, rec) in enumerate(records) if is_real_user(rec)]
    rounds: list[Round] = []
    for n, begin in enumerate(starts):
        end = starts[n + 1] - 1 if n + 1 < len(starts) else len(records) - 1
        chunk = records[begin : end + 1]
        user_rec = chunk[0][1]
        asst = next(
            (
                _assistant_text(rec)
                for _, rec in chunk[1:]
                if rec.get("type") == "assistant" and not _is_synthetic(rec)
            ),
            "",
        )
        nbytes = sum(len(json.dumps(rec, ensure_ascii=False)) for _, rec in chunk)
        rounds.append(
            Round(
                index=n + 1,
                start=chunk[0][0],
                end=chunk[-1][0],
                ts=str(user_rec.get("timestamp", "")),
                user=_block_text((user_rec.get("message") or {}).get("content")),
                assistant=asst,
                nbytes=nbytes,
            )
        )
    return rounds


#: 「这行可能是真实 user 记录」的廉价标记（**先子串筛、再 JSON 解析**）。
#: 原生档案是紧凑 JSON（实测 ``{"type":"user",...}``，冒号后无空格），两种分隔都收，
#: 免得平台哪天改成带空格的写法就静默退化成「0 回合」。
_USER_MARKERS: tuple[str, ...] = ('"type":"user"', '"type": "user"')
#: 同上，bytes 形态（二进制读路径用；见 :func:`session_stats`）。
_USER_MARKERS_B: tuple[bytes, ...] = (b'"type":"user"', b'"type": "user"')

#: 取「最后活跃」用的尾部窗口（同 :data:`sessions.TAIL_BYTES` 的量级）：档尾才是最新
#: 时间戳，读全档只为拿一个时间戳不值。
STATS_TAIL_BYTES: int = 1 << 18


def _looks_like_user(line: str) -> bool:
    """行级廉价预筛：不含 user 类型标记的行**不必** JSON 解析。

    为什么值得加（backlog 20261013-182427-devops 实测）：真实档案里簿记行占近半——
    devops 单档 105.9 MB / 23663 assistant + 23610 ``active-leaf`` + 1605 ``attachment``，
    而后两类与回合数**无关**。原实现逐行 ``json.loads`` 全量解析，1.0s/档，而 inspector
    每轮都要跑一次。预筛是**保守**的（宁可多解析几行，绝不漏掉真回合），只在筛出候选后
    才付解析代价。
    """
    return any(marker in line for marker in _USER_MARKERS)


def session_stats(path: Path) -> tuple[int, float]:
    """某会话档的 ``(回合数, 最后活跃 epoch 秒)``（读不出 → ``(0, 0.0)``）。

    单次顺序读 + 行级预筛（:func:`_looks_like_user`）：只对候选行做 JSON 解析；
    「最后活跃」改读**尾部窗口**（最新时间戳必在档尾），不再为它全量解析一遍。
    回合数与全量实现逐值一致（预筛只跳行、不改判定：命中子串的行仍走
    :func:`is_real_user` 的完整判据）。
    """
    rounds = 0
    try:
        # **二进制读 + 迟到解码**：105 MB 档全量 decode 成 str 本身就要 ~0.3s，而我们
        # 只对候选行付解码与解析——先在 bytes 上做子串筛，命中的行才 decode。
        with path.open("rb") as fh:
            for raw in fh:
                if not any(marker in raw for marker in _USER_MARKERS_B):
                    continue
                # 工具结果回灌也走 type=user，但**不含 text 块** ⇒ 必非真实 user 回合
                # （:func:`is_real_user` 只认含 text 块的 user 记录）。带 text 块的记录
                # 照常解析，故这条跳过是**保守**的：含 text 就绝不跳。
                if b'"type":"tool_result"' in raw and b'"type":"text"' not in raw:
                    continue
                try:
                    rec = json.loads(raw.decode("utf-8", errors="replace"))
                except (json.JSONDecodeError, ValueError):
                    continue
                if isinstance(rec, dict) and is_real_user(rec):
                    rounds += 1
    except OSError:
        return 0, 0.0
    return rounds, _last_timestamp(path)


def _last_timestamp(path: Path) -> float:
    """档内**最后一个**可解析时间戳（先读尾部窗口；窗口内没有才退回全量）。

    尾部窗口起点可能切断一行（残片不是合法 JSON）——按 :func:`read_records` 的老规矩
    丢掉首行即可。回退分支只在「尾部窗口里一个时间戳都没有」这种异常档上走，属罕见路径，
    故保留全量兜底以**不改变**旧语义。
    """
    try:
        size = path.stat().st_size
        with path.open("rb") as fh:
            fh.seek(max(0, size - STATS_TAIL_BYTES))
            chunk = fh.read().decode("utf-8", errors="replace")
    except OSError:
        return 0.0
    lines = chunk.splitlines()
    if size > STATS_TAIL_BYTES and lines:
        lines = lines[1:]
    for line in reversed(lines):
        if '"timestamp"' not in line:
            continue
        try:
            rec = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(rec, dict):
            stamp = _epoch(rec.get("timestamp"))
            if stamp:
                return stamp
    if size > STATS_TAIL_BYTES:
        for _, rec in reversed(read_records(path)):
            stamp = _epoch(rec.get("timestamp"))
            if stamp:
                return stamp
    return 0.0


def _epoch(raw: Any) -> float:
    """时间戳归一为 epoch 秒：毫秒整数（runtime-config）与 ISO 串（user/assistant）都吃。"""
    if isinstance(raw, (int, float)) and raw > 0:
        return float(raw) / 1000.0 if raw > 1e11 else float(raw)
    if isinstance(raw, str) and raw:
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return 0.0
    return 0.0


# ---------------------------------------------------------------------------
# 三态游标：可检查 / 已检查 / 太短
# ---------------------------------------------------------------------------
CHECKABLE = "checkable"
CHECKED = "checked"
SHORT = "short"


def load_cursor(path: Path | None = None) -> dict[str, dict[str, Any]]:
    """读游标（``{sid: {...}}``；缺失/损坏 → 空表，fail-open）。"""
    p = CURSOR_PATH if path is None else path
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(raw, dict):
        return {}
    sessions = raw.get("sessions")
    return sessions if isinstance(sessions, dict) else {}


def save_cursor(data: dict[str, dict[str, Any]], path: Path | None = None) -> None:
    """写游标（原子替换：半写的游标会让下轮把已检会话当没检过）。"""
    p = CURSOR_PATH if path is None else path
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(
        json.dumps({"version": 1, "sessions": data}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    tmp.replace(p)


def classify(
    prev: dict[str, Any] | None,
    *,
    rounds: int,
    last_active: float,
    now: float,
) -> str:
    """三态判定（纯函数，全矩阵可单测）。

    规则（用户设计 2026-10-13）：

    - **首达 5 回合且未检过** → :data:`CHECKABLE`；
    - 检查后 → :data:`CHECKED`；
    - **重可检**（已检会话回到 CHECKABLE）两条通路：新增回合 ≥5；**或**会话静止
      :data:`STATIC_S`（每有新回合刷新计时器）且期间新增 ≥1 回合——静止的会话即使
      只多了一轮也值得复看（它大概率已经收尾，是干净的审计样本）；
    - 短于 5 回合且未静止 → :data:`SHORT`（不检查）。

    Args:
        prev: 该会话此前的游标行（``None``=从未见过）。
        rounds: 当前回合数。
        last_active: 最后活跃 epoch 秒。
        now: 当前 epoch 秒。

    Returns:
        ``checkable`` / ``checked`` / ``short`` 之一。
    """
    if prev is None or not str(prev.get("checked_at") or "").strip():
        return CHECKABLE if rounds >= MIN_ROUNDS else SHORT
    new_rounds = rounds - int(prev.get("rounds_seen", 0) or 0)
    if new_rounds >= MIN_ROUNDS:
        return CHECKABLE
    if new_rounds >= 1 and last_active and (now - last_active) >= STATIC_S:
        return CHECKABLE
    return CHECKED


def session_kind(project_key: str, *, leader_key: str) -> str:
    """项目键 → 会话归属（``leader``/``member``/``other``）。

    判据是**路径形态**而非成员名单：leader 的 TUI 跑在仓库根（键 = 根目录的
    :func:`registry.pod_session_key`），组员的 cwd 是 ``orchestration/pods/<成员>``
    （含 ``-orchestration-pods-``），worktree 里的组员会话同样含该段故仍归 member。
    """
    if project_key == leader_key:
        return "leader"
    if "-orchestration-pods-" in project_key:
        return "member"
    return "other"


def list_sessions(
    *,
    root: Path | None = None,
    cursor_path: Path | None = None,
    leader_key: str = "",
    now: float | None = None,
    budget: int = ROUND_BUDGET,
    only_checkable: bool = True,
) -> list[SessionRef]:
    """按优先级列出会话（默认只列可检查者，且只取头部 :data:`ROUND_BUDGET` 个）。

    优先级：**leader 会话 > 组员**，同组内**新 > 旧**（最后活跃倒序）。预算在
    排序**之后**截断——先排序再取头部，才能保证每轮巡检拿到的是最该看的那几个。

    Args:
        root: 项目根覆盖（测试注入）。
        cursor_path: 游标路径覆盖（测试注入）。
        leader_key: leader 项目键（空则不判 leader，全部按 member/other 分）。
        now: 当前 epoch 秒（注入以便单测；缺省取系统时间）。
        budget: 头部保留数（0 或负 = 不限）。
        only_checkable: 只列 :data:`CHECKABLE`（缺省；``False`` 时列全部并存 state）。
    """
    import time as _time

    stamp = _time.time() if now is None else now
    cursor = load_cursor(cursor_path)
    rows: list[SessionRef] = []
    for project in project_dirs(root):
        key = project.name
        kind = session_kind(key, leader_key=leader_key)
        for path in session_files(project):
            sid = path.stem
            rounds, last = session_stats(path)
            prev = cursor.get(sid)
            state = classify(prev, rounds=rounds, last_active=last, now=stamp)
            if only_checkable and state != CHECKABLE:
                continue
            rows.append(
                SessionRef(
                    sid=sid,
                    project=key,
                    kind=kind,
                    rounds=rounds,
                    last_active=last,
                    state=state,
                )
            )
    rows.sort(key=lambda r: (r.rank, -r.last_active, r.sid))
    return rows[:budget] if budget and budget > 0 else rows


def mark_checked(
    cursor: dict[str, dict[str, Any]],
    ref: SessionRef,
    *,
    now_iso: str,
) -> None:
    """把某会话记成「已检查」（就地更新传入的游标 dict，落盘由调用方负责）。"""
    cursor[ref.sid] = {
        "project": ref.project,
        "sid": ref.sid,
        "rounds_seen": ref.rounds,
        "last_round_ts": _iso_from_epoch(ref.last_active) or now_iso,
        "checked_at": now_iso,
    }


def _iso_from_epoch(value: float) -> str:
    if not value:
        return ""
    return datetime.fromtimestamp(value).astimezone().isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# scan：简版（噪声过滤 + 折叠 + 行锚）
# ---------------------------------------------------------------------------
def _one_line(text: str, width: int = SLICE_CHARS) -> str:
    """压成单行并截断（超长标注实际长度——「截断了」这件事必须可见）。"""
    flat = " ".join(str(text).split())
    if len(flat) <= width:
        return flat
    return f"{flat[:width]}…（共 {len(flat)} 字）"


def _tool_line(block: dict[str, Any]) -> str:
    """工具调用折叠为单行：``[tool: 名称, 尺寸]``。"""
    name = str(block.get("name", "?"))
    size = len(json.dumps(block.get("input"), ensure_ascii=False))
    return f"[tool: {name}, {size}B]"


def simple_lines(
    records: Iterable[tuple[int, dict[str, Any]]], *, width: int = SLICE_CHARS
) -> list[str]:
    """把一段记录渲染成**简版**（滤噪声、折叠思考与工具调用）。

    - 簿记行（attachment/active-leaf/runtime-config/…）直接丢；
    - 合成回合（``<synthetic>``）整条丢——那是渠道故障注入的形状，不是组员说的话；
    - assistant 的 ``thinking`` → ``[thinking]`` 单行、``tool_use`` → 单行；
    - 只留 user 与 assistant 的**正式回复**（text 块）。

    ``width``：正文截断宽度（索引行用 :data:`SLICE_CHARS`，``--verbose`` 正文用
    :data:`VERBOSE_SLICE_CHARS`）。
    """
    out: list[str] = []
    for _, rec in records:
        rtype = rec.get("type")
        if rtype not in CONTENT_TYPES or _is_synthetic(rec):
            continue
        if rtype == "user":
            text = _block_text((rec.get("message") or {}).get("content"))
            if text.strip():
                out.append(f"  user: {_one_line(text, width)}")
            continue
        msg = rec.get("message") or {}
        for block in msg.get("content") or []:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text":
                text = str(block.get("text", ""))
                if text.strip():
                    out.append(f"  asst: {_one_line(text, width)}")
            elif btype == "thinking":
                out.append("  [thinking]")
            elif btype == "tool_use":
                out.append(f"  {_tool_line(block)}")
    return out


def anchor_of(sid: str, rnd: Round) -> str:
    """行锚：``<sid>.jsonl:L<a>-L<b>``（findings 的 ``--repro`` 直接引它）。"""
    return f"{sid}.jsonl:L{rnd.start}-L{rnd.end}"


def scan_session(
    path: Path,
    *,
    turns: tuple[int, int] | None = None,
    verbose: bool = False,
) -> str:
    """渲染会话简版：每回合一行**索引** +（``verbose``）简版正文。

    索引行 = id / 时间戳 / user 首段 / assistant 首段 / 字节量 + 行锚——先看索引定位
    疑点，再按锚 fetch 取原文（摘要先行、锚点深挖）。

    Args:
        path: 会话档。
        turns: ``(起, 止)`` 回合区间（1 起，含；``None``=全部）。
        verbose: 附简版正文（user/assistant 正式回复 + 折叠的思考/工具行）。
    """
    sid = path.stem
    rounds = build_rounds(path)
    if turns is not None:
        lo, hi = turns
        rounds = [r for r in rounds if lo <= r.index <= hi]
    if not rounds:
        return f"（{sid}.jsonl 无回合——空会话或全是簿记行）"
    head = [
        f"# {sid}.jsonl  回合 {len(rounds)} 个"
        + (f"（{turns[0]}-{turns[1]}）" if turns else "")
    ]
    records = read_records(path) if verbose else []
    for rnd in rounds:
        head.append(
            f"[{rnd.index}] {rnd.ts or '-'}  user: {_one_line(rnd.user)}  |  "
            f"asst: {_one_line(rnd.assistant)}  |  {rnd.nbytes}B  "
            f"anchor={anchor_of(sid, rnd)}"
        )
        if verbose:
            chunk = [(no, rec) for no, rec in records if rnd.start <= no <= rnd.end]
            head.extend(simple_lines(chunk, width=VERBOSE_SLICE_CHARS))
    return "\n".join(head)


# ---------------------------------------------------------------------------
# fetch：按锚或按模式精确取段
# ---------------------------------------------------------------------------
#: 行锚形状：``<sid>.jsonl:L<a>-L<b>``。sid 段**不写死 UUID 形态**——真 sid 是 UUID，
#: 但锚也可能来自自造样本/前缀，写死只会在边界上误拒（不写死也不会误收：``.jsonl``
#: 后缀与行号形状已足够特异）。
ANCHOR_RE = re.compile(r"^(?P<sid>.+?)\.jsonl:L(?P<a>\d+)-L(?P<b>\d+)$")


def parse_anchor(text: str) -> tuple[str, int, int]:
    """解析 ``<sid>.jsonl:L<a>-L<b>``；形状不对抛 :class:`ValueError`。"""
    m = ANCHOR_RE.match(text.strip())
    if not m:
        raise ValueError(
            f"锚形状非法：{text!r}（应为 <sid>.jsonl:L<a>-L<b>，如 "
            "6f0cc8e5-9aa6-41f0-86ff-58a49b002704.jsonl:L120-L160）"
        )
    a, b = int(m.group("a")), int(m.group("b"))
    if a < 1 or b < a:
        raise ValueError(f"锚行号非法：L{a}-L{b}（须 1 ≤ a ≤ b）")
    return m.group("sid"), a, b


def find_session(sid: str, *, root: Path | None = None) -> Path | None:
    """按 sid（或唯一前缀）在全部项目目录里定位会话档；找不到/歧义 → ``None``。"""
    hits: list[Path] = []
    for project in project_dirs(root):
        for path in session_files(project):
            if path.stem == sid or path.stem.startswith(sid):
                hits.append(path)
    return hits[0] if len(hits) == 1 else None


def _render_record(no: int, rec: dict[str, Any]) -> str:
    """把一条记录渲染成有界文本（工具结果/长正文截断并标注实际大小）。"""
    body = json.dumps(rec, ensure_ascii=False)
    label = str(rec.get("type", "?"))
    if rec.get("type") == "assistant":
        blocks = (rec.get("message") or {}).get("content") or []
        parts = []
        for block in blocks:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use":
                parts.append(_tool_line(block))
        if parts:
            label += " " + " ".join(parts)
    if len(body) > FETCH_SLICE_CHARS:
        return (
            f"L{no:<6} [{label}] {body[:FETCH_SLICE_CHARS]}…（截断，原 {len(body)} 字）"
        )
    return f"L{no:<6} [{label}] {body}"


def fetch_anchor(anchor: str, *, root: Path | None = None) -> str:
    """按行锚取原始段并渲染（**有界返回**：单次上限 :data:`FETCH_MAX_CHARS`）。"""
    sid, a, b = parse_anchor(anchor)
    path = find_session(sid, root=root)
    if path is None:
        return f"[✗] 找不到会话 {sid}.jsonl（或前缀有歧义）"
    records = read_records(path)
    if not records:
        return f"[✗] {sid}.jsonl 读不出记录"
    lines: list[str] = [f"# {anchor}（共 {len(records)} 行，取 L{a}-L{b}）"]
    total = 0
    for no, rec in records:
        if not (a <= no <= b):
            continue
        text = _render_record(no, rec)
        total += len(text)
        if total > FETCH_MAX_CHARS:
            lines.append(f"…（已达单次上限 {FETCH_MAX_CHARS} 字，缩小锚区间再取）")
            break
        lines.append(text)
    return "\n".join(lines)


def fetch_grep(
    sid: str, pattern: str, *, root: Path | None = None, limit: int = 40
) -> str:
    """在会话档里按正则找记录，返回命中行的行号与**所在回合**的锚（便于接着 fetch）。"""
    path = find_session(sid, root=root)
    if path is None:
        return f"[✗] 找不到会话 {sid}.jsonl（或前缀有歧义）"
    try:
        cre = re.compile(pattern)
    except re.error as exc:
        return f"[✗] 正则非法：{exc}"
    records = read_records(path)
    rounds = build_rounds(path)
    spans = [(r.start, r.end, anchor_of(sid, r)) for r in rounds]
    hits: list[str] = []
    for no, rec in records:
        body = json.dumps(rec, ensure_ascii=False)
        if not cre.search(body):
            continue
        owner = next((a for s, e, a in spans if s <= no <= e), "-")
        hits.append(
            f"L{no:<6} [{rec.get('type', '?')}] {owner}  {_one_line(body, 120)}"
        )
        if len(hits) >= limit:
            hits.append(f"…（已到 {limit} 条上限，收紧模式再查）")
            break
    if not hits:
        return f"（{sid}.jsonl 无匹配 {pattern!r} 的记录）"
    return "\n".join([f"# {sid}.jsonl grep {pattern!r}"] + hits)


# ---------------------------------------------------------------------------
# 缓存（落盘只是缓存，保留最近 N 期）
# ---------------------------------------------------------------------------
def save_cache(name: str, text: str, *, keep: int = CACHE_KEEP) -> Path:
    """把一份扫描结果落进缓存目录并清理旧期（缓存不是资产，丢得起）。"""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S")
    path = CACHE_DIR / f"{name}-{stamp}.txt"
    path.write_text(text, encoding="utf-8")
    prune_cache(keep=keep)
    return path


def prune_cache(*, keep: int = CACHE_KEEP) -> int:
    """只保留最近 ``keep`` 期缓存，返回删除数（读不到目录 → 0）。"""
    try:
        files = sorted(
            CACHE_DIR.glob("*.txt"), key=lambda p: p.stat().st_mtime, reverse=True
        )
    except OSError:
        return 0
    dropped = 0
    for old in files[max(0, keep) :]:
        try:
            old.unlink()
            dropped += 1
        except OSError:
            continue
    return dropped
