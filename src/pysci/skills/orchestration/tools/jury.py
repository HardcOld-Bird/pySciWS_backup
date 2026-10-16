"""Tier C 仲裁团（README §5.3 J 通道）：家族外、直连、阻塞、双序、多数、材料冻结。

**共享库，不建独立 CLI**（§0.9）：``orch jury <case-file>`` 子命令封装本模块
（:func:`cmd_jury`），E6 的 pysci-inspect 复用同一条通路。渠道 = **逐席 provider 直连**
（E3-live，§4 用户裁决 2026-10-12 修订）：deepseek / bigmodel / moonshot 三家各一席，
均为 OpenAI 兼容接口，Key 只进请求头。

判决语义（每条对应 ``tests/skills/orchestration/test_jury.py`` 的一例）：

- **法院模式**（§4 用户裁决 v0.4：jury 不自行读代码）：裁判无工具、不看仓库，只读
  案卷。案卷由本模块**自动附加确定性材料**（``git diff`` / ``git log --oneline -5`` /
  涉事文件字节读数 / 相关台账行，按案卷头部 ``targets:`` 声明抓取），全文哈希入档
  ＝材料冻结——事后证据推翻裁决时能核对「裁判当时看到的是什么」。
- **面板构成随预设换血**（用户裁决 2026-10-12 22:4x）：本庭 = 当前预设点名的三家族
  （qwen→glm+kimi+mimo、mimo→glm+kimi+qwen、deepseek→glm+kimi+mimo），配置里其余席
  **不入面板**但留痕报数，面板缺席的家族在读数行点名让用户去补 ``registry.jury.seats``。
- **谷时隐性态换面板**（路由 v3 规格 3，backlog 20261013-093000-leader）：谷时主席是
  DeepSeek，面板换成固定三席 :data:`~.routing.DEFAULT_JURY_PANEL`（GLM+KIMI+Mimo，用户
  终裁 v4）——被评模型换了同族就换（:func:`.routing.producer_family`），否则庭里坐进的
  就是被评方自己。
- **仲裁团一律按量付费**（用户终裁 2026-10-13）：TokenPlan 套餐端点**整个不做**——两家
  个人版条款都禁止 API 自动化调用（违规封 Key），而仲裁团正是自动化调用。故一席只有
  「一条端点」一种形状：代码里不存端点链，也不写下探回退。
- **预设同族硬规则**：面板型号命中**被评模型同族**即拒开庭（排除同族自恋偏差）——
  绝对化的「不得含 Qwen」已被 mimo 预设的合法 Qwen 席推翻，规则改为相对预设。选型权
  归用户，代码不内置候选清单、也不静默剔除——违约要被看见。
- **逐席可用性**（§4 三席直调的减平票前提）：一席 Key 缺失只作废该席（标 unavailable +
  **双面告警**：调用者面 :func:`_court_alert`、读数面 :func:`readiness_summary`），
  **不拖垮整案**——剩下两席照常开庭计票，平票由 ``escalate="twin"`` 终裁兜底。全部主席
  不可达且回退席也未就位时才降级入队。
- **回退席**（``registry.jury.fallback``，独立开关）：仅当 ≥1 主席不可达时上场，
  上场即 :data:`FALLBACK_NOTICE` 告警，三处可见（留痕案卷 / ``orch status`` 判决包健康段 /
  mirror 概览）——回退是权宜，不是常态，用户必须知道主席掉了。**被评模型同族的回退席不得
  上场**：它一进场就是 §4 违约，会把「该降级」的案子误判成「拒开庭」。
- **双序**：每裁判跑 AB 与 BA 两局。BA 局把两个选项内容对调、仍以 A/B 标签提问，
  答案经 :func:`flip` 映射回真实选项；两局映射不一致 → 该裁判记 inconclusive
  **不计票**（被位置偏好牵着走的裁判自动出局）。
- **多数 / 模棱两可**：计票少数服从多数；**无有效票 / 平票 / inconclusive 过半**
  → ``escalate="twin"``（§5.5 终裁链；E6 twin 未上线前挂起呈组长，不静默丢弃）。
- **降级不阻塞生产**：Key 全缺、席位未选型、调用失败或超时（``deadline_s`` →
  重试一次 → 仍败）一律把案卷落 ``state/evals/jury-queue/`` 待重裁，**绝不 raise**
  ——基建决策不因仲裁渠道不可达而停摆。
- **Key 安全**：Key 永不进日志、留痕、异常文本；自由文本出模块前统一过 :func:`scrub`。
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import subprocess
import threading
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
from dotenv import dotenv_values

from pysci.paths import ORCH_STATE_ROOT, PROJECT_ROOT

from . import presets, procs, routing
from .ledger import read_all
from .registry import DEFAULT_JURY, Registry, orchestration_rel

#: 裁决留痕目录（每案一份 JSON：案卷哈希、各裁判输出、双序一致性、计票、usage、成本）。
JURY_DIR: Path = ORCH_STATE_ROOT / "evals" / "jury"
#: 降级队列（渠道不可达 / 超时 / 未选型的案卷落这里待重裁）。
JURY_QUEUE_DIR: Path = ORCH_STATE_ROOT / "evals" / "jury-queue"
#: 冻结简报（护栏级：出现 ``approved:`` 行才生效，否则运行时警告「简报未批准」）。
JURY_BRIEF_PATH: Path = ORCH_STATE_ROOT / "evals" / "jury-brief.md"

#: 回退席上场时的告警语（§4 用户裁决：触发回退必须显式告警）。三处可见＝留痕案卷、
#: ``orch status`` 判决包健康段、mirror 概览——回退是权宜不是常态，主席掉了要有人处理。
FALLBACK_NOTICE = "回退席已激活——请用户处理主席"
#: 双序两局的标签序。BA 局内容对调，答案映射回真实选项后才可比（见 :func:`flip`）。
ORDERS: tuple[str, ...] = ("AB", "BA")
#: 出厂请求 ``max_tokens``（registry.jury.max_output_tokens 可覆盖）。三席皆推理型，
#: 真案卷上 reasoning_tokens 是重尾的（实测 432～3986），额度不足会把正文截成空——
#: :func:`_ballot` 对此报 ``截断`` 错误（降级入队），不让它伪装成裁判的无效票。
DEFAULT_MAX_TOKENS: int = int(DEFAULT_JURY["max_output_tokens"])

#: 自动附着材料里 ``git diff`` 正文的字节上限——案卷要进每次调用的 prompt，一次大重构
#: 不能把材料面撑成十万字（超出截断并标注原大小，哈希仍按截断后正文计算）。
DIFF_CAP_BYTES = 4000
#: 每个涉事文件的 excerpt 不再抓取（只给字节读数），targets 条目数上限纯为止损。
MAX_TARGETS = 20

#: 面板家族词（预设构成与同族排除的匹配口径）：席的 name/model 命中其一即归该家族。
FAMILY_TOKENS: tuple[str, ...] = ("deepseek", "glm", "kimi", "mimo", "qwen")


def seat_family(*parts: str) -> str:
    """席 → 家族名（``deepseek``/``glm``/``kimi``/``mimo``/``qwen``；认不出返回 ``""``）。

    name 与 model 一起看：配置可能只在 model 里带族名（如 ``deepseek-ai/DeepSeek-V4-Pro``），
    认不出家族的席**不参与**同族排除（宁可放行也不误伤陌生配置）。
    """
    blob = " ".join(p for p in parts if p).lower()
    for token in FAMILY_TOKENS:
        if token in blob:
            return token
    return ""


_BRIEF_APPROVED_RE = re.compile(
    r"^\s*>?\s*#*\s*approved\s*:", re.IGNORECASE | re.MULTILINE
)
_HEADING_RE = re.compile(r"^##\s*(?P<title>.+?)\s*$", re.MULTILINE)
_TARGETS_RE = re.compile(r"^targets\s*:\s*(?P<val>.*)$", re.IGNORECASE | re.MULTILINE)
_JSON_BLOCK_RE = re.compile(r"\{[\s\S]*\}")

#: 疑似 API Key 形态（:func:`scrub` 用；同一套形状的仓库级扫描见 ``pysci-dev doctor``
#: 的 keyscan 哨兵，两边刻意各写各的——扫描要宽撒网，脱敏要准命中）。
_SECRET_RES: tuple[re.Pattern[str], ...] = (
    re.compile(r"sk-[A-Za-z0-9_\-]{16,}"),
    re.compile(
        r"(?i)(api[_-]?key|secret|token|password)(\s*[:=]\s*['\"]?)([A-Za-z0-9_\-]{16,})"
    ),
)


class JuryError(RuntimeError):
    """仲裁调用失败（HTTP 非 2xx、响应体不合形状）。消息已脱敏，可安全入日志。"""


# ---------------------------------------------------------------------------
# 案卷：解析 → 自动附着材料 → 哈希冻结
# ---------------------------------------------------------------------------
@dataclass
class Case:
    """一份仲裁案卷（问题 + 两选项全文 + 证据 + 涉事路径声明）。"""

    case_id: str
    question: str
    option_a: str
    option_b: str
    evidence: str = ""
    targets: list[str] = field(default_factory=list)
    path: str = ""


def _split_sections(text: str) -> dict[str, str]:
    """按 ``## `` 小节切分正文（同名小节取首次出现；返回标题→正文，标题原样保留空格）。"""
    marks = list(_HEADING_RE.finditer(text))
    out: dict[str, str] = {}
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        out.setdefault(m.group("title").strip(), text[m.end() : end])
    return out


def _lookup(sections: dict[str, str], *names: str) -> str:
    """按去空格小写名在小节表里取值（``选项 A``/``Option A``/``a`` 都能命中）。"""
    norm = {k.replace(" ", "").lower(): v for k, v in sections.items()}
    for name in names:
        if (val := norm.get(name.replace(" ", "").lower())) is not None:
            return val.strip()
    return ""


def parse_case(path: Path | str) -> Case:
    """读案卷文件 → :class:`Case`。

    约定（与组员提案文件同族：标记行 + ``## `` 小节）：``## 问题``（或 H1 标题兜底）、
    ``## 选项 A``、``## 选项 B`` 必需，``## 证据`` 可选，头部 ``targets: a, b`` 声明
    涉事路径供自动抓取材料。

    Raises:
        ValueError: 缺问题或任一选项小节（案卷不可用＝组不出案卷，不降级入队）。
    """
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    sections = _split_sections(text)
    question = _lookup(sections, "问题", "question")
    if not question:
        for line in text.splitlines():
            if line.startswith("# ") and line[2:].strip():
                question = line[2:].strip()
                break
    option_a = _lookup(sections, "选项 a", "option a", "a")
    option_b = _lookup(sections, "选项 b", "option b", "b")
    missing = [
        n
        for n, v in (("问题", question), ("选项 A", option_a), ("选项 B", option_b))
        if not v
    ]
    if missing:
        raise ValueError(
            f"案卷缺小节：{missing}（需要 ## 问题 / ## 选项 A / ## 选项 B）"
        )
    targets = (
        [t.strip() for t in re.split("[;,]", tm.group("val")) if t.strip()]
        if (tm := _TARGETS_RE.search(text))
        else []
    )
    return Case(
        case_id=p.stem,
        question=question,
        option_a=option_a,
        option_b=option_b,
        evidence=_lookup(sections, "证据", "evidence"),
        targets=targets[:MAX_TARGETS],
        path=str(p),
    )


def run_git(*args: str) -> str:
    """在项目根跑一条 git 命令取材料（法院模式：材料缺失只记录原因，不崩整案）。"""
    cmd = " ".join(["git", *args])
    try:
        proc = procs.run_hidden(
            ["git", *args],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"（{cmd} 不可用：{type(exc).__name__}: {exc}）"
    if proc.returncode != 0:
        return f"（{cmd} rc={proc.returncode}）"
    return (proc.stdout or "").strip()


def file_reading(target: str) -> str:
    """涉事文件的**字节读数**（大小 / 行数 / 内容哈希）——不给正文，避免案卷被文件吞掉。

    哈希是「裁判看到的这一版」的指纹：文件随后被改，回访时能据哈希证明当时不是这个样子。
    """
    p = Path(os.path.expanduser(target))
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    if not p.exists():
        return f"- {target}: 缺席"
    if p.is_dir():
        files = [f for f in p.rglob("*") if f.is_file()]
        return (
            f"- {target}: 目录，{len(files)} 文件 / "
            f"{sum(f.stat().st_size for f in files)} B"
        )
    data = p.read_bytes()
    text = data.decode("utf-8", errors="replace")
    return (
        f"- {target}: {len(data)} B / {len(text.splitlines())} 行 / sha256={_sha(text)}"
    )


def _ledger_needles(case: Case) -> list[str]:
    """台账相关性判据的检索串：涉事路径的文件名 + 案卷 id（命中任一即算相关）。"""
    needles = [Path(t).name for t in case.targets if Path(t).name]
    needles.append(case.case_id)
    return [n for n in dict.fromkeys(needles) if n]


def relevant_ledger_rows(case: Case, *, limit: int = 10) -> list[str]:
    """``state/ledger/deliveries.jsonl`` 里与本案涉事物相关的行（近 N 条，供裁判看既往表现）。"""
    needles = _ledger_needles(case)
    rows: list[str] = []
    for e in read_all():
        blob = json.dumps(asdict(e), ensure_ascii=False)
        if any(n in blob for n in needles):
            rows.append(
                f"- {e.ts} {e.member} kind={e.kind or '-'} plan={e.plan or '-'} "
                f"turns={e.num_turns} tags={' '.join(e.tags) or '-'}"
            )
    return rows[-limit:]


def build_material(case: Case) -> str:
    """CLI 自动附着的**确定性材料**（§4 对「jury 不自行读代码」的补偿条款）。

    抓取项全部可重放：``git diff --stat`` / ``git diff``（截断）/ ``git log
    --oneline -5`` / 案卷 ``targets:`` 涉事文件的字节读数 / 相关台账行。
    """
    diff = run_git("diff")
    if len(diff) > DIFF_CAP_BYTES:
        diff = diff[:DIFF_CAP_BYTES] + f"\n…（截断：git diff 全文 {len(diff)} B）"
    targets = (
        "\n".join(file_reading(t) for t in case.targets)
        if case.targets
        else "（案卷未声明 targets: 行）"
    )
    ledger_rows = "\n".join(relevant_ledger_rows(case)) or "（无相关台账行）"
    return (
        "## 自动附着材料（CLI 确定性抓取，非撰写者所写）\n\n"
        f"### git diff --stat\n{run_git('diff', '--stat') or '（空）'}\n\n"
        f"### git diff\n{diff or '（空）'}\n\n"
        f"### git log --oneline -5\n{run_git('log', '--oneline', '-5') or '（空）'}\n\n"
        f"### 涉事文件字节读数\n{targets}\n\n"
        f"### 相关台账行\n{ledger_rows}"
    )


def render_dossier(case: Case, material: str, *, order: str = "AB") -> str:
    """案卷全文 = 撰写者侧内容 + 自动附着材料，按标签序摆放两选项。

    ``order="BA"`` 只把两选项内容对调（标签仍为 A/B），材料与问题原样不动——因此
    双序两局所见是**同一份材料的两种摆法**，材料哈希按 AB 规范序计算即足以冻结全部
    所见信息。
    """
    a, b = (
        (case.option_a, case.option_b)
        if order == "AB"
        else (case.option_b, case.option_a)
    )
    parts = [
        f"# 案卷 {case.case_id}",
        f"## 问题\n{case.question}",
        f"## 选项 A\n{a}",
        f"## 选项 B\n{b}",
    ]
    if case.evidence:
        parts.append(f"## 证据\n{case.evidence}")
    parts.append(material)
    return "\n\n".join(parts) + "\n"


def build_dossier(case: Case) -> str:
    """规范序（AB）案卷全文——落留痕、算材料哈希的那一份。"""
    return render_dossier(case, build_material(case))


def _sha(text: str) -> str:
    """内容哈希（CRLF→LF 规范化后取 sha256 前 16 位）——与 sync.tree_hash 同一课：
    行尾是检出环境的产物，不该进哈希。"""
    return hashlib.sha256(text.replace("\r\n", "\n").encode("utf-8")).hexdigest()[:16]


def dossier_hash(dossier: str) -> str:
    """材料冻结指纹（进留痕，供「裁决被事后证据推翻」回访计数）。"""
    return _sha(dossier)


# ---------------------------------------------------------------------------
# 提示与解析（结构化输出的强制约束）
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "你是外部仲裁团的独立裁判。只依据案卷文本判决，案卷之外一无所知，"
    "不得调用工具、不得臆测未写出的事实。必须二选一，不许弃权、不许判并立。"
    "只输出一个 JSON 对象，不要任何前后缀或代码块之外的文字：\n"
    '{"choice": "A" 或 "B", "reason": "不超过 60 字的中文理由", "confidence": 0 到 1 的小数}'
)

_USER_FOOTER = "请在选项 A 与选项 B 之间做出判决（按上面的 JSON 形状回答）。"


def render_prompt(
    case: Case, material: str, brief: str, *, order: str = "AB"
) -> list[dict[str, str]]:
    """一次调用的 messages：简报（既定规则）+ 按标签序摆放的案卷 + 判决指令。

    Args:
        order: ``"AB"`` 原序；``"BA"`` 把两选项内容对调摆放（标签仍是 A/B），
            用于检出位置偏好——答案回来须经 :func:`flip` 才与 AB 局可比。
    """
    blocks = [
        f"# 评审简报（既定规则，必须遵守）\n{brief}" if brief else "",
        render_dossier(case, material, order=order),
        _USER_FOOTER,
    ]
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n\n".join(x for x in blocks if x)},
    ]


def flip(choice: str, order: str) -> str:
    """把当面标签映射回真实选项：BA 局的标签 A 挂的是选项 B 的内容。"""
    if choice not in ("A", "B"):
        return ""
    return choice if order == "AB" else ("B" if choice == "A" else "A")


def parse_vote(content: str) -> dict[str, Any]:
    """裁判输出 → ``{choice, reason, confidence}``；不合形状则 ``choice=""``。

    强制 JSON 只是请求参数（``response_format``），并非所有型号都守得住，故解析侧
    兜底：抓正文里第一个 ``{…}`` 块；choice 非 A/B 视为无效票（计入 inconclusive）。
    """
    out: dict[str, Any] = {
        "choice": "",
        "reason": (content or "").strip()[:200],
        "confidence": 0.0,
    }
    m = _JSON_BLOCK_RE.search(content or "")
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
    out["reason"] = str(data.get("reason", "")).strip()[:300]
    try:
        conf = float(data.get("confidence"))
    except (TypeError, ValueError):
        conf = 0.0
    out["confidence"] = min(1.0, max(0.0, conf))
    return out


# ---------------------------------------------------------------------------
# 席位（provider 直调）：Key 读取 → 可用性 → 回退席上场
# ---------------------------------------------------------------------------
@dataclass
class Seat:
    """一席裁判的传输面：端点 + 型号 + Key 变量名 + 逐席请求参数（§4 三席直调）。

    ``usable`` 由 :func:`seat_readiness` 判定（配置齐全 且 Key 就位）。不可用的席位照样
    进留痕与读数面并带 ``reason``：「这席这次没投票」必须能区分「没配好」「没 Key」「回退
    席待命」——混为一谈就没法处理主席。

    一席只有一条端点（用户终裁 2026-10-13：仲裁团一律按量付费，TokenPlan 套餐路径整个
    不做），所以没有端点链、没有下探，``base_url``/``model``/``key_env`` 就是本庭落点。
    """

    name: str
    model: str
    base_url: str
    key_env: str
    params: dict[str, Any] = field(default_factory=dict)
    role: str = "primary"
    defect: str = ""
    reason: str = ""
    usable: bool = False

    def public(self) -> dict[str, Any]:
        """留痕/读数用的一行状态（Key 值永不进这里，只有变量名与布尔）。"""
        d = asdict(self)
        d.pop("params", None)
        return d


def _to_seat(raw: dict[str, Any], role: str) -> Seat:
    """registry 规范化席位 dict → :class:`Seat`（:func:`~.registry.normalize_seat` 的形状）。"""
    return Seat(
        name=str(raw.get("name") or role),
        model=str(raw.get("model") or ""),
        base_url=str(raw.get("base_url") or ""),
        key_env=str(raw.get("key_env") or ""),
        params=dict(raw.get("params") or {}),
        role=role,
        defect=str(raw.get("defect") or ""),
    )


def read_key(key_env: str) -> str:
    """取一席的 Key：进程环境变量优先，回落项目根 ``.env``（gitignored，永不入库）。

    ``.env`` 用 :func:`dotenv_values` **读文件**而不是 ``load_dotenv``：后者会把值灌进
    ``os.environ``，让一次读数污染整个进程环境（Key 的可见面凭空变大）。返回值只进请求头；
    任何要打印/落盘的文本都先过 :func:`scrub`。
    """
    if not key_env:
        return ""
    if value := os.environ.get(key_env, "").strip():
        return value
    env_file = PROJECT_ROOT / ".env"
    if not env_file.is_file():
        return ""
    try:
        return (dotenv_values(env_file).get(key_env) or "").strip()
    except OSError:
        return ""


def seat_readiness(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """逐席读 Key 判可用性；**≥1 主席不可达**才让回退席上场（§4 回退必须告警）。

    Args:
        cfg: :func:`load_config` 的产物；缺席即自己读 registry（读数面 ``orch status``／
            mirror 直接可用，不必先开庭）。

    Returns:
        ``{"seats": [Seat…主席+已启用的回退席], "voting": [Seat…本庭实际开庭的席],
        "unavailable": [Seat…], "missing_primary": [Seat…], "fallback_enabled": bool,
        "fallback_used": bool, "fallback_blocked": str, "notice": str}``。**不返回 Key 值。**
    """
    out = cfg if cfg is not None else load_config()
    act = presets.active()
    panel = list(routing.jury_panel(act.preset.jury))
    producer = routing.producer_family()
    all_primary = [_to_seat(raw, "primary") for raw in out["seats"]]

    # 预设构成换血（用户裁决 2026-10-12 22:4x）：面板=预设点名的三家族；谷时隐性态换成
    # 固定三席 GLM+KIMI+Mimo（终裁 v4，见 :func:`.routing.jury_panel`）。**认得出家族却不在
    # 面板**的配置席不入 panel（仍留痕报数，静默剔除会让人以为庭里真有那席）；认不出家族的席
    # 照常入 panel——名字不带族名（测试桩/人自起的席名）不构成换血证据，宁可放行也不误伤。
    # 缺席的**已点名家族**要出声：那是用户该去补 registry.jury.seats 的信号。
    def _family(s: Seat) -> str:
        return seat_family(s.name, s.model)

    seats = [s for s in all_primary if not _family(s) or _family(s) in panel]
    out_of_panel = [s for s in all_primary if _family(s) and _family(s) not in panel]
    present = {_family(s) for s in seats}
    missing_panel = [family for family in panel if family not in present]
    fallback_enabled = bool(out["fallback"].get("enabled"))
    fallback = _to_seat(out["fallback"], "fallback")
    planned = seats + ([fallback] if fallback_enabled else [])
    for s in planned:
        if s.defect:
            s.reason = f"配置{s.defect}"
        elif not read_key(s.key_env):
            s.reason = f"缺 {s.key_env}"
        else:
            s.usable = True
    missing = [s for s in seats if not s.usable]
    voting = [s for s in seats if s.usable]
    # §4 同族硬规则对回退席同样成立：被评模型的同族回退席一进场就是违约，会把「该降级」
    # 的案子变成 declined。缺席时宁可 2 席运行（平票由 twin 终裁兜底），也不请同族进来。
    fallback_blocked = ""
    if fallback_enabled and _family(fallback) and _family(fallback) == producer:
        fallback.reason = f"被评模型 {producer} 同族，依 §4 不得入庭"
        fallback_blocked = fallback.reason
    fallback_used = (
        fallback_enabled and bool(missing) and fallback.usable and not fallback_blocked
    )
    if fallback_used:
        voting.append(fallback)
    return {
        "seats": planned,
        "voting": voting,
        "unavailable": [s for s in planned if not s.usable],
        "missing_primary": missing,
        "fallback_enabled": fallback_enabled,
        "fallback_used": fallback_used,
        #: 回退席被 §4 同族硬规则挡下的原因（''=没被挡）——缺席告警要说清「为何不上回退」
        "fallback_blocked": fallback_blocked,
        "notice": FALLBACK_NOTICE if fallback_used else "",
        "panel": panel,
        "preset": act.name,
        #: 被评模型家族（谷时=隐性预设，平时=toml 那一行）——同族硬规则的锚点
        "producer": producer,
        #: 谷时隐性态是否在生效（读数面据此说明「面板为什么不是预设自带的那三族」）
        "hidden": bool(routing.hidden_valley_preset()),
        "out_of_panel": out_of_panel,
        "missing_panel": missing_panel,
    }


def _absence_note(plan: dict[str, Any]) -> str:
    """主席缺席的告警一句（调用者面 :func:`_court_alert` 与读数面 :func:`readiness_summary`
    共用一份文案——双面告警说的是同一件事，两处各写一份必然漂移）。

    终裁口径（2026-10-13）：缺 Key 的席**标 unavailable + 出声**，本庭减员运行，平票由
    ``escalate="twin"`` 终裁兜底；回退席被同族硬规则挡下时一并说明**为什么不上回退**。
    """
    missing = plan.get("missing_primary") or []
    if not missing or plan.get("fallback_used"):
        # 回退席顶上了就不叫「缺席」——那句 FALLBACK_NOTICE 已点名不可达主席，
        # 再报一遍「缺席 X」会让读数面像是同时回退又减员。
        return ""
    names = "、".join(f"{s.name}（{s.reason}）" for s in missing)
    note = (
        f"[!] 缺席 {names} → 本庭 {len(plan.get('voting') or [])} 席运行"
        "（减员照常开庭，平票由 twin 终裁兜底）"
    )
    blocked = str(plan.get("fallback_blocked") or "")
    if blocked:
        note += f"｜回退席不上场：{blocked}"
    return note


def readiness_summary(plan: dict[str, Any]) -> str:
    """席位读数一行话（``orch status`` 判决包健康段与 mirror 概览共用，防两处文案漂移）。"""
    panel = plan.get("panel") or []
    panel_note = (
        f"｜面板 {'+'.join(panel)}（预设 {plan.get('preset', '?')}"
        f"{'，谷时隐性态换面板' if plan.get('hidden') else ''}）"
        if panel
        else ""
    )
    if not plan["seats"]:
        line = "仲裁团未选型（registry.jury.seats 为空，开庭即入队）"
        if plan.get("out_of_panel"):
            names = "、".join(s.name for s in plan["out_of_panel"])
            line += f"｜预设外配置席未入面板：{names}"
        return line + panel_note
    in_court = {s.name for s in plan["voting"]}
    parts: list[str] = []
    for s in plan["seats"]:
        tag = f"{s.name}{'[回退]' if s.role == 'fallback' else ''}"
        if s.name in in_court:
            parts.append(f"{tag}✓")
        elif s.role == "fallback":
            parts.append(f"{tag}待命")
        else:
            parts.append(f"{tag}✗({s.reason})")
    line = f"席位 {' '.join(parts)}｜本庭 {len(plan['voting'])} 席"
    if plan["fallback_used"]:
        dropped = "、".join(s.name for s in plan["missing_primary"])
        line += f"｜[!] {FALLBACK_NOTICE}（不可达主席：{dropped}）"
    if absent := _absence_note(plan):
        line += f"｜{absent}"
    line += panel_note
    if plan.get("out_of_panel"):
        producer = str(plan.get("producer") or "")
        names = "、".join(
            f"{s.name}（被评同族，依 §4 不得入庭）"
            if seat_family(s.name, s.model) == producer
            else s.name
            for s in plan["out_of_panel"]
        )
        line += f"｜预设外未入面板：{names}"
    if plan.get("missing_panel"):
        line += f"｜[!] 面板缺席：{'、'.join(plan['missing_panel'])}（registry.jury.seats 未配置）"
    return line


def load_config() -> dict[str, Any]:
    """registry ``jury`` 块（缺失即出厂默认，见 :data:`DEFAULT_JURY`）。"""
    return Registry.load().jury


def load_brief() -> tuple[str, bool]:
    """冻结简报正文 + 是否已批准（``approved:`` 行）。文件缺席 → ``("", False)``。"""
    if not JURY_BRIEF_PATH.exists():
        return "", False
    text = JURY_BRIEF_PATH.read_text(encoding="utf-8")
    return text, bool(_BRIEF_APPROVED_RE.search(text))


def scrub(text: str, *, secrets: tuple[str, ...] = ()) -> str:
    """脱敏自由文本：已知 Key 值与疑似密钥形态一律换成 ``[REDACTED]``。

    Args:
        secrets: 本次会话实际使用的密钥（连它都不在的话，模式匹配就是唯一防线）。
    """
    out = text or ""
    for secret in (s for s in secrets if s):
        out = out.replace(secret, "[REDACTED]")
    out = _SECRET_RES[0].sub("[REDACTED]", out)
    return _SECRET_RES[1].sub(lambda m: f"{m.group(1)}{m.group(2)}[REDACTED]", out)


def family_violations(models: list[str], family: str) -> list[str]:
    """§4 硬规则的**被评模型相对化**版：本庭不得含被评模型同族型号（同族自恋偏差）。

    档位退役前这里写死「不得含 Qwen」——mimo 预设的面板恰恰要点名 Qwen 旗舰（用户裁决
    2026-10-12 22:4x），绝对规则会把合法面板拒之门外。路由 v3 再相对化一步：锚点从
    「toml 那一行的预设」换成「本庭实际在评谁的模型」（:func:`.routing.producer_family`）
    ——谷时被评的是 DeepSeek，请 Qwen/Mimo 旗舰回庭当第三席正是设计，不是违规。
    返回违约型号清单；``family`` 为空（认不出被评模型）不判。
    """
    if not family:
        return []
    return [m for m in models if seat_family(m) == family]


# ---------------------------------------------------------------------------
# 调用（并发）与计票
# ---------------------------------------------------------------------------
@dataclass
class Ballot:
    """一局（某席位 × 某标签序）的结果。``error`` 非空 = 该局没拿到可用答案。"""

    seat: str
    model: str
    order: str
    choice: str = ""
    real: str = ""
    reason: str = ""
    confidence: float = 0.0
    usage: dict[str, Any] = field(default_factory=dict)
    error: str = ""


def chat_once(
    model: str,
    messages: list[dict[str, str]],
    *,
    base_url: str,
    key: str,
    timeout_s: int,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """向 OpenAI 兼容端点发一次非流式对话，返回解析后的响应 JSON。

    ``params`` 是**逐席请求参数覆盖**（registry 席位 ``params`` 键）：三家 provider 的参数
    约束实测不一致——``kimi-k2.6`` 只接受 ``temperature=1``，共享一个写死的 0 会让那席
    每次 400，于是「主席不可达」其实是请求体不合席位胃口。

    Raises:
        JuryError: HTTP 非 2xx 或响应体不是 JSON（消息已脱敏）。
        requests.RequestException: 网络层失败（超时/连接），由 :func:`_ballot` 收敛成
            该局 error，不外抛。
    """
    resp = requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
            **(params or {}),
        },
        timeout=timeout_s,
    )
    if resp.status_code != 200:
        raise JuryError(f"HTTP {resp.status_code}: {scrub(resp.text)[:200]}")
    try:
        data = resp.json()
    except ValueError as exc:
        raise JuryError(f"响应非 JSON：{scrub(str(exc))[:200]}") from exc
    if not isinstance(data, dict) or "choices" not in data:
        raise JuryError("响应缺 choices 字段")
    return data


def _ballot(
    seat: Seat,
    order: str,
    case: Case,
    material: str,
    brief: str,
    *,
    cfg: dict[str, Any],
    lock: threading.Lock | None = None,
) -> Ballot:
    """一局的全部失败路径都收敛到 :attr:`Ballot.error`（一案多裁判，单点不可达不崩盘）。

    Key 在本局内现取（:func:`read_key`）：逐席 provider 各有各的变量名，且它只进请求头，
    不进 :class:`Seat`——否则 :meth:`Seat.public` 的留痕字典就成了泄漏面。

    ``lock`` = 同提供商组织的串行闸（见 :func:`collect_votes`）；缺席即不加闸。
    """
    key = read_key(seat.key_env)
    messages = render_prompt(case, material, brief, order=order)
    gate = lock if lock is not None else nullcontext()
    with gate:
        try:
            data = chat_once(
                seat.model,
                messages,
                base_url=seat.base_url,
                key=key,
                timeout_s=cfg["call_timeout_s"],
                max_tokens=cfg["max_output_tokens"],
                params=seat.params,
            )
        except (
            Exception
        ) as exc:  # 网络/HTTP/响应形状都只作废这一局，不外抛（§4 降级入队）
            return Ballot(
                seat=seat.name,
                model=seat.model,
                order=order,
                error=scrub(
                    f"{seat.name}/{order} {type(exc).__name__}: {exc}", secrets=(key,)
                )[:300],
            )
    first = (data.get("choices") or [{}])[0]
    content = (first.get("message") or {}).get("content") or ""
    # 截断是**预算不合**（改 registry 即可修因），记成无效票会把渠道故障伪装成「裁判模棱两可」
    if first.get("finish_reason") == "length":
        det = (data.get("usage") or {}).get("completion_tokens_details") or {}
        return Ballot(
            seat=seat.name,
            model=seat.model,
            order=order,
            usage=data.get("usage") or {},
            error=(
                f"{seat.name}/{order} 截断：正文未出"
                f"（max_output_tokens={cfg['max_output_tokens']}，"
                f"reasoning_tokens={det.get('reasoning_tokens', '?')}）"
            ),
        )
    vote = parse_vote(content)
    return Ballot(
        seat=seat.name,
        model=seat.model,
        order=order,
        choice=vote["choice"],
        real=flip(vote["choice"], order),
        reason=scrub(vote["reason"], secrets=(key,)),
        confidence=vote["confidence"],
        usage=data.get("usage") or {},
    )


def collect_votes(
    seats: list[Seat],
    case: Case,
    dossier: str,
    brief: str,
    *,
    cfg: dict[str, Any],
) -> list[Ballot]:
    """并发跑 ``len(seats) × 2`` 局，按 (席位, 序) 原序返回（留痕顺序须稳定可重放）。

    **同一提供商组织内串行**：席内两局（AB/BA）打的是同一个账号，而官方直调的并发额度
    按组织算——实测 DeepSeek 免费档 ``max organization concurrency: 1``，同席两局并发必
    429（2026-10-12 三席真实 smoke 首爆）。不同席是不同提供商，照常并发，故整案墙钟 ≈
    单局时长 × 2 而非 × 6。

    总等待受 ``cfg["deadline_s"]`` 约束：超时未归的局记 error 并放弃等待（线程随进程
    结束），**不在这里重试**——重试是 :func:`adjudicate` 的 attempt 层职责。
    """
    jobs = [(s, o) for s in seats for o in ORDERS]
    model_of = {s.name: s.model for s in seats}
    #: 闸按 (端点, Key 变量名) 归组而非按席位：两席挂同一账号（如官方 deepseek 旗舰 +
    #: 同家 flash）时也必须串行，按席位分锁会让它们照旧并发撞额度。
    locks: dict[tuple[str, str], threading.Lock] = {}
    for seat in seats:
        locks.setdefault((seat.base_url, seat.key_env), threading.Lock())
    got: dict[tuple[str, str], Ballot] = {}
    ex = concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(jobs)))
    try:
        futs = {
            ex.submit(
                _ballot,
                seat,
                order,
                case,
                dossier,
                brief,
                cfg=cfg,
                lock=locks[(seat.base_url, seat.key_env)],
            ): (seat.name, order)
            for seat, order in jobs
        }
        done, pending = concurrent.futures.wait(
            futs, timeout=cfg["deadline_s"], return_when="ALL_COMPLETED"
        )
        for fut in done:
            got[futs[fut]] = fut.result()
        for fut in pending:
            name, order = futs[fut]
            got[(name, order)] = Ballot(
                seat=name,
                model=model_of[name],
                order=order,
                error=f"timeout(>{cfg['deadline_s']}s)",
            )
    finally:
        # wait=False：已经放弃等待了，就别在解释器退出时把整案再拖回超时窗口
        ex.shutdown(wait=False, cancel_futures=True)
    return [got[(seat.name, order)] for seat, order in jobs]


def _sum_usage(ballots: list[Ballot]) -> dict[str, int]:
    keys = ("prompt_tokens", "completion_tokens", "total_tokens")
    return {k: sum(int(b.usage.get(k, 0) or 0) for b in ballots) for k in keys}


def usage_by_model(ballots: list[Ballot]) -> dict[str, dict[str, int]]:
    """按型号聚合 usage（成本折算的输入；失败局自然为 0）。"""
    out: dict[str, list[Ballot]] = {}
    for b in ballots:
        out.setdefault(b.model, []).append(b)
    return {m: _sum_usage(bs) for m, bs in out.items()}


def estimate_cost(
    usage: dict[str, dict[str, int]], pricing: dict[str, Any]
) -> dict[str, Any]:
    """按 ``registry.jury.pricing``（元/百万 token）折算成本。

    无价目表时 ``amount=None`` 并附 note——**不猜价**：落进留痕的成本数字要能对外负责。
    """
    total = 0.0
    priced: list[str] = []
    unpriced: list[str] = []
    for model in sorted(usage):
        p = pricing.get(model)
        if not isinstance(p, dict):
            unpriced.append(model)
            continue
        try:
            pin = float(p.get("input", 0))
            pout = float(p.get("output", 0))
        except (TypeError, ValueError):
            unpriced.append(model)
            continue
        u = usage[model]
        total += (u["prompt_tokens"] * pin + u["completion_tokens"] * pout) / 1e6
        priced.append(model)
    note = (
        "无价目表：registry.jury.pricing 未覆盖任何 panel 型号（成本不猜）"
        if not priced
        else (f"部分型号无价目表，成本不含：{unpriced}" if unpriced else "")
    )
    return {
        "amount": round(total, 6) if priced else None,
        "unit": "CNY/百万 token 折算" if priced else "",
        "note": note,
        "priced": priced,
        "unpriced": unpriced,
    }


def tally(ballots: list[Ballot]) -> dict[str, Any]:
    """双序一致性 → 计票 → 多数裁决 / 平票或大量 inconclusive 时 ``escalate="twin"``。

    计票身份是**席位名**而非型号：逐席 provider 下同一型号可能挂在两个端点上，「谁投的」
    要以席为准（两席同型号各投一票才是两票，合并成一票会悄悄改变多数）。

    Returns:
        ``{"jurors": [{seat, model, consistent, vote, ballots, usage}…],
        "votes": {seat: 真实选项}, "tally": {"A": n, "B": n}, "inconclusive": [seat…],
        "verdict": "A"|"B"|"", "escalate": "twin"|""}``。
    """
    by_seat: dict[str, list[Ballot]] = {}
    for b in ballots:
        by_seat.setdefault(b.seat, []).append(b)
    jurors: list[dict[str, Any]] = []
    votes: dict[str, str] = {}
    inconclusive: list[str] = []
    for seat in sorted(by_seat):
        bs = sorted(by_seat[seat], key=lambda b: b.order)
        valid = [b for b in bs if not b.error and b.real in ("A", "B")]
        consistent = (
            len(bs) == 2 and len(valid) == 2 and len({b.real for b in valid}) == 1
        )
        jurors.append(
            {
                "seat": seat,
                "model": bs[0].model,
                "consistent": consistent,
                "vote": valid[0].real if consistent else "",
                "ballots": [asdict(b) for b in bs],
                "usage": _sum_usage(bs),
            }
        )
        if consistent:
            votes[seat] = valid[0].real
        else:
            inconclusive.append(seat)
    n_a = sum(1 for v in votes.values() if v == "A")
    n_b = len(votes) - n_a
    escalate = (
        "twin" if (not votes or n_a == n_b or len(inconclusive) > len(votes)) else ""
    )
    return {
        "jurors": jurors,
        "votes": votes,
        "tally": {"A": n_a, "B": n_b},
        "inconclusive": inconclusive,
        "verdict": "" if escalate else ("A" if n_a > n_b else "B"),
        "escalate": escalate,
    }


# ---------------------------------------------------------------------------
# 裁决书（verdict_summary）与 E6 twin 终裁（PSEP_3 §4.4）
# ---------------------------------------------------------------------------
def _seat_confidence(juror: dict[str, Any]) -> float | None:
    confs = [
        b.get("confidence")
        for b in juror.get("ballots", [])
        if b.get("confidence") is not None
    ]
    return max(confs) if confs else None


def render_verdict_summary(summary: dict[str, Any]) -> str:
    """裁决书**文本**（一行要点 + 各席 confidence + 多数理由汇总 +（若有）twin 终裁）。"""
    winner = str(summary.get("winner") or "")
    by = str(summary.get("decided_by") or "")
    tally = summary.get("tally") or {}
    if winner:
        head = (
            f"裁决={winner}（{by or 'jury'} 定案；"
            f"计票 A:{tally.get('A')} B:{tally.get('B')}）"
        )
    else:
        head = f"未定案（计票 A:{tally.get('A')} B:{tally.get('B')}）"
    parts = [head]
    confs = []
    for s in summary.get("seats", []):
        run = f"{s.get('seat')}={s.get('vote') or '-'}"
        conf = s.get("confidence")
        if isinstance(conf, (int, float)):
            run += f"/{float(conf):.2f}"
        confs.append(run)
    if confs:
        parts.append("各席：" + " ".join(confs))
    if summary.get("majority_reasons"):
        parts.append("多数理由：" + "；".join(summary["majority_reasons"]))
    twin = summary.get("twin")
    if twin:
        conf = twin.get("confidence")
        cs = f"{float(conf):.2f}" if isinstance(conf, (int, float)) else "-"
        parts.append(
            f"twin 终裁：{twin.get('choice')}（置信度 {cs}）：{twin.get('reason')}"
        )
    return "｜".join(parts)


def build_verdict_summary(record: dict[str, Any]) -> dict[str, Any]:
    """综合**裁决书**（PSEP_3 §4.4）：胜方 + 多数理由汇总 + 各席 confidence +（若有）twin 终裁。

    由判决记录**纯推导**（含 ``final_verdict``/``decided_by``/``twin`` 字段），不读盘——
    四处落点（jury ``<case>.json`` / preference-journal / ``pysci decisions`` / TUI）共用同一份，
    杜绝各写各的。
    """
    decided_by = str(
        record.get("decided_by") or ("jury" if record.get("verdict") else "")
    )
    winner = str(record.get("final_verdict") or record.get("verdict") or "")
    seats: list[dict[str, Any]] = []
    majority: list[str] = []
    for j in record.get("jurors", []):
        seat = {
            "seat": j.get("seat"),
            "model": j.get("model"),
            "vote": j.get("vote") or "",
            "consistent": bool(j.get("consistent")),
            "confidence": _seat_confidence(j),
        }
        seats.append(seat)
        if winner and seat["vote"] == winner:
            reason = next(
                (
                    str(b.get("reason") or "").strip()
                    for b in j.get("ballots", [])
                    if b.get("real") == winner and str(b.get("reason") or "").strip()
                ),
                "",
            )
            if reason:
                majority.append(f"{seat['seat']}：{reason}")
    twin = None
    if decided_by == "twin":
        info = record.get("twin") or {}
        twin = {
            "choice": winner,
            "reason": str(info.get("reason") or ""),
            "confidence": info.get("confidence"),
        }
    summary: dict[str, Any] = {
        "winner": winner,
        "decided_by": decided_by,
        "tally": dict(record.get("tally") or {}),
        "inconclusive": list(record.get("inconclusive") or []),
        "seats": seats,
        "majority_reasons": majority,
        "twin": twin,
    }
    summary["text"] = render_verdict_summary(summary)
    return summary


def _resolve_twin(case: Case, *, model_tier: str | None = None) -> dict[str, Any]:
    """E6：平票/inconclusive 过半 → **自动** twin 终裁（复用 taste 机制，PSEP_3 §4.4）。

    Returns:
        ``{decided, final_verdict, decided_by, reason, confidence, sid, available}``：
        ``decided=True`` 时 ``final_verdict`` ∈ {A,B}（**收敛确定选项、不给平局**，REFACTOR §5）；
        否则为**保守回退**（twin 不可达 / 读不出选择）——调用方据此标挂起并**出声**，不静默放行。
    """
    from . import taste  # 延迟导入：taste 顶层引 jury，避免成环

    code, rec = taste.ruling_for_case(case, model_tier=model_tier)
    choice = str(rec.get("choice") or "")
    if code == 0 and choice in ("A", "B"):
        return {
            "decided": True,
            "final_verdict": choice,
            "decided_by": "twin",
            "reason": str(rec.get("reason") or ""),
            "confidence": rec.get("confidence"),
            "sid": str(rec.get("sid") or ""),
            "available": True,
        }
    available = code != 2  # code 2＝通道不可用（taste 未生效 / 宪章缺失）
    reason = str(rec.get("reason") or "") if not available else "twin 未给出确定选择"
    return {
        "decided": False,
        "final_verdict": "",
        "decided_by": "",
        "reason": reason,
        "confidence": None,
        "sid": str(rec.get("sid") or ""),
        "available": available,
    }


def verdict_summary_lines(record: dict[str, Any]) -> list[str]:
    """裁决书**多行呈现**（CLI/TUI 共用一份）：要点 + 逐席 + 多数理由 +（若有）twin 终裁。"""
    s = record.get("verdict_summary") or build_verdict_summary(record)
    out = [f"  裁决书：{s.get('text') or '（空）'}"]
    for seat in s.get("seats", []):
        conf = seat.get("confidence")
        cs = f"{float(conf):.2f}" if isinstance(conf, (int, float)) else "-"
        tag = "一致" if seat.get("consistent") else "inconclusive"
        mark = "√" if seat.get("vote") and seat.get("vote") == s.get("winner") else " "
        out.append(
            f"    {mark} {seat.get('seat')} 票={seat.get('vote') or '-'} "
            f"[{tag}] conf={cs}"
        )
    if s.get("majority_reasons"):
        out.append("    多数理由：" + "；".join(s["majority_reasons"]))
    if s.get("twin"):
        t = s["twin"]
        conf = t.get("confidence")
        cs = f"{float(conf):.2f}" if isinstance(conf, (int, float)) else "-"
        out.append(
            f"    twin 终裁：{t.get('choice')}（置信度 {cs}）：{t.get('reason')}"
        )
    return out


def _journal_verdict_summary(record: dict[str, Any]) -> None:
    """裁决书入**单一账目源** preference-journal（channel 记账；twin 定案标 decided_by）。

    ``kind="verdict_summary"``：它是同一判决的**文书面**（非又一票/又一次裁决），故
    :func:`~.verdict.health` 按 kind 跳过它，不重复计入通道用量。写失败 fail-soft——
    ``evals/jury/<case>.json`` 才是硬留痕。
    """
    try:
        from . import journal

        summary = record.get("verdict_summary") or {}
        decided_by = str(record.get("decided_by") or "")
        case_id = str(record.get("case_id") or "")
        journal.append(
            {
                "ts": str(record.get("ts") or ""),
                "situation": (
                    f"jury 裁决 {case_id}（{str(record.get('question') or '')[:80]}）"
                ),
                "options": "A=采纳 B=维持现状",
                "choice": str(record.get("verdict") or ""),
                "reason": str(summary.get("text") or ""),
                "source": "jury",
                "kind": "verdict_summary",
                "channel": "twin" if decided_by == "twin" else "jury",
                "case_id": case_id,
                "finding_id": case_id,  # 巡检 J 案卷 case_id==finding id：decisions 面据此可查
                "decided_by": decided_by,
                "final_verdict": str(record.get("final_verdict") or ""),
            }
        )
    except Exception:  # noqa: BLE001 - 账目写失败不炸判决（jury JSON 才是硬留痕）
        pass


# ---------------------------------------------------------------------------
# 留痕与队列
# ---------------------------------------------------------------------------
def _atomic_write_json(path: Path, obj: Any) -> None:
    """原子写 JSON（``*.tmp`` → replace，与 registry 同法：中断不留半文件）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    tmp.replace(path)


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def record_path(case_id: str) -> Path:
    """裁决留痕文件路径（``state/evals/jury/<case-id>.json``；重裁覆盖＝以最新为准）。"""
    return JURY_DIR / f"{case_id}.json"


def queue_path(case_id: str) -> Path:
    """降级队列条目路径（``state/evals/jury-queue/<case-id>.json``）。"""
    return JURY_QUEUE_DIR / f"{case_id}.json"


def write_record(record: dict[str, Any]) -> Path:
    """裁决落盘并销掉同案队列条目（渠道恢复后重裁，队列不留僵尸）。"""
    p = record_path(str(record["case_id"]))
    _atomic_write_json(p, record)
    queue_path(str(record["case_id"])).unlink(missing_ok=True)
    return p


def enqueue(case: Case, *, reason: str, key_hints: tuple[str, ...] = ()) -> Path:
    """降级留痕：案卷进队列待重裁（§4 渠道单点不可达时基建决策不停摆）。"""
    p = queue_path(case.case_id)
    _atomic_write_json(
        p,
        {
            "case_id": case.case_id,
            "case_file": case.path,
            "queued_at": _now(),
            "reason": scrub(reason, secrets=key_hints),
            "question": scrub(case.question),
        },
    )
    return p


# ---------------------------------------------------------------------------
# 门面：adjudicate / cmd_jury
# ---------------------------------------------------------------------------
@dataclass
class Run:
    """一次 ``orch jury`` 的结果（``cmd_jury`` 只负责打印，逻辑都在 :func:`adjudicate`）。"""

    status: str  # dry | recorded | queued | declined
    message: str
    record: dict[str, Any] | None = None
    path: Path | None = None
    dossier: str = ""


def _court_alert(plan: dict[str, Any]) -> str:
    """开庭告警一句（进 Run.message 开头，也进留痕 ``fallback.notice``）。

    两件事都要出声：**主席掉线后回退席顶上了**，以及**主席缺席、本庭减员运行**（终裁
    2026-10-13：缺 Key 席标 unavailable + 双面告警 + 剩余席照常开庭，平票由 twin 兜底）。
    文案与读数面 :func:`readiness_summary` 共用 :func:`_absence_note`，防两处漂移。
    """
    out: list[str] = []
    if plan["fallback_used"]:
        dropped = "、".join(s.name for s in plan["missing_primary"])
        out.append(f"[!] {FALLBACK_NOTICE}（不可达主席：{dropped}）")
    if absent := _absence_note(plan):
        out.append(absent)
    return "".join(f"{line}；" for line in out)


def adjudicate(case: Case, *, dry: bool = False, deadline_s: int | None = None) -> Run:
    """组卷 → 开庭 → 计票 → 落留痕；任何渠道故障降级入队（不 raise）。

    Args:
        case: :func:`parse_case` 的产物。
        dry: 只组卷并返回案卷文本，不发任何 API 调用、不落裁决。
        deadline_s: 覆盖 ``registry.jury.deadline_s`` 的整体等待上限（CLI ``--timeout``）。
    """
    brief, approved = load_brief()
    warn = "" if approved else "简报未批准（jury-brief.md 缺席或无 approved: 行）；"
    cfg_error = ""
    cfg: dict[str, Any] | None = None
    plan: dict[str, Any] | None = None
    try:
        cfg = load_config()
        plan = seat_readiness(cfg)
    except (OSError, ValueError) as exc:  # registry.json 坏掉也不该崩 CLI
        cfg_error = f"registry 读取失败 {type(exc).__name__}: {exc}"
    #: 本次涉及的 Key 值只用于 scrub 兜底（异常文本常把请求上下文带出来），不落任何盘。
    secrets: tuple[str, ...] = tuple(
        k for s in (plan["seats"] if plan else []) if (k := read_key(s.key_env))
    )
    #: material＝自动附着段；dossier＝按规范序摆好选项的全文（落留痕、算哈希的那一份）。
    #: 分开存是对的：双序两局各渲染一次 dossier，材料只抓一次。
    material = scrub(build_material(case), secrets=secrets)
    dossier = render_dossier(case, material)
    if dry:
        return Run(
            "dry",
            f"{warn}案卷组装完成（--dry：未调用任何 API）",
            record={"case_id": case.case_id, "dossier_hash": dossier_hash(dossier)},
            dossier=dossier,
        )
    if cfg_error or cfg is None or plan is None:
        return Run(
            "queued",
            f"{warn}降级入队：{cfg_error}",
            path=enqueue(case, reason=cfg_error, key_hints=secrets),
        )
    preset_name = presets.active().name
    producer = plan["producer"]
    # 硬规则的**实质条件**是「被评模型同族坐进了本庭」，不是「席位表里配了同族席」。
    # 路由 v3 之前两者等价（三家族面板 = 全部席位），之后不再等价：席位表跨预设共用，
    # 谷时 DeepSeek 是被评模型而 deepseek 席必然在册、又必然被面板（glm+kimi+本预设旗舰）
    # 挡在庭外——拿「在册」当违约会把谷时每一庭都锁死成 declined，而 that 恰是本单要开的庭。
    # 被挡在庭外的同族席照样在读数面点名（:func:`readiness_summary`），违约要被看见，
    # 但看见的位置在「这席没进庭」那一段，不在拒庭那条上。
    #
    # **回退席不参与本判据**（backlog 20261013-173000-leader）：它同族时已被
    # :func:`_plan_court` 按 §4 钉在庭外（``fallback_blocked``）、**永远不会投票**，
    # 而 ``plan["seats"]`` 是「主席 + 已启用的回退席」的规划面——照它审就等于
    # 「没上场也拒庭」。实测 2026-10-13 17:2x `approve 131505 --jury` 正是该误判：
    # 三主席 glm/kimi/qwen 全就绪，却因回退席 `deepseek-ai/DeepSeek-V4-Pro` 命中被评族
    # 被拒庭（jury.py 自身注释早已预警该形态）。回退席同族时的正确处理是**该席禁用**
    # （真需要回退时按设计走降级入队 + 双面告警），不是拒开庭。
    in_court = [s.model for s in plan["seats"] if s.role != "fallback"]
    if bad := family_violations(in_court, producer):
        return Run(
            "declined",
            f"拒开庭：本庭含被评模型 {producer} 的同族型号 {bad}"
            f"（§4 硬规则，排除同族自恋偏差；配置预设 {preset_name}）",
        )
    if not cfg["seats"]:
        reason = "registry.jury.seats 未选型（席位清单为空）"
        return Run(
            "queued",
            f"{warn}降级入队：{reason}",
            path=enqueue(case, reason=reason, key_hints=secrets),
        )
    if not plan["voting"]:
        reason = "无一席可开庭：" + "；".join(
            f"{s.name}({s.reason})" for s in plan["seats"]
        )
        return Run(
            "queued",
            f"{warn}降级入队：{reason}",
            path=enqueue(case, reason=reason, key_hints=secrets),
        )
    if deadline_s:
        cfg["deadline_s"] = deadline_s
    alert = _court_alert(plan)

    seats = plan["voting"]
    ballots: list[Ballot] = []
    errors: list[str] = []
    attempts = 0
    try:
        for attempt in (1, 2):
            attempts = attempt
            ballots = collect_votes(seats, case, material, brief, cfg=cfg)
            errors = [b.error for b in ballots if b.error]
            if not errors:
                break
    except Exception as exc:  # noqa: BLE001 — §4 硬要求：仲裁面绝不 raise
        errors = [f"{type(exc).__name__}: {exc}"]
        ballots = ballots or []
    if errors:
        reason = f"{len(errors)}/{len(ballots) or len(seats) * 2} 局失败：{errors[0]}"
        return Run(
            "queued",
            f"{alert}{warn}降级入队：{scrub(reason, secrets=secrets)}（已尝试 {attempts} 轮）",
            path=enqueue(case, reason=reason, key_hints=secrets),
        )

    t = tally(ballots)
    usage = usage_by_model(ballots)
    # E6（PSEP_3 §4.4）：平票/inconclusive 过半 → **自动** twin 终裁（收敛确定选项，不给平局）。
    twin: dict[str, Any] = _resolve_twin(case) if t["escalate"] else {}
    final = str(twin.get("final_verdict") or "")
    record = {
        "case_id": case.case_id,
        "case_file": case.path,
        "ts": _now(),
        "question": scrub(case.question),
        "options": {
            "A": scrub(case.option_a[:500], secrets=secrets),
            "B": scrub(case.option_b[:500], secrets=secrets),
        },
        "targets": case.targets,
        "dossier_hash": dossier_hash(dossier),
        "dossier_chars": len(dossier),
        "dossier": dossier,
        "brief": {
            "path": str(JURY_BRIEF_PATH),
            "approved": approved,
            "hash": dossier_hash(brief),
        },
        "seats": [s.public() for s in plan["seats"]],
        #: 路由面（谷时隐性态）：留痕要说得出「这一庭是谁开的庭」——面板构成随预设与
        #: 峰谷换血，缺了这一段，事后无法核对裁判当时看的是哪套席。
        "routing": {
            "preset": preset_name,
            "hidden": plan["hidden"],
            "producer": producer,
            "panel": plan["panel"],
        },
        "fallback": {
            "enabled": plan["fallback_enabled"],
            "used": plan["fallback_used"],
            "notice": plan["notice"],
            "missing_primary": [s.name for s in plan["missing_primary"]],
        },
        "panel": [s.model for s in seats],
        "attempts": attempts,
        "jurors": t["jurors"],
        "votes": t["votes"],
        "tally": t["tally"],
        "inconclusive": t["inconclusive"],
        "verdict": t["verdict"] or final,
        "escalate": t["escalate"],
        # E6 终裁字段（twin 定案则 verdict 收敛到确定选项；不可达则空 + 保守回退）
        "final_verdict": final,
        "decided_by": "twin" if final else "",
        "twin": (
            {
                "choice": final,
                "reason": str(twin.get("reason") or ""),
                "confidence": twin.get("confidence"),
                "sid": str(twin.get("sid") or ""),
                "available": bool(twin.get("available")),
            }
            if twin
            else None
        ),
        "usage": _sum_usage(ballots),
        "usage_by_model": usage,
        "cost": estimate_cost(usage, cfg["pricing"]),
    }
    record["verdict_summary"] = build_verdict_summary(record)
    p = write_record(record)
    _journal_verdict_summary(record)
    if t["escalate"]:
        head = f"模棱两可（tally={t['tally']} inconclusive={t['inconclusive']}）→ escalate=twin"
        if final:
            reason = str((record.get("twin") or {}).get("reason") or "")
            msg = f"{head} → twin 终裁收敛：{final}（理由：{reason[:80]}）"
        elif twin.get("available"):
            msg = f"{head}：twin 未给出确定选择 → 保守回退挂起（未通过，待补充指引）"
        else:
            msg = (
                f"{head}：twin 不可达（{twin.get('reason')}）→ 保守回退挂起，不静默放行"
            )
    else:
        msg = f"裁决={t['verdict']}（tally A:{t['tally']['A']} B:{t['tally']['B']}）"
    return Run("recorded", f"{alert}{warn}{msg}", record=record, path=p)


def cmd_jury(args: argparse.Namespace) -> int:
    """``orch jury <case-file> [--dry] [--timeout S]``——Tier C 仲裁团（README §5.3 J 通道）。

    阻塞执行（并行单轮，默认整体上限 5 min）；退出码 0=已留痕（裁决或 escalate=twin）、
    1=渠道降级（案卷入队待重裁）、2=案卷不可用或违反硬规则（拒开庭）。
    """
    try:
        case = parse_case(args.case_file)
    except (OSError, ValueError) as exc:
        print(f"✗ 案卷不可用：{exc}")
        print(
            "[NEXT] 案卷格式：## 问题 / ## 选项 A / ## 选项 B（必需）+ ## 证据（可选）"
            "，头部 targets: 路径1, 路径2 声明涉事文件"
        )
        return 2
    run = adjudicate(case, dry=args.dry, deadline_s=args.timeout)
    if run.status == "dry":
        print(run.dossier)
        print(f"— 材料哈希 {run.record['dossier_hash']}")
        print(
            "[NEXT] 去掉 --dry 即开庭（阻塞 ≈1 min，需 registry.jury.seats 选型 + "
            "逐席 Key 就位）"
        )
        return 0
    mark = {"queued": "!", "declined": "✗"}.get(run.status, "√")
    print(f"[{mark}] {run.message}")
    if run.status == "recorded":
        rec = run.record or {}
        for j in rec.get("jurors", []):
            conf = max((b["confidence"] for b in j["ballots"]), default=0.0)
            tag = "一致" if j["consistent"] else "inconclusive"
            print(
                f"    {j['seat']:<12} {j['model']:<26} 票={j['vote'] or '-'} "
                f"[{tag}] conf={conf:.2f}"
            )
        cost = rec.get("cost", {})
        amount = cost.get("amount")
        print(
            f"    留痕 {orchestration_rel(run.path)}｜材料哈希 "
            f"{rec.get('dossier_hash')}｜usage={rec.get('usage', {}).get('total_tokens')}"
            f"｜cost={'无价目表' if amount is None else amount}"
        )
        for line in verdict_summary_lines(rec):
            print(line)
        if rec.get("verdict"):
            action = "approve" if rec["verdict"] == "A" else "reject"
            by = "twin 终裁" if rec.get("decided_by") == "twin" else "仲裁多数"
            print(
                f"[NEXT] 裁决自动生效（{by}）：{action} 对应提案，"
                "组长只写任务书派 devops（§5.1 J 路）"
            )
        else:
            print(
                "[NEXT] escalate=twin 且 twin 不可达/未表态 → **保守回退挂起**："
                "提案维持待审批、未入队；请人工终裁（orch twin consult 或 --override）"
            )
        return 0
    if run.status == "queued":
        print(f"    队列 {orchestration_rel(run.path)}")
        print(f"    席位 {readiness_summary(seat_readiness())}")
        print(
            "[NEXT] 修好渠道（registry.jury.seats 选型 + 该席 Key 进环境或项目根 .env）"
            f"后重裁：orch jury {args.case_file}"
        )
        return 1
    print("[NEXT] 改 registry.jury.seats 去掉预设同族型号后重裁（§4 硬规则不可放宽）")
    return 2


__all__ = [
    "Ballot",
    "Case",
    "DEFAULT_MAX_TOKENS",
    "FALLBACK_NOTICE",
    "JuryError",
    "Run",
    "Seat",
    "adjudicate",
    "build_dossier",
    "chat_once",
    "cmd_jury",
    "dossier_hash",
    "estimate_cost",
    "load_brief",
    "parse_case",
    "parse_vote",
    "family_violations",
    "seat_family",
    "read_key",
    "readiness_summary",
    "scrub",
    "seat_readiness",
    "tally",
    "build_verdict_summary",
    "render_verdict_summary",
    "verdict_summary_lines",
]
