"""人机运行时配置真源 ``data/用户的inbox/pysci_settings.toml`` 的读写面。

用户裁决 2026-10-12 22:4x（backlog 20261012-225500-leader）：运行时旋钮收敛到本文件，
**手改一行即在下一次派发生效**，在途会话不受影响。2026-10-22 用户裁决：文件随目录
划分迁至 ``data/用户的inbox/``（用户自管配置真源——inbox 禁依赖规则的显式例外，
读侧 fail-open 即缺席/被改名时回落出厂值，不阻塞生产）。本模块只承担两件事：

- **读侧 fail-open**：文件缺席 / 键缺席 / 解析失败一律回落出厂值，并把原因作为警告
  带回调用方——一个人工配置文件不该能把生产阻塞死；
- **写侧外科式**：机器只改 ``[inspector] enabled`` 一行，其余内容（含用户写的全部
  注释——注释即规格）逐字保留。不做 TOML round-trip 序列化：那会把注释整篇吃掉。
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pysci.paths import DATA_ROOT

SETTINGS_PATH: Path = DATA_ROOT / "用户的inbox" / "pysci_settings.toml"

#: ``[model] preset`` 合法值（精确串，全小写；大小写敏感是 toml 注释里的原话）。
PRESET_NAMES: tuple[str, ...] = ("qwen", "mimo", "deepseek")

#: 文件缺席 / 键缺席 / 非法值时的出厂预设（= 当前观察期主力，改它属护栏级）。
DEFAULT_PRESET: str = "mimo"


def read_settings(path: Path | None = None) -> dict[str, Any]:
    """读整份配置；任何异常都返回 ``{}``（读侧 fail-open，警告由消费方按需补）。"""
    target = path or SETTINGS_PATH
    try:
        with target.open("rb") as fh:
            return tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError, ValueError):
        return {}


def preset_name(data: dict[str, Any] | None = None) -> tuple[str, list[str]]:
    """当前预设名。

    Args:
        data: 已读到的配置；``None`` 时自己读盘。

    Returns:
        ``(预设名, 警告列表)``。键缺席 / 值非法 / 文件读不出 → 出厂预设 + 一条警告
        （警告要指明「改哪一行、合法串是什么」，否则用户只能靠猜）。
    """
    if data is None:
        data = read_settings()
    raw = None
    try:
        raw = data.get("model", {}).get("preset")
    except AttributeError:
        raw = None
    if raw is None:
        return DEFAULT_PRESET, [
            f"[model] preset 未配置（或 {SETTINGS_PATH.name} 读不出）→ 出厂预设 {DEFAULT_PRESET!r}"
        ]
    name = str(raw).strip()
    if name not in PRESET_NAMES:
        return DEFAULT_PRESET, [
            f"[model] preset = {raw!r} 不是合法预设（合法串：{'/'.join(PRESET_NAMES)}，"
            f"全小写）→ 回落 {DEFAULT_PRESET!r}"
        ]
    return name, []


def inspector_flag(data: dict[str, Any] | None = None) -> bool | None:
    """``[inspector] enabled`` 的读数；键/文件缺席返回 ``None``（调用方回落 legacy 文件）。

    显式写 ``false``/``0``/``off``/``no`` 视为关，其余非空串视为开（与旧
    ``state/inspector.enabled`` 的宽容读法同口径）。
    """
    if data is None:
        data = read_settings()
    try:
        raw = data.get("inspector", {}).get("enabled")
    except AttributeError:
        return None
    if raw is None:
        return None
    if isinstance(raw, bool):
        return raw
    return str(raw).strip().lower() not in ("false", "0", "off", "no", "")


#: 夜航开关的宽容真假值表（字符串/数字手写面的常见写法；与 inspector 同精神）。
_NIGHT_TRUE: tuple[str, ...] = ("true", "1", "on", "yes")
_NIGHT_FALSE: tuple[str, ...] = ("false", "0", "off", "no", "")


def night_enabled(data: dict[str, Any] | None = None) -> tuple[bool, list[str]]:
    """``[night] enabled`` 夜航总开关（PSEP_2 §3.1；读取面 fail-open）。

    **文件缺席 / 解析失败 / ``[night]`` 段或键缺席 / 非法值一律回落 ``False``**，并把
    原因作为**警告**带回调用方（读侧 fail-open——一个人工配置不该把注入钩子阻塞死；
    非法值尤其**要出声**，否则「写了没生效」只能靠猜）。布尔按原值；字符串/数字按宽容
    真值表；其余一切（含任意其它串）回落 False + 警告。

    Args:
        data: 已读到的配置；``None`` 时自己读盘（此时能分辨「文件缺席」与「解析失败」）。

    Returns:
        ``(是否开启, 警告列表)``——警告要指明「改哪一行、合法写法是什么」。
    """
    if data is None:
        if not SETTINGS_PATH.exists():
            return False, [f"{SETTINGS_PATH.name} 缺席 → 夜航按关（false）"]
        data = read_settings()
        if not data:
            return False, [f"{SETTINGS_PATH.name} 解析失败 → 夜航按关（false）"]
    try:
        raw = data.get("night", {}).get("enabled")
    except AttributeError:
        raw = None
    if raw is None:
        return False, [
            f"[night] enabled 未配置（在 {SETTINGS_PATH.name} 写 `[night] enabled = true` 即开）"
            " → 夜航按关（false）"
        ]
    if isinstance(raw, bool):
        return raw, []
    if isinstance(raw, int):
        if raw in (0, 1):
            return bool(raw), []
        return False, [
            f"[night] enabled = {raw!r} 不是合法开关（合法：true/false）→ 回落 false"
        ]
    text = str(raw).strip().lower()
    if text in _NIGHT_TRUE:
        return True, []
    if text in _NIGHT_FALSE:
        return False, []
    return False, [
        f"[night] enabled = {raw!r} 不是合法开关（合法：true/false）→ 回落 false"
    ]


#: ``[priority] order`` 的出厂优先级序（用户裁决 2026-10-13，backlog 20261013-193000-leader）：
#: 外层列表序＝优先级（前＝高），内层同组＝**同档**。列表项是**档名**（活跃会话注册表的
#: ``tier`` 字段），不是成员 id——「research」一档覆盖 7 专职 + reviewer + deputy。
#: 未登记档名一律取最低档：新档没进表就不该插队（fail-open，且方向是保守的）。
DEFAULT_PRIORITY_ORDER: tuple[tuple[str, ...], ...] = (
    ("leader",),
    ("research",),
    ("devops",),
    ("bench-pressure",),
    ("bencher", "twin"),
    ("inspector",),
)


def priority_order(data: dict[str, Any] | None = None) -> tuple[tuple[str, ...], ...]:
    """读 ``[priority] order``：外层序＝优先级、内层同档；缺席/非法 → 出厂表。

    值形是**数组的数组**（``[["leader"], ["bencher","twin"], …]``）——同档的档名写在同一个
    内层数组里，否则「bencher 与 twin 同档」这条用户裁决会被列表顺序偷偷拆成两档。
    单项数组也收（``["leader"]`` 与 ``"leader"`` 等价），免得写法收紧后旧配置突然失效。
    """
    if data is None:
        data = read_settings()
    try:
        raw = data.get("priority", {}).get("order")
    except AttributeError:
        return DEFAULT_PRIORITY_ORDER
    if not isinstance(raw, list) or not raw:
        return DEFAULT_PRIORITY_ORDER
    groups: list[tuple[str, ...]] = []
    for item in raw:
        names = item if isinstance(item, (list, tuple)) else [item]
        cleaned = tuple(str(x).strip() for x in names if str(x).strip())
        if cleaned:
            groups.append(cleaned)
    return tuple(groups) or DEFAULT_PRIORITY_ORDER


def priority_rank(tier: str, data: dict[str, Any] | None = None) -> int:
    """档名的优先级序（**小者高**）；未登记 → 末位（=最低档，不插队）。

    未登记取末位而不是 0（最高）是刻意的：档名拼错/新档没进表时，宁可不给调度权重，
    也不能让一个陌生名字压过 leader。
    """
    order = priority_order(data)
    key = str(tier or "").strip()
    for index, group in enumerate(order):
        if key in group:
            return index
    return len(order)


# ---------------------------------------------------------------------------
# [taskbooks]：大任务执行模式 + TUI 新建/驳回表单默认值（用户 2026-10-17 裁决）
# ---------------------------------------------------------------------------
#: ``[taskbooks] execution_mode`` 合法值。``parallel``＝多大任务并行交替（既有行为）；
#: ``serial``＝按任务书产生时序严格排队、一次只允许一个大任务进行（解锁点＝更早任务书
#: 离开 new/planning/active，即 delivered/failed/accepted/rejected）。
TASKBOOK_EXECUTION_MODES: tuple[str, ...] = ("parallel", "serial")
DEFAULT_TASKBOOK_EXECUTION_MODE: str = "parallel"

#: 大任务优先级合法值（settings 是叶子模块，不 import taskbooks；与 taskbooks.PRIORITIES 同字面）。
TASKBOOK_PRIORITIES: tuple[str, ...] = ("high", "normal", "low")
DEFAULT_TASKBOOK_PRIORITY: str = "normal"


@dataclass(frozen=True)
class TaskbookFormDefaults:
    """TUI 新建/驳回面板的表单默认值（``[taskbooks]`` 逐键读数，非法回落出厂）。"""

    new_priority: str = DEFAULT_TASKBOOK_PRIORITY
    new_leader: bool = False
    reject_priority: str = DEFAULT_TASKBOOK_PRIORITY
    reject_infra: bool = True
    reject_leader: bool = False
    reject_inspector: bool = True


#: 出厂默认（≡ 旧驳回默认 full 档：基础设施改进＋监察改进、无组长改进）。
DEFAULT_TASKBOOK_FORM: TaskbookFormDefaults = TaskbookFormDefaults()


def taskbook_execution_mode(
    data: dict[str, Any] | None = None,
) -> tuple[str, list[str]]:
    """``[taskbooks] execution_mode`` 大任务执行模式。

    **热切换**：本读数每次现算（无缓存），手改 toml 保存即生效于下一次派发 / 组长
    下一回合首个命令——串行谓词无状态，不存在在途锁定。

    Args:
        data: 已读到的配置；``None`` 时自己读盘。

    Returns:
        ``(模式, 警告列表)``。键缺席 / 值非法 → 出厂 ``parallel`` + 警告（回落方向保守：
        默认即既有并行行为，配置写坏也不会锁死队列）。
    """
    if data is None:
        data = read_settings()
    raw = None
    try:
        raw = data.get("taskbooks", {}).get("execution_mode")
    except AttributeError:
        raw = None
    if raw is None:
        return DEFAULT_TASKBOOK_EXECUTION_MODE, [
            f"[taskbooks] execution_mode 未配置（或 {SETTINGS_PATH.name} 读不出）"
            f" → 出厂 {DEFAULT_TASKBOOK_EXECUTION_MODE!r}"
        ]
    text = str(raw).strip()
    if text in TASKBOOK_EXECUTION_MODES:
        return text, []
    return DEFAULT_TASKBOOK_EXECUTION_MODE, [
        f"[taskbooks] execution_mode = {raw!r} 不是合法模式"
        f"（合法：{'/'.join(TASKBOOK_EXECUTION_MODES)}）"
        f" → 回落 {DEFAULT_TASKBOOK_EXECUTION_MODE!r}"
    ]


def _bool_or_none(raw: Any) -> bool | None:
    """宽容布尔解析（与 night_enabled 同精神）：bool 原值；0/1；宽容真值表；其余 None。"""
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, int):
        return bool(raw) if raw in (0, 1) else None
    text = str(raw).strip().lower()
    if text in _NIGHT_TRUE:
        return True
    if text in _NIGHT_FALSE:
        return False
    return None


def taskbook_form_defaults(
    data: dict[str, Any] | None = None,
) -> tuple[TaskbookFormDefaults, list[str]]:
    """``[taskbooks]`` 六个表单默认键（TUI 新建/驳回面板装载）。

    Returns:
        ``(默认值集, 警告列表)``——逐键回落出厂 + 逐键警告（警告指明合法值）。
    """
    if data is None:
        data = read_settings()
    try:
        section = data.get("taskbooks", {})
    except AttributeError:
        section = {}

    def _priority(key: str, default: str) -> tuple[str, str | None]:
        raw = section.get(key)
        if raw is None:
            return default, f"[taskbooks] {key} 未配置 → 回落 {default!r}"
        text = str(raw).strip()
        if text in TASKBOOK_PRIORITIES:
            return text, None
        return (
            default,
            f"[taskbooks] {key} = {raw!r} 不是合法优先级"
            f"（合法：{'/'.join(TASKBOOK_PRIORITIES)}）→ 回落 {default!r}",
        )

    def _flag(key: str, default: bool) -> tuple[bool, str | None]:
        raw = section.get(key)
        if raw is None:
            return default, f"[taskbooks] {key} 未配置 → 回落 {default!r}"
        parsed = _bool_or_none(raw)
        if parsed is None:
            return default, (
                f"[taskbooks] {key} = {raw!r} 不是合法开关（合法：true/false）"
                f" → 回落 {default!r}"
            )
        return parsed, None

    new_priority, w1 = _priority(
        "new_default_priority", DEFAULT_TASKBOOK_FORM.new_priority
    )
    new_leader, w2 = _flag("new_default_leader", DEFAULT_TASKBOOK_FORM.new_leader)
    rej_priority, w3 = _priority(
        "reject_default_priority", DEFAULT_TASKBOOK_FORM.reject_priority
    )
    rej_infra, w4 = _flag("reject_default_infra", DEFAULT_TASKBOOK_FORM.reject_infra)
    rej_leader, w5 = _flag("reject_default_leader", DEFAULT_TASKBOOK_FORM.reject_leader)
    rej_inspector, w6 = _flag(
        "reject_default_inspector", DEFAULT_TASKBOOK_FORM.reject_inspector
    )
    warnings = [w for w in (w1, w2, w3, w4, w5, w6) if w]
    return (
        TaskbookFormDefaults(
            new_priority=new_priority,
            new_leader=new_leader,
            reject_priority=rej_priority,
            reject_infra=rej_infra,
            reject_leader=rej_leader,
            reject_inspector=rej_inspector,
        ),
        warnings,
    )


def _detect_eol(text: str) -> str:
    """取文件主导换行（新插入的行必须与既有行同制式，否则整文件被改脏）。"""
    match = re.search(r"\r\n|\n", text)
    return match.group(0) if match else "\n"


def set_inspector_enabled(on: bool, path: Path | None = None) -> str:
    """外科式改写 ``[inspector] enabled``，返回写入的字面值（``"true"``/``"false"``）。

    只动目标行：定位 ``[inspector]`` 段 → 替换段内首个 ``enabled =`` 行的值（行尾注释
    保留）；段在而键缺席 → 在段首后插入一行；段本身缺席 → 文件尾追加新段。文件读写
    异常向上抛，由调用方决定回落面（``inspectd`` 回落 legacy 开关文件）。
    """
    target = path or SETTINGS_PATH
    value = "true" if on else "false"
    with target.open("r", encoding="utf-8", newline="") as fh:
        text = fh.read()
    eol = _detect_eol(text)
    lines = text.splitlines(keepends=True)

    section = -1
    for i, line in enumerate(lines):
        if line.lstrip().startswith("["):
            head = line.split("#", 1)[0].strip()
            if head == "[inspector]":
                section = i
                break

    if section == -1:
        # 段缺席：文件尾补一段（保持与既有内容空一行的排版）。
        if text and not text.endswith(("\n", "\r")):
            lines.append(eol)
        if lines and lines[-1].strip():
            lines.append(eol)
        lines.append(f"[inspector]{eol}")
        lines.append(f"enabled = {value}{eol}")
        section = len(lines) - 2
    else:
        end = len(lines)
        for j in range(section + 1, len(lines)):
            if lines[j].lstrip().startswith("["):
                end = j
                break
        for j in range(section + 1, end):
            stripped = lines[j].lstrip()
            if stripped.startswith("#") or not stripped:
                continue
            if re.match(r"^enabled\s*=", stripped):
                indent = lines[j][: len(lines[j]) - len(stripped)]
                key = stripped.split("#", 1)[0].split("=", 1)[0].rstrip()
                tail = stripped.split("#", 1)
                comment = (
                    f"  # {tail[1].strip()}"
                    if len(tail) > 1 and tail[1].strip()
                    else ""
                )
                lines[j] = f"{indent}{key} = {value}{comment}{eol}"
                break
        else:
            # 段内无 enabled 键：紧接段头插入（头行的行尾注释不动）。
            lines.insert(section + 1, f"enabled = {value}{eol}")

    with target.open("w", encoding="utf-8", newline="") as fh:
        fh.write("".join(lines))
    return value
