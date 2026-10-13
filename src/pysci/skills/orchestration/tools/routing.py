"""谷时隐性 DeepSeek 路由：时间驱动的预设覆盖（用户设计 2026-10-13，backlog 20261013-093000-leader）。

官方 DeepSeek 定价分峰谷（峰时 2 倍价）。用户的裁决是：**谷时段自动落到 DeepSeek 席**，
而且「隐性」——``data/pysci_settings.toml`` 的 ``[model] preset`` 那一行**不动**、不新增
第四个预设，只在谷时把 qwen/mimo 预设的**主力席（main）**与仲裁团面板换血，到峰时自动
切回。升档席（pro）成员**不受覆盖**（用户终裁 2026-10-13：隐性态下 pro 席保持原席，
提高工作质量）——被时间换血的只有普通成员那一档。

时段口径（多源核实 2026-08/09，权威抄录见任务书证据 ``state/suggestions/approved/``）：

- **峰** = 工作日（周一至周五）9:00–12:00、14:00–18:00；
- **谷** = ① 周六日**全天**（调休上班的周六日照样算谷：按星期几判定，``is_in_lieu`` 不参与）；
  ② 法定节假日**全天**（哪怕落在工作日）；③ 工作日其余时段（含 12:00–14:00 午休与夜间）。

两条不可省的机制：

- **早切**（进峰提前 :data:`DEFAULT_LEADTIME_S`）：一跳一旦起跑就不能中途换模型，所以
  「什么时候切」按**这一跳会不会跨过峰界**决定，而不是按此刻跨没跨——谷段末尾那一小时
  已经切回原预设，宁可在峰时多烧一会儿，也不要一跳烧两个渠道；
- **保守不切换**：节假日库（``chinesecalendar``）只内嵌 2004–2026，读不出的日期**一律按
  原预设走**并出声告警（mirror/``orch status``/派发面同一条文案）——宁可多付峰时价，
  也不凭猜测切渠道。

时钟口径：按用户指示用**本机时间**判定（不联网校时），读数一律折到北京时间
（:data:`CN_TZ_NAME`）；机器时区不是 UTC+8 时归一换算并注记一句。

配置面（全部可选，非法值回落出厂并警告，读侧 fail-open）：``[routing]`` 块在
``data/pysci_settings.toml``——``enabled`` / ``valley_preset`` / ``leadtime_s`` /
``peak`` / ``jury_panel``。席位本身不在这里：谷时主席取 ``valley_preset`` 的席位链
（跟踪 ``registry.json presets.<name>`` 整链覆盖 ← 出厂表），仲裁团席位取
``registry.jury.seats``（三席一律**按量付费**直调，用户终裁 2026-10-13：TokenPlan 套餐
路径整个不做——两家个人版条款禁止 API 自动化调用，违规封 Key）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from pysci.paths import ORCH_STATE_ROOT

from . import presets, settings

#: 判定口径的时区名（北京时间；机器时区不同时以它归一换算）。
CN_TZ_NAME: str = "Asia/Shanghai"

#: 峰时段出厂口径（半开区间 ``[起, 止)``，小时）。谷=其余全部时刻。
DEFAULT_PEAK_WINDOWS: tuple[tuple[int, int], ...] = ((9, 12), (14, 18))

#: 进峰前提前切换的秒数（见模块 docstring「早切」）。
DEFAULT_LEADTIME_S: int = 3600

#: 谷时主席来自哪个预设的席位表。qwen/mimo 是「被覆盖方」，deepseek 是「覆盖来源」——
#: 显式选了 deepseek 预设时不存在覆盖（本来就是它），隐性态自然不生效。
DEFAULT_VALLEY_PRESET: str = "deepseek"

#: 隐性态下仲裁团的**整副面板**（用户终裁 v4 2026-10-13：隐性态=GLM+KIMI+Mimo，全按量）。
#: 与 :func:`jury_panel` 同名，是因为这一串就是「谷时那三席是谁」的唯一答案——前两席
#: 跨预设固定，第三席也固定（不随配置预设漂），故不再拆成「前两席 + 本预设旗舰」。
DEFAULT_JURY_PANEL: tuple[str, ...] = ("glm", "kimi", "mimo")

#: 向后找切换点的最大天数（覆盖最长黄金周+调休；超出即按「判不出」保守不切）。
LOOKAHEAD_DAYS: int = 45

#: 上次读数留痕（gitignored 运行态）。切换事件只有**跨进程**比较才看得见：drain 一跳
#: 一进程，18:00 的入谷往往发生在两跳之间，进程内变量必然漏记，故落盘比对这个事实。
ROUTING_STATE_PATH: Path = ORCH_STATE_ROOT / "routing-last.json"

#: 决策日志的 kind 值（``journal`` 的 kind 是约定不是枚举，新增一词只在这一处声明）。
JOURNAL_KIND: str = "routing"

#: 星期几中文名（判据读数用，避免「weekday=5」这种没人读的串）。
_WEEKDAYS: tuple[str, ...] = ("周一", "周二", "周三", "周四", "周五", "周六", "周日")


@dataclass(frozen=True)
class Face:
    """``[routing]`` 块的**已校验**读数（出厂值 + 警告，绝不为非法配置停摆）。"""

    enabled: bool = True
    valley_preset: str = DEFAULT_VALLEY_PRESET
    leadtime_s: int = DEFAULT_LEADTIME_S
    peak: tuple[tuple[int, int], ...] = DEFAULT_PEAK_WINDOWS
    jury_panel: tuple[str, ...] = DEFAULT_JURY_PANEL
    warnings: tuple[str, ...] = ()

    def windows_text(self) -> str:
        """峰段的人类可读串（告警与「已关闭」读数里都要它，别两处各拼一遍）。"""
        return "、".join(f"{a:02d}:00–{b:02d}:00" for a, b in self.peak)


def _zone() -> ZoneInfo | None:
    """北京时区的 tzinfo；tzdata 缺席时返回 ``None``（调用方回落本机时间并警告）。"""
    try:
        return ZoneInfo(CN_TZ_NAME)
    except Exception:  # noqa: BLE001 — 平台缺 tzdata 属环境事实，不崩派发
        return None


def now_cn() -> datetime:
    """此刻的北京时间读数（本机时钟，不联网校时；非 UTC+8 的机器按 :data:`CN_TZ_NAME` 归一）。"""
    local = datetime.now().astimezone()
    tz = _zone()
    return local.astimezone(tz) if tz is not None else local


def as_cn(value: datetime | None) -> datetime:
    """把给定时刻折到北京时间（naive 按北京时间理解——测试可直接写 ``datetime(2026, 10, 17, 12, 30)``）。"""
    moment = now_cn() if value is None else value
    tz = _zone()
    if moment.tzinfo is None:
        return moment.replace(tzinfo=tz) if tz is not None else moment.astimezone()
    return moment.astimezone(tz) if tz is not None else moment.astimezone()


def tz_note() -> str:
    """机器时区注记（非 UTC+8 时说明「已归一」，否则读数会被当成机器本地时段）。"""
    offset = datetime.now().astimezone().utcoffset()
    if offset is None:
        return "本机时区读数不可用（按系统默认判定）"
    hours = offset.total_seconds() / 3600
    return (
        ""
        if hours == 8
        else f"本机时区 UTC{hours:+.1f} 非 +8 → 判据已按 {CN_TZ_NAME} 归一"
    )


def _valid_pair(item: Any) -> tuple[int, int] | None:
    try:
        start, end = int(item[0]), int(item[1])  # type: ignore[index]
    except (TypeError, ValueError, IndexError, KeyError):
        return None
    return (start, end) if 0 <= start < end <= 24 else None


def face(data: dict[str, Any] | None = None) -> Face:
    """读 ``[routing]`` 块并逐键校验（非法键回落出厂值 + 一条警告，不影响其余键）。

    读侧 fail-open 与 :func:`.settings.preset_name` 同口径：一个手写的配置文件不该能把
    派发面卡死，但**回落必须出声**——静默回落等于「用户改了却没生效」。
    """
    if data is None:
        data = settings.read_settings()
    raw = data.get("routing") if isinstance(data, dict) else None
    if raw is None:
        return Face()
    if not isinstance(raw, dict):
        return Face(
            warnings=(f"[routing] 不是表（收到 {type(raw).__name__}）→ 全部回落出厂值",)
        )
    warn: list[str] = []

    enabled = raw.get("enabled", True)
    if not isinstance(enabled, bool):
        warn.append(f"[routing] enabled = {raw.get('enabled')!r} 非布尔 → 按 true")
        enabled = True

    valley = str(raw.get("valley_preset", DEFAULT_VALLEY_PRESET) or "").strip()
    if valley not in presets.PRESETS:
        warn.append(
            f"[routing] valley_preset = {valley!r} 不是合法预设（"
            f"{'/'.join(presets.PRESETS)}）→ 回落 {DEFAULT_VALLEY_PRESET!r}"
        )
        valley = DEFAULT_VALLEY_PRESET

    leadtime = DEFAULT_LEADTIME_S
    if raw.get("leadtime_s") is not None:
        try:
            number = int(raw["leadtime_s"])
            if not 0 <= number <= 86400:
                raise ValueError
            leadtime = number
        except (TypeError, ValueError):
            warn.append(
                f"[routing] leadtime_s = {raw.get('leadtime_s')!r} 非法（0–86400）"
                f"→ 回落 {DEFAULT_LEADTIME_S}"
            )

    peak = DEFAULT_PEAK_WINDOWS
    if raw.get("peak") is not None:
        parsed: list[tuple[int, int]] = []
        for item in raw.get("peak") or []:
            pair = _valid_pair(item)
            if pair is None:
                warn.append(
                    f"[routing] peak 条目 {item!r} 非法（须 [起, 止] 且 0≤起<止≤24）"
                    "→ 丢弃该条"
                )
                continue
            parsed.append(pair)
        if parsed:
            peak = tuple(sorted(parsed))
        else:
            warn.append(
                f"[routing] peak = {raw.get('peak')!r} 无合法区间 → 回落出厂峰段"
            )
            peak = DEFAULT_PEAK_WINDOWS

    panel = DEFAULT_JURY_PANEL
    if raw.get("jury_panel") is not None:
        items = raw.get("jury_panel")
        if (
            isinstance(items, (list, tuple))
            and items
            and all(str(x).strip() for x in items)
        ):
            panel = tuple(str(x).strip() for x in items)
        else:
            warn.append(
                f"[routing] jury_panel = {items!r} 非法（须家族名列表）→ 回落 "
                f"{'+'.join(DEFAULT_JURY_PANEL)}"
            )

    return Face(
        enabled=bool(enabled),
        valley_preset=valley,
        leadtime_s=int(leadtime),
        peak=peak,
        jury_panel=panel,
        warnings=tuple(warn),
    )


# ---------------------------------------------------------------------------
# 日期与时段判定（纯函数：给时刻出面额，不读盘不猜）
# ---------------------------------------------------------------------------
def holiday_off(day: date) -> tuple[bool, str, str]:
    """该日是否**法定节假日**：``(放假, 节日名, 警告)``。

    库缺席 / 年份不覆盖（``chinesecalendar`` 内嵌 2004–2026）/ 任何读数异常 →
    ``(False, "", 警告)``：保守按「不是假日」处理，也就是**不切换**，由调用方出声。
    """
    try:
        # 延迟导入：库缺席不许拖垮 registry / 派发
        from chinese_calendar import get_holiday_detail
    except ImportError:
        return (
            False,
            "",
            "节假日库 chinesecalendar 未安装 → 保守不切换（改 pyproject 后 pysci-dev venv-sync）",
        )
    try:
        off, name = get_holiday_detail(day)
    except NotImplementedError as exc:
        return False, "", f"节假日库不覆盖 {day.year} 年（{exc}）→ 保守不切换"
    except Exception as exc:  # noqa: BLE001 — 判定失败不能挡住派发
        return False, "", f"节假日库读数失败（{type(exc).__name__}: {exc}）→ 保守不切换"
    return bool(off), str(name or ""), ""


def day_reading(day: date, face_value: Face) -> tuple[str, str, str]:
    """该日的**日型**：``(态, 判据读数, 警告)``，态 ∈ ``valley|peakable|unknown``。

    ``valley``=全天谷（周末或法定节假日）；``peakable``=工作日，峰段按 :attr:`Face.peak`；
    ``unknown``=判不出（节假日库不覆盖该日期）→ 调用方按保守不切换处理。
    **周末优先**：调休上班的周六日照样全天谷（用户裁决：按星期几判定，``is_in_lieu`` 不参与）。
    """
    weekday = day.weekday()
    if weekday >= 5:
        return "valley", f"{_WEEKDAYS[weekday]}（周末全天）", ""
    off, name, warn = holiday_off(day)
    if warn:
        return "unknown", "工作日（节假日判不出）", warn
    if off:
        return "valley", f"法定节假日：{name or '未命名'}（全天）", ""
    return "peakable", f"工作日（{_WEEKDAYS[weekday]}）", ""


def peak_slots(day: date, face_value: Face) -> list[tuple[datetime, datetime]]:
    """该日的峰段区间（北京时间的 aware datetime）；谷全日型/判不出的日期返回空表。"""
    kind, _, _ = day_reading(day, face_value)
    if kind != "peakable":
        return []
    tz = _zone()
    if tz is None:
        tz = datetime.now().astimezone().tzinfo
    return [
        (
            datetime.combine(day, time(start_h), tzinfo=tz),
            datetime.combine(day, time(end_h), tzinfo=tz),
        )
        for start_h, end_h in face_value.peak
    ]


def next_peak_start(moment: datetime, face_value: Face) -> datetime | None:
    """此刻之后最近的峰段**起点**（跨日向后找，最多 :data:`LOOKAHEAD_DAYS` 天）。

    向后找不到（节假日库把这几天全判成 unknown，且中间没有工作日）→ ``None``；
    调用方按保守处理——「判不出下次峰」等于「不敢切」。
    """
    day = moment.date()
    for _ in range(LOOKAHEAD_DAYS + 1):
        for start, _end in peak_slots(day, face_value):
            if start > moment:
                return start
        day = day + timedelta(days=1)
    return None


def switch_moments(day: date, face_value: Face) -> list[tuple[datetime, str]]:
    """该日的路由切换时刻 ``(时刻, 方向)``，方向 ∈ ``入峰|入谷``。

    入峰=峰段起点**减**提前量（早切）；入谷=峰段终点（即时）。谷全日型日期没有切换点：
    它前一晚 18:00 起就已是谷，跨零点不产生状态变化——这就是任务书把「周末/节假日起点」
    列为切换时机、而实现里却不出现这一条的原因（状态机在周五 18:00 已经切过了）。
    """
    out: list[tuple[datetime, str]] = []
    for start, end in peak_slots(day, face_value):
        out.append((start - timedelta(seconds=face_value.leadtime_s), "入峰"))
        out.append((end, "入谷"))
    return sorted(out)


def next_switch(
    moment: datetime, face_value: Face
) -> tuple[datetime | None, str, tuple[str, ...]]:
    """下一次切换的 ``(时刻, 方向, 警告)``；找不到 → 时刻为 ``None`` 并带原因警告。"""
    warn: list[str] = []
    day = moment.date()
    for _ in range(LOOKAHEAD_DAYS + 1):
        kind, _, reading_warn = day_reading(day, face_value)
        if reading_warn:
            warn.append(reading_warn)
            break
        if kind == "peakable":
            for when, mode in switch_moments(day, face_value):
                if when > moment:
                    return when, mode, tuple(warn)
        day = day + timedelta(days=1)
    return (
        None,
        "",
        tuple(warn) or (f"{LOOKAHEAD_DAYS} 天内无切换时刻（配置或库异常）",),
    )


# ---------------------------------------------------------------------------
# 读数面
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Routing:
    """一次路由读数（``orch status`` 路由段、mirror 概览、派发日志共读同一个对象）。"""

    #: 读数时刻（北京时间，分钟级）
    at: str = ""
    #: 隐性 DeepSeek 态是否生效
    hidden: bool = False
    #: toml 里配置的预设（覆盖的**被覆盖方**）
    preset: str = ""
    #: 谷时主席来自哪个预设
    valley_preset: str = DEFAULT_VALLEY_PRESET
    #: 判据读数（周末 / 节假日 / 时段，含峰段边界与提前量）
    basis: str = ""
    #: 下次切换时刻（'' = 判不出）
    next_at: str = ""
    #: 下次切换方向（入谷 / 入峰）
    next_mode: str = ""
    #: 提前切换量（展示用）
    leadtime_s: int = DEFAULT_LEADTIME_S
    #: 出厂峰段（展示用，让读数自证判据）
    peak: tuple[tuple[int, int], ...] = DEFAULT_PEAK_WINDOWS
    #: 隐性态仲裁团面板（展示与 :func:`jury_panel` 共用同一串）
    jury_panel: tuple[str, ...] = DEFAULT_JURY_PANEL
    #: 警告（非法配置、库不覆盖、时区归一）
    warnings: tuple[str, ...] = ()

    @property
    def mode(self) -> str:
        """生效路由的人类名（``隐性 DeepSeek`` 或预设名）。"""
        return "隐性 DeepSeek" if self.hidden else self.preset

    @property
    def state_key(self) -> str:
        """状态标识（跨进程比对切换用；只有这个串变了才算「切了一次」）。"""
        return f"{self.valley_preset}-隐性" if self.hidden else f"{self.preset}-原样"

    def one_line(self) -> str:
        """一行读数（mirror 概览项用；``[路由]`` 前缀由 mirror 加）。"""
        nxt = (
            f"｜下次 {self.next_at} {self.next_mode}"
            if self.next_at
            else "｜下次切换判不出（保守维持）"
        )
        warn = "".join(f" [!] {w}" for w in self.warnings)
        return f"〔{self.mode}〕 {self.basis}{nxt}{warn}"


def state(
    *, moment: datetime | None = None, data: dict[str, Any] | None = None
) -> Routing:
    """当前路由读数（谷/峰判定 + 下次切换预告 + 警告）。

    Args:
        moment: 注入的判定时刻（单测/复算用）；缺省取 :func:`now_cn`。
        data: 注入的已读配置（缺省读盘）——同时喂给 ``[routing]`` 与 ``[model] preset``，
            两者本就在同一份文件里，分开注入会出现「半新半旧」的假状态。
    """
    face_value = face(data)
    act = presets.active(data)
    at = as_cn(moment)
    warn: list[str] = list(face_value.warnings)
    if data is None:
        note = tz_note()
        if note:
            warn.append(note)
    stamp = at.strftime("%Y-%m-%d %H:%M")
    common: dict[str, Any] = {
        "at": stamp,
        "preset": act.name,
        "valley_preset": face_value.valley_preset,
        "leadtime_s": face_value.leadtime_s,
        "peak": face_value.peak,
        "jury_panel": face_value.jury_panel,
    }
    if not face_value.enabled:
        return Routing(
            **common,
            hidden=False,
            basis=(
                f"[routing] enabled=false → 峰谷判定已关闭"
                f"（出厂口径峰段 {face_value.windows_text()}）"
            ),
            warnings=tuple(dict.fromkeys(warn)),
        )
    kind, day_basis, reading_warn = day_reading(at.date(), face_value)
    if reading_warn:
        warn.append(reading_warn)
    if kind == "unknown":
        return Routing(
            **common,
            hidden=False,
            basis=f"{day_basis} → 保守沿用原预设",
            warnings=tuple(dict.fromkeys(warn)),
        )
    slot = next(
        ((s, e) for s, e in peak_slots(at.date(), face_value) if s <= at < e), None
    )
    if slot is not None:
        hidden = False
        basis = f"{day_basis} 峰段 {slot[0]:%H:%M}–{slot[1]:%H:%M}（2 倍价）"
    else:
        nxt_peak = next_peak_start(at, face_value)
        if nxt_peak is None:
            hidden = False
            basis = f"{day_basis} 谷段（向后判不出峰段 → 保守沿用原预设）"
            warn.append(f"{LOOKAHEAD_DAYS} 天内找不到峰段起点 → 维持原预设")
        else:
            gap = (nxt_peak - at).total_seconds()
            if gap <= face_value.leadtime_s:
                hidden = False
                basis = (
                    f"{day_basis} 谷段，距峰 {nxt_peak:%m-%d %H:%M} 不足提前量 "
                    f"{face_value.leadtime_s // 60} 分钟 → 已提前切回"
                )
            else:
                hidden = True
                basis = f"{day_basis} 谷段（距峰 {nxt_peak:%m-%d %H:%M} 还有 {gap / 3600:.1f} 小时）"
    when, mode, switch_warn = next_switch(at, face_value)
    warn.extend(w for w in switch_warn if w not in warn)
    if hidden and act.name == face_value.valley_preset:
        hidden = False
        basis = f"{basis}｜预设已是 {act.name}（无需覆盖，席位链本来就走这条）"
    return Routing(
        **common,
        hidden=hidden,
        basis=basis,
        next_at=when.strftime("%m-%d %H:%M") if when else "",
        next_mode=mode,
        warnings=tuple(dict.fromkeys(warn)),
    )


def hidden_valley_preset(
    *, moment: datetime | None = None, data: dict[str, Any] | None = None
) -> str:
    """隐性态生效时返回**该落哪个预设的席位表**，否则 ``""``（派发面的唯一入口）。

    「显式 deepseek 预设」在这里返回空串（见 :func:`state` 末段）：那时席位链本来就走
    deepseek，把它说成一次覆盖会让日志虚报一次切换——用户自己选的预设，不是时间替用户做的决定。
    """
    reading = state(moment=moment, data=data)
    return reading.valley_preset if reading.hidden else ""


def producer_family(
    *, moment: datetime | None = None, data: dict[str, Any] | None = None
) -> str:
    """**被评模型**的家族（§4「不得同族自裁」的相对化锚点）。

    隐性态下主席是 DeepSeek，被排除的同族就是 deepseek——而 toml 那一行写着 qwen/mimo。
    硬拿 ``presets.active().name`` 当同族，会把「谷时请 Qwen 旗舰当第三席」误判成违规拒开庭。
    """
    reading = state(moment=moment, data=data)
    return reading.valley_preset if reading.hidden else reading.preset


def jury_panel(
    base_panel: tuple[str, ...] | list[str],
    *,
    moment: datetime | None = None,
    data: dict[str, Any] | None = None,
) -> tuple[str, ...]:
    """隐性态下的仲裁团**面板构成**覆盖：固定三席 :data:`DEFAULT_JURY_PANEL`。

    非隐性态原样返回 ``base_panel``（预设自带的三家族）。设计理由见 :func:`producer_family`：
    谷时被评的是 DeepSeek，庭里就不得再有 DeepSeek 席——glm+kimi 固定，第三席也固定成
    Mimo（用户终裁 v4 2026-10-13：隐性态→GLM+KIMI+Mimo，不随配置预设漂）。家族名即预设名
    （``qwen``/``mimo``/``deepseek`` 与 ``jury.FAMILY_TOKENS`` 同一套串），故直接取家族名，
    不经 ``jury.seat_family``——那一层会引 jury↔routing 环。
    """
    reading = state(moment=moment, data=data)
    return tuple(reading.jury_panel) if reading.hidden else tuple(base_panel)


def status_lines(*, moment: datetime | None = None) -> list[str]:
    """``orch status`` 的「路由模式」段（与 mirror 概览共读 :func:`state`，文案不另起炉灶）。"""
    reading = state(moment=moment)
    out = [
        "== 路由模式（谷时隐性 DeepSeek，真源 data/pysci_settings.toml [routing]）==",
        f"  配置预设 {reading.preset}｜生效路由 {reading.mode}"
        f"（谷时主席取 {reading.valley_preset} 席链）",
        f"  判据读数：{reading.basis}",
        (
            f"  下次切换：{reading.next_at} {reading.next_mode}"
            if reading.next_at
            else "  下次切换：判不出（保守维持当前路由）"
        ),
        f"  峰段口径 {Face(peak=reading.peak).windows_text()}"
        f"｜提前量 {reading.leadtime_s // 60} 分钟（单跳不可中途换模型）",
        f"  上次切换：{last_switch_line()}",
    ]
    if reading.hidden:
        out.append(
            f"  仲裁团面板：{'+'.join(reading.jury_panel)}"
            "（隐性态固定三席，全按量；被评模型是 DeepSeek，庭内不得有同族席）"
        )
    for warn in reading.warnings:
        out.append(f"[!] {warn}")
    return out


def summary_line(*, moment: datetime | None = None) -> str:
    """mirror 概览「路由」项的一行读数（单项失败由 mirror 的逐项容错兜住）。

    带上上次切换：mirror 是用户「一天看一次」的入口，只报此刻的状态就看不出「昨晚上
    切过没有」——而切换本身是要被审阅的事件（规格 4「切换记录入 mirror」）。
    """
    return f"{state(moment=moment).one_line()}｜上次切换 {last_switch_line()}"


# ---------------------------------------------------------------------------
# 切换留痕（决策日志 kind=routing + run 日志一行）
# ---------------------------------------------------------------------------
def _read_last(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return raw if isinstance(raw, dict) else {}


def _write_last(
    reading: Routing, path: Path, *, switched_at: str = "", from_mode: str = ""
) -> None:
    """落「上次读数」（gitignored 运行态）。

    ``switched_at``/``from_mode`` 只在**真的跨过一次切换**时写：立基线那次没有「切换」这
    件事，把读数时刻冒充成切换时刻，mirror 就会报出一场没发生过的切换。
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "mode": reading.state_key,
                    "at": reading.at,
                    "basis": reading.basis,
                    "switched_at": switched_at,
                    "from_mode": from_mode,
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
    except OSError:
        pass


def last_switch_line(path: Path | None = None) -> str:
    """上次切换的一句话读数（无记录 → 明说「尚无切换」，别让空面看起来像坏了）。"""
    raw = _read_last(path or ROUTING_STATE_PATH)
    if not raw:
        return "尚无切换记录（路由首次读数即立基线）"
    stamp = str(raw.get("switched_at") or "")
    if not stamp:
        return f"尚未跨过峰谷界（基线 {raw.get('mode', '?')} @ {raw.get('at', '?')}）"
    return f"{stamp} 由 {raw.get('from_mode', '?')} 切至 {raw.get('mode', '?')}"


def note_switch(*, run_log: Path | None = None, moment: datetime | None = None) -> str:
    """检测路由状态变化：变了就写决策日志（``kind="routing"``）并回一行日志文案。

    首次读数只立基线不出声（drain 一跳一新进程，否则每次开工都多一条日志进 git 跟踪面）。
    ``run_log`` 给了就把同一行也写进 run 日志；观察面写失败一律不炸主流程
    （同 :func:`.channels.append_event` 的 fail-open 红线）。
    """
    reading = state(moment=moment)
    path = ROUTING_STATE_PATH
    last = str(_read_last(path).get("mode", ""))
    if not last:
        _write_last(reading, path)
        return ""
    if last == reading.state_key:
        return ""
    line = (
        f"[i] 路由切换：{last} → {reading.state_key}（{reading.basis}；"
        f"下次切换 {reading.next_at or '判不出'} {reading.next_mode}）"
    )
    from . import journal as _journal

    # 决策日志是**观察面**：写不进去（磁盘/权限/仓库被只读挂载）只意味着这一条留痕缺失，
    # 不能把派发炸掉——但也不许因此把状态推进：读数面停在旧基线，下一跳会重试同一次切换，
    # 宁可重复出声也不静默吞掉一次跨越。
    try:
        _journal.append(
            {
                "ts": datetime.now().astimezone().isoformat(timespec="seconds"),
                "situation": f"DeepSeek 峰谷跨越（{reading.basis}）",
                "options": (
                    f"维持 {reading.preset} 席位链 / 落隐性 {reading.valley_preset} 席位链"
                ),
                "choice": line.replace("[i] 路由切换：", ""),
                "reason": (
                    f"配置预设 {reading.preset}，谷时主席取 {reading.valley_preset}，"
                    f"提前量 {reading.leadtime_s // 60} 分钟；只影响新派发，在途单跳不换模型"
                ),
                "source": "routing",
                "kind": JOURNAL_KIND,
            }
        )
    except OSError:
        return line
    _write_last(reading, path, switched_at=reading.at, from_mode=last)
    if run_log is not None:
        try:
            run_log.parent.mkdir(parents=True, exist_ok=True)
            with run_log.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except OSError:
            pass
    return line


__all__ = [
    "CN_TZ_NAME",
    "DEFAULT_JURY_PANEL",
    "DEFAULT_LEADTIME_S",
    "DEFAULT_PEAK_WINDOWS",
    "DEFAULT_VALLEY_PRESET",
    "JOURNAL_KIND",
    "LOOKAHEAD_DAYS",
    "ROUTING_STATE_PATH",
    "Face",
    "Routing",
    "as_cn",
    "day_reading",
    "face",
    "hidden_valley_preset",
    "holiday_off",
    "jury_panel",
    "last_switch_line",
    "next_peak_start",
    "next_switch",
    "note_switch",
    "now_cn",
    "peak_slots",
    "producer_family",
    "state",
    "status_lines",
    "summary_line",
    "switch_moments",
    "tz_note",
]
