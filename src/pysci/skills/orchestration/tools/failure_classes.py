"""失败分类器：bandwidth / outage / quota / model_glitch / unknown（用户设计 2026-10-13）。

**为什么需要它**（backlog 20261013-180000-leader）：所有提供商的并发带宽限制与项目需求
同量级（mimo 明确 10 RPM / 10M TPM），「带宽不足」与「渠道真不可用」必须分开——前者
正确处置是**等**（延期），后者才是换席/停链。而区分二者只能看 provider 报错原文，故本单
同时把原文入台账（``error_text``）并给出分类器。

**model_glitch**（backlog 20261016-042500-devops）：mid-run 工具回合中断（模型发出畸形/
未名的工具调用、流式截断）——stop_reason 落在 ``tool_use``/``stop_sequence`` 而原文常为空。
席位与渠道都活着，正确处置是**重试一次**（resume 续跑），故必须与上面三类分开。

关键词表落 :data:`CLASSES_PATH`（``state/failure-classes.json``，git 跟踪），**文件形状
即协议**，纪律同 :mod:`failure_tags`：增补 pattern 须附定义（``definition`` 非空）。
dispatch 的 quota/channel 识别与后续延期机制都只从这里取词，不在别处硬编码。

分类优先级 :data:`PRECEDENCE`：额度耗尽要充值、带宽不足只要等——两条同现时取更硬的那个。

**本模块只懂分类**：不写台账、不改派发策略；调用方拿 ``str`` 自己决定后果（drain 的
延期/回退策略在 backlog 20261013-180500-leader，不在本模块）。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from pysci.paths import ORCH_STATE_ROOT

CLASSES_PATH: Path = ORCH_STATE_ROOT / "failure-classes.json"

#: 无匹配时的类名（不是「第四类」，是「判不出」——调用方必须自己决定保守方向）。
UNKNOWN: str = "unknown"

#: 词表缺席/损坏时的兜底 pattern（只保**能跑**，真词在文件里）。
#: 与随附文件的一致性由 ``test_failure_classes.py`` 钉住（改一边必改另一边）；
#: ``quota`` 三条与原 ``dispatch.QUOTA_ERROR_RE`` 逐字一致——重构不改行为。
DEFAULT_CLASSES: dict[str, tuple[str, ...]] = {
    "quota": (r"credit usage limit", r"out of credits?", r"insufficient credits?"),
    "bandwidth": (
        r"custom model service error",
        r"please retry later",
        r"retry later or switch to another model",
        r"please retry",
        r"rate[ _-]?limit",
        r"too many requests",
        r"concurrency",
        r"\b429\b",
        r"\bTPM\b",
        r"\bRPM\b",
        r"overload(ed)?",
        r"capacity",
    ),
    "outage": (
        r"unauthorized",
        r"invalid api key",
        r"authentication failed",
        r"model not found",
        r"unknown model",
        r"does not exist",
        r"no such model",
        r"connection refused",
        r"connection reset",
        r"bad gateway",
        r"service unavailable",
        r"gateway timeout",
        r"internal server error",
        r"\b(500|502|503|504)\b",
    ),
    "model_glitch": (r"\btool_use\b", r"\bstop_sequence\b"),
}

#: 判定优先级（同现时取**更硬**者：额度耗尽须充值、带宽只要等）。文件可覆盖。
#: ``model_glitch`` 排最后：它是「模型侧打嗝」（mid-run 工具回合中断），原文通常是
#: stop_reason 短码本身——只有没有更硬的 provider 报错在场时才归它。
PRECEDENCE: tuple[str, ...] = ("quota", "bandwidth", "outage", "model_glitch")

#: 编译缓存：键 = 该类当前 patterns 元组（词表改了就自动重编，不必重启）。
_RE_CACHE: dict[tuple[str, ...], re.Pattern[str]] = {}


def load_classes(path: Path | None = None) -> dict[str, tuple[str, ...]]:
    """读词表为 ``{类: (pattern, …)}``；缺失/损坏 → 回落 :data:`DEFAULT_CLASSES`。

    逐键校验：只要文件给出**非空 pattern 列表**就用它，否则该类回落默认（一个手写的
    配置文件不该让分类器失效，但也绝不回落成空表——那会把「一切都不匹配」伪装成正常）。
    """
    p = CLASSES_PATH if path is None else path
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return dict(DEFAULT_CLASSES)
    classes = raw.get("classes") if isinstance(raw, dict) else None
    if not isinstance(classes, dict):
        return dict(DEFAULT_CLASSES)
    out = dict(DEFAULT_CLASSES)
    for name, spec in classes.items():
        patterns = spec.get("patterns") if isinstance(spec, dict) else None
        if isinstance(patterns, list) and patterns:
            cleaned = tuple(str(x) for x in patterns if str(x).strip())
            if cleaned:
                out[str(name)] = cleaned
    return out


def precedence(path: Path | None = None) -> tuple[str, ...]:
    """判定优先级（文件 ``precedence`` 可覆盖；缺省 :data:`PRECEDENCE`）。"""
    p = CLASSES_PATH if path is None else path
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return PRECEDENCE
    order = raw.get("precedence") if isinstance(raw, dict) else None
    if isinstance(order, list) and order:
        cleaned = tuple(str(x) for x in order if str(x).strip())
        if cleaned:
            return cleaned
    return PRECEDENCE


def combined_re(
    cls: str, *, path: Path | None = None, extra: tuple[str, ...] = ()
) -> re.Pattern[str]:
    """某类的合并正则（缓存；``extra`` 追加 pattern，供联合视图复用缓存键）。"""
    patterns = tuple(load_classes(path).get(cls, ())) + tuple(extra)
    cached = _RE_CACHE.get(patterns)
    if cached is None:
        cached = re.compile("|".join(patterns), re.IGNORECASE)
        _RE_CACHE[patterns] = cached
    return cached


def provider_error_re(*, path: Path | None = None) -> re.Pattern[str]:
    """「**任何** provider 侧报错签名」的联合正则（带宽 ∪ 故障）。

    dispatch 的 channel_error 文本支用它（原先只认 mimo 那一句固定文案）。**这是有意的
    放宽**：真正把「渠道级故障」与「组员自己复述了错误串」分开的是形状支（会话档末轮
    ``<synthetic>`` + 空 stop_sequence），文本支只需回答「像不像 provider 报错」。放宽后
    rate-limit / 429 / 5xx 这些同样算渠道级异常——正是带宽治理要接的面。
    """
    classes = load_classes(path)
    return combined_re("bandwidth", path=path, extra=tuple(classes.get("outage", ())))


def classify_failure(
    error_text: str, stop_reason: str = "", is_error: bool = True
) -> str:
    """把失败跳的原文判成 ``bandwidth`` / ``outage`` / ``quota`` / ``unknown``。

    Args:
        error_text: provider 报错原文（台账 ``error_text`` 同源；envelope result / stderr）。
        stop_reason: 失败跳的 envelope stop_reason（原文缺席时的次级线索）。
        is_error: 该跳是否确为失败（**缺省 True**：本函数的名字就是「分类失败」）。
            传 False = 「这跳成功了，只是想拿这段文字问问」→ 一律 :data:`UNKNOWN`。
            为什么要有这个闸：成功交付的正文可能只是**提及**这些词（复盘渠道故障的任务书
            就是活样本），把成功判成故障会污染统计与后续策略。

    Returns:
        类名（:data:`UNKNOWN` 表示无匹配）。
    """
    if not is_error:
        return UNKNOWN
    blob = (error_text or "").strip() or (stop_reason or "").strip()
    if not blob:
        return UNKNOWN
    for cls in precedence():
        if combined_re(cls).search(blob):
            return cls
    return UNKNOWN
