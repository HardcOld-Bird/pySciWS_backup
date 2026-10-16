"""交付协议解析：从组员最终回复中提取 XML 标记块（README §3.4）。

协议：``<result>`` 与 ``<blocked>`` 互斥且必居其一（delivery-gate hook 已在成员侧
强制，本模块是第二道防线）；``<infra_suggestion>`` 可选；``<artifact>`` 内嵌于
result，声明机械验收类型；``<blocked>`` 可选 ``tags="a,b"`` 失败标签（受控词表见
:mod:`.failure_tags`，README §5.4）。

**协议扩展有滚动升级窗口**：解析器是派发进程**启动时 import** 的快照，本模块合并进
HEAD 对此后**新起**的进程才生效。旧进程遇到新属性形态会整块看不见 → 判 ``parse_error``
→ 组员白跑一跳、失败原因失真（E1 ``tags`` 实测）。故 :func:`parse_delivery` 在判
``parse_error`` 之前先走一遍**宽松兜底面**（只认标签名，不校验属性语法），命中即按兜底
结果解析并在 :attr:`Delivery.warning` 附一行告警——告警不静默，组长看得见。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .failure_tags import parse_reason, parse_tags

#: 标签属性面（**引号感知**）。旧写法 ``[^>]*`` 把属性值里的 ``>`` 当标签结束——
#: ``reason="优于 > 阈值"`` 会在引号内部提前停下，剩余部分被当成正文起点：kind 仍判
#: result、warning 仍为空（严格面自认为「匹配上了」，兜底面不触发），而产物路径被污染成
#: ``阈值">p/q.pdf`` → 机械验收去 check 一个不存在的路径，比 parse_error 更隐蔽
#: （backlog 20261012-025325-devops）。改法：先吃成对的引号段（值内 ``>``/``</`` 一律
#: 放行），再吃不参与引号与 ``>`` 的普通字符。引号段禁跨行（普通字符仍可跨行，多行标签
#: 形态照旧收下）：未闭合的引号若允许跨行找配对，会把到下一个孤立引号之间的整段交付
#: 吞进属性面——那比截断更糟；禁跨行时最多吞到行尾，严格面随即失败并走兜底告警。
_ATTRS = r"""(?:"[^"\n]*"|'[^'\n]*'|[^>"'])*"""

_ARTIFACT_RE = re.compile(
    rf"<artifact\b(?P<attrs>{_ATTRS})>(?P<path>.*?)</artifact>", re.DOTALL
)
_ATTR_RE = re.compile(r'(?P<key>\w+)\s*=\s*"(?P<val>[^"]*)"')
# E0-B：infra_suggestion 现在可带 kind 属性，通用 _extract（`<tag>` 字面量）匹配不到
# 属性形式，故单列一条属性感知的正则（与 delivery-gate 的 INFRA_RE 同语义）。
_INFRA_RE = re.compile(
    rf"<infra_suggestion\b(?P<attrs>{_ATTRS})>(?P<body>.*?)</infra_suggestion>",
    re.DOTALL,
)
# ---- 严格面：本进程所认识的**那一版**协议形状（随协议版本演进）----
_RESULT_RE = re.compile(
    rf"<result\b(?P<attrs>{_ATTRS})>(?P<body>.*?)</result>", re.DOTALL
)
# E1：同上理由，<blocked> 可带 tags 属性——用 _extract 会**整体漏匹配**（解析器
# 看不见带属性的交付块，交付就被判成 parse_error）。delivery-gate 的 BLOCKED_RE 同语义。
_BLOCKED_RE = re.compile(
    rf"<blocked\b(?P<attrs>{_ATTRS})>(?P<body>.*?)</blocked>", re.DOTALL
)
# ---- 宽松兜底面：协议形状之外的那一层 ----
# 只认「交付块标签名 + 任意属性面 + 正文」：不校验属性语法（单/双引号、裸值、未知属性
# 一律收下），闭合标签容忍尾部空白。严格面两者皆缺时先来这里捞一把（见模块 docstring
# 的滚动升级窗口；backlog 20261012-002857-devops）。捞到就**必须**附告警——兜底面比协议
# 宽，让它静默工作等于把协议漂移藏起来。
_LENIENT_BLOCK_RE = re.compile(
    r"<(?P<tag>result|blocked)\b(?P<attrs>[^>]*)>\s*(?P<body>.*?)\s*</(?P=tag)\s*>",
    re.DOTALL,
)
#: 宽松属性面：双引号 / 单引号 / 裸值三种形态都取值（严格面 :data:`_ATTR_RE` 只认双引号）。
_LOOSE_ATTR_RE = re.compile(
    r"""(?P<key>\w+)\s*=\s*(?:"(?P<dq>[^"]*)"|'(?P<sq>[^']*)'|(?P<bare>[^\s>]+))"""
)
#: 兜底命中时的告警文案（逐字进 :attr:`Delivery.warning`，由 dispatch/drain 呈现）。
PROTOCOL_LAG_HINT = "解析器早于协议：交付块按宽松兜底形态解析，建议重跑本单"
# 建议正文里的结构化标记行（本批 leader 建议自身即样例）：`kind: defect`、
# `success_criteria：` 后紧跟要点行。kind 值面**含连字符与数字**：组员自造的分类词
# （实测 `defusability-placeholder`）必须能被读出来再交给词表判定，否则「标错分类」会被
# 误报成「没标分类」，提示文案指错方向（backlog 20261012-120001-leader）。
#: 结构化标记行的**前缀盲区**（backlog adhoc-criteria-parse / 20261016-215944-devops）：
#: 建议正文里这些标记常写成 **列表项**（``- success_criteria: …``）、**粗体**
#: （``**success_criteria**:``）或 **markdown 标题**（``## success_criteria:``——文档原话说
#: 「段」，读者自然这么写，首写必踩），而旧口径只认行首裸标记 → 整条漏读、真判据被当
#: 「缺判据」误拒（实案 20261015-130838-writing、20261016-203600-devops 各一次）。
#: 三处统一放宽：可选标题井号 + 可选列表标记 + 可选粗体包裹（同一前缀，``kind`` 同享）。
_MARKER_PREFIX = r"^\s*(?:#{1,6}\s*)?(?:[-*+•]\s*)?(?:\*\*|__)?"
_MARKER_SUFFIX = r"(?:\*\*|__)?"
_KIND_LINE_RE = re.compile(
    _MARKER_PREFIX + r"kind[:：]\s*([A-Za-z][\w-]*)\s*$", re.MULTILINE
)
_SC_HEADER_RE = re.compile(
    _MARKER_PREFIX + r"success_criteria" + _MARKER_SUFFIX, re.IGNORECASE
)
#: 标记行**本身**（含冒号）——用于剥出「行内判据值」。
_SC_INLINE_RE = re.compile(
    _MARKER_PREFIX + r"success_criteria" + _MARKER_SUFFIX + r"\s*[:：]?\s*",
    re.IGNORECASE,
)
#: 判据段**终止面**（backlog 20261016-224400）：下一节标题 或 另一条结构化标记行。
#: 没有它，「标记后真空」的正文会把后文当判据吃掉（假阳性）。
_SC_SECTION_BREAK_RE = re.compile(
    r"^\s*(?:#{1,6}\s|kind[:：]|source[:：]|fingerprint[:：])"
)


@dataclass
class Artifact:
    """一条产物声明及其机械验收意图。"""

    path: str
    check: str = "none"
    reason: str = ""


@dataclass
class Delivery:
    """一次交付的结构化解析结果。"""

    kind: str  # "result" | "blocked" | "parse_error"
    body: str = ""
    artifacts: list[Artifact] = field(default_factory=list)
    infra_suggestion: str = ""
    infra_kind: str = (
        ""  # E0-B：非空 infra_suggestion 的 kind 属性（defect/usability/""）
    )
    #: E1（README §5.4）：<blocked> 的 tags 属性 → 失败标签列表（已规范化，未校词表——
    #: 校验在 gate 侧，本侧只负责把组员标的内容如实记进台账）。缺省空列表。
    tags: list[str] = field(default_factory=list)
    #: <blocked reason="…"> 的**成因档位**（backlog 20261013-191213-devops）：单值受控键
    #: （needs_decision/capacity/blocked_by_env/blocked_by_upstream，词表见
    #: state/failure-tags.json 的 reasons 段）。缺省空串=未标注（合法）。与 tags 同纪律：
    #: 本侧只如实解析，校验在 gate 侧——drain/status 据它把 blocked **分列**计数。
    reason: str = ""
    #: 宽松兜底命中时的告警（严格面什么都没扫到、宽松面扫到一块）。**空串=正常路径**，
    #: 非空即「解析器早于协议」：交付照样判读，但派发/巡检面必须打出一行提示重跑本单
    #: （backlog 20261012-002857-devops）。
    warning: str = ""
    raw: str = ""

    @property
    def ok(self) -> bool:
        """交付是否表示任务成功（result 块存在即成功——用户裁决的清晰语义）。"""
        return self.kind == "result"


#: 严格面登记表：交付块 → 本进程那一版协议的形状。:func:`_strict_match` 单点查这里，
#: 好处是「协议版本」可整体替换（测试用它钉住旧版正则，验证兜底面真能救回当次交付）。
_STRICT_BLOCKS: dict[str, re.Pattern[str]] = {
    "result": _RESULT_RE,
    "blocked": _BLOCKED_RE,
}


def _attrs_of(m: re.Match[str]) -> str:
    """匹配体的属性面文本；旧版正则没有 ``attrs`` 组时返回空串（不抛）。"""
    return m.groupdict().get("attrs") or ""


def _strict_match(text: str, tag: str) -> re.Match[str] | None:
    """按 :data:`_STRICT_BLOCKS` 里登记的协议形状搜第一处交付块。"""
    return _STRICT_BLOCKS[tag].search(text)


def _lenient_blocks(text: str) -> dict[str, tuple[str, str]]:
    """宽松面扫全文：``{标签名: (属性面, 正文)}``，同名只留第一处。

    与严格面的分工：严格面**校验**协议（属性写法、闭合形态按当前版本定义），宽松面只
    **认标签名**。前者是协议的执行端，后者是版本错位的救援端——兜底命中一律附告警，
    不让协议漂移静默通过。
    """
    found: dict[str, tuple[str, str]] = {}
    for m in _LENIENT_BLOCK_RE.finditer(text):
        found.setdefault(m.group("tag"), (_attrs_of(m), m.group("body").strip()))
    return found


def _loose_attr(attrs: str, key: str) -> str:
    """宽松属性取值：双引号 / 单引号 / 裸值三种形态都认（严格面只认双引号）。"""
    for m in _LOOSE_ATTR_RE.finditer(attrs):
        if m.group("key") != key:
            continue
        return m.group("dq") or m.group("sq") or m.group("bare") or ""
    return ""


def _extract(text: str, tag: str) -> str | None:
    """提取第一个 ``<tag ...>...</tag>`` 块的正文（DOTALL，属性一律跳过）；不存在 → None。

    属性跳过是必需的：交付块现在可以带 ``tags=``/``kind=``（E0-B、E1），只匹配裸标签会
    **静默漏匹配**——带属性的成功交付会被判成 parse_error。``\\b`` 保证 ``<results>``
    这类同名前缀的标签不被误吞；需要读属性时用各自的属性感知正则（``_BLOCKED_RE``/
    ``_INFRA_RE``）。
    """
    m = _strict_match(text, tag)
    return m.group("body").strip() if m else None


def parse_delivery(result_text: str) -> Delivery:
    """解析组员最终回复文本。

    Args:
        result_text: envelope 的 result 字段（组员最终回复全文）。

    Returns:
        Delivery：kind 为 result/blocked/parse_error（两者皆缺或并存时 parse_error，
        正文取先出现者供人工判读）；``tags`` 取 <blocked> 的 tags 属性（result 交付
        不带标签——标签是失败分类面，成功没有可分类的对象）；``reason`` 取 <blocked>
        的成因档位（同上，只属于失败面）；``warning`` 仅在严格面全缺、宽松面救回时
        非空（见 :data:`PROTOCOL_LAG_HINT`）。
    """
    result_body = _extract(result_text, "result")
    blocked = _strict_match(result_text, "blocked")
    blocked_body = blocked.group("body").strip() if blocked else None
    tags: list[str] = []
    reason = ""
    if blocked is not None:
        blocked_attrs = {k: v for k, v in _ATTR_RE.findall(_attrs_of(blocked))}
        tags = parse_tags(blocked_attrs.get("tags", ""))
        reason = parse_reason(blocked_attrs.get("reason", ""))
    infra = ""
    infra_kind = ""
    im = _INFRA_RE.search(result_text)
    if im:
        infra = im.group("body").strip()
        attrs = {k: v for k, v in _ATTR_RE.findall(im.group("attrs"))}
        infra_kind = attrs.get("kind", "")

    warning = ""
    if result_body is not None and blocked_body is None:
        kind, body = "result", result_body
    elif blocked_body is not None and result_body is None:
        kind, body = "blocked", blocked_body
    else:
        # 严格面「两者皆缺」→ 先走宽松兜底面：旧解析器读新属性交付的唯一救援路径。
        # 「并存」不走兜底：两块并存是协议违规，宽松面救它等于把违规变成告警。
        lenient = _lenient_blocks(result_text)
        if len(lenient) == 1:
            kind, (attrs, body) = next(iter(lenient.items()))
            warning = PROTOCOL_LAG_HINT
            if kind == "blocked":
                tags = parse_tags(_loose_attr(attrs, "tags"))
                reason = parse_reason(_loose_attr(attrs, "reason"))
        else:
            kind = "parse_error"
            body = result_body or blocked_body or ""
    if kind != "blocked":
        tags = []
        reason = ""

    artifacts: list[Artifact] = []
    if kind == "result":
        # 兜底救回的 result 也要抽产物：正文面取实际判出的那块（严格面命中时
        # result_body 与 body 同值，走兜底时 result_body 是 None）
        for m in _ARTIFACT_RE.finditer(body or ""):
            attrs = {k: v for k, v in _ATTR_RE.findall(m.group("attrs"))}
            artifacts.append(
                Artifact(
                    path=m.group("path").strip(),
                    check=attrs.get("check", "none"),
                    reason=attrs.get("reason", ""),
                )
            )

    return Delivery(
        kind=kind,
        body=body,
        artifacts=artifacts,
        infra_suggestion=infra,
        infra_kind=infra_kind,
        tags=tags,
        reason=reason,
        warning=warning,
        raw=result_text,
    )


def parse_suggestion_fields(body: str) -> tuple[str, str]:
    """从建议正文解析结构化字段 kind 与 success_criteria（E0-B 判据 2/3 的解析端）。

    约定：正文含独立一行 ``kind: defect``（或 ``kind：usability``），以及
    ``success_criteria`` 标记行——其后紧跟的要点行（直到空行）即判据正文。
    样例见本批 leader 建议自身（state/suggestions/approved/20261011-210317-leader.md）。

    Args:
        body: infra_suggestion 块正文（或 suggestion 文件全文）。

    Returns:
        ``(kind, success_criteria)``：任一缺失即返回空串。kind 取标记行值（小写规范化），
        success_criteria 取标记行后（**先跳过前导空行**）连续非空行的拼接（保留 ``- ``
        要点前缀供人读）；撞到下一节标题/结构化标记行即止（backlog 20261016-224400）。
    """
    kind = ""
    m = _KIND_LINE_RE.search(body)
    if m:
        kind = m.group(1).lower()

    criteria: list[str] = []
    lines = body.splitlines()
    for i, ln in enumerate(lines):
        if _SC_HEADER_RE.match(ln):
            # **行内值也算**：原件常把判据直接写在标记行上（`- success_criteria: <全文>`）——
            # 旧口径只看其后的行，于是「有判据」被判成「缺判据」（实案 20261015-130838-writing）。
            inline = _SC_INLINE_RE.sub("", ln).strip()
            if inline:
                criteria.append(inline)
            # **标记行后的前导空行跳过**（backlog 20261016-224400）：markdown 惯例里标题后
            # 必有一空行，「## success_criteria:」＋空行＋要点行才自然——旧口径「其后直到空行」
            # 于是把标准写法读成**空判据**，approve 照旧拒绝（第五例实证）。两条护栏防过度吸收：
            #   ① 有行内值时按旧口径「空行即止」（不越过空行把后文当判据）；
            #   ② 撞到下一节标题/结构化标记行即止（标记后真空 → 仍判「缺判据」）。
            skip_leading = not inline
            started = False
            for follow in lines[i + 1 :]:
                if not follow.strip():
                    if started or not skip_leading:
                        break
                    continue
                if _SC_SECTION_BREAK_RE.match(follow):
                    break
                started = True
                criteria.append(follow.strip())
            break
    return kind, "\n".join(criteria)
