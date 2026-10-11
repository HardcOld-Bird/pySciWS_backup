"""交付协议解析：从组员最终回复中提取 XML 标记块（README §3.4）。

协议：``<result>`` 与 ``<blocked>`` 互斥且必居其一（delivery-gate hook 已在成员侧
强制，本模块是第二道防线）；``<infra_suggestion>`` 可选；``<artifact>`` 内嵌于
result，声明机械验收类型。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_ARTIFACT_RE = re.compile(
    r"<artifact\b(?P<attrs>[^>]*)>(?P<path>.*?)</artifact>", re.DOTALL
)
_ATTR_RE = re.compile(r'(?P<key>\w+)\s*=\s*"(?P<val>[^"]*)"')
# E0-B：infra_suggestion 现在可带 kind 属性，通用 _extract（`<tag>` 字面量）匹配不到
# 属性形式，故单列一条属性感知的正则（与 delivery-gate 的 INFRA_RE 同语义）。
_INFRA_RE = re.compile(
    r"<infra_suggestion\b(?P<attrs>[^>]*)>(?P<body>.*?)</infra_suggestion>",
    re.DOTALL,
)
# 建议正文里的结构化标记行（本批 leader 建议自身即样例）：`kind: defect`、
# `success_criteria：` 后紧跟要点行。
_KIND_LINE_RE = re.compile(r"^\s*kind[:：]\s*([A-Za-z_]+)\s*$", re.MULTILINE)
_SC_HEADER_RE = re.compile(r"^\s*success_criteria", re.IGNORECASE)


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
    raw: str = ""

    @property
    def ok(self) -> bool:
        """交付是否表示任务成功（result 块存在即成功——用户裁决的清晰语义）。"""
        return self.kind == "result"


def _extract(text: str, tag: str) -> str | None:
    """提取第一个 <tag>...</tag> 块的正文（DOTALL）；不存在返回 None。"""
    m = re.search(rf"<{tag}>(?P<body>.*?)</{tag}>", text, re.DOTALL)
    return m.group("body").strip() if m else None


def parse_delivery(result_text: str) -> Delivery:
    """解析组员最终回复文本。

    Args:
        result_text: envelope 的 result 字段（组员最终回复全文）。

    Returns:
        Delivery：kind 为 result/blocked/parse_error（两者皆缺或并存时 parse_error，
        正文取先出现者供人工判读）。
    """
    result_body = _extract(result_text, "result")
    blocked_body = _extract(result_text, "blocked")
    infra = ""
    infra_kind = ""
    im = _INFRA_RE.search(result_text)
    if im:
        infra = im.group("body").strip()
        attrs = {k: v for k, v in _ATTR_RE.findall(im.group("attrs"))}
        infra_kind = attrs.get("kind", "")

    if result_body is not None and blocked_body is None:
        kind, body = "result", result_body
    elif blocked_body is not None and result_body is None:
        kind, body = "blocked", blocked_body
    else:
        kind = "parse_error"
        body = result_body or blocked_body or ""

    artifacts: list[Artifact] = []
    if kind == "result":
        for m in _ARTIFACT_RE.finditer(result_body or ""):
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
        success_criteria 取标记行后连续非空行的拼接（保留 ``- `` 要点前缀供人读）。
    """
    kind = ""
    m = _KIND_LINE_RE.search(body)
    if m:
        kind = m.group(1).lower()

    criteria: list[str] = []
    lines = body.splitlines()
    for i, ln in enumerate(lines):
        if _SC_HEADER_RE.match(ln):
            for follow in lines[i + 1 :]:
                if not follow.strip():
                    break
                criteria.append(follow.strip())
            break
    return kind, "\n".join(criteria)
