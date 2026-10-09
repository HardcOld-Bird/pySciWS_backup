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
    infra = _extract(result_text, "infra_suggestion") or ""

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
        raw=result_text,
    )
