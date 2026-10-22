"""watch 事件枚举的活文档一致性（insp-20261013-165608：补 SALVAGE，与 drain-watch 同口径）。

README v3.0 拆分后旧枚举移居，活残留只有 `references/quality-and-improvement.md`；
本测试把「六类口径」钉在 drain-watch ↔ quality 文档 ↔ watch.PRIORITY 三面，防再漂移。
"""

from __future__ import annotations

import re

from pysci.paths import ORCHESTRATION_ROOT
from pysci.skills.orchestration.tools import watch

DRAIN_WATCH = (
    ORCHESTRATION_ROOT / "skills" / "orchestration" / "references" / "drain-watch.md"
)
QUALITY = ORCHESTRATION_ROOT / "references" / "quality-and-improvement.md"

_SIX = [
    "QUEUE-COMPLETE",
    "NEEDS-LEADER",
    "QUOTA-STOP",
    "CHANNEL-STOP",
    "CRASH",
    "SALVAGE",
]

_ENUM_RE = re.compile(r"([A-Z][A-Z-]*)")


def _drain_watch_six() -> list[str]:
    lines = DRAIN_WATCH.read_text(encoding="utf-8").splitlines()
    start = next(i for i, ln in enumerate(lines) if "基线差分六类" in ln)
    blob = " ".join(lines[start : start + 2]).split("（", 1)[0]
    return _ENUM_RE.findall(blob)


def _quality_enum() -> list[str]:
    lines = QUALITY.read_text(encoding="utf-8").splitlines()
    start = next(i for i, ln in enumerate(lines) if "命中 QUEUE-COMPLETE" in ln)
    blob = " ".join(lines[start : start + 2]).split("（", 1)[0]
    return _ENUM_RE.findall(blob)


def test_drain_watch_declares_six_classes():
    assert _drain_watch_six() == _SIX, _drain_watch_six()


def test_quality_doc_enum_matches_six_classes():
    """success_criteria：quality 文档枚举含 SALVAGE，且与 drain-watch 六类一致。"""
    assert _quality_enum() == _SIX, _quality_enum()
    assert _quality_enum() == _drain_watch_six()


def test_watch_priority_includes_salvage():
    assert "SALVAGE" in watch.PRIORITY, "代码侧 watch 事件须含 SALVAGE（六类口径）"
