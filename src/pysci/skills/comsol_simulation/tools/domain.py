"""仿真「域几何声明」机械检查：节存在性 + 四问键齐全 + 每问带数值。

任务书/交付物须含 ``## 域几何声明`` 节，四问以 ``- **键**：值`` 逐行给出：

- 兼容性：目标场区/标注范围完整位于有效域内 + 边距值
- 覆盖率：散射体占入射面比例；<100% 须显式声明
- 边界适配：外场边界/PML 选择 + 背景压力场不覆盖散射体内部
- 激励口径：波束宽度 vs 孔径 + 实验口径互证

缺节、缺键、无数字或占位词（待补/TBD/…）均判 FAIL。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

SECTION_KEY = "域几何声明"
KEYS: tuple[str, ...] = ("兼容性", "覆盖率", "边界适配", "激励口径")

_HEADING_RE = re.compile(r"^(#{1,6})\s*(.*?)\s*$")
_PLACEHOLDER_RE = re.compile(r"待补|待定|待核|待确认|TBD|TODO|N/?A|—|\.\.\.|…", re.I)
_DIGIT_RE = re.compile(r"\d")


def _section_lines(text: str) -> list[str] | None:
    """``域几何声明`` 节正文（到同级/更高级标题为止）；缺节返回 None。"""
    lines = text.splitlines()
    start = None
    level = 0
    for i, line in enumerate(lines):
        m = _HEADING_RE.match(line)
        if m and SECTION_KEY in m.group(2):
            start, level = i, len(m.group(1))
            break
    if start is None:
        return None
    out = []
    for line in lines[start + 1 :]:
        m = _HEADING_RE.match(line)
        if m and len(m.group(1)) <= level:
            break
        out.append(line)
    return out


def _value_for(lines: list[str], key: str) -> str | None:
    """``- **键**：值`` / ``### 键`` 后的值；未出现返回 None。"""
    pat = re.compile(
        rf"^\s*(?:#{{1,6}}\s*|[-*]\s*)?(?:\*\*)?{re.escape(key)}"
        r"(?:\*\*)?\s*[:：]?\s*(.*)$"
    )
    for line in lines:
        m = pat.match(line)
        if m:
            return m.group(1).strip()
    return None


def check_domain_declaration(text: str) -> dict:
    """检查声明节；返回 ``{ok, section, questions, missing, non_numeric, detail}``。"""
    lines = _section_lines(text)
    if lines is None:
        return {
            "ok": False,
            "section": False,
            "questions": {
                k: {"present": False, "numeric": False, "value": ""} for k in KEYS
            },
            "missing": list(KEYS),
            "non_numeric": [],
            "detail": f"缺「{SECTION_KEY}」节",
        }
    questions: dict[str, dict] = {}
    missing: list[str] = []
    non_numeric: list[str] = []
    for key in KEYS:
        val = _value_for(lines, key)
        present = val is not None
        numeric = bool(
            val
            and len(val) >= 8
            and _DIGIT_RE.search(val)
            and not _PLACEHOLDER_RE.search(val)
        )
        questions[key] = {
            "present": present,
            "numeric": numeric,
            "value": val or "",
        }
        if not present:
            missing.append(key)
        elif not numeric:
            non_numeric.append(key)
    parts = []
    if missing:
        parts.append("缺键：" + "、".join(missing))
    if non_numeric:
        parts.append("无有效数值/占位：" + "、".join(non_numeric))
    return {
        "ok": not missing and not non_numeric,
        "section": True,
        "questions": questions,
        "missing": missing,
        "non_numeric": non_numeric,
        "detail": "；".join(parts) or "四问齐全且带数值",
    }


def cmd_domain_check(args) -> int:
    path = Path(args.file)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"[domain-check] 读取失败：{path}（{exc}）")
        return 2
    rep = check_domain_declaration(text)
    if getattr(args, "json", False):
        print(json.dumps({"file": str(path), **rep}, ensure_ascii=False, indent=2))
    else:
        print(f"[domain-check] {path}：{'PASS' if rep['ok'] else 'FAIL'}")
        for key, q in rep["questions"].items():
            mark = "√" if q["present"] and q["numeric"] else "✗"
            print(f"  [{mark}] {key}：{q['value'] or '（缺）'}")
        print(f"  {rep['detail']}")
    return 0 if rep["ok"] else 1
