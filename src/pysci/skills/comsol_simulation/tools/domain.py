"""仿真「域几何声明」机械检查：节存在性 + 四问键齐全 + 每问带数值。

任务书/交付物须含 ``## 域几何声明`` 节，四问以 ``- **键**：值`` 逐行给出：

- 兼容性：目标场区/标注范围完整位于有效域内 + 边距值
- 覆盖率：散射体占入射面比例；<100% 须显式声明
- 边界适配：外场边界/PML 选择 + 背景压力场不覆盖散射体内部
- 激励口径：波束宽度 vs 孔径 + 实验口径互证；并含**工作点 (cr,ci) ↔ 最近异常点(EP)
  偏移**与寻点判据出处（理论小结/探测脚本）

缺节、缺键、无数字或占位词（待补/TBD/…）均判 FAIL；激励口径缺 EP 子问亦 FAIL
（tb-3 R5 M4）。域版本与寻点版本不一致且无重寻读数时给**软告警**（不判 FAIL，
tb-3 R6 M5：扩域/换形后须在新域重寻 EP）。
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

#: 激励口径的「工作点-异常点偏移」子问（tb-3 R5 M4）：(cr,ci)、EP 偏移、寻点出处。
_EXC_CR_RE = re.compile(r"(?<![A-Za-z])cr(?![A-Za-z])\s*(?:[=:：]|\s)\s*-?\d", re.I)
_EXC_CI_RE = re.compile(r"(?<![A-Za-z])ci(?![A-Za-z])\s*(?:[=:：]|\s)\s*-?\d", re.I)
_EXC_PAIR_RE = re.compile(
    r"\(\s*cr\s*,\s*ci\s*\)\s*[=:：]\s*\(?\s*-?\d+(?:\.\d+)?\s*,\s*-?\d", re.I
)
_EXC_OFFSET_RE = re.compile(
    r"(?:偏移|Δ|delta|距(?:离)?[^。；\n]{0,24}(?:EP|异常点|谱奇点))"
    r"[^。；\n]{0,48}\d",
    re.I,
)
_EXC_SOURCE_RE = re.compile(
    r"源[：:]|出处|依据|判据|理论小结|探测脚本|探针脚本"
    r"|\.(?:md|py|json|yaml)(?![A-Za-z])",
    re.I,
)

#: 域版本/寻点版本（R5 批语 3：扩域/换形后须在新域重寻 EP）。
_DOMAIN_VER_RE = re.compile(r"域版本\s*[：:=]\s*([^\s；;，,。）)]+)")
_EP_VER_RE = re.compile(r"寻点版本\s*[：:=]\s*([^\s；;，,。）)]+)")
_RESEARCH_RE = re.compile(r"重寻[^。；\n]{0,48}\d|重寻偏移|re-?search", re.I)


def _check_excitation_ep(value: str) -> dict:
    """激励口径的 EP 子问：工作点、偏移、出处；域版本不一致时软告警。"""
    missing: list[str] = []
    point = bool(_EXC_PAIR_RE.search(value)) or (
        bool(_EXC_CR_RE.search(value)) and bool(_EXC_CI_RE.search(value))
    )
    if not point:
        missing.append("工作点(cr,ci)")
    if not _EXC_OFFSET_RE.search(value):
        missing.append("最近EP偏移")
    if not _EXC_SOURCE_RE.search(value):
        missing.append("寻点判据出处")
    domain_ver = (_DOMAIN_VER_RE.search(value) or [None, ""])[1]
    ep_ver = (_EP_VER_RE.search(value) or [None, ""])[1]
    warnings: list[str] = []
    if (
        domain_ver
        and ep_ver
        and domain_ver != ep_ver
        and not _RESEARCH_RE.search(value)
    ):
        warnings.append(
            f"域版本 {domain_ver} ≠ 寻点版本 {ep_ver} 且未见重寻读数——"
            "扩域/换形后须在新域重寻 EP 并报偏移（R5 批语 3）"
        )
    return {
        "ok": not missing,
        "missing": missing,
        "value": value,
        "domain_version": domain_ver,
        "ep_version": ep_ver,
        "warnings": warnings,
    }


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
    """``- **键**：值`` / ``### 键`` 后的值（含缩进续行）；未出现返回 None。"""
    pat = re.compile(
        rf"^\s*(?:#{{1,6}}\s*|[-*]\s*)?(?:\*\*)?{re.escape(key)}"
        r"(?:\*\*)?\s*[:：]?\s*(.*)$"
    )
    for i, line in enumerate(lines):
        m = pat.match(line)
        if not m:
            continue
        parts = [m.group(1).strip()]
        for nxt in lines[i + 1 :]:
            s = nxt.strip()
            if not s:
                break
            if re.match(r"^\s*(?:#{1,6}\s|[-*]\s)", nxt):
                break
            if not nxt.startswith((" ", "\t")):
                break
            parts.append(s)
        return " ".join(p for p in parts if p)
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
            "excitation_ep": {
                "ok": False,
                "missing": ["工作点(cr,ci)", "最近EP偏移", "寻点判据出处"],
                "value": "",
                "warnings": [],
            },
            "missing": list(KEYS),
            "non_numeric": [],
            "warnings": [],
            "detail": f"缺「{SECTION_KEY}」节",
        }
    questions: dict[str, dict] = {}
    missing: list[str] = []
    non_numeric: list[str] = []
    excitation: dict = {"ok": True, "missing": [], "value": ""}
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
        elif key == "激励口径":
            excitation = _check_excitation_ep(val or "")
    parts = []
    if missing:
        parts.append("缺键：" + "、".join(missing))
    if non_numeric:
        parts.append("无有效数值/占位：" + "、".join(non_numeric))
    if excitation["missing"]:
        parts.append("激励口径缺：" + "、".join(excitation["missing"]))
    return {
        "ok": not missing and not non_numeric and excitation["ok"],
        "section": True,
        "questions": questions,
        "excitation_ep": excitation,
        "missing": missing,
        "non_numeric": non_numeric,
        "warnings": list(excitation.get("warnings") or []),
        "detail": "；".join(parts) or "四问齐全且带数值（含工作点-EP 偏移）",
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
        exc = rep.get("excitation_ep") or {}
        if exc.get("value") and exc.get("ok"):
            print("  [√] 激励口径·工作点-EP 偏移：工作点/偏移/出处齐全")
        elif exc.get("value"):
            print(
                "  [✗] 激励口径·工作点-EP 偏移：缺 "
                + "、".join(exc.get("missing") or [])
            )
        for warn in rep.get("warnings") or []:
            print(f"  [!] {warn}")
        print(f"  {rep['detail']}")
    return 0 if rep["ok"] else 1
