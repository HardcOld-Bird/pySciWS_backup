"""技能真本的 ``--model-tier`` 示例值须与 CLI 可选集一致（backlog 20261029-215130-leader）。

事故：SKILL.md 原文「升 ``--model-tier max``」与 ``--help`` 的 ``{main,pro,minimum}``
不一致（旧档名残留；``max`` 仅作读路别名、帮助不列），照抄易踩空。本钉把「技能文本
里出现的取值 ⊆ 角色主集」变成机械判据——今后照抄即可执行。
"""

from __future__ import annotations

import re

from pysci.paths import ORCHESTRATION_ROOT
from pysci.skills.orchestration.tools import presets

SKILL_DIR = ORCHESTRATION_ROOT / "skills" / "orchestration"

#: ``--model-tier <值>``：值可为 a、a|b|c 或 {a,b,c}；遇空白/反引号/中文标点/括号即止。
_TOKEN_RE = re.compile(r"--model-tier[\s=]+([^\s`，。；、（）()\[\]【】]+)")


def _tier_values(text: str) -> list[str]:
    """抽出文本里 ``--model-tier`` 的示例取值（``-`` 开头视为下一旗标，跳过）。"""
    values: list[str] = []
    for match in _TOKEN_RE.finditer(text):
        raw = match.group(1).strip().strip("{}")
        for token in re.split(r"[|,]", raw):
            token = token.strip().strip("`")
            if token and not token.startswith("-"):
                values.append(token)
    return values


def test_skill_model_tier_examples_are_cli_choices():
    valid = set(presets.ROLE_CHOICES)
    files = [SKILL_DIR / "SKILL.md", *sorted((SKILL_DIR / "references").glob("*.md"))]
    seen: list[str] = []
    offenders: list[str] = []
    for path in files:
        values = _tier_values(path.read_text(encoding="utf-8"))
        seen.extend(values)
        offenders.extend(f"{path.name}: {v}" for v in values if v not in valid)
    assert seen, "未解析到任何 --model-tier 示例（判据失效，检查正则/文本形态）"
    assert not offenders, (
        f"技能文本的 --model-tier 取值不在 CLI 可选集 {sorted(valid)} 内：{offenders}"
    )
