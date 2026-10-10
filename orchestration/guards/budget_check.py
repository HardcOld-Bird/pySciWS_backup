#!/usr/bin/env python3
"""pre-commit 钩子：真本侧 harness 预算硬拦（basic.md §1；backlog 20261010-budget-precommit）。

三层执法之**真本侧**（管组长与 devops 的编辑）：技能真本的 ``SKILL.md`` 与
``references/*.md``、根 ``.qoder/rules/basic.md`` 与 ``leader-only.md`` 每文件 ≤8192B，
超限即拦下提交并打印 文件 + 实测字节 + 上限 + 档位 + 整改动作。

计量口径是**落盘字节**：pre-commit 传进来的是工作区文件（Windows 经 smudge 为 CRLF），
``stat().st_size`` 与 ``wc -c`` / ``find -size +8192c`` / delivery-gate / ``pysci-dev
doctor`` 的预算审计同口径——harness 注入所付的税按字节算，不按逻辑字符数。

范围判定只依赖路径形状（不依赖 ``pysci`` 包）：钩子跑在 pre-commit 自建的空虚拟环境里，
那里没装本项目的包，stdlib-only 是唯一不会崩的写法。与 ``budget.py`` 的一致性由
``tests/skills/devops/test_budget_precommit.py`` 钉住（上限值、档位集合、真实仓库现状）。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path, PurePosixPath

LIMIT = 8192

# (档位标签, 路径判定, 整改动作)——顺序即优先级，先命中者定性。
TIERS: tuple[tuple[str, re.Pattern[str], str], ...] = (
    (
        "全量注入档（SKILL.md 每跳全文进上下文）",
        re.compile(r"(^|/)orchestration/skills/[^/]+/SKILL\.md$"),
        "把细节移到同目录 references/*.md（按需档），SKILL.md 只留决策与调用面 + 一行索引",
    ),
    (
        "按需档（references 分册只在被指向时读）",
        re.compile(r"(^|/)orchestration/skills/[^/]+/references/[^/]+\.md$"),
        "按 `##` 主题拆成 <stem>-0N-<slug>.md 分册，原名留作分册索引（见 devops 技能）",
    ),
    (
        "全量注入档（根 rules 全文进每个会话）",
        re.compile(r"(^|/)\.qoder/rules/(?:basic|leader-only)\.md$"),
        "精简措辞；权威细节移到 orchestration/README.md 或技能真本的 references 分册",
    ),
)


def tier_for(path_str: str) -> tuple[str, str] | None:
    """返回 (档位标签, 整改动作)；不在执法范围内 → None。"""
    norm = path_str.replace("\\", "/")
    for label, pattern, action in TIERS:
        if pattern.search(norm):
            return label, action
    return None


def main(argv: list[str]) -> int:
    offenders: list[str] = []
    checked = 0
    for arg in argv:
        hit = tier_for(arg)
        if hit is None:
            continue
        checked += 1
        p = Path(arg)
        if not p.is_file():  # 暂存删除/重命名的旧路径：无内容可注入，不算违规
            continue
        size = p.stat().st_size
        if size > LIMIT:
            label, action = hit
            rel = PurePosixPath(arg.replace("\\", "/"))
            offenders.append(
                f"  ✗ {rel}  实测 {size}B > 上限 {LIMIT}B（{label}）\n"
                f"     超出 {size - LIMIT}B。动作：{action}。"
            )
    if not offenders:
        print(f"harness 预算：{checked} 个在范围内的文件全部 ≤{LIMIT}B ✓")
        return 0
    print(
        f"harness 预算硬拦：{len(offenders)} 个文件超出 basic.md §1 的每文件上限 "
        f"{LIMIT}B（落盘字节，CRLF 计税；口径同 wc -c / find -size +8192c）：\n"
        + "\n".join(offenders)
        + "\n超限说明该文档过于复杂——正确反应是拆分或重新设计其使用方式，不是放宽预算。"
        "改完跑 `pysci-dev doctor` 的预算审计复测。"
    )
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
