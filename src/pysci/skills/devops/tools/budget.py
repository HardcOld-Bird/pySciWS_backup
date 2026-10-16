"""harness 预算三档审计（`.qoder/rules/basic.md` §1 的机械哨兵）。

三档口径与 basic.md §1 一致：**全量注入档**（charter / AGENTS.md 等每跳全文进上下文的
文件）、**常驻暴露档**（技能 `description:` 行——不进正文但每跳占位，按类合计）、
**按需档**（`references/*.md`、自建 rules/skills 等被指向时才读的单文件）。每档上限同为
8192B，**按落盘面（CRLF）计量**——即 `wc -c` / `find -size` 的口径，超限即违规。

本模块只读不写；`pysci-dev doctor` 调 :func:`audit` 打印审计报告并据违规数决定退出码。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pysci.paths import ORCHESTRATION_ROOT, PODS_ROOT, PROJECT_ROOT

#: 预算上限（字节，落盘面）。basic.md §1 三档同为 8KB。
LIMIT = 8192

#: 余量预警阈值（字节）：文档族须留出 **≥512B 可用余量**（backlog 20261016-030200——
#: orchestration 真本多文件已贴顶：dispatch-playbook 曾只剩 64B，任何新规则都只能挤压
#: 他人正文）。占用 > ``LIMIT − WARN_HEADROOM`` 即提示「按 basic.md §1 拆分/重设计」。
WARN_HEADROOM = 512

#: 预警起点（落盘字节）：**大于**它且未超限即提示余量吃紧。全员真源，pre-commit 钩子
#: ``guards/budget_check.py`` 复制同一常量（钩子 stdlib-only），一致性由测试钉住。
WARN_AT = LIMIT - WARN_HEADROOM
#: 兼容读数（= WARN_AT / LIMIT，仅用于百分比展示）。
WARN_RATIO = WARN_AT / LIMIT

#: 审计根（模块级常量，便于测试 monkeypatch 到 tmp_path）。
SKILLS_ROOT = ORCHESTRATION_ROOT / "skills"
RULES_ROOT = PROJECT_ROOT / ".qoder" / "rules"

#: 常驻暴露档的计量对象：各技能真本的 description 行。
DESCRIPTION_GLOB = "*/SKILL.md"


@dataclass(frozen=True)
class Finding:
    """一条审计发现：档位、路径、落盘字节数（超限判据用 ``over`` 而非现场重算）。"""

    tier: str
    path: Path
    size: int

    @property
    def over(self) -> bool:
        return self.size > LIMIT

    @property
    def headroom(self) -> int:
        """距上限的剩余字节（可为负=超限）。"""
        return LIMIT - self.size

    @property
    def pct(self) -> int:
        """已用百分比（向下取整）。"""
        return self.size * 100 // LIMIT

    @property
    def near(self) -> bool:
        """余量吃紧：占用 >95% 上限且未超限（预警不拦截）。"""
        return not self.over and self.size > WARN_AT


def disk_size(path: Path) -> int:
    """按落盘面计量：``wc -c`` 口径（本机 core.autocrlf 下文本为 CRLF）。"""
    return len(path.read_bytes())


def _desc_line_size(skill_md: Path) -> int:
    """frontmatter 里 ``description:`` 那一行的字节数（常驻暴露档的征税面）。"""
    try:
        lines = skill_md.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return 0
    for ln in lines[:40]:
        if ln.startswith("description:"):
            return len(ln.encode("utf-8"))
    return 0


def audit() -> list[Finding]:
    """扫描三档全部对象，返回发现列表（含未超限者，供 doctor 打印汇总）。"""
    out: list[Finding] = []
    if PODS_ROOT.exists():
        for pod in sorted(p for p in PODS_ROOT.iterdir() if p.is_dir()):
            for f in (pod / ".qoder" / "rules" / "charter.md", pod / "AGENTS.md"):
                if f.exists():
                    out.append(Finding("全量注入", f, disk_size(f)))
            for f in sorted((pod / ".qoder" / "rules").glob("*.md")):
                if f.name != "charter.md":
                    out.append(Finding("按需", f, disk_size(f)))
            for f in sorted((pod / ".qoder" / "skills").rglob("*.md")):
                out.append(Finding("按需", f, disk_size(f)))
    for f in sorted(RULES_ROOT.glob("*.md")) if RULES_ROOT.exists() else []:
        out.append(Finding("全量注入", f, disk_size(f)))
    if SKILLS_ROOT.exists():
        for f in sorted(SKILLS_ROOT.rglob("*.md")):
            tier = "全量注入" if f.name == "SKILL.md" else "按需"
            out.append(Finding(tier, f, disk_size(f)))
    return out


def description_findings() -> list[Finding]:
    """常驻暴露档：每个技能真本的 description 行（单条即一份征税面）。"""
    if not SKILLS_ROOT.exists():
        return []
    return [
        Finding("常驻暴露", f, _desc_line_size(f))
        for f in sorted(SKILLS_ROOT.glob(DESCRIPTION_GLOB))
    ]


def report(findings: list[Finding], desc: list[Finding]) -> list[str]:
    """人类可读审计行：超限 [!] 逐条、余量吃紧 [~] 逐条、各档计数 + 余量读数。"""
    lines: list[str] = []
    over = [f for f in findings if f.over]
    for f in over:
        lines.append(f"[!] 超预算 {f.tier} 档：{_rel(f)}（{f.size}B > {LIMIT}B）")
    total_desc = sum(d.size for d in desc)
    if total_desc > LIMIT:
        lines.append(
            f"[!] 超预算 常驻暴露 档：技能 description 合计 {total_desc}B > {LIMIT}B"
        )
    near = sorted((f for f in findings if f.near), key=lambda f: f.headroom)
    for f in near:
        lines.append(
            f"[~] 余量吃紧 {f.tier} 档：{_rel(f)}"
            f"（{f.size}B / {LIMIT}B，已用 {f.pct}%，余 {f.headroom}B）"
            "——加一句即可能超限，宜先精简或移分册"
        )
    for tier in ("全量注入", "按需"):
        n = sum(1 for f in findings if f.tier == tier)
        bad = sum(1 for f in findings if f.tier == tier and f.over)
        lines.append(f"  {tier} 档：{n} 个文件，违规 {bad}")
    lines.append(
        f"  常驻暴露 档：{len(desc)} 条 description，合计 {total_desc}B / 上限 {LIMIT}B"
    )
    lines.append(_headroom_readout(findings, near))
    if not over and total_desc <= LIMIT:
        lines.insert(0, "  [√] 三档全部在预算内")
    return lines


def _rel(f: Finding):
    """相对仓库根的短路径（不在仓内时原样返回）。"""
    try:
        return f.path.relative_to(PROJECT_ROOT)
    except ValueError:
        return f.path


def _headroom_readout(findings: list[Finding], near: list[Finding]) -> str:
    """余量读数一行：最紧文件距上限多少字节 + 预警计数（决定能否再加一句）。"""
    within = [f for f in findings if not f.over]
    if not within:
        return "  余量读数：无可计量文件"
    tight = min(within, key=lambda f: f.headroom)
    warn = f"，{len(near)} 个余量 <{WARN_HEADROOM}B 预警" if near else ""
    return (
        f"  余量读数：最紧 {_rel(tight)} 余 {tight.headroom}B"
        f"（已用 {tight.pct}%）{warn}"
    )
