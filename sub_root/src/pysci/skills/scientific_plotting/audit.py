"""轻量出版自检 audit（用户裁决 2026-11-17：轻量版保留）。

检查面（SKILL.md 工作流第 6 步；判据细节见 references/export-acceptance.md）：

1. **宽度契约**：成品宽度 vs 期刊契约 ±2mm（防 tight bbox 裁切后的"假绿"名义宽度）；
2. **最小字号**：文本 artist 字号 ≥ 5pt（低于期刊可读下限）；
3. **字体嵌入**：PDF FontFile2 / EPS /sfnts（fonttype=42）/ SVG 保留 ``<text>`` 节点；
4. **面板号计数**：``(a)(b)…`` 标号个数 == 面板轴个数（面板轴 = 可见 + 携带数据
   artist；排除 colorbar 轴与 ultraplot 图例/caption 自动轴）；
5. **色循环**：prop_cycle == 家风循环（Okabe-Ito 去黄版，色盲安全）。

裁剪说明（ADR 2026-11-17）：旧 22KB audit 的色盲 deuteranopia 模拟、假绿防护细则、
标注语义四判据、零遮挡像素判据**砍**（判据规则文字进 SKILL.md 指南，人工/视觉层兜底）。
判定口径：**依据不足记 n/a，不判 FAIL**（工具链不可用不连坐）。

判定状态：``"pass" | "fail" | "na"``；:class:`AuditReport.passed` 只要有 fail 即 False。
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from pysci.skills.scientific_plotting.styles import HOUSE_CYCLE

#: 宽度容差（mm）：期刊社排版重排可接受的浮动。
WIDTH_TOL_MM = 2.0

#: 最小正文字号（pt）：低于此值审稿可读性不达标。
MIN_FONT_PT = 5.0

_PANEL_LABEL_RE = re.compile(r"^\([a-z]\)$")


@dataclass(frozen=True)
class Finding:
    """一条自检结论。

    Attributes:
        check: 检查名（width / fontsize / font-embedding / panel-labels / color-cycle）。
        status: ``"pass"`` / ``"fail"`` / ``"na"``（依据不足，不判 FAIL）。
        detail: 人类可读说明（含修法提示）。
    """

    check: str
    status: str
    detail: str


@dataclass(frozen=True)
class AuditReport:
    """一组自检结论 + 总判定。

    Attributes:
        findings: 逐条结论。
    """

    findings: tuple[Finding, ...]

    @property
    def passed(self) -> bool:
        """无 fail 即通过（na 不连坐）。"""
        return all(f.status != "fail" for f in self.findings)

    def format(self) -> str:
        """人类可读报告（CLI 输出用）。"""
        head = "PASS" if self.passed else "FAIL"
        lines = [f"[{head}]"]
        for f in self.findings:
            lines.append(f"  {f.status.upper():4s} {f.check}: {f.detail}")
        return "\n".join(lines)


def check_width(
    actual_mm: float | None,
    expected_mm: float | None,
    *,
    tol_mm: float = WIDTH_TOL_MM,
) -> Finding:
    """宽度契约判定（±tol_mm）。

    Args:
        actual_mm: 成品实测宽度（mm）；None → na（未测量）。
        expected_mm: 契约宽度（mm）；None → na（未给契约）。
        tol_mm: 容差。

    Returns:
        width 结论。
    """
    if actual_mm is None or expected_mm is None:
        return Finding("width", "na", "宽度依据不足（实测/契约缺一）")
    diff = abs(actual_mm - expected_mm)
    if diff <= tol_mm:
        return Finding(
            "width", "pass", f"实测 {actual_mm:.1f}mm vs 契约 {expected_mm:.1f}mm（±{tol_mm}）"
        )
    return Finding(
        "width",
        "fail",
        f"实测 {actual_mm:.1f}mm 偏离契约 {expected_mm:.1f}mm 达 {diff:.1f}mm"
        f"（容差 ±{tol_mm}；注意 tight bbox 会裁切，勿以名义 figsize 冒充）",
    )


def check_min_font(
    sizes_pt: Iterable[float], *, min_pt: float = MIN_FONT_PT
) -> Finding:
    """最小字号判定。

    Args:
        sizes_pt: 文本字号集合（pt）。
        min_pt: 下限。

    Returns:
        fontsize 结论（空集合 → na）。
    """
    sizes = [float(s) for s in sizes_pt]
    if not sizes:
        return Finding("fontsize", "na", "无文本对象（依据不足）")
    offenders = sorted({s for s in sizes if s < min_pt})
    if not offenders:
        return Finding("fontsize", "pass", f"最小字号 {min(sizes):.1f}pt ≥ {min_pt}pt")
    return Finding(
        "fontsize",
        "fail",
        f"字号低于 {min_pt}pt：{offenders}（移位/缩版式，不缩字号硬挤）",
    )


def check_pdf_fonttype(data: bytes) -> Finding:
    """PDF 字体嵌入判定（fonttype=42 → TrueType 软字体 ``/FontFile2``）。

    Args:
        data: PDF 文件字节。

    Returns:
        font-embedding 结论（无字体对象 → na）。
    """
    if b"/FontFile2" in data:
        return Finding("font-embedding", "pass", "PDF：TrueType 软字体已嵌入（FontFile2）")
    if b"/Font" not in data:
        return Finding("font-embedding", "na", "PDF 无字体对象（依据不足）")
    return Finding(
        "font-embedding",
        "fail",
        "PDF 字体未按 fonttype=42 嵌入（缺 FontFile2）——确认 pdf.fonttype=42 或经契约层导出",
    )


def check_eps_fonttype(data: bytes) -> Finding:
    """EPS 字体嵌入判定（fonttype=42 → Type 42 ``/sfnts``；Type3 位图 = 硬伤）。

    Args:
        data: EPS 文件字节。

    Returns:
        font-embedding 结论（无字体痕迹 → na）。
    """
    if b"/sfnts" in data:
        return Finding("font-embedding", "pass", "EPS：Type 42 软字体已嵌入（/sfnts）")
    if b"/CharStrings" in data:
        return Finding(
            "font-embedding",
            "fail",
            "EPS 为 Type3 位图字体（/CharStrings）——设 ps.fonttype=42 重新导出",
        )
    if b"%%BeginResource: font" not in data and b"/FontName" not in data:
        return Finding("font-embedding", "na", "EPS 无字体资源（依据不足）")
    return Finding("font-embedding", "fail", "EPS 字体资源未按 Type 42 嵌入（缺 /sfnts）")


def check_svg_text(data: bytes) -> Finding:
    """SVG 文字可编辑判定（``svg.fonttype=none`` → 保留 ``<text>`` 节点）。

    Args:
        data: SVG 文件字节。

    Returns:
        font-embedding 结论（无文字痕迹 → na）。
    """
    if b"<text" in data:
        return Finding("font-embedding", "pass", "SVG：<text> 节点保留（Illustrator/Inkscape 可改字）")
    if b"<path" in data or b"<g" in data:
        return Finding(
            "font-embedding",
            "fail",
            "SVG 文字已转路径——确认 svg.fonttype=none 重新导出",
        )
    return Finding("font-embedding", "na", "SVG 无文字/图形节点（依据不足）")


def check_panel_label_count(
    n_panels: int, label_texts: Iterable[str], *, n_expected: int | None = None
) -> Finding:
    """面板号计数判定：``(a)`` 形标号个数 == 面板轴个数。

    Args:
        n_panels: 面板轴个数（可见 + 携带数据；排除 colorbar 轴与
            ultraplot 图例/caption 自动轴）。
        label_texts: 图内全部文本（标号形如 ``"(a)"``）。
        n_expected: 期望面板数；None → 以 ``n_panels`` 为准。

    Returns:
        panel-labels 结论（两者皆 0 → na）。
    """
    expected = n_panels if n_expected is None else n_expected
    found = sorted(t for t in label_texts if _PANEL_LABEL_RE.match(t.strip()))
    if expected == 0 and not found:
        return Finding("panel-labels", "na", "无面板/无标号（依据不足）")
    if len(found) == expected:
        return Finding("panel-labels", "pass", f"标号 {found or '无'} 计数 == 面板数 {expected}")
    return Finding(
        "panel-labels",
        "fail",
        f"标号计数不符：面板 {expected} 个，标号 {len(found)} 个 {found}"
        "（panel_label_text 逐面板打标；colorbar/图例自动轴不算面板）",
    )


def _norm_hex(color: str) -> str:
    """归一化 hex 色：小写 + 补 ``#``。"""
    c = color.strip().lower()
    return c if c.startswith("#") else f"#{c}"


def check_color_cycle(
    colors: Iterable[str], *, expected: Sequence[str] = HOUSE_CYCLE
) -> Finding:
    """色循环判定：与家风循环（色盲安全）逐位一致。

    Args:
        colors: 当前 prop_cycle 色序列。
        expected: 期望序列（默认 :data:`~styles.HOUSE_CYCLE`）。

    Returns:
        color-cycle 结论（空序列 → na）。
    """
    got = [_norm_hex(c) for c in colors]
    if not got:
        return Finding("color-cycle", "na", "无色循环信息（依据不足）")
    want = [_norm_hex(c) for c in expected]
    if got == want:
        return Finding("color-cycle", "pass", f"色循环与家风 {len(want)} 色一致（Okabe-Ito 去黄）")
    return Finding(
        "color-cycle",
        "fail",
        f"色循环不符：{got} != {want}（铺 house.mplstyle；色板改家风须过色盲核验）",
    )


def audit_bytes(path: str | Path) -> list[Finding]:
    """按后缀对导出文件做字体嵌入判定（pdf/eps/svg；png → na）。

    Args:
        path: 导出文件路径。

    Returns:
        结论列表（未知后缀 → na 一条）。
    """
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".png":
        return [Finding("font-embedding", "na", "PNG 为位图，字体嵌入不适用")]
    data = p.read_bytes()
    if suffix == ".pdf":
        return [check_pdf_fonttype(data)]
    if suffix == ".eps":
        return [check_eps_fonttype(data)]
    if suffix == ".svg":
        return [check_svg_text(data)]
    return [Finding("font-embedding", "na", f"未知后缀 {suffix}，跳过")]


def audit_figure(
    fig: object,
    *,
    expected_width_mm: float | None = None,
    cycle_colors: Iterable[str] | None = None,
    min_font_pt: float = MIN_FONT_PT,
) -> AuditReport:
    """对 Figure 做轻量自检（鸭子类型，不 import matplotlib）。

    检查：宽度（get_size_inches 实测换算）、字号（非空 Text artist）、
    面板号计数（可见轴 - colorbar 轴）、色循环（可选传入）。

    Args:
        fig: Figure（``get_size_inches``/``findobj``/``axes``）。
        expected_width_mm: 契约宽度（mm）；None → width 记 na。
        cycle_colors: 当前 prop_cycle 色；None → color-cycle 记 na。
        min_font_pt: 最小字号。

    Returns:
        :class:`AuditReport`。
    """
    actual_mm = None
    try:
        w_in, _ = fig.get_size_inches()
        actual_mm = float(w_in) * 25.4
    except AttributeError:
        pass

    sizes: list[float] = []
    texts: list[str] = []
    for obj in fig.findobj():
        get_text = getattr(obj, "get_text", None)
        if get_text is None or obj.__class__.__name__ not in ("Text", "Annotation"):
            continue
        text = get_text()
        if not text:
            continue
        texts.append(text)
        get_size = getattr(obj, "get_fontsize", None)
        if get_size is not None:
            sizes.append(float(get_size()))

    n_panels = 0
    for ax in getattr(fig, "axes", ()):
        if not ax.get_visible() or ax.get_label() == "<colorbar>":
            continue
        has_data = getattr(ax, "has_data", None)
        if callable(has_data) and not has_data():
            continue  # 无数据轴（ultraplot 图例/caption 自动轴，不算面板）
        n_panels += 1

    findings = [
        check_width(actual_mm, expected_width_mm),
        check_min_font(sizes, min_pt=min_font_pt),
        check_panel_label_count(n_panels, texts),
        check_color_cycle(cycle_colors or ()),
    ]
    return AuditReport(tuple(findings))
