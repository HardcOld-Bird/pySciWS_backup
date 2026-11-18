"""多子图面板标号 (a)(b)(c)（纯文本生成可测；轴上放置懒用 matplotlib 鸭子类型）。

工作流位（SKILL.md 第 4 步）：多子图面板统一角标。排版约定（references/
export-acceptance.md）：标号锚定**面板轴左上角外侧**（整体 bbox 含图例/colorbar 时
以整体左上角为编号锚点），与面板内容零遮挡。
"""

from __future__ import annotations

#: 标号风格：``paren`` = "(a)"（期刊默认）；``plain`` = "a"；``paren_after`` = "a)"。
LABEL_STYLES: tuple[str, ...] = ("paren", "plain", "paren_after")

_ALPHABET = "abcdefghijklmnopqrstuvwxyz"


def panel_label_text(index: int, style: str = "paren") -> str:
    """生成第 ``index`` 个面板的标号文本（纯函数）。

    Args:
        index: 面板序号（0-based；0 → "a"，25 → "z"）。
        style: :data:`LABEL_STYLES` 之一。

    Returns:
        标号文本，如 ``"(a)"``。

    Raises:
        ValueError: 序号越界（仅支持 0–25）或风格未知。
    """
    if not 0 <= index < len(_ALPHABET):
        raise ValueError(f"面板序号越界：{index}（支持 0–{len(_ALPHABET) - 1}）")
    letter = _ALPHABET[index]
    if style == "paren":
        return f"({letter})"
    if style == "plain":
        return letter
    if style == "paren_after":
        return f"{letter})"
    raise ValueError(f"未知标号风格：{style!r}（可用：{', '.join(LABEL_STYLES)}）")


def label_panel(
    ax: object,
    index: int,
    *,
    style: str = "paren",
    x: float = -0.02,
    y: float = 1.02,
    **text_kwargs: object,
) -> object:
    """给面板轴打角标（鸭子类型，不 import matplotlib）。

    默认位置：轴坐标系左上角外侧（``x=-0.02, y=1.02``，ha=right / va=bottom），
    粗体——多子图期刊惯例。

    Args:
        ax: 有 ``text``/``transAxes`` 的 Axes。
        index: 面板序号（见 :func:`panel_label_text`）。
        style: 标号风格。
        x / y: 标号锚点（轴坐标系，可按版式微调）。
        **text_kwargs: 透传给 ``ax.text``（如 ``fontsize``）。

    Returns:
        创建的 Text artist。
    """
    label = panel_label_text(index, style=style)
    opts: dict[str, object] = {
        "transform": ax.transAxes,
        "fontweight": "bold",
        "ha": "right",
        "va": "bottom",
        "clip_on": False,
    }
    opts.update(text_kwargs)
    return ax.text(x, y, label, **opts)
