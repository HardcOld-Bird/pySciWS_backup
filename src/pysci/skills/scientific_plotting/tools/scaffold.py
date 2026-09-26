"""脚手架：从内嵌模板新建一幅图的生产管线目录。

``scaffold_figure`` 在研究资产目录 ``data/research/<n>_<name>/article/figures/<slug>/`` 下
生成一份可直接运行的管线（``build_figure``）、一份 ``notes.md`` 迭代日志，并在 figures 根
放一份 ``STYLE.yaml``（若不存在）绑定默认期刊预设与宽度。生成后即可::

    pysci-figures build data/research/1_gain_ep/article/figures/<slug>

内置三套模板：
- ``single``              —— 单子图。
- ``multi_panel``         —— 2×2 多子图 + (a)(b)(c)(d) 角标。
- ``concept_plus_data``   —— 左概念图 / 右数据子图的嵌套布局。

模板以字符串内嵌于本模块（随代码版本化）；``data/skills/scientific_plotting/templates/``
保留给用户自定义模板的后续扩展。
"""

from __future__ import annotations

from pathlib import Path

from .runner import figures_root

# ---------------------------------------------------------------------------
# 模板：管线模块源码
# ---------------------------------------------------------------------------
# 模板以 raw 字符串书写：其中的反斜杠（mathtext 的 \sin / \mathrm）与花括号都按字面
# 写入生成的管线文件；仅 ``__SLUG__`` 占位符会被 .replace 替换为图目录名。
# （不用 str.format：模板正文含大量给生成代码用的 {}，会与 format 语法冲突。）
_TPL_SINGLE = r'''\
"""__SLUG__ — 单子图管线（pysci-figures 脚手架生成，按需修改）。

运行：pysci-figures build <此目录>
build_figure 会被 runner 在期刊 style_context 下调用，figsize/字体/配色/字号已生效。
可选入参：style(StyleInfo)、research_dir(figures 根目录)；其余 kwargs 会被安全忽略。
"""

import matplotlib.pyplot as plt
import numpy as np


def build_figure(style=None, research_dir=None, **kwargs):
    """构建并返回一个 matplotlib Figure。"""
    fig, ax = plt.subplots()

    # --- 示例数据（替换为真实仿真/实验数据）---
    x = np.linspace(0.0, 2.0 * np.pi, 200)
    ax.plot(x, np.sin(x), label=r"$\sin(x)$")

    # 正斜体约定：变量用 $...$ 斜体，单位用 \mathrm{} 正体
    ax.set_xlabel(r"$x$ / $\mathrm{rad}$")
    ax.set_ylabel(r"$y$ / $\mathrm{a.u.}$")
    ax.legend()
    return fig
'''

_TPL_MULTI_PANEL = r'''\
"""__SLUG__ — 2×2 多子图管线（pysci-figures 脚手架生成，按需修改）。"""

import matplotlib.pyplot as plt
import numpy as np

from pysci.skills.scientific_plotting.tools import layout, palette


def build_figure(style=None, research_dir=None, **kwargs):
    """构建 2×2 复合图并返回 Figure。"""
    fig, axes = layout.grid(2, 2)

    x = np.linspace(0.0, 2.0 * np.pi, 200)
    for i, ax in enumerate(layout.flatten_axes(axes)):
        ax.plot(x, np.sin(x + i * 0.5), color=palette.color(i + 1))
        ax.set_xlabel(r"$x$ / $\mathrm{rad}$")
        ax.set_ylabel(rf"$y_{{{i + 1}}}$")

    layout.label_panels(axes)  # 自动补 (a)(b)(c)(d)
    return fig
'''

_TPL_CONCEPT_PLUS_DATA = r'''\
"""__SLUG__ — 左概念图 / 右数据子图的嵌套管线（pysci-figures 脚手架生成）。"""

import matplotlib.pyplot as plt
import numpy as np

from pysci.skills.scientific_plotting.tools import layout


def build_figure(style=None, research_dir=None, **kwargs):
    """左：概念示意（可放外部媒体）；右：2×1 数据子图。"""
    fig = plt.figure()
    # 外层 1×2；左格 1×1（概念），右格 2×1（数据）
    left, right = layout.nested(fig, outer=(1, 2), inner=[(1, 1), (2, 1)],
                                outer_ratios=[1.0, 1.0])

    ax_concept = left if hasattr(left, "plot") else left.ravel()[0]
    # 概念图占位：实际可用 ax_concept.imshow(plt.imread(<blender 渲染 png>))
    ax_concept.set_title("concept")
    ax_concept.axis("off")

    x = np.linspace(0.0, 1.0, 100)
    for j, ax in enumerate(np.atleast_1d(right)):
        ax.plot(x, x ** (j + 1))
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")

    layout.label_panels([ax_concept, *np.atleast_1d(right).ravel()])
    return fig
'''

_TEMPLATES: dict[str, str] = {
    "single": _TPL_SINGLE,
    "multi_panel": _TPL_MULTI_PANEL,
    "concept_plus_data": _TPL_CONCEPT_PLUS_DATA,
}

#: 默认模板名。
DEFAULT_TEMPLATE = "multi_panel"

_TPL_NOTES = """\
# {slug} — 迭代日志

- **所属研究**：{research}
- **风格预设**：{style} / 宽度 {width}
- **产物**：`out/{slug}.eps`（投稿）、`.pdf`、`.svg`（人工微调）、`_preview.png`（Agent 视觉校验）

## 迭代记录

<!-- 每次修改在此追加：日期 / 需求（换数据|换画法|换风格）/ 结果 -->
"""

_TPL_STYLE_YAML = """\
# 本 figures 根目录的默认期刊风格绑定。
# 命令行 --style/--width 优先级高于此文件；单个图目录也可放自己的 STYLE.yaml 覆盖。
style: {style}
width: {width}
aspect: 0.618
palette: okabe-ito
"""


def list_templates() -> list[str]:
    """列出可用模板名。"""
    return list(_TEMPLATES)


def scaffold_figure(
    research: str,
    slug: str,
    *,
    style: str = "aps",
    width: str = "double",
    template: str = DEFAULT_TEMPLATE,
    figures_dir: Path | str | None = None,
    overwrite: bool = False,
) -> Path:
    """新建一幅图的生产管线目录。

    Args:
        research: 研究线名（不含数字前缀），如 ``gain_ep``。
        slug: 图目录名（如 ``fig1_ep_band``）。
        style: 默认期刊预设（写入 STYLE.yaml 与 notes）。
        width: 默认设计宽度（single/double）。
        template: 模板名（见 list_templates）。
        figures_dir: 显式指定 figures 根目录（默认由 research 解析）。
        overwrite: 目标已存在时是否覆盖管线文件。

    Returns:
        新建的图目录路径。

    Raises:
        KeyError: 模板名未知。
        FileExistsError: 目录已存在且未指定 overwrite。
    """
    if template not in _TEMPLATES:
        raise KeyError(
            f"未知模板 {template!r}；可选：{', '.join(_TEMPLATES)}"
        )
    root = Path(figures_dir) if figures_dir else figures_root(research)
    figdir = root / slug
    if figdir.exists() and not overwrite:
        raise FileExistsError(f"图目录已存在：{figdir}（加 overwrite=True 覆盖）")
    (figdir / "out").mkdir(parents=True, exist_ok=True)

    # 管线模块（仅替换 __SLUG__ 占位符，模板正文的花括号/反斜杠原样写入）
    pipeline = figdir / f"{slug}.py"
    pipeline.write_text(
        _TEMPLATES[template].replace("__SLUG__", slug), encoding="utf-8"
    )

    # 迭代日志
    notes = figdir / "notes.md"
    if not notes.exists() or overwrite:
        notes.write_text(
            _TPL_NOTES.format(slug=slug, research=research, style=style, width=width),
            encoding="utf-8",
        )

    # figures 根的风格绑定（不覆盖已有）
    style_yaml = root / "STYLE.yaml"
    if not style_yaml.exists():
        root.mkdir(parents=True, exist_ok=True)
        style_yaml.write_text(
            _TPL_STYLE_YAML.format(style=style, width=width), encoding="utf-8"
        )

    return figdir
