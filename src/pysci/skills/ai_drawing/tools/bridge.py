"""审美参考 → 数据复现桥（工作流 A 的杀手锏）。

AI 生成的图**好看但数据/文字不可信**。本模块把一张审美范本"翻译"成 scientific_plotting 的
生产管线骨架：

1. 抽取范本的**主色板**（复用 :func:`postprocess.extract_palette`，Pillow 中位切色彩量化）+
   **尺寸/长宽比/朝向/亮度**等构图线索；
2. 调 ``scaffold_figure(research, slug, ...)`` 脚手架出一幅图的生产管线（代码 + 数据目录）；
3. 把范本图复制进图目录，并写 ``design_spec.md``（内嵌范本 + hex 配色 + 布局提示 + 下一步）；
4. 可选：把配色注册进 scientific_plotting 的调色板并绑定到该图目录的 ``STYLE.yaml``。

产出的是**待填真实数据**的骨架，不是最终图：随后在管线里填真实数据 → ``figures build`` →
``Read`` 预览对照范本迭代（视觉闭环）。

.. note::
   对 scientific_plotting 的跨技能依赖（``scaffold`` / ``palette`` / ``runner``）都在函数内
   **惰性导入**，以保持 ai_drawing 可独立导入、离线单测无需触发绘图栈。
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from PIL import Image

from pysci.paths import PROJECT_ROOT, assert_within_data

from . import postprocess as _pp

#: design_spec.md 里 STYLE.yaml 的默认长宽比（黄金比，与 scientific_plotting 一致）。
#: 期刊规范优先于范本比例——范本的实际比例仅作**提示**，不强行写进图。
_DEFAULT_ASPECT = 0.618


# ---------------------------------------------------------------------------
# 范本分析
# ---------------------------------------------------------------------------
def analyze_reference(src: str | Path, *, n_colors: int = 6) -> dict[str, Any]:
    """分析一张审美范本：尺寸/长宽比/朝向/平均亮度 + 主色板（hex，按占比降序）。

    Args:
        src: 范本图路径。
        n_colors: 抽取主色数量（1-16）。

    Returns:
        dict：``path/width/height/aspect(w÷h)/aspect_hw(h÷w)/orientation/mean_brightness/palette``。

    Raises:
        FileNotFoundError: 范本不存在。
    """
    p = Path(src).expanduser()
    if not p.is_file():
        raise FileNotFoundError(f"审美范本不存在：{p}")
    with Image.open(p) as img:
        w, h = img.size
        gray = img.convert("L")
        gray.thumbnail((64, 64))
        pixels = list(gray.getdata())
        mean_brightness = round(sum(pixels) / max(1, len(pixels)), 1)
    if w > h:
        orientation = "landscape"
    elif h > w:
        orientation = "portrait"
    else:
        orientation = "square"
    return {
        "path": str(p),
        "width": w,
        "height": h,
        "aspect": round(w / h, 4),
        "aspect_hw": round(h / w, 4),
        "orientation": orientation,
        "mean_brightness": mean_brightness,
        "palette": _pp.extract_palette(p, n_colors),
    }


# ---------------------------------------------------------------------------
# design_spec.md 渲染
# ---------------------------------------------------------------------------
def _rel(path: Path) -> str:
    """尽量给出相对项目根的 POSIX 路径（便于 design_spec.md 里的可复制命令）。"""
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _render_design_spec(
    *,
    slug: str,
    research: str,
    info: dict[str, Any],
    figdir: Path,
    pipeline: Path,
    ref_copy: Path | None,
    style: str,
    width: str,
    template: str,
    palette_name: str | None,
) -> str:
    pal = info["palette"]
    orient_hint = {
        "landscape": "范本为横构图——适合并排多子图（multi_panel）或宽幅概念图。",
        "portrait": "范本为竖构图——适合堆叠子图或纵向示意。",
        "square": "范本近方形——适合单子图或对称布局。",
    }[info["orientation"]]
    bright_hint = (
        "偏亮背景——注意深色文字/曲线的对比度。"
        if info["mean_brightness"] > 128
        else "偏暗背景——若用于投稿，考虑反转为浅色背景以省墨并提升可读性。"
    )
    L: list[str] = []
    L.append(f"# {slug} — 设计规格（AI 审美范本 → 数据复现）")
    L.append("")
    L.append("> 由 `imagine bridge` 自动抽取。**范本是 AI 生成的：视觉可借鉴，但其中的数据、坐标、")
    L.append("> 文字一律不可信**。请只借用其**配色 / 构图 / 长宽比**，在管线里用**真实数据**复现。")
    L.append("")
    L.append("## 审美范本")
    if ref_copy is not None:
        L.append(f"![reference]({ref_copy.name})")
        L.append("")
    L.append(f"- 源文件：`{info['path']}`")
    L.append(f"- 尺寸：{info['width']}×{info['height']} px（朝向：{info['orientation']}）")
    L.append(f"- 长宽比 w/h：{info['aspect']}（h/w：{info['aspect_hw']}）")
    L.append(f"- 平均亮度：{info['mean_brightness']}/255")
    L.append("")
    L.append(f"## 抽取的主色（{len(pal)} 色，按占比降序）")
    L.append("")
    L.append("| # | hex | 近似角色 |")
    L.append("|---|-----|---------|")
    for i, hx in enumerate(pal):
        role = "主色 / 背景基调" if i == 0 else f"强调色 {i}"
        L.append(f"| {i + 1} | `{hx}` | {role} |")
    L.append("")
    L.append("**在管线里用这些颜色**（`palette.color()` 接受直接传 hex）：")
    L.append("")
    L.append("```python")
    L.append("from pysci.skills.scientific_plotting.tools import palette")
    if pal:
        L.append(f'ax.plot(x, y, color=palette.color("{pal[0]}"))   # 主色')
        if len(pal) > 1:
            L.append(f'ax.plot(x, y2, color=palette.color("{pal[1]}"))  # 强调色 1')
    L.append("```")
    if palette_name:
        L.append("")
        L.append(
            f"本范本配色已注册为调色板 **`{palette_name}`**（持久化到 "
            "`data/skills/scientific_plotting/palettes.json`），并写进本图目录的 "
            f"`STYLE.yaml`（`palette: {palette_name}`）。也可显式取色："
        )
        L.append("")
        L.append("```python")
        L.append(f'palette.color(0, "{palette_name}")  # 取注册调色板第 0 色')
        L.append("```")
        L.append("")
        L.append(
            "> 注意：AI 配色**不保证色盲安全**；`pysci-figures audit` 会照常告警。"
            "重要投稿图请酌情换回 Okabe-Ito。"
        )
    L.append("")
    L.append("## 布局提示")
    L.append("")
    L.append(f"- 风格预设：`{style}` / 宽度 `{width}`（已写入脚手架；期刊规范优先于范本比例）。")
    L.append(f"- 构图：{orient_hint}")
    L.append(f"- 明暗：{bright_hint}")
    L.append(f"- 脚手架模板：`{template}`（换布局：编辑管线，或 `pysci-figures new --template ...`）。")
    L.append("")
    L.append("## 下一步（视觉闭环）")
    L.append("")
    L.append(f"1. 编辑管线 `{_rel(pipeline)}`，把模板示例**替换为真实数据**。")
    L.append(f"2. `uv run pysci-figures build {_rel(figdir)}`")
    L.append(f"3. `Read` 预览 `{_rel(figdir)}/out/{slug}_preview.png`，对照本范本迭代（配色/构图逼近，数据用真的）。")
    L.append("")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------
# bridge：脚手架 + design_spec.md（+ 可选调色板注册）
# ---------------------------------------------------------------------------
def _write_figdir_style(
    figdir: Path, *, style: str, width: str, palette_name: str, aspect: float = _DEFAULT_ASPECT
) -> Path:
    """在图目录写一份 STYLE.yaml，绑定注册好的调色板（figdir 级优先于 figures 根级）。"""
    p = figdir / "STYLE.yaml"
    p.write_text(
        "# 由 ai_drawing bridge 生成：绑定从审美范本抽取的配色。\n"
        "# 命令行 --palette 优先级更高；删除本文件即回退到 figures 根的 STYLE.yaml。\n"
        f"style: {style}\n"
        f"width: {width}\n"
        f"aspect: {aspect}\n"
        f"palette: {palette_name}\n",
        encoding="utf-8",
    )
    return p


def bridge(
    ref: str | Path,
    research: str,
    slug: str,
    *,
    style: str = "aps",
    width: str = "double",
    template: str | None = None,
    n_colors: int = 6,
    palette_name: str | None = None,
    copy_ref: bool = True,
    overwrite: bool = False,
) -> dict[str, Any]:
    """把一张审美范本桥接成 scientific_plotting 的生产管线骨架（工作流 A）。

    Args:
        ref: 审美范本图路径（AI 生成或任意图）。
        research: 研究线名（不含数字前缀），如 ``gain_ep``。
        slug: 图目录名，如 ``fig1_cover``。
        style: 期刊风格预设（写入脚手架 STYLE.yaml）。
        width: 设计宽度（single/double 或毫米数）。
        template: 脚手架模板名（None → scientific_plotting 默认 ``multi_panel``）。
        n_colors: 抽取主色数量。
        palette_name: 给则把抽取的配色注册为该名调色板（持久化）并绑定到图目录 STYLE.yaml。
        copy_ref: 是否把范本图复制进图目录（供 design_spec.md 内嵌 + 对照）。
        overwrite: 图目录/管线已存在时是否覆盖（透传 scaffold_figure）。

    Returns:
        dict：``figdir / pipeline / design_spec / ref_copy / palette_name / reference``。

    Raises:
        FileNotFoundError: 范本不存在。
        KeyError: 模板名未知（透传 scaffold_figure）。
    """
    # 惰性导入 scientific_plotting（跨技能；保持 ai_drawing 可独立导入）
    from pysci.skills.scientific_plotting.tools import scaffold as _scaffold
    from pysci.skills.scientific_plotting.tools.runner import figures_code_root

    ref = Path(ref).expanduser()
    info = analyze_reference(ref, n_colors=n_colors)

    tmpl = template or _scaffold.DEFAULT_TEMPLATE
    figdir = _scaffold.scaffold_figure(
        research, slug, style=style, width=width, template=tmpl, overwrite=overwrite
    )
    assert_within_data(figdir, what="桥接图目录")
    pipeline = figures_code_root(research) / f"{slug}.py"

    ref_copy: Path | None = None
    if copy_ref:
        ref_copy = figdir / f"_reference{ref.suffix.lower() or '.png'}"
        shutil.copy2(ref, ref_copy)

    registered: str | None = None
    if palette_name:
        from pysci.skills.scientific_plotting.tools import palette as _palette

        _palette.register_palette(palette_name, info["palette"])
        _write_figdir_style(figdir, style=style, width=width, palette_name=palette_name)
        registered = palette_name

    spec = figdir / "design_spec.md"
    spec.write_text(
        _render_design_spec(
            slug=slug,
            research=research,
            info=info,
            figdir=figdir,
            pipeline=pipeline,
            ref_copy=ref_copy,
            style=style,
            width=width,
            template=tmpl,
            palette_name=registered,
        ),
        encoding="utf-8",
    )

    return {
        "figdir": figdir,
        "pipeline": pipeline,
        "design_spec": spec,
        "ref_copy": ref_copy,
        "palette_name": registered,
        "reference": info,
    }
