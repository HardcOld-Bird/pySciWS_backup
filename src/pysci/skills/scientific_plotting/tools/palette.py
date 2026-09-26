"""色盲友好调色板与颜色循环。

科研插图对配色有硬性要求：需在常见色觉障碍（红绿色盲 deuteranopia / protanopia、
蓝黄色盲 tritanopia）下仍可区分。这里内置两套业界标准的色盲安全调色板：

- **Okabe-Ito**（默认）：8 色，最广泛推荐的色盲安全方案。
- **Tol bright / Tol vibrant**：Paul Tol 设计的定性调色板，色彩更饱和。

``get_palette(name)`` 返回颜色（hex）列表，供 style.build_rcparams 组装 ``axes.prop_cycle``；
``color(i)`` 提供按索引取色的小工具，方便管线里显式指定“这条曲线用第 2 色”。
"""

from __future__ import annotations

# Okabe & Ito (2008) "Color Universal Design" —— 色盲安全定性调色板。
# 顺序：黑、橙、天蓝、蓝绿、黄、蓝、朱红、红紫。
OKABE_ITO: tuple[str, ...] = (
    "#000000",  # black
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
)

# Paul Tol "bright" 定性方案。
TOL_BRIGHT: tuple[str, ...] = (
    "#4477AA",  # blue
    "#EE6677",  # red
    "#228833",  # green
    "#CCBB44",  # yellow
    "#66CCEE",  # cyan
    "#AA3377",  # purple
    "#BBBBBB",  # grey
)

# Paul Tol "vibrant" 定性方案。
TOL_VIBRANT: tuple[str, ...] = (
    "#EE7733",  # orange
    "#0077BB",  # blue
    "#33BBEE",  # cyan
    "#EE3377",  # magenta
    "#CC3311",  # red
    "#009988",  # teal
    "#EEEEEE",  # grey
)

_PALETTES: dict[str, tuple[str, ...]] = {
    "okabe-ito": OKABE_ITO,
    "okabe": OKABE_ITO,
    "tol-bright": TOL_BRIGHT,
    "tol": TOL_BRIGHT,
    "tol-vibrant": TOL_VIBRANT,
    "vibrant": TOL_VIBRANT,
}

#: 默认调色板名（style.build_rcparams 的缺省值）。
DEFAULT_PALETTE = "okabe-ito"


def get_palette(name: str = DEFAULT_PALETTE) -> tuple[str, ...]:
    """按名取调色板；未知名回退到默认（Okabe-Ito）而不报错。"""
    key = (name or DEFAULT_PALETTE).lower()
    return _PALETTES.get(key, OKABE_ITO)


def list_palettes() -> list[str]:
    """列出全部调色板名（去重后，供 CLI 展示）。"""
    seen: list[str] = []
    for k, v in _PALETTES.items():
        if v not in [ _PALETTES[s] for s in seen ]:
            seen.append(k)
    return seen


def color(name_or_index: int | str = 0, palette: str = DEFAULT_PALETTE) -> str:
    """取某调色板的第 i 个颜色（越界则循环）。便于管线显式指定曲线颜色。"""
    pal = get_palette(palette)
    if isinstance(name_or_index, str):
        # 允许直接传 hex
        return name_or_index if name_or_index.startswith("#") else pal[0]
    return pal[name_or_index % len(pal)]
