"""色盲友好调色板与颜色循环。

科研插图对配色有硬性要求：需在常见色觉障碍（红绿色盲 deuteranopia / protanopia、
蓝黄色盲 tritanopia）下仍可区分。这里内置两套业界标准的色盲安全调色板：

- **Okabe-Ito**（默认）：8 色，最广泛推荐的色盲安全方案。
- **Tol bright / Tol vibrant**：Paul Tol 设计的定性调色板，色彩更饱和。

``get_palette(name)`` 返回颜色（hex）列表，供 style.build_rcparams 组装 ``axes.prop_cycle``；
``color(i)`` 提供按索引取色的小工具，方便管线里显式指定“这条曲线用第 2 色”。

``register_palette(name, colors)`` 允许注册**自定义调色板**（如 ai_drawing 桥接从审美范本
抽取的配色），并**持久化**到 ``data/skills/scientific_plotting/palettes.json``，使其在后续的
``figures build``（独立进程）中也能被 ``get_palette(name)`` / STYLE.yaml 的 ``palette: <name>``
解析到。注意：自定义配色**不保证色盲安全**——``figures audit`` 仍会照常对其做色觉可读性告警。
"""

from __future__ import annotations

import json
from pathlib import Path

from pysci.paths import PLOTTING_ROOT

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
    _load_user_palettes()
    key = (name or DEFAULT_PALETTE).lower()
    return _PALETTES.get(key, OKABE_ITO)


def list_palettes() -> list[str]:
    """列出全部调色板名（去重后，供 CLI 展示）。"""
    _load_user_palettes()
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


# ---------------------------------------------------------------------------
# 自定义调色板注册（持久化）——供 ai_drawing 桥接把 AI 抽取的配色接入绘图管线
# ---------------------------------------------------------------------------
#: 用户/AI 调色板持久化文件（跨进程可见；``figures build`` 也会加载）。
_USER_PALETTE_FILE: Path = PLOTTING_ROOT / "palettes.json"
_user_loaded: bool = False

_HEX_DIGITS = set("0123456789abcdefABCDEF")


def _normalize_hex(c: str) -> str:
    """把一个颜色串规范成 ``#RRGGBB``（大写）。接受 ``#rgb``/``#rrggbb``/``rgb``/``rrggbb``。

    Raises:
        ValueError: 非法 hex。
    """
    s = str(c).strip()
    if s.startswith("#"):
        s = s[1:]
    if len(s) == 3 and all(ch in _HEX_DIGITS for ch in s):
        s = "".join(ch * 2 for ch in s)
    if len(s) != 6 or not all(ch in _HEX_DIGITS for ch in s):
        raise ValueError(f"非法 hex 颜色：{c!r}（应形如 #RRGGBB 或 #RGB）")
    return "#" + s.upper()


def _load_user_palettes() -> None:
    """惰性加载持久化的自定义调色板（仅一次；文件缺失/损坏则静默跳过）。"""
    global _user_loaded
    if _user_loaded:
        return
    _user_loaded = True
    try:
        if _USER_PALETTE_FILE.is_file():
            data = json.loads(_USER_PALETTE_FILE.read_text(encoding="utf-8"))
        else:
            return
    except (OSError, json.JSONDecodeError, ValueError):
        return
    if not isinstance(data, dict):
        return
    for k, v in data.items():
        if isinstance(v, list) and v:
            try:
                _PALETTES.setdefault(str(k).lower(), tuple(_normalize_hex(x) for x in v))
            except ValueError:
                continue


def register_palette(name: str, colors: list[str] | tuple[str, ...], *, persist: bool = True) -> tuple[str, ...]:
    """注册一个自定义调色板（如 AI 抽取的配色），并（默认）持久化到 ``palettes.json``。

    注册后即可用 ``get_palette(name)`` / ``color(i, name)`` 取色，或把 ``palette: <name>`` 写进
    某 figures 根的 ``STYLE.yaml``；因持久化，独立进程的 ``figures build`` 也能解析到。

    Args:
        name: 调色板名（大小写不敏感，内部小写存储）。
        colors: hex 颜色序列（如 ``["#1B2A4A", "#E0B050"]``）；至少 1 个，自动规范为 ``#RRGGBB``。
        persist: 是否写入 ``palettes.json``（默认 True；False 则仅本进程内可见）。

    Returns:
        规范化后的颜色元组。

    Raises:
        ValueError: ``name`` 为空 / ``colors`` 为空 / 含非法 hex。
    """
    key = (name or "").strip().lower()
    if not key:
        raise ValueError("register_palette：name 不能为空")
    if not colors:
        raise ValueError("register_palette：colors 不能为空")
    norm = tuple(_normalize_hex(c) for c in colors)
    _PALETTES[key] = norm
    if persist:
        _persist_user_palettes()
    return norm


def _persist_user_palettes() -> None:
    """把当前**非内置**调色板写回 ``palettes.json``（内置三套不落盘，保持文件精简）。"""
    builtin = {OKABE_ITO, TOL_BRIGHT, TOL_VIBRANT}
    user = {k: list(v) for k, v in _PALETTES.items() if v not in builtin}
    try:
        _USER_PALETTE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _USER_PALETTE_FILE.write_text(
            json.dumps(user, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except OSError:
        # 持久化失败不应阻断内存注册（调用方仍可本进程内使用）。
        pass
