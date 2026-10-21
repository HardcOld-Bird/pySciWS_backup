"""``gain_ep.field_export`` 契约：键名单一真源 + 两代 field_schema 同源解析。

两代实况（F6，tb-3 R4）：

- ``field_8period_1m``：``geometry.inner_diamond``（单数 dict）、
  ``geometry.display_semicircle``（仅 ``max_inscribed_radius_m`` 与说明）；
- ``field_8period_1p5m``：``geometry.diamonds``（list）、**根级**
  ``display_semicircle``（``radius_m`` / ``center_m`` / 边距）。

本模块把二者归一为 :class:`FieldSchema`；别名只在 ``_LEGACY_*`` 集中登记，新导出
一律用正口径，**禁再增新同义键**（:func:`key_drift_warnings` 可机械点名）。消费端
（f2/f3 等）统一走 :func:`load_field_schema`。

新导出用 ``geometry.scan_regions``（**0.31 m 正方形 @ 近边 0.25 m**，用户口头反馈
2026-10-20 覆盖 twin A 的 60×31 矩形读法；见 ``scan_geometry.scan_geometry_block``）；
1m/1.5m 历史 schema 的 ``diamonds``/``inner_diamond`` 继续可读。

``files.*.display_y_positive_max_abs_real_p_t`` 的语义是**严格 y>0 气域**（排除
y=0 口径线行与管槽），导出端一律用 :func:`display_declaration`（与 `vmax-check`
同源）；不得再写 y≥0 口径。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from pysci.skills.scientific_plotting.tools.field import display_region_max

SCHEMA_KIND = "gain_ep.field_export"
#: ``files.*`` 的显示区声明：严格 y>0 气域 max|Re p_t|（与 `vmax-check` 同源）。
KEY_DISPLAY_MAX = "display_y_positive_max_abs_real_p_t"
KEY_DISPLAY_POINTS = "display_points"
KEY_GEOMETRY = "geometry"
KEY_DIAMONDS = "diamonds"
KEY_DISPLAY_SEMICIRCLE = "display_semicircle"
#: 扫场区（用户口头反馈 2026-10-20：0.31 m 正方形 @ 近边 0.25 m）。
KEY_SCAN_REGIONS = "scan_regions"

#: 历史别名（只为两代现存文件保留；新导出禁止使用）。
_LEGACY_DIAMOND_KEYS = ("inner_diamond", "inner_diamonds")
_LEGACY_SEMICIRCLE_SCOPES = ("geometry", "root")


@dataclass(frozen=True)
class DiamondSpec:
    """菱形/扫描区读数（归一后）。``radial_deg`` 在单菱形旧 schema 中可能缺失。"""

    radial_deg: float | None
    center_m: tuple[float, float] | None
    vertices_m: tuple[tuple[float, float], ...]
    side_m: float | None
    source_key: str
    raw: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)


@dataclass(frozen=True)
class ScanRegionSpec:
    """扫场区读数（正方形口径；旧 schema 无此键时为空）。"""

    radial_deg: float | None
    shape: str
    radial_m: float | None
    transverse_m: float | None
    near_edge_m: float | None
    center_m: tuple[float, float] | None
    vertices_m: tuple[tuple[float, float], ...]
    source_key: str
    raw: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)


@dataclass(frozen=True)
class SemicircleSpec:
    """显示半圆读数（``max_inscribed_radius_m`` 归一为 ``radius_m``）。"""

    radius_m: float | None
    center_m: tuple[float, float] | None
    source_key: str
    note: str = ""
    raw: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)


@dataclass(frozen=True)
class FieldSchema:
    kind: str
    version: int
    diamonds: tuple[DiamondSpec, ...]
    scan_regions: tuple[ScanRegionSpec, ...]
    semicircle: SemicircleSpec | None
    raw: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)


def _pair(value: Any) -> tuple[float, float] | None:
    try:
        pair = [float(v) for v in value]
    except (TypeError, ValueError):
        return None
    return (pair[0], pair[1]) if len(pair) == 2 else None


def _vertices(value: Any) -> tuple[tuple[float, float], ...]:
    out: list[tuple[float, float]] = []
    for item in value or []:
        pair = _pair(item)
        if pair is not None:
            out.append(pair)
    return tuple(out)


def _side_from_vertices(vertices: tuple[tuple[float, float], ...]) -> float | None:
    if len(vertices) != 4:
        return None
    pts = np.asarray(vertices, dtype=float)
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    ordered = pts[np.argsort(ang)]
    edges = np.linalg.norm(np.roll(ordered, -1, axis=0) - ordered, axis=1)
    if not np.allclose(edges, edges.mean(), rtol=1e-6, atol=1e-9):
        return None
    return float(edges.mean())


def _diamond(doc: dict[str, Any], source_key: str) -> DiamondSpec:
    vertices = _vertices(doc.get("vertices_m"))
    side = doc.get("side_m")
    side_f = float(side) if side is not None else _side_from_vertices(vertices)
    radial = doc.get("radial_deg")
    return DiamondSpec(
        radial_deg=float(radial) if radial is not None else None,
        center_m=_pair(doc.get("center_m")),
        vertices_m=vertices,
        side_m=side_f,
        source_key=source_key,
        raw=dict(doc),
    )


def _as_diamond_docs(value: Any) -> list[Any]:
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list):
        return [v for v in value if isinstance(v, dict)]
    return []


def _scan_region(doc: dict[str, Any], source_key: str) -> ScanRegionSpec:
    def _f(key: str) -> float | None:
        value = doc.get(key)
        return float(value) if value is not None else None

    radial = doc.get("radial_deg")
    return ScanRegionSpec(
        radial_deg=float(radial) if radial is not None else None,
        shape=str(doc.get("shape") or ""),
        radial_m=_f("radial_m"),
        transverse_m=_f("transverse_m"),
        near_edge_m=_f("near_edge_m"),
        center_m=_pair(doc.get("center_m")),
        vertices_m=_vertices(doc.get("vertices_m")),
        source_key=source_key,
        raw=dict(doc),
    )


def _semicircle(doc: dict[str, Any] | None, source_key: str) -> SemicircleSpec | None:
    if not isinstance(doc, dict):
        return None
    radius = doc.get("radius_m", doc.get("max_inscribed_radius_m"))
    return SemicircleSpec(
        radius_m=float(radius) if radius is not None else None,
        center_m=_pair(doc.get("center_m")),
        source_key=source_key,
        note=str(doc.get("note") or ""),
        raw=dict(doc),
    )


def parse_field_schema(doc: dict[str, Any]) -> FieldSchema:
    """把任一代 ``gain_ep.field_export`` schema 归一为 :class:`FieldSchema`。

    正口径优先（``geometry.diamonds`` / 根级 ``display_semicircle``）；旧别名仅在
    正口径缺失时兜底。两代皆无菱形/半圆读数时对应字段为空，不臆造。
    """
    if not isinstance(doc, dict):
        raise ValueError("field_schema 须为映射")
    geom = doc.get(KEY_GEOMETRY)
    geom = geom if isinstance(geom, dict) else {}

    diamonds: list[DiamondSpec] = []
    if isinstance(geom.get(KEY_DIAMONDS), list):
        diamonds = [
            _diamond(d, f"{KEY_GEOMETRY}.{KEY_DIAMONDS}")
            for d in _as_diamond_docs(geom[KEY_DIAMONDS])
        ]
    else:
        for key in _LEGACY_DIAMOND_KEYS:
            if key in geom:
                diamonds = [
                    _diamond(d, f"{KEY_GEOMETRY}.{key}")
                    for d in _as_diamond_docs(geom[key])
                ]
                break

    semicircle = _semicircle(doc.get(KEY_DISPLAY_SEMICIRCLE), KEY_DISPLAY_SEMICIRCLE)
    if semicircle is None:
        semicircle = _semicircle(
            geom.get(KEY_DISPLAY_SEMICIRCLE), f"{KEY_GEOMETRY}.{KEY_DISPLAY_SEMICIRCLE}"
        )

    scan_regions: tuple[ScanRegionSpec, ...] = ()
    if isinstance(geom.get(KEY_SCAN_REGIONS), list):
        scan_regions = tuple(
            _scan_region(d, f"{KEY_GEOMETRY}.{KEY_SCAN_REGIONS}")
            for d in _as_diamond_docs(geom[KEY_SCAN_REGIONS])
        )

    return FieldSchema(
        kind=str(doc.get("kind") or ""),
        version=int(doc.get("version") or 0),
        diamonds=tuple(diamonds),
        scan_regions=scan_regions,
        semicircle=semicircle,
        raw=doc,
    )


def load_field_schema(path: str | Path) -> FieldSchema:
    """读 ``field_schema.json`` 并归一解析。"""
    p = Path(path)
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{p} 不是合法 JSON：{exc}") from exc
    return parse_field_schema(doc)


def display_declaration(
    xy: np.ndarray,
    values: np.ndarray,
    *,
    center: tuple[float, float] = (0.0, 0.0),
    radius: float | None = None,
) -> dict[str, Any]:
    """``files.*`` 的显示区声明（严格 y>0 气域；与 ``vmax-check`` 同源）。

    导出端用它写 ``display_y_positive_max_abs_real_p_t`` / ``display_points``，
    禁止另写一套 y≥0 口径（F9）。
    """
    stats = display_region_max(xy, values, center=center, radius=radius)
    return {
        KEY_DISPLAY_MAX: stats.max_abs,
        KEY_DISPLAY_POINTS: stats.n_points,
    }


def recompute_display_declarations(
    schema_path: str | Path,
    *,
    csv_dir: str | Path | None = None,
) -> dict[str, dict[str, Any]]:
    """把 schema 的 ``files.*`` 显示区声明按**严格 y>0** 重算并落盘（幂等）。

    显示区＝根级 ``display_semicircle``（半径/圆心）；每个 ``field_<key>.csv`` 经
    :func:`display_declaration`（＝``vmax-check`` 同源）重算
    ``display_y_positive_max_abs_real_p_t`` / ``display_points``。无变更不写盘。

    Args:
        schema_path: ``field_schema.json`` 路径。
        csv_dir: CSV 目录（缺省＝ schema 同目录）。

    Returns:
        ``{series: {old, new, old_points, new_points}}``（只含实际变更的 series）。

    Raises:
        ValueError: schema 缺 ``display_semicircle`` 或缺 ``files``。
    """
    path = Path(schema_path)
    doc = json.loads(path.read_text(encoding="utf-8"))
    parsed = parse_field_schema(doc)
    if parsed.semicircle is None or parsed.semicircle.radius_m is None:
        raise ValueError(
            f"{path} 缺根级 display_semicircle（radius/center），无法定显示区"
        )
    center = parsed.semicircle.center_m or (0.0, 0.0)
    radius = float(parsed.semicircle.radius_m)
    files = doc.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError(f"{path} 缺 files 段")
    directory = Path(csv_dir) if csv_dir is not None else path.parent

    from pysci.skills.scientific_plotting.tools.field import (  # noqa: PLC0415 - 仅重算需要
        load_field_points,
    )

    summary: dict[str, dict[str, Any]] = {}
    for key in sorted(files):
        xy, vals = load_field_points(directory / f"field_{key}.csv")
        decl = display_declaration(xy, vals, center=center, radius=radius)
        entry = files.setdefault(key, {})
        old_max = entry.get(KEY_DISPLAY_MAX)
        old_points = entry.get(KEY_DISPLAY_POINTS)
        if old_max == decl[KEY_DISPLAY_MAX] and old_points == decl[KEY_DISPLAY_POINTS]:
            continue
        entry[KEY_DISPLAY_MAX] = decl[KEY_DISPLAY_MAX]
        entry[KEY_DISPLAY_POINTS] = decl[KEY_DISPLAY_POINTS]
        summary[key] = {
            "old": old_max,
            "new": decl[KEY_DISPLAY_MAX],
            "old_points": old_points,
            "new_points": decl[KEY_DISPLAY_POINTS],
        }
    if summary:
        path.write_text(
            json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return summary


def key_drift_warnings(doc: dict[str, Any]) -> list[str]:
    """点名**新同义键**（geometry/根级里含 diamond/semicircle 但不在允许集）。"""
    allowed = {KEY_DIAMONDS, KEY_DISPLAY_SEMICIRCLE, *_LEGACY_DIAMOND_KEYS}
    out: list[str] = []
    for scope, container in (("geometry", doc.get(KEY_GEOMETRY)), ("root", doc)):
        if not isinstance(container, dict):
            continue
        for key in container:
            low = str(key).lower()
            if ("diamond" in low or "semicircle" in low) and key not in allowed:
                out.append(f"{scope}.{key}")
    return out


def schema_contract() -> dict[str, Any]:
    """契约摘要（导出/加载/消费端共读；新导出只用正口径）。"""
    return {
        "kind": SCHEMA_KIND,
        "geometry": {
            KEY_DIAMONDS: "list[Diamond]（1m/1.5m 历史 schema）",
            KEY_SCAN_REGIONS: "list[ScanRegion]（R6 矩形口径，新导出正口径）",
            "display_semicircle": "根级对象（正口径）",
        },
        "scan_region": {
            "keys": [
                "radial_deg",
                "shape",
                "radial_m",
                "transverse_m",
                "near_edge_m",
                "edges_aligned_radial",
                "center_m",
                "vertices_m",
            ],
            "source": "pysci.research.gain_ep.scan_geometry.scan_geometry_block",
            "caliber": (
                "用户口头反馈 2026-10-20：0.31 m 正方形 @ 近边 0.25 m（远边 0.56 m），"
                "菱形永远正方形；覆盖 twin A 的 60×31 矩形读法与 R1 旧近边 30 cm"
            ),
        },
        "aliases": {
            f"{KEY_GEOMETRY}.inner_diamond": "单菱形旧 schema（1m）",
            f"{KEY_GEOMETRY}.inner_diamonds": "list 别名（历史）",
            f"{KEY_GEOMETRY}.display_semicircle": "半圆旧位置（1m）",
        },
        "files": {
            KEY_DISPLAY_MAX: (
                "严格 y>0 气域实测 max|Re p_t|（排除 y=0 口径线行与管槽；"
                "由 field.display_region_max 计算，与 `vmax-check` 同源）"
            ),
            KEY_DISPLAY_POINTS: "参与计算的显示区点数",
        },
        "normalized": {
            "diamond": ["radial_deg", "center_m", "vertices_m", "side_m"],
            "scan_region": [
                "radial_deg",
                "radial_m",
                "transverse_m",
                "near_edge_m",
                "center_m",
                "vertices_m",
            ],
            "semicircle": ["radius_m", "center_m", "note"],
        },
        "policy": "新导出只用正口径；解析器只为两代现存文件保留别名，禁再增新同义键",
    }
