"""远场雷达数据导出：1° 网格 × 三组批量落盘（CSV/MAT + schema）。

sim1 从 COMSOL 导出远场（角度 + ``|p_ext|``）后，用本模块把三组数据统一到
0–360°、1°（361 点）网格，批量写 CSV/MAT 并附 ``schema.json``；f1 直接读输出。
既有 ``raw/far_<cfg>_<inc>.csv``（0–180°、181 点）仍可按 ``full_circle=False``
读取，供旧图对照。

COMSOL 表口径（与 ``beam_scatter/raw/far_*.csv`` 一致）：``x, y, angle_deg,
|p_ext|``，行首 ``%`` 注释。

导出→读取闭环：宽表用 :func:`load_radar_csv`（``(angle, {series: amp})``），
COMSOL/纯数值表用 :func:`load_comsol_csv`（注释与表头自动跳过）。
"""

from __future__ import annotations

import io
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_POINTS = 361
DEFAULT_STEP_DEG = 1.0
ANGLE_START = 0.0
ANGLE_STOP = 360.0
VALUE_NAME = "abs_p_ext"
VALUE_UNIT = "Pa"

#: 本轮三组（b1 口径）与旧图 cfg 键的映射。
LEGACY_CASE_KEYS: dict[str, str] = {
    "full": "groove_gain",
    "nogain": "groove_nogain",
    "nostruct": "gain_nogroove",
}
CASE_LABELS: dict[str, str] = {
    "groove_gain": "groove + gain",
    "groove_nogain": "groove, no gain",
    "gain_nogroove": "gain, no groove",
}


@dataclass(frozen=True)
class RadarSeries:
    """一条远场曲线（单入射/通道）。"""

    name: str
    angles_deg: Sequence[float]
    amplitude_pa: Sequence[float]


@dataclass(frozen=True)
class RadarGroup:
    """一组对照（一个物理情形），含一条或多条曲线。"""

    key: str
    series: tuple[RadarSeries, ...]
    label: str = ""


def _numeric_rows(path: str | Path) -> list[str]:
    """跳过空行、``%`` 注释与表头，返回从首个数值行起的原始行。

    表头判定：首个非注释行的首列不是数字（如本模块导出的 ``angle_deg,...``）。
    数据区中间再出现非数值行仍交给 ``np.loadtxt`` 报错——不静默吞坏行。
    """
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    rows: list[str] = []
    started = False
    for line in lines:
        text = line.strip()
        if not text or text.startswith("%"):
            continue
        if not started:
            try:
                float(text.split(",", 1)[0].strip())
            except ValueError:
                continue  # 表头
            started = True
        rows.append(line)
    if not rows:
        raise ValueError(f"{path} 无数据行")
    return rows


def load_comsol_csv(
    path: str | Path, *, angle_col: int = 2, value_col: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    """读数值表 -> (角度 deg, |p_ext| Pa)。

    ``%`` 注释与**表头行**自动跳过（本模块导出的宽表 ``angle_deg,L,R`` 也可用
    ``angle_col=0, value_col=1`` 直接读）；更自然的宽表读法见 :func:`load_radar_csv`。
    """
    arr = np.loadtxt(
        io.StringIO("\n".join(_numeric_rows(path))), delimiter=",", ndmin=2
    )
    if arr.shape[1] <= max(angle_col, value_col):
        raise ValueError(
            f"{path} 列数 {arr.shape[1]} 不足（需要第 {angle_col}/{value_col} 列）"
        )
    return arr[:, angle_col].astype(float), arr[:, value_col].astype(float)


def load_radar_csv(path: str | Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """读 :func:`export_radar_groups` 的宽表 CSV -> ``(角度 deg, {曲线名: 幅值})``。

    首行须为表头（首列 ``angle_deg``，其后各列为曲线名）；``%`` 注释行跳过。
    纯数值/COMSOL 表用 :func:`load_comsol_csv`。
    """
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    header: list[str] | None = None
    rows: list[str] = []
    for line in lines:
        text = line.strip()
        if not text or text.startswith("%"):
            continue
        if header is None:
            header = [c.strip() for c in text.split(",")]
            if header[0] != "angle_deg":
                raise ValueError(
                    f"{path} 首列表头 {header[0]!r} ≠ 'angle_deg'"
                    "（无表头表请用 load_comsol_csv）"
                )
            if len(header) < 2 or len(set(header)) != len(header):
                raise ValueError(f"{path} 表头须唯一且至少含一列曲线：{header}")
            if any(not name for name in header):
                raise ValueError(f"{path} 表头含空列名：{header}")
            continue
        rows.append(line)
    if header is None:
        raise ValueError(f"{path} 缺表头")
    if not rows:
        raise ValueError(f"{path} 无数据行")
    arr = np.loadtxt(io.StringIO("\n".join(rows)), delimiter=",", ndmin=2)
    if arr.shape[1] != len(header):
        raise ValueError(f"{path} 列数 {arr.shape[1]} ≠ 表头 {len(header)}")
    return arr[:, 0].astype(float), {
        name: arr[:, i].astype(float) for i, name in enumerate(header[1:], 1)
    }


def groups_from_comsol_csvs(
    sources: Mapping[str, Mapping[str, str | Path]],
    *,
    labels: Mapping[str, str] | None = None,
) -> tuple[RadarGroup, ...]:
    """按 ``{group_key: {series_name: csv_path}}`` 批量读 COMSOL 表成组。"""
    groups: list[RadarGroup] = []
    for key, series_map in sources.items():
        series = tuple(
            RadarSeries(name, *load_comsol_csv(path))
            for name, path in series_map.items()
        )
        if not series:
            raise ValueError(f"组 {key} 没有曲线")
        label = (labels or {}).get(key) or CASE_LABELS.get(key, key)
        groups.append(RadarGroup(key=key, series=series, label=label))
    if not groups:
        raise ValueError("至少需要一组远场数据")
    return tuple(groups)


def validate_angle_grid(
    angles_deg: Sequence[float],
    *,
    expected_points: int = DEFAULT_POINTS,
    step_deg: float = DEFAULT_STEP_DEG,
    full_circle: bool = True,
    tol: float = 1e-6,
) -> np.ndarray:
    """校验 1° 网格并返回 float 数组；违反任一条件抛 ``ValueError``。"""
    angles = np.asarray(angles_deg, dtype=float).ravel()
    if angles.size != expected_points:
        raise ValueError(f"角度点数 {angles.size} ≠ {expected_points}（1° 网格）")
    if not np.all(np.isfinite(angles)):
        raise ValueError("角度含 NaN/Inf")
    if not np.all(np.diff(angles) > 0):
        raise ValueError("角度须严格递增（重复/倒序网格不允许）")
    if not np.allclose(np.diff(angles), step_deg, atol=tol, rtol=0.0):
        raise ValueError(f"角度步长不统一（期望 {step_deg}°）")
    if abs(angles[0] - ANGLE_START) > tol:
        raise ValueError(f"角度起点 {angles[0]} ≠ {ANGLE_START}°")
    stop = ANGLE_STOP if full_circle else ANGLE_START + step_deg * (angles.size - 1)
    if abs(angles[-1] - stop) > tol:
        raise ValueError(f"角度终点 {angles[-1]} ≠ {stop}°")
    return angles


def validate_group(
    group: RadarGroup,
    *,
    expected_points: int = DEFAULT_POINTS,
    step_deg: float = DEFAULT_STEP_DEG,
    full_circle: bool = True,
) -> None:
    """校验组内每条曲线：角度网格合规、幅值有限且非负、点序与网格一致。"""
    if not group.key:
        raise ValueError("组缺少 key")
    if not group.series:
        raise ValueError(f"组 {group.key} 没有曲线")
    names = [s.name for s in group.series]
    if len(set(names)) != len(names):
        raise ValueError(f"组 {group.key} 曲线名重复：{names}")
    base: np.ndarray | None = None
    for s in group.series:
        angles = validate_angle_grid(
            s.angles_deg,
            expected_points=expected_points,
            step_deg=step_deg,
            full_circle=full_circle,
        )
        amp = np.asarray(s.amplitude_pa, dtype=float).ravel()
        if amp.size != angles.size:
            raise ValueError(
                f"组 {group.key}/{s.name}：幅值 {amp.size} 点 ≠ 角度 {angles.size} 点"
            )
        if not np.all(np.isfinite(amp)):
            raise ValueError(f"组 {group.key}/{s.name}：幅值含 NaN/Inf")
        if np.any(amp < 0):
            raise ValueError(f"组 {group.key}/{s.name}：幅值须非负")
        if base is None:
            base = angles
        elif not np.allclose(angles, base, atol=1e-6, rtol=0.0):
            raise ValueError(f"组 {group.key} 各曲线角度网格不一致")


def _schema(
    groups: Sequence[RadarGroup],
    *,
    expected_points: int,
    step_deg: float,
    full_circle: bool,
    stem: str,
    outputs: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    stop = ANGLE_STOP if full_circle else ANGLE_START + step_deg * (expected_points - 1)
    return {
        "kind": "gain_ep.radar_export",
        "version": 1,
        "stem": stem,
        "angle": {
            "unit": "deg",
            "start": ANGLE_START,
            "stop": stop,
            "step": step_deg,
            "points": expected_points,
            "convention": "counterclockwise from +x",
        },
        "value": {"name": VALUE_NAME, "unit": VALUE_UNIT},
        "legacy_case_keys": LEGACY_CASE_KEYS,
        "groups": {
            g.key: {
                "label": g.label or CASE_LABELS.get(g.key, g.key),
                "series": [s.name for s in g.series],
                "files": dict(outputs.get(g.key, {})),
            }
            for g in groups
        },
        "csv": {
            "format": "wide",
            "columns": ["angle_deg", "<series names in groups.*.series>"],
            "loader": "pysci.research.gain_ep.simulation.farfield.load_radar_csv",
        },
        "mat": {
            "variables": ["angle_deg", "amp_<series>"],
            "shape": [1, expected_points],
            "note": "scipy.io.savemat 以 1×N 行向量落盘",
        },
    }


def export_radar_groups(
    groups: Sequence[RadarGroup],
    out_dir: str | Path,
    *,
    stem: str = "radar_1deg",
    formats: Sequence[str] = ("csv", "mat"),
    expected_points: int = DEFAULT_POINTS,
    step_deg: float = DEFAULT_STEP_DEG,
    full_circle: bool = True,
) -> dict[str, Path]:
    """批量导出三组远场数据；返回 ``{group.key[.csv|.mat] / schema: Path}``。

    CSV 为宽表（``angle_deg`` + 各曲线一列）；MAT 变量为 ``angle_deg`` 与
    ``amp_<series>``；同目录写 ``<stem>_schema.json``（列名/单位/角网格/文件映射）。
    """
    fmt = {f.lower() for f in formats}
    unknown = fmt - {"csv", "mat"}
    if unknown:
        raise ValueError(f"不支持的导出格式：{sorted(unknown)}（csv/mat）")
    if not fmt:
        raise ValueError("至少需要一种导出格式")
    if not groups:
        raise ValueError("至少需要一组远场数据")
    keys = [g.key for g in groups]
    if len(set(keys)) != len(keys):
        raise ValueError(f"组 key 重复：{keys}")
    for g in groups:
        validate_group(
            g,
            expected_points=expected_points,
            step_deg=step_deg,
            full_circle=full_circle,
        )

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, dict[str, str]] = {}
    written: dict[str, Path] = {}
    for g in groups:
        angles = np.asarray(g.series[0].angles_deg, dtype=float).ravel()
        names = [s.name for s in g.series]
        files: dict[str, str] = {}
        if "csv" in fmt:
            path = out / f"{stem}_{g.key}.csv"
            data = np.column_stack(
                [angles, *[np.asarray(s.amplitude_pa, dtype=float) for s in g.series]]
            )
            header = ",".join(["angle_deg", *names])
            np.savetxt(
                path,
                data,
                delimiter=",",
                header=header,
                comments="",
                fmt="%.10g",
            )
            files["csv"] = path.name
            written[f"{g.key}.csv"] = path
        if "mat" in fmt:
            from scipy.io import savemat  # noqa: PLC0415 - 仅 MAT 导出需要

            path = out / f"{stem}_{g.key}.mat"
            payload: dict[str, Any] = {"angle_deg": angles}
            for s in g.series:
                payload[f"amp_{s.name}"] = np.asarray(s.amplitude_pa, dtype=float)
            savemat(path, payload, do_compression=True)
            files["mat"] = path.name
            written[f"{g.key}.mat"] = path
        outputs[g.key] = files

    schema_path = out / f"{stem}_schema.json"
    schema_path.write_text(
        json.dumps(
            _schema(
                groups,
                expected_points=expected_points,
                step_deg=step_deg,
                full_circle=full_circle,
                stem=stem,
                outputs=outputs,
            ),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    written["schema"] = schema_path
    return written


def main(argv: Sequence[str] | None = None) -> int:
    """CLI：``python -m pysci.research.gain_ep.simulation.farfield``。

    每个 ``--source GROUP SERIES CSV`` 指定一条曲线；同一 GROUP 的多次出现合成一组。
    """
    import argparse  # noqa: PLC0415 - 仅 CLI 需要

    parser = argparse.ArgumentParser(description="远场雷达 1° 网格批量导出")
    parser.add_argument(
        "--source",
        nargs=3,
        action="append",
        metavar=("GROUP", "SERIES", "CSV"),
        required=True,
        help="组名 曲线名 COMSOL CSV（可重复）",
    )
    parser.add_argument("--out", required=True, help="输出目录")
    parser.add_argument("--stem", default="radar_1deg", help="输出文件前缀")
    parser.add_argument("--formats", default="csv,mat", help="csv,mat")
    parser.add_argument(
        "--points", type=int, default=None, help="点数（缺省：全圆 361 / 半圆 181）"
    )
    parser.add_argument(
        "--half-circle",
        action="store_true",
        help="按旧图 0–180°、181 点网格读取",
    )
    args = parser.parse_args(argv)

    full_circle = not args.half_circle
    points = args.points or (181 if args.half_circle else DEFAULT_POINTS)
    sources: dict[str, dict[str, str]] = {}
    for group, series, path in args.source:
        sources.setdefault(group, {})[series] = path
    groups = groups_from_comsol_csvs(sources)
    written = export_radar_groups(
        groups,
        args.out,
        stem=args.stem,
        formats=tuple(f.strip() for f in args.formats.split(",") if f.strip()),
        expected_points=points,
        full_circle=full_circle,
    )
    for name, path in written.items():
        print(f"[radar-export] {name}: {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - 入口
    raise SystemExit(main())
