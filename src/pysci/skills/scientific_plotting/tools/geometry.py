"""标注几何语义机械检查：``ANNOTATIONS.yaml`` ↔ 图中实际 Artist。

用途：把任务书声明的**可计数语义模型**（如「每侧 2 个框、外框大于内框、二者虚线相连」）
变成 ``pysci-figures audit`` 可复跑的判据，取代人眼核对。判据四类：

1. **计数一致性**（``counts``）：按 ``view × kind × role`` 数图中原语，须与声明精确相等；
2. **完整包含 + 留边距**（``contain``）：内原语所有顶点在外原语内、边界不相交，且最短
   边界距离 ≥ 声明边距；
3. **成对相对大小**（``sizes``）：闭合原语的面积比须严格大于声明下限（默认 >1）；
4. **连接线存在性**（``connectors``）：存在一条 ``line`` 原语，两端点分别落在两个元素上
   或距其边界 ≤ 容差。

**几何来源是实际绘制对象**，不是声明里的数值：管线给标注原语打
``gid="annot:<id>[;role=…][;view=…]"``，本模块从 ``Rectangle`` / ``Polygon`` /
``PathPatch`` / ``Line2D`` 抽取顶点并统一换算到显示像素后比较（坐标系统无关）。
声明 schema、gid 约定与示例见
``orchestration/skills/scientific-plotting/references/plotting-04-annotation-semantics.md``。

**通用性**：本模块不含任何具体形状/物理尺寸；边距、面积比、容差全部由声明参数化。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.artist import Artist
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch, Polygon, Rectangle

#: 标注原语的 gid 前缀与视图轴 gid 前缀（约定见 plotting-04 分册）。
ANNOT_PREFIX = "annot:"
VIEW_PREFIX = "view:"

#: 图目录声明文件的查找顺序（大小写/扩展名容错）。
_SIDECAR_STEMS = ("ANNOTATIONS", "annotations")
_SIDECAR_SUFFIXES = (".yaml", ".yml", ".json")

ERROR, WARN = "ERROR", "WARN"

_Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class Primitive:
    """一个标注原语的几何（顶点已换算到显示像素）。

    Attributes:
        id: 声明中的唯一标识（``gid`` 里 ``annot:`` 之后、``;`` 之前的部分）。
        kind: ``box`` / ``polygon`` / ``line``。
        view: 所属视图名（轴 ``gid="view:<name>"`` 或元数据 ``;view=``；缺省 ``ax<N>``）。
        role: 可选角色（``;role=``），用于计数过滤。
        vertices: ``(N, 2)`` 显示像素坐标；闭合原语首尾可重复。
    """

    id: str
    kind: str
    view: str
    role: str
    vertices: np.ndarray

    @property
    def closed(self) -> bool:
        """是否为闭合原语（box / polygon）。"""
        return self.kind != "line"

    def bbox(self) -> _Box:
        """显示像素下的外接框 ``(x0, y0, x1, y1)``。"""
        x = self.vertices[:, 0]
        y = self.vertices[:, 1]
        return float(x.min()), float(y.min()), float(x.max()), float(y.max())

    def area(self) -> float:
        """闭合原语的多边形面积；``line`` 退化为外接框面积。"""
        if self.closed:
            x, y = self.vertices[:, 0], self.vertices[:, 1]
            return float(
                abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0
            )
        x0, y0, x1, y1 = self.bbox()
        return float((x1 - x0) * (y1 - y0))


@dataclass
class GeometryIssue:
    """一条几何语义检查判定。"""

    level: str
    code: str
    message: str


@dataclass
class GeometryReport:
    """几何语义检查报告（供 audit 合并）。"""

    issues: list[GeometryIssue] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def errors(self) -> list[GeometryIssue]:
        """ERROR 级判定。"""
        return [i for i in self.issues if i.level == ERROR]

    @property
    def ok(self) -> bool:
        """无 ERROR 即通过。"""
        return not self.errors

    def add(self, level: str, code: str, message: str) -> None:
        """追加一条判定。"""
        self.issues.append(GeometryIssue(level, code, message))


# ---------------------------------------------------------------------------
# 声明文件
# ---------------------------------------------------------------------------
def find_sidecar(figdir: Path | str) -> Path | None:
    """在图目录内查找标注语义声明文件（``ANNOTATIONS.yaml|yml|json``）。

    Args:
        figdir: 图目录。

    Returns:
        首个命中的声明文件路径；不存在返回 ``None``。
    """
    figdir = Path(figdir)
    for stem in _SIDECAR_STEMS:
        for ext in _SIDECAR_SUFFIXES:
            p = figdir / f"{stem}{ext}"
            if p.is_file():
                return p
    return None


def load_spec(path: Path | str) -> dict[str, Any]:
    """读取声明文件为 dict。

    Args:
        path: ``.yaml`` / ``.yml`` / ``.json`` 声明文件。

    Returns:
        声明 dict（空文件返回空 dict）。

    Raises:
        ValueError: 文件内容不是 mapping，或 YAML 解析不可用。
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        data = json.loads(text) if text.strip() else {}
    else:
        try:
            import yaml  # type: ignore
        except ImportError as e:  # pragma: no cover - pyyaml 是显式依赖
            raise ValueError(f"读取 {path.name} 需要 pyyaml：{e}") from e
        data = yaml.safe_load(text) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path.name} 必须是 mapping（实际 {type(data).__name__}）")
    return data


# ---------------------------------------------------------------------------
# 从 Figure 抽取原语
# ---------------------------------------------------------------------------
def _parse_gid(gid: str) -> tuple[str, dict[str, str]]:
    """解析 ``annot:<id>[;k=v…]`` → ``(id, metadata)``。"""
    body = gid[len(ANNOT_PREFIX) :]
    head, *rest = body.split(";")
    meta: dict[str, str] = {}
    for item in rest:
        key, sep, value = item.partition("=")
        if sep:
            meta[key.strip()] = value.strip()
    return head.strip(), meta


def _view_of(artist: Artist) -> str:
    """取原语所属视图：轴 gid ``view:<name>``，否则按轴序号 ``ax<N>``。"""
    axes = getattr(artist, "axes", None)
    if axes is None:
        return ""
    gid = axes.get_gid()
    if isinstance(gid, str) and gid.startswith(VIEW_PREFIX):
        return gid[len(VIEW_PREFIX) :]
    try:
        fig = axes.figure
        return f"ax{list(fig.axes).index(axes)}"
    except (AttributeError, ValueError):
        return "ax0"


def _artist_geometry(artist: Artist) -> tuple[str, np.ndarray] | None:
    """把一个 Artist 转成 ``(kind, display-vertices)``；不支持的类型返回 None。

    补丁类用 ``Artist.get_transform``（含自定义 ``transform=ax.transAxes`` 等）；
    ``Patch.get_transform`` 会叠加补丁自身的平移，不能用于已经绝对化的顶点。
    """
    if isinstance(artist, Rectangle):
        (x0, y0), (x1, y1) = artist.get_bbox().get_points()
        verts = np.array(
            [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]], dtype=float
        )
        return "box", Artist.get_transform(artist).transform(verts)
    if isinstance(artist, Polygon):
        verts = np.asarray(artist.get_xy(), dtype=float)
        return "polygon", Artist.get_transform(artist).transform(verts)
    if isinstance(artist, PathPatch):
        verts = np.asarray(artist.get_path().vertices, dtype=float)
        return "polygon", Artist.get_transform(artist).transform(verts)
    if isinstance(artist, Line2D):
        verts = np.asarray(artist.get_xydata(), dtype=float)
        return "line", artist.get_transform().transform(verts)
    return None


def collect_primitives(fig: Figure, *, prefix: str = ANNOT_PREFIX) -> list[Primitive]:
    """抽取图中所有带 ``annot:`` gid 的标注原语。

    Args:
        fig: 已构建的 Figure（需已完成轴布局；建议先 ``fig.canvas.draw()``）。
        prefix: gid 前缀，默认 ``annot:``。

    Returns:
        原语列表；排序稳定（按抽取顺序）。
    """
    out: list[Primitive] = []
    for artist in fig.findobj():
        gid = artist.get_gid()
        if not isinstance(gid, str) or not gid.startswith(prefix):
            continue
        ident, meta = _parse_gid(gid)
        if not ident:
            continue
        parsed = _artist_geometry(artist)
        if parsed is None:
            continue
        kind, verts = parsed
        if verts is None or len(verts) < 2:
            continue
        if meta.get("kind"):
            kind = meta["kind"].strip().lower()
        out.append(
            Primitive(
                id=ident,
                kind=kind,
                view=meta.get("view") or _view_of(artist),
                role=meta.get("role", ""),
                vertices=verts,
            )
        )
    return out


# ---------------------------------------------------------------------------
# 几何工具（显示像素坐标系）
# ---------------------------------------------------------------------------
def _segments(
    vertices: np.ndarray, *, closed: bool
) -> list[tuple[np.ndarray, np.ndarray]]:
    """把顶点序列拆成线段；闭合原语补首尾段。"""
    segs = [(vertices[i], vertices[i + 1]) for i in range(len(vertices) - 1)]
    if closed and len(vertices) >= 3 and not np.allclose(vertices[0], vertices[-1]):
        segs.append((vertices[-1], vertices[0]))
    return segs


def _cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """二维叉积。"""
    return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))


def _on_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> bool:
    """点 p 是否落在线段 ab 上（含端点）。"""
    if abs(_cross(a, b, p)) > 1e-9:
        return False
    return bool(
        min(a[0], b[0]) - 1e-9 <= p[0] <= max(a[0], b[0]) + 1e-9
        and min(a[1], b[1]) - 1e-9 <= p[1] <= max(a[1], b[1]) + 1e-9
    )


def _segments_intersect(
    p1: np.ndarray, p2: np.ndarray, q1: np.ndarray, q2: np.ndarray
) -> bool:
    """两条线段是否相交（含端点接触/共线重叠）。"""
    d1 = _cross(q1, q2, p1)
    d2 = _cross(q1, q2, p2)
    d3 = _cross(p1, p2, q1)
    d4 = _cross(p1, p2, q2)
    if ((d1 > 0 > d2) or (d1 < 0 < d2)) and ((d3 > 0 > d4) or (d3 < 0 < d4)):
        return True
    return (
        _on_segment(p1, q1, q2)
        or _on_segment(p2, q1, q2)
        or _on_segment(q1, p1, p2)
        or _on_segment(q2, p1, p2)
    )


def _point_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """点到线段的最短距离。"""
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom <= 1e-18:
        return float(np.linalg.norm(p - a))
    t = float(np.dot(p - a, ab) / denom)
    t = min(1.0, max(0.0, t))
    return float(np.linalg.norm(p - (a + t * ab)))


def _segment_distance(
    p1: np.ndarray, p2: np.ndarray, q1: np.ndarray, q2: np.ndarray
) -> float:
    """两条线段的最短距离；相交为 0。"""
    if _segments_intersect(p1, p2, q1, q2):
        return 0.0
    return min(
        _point_segment_distance(p1, q1, q2),
        _point_segment_distance(p2, q1, q2),
        _point_segment_distance(q1, p1, p2),
        _point_segment_distance(q2, p1, p2),
    )


def _boundary_distance(a: Primitive, b: Primitive) -> float:
    """两原语边界的最短距离（闭合原语含闭合段）。"""
    segs_a = _segments(a.vertices, closed=a.closed)
    segs_b = _segments(b.vertices, closed=b.closed)
    if not segs_a or not segs_b:
        return float("inf")
    return min(
        _segment_distance(p1, p2, q1, q2) for p1, p2 in segs_a for q1, q2 in segs_b
    )


def _point_inside(point: np.ndarray, poly: Primitive) -> bool:
    """点是否在闭合原语内部（射线法；边界点按内部处理由调用方另行判距）。"""
    if not poly.closed:
        return False
    x, y = float(point[0]), float(point[1])
    verts = poly.vertices
    inside = False
    j = len(verts) - 1
    for i in range(len(verts)):
        xi, yi = float(verts[i, 0]), float(verts[i, 1])
        xj, yj = float(verts[j, 0]), float(verts[j, 1])
        if (yi > y) != (yj > y):
            x_cross = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < x_cross:
                inside = not inside
        j = i
    return inside


def _contained(inner: Primitive, outer: Primitive) -> bool:
    """内原语是否完整落在外原语内且边界不相交。"""
    if not outer.closed:
        return False
    if not all(_point_inside(np.asarray(p, float), outer) for p in inner.vertices):
        return False
    return not any(
        _segments_intersect(p1, p2, q1, q2)
        for p1, p2 in _segments(inner.vertices, closed=inner.closed)
        for q1, q2 in _segments(outer.vertices, closed=True)
    )


def _required_margin(rule: dict[str, Any], outer: Primitive) -> float:
    """声明的边距（像素）：``margin_px`` 优先，否则按外框 bbox 短边的比例。"""
    if "margin_px" in rule:
        return float(rule["margin_px"])
    ratio = float(rule.get("margin_ratio", 0.02))
    x0, y0, x1, y1 = outer.bbox()
    return ratio * min(x1 - x0, y1 - y0)


def _required_tolerance(rule: dict[str, Any], a: Primitive, b: Primitive) -> float:
    """连接线端点容差（像素）：``tolerance_px`` 优先，否则按两元素较小短边比例。"""
    if "tolerance_px" in rule:
        return float(rule["tolerance_px"])
    ratio = float(rule.get("tolerance_ratio", 0.05))
    shorts: list[float] = []
    for p in (a, b):
        x0, y0, x1, y1 = p.bbox()
        short = min(x1 - x0, y1 - y0)
        if short > 0:
            shorts.append(short)
    return ratio * min(shorts) if shorts else 0.0


def _endpoint_near(point: np.ndarray, prim: Primitive, tolerance: float) -> bool:
    """点是否落在原语内或距其边界 ≤ 容差。"""
    if _point_inside(point, prim):
        return True
    if prim.closed:
        dist = _boundary_distance(
            Primitive(prim.id, "line", prim.view, prim.role, np.array([point, point])),
            prim,
        )
    else:
        dist = min(
            _point_segment_distance(point, a, b)
            for a, b in _segments(prim.vertices, closed=False)
        )
    return dist <= tolerance + 1e-9


# ---------------------------------------------------------------------------
# 四类判据
# ---------------------------------------------------------------------------
def _rules(spec: dict[str, Any], key: str) -> list[dict[str, Any]]:
    """把声明项统一成 list[dict]（允许单条 mapping 简写）。"""
    raw = spec.get(key)
    if raw is None:
        return []
    if isinstance(raw, dict):
        return [raw]
    return [r for r in raw if isinstance(r, dict)]


def _check_require(
    by_id: dict[str, Primitive], spec: dict[str, Any], rep: GeometryReport
) -> None:
    for ident in _list_str(spec.get("require")):
        if ident not in by_id:
            rep.add(
                ERROR,
                "annot-missing",
                f"声明要求原语 {ident!r}，但图中不存在（gid 漏打或拼写不一致？）",
            )


def _list_str(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw]
    return [str(x) for x in raw]


def _check_counts(
    primitives: list[Primitive], spec: dict[str, Any], rep: GeometryReport
) -> None:
    for rule in _rules(spec, "counts"):
        view = str(rule.get("view", ""))
        kind = str(rule.get("kind", "box")).lower()
        role = rule.get("role")
        expected = int(rule.get("expected", rule.get("count", 0)))
        actual = [
            p
            for p in primitives
            if p.view == view
            and (kind == "any" or p.kind == kind)
            and (role is None or p.role == str(role))
        ]
        key = f"annot_count[{view}/{kind}{'/' + str(role) if role else ''}]"
        rep.metrics[key] = len(actual)
        if len(actual) != expected:
            ids = ", ".join(p.id for p in actual) or "无"
            rep.add(
                ERROR,
                "annot-count",
                f"view={view!r} kind={kind!r}"
                f"{f' role={role!r}' if role else ''} 声明 {expected} 个，"
                f"实际 {len(actual)} 个（ids: {ids}）；与任务书语义模型不一致",
            )


def _lookup(
    by_id: dict[str, Primitive], ident: Any, rep: GeometryReport, rule_name: str
) -> Primitive | None:
    key = str(ident)
    prim = by_id.get(key)
    if prim is None:
        rep.add(
            ERROR,
            "annot-missing",
            f"{rule_name} 引用的原语 {key!r} 在图中不存在（gid 漏打或拼写不一致？）",
        )
    return prim


def _check_contain(
    by_id: dict[str, Primitive], spec: dict[str, Any], rep: GeometryReport
) -> None:
    for rule in _rules(spec, "contain"):
        inner = _lookup(by_id, rule.get("inner"), rep, "contain")
        outer = _lookup(by_id, rule.get("outer"), rep, "contain")
        if inner is None or outer is None:
            continue
        if not outer.closed:
            rep.add(
                ERROR,
                "annot-contain",
                f"contain: 外原语 {outer.id!r} 必须是 box/polygon（实际 {outer.kind}）",
            )
            continue
        if not _contained(inner, outer):
            rep.add(
                ERROR,
                "annot-contain",
                f"{inner.id!r} 未完整位于 {outer.id!r} 内"
                "（顶点越界或边界相交）；约束冲突须按任务书处置分支扩大区域，"
                "禁裁剪/溢出/贴边或另画替代框",
            )
            continue
        required = _required_margin(rule, outer)
        actual = _boundary_distance(inner, outer)
        rep.metrics[f"annot_margin[{inner.id}→{outer.id}]"] = round(actual, 2)
        if actual < required - 1e-9:
            rep.add(
                ERROR,
                "annot-contain",
                f"{inner.id!r} 与 {outer.id!r} 的边界距离 {actual:.1f}px < "
                f"声明边距 {required:.1f}px（贴边/临界间距＝缺陷）",
            )


def _check_sizes(
    by_id: dict[str, Primitive], spec: dict[str, Any], rep: GeometryReport
) -> None:
    for rule in _rules(spec, "sizes"):
        outer = _lookup(by_id, rule.get("outer"), rep, "sizes")
        inner = _lookup(by_id, rule.get("inner"), rep, "sizes")
        if outer is None or inner is None:
            continue
        area_outer, area_inner = outer.area(), inner.area()
        if area_inner <= 0:
            rep.add(
                ERROR,
                "annot-size",
                f"sizes: 内原语 {inner.id!r} 面积为 0（退化几何），无法比较相对大小",
            )
            continue
        ratio = area_outer / area_inner
        minimum = float(rule.get("min_ratio", 1.0))
        rep.metrics[f"annot_size_ratio[{outer.id}/{inner.id}]"] = round(ratio, 3)
        if not ratio > minimum + 1e-12:
            rep.add(
                ERROR,
                "annot-size",
                f"成对相对大小不满足： {outer.id!r}/{inner.id!r} 面积比 "
                f"{ratio:.3f} 未严格大于 {minimum:g}（外框应大于内框；检查是否倒置）",
            )


def _check_connectors(
    primitives: list[Primitive],
    by_id: dict[str, Primitive],
    spec: dict[str, Any],
    rep: GeometryReport,
) -> None:
    lines = [p for p in primitives if p.kind == "line" and len(p.vertices) >= 2]
    for rule in _rules(spec, "connectors"):
        source = _lookup(by_id, rule.get("source"), rep, "connectors")
        target = _lookup(by_id, rule.get("target"), rep, "connectors")
        if source is None or target is None:
            continue
        tolerance = _required_tolerance(rule, source, target)
        found: list[str] = []
        for line in lines:
            a, b = line.vertices[0], line.vertices[-1]
            forward = _endpoint_near(a, source, tolerance) and _endpoint_near(
                b, target, tolerance
            )
            backward = _endpoint_near(b, source, tolerance) and _endpoint_near(
                a, target, tolerance
            )
            if forward or backward:
                found.append(line.id)
        rep.metrics[f"annot_connectors[{source.id}↔{target.id}]"] = len(found)
        if not found:
            rep.add(
                ERROR,
                "annot-connect",
                f"缺少连接线：没有任何 line 原语的两端点分别落在 "
                f"{source.id!r} 与 {target.id!r} 上或距边界 ≤ {tolerance:.1f}px",
            )


# ---------------------------------------------------------------------------
# 对外入口
# ---------------------------------------------------------------------------
def check_annotations(
    primitives: list[Primitive], spec: dict[str, Any]
) -> GeometryReport:
    """按声明核验一组标注原语。

    Args:
        primitives: 从 Figure 抽取的原语（``collect_primitives``）。
        spec: ``ANNOTATIONS.yaml`` 解析后的声明 dict。

    Returns:
        含判定与读数指标的几何报告。
    """
    rep = GeometryReport()
    by_id: dict[str, Primitive] = {}
    for p in primitives:
        by_id.setdefault(p.id, p)
    rep.metrics["annot_primitives"] = len(primitives)
    rep.metrics["annot_views"] = sorted({p.view for p in primitives})
    _check_require(by_id, spec, rep)
    _check_counts(primitives, spec, rep)
    _check_contain(by_id, spec, rep)
    _check_sizes(by_id, spec, rep)
    _check_connectors(primitives, by_id, spec, rep)
    return rep


def audit_annotations(fig: Figure, spec: dict[str, Any] | None) -> GeometryReport:
    """对已构建的 Figure 执行几何语义检查。

    无声明时：图中存在 ``annot:`` 原语 → WARN 提示补声明；否则静默（空报告）。
    声明存在但图中无任何原语 → WARN（避免「声明了却从未打 gid」静默假绿）。

    Args:
        fig: 目标 Figure（有声明或标注原语时先 ``canvas.draw()`` 定格布局/变换）。
        spec: 声明 dict；``None`` 表示图目录无声明文件。

    Returns:
        几何报告。
    """
    primitives = collect_primitives(fig)
    rep = GeometryReport()
    rep.metrics["annot_primitives"] = len(primitives)
    if spec is None and not primitives:
        return rep
    # 布局引擎（constrained/tight）在 draw 后才定格；画一次再重抽，保证轴位置稳定。
    try:
        fig.canvas.draw()
    except Exception:  # noqa: BLE001 - 无 canvas 的合成 Figure 仍可按当前变换核验
        pass
    primitives = collect_primitives(fig)
    rep.metrics["annot_primitives"] = len(primitives)
    if spec is None:
        if primitives:
            rep.add(
                WARN,
                "annotations",
                f"图中存在 {len(primitives)} 个 annot:* 标注原语，但图目录无 "
                "ANNOTATIONS.yaml——语义模型未机械声明，无法核验计数/包含/大小/连线",
            )
        return rep
    if not primitives:
        rep.add(
            WARN,
            "annot-empty",
            "ANNOTATIONS.yaml 已声明语义模型，但图中没有任何 annot:* 原语"
            "（gid 未打或声明为空）；若确无标注，请删除声明文件",
        )
        return rep
    checked = check_annotations(primitives, spec)
    rep.issues.extend(checked.issues)
    rep.metrics.update(checked.metrics)
    return rep
