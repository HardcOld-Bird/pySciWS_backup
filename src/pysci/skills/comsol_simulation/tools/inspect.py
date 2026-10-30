"""模型自省：让 Agent「读懂」一个 .mph 的结构与任意 Java 节点的能力。

四类能力：
- :func:`dump_tree` — 紧凑文本化模型树（组件/几何/材料/物理场/网格/研究/求解/结果 + 参数），供 Agent 读结构。
- :func:`list_parameters` / :func:`inventory` — 参数表与按子系统的节点清单。
- :func:`resolve_path` / :func:`dump_node` — 按路径定位任意活体节点并 dump 其**状态**
  （类型/标签/属性当前值/枚举允许值/选择实体数），是建模试错时的第一诊断入口。
- :func:`introspect` — 对任意活体 Java 节点用反射列出类名与方法签名（**活体 javadoc**，弥补未安装的 javadoc 插件）。
- :func:`summarize_java` — 把 GUI 导出的 ``.java`` 模型文件解析成结构化摘要（GUI→Java 兜底路径与 knowledge 种子用）。

所有 live COMSOL 调用都做了防御式包裹：探测方法存在性、逐个 try/except，任何单点失败都降级为占位符，
绝不因某个节点不可读而中断整棵树的 dump。
"""

from __future__ import annotations

import re
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# 通用小工具（防御式 Java 调用）
# ---------------------------------------------------------------------------
def _jmodel(model: Any) -> Any:
    """接受 mph.Model 或裸 Java model，统一返回 Java model 对象。"""
    return getattr(model, "java", model)


def _safe(fn: Callable[..., Any], *args: Any, default: Any = None, **kw: Any) -> Any:
    """调用可能抛异常的 Java 方法，失败返回 default。"""
    try:
        return fn(*args, **kw)
    except Exception:  # noqa: BLE001 - 自省必须对任意节点健壮
        return default


def _call_if(obj: Any, method: str, *args: Any, default: Any = None) -> Any:
    """若 obj 有该方法则调用，否则返回 default。"""
    fn = getattr(obj, method, None)
    if not callable(fn):
        return default
    return _safe(fn, *args, default=default)


def _tags(seq: Any) -> list[str]:
    """读取一个 COMSOL 序列节点的 tag 列表。"""
    if seq is None:
        return []
    raw = _call_if(seq, "tags", default=[]) or []
    return [str(t) for t in raw]


def _str_list(raw: Any) -> list[str]:
    return [str(x) for x in (raw or [])]


# ---------------------------------------------------------------------------
# 参数
# ---------------------------------------------------------------------------
def list_parameters(model: Any) -> OrderedDict[str, dict[str, str]]:
    """返回全局参数表 ``name -> {value, descr}``。

    优先用 mph 的 ``Model.parameters`` 便捷属性；回退到 Java ``param()`` 序列。
    """
    out: OrderedDict[str, dict[str, str]] = OrderedDict()

    # 途径 1：mph Model.parameters（OrderedDict[str, str]）
    params = getattr(model, "parameters", None)
    if isinstance(params, dict) and params:
        for k, v in params.items():
            out[str(k)] = {"value": str(v), "descr": ""}
        return out

    # 途径 2：Java param() 序列
    jm = _jmodel(model)
    p = _call_if(jm, "param", default=None)
    if p is None:
        return out
    names = _str_list(_call_if(p, "varnames", default=[]))
    for n in names:
        val = _call_if(p, "get", n, default="")
        descr = _call_if(p, "descr", n, default="") or _call_if(
            p, "getdescr", n, default=""
        )
        out[n] = {"value": str(val), "descr": str(descr or "")}
    return out


# ---------------------------------------------------------------------------
# 模型树 dump
# ---------------------------------------------------------------------------
#: 顶层（model 级）序列访问器 → 展示名。
_MODEL_SEQS: tuple[tuple[str, str], ...] = (
    ("param", "Global Definitions / Parameters"),
    ("func", "Functions"),
    ("material", "Materials (global)"),
    ("study", "Study"),
    ("sol", "Solver Configurations"),
    ("result", "Results"),
    ("batch", "Batch Sequences"),
)

#: 组件（component）级序列访问器 → 展示名。
_COMP_SEQS: tuple[tuple[str, str], ...] = (
    ("geom", "Geometry"),
    ("material", "Materials"),
    ("physics", "Physics"),
    ("mesh", "Mesh"),
    ("view", "Views"),
)


def _node_line(seq: Any, tag: str) -> str:
    """为某个 tag 生成一行摘要：尽量附带类型/标签信息（全部防御式读取）。"""
    node = _call_if(seq, "get", tag, default=None)
    # 类型：先试序列级 getType(tag)，再试节点级 getType()/type()
    typ = str(
        _call_if(seq, "getType", tag, default="")
        or _call_if(node, "getType", default="")
        or _call_if(node, "type", default="")
        or ""
    )
    label = str(_call_if(node, "label", default="") or "")
    if label and label != tag:
        typ = f"{typ} [{label}]" if typ else f"[{label}]"
    return f"    - {tag}" + (f"  ::  {typ}" if typ else "")


def _dump_seq(seq: Any, title: str, lines: list[str]) -> None:
    tags = _tags(seq)
    lines.append(f"  {title}: ({len(tags)})")
    for t in tags:
        lines.append(_node_line(seq, t))


def dump_tree(model: Any, *, include_params: bool = True) -> str:
    """把模型树 dump 成紧凑文本，供 Agent 阅读理解模型结构。"""
    jm = _jmodel(model)
    lines: list[str] = ["Model Tree", "=========="]

    # 顶层序列
    for accessor, title in _MODEL_SEQS:
        seq = _call_if(jm, accessor, default=None)
        if seq is None:
            continue
        if accessor == "param" and include_params:
            params = list_parameters(model)
            lines.append(f"  Parameters: ({len(params)})")
            for name, meta in params.items():
                d = f"  # {meta['descr']}" if meta.get("descr") else ""
                lines.append(f"    - {name} = {meta['value']}{d}")
            continue
        _dump_seq(seq, title, lines)

    # 组件
    comps = _call_if(jm, "component", default=None)
    comp_tags = _tags(comps)
    lines.append(f"  Components: ({len(comp_tags)})")
    for c in comp_tags:
        comp = _safe(getattr(comps, "get", lambda *_: None), c)
        if comp is None:
            lines.append(f"    * {c}")
            continue
        lines.append(f"    * component '{c}'")
        for accessor, title in _COMP_SEQS:
            seq = _call_if(comp, accessor, default=None)
            if seq is None:
                continue
            tags = _tags(seq)
            lines.append(f"      {title}: ({len(tags)}) " + ", ".join(tags))

    return "\n".join(lines)


def inventory(model: Any) -> dict[str, list[str]]:
    """按子系统返回节点 tag 清单（供程序化遍历）。"""
    jm = _jmodel(model)
    inv: dict[str, list[str]] = {}
    for accessor, _ in _MODEL_SEQS:
        seq = _call_if(jm, accessor, default=None)
        if seq is not None:
            inv[accessor] = _tags(seq)
    comps = _call_if(jm, "component", default=None)
    inv["component"] = _tags(comps)
    for c in inv["component"]:
        comp = _safe(getattr(comps, "get", lambda *_: None), c)
        if comp is None:
            continue
        for accessor, _ in _COMP_SEQS:
            seq = _call_if(comp, accessor, default=None)
            if seq is not None:
                inv[f"{c}.{accessor}"] = _tags(seq)
    return inv


# ---------------------------------------------------------------------------
# 节点级状态 dump（建模试错的第一诊断入口）
# ---------------------------------------------------------------------------
#: 路径段语法：``name`` 或 ``name(arg)``；arg 恒以字符串传入（tag/选择名）。
_PATH_SEG = re.compile(r"^([A-Za-z_]\w*)(?:\(([^)]*)\))?$")


def _result_container_lookup(node: Any, tag: str) -> Any | None:
    """``result(tag)`` 落空时回退表格/数据集容器层——``result.table(tag)`` / ``result.dataset(tag)``。

    COMSOL 结果树里表格（``tbl*``）与数据集（``dset*``）**不在** ``result(tag)`` 的同类
    （plot group ``pg*``）下，而挂在 ``result.table(tag)`` / ``result.dataset(tag)``。
    两个容器都取不到该 tag 时返回 ``None``，调用方照原样抛首次异常（其余路径行为不变）。
    """
    if node is None:
        return None
    for accessor in ("table", "dataset"):
        seq = getattr(node, accessor, None)
        if not callable(seq):
            continue
        hit = _safe(seq, tag, default=None)
        if hit is not None:
            return hit
    return None


def resolve_path(model: Any, path: str) -> Any:
    """按点分路径从 Java model 定位任意节点。

    路径段语法：``accessor`` 或 ``accessor(tag)``，逐段调用。无参段对可调用对象
    做无参调用（如 ``selection``），否则取属性值。示例::

        component(comp1).physics(acpr).feature(bpf1)
        result(pg_field) / component(comp1).selection(sel_arc)
        result(tblSL)      # 表格：自动回退 result.table(tblSL)
        result(dset1)      # 数据集：自动回退 result.dataset(dset1)

    结果树容器层说明：``result(tag)`` 只解析 plot group；表格与数据集分别在
    ``result.table(tag)`` / ``result.dataset(tag)`` 下——本函数在 ``result(tag)``
    落空时自动回退试这两层（取不到仍抛首次异常）。

    Raises:
        ValueError: 路径段语法非法。
        AttributeError: 某段在上一步对象上不存在。
    """
    obj = _jmodel(model)
    for part in (p.strip() for p in path.split(".")):
        if not part:
            continue
        m = _PATH_SEG.match(part)
        if m is None:
            raise ValueError(f"路径段语法非法：{part!r}（应为 name 或 name(arg)）")
        name, arg = m.group(1), (m.group(2) or "").strip()
        fn = getattr(obj, name, None)
        if fn is None:
            raise AttributeError(f"路径段 {part!r}：对象无属性 '{name}'")
        if arg:
            try:
                obj = fn(arg)
            except Exception:  # noqa: BLE001 - 仅回退结果族表格/数据集容器层
                alt = _result_container_lookup(_safe(fn, default=None), arg)
                if alt is None:
                    raise
                obj = alt
        elif callable(fn):
            obj = fn()
        else:
            obj = fn
    return obj


@dataclass
class FeatureTypesReport:
    """``inspect feature-types`` 的读数：既有子特征 + 候选类型**探针**结果。"""

    #: 该节点已有子特征 ``(tag, getType())``
    existing: list[tuple[str, str]] = field(default_factory=list)
    #: trial-create 成功的候选类型（本模型实测可建）
    creatable: list[str] = field(default_factory=list)
    #: trial-create 报「未知特征 ID」的候选
    unknown: list[str] = field(default_factory=list)
    #: 建出来后未能删除的探针数（>0 → 模型可能留痕）
    probes_left: int = 0


def existing_feature_types(node: Any) -> list[tuple[str, str]]:
    """该节点**已有**子特征的 ``(tag, getType())`` 清单（无 ``feature()`` 能力 → 空）。"""
    fl = _call_if(node, "feature", default=None)
    if fl is None:
        return []
    out: list[tuple[str, str]] = []
    for tag in _tags(fl):
        feat = _safe(fl, tag, default=None)
        typ = _safe(getattr(feat, "getType", None), default="?")
        out.append((tag, str(typ)))
    return out


def probe_feature_types(node: Any, candidates: list[str]) -> FeatureTypesReport:
    """逐个 **trial-create** 候选类型（建完即删），回报「可创建 / 未知」。

    COMSOL **没有**枚举「接口允许的子特征类型」的 API（Programming Reference 只有
    ``feature().create(tag, type)`` / ``tags()`` / ``getType()``；文档里也只写自然语言名）。
    故把组员手工的「试建猜」机械化为一次批量探针——结论以**本模型实测**为准，不靠猜名。
    """
    rep = FeatureTypesReport(existing=existing_feature_types(node))
    fl = _call_if(node, "feature", default=None)
    if fl is None:
        rep.unknown = list(candidates)
        return rep
    used = {t for t, _ in rep.existing}
    for i, cand in enumerate(candidates):
        tag = f"_ftprobe{i}"
        while tag in used:  # 探针标签不得撞既有节点
            tag += "_"
        try:
            feat = fl.create(tag, cand)
        except Exception:  # noqa: BLE001 — COMSOL 对未知类型抛 Java 异常
            feat = None
        if feat is None:
            rep.unknown.append(cand)
            continue
        rep.creatable.append(cand)
        if not _remove_feature(node, fl, tag):
            rep.probes_left += 1
    return rep


def _remove_feature(node: Any, fl: Any, tag: str) -> bool:
    """尽力删除探针特征（两种常见 API 形态依次试）；全失败返回 False（调用方告警）。"""
    for attempt in (lambda: fl.remove(tag), lambda: node.feature(tag).remove()):
        try:
            attempt()
            return True
        except Exception:  # noqa: BLE001
            continue
    return False


def format_feature_types(rep: FeatureTypesReport, *, path: str = "") -> str:
    lines = [f"[feature-types] {path}", "  既有子特征："]
    lines += [f"    {tag} :: {typ}" for tag, typ in rep.existing] or ["    （无）"]
    if rep.creatable or rep.unknown:
        lines.append("  候选类型探针（trial-create，建完即删）：")
        lines += [f"    [√] {c}" for c in rep.creatable]
        lines += [f"    [✗] {c}（未知特征 ID）" for c in rep.unknown]
        if rep.probes_left:
            lines.append(
                f"  [!] {rep.probes_left} 个探针未能删除——模型可能留痕，勿直接保存"
            )
    else:
        lines.append(
            "  提示：COMSOL 无「枚举允许子类型」API；用 `--candidates a b c` 传候选名批量试建即得实测结论"
        )
    return "\n".join(lines)


def _longest_allowed(node: Any, prop: str, cap: int = 12) -> list[str] | None:
    """读枚举属性的允许值列表；非枚举/超长返回 None（避免 dump 爆炸）。"""
    allowed = _call_if(node, "getAllowedPropertyValues", prop, default=None)
    if allowed is None:
        return None
    vals = [str(a) for a in allowed]
    return vals if len(vals) <= cap else None


#: 子特征清单最多列出多少个（网格/物理场序列可达数百个，避免 dump 爆炸）。
_MAX_LISTED_FEATURES = 60


def describe_selection(sel: Any) -> dict[str, Any]:
    """读一个选择对象的实体/对象读数，单点失败降级为空。

    COMSOL 的选择对象不止一种：几何特征的 ``LocalGeomObjectSelectionClient`` 用
    ``objects()`` 给几何对象 tag、没有 ``entities``；网格/物理场的
    ``MeshSelectionClient``/``XDLocalSelectionClient`` 用 ``entities(dim)``；几何序列的
    ``selection()`` 则是**选择特征列表**（``tags``/``size``）。这里统一成 dict。
    """
    out: dict[str, Any] = {}
    entities = getattr(sel, "entities", None)
    if callable(entities):
        for dim in (0, 1, 2, 3):
            ents = _safe(entities, dim, default=None)
            if ents is None:
                continue
            try:
                out[f"dim{dim}"] = len(ents)
            except TypeError:  # 某些实现返回计数而非序列
                out[f"dim{dim}"] = int(ents)
    objects = getattr(sel, "objects", None)
    if callable(objects):
        objs = _safe(objects, default=None)
        if objs is not None:
            out["objects"] = [str(o) for o in objs]
    tags = _tags(sel)
    if tags:
        out["tags"] = tags
    if not out and not callable(entities):
        # 无 entities 的列表型选择（几何序列的选择特征列表）才把 size 当条目数。
        size = getattr(sel, "size", None)
        if callable(size):
            n = _safe(size, default=None)
            if n is not None:
                out["feature_count"] = int(n)
    return out


def selection_inputs(node: Any) -> list[dict[str, Any]]:
    """列节点的**命名选择输入**与当前实体（COMSOL 无统一枚举 API，这里组合两路）。

    - 几何特征（如 ``PartitionDomains``）：选择输入是 ``Selection`` 型属性
      （``getValueType(prop) == "Selection"``，如 domain/edge/object）；
    - 附加选择（如 ``PeriodicCondition`` 的配对域）：``getExtraSelectionNames()``；
    - 逐个 ``selection(name)`` 读当前实体（几何特征给 objects，其他给各维实体数）。

    未知/不可读之名一律跳过，不抛异常。
    """
    sel_fn = getattr(node, "selection", None)
    if not callable(sel_fn):
        return []
    names: list[str] = []
    for p in _str_list(_call_if(node, "properties", default=[])):
        if str(_call_if(node, "getValueType", p, default="")) == "Selection":
            names.append(p)
    names += _str_list(_call_if(node, "getExtraSelectionNames", default=[]))
    out: list[dict[str, Any]] = []
    for name in dict.fromkeys(names):
        sel = _safe(sel_fn, name, default=None)
        if sel is None:
            continue
        out.append({"name": name, **describe_selection(sel)})
    return out


def _child_features(node: Any) -> tuple[list[dict[str, Any]], int]:
    """列 ``node.feature()`` 子特征（tag/type/label），并返回子特征总数。"""
    feat_fn = getattr(node, "feature", None)
    if not callable(feat_fn):
        return [], 0
    seq = _safe(feat_fn, default=None)
    tags = _tags(seq)
    getter = getattr(seq, "get", None)
    rows: list[dict[str, Any]] = []
    for tag in tags[:_MAX_LISTED_FEATURES]:
        if callable(getter):
            child = _safe(getter, tag, default=None)
        elif callable(seq):
            child = _safe(seq, tag, default=None)
        else:
            child = None
        rows.append(
            {
                "tag": tag,
                "type": str(_call_if(child, "getType", default="") or ""),
                "label": str(_call_if(child, "label", default="") or ""),
            }
        )
    return rows, len(tags)


def _geometry_stats(node: Any) -> dict[str, int]:
    """几何序列/特征的实体计数（其他节点返回空表）。"""
    out: dict[str, int] = {}
    for key, meth in (
        ("vertices", "getNVertices"),
        ("edges", "getNEdges"),
        ("faces", "getNFaces"),
        ("domains", "getNDomains"),
    ):
        val = _call_if(node, meth, default=None)
        if val is not None:
            out[key] = int(val)
    return out


def dump_node(node: Any, *, include_methods: bool = False) -> dict[str, Any]:
    """dump 任意活体节点的状态：类型/标签/属性当前值/枚举允许值/选择与子特征清单。

    与 :func:`introspect`（列方法签名＝活体 javadoc）互补：本函数回答
    “这个节点**现在**是什么类型、每个属性设成了什么、还能设成什么、选择集里有几个实体”。
    全部防御式读取，单点失败降级为占位符（几何节点没有 ``selection().entities`` 属
    性，只按选择特征列表/子特征清单渲染）。
    """
    info: dict[str, Any] = {
        "class": str(
            _call_if(_call_if(node, "getClass", default=None), "getName", default="")
            or ""
        ),
        "type": str(
            _call_if(node, "getType", default="")
            or _call_if(node, "type", default="")
            or ""
        ),
        "label": str(_call_if(node, "label", default="") or ""),
    }
    tags = _tags(node)
    if tags:
        info["tags"] = tags

    props: list[dict[str, Any]] = []
    for p in _str_list(_call_if(node, "properties", default=[])):
        row: dict[str, Any] = {
            "name": p,
            "value": _call_if(node, "getString", p, default=None),
        }
        allowed = _longest_allowed(node, p)
        if allowed:
            row["allowed"] = allowed
        props.append(row)
    info["properties"] = props

    # 默认选择：物理场/网格特征可读各维实体数；几何序列返回的是选择特征列表。
    sel = _call_if(node, "selection", default=None)
    if sel is not None:
        sinfo = describe_selection(sel)
        if sinfo:
            info["selection"] = sinfo

    named = selection_inputs(node)
    if named:
        info["selection_inputs"] = named

    features, n_features = _child_features(node)
    if features:
        info["features"] = features
        if n_features > len(features):
            info["features_truncated"] = n_features - len(features)

    stats = _geometry_stats(node)
    if stats:
        info["geometry"] = stats

    objs = _call_if(node, "objectNames", default=None)
    if objs is not None:
        info["objects"] = [str(o) for o in objs]

    if include_methods:
        info["methods"] = introspect(node)["methods"]
    return info


def _fmt_selection(row: dict[str, Any]) -> str:
    """把选择/计数读数压成一行（``name`` 由调用方另行打印）。"""
    return " ".join(f"{k}={v}" for k, v in row.items() if k != "name") or "-"


def format_node_dump(info: dict[str, Any], *, path: str = "") -> str:
    """把 :func:`dump_node` 的结果格式化为可读文本。"""
    head = f"node: {path}" if path else "node"
    lines = [head, f"  class: {info.get('class') or '?'}"]
    typ, label = info.get("type") or "", info.get("label") or ""
    if typ or label:
        lines.append(
            f"  type : {typ}" + (f"  [{label}]" if label and label != typ else "")
        )
    if info.get("tags"):
        lines.append(f"  tags : {info['tags']}")
    props = info.get("properties") or []
    lines.append(f"  properties: ({len(props)})")
    for row in props:
        line = f"    - {row['name']} = {row['value']!r}"
        if row.get("allowed"):
            line += f"   allowed: {'|'.join(row['allowed'])}"
        lines.append(line)
    if info.get("geometry"):
        lines.append(f"  geometry: {_fmt_selection(info['geometry'])}")
    if info.get("objects") is not None:
        lines.append(f"  objects: {info['objects']}")
    if info.get("features"):
        rows = info["features"]
        lines.append(f"  features: ({len(rows)})")
        for row in rows:
            line = f"    - {row['tag']} :: {row['type'] or '?'}"
            if row.get("label") and row["label"] != row["type"]:
                line += f"  [{row['label']}]"
            lines.append(line)
        if info.get("features_truncated"):
            lines.append(f"    … 另有 {info['features_truncated']} 个未列出")
    if info.get("selection"):
        lines.append(f"  selection: {_fmt_selection(info['selection'])}")
    if info.get("selection_inputs"):
        lines.append(f"  selection inputs: ({len(info['selection_inputs'])})")
        for row in info["selection_inputs"]:
            lines.append(f"    - {row['name']}: {_fmt_selection(row)}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 通用节点属性 setter（与 resolve_path/dump_node 对称的写入端）
# ---------------------------------------------------------------------------
def _coerce_value(raw: str) -> Any:
    """把 CLI 传入的字符串值转成 Java ``set`` 友好的类型。

    整数值用 ``jpype.JInt`` 包裹——COMSOL 的 ``set(String, int)`` 重载在 JPype 下
    传 Python int 可能因重载歧义失败（如 Box 选择的 ``entitydim``）；其余一律按
    字符串传入（COMSOL 属性 setter 对 double/枚举/on-off 都接受字符串形态）。
    """
    try:
        iv = int(raw)
    except ValueError:
        return raw
    try:
        import jpype

        return jpype.JInt(iv)
    except Exception:  # noqa: BLE001 - 无 JPype（测试桩）时退回原字符串
        return raw


@dataclass
class SetPropsResult:
    """:func:`set_node_props` 的结果：逐条成功/失败汇总（失败不中断其余条目）。"""

    path: str
    node_class: str = ""
    ok: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)

    @property
    def all_ok(self) -> bool:
        return not self.failed

    def report(self) -> str:
        lines = [
            f"node set: {self.path}"
            + (f"  ({self.node_class})" if self.node_class else "")
        ]
        lines += [f"  ok   : {k}" for k in self.ok]
        lines += [f"  FAIL : {f}" for f in self.failed]
        lines.append(f"  => {len(self.ok)} ok, {len(self.failed)} failed")
        return "\n".join(lines)


def set_node_props(model: Any, path: str, pairs: list[str]) -> SetPropsResult:
    """按路径定位节点后逐条 ``set(key, value)``（``pairs`` 形如 ``["k=v", ...]``）。

    与 :func:`resolve_path`/:func:`dump_node` 对称的**写入端**：让"改一个属性"零自定义
    代码。单条失败记录后继续（汇总返回），绝不因一个坏属性中断整批设置。

    Raises:
        ValueError: ``pairs`` 为空，或某条缺少 ``=`` 分隔符。
        AttributeError: 路径无法定位（透传 :func:`resolve_path` 的异常）。
    """
    if not pairs:
        raise ValueError("set_node_props：至少需要一条 key=value")
    parsed: list[tuple[str, str]] = []
    for kv in pairs:
        k, sep, v = kv.partition("=")
        if not sep or not k.strip():
            raise ValueError(f"--set 参数应为 key=value 形态：{kv!r}")
        parsed.append((k.strip(), v.strip()))

    node = resolve_path(model, path)
    res = SetPropsResult(
        path=path,
        node_class=str(
            _call_if(_call_if(node, "getClass", default=None), "getName", default="")
            or ""
        ),
    )
    for k, v in parsed:
        try:
            node.set(k, _coerce_value(v))
            res.ok.append(f"{k}={v}")
        except Exception as e:  # noqa: BLE001 - 单条失败汇总，不中断
            res.failed.append(f"{k}={v} :: {type(e).__name__}: {e}")
    return res


# ---------------------------------------------------------------------------
# JPype 运行时自省（活体 javadoc）
# ---------------------------------------------------------------------------
def introspect(node: Any, *, limit: int | None = None) -> dict[str, Any]:
    """用 Java 反射列出任意活体节点的类名与方法签名。

    弥补本机未安装的 ``com.comsol.help.api`` javadoc 插件：对任意 Java 对象，返回其
    运行时类名 + 全部 public 方法签名（去重排序），等价于一份"活体 javadoc"。

    Args:
        node: 一个 JPype 包裹的 Java 对象（如 ``model.java.geom('geom1')``）。
        limit: 最多返回多少个方法签名（None = 全部）。

    Returns:
        ``{"class": str, "methods": [str, ...], "python_attrs": [str, ...]}``。
    """
    result: dict[str, Any] = {"class": None, "methods": [], "python_attrs": []}

    cls = _call_if(node, "getClass", default=None)
    if cls is not None:
        result["class"] = str(_call_if(cls, "getName", default=""))
        methods = _call_if(cls, "getMethods", default=[]) or []
        sigs = sorted({str(m) for m in methods})
        result["methods"] = sigs[:limit] if limit else sigs

    result["python_attrs"] = sorted(a for a in dir(node) if not a.startswith("_"))
    return result


def format_introspection(info: dict[str, Any]) -> str:
    """把 :func:`introspect` 的结果格式化为可读文本。"""
    lines = [f"class: {info.get('class')}", "methods:"]
    lines += [f"  {m}" for m in info.get("methods", [])]
    attrs = info.get("python_attrs", [])
    if attrs:
        lines.append("python attrs: " + ", ".join(attrs))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# .java 模型文件摘要（GUI→Java 兜底 + knowledge 种子）
# ---------------------------------------------------------------------------
# 匹配形如 model.geom("geom1").feature().create("blk1", "Block"); 的语句
_JAVA_MODEL_STMT = re.compile(r"\bmodel\s*\.\s*([A-Za-z_][\w]*)\s*\(")
_JAVA_CREATE = re.compile(r'\.create\s*\(\s*"([^"]+)"\s*,\s*"([^"]+)"')
_JAVA_SET = re.compile(r'\.set\s*\(\s*"([^"]+)"\s*,\s*(.+?)\)\s*;')

#: 顶层访问器 → 归类桶。
_JAVA_BUCKETS: dict[str, str] = {
    "param": "parameters",
    "func": "functions",
    "variable": "variables",
    "material": "materials",
    "modelNode": "components",
    "component": "components",
    "geom": "geometry",
    "physics": "physics",
    "mesh": "mesh",
    "study": "study",
    "sol": "solvers",
    "result": "results",
    "batch": "batch",
    "view": "views",
}


def summarize_java(path: str | Path) -> dict[str, Any]:
    """解析 GUI 导出的 ``.java`` 模型文件，返回结构化摘要（不加载模型）。

    用于：(1) GUI→Java 兜底建模前快速了解模型骨架；(2) 把 ``mphs/*.java`` 转成
    knowledge 库的首批 Java→Python 对照笔记。

    Returns:
        ``{"path", "n_lines", "buckets": {桶: 命中次数}, "created": [(tag, type)], "params": [(name, value)]}``。
    """
    p = Path(path)
    text = p.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    buckets: dict[str, int] = {}
    for m in _JAVA_MODEL_STMT.finditer(text):
        acc = m.group(1)
        bucket = _JAVA_BUCKETS.get(acc, acc)
        buckets[bucket] = buckets.get(bucket, 0) + 1

    created = [(tag, typ) for tag, typ in _JAVA_CREATE.findall(text)]

    params: list[tuple[str, str]] = []
    for name, val in _JAVA_SET.findall(text):
        # 仅收集 param().set("name", "value") 形态（启发式：值以引号或数字开头）
        v = val.strip()
        if v.startswith('"') or re.match(r"^-?\d", v):
            params.append((name, v.strip('"')))

    return {
        "path": str(p),
        "n_lines": len(lines),
        "buckets": buckets,
        "created": created,
        "params": params[:200],
    }


def format_java_summary(info: dict[str, Any]) -> str:
    """把 :func:`summarize_java` 结果格式化为可读文本。"""
    lines = [
        f".java summary: {info['path']}",
        f"  lines: {info['n_lines']}",
        "  buckets (accessor hits):",
    ]
    for b, n in sorted(info["buckets"].items(), key=lambda kv: -kv[1]):
        lines.append(f"    {b:14}: {n}")
    created = info.get("created", [])
    if created:
        lines.append(f"  created nodes ({len(created)}):")
        for tag, typ in created[:80]:
            lines.append(f"    {tag:16} :: {typ}")
    params = info.get("params", [])
    if params:
        lines.append(f"  params set ({len(params)}):")
        for name, val in params[:60]:
            lines.append(f"    {name} = {val}")
    return "\n".join(lines)
