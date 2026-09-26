"""模型自省：让 Agent「读懂」一个 .mph 的结构与任意 Java 节点的能力。

四类能力：
- :func:`dump_tree` — 紧凑文本化模型树（组件/几何/材料/物理场/网格/研究/求解/结果 + 参数），供 Agent 读结构。
- :func:`list_parameters` / :func:`inventory` — 参数表与按子系统的节点清单。
- :func:`introspect` — 对任意活体 Java 节点用反射列出类名与方法签名（**活体 javadoc**，弥补未安装的 javadoc 插件）。
- :func:`summarize_java` — 把 GUI 导出的 ``.java`` 模型文件解析成结构化摘要（GUI→Java 兜底路径与 knowledge 种子用）。

所有 live COMSOL 调用都做了防御式包裹：探测方法存在性、逐个 try/except，任何单点失败都降级为占位符，
绝不因某个节点不可读而中断整棵树的 dump。
"""

from __future__ import annotations

import re
from collections import OrderedDict
from collections.abc import Callable
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
def list_parameters(model: Any) -> "OrderedDict[str, dict[str, str]]":
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
        descr = _call_if(p, "descr", n, default="") or _call_if(p, "getdescr", n, default="")
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

    result["python_attrs"] = sorted(
        a for a in dir(node) if not a.startswith("_")
    )
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
