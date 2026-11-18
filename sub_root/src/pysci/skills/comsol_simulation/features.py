"""feature-types 试建探测 + node set 提取/批量属性写入。

COMSOL Java API **没有**「枚举某节点允许的子特征类型」接口（旧项目实测），agent 只能
试建猜名——本模块把「试建猜」机械化为一次批量探针（:func:`probe_feature_types`）：
候选类型逐个 create，成功即合法，随后删除探针节点；结论以**本模型实测**为准。

node set（:func:`collect_nodes` + :func:`set_node_props`）：按路径模式批量收集节点、
对称写属性（``--set k=v`` 语义）——改 N 个同类特征的同一属性是 agent 高频刚需。

路径语法（:func:`parse_node_path`）：``component(comp1).physics(acpr).feature(bpf1)``
——段名（component/physics/feature…）只作语义标注，查找以括号内 **tag** 为准；
``*`` 通配段可配 :func:`collect_nodes` 的模式匹配。
"""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass, field

_SEGMENT_RE = re.compile(r"([A-Za-z_]\w*)\(([^()]+)\)|([A-Za-z_][\w*]*|\*)")


def parse_node_path(path: str) -> list[tuple[str | None, str]]:
    """解析节点路径 → ``[(段名, tag)]``（纯函数）。

    ``"component(comp1).physics(acpr)"`` → ``[("component", "comp1"),
    ("physics", "acpr")]``；裸段 ``"result"`` → ``[(None, "result")]``。

    Args:
        path: 节点路径。

    Returns:
        段元组列表。

    Raises:
        ValueError: 路径为空或含无法解析的段。
    """
    if not path.strip():
        raise ValueError("节点路径为空")
    segments: list[tuple[str | None, str]] = []
    for part in path.split("."):
        m = _SEGMENT_RE.fullmatch(part.strip())
        if not m:
            raise ValueError(f"无法解析路径段：{part!r}（全路径 {path!r}）")
        kind, tag, bare = m.groups()
        segments.append((kind, tag) if tag is not None else (None, bare))
    return segments


def match_path(path: str, pattern: str) -> bool:
    """路径通配匹配（纯函数）：段 tag 逐段 fnmatch（``*``/``?`` 通配，``pg*`` 可用）。

    Args:
        path: 具体节点路径。
        pattern: 通配模式（如 ``result(pg*).feature(*)``）。

    Returns:
        是否匹配。
    """
    segs = parse_node_path(path)
    pats = parse_node_path(pattern)
    if len(segs) != len(pats):
        return False
    return all(
        fnmatch.fnmatchcase(stag, ptag)
        for (_, stag), (_, ptag) in zip(segs, pats, strict=True)
    )


@dataclass
class FeatureTypesReport:
    """feature-types 探测结论。

    Attributes:
        existing: 既有子特征 ``[(tag, 类型名)]``。
        valid: 探测成功的候选类型。
        invalid: 试建失败的候选类型。
        unknown: 未探测的候选（父节点无 create 能力时原样保留）。
        probes_left: 未能删除的探针节点数（>0 ⇒ 模型留痕，勿直接保存）。
    """

    existing: list[tuple[str, str]] = field(default_factory=list)
    valid: list[str] = field(default_factory=list)
    invalid: list[str] = field(default_factory=list)
    unknown: list[str] = field(default_factory=list)
    probes_left: int = 0


def existing_feature_types(node: object) -> list[tuple[str, str]]:
    """列出节点的既有子特征 ``(tag, 类型名)``（驱动；鸭子类型）。

    Args:
        node: 父节点（``children()`` 或 ``features()``，元素有 tag/name）。

    Returns:
        子特征元组列表。
    """
    out: list[tuple[str, str]] = []
    for attr in ("children", "features"):
        fn = getattr(node, attr, None)
        if not callable(fn):
            continue
        for child in fn():
            tag = _member(child, "tag") or str(child)
            typ = _member(child, "type") or type(child).__name__
            out.append((str(tag), str(typ)))
        break
    return out


def _member(obj: object, name: str) -> object | None:
    """取对象成员（属性或零参方法），鸭子类型通用。"""
    v = getattr(obj, name, None)
    if callable(v):
        try:
            return v()
        except Exception:  # noqa: BLE001 - 鸭子类型探测，失败即视为无
            return None
    return v


def probe_feature_types(
    node: object, candidates: list[str], *, probe_name: str = "_ftprobe"
) -> FeatureTypesReport:
    """批量试建探测合法子特征类型（驱动；结论以本模型实测为准）。

    每个候选：``node.create(候选, name=<probe_name><i>)``（mph Node API：首参即类型名）
    ——成功即合法并随即删除探针节点；失败即非法。删除不掉的探针计入 ``probes_left``
    警示模型留痕。

    Args:
        node: 父节点（mph ``Node``：``create(*type, name=…)`` 返回子节点句柄）。
        candidates: 候选类型名列表。
        probe_name: 探针 label 前缀（避免与用户特征混淆）。

    Returns:
        :class:`FeatureTypesReport`。
    """
    rep = FeatureTypesReport(existing=existing_feature_types(node))
    create = getattr(node, "create", None)
    if not callable(create):
        rep.unknown = list(candidates)
        return rep
    for i, cand in enumerate(candidates):
        try:
            handle = create(cand, name=f"{probe_name}{i}")
        except Exception:  # noqa: BLE001 - 类型名不合法即 Java 异常，属预期探测
            rep.invalid.append(cand)
            continue
        rep.valid.append(cand)
        if not _remove_feature(handle):
            rep.probes_left += 1
    return rep


def _remove_feature(handle: object) -> bool:
    """删除探针节点（mph ``Node.remove()`` 兜底 ``parent.remove(tag)``）；成功返回 True。"""
    remove = getattr(handle, "remove", None)
    if callable(remove):
        try:
            remove()
            return True
        except Exception:  # noqa: BLE001 - API 形态差异，换下一招
            pass
    return False


def format_feature_types(rep: FeatureTypesReport, *, path: str = "") -> str:
    """把探测结论渲染成人话报告。"""
    lines = [f"[feature-types] {path}", "  既有子特征："]
    lines += [f"    {tag} : {typ}" for tag, typ in rep.existing] or ["    （无）"]
    lines.append(f"  合法候选：{', '.join(rep.valid) or '（无）'}")
    lines.append(f"  非法候选：{', '.join(rep.invalid) or '（无）'}")
    if rep.unknown:
        lines.append(f"  未探测：{', '.join(rep.unknown)}（父节点无 create 能力）")
    if rep.probes_left:
        lines.append(f"  [!] {rep.probes_left} 个探针未能删除——模型可能留痕，勿直接保存")
    return "\n".join(lines)


def coerce_value(raw: str) -> tuple[object, str]:
    """``k=v`` 的 v 值强制转换（纯函数）：int/float/bool 优先，否则原字符串。

    Args:
        raw: 原始值字符串。

    Returns:
        ``(值, 种类)``——种类 ∈ ``{"int","float","bool","str"}``。
    """
    low = raw.strip().lower()
    if low in ("true", "false"):
        return (low == "true", "bool")
    try:
        return (int(raw), "int")
    except ValueError:
        pass
    try:
        return (float(raw), "float")
    except ValueError:
        return (raw, "str")


@dataclass
class SetPropsResult:
    """node set 批量写属性的结果。

    Attributes:
        applied: 成功的 ``"k=v"``。
        failed: 失败的 ``"k=v"``（单条失败不中断，汇总报告）。
    """

    applied: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)

    @property
    def all_ok(self) -> bool:
        """是否全部成功。"""
        return not self.failed


def set_node_props(node: object, pairs: list[str]) -> SetPropsResult:
    """对称写节点属性（``["k=v", …]``；mph ``Node.property(name, value)``）。

    JPype 坑（java-mph-pitfalls §JPype）：Python int 传 Java ``set(String, int)`` 重载
    歧义 → JVM 运行中时用 ``JInt`` 包裹；mock 测试无 JVM 则原样传 int。
    单条失败汇总不中断（Java 端属性缺失属常见探测结果）。

    Args:
        node: 目标节点（mph ``Node``）。
        pairs: ``"k=v"`` 列表。

    Returns:
        :class:`SetPropsResult`。

    Raises:
        ValueError: 条目不含 ``=``。
    """
    res = SetPropsResult()
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"属性条目须为 k=v 形式：{pair!r}")
        key, raw = pair.split("=", 1)
        value, kind = coerce_value(raw)
        if kind == "int":
            value = _jint(value)
        try:
            node.property(key.strip(), value)
            res.applied.append(pair)
        except Exception:  # noqa: BLE001 - 单条失败汇总不中断（Java 端属性缺失）
            res.failed.append(pair)
    return res


def _jint(value: int) -> object:
    """JVM 运行中时包裹 ``JInt``（消 set(String,int) 重载歧义）；否则原样传 int。

    注意：``jpype.JInt(...)`` 只在 JVM 启动后可用（mock 测试/无 JVM 场景直接传 int）。
    """
    try:
        import jpype
    except ImportError:
        return value
    try:
        if jpype.isJVMStarted():
            return jpype.JInt(value)
    except Exception:  # noqa: BLE001 - jpype 形态差异兜底
        pass
    return value


def collect_nodes(model: object, pattern: str) -> list[str]:
    """按通配路径模式收集模型内匹配节点的路径（驱动；树遍历 + :func:`match_path`）。

    Args:
        model: 模型根节点（``children()`` 递归）。
        pattern: 通配路径模式（``*`` 单段通配）。

    Returns:
        匹配到的具体节点路径列表。
    """
    hits: list[str] = []

    def walk(node: object, prefix: str) -> None:
        tag = _member(node, "tag") or _member(node, "name")
        here = f"{prefix}.{tag}" if prefix and tag else (prefix or str(tag or ""))
        if here and match_path(here, pattern):
            hits.append(here)
        fn = getattr(node, "children", None)
        if callable(fn):
            for child in fn():
                walk(child, here)

    for child in _root_children(model):
        walk(child, "")
    return hits


def _root_children(model: object) -> list[object]:
    """模型根的直接子节点（mph ``children()``；鸭子类型）。"""
    fn = getattr(model, "children", None)
    return list(fn()) if callable(fn) else []
