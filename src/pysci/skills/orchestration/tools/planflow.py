"""plan 编排树（superstep）解析与**递归就绪谓词**（PSEP_3 §2.1/§2.2，M-A 引擎）。

plan YAML ``steps[]`` 元素二型（递归）：

- **普通 step**：现有字段不变（``id/member/brief/dirs/model/effort/review``）；
- **superstep**：``{id, mode: serial|parallel, children: [...]}``，``children`` 可继续嵌套。

**叶子在 state.json 扁平化为点路径 id**（``a2.a2a.a2a-1``）——``_reconcile`` 按 id 对账
机制不变；superstep 的聚合状态由叶子**推导、不落盘**；顶层 ``steps[]`` 视为**隐式
serial 组**——故**存量扁平 plan 行为完全不变**（等价单层串行）。

就绪谓词（递归适用）：

- ``started(node)``：step＝已入队（queued/done/blocked/failed 皆算「已派发」）；
  superstep＝其任一叶子 started；
- ``done(node)``：step＝done/skipped；superstep＝其全部叶子 done（空组视为 done）；
- **叶子派发就绪**＝对它所在**每个**分组（顶层组或任一祖先 superstep）：组 ``mode:
  serial`` → 其**前缀兄弟**逐个 ``done``（完成级依赖）；组 ``mode: parallel`` → 逐个
  ``started``（**派发级依赖**：id 靠前分支派发后即可派发靠后分支，不等其完成）。

由此「一条串行链整体与兄弟并行」＝并行组里放一个 serial superstep；更深依赖靠嵌套
表达，**不引入第三种 gate 语义**。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

#: 「已派发」状态族（started 的 step 判据）——凡离开 pending 即已在途/已有结果。
STARTED_STATES: tuple[str, ...] = (
    "queued",
    "done",
    "skipped",
    "blocked",
    "failed",
    "run_failed",
)

#: 「完成」状态族（done 的 step 判据）。
DONE_STATES: tuple[str, ...] = ("done", "skipped")


@dataclass(frozen=True)
class Node:
    """编排树节点（step 或 superstep）；``path`` 为点路径（顶层 == name）。"""

    name: str
    path: str
    is_group: bool
    mode: str  # 组：serial|parallel；叶子：""
    yaml: dict[str, Any] = field(default_factory=dict)
    index: int = 0
    parent_path: str = ""


class PlanTree:
    """解析后的编排树：叶子清单/父链/就绪谓词（全部纯查询，无 I/O）。"""

    def __init__(self, steps_yaml: list | None) -> None:
        self.nodes: list[Node] = []
        self.leaves: list[Node] = []
        self._by_path: dict[str, Node] = {}
        self._children: dict[str, list[Node]] = {}  # 组 path（顶层 ""）→ 子节点
        self._prefix: dict[str, list[tuple[str, list[str]]]] = {}
        self._parse(list(steps_yaml or []), parent_path="")

    # ------------------------------------------------------------------ 解析
    def _parse(self, items: list, *, parent_path: str) -> None:
        children: list[Node] = []
        for i, raw in enumerate(items):
            if not isinstance(raw, dict):
                continue
            name = str(raw.get("id") or f"s{i + 1}")
            path = f"{parent_path}.{name}" if parent_path else name
            kids = raw.get("children")
            is_group = isinstance(kids, list) and bool(kids)
            mode = str(raw.get("mode") or "serial") if is_group else ""
            if is_group:
                mode = "parallel" if mode == "parallel" else "serial"
            node = Node(
                name=name,
                path=path,
                is_group=is_group,
                mode=mode,
                yaml=raw,
                index=i,
                parent_path=parent_path,
            )
            self.nodes.append(node)
            self._by_path[path] = node
            children.append(node)
            if is_group:
                self._parse(kids, parent_path=path)
            else:
                self.leaves.append(node)
        self._children[parent_path] = children

    # ------------------------------------------------------------- 只读面
    def leaf_paths(self) -> list[str]:
        """叶子点路径（**深度优先序**＝推进/引导序）。"""
        return [n.path for n in self.leaves]

    def leaf_map(self) -> dict[str, dict[str, Any]]:
        return {n.path: n.yaml for n in self.leaves}

    def is_leaf(self, path: str) -> bool:
        n = self._by_path.get(path)
        return bool(n and not n.is_group)

    def prefix_siblings(self, path: str) -> list[tuple[str, list[str]]]:
        """该叶子的就绪校验清单：``[(组 mode, 前缀兄弟 path 列表), …]``（顶层组在前）。

        顶层 ``steps[]`` 视为**隐式 serial 组**；superstep 组用自身 ``mode``。
        """
        if path in self._prefix:
            return self._prefix[path]
        chain: list[tuple[str, list[str]]] = []
        node = self._by_path[path]
        seen: set[str] = set()
        cur = node
        while cur is not None:
            group_path = cur.parent_path
            if group_path in seen:  # 防御：病态环（YAML 不会，但别死循环）
                break
            seen.add(group_path)
            sibs = self._children.get(group_path, [])
            group_mode = (
                "serial" if group_path == "" else self._by_path[group_path].mode
            )
            chain.append((group_mode, [s.path for s in sibs[: cur.index]]))
            cur = self._by_path.get(group_path)
        chain.reverse()  # 顶层组在前（顺序无关判定，仅为可读性）
        self._prefix[path] = chain
        return chain

    # ------------------------------------------------------------- 谓词面
    @staticmethod
    def _status_of(status_of: Callable[[str], str], path: str) -> str:
        return str(status_of(path) or "pending")

    def started(self, path: str, status_of: Callable[[str], str]) -> bool:
        """``started(node)``：step=已入队；superstep=任一叶子已入队（递归）。"""
        node = self._by_path.get(path)
        if node is None:
            return False
        if not node.is_group:
            return self._status_of(status_of, path) in STARTED_STATES
        return any(
            self.started(leaf.path, status_of)
            for leaf in self.leaves
            if leaf.path == path or leaf.path.startswith(path + ".")
        )

    def done(self, path: str, status_of: Callable[[str], str]) -> bool:
        """``done(node)``：step=done/skipped；superstep=全部叶子 done（递归）。"""
        node = self._by_path.get(path)
        if node is None:
            return False
        if not node.is_group:
            return self._status_of(status_of, path) in DONE_STATES
        subs = [
            leaf.path
            for leaf in self.leaves
            if leaf.path == path or leaf.path.startswith(path + ".")
        ]
        return all(self.done(p, status_of) for p in subs)

    def ready(self, path: str, status_of: Callable[[str], str]) -> bool:
        """叶子派发就绪：对其**每个**分组，前缀兄弟按组 mode 满足 done/started。"""
        for mode, prefix in self.prefix_siblings(path):
            if mode == "parallel":
                if not all(self.started(p, status_of) for p in prefix):
                    return False
            else:
                if not all(self.done(p, status_of) for p in prefix):
                    return False
        return True

    def dispatchable(
        self, status_of: Callable[[str], str], *, status: str = "pending"
    ) -> list[str]:
        """可派发叶子（DFS 序）：状态为 ``status`` 且 :func:`ready`。"""
        return [
            leaf.path
            for leaf in self.leaves
            if self._status_of(status_of, leaf.path) == status
            and self.ready(leaf.path, status_of)
        ]

    def aggregate(self, path: str, status_of: Callable[[str], str]) -> dict[str, bool]:
        """superstep 聚合状态（由叶子推导，不落盘）。"""
        return {
            "started": self.started(path, status_of),
            "done": self.done(path, status_of),
        }

    def progress(self, path: str, status_of: Callable[[str], str]) -> dict[str, int]:
        """子树进度读数（叶子 done/总、在途、待决策）——[NEXT] 卡片用。"""
        subs = [
            leaf.path
            for leaf in self.leaves
            if leaf.path == path or leaf.path.startswith(path + ".")
        ]
        st = [self._status_of(status_of, p) for p in subs]
        return {
            "total": len(subs),
            "done": sum(1 for s in st if s in DONE_STATES),
            "inflight": sum(1 for s in st if s == "queued"),
            "need_leader": sum(
                1 for s in st if s in ("blocked", "failed", "run_failed")
            ),
        }


__all__ = [
    "DONE_STATES",
    "STARTED_STATES",
    "Node",
    "PlanTree",
]
