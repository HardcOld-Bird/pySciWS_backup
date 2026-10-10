"""成员注册表与机械验收检查表的读写。

``orchestration/state/registry.json`` 是编排体系的单一配置事实源：成员 → pod 路径、
模型档位、参数模板、会话池索引；check 类型 → 确定性验收命令。全部为可读 JSON
（fail-open 红线，README §4.3）。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pysci.paths import ORCH_STATE_ROOT, ORCHESTRATION_ROOT, PODS_ROOT, PROJECT_ROOT

REGISTRY_PATH: Path = ORCH_STATE_ROOT / "registry.json"

#: 档位 → 具体型号映射（README §3.5）；registry.json 的 models 块覆盖这里的出厂默认。
#: 渠道策略（用户裁决 2026-10-10，credit 耗尽事故复盘）：
#: - ``max`` = ''（不传 -m → 走用户级默认 BYOK；**禁用内置同名模型**——烧订阅 credit）；
#: - ``flash`` = 内置 Qwen3.8-Flash（限时免费，零额度消耗）；
#: - ``<tier>_fallback`` = 该档额度耗尽时的备用渠道（do_dispatch 自动换档重试一次）。
#:   BYOK modelID 账号相关，出厂默认为空=无回退；实际值由组长在 registry.json 配置
#:   （flash_fallback=BYOK Qwen-3.8-Flash，护栏级、须经用户）。
DEFAULT_MODELS: dict[str, str] = {
    "max": "",
    "flash": "Qwen3.8-Flash",
    "flash_fallback": "",
}


def resolve_exe(registry: dict[str, Any] | None = None) -> Path:
    """定位 qoderclicn 原生 exe（`.cmd` 包装器剥引号，程序化调用必须直连 exe）。

    Args:
        registry: 已加载的注册表（可选 ``exe`` 字段覆盖）。

    Returns:
        可执行文件路径。

    Raises:
        FileNotFoundError: 所有候选位置都不存在时。
    """
    candidates: list[Path] = []
    if registry and registry.get("exe"):
        candidates.append(Path(os.path.expanduser(str(registry["exe"]))))
    if env_exe := os.environ.get("PYSCI_ORCH_EXE"):
        candidates.append(Path(os.path.expanduser(env_exe)))
    candidates.append(
        Path.home() / ".qoder-cn" / "bin" / "qoderclicn" / "qoderclicn.exe"
    )
    for cand in candidates:
        if cand.exists():
            return cand
    raise FileNotFoundError(
        f"未找到 qoderclicn 原生 exe，候选：{[str(c) for c in candidates]}"
    )


@dataclass
class Member:
    """一个组员（pod + 会话池 + 参数模板）的注册表条目。"""

    member_id: str
    pod: Path
    model_tier: str = "flash"
    max_turns: int = 60
    timeout_s: int = 7200
    mcp_config: str | None = ".qoder/mcp.json"
    sessions: list[dict[str, Any]] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def active_sessions(self) -> list[dict[str, Any]]:
        """未归档的会话（按 last_active 降序，最近在前）。"""
        act = [s for s in self.sessions if s.get("status") != "archived"]
        return sorted(act, key=lambda s: s.get("last_active", ""), reverse=True)

    def find_session(self, sid: str) -> dict[str, Any] | None:
        """按 sid 前缀匹配会话条目（允许缩写）。"""
        hits = [s for s in self.sessions if str(s.get("sid", "")).startswith(sid)]
        return hits[0] if len(hits) == 1 else None

    def update_session(self, sid: str, **fields: Any) -> None:
        """更新指定会话条目的字段（不存在则忽略）。"""
        for s in self.sessions:
            if s.get("sid") == sid:
                s.update(fields)


@dataclass
class Registry:
    """registry.json 的内存表示与持久化。"""

    data: dict[str, Any]
    path: Path = REGISTRY_PATH

    @classmethod
    def load(cls) -> Registry:
        """读取注册表；不存在时返回带骨架默认值的实例（不落盘）。"""
        if REGISTRY_PATH.exists():
            data = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        else:
            data = {
                "version": 1,
                "exe": None,
                "models": dict(DEFAULT_MODELS),
                "members": {},
                "checks": {},
            }
        return cls(data=data)

    def save(self) -> None:
        """原子写回注册表（先写临时文件再替换，防中断损坏）。"""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(self.data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        tmp.replace(self.path)

    @property
    def models(self) -> dict[str, str]:
        """双档 → 具体型号映射。"""
        return {**DEFAULT_MODELS, **self.data.get("models", {})}

    @property
    def checks(self) -> dict[str, dict[str, Any]]:
        """check 类型 → {cmd: [...含 {path} 占位], cwd: project|pod}。"""
        return self.data.get("checks", {})

    def member(self, member_id: str) -> Member:
        """按 id 取成员条目。

        Raises:
            KeyError: 成员未注册（报错信息附可用成员清单，便于 flash 级模型自纠）。
        """
        members = self.data.get("members", {})
        if member_id not in members:
            raise KeyError(f"成员 '{member_id}' 未注册；可用：{sorted(members)}")
        raw = members[member_id]
        pod_rel = raw.get("pod", f"orchestration/pods/{member_id}")
        pod = Path(pod_rel) if Path(pod_rel).is_absolute() else PROJECT_ROOT / pod_rel
        return Member(
            member_id=member_id,
            pod=pod,
            model_tier=raw.get("model_tier", "flash"),
            max_turns=int(raw.get("max_turns", 60)),
            timeout_s=int(raw.get("timeout_s", 7200)),
            mcp_config=raw.get("mcp_config", ".qoder/mcp.json"),
            sessions=list(raw.get("sessions", [])),
            raw=raw,
        )

    def write_member(self, m: Member) -> None:
        """把 Member 的会话池等可变状态写回底层 dict（save() 前调用）。"""
        entry = self.data.setdefault("members", {}).setdefault(m.member_id, {})
        entry.update(
            {
                "pod": str(m.pod.relative_to(PROJECT_ROOT))
                if m.pod.is_relative_to(PROJECT_ROOT)
                else str(m.pod),
                "model_tier": m.model_tier,
                "max_turns": m.max_turns,
                "timeout_s": m.timeout_s,
                "mcp_config": m.mcp_config,
                "sessions": m.sessions,
            }
        )

    def ensure_member_defaults(self, member_id: str) -> None:
        """为尚未注册的成员生成默认条目（pod 目录存在时才可用，dispatch 前调用）。"""
        members = self.data.setdefault("members", {})
        if member_id not in members:
            pod = PODS_ROOT / member_id
            members[member_id] = {
                "pod": str(pod.relative_to(PROJECT_ROOT)),
                "model_tier": "flash",
                "max_turns": 60,
                "timeout_s": 7200,
                "mcp_config": ".qoder/mcp.json",
                "sessions": [],
            }


def pod_session_key(pod: Path) -> str:
    """把 pod 绝对路径映射为 Qoder 的项目存储键（实测规则：非字母数字逐字符替换为 '-'）。

    Args:
        pod: pod 目录绝对路径。

    Returns:
        ``~/.qoder-cn/projects/`` 下的目录名，如
        ``D--XXXIIIGGG-projects-pySci-pySciWS-orchestration-pods-figure``。
    """
    import re

    return re.sub(r"[^A-Za-z0-9]", "-", str(pod.resolve()))


def pod_sessions_dir(pod: Path) -> Path:
    """pod 对应的项目级会话存储目录（jsonl 落盘处，Phase 0 实测确认按 cwd 键分储）。"""
    return Path.home() / ".qoder-cn" / "projects" / pod_session_key(pod)


def orchestration_rel(path: Path) -> str:
    """把绝对路径转为相对项目根的显示串（越界时原样返回）。"""
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


#: 便于 doctest/调试的常量再导出。
__all__ = [
    "DEFAULT_MODELS",
    "Member",
    "ORCHESTRATION_ROOT",
    "REGISTRY_PATH",
    "Registry",
    "orchestration_rel",
    "pod_session_key",
    "pod_sessions_dir",
    "resolve_exe",
]
