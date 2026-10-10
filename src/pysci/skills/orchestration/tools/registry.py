"""成员注册表与机械验收检查表的读写。

``orchestration/state/registry.json`` 是编排体系的单一配置事实源：成员 → pod 路径、
模型档位、参数模板、会话池索引；check 类型 → 确定性验收命令。全部为可读 JSON
（fail-open 红线，README §4.3）。
"""

from __future__ import annotations

import json
import os
import re
import subprocess
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

#: 档位 → ``--list-models`` 人类名精确匹配表（用户裁决 2026-10-10，模型映射自动刷新）；
#: registry.json 的 ``model_patterns`` 块可逐档覆盖。BYOK 命名是账号级配置（当前
#: Qwen-3.8-Max/Flash），改名后更新这里或 registry——**精确匹配**是硬要求：内置
#: ``Qwen3.8-Flash`` 与 BYOK ``Qwen-3.8-Flash`` 仅差一个连字符，模糊匹配必串档。
DEFAULT_MODEL_PATTERNS: dict[str, str] = {
    "max": "Qwen-3.8-Max",
    "flash_fallback": "Qwen-3.8-Flash",
    "flash": "Qwen3.8-Flash",
}

#: --list-models 的 BYOK 行：``人类名 (uuid)``；内置行为裸名（无 UUID）。
_MODEL_LINE_RE = re.compile(
    r"^(?P<name>.+?)\s+\((?P<uuid>[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}"
    r"-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})\)\s*$"
)


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
    #: 成员级推理强度标注（可选，档位名见 runner.VALID_EFFORTS）。空 = 不覆盖，跟随
    #: 用户级默认（中）。派发旗标 --effort / 计划环节 effort 覆盖它（backlog
    #: 20261010-orch-effort-per-item，用户裁决 2026-10-10：不做全局 high，按项标注）。
    effort: str = ""
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
            effort=str(raw.get("effort", "") or ""),
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
        # effort 是**可选**标注：非空才落键，空则移除——registry 里少一个恒为 "" 的
        # 噪声键，且组长手改清空后不会残留旧档位。
        if m.effort:
            entry["effort"] = m.effort
        else:
            entry.pop("effort", None)

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


# ---------------------------------------------------------------------------
# 模型映射自动刷新（用户裁决 2026-10-10：--list-models 是唯一 name→UUID 目录，
# settings.json 只有当前激活模型 UUID，无法监听；重配 BYOK 后下次 drain 自愈，零 daemon）
# ---------------------------------------------------------------------------
def list_models_output(*, timeout_s: int = 60) -> str:
    """运行原生 exe ``--list-models``，返回合并输出（账户级查询，实测秒级）。

    Raises:
        FileNotFoundError: exe 未找到（resolve_exe）。
        OSError / subprocess.TimeoutExpired: 调用失败/超时——是否 fail-open 由调用方决定。
    """
    exe = resolve_exe()
    proc = subprocess.run(
        [str(exe), "--list-models"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_s,
    )
    return (proc.stdout or "") + (proc.stderr or "")


def parse_model_catalog(output: str) -> dict[str, str]:
    """解析 --list-models 输出 → ``{人类名: UUID 或名自身}``。

    实测格式（2026-10-10）：表头行 ``MODEL``；BYOK 行 ``Name (uuid)``；内置行裸名
    （无 UUID）→ 值即人类名本身（-m 直接可用）。空行/表头跳过，其余裸名行按内置收录。
    """
    catalog: dict[str, str] = {}
    for line in (output or "").splitlines():
        line = line.strip()
        if not line or line.upper() == "MODEL":
            continue
        m = _MODEL_LINE_RE.match(line)
        if m:
            catalog[m.group("name").strip()] = m.group("uuid")
        else:
            catalog[line] = line
    return catalog


def refresh_models(*, timeout_s: int = 60) -> dict[str, Any]:
    """按 --list-models 刷新 registry.models（drain 启动自愈 + 手动 refresh-models）。

    以 model_patterns（registry.json 可覆盖 :data:`DEFAULT_MODEL_PATTERNS`）**精确匹配**
    人类名；命中且与现值不同 → 写回 registry。未命中的档**保持现值不清空**——目录缺名
    多为账户瞬时状态，宁旧勿空（清空 max 会静默改变计费渠道）。

    Returns:
        ``{"changed": {tier: {"old":…, "new":…}}, "missing": ["tier←名"…],
        "catalog_size": n}``。

    Raises:
        ValueError: 目录解析为空（输出格式变化/exe 异常）；
        以及 :func:`list_models_output` 的子进程异常——调用方 fail-open。
    """
    catalog = parse_model_catalog(list_models_output(timeout_s=timeout_s))
    if not catalog:
        raise ValueError("--list-models 输出解析为空目录（格式变化或 exe 异常？）")
    reg = Registry.load()
    patterns = {**DEFAULT_MODEL_PATTERNS, **reg.data.get("model_patterns", {})}
    models = reg.data.setdefault("models", {})
    changed: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for tier, name in patterns.items():
        if name not in catalog:
            missing.append(f"{tier}←{name}")
            continue
        new = catalog[name]
        old = models.get(tier)
        if old != new:
            models[tier] = new
            changed[tier] = {"old": old, "new": new}
    if changed:
        reg.save()
    return {"changed": changed, "missing": missing, "catalog_size": len(catalog)}


def pod_session_key(pod: Path) -> str:
    """把 pod 绝对路径映射为 Qoder 的项目存储键（实测规则：非字母数字逐字符替换为 '-'）。

    Args:
        pod: pod 目录绝对路径。

    Returns:
        ``~/.qoder-cn/projects/`` 下的目录名，如
        ``D--XXXIIIGGG-projects-pySci-pySciWS-orchestration-pods-figure``。
    """
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
