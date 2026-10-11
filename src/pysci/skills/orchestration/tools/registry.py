"""成员注册表与机械验收检查表的读写。

配置面与运行态**分储**（根治 163347/220711 两笔同构合并阻塞，backlog
20261010-registry-runtime-split）：

- ``orchestration/state/registry.json`` = **低频配置事实源**（git 跟踪）：成员 → pod
  路径、模型档位、参数模板、checks；随代码评审演进，人手编辑。
- ``orchestration/state/sessions-runtime.json`` = **高频运行态**（gitignore）：会话池
  索引与心跳字段（sid/hops/last_active/chars_offset/status）。每次 dispatch 一跳都改写
  它——留在 registry.json 里会让 main 工作区每派发必 dirty，任何碰 registry 的 worktree
  交付 merge 前检查即被心跳拦下。

首次 :meth:`Registry.load` 检测到旧配置内嵌 ``sessions`` 且运行态文件缺席，即一次性
迁移：备份原文件（``registry.presplit.json``，已存在则不覆盖）→ 抬高会话池到运行态 →
从配置剥离，两份原子落盘。全部为可读 JSON（fail-open 红线，README §4.3）。

**配置面磁盘权威**（backlog 20261011-220621-leader，lost-update 修复）：分储只搬走了心跳，
没搬走「整体回写内存副本」这个坏习惯——长驻 drain worker 收尾时仍把启动时加载的
registry.json 快照原样写回，后写覆盖，抹掉了运行期间合并进来的配置改动。故
:meth:`Registry.save` 改为**读-改-写按键合并**：落盘前重读磁盘，配置面只叠加本进程
*显式声明*过的改动（:meth:`Registry.declare_config` 改 models/checks，
:meth:`Registry.ensure_member_defaults` 新增成员），其余键一律以磁盘为准；无任何声明时
配置文件**一个字节都不写**。会话池经 :meth:`Registry.write_sessions` 按成员、按 sid
合并进运行态。机器写配置面因此只剩「声明」这一条通路。
"""

from __future__ import annotations

import copy
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pysci.paths import ORCH_STATE_ROOT, ORCHESTRATION_ROOT, PODS_ROOT, PROJECT_ROOT

REGISTRY_PATH: Path = ORCH_STATE_ROOT / "registry.json"

#: 运行态会话池文件名（落在 registry.json 同目录；gitignore，见模块 docstring）。
RUNTIME_NAME: str = "sessions-runtime.json"

#: 一次性迁移的原文备份文件名（首次 cutover 时创建，已存在则不覆盖——保留真·pre-split 快照）。
BACKUP_NAME: str = "registry.presplit.json"

#: 档位 → 具体型号映射（README §3.5）；registry.json 的 models 块覆盖这里的出厂默认。
#: 渠道策略（用户裁决 2026-10-10，credit 耗尽事故复盘）：
#: - ``max`` = ''（不传 -m → 走用户级默认 BYOK；**禁用内置同名模型**——烧订阅 credit）；
#: - ``flash`` = 内置 Qwen3.8-Flash（限时免费，零额度消耗）；
#: - ``<tier>_fallback`` = 该档额度耗尽时的备用渠道（do_dispatch 自动换档重试一次）。
#:   BYOK modelID 账号相关，出厂默认为空=无回退；实际值由组长在 registry.json 配置
#:   （flash_fallback=BYOK Qwen-3.8-Flash，护栏级、须经用户）。
#: - ``minimum`` = harness 压测专用最低成本档（REFACTOR §1.1，驱动 ``orch eval run``）。
#:   ``registry.models`` 改动属护栏级：本档的**设立**经用户裁决（§1.1），而**型号选择**
#:   原属专门环节，委托期内由组长代决（§9.4）——暂映射内置 ``Qwen3.8-Flash``（现唯一
#:   零成本渠道，credits 恒 0 已实锤；决策已入 state/preference-journal.jsonl）。
#:   **重估条件**：E3 SiliconFlow 接入后，若拿到更廉价的压测渠道即改这里或写 registry.json
#:   的 ``models.minimum``（registry 覆盖优先）；档位抽象与本型号解耦，套件案不写死型号。
DEFAULT_MODELS: dict[str, str] = {
    "max": "",
    "flash": "Qwen3.8-Flash",
    "minimum": "Qwen3.8-Flash",
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
    "minimum": "Qwen3.8-Flash",
}

#: --list-models 的 BYOK 行：``人类名 (uuid)``；内置行为裸名（无 UUID）。
_MODEL_LINE_RE = re.compile(
    r"^(?P<name>.+?)\s+\((?P<uuid>[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}"
    r"-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})\)\s*$"
)

#: est_tokens 换算系数（**token/字符**）——启发式，非测量值。用户裁决 2026-10-10
#: （backlog 20261010-ledger-est-tokens）：双渠道 envelope/jsonl 逐轮 usage 实锤全为 0、
#: 平台网关无 usage/balance 路由 → 绝对精度不可得，改用「会话 jsonl 字符增量」做
#: **相对排名**。既然只做排名，系数取任何固定值都不改变排序结果，只影响绝对数字的可读性；
#: 关键是**双渠道同法同尺**（BYOK 与内置跳用同一系数，才可比）。
#: 标定依据（2026-10-10 实测 devops 主会话 jsonl，6.28 M 字符）：非 ASCII 占比 4.0%、
#: 中文以原生 UTF-8 存储（全文件仅 6 处 ``\uXXXX`` 转义）→ 字符数不被转义膨胀扭曲，
#: 混合中英取 0.6 token/字符作粗估。registry.json 顶层 ``est_tokens_per_char`` 可覆盖。
#: **绝对量偏高约 3–4×**（同会话实测对照：整段 est≈3.52 M tok，而 ctx_ratio 峰 0.9246
#: × jsonl 内 contextWindow 1,000,000 ≈ 925 k 真实上下文 token）——jsonl 里的 uuid/
#: timestamp/工具元数据不进上下文。常系数在**相对排名**中抵消，不影响裁决用途；若要拿
#: 绝对值对外汇报，先按上面这条对照重新标定，别默认 0.6 是测量值。
DEFAULT_TOKENS_PER_CHAR = 0.6

#: Tier A 阈值自动立案参数（REFACTOR §2.4，用户裁决 v0.5）：同一 ``(标签, 成员)`` 在
#: **滑窗内**（``window_days`` 天 或 ``window_hops`` 跳，取并集）失败标签出现
#: ``threshold`` 次 → orch 在 status/approve 入口自动登记基建提案。设计要点：
#: 计数、立案全代码完成，人只在强制选择点出现——「组长有空去看」不是机制。
#: registry.json 顶层 ``tier_a`` 块可逐键覆盖（非法值回落默认，见 :meth:`Registry.tier_a`）。
DEFAULT_TIER_A: dict[str, int] = {
    "window_days": 14,
    "window_hops": 20,
    "threshold": 3,
}


#: Tier C 仲裁团配置（REFACTOR §4，E3-prep）。**panel 默认空**：选型权归用户
#: （SiliconFlow 账号与付费面是用户的），代码只负责一条硬规则——型号串含 ``qwen``
#: 即拒开庭（排除同族自恋偏差），不做模型能力判断也不内置候选清单。空 panel 的语义
#: 是「尚未选型」，``orch jury`` 据此降级入队而不是报错。
#: registry.json 顶层 ``jury`` 块逐键覆盖（非法值回落默认，见 :meth:`Registry.jury`）：
#:
#: .. code-block:: json
#:
#:    {"jury": {"panel": ["deepseek-ai/DeepSeek-V3", "moonshotai/Kimi-K2-Instruct"],
#:              "pricing": {"deepseek-ai/DeepSeek-V3": {"input": 2.0, "output": 8.0}}}}
#:
#: ``pricing`` 单位 = 元/百万 token（SiliconFlow 刊例按型号而定）；缺席 → 记录里
#: ``cost.amount`` 为 null 并附 note，**不猜价**——成本数字要能对外负责。
DEFAULT_JURY: dict[str, Any] = {
    "panel": [],
    "base_url": "https://api.siliconflow.cn/v1",
    "call_timeout_s": 120,
    "deadline_s": 300,
    "pricing": {},
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


def _atomic_write(path: Path, obj: Any) -> None:
    """原子写 JSON（先写 ``*.tmp`` 再 replace，防中断损坏）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    tmp.replace(path)


def _empty_runtime() -> dict[str, Any]:
    """运行态骨架（会话池唯一落点）。"""
    return {"version": 1, "members": {}}


def _blank_config() -> dict[str, Any]:
    """配置面骨架（文件缺席时的合并底本，与 :meth:`Registry.load` 的首次形状一致）。"""
    return {
        "version": 1,
        "exe": None,
        "models": dict(DEFAULT_MODELS),
        "members": {},
        "checks": {},
    }


def _runtime_member(runtime: dict[str, Any], member_id: str) -> dict[str, Any]:
    """取/建运行态里某成员的条目（含 sessions 列表）。"""
    entry = runtime.setdefault("members", {}).setdefault(member_id, {})
    entry.setdefault("sessions", [])
    return entry


def _read_for_merge(path: Path, blank: dict[str, Any]) -> dict[str, Any]:
    """读盘作为合并底本：文件缺席→``blank`` 深拷贝；损坏→照原样抛。

    损坏不静默兜底成骨架：那等于用空数据覆盖一份「可能还能救」的 state 文件，正是本
    次要根治的 lost-update 的反面（与 :meth:`Registry.load` 同纪律——坏 JSON 硬失败）。
    """
    if not path.exists():
        return copy.deepcopy(blank)
    return json.loads(path.read_text(encoding="utf-8"))


def _merge_sessions(
    disk: list[dict[str, Any]], ours: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """会话池按键（sid）合并：本进程条目为准，磁盘独有 sid 保留（顺序=本进程原序+磁盘补尾）。

    两个进程各自持有同一成员会话池的副本（组长派发 + drain worker 并发时可能）：整体
    覆盖会让后写者抹掉先写者的新会话，按键合并至少不丢条目。
    """
    merged = list(ours)
    known = {str(s.get("sid", "")) for s in merged}
    merged.extend(s for s in disk if str(s.get("sid", "")) not in known)
    return merged


def _warn_missing_runtime(config_data: dict[str, Any], runtime_file: Path) -> None:
    """(a) 轻哨兵（backlog 20261010-worktree-merge-cp ④）：members 非空 + 无内嵌 sessions
    + 运行态文件缺席 → stderr 打印显著警告，不反推数据（避免掩盖跨合并 cp 失败的真因）。

    症状来源：`--no-ff` 只带 git 面，sessions-runtime.json 是 gitignore 的运行态、
    须由 `pysci-dev worktree merge <id>` 或手工 cp 搬到 main——没搬就是本次警告。
    首次安装（members 空）与已完成迁移（运行态在盘）均静默。
    """
    members = config_data.get("members", {})
    if not members:
        return
    if runtime_file.exists():
        return
    print(
        f"[!] registry 加载：配置面有 {len(members)} 位成员，但运行态文件缺席"
        f"（{runtime_file}）。跨合并的 sessions-runtime.json 不会随 --no-ff 迁移——"
        f"若这是刚合并过 registry-runtime-split 的 main，请跑 `pysci-dev worktree "
        f"merge <id>` 或手工 cp 从 worktree 把运行态搬来；否则 orch status 与"
        f" dispatch 会看到 0 会话。首次安装可忽略本警告。",
        file=sys.stderr,
    )


@dataclass
class Registry:
    """registry.json（配置面）+ sessions-runtime.json（运行态）的内存表示与持久化。"""

    data: dict[str, Any]
    runtime: dict[str, Any] = field(default_factory=_empty_runtime)
    path: Path = REGISTRY_PATH
    #: 本进程**声明**要写回配置面的顶层改动（models/checks…）；:meth:`save` 按键叠加到
    #: 磁盘当前内容之上。未声明的键一律以磁盘为准（磁盘权威，lost-update 修复）。
    cfg_overlay: dict[str, Any] = field(default_factory=dict)
    #: 本进程声明「新增」的成员条目；:meth:`save` 仅当磁盘也无该成员时才写入。
    cfg_add_members: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: 本进程拥有会话池的成员 id（:meth:`save` 按 sid 键合并进运行态这些成员的池）。
    rt_members: set[str] = field(default_factory=set)

    @property
    def runtime_path(self) -> Path:
        """运行态文件路径（同目录，随 ``self.path`` 移动——测试 patch REGISTRY_PATH 即两侧隔离）。"""
        return self.path.parent / RUNTIME_NAME

    @property
    def backup_path(self) -> Path:
        """一次性迁移备份路径。"""
        return self.path.parent / BACKUP_NAME

    @classmethod
    def load(cls) -> Registry:
        """读取配置面与运行态；首次遇到内嵌 sessions 的旧配置时自动迁移 + 备份。

        路径显式取模块级 ``REGISTRY_PATH``（非 dataclass 默认值——后者在类定义时已绑定
        真实路径，无视测试的 monkeypatch）；``runtime_path``/``backup_path`` 随 ``path``
        的父目录移动，故 patch 一处即两面隔离。
        """
        if REGISTRY_PATH.exists():
            data = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        else:
            data = _blank_config()
        runtime_file = REGISTRY_PATH.parent / RUNTIME_NAME
        if runtime_file.exists():
            runtime = json.loads(runtime_file.read_text(encoding="utf-8"))
        else:
            runtime = _empty_runtime()
        reg = cls(data=data, runtime=runtime, path=REGISTRY_PATH)
        reg._cutover_if_needed(runtime_file)
        return reg

    def _cutover_if_needed(self, runtime_file: Path) -> None:
        """一次性迁移：配置内嵌 sessions + 运行态文件缺席 → 备份 + 抬高 + 剥离 + 双写。

        只在「运行态文件尚不存在」且「配置里任一成员带 sessions 键」时触发；迁移后配置
        不再内嵌 sessions、运行态文件已在盘，故第二次 load 天然跳过（幂等）。备份文件
        已存在则不覆盖——保留真·pre-split 原文快照。
        """
        if runtime_file.exists():
            return
        members = self.data.get("members", {})
        embedded = {
            mid: list(raw.get("sessions", []))
            for mid, raw in members.items()
            if isinstance(raw, dict) and "sessions" in raw
        }
        if not embedded:
            _warn_missing_runtime(self.data, runtime_file)
            return
        if not self.backup_path.exists():
            _atomic_write(self.backup_path, self.data)
        for mid, sessions in embedded.items():
            _runtime_member(self.runtime, mid)["sessions"] = sessions
            members[mid].pop("sessions", None)
        # 全量写是这里的特例：数据就是本次 load 刚读到的磁盘内容（同一毫秒级窗口），
        # 且一次性触发后即永不再走（幂等）。常规持久化一律走 save() 的按键合并。
        _atomic_write(self.path, self.data)
        _atomic_write(runtime_file, self.runtime)

    def declare_config(self, **overlay: Any) -> None:
        """声明本进程对配置面的改动（同时写进内存副本）——机器写配置面的唯一通路。

        Args:
            **overlay: 键 = 顶层配置键（``models``/``checks``…），值 = 该键的**改动子集**。
                dict 值按键合并，故 :meth:`save` 只把这几个键叠到磁盘当前内容之上，
                同键下他人改过的兄弟项不受牵连。
        """
        for key, value in overlay.items():
            declared = self.cfg_overlay.get(key)
            if isinstance(value, dict) and isinstance(declared, dict):
                declared.update(value)
            else:
                self.cfg_overlay[key] = value
            memory = self.data.get(key)
            if isinstance(value, dict) and isinstance(memory, dict):
                memory.update(value)
            else:
                self.data[key] = value

    def save(self) -> None:
        """读-改-写落盘：配置面**磁盘权威**，只叠加本进程声明过的改动。

        落盘前重读磁盘，绝不把内存副本整体写回。长驻进程（drain worker）的内存是启动
        时的快照，整体回写会抹掉运行期间合并进来的配置裁决——backlog 20261011-220621-leader
        首战实锤：21:05 起 worker，21:42 合并的 deputy/reviewer 降档在收尾时被回退成 max。

        - 配置面：无声明 → 一个字节不写；有声明 → 磁盘为底，:attr:`cfg_overlay` 顶层键
          覆盖（dict 值按键合并），:attr:`cfg_add_members` 仅补磁盘缺席的成员。
        - 运行态：只动 :attr:`rt_members` 声明过的成员，会话池按 sid 合并；其余成员、
          其余字段原样保留。
        """
        if self.cfg_overlay or self.cfg_add_members:
            base = _read_for_merge(self.path, self.data)
            members = base.setdefault("members", {})
            for key, value in self.cfg_overlay.items():
                if isinstance(value, dict) and isinstance(base.get(key), dict):
                    base[key].update(value)
                else:
                    base[key] = value
            for mid, entry in self.cfg_add_members.items():
                members.setdefault(mid, copy.deepcopy(entry))
            _atomic_write(self.path, base)
        if not self.rt_members:
            return
        rt = _read_for_merge(self.runtime_path, _empty_runtime())
        rt_members = rt.setdefault("members", {})
        for mid in self.rt_members:
            target = rt_members.setdefault(mid, {})
            target["sessions"] = _merge_sessions(
                list(target.get("sessions") or []),
                list(
                    self.runtime.get("members", {}).get(mid, {}).get("sessions") or []
                ),
            )
        _atomic_write(self.runtime_path, rt)

    @property
    def models(self) -> dict[str, str]:
        """双档 → 具体型号映射。"""
        return {**DEFAULT_MODELS, **self.data.get("models", {})}

    @property
    def checks(self) -> dict[str, dict[str, Any]]:
        """check 类型 → {cmd: [...含 {path} 占位], cwd: project|pod}。"""
        return self.data.get("checks", {})

    @property
    def tier_a(self) -> dict[str, int]:
        """Tier A 滑窗与阈值（见 :data:`DEFAULT_TIER_A`，REFACTOR §2.4）。

        逐键 fail-open：缺失/非数/非正 → 该键回落默认。统计与自动立案都靠这三个数
        决定「何时算反复出现」，读不到配置不该让它们停摆——宁可按出厂阈值跑。

        Returns:
            ``{"window_days": int, "window_hops": int, "threshold": int}``。
        """
        out = dict(DEFAULT_TIER_A)
        raw = self.data.get("tier_a")
        if not isinstance(raw, dict):
            return out
        for key, default in DEFAULT_TIER_A.items():
            try:
                value = int(raw.get(key, default))
            except (TypeError, ValueError):
                continue
            if value > 0:
                out[key] = value
        return out

    @property
    def jury(self) -> dict[str, Any]:
        """Tier C 仲裁团配置（见 :data:`DEFAULT_JURY`，REFACTOR §4）。

        逐键 fail-open：类型不符 → 该键回落默认（宁按出厂值跑，不让配置笔误停摆）。
        ``panel`` 只保证「字符串列表」这一形状，**不在此处过滤 Qwen**——硬规则的违约
        必须被看见（``orch jury`` 拒开庭并点名型号），静默剔除会让用户以为评审团里
        真有那个模型。

        Returns:
            ``{"panel": [str…], "base_url": str, "call_timeout_s": int,
            "deadline_s": int, "pricing": {model: {"input": x, "output": y}}}``。
        """
        out: dict[str, Any] = {
            "panel": list(DEFAULT_JURY["panel"]),
            "base_url": str(DEFAULT_JURY["base_url"]),
            "call_timeout_s": int(DEFAULT_JURY["call_timeout_s"]),
            "deadline_s": int(DEFAULT_JURY["deadline_s"]),
            "pricing": dict(DEFAULT_JURY["pricing"]),
        }
        raw = self.data.get("jury")
        if not isinstance(raw, dict):
            return out
        if isinstance(raw.get("panel"), list):
            out["panel"] = [str(m).strip() for m in raw["panel"] if str(m).strip()]
        if isinstance(raw.get("base_url"), str) and raw["base_url"].strip():
            out["base_url"] = raw["base_url"].strip()
        for key in ("call_timeout_s", "deadline_s"):
            try:
                value = int(raw.get(key, out[key]))
            except (TypeError, ValueError):
                continue
            if value > 0:
                out[key] = value
        if isinstance(raw.get("pricing"), dict):
            out["pricing"] = raw["pricing"]
        return out

    @property
    def est_tokens_per_char(self) -> float:
        """est_tokens 换算系数（见 :data:`DEFAULT_TOKENS_PER_CHAR`，registry 可配）。

        fail-open：缺失/非数/非正 → 回落默认。系数只让估算失准，不该拖垮派发或
        台账写入（写进台账的是**已换算值**，故还须把字符增量原值一并记下，见
        LedgerEntry.delta_chars——改系数后可重算历史而不必重跑派发）。
        """
        try:
            value = float(self.data.get("est_tokens_per_char"))
        except (TypeError, ValueError):
            return DEFAULT_TOKENS_PER_CHAR
        return value if value > 0 else DEFAULT_TOKENS_PER_CHAR

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
        rt = self.runtime.get("members", {}).get(member_id, {})
        return Member(
            member_id=member_id,
            pod=pod,
            model_tier=raw.get("model_tier", "flash"),
            max_turns=int(raw.get("max_turns", 60)),
            timeout_s=int(raw.get("timeout_s", 7200)),
            mcp_config=raw.get("mcp_config", ".qoder/mcp.json"),
            effort=str(raw.get("effort", "") or ""),
            sessions=list(rt.get("sessions", [])),
            raw=raw,
        )

    def write_sessions(self, m: Member) -> None:
        """把 Member 的会话池写进运行态并声明该成员的会话池所有权（**不碰配置面**）。

        派发一跳要记账的是心跳（hops/last_active/chars_offset/status），配置面
        （model_tier/pod 路径/参数模板/effort/tier_note）是人手与代码评审的领地。旧名
        ``write_member`` 顺手把内存 Member 的配置字段整体抄回 registry.json——长驻 worker
        因此会在收尾时用自己的启动快照覆盖掉期间合并进来的降档裁决（backlog
        20261011-220621-leader）。改此只写运行态，配置面回写通路就此消失。
        """
        _runtime_member(self.runtime, m.member_id)["sessions"] = m.sessions
        self.rt_members.add(m.member_id)

    def ensure_member_defaults(self, member_id: str) -> None:
        """为尚未注册的成员生成默认条目（pod 目录存在时才可用，dispatch 前调用）。

        新成员条目按「新增」声明（:attr:`cfg_add_members`）：save 时仅当磁盘也无该成员
        才落，故不会用默认值盖掉同窗口内他人注册的真条目。
        """
        members = self.data.setdefault("members", {})
        if member_id in members:
            return
        pod = PODS_ROOT / member_id
        entry = {
            "pod": str(pod.relative_to(PROJECT_ROOT)),
            "model_tier": "flash",
            "max_turns": 60,
            "timeout_s": 7200,
            "mcp_config": ".qoder/mcp.json",
        }
        members[member_id] = entry
        _runtime_member(self.runtime, member_id)
        self.cfg_add_members[member_id] = entry
        self.rt_members.add(member_id)


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
    models = reg.data.get("models", {})
    changed: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for tier, name in patterns.items():
        if name not in catalog:
            missing.append(f"{tier}←{name}")
            continue
        new = catalog[name]
        old = models.get(tier)
        if old != new:
            changed[tier] = {"old": old, "new": new}
    if changed:
        # 只声明变更的档位：save 时这些键叠加到磁盘当前 models 之上，同期他人改过的
        # 兄弟档（如手工配的 flash_fallback）不被本进程的旧值盖掉
        reg.declare_config(models={t: d["new"] for t, d in changed.items()})
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
    "BACKUP_NAME",
    "DEFAULT_JURY",
    "DEFAULT_MODELS",
    "DEFAULT_TOKENS_PER_CHAR",
    "Member",
    "ORCHESTRATION_ROOT",
    "REGISTRY_PATH",
    "RUNTIME_NAME",
    "Registry",
    "orchestration_rel",
    "pod_session_key",
    "pod_sessions_dir",
    "resolve_exe",
]
