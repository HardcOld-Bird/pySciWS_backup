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

运行态文件里除会话池外还有 **``models_resolved`` 档位面**（:func:`refresh_models` 的
落点）、**``model_seat_fail`` 席位故障粘滞窗**与 **``model_seat_names`` 席→名反查表**：
模型映射自愈是机器行为，写进 git 跟踪的 ``models`` 会把跟踪配置周期改脏（每
drain 必现 `` M registry.json``，堵死一切碰 registry 的单，backlog 20261012-080410-devops），
而 ``models`` 按 charter 属护栏级——机器静默改写它本身就越权。故读取按
「跟踪值 → 运行态解析值 → 出厂默认」三级回退（:attr:`Registry.models` /
:meth:`Registry.model_chain`），跟踪面只留人写的覆盖。档位值自 2026-10-12 21:1x 起是
**有序回退链**（一档多席，用户裁决 backlog 20261012-211500-leader），旧单值写兼容读。

**配置面磁盘权威**（backlog 20261011-220621-leader，lost-update 修复）：分储只搬走了心跳，
没搬走「整体回写内存副本」这个坏习惯——长驻 drain worker 收尾时仍把启动时加载的
registry.json 快照原样写回，后写覆盖，抹掉了运行期间合并进来的配置改动。故
:meth:`Registry.save` 改为**读-改-写按键合并**：落盘前重读磁盘，配置面只叠加本进程
*显式声明*过的改动（:meth:`Registry.declare_config` 改声明过的顶层键、
:meth:`Registry.ensure_member_defaults` 新增成员），其余键一律以磁盘为准；无任何声明时
配置文件**一个字节都不写**。会话池经 :meth:`Registry.write_sessions` 按成员、按 sid
合并进运行态。机器写配置面因此只剩「声明」这一条通路——而档位映射自愈**不走它**
（见上文 ``models_resolved``）。
"""

from __future__ import annotations

import copy
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pysci.paths import ORCH_STATE_ROOT, ORCHESTRATION_ROOT, PODS_ROOT, PROJECT_ROOT

from . import presets, routing

REGISTRY_PATH: Path = ORCH_STATE_ROOT / "registry.json"

#: 运行态会话池文件名（落在 registry.json 同目录；gitignore，见模块 docstring）。
RUNTIME_NAME: str = "sessions-runtime.json"

#: 一次性迁移的原文备份文件名（首次 cutover 时创建，已存在则不覆盖——保留真·pre-split 快照）。
BACKUP_NAME: str = "registry.presplit.json"

#: 运行态文件里的档位映射键（:func:`refresh_models` 的写点；gitignored）。
RUNTIME_MODELS_KEY: str = "models_resolved"

#: 运行态文件里的**席位故障粘滞窗**键（``{档: {席: {"ts":…, "kind":…, "member":…}}}``）。
#: 用户裁决 2026-10-12 21:3x：BYOK Qwen 渠道暂不充值、将持续不可用 → 探测失败的席在短窗内
#: 直接跳过（省探测跳数），窗口过期自动恢复探测（充值/修好后自愈，无需改代码）。
#: 落运行态而非跟踪面同一纪律：心跳每跳必变的机器事实不往 git 跟踪配置里写（免堵合并）。
RUNTIME_SEAT_FAIL_KEY: str = "model_seat_fail"

#: 运行态文件里的 **席 → 人类名** 反查表（:func:`refresh_models` 顺带落点；mirror/status
#: 告警行要给人看名字而不是 UUID）。
RUNTIME_SEAT_NAMES_KEY: str = "model_seat_names"

#: 回退策略默认参数（``registry.json`` 顶层 ``model_fallback`` 块可逐键覆盖）。
#: ``skip_window_s``：某席探测失败后跳过重试的时长（0=禁用粘滞跳过，每跳都重新探测）。
#: ``event_tail``：mirror/status 面「最近 N 条」的 N。
DEFAULT_MODEL_FALLBACK: dict[str, int] = {"skip_window_s": 3600, "event_tail": 5}

#: 脏跟踪面哨兵扫描的相对路径（见 :func:`dirty_tracked_state`）。
STATE_REL_PATH: str = "orchestration/state"

#: 角色 → 出厂首席型号（平读面 :attr:`Registry.models` 的保守底）。main/pro 的真实席位链
#: **随预设**（:mod:`presets` 与跟踪 registry 的 ``presets`` 块），这里两个空串只在
#: 「预设不可激活」时兜底——空 = 不传 ``-m``，宁窄不猜（未解析的人名当默认会让 exe 启动即失败）。
#: ``minimum`` = harness 压测专用最低成本档（README §3.5），**独立链、不随预设走**。
DEFAULT_MODELS: dict[str, str] = {
    "main": "",
    "pro": "",
    "minimum": "Qwen3.8-Flash",
}

#: 档 → **有序人类名模板**：档位模式退役（用户裁决 2026-10-12 22:4x，backlog
#: 20261012-225500-leader）后本表**只承载 minimum**——它的型号选择属护栏级且与预设解耦
#: （E3-live 实测火山方舟不可行，维持「Mimo-Flash → 内置」）。``main``/``pro`` 的模板随
#: 预设（``presets.PRESETS``），:func:`refresh_models` 不再代它解析。
DEFAULT_MODEL_CHAINS: dict[str, tuple[str, ...]] = {
    "minimum": ("Mimo-V2.6-Flash", "Qwen3.8-Flash"),
}

#: 角色键（与 ``presets.ROLE_CHOICES`` 同源：main=预设主力席、pro=预设 pro 席、
#: minimum=独立压测档）。
TIER_KEYS: tuple[str, ...] = ("main", "pro", "minimum")

#: 旧档名 → 角色（跟踪面 / 运行态 / plan YAML ``model:`` / 老案卷的兼容读：flash 当年是
#: 主力、max 是升档席）。读路统一过这张表，写路不再产生旧键。
LEGACY_TIER_ALIASES: dict[str, str] = {"flash": "main", "max": "pro"}


def normalize_seats(value: Any) -> list[str]:
    """单值 / 链 / None → 席位列表（旧写 ``"a"`` 与新写 ``["a","b"]`` 同一条读路）。

    ``""`` 保留（＝不传 ``-m`` 的合法席位裁决）；``None`` 丢弃（JSON null 容错）。
    非列表标量按单席处理；其它类型（数字等）串化后当一席——读侧宽容，写侧只有链。
    **嵌套序列摊平**（backlog 20261013-003000-leader）：配置形状迁移期会出现「链中链」
    （如 ``[["a","b"]]``），旧口径把它 ``str()`` 成 ``"['a', 'b']"`` 垃圾席位——一路
    带到 ``-m`` 上「启动即失败」白烧一跳；摊平后它仍是一条合法链。dict 无席位语义，
    丢弃（宁可退化成不传 ``-m`` 走用户级默认，也不把容器串塞进 argv）。
    """
    if value is None:
        return []
    if isinstance(value, dict):
        return []
    if isinstance(value, (list, tuple)):
        out: list[str] = []
        for item in value:
            if item is None or isinstance(item, dict):
                continue
            if isinstance(item, (list, tuple)):
                out.extend(normalize_seats(item))
                continue
            out.append(str(item))
        return out
    return [str(value)]


def dedupe_seats(seats: list[str]) -> list[str]:
    """保序去重（同一席在链里出现两次只会白烧一跳）。"""
    seen: set[str] = set()
    out: list[str] = []
    for seat in seats:
        if seat not in seen:
            seen.add(seat)
            out.append(seat)
    return out


def model_tier_rules(
    chains: dict[str, Any] | None = None, extra: dict[str, Any] | None = None
) -> dict[str, set[str]]:
    """反读「名 → 属于哪些档」的规则表（防自愈把某个名塞进错误的档）。

    Args:
        chains: 档 → 名链；缺省用 :data:`DEFAULT_MODEL_CHAINS`。
        extra: 追加的 名→档 规则（``registry.model_patterns`` 的历史倒表形态）。

    Returns:
        ``{人类名: {档, …}}``；一名的规则互相合并，供 :func:`refresh_models` 比对。
    """
    rules: dict[str, set[str]] = {}
    for tier, names in (chains or DEFAULT_MODEL_CHAINS).items():
        for name in normalize_seats(names):
            if name:
                rules.setdefault(name, set()).add(str(tier))
    for name, tier in (extra or {}).items():
        if name and tier:
            rules.setdefault(str(name), set()).add(str(tier))
    return rules


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

#: Tier A 阈值自动立案参数（README §5.4，用户裁决 v0.5）：同一 ``(标签, 成员)`` 在
#: **滑窗内**（``window_days`` 天 或 ``window_hops`` 跳，取并集）失败标签出现
#: ``threshold`` 次 → orch 在 status/approve 入口自动登记基建提案。设计要点：
#: 计数、立案全代码完成，人只在强制选择点出现——「组长有空去看」不是机制。
#: registry.json 顶层 ``tier_a`` 块可逐键覆盖（非法值回落默认，见 :meth:`Registry.tier_a`）。
DEFAULT_TIER_A: dict[str, int] = {
    "window_days": 14,
    "window_hops": 20,
    "threshold": 3,
}


#: Tier C 仲裁团配置（README §5.3；E3-prep 单渠道，E3-live 改**逐席 provider**）。
#: **seats 默认空**：选型权归用户（各提供商的账号与付费面是用户的），代码只负责一条硬
#: 规则——面板不得含**被评模型同族**型号（排除同族自恋偏差；「同族」随预设/谷时隐性态
#: 相对化，见 :func:`.routing.producer_family`），不做模型能力判断也不内置候选清单。
#: 空 seats 的语义是「尚未选型」，``orch jury`` 据此降级入队而不是报错。
#: registry.json 顶层 ``jury`` 块逐键覆盖（非法值回落默认，见 :meth:`Registry.jury`）：
#:
#: .. code-block:: json
#:
#:    {"jury": {"seats": [{"name": "deepseek", "key_env": "DEEPSEEK_API_KEY",
#:                         "base_url": "https://api.deepseek.com/v1",
#:                         "model": "deepseek-v4-pro"}],
#:              "fallback": {"enabled": true, "name": "siliconflow",
#:                           "key_env": "SILICONFLOW_API_KEY",
#:                           "base_url": "https://api.siliconflow.cn/v1",
#:                           "model": "deepseek-ai/DeepSeek-V3"},
#:              "pricing": {"deepseek-v4-pro": {"input": 2.0, "output": 8.0}}}}
#:
#: 每席四键 ``name``/``key_env``/``base_url``/``model`` 缺一不可：缺了不剔除该席，而是标
#: ``defect`` 让它以 unavailable 身份被看见（静默剔除会让用户以为评审团里真有那席）。
#: **一席一条端点、一律按量付费**（用户终裁 2026-10-13，路由 v3 返工）：TokenPlan 套餐
#: 路径整个不做——两家个人版条款都禁止 API 自动化调用（违规封 Key），而仲裁团正是自动化
#: 调用。故席位没有端点链、没有 ``billing``/``endpoints`` 键，代码里也不存下探回退。
#: 可选 ``params`` 是逐席请求参数覆盖——实测三席传输面并不齐（``kimi-k2.6`` 只接受
#: ``temperature=1``，共享的 ``temperature: 0`` 请求会 400），故参数必须能按席覆盖。
#: ``fallback`` 是带独立开关的**回退席**，仅当 ≥1 主席不可达时才上场，上场必告警（§4）。
#: ``pricing`` 单位 = 元/百万 token（各提供商刊例按型号而定）；缺席 → 记录里
#: ``cost.amount`` 为 null 并附 note，**不猜价**——成本数字要能对外负责。
#: ``max_output_tokens`` 是请求的 ``max_tokens``：三席皆推理型，正文之前先烧一大段
#: reasoning，且这段的长度是**重尾**的——真案卷（prompt ≈4 k）上实测 432～1500 浮动，
#: 4000 仍有一席烧到 3986 空手而归。1500/4000 都会**随机**把正文截成空；出厂给 8000。
#: 截断不再计为无效票（:func:`.jury._ballot` 报错误并降级入队，点名是哪一席）。
DEFAULT_JURY: dict[str, Any] = {
    "seats": [],
    "fallback": {"enabled": False},
    "call_timeout_s": 120,
    "deadline_s": 300,
    "max_output_tokens": 8000,
    "pricing": {},
}

#: 席位必填键（``name`` 可缺，缺则回落型号名）。
SEAT_REQUIRED_KEYS: tuple[str, ...] = ("base_url", "model", "key_env")


def normalize_seat(raw: Any) -> dict[str, Any] | None:
    """配置里的一个席位 → 规范化 dict；非 dict 直接作废（返回 None）。

    缺必填键**不剔除**该席，而是落 ``defect`` 说明缺什么：调用侧据此标 unavailable，
    读数面（``orch status``／mirror）点名——违约要被看见，这是 §4 非 Qwen 硬规则的同一条课。
    """
    if not isinstance(raw, dict):
        return None
    out: dict[str, Any] = {
        "base_url": "",
        "model": "",
        "key_env": "",
        "name": "",
        "params": {},
    }
    for key in ("base_url", "model", "key_env", "name"):
        if isinstance(raw.get(key), str):
            out[key] = raw[key].strip()
    if isinstance(raw.get("params"), dict):
        out["params"] = dict(raw["params"])
    if not out["name"]:
        out["name"] = out["model"] or "seat"
    missing = [k for k in SEAT_REQUIRED_KEYS if not out[k]]
    out["defect"] = f"缺 {','.join(missing)}" if missing else ""
    return out


#: twin 判断通道（``orch taste``）的生效开关默认值。**False 是设计而不是缺省占位**：
#: PREFERENCE.md v1 是草案，生效确认权归用户（README §1 代理裁决「上线留用户」），故代码
#: 不替用户按下这个开关；用户确认草案后把 registry.json 顶层 ``taste_enabled`` 置 true。
#: 值非布尔（笔误/字符串）→ 回落 False：这条通道宁可不开，不可误开。
DEFAULT_TASTE_ENABLED = False

#: inspector 闲时巡检引擎参数（README §6，E5a）。设计纪律同 :data:`DEFAULT_TIER_A`：
#: 节流与预算是**成本控制机制**，读不到配置就按出厂值跑（fail-open），不许因笔误停摆。
#: registry.json 顶层 ``inspector`` 块逐键覆盖。
#:
#: - ``max_hops`` / ``wallclock_s`` —— 单次激活的硬预算：到顶自停（§7「预算硬上限」）。
#: - ``cooldown_s`` —— 两次激活的最小间隔（读最近一次 ``.done``）。没有这一条，组长每敲
#:   一次 ``orch status`` 都会在内嵌拉起里重新激活一轮，成本按敲击次数放大。
#: - ``yield_window_s`` —— 「plan run 活跃」的判定窗：计划 ``.state.json`` 的 mtime 距今
#:   小于此值即认定生产侧在派跳，本轮只跑只读巡检（§7 让路）。
#: - ``pause_*`` / ``resume_*`` —— 双队列迟滞闸（§5.2 用户裁决 v0.5）：backlog pending 或
#:   bencher 队列长度**超过** pause 值 → 暂停产出；两者**回落到 resume 值及以下** → 恢复。
#:   恢复阈值必须严格低于暂停阈值（迟滞防振荡），否则这四键整体回落出厂值并在
#:   :meth:`Registry.inspector` 的 ``hysteresis_valid`` 里报假 —— 把闸配成 ``pause<=resume``
#:   等于配了一个每轮都翻转的开关，那不是节流。
DEFAULT_INSPECTOR: dict[str, Any] = {
    "max_hops": 3,
    "wallclock_s": 1800,
    "cooldown_s": 3600,
    "yield_window_s": 900,
    "pause_backlog": 12,
    "pause_bench": 6,
    "resume_backlog": 8,
    "resume_bench": 4,
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
    """一个组员（pod + 会话池 + 参数模板）的注册表条目。

    模型席位**不在成员身上**（档位模式已退役，用户裁决 2026-10-12 22:4x）：本跳用哪个席
    由「预设的席位表 + 成员是否在 pro 名单」解析，见 :func:`presets.resolve_role`。
    """

    member_id: str
    pod: Path
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


def note_hop_pid(member_id: str, sid: str, *, pid: int, task: str = "") -> None:
    """把**在途跳**的会话进程 pid 盖进运行态会话池（孤儿回收的第四面判据）。

    为什么必须在派生瞬间写盘、而不是跳结束时记进台账：worker 若在跳中途被杀，台账那行
    根本不会存在，可它派生的会话进程还活着并继续写 worktree（backlog
    20261012-184000-leader 的并发双写根因）。反之跳正常收尾时，派发侧的常规
    :meth:`Registry.write_sessions` 会用「本进程条目为准」的合并语义把它覆盖掉
    （见 :func:`_merge_sessions`）——所以 **pid 键的存活期恰好等于这一跳的在途期**，
    回收者看到的就是一个天然的「有谁还在写」标记，无需再加状态机。``pid=0`` 是显式收工
    写法（回收者处置完一个孤儿后用它抹掉标记，免得下次对着已复用的 pid 误动手）。

    ``task`` 记任务书文件名（含条目 id），供 drain 把孤儿条目对上会话。

    **fail-open**：运行态写不动（磁盘/权限/并发损坏）只静默放弃——记账面不得拖垮真实
    派发；代价是该跳失去第四面保护，回收会退化为异名隔离重建（仍安全）。
    """
    try:
        reg = Registry.load()
        sessions = _runtime_member(reg.runtime, member_id).setdefault("sessions", [])
        for s in sessions:
            if s.get("sid") == sid:
                s["pid"] = int(pid)
                s["pid_task"] = task
        reg.rt_members.add(member_id)
        reg.save()
    except Exception:  # noqa: BLE001 - 记账旁路，任何异常都不该影响派发
        return


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
    def runtime_model_chains(self) -> dict[str, list[str]]:
        """运行态档位链（``models_resolved``，gitignored）——机器自愈的唯一落点。

        面缺失/被手工写坏成非字典 → 空 dict（读侧自然回退到出厂默认，不抛）。值兼容
        两种历史形态：单值字符串（旧写）与席位列表（链）。
        """
        raw = self.runtime.get(RUNTIME_MODELS_KEY)
        if not isinstance(raw, dict):
            return {}
        return {str(k): normalize_seats(v) for k, v in raw.items()}

    @property
    def runtime_models(self) -> dict[str, str]:
        """运行态档位的**首席平读面**（向后兼容读法：消费方仍拿到一个字符串）。"""
        return {
            tier: (seats or [""])[0]
            for tier, seats in self.runtime_model_chains.items()
        }

    def _merged_model_faces(self) -> dict[str, list[str]]:
        """逐键取「跟踪 > 运行态 > 出厂」中**最高优先且存在该键**的那一面的链。

        合并粒度是**整条链**（不是逐席）：跟踪面写了 ``minimum`` 就是人对这一档的完整裁决，
        运行态解析值不得往中间插席——那正是 #74「机器静默顶掉裁决」的链形态版本。
        ``main``/``pro`` 的真值**不走这条合并**（随预设，见 :meth:`Registry.preset_chain`），本函数
        只在预设缺席/不可激活时给它们出厂空位兜底。
        """
        faces = (
            {
                str(k): normalize_seats(v)
                for k, v in (self.data.get("models") or {}).items()
            },
            self.runtime_model_chains,
            {k: normalize_seats(v) for k, v in DEFAULT_MODELS.items()},
        )
        merged: dict[str, list[str]] = {}
        for face in faces:
            for tier, seats in face.items():
                merged.setdefault(tier, seats)
        return merged

    def preset_chain(self, role: str, *, preset: str | None = None) -> list[str]:
        """``main``/``pro`` 的席位链（跟随当前预设；谷时隐性态跟随隐性预设）。

        三级：跟踪 ``presets.<预设>.<role>`` **整链覆盖**（账号级 BYOK UUID 落这里，选型权
        归用户）← 出厂预设表 ``presets.PRESETS``（人类名）。跟踪面**没有** ``presets`` 块的
        旧部署读 ``models.flash``/``models.max`` 兼容（旧档=当年的主力/升档席，本仓库迁移
        后走新块，此路径只服务回退/异机）。

        **谷时隐性 DeepSeek**（路由 v3 规格 3，backlog 20261013-093000-leader）：判定生效时
        **只有 ``main``** 改读隐性预设的席表——``pro`` 保持配置预设的升档席不变（用户终裁
        2026-10-13：「pro 席成员保持原 pro 席不变，提高工作质量」，原「全员同档无 pro 席」
        作废）。时间只覆盖**选哪张席表**，不绕过跟踪整链覆盖纪律：跟踪面写了
        ``presets.deepseek.main`` 就以覆盖为准，机器不往链里插席。

        ``preset`` 显式给定时读**那一个**预设，且**不再**做时间覆盖（隐性链本身就走这条
        路，再覆盖一次等于自指）。
        """
        if role == "main" and preset is None:
            valley = routing.hidden_valley_preset()
            if valley:
                return self.preset_chain("main", preset=valley)
        active = presets.active()
        name = preset or active.name
        tracked_presets = self.data.get("presets")
        if isinstance(tracked_presets, dict):
            table = tracked_presets.get(name)
            if isinstance(table, dict) and role in table:
                return dedupe_seats(normalize_seats(table[role]))
            if preset:
                return dedupe_seats(presets.factory_chain(name, role))
        elif not preset:
            key = "flash" if role == "main" else "max"
            models_face = self.data.get("models") or {}
            legacy = models_face.get(key)
            if legacy is not None:
                seats = normalize_seats(legacy)
                for seat in normalize_seats(models_face.get(f"{key}_fallback")):
                    if seat and seat not in seats:
                        seats.append(seat)
                return dedupe_seats(seats)
        return dedupe_seats(
            presets.factory_chain(name, role) if preset else active.chain(role)
        )

    @property
    def models(self) -> dict[str, str]:
        """角色 → 具体型号的**平读面**（首席）。

        ``main``/``pro`` 首席随当前预设（:meth:`Registry.preset_chain`），``minimum`` 与旧档键走
        :meth:`_merged_model_faces` 的三级回退。保留这个视图是为了不惊动既有消费方
        （``eval``/``orch`` 蒸馏跳/大量断言），它们要的就是「这一角色现在用哪个渠道」。
        需要回退链的代码请走 :meth:`model_chain`。
        """
        merged = self._merged_model_faces()
        out = {tier: (seats or [""])[0] for tier, seats in merged.items()}
        # 角色是读数面的正名：旧档键（flash/max）归一进 main/pro 后从平读面摘掉，
        # 免得消费方同时看到两把钥匙、哪把是真值全靠猜。
        for role, legacy in (("main", "flash"), ("pro", "max")):
            chain = self.preset_chain(role)
            out[role] = chain[0] if chain else ""
            out.pop(legacy, None)
        return out

    def model_chain(self, tier: str) -> list[str]:
        """该角色的有序回退链（首席在前）；``tier`` 接受旧档名（flash/max 别名到 main/pro）。

        ``main``/``pro`` 跟随预设（:meth:`Registry.preset_chain`，谷时隐性态跟随隐性预设）；
        ``minimum`` 及其余键走三级回退且旧 ``<tier>_fallback`` 键**并入链尾**（兼容读）。
        至少返回一席：链为空时回落 ``[""]``＝不传 ``-m``（走用户级默认渠道）。
        """
        role = LEGACY_TIER_ALIASES.get(tier, tier)
        if role in ("main", "pro"):
            return self.preset_chain(role) or [""]
        merged = self._merged_model_faces()
        seats = list(merged.get(tier) or merged.get(role) or [""])
        legacy = merged.get(f"{tier}_fallback") or []
        for seat in legacy:
            if seat and seat not in seats:
                seats.append(seat)
        return dedupe_seats(seats) or [""]

    @property
    def seat_names(self) -> dict[str, str]:
        """席（UUID 或内置名）→ 人类名反查表（``model_seat_names``，告警面可读性用）。"""
        raw = self.runtime.get(RUNTIME_SEAT_NAMES_KEY)
        if not isinstance(raw, dict):
            return {}
        return {str(k): str(v) for k, v in raw.items()}

    def seat_label(self, seat: str) -> str:
        """席位的显示名（反查不到就原样返回，告警面绝不因缺表而空转）。"""
        if not seat:
            return "(用户级默认)"
        return self.seat_names.get(seat, seat)

    @property
    def fallback_policy(self) -> dict[str, int]:
        """回退策略参数（``registry.json`` 顶层 ``model_fallback`` 可覆盖，逐键 fail-open）。

        口径同 :meth:`tier_a`：读不到/非法值 → 回落出厂默认，不让参数把派发面卡死。
        """
        raw = self.data.get("model_fallback") or {}
        out = dict(DEFAULT_MODEL_FALLBACK)
        if not isinstance(raw, dict):
            return out
        for key in DEFAULT_MODEL_FALLBACK:
            try:
                number = int(raw.get(key))
            except (TypeError, ValueError):
                continue
            if number >= 0:
                out[key] = number
        return out

    @property
    def seat_failures(self) -> dict[str, dict[str, dict[str, Any]]]:
        """席位故障粘滞窗（``model_seat_fail``）：``{角色: {席: {"ts":…, "kind":…, "member":…}}}``。

        旧键（``flash``/``max``）读时归一到角色，免得档位退役后在途的窗变成孤儿。
        """
        raw = self.runtime.get(RUNTIME_SEAT_FAIL_KEY)
        if not isinstance(raw, dict):
            return {}
        out: dict[str, dict[str, dict[str, Any]]] = {}
        for tier, seats in raw.items():
            if isinstance(seats, dict):
                key = LEGACY_TIER_ALIASES.get(str(tier), str(tier))
                out.setdefault(key, {}).update(
                    {
                        str(seat): info
                        for seat, info in seats.items()
                        if isinstance(info, dict)
                    }
                )
        return out

    def note_seat_failure(
        self,
        tier: str,
        seat: str,
        *,
        kind: str,
        member: str = "",
        now: float | None = None,
    ) -> None:
        """记一次席位探测失败（粘滞窗起点）；读-改-写与会话池同纪律，只动这一格。"""
        tier = LEGACY_TIER_ALIASES.get(tier, tier)
        rt = _read_for_merge(self.runtime_path, _empty_runtime())
        face = rt.get(RUNTIME_SEAT_FAIL_KEY)
        face = dict(face) if isinstance(face, dict) else {}
        seats = dict(face.get(tier) or {})
        seats[seat] = {
            "ts": float(now if now is not None else time.time()),
            "kind": kind,
            "member": member,
        }
        face[tier] = seats
        rt[RUNTIME_SEAT_FAIL_KEY] = face
        _atomic_write(self.runtime_path, rt)
        self.runtime[RUNTIME_SEAT_FAIL_KEY] = face

    def note_seat_success(self, tier: str, seat: str) -> None:
        """该席跑通即销故障记录（充值/修好后自愈的落点：下一次成功就不再有跳过窗）。"""
        tier = LEGACY_TIER_ALIASES.get(tier, tier)
        rt = _read_for_merge(self.runtime_path, _empty_runtime())
        face = rt.get(RUNTIME_SEAT_FAIL_KEY)
        if not isinstance(face, dict):
            return
        seats = face.get(tier)
        if not isinstance(seats, dict) or seat not in seats:
            return
        rest = {k: v for k, v in seats.items() if k != seat}
        face = dict(face)
        if rest:
            face[tier] = rest
        else:
            face.pop(tier, None)
        rt[RUNTIME_SEAT_FAIL_KEY] = face
        _atomic_write(self.runtime_path, rt)
        self.runtime[RUNTIME_SEAT_FAIL_KEY] = face

    def merge_runtime_models(
        self, resolved: dict[str, Any]
    ) -> dict[str, dict[str, Any]]:
        """把解析出的档位链合并写进**运行态**文件（跟踪配置一个字节不碰）。

        读-改-写与 :meth:`save` 同纪律：落盘前重读磁盘，只更新 ``models_resolved`` 里
        出现的档，其余档与会话池原样保留（会话池每跳都在被别的进程改写）。

        Args:
            resolved: 档 → 席位链（本次 --list-models 解析出的值；单值写也归一成链）。

        Returns:
            ``{tier: {"old": 运行态旧值|None, "new": 新链}}``，只含**确实变了**的档；
            无变化时不写盘并返回 ``{}``（免 churn）。比较按**归一后的链**做，免得
            ``"a"`` 与 ``["a"]`` 被当成两回变化。
        """
        rt = _read_for_merge(self.runtime_path, _empty_runtime())
        face = rt.get(RUNTIME_MODELS_KEY)
        face = dict(face) if isinstance(face, dict) else {}
        normalized = {tier: normalize_seats(value) for tier, value in resolved.items()}
        changed = {
            tier: {"old": face.get(tier), "new": seats}
            for tier, seats in sorted(normalized.items())
            if normalize_seats(face.get(tier)) != seats
        }
        if not changed:
            return {}
        face.update({tier: normalized[tier] for tier in changed})
        rt[RUNTIME_MODELS_KEY] = face
        _atomic_write(self.runtime_path, rt)
        self.runtime[RUNTIME_MODELS_KEY] = face
        return changed

    def merge_runtime_seat_names(self, names: dict[str, str]) -> None:
        """把「席 → 人类名」反查表并进运行态（只增不删，UUID 稳定无须清理）。"""
        if not names:
            return
        rt = _read_for_merge(self.runtime_path, _empty_runtime())
        face = rt.get(RUNTIME_SEAT_NAMES_KEY)
        face = dict(face) if isinstance(face, dict) else {}
        if all(face.get(seat) == name for seat, name in names.items()):
            return
        face.update(names)
        rt[RUNTIME_SEAT_NAMES_KEY] = face
        _atomic_write(self.runtime_path, rt)
        self.runtime[RUNTIME_SEAT_NAMES_KEY] = face

    @property
    def checks(self) -> dict[str, dict[str, Any]]:
        """check 类型 → {cmd: [...含 {path} 占位], cwd: project|pod}。"""
        return self.data.get("checks", {})

    @property
    def tier_a(self) -> dict[str, int]:
        """Tier A 滑窗与阈值（见 :data:`DEFAULT_TIER_A`，README §5.4）。

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
        """Tier C 仲裁团配置（见 :data:`DEFAULT_JURY`，README §5.3，E3-live 逐席 provider）。

        逐键 fail-open：类型不符 → 该键回落默认（宁按出厂值跑，不让配置笔误停摆）。
        ``seats`` 只保证「规范化席位表」这一形状，**不在此处过滤 Qwen**——硬规则的违约
        必须被看见（``orch jury`` 拒开庭并点名型号），静默剔除会让用户以为评审团里
        真有那个模型。席位自身缺必填键同样不剔除，带 ``defect`` 交给读数面点名。

        Returns:
            ``{"seats": [seat…]（主席）, "fallback": {seat…, "enabled": bool},
            "call_timeout_s": int, "deadline_s": int, "max_output_tokens": int,
            "pricing": {model: {"input": x, "output": y}}}``。
        """
        out: dict[str, Any] = {
            "seats": [],
            "fallback": {
                **normalize_seat(DEFAULT_JURY["fallback"]),
                "enabled": bool(DEFAULT_JURY["fallback"].get("enabled")),
            },
            "call_timeout_s": int(DEFAULT_JURY["call_timeout_s"]),
            "deadline_s": int(DEFAULT_JURY["deadline_s"]),
            "max_output_tokens": int(DEFAULT_JURY["max_output_tokens"]),
            "pricing": dict(DEFAULT_JURY["pricing"]),
        }
        raw = self.data.get("jury")
        if not isinstance(raw, dict):
            return out
        if isinstance(raw.get("seats"), list):
            out["seats"] = [
                s for s in (normalize_seat(x) for x in raw["seats"]) if s is not None
            ]
        fb = normalize_seat(raw.get("fallback"))
        if fb is not None:
            fb["enabled"] = raw["fallback"].get("enabled") is True
            out["fallback"] = fb
        for key in ("call_timeout_s", "deadline_s", "max_output_tokens"):
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
    def taste_enabled(self) -> bool:
        """twin 宪章是否生效（``orch taste`` 能否派发，见 :data:`DEFAULT_TASTE_ENABLED`）。

        只认布尔真值：缺失、非布尔、字符串 ``"false"`` 一律判 False——这条通道一旦开启
        就以「用户的声音」对外服务，读错方向必须是关闭而非猜测。
        """
        return self.data.get("taste_enabled") is True

    @property
    def inspector(self) -> dict[str, Any]:
        """inspector 巡检引擎参数（见 :data:`DEFAULT_INSPECTOR`，README §6）。

        逐键 fail-open（非数/非正 → 该键回落默认），但迟滞四键是**成对校验**：
        ``resume_*`` 必须严格小于 ``pause_*``，否则整对回落出厂值——配成 ``pause<=resume``
        的闸会在「暂停/恢复」之间每轮翻转，那不再是节流而是抖动。

        Returns:
            :data:`DEFAULT_INSPECTOR` 的形状，外加 ``hysteresis_valid``（迟滞配置是否成立）。
        """
        out: dict[str, Any] = dict(DEFAULT_INSPECTOR)
        raw = self.data.get("inspector")
        if not isinstance(raw, dict):
            raw = {}
        for key in DEFAULT_INSPECTOR:
            if key in (
                "pause_backlog",
                "pause_bench",
                "resume_backlog",
                "resume_bench",
            ):
                continue
            try:
                value = int(raw.get(key, out[key]))
            except (TypeError, ValueError):
                continue
            if value > 0:
                out[key] = value
        for key in ("pause_backlog", "pause_bench", "resume_backlog", "resume_bench"):
            try:
                value = int(raw.get(key, out[key]))
            except (TypeError, ValueError):
                continue
            if value >= 0:
                out[key] = value
        out["hysteresis_valid"] = (
            out["resume_backlog"] < out["pause_backlog"]
            and out["resume_bench"] < out["pause_bench"]
        )
        if not out["hysteresis_valid"]:
            for key in (
                "pause_backlog",
                "pause_bench",
                "resume_backlog",
                "resume_bench",
            ):
                out[key] = DEFAULT_INSPECTOR[key]
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
        （pod 路径/参数模板/effort/tier_note）是人手与代码评审的领地。旧名
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


def split_model_tables(
    raw_chains: dict[str, Any] | None, raw_patterns: dict[str, Any] | None
) -> tuple[dict[str, list[str]], dict[str, str]]:
    """把配置面的档位模板拆成（档 → 人类名链, 名 → 档规则）。

    ``model_patterns`` 历史上**两种写法并存**（backlog 20261012-211500-leader 收口）：真实
    ``registry.json`` 写的是**名→档**倒表（``{"Qwen3.8-Flash": "flash"}``），文档与单测却按
    **档→名**模板用它（``{"max": "Qwen3.8-Flash"}``）。旧实现把两张表直接 merge 成一张，
    倒表的键被当成档位名 ⇒ 每个 drain 都报一堆 ``Qwen3.8-Flash←flash`` 的假「未命中」。
    分流判据：值是**已知档名** ⇒ 名→档规则；否则 ⇒ 档→名模板（单值或链皆可）。

    Returns:
        ``(templates, rules)``：templates 已含 :data:`DEFAULT_MODEL_CHAINS` 打底，
        rules 是补充的 名→档 断言（供 :func:`refresh_models` 查串档冲突）。
    """
    known = (
        set(TIER_KEYS)
        | set(LEGACY_TIER_ALIASES)
        | {f"{tier}_fallback" for tier in TIER_KEYS}
        | {f"{tier}_fallback" for tier in LEGACY_TIER_ALIASES}
    )
    templates: dict[str, list[str]] = {
        tier: list(names) for tier, names in DEFAULT_MODEL_CHAINS.items()
    }
    for tier, names in (raw_chains or {}).items():
        templates[str(tier)] = normalize_seats(names)
        known.add(str(tier))
    rules: dict[str, str] = {}
    for key, value in (raw_patterns or {}).items():
        name = str(key)
        if not isinstance(value, (list, tuple)) and str(value) in known:
            rules[name] = str(value)  # 名→档（历史倒表）
            continue
        templates[name] = normalize_seats(value)  # 档→名（模板写法）
        known.add(name)
    return templates, rules


def refresh_models(*, timeout_s: int = 60) -> dict[str, Any]:
    """按 --list-models 刷新档位**回退链**的运行态面（跟踪 registry.json 一个字节不写）。

    以链模板（:data:`DEFAULT_MODEL_CHAINS`，``registry.json`` 的 ``model_chains``/
    ``model_patterns`` 可覆盖）**精确匹配**人类名，逐席解析成 UUID/内置名；该档在跟踪面
    **无人写值**时，解析出的整条链合并进 sessions-runtime.json 的 ``models_resolved``
    （gitignored，见 :meth:`Registry.merge_runtime_models`）。顺带把「席→人类名」写进
    ``model_seat_names``，让告警面能报名字而不是裸 UUID。

    为什么不写跟踪面（backlog 20261012-080410-devops 方案 A）：自愈每个 drain 都跑，写
    registry.json 就把 main 工作区周期改脏——任何碰 registry 的 worktree 交付在 merge
    前检查即被拦（本单自身即受害者），且 ``models`` 按 charter 属护栏级、机器静默改写
    本身越权。故跟踪值优先遮蔽解析值，遮蔽情况报进返回值 ``shadowed``（要换档由人决定：
    改跟踪值，或删掉该键让机器接管）。合并粒度是**整条链**，运行态值不会往人写的链中间插席。

    目录里缺的名**跳过该席、其余照旧**（未命中的档保持现值不清空——目录缺名多为账户瞬时
    状态，宁旧勿空；清空 max 会静默改变计费渠道）。因此「跟踪链含账户已摘除的席」只报
    ``missing``、不报 ``shadowed``（那不是人能改的漂移，报了反而淹没真信号）。

    Returns:
        ``{"changed": {tier: {"old":…, "new":…}}, "missing": ["tier←名"…],
        "shadowed": {tier: {"tracked":…, "resolved":…}},
        "conflicts": ["名∈档A，规则说档B"…], "catalog_size": n}``。

    Raises:
        ValueError: 目录解析为空（输出格式变化/exe 异常）；
        以及 :func:`list_models_output` 的子进程异常——调用方 fail-open。
    """
    catalog = parse_model_catalog(list_models_output(timeout_s=timeout_s))
    if not catalog:
        raise ValueError("--list-models 输出解析为空目录（格式变化或 exe 异常？）")
    reg = Registry.load()
    templates, rules = split_model_tables(
        reg.data.get("model_chains"), reg.data.get("model_patterns")
    )
    tracked = reg.data.get("models") or {}
    missing: list[str] = []
    shadowed: dict[str, dict[str, Any]] = {}
    machine: dict[str, list[str]] = {}
    seat_names: dict[str, str] = {}
    resolved_names: dict[str, set[str]] = {}
    for tier, names in sorted(templates.items()):
        seats: list[str] = []
        tier_names: set[str] = set()
        dropped = False
        for name in names:
            if not name:
                seats.append("")  # 不传 -m 的裁决席：目录里没有它，原样占一席
                continue
            tier_names.add(name)
            if name not in catalog:
                missing.append(f"{tier}←{name}")
                dropped = True
                continue
            seats.append(catalog[name])
            seat_names[catalog[name]] = name
        resolved_names[tier] = tier_names
        if not seats:
            continue
        if dropped and not any(seats):
            # 链里的真名**一个都没**在目录里命中，只剩「不传 -m」的占位席。那不是解析结果，
            # 写成运行态等于替人裁决这一档（且每个 drain 都在无意义 churn）→ 保持现值。
            continue
        seats = dedupe_seats(seats)
        if tier in tracked:
            # 人写的覆盖优先，机器不碰；只在**账户席齐全**时才比对，否则差异来自缺名而非漂移
            human = normalize_seats(tracked[tier])
            if not dropped and human != seats:
                shadowed[tier] = {"tracked": tracked[tier], "resolved": seats}
            continue
        machine[tier] = seats
    # 名→档 规则与模板链互相矛盾 = 串档（#74 教训的正牌哨兵：Mimo 被塞进 minimum 之外之类）。
    # 一席在同档的多条链里出现（Qwen3.8-Flash 同时在 flash 与 minimum）不算矛盾：规则说的
    # 档**属于**它出现的档集就是自洽的，否则历史倒表每跑一次都刷一遍假警报。
    conflicts: list[str] = []
    for name, tier in sorted(rules.items()):
        members = {other for other, names in resolved_names.items() if name in names}
        if members and tier not in members:
            conflicts.append(f"{name}∈{'、'.join(sorted(members))}，规则说 {tier}")
    changed = reg.merge_runtime_models(machine) if machine else {}
    # 预设席位（跟踪 presets 块，常为账号级 UUID）也进「席 → 人类名」表：告警面与
    # status 要报人名，而不是让用户对着 UUID 找是哪一席。
    by_id = {sid: name for name, sid in catalog.items()}
    for table in (reg.data.get("presets") or {}).values():
        if not isinstance(table, dict):
            continue
        for role in ("main", "pro"):
            for seat in normalize_seats(table.get(role)):
                if seat and seat not in seat_names and seat in by_id:
                    seat_names[seat] = by_id[seat]
    reg.merge_runtime_seat_names(seat_names)
    return {
        "changed": changed,
        "missing": missing,
        "shadowed": shadowed,
        "conflicts": conflicts,
        "catalog_size": len(catalog),
    }


def dirty_tracked_state(
    *, path: str = STATE_REL_PATH, timeout_s: int = 10
) -> list[str]:
    """git 跟踪的 ``path`` 范围（默认 ``orchestration/state/**``）中有未提交改动的文件路径。

    为什么要哨兵（backlog 20261012-080410-devops 方案 C）：运行时自愈把跟踪配置改脏时，
    组长只在组员交上来的 ``<blocked>`` 里才发现——「本次合并会触碰的脏面」已经变成了
    周期墙。此函数把同一事实搬到观测面（``pysci-dev doctor`` / ``orch status``）。

    **advisory，不判失败**：脏跟踪 state 常常正是组长在飞的改动或本单自己要改的文件，
    硬失败会把正常作业挡成事故。

    Args:
        path: 扫描范围（项目根相对的目录/文件形参）。默认只看跟踪的 state 面；压测隔离面
            复用同一份解析，改传 ``orchestration/pods/<member>``。
        timeout_s: git 调用上限（挂起即放弃）。

    Returns:
        相对项目根的路径列表，如 ``["orchestration/state/registry.json"]``；干净或
        **查不到**（非仓库/git 缺失/超时/任何异常）一律返回 ``[]``——哨兵失灵不该
        影响主流程（fail-open，README §4.3）。
    """
    try:
        proc = subprocess.run(
            [
                "git",
                "status",
                "--porcelain",
                "--untracked-files=no",
                "--",
                path,
            ],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=timeout_s,
            check=False,
        )
    except Exception:
        return []
    if proc.returncode != 0:
        return []
    paths: list[str] = []
    for line in proc.stdout.splitlines():
        if len(line) <= 3:
            continue
        # XY<空格>路径；重命名形如 ``old -> new`` 取箭头后（新面才是现存文件）
        target = line[3:].strip().strip('"')
        paths.append(target.split(" -> ")[-1].strip('"'))
    return paths


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
    "DEFAULT_MODEL_CHAINS",
    "DEFAULT_MODEL_FALLBACK",
    "DEFAULT_TOKENS_PER_CHAR",
    "Member",
    "ORCHESTRATION_ROOT",
    "REGISTRY_PATH",
    "RUNTIME_MODELS_KEY",
    "RUNTIME_NAME",
    "RUNTIME_SEAT_FAIL_KEY",
    "RUNTIME_SEAT_NAMES_KEY",
    "Registry",
    "SEAT_REQUIRED_KEYS",
    "STATE_REL_PATH",
    "TIER_KEYS",
    "dedupe_seats",
    "dirty_tracked_state",
    "model_tier_rules",
    "normalize_seat",
    "normalize_seats",
    "orchestration_rel",
    "pod_session_key",
    "pod_sessions_dir",
    "refresh_models",
    "resolve_exe",
    "split_model_tables",
]
