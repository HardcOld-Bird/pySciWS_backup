"""提供商预设：qwen / mimo / deepseek 一键切换（backlog 20261012-225500-leader）。

用户裁决 2026-10-12 22:4x：**档位模式（max/flash）退役**。预设把「主力席 + pro 席 +
仲裁团构成」打包成一个字符串，改 ``data/pysci_settings.toml`` 的 ``[model] preset``
一行即整体换血；生效粒度=下一次派发，在途会话不受影响。

三条边界：

- **角色（main/pro/minimum）≠ 预设**：预设决定「这一席是什么型号」，角色决定「这个成员
  本跳用哪一席」。旧档名 ``flash``/``max`` 经 :data:`ROLE_ALIASES` 兼容读到 main/pro
  （plan YAML 与老案卷里还写得有），``minimum`` 独立、不随预设走（E3-live 已判
  火山方舟不可行，维持现有链）；
- **pro 席名单由预设自带**（qwen 刻意只给 twin，mimo 给四人组——两预设不对称是用户
  2026-10-12 确认的有意设计，不是漏配）；
- **deepseek 过渡期不可激活**：BYOK 精确名尚未回填（toml 注释标 ⚠）。空席表 + 未配置
  声明 → 派发拒发、status 出声，**不**静默落到别的预设上烧错渠道。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from . import settings

#: ``--model-tier`` 的合法角色（argparse choices 与角色覆盖同源）。
ROLE_CHOICES: tuple[str, ...] = ("main", "pro", "minimum")

#: 旧档名 → 角色（只做已知别名的换算；未知串原样透传，交给 argparse choices 判非法）。
ROLE_ALIASES: dict[str, str] = {"flash": "main", "max": "pro"}


@dataclass(frozen=True)
class Preset:
    """一个预设的全部打包内容（席位表是**有序回退链**，首席在前）。"""

    name: str
    #: 主力席链：绝大多数成员的默认（组长默认模型同源，现场 /model 管理优先于预设）。
    main: tuple[str, ...] = ()
    #: pro 席链：仅 :attr:`pro_members` 名单内的成员默认取此席。
    pro: tuple[str, ...] = ()
    #: 仲裁团**构成**（家族名顺序，官方直调选型权归用户）：面板=这三家，且天然不含本族。
    jury: tuple[str, ...] = ()
    #: 本预设下取 pro 席的成员 id 名单。
    pro_members: tuple[str, ...] = ()
    #: 非空 = 过渡期不可激活的原因（派发拒发 / status 告警共读这一句）。
    unavailable: str = ""
    #: 设计注记（status 与文档引用，说明「为什么是这几家」）。
    note: str = ""


#: 出厂预设表（席位值=人类名/BYOK 名；跟踪 registry.json 的 ``presets`` 块可整链覆盖，
#: 生产账号的 BYOK UUID 落那里——选型权归用户，代码不内置账号级 ID）。
PRESETS: dict[str, Preset] = {
    "qwen": Preset(
        name="qwen",
        main=("Qwen-3.8-Flash", "Qwen3.8-Flash"),
        pro=("Qwen-3.8-Max", "Qwen3.8-Flash"),
        jury=("glm", "kimi", "mimo"),
        pro_members=("twin",),
        note="pro 席仅 twin（Qwen pro 贵，刻意不规模化）；仲裁团排除 Qwen 同族",
    ),
    "mimo": Preset(
        name="mimo",
        main=("Mimo-V2.6-Flash", "Qwen3.8-Flash"),
        pro=("Mimo-V2.6-Pro", "Qwen3.8-Flash"),
        jury=("glm", "kimi", "qwen"),
        pro_members=("reviewer", "deputy", "inspector", "twin"),
        note="pro 席四人组；仲裁团排除 Mimo 同族",
    ),
    "deepseek": Preset(
        name="deepseek",
        main=(),
        pro=(),
        jury=("glm", "kimi", "mimo"),
        pro_members=(),
        unavailable=(
            "BYOK 精确名待回填（data/pysci_settings.toml [model] 注释标 ⚠；"
            "--list-models 现无 DeepSeek-V4-Flash 条目）"
        ),
        note="全员同档无 pro 席；仲裁团排除 DeepSeek 同族",
    ),
}


@dataclass(frozen=True)
class ActivePreset:
    """当前生效的预设读数（status/派发/仲裁团三面共用同一对象，防口径漂移）。"""

    preset: Preset
    name: str
    #: 读配置时攒下的警告（文件缺席 / 非法串），随读数面一起出声。
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def available(self) -> bool:
        """过渡期预设（席表未配置）= 不可激活。"""
        return not self.preset.unavailable

    @property
    def unavailable_reason(self) -> str:
        return self.preset.unavailable

    def chain(self, role: str) -> list[str]:
        """该角色的出厂席位链（跟踪 registry 覆盖在 registry 侧合并，本函数只给出厂值）。"""
        seats = self.preset.pro if role == "pro" else self.preset.main
        return list(seats)


def normalize_role(value: Any) -> str:
    """argparse ``type=``：旧档名归一到角色（未知串原样返回，交 choices 判非法）。"""
    key = str(value).strip().lower()
    return ROLE_ALIASES.get(key, key)


def active(data: dict[str, Any] | None = None) -> ActivePreset:
    """当前预设（含配置读取警告）；``data`` 可注入已读配置（测试免读盘）。"""
    if data is None:
        data = settings.read_settings()
    name, warnings = settings.preset_name(data)
    return ActivePreset(
        preset=PRESETS.get(name, PRESETS[settings.DEFAULT_PRESET]),
        name=name,
        warnings=tuple(warnings),
    )


def resolve_role(member_id: str, override: Any = None) -> str:
    """本跳该成员用哪个角色。

    优先级：派发旗标/案卷覆盖 > 预设的 pro 名单 > main。覆盖值先过
    :func:`normalize_role`（plan YAML 里的 ``model: flash`` 走这条），非法值抛
    ``ValueError``——调用方（dispatch）据此拒发，不静默回落烧错档。
    """
    if override:
        role = normalize_role(override)
        if role not in ROLE_CHOICES:
            raise ValueError(
                f"未知角色/档位 {override!r}（合法值 {'/'.join(ROLE_CHOICES)}；"
                f"旧档名 flash/max 已别名到 main/pro）"
            )
        return role
    active_preset = active()
    return "pro" if member_id in active_preset.preset.pro_members else "main"


def role_chain_source(role: str) -> str:
    """角色 → 出厂席表字段名（``main``/``pro``；``minimum`` 不走预设，调用方自取）。"""
    return "pro" if role == "pro" else "main"


def _chain_text(reg: Any, role: str) -> str:
    """席位链的展示串（席显示名经 ``seat_label`` 反查；空链显式出声，不留空串骗人）。"""
    seats = reg.model_chain(role)
    return " → ".join(reg.seat_label(s) for s in seats) if seats else "(空)"


def status_lines(reg: Any) -> list[str]:
    """``orch status`` 的「模型预设」段：预设名 + 各席解析值 + 构成 + 告警。

    单一源（success_criteria 要求 status 与派发读同一张表）：链读数走
    ``Registry.model_chain``，预设读数走 :func:`active`，别处不得另拼一份。
    """
    act = active()
    preset = act.preset
    out = ["== 模型预设（真源 data/pysci_settings.toml [model] preset）=="]
    out.append(f"  预设 {act.name}｜{_chain_source_note(preset)}")
    out.append(
        f"  main 席（大部分成员+组长默认，现场 /model 优先）：{_chain_text(reg, 'main')}"
    )
    if preset.pro:
        out.append(
            f"  pro 席（{'/'.join(preset.pro_members)}）：{_chain_text(reg, 'pro')}"
        )
    else:
        out.append("  pro 席：—（本预设全员同档）")
    out.append(f"  仲裁团构成：{'+'.join(preset.jury)}（预设同族排除）")
    out.append(f"  minimum（独立，不随预设走）：{_chain_text(reg, 'minimum')}")
    if not act.available:
        out.append(f"[!] 预设 {act.name} 未激活：{act.unavailable_reason}")
    for warn in act.warnings:
        out.append(f"[!] {warn}")
    return out


def _chain_source_note(preset: Preset) -> str:
    """预设注记（设计意图一句话，避免 status 只报型号不报为什么）。"""
    return preset.note or ""
