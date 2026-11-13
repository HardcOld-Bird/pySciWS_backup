---
name: blueprint-canon
description: PySci 蓝图正典（meta 视角裁剪版）：蓝图法则、分册导航与只读边界。meta 需要查阅设计哲学、分层测试目标层、做 harness 选型（自研 vs 社区）判断、或理解设计背后的“Why”时使用；本技能只导航，正文在共享正典里。
whenToUse: meta 改进 harness 前需要理解系统设计（分层测试、社区优先、原则冲突裁决）时；仅需操作规程时读 AGENTS.meta.md，不必加载本技能。
---

# 蓝图正典（Blueprint Canon）· meta 视角裁剪版

> **出处**：`.agents/shared-skills/blueprint-canon/SKILL.md`（worker/leader 共享原件，只读，由人类维护）。
> **裁剪理由**：删去“项目概述”长文（科研能力与工作区设计对 harness 工作非必需，分层测试目标层已摘入 `AGENTS.meta.md`）；保留法则、分册导航与协议。本技能**只做导航，不复制正文**——一切正文读原件（单一真源，防漂移）；冲突时以原件为准。

## 法则

**蓝图是唯一的事实来源。** 如果代码注释、旧计划或建议与蓝图文档冲突，**一律以蓝图为准**。

**meta 三不碰注记**：本技能供 meta **读**蓝图以理解系统（允许）；**改**蓝图禁止——蓝图是人类维护面（见 `AGENTS.meta.md` §2）。

## 蓝图目录（正文全在共享正典，只读）

- [philosophy-detailed.md](../../shared-skills/blueprint-canon/references/philosophy-detailed.md) — 设计哲学详版：`AGENTS.md` 各条原则的细微差别，用于设计权衡与冲突裁决。
- [philosophy-community-first.md](../../shared-skills/blueprint-canon/references/philosophy-community-first.md) — **社区优先与薄胶水（分册）**：完整正文、社区引入判定标准、全生态扩展点地图；harness 扩展选型（工具/技能/MCP/插件/preset）“自研 vs 社区”判断前先读。
- [abstract-support-systems.md](../../shared-skills/blueprint-canon/references/abstract-support-systems.md) — **项目支撑系统**：依赖管理、**分层测试系统**（可机械断言层 / 可基准测试层 / 主观层，meta 即第 3 层）。
- [abstract-harness.md](../../shared-skills/blueprint-canon/references/abstract-harness.md) — 三 preset、会话模型、子代理规则、DSH 接线；meta 改进 worker/leader harness 时的目标面。
- [abstract-core.md](../../shared-skills/blueprint-canon/references/abstract-core.md) — 总纲与目录（跨模块架构联系）。
- [abstract-workspace.md](../../shared-skills/blueprint-canon/references/abstract-workspace.md) — 工作区布局与入库边界；定位 harness 配置、技能目录、eval 产物时参考。

## 协议

1. 先读 `AGENTS.meta.md`（meta 的规程常驻；项目宪法摘录在其 §1）。
2. 需要“深度/背景/Why”时，按上表直接读共享正典对应分册；跨模块架构读 `abstract-core.md`。
3. **涉及 harness 扩展选型（自研 vs 社区）时**：先读 `philosophy-community-first.md` 的扩展点地图，按生态定位接缝并调查社区，再提出方案。
4. **尊重边界**：本技能与一切蓝图文档内不写实现细节；meta 只读不改。
