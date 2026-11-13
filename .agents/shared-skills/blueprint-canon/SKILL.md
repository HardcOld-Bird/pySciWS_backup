---
name: blueprint-canon
description: PySci 的“蓝图正典”。包含项目概述、详细的设计哲学与全项目设计摘要。当需要概览项目全貌、规划新功能、解决需求冲突、评估“自研 vs 社区”、理解设计背后的“原因（Why）”、查阅项目长期愿景或技能重建规则时使用。
---

# 蓝图正典 (Blueprint Canon)

本技能是**分层蓝图**的档案馆。它由人类维护。作为 AI，你必须将其视为最高权威。

## 项目概述

PySci 是一个**个人科研工作区**：把科研全流程中可自动化的环节——文献调研、理论计算、有限元仿真、科研绘图、论文与办公文档写作、AI 绘图、3D 建模——交给 LLM Agent 驱动；人只保留真实物理世界的实验、研究判断与最终验收。

项目由三部分组成：

1. **项目支撑系统**：依赖管理（一切工具链与依赖在项目目录内，统一更新器维护）、分层测试（可机械断言层 / 可基准测试层 / 主观层）与三 preset 的 DSH harness。
2. **科研生产能力**：7 个科研能力域，每个以“技能文档 + 社区工具/薄胶水”的形式交付；不追求单一自研大包，优先组合社区依赖。
3. **研究线**：用户的具体研究内容（代码、数据、脚本、测试）只在本地保留，不入仓库；仓库只含可重建的框架与骨架。

核心工程原则：**维护面最小化**。能用社区工具的不自研；自研只写社区表达不了的薄胶水；依赖、配置、DSH 与 Python 环境都留在项目目录内；项目外只保留一层路由 shim。

## 法则

**蓝图是唯一的事实来源。**

如果代码注释、旧计划、旧技能文档或子代理的建议与本页或 `references/` 中的文档冲突，**一律以蓝图为准**。

## 蓝图目录

蓝图由 `AGENTS.md`（设计哲学·简版）、`AGENTS.leader.md`、`AGENTS.meta.md`、本页与设计摘要分册组成。需要全局理解时先读总纲；只需要某部分细节时可直接打开对应分册。

### 设计哲学（详版）

- [philosophy-detailed.md](references/philosophy-detailed.md) — 项目哲学的“长文版”：解释 `AGENTS.md` 中各条原则的细微差别，用于设计权衡与冲突裁决。
- [philosophy-community-first.md](references/philosophy-community-first.md) — **社区优先与薄胶水（分册）**：社区引入判定标准与按生态组织的全生态扩展点地图；新需求选型、技能重建、依赖引入前必读。

### 设计摘要

- [abstract-core.md](references/abstract-core.md) — **总纲**：项目支撑系统、科研生产能力、研究线的顶层关系与分册入口。
- [abstract-harness.md](references/abstract-harness.md) — **三 preset**：worker（默认）/ leader（可选）/ meta（仅 harness）、会话模型、子代理规则与 DSH 接线。
- [abstract-support-systems.md](references/abstract-support-systems.md) — **项目支撑系统**：依赖管理（项目内自包含、统一更新器）与分层测试（三层）。
- [abstract-workspace.md](references/abstract-workspace.md) — **工作区布局**：`src/`、`scripts/`、`data/`、`sub_root/`、`.agents/` 的职责、镜像关系与入库边界。

## 协议

1. **先读 `AGENTS.md`**（它已在你的上下文中）。
2. **需要“深度/背景/Why”时**，先读本页的“项目概述”和目录，再按需打开分册；需要跨模块关系时读 `abstract-core.md`。
3. **涉及新需求、技能重建、依赖选型或“自研 vs 社区”判断时**：先读 [philosophy-community-first.md](references/philosophy-community-first.md) 的扩展点地图，按生态定位接缝并调查社区，再提出方案；技能重建必须与用户逐轮确认。
4. **尊重边界**：蓝图只写设计目标与约束，不写实现细节。实现细节以源码、`paths.py` 与工具文档为准。
