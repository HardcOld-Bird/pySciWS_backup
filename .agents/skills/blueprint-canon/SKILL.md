---
name: blueprint-canon
description: PySci 的“蓝图正典”。包含项目概述、详细的设计哲学和全项目设计摘要。当你需要概览项目全貌、规划新功能、解决需求冲突、评估“自研 vs 社区”、理解设计背后的“原因（Why）”、或查阅项目长期愿景时使用。
---

# 蓝图正典 (Blueprint Canon)

本技能是**分层蓝图** 的档案馆。它由人类维护。作为 AI，你必须将其视为最高权威。

## 项目概述



## 法则

**蓝图是唯一的事实来源。**

如果代码注释、旧计划或子代理的建议与本页或 `references/` 中的文档冲突，**一律以蓝图为准**。

## 蓝图目录

蓝图由 `AGENTS.md`（设计哲学·简版）、`philosophy-detailed.md`（设计哲学·详版）、`philosophy-community-first.md`（社区优先分册）与设计摘要分册组成。以下为一句话目录：需要全局理解时，先读总纲；仅需要对应部分细节时，可跳过总纲直接打开对应分册。

### 设计哲学（详版）

- [philosophy-detailed.md](references/philosophy-detailed.md) — 项目哲学的“长文版”：解释 `AGENTS.md` 中各条原则的细微差别，用于设计权衡与冲突裁决。
- [philosophy-community-first.md](references/philosophy-community-first.md) — **社区优先与薄胶水（分册）**：2.1 的完整正文、社区引入判定标准，以及按生态组织的全生态扩展点地图；新需求选型、依赖引入或“自研 vs 社区”判断前先读。

### 设计摘要



## 协议

1. **先读 `AGENTS.md`**（它已在你的上下文中）。
2. **需要“深度”或“背景”时，先读本页的“项目概述”和“蓝图目录”**，再按需打开分册；需要跨模块的架构与联系时，读 `abstract-core.md`。
3. **涉及新需求、依赖选型或“自研 vs 社区”判断时**：先读 [philosophy-community-first.md](references/philosophy-community-first.md) 的扩展点地图，按生态定位接缝并调查社区，再提出方案。
4. **尊重边界**：不要在这里写实现细节。如果你发现自己在这些文件里写代码逻辑，说明你来错地方了。
