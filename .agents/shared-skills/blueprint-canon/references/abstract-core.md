# 详细设计摘要（总纲）

> **所属**：PySci 蓝图 · 设计摘要
> **上位**：[SKILL.md](../SKILL.md)
> **效力**：与实现、旧计划、旧技能文档冲突时，一律以蓝图为准。

本文件是 PySci 蓝图“设计摘要”的总纲与目录。项目概述与全蓝图简目见 [SKILL.md](../SKILL.md)；各分册是设计摘要的正文。

PySci 在顶层分为三部分：**项目支撑系统**、**科研生产能力**与**研究线**。前者让中者与后者可构建、可迁移、可复现、可验证；中者是项目交付的科研能力；后者是用户的具体研究内容，只在本地保留。

## 项目支撑系统

分册：[项目支撑系统](abstract-support-systems.md)

项目支撑系统由三部分组成：

- **依赖管理**：让项目目录在目录边界内自足、可复制、可重建；工具链、DSH、Python 包与环境全部在 `sub_root/deps_management/` 内，项目外只保留路由 shim。
- **分层测试**：按“测试层级 = 抽象层级”分三层——可机械断言层（`sub_root/tests`）、可基准测试层（`sub_root/evals`）、主观层（meta）。让代码、技能契约与工作 Agent 的 harness 改进都能被相应手段检验。
- **DSH harness**：三个 preset（worker / leader / meta）与对应工作说明、技能面、子代理规则；用户按任务规模选择 worker 直做或 leader 组织。

## 科研生产能力

科研生产能力由 7 个**能力域**组成：文献研究、有限元仿真、文档写作、科研绘图、理论计算、AI 绘图、3D 建模。

每个能力域的交付形态是“**技能文档 + 社区工具/薄胶水**”：

- 技能文档在 `.agents/skills/<name>/`，只写领域决策、调用面与索引；细节进 `references/` 或源码。
- 后端优先由社区依赖承担；自研只写社区表达不了的薄胶水，放在 `sub_root/src/pysci/skills/<name>/` 或 `sub_root/src/pysci/support/`。
- 每个能力域按“新需求三段式”重建：选型（社区调研，用户在位）→ 计划与实施 → QA 同步。**不直接复制旧 pySciWS 实现**；允许一个能力域由多个社区依赖组合替代。

能力域之间不强制统一 CLI 形状；对外入口以该技能 `SKILL.md` 的 Invocation 为准。

## 研究线

研究线是用户的具体科研内容：

- 代码：`sub_root/src/pysci/research/<line>/`
- 脚本：`sub_root/scripts/research/<line>/`
- 资产与产物：`sub_root/data/research/<n>_<line>/`
- 测试：`sub_root/tests/research/<line>/`（本地）

研究线只在本地保留，不入仓库。仓库只含框架、技能文档、支撑系统与最小资产骨架。用户的研究判断、真实实验与最终验收不进入 Agent 的自动化范围。
