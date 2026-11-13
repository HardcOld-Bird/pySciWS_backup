---
name: harness-dev
description: meta 改进 worker/leader harness 的操作知识：preset/profile/技能/工作说明的修改、备份、冒烟、回滚与热改风险。需要改 DSH preset、技能目录、AGENTS 规程或排查 harness 不生效时使用。
---

# Harness 开发

## 目标面

worker/leader 的 harness = preset 组合 + 技能 + 工作说明 + 规程 + 工具面。你只改这些；不改蓝图、项目代码、测试判据、研究内容。

## 配置位置

- 项目 DSH：`sub_root/deps_management/deps/node/modules/dsh_home`
- web profile：`.../dsh_home/profiles/web/`
- preset 声明：`.../profiles/web/cordis.patch.yml`
- 上游片段：`.../profiles/web/pysci/{standard,cordis}.plugins.yml`（生成物，不手改）
- 技能根：`.agents/skills`（worker）、`.agents/leader-skills`（leader）、`.agents/meta-skills`（meta）、`.agents/shared-skills`（共享蓝图）
- 工作说明：`AGENTS.md`、`AGENTS.leader.md`、`AGENTS.meta.md`

## 修改流程

1. **先读现状**：`dsh --profile web --dump-config`、`cordis.patch.yml`、相关技能 `SKILL.md`；不要凭记忆。
2. **写清假设**：改什么、为什么、预期可观测变化、如何回滚。
3. **备份**：`cordis.patch.yml` 复制为 `.bak-<date>-<reason>`；技能/AGENTS 用 git 历史。
4. **小步**：一次只改一个可验证增量；preset 声明变更只在无运行会话时做。
5. **冒烟**：新建会话，检查工具表、技能目录、工作说明、子代理枚举；必要时 `dsh --profile web --dump-config`。
6. **记录**：改动、冒烟结果、回滚点写入 `sub_root/evals/reports/improvement-ledger.md`。

## 热改风险

- preset 声明在运行中改可能导致工具表坍缩/代际缓存不一致（mySimsWS2 有事故前科）。
- 规则：无运行会话时改；改后重启；保留备份；不要用 `!!js` 写跨代际配置值（除非已验证）。
- 技能目录增删通常可被 watcher 发现，但新技能在下一轮 catalog 才可见；改后新建会话验证。

## 常见故障

- `patch: entry "X" not found`：patch 的 id 在当前 DSH 版本不存在；查 `--dump-config` 的真实 id，删掉/改写。
- 技能不显示：frontmatter 非法、名称非 kebab-case、目录不在挂载的 `customSkillDirs`、`includeDefaultRoots: false` 没显式列。
- 子代理不能选 preset：`allowedAgentPresets` 未包含该 id，或 `agentPreset` 默认值不在白名单。
- leader 出现 leader 子代理：fork/workflow 未禁用；它们继承父 preset。
- 工具表异常：立即停止改，回滚备份，重启，记录。

## 边界

- 你只能改 harness；需要改项目代码/测试判据/蓝图时，向用户提出，由用户决定是否派给 worker。
- 不要为了让 eval 通过而改判据；发现判据问题记台账，交用户。
