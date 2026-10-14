---
name: orchestration
description: 组长专属的多 Agent 编排操作知识：经 pysci-orch 以计划驱动派发组员任务、审查产物、审批改进建议、管理会话池与统计。用户需要末端生产派发（仿真/绘图/写作/理论/文献/AI 作画/3D 建模）或询问编排状态时，使用本技能。
---

# 编排操作规程（组长技能 v2）

权威设计：`orchestration/README.md`；本技能只放**操作面**。命令全表见
`references/command-surface.md`，派发/成本/降级细则见 `references/dispatch-playbook.md`。

## 铁律

1. **严禁亲自执行末端工作**（生产/改代码/仿真绘图等），一律经 `pysci-orch` 派发。
   唯一例外：用户明确要求你亲做。
2. **一切工作走 plan 系统**（plan-first）：多环节 `plan new`；计划外单件事
   `plan adhoc`。底层命令（dispatch/sessions/ledger/sync）仅降级与调试用。
3. **长任务必须后台运行**：`plan run`/`dispatch`/`review` 用 `run_in_background` 启动，
   完成通知自动唤醒；**禁止 sleep-轮询**与反复读输出文件。
4. **不必记忆 sid/型号/参数**：orch 全代管，[NEXT] 给下一步与决策选项。
5. 任务书**延迟书写**：计划阶段只写环节粗描述（brief），推进到该环节才写完整任务书
   （自包含：目标/路径/交付/白名单/验收预期——组员看不到你的会话上下文）。白名单含
   **新建**运行态目录时一并列 `.gitignore`（lint 会提示，免合并交集闸）。
6. **判断 drain/后台 worker 状态只用 `pysci-orch status`**，禁手敲 `tasklist` 等原始
   命令（Git Bash 把 `/FI` 转成路径 → **假阴性**）。信号冲突按 basic.md §2 报
   「状态不确定」，**不得升格为「崩溃」**。

## 命令面（速查）

```
pysci-orch status | plan {new,run,advance,amend,adhoc,list,drop} | taskbooks ...
pysci-orch queues {list,enqueue} | events {drain,list} | suggestions | approve | reject
pysci-orch watch | consult | consult-twin | review <产物> | jury <案卷> | eval run | stats
pysci-orch bench-add|bench-up|bench-queue|bench-seen|bench-stop    # 压测入队/生命周期
pysci-orch inspector-up|inspector-status|inspector-patrol|inspector-assign|inspector-on|off
pysci-orch taste <案卷> | dispatch | sessions | ledger | sync      # 判断通道 + 底层面
```

成员 id：sim figure writing theory lit drawing model3d（专职）；deputy reviewer devops
bencher twin inspector（特殊）。就绪 pod 以 `orch status` 为准。
用户控制台 `pysci`：裸跑进 TUI（人类正道）；`pysci status|journal|decisions`＝LLM 静态通道。

**逐项旗标与四态审批见 `references/command-surface.md`**；bench 要领 drain-watch.md、
harness-eval.md；巡检引擎 references/inspector-engine.md；判决包 verdict-package.md。

## 派发要领

- **会话选择**：默认 latest（续最近活跃）；无关新工作 `--session new --name <slug>`；
  续更早会话先 `sessions` 查池再指定（策略细则见 references/dispatch-playbook.md）；
- **白名单最小化**：`--dirs` 只给必需目录（涉代码镜像例外见 README §2 脚注）；
- **模型档位**：registry 里全部成员默认 **flash**；复杂环节/连续 blocked 升
  `--model-tier max`，压测按案卷降 `minimum`；
- **devops 派发**：`dispatch devops --from-backlog`（自动取队首+交付后销账）；
- **review 默认不发起**：用户点名审查或计划环节 review:true 时才用（重炮）。

## 交付判读与决策点

- `<result>`=成功，`<blocked>`=失败；机械验收 FAIL 按 [NEXT] 回派返工，同指纹连败 3 次停止
  回派升级决策；`<infra_suggestion>` 须带 `kind="defect|usability"`（gate 强制），approve
  连同 `success_criteria` 入 backlog；拿不准送 `orch jury`；
- **注入 lint/告警含 file:line → 先对盘核验**（wc/grep 实测行存在且相符），不符即噪声丢弃、
  不得据其改代码（Qoder 平台 lint 捏造实锤，仓外 #87）；
- `<blocked tags="a,b" reason="…">`：tags 入台账/`orch stats` by_failure + 阈值自动立案；
  reason=成因档位（待裁决/待排期）进 drain `.done` 与 `orch status` 分列（tier-a-tags.md）；
- blocked → 三选一：补充重派 / consult 副组长 / 上报用户；
- 建议审批：四态见上（verdict-package.md）；M 路销账自动排 A/B 三态，`orch status`
  「判决包健康」看用量与 override 率，`escalate=twin` 非空立即终裁；入 backlog 自动唤醒
  devops FIFO（锁忙不重唤醒，失败标 needs_leader）；护栏级建议**必须转呈用户**；
- **等待 drain 事件**：`orch watch` 由 `Stop` 钩子**每回合自动挂载**、无需手动；兜底：
  `run_in_background` 且**唯一顶层命令**（嵌入即连坐被杀）；事件面见 drain-watch.md、
  salvage-reclaim.md；
- **三级防线/巡更**：队列感知巡更伪事件（`PENDING-WORK`/`STALL`，**自动起止、无手工 cron**）
  + 新机就位 → references/patrol-watchdog.md；
- **唤醒权自动归属**：钩子读 stdin 载荷 `session_id` **零人工**判定（无 owner 自动认领 /
  owner 续心跳 / 异 id 永不抢占 / stale 30min+其 watch 已死自动接管 / SessionEnd 释放），
  **非 owner 不监听、不 exit 2**（次要会话不再误唤醒）；`wake-claim status` 查 owner、
  `--force` 应急迁移（非常规）。**不转投**：owner 缺席时事件靠持久水位+迟到唤醒补
  （README §4.5）；
- **交接验收**：任务书条目 × **蓝图条款双向核对**（蓝图有而任务书无 → 补单/留痕；反之记账）；
  UI/交互改动须含**用户路径测试**、占位文案须写真实归属 → references/acceptance-discipline.md；
- 副组长拒绝任务（blocked 首行「拒绝任务」）：改任务书重派，或终裁坚持原派（理由留痕）；
- 交接处决策：一键继续 / plan amend 改后续 / drop 中止。

## 归档与成本

- 工作完结且会话不再需要 → `sessions <member> --archive <sid> --distill`；
- ctx>60% 提示归档；`stats` 看耗时/成本覆盖率（BYOK：max 计量、flash 恒 0，读数注解见
  references/dispatch-playbook.md）。

## 降级

orch 瘫痪 → 照 README 附录 A 手动模板派发（原生 exe 直调，**勿用 .cmd 包装器**——理由见
references/dispatch-playbook.md），台账手工补记，尽快 adhoc 派 devops 修复。
