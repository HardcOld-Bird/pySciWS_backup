---
name: orchestration
description: 组长专属的多 Agent 编排操作知识：经 pysci-orch 以计划驱动派发组员任务、审查产物、审批改进建议、管理会话池与统计。当用户提出任何需要末端生产的工作（仿真/绘图/写作/理论/文献/AI 作画/3D 建模）需派发组员时，或询问编排体系状态时，使用本技能。组长自己严禁执行末端工作。
---

# 编排操作规程（组长技能 v2）

权威设计：`orchestration/README.md`。本技能只放**操作面**；架构疑问读 README。

## 铁律

1. **严禁亲自执行末端工作**（生产/改代码/跑仿真绘图等），一律经 `pysci-orch` 派发。
   唯一例外：用户明确要求你亲做。
2. **一切工作走 plan 系统**（plan-first）：多环节工作 `plan new`；计划外单件事
   `plan adhoc`。底层命令（dispatch/sessions/ledger/sync）仅降级与调试用，勿养成
   绕过 plan 的习惯。
3. **长任务必须后台运行**：`plan run`/`dispatch`/`review` 用 Bash 工具的
   `run_in_background` 启动，完成通知自动唤醒；等待期间**禁止 sleep-轮询**、禁止反复
   读输出文件。
4. 派发前不需要记忆 sid/型号/参数——orch 全代管，[NEXT] 段给下一步；需要决策时照
   选项执行。
5. 任务书**延迟书写**：计划阶段只写环节粗描述（brief），推进到该环节时才写完整任务书
   （自包含：目标/输入路径/交付要求/白名单/验收预期——组员看不到你的会话上下文）。

## 命令面

```
uv run pysci-orch status                        # 全景：计划/成员/队列/devops worker/台账
# —— 日常面 ——
uv run pysci-orch plan new <计划.yaml> [--id X]  # 登记计划（格式见 README §4.2）
uv run pysci-orch plan run <id> --text "<当前环节任务书>"   # 推进（交接处暂停）
uv run pysci-orch plan amend <id>               # 直接编辑 plan YAML 后校验对账
uv run pysci-orch plan adhoc <member> --text "..." [--dirs d1,d2]  # 计划外单步
uv run pysci-orch plan list | plan drop <id>
uv run pysci-orch suggestions                   # 待审批改进建议
uv run pysci-orch approve <建议id> --note "..." [--no-wake]
        # 采纳→入 backlog，默认自动唤醒 devops 后台 FIFO 消化；--no-wake 只入队不唤醒
uv run pysci-orch reject <建议id> --note "..."   # 否决（仅回复附送，不唤醒）
uv run pysci-orch consult "<问题>"               # 咨询副组长（平级：异议义务已注入）
uv run pysci-orch review <产物> --origin <member> [--rubric generic]
        # 审查链：VERDICT → FAIL 自动回派返工 → 复审一次 → 二次 FAIL 升级你仲裁
uv run pysci-orch stats [--plan id|--member m|--days N]
# —— 底层命令面（降级/调试）——
uv run pysci-orch dispatch <member> (--task F|--text T|--from-backlog) [--session latest|new|<sid>] ...
uv run pysci-orch sessions <member> [--all|--archive <sid> [--distill]|--prune <sid>]
uv run pysci-orch ledger [--member --days --stats] | sync [--check]
```

成员 id：`sim` `figure` `writing` `theory` `lit` `drawing` `model3d`（专职）；
`deputy` `reviewer` `devops`。已就绪 pod 以 `orch status` 为准。

## 派发要领

- **会话选择**：默认 latest（续最近活跃会话）；全新无关工作 `--session new --name <slug>`；
  续更早会话先 `sessions` 查池再显式指定；
- **白名单最小化**：`--dirs` 只给任务需要的目录；生产型组员涉代码镜像时加
  `src/pysci/research/<线>/`（任务级授权例外，README §2 脚注）；
- **模型档位**：专职组员默认 flash；复杂环节/连续 blocked 升 `--model-tier max`；
  deputy/reviewer/devops 默认 max；
- **devops 派发**：`dispatch devops --from-backlog`（自动取队首+交付后销账）；
- **review 默认不发起**：用户点名审查或计划环节 review:true 时才用（重炮）。

## 交付判读与决策点

- `<result>`=成功，`<blocked>`=失败（含卡点）；机械验收 FAIL 按 [NEXT] 回派返工，
  同指纹连败 3 次停止回派升级决策；
- blocked → 三选一：补充重派 / consult 副组长 / 上报用户；
- 建议审批：approve 入 backlog 并**自动唤醒 devops 后台消化**（分离进程持锁 FIFO 串行；
  锁忙不重复唤醒，单项失败标 needs_leader 跳过、不阻塞队列），reject 附理由——回复都会
  自动附送提议人；`orch status` 看 devops worker 态 / 最近 run / needs_leader 提醒；
  护栏级建议（README/角色边界/审批权/guards）**必须转呈用户**；
- 副组长拒绝任务（blocked 首行「拒绝任务」）：修改任务书重派，或行使最终裁定权
  坚持原派（理由留痕台账）；
- 交接处决策：一键继续 / plan amend 改后续 / drop 中止。

## 归档与成本

- 工作完结且会话不再需要 → `sessions <member> --archive <sid> --distill`；
- ctx>60% 的会话提示归档；`stats` 看成员耗时占比（BYOK 下 credits 恒 0，以耗时/轮数
  为准）；成本曲线异常 → 考虑 harness 改进或档位调整。

## 降级

orch 瘫痪 → 照 README 附录 A 手动模板派发（原生 exe 直调，**勿用 .cmd 包装器**），
台账手工补记，尽快以 adhoc 计划派 devops 修复。
