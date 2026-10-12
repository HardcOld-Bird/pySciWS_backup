---
name: orchestration
description: 组长专属的多 Agent 编排操作知识：经 pysci-orch 以计划驱动派发组员任务、审查产物、审批改进建议、管理会话池与统计。当用户提出任何需要末端生产的工作（仿真/绘图/写作/理论/文献/AI 作画/3D 建模）需派发组员时，或询问编排体系状态时，使用本技能。组长自己严禁执行末端工作。
---

# 编排操作规程（组长技能 v2）

权威设计：`orchestration/README.md`；本技能只放**操作面**。

## 铁律

1. **严禁亲自执行末端工作**（生产/改代码/跑仿真绘图等），一律经 `pysci-orch` 派发。
   唯一例外：用户明确要求你亲做。
2. **一切工作走 plan 系统**（plan-first）：多环节工作 `plan new`；计划外单件事
   `plan adhoc`。底层命令（dispatch/sessions/ledger/sync）仅降级与调试用。
3. **长任务必须后台运行**：`plan run`/`dispatch`/`review` 用 Bash 的
   `run_in_background` 启动，完成通知自动唤醒；**禁止 sleep-轮询**与反复读输出文件。
4. **不必记忆 sid/型号/参数**：orch 全代管，[NEXT] 给下一步与决策选项。
5. 任务书**延迟书写**：计划阶段只写环节粗描述（brief），推进到该环节时才写完整任务书
   （自包含：目标/输入路径/交付要求/白名单/验收预期——组员看不到你的会话上下文）。
6. **判断 drain/后台 worker 状态只用 `pysci-orch status`**，禁手敲 `tasklist` 等原始
   命令（Git Bash 把 `/FI` 转成路径 → **假阴性**）。信号冲突按 basic.md §2 报
   「状态不确定」，**不得升格为「崩溃」**。

## 命令面

```
pysci-orch status                        # 全景：计划/成员/队列/worker/台账
# —— 日常面 ——
pysci-orch plan new <计划.yaml> [--id X]  # 登记计划（格式见 README §4.2）
pysci-orch plan run <id> --text "<当前环节任务书>"   # 推进（交接处暂停）
pysci-orch plan amend <id>               # 编辑 plan YAML 后校验对账
pysci-orch plan adhoc <member> --text "..." [--dirs d1,d2]  # 计划外单步
pysci-orch plan list | plan drop <id>
pysci-orch suggestions                   # 待审批改进建议
pysci-orch approve <建议id> (--jury | --bench <案id[,案id]> | --override "<理由>")
        [--note "..."] [--no-wake] [--criteria "<判据>"]  # 四态判决包必居其一（例外=缺陷类
        # ×该跳 blocked）：J 出 B 自动驳回；M 锚基线、销账时自动排 A/B。缺 kind/
        # success_criteria 亦拒入队。规程 references/verdict-package.md
pysci-orch watch [--timeout 14400]
        # 等 drain 五类事件（QUEUE-COMPLETE/NEEDS-LEADER/QUOTA/CHANNEL-STOP/CRASH）；
        # run_in_background 挂起，漏挂由 status 补账
pysci-orch reject <建议id> --override "<判决依据>"   # 自裁否决：必填依据、留痕（J/M 判决的驳回不计 override）
pysci-orch consult "<问题>"               # 咨询副组长（平级：异议义务已注入）
pysci-orch consult-twin <案卷.md>       # 咨询 twin（taste_enabled=false 即拒 rc=2）
pysci-orch review <产物> --origin <member> [--rubric generic]
        [--goal-from <原生产任务书> | --goal "<目标陈述>"]
        # 审查链：VERDICT → FAIL 回派返工 → 复审一次 → 二次 FAIL 升级仲裁；
        # --goal* 把组长目标陈述送任务书，供 reviewer G2 独立核验
pysci-orch jury <案卷.md> [--dry|--timeout S]  # Tier C 仲裁团（非 Qwen 双序对局，阻塞
        # 出裁决；不可达入队 rc=1；格式 jury-panel.md，渠道面 jury-channels.md）
pysci-orch eval run (--case <id>|--all) [--label L]
        # Tier B 压测**抽检面**（runner=leader）；套件本体归 bencher
pysci-orch bench-add <run|capture|ab|check> [--cases a,b] [--ref-a R --ref-b R]
        [--label L] [--tier minimum] [--brief T] [--start]  # 测量任务入队（不阻塞）
pysci-orch bench-up [--dry] | bench-queue [--all] | bench-seen <id> | bench-stop
        # bencher 生命周期（FIFO 非常驻）；判决进 status「待查看」面不即时唤醒
        # （§5.1-B）。要领/案卷格式 references/drain-watch.md、harness-eval.md
pysci-orch taste <案卷.md> [--dry]  # twin 判断通道（§9 冷启动+宪章注入；
        # taste_enabled 未生效即拒派发，references/twin-taste-mirror.md）
pysci-orch inspector-up [--dry|--force] | inspector-status | inspector-patrol [--target 菜单id]
        [--start] | inspector-assign --brief T [--start] | inspector-on|off
        # 巡检引擎生命周期（§7）：拉起已内嵌 status/approve/dispatch（幂等；开关/让路/
        # 迟滞/预算在代码与 registry.inspector，无 CLI 旋钮）。口径 references/inspector-engine.md
pysci-orch stats [--plan id|--member m|--days N]
# —— 底层命令面（降级/调试）——
pysci-orch dispatch <member> (--task F|--text T|--from-backlog) [--session latest|new|<sid>] ...
pysci-orch sessions <member> [--all|--archive <sid> [--distill]|--prune <sid>]
pysci-orch ledger [--member --days --stats] | sync [--check]
```

成员 id：sim figure writing theory lit drawing model3d（专职）；deputy reviewer devops
bencher twin inspector（特殊）。就绪 pod 以 `orch status` 为准。

## 派发要领

- **会话选择**：默认 latest（续最近活跃会话）；全新无关工作 `--session new --name <slug>`；
  续更早会话先 `sessions` 查池再指定；
- **白名单最小化**：`--dirs` 只给任务需要的目录（生产型组员涉代码镜像的例外见
  README §2 脚注）；
- **模型档位**：registry 里全部成员默认 **flash**；复杂环节/连续 blocked 升
  `--model-tier max`，压测按案卷降 `minimum`；
- **devops 派发**：`dispatch devops --from-backlog`（自动取队首+交付后销账）；
- **review 默认不发起**：用户点名审查或计划环节 review:true 时才用（重炮）。

## 交付判读与决策点

- `<result>`=成功，`<blocked>`=失败（含卡点）；机械验收 FAIL 按 [NEXT] 回派返工，同指纹
  连败 3 次停止回派升级决策；`<infra_suggestion>` 须带 `kind="defect|usability"`
  （delivery-gate 强制），approve 把它与正文 `success_criteria` 一并写入 backlog；
  拿不准的送 `orch jury`（references/jury-panel.md）；
- `<blocked tags="a,b">` 失败标签入台账/`orch stats` by_failure；滑窗内同 (tag,member)
  达 registry `tier_a` 阈值 → 自动立 source=tier-a 提案（references/tier-a-tags.md）；
- blocked → 三选一：补充重派 / consult 副组长 / 上报用户；
- 建议审批：判决包四态见上（细则 references/verdict-package.md）；M 路销账自动排 A/B
  三态落地，`orch status`「判决包健康」段看通道用量与 override 率，`escalate=twin`
  非空须立即终裁；入 backlog 后**自动唤醒 devops FIFO**（锁忙不重复唤醒，单项失败标
  needs_leader 跳过）；护栏级建议（README/角色边界/审批权/guards）**必须转呈用户**；
- **挂 `orch watch` 等事件**（照 [NEXT]，`run_in_background`）：watch 必须是该后台调用的
  **唯一顶层命令**（嵌进 `&`/`&&`/管道会被父任务连坐杀死）；
  挂载纪律与 `bench-*` 见 references/drain-watch.md、salvage-reclaim.md（抢救面）；
- 副组长拒绝任务（blocked 首行「拒绝任务」）：修改任务书重派，或行使最终裁定权
  坚持原派（理由留痕台账）；
- 交接处决策：一键继续 / plan amend 改后续 / drop 中止。

## 归档与成本

- 工作完结且会话不再需要 → `sessions <member> --archive <sid> --distill`；
- ctx>60% 提示归档；`stats` 看耗时占比与成本覆盖率（BYOK：max 档计量、flash 恒 0，
  故标注「覆盖 M/N 跳」）。

## 降级

orch 瘫痪 → 照 README 附录 A 手动模板派发（原生 exe 直调，**勿用 .cmd 包装器**），
台账手工补记，尽快 adhoc 派 devops 修复。
