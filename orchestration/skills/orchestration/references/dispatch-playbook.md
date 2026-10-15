# 派发 / 成本 / 降级细则（技能 SKILL.md 的展开）

SKILL.md「派发要领 / 归档与成本 / 降级」是压缩版；本文件补回被压缩掉的细节。动派发参数、
读成本账、或 orch 降级手动派发前先读这里。

## 会话选择策略

- **默认 latest（续最近活跃）**：`dispatch <member>` 不带 `--session` 时续该成员最近活跃的
  会话——同一工作的连续环节靠它攒上下文，省重复交代。
- **无关新工作**用 `--session new --name <slug>`：另起会话，别把不相干的活塞进活跃会话
  （上下文互相污染）。
- **续更早的某会话**：先 `sessions <member> --all` 查池，再 `--session <sid>` 指定；会话池
  水位/归档见 README 与 `orch sessions --archive <sid> --distill`。

## 成本读数（BYOK）

- BYOK 模式：**max 档计量、flash 档恒 0**（flash 走包月额度，账单上不计增量）。
- `orch stats` 的覆盖率读数写作 **「覆盖 M/N 跳」**：M=有计量的跳数、N=总跳数——flash 跳
  不进 M，故 M<N 是正常形态，不是漏记。
- 耗时/成本异常增长的成员会话 → 提请组长归档轮换（见 devops 「巡检清单」）。

## devops 派发（双通道，task-20261014-devops-gap-d1/d2）

devops 是**队列成员**，两条通道**并存、非二选一**：

- **plan 环节直派（首选）**：基建改进本身是计划的一部分时，步骤写 `member: devops` 走
  `plan run`（同步推进到交接）或 `plan advance`（事件驱动、入队即返回）——job 为
  `kind=produce`，与其它成员同队列同 worker，上下文与交接进 plan 状态机。
- **backlog FIFO 组包（改进建议 / 批次通道）**：`approve` 采纳的建议入 backlog，由计划器
  组包入队（`kind=backlog-batch`），走 drain 账务（销账 / needs_leader / 延期 / 系统性停止）。

一句话关系：**plan 直派＝环节视角（计划内基建）；backlog＝建议/批次视角（计划外改进）；
同一个 devops FIFO 与 worker，执行只一份实现**（口径权威＝`queues.QUEUE_MEMBERS` 的 ADR 注记）。
完成唤醒：plan 环节 job 收 **JOB-DONE**；backlog 批次消化完另收 **QUEUE-EMPTY**
（`orch events list` 水位可查；watch 事件到达即唤醒组长）。

## 降级派发（orch 瘫痪时）

- orch 不可用 → 照 README **附录 A 手动模板派发**：**原生 exe 直调**，
  **勿用 .cmd 包装器**（`.cmd` 包装器会在 Git Bash 下把参数当路径改写、并吞掉退出码，
  令手动派发的读数与真实不符）；台账手工补记，随后尽快 `plan adhoc` 派 devops 修复。
- 手动派发同样**长任务后台运行**；不要 sleep-轮询。

## 任务书（`plan run/advance --task`）格式与落点

`--task <文件>` 是「本环节任务书」的**正文来源**（与 `--text` 二选一）。文件**只读内容**：
编排不按文件名找它——路径原样存进 plan 状态的 `task_file`，派发时 orch 再**复制**进成员
pod 的 `inbox/task-<时间戳>-<plan>-<step>.task.md`（**别手写 inbox**）。沿用现状的
**建议约定**（在册 35 份）：落点 `orchestration/state/plans/`（与 plan YAML 同处好查），
命名 `<plan-id>.task.md`，分环节再带步骤名 `<plan-id>-<step>.task.md`。

模板（五段 ↔ SKILL 铁律 5 的「自包含」一一对应）：

```markdown
# 任务书 <plan-id> · <step-id>：<一句话标题>

## 背景 / 目标
<为什么做、要做到什么；含必要上下文与已知事实>

## 输入（只读）
- <路径>：<为什么读它>

## 交付
- <产物路径>：<形态与要点>

## 白名单（写权限）
- 可写：<目录/文件>；新建运行态目录时一并列 `.gitignore`

## 验收预期
1. <可机械复核的判据：命令 + 期望输出>
```

组员看不到你的会话上下文——**这五段就是全部交代**；写不出的段 = 这一刀切错了
（粒度判据见 `plan-decomposition.md` §2）。

## 三种审查表达（`review: true` / `member: reviewer` / `orch review`）

backlog 20261015-202600：三者关系原先只在单行提及里，组长无从评估用哪个。

| 表达 | 形态 | 机械行为（实现为准） | 何时用 |
|---|---|---|---|
| 叶子 `review: true` | 该叶子派发时给**宿主 job** 带 review 标记 | 宿主**交付成功后 worker 自动**向 reviewer 队投**附属子 job**（`queues.spawn_review`）；宿主**不发** JOB-DONE——等宿主与 review **都收尾**才发一次 `STEP-REVIEW-DONE`，中间态 `awaiting_review`（由 `queues.completion` 派生） | 环节**必须过审才算完**，且不希望多一个环节位 |
| 叶子 `member: reviewer` | 审查是一个**独立环节**（自己的 id/brief/dirs 与任务书） | 与普通叶子同：占工作清单位、参与依赖；不带 review 标记 → 不再套一层附属子 job | 审查要有**独立位置/先后依赖**（视觉审核、跨产物比对），或结论要进 `plan advance` 工作清单 |
| `orch review <产物> --origin <member> [--goal…]` | **手动一次性**命令 | 由**当前会话**跑审查链（VERDICT → FAIL 回派返工 → 复审一次 → 二次 FAIL 升级仲裁） | 计划外产物复核；或要**马上**拿结论、不等队列 |

`queues enqueue <member> … --review` 是同一机制的**队列入口**（无 plan 时手投宿主 job 用）。
