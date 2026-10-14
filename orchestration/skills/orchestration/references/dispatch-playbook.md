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
