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

## 交付全文留档（`orchestration/state/deliveries/`）

每个 job 的**完整交付正文**落 `orchestration/state/deliveries/<job-id>.md`（头部元数据 + 正文
原文；**失败交付同样留档**——对返工分析有价值），落盘即提交入 git，远端/新克隆可取。
**JOB-DONE / STEP-REVIEW-DONE 事件细节自带该留档的绝对路径**：watch 唤醒行与 `events drain`
输出都能直接看到，故组长想在 `plan advance`（结算+派发）**之前**先读全文把关时，**不必**再猜
目录或翻 ledger（backlog 20261015-205906）。`plan advance` 的结算段也打印同一路径。

## devops 派发（双通道，task-20261014-devops-gap-d1/d2）

devops 是**队列成员**，两条通道**并存、非二选一**：

- **plan 环节直派（首选）**：基建改进本身是计划的一部分时，步骤写 `member: devops` 走
  `plan advance`（事件驱动、入队即返回）——job 为
  `kind=produce`，与其它成员同队列同 worker，上下文与交接进 plan 状态机。
- **backlog FIFO 组包（改进建议 / 批次通道）**：`approve` 采纳的建议入 backlog，由计划器
  组包入队（`kind=backlog-batch`），走 drain 账务（销账 / needs_leader / 延期 / 系统性停止）。

一句话关系：**plan 直派＝环节视角（计划内基建）；backlog＝建议/批次视角（计划外改进）；
同一个 devops FIFO 与 worker，执行只一份实现**（口径权威＝`queues.QUEUE_MEMBERS` 的 ADR 注记）。
完成唤醒：plan 环节 job 收 **JOB-DONE**；backlog 批次消化完另收 **QUEUE-EMPTY**
（`orch events list` 水位可查；watch 事件到达即唤醒组长）。

## 派发前：清面与交集闸（backlog 20261016-022657）

`pysci-dev worktree merge` 的**合并前交集检查**在「本次提交面 ∩ main 未提交面」非空时中止（rc=2）
——组员那边其实已做完，白烧一跳。高频脏面：`orchestration/state/registry.json`（每次成员注册/
档位/check 调整都改）、他人 pod 的自维护层（`AGENTS.md` 等，正被该成员编辑时）。

**派发纪律**：给「需改这些面」的基建单写任务书/派发前，**先提交或还原**这些 main 侧未提交面；
确需让组员在脏面上作业时，**任务书写一行「本单由组长先清面」**认账（写了即视为已知，派发侧不再
重复告警）。派发命令会**机械提示**声明面与脏面的交集（`dispatch` 与 `plan advance`/`plan adhoc`
的告警行）；merge 工具另把交集清单原样摊开，且**缺省自动搁置**这批相交改动（`git stash -u`）→
合并 → 三路应回（冲突则保留条目、不丢在飞改动）——故「交集 rc=2」不再是默认（`--no-shelve` 才
退回；backlog 20261017-051318 治审批/销账流在飞写 state 与合并并发）。

## run_failed 恢复清单（基础设施终止；backlog 20261017-104502）

组员被 harness 终止（如「Background tasks are still running after 600s」）时**交付正文为空**、
错误正文只有终止语，但**半成品往往已完整落盘**。按序处置：

1. **先盘点半成品**：读该环节产物目录、交付说明/报告、管线脚本的在盘情况；
2. **完整** → 续作任务书写明「**只核对与正式交付、不重做**」（把已有产物剪进交付）；
3. **会话选择**：按 capacity 纪律用 `--session new`（run_failed 后原会话不可复用）；
4. **缺失** → 才整单重做。

`plan advance` 对 run_failed 环节的 `[NEXT]` 会自动带一行本清单提示。

## 降级派发（orch 瘫痪时）

- orch 不可用 → 照 README **附录 A 手动模板派发**：**原生 exe 直调**，
  **勿用 .cmd 包装器**（`.cmd` 包装器会在 Git Bash 下把参数当路径改写、并吞掉退出码，
  令手动派发的读数与真实不符）；台账手工补记，随后尽快 `plan adhoc` 派 devops 修复。
- 手动派发同样**长任务后台运行**；不要 sleep-轮询。

## 任务书格式与落点（模板）

任务书骨架（背景/目标、输入、参考材料语义对齐、交付、白名单、验收预期）与落点约定
已按 §1 拆为分册：**`references/taskbook-template.md`**（写任务书前读它）。

## 三种审查表达（`review: true` / `member: reviewer` / `orch review`）

backlog 20261015-202600：三者关系原先只在单行提及里，组长无从评估用哪个。

| 表达 | 形态 | 机械行为（实现为准） | 何时用 |
|---|---|---|---|
| 叶子 `review: true` | 该叶子派发时给**宿主 job** 带 review 标记 | 宿主**交付成功后 worker 自动**向 reviewer 队投**附属子 job**（`queues.spawn_review`）；宿主**不发** JOB-DONE——等宿主与 review **都收尾**才发一次 `STEP-REVIEW-DONE`，中间态 `awaiting_review`（由 `queues.completion` 派生） | 环节**必须过审才算完**，且不希望多一个环节位 |
| 叶子 `member: reviewer` | 审查是一个**独立环节**（自己的 id/brief/dirs 与任务书） | 与普通叶子同：占工作清单位、参与依赖；不带 review 标记 → 不再套一层附属子 job | 审查要有**独立位置/先后依赖**（视觉审核、跨产物比对），或结论要进 `plan advance` 工作清单 |
| `orch review <产物> --origin <member> [--goal…]` | **手动一次性**命令 | 由**当前会话**跑审查链（VERDICT → FAIL 回派返工 → 复审一次 → 二次 FAIL 升级仲裁；**缺陷指纹口径**见 `verdict-package.md` §8） | 计划外产物复核；或要**马上**拿结论、不等队列 |

`queues enqueue <member> … --review` 是同一机制的**队列入口**（无 plan 时手投宿主 job 用）。
