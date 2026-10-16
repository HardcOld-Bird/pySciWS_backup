# 计划驱动与跨任务路由（README §4.2 全量）

> 由 `orchestration/README.md §4.2` 迁入（细则块；**唯一权威仍是 README**）。命令表见
> `command-surface.md`；拆解方法论与 **plan YAML 字段速查表（§8）**见 `plan-decomposition.md`。

**plan-first**：一切工作走 plan。`steps[]` 元素二型**递归**——普通 step 与 **superstep**
（`{id, mode: serial|parallel, children:[…]}`）；叶子在 state.json 扁平化为**点路径 id**
（`A.P1.p1-1`），`_reconcile` 按 id 对账（YAML 增删天然兼容）。顶层 `steps[]` 即隐式 serial 组，
存量扁平 plan 行为不变。

- **就绪谓词**（递归）：`serial` 组内前兄弟须 **done（完成级）**、`parallel` 组内前兄弟须
  **started（派发级）**——「一条串行链整体与兄弟并行」＝并行组里放一个 serial superstep；更深依赖
  靠嵌套表达（**不引入第三种 gate 语义**）。聚合态（started/done）由叶子推导、**不落盘**。
- **kind:self（组长自办环节）**：不入成员队列；`plan advance <id> --evidence <文件>` 收尾（文件存在
  + 标题段 lint 通过才置 done），把组长里程碑纳入状态机以硬性 gate 后续环节。无 kind 等价 dispatch。
- **推进面**：`advance` 收尾**全部**已终态队列条目（多 job 并行乱序到达时逐项交接决策点）→ [NEXT]
  **三栏工作清单**（可派发/在途/待验收），可派发按 id 序**一次引导一个**；`run` 等**全部在途** job
  收尾。交接处暂停，精准 recall 下一环节粗描述原文。
- **护栏**（PSEP_3 §2.4）：amend 对**已 started 的 superstep 结构冻结**（删/挪已派发或已 started
  子树内节点即拒；未派发部分放行）；并行分支同 member、宽度>3 或深度>3 → **软提示**（不阻断）；
  大任务拆解前必读 `plan-decomposition.md`（`plan new` 打印 + `read:` 标记 lint）。
- **返工流（PSEP_3 §4）**：TUI 对 delivered 交付按 `r` 驳回 → **三段式**（① 返工模式：完整／无监察
  改进／仅返工 → ② 优先级：高/中/低 → ③ 批语，均在**输入批语前**）→ 生成**新返工任务书**（继承原
  名称/简介；字段 `rework_of`/`rework_round`/`mode`/`priority`；原件 `rejected` 终态留档）。
  `plan rework <tb-id> [--mode]` 生成三档骨架：**完整** = A 综合项目改进(serial: a1 成因分析→a2 解决
  方案→a3 送审→a4 拆解派单→a5 改进实施) → BC(parallel: B 返工执行 / C 监察改进，B id 靠前优先、
  C 派发级跟随)；**无监察改进** = A→B；**仅返工** = B。
  - **硬性强制点**：模板**必留节点**缺失/乱序 → advance 拒绝；**送审 gate** = a4/c3 前置校验对应
    案卷 `verdict=A`（含 twin 终裁=A）才派单，否则拒绝；**两次送审不过** → plan `abandoned` + 任务书
    `failed` + 附两轮裁决书；`rework_round≥2` 时 a1 引导**全文展开全部前轮报告**且本轮成因分析报告须含
    「前次改进复盘」段（lint）。
  - **交付=全流程报告**：`taskbooks report <id>` 程序化拼装（任务概述/历轮摘要/本轮成因分析/解决方案/
    裁决书/监察判断/改进清单/返工产物报告）→ `data/taskbooks/<id>.report.md`；delivered 交付物即该
    报告。**循环语义**：驳回→新返工大任务→全流程→交付→再驳回……任一轮 accept 即终。
- **跨任务路由 `next`**（PSEP_3 §3）：聚合全部活动 plan 的就绪动作（blocked 决策／待验收／可派发）＋
  任务书待办，排序 = **（任务优先级 high>normal>low → 动作类别 blocked 决策>待验收>可派发 → plan/step
  序）**；输出置顶一张**任务卡片**（大任务 id/名称/优先级/模式轮次、一行简介、plan 内位置树
  〔如 `A 综合项目改进 > a1 成因分析报告`〕+ 叶子进度 k/n·在途·待验收、动作、命令模板），其余就绪项
  简表附后。**`priority` 仅影响引导顺序，不抢占/不中断在途**；无关联任务书的 plan 按 `normal`。
  **机械推送点**（§3.3）：`plan advance` 尾部、`events drain` 尾部、watch 唤醒 detail、plan
  交接报告**一律以路由卡片收尾**；TUI 概览面板只作补充视图。
