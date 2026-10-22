# PSEP 3 — plan 流程编排升级（superstep）＋ 返工驱动改进流

> Status: **已批准（2026-10-14 审阅通过，修订⑦已补记）**，待实施｜类型: Enhancement（PySci Enhancement Proposal，戏拟 PEP）
> 作者: 组长（综合用户提案两则 2026-10-14 与实查现状、两轮问答裁决）｜关联:
> PSEP 1（队列化/plan 基座）、PSEP 2（夜航，twin 授权链）｜依赖基建: plan 系统
> （plans.py）、成员队列（queues/queueworker）、jury 库、判决包、taskbooks、pysci TUI
> （均已落地）
>
> 本文件为细化方案，供用户完整审阅；批准后按 §8 分期入 devops backlog 实施，
> 实施完成后裁决并入 `orchestration/README.md`，本文件退役（PSEP 惯例）。

---

## 0. 动机（用户提案 2026-10-14，两则）

**提案一：plan 流程编排升级。** 现 plan 只支持单个大任务的串行组织。大任务的诸多环节
在工序上常互不依赖，应引入并行编排：plan 本质是串行长串，但每个单元可以是
"superstep"——将若干互不依赖的子元素并行执行；子元素本身又可以是 superstep（递归，
子元素可并行亦可串行），从而表达任意复杂流程。**组长本质仍然串行**（一次只写一份
任务书）——并行仅改变 orch 推进 plan 的行为：串行步骤须前环节完成并组长验收后才
引导书写下一环节任务书；并行步骤在 id 靠前分支派发后即引导派发 id 靠后分支，不等
兄弟分支完成。orch 自动解析 plan 结构并按最优策略引导组长推进。另：大任务拆解由此
成为技巧性工作，须在组长 harnessing 中加入专门编排指导，并确保组长上手大任务时
必定阅读。

**提案二：大任务返工直接促进迭代优化。** 用户工作方式两轨：组长会话直聊（灵活、
快、吸收差，适合小活/急活）与 pysci 控制台标准派发（规范、深度耦合、项目充分吸收，
适合不急的大活）。应强化后者的吸收能力：对返工批复（用户 reject 已交付大任务），
作为新大任务回大任务待办队列，并**硬性**强制拆分为「综合项目改进 → 返工执行」的
流程：先系统性分析质量缺陷成因、提出机制性消灭方案、送仲裁团裁决、派单实施改进，
再基于新基础设施返工；监察改进（inspector 能否自行发现该问题）以更高抽象层次并行
推进。返工最终交付为**全流程报告**；再次被驳回时循环本流程，且强制展示历次改进
报告引导反思。二元判断交仲裁团等机关，已确定该做而仅需考虑如何做的问题才由组长
敲定细节。

**用户已裁决（2026-10-14 两轮问答，全文登记 §7）**：① E6 twin 终裁补实施；② 拆解
指导硬保证取三级方案（必读文档 + `plan new` 打印 + `read:` 标记行 lint）；③ 返工
三档模式（完整／无监察改进／仅返工）在 TUI reject 时选择；④ 监察改进整体移出
「综合项目改进」，与返工执行并行（id 靠后、优先级低于返工执行）；⑤ 返工 reject 时
另选优先级（高/中/低，默认中），多大任务并行时按优先级决定推进顺序；⑥ orch 一切
引导输出必须携带大任务名称/简介/plan 内位置，机械流程强推，不能只靠概览面板；
⑦（审阅补记）**首次派发（非返工）的大任务同样有优先级**（但无返工模式）——pysci
TUI 投递大任务时，在输入任务信息串**之前**交互式引导用户选择优先级（高/中/低，
默认中）。

---

## 1. 现状基线（2026-10-14 实查）

- **plan 系统**（`src/pysci/skills/orchestration/tools/plans.py`，668 行）：扁平
  `steps[]`（id/member/brief/dirs/model/effort/review），双文件存储
  `orchestration/state/plans/<id>.plan.yaml` + `<id>.state.json`；选环逻辑两处
  `next((s for s in state["steps"] if s["status"] not in _DONE), None)`
  （plans.py:249、426）——严格串行，全仓无 superstep 概念。`run` 阻塞等单 job
  （_wait_job，上限 3h），`advance` 事件驱动（先收尾在途条目再入队下一环节即返回）。
  `_reconcile`（:69-98）按 id 对账，YAML 增删节点天然兼容。
- **并行已存在于执行层**：per-member FIFO 队列（`queues.py`，7 专职+deputy+reviewer），
  不同成员天然并行；plan 层缺的只是依赖调度与引导语义。多 plan 各自 advance 互不
  阻塞，但**无跨 plan 的优先级排序与统一引导**。
- **jury**（`tools/jury.py`）：二元 A/B 案卷、双序防位置偏好、每席
  `{choice, reason≤60字, confidence}` 留痕 `state/evals/jury/<case>.json`；**有逐席
  理由但无综合裁决书**；平票/inconclusive 过半 → `escalate="twin"`，**E6 twin 终裁
  未接线**（现为人工挂起，tally jury.py:906-955）。
- **判决包**：J 路 `assemble_case`（verdict.py:143-180）A=采纳/B=维持 组卷开庭——
  与返工送审（A=改进后状态/B=现状）形状完全同构，可直接复用。
- **taskbooks**（`tools/taskbooks.py`）：大任务=任务书，状态机
  `new→planning→active→delivered→accepted|rejected`（:40-47）；TUI `r` 驳回 =
  记批语（`rejections` 持久）→ 回 `new`（:241-248）——**这就是全部现有返工机制**，
  无模式/优先级/强制流程/报告组装。`sync_delivered`（:251-276）：所链 plan 完成
  → delivered。
- **inspector**：巡检菜单 8 项/发现/分流俱全；**「某问题能否被巡检发现」的评估器
  不存在**——由本提案的监察改进案卷环节补（不建独立组件）。
- **组长 harnessing**：orchestration skill 有 dispatch-playbook/command-surface 等
  references；**无大任务拆解指导**。

---

## 2. Part I：plan 流程编排升级（superstep）

### 2.1 数据结构

plan YAML `steps[]` 元素二型，递归：

```yaml
steps:
- id: a1                       # ① 普通 step：现有字段不变
  kind: self                   #    新增 kind: dispatch（默认，入成员队列）| self（组长环节）
  brief: 成因分析报告
- id: a2                       # ② superstep：mode + children（children 可继续嵌套）
  mode: parallel               #    serial | parallel
  children:
  - id: a2a
    mode: serial
    children: [{id: a2a-1, member: sim, brief: …}, {id: a2a-2, member: sim, brief: …}]
  - id: a2b
    member: devops             #    叶子即 step
    brief: …
```

- **叶子扁平化**：state.json 中叶子 id 用点路径（`a2.a2a.a2a-1`），`_reconcile` 按 id
  对账机制不变；superstep 聚合状态由叶子推导（started/done，不落盘）。
- **`kind: self`（组长环节）**：返工流的「成因分析/解决方案/送审/拆解」都是组长
  自己的工作（写报告、组案卷），不入成员队列。其 done 判据 =
  `plan advance --evidence <path>`（文件存在 + lint 校验模板要求的标题段），把组长
  环节纳入 plan 状态机才能硬性 gate 后续环节。这也顺带惠及普通 plan 的组长里程碑。
- **兼容存量**：无 superstep 的扁平 plan 等价于单元素串行 superstep，行为不变；
  无 `kind` 等价 `dispatch`。

### 2.2 推进语义（核心定义）

两个谓词，递归适用：

- `started(node)`：step＝已入队（self：已开始，即 evidence 未提交但已被引导）；
  superstep＝其首个可派发叶子已入队。
- `done(node)`：step＝done 且验收通过（self：evidence 通过 lint）；superstep＝全部
  叶子 done。

**叶子派发就绪条件**——对它所在每个 superstep：

- `mode: serial`：前一兄弟 `done`（完成级依赖＝「前面环节完成并组长验收后」）；
- `mode: parallel`：前一兄弟 `started`（**派发级依赖**＝「id 靠前分支派发后即可派发
  id 靠后分支」，不等兄弟完成）。

由此，「一条串行链整体与兄弟并行」＝并行组里放一个 serial superstep；「监察改进
在生产改进派单后才起步」等深度依赖＝嵌套结构表达（见 §4.2），**不引入第三种
gate 语义**。

### 2.3 advance / run 行为变化与 [NEXT] 工作清单

- `plan advance` 从「推进一个环节」变为「刷新工作清单」：收尾**全部**已终态队列条目
  （多 job 乱序到达，逐项给交接决策点），再按就绪条件计算可派发项。
- [NEXT] 三栏工作清单：**可派发项**（按 id 序，一次引导一个；写完任务书入队后即可
  连续引导下一个）／**在途项**（多 job 并行中）／**待验收项**（含 blocked 决策点，
  可能同时挂多个）。
- `plan run` 阻塞语义改为「等全部在途 job 收尾」。
- 交接决策点（`_report_handoff`）按到达序逐项报告，与并行不冲突；`review:true` 的
  子 job（spawn_review）机制不动。

### 2.4 护栏

- **amend 冻结**：`started` 的 superstep 结构冻结，amend 只能改未派发部分（lint 校验）。
- **同成员告警**：并行叶子同 member 时 lint 软提示（per-member FIFO 下实际串行）。
- **复杂度提示**：并行组宽度 >3 或深度 >3 时 [NEXT] 提示拆解过碎（软提示）。
- **返工模板必留节点**：template=rework 的 plan，必留节点缺失或乱序时 `advance`
  拒绝推进（§4.3）。

### 2.5 大任务拆解指导（harnessing 注入，三级——用户已裁决）

1. **内容**：orchestration skill 新增 `references/plan-decomposition.md`（按需档
   ≤8KB）：依赖分析方法、切分粒度（一个叶子＝一份任务书＝一次可验收交付）、并行
   判定标准（**产物无交集才并行**：白名单目录不重叠、不改同一文件）、`kind: self`
   使用时机、superstep 嵌套表达选择、验收点设置。
2. **索引**：SKILL.md 加一行「多环节大任务拆解前必读 references/plan-decomposition.md」。
3. **机械提示**：`plan new` 输出头部固定打印必读指引；plan 含 superstep 或环节 ≥5
   时 lint 校验 plan YAML 含 `read: plan-decomposition.md` 标记行（缺则提示补，
   不阻断）。

---

## 3. Part II：跨任务优先级与推进路由

### 3.1 优先级

- taskbook 新增 `priority: high|normal|low`（默认 `normal`）。设置面两处（用户
  裁决⑤⑦）：① **TUI 投递大任务时**（`n`），在输入任务信息串之前交互式引导选择
  优先级（首次派发无返工模式，仅优先级）；② TUI reject 时随返工模式一并选择。
  任务书详情可改（本期最小实现）。
- 优先级仅影响**引导顺序**，不抢占、不中断在途 job。

### 3.2 `pysci-orch next`（跨 plan 路由器，新命令）

- 聚合全部活动 plan 的就绪动作（可派发／待验收／blocked 决策）＋任务书待办，排序：
  **（任务优先级 → 动作类别〔blocked 决策 > 待验收 > 可派发〕→ plan/step 序）**。
  〔默认决策，见 §7〕输出置顶的一张**任务卡片**，其余就绪项以简表附后。
- **任务卡片格式**（组长上下文切换的信息刚需——用户裁决⑥）：

  ```
  ┌─【大任务 <tb-id>：<名称>】优先级=<high|normal|low> 模式=<完整|…> 第 N 轮返工
  │ 简介：<一行摘要>
  │ 位置：plan <plan-id> @ A 综合项目改进 > A1 生产质量改进-策划 > 成因分析报告
  │       （叶子进度 3/9，在途 1，待验收 1）
  │ 动作：书写本环节任务书并派发（或：验收 X / blocked 决策 X）
  └─ 命令模板：pysci-orch plan advance <plan-id> --task …
  ```

### 3.3 机械推送点（不能只靠面板）

- `plan advance` 尾部、`events drain` 尾部、watch 唤醒（JOB-DONE/NEEDS-LELE/
  PENDING-WORK 等）detail、`plan run` 交接报告——**一律以路由卡片收尾**；
- TUI 概览面板只作补充视图（用户裁决⑥：信息推送必须进机械流程）。

---

## 4. Part III：返工驱动改进流

### 4.1 触发：TUI reject 对话（三段式）

用户在 pysci 任务书面板对 delivered 大报告按 `r` 驳回时，**输入批语前**依次选：

1. **模式**（三档）：
   - **完整**——综合项目改进＋监察改进＋返工执行（§4.2 骨架全量）；
   - **无监察改进**——省略监察改进（A→B）；
   - **仅返工**——不含任何改进环节，只余返工执行（B），快速返工；
2. **优先级**（高/中/低，默认中）；
3. **批语**（完整批复要求，现状 `rejections` 持久机制不变）。

随后生成**新任务书**入大任务待办队列（用户定调：返工=新大任务），字段：
`rework_of: <原tb-id>`、`rework_round: N`（=原任务书历史轮次+1）、`mode`、
`priority`、继承原任务书名称/简介；原任务书终态（rejected）留档。

### 4.2 plan 模板骨架（三档）

```
【完整】
├─ A 综合项目改进 (serial)                 ← 生产质量改进全链
│   ├─ a1 成因分析报告        (self)
│   ├─ a2 解决方案报告        (self)
│   ├─ a3 送审 jury           (self, gate: verdict=A 才解锁 a4)
│   ├─ a4 拆解改进派单        (self)
│   └─ a5 改进实施            (dispatch, 跟进成员至完成验收)
├─ B 返工执行 (serial, id 靠前)            ← A 全部 done 后 started
│   ├─ b1 重拆解返工          (self；b2+ 占位待 amend 填入具体生产叶子)
│   ├─ b2…bn 返工生产环节     (dispatch，允许基于旧成果修改)
│   └─ b_last 全流程报告组装  (self)
└─ C 监察改进 (serial, id 靠后)            ← 与 B 并行；B started 后可推进
    ├─ c1 可发现性案卷送审    (self, jury 二元：现有机制能否高效自行发现)
    ├─ c2 [若判需改进] 改进方案 (self)
    ├─ c3 [若需改进] 拆解派单  (self)
    └─ c4 [若需改进] 改进实施  (dispatch)

【无监察改进】= A → B（删 C）
【仅返工】    = B（删 A、C）
```

- **监察改进移出后的位置（用户裁决④）**：B/C 并行组内派发级依赖（C 在 B started
  后可推进）＋ id 序＋路由排序（B 项先于 C 项）共同保证「返工执行优先、监察改进
  尽力跟随」；返工执行**不等**监察改进。
- **二元判断交仲裁、如何做归组长**：监察改进的可发现性判断送 jury；一旦判「需改进」，
  改进方案**不再二次送审**（属「如何做」，组长敲定），直接拆解派单——与生产质量
  改进的方案须送审（那是「改不改」的二元判断：改进后状态 vs 现状）严格区分。
- **b1 后的 amend 填充**：模板生成时 B 的返工叶子为占位；b1（重拆解）完成后组长
  `plan amend` 补入具体生产叶子（lint：B 至少一个 dispatch 叶子才允许 b_last 前进）。
  B 未 started 前 amend 合法（§2.4 冻结规则）。
- **交付语义**：`sync_delivered` 不变＝**全 plan 完成**才 delivered；完整模式下
  交付等待 C 完成（全流程报告含监察改进结果），但 B 的执行与验收推进不等 C。
  〔默认决策，见 §7〕

### 4.3 硬性强制点（汇总，全部机械落地）

| # | 强制内容 | 落地机制 |
|---|---|---|
| 1 | 返工必走模板、必留节点在序 | reject 生成任务书 → `plan new --template rework`；lint 校验必留节点，advance 拒绝乱序/缺节点推进 |
| 2 | 改进方案送审通过才拆解派单 | a4 派发前置校验：`state/evals/jury/` 对应案卷 `verdict=A`（或 twin 终裁=A），否则拒绝 |
| 3 | 两次送审不过即中止 | 第 2 次 verdict=B → plan 标 abandoned，任务书置新终态 **`failed`**，TUI 标「执行失败，待用户补充指引」，附两轮裁决书（taskbooks.py:40-47 ALLOWED 加 `active→failed`） |
| 4 | 历次改进强制展示 | rework_round≥2 时，a1 引导输出（[NEXT]）自动展开**全部前轮**改进报告与上轮全流程报告（全文，不只给路径）；lint：本轮成因分析报告必须含「前次改进复盘」标题段（机械验收 grep） |
| 5 | 交付=全流程报告 | `pysci-orch taskbooks report <id>` 组装器：程序化拼装（任务概述/历轮摘要/本轮成因分析/解决方案/裁决书/监察判断/改进清单/返工产物报告）→ `data/taskbooks/<id>.report.md`；delivered 交付物即该报告 |
| 6 | 监察改进的自我评估 | c1 案卷（证据：巡检菜单覆盖、findings 历史、问题特征）送 jury 二元裁决——补上「可发现性评估」设计缺口，不建独立组件 |
| 7 | 咨询通道 | a1 成因分析允许向涉事组员发问询型小任务（走既有队列，`kind` 标问询）与 `consult` deputy |

### 4.4 jury 增强：裁决书（verdict_summary）＋ E6 twin 终裁

- **裁决书（用户提案二附带要求）**：jury 收尾生成综合裁决书＝胜方＋多数理由汇总
  （逐席 reason 聚合）＋各席 confidence＋（若有）twin 终裁及理由；落
  `state/evals/jury/<case>.json`、`preference-journal.jsonl`、`pysci decisions`、
  TUI 显示。理由从「逐席 60 字散票」升格为可复盘文书，覆盖判决包 J 路与返工送审。
- **E6 twin 终裁（用户裁决①，补实施）**：tally 平票/inconclusive 过半 →
  `escalate=twin` → 自动调用 twin 终裁通道（复用 taste 机制：PREFERENCE.md 宪章
  注入、单跳冷启动、二选一＋理由＋置信度）→ 落 `final_verdict`＋`decided_by:
  twin`＋终裁理由，入 journal/decisions（REFACTOR §4 高风险授权留痕）。REFACTOR
  §5「twin 终裁不给平局」由此落地，一切判决链收敛到确定选项。适用范围：全部
  jury 使用点（判决包 J 路＋返工送审＋监察可发现性）。twin 不可达
  （taste_enabled=false）→ 保守回退：视为「未通过/挂起」并出声，不静默。
  返工流中 twin 终裁即最终 pass/fail。

### 4.5 循环语义

驳回 →（选模式/优先级）→ 新返工大任务 → 全流程 → 交付 → 再驳回…… 无限循环；
每轮 `rework_round+1`，§4.3 #4 保证每轮都带着全部历史改进报告反思「此前改进为何
效果不佳、方向哪里不对」。任一轮 accept 即终。

---

## 5. 触点清单（实施地图）

| 组件 | 改动 |
|---|---|
| `tools/plans.py` | superstep schema 解析/递归就绪计算/多 job 收尾/`kind: self` evidence 收尾/[NEXT] 工作清单/amend 冻结 lint/模板必留节点 lint（选环逻辑 :249/:426 重构） |
| `tools/orch.py` | `plan next`（或 `orch next`）路由命令；advance 尾部附路由卡片 |
| `tools/taskbooks.py` | `priority`/`mode`/`rework_of`/`rework_round` 字段、`failed` 状态迁移、`report` 组装命令 |
| `tools/jury.py` | tally 后 E6 twin 终裁接线；`verdict_summary` 裁决书生成 |
| `tools/taste.py` | twin 终裁通道复用（可能仅函数级复用，不新建通道） |
| `tools/verdict.py` | J 路 escalate=twin 顺带生效（无需新逻辑，验收覆盖） |
| `tools/queueworker.py` / `watch.py` | JOB-DONE/唤醒 detail 附任务卡片上下文 |
| `tui/app.py` | reject 对话三段式（模式→优先级→批语） |
| `tui/modules.py` | 任务书卡片（轮次/模式/优先级）、superstep 进度树、裁决书摘要、改进清单 |
| skill 文档 | `references/plan-decomposition.md`（新）＋SKILL.md 索引行＋command-surface 补 `next`/`report` |
| 测试 | 就绪谓词递归（serial/parallel/嵌套/self）、多 job 乱序收尾、amend 冻结、模板 lint、reject 三段式落字段、两次送审中止、twin 终裁回退、裁决书聚合、报告组装、路由排序 |

---

## 6. 实施分期

- **M-A superstep 引擎**：§2 全部（schema/语义/收尾/lint/拆解指导）。独立可用。
- **M-B 跨任务路由**：§3 全部（priority/next/卡片推送）。
- **M-C jury 增强**：§4.4（裁决书＋E6 twin 终裁）。独立可用，M-D 前置。
- **M-D 返工流**：§4 其余（reject 三段式/模板/gate/failed/报告组装/TUI/循环）。

依赖：M-A ← M-D；M-C ← M-D；M-B 与其余松耦合。M-A/M-B/M-C 可并行派发
（不同成员/工序无交集）。

## 7. 裁决记录与默认决策

**用户裁决（2026-10-14）**：见 §0 末六条（①–⑥）。

**组长默认决策（审阅可改）**：

1. **交付=全 plan 完成**：完整模式下 delivered 等 C（监察改进）完成——全流程报告
   含其结果；B 的执行推进不等 C。
2. **跨任务动作排序**：（任务优先级 → 动作类别〔blocked>待验收>可派发〕→ 序）——
   优先级决定任务间顺序为用户裁决，动作类别权重为组长默认。
3. **监察改进方案不二次送审**：可发现性判断是二元判断已交 jury；改进内容属「如何做」
   归组长（依用户「二元判断交仲裁、如何做归组长」总原则推得）。
4. **twin 不可达回退**：终裁链断时视为未通过/挂起并出声（保守），不静默放行。
5. **返工叶子占位 amend 填充**（§4.2）：模板 B 段留占位，b1 后补齐。

## 8. 验收判据（机械可核验）

- 嵌套 plan（serial⊃parallel⊃serial）推进顺序符合 §2.2 谓词（测试用例全绿）；
  并行分支「派发后即引导兄弟」实测成立（dispatch 记录时间序）；
- `plan new` 多环节输出含必读指引；缺 `read:` 标记行时 lint 提示；
- reject 对话产生 `rework_of/rework_round/mode/priority` 齐全的新任务书；
- 完整模式 plan 骨架生成后，删除 C 或 A 任一必留节点 → advance 拒绝推进；
- a4 在 verdict≠A 时拒绝派单；第 2 次 verdict=B → 任务书 `failed` 且 TUI 显示；
- rework_round=2 的 a1 引导输出含第 1 轮报告全文；缺「前次改进复盘」段的报告
  机械验收 FAIL；
- `taskbooks report` 产出含全部规定段落的大报告；delivered 交付物为该报告；
- jury 平票案卷经 twin 终裁收敛到确定选项并留痕 decisions；twin 关闭时保守回退出声；
- `orch next` 在两任务（high/normal）就绪时先推 high，卡片含名称/简介/位置；
- 全仓测试全绿。

## 9. 残余风险与非目标

- **组长注意力负担**：并行分支多时工作清单变长、上下文切换成本升——宽度/深度软
  提示（§2.4）＋卡片信息完整是补偿；接受此残余。
- **返工流程刚性成本**：三档模式（用户裁决③）已对冲「小缺陷大流程」；不再分级细化。
- **`kind: self` 的 done 依赖组长如实提交 evidence**：lint 校验标题段是机械下限，
  内容质量由裁决书/验收环节兜底。
- **非目标**：不引 LangGraph 等执行引擎（并行早已由成员队列承担，plan 层只缺依赖
  调度与引导语义，自研胶水即够——与 leader-only.md §2「默认社区、自研只写胶水」
  相符）；不建独立「可发现性评估器」组件；不改队列层/成员 FIFO；inspector 本体
  不动（改进经 backlog 派单实施）。
