# 大任务拆解指导（PSEP_3 §2.5；必读）

本册解决一件事：把用户的一个大任务，拆成 plan YAML 里可被 orch 机械推进的
`steps[]` 树。**一个叶子 = 一份任务书 = 一次可验收交付**；拆解的产物是 YAML，
不是散文。plan 只负责**依赖调度与引导**——真正的并行早已由 per-member FIFO 队列
承担（不同成员天然并行）。

## 1. 依赖分析方法（先画链，再定序）

1. **列产物**：任务最终要交什么（报告/图/代码/模型）？逐条写下来。
2. **列前置**：每个产物依赖哪些先行产物？「B 用到 A 的产出」即 A→B。
3. **判依赖级**（决定放在哪类组里，见 §4）：
   - **完成级**（B 要读 A 的**结果正文**才能开工）→ 串行组（serial）；
   - **派发级**（B 只需知道 A **已派发**即可并行起跑，不必等 A 结果）→ 并行组
     （parallel，靠后分支在靠前分支**派发后**即解锁）。
4. **无依赖的横排、有依赖的串起来**，得到一棵树。**默认尽量平**——扁平串行是
   基线，只有确要并行才引入 superstep。

判据自查：把树交给自己「逆推」一遍——任一叶子缺它的前置，能不能开工？能则依赖
漏画。

## 2. 切分粒度（一个叶子 = 一份任务书 = 一次可验收交付）

- **一个叶子一件事**且**一次能验收**：交付面自包含（目标/路径/白名单/验收预期）——
  做不到自包含 = 切太细 + 任务书没写全；两件事塞进一个叶子 = 切太粗。
- **粒度对齐任务书**：你写不出这份环节的完整任务书 → 说明这一刀切错了。
- **过碎信号**：并行组宽度 >3、嵌套深度 >3——orch 会软提示。常见于「把同一成员
  的连续工序硬拆成并行」；此时应合并回串行链。
- **过粗信号**：一个叶子要写多份任务书、或验收标准含「以及」。继续拆。
- 叶子建议 5–15 个；>20 拆 plan。

## 3. 并行判定标准（**产物无交集才并行**）

并行分支之间必须**互不碰同一份可写产物**，否则是伪并行（合并即冲突）。判定三条：

1. **白名单目录不重叠**：两个分支的 `dirs` 不相交（尤其 `tests/`、`src/` 这类
   共享目录——两个分支都写 `tests/` 即冲突）。
2. **不改同一文件**：即便目录不同，也要确认没有共同的可写文件（含生成的部署副本、
   台账文件）。
3. **同成员不并行**：per-member FIFO 下，两个叶子若 `member` 相同，**实际串行**
   （队列排队），写进并行组只会误导。orch 会软提示——应改放串行链，或换成员。

反例：同成员两叶同组（实为排队）；正例：不同成员 + `dirs` 不交。

## 4. superstep 嵌套表达选择

- **扁平串行**（默认）：顶层 `steps[]` 即隐式 serial 组，行为等同现状，无需 mode。
- **整组并行**：`{id: G, mode: parallel, children: [...]}`——children 各自的**首个
  可派发叶子**派发后，兄弟分支即解锁。
- **「一条串行链整体与兄弟并行」**：在并行组里放一个 **serial superstep**——
  ```yaml
  steps:
  - id: a1                 # 前置（顶层串行）
  - id: G
    mode: parallel
    children:
    - id: P1               # 串行链整体（对内串行、对外与 P2 并行）
      mode: serial
      children: [{id: p1-1, member: sim}, {id: p1-2, member: figure}]
    - id: p2-1             # 兄弟分支（叶子）
      member: devops
  ```
- **深度依赖**（C 在 B started 后起步、B 又依赖 A）＝嵌套表达，**不要**造第三种
  gate 语义——只用 serial（完成级）与 parallel（派发级）两把尺子。
- 叶子 id 扁平化为点路径（`G.P1.p1-1`）；superstep 聚合状态由叶子推导，**不落盘**。

## 5. `kind: self` 使用时机（组长自办环节）

`kind: self` = **组长自己的工作**（写报告/组案卷/送审/拆解），**不入成员队列**，
故必须显式收尾：`pysci-orch plan advance <plan> --evidence <报告.md>`（文件存在 +
标题段 lint 通过才置 done）。用于把组长里程碑**纳入状态机**，硬性 gate 后续环节。

- **该用**：成因分析报告、解决方案报告、送审/组卷、返工拆解——这些是「组长产出」，
  不是派给成员的活。
- **不该用**：任何有专职成员的末端生产——一律 `kind: dispatch`（默认；无 kind 等价）。
- **evidence 规格**：写成 Markdown，含 `# 标题段`；YAML 可用
  `evidence_sections: [成因分析, 解决方案, 结论]` 声明**必含**标题段，lint 逐个校验。
- 与 dispatch 叶子一致参与依赖：self 一被引导（`self_awaiting`）即算 `started`，
  但其 `done` 须 evidence 通过——串行后继仍等它 done。

## 6. 验收点设置

- 每个**叶子的任务书**写成可证伪：交付路径具体、验收命令可重跑（测试/检查脚本）。
- 高风险/不可逆环节后设**验收叶子**：其任务书=「核对上一环节产物 + 出具结论」，
  验收不过 → 交接处 blocked，组长决策。
- 机制性 gate（如「送审通过才拆解」）由上游 self 环节的 evidence + 后续叶子依赖
  共同表达——**别靠口头约定**。
- 交付类（总结 / 报告 / 综述）环节的任务书另须含**信息架构**判据：见验收纪律
  `acceptance-discipline.md` §4「高频查阅维度集中呈现」（原则与判据形态只在该节维护）。

## 7. 交付前自查清单

- [ ] 每个叶子 = 一份可自包含写成的任务书、一次可验收交付；
- [ ] 依赖级（完成级/派发级）判对，串行/并行组选对；
- [ ] 并行分支**产物无交集**（目录不重叠、不改同一文件、不同成员）；
- [ ] 宽度 ≤3、深度 ≤3；
- [ ] 含 superstep 或叶子 ≥5 时，plan YAML 补 `read: plan-decomposition.md` 标记行
      （`plan new`/`plan amend` 会 lint 提示）。
- [ ] 复杂视觉任务按视觉检验单元拆叶；单叶检验量 ≤ 一轮舒适量：`plan-decomposition-visual.md`。

## 8. plan YAML 字段速查表（backlog 20261015-202558）

一处查全：顶层与树节点共用一套键；**有 `children` 的节点是 superstep**（须 `mode`、不被
派发），其余是**叶子**（须 `member`，除非 `kind: self`）。

| 字段 | 合法值 / 形态 | 何时用（缺省=不写） |
|---|---|---|
| `id` | 字符串 | 必填；叶子在 state 里扁平化为**点路径**（`G.P1.p1-1`） |
| `goal` | 字符串 | 顶层必填：一句话计划目标 |
| `read` | `plan-decomposition.md`（可列表） | 含 superstep 或叶子 ≥5 时**建议**加；缺则 `plan new` lint 软提示 |
| `steps` | 列表 | 顶层必填：树根 |
| `mode` | `serial` \| `parallel` | 仅 superstep：serial=完成级、parallel=派发级 |
| `children` | 列表 | 仅 superstep |
| `brief` | 字符串 | 环节一句话描述（引导时回显） |
| `member` | 成员 id（清单见 `command-surface.md`） | 叶子派发目标；`kind: self` 时省略 |
| `kind` | `dispatch`（默认）\| `self` | `self`=组长自办（不入队列，须 `--evidence` 收尾） |
| `dirs` | `[a, b]` 或 `"a,b"` | 叶子白名单；并行组互斥判据 |
| `effort` | `auto`\|`none`\|`low`\|`medium`\|`high` | 环节推理强度；CLI `--effort` 优先于它 |
| `role` | `cause`\|`solution`\|`review`\|`improvement_plan`\|`rework_plan`\|`final_report`\|`leader_reflection`\|`leader_improvement_plan` | `kind: self` 的 evidence 缺省特征段（真源 `plans.ROLE_EVIDENCE_SECTIONS`；L：leader-improvement.md） |
| `evidence_sections` | `[标题段, …]` | 显式声明 evidence 必含的 Markdown 标题段（优先于 `role` 缺省） |
| `gate_jury` | 案卷 id | 送审 gate：本叶派发前须案卷 `verdict=A` |
| `review` | `true` | 重炮审查（见 SKILL.md 派发要领） |
| `expand_history` | `true` | 返工 a1（round≥2）：展示要求与历史总览 |
