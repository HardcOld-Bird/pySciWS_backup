# REFACTOR：评价驱动的自改进体系（定稿候选 v1.0）

**状态**：定稿候选——E0–E7 成功判据已写实（§12），待用户终审。终审通过 →
裁决登记 → 按 E0 起入 backlog 实施；全部完成后融入 README.md 并删除本文。
v1.0 变更：§12 各阶段成功判据由占位补为可机械核验文本（触发命令 + 预期
落盘物/输出）。历史：v0.6 收敛轮（jury-brief 直通规则确认、inspector 设计
审查 rubric、mirror status 全景 + 注册纪律、journal 机械分流）；v0.5 判决包
连带决策（J/M/J+M/不选=override；实施前实测删除）；交付后核验零新裁决点、
只实测不再仲裁、失败自动立案纠正；基线捕获；tester 定名 **bencher**；
inspector 双队列迟滞节流；twin 免阈值直接上线；distill 延期；CLI 人体工学
范本（Qoder CLI/commitizen）。

**范围（用户裁决）**：注意力完全在「自我改进」功能；生产管道与星形拓扑不变。

**目标优先级（用户裁决）**：① 客观上的优质；② 符合用户审美（客观打平时的
tiebreaker，或客观通道回退时的终裁；冲突时服从①）。

---

## 0. 设计纪律

1. **不新增抽象层级，允许新增抽象要素**。禁止在 orchestration 之上再套「管理
   orchestration 的系统」。
2. **改进管道自覆盖（用户裁决）**：管道一切组件（orch、guards、inspector/
   bencher 工具链、jury 库、mirror、twin pod harness、本纪律的执行机制自身）
   都在被评估与被改进范围内，走同一条管道。**防自指豁口**：inspector 的 CLI
   不在其自身巡检范围（工具保持精简，维护与质量责任在用户与组长）。
3. **硬代码优先（用户裁决）**：流程贯彻由代码强制，不依赖「按需/有空」类软
   条款（教训：consult 软选项几乎从不被使用；Tier A 统计必须被流程硬性依赖，
   §2.4）。
4. **高风险授权操作必留痕（用户裁决）**：一切代理裁决（twin 代用户、twin 终裁、
   jury 裁决、override）落盘且 mirror 可事后详查——「先决策，后问责」。
5. **决策形式统一二选一（用户裁决）**：实测、仲裁、twin 三类判决通道输出一律
   二元；确需多选时拆为多个二元测试依次处理。**twin 终裁不给平局**——一切
   判决链最终收敛到确定选项。
6. **元风险控制**：组件可单个重建、可独立停用（fail-open）；评价绝不阻塞生产
   管道；退役机制见 §14。
7. **两目标解耦**：客观机构与审美机构永不互投对方的票；twin 仅在客观通道
   回退时具强制性（§5.5）。
8. **Goodhart 防护 + 证据分层**：提案者/判决者/执行者三权分立；机械可查的不
   交模型，台账可统计的不跑实验。
9. **一成员一技能一 CLI，共享逻辑做库不做新 CLI（用户裁决）**：组长决策流
   内嵌 orch、inspector 决策流内嵌 pysci-inspect；共享逻辑（jury 调用、twin/
   deputy 咨询、backlog 入队与 devops 唤醒、判决包解析）做内部库模块，由各
   工具封装为子命令。**CLI 人体工学范本（用户建议 v0.5）**：Qoder CLI 与
   commitizen——面向人类使用、交互美观清晰；实施阶段以此为参照。

## 1. 总览

**成员花名册（新增 3）**：7 专职 + deputy + reviewer + devops（既有）；
+ **inspector**（巡检找茬）+ **bencher**（harness 实测）+ **twin**（审美孪生/
用户代理）。

**两种测试的分野（用户裁决）**：普通测试（src 代码库，pytest，根 tests/）由
devops 管理编辑；**Agent harnessing 测试**（harness 设计的特殊套件）由 bencher
管理编辑——命名同源 benchmark，与 tests/ 清晰区隔（用户定名 v0.5）。

```
用户 ──任务──▶ 组长 ──orch plan──▶ 组员 pods ──交付──▶ 用户【最终验收，不代理】
用户 ──闲时──▶ pysci-mirror：盲审 twin 决策（=训练）／审阅全部留痕／口述偏好

改进管道（覆盖自身）：
 提案三源：组员 infra_suggestion ／ inspector 找茬 ／ Tier A 阈值自动立案（§2.4）
 ├─ 组员提案 → 组长强制选择：判决包 {J,M,J+M} approve ／ reject=override（§5.1）
 ├─ inspector 发现 → 自行 {J,M,J+M}（不经组长；机械证据类自动晋升；无 override）
 └─ 偏差报告（mirror 盲审）→ 组长绿色通道（§5.4）
 判决机构：jury 仲裁（轻、阻塞≈1min、实施前）／ bencher 实测（重、异步、
 实施后 A/B）／ twin 终裁（回退必选、不给平局）
 → 判决自动 approve/reject → devops 实施 → 核验（§5.6）→ 销账或自动立案纠正
```

**组长职责重新聚焦**：直接决策 = 生产调度 + 改进流程中的「撰写仲裁文本块/
任务书 + 选择判决包 + 执行判决 + 撰写回复」。基建改进的**判断**由机构完成。

### 1.1 模型策略（用户裁决）

- 新增 `minimum` 档（成本最低模型，驱动 harness 压测；型号留待专门环节，
  registry.models 改动属护栏级——按 §9.4 由 twin 代理裁决）。
- 全项目降档：flash 主力，max 仅顶层按需、无组员常驻；deputy/reviewer 降档
  随 E0/E1 执行（几乎无台账存量，用户确认 v0.4）。

## 2. 客观层 Tier A：失败标签、统计与硬性接入

- 词表 `state/failure-tags.json`（受控，devops 增补附定义，护栏级）；组员
  `<blocked>` 可选 `tags="a,b"`（delivery-gate 校验词表，缺省不阻断）；
  needs_leader 裁决时组长补标/改标。
- 聚合：`orch stats` by_failure 段；**同一标签反复出现 = 深层结构问题的客观
  信号**。

### 2.4 硬性依赖（用户裁决：不指望组长「有空」去看）

- **阈值自动立案**：orch 在 approve/status 等入口机械重算滑窗统计（参数已定
  v0.5：同一 (tag, member) 于 14 天或 20 跳内 ≥3 次）；越限 → 自动注册基建
  提案（kind=defect，证据 = 台账行与计数）→ 进 §5.2 inspector 流程。检查、
  计数、立案全代码完成，人只在强制选择点出现。
- 此类提案带 `source=tier-a`；其 success_criteria 天然可测（销账时对应计数
  应回落）。

## 3. 客观层 Tier B：harness 测试套件（bencher 所有）

- **套件** `state/evals/harness/`（bencher 完全写权）：历史真实任务提炼，首批
  3~5 个；每案 = 任务书 + 白名单 + 机械成功判据（确定性断言）+ 预期指标
  基线。**输出统一二元判决**（§0.5）；模棱两可 → §5.5 twin 终裁回退。
- **执行**：dispatch 特化封装（全新会话、minimum 档、不附审批回复），结果入
  `state/ledger/evals.jsonl`。
- **副本 A/B 与基线捕获（v0.5 细化）**：
  - **基线捕获**（判决包选 M 时由 bencher 立即执行）：`git worktree` @ 审批
    时 HEAD + 对 pods 内 **gitignored harness 文件**（AGENTS.md、自建 rules/
    skills 等，按各 pod .gitignore 细则）做物理归档 → 存
    `state/evals/baselines/<case>/`。git 已保住 tracked 面，物理归档只补
    gitignored 面；
  - **实施后副本**：devops 合并后，bencher 以合并 commit 另建新副本（同法）；
  - **A/B 判决**：双副本各跑 success_criteria 关联的套件案，对比台账指标，
    二元判决「改善/未改善」；用后即弃、无损回退（worktree 纪律有先例）。
  - 副本机制同样适用于纯比较型实验（charter 变体等，可覆盖全体成员含组长
    与 bencher/inspector 自身——会话按 cwd 项目键分储，副本天然隔离）。
    实现细节（hooks 接线、清理纪律）E4 实测商榷。
- **多选拆分**：>2 方案比较拆为依序二元对局（§0.5）。

## 4. 客观层 Tier C：仲裁团（家族外、直连、阻塞）

- **定位**：成对比较；**实施前**的常态判决模式（实测昂贵缓慢且实施前不可用，
  §5.1）。
- **渠道（用户裁决）**：SiliconFlow（国内直连、模型全、用户有账号与 Key、
  单轮冷咨询按量开销极小）；实施时据平台模型自由选型；**硬性规则：不含
  Qwen 系**（排除同族自恋偏差）。
- **架构**：不经 Qoder；共享库直连 OpenAI 兼容接口并发调用，orch /
  pysci-inspect 各自封装子命令。Key 环境变量注入（永不入库；doctor 泄漏
  哨兵）。
- **耗时与模式**：并行单轮 ≈1 分钟，**阻塞实现**；超时（5 min）→重试一次→
  仍失败该案「入队待重裁」留痕（渠道单点不可达时基建决策降级入队，不阻塞
  生产）。
- **冻结简报** `state/evals/jury-brief.md`（护栏级）。**裁决规则（用户确认
  v0.6）**：「仅涉提议者自身专有基础设施的提案，除非有明确危害，默认
  approve」——组长主观直通豁免取消，移入简报成为统一裁决规则；代价是慢一点
  贵一点，与整体重判决风格一致（用户原话）。快速通道保留（J 仅 ≈1 min）且
  统一留痕。
- **材料组装责任在撰写者侧（用户裁决 v0.4：jury 不自行读代码）**：法院模式；
  工具访问会引入 agent loop（慢/贵/违反非 Qwen 规则），且各裁判所见不同 →
  破坏双序一致性与材料哈希冻结。补偿：CLI 组装案卷时**自动附加确定性材料**
  （git diff、预算读数、相关台账行、涉事 charter 段落）。**回访触发条件**：
  裁决反复被事后证据推翻（材料不足实锤）时重议。
- **解析（代码强制）**：结构化返回（选 A/B + 理由 + 置信度）；双序各跑一次，
  单裁判不一致记 inconclusive（不计票）；少数服从多数；**平票/inconclusive
  过半 = 模棱两可 → §5.5 twin 终裁**。裁决落 `state/evals/jury/<case>.json`
  （各裁判输出、usage、成本、材料哈希）。

## 5. 决策流（v0.5：判决包连带决策）

### 5.1 组长处理组员基建提案

**结构化分类**：`<infra_suggestion>` 必须标 `kind="defect"`（实际阻塞+证据）或
`kind="usability"`（改进/重构）；delivery-gate 校验。

**分流规则**：
- **缺陷类 + 同条交付含 `<blocked>`**：当条工作本身即 in-vivo 实测 →
  **默认 approve**，orch 直接要求组长写任务书派 devops。规程建议连带选 M
  （复跑原失败案 = 最自然的核验）。组长拒绝 = **override**（留痕、计入
  override 率）。
- **其余（缺陷类+result、难用类）**：组长强制选择**判决包**（v0.5，实施前
  实测不可行——改动文本未落实——原 B 选项删除）：
  - **J**：现在送仲裁团（阻塞 ≈1 min）→ 裁决自动 approve/reject，组长仅写
    回复；
  - **M**：approve 并实施 + 基线捕获 + 实施后 A/B 实测核验（§3/§5.6）；
  - **J+M**：仲裁把关实施、实测核验效果；**实测结果覆盖仲裁结果**（未改善
    → 自动立案纠正，§5.6）；
  - **都不选 = override**（approve 或 reject 均是），须附详细说明，mirror
    留痕，计入 override 率。
  - 经 J 或 M 判决返回的 reject/纠正**不算 override**（有判决依据；v0.4
    调和解释经用户确认 v0.5）。
- **可选咨询环节**（任意点、可跳过、由组长判断）：高风险事项 → twin 咨询
  （仅建议、无强制；阻塞单跳；mirror 留痕）；深究技术问题 → deputy 咨询
  （`orch consult`）。经共享库同时提供组长与 inspector（§0.9）。

### 5.2 inspector 处理自身发现（不经组长）

- **机械证据类**（harness 测试失败、doctor ✗、预算超限、实锤复现）→ 自动
  晋升 backlog（source=inspector，附证据），无需判决。
- **判断类** → 强制选择判决包 {J, M, J+M}（同 §5.1 语义）；approve →
  inspector 撰写 devops 任务书并派发（共享库：backlog 入队 + drain 唤醒）。
  **全程不经组长**（devops commit 后组长经 git 历史可查）；**inspector 不
  允许 override**（其提案被拒不阻塞任何工作）。
- **双队列迟滞节流（用户裁决 v0.5）**：监视 devops backlog pending 与
  bencher 队列长度——任一 > 积压上限（初始建议：backlog 12 / bench 队列 6）
  → inspector **暂停产出**（巡检只读部分可继续，不生成新请求/提案）；两队列
  均回落至**恢复阈值**（低于积压上限，初始建议 8 / 4）→ 恢复。**恢复阈值
  严格低于暂停阈值 = 延迟恢复（迟滞），防振荡**。

### 5.3 Tier A 阈值提案（§2.4）→ 进 §5.2 流程。

### 5.4 偏差报告队列（mirror 盲审 → 组长处理）

用户盲审 twin 决策，不一致时产出**偏差报告**（用户选择 + 不一致原因 + 是否
纠正现有结果〔可附实施要求；无要求 = 原样恢复〕）入队列；`mirror journal` 的
**行动指令**（§10）入同一队列且同「强制派发」待遇；组长择机在 orch 处理：
- **「twin 不对」→ twin harness 改进，绿色通道默认通过**（用户反馈即仲裁
  结果，已授权实施）。组长二选一：写任务书派 devops ／ override（认为无需
  改动；留痕、计 override 率）。**允许批量处理多个不一致案**（类比 devops
  组包），批量不算 override。
- **「纠正现有结果」→ 强制写任务书派 devops，不得 override**。
- 两类原则上分别派发（主题不同）。

### 5.5 模棱两可回退链

bencher 实测无显著指导意义、或 jury 平票/inconclusive 过半 → **自动回退
twin 终裁**（结构化输入自动组装 prompt；不给平局）→ 结果必为选项之一。
twin 仅在回退时具强制性。所有 twin 裁决 mirror 留痕。

### 5.6 交付后核验（零新裁决点，v0.5 重设计）

- **核验方式在审批时一次定死，交付后无人再裁决**：机械判据（pytest/doctor/
  checks）由既有交付协议自动执行，永远在场、免费；**重型 A/B 实测仅当审批
  选了 M** 才由 bencher 执行（基线捕获已在审批时完成，§3）。
- **防重复规则（用户裁决 v0.5）**：同一提案实施前至多一次 jury；实施后
  **只实测、不再仲裁**——仲裁裁「该不该做」、实测裁「做了对不对」，不构成
  重复判决。
- **判决后果**：改善 → orch 销账（success_criteria 核验通过）；**未改善 →
  自动生成新的纠正提案**（新证据 = A/B 数据、新案卷，进 §5.2 管道；纠正
  方式〔修复 vs 回滚〕作为新案可经 J）。J+M 情形下实测覆盖仲裁：仲裁
  approve 但实测未改善，同样自动立案纠正。
- 核验记录自动入决策日志。

## 6. 反馈回路总闸：可证伪成功判据

backlog 条目必填 `success_criteria`（X 指标应如何变化／判决通道应如何裁决／
用户确认 Z）。三权分立：提案者写、判决案卷审定、devops 只执行。判据同时是
M 实测的靶点（§5.6）。E0 生效。

## 7. inspector 组员（专职巡检找茬）

**设计理念（用户裁决）**：「**与用户视角相同，但知识不同**」——覆盖用户视角
的观察能力（查阅所有会话 transcript、项目文件、各类产物），像专业开发者一样
抓虫：主动检查矛盾点、与设计意图不符之处，产出结构化基建提案（走 §5.2，
不经组长）。

- pod `orchestration/pods/inspector/` + charter + 技能 + CLI `pysci-inspect`
  （封装：发现提交、判决包流、twin/deputy 咨询、backlog 入队与 devops 唤醒）。
  全项目**只读**（代码树零写权限；产出只落 state/inspector/ 与 state/evals/）。
  flash/minimum 档。
- **全局开关（用户裁决）**：`state/inspector.enabled` 默认开启，可随时完全
  关闭（成本控制）；orch 拉起逻辑尊重开关。
- **生命周期（代码保证）**：`orch inspector-up` 幂等拉起（分离进程+lock），
  内嵌进 orch 常用命令入口；单次激活预算硬上限；**让路机制**：生产活跃时
  只做只读巡检。
- **工作循环（queue-first，空则巡检，不退出）**：① 队列（Tier A 阈值案文本
  块撰写、组长指派核查）；② 巡检菜单随机抽取（doctor 全域、随机 pod
  transcript/自维护层抽查、预算审计、台账成本曲线异常、随机代码走查、**改进
  管道自身组件**〔§0.2 自覆盖〕、设计意图一致性核对〔README/charter vs 实际
  行为〕、**设计审查 rubric**〔v0.6，用户提供〕）；③ 发现落 findings.jsonl
  → §5.2；④ 双队列迟滞节流（§5.2）。
- **设计审查 rubric** `pods/inspector/rubrics/design-review.md`（只读层、
  护栏级、修订权归用户——参照 reviewer rubrics 先例；用户指引 v0.6）：
  ① **社区优先**：调研社区/GitHub 上有影响力且维护积极的工具，评估能否完全
  或部分替代自研技能模块——能则优先替代（维护负担转移给社区，项目整体负担
  减轻）；② **反冗余**：避免能力重叠（社区依赖与自研代码实现同类功能），
  积极提议优化掉冗余部分；③ **LLM 直觉友好**：模块设计是否让 LLM 认知负担
  轻、无需深入内部实现即可高效使用——别扭/反直觉处积极提议重构。此类发现
  属判断类，走 §5.2 判决包。
- **豁口**：pysci-inspect 自身不在巡检范围（§0.2）。

## 8. bencher 组员（专职 harness 实测；用户定名 v0.5）

**与 inspector 拆分理由（用户裁决）**：harnessing 测试与巡检是不相干的工作，
上下文不必互相污染。

- **工作形态类 devops**：FIFO 队列（`state/bencher/queue.json`），非空拉起、
  消化完销毁（不常驻）；完成项经 watch/status **择机查看型**通知提请人
  （组长或 inspector；不即时打断，用户裁决 v0.4）。
- **权限**：harness 测试套件与基线区（`state/evals/**`）**完全写权**；外部
  项目代码与自身 harness 只读层零写权。关注点 = **harnessing 设计**（与
  devops 的 src 测试错开，§1）。
- **职责**：§3 全部——套件维护与扩充、基线捕获、副本 A/B、二元判决、
  模棱两可回退 twin（§5.5）；交付可附 `<infra_suggestion>`（走 §5.1）。

## 9. twin 组员（审美孪生 → 回退终裁者 + 用户代理）

- pod `orchestration/pods/twin/`；PREFERENCE.md 置 pod 内（readonly_extra
  只读，护栏级）。**无记忆纪律**：每次判断冷启动新会话；「记忆」只有
  PREFERENCE.md 与决策日志。**无成员 CLI**（判断经 orch taste/共享库；训练
  审阅经 pysci-mirror）。
- **三重角色**：① 可选咨询（仅建议）；② 客观通道回退终裁（强制、不给
  平局）；③ 用户代理裁决（§9.4）。
- **判断执行**：`orch taste <case>` 冷启动单跳（flash 起步），注入
  PREFERENCE.md + 案件，输出「选择+理由+置信度」，落 state/evals/taste/。
  回测 `pysci-mirror backtest`：一致率——**v0.5 起为监测指标而非上线门**
  （趋势异常时触发训练冲刺，§10）。

### 9.4 代理裁决（先决策，后问责）

- **上线（用户裁决 v0.5）**：充分吸收现有语料（README §13 存量 + 用户
  prompt 总集 + 日志）产出 PREFERENCE.md v1、经用户确认后**直接上线**——
  不设数值门槛，后续问题在实际运行中经盲审纠正完善（§10），不等待上线。
- **授权范围**：上线后原「需用户裁决」事项（护栏级审批、升级仲裁、Phase
  验收等）由 twin 即时代理，项目运转永不因用户缺席阻塞。一切代理裁决
  mirror 留痕；用户异议事后追溯——**责任在用户（监督缺席），不在系统与
  agent**；修正方式是训练，不是回滚。
- PREFERENCE.md 改动授权链：用户盲审纠正 → 偏差报告绿色通道（§5.4）→
  devops 实施——twin 永不自改权重。
- twin 永不投客观质量票（§0.7）。

## 10. 用户控制台 `pysci-mirror`（CLI；自然语言 + 编号菜单，永不要求打路径）

- **`mirror status`（全景概览，v0.6）= 用户侧唯一关注入口**（类比 `orch
  status` 之于组长）。**注册纪律（硬要求）**：未来任何新增的「需用户关注
  事项类型」必须注册进本概览。首批概览项：待盲审案件数、待确认宪章 diff、
  一致率/override 率趋势告警、待决事项（minimum 选型等）、inspector 开关与
  双队列水位、待用户验收的最终交付提醒。
- **`mirror review`（盲审 = 训练，反转设计）**：twin 决策逐案呈现——先隐去
  twin 答案，用户自选；一致 → 通过；不一致 → 追问：① 原因 ② 是否纠正现有
  结果（可附实施要求）。打包为**偏差报告**入 §5.4 队列。审核即最高质量训练
  采样。
- **`mirror journal`（写入时机械分流，v0.6）**：口述内容后菜单二选（可复选）：
  ① **偏好陈述** → 入决策日志作训练语料（twin 判断时可注入参考；经偏差
  报告/未来 distill 逐步归纳进宪章）——不进处理队列（不要求行动）；
  ② **行动指令** → 入**强制处理队列**（与偏差报告「纠正现有结果」同类）：
  组长必须写任务书派 devops，**不得 override**（用户明确要求即授权，§5.4
  同一法理）；指令含糊/涉护栏疑虑时组长经 mirror 回问（回问留痕）。
- `mirror overrides` / `stats`：override 详情视图与比率、判决通道用量、twin
  代理裁决总览、**一致率趋势**（跌破态势 → 提示训练冲刺）。
- **distill（延期项，v0.5 简化）**：批量蒸馏（日志→宪章原则归纳）的职能已
  大部分被偏差报告绿色通道替代（增量式纠正）；E6 只做一次性 v1 宪章产出
  （存量语料蒸馏 + 用户确认），日常更新走绿色通道；若运行中显现批量归纳
  需求再实现 `mirror distill`。

## 11. 决策日志（原料层，E0 启动）

`state/preference-journal.jsonl`：`{ts, situation, options, choice, reason,
source, kind}`。来源：① README §13 存量回灌；② **用户 prompt 持久语料**
（实施时用户提供）；③ mirror 盲审/口述（自动）；④ override（自动）；
⑤ 核验记录（自动）；⑥ twin 代理裁决与终裁（自动）。

## 12. 实施阶段与成功判据（v1.0 写实；每判据 = 触发方式 + 预期可核验结果）

### E0 —— 纪律与原料层（依赖：无）

内容：§6 判据纪律；§11 日志启动（回灌 README §13 存量 + 用户 prompt 语料）；
提案 kind 结构化（delivery-gate 校验）；approve 校验雏形（override 留痕）；
deputy/reviewer 降档 flash。

成功判据：
1. `grep kind orchestration/state/backlog.json` 显示新登记提案均含
   `kind∈{defect,usability}` 与非空 `success_criteria`；构造一条缺 kind 的
   `<infra_suggestion>` 交付 → delivery-gate exit 2 并注入改正指令。
2. 构造一次拒绝：`orch reject <id>`（无判决依据）→ 拒绝被代码要求附
   `--override "<理由>"`，提交后 `tail -1 orchestration/state/preference-journal.jsonl`
   出现 `source=override` 行且理由完整。
3. `wc -l orchestration/state/preference-journal.jsonl` ≥ README §13 裁决条目
   数（存量回灌完成的下界核验）。
4. `orch dispatch deputy --text "档位自检"` 后台账新行 `model` 字段为 flash
   档型号；registry.json 中 deputy/reviewer 默认档 = flash。

### E1 —— Tier A 全链（依赖：E0）

内容：failure-tags 词表；`<blocked>` tags 属性解析；stats by_failure；
§2.4 阈值自动立案。

成功判据：
1. 组员交付 `<blocked tags="cli_missing">` → `tail -1 state/ledger/deliveries.jsonl`
   含该 tag；构造非法 tag（词表外）→ delivery-gate exit 2。
2. `orch stats` 输出含 by_failure 段（标签×成员计数与台账一致）。
3. 注入 3 条同 (tag, member) 台账行（14 天窗口内）→ 下次 `orch status` 或
   approve 入口自动登记 `source=tier-a` 提案，证据字段含 3 行台账引用，且
   该提案出现在 inspector 队列。

### E2 —— minimum 档 + harness 套件骨架（依赖：E1；前置：用户选定 minimum 型号）

内容：registry 增 minimum 档；`state/evals/harness/` 首批 ≥3 案（任务书+
白名单+机械判据+基线指标）；`orch eval run`；baseline.json。

成功判据：
1. `orch eval run --case <id> --model-tier minimum` 完成 → `state/ledger/evals.jsonl`
   新行 model=minimum 型号，stdout 输出二元判决（通过/失败）+ 指标表。
2. 对一次真实 harness 改动（任一 E1 合并项）执行改动前后各一轮 → orch 输出
   before/after 对比表且判决为「改善/未改善」二者之一（非平局，§0.5）。
3. `orch dispatch <任一专职组员>` 台账 model 字段核验降档面无回归（仍 flash）。

### E3 —— 仲裁团（依赖：用户提供 SiliconFlow Key）

内容：jury 共享库（直连、并发、双序、多数解析、自动附材料）；orch jury 封装；
jury-brief v1（含「仅涉自身基建直通」规则）；doctor Key 泄漏哨兵。

成功判据：
1. 真实争议案（取 E1/E2 期间一条实际 usability 提案）：`orch jury <case>`
   ≤5 min 返回结构化裁决；`state/evals/jury/<case>.json` 含各裁判
   {选择,理由,置信度}、双序一致性标记、usage、成本、材料哈希；评审团型号
   清单不含 Qwen 系（配置文件核验）。
2. Key 安全：`git grep -l <Key 前 8 位>`（全仓）零命中；doctor 泄漏哨兵对
   构造的假 Key 测试文件报 ✗、对干净树报绿。
3. 直通规则实测：一条「仅涉提议者自身专有基建」的 usability 提案经 J →
   裁决 approve（简报规则生效的行为证据）。
4. 构造平票案（材料刻意均衡）→ 结果标记 `escalate=twin`（E6 前挂起呈组长，
   不静默丢弃）。

### E4 —— bencher 组员（依赖：E2）

内容：bencher pod（charter/技能/pysci-bench）；FIFO 队列与进程生命周期；套件
与基线区所有权移交；基线捕获；副本 A/B；择机查看型通知。

成功判据：
1. `pysci-dev doctor` 对 bencher pod 全绿；队列注入 2 项 → 进程自动拉起 →
   消化完毕自动退出（lock 文件消失、.done 摘要落盘），`orch status` 显示
   2 项待查看结果（非即时唤醒）。
2. **M 判决包全程**：一条真实提案 approve 时选 M → `state/evals/baselines/<case>/`
   出现基线（worktree commit 记录 + gitignored harness 归档清单）→ devops
   实施合并 → bencher 建新副本跑 A/B → 二元判决落盘；「改善」→ orch 销账；
   构造「未改善」案 → backlog 自动出现 `source=remediation` 纠正提案。
3. 归因污染防护：构造窗口期插入另一 harness commit 的案 → bencher 判
   `inconclusive`（附窗口期 git log 证据）→ 标记 escalate=twin。
4. 副本无损：A/B 完成后 `git worktree list` 无残留、主树 `git status` 干净。

### E5 —— 决策流硬化全量（依赖：E3、E4）

内容：§5.1 组长判决包流（J/M/J+M/override）；§5.2 inspector 流（判决包 +
机械晋升 + 双队列迟滞 + 全局开关 + 巡检菜单含设计审查 rubric）；§5.6 核验
闭环；咨询环节封装（twin 咨询占位接口，E6 接通）。

成功判据：
1. `orch approve <id>` 不带判决包参数与 override → 代码拒绝（非零退出 +
   可读提示列出合法选项）。
2. 一次完整改进循环实录：提案（kind 结构化）→ J 裁决 → approve → devops
   实施 → 机械核验 → 销账；全程 preference-journal 无未留痕 override。
3. inspector 自主闭环：一条自主发现（巡检产出）→ 自选 J/M → approve →
   自撰任务书派 devops → 合并；核验组长介入次数 = 0（backlog 条目
   actor 字段 + git log 全程无组长会话痕迹）。
4. 迟滞闸：backlog 注入至 pending >12 → inspector state 出现 paused 标记且
   findings 停止晋升；回落至 ≤8 → 恢复标记；两阈值不等（配置核验）。
5. `orch inspector-off` → 后续 orch 入口命令不再拉起 inspector（lock 不出现）。
6. 设计审查 rubric 就位：`pods/inspector/rubrics/design-review.md` 存在且
   含三原则；一次巡检产出引用 rubric 条款的判断类发现。

### E6 —— twin + mirror + 回退链（依赖：E0、E3）

内容：twin pod（PREFERENCE.md v1 经用户确认、readonly_extra、冷启动纪律）；
orch taste；§5.5 回退链接通（E3/E4 的 escalate=twin 由挂起转自动）；
pysci-mirror（status/review/journal/overrides）；§5.4 偏差报告队列与绿色
通道；咨询环节接通。

成功判据：
1. twin 写 PREFERENCE.md 被 readonly_extra 拦截（pod 内实测 exit 2）；
   `orch taste <case>` 两次调用 session-id 不同（冷启动核验），输出
   {选择,理由,置信度} 落 `state/evals/taste/`。
2. 回退链：E3 判据 4 与 E4 判据 3 的挂起案重放 → 自动 twin 终裁 → 非平局
   结果 → 自动 approve/reject；mirror 留痕可查。
3. 盲审闭环：`mirror review` 呈现案件时 twin 答案先隐去（交互实录）；构造
   一次不一致 → 两项追问 → 偏差报告入队 → 组长 orch 处理：选「派任务书」
   路径落地；选「纠正现有结果」的案 → `orch approve` 拒绝 override 参数
   （代码强制）。
4. `mirror journal` 分流：选①仅 journal 新行；选②入强制队列且组长侧可见。
5. `mirror status` 输出首批全景项（待盲审数/宪章 diff/趋势告警/待决事项/
   inspector 水位/交付提醒）。

### E7 —— twin 上线（依赖：E6；前置：语料吸收完成 + 用户宣告）

内容：用户 prompt 总集吸收；上线宣告登记；代理裁决接入 orch 各「呈用户」
升级点（护栏级审批、升级仲裁、Phase 验收）。

成功判据：
1. 上线宣告入 preference-journal（source=user）并登记 README 裁决记录表。
2. 一次真实护栏级事项（如 minimum 档型号确认）：原「呈用户等待」点自动路由
   twin 代理裁决 → `state/evals/taste/` 留痕 {案件,选择,理由,置信度} →
   `mirror review` 该案可盲审；事项从发生到裁决完成无等待用户的阻塞
   （时间戳对比）。
3. 例外核验：最终交付验收点仍呈用户本人（不被代理）；PREFERENCE.md 修订
   仍走 §5.4 用户纠正链。

### 收尾 —— 融合与退役（依赖：E1–E7）

内容：README 重写（融入本方案全部生效机制、裁决登记）；删除本文；各组件
首轮退役自评。成功判据：用户验收。

## 13. 待决问题（均不阻塞定稿，属实施期输入）

1. **minimum 档型号**（E2 前置，专门环节选定；上线后属护栏级，可由 twin
   代理裁决——E7 判据 2 即以此为例）。
2. 基线捕获的 gitignored 面范围细则（E4 实测商榷）。

**终审后动作**：用户确认定稿 → 本文裁决要点登记 README §13 → E0 拆分为
backlog 条目入队（附各自 success_criteria，自举 §6 纪律）→ devops 实施。

## 14. 已知局限与退役机制

- Tier B 判别力下限 = 断言可机械化程度；纯质量差异归 jury/twin。
- **jury 材料质量风险**（撰写者侧组装）：自动附加确定性材料缓解；回访触发
  条件已登记（§4）。
- **M 的时间窗**：基线捕获与实施后实测之间若发生其他 harness 改动，A/B 归因
  会被污染——bencher 判决时应检查窗口期 git 历史，受污染则判 inconclusive
  → twin 终裁（§5.5），不假装可信。
- **twin 代理风险**：错误裁决在盲审前即为系统行为——用户明确选择的效率-风险
  交换；缓解 = 全量留痕 + 盲审动线 + 一致率趋势监测 + 下线权归用户。
- inspector 成本：激活预算/让路/双队列迟滞节流 + 全局开关；bencher 不常驻。
- SiliconFlow 单点：不可达时基建决策降级「入队待裁」；Key 永不入库 + doctor
  泄漏哨兵。
- twin 一致率有天花板（不自洽残余不建模）；override 率只有趋势意义。
- **退役机制**：每组件每 Phase 收尾以 Tier B/C 自评——连续两个 Phase 未产生
  一次改变决策的读数，删除它。

## 15. 委托记录与实施日志（滚动更新，用户 2026-10-11 23:2x 委托）

**委托（用户书面，全文要点已入 preference-journal kind=delegation）**：用户
离线至 **2026-10-12 07:00**；期间组长按自身判断**尽可能推进本方案实施**，
尽可能避免停下来汇报/请示/求批准，汇报与咨询推迟至 7 时后一次性呈报；除非
遇到切实阻碍计划的严重事项，不因等待用户而停摆。首次实践「先决策，后追溯」
——用户对异议实现事后追溯，责任在用户。**本节为委托之防遗忘锚点：任何
后续会话推进本方案前须读本节。**

**委托期内自守边界（组长自律，非用户要求）**：
- twin 上线宣告、PREFERENCE.md 生效确认——留待用户早晨确认，推进至「待确认」
  状态为止；
- E3 真实连通需用户提供 SiliconFlow Key（环境变量 `SILICONFLOW_API_KEY`）——
  代码与 mock 测试先行，真实连通列待办；
- 一切行使护栏级授权的决定（registry model 配置、jury-brief 草案、charter
  真本改动）逐条落决策日志 + 本节实施日志，供早晨追溯。

**实施日志**：
- 2026-10-11 21:42 E0 三项完成（439e7b7/e7340cd+f2e0840/afff18a 等，commit
  7817e0e 起）；判据核验通过（journal 回灌、kind 解析生效、override 强制留痕、
  降档 flash）；
- 21:42–23:1x 首战暴露并修复：#41 registry lost-update（drain 陈旧快照回写
  抹配置——修复经 23:01 run 实战验证：收尾工作树干净）；#42 watch QUEUE-COMPLETE
  脱靶吞账（身份三元组+转移帧必报+抑制帧不推进基线+黑匣子）；#43 锁面 released
  墓碑（release_lock unlink 静默失败→假 running）；#44 锁面 .done 收尾硬证据
  否决判活（进行中，含 220622 观察面契约重述授权）；
- 待用户早晨处理：① SiliconFlow Key 提供（E3 真实连通前置）；② 用户 prompt
  语料文件路径（E6 twin 原料）；③ PREFERENCE.md v1 确认与 twin 上线宣告
  （E6/E7，届时呈「待确认」包）；④ 本委托期内全部护栏级决定的追溯清单。

**委托期夜航日志**：
- 23:33 run 起（45-48 位）：E1+E2 组包批次于 00:04 `run_failed(tool_use)`——
  运行时中断非规格障碍，E1 worktree 留大量未提交半成品；处置=E1 标
  needs_leader（superseded 留档），另立 **001 续作单**（20261012-001000）授权
  devops 在遗留 worktree 内续完（不重做）；E2 已转独立批次于 00:04 起消化；
- 00:28 E2 批次 blocked——devops 会话自行判断先抢救 E1（续作单 scope），
  完成并合并推送（7008a2e→f772bdf，组长验收测试 55 passed，E1 全链在
  HEAD）；组长手工处置：233001/001000 销账（done，防重做），233002（E2）
  回排 pending（附「E1 已入 HEAD，若再 blocked 须写明卡点」指示）；另批准
  devops 新发现（002857：协议扩展滚动窗口——新属性交付遇旧解析器判
  parse_error，批准宽松兜底）；
- 队列余量（12 日 00:5x）：233003（E3-prep 在途）、233248（互斥修复）、
  002857（解析兜底）、233002（E2 重排队）。
