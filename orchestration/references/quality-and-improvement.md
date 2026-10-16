# 质量保证与基础设施改进循环（README §5/§6 全量）

> 机械验收/reviewer/判决包/Tier A·B 与改进循环细节；由 README §5/§6 迁入。

## 5. 质量保证：两层

### 5.1 机械验收（成员声明式，零 LLM）

组员工作形态多样（仿真组员可能只答疑问，绘图组员可能只画草图），**是否触发由组员在
交付中声明**（§3.4 `check` 属性），机械验收不得阻碍灵活任务：

- `check="<type>"` → orch 自动运行对应确定性脚本（注册于 registry checks 表；如图件→
  figures CLI 合规审计，orch CLI/guards/pod 只读层等基础设施类→`orch-tests` 跑
  `pytest tests/skills/orchestration`，guards 面变更→`guards-in-force` 定向跑门禁在法
  探针用例；FAIL→自动回派返工（≤3 次，同指纹连败升级组长）；
- `check="none" reason="..."` → 放行，台账记「未验收（成员判断）」及理由；
- 无 `<artifact>` 的交付（纯咨询类）→ 天然不触发；
- 制衡：台账透明 + 组长 `orch check` 随时抽检 + reviewer 可见验收状态；
- **UI/交互类改动的验收必须含「用户路径测试」**（菜单→Enter→视图→footer 广告**键位矩阵**
  逐键断言可观察效果 + 未广告键无副作用），**函数直调测试不作数**——2026-10-14 `n` 键用户
  路径断裂穿过验收，根因正是「测试只走面板路径（`t`→`n`），菜单→Enter 路径零覆盖」；范例
  `tests/skills/orchestration/test_tui_user_paths.py`（f1 立）与 `test_tui_visual_structure.py`（f2）；
- 脚本所有权在 devops，组员禁改（只读层）——独立性由所有权保证；
- unrouted（未知 check 类型）→ 放行 + 记台账（门禁故障不阻塞生产）。

### 5.2 reviewer（主观学术质量，可选重炮）

- 默认不触发；用户点名或组长主动时 `orch review`；
- 有权执行昂贵动作（文献检索/下载/转换、已发表对照样本视觉对比）；
- VERDICT：`PASS | FAIL` + 分维度评分（科学正确性/规范性/美观度/创新性）+ 逐条证据；
- **G2 独立性**：`orch review --goal-from <原生产任务书>`（或 `--goal "<文本>"`）把组长
  撰写的目标陈述嵌入审查任务书，使「与任务书目标一致」可据组长原意独立核验，不依赖
  生产者 notes.md 自述（消除与 charter 独立性纪律的张力）；缺省时告警但不中断；
- FAIL→自动回派返工（附报告）→复审一次→再 FAIL 升级组长仲裁（改派副组长/换方案/
  呈报用户）；
- rubric 存于 reviewer pod 只读层，修订须经用户（量表质量决定审查质量）。

### 5.3 判决包（基建提案的判决通道；四态强制 + 例外通道 + 修复形状分流）

**四态强制（用户裁决，实施前实测不可行故有此设计）**：组长 `orch approve/reject` 时
判决包必居其一，缺参数 → 代码拒绝（rc≠0 + 列出合法选项）：

- **J**：案卷送 Tier C 仲裁团（家族外三席官方直调、双序对局、并行单轮阻塞 ≈1 min、
  结构化返回「选择+理由+置信度」、少数服从多数、材料由**撰写者侧**组装并自动附 git
  diff/预算读数/台账行，`orch jury` 见 §4.1）→ 裁决自动 approve/reject，组长只写回复；
- **M**：approve 并实施 + **基线捕获**（审批时 git worktree @ HEAD + gitignored harness
  面物理归档进 `state/evals/baselines/<case>/`）+ 实施后 bencher **副本 A/B 实测**（§5.5）；
- **J+M**：仲裁把关实施、实测核验效果，**实测覆盖仲裁**（仲裁 approve 但实测未改善 →
  自动立案纠正）；
- **都不选 = override**（approve 或 reject 均是）：附详细理由、`mirror` 留痕、计入
  **override 率**（`orch stats` 判决包健康段 / `pysci status`）；`orch reject` 无
  `--override` 参数即代码拒绝——**reject=override** 是数据事实，不是礼仪。

**例外通道**：缺陷类 × 同条交付含 `<blocked>` → 当条工作本身即 in-vivo 实测，**默认
approve**（规程建议连带选 M 复跑原失败案）；组长拒绝仍记 override。

**修复形状分流（remedy_shape，用户裁决 2026-10-13）**：inspector 机械通道的**免判决
只剩 `restore`**（复原既有声明行为：恢复被删行、修断正则、回滚误删配置）；凡
**`change`**（改声明行为/策略/权衡面——会话生命周期、阈值语义、charter 条款、路由语义）
而**无判决记录** → drain 任务书注入约束，devops 必须交付 `<blocked>` 上报而非自行实施
（裁定权在判决包与用户，不在执行端）。落点三面：登记字段缺省 change、任务书模板注入、
决策审计面高亮（§6 `pysci decisions`）。

**核验闭环（零新裁决点）**：核验方式在审批时一次定死——机械判据永远由交付协议自动执行，
重型 A/B 仅当审批选了 M 才跑；**实施前至多一次 J、实施后只实测不再仲裁**；改善 → 销账，
未改善 → 自动生成纠正提案（`source=remediation`）进 §6 管道。**模棱两可回退**：bencher
实测无显著指导意义、或 jury 平票/inconclusive 过半 → 自动回退 **twin 终裁**（不给平局，
结果必为选项之一，全量 mirror 留痕）。

### 5.4 Tier A：失败标签与阈值自动立案

- 词表 `orchestration/state/failure-tags.json`（**单一源，护栏级**，devops 增补附定义）：
  组员 `<blocked>` 可带 `tags="a,b"`（delivery-gate 校验词表，词表外退回、缺省不阻断）；
  聚合读数 `orch stats` 的 by_failure 段——**同一标签反复出现 = 深层结构问题的客观信号**；
- **成因档位 `reason="…"`**（同一词表的 `reasons` 段，backlog 20261013-191213-devops）：
  `<blocked>` **语义过载**——「已侦察、待裁决」（`needs_decision`，重派改变不了它，停链等
  裁决）与「执行端余量不足」（`capacity`，补足余量后重派即可落地）**最优处置相反**，却都
  长成同一个 blocked；`tags` 说「失败是哪一类」、`reason` 说「该谁怎么处置」，二者分开。
  档位随交付落进 drain 的 `.done`，`orch status` 与之**分列**计数（`待裁决=N 待排期=N`），
  链路策略亦可按档分化；缺省不标注合法（同 tags 纪律）。
- **阈值自动立案（硬性依赖，不指望组长「有空」）**：`orch status`/`approve` 入口机械
  重算滑窗——同一 `(tag, member)` 14 天或 20 跳内 ≥3 次 → 自动注册基建提案
  （`source=tier-a`，证据=台账行与计数，`success_criteria` 天然可测=销账时计数回落）
  → 进 §6 改进管道。检查、计数、立案全代码完成，人只在强制选择点出现。

### 5.5 Tier B：harness 测试套件（bencher 所有）

- **两种测试的分野（用户裁决）**：普通测试（src，`tests/`，pytest）由 devops 管理；
  **Agent harnessing 测试**（历史真实任务提炼的特殊套件）由 bencher 管理——命名同源
  benchmark，清晰区隔；
- **套件** `state/evals/harness/`（bencher 完全写权）：每案 = 任务书 + 白名单 + 机械
  成功判据 + 预期指标基线；输出统一**二元判决**（通过/失败、改善/未改善），执行经
  `dispatch` 特化封装（全新会话、`minimum` 压测档），结果入 `state/ledger/evals.jsonl`；
- **基线与副本 A/B**：判决包选 M 时捕获基线 → devops 合并后 bencher 以合并 commit 另建
  副本 → 双副本各跑关联案、对比台账、二元判决，用后即弃、无损回退；窗口期混入他人
  harness 提交 → 判 `inconclusive` 附 git log 证据 → 回退 twin（不假装可信）；
- **生命周期**：FIFO 队列（`orch bench add`），非空拉起、消化完销毁（**非常驻**）；
  完成项经 `orch status` **择机查看**，不即时唤醒组长；外部代码树零写权（设施缺陷提
  `<infra_suggestion>` 交 devops）。

---

## 6. 基础设施改进循环

**全员均有提议资格**（7 专职 + deputy + reviewer + devops）；建议从宽受理（§3.4：
缺陷类附证据，难用类描述场景即可）；采纳与否组长裁定。组长自己也可提议（审批人
是用户）。

```
组员交付含 <infra_suggestion>
→ orch 登记待审批项，呈组长：建议摘要 + git 历史相关改动（防往复翻转；必要时
  orch deputy consult 副组长评估）
→ 组长裁决 approve（入 backlog FIFO；推理密集项可 `--effort high` 按项标注，§3.5）
  / reject（附理由）
→ 审批回复存 state/replies/，下次派发该成员时自动附送（闭环告知）
→ approve 默认**自动唤醒** devops **batch 计划器**：分离后台进程 `_drain-devops` 持
  state/devops.lock 组包并把批次**入队** `state/queues/devops.json`（`kind=backlog-batch`），
  再由 **devops FIFO worker**（`orch _queue-worker devops`）逐个执行——**执行只留队列
  worker 一份实现**（backlog-fifo-20261014：drain 自持的 do_dispatch 循环退役；组长
  零阻塞、零轮询，对齐 §4.5 等待模型）
  · --no-wake 只入队不唤醒；计划器锁忙不重复 spawn——条目在**持久 backlog**里由在跑那位
    一并计划/FIFO 排队消化（**锁忙语义消失：排队而非丢弃待拾**）；计划器**一次一批**，
    下一批由 worker 收尾后的续计划链上（未取项保持 pending）
  · 锁 stale（持有进程死 / 锁龄 >6h）自动接管，防计划器崩溃后死锁
  · 释放锁时 unlink 删不掉（Windows 他方持句柄 → 共享冲突）**不得静默**：锁记录改写为
    `released` **墓碑** + run 日志留一行 `[!]`。墓碑不作互斥（`_lock_live` 判非活、
    `lock_status` 报 stale 且非 crashed），下一任 drain 就地改写接管——否则「删不掉的残骸
    + 判活误真」会把队列挡成假 running，最长阻塞到 6h 超龄才解（backlog 20261011-225933-devops）
  · 墓碑也写不进（他方连写都不共享）时仍有**第二道收尾证据**：判活采纳「该 run 已有
    `.done`」即非活（`_run_concluded`；`.done` 在 `release_lock` 之后才可能出现，故与墓碑
    同为收尾宣告）。该证据须过**两道闸**：摘要自证 `run_log` 同名（身份）+ 摘要
    `completed` 不早于锁 `started`（时序，`_done_predates_lock`）——否则上一任的尾会把
    同秒续跑的活锁判成残骸并被抢走，单 worker 互斥击穿。命名面同时查重：`new_run_log()`
    以 `O_CREAT|O_EXCL` 当场占位，撞名顺延 `<ts>_002`…（分隔符取 `_` 不取 `-`，否则顺延名
    字典序早于被占名，`latest_done` 会读旧摘要）（backlog 20261011-231237-devops +
    20261011-233248-devops，与 §4.5「收尾事实优先于锁的自称」同一纪律）。`lock_status`
    对这类残骸报 `residue="done-exists"`，`orch status` 呈现**残骸已收尾**
  · 获锁即回收孤儿 in_progress 条目（前任 worker 中途被杀的遗留）重置 pending，并顺带
    清理其残留 worktree/branch（`git worktree remove --force wt-<id>` + `branch -D` +
    `prune` 兜底）——worktree 名 = 条目 id（take_first 记录、任务书指定），只清孤儿名、
    不误删活跃/手动 worktree
  · `orch devops stop` 协作式停止：写**计划器** state/devops.cancel **与 devops worker**
    state/queues/devops.cancel（指向各自锁持有者 pid），两侧在条目间隙消费→干净退出
    （释放锁、写部分 .done 含 stopped=cooperative、cancel 自删）；陈旧信号（pid≠本进程）
    清除且不误停接棒 worker；锁空闲不写信号（no-op）
→ 计划器每批：取队首为**种子** + 收集同提请者(member)**且同 effort 档位**的 pending 为
  **组包**（含种子 ≤4 项，同提请者≈同模块/同视角；一跳只有一个跳级档位，混档会静默
  改变被标项的强度）→ 批量任务书（种子必做 + 组包菜单，devops 自主选取合并实施，
  单 worktree=种子 id）→ 批次正文**入队**（含元数据）→ **worker 执行**：dispatch devops
  （§7）→ 测试验证 → git 提交（+ `timeout 25 git push origin main` best-effort，失败忽略，
  用户裁决 2026-10-10）→ 交付列 `backlog id=<ids>`；**账务**（销账/needs_leader/延期/系统性
  停止/抢救）在 worker 侧结算（单一实现 `drain._settle_outcome`，计划器 dry 支与 worker
  真派发支共用）；选组包项保持 pending 的项由 worker 收尾后的**续计划**当新种子再入队
  · 成功 → 按交付 id 清单**逐 id 销账**（done，记 commit hash + batch_size/batch_seed）；
    组包中未选项保持 pending 留待下轮（devops 可只吃种子）
  · 失败（blocked/run_failed）→ 仅**种子** needs_leader、组包项保持 pending（**不连坐**，
    一项卡住不阻塞全队列）；其中交付带 `reason="needs_decision"`（已侦察待裁决）的条目额外
    挂 `awaiting_decision` 标记——`orch status` 单列「待补授权」，重排任务书自动注入
    「原点名事项 ↔ 本次补齐情况」对照行（backlog 20261013-193058-devops），使「执行端点名
    要授权 → 组长补授权 → 重排」这条闭环在机械面上可见、不必靠人记得写附注
  · 系统性故障（`quota_exhausted` 额度耗尽 / `channel_error` 渠道级故障）→ **全局停止**：
    种子复位 pending（非 needs_leader）、组包与剩余条目保持 pending、释放锁、.done 注明
    停止类别；恢复后下次唤醒自动续消化（两类只恢复动作不同：充值/换渠道 vs 等渠道恢复）
→ **两本账**（backlog-fifo-20261014）：**计划账** devops-runs/<ts>.{log,done}
  （`kind="plan"`：计划了 N 批；watch 对 plan-done 不报 QUEUE-COMPLETE）；**执行账**
  queues-runs/devops/<ts>.{log,done}（销账/停止/by_reason 档位分列在此）。orch status
  展示：计划器 worker 态（running/idle/stale，stale 再分**崩溃**〔pid 死且缺该 run 的
  .done〕、**残留已标记**〔released 墓碑〕与**残骸已收尾**〔该 run 已有 .done〕）、
  **devops FIFO worker 态 + 队列水位行**（待/在做/完成/失败）、
  最近 run 摘要（**执行账优先**；含停止方式：自然跑完/协作停止/额度耗尽/渠道级故障/异常中断）、
  needs_leader 提醒（组长据此 `orch requeue` 回排重派 / 改任务书 / `orch resolve` 销账 / 上报）
→ 组长的等待面（代码化，§4.5）：唤醒/采纳/交办三处输出固定附 `orch watch` 挂载指令；
  watch 后台阻塞轮询 state 文件与锁，命中 QUEUE-COMPLETE / NEEDS-LEADER / QUOTA-STOP /
  CHANNEL-STOP / CRASH 即单行退出唤醒会话（devops backlog 的**完成唤醒自 backlog-fifo 起
  走事件流 `QUEUE-EMPTY`**，由执行侧 worker 退出时发）；watch 随会话消亡、不影响 drain，漏掉的账由下次 `orch status`
  的 watch 段一次性补账
→ 涉及发起人 pod 的改动完成后，组长可立即重派该成员验证
```

**提案三源与机构分工**（自改进体系收官并入；上图是组员源的判决包流）：

- **组员 `<infra_suggestion>`** → 上图组长判决包流（§5.3）；
- **inspector 发现（不经组长）**：机械证据类（harness 测试失败、doctor ✗、预算超限、
  实锤复现）→ **自动晋升** backlog（`source=inspector`，附复现命令与实际输出 + 可证伪
  success_criteria）；**判断类** → 强制判决包 {J, M, J+M}（§5.3）→ approve 后 inspector
  自撰任务书派 devops，全程零组长介入；**inspector 无 override**（其提案被拒不阻塞任何
  工作）；设计审查 rubric（社区优先 / 反冗余 / LLM 直觉友好）产出的发现属判断类；
- **Tier A 阈值自动立案**（§5.4）→ 进 inspector 同一管道；
- **偏差报告绿色通道（mirror 盲审 → 组长处理）**：用户盲审 twin 决策不一致 → 偏差报告
  入队；「twin 不对」→ twin harness 改进**绿色通道默认通过**（用户反馈即仲裁结果），
  组长二选一：写任务书派 devops ／ override 留痕（批量处理多个不一致案不算 override）；
  「纠正现有结果」→ **强制写任务书派 devops，不得 override**。`mirror journal` 的行动
  指令入同一队列、同「强制派发」待遇（§4.1 用户控制台）；
- **留痕与审计面**：override 一律入决策日志并计入 **override 率**（`orch stats` 判决包
  健康段；趋势意义，非绝对指标）；核验记录、twin 代理裁决与终裁自动入日志（原料层
  `state/preference-journal.jsonl`）。**`pysci decisions` = 决策链全貌**（只读拼装：
  findings × backlog × run 摘要 × jury 案卷 × 决策日志）——发现 → class → remedy_shape
  → 判决记录 → 实施 commit → 核验状态，**免判决+change 置顶**（最需审计者优先）；
  `mirror status` 的「决策」项实时报水位。重要决策必须汇总进概览并可一屏看全
  （用户裁决 2026-10-13；ctx 水位闸绕过判决通道案即教训，§13 登记）。

**双队列迟滞节流（用户裁决）**：inspector 监视 devops backlog pending 与 bencher 队列
长度——任一 > 积压上限（backlog 12 / bench 6）→ **暂停产出**新提案（只读巡检继续）；
两队列均回落至**恢复阈值**（8 / 4）→ 恢复。**恢复阈值严格低于暂停阈值 = 迟滞**，防振荡。

**闲时自调度巡检（autonomy-fix a2）**：此前只有 `orch inspector start` 拉起式，2026-10-14 实测
02:08–07:42 **零巡检**、02:21 入队的 patrol 积压整夜（唤醒链断 + 无命令 = 永久沉睡）。
现由 a1 看门狗 tick（60 s）顺带敲 `inspectd.self_patrol_tick`——**唯一调度耦合点，不新造第二套
循环**：全局开关关 / worker 在跑 / 冷却窗内（`cooldown_s`=3600）任一为真都不动；队列空才自入队
一项 `patrol` 再 `wake_inspector`。因此 §1 那句「闲时自主激活」是**代码保证**而非纪律，且
让路/节流/cooldown/预算闸全部照旧。

**队列入队超时告警（不许静默积压）**：同一 tick 还敲 `inspectd.notify_stale_queue`——队列项
入队 > 1800 s 仍未被取走即进统一事件层发 `QUEUE-TIMEOUT`（词表单一源 `events.WAKE`；每项只报
一次，靠 `timeout_notified_at` 去重），并在 TUI「盲审与审查」/「队列与运行」详情点名；`orch status`
inspector 段单列「队列超时 N 项」。

**backlog 判据纪律（三权分立）**：条目必填可证伪 `success_criteria`——提案者写、判决
案卷审定、devops 只执行；**无判据不入队**，判据同时是 M 实测的靶点（§5.3/§5.5）。

**组长自提改进**：组长向用户交付成果时可一并提出自己的改进建议；用户回复认可后，
组长以同一 approve 通道入 backlog。**此流程不设 orch 硬保护**（裁决理由：管理工具的
改进若强制经过管理工具自身会引入自举问题；且用户无强制否决权，硬保护无意义）——仅由
orchestration 技能软性约束「提议→告知用户→入 backlog」的默认顺序。极端情况下组长有
权直接改基础设施（权力结构裁决），但不推荐、须在台账留痕。

触碰护栏（本文档、角色边界、审批权、guards、只读层定义）的建议：组长必须转呈用户，
不得自行 approve。

---
