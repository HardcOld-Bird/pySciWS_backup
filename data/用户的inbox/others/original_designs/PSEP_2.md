# PSEP 2 — 夜航模式（无人值守推进）：单开关 + 机械政策注入 + 判据拦截

> **备忘（2026-10-14 补，优先实施项）**：plan 系统不支持派发 devops 环节
> （`queues.QUEUE_MEMBERS` 不含 devops，`plan run/advance` 成员闸拒绝——系早期
> 决策「基建条目走 devops backlog」未考虑大任务显式含基建改进的情景，实现不
> 彻底）。**先补齐此缺口**（收编 devops 入队列成员、成员闸反转、ADR 注记、
> backlog-batch 不回归、全链实测），随后 PSEP 2 原内容的基建环节即可直接经
> plan 派发 devops，受益于此。此备忘属 PSEP 2 实施序的一部分，实施完成后
> 并入正文或随本文退役。

> Status: **已批准（方向）**，待实施｜类型: Enhancement（PySci Enhancement Proposal，戏拟 PEP）
> 作者: 组长（综合用户需求 2026-10-14 与次要会话调查评估）｜关联: PSEP 1（原 REFACTOR
> 收官版，已更名存档）｜依赖基建: wake-claim / cronless 队列感知巡更 / 配置实时化
> （均已落地）、twin taste_enabled=true

## 0. 动机（用户工作模式，原文即需求）

> 「接下来一段时间，我将离开电脑，无法即时回复你的消息，因此请你尽可能持续推进
> 目前事项/大任务，除非遇到不得不停下的严重阻碍（或者所有事项完成），否则你自行
> 执行决策并推进即可。对于确实需要用户口头授权的高风险事项，请咨询 twin，我完全
> 授权 twin 的决策具有与我等同的决策权力。如果我真的对你的决策有异议，事后追溯
> 责任在我，不在你。」

需要一个**手动开关**控制这种「夜航」模式：开启时项目持续自动化运转，**不因等待
用户回复产生任何阻塞**；关闭即恢复正常交互。

## 1. 调查评估结论（次要会话方案的采纳与修正）

**采纳**：三层架构（政策/判据/保险丝）；「hook 延迟打回」即 watch 钩子既有语义；
条件化周期唤醒大半已存在（PENDING-WORK/STALL + 30min 节流）；「LLM 当定时器」铁律
（patrol §2）；PermissionRequest 权限弹窗是真正硬阻塞、最简解法是用户离开前调权限档。

**修正（四处）**：

1. **政策注入不用 rules 手动开关文件**，改钩子机械注入（§3.2）——单一开关、热切换、
   消灭「模型没读到政策」失效模式；
2. **WAITING-USER 自动捕获优先**（§3.3）——`AskUserQuestion` 被 PreToolUse 钩子
   机械拦截转 twin，把「不等用户」从劝告变成机制；软写入兜底**复用 `mirror-pending.json`**
   （它本来就是「此刻谁在等用户」的真源），不新建 night-hold；
3. **twin 代行授权书显式化**（§3.2 政策文本内置）——leader-only.md §3 要求裁决须用户
   授权，政策文本即用户签发的**限时授权书**（开关开启期间有效），含护栏级/规则级裁决；
4. **保险丝层缓行**（§3.4）——若加装必须条件化（未竟事项+无活动才打回；终态静默），
   否则「全部完成后的定时打回」就是纯烧 token。

**审计面（用户裁定）**：夜航逐条总结**可免**——项目已有 **twin 仲裁盲测追溯板块**
（pysci「盲审与审查」/决策审计面）作为既成审计设施。**配套要求：把该板块落实做好**
（盲测案件、twin 判断、用户复核链路可追溯），这是本方案唯一的软依赖。

## 2. 总体设计

```
data/pysci_settings.toml [night]  ←── 唯一手动开关（用户改一行即生效）
        │
        ├── ① 政策注入层：SessionStart / PostCompact 钩子读开关 →
        │      hookSpecificOutput.additionalContext 注入政策全文（含授权书）
        ├── ② 判据层：
        │      a. PreToolUse(AskUserQuestion) 拦截：夜航开 → deny + 注入「转 twin 终裁」
        │      b. queue_pressure 夜航判据：mirror-pending 非空 → PENDING-WORK 伪事件
        │         （30min 节流既有；单一实现扩展，不动四处消费面）
        └── ③ 保险丝层（缓行，规格预留）：见 §3.4
```

## 3. 详细设计

### 3.1 单开关（真源 `data/pysci_settings.toml`，配置实时化同一哲学）

```toml
[night]
enabled = false          # 夜航总开关（手动）：true = 政策注入 + 判据拦截生效
# periodic_wake_s = 3600 # 保险丝打回间隔（预留；未实施前勿设——见 §3.4）
```

钩子每回合现读文件（热切换，无需重启会话）；非法值回落 false 并出声。

### 3.2 政策注入层（SessionStart + PostCompact 钩子）

- 触发点选 **SessionStart + PostCompact**（而非每回合 Stop）：逐回合注入烧 token；
  而上下文压缩正是政策最容易丢失的时刻——PostCompact 补一针即覆盖真风险。
- 载体：既有钩子体系 `hookSpecificOutput.additionalContext`（exit 0 + JSON 注出），
  不新建平行钩子（token 竞争/wake-claim 均在 watch-arm.sh 链上）。
- **注入的政策全文草案**（即用户授权书，可直接使用）：

  > **【夜航模式生效】** 用户已离机并签发以下授权（本授权随 [night] enabled=true
  > 有效，关闭即失效）：① 尽可能持续推进所有事项/大任务，不得因等待用户回复而
  > 停滞或结束回合观望；② 一般决策你自行执行，事后追溯责任在用户；③ 需用户
  > 口头授权的高风险事项（含护栏级/规则级裁决）**咨询 twin 并依其裁决执行**——
  > twin 的决策权等同用户（本条为 leader-only.md §3 的限时预授权）；twin 不可达时
  > 降级为「副组长咨询 + 自行决断 + 留痕」，仍不等用户；④ 终态仅两种：**全部
  > 完成**或**不得不的严重阻碍**——达成即写一段收尾总结、将 enabled 改回 false、
  > 结束推进；⑤ 禁止把 LLM 当定时器（空转打回）；⑥ 工具权限弹窗属硬阻塞，用户
  > 离机前应自调权限档（见 §3.6）。

### 3.3 判据层

- **a. `AskUserQuestion` 机械拦截**（硬主力）：PreToolUse 钩子（`if:
  "AskUserQuestion"`）读开关——夜航开 → `exit 2` deny 并附「夜航模式：该决策转
  twin 终裁（`orch consult-twin`），依其裁决执行」。提问工具本身成为不可能，
  不依赖模型自觉。`PermissionRequest` 暂不接管（§3.6）。
- **b. WAITING-USER 伪事件**（兜底）：`watch.queue_pressure`（判据单一实现处）增
  一条夜航判据——`mirror-pending.json` 非空 → PENDING-WORK（detail 注明「夜航：
  待决事项应转 twin 或自行裁决，不得等用户」），沿既有 30min 节流。非夜航时段
  不加此判据（白天等用户是正常的）。

### 3.4 保险丝层（缓行，规格先立）

仅当出现「有未竟事项（计划/队列/任务书）且组长无活动」的停滞才打回（建议
`periodic_wake_s ≥ 3600`）；**全部完成/严重阻碍两个终态一律静默**。鉴于
PENDING-WORK 已覆盖大部分停滞面，本层残余价值薄——**观察一期后再决定是否加装**
（实施单另立）。

### 3.5 终态语义

「全部完成」与「严重阻碍」都是合法终态：组长写**一段**收尾总结（进展/遗留/关键
决策点，供晨间阅读）、`enabled` 改回 false、然后停。夜航不空转、不悬挂。

### 3.6 权限弹窗（人工 checklist，不写钩子）

工具权限弹窗是唯一会硬卡死夜航的非模型因素。**用户离机前手动将会话权限调至
宽松档**即解——为它写 PermissionRequest 自动放行钩子收益低、风险高，明确不做。

## 4. 实现注意（给未来实施单）

- 全部钩子 **fail-open**（异常 exit 0，不塌唤醒主链路）；`[night]` 读取失败按 false；
- 复用不新建：政策注入走既有钩子体系、WAITING-USER 判据进 `queue_pressure` 单点、
  拦截钩子与 watch-arm.sh 同目录同风格（`.qoder/hooks/`）；
- 测试：开关热切换、注入触发点（SessionStart/PostCompact）、AskUserQuestion deny
  双向（夜航开/关）、mirror-pending 判据节流、政策文本含授权书关键词；
- 文档：README §4.5/§6 或 patrol-watchdog.md 一段 + `[night]` 注释自解释；
  预算（≤8192B 注入文件）与 grep 判据约定（20261014-075230）照旧。

## 5. 验收判据

- `[night] enabled=true` 后：新会话/压缩后自动收到政策全文；`AskUserQuestion` 被
  拒绝且收到转 twin 指令；mirror-pending 非空 30min 内触发一次夜航 PENDING-WORK；
- `enabled=false` 全部行为回到现状（零残留）；
- 全仓测试全绿；twin 仲裁盲测追溯板块可完整追溯一次夜航 twin 裁决（配套要求）。

## 6. 残余风险与开放项

- twin 质量即夜航决策上限（用户显式授权的选择）——配套审计面（盲测追溯）做实
  是补偿手段；
- 软写入侧（模型不写 mirror-pending 就停）由 AskUserQuestion 拦截 + 保险丝（缓行）
  双层兜底，但非绝对硬保证——接受此残余或日后加装保险丝，观察后定；
- 双会话场景：夜航开关是项目级配置，次要会话同样读到政策注入——政策文本已含
  「持续推进」语义，次要会话若被唤醒也应按夜航政策处置或立即让位（wake-claim
  已保证它不收唤醒，风险很小）。
