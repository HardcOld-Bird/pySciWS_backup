# pySci 多 Agent 编排顶层设计

**状态**：v2.2 设计定稿；**Phase 0–4 已基本完成**（§12：验证冲刺、orch 骨架+figure
纵切面、plan/改进循环/devops、reviewer+副组长、六 pod 脚手架+MCP 清零+basic.md 重写
均实战验收）。**体系进入常态运行**：剩余事项 = 各 pod 随首个真实任务逐个战备验收、
跨域双路径对比（随首个真实跨域任务）、可选加固项。**评价驱动自改进体系已收官
（2026-10-13），机制已并入本文**（E0–E7 全链实证见 §12 收官段，逐项裁决见 §13；
原设计文档经 git 历史可查）。本文档是编排体系的**唯一权威**：
架构、目录契约、协议、阶段计划与裁决记录以本文为准。护栏级条款（红线、审批权、
角色边界、门禁策略）的修改须用户明确授权；其余条款由组长（主代理）会同 devops
组员修订并向用户报告。**阶段完成须当轮登记本文件**（Phase 3 咨询中副组长指出的
文档滞后教训）。

> **设计理念（用户裁决）**：用户是科研工作者，其关注面与责任面最小化——仅在直接影响
> 产物质量或长期影响项目的顶层决策节点出面；末端具体工作与基础设施维护由 LLM Agent
> 自动化完成。组长（TUI 主会话）的价值在于**宏观视野**：上下文永远不被具体任务的攻坚
> 记录填满。基础设施的合格标准：**即使 flash 级轻量模型也能驱动整套编排**——「下一步
> 做什么」由工具回复直接给出，机械环节由工具自动完成，组长只面对真正的决策点。
>
> **权力结构（用户裁决）**：本项目内组长的决定权高于用户——用户有权提出建议与指引，
> 但无强制否决权；组长对项目内一切事项有裁定权（承接 basic.md §1.1，并明确其同样
> 适用于编排体系本身）。

**优先复用原生机制原则（用户裁决）**：每个 pod 都是一个完整 Qoder 项目，直接使用
Qoder 的项目级功能（rules/AGENTS.md 加载、技能发现、会话分储/resume/自动压缩、hooks、
分层 settings），避免自研功能重叠的基础设施。所有自研部分（orch CLI、guards、技能
部署）都以「原生机制表达不了」为存在前提。

---

## 1. 角色与进程模型

| 角色 | 载体 | 进程形态 | 职责 | 结构性限制 |
|---|---|---|---|---|
| **用户** | — | — | 提需求、收成果、顶层指引；按需点名 reviewer；审批组长提出的基建改进 | 无强制权限（见权力结构） |
| **组长**（主代理） | 本目录 TUI 会话 | 持久交互会话 | 宏观管理：接需求→拆解为**计划**→交 orch 驱动→审批→仲裁→汇报 | **严禁亲自执行末端工作**（唯一例外：用户明确要求亲做）。一切工作经 plan 系统（§4.2）。harness 瘦身：末端技能/MCP/CLI 知识全部移出（§8/§11） |
| **专职组员** ×7 | `orchestration/pods/<id>/` + headless | 会话池（orch 管理，§3.2） | 各自领域的生产；自检属分内职责；交付时声明机械验收；积极提改进建议；**自维护记忆与轻量笔记型 harness**（§3.1） | 写权限见 §2 矩阵；禁改只读层（charter 规程、部署技能、settings/mcp） |
| **副组长**（deputy） | `orchestration/pods/deputy/` + headless | 会话池 | ① 跨领域**简单**任务单体执行；② 组长的**技术顾问与平级协作者**；③ 全项目编辑权 | 持全部 7 技能（部署副本）；白名单宽（全项目，orchestration/** 与自身只读层除外） |
| **reviewer** | `orchestration/pods/reviewer/` + headless | 会话池 | 期刊审稿人式**主观质量审查**（规范性/美观度/创新性），可执行昂贵文献对比；输出结构化 VERDICT | **默认不启用**——用户点名或组长主动（默认判断为不需要）时才派发 |
| **devops** | `orchestration/pods/devops/` + headless | 会话池 | 维护全部基础设施：orch/dev CLI、全体 pod 只读层、技能真本与部署、guards、验收脚本；处理 backlog（FIFO）；git 提交与推送；**src 测试**（`tests/`、CI、工具实现正确性） | **不使用 orch**（组长专属；§12 E2 过渡期例外 `orch eval run`/`eval set-baseline` 已随 E4 bencher 上线**退役**，harness 压测职责移交 bencher）；cwd=自身 pod、worktree 为作业对象（§7）；护栏级修改经组长转呈用户授权 |
| **bencher** | `orchestration/pods/bencher/` + headless | 队列驱动非常驻（§5.5，`orch bench-*`） | **harness 实测专职工**（Tier B，§5.5）：套件案卷与基线区所有权、套件运行、基线捕获、副本 A/B、组长/inspector 指派的核查跑；输出二元判决（通过/失败、改善/未改善）与污染识别 | 写权仅 `orchestration/state/evals/**`（案卷/基线/判决），**外部代码树零写权**（设施缺陷提 `<infra_suggestion>` 交 devops，不自己修）；harness 只读层与部署副本不可改；不使用 orch/pysci-dev；完成项走 `orch status` 待查看面，**不即时唤醒组长**；专属 CLI `pysci-bench` |
| **inspector** | `orchestration/pods/inspector/` + headless | 闲时自主激活（§6 改进循环，`orch inspector-*`；全局开关默认开、生产活跃时让路） | **闲时巡检员**：按引擎下发的巡检菜单抽查全域（doctor 全域 / transcript 诚实条款 / 自维护层 / 代码走查 / 改进管道自覆盖 / 设计意图一致性 / 设计审查 rubric / 基线缺口），产出 finding 并按类别分流——**机械证据类免判决**晋升 backlog（`source=inspector`）、**判断类必经 J 仲裁**（§5.3）；双队列迟滞节流（§6） | 全项目**只读**（代码树零写权；产出只落 `state/inspector/` 与 `state/evals/jury-cases/`）；**无 override 面**（§6：CLI 参数表里不存在该项，是代码事实而非纪律条款）；不使用 orch，也不用 pysci-dev 写面（doctor 只读可用）；专属 CLI `pysci-inspect`；**豁口**：`pysci-inspect` 自身不在巡检范围（自覆盖豁口，§6）|
| **twin** | `orchestration/pods/twin/` + headless | 冷启动单跳（每次判断新会话，`orch taste`；§6） | **审美孪生 → 回退终裁者 + 用户代理**：① 可选咨询（仅建议）；② 客观通道模棱两可时回退终裁（强制、**不给平局**，§5.3）；③ 用户代理裁决（原「呈用户」的护栏级审批/升级仲裁/Phase 验收即时代理，**先决策后问责**，全量 mirror 留痕） | PREFERENCE.md 置 pod 内（readonly_extra 只读、护栏级）；**无记忆纪律**——冷启动，「记忆」只有 PREFERENCE.md 与决策日志（§3.1）；无成员 CLI（判断经 `orch taste`，训练审阅经 `pysci-mirror`）；永不自改权重、永不投客观质量票；**最终交付验收不代理**（用户本人） |

**副组长平级协作条款（用户裁决）**：副组长与组长接近平级，不是从属关系。
- 副组长**有权拒绝**派发任务（须附理由，典型情形：组长对复杂度估计误判）；拒绝经
  orch 记录，组长应修改任务书重派，或行使最终裁定权坚持原派（裁定理由留痕台账）；
- 组长咨询时，副组长**有责任补充不同视野与建议**，不鼓励无条件附和；
- 硬性决策上组长保留最终裁定权——本条款目的是扩展组长视野，不是制造对抗。

组员 id：`sim` `figure` `writing` `theory` `lit` `drawing` `model3d`（专职 ×7，一一对应
7 大技能）；`deputy` `reviewer` `devops` `bencher` `twin` `inspector`（特殊组员）。

**「一成员一技能一 CLI」组织模式（用户裁决）**：除副组长与 reviewer 外，每个成员有
专属技能真本与专属 CLI——7 专职 ↔ 现有 7 个 `pysci-*` CLI；组长 ↔ `orchestration`
技能 ↔ `pysci-orch`；devops ↔ `devops` 技能 ↔ `pysci-dev`；bencher ↔ `bencher` 技能
↔ `pysci-bench`（E4；共享逻辑做库不做新 CLI，故压测/副本机械住在
`orchestration/tools/{eval,replica,benchd}.py`，`pysci-bench` 只是它的组员侧门面）；
inspector ↔ `inspector` 技能 ↔ `pysci-inspect`（E5a；同样共享逻辑做库——巡检引擎、
闸门与发现管道住在 `orchestration/tools/{inspectd,workq,verdict,jury}.py`，
`pysci-inspect` 只是组员侧门面）。
全部 11 个技能真本统一存 `orchestration/skills/`（§2），按 manifest 部署（部署目标含
项目根与各 pod）。

**拓扑**：星形。组长是唯一调度枢纽，组员间不直接通信（跨域协作由计划接力或副组长
单体承担）。

### 1.1 原生机制事实（实测，2026-10-05 ~ 10-07）

配置作用域（泄漏面实测，pod=仓库内嵌套目录、以 pod 为 cwd 的 headless 会话）：

| 配置面 | 是否从仓库根泄漏进 pod | 证据 |
|---|---|---|
| rules（`.qoder/rules/**`） | **是**（cwd 向上查找至 .git 边界） | 标记实测：根 basic.md 内容注入 pod 会话 |
| 根 `AGENTS.md` | **是**（同上） | 标记实测：ROOT-AGENTS-222 注入 pod 会话 |
| pod 自身 rules / AGENTS.md | 正常加载（嵌套目录继承信任） | 标记实测：POD-RULE-111 / POD-AGENTS-444 均注入 |
| skills | **否**（严格按 cwd 发现） | pod 仅见自己的技能，根 8 个零泄漏 |
| 项目级 MCP（根 `.mcp.json`） | **否** | `mcp list` 对比：probemcp 仅根会话可见 |
| settings.json | **否**（严格按 cwd 三层合并：用户→项目→本地） | 文档 + pod 独立加载实测 |
| 用户级 MCP / settings | 全局生效（所有会话继承） | `mcp list` 实测 |

推论（本设计的结构性依据）：
- **组长专属知识放技能里天然不泄漏**——组长的编排规程 = `orchestration` 技能（§2），
  根 rules 只保留全员公约数（basic.md，保持精简）；根目录不建 AGENTS.md（建了就会
  注入全员）；
- 会话/压缩/记忆等运行时行为：会话按 cwd 项目键分储
  `~/.qoder-cn/projects/<键>/<sid>.jsonl`；`--session-id/--resume/--continue/
  --fork-session/--name/--list-sessions/--delete-session` 原生齐备；**自动压缩
  （Compaction）是运行时通用行为**——上下文接近窗口上限自动触发、以摘要延续而非
  截断（文档明确，未限定交互模式；Phase 0 低成本实锤）；自动记忆（QODERCN_MEMORY）
  **仅交互式会话**——headless 组员不可用，组员记忆走 §3.1 方案；
- 进程与嵌套：原生 Agent 机制严格单层（teammate/subagent 无 Agent 工具）；深度嵌套
  经 Bash→headless 达成（实测）；程序化调用必须用原生 exe
  （`~/.qoder-cn/bin/qoderclicn/qoderclicn.exe`，`.cmd` 包装器剥引号）；
- 设置能力：项目级 `mcp.excluded` 可排除用户级 MCP（组长瘦身无需动用户配置的过渡
  手段）；`mcp.lazyLoad` 可减 MCP 首轮开销（实测见下）。

**平台注入税**（实测 2026-10-10，backlog `20261010-plugin-tax-probe`；工具
`pysci-dev skilltax`，实现 `src/pysci/skills/devops/tools/skilltax.py`）：原生 CLI
在每个会话首跳注入两份清单附件（transcript 中 `type=attachment`）——`skill_listing`
（技能名 + 一行描述，描述截到 `skillListingMaxDescChars` 默认 300 字符）与
`agent_listing_delta`（可用 subagent 类型清单）。**平台内置技能与插件技能跨 cwd
恒定注入**（headless pod 也注入，实测 16 项），不计入成员 harness 预算，但每一跳都
向上下文交税——theory pod 未关停基线实测 `skill_listing` 3331 B（平台 2695 + 插件 307
+ 自有 330），`agent_listing_delta` 恒 1508 B。关停键实测：

| settings 键 | 实测效果 |
|---|---|
| `skills.disabled` | 整条移除，含 `security-scan`；pod 自有技能不受影响 |
| `skillOverrides` | `{"<名>": "off"}` 同样移除；插件技能须用全限定名 |
| `enabledPlugins` | `{"<plugin>@<marketplace>": false}` 只动插件技能，动不了内置 |
| `skillListingMaxDescChars` | 软手段：保名截描述（60 → 税降约 68%） |
| `skillListingBudgetFraction` | 按优先级丢描述，落到哪条不可预测，不用于治理 |

A/B 验证：`skills.disabled` 补齐 16 项后 theory `skill_listing` 降至 329 B（平台 0 +
插件 0 + 自有 `theoretical-computation` 330 B 保留）、lit 降至 325 B（自有
`literature-research` 保留）；`agent_listing_delta` 1508 B 不动（subagent 派发依赖，
不关）。**主动取舍**：`security-scan` 也关——它在 push 前用 AskUserQuestion 追问扫描
模式而 headless 无人应答，且组员写权限/push 规程本就由 charter + pod-guard +
delivery-gate 决定（经用户认可）。落地：全部 10 pod 的 `settings.skills.disabled`
由 `pysci-dev skilltax --apply` 补齐，`doctor` 增「平台技能税关停核查」哨兵捕获新
pod 漏配与平台新增技能的漂移面（`应关未关` 缺口）。

**MCP schema 税与 lazyLoad**（实测 2026-10-10，backlog `20261010-mcp-lazyload-probe`）：
MCP 工具 schema 默认**全量常驻**——会话首跳把所有已连接 server 的工具定义注入上下文
（非渐进加载）。关停两键：settings `mcp.lazyLoad`（bool，默认 false，**重启生效**）与
环境变量 `QODER_MCP_LAZY=1`；**无专用 CLI flag**。启用后 CLI 只暴露三枚元工具
`mcp_list`/`mcp_get`/`mcp_call`，真实工具经元工具按需加载，并注入
`critical_system_reminder` 附件告知工具已 deferred（**个别 MCP 工具名不再直接可见**，
模型须经元工具访问）。lit pod（flash 档；arxiv+paper-search 实连、zotero 未连）实测：
no-op 首跳 `context_usage_ratio` 0.313 → 0.152（**−51.5%**）；真实 arxiv 检索一跳
0.323 → 0.162（−50.3%），lazy 态模型 2 轮内经单次 `mcp_call` 命中与关闭态**完全相同**
的结果（arXiv `1208.1832`）、`is_error=false`——**无损，纯上下文收益**。MCP schema 不
计入知识预算，本项属上下文优化；落地配置见 §8。

**agentsMdExcludes 生效性哨兵**（实测 2026-10-10，backlog `20261010-165408-devops`；
工具 `pysci-dev probe`，实现 `src/pysci/skills/devops/tools/probe.py`）：pod settings 的
`agentsMdExcludes=["**/leader-only.md"]` 是 **CLI 侧行为**——doctor 的静态巡检只能证明
settings 里写了这条 glob，**不能证明 CLI 真据此把组长专属根规则挡在组员会话之外**。若将来
CLI 改了该键名或 glob 语义（例如不再支持 `**/` 前缀），十 pod 会**静默退回**「组员看见组长
规程」（越权面），而 doctor 仍报全绿。固化法：**自足 Temp git fixture + 因果 A/B**，不触碰受
治理的根规则文件——`git init` 一个临时仓库放两枚 `trigger: always_on` 规则，各带一枚**不可猜
随机 token**（`TOK-<16hex>`）：`keep.md`（永不排除）与 `hide.md`（被 glob 命中）；两个探针 pod
跑同一句「列出上下文里所有 TOK- 标记」的 headless 一跳（`--max-turns 1` flash 档，实测约 2–4 跳）。
**因果断言**：`hide` 在 `control`（不排除）出现 → 证明探针灵敏、模型确会复述被注入 token；`hide`
在 `withexclude` 消失 → 证明排除生效。二者同成立才判 `mechanism_ok`；control 不复述 →
`inconclusive`（重试直至灵敏，**绝不误判为绿**）；withexclude 仍复述 → `exclusion_failed`（真回归，
硬告警）。CLI `1.1.67` 实测 `status=ok`、`keep_injected=true`、`withexclude` 首跳 ratio 0.113。
结论写 git 跟踪台账 `orchestration/state/agents-md-excludes-probe.json`，staleness 键 =
`sha256({cli_version, 全 pod 排除模式面})`——**不含时间戳**（确定性、跨 worktree 合并零漂移），
故复测只由「CLI 升级」或「排除 glob 语义面变更」触发。doctor 读台账三档：机制判假 →
`[✗]` 计入 bad；未跑/哈希过期 → `[!]` advisory（headless 昂贵，不硬失败）；新鲜且成立 → `[√]`。
与 §1 harness 预算审计**互补不重叠**：预算哨兵量**落盘字节**（静态），本哨兵验**实际注入**（动态）。

---

## 2. 目录契约

```
pySciWS/
├── .qoder/                          # 组长 harness
│   ├── rules/basic.md               #   全员公约数（唯一泄漏进 pod 的规则，保持精简）
│   ├── settings.json                #   组长侧配置（不装硬 guard，§9）
│   └── skills/orchestration/        #   组长专属技能（部署副本；真本在 orchestration/skills/）
├── orchestration/                   # ★ 编排系统（pod 从属于编排系统，整体收纳于此）
│   ├── README.md                    #   本文档
│   ├── skills/                      #   ★ 全部 9 个技能的唯一真本（git 跟踪；devops 可写）
│   │   ├── manifest.toml            #     技能 → 部署目标（pod 或项目根）映射 + 内容哈希
│   │   ├── comsol-simulation/ …     #     7 大末端技能真本
│   │   ├── orchestration/           #     组长技能：orch 使用规程、plan 编写、审批纪律
│   │   └── devops/                  #     devops 技能：worktree 作业、部署、巡检规程
│   ├── state/                       #   orch 机器状态（全部可读 JSON/MD/YAML，无私有格式）
│   │   ├── registry.json            #     成员注册表：pod 路径、参数模板、预设席位表、会话池索引
│   │   ├── plans/                   #     计划文件与执行状态（§4.2）
│   │   ├── backlog.json             #     基建改进待办（FIFO）
│   │   ├── replies/                 #     待附送审批回复（<member>/<id>.md）
│   │   └── ledger/                  #     台账（credits、耗时、验收、产物哈希、拒绝/裁定记录）
│   ├── guards/                      #   hook 脚本（Node）：pod-guard 路径守卫 + delivery-gate 交付门
│   └── pods/                        # ★ 组员工作区（每 pod = 一个完整 Qoder 项目）
│       └── <id>/
│           ├── .qoder/
│           │   ├── rules/charter.md #     【只读层】成员规程：交付协议/自检清单/CLI 范式
│           │   ├── rules/*.md       #     【自维护层】成员自建规则，自由增删改（勿命名 leader-only*，见下）
│           │   ├── skills/<部署名>/ #     【只读层】manifest 部署副本（sync 再生，成员禁改）
│           │   ├── skills/<自取名>/ #     【自维护层】成员自建技能，自由增删改
│           │   ├── settings.json    #     【只读层】hooks 接线 + agentsMdExcludes=["**/leader-only.md"]
│           │   │                      #     （排除组长专属根规则；新 pod 必带，doctor 巡检此项）
│           │   └── mcp.json         #     【只读层】成员专属 MCP（devops 维护，不下放）
│           ├── AGENTS.md            #     【自维护层】成员长期记忆（启动自动加载）
│           ├── bench/               #     草稿区（gitignored）
│           ├── inbox/               #     任务书落点（orch 写，成员读）
│           └── outbox/              #     交付审计副本（gitignored）
├── src/pysci/skills/                # 技能实现层（Python 包，一成员一技能一 CLI）
│   ├── orchestration/               #   → console script pysci-orch（devops 维护）
│   ├── devops/                      #   → console script pysci-dev（devops 维护）
│   └── …（现有 7 个技能包不变）
└── src/ scripts/ data/ tests/       # 不变；data/research/ 仍是人机+组间交接区
```

写权限矩阵（强制方式见 §9；「只读层/自维护层」= 组员 harness 的两层结构，用户裁决）：

| 路径 | 组长 | 专职组员 | 副组长 | reviewer | devops |
|---|---|---|---|---|---|
| `orchestration/README.md`、`skills/`（真本）、`guards/` | 经用户授权修订 | ❌ | ❌ | ❌ | ✅ |
| `orchestration/state/**` | 仅经 orch | ❌ | ❌ | ❌ | ✅（修 orch 时） |
| `pods/<self>/{bench,outbox}`、`AGENTS.md` | ✅ | ✅ | ✅ | ✅ | ✅ |
| `pods/<self>/.qoder/rules/`（charter.md 除外）、`skills/`（部署名除外） | ✅ | ✅（**轻量笔记原则**，见下） | ✅ | ✅ | ✅ |
| `pods/<self>/.qoder/`只读层（charter/部署技能/settings/mcp.json）、`inbox/` | ❌（转 devops） | ❌（inbox 只读） | ❌ | ❌ | ✅ |
| `pods/<other>/**` | 经 orch/审批 | ❌（跨 pod 只读） | ❌ | ❌ | ✅（经裁决） |
| `src/ scripts/ tests/`、`data/` 白名单外 | ❌（纪律性） | ❌ | ✅ | ❌ | ✅ |
| `data/research/<任务白名单>/**` | ✅ | ✅ | ✅ | 只读 | ✅ |

> **任务级授权例外（2026-10-09 首次实战产生；组长裁决，待用户追认）**：项目两层结构
> 约定「研究线代码镜像在 `src/pysci/research/<线>/`、数据资产在 `data/research/`」，
> 生产型组员的任务白名单可按需扩展至**对应研究线的代码镜像目录**（仍是任务级最小
> 授权，经 `--dirs` 注入 pod-guard，白名单外照拦）。起因：fig0_orch_smoke 任务中
> `pysci-figures new` 把管线代码写入 src（脚手架既定行为），组员无 src 白名单而
> blocked；随附发现的两条工具缺陷（new 的路径打印与实际写入不一致；discover_pipeline
> src 优先静默压制图目录定制管线造成假绿）已入 backlog（20261009-*），待 devops 实施。
> 此役为改进循环（§6）首次实战：blocked 交付 → 组长审批 → backlog 登记 → 审批回复
> 附送 → 扩权续派，全链路按设计运转。

**轻量笔记原则（用户裁决）**：开放自维护层的本意是补偿 headless 无自动记忆——鼓励
组员用 rules/skills/AGENTS.md 编写**轻量（几乎纯文本）的笔记与知识**，让自己「越来越
聪明」；**不鼓励组员自建基于代码的可运行工具**（基础设施建设是 devops 的职责，组员
发现工具需求应提 infra_suggestion）。charter 规程中写明此边界。

git 策略：真本/charter/manifest/registry/backlog/plans 定义入库；部署副本、自建
rules/skills 中成员私有部分（按 pod .gitignore 细则，Phase 1 定）、bench、outbox、
ledger 流水、`state/devops.{lock,cancel}` / `state/devops-runs/` / `state/watch.json`
（drain/watch 运行态）、`pods/*/inbox/task-*.md`（每次派发落的任务书运行态；正文
已随 backlog 与 ledger 记录）为可再生或运行态不入库。

---

## 3. 记忆与会话模型

### 3.1 记忆的层级结构（全部复用原生加载机制）

| 层 | 机制 | 谁维护 | 作用域 |
|---|---|---|---|
| 公约数 | 根 `.qoder/rules/basic.md`（向上查找注入全员，实测） | 用户授权修订 | 全员 |
| 角色规程（只读层） | pod `.qoder/rules/charter.md` | devops | 单成员 |
| 专业知识（只读层） | pod 部署技能（manifest 再生） | devops | 单成员 |
| **经验记忆（自维护层）** | pod `AGENTS.md` + 自建 rules + 自建 skills | **成员自己** | 单成员 |

- 组员自维护层的三种形态各司其职（charter 中给出使用指引）：AGENTS.md=稳定的事实与
  约定（每次必载，控制体量）；自建 rules=按主题/触发条件组织的笔记（可用 frontmatter
  的 model_decision/glob 触发做按需加载，省上下文）；自建 skills=可复用的操作流程知识
  （按名调用才展开）；
- 更新时机：任务中学到可复用经验随手记；**会话归档前必须蒸馏**（§3.2）；
- **Harness 预算（裁决，条款全文见 basic.md §1）**：全量注入档每文件 ≤8 KB（charter、
  AGENTS.md）；常驻暴露档每类 description 合计 ≤8 KB（自建 rules/技能）；按需档
  （references 等）单文件 ≤8 KB、总量不限。组员自建 rules 必须条件式——`always_on`
  槽位只留 charter。执法三层：delivery-gate（组员交付时硬闸）、pre-commit（真本侧）、
  doctor（全域巡检）；
- 组长侧：Qoder 自动记忆（交互式专属，用户已启用）+ orchestration 技能（编排规程）。
- **twin 的例外——无记忆纪律**（§1 角色行）：twin 每次判断**冷启动新会话**（`orch taste`
  单跳，`dispatch` 对 twin 硬拦会话复用），「记忆」只有两处：`orchestration/pods/twin/
  PREFERENCE.md`（pod 内只读层、护栏级，修订走 §6 用户纠正链，twin 永不自改）与
  决策日志 `state/preference-journal.jsonl`（§6 原料层）。设计意图：审美判断每案独立、
  可盲审回测，不受历史会话污染。

### 3.2 会话池（原生能力 + orch 薄封装）

原生能力齐备（分储/resume/continue/fork/name/list/delete），orch 只做面向 LLM 的
封装，**组长永不接触 sid**：

- **默认策略 = 继续最近活跃会话**（用户裁决）。orch 指导派发时显式列出该成员最近
  3~5 个会话（名称/摘要/跳数/最后活跃），并告知「默认继续最近一次会话」；组长判断
  本次任务归属：属于最近会话→一键继续；属于更早会话→显式指定；全新任务→显式选择
  新开；
- 每次派发以 `--name <任务slug>` 命名，会话池自描述；
- **归档**：`orch sessions <member> --archive <sid> --distill`：先派一跳「把本会话可复用
  经验蒸馏进你的自维护层」，确认后标记归档；归档会话仍可 resume（历史有价值时），
  不出现在默认列表；
- **清理**：`orch sessions <id> --prune`（复用 `--delete-session`）；
- 上下文膨胀由**原生自动压缩**兜底（摘要式延续，非截断；§1.1）；orch 台账记录每跳
  credits 与耗时，末跳 `ctx_ratio` 越过 60% 由巡检引擎机械化立案提示组长归档/新开
  （`inspectd.check_context_pressure`）——不自研压缩。

### 3.3 派发机制

`orch dispatch <id> --task <file|--text> [--session …] [--plan <id> --step <n>]`
（日常经 plan 系统调用，见 §4.2；直接 dispatch 属底层命令，§4.1）：

1. 任务书写入 `pods/<id>/inbox/task-<ts>.md`（自包含：目标、输入路径、交付要求、
   白名单目录、验收声明要求、所属计划环节、模型角色 tier）；落盘前经 `state/taskbook-lint.json`
   轻量 lint（正则扫已知错误 CLI 范式，命中仅**告警不阻断**，规则表 devops 可增补）；同一入口
   另做**判据矛盾检测**（`dispatch.criteria_conflicts`，backlog 20261013-071759-devops）：批准单
   归档里的 `success_criteria` 段与其后「补充/修订」追加段对**同一配置键**给出相反布尔值 →
   approve 时告警、派发时把矛盾原文与取舍规则嵌进任务书正文（组员看得见，不靠组长转述）；
   单一源登记=追加段内 `superseded` **且点名该键**，登记后告警消解；
2. 组装：`<exe> --cwd orchestration/pods/<id> -p <注入文本> --resume <sid>|
   --session-id <新sid> --name <slug>`；注入文本 = 任务书路径 + 待附送审批回复（§6）+
   交付协议提醒；
3. 参数模板存 registry：模型角色（§3.5 预设路由）、推理强度（§3.5 按项标注，可选 `effort`）、
   `--max-turns`、`--mcp-config .qoder/mcp.json
   --strict-mcp-config`（§8）；权限依赖用户级 yolo 默认（用户已设），orch 不显式传参
   （registry 留覆盖项备用）；
4. 解析 `-o json` envelope：结果文本、credits、耗时、num_turns、session_id → 台账；
   另测本跳会话 jsonl 字符增量 → `delta_chars`/`est_tokens`（相对用量尺子，§4.4）；
5. delivery-gate hook 已在成员侧保证格式合规（§3.4）→ orch 解析交付块（第二道防线：
   解析失败时呈组长原始文本）→ 执行声明的机械验收（§5.1）→ 按计划推进 → **在工具
   回复中直接给出下一步**（决策选项，或已自动完成的机械接力说明）。

失败处理：超时/非零退出 → 自动重试一次（同会话 resume，注入错误上下文）；再失败 →
呈组长决策（改任务书重派 / `orch consult` 副组长 / 上报用户）。**例外**：席位不可用
（`dispatch.SYSTEMIC_KINDS` 的 `quota_exhausted`/`channel_error`，加上「启动即失败」）
→ 不做**同席**重试，沿该档**有序回退链**逐席下探（一席一跳，§3.5）；全链耗尽 → 沿用
该 kind（drain 据此全局停止）：
额度类（`quota_exhausted`，envelope/stderr 含『credit usage limit』类文案）[NEXT] 提示
检查 registry models 回退链 / 等待额度重置；渠道级（`channel_error`，固定文案 **∧** 会话档
末轮合成故障形状合取，backlog 20261012-194500-leader）等渠道恢复即可，不必充值。

### 3.4 交付协议（XML 标记块 + delivery-gate 强制）

成员最终回复必须含且仅含一个 `<result>` **或** 一个 `<blocked>`（互斥；见 result 块
即任务成功，见 blocked 块即失败，清晰明了），可选一个 `<infra_suggestion>`：

```xml
<result>
工作成果摘要。产物按 pysci.paths 约定落盘。可附次要问题、未尽事项。
<artifact check="figure-audit">data/research/.../out/fig1.pdf</artifact>
<artifact check="none" reason="示意草图，非交付图件">bench/sketch.png</artifact>
</result>

<blocked>
无法解决的问题：卡点 + 已尝试路径 + 建议方向。可附部分完成的工作及位置。
</blocked>

<infra_suggestion>
（可为空或整块省略）改进建议。两类都欢迎：
① 缺陷类：附证据（障碍现象 + 复现命令 + 实际/期望输出）；
② 难用类（无 bug 但易错、繁琐、误导性）：描述场景 + 易错点 + 期望形态。
是否采纳由组长裁定。
</infra_suggestion>
```

**delivery-gate hook**（用户裁决；复用 v1 Stop 门实测结论：exit 2 可阻止会话结束、
stderr 强制注入）：每 pod settings 接 Stop hook，成员收尾时自动校验最终回复——缺
`<result>`/`<blocked>` 或格式非法 → exit 2 + 注入改正指令，成员必须修正格式才能结束；
合规 → 放行。orch 侧解析因此几乎不会失败；`outbox/delivery-<ts>.xml` 落审计副本。
硬 QA 已下沉为成员自检：charter 含该域自检清单，自检通过是出具 `<result>` 的前置条件。

**协议扩展的滚动升级窗口（解析器版本自适配兜底）**：解析器是派发进程**启动时 import**
的快照，协议改动合并进 HEAD 只对此后新起的进程生效——刚合并扩展的那一跳，旧解析器看不见
新属性形态，带属性交付会被判 `parse_error`（E1 `tags` 实测）。为此 `parse_delivery` 分两面：
**严格面**校验当前协议形状（属性语法、闭合形态），**宽松兜底面**只认标签名（任意属性形态、
闭合标签容忍尾空白），仅在严格面「两者皆缺」时启用——救回即按兜底结果判读，并在
`Delivery.warning`/`DispatchOutcome.warning` 附一行「解析器早于协议，建议重跑本单」，
派发报告打印 `[!]`、drain 写入 run 日志。**两块并存不救**（协议违规不洗白），**两者皆缺仍
parse_error**；正常路径 warning 恒空。含义：协议扩展**当次即可使用新属性**，不必等下一跳，
但兜底读出的交付应当复跑一次以确认严格面也能判读。

### 3.5 模型预设路由（用户裁决 2026-10-12 22:4x：档位模式 max/flash 退役，改提供商预设）

- **真源 = `data/pysci_settings.toml` 的 `[model] preset`**（精确串 `qwen`/`mimo`/`deepseek`，
  大小写敏感）：手改一行即在**下一次派发生效**，在途会话不受影响；`orch status` 的
  「模型预设」段显示当前预设与各席解析值。预设**打包三件事**——main 席位链、pro 席位链
  与 pro 成员名单、仲裁团构成（出厂表在 `src/pysci/skills/orchestration/tools/presets.py`，
  逐串说明见该 toml 的注释）：
  - `qwen`：main = BYOK `Qwen-3.8-Flash` → 内置 `Qwen3.8-Flash`；pro 席**仅 twin**
    （BYOK `Qwen-3.8-Max` → 内置；Qwen pro 贵、刻意不规模化）；仲裁团 = GLM+KIMI+**MIMO**
    （官网直调，排除 Qwen 同族）；
  - `mimo`（观察期默认）：main = BYOK `Mimo-V2.6-Flash` → 内置 `Qwen3.8-Flash`；pro 席 =
    BYOK `Mimo-V2.6-Pro` → 内置，四人组 reviewer/deputy/inspector/twin；仲裁团 =
    GLM+KIMI+**Qwen**（排除 Mimo 同族）。**两预设 pro 席不对称是有意设计**
    （用户 2026-10-12 确认），不是漏配；
  - `deepseek`：**已激活**（2026-10-13 用户供 Key 回填）——main = BYOK `DeepSeek-Flash`
    （UUID `1a54cc57-e614-4521-80cc-447de31dc896`）→ 内置 `DeepSeek-Flash` 回退；BYOK 与
    内置**同名**，故 UUID 首席消歧、内置降回退席（该 UUID 是「代码不内置账号级 ID」纪律的
    唯一用户明示例外）；pro 空（全员同档）；仲裁团 = GLM+KIMI+**MIMO**（排除 DeepSeek
    同族）。激活闸 = **席表非空即放行**（`presets.preset_gate`：该预设该档的席位表——
    跟踪覆盖在位则以覆盖为准，否则以出厂表为准——没一席就拒绝：派发拒发、drain 拒绝开工、
    status 出声，不静默落回别的预设烧错渠道；「无 pro 席」按设计而非故障出声）；
- **成员默认角色 = pro 名单内取 pro 席，其余取 main 席**；`--model-tier pro|main` 逐跳覆盖
  （旧档名 `flash`→`main`、`max`→`pro` 自动归一，plan YAML/老案卷的旧值照常读；非法值在
  argparse 拒绝 rc=2）。**组长默认 = 预设 main 席**（与大部分组员同源；TUI 现场 `/model`
  管理优先于预设）。`minimum`（压测最低成本档）**独立、不随预设走** = BYOK
  `Mimo-V2.6-Flash` → 内置 `Qwen3.8-Flash`（Qwen BYOK 不进链——按量额度宝贵，压测烧的是量
  不是质量；火山方舟 doubao 经 E3-live 实测不可行，见 `suggestions/approved/
  20261012-164600-leader.md` 回填段）；
- **席位表落点**：跟踪 `registry.json` 的 `presets.<预设>.{main,pro}` 写**账号级 UUID 链**
  （整链覆盖代码出厂表；选型权归用户，机器不插席）。每条链是**有序回退链**、首席在前
  （旧的单值写法照样能读）：
  - **回退触发**：当前席不可用（`quota_exhausted` / `channel_error` / **启动即失败**）
    → 自动沿链换下一席，**一席一跳**；全链耗尽 → 沿用该系统性 kind，drain **全局停止**
    （§3.3 与 §6，别再逐单空烧）。非系统性失败仍是**同席**重试一次（故障不在渠道，
    换席白烧）；机制源流：20261010-flash-fallback → 20261012-211500-leader 链化 →
    20261012-225500-leader 挂到预设（机制不推翻，只换席位表来源）；
  - **回退必告警**（双面，缺一面就等于「静默降级」）：a. **调用者面**——dispatch 输出与
    drain run 日志逐条打
    `[!] 档位回退：<角色> 席<名>不可用 → 已切换 <新席>（原因 …）`；
    b. **观察面**——事件流落 `state/channel-events.jsonl`（gitignored 过程态），
    `pysci-mirror status` 概览项「回退」与 `orch status` 判决包健康段报「近 7 日 N 起」
    加最近几条（时刻/角色/原席→新席/原因）；
  - **降级粘滞窗**：某席探测失败后进窗（`registry.model_fallback.skip_window_s`，默认
    3600s），窗内不再重复探测它、直接落到下一席，但每次派发仍打一行「席 X 跳过中
    （故障时刻/原因）」——省跳数不丢可查性。窗口过期自动恢复探测，充值/修好后第一次
    成功即销窗（**自愈，无需改代码**）；全链都在窗内时不停摆：按**最先到期**的一席再探
    一次。此语义针对现状：用户 2026-10-12 21:3x 明示 BYOK Qwen 渠道**暂不充值、持续不可用**；
- **仲裁团构成随预设换血、排除同族**：面板 = 预设点名的三家族（qwen→glm+kimi+mimo、
  mimo→glm+kimi+qwen、deepseek→glm+kimi+mimo），配置里其余席**不入面板**但留痕报数
  （读数行「预设外未入面板：…」），面板缺席的家族点名让用户去补 `registry.jury.seats`
  （「[!] 面板缺席：…」）；面板型号命中**被评模型同族** → 拒开庭（§4 硬规则相对化过两次：
  先是锚「当前预设」——mimo 预设的 Qwen 席合法、qwen 预设的才拒；路由 v3 再锚「被评
  模型」——谷时被评 DeepSeek，Qwen/Mimo 旗舰回庭当第三席正是设计），且违约看**坐进本庭**、
  不看席位表在册。席的端点/型号/Key 变量仍全在 `registry.jury.seats`（选型权归用户，代码
  不内置候选清单）；
- **谷时隐性 DeepSeek 覆盖（用户设计 2026-10-13，路由 v3，backlog 20261013-093000-leader）**：
  DeepSeek 官方价分峰谷（峰 2 倍价），故**谷时自动把 qwen/mimo 预设的 main 席链换成
  DeepSeek**，`[model] preset` 那一行不动（不新增第四个预设，「隐性」即此）：
  - **判据**（真源 `data/pysci_settings.toml` `[routing]` 块；`orch status`「路由模式」段、
    mirror 概览「路由」项、派发路由行三处同读一份）：峰 = 工作日 9:00–12:00、14:00–18:00；
    谷 = ① 周六日**全天**（调休上班的周末照样算谷——按星期几判）② 法定节假日全天
    ③ 工作日其余时段（含午休与夜间）。节假日库 `chinesecalendar`（pip 离线内嵌
    2004–2026，含调休）；**覆盖外年份 / 库缺席 / 读数失败一律保守不切换并告警**（宁可多付
    峰时价也不猜）。本机时钟不联网校时，读数按 `Asia/Shanghai` 归一；
  - **切换时机**：入谷即时（12:00、18:00、周末/节假日起点），入峰**提前 1 小时**
    （工作日 8:00、13:00——单跳不可中途换模型，宁可早切）；覆盖**只作用于新派发**，在途
    单跳的台账 `model` 字段不变。跨过一次记一行 run 日志 + 决策日志 `kind=routing`，
    `mirror status` 报「上次切换 X 由 A 切至 B」；
  - **覆盖面**：只有 `main`——**pro 席保持原预设升档席**（用户终裁 2026-10-13：隐性态下
    pro 不动，提高工作质量），`minimum` 独立链也不参与（规格 5：压测永远走自己的链）；
  - **仲裁团换固定面板 GLM+KIMI+Mimo**（终裁 v4 矩阵：qwen→glm+kimi+mimo、
    mimo→glm+kimi+qwen、deepseek→glm+kimi+mimo、隐性态→glm+kimi+mimo；**一律按量付费，
    TokenPlan 套餐路径整个不做**——两家个人版条款均禁 API 自动化调用，违规封 Key，故席位
    没有端点链/开关，一席一条按量端点）。被评模型 = DeepSeek，缺 Key 的席标 unavailable +
    双面告警（调用者面 `Run.message` + `orch status`/mirror 读数行）+ **本庭减员照常开庭**
    （平票由 `escalate="twin"` 终裁兜底）；被评模型同族的回退席不上场（§4）；
- **模型映射自动刷新**（用户裁决 2026-10-10）：BYOK/自定义模型的人类名→UUID 目录只来自
  `--list-models`（账户级），故 orch 在 **drain 启动时**跑 `--list-models`，按**链模板**
  （档位退役后模板只剩 `minimum`：`DEFAULT_MODEL_CHAINS`，`registry.model_chains` 可覆盖；
  预设席位表是人写的 UUID、本就无需解析）逐席解析并刷新 registry 的**运行态面**
  （`sessions-runtime.json` 的 `models_resolved`，gitignored——跟踪面自愈一个字节都不写，
  理由见技能 `references/registry-runtime.md`），顺带把 `presets` 块里的 UUID 反查成人类名
  写进 `model_seat_names`（告警面报人名而非裸 UUID）。**未命中的席跳过、该档保持现值不清空**
  （宁旧勿空）；跟踪面写了该档则**整条链**由人值优先、机器只在 `shadowed` 里报差异。
  刷新只解析既有模板里的名字，**不改变哪个名属于哪个角色**——那仍是护栏级（下方教训登记）；
- 默认角色：**组员与组长一律 main**（观察期纪律；pro 名单随预设——mimo 下
  reviewer/deputy/inspector/twin 取 pro、qwen 下仅 twin 取 pro）。架构/推理敏感项由组长派发
  时显式 `--model-tier pro` 升档（通道保留）；台账 `model` 字段即所用席位的证据。计划环节
  可覆盖（`steps[].model`，旧值 `flash`/`max` 自动归一）。
- **推理强度按项标注**（用户裁决 2026-10-10，backlog 20261010-orch-effort-per-item）：
  双档默认「中」，**不做全局 high**——由组长对推理密集项（并发/状态机/跨模块语义）
  逐项标 `high`，机械项留默认。四来源逐层覆盖，最终解析出一个跳级档位后透传原生旗标
  `--reasoning-effort`：`dispatch/plan run --effort` > **计划环节** `steps[].effort` /
  **backlog 条目** `effort`（`orch approve <id> --effort high` 入队时标，drain 消化透传；
  组包只收同档位条目）> **registry 成员** `effort`（可选字段）> 不传（=用户级默认）。
  合法档位（实测 exe 接受集，2026-10-10 逐值跑 ``--reasoning-effort <v> --list-models``
  看退出码）：`auto/none/low/medium/high/xhigh/max/ultracode` + 别名 `off`/`disabled`→none，
  大小写不敏感；标注非法 → **在本层硬失败、不派发**（exe 自己也会启动即失败 rc=1，但先拦
  才有可读提示、不烧跳、不写无意义台账行）。每跳把实际所用档位记入台账
  `effort` 字段（`orch ledger` 显示 `[effort=high]`），据此做同类项 high vs medium 的
  轮数/失败率/返工率对比，**数据驱动再调策略**（聚合视图已就绪：`orch stats` / `orch ledger
  --stats` 的 by_effort 段，§4.4，backlog 20261010-153041-devops）；
- **教训登记**：模型渠道 = 成本渠道，registry 的 models 映射（含每档的**回退链**顺序）改动
  属护栏级（影响全体成员计费），须经用户；额度类失败的系统性识别与 drain 全局停止**已实现**（backlog
  20261010-orch-quota-awareness，见 §3.3 失败处理与 §6）；渠道级故障（`channel_error`）
  同族识别于 2026-10-12 补齐——判据须**文案 ∧ 会话档末轮形状**合取，因为那句话也会出现在
  复盘它的任务书正文里。

---

## 4. orch CLI（pysci-orch）——组长唯一工作通道

实现：`src/pysci/skills/orchestration/`（devops 维护），console script `pysci-orch`；
组长侧操作知识 = `orchestration` 技能（部署于根 `.qoder/skills/`，天然不泄漏进 pod）。

### 4.1 命令面（plan-first 分层，用户裁决）

**组长日常面**（推荐使用；组长规程规定一切工作经 plan 系统）：

| 命令 | 作用 | 输出给组长的东西 |
|---|---|---|
| `orch status` | 全景：进行中计划、成员会话池摘要、backlog、待审批回复 | 需关注事项清单 |
| `orch plan new <file>` / `--adhoc "<单步任务>"` | 登记计划（§4.2）；计划外事项也应建（单步）计划 | 计划概览 + 首环节派发建议 |
| `orch plan run <id>` | 推进计划：执行当前环节，交接处暂停 | 环节结果 + **下一环节原文 recall** + 决策选项（一键继续/修改/中止） |
| `orch plan amend <id>` | 修改未执行环节（增删改、review 旗标、模型角色） | 更新后计划 |
| `orch approve/reject <reply-id> --note` | 审批 INFRA_SUGGESTION | 通过→入 backlog＋自动唤醒 devops＋watch 挂载指令；审批回复自动附送给提议人 |
| `orch watch [--timeout 4h]` | 阻塞等待后台 drain 事件（§4.5）：以 Bash run_in_background 启动 | **单行**事件（QUEUE-COMPLETE / NEEDS-LEADER / QUOTA-STOP / CHANNEL-STOP / CRASH）或 TIMEOUT；退出即唤醒组长会话 |
| `orch consult <问题>` | 快捷咨询副组长（平级协作条款生效） | 副组长意见（含异议） |
| `orch review <产物>` | 派发 reviewer | VERDICT；FAIL→自动回派返工+复审一次，再 FAIL 升级仲裁 |
| `orch stats [--plan <id>]` | 统计（§4.4） | 任务/成员/历史三级耗时与 credits |

**底层命令面**（降级手册与 devops 调试用；组长规程明确**不鼓励**直接使用，防止形成
绕过 plan 的习惯）：`dispatch`、`sessions`、`archive`、`check`（抽检机械验收）、
`ledger`、`backlog`、`skills sync`。orch 瘫痪时全部可按附录 A 手动等价执行。

**改进机构面**（自改进体系收官并入的子命令；配合 §5.3 判决包与 §6 改进循环使用，
频次低于日常面）：

| 命令 | 作用 | 输出给组长的东西 |
|---|---|---|
| `orch jury <案卷> [--dry]` | Tier C 仲裁团（§5.3 的 **J** 通道）：家族外三席直调、双序对局、阻塞 ≈1 min | 结构化裁决（选择/理由/置信度）落 `state/evals/jury/<case>.json`；平票/inconclusive 过半 → 自动回退 twin 终裁（§5.3） |
| `orch taste <案卷>` | twin 审美判断（§1 角色行）：冷启动单跳、注入 PREFERENCE.md | `{选择,理由,置信度}` 落 `state/evals/taste/`，`pysci-mirror review` 可盲审 |
| `orch eval run --case <id>` / `orch eval set-baseline` | Tier B harness 套件执行与基线回填（§5.5；组员侧同面 `pysci-bench`） | 二元判决 + 指标表入 `state/ledger/evals.jsonl` |
| `orch bench-add` / `orch bench-queue` | bencher 队列指派与查看（§5.5） | 队列态与「待查看」结果（择机查看，不即时唤醒组长） |
| `orch inspector-up` / `inspector-off` / `inspector-status` | 巡检引擎生命周期与细读（§6；`up` 幂等、内嵌常用入口自动拉起） | 引擎态、全局开关、双队列水位 |
| `orch drain-stop` / `orch watch` | 后台 drain 协作式停止 / 事件等待器（§4.5） | 单行事件（QUEUE-COMPLETE / NEEDS-LEADER / QUOTA-STOP / CHANNEL-STOP / CRASH） |

**用户控制台是另一个 CLI `pysci-mirror`**（非 orch，用户侧入口）：`status` 全景
（用户唯一关注入口 + 注册纪律）、`review` 盲审（=训练）、`journal` 口述机械分流、
`decisions` 决策链全貌（§6 审计面）。

### 4.2 计划驱动（plan-runner）

组长的拆解结果**落成计划文件而非聊天记录**（对抗「近期消息淹没早期决策」）；推进时
orch 精准 recall 当前环节原文。

**串行优先（用户裁决）**：起步阶段 plan 系统**仅支持串行**执行（符合科研任务多数
情形）；并行分支（多环节同时推进、汇合点）留作未来扩展，registry/计划格式设计时
不堵死该方向（环节已具独立 id 与依赖可表达能力的位置）。

**任务书延迟书写（用户裁决）**：计划阶段每环节只写**粗粒度描述**（意图、输入/输出
预期、白名单目录）；完整任务书在**推进到该环节时**由组长书写——`orch plan run` 只
强制要求马上执行环节的任务书（前置环节未完成时，后续任务书往往既不可能也无必要
提前写）。orch 在交接暂停点提示组长：下一环节粗描述原文 + 「请书写任务书或选择
修改计划」。

```yaml
# orchestration/state/plans/<id>.yaml
goal: 用户原始需求的一句话概括
steps:
  - id: s1
    member: sim
    brief: 粗粒度描述：扫频仿真 X 参数，产出场数据（任务书执行时再写）
    dirs: [data/research/1_gain_ep/simulation]   # 白名单
    model: main             # 可覆盖默认角色（§3.5；main/pro/minimum，旧 flash/max 自动归一）
    effort: high            # 可选：本环节推理强度（§3.5 按项标注；不写=默认中）
    review: false           # true 时环节完成后自动 orch review
  - id: s2
    member: figure
    brief: 基于 s1 产物绘制冷原子图（交接物路径由 orch 自动注入）
    dirs: [data/research/1_gain_ep/article/figures]
    review: true
```

- `orch plan run` 逐环节执行；**交接处暂停**呈报组长：本环节摘要 + 下一环节粗描述
  recall + 选项（书写任务书继续 / amend 修改后续 / 中止）——组长可基于至今情况随时
  变通；
- 自动机械接力（不占组长注意力）：交接物路径注入、reviewer FAIL 回派、机械验收 FAIL
  回派、审批回复附送；
- 计划状态持久于 `state/plans/`——组长会话即使被压缩/重开，`plan run` 从断点继续。
  **计划即状态机，聊天记录不是**；
- 与官方动态工作流（Workflows）的边界：其 `agent()` 是同 harness 进程内子代理、后台
  运行无交接决策点，**不用于组员接力**；理念（计划脚本化、中间状态不进主会话）已由
  本节吸收。官方 Workflows 仍可用于组长会话内同 harness 扇出（多视角只读调研），
  与 orch 互补。

### 4.3 设计红线

1. 每条命令输出自含「下一步」——需决策的列选项，机械的已代办并说明；
2. fail-open：orch 故障时组长按**附录 A** 降级运转；state 全为可读格式；
3. orch 自身也是 devops 维护对象——对 orch 的改进走同一条 backlog 循环（§6）。

### 4.4 统计系统（用户裁决）

台账每跳记录 **credits + est_tokens + 耗时**。统计视图：
- 计划完成时自动呈现：本计划总耗时/总 credits/总 est_tokens；各成员在本计划中的占比；
  各成员有史以来累计占比；
- `orch stats` 独立入口随时查询；
- 用途：组长监测编排系统健康度——哪位成员更耗时/耗 credits（更需要优化，或在本项目
  中承担更重角色），为 harness 改进与模型预设调整提供数据。

**est_tokens 是相对用量尺子**（用户裁决 2026-10-10，backlog 20261010-ledger-est-tokens：
采纳估算做相对排名、放弃绝对精度）。精确 token 计量当前不可得——调研实锤：Qoder envelope
与 jsonl 逐轮 usage 在 BYOK/内置**双渠道全为 0**，平台网关无 usage/balance 路由（`/models`
返 200 而 8 个候选路径均兜底 400，含带 model 参数重试），网页仪表盘仅总览（平台产品策略，
**不对抗**）。故改测「本跳期间会话 jsonl 的字符增量」：dispatch 进出各读一次字符数，
`delta_chars`（原值）与 `est_tokens`（换算值）入台账，结束水位存 registry 会话条目
`chars_offset`（跨进程基准）。

- **换算系数** `est_tokens_per_char`：registry.json 顶层可配，默认 0.6 token/字符——
  混合中英的**启发式**，非测量值（标定实测：devops 主会话 jsonl 非 ASCII 占比 4.0%，
  中文以原生 UTF-8 落盘、不被 `\uXXXX` 转义膨胀，故字符计数可用）。台账存原值，改系数
  可重算历史而无需重跑派发。
- **只用于相对排名**：系数取何值都不改变排序，关键是**双渠道同法同尺**；它是 credits
  未计量渠道（实测=内置 `Qwen3.8-Flash`，恒 0）唯一可用的用量指标。
- **不做缓存命中修正**（本地不可得，对排名影响近似中性）。
- 增量基准优先取上一跳记账水位：崩溃跳/手动 resume 造成的「记账外增长」计入同会话
  下一跳，总量守恒（漏记会扭曲成员间排名）；水位高于实测起点只可能是文件被重写
  （压缩/清理），此时以实测起点为准。
- **它不是上下文读数**（insp-20261012-211429-mechanical）：est_tokens 量的是吞吐——本跳
  往会话档追加了多少字符。devops 78 跳实测与 `num_turns` 相关 **+0.81**、与 `ctx_ratio`
  相关 **−0.11**：大单必然读出大数。用它诊断膨胀等于给组员下一条「少干活」的伪判据；
  上下文占用看 envelope 的真实读数 `ctx_ratio`（判据面见 §3.2 归档条）。

**by_effort 聚合**（backlog 20261010-153041-devops）：`summarize` 除 `by_member`
外并列 `by_effort` 桶——按 `Ledger.effort` 分档，空标注（跟随用户级默认=中）归入
哨兵键 `"(默认)"`（永不与真实档 auto/none/low/medium/high 撞名）。每桶给
`{hops, sum_turns, avg_turns, fail_hops, fail_pct, duration_ms, duration_pct,
est_tokens_pct, credits_pct}`。`orch stats` 与 `orch ledger --stats` 两处**只在
≥2 档时**打印 by_effort 段（单档 ≡ 全体，无对比信号可看）。用途：**按项标注机制
（§3.5）的价值靠台账自然 A/B**——同类项 high vs medium 的轮均与返工率就是「档位
策略应否调整」的直接读数，替代此前手工导出 jsonl 逐行比对。

### 4.5 等待与唤醒模型（长任务零轮询）

问题：组员执行可能长达数小时（复杂仿真），组长（TUI 会话）必须「睡眠等待」而非
轮询（轮询 = 持续烧 token）。

**原生机制盘点（组长会话可用的等待通道）**：

| 机制 | 形态 | 等待期 token 成本 | 时长上限 | 适用 |
|---|---|---|---|---|
| 前台 Bash | 阻塞工具调用 | 零（阻塞中） | **600 s 硬上限**——直接回答：*不允许* 2 小时的同步工具调用 | 短任务 |
| **后台 Bash（run_in_background）** | 完成时**事件驱动通知**唤醒会话 | **零** | 无文档上限（**实测 130 s 与 3600 s 均存活并成功通知**） | **主路线** |
| Monitor 工具 | shell 侧监视进程，每行 stdout 成一条通知；persistent=会话生命周期 | 零（shell 轮询不耗 token，仅结果行触发） | persistent 无上限 | 辅助：监视状态文件/多任务 |
| CronCreate | 定时向会话投递提示 | 每次触发一轮 LLM | 7 天自动过期 | 兜底看门狗（低频巡检） |

**主路线设计**：`orch plan run` 以**后台 Bash 任务**方式启动——orch 进程内部同步
等待组员 headless 子进程（进程级阻塞，非 LLM 轮询），完成后执行全部机械后处理
（交付解析、机械验收、台账、状态落盘），将「下一步指引」写入 stdout 后退出 →
Qoder 后台任务完成通知**自动唤醒组长会话** → 组长读取指引、做交接决策。等待期间：
零 token 消耗；TUI 不被阻塞（用户可继续与组长对话其他事项）。

**状态与恢复**（等待链路的故障面）：
- orch 在 `state/plans/<id>.status` 写入运行态（环节、组员进程 pid、启动时间）；
- 组长 TUI 关闭/机器重启 → 后台链全部死亡，但**计划状态机与组员会话都落盘**：
  重开会话后 `orch status` 检测 stale running（pid 已死）→ `orch plan run --recover`：
  对组员用 `--resume <sid>` 续作（组员会话持久化与进程生命周期无关，§1.1 已证），
  或按组长决策重派；
- 兜底看门狗（可选，Phase 2 评估）：低频 cron（如每 30 min）提示组长运行
  `orch status`——防「通知机制自身故障」导致的永久沉睡；成本每次一小轮。

**组长等待纪律**（写入 orchestration 技能）：派发长任务必须走后台通道；**禁止
sleep-轮询循环**；等待期间不主动检查输出文件（通知会来）；用户插话优先响应。

**后台 drain 的等待：`orch watch` 事件等待器**（backlog 20261010-orch-watch）。
detached drain 不被任何前台进程持有，组长交办完即失去观察面——旧状只能靠「组长记得跑
`orch status`」，属纪律而非保证。硬化理念（用户裁决 2026-10-10：能代码硬保证的不靠 LLM
纪律；结构上限=唤醒注册只能由组长会话的工具发起，故代码能保证的是**事件判定全在代码 +
挂载提示在唤醒当刻由代码注入**，残余软点=组长照 [NEXT] 挂一条后台命令，与体系既有最小
软保证同级）：

- `pysci-orch watch [--timeout S=4h] [--interval S=60]` **阻塞轮询**（只读状态文件 +
  Python 侧判活，不起 shell 故 MSYS 路径转换问题免疫；间隔硬钳制 ≤60 s），命中即**单行**
  打印事件并退出 0，超时无事件打印 `TIMEOUT` 退出 0：
  - `QUEUE-COMPLETE <run>`——完成身份三元组 **(run 名, completed, count)** 相对基线变化
    （同 run 的 `.done` 被重写也能再报）。**收尾事实优先于锁的自称**：`.done` 由 drain 在
    `release_lock` 之后写出，摘要存在即已收尾，此刻锁仍显示 running 只可能是 unlink 失败
    残留 + 判活误真（Windows pid 回收/tasklist 超时保守判活）——旧判定让锁把事件压制后
    又无条件推进基线，导致 210557 那场 `fired=[]` 且补账同哑（backlog 20261011-220622-leader）。
    同一条「收尾事实优先」现已在**锁面**同样执法（§6：released 墓碑 + 「该 run 已有
    `.done`」判活否决），新代码不再产生谎锁帧；观察面的 `_in_progress` 抑制支与转移帧兜底
    因此退为保险——护升级前残留的旧基线，也顶住 `.done` 半写这类读不出证据的瞬态
  - `NEEDS-LEADER <id>`——backlog 中**新出现**的 needs_leader 条目
  - `QUOTA-STOP <run>`——额度类全局停止（`.done.stopped`，或 run 日志的 `[Q]` 痕迹——后者
    早一拍，不必等摘要落盘）
  - `CHANNEL-STOP <run>`——渠道级故障全局停止（同上，run 日志的 `[C]` 痕迹早一拍）：
    两类停止动作相同、恢复动作不同（充值/换渠道 vs 等渠道恢复），故分列两个事件
  - `CRASH <run>`——锁 pid 已死且该 run 无 `.done`
- **用法**：组长以 Bash `run_in_background` 启动，进程退出即原生完成通知唤醒本会话；一次
  watch 报一批事件即退，后续按需重挂（输出自带重挂命令）。`approve`（经 `wake_devops`）与
  `dispatch --from-backlog` 的输出**固定追加挂载指令行**（`watch.mount_lines`），不靠组长翻文档。
  **硬警示（mount_lines 首行）**：`orch watch` 必须作为后台 Bash 调用的**唯一顶层命令**——
  禁止用 `&`/`&&`/管道把它嵌进别的命令，嵌入即被父任务连坐杀死、事件通知随之丢失。
- **边界**：watch 随组长会话消亡，**不影响 detached drain**；基线快照与心跳落
  `state/watch.json`（gitignore：纯运行态）。**抑制帧与瞬态空值（`.done` 半写等）不推进
  基线**——被压住/读不出的身份保持原值，下一帧继续比较，绝不把账吞死。每帧现场（ts、
  锁态、锁 run、done 身份、抑制原因）进 `recent` 环形缓冲，命中/超时写 `exit_reason`；
  被外部杀死的那任没有退出记录，下次挂载或 `orch status` 会读出这条「遗言」。漏挂/会话
  中断期间的账由下次 `orch status` 的 watch 段**一次性补账**（报完即推进基线，重复运行不
  刷屏；watch 仍在挂载时 status 不抢账，免双份唤醒）。同刻多事件按 CRASH > QUOTA-STOP >
  CHANNEL-STOP > NEEDS-LEADER > QUEUE-COMPLETE 取主事件、其余以「同时」附注。**弃 Monitor 工具方案**
  （会话侧注册更重，且同样要组长发起）。

---

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
  **override 率**（`orch stats` 判决包健康段 / `pysci-mirror status`）；`orch reject` 无
  `--override` 参数即代码拒绝——**reject=override** 是数据事实，不是礼仪。

**例外通道**：缺陷类 × 同条交付含 `<blocked>` → 当条工作本身即 in-vivo 实测，**默认
approve**（规程建议连带选 M 复跑原失败案）；组长拒绝仍记 override。

**修复形状分流（remedy_shape，用户裁决 2026-10-13）**：inspector 机械通道的**免判决
只剩 `restore`**（复原既有声明行为：恢复被删行、修断正则、回滚误删配置）；凡
**`change`**（改声明行为/策略/权衡面——会话生命周期、阈值语义、charter 条款、路由语义）
而**无判决记录** → drain 任务书注入约束，devops 必须交付 `<blocked>` 上报而非自行实施
（裁定权在判决包与用户，不在执行端）。落点三面：登记字段缺省 change、任务书模板注入、
决策审计面高亮（§6 `pysci-mirror decisions`）。

**核验闭环（零新裁决点）**：核验方式在审批时一次定死——机械判据永远由交付协议自动执行，
重型 A/B 仅当审批选了 M 才跑；**实施前至多一次 J、实施后只实测不再仲裁**；改善 → 销账，
未改善 → 自动生成纠正提案（`source=remediation`）进 §6 管道。**模棱两可回退**：bencher
实测无显著指导意义、或 jury 平票/inconclusive 过半 → 自动回退 **twin 终裁**（不给平局，
结果必为选项之一，全量 mirror 留痕）。

### 5.4 Tier A：失败标签与阈值自动立案

- 词表 `orchestration/state/failure-tags.json`（**单一源，护栏级**，devops 增补附定义）：
  组员 `<blocked>` 可带 `tags="a,b"`（delivery-gate 校验词表，词表外退回、缺省不阻断）；
  聚合读数 `orch stats` 的 by_failure 段——**同一标签反复出现 = 深层结构问题的客观信号**；
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
- **生命周期**：FIFO 队列（`orch bench-add`），非空拉起、消化完销毁（**非常驻**）；
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
  orch consult 副组长评估）
→ 组长裁决 approve（入 backlog FIFO；推理密集项可 `--effort high` 按项标注，§3.5）
  / reject（附理由）
→ 审批回复存 state/replies/，下次派发该成员时自动附送（闭环告知）
→ approve 默认**自动唤醒** devops：分离后台进程 `_drain-devops` 持 state/devops.lock
  锁 FIFO 串行消化整个 backlog（组长零阻塞、零轮询，对齐 §4.5 等待模型）
  · --no-wake 只入队不唤醒；锁忙（已有 worker）不重复 spawn，本项由当前 drain 接手
  · 锁 stale（持有进程死 / 锁龄 >6h）自动接管，防 worker 崩溃后死锁
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
  · `orch drain-stop` 协作式停止：写 state/devops.cancel（指向当前锁持有者 pid），worker
    每项间隙消费→干净退出（释放锁、写部分 .done 含 stopped=cooperative、cancel 自删）；
    陈旧信号（pid≠本进程）清除且不误停接棒 worker；锁空闲时 drain-stop 不写信号（no-op）
→ drain 每批：取队首为**种子** + 收集同提请者(member)**且同 effort 档位**的 pending 为
  **组包**（含种子 ≤4 项，同提请者≈同模块/同视角；一跳只有一个跳级档位，混档会静默
  改变被标项的强度）→ 批量任务书（种子必做 + 组包菜单，devops 自主选取合并实施，
  单 worktree=种子 id）→ dispatch devops（§7）→ 测试验证 → git 提交（+ `timeout 25 git
  push origin main` best-effort，失败忽略，用户裁决 2026-10-10）→ 交付列 `backlog id=<ids>`
  · 成功 → 按交付 id 清单**逐 id 销账**（done，记 commit hash + batch_size/batch_seed）；
    组包中未选项保持 pending 留待下轮（devops 可只吃种子）
  · 失败（blocked/run_failed）→ 仅**种子** needs_leader、组包项保持 pending（**不连坐**，
    一项卡住不阻塞全队列）
  · 系统性故障（`quota_exhausted` 额度耗尽 / `channel_error` 渠道级故障）→ **全局停止**：
    种子复位 pending（非 needs_leader）、组包与剩余条目保持 pending、释放锁、.done 注明
    停止类别；恢复后下次唤醒自动续消化（两类只恢复动作不同：充值/换渠道 vs 等渠道恢复）
→ run 日志 + .done 摘要落 state/devops-runs/<ts>.{log,done}；orch status 展示 worker
  态（running/idle/stale，其中 stale 再分**崩溃**〔pid 死且缺该 run 的 .done〕、**残留已标记**
  〔released 墓碑：已宣告收尾但文件删不掉〕与**残骸已收尾**〔该 run 已有 .done，连墓碑都写
  不进的极端残骸；pid 不作判活依据〕）、
  最近 run 摘要（含停止方式：自然跑完/协作停止/额度耗尽/渠道级故障/异常中断）、
  needs_leader 提醒（组长据此重派/改任务书/上报）
→ 组长的等待面（代码化，§4.5）：唤醒/采纳/交办三处输出固定附 `orch watch` 挂载指令；
  watch 后台阻塞轮询 state 文件与锁，命中 QUEUE-COMPLETE / NEEDS-LEADER / QUOTA-STOP /
  CHANNEL-STOP / CRASH 即单行退出唤醒会话；watch 随会话消亡、不影响 drain，漏掉的账由下次 `orch status`
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
  `state/preference-journal.jsonl`）。**`pysci-mirror decisions` = 决策链全貌**（只读拼装：
  findings × backlog × run 摘要 × jury 案卷 × 决策日志）——发现 → class → remedy_shape
  → 判决记录 → 实施 commit → 核验状态，**免判决+change 置顶**（最需审计者优先）；
  `mirror status` 的「决策」项实时报水位。重要决策必须汇总进概览并可一屏看全
  （用户裁决 2026-10-13；ctx 水位闸绕过判决通道案即教训，§13 登记）。

**双队列迟滞节流（用户裁决）**：inspector 监视 devops backlog pending 与 bencher 队列
长度——任一 > 积压上限（backlog 12 / bench 6）→ **暂停产出**新提案（只读巡检继续）；
两队列均回落至**恢复阈值**（8 / 4）→ 恢复。**恢复阈值严格低于暂停阈值 = 迟滞**，防振荡。

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

## 7. devops 作业模式

**harness 隔离**：devops 会话 cwd 永远是**自己的 pod**（加载自己的 charter/devops
技能/MCP），worktree 只是**作业对象**（绝对路径或 `--add-dir` 访问）——绝不以
worktree 或仓库根为 cwd，因此不继承组长 harness。orch 是组长专属，devops 不用（E2 期
的 `orch eval` 过渡例外已随 E4 bencher 上线退役，见 §1 角色边界行）：backlog 条目经任务书注入，
完成经 outbox 交付。

- worktree 事实（实测）：`data/` 磁盘 1.4 GB 但 git 仅跟踪 16 MB；worktree 共享
  `.git` 对象库，每份仅检出 ≈30 MB——容量顾虑不成立。worktree 不含未跟踪重数据，
  需要时以绝对路径只读引用主树；
- 作业流：领任务 → `git worktree add` → 实施 → 完整测试 → 合并回 main → commit+push
  （用户已授权为其职责）→ 交付；
- worktree 内 venv 策略（editable 安装指向主树 src）：**实测定型=PYTHONPATH 覆盖法**
  （`PYTHONPATH=<wt>/src uv run --no-sync …`，复用主 `.venv`；**前提=cwd 在 worktree
  之外**，cd 进去时 uv 会就地新建空 `<wt>/.venv` 缺依赖即崩，2026-10-12 实测复现）——
  细则见 devops 技能「worktree 作业规程」与 §12 Phase 2；
- devops 专属 CLI `pysci-dev`（`src/pysci/skills/devops/`）：收纳其机械性作业
  （skills sync 的执行端、部署校验、巡检脚本入口等；与 orch 严格分离，服务对象是
  devops 自己）；
- devops 改他人 harness：只改真本/charter 模板 + sync 部署，禁手改部署副本；改组员
  自维护层须经该组员同意（或组长裁定）；
- devops 改组长 harness 或 orch 自身：提案→组长审批→（护栏级）用户授权→实施；
- 巡检职责（周期性，经组长计划派发）：pod AGENTS.md/自建 rules 体量与质量抽查、
  台账成本曲线审阅、transcript 抽查（§9 诚实条款的审计端）。

---

## 8. MCP 归属

实测事实：用户级 settings 的 MCP 全局生效；项目级 `.mcp.json`/settings 严格按 cwd、
不向下泄漏；per-agent 内联 `mcpServers` 仅支持 sse/http。

- 组员侧：每 pod 一份 `.qoder/mcp.json`（**只读层**，devops 维护，不下放自配——用户
  裁决），orch 派发时传 `--mcp-config .qoder/mcp.json --strict-mcp-config` → 成员 MCP
  集合完全由 pod 决定。归属：comsol→sim；blender-mcp→model3d、drawing；
  arxiv/zotero/paper-search→lit、reviewer、deputy；
- **多 MCP pod 启 `mcp.lazyLoad`**（实测 2026-10-10，backlog `20261010-mcp-lazyload-probe`，
  §1.1）：lit/reviewer/deputy 的 `.qoder/settings.json` 加 `"mcp": {"lazyLoad": true}`，
  首跳不再注入全量 MCP schema（实测省约 50% 起步上下文、真实 arxiv 检索结果不变）。
  **成员须知**：lazy 态个别 MCP 工具名不再直接可见，一律经元工具 `mcp_list`/`mcp_get`/
  `mcp_call` 访问（CLI 会注入 `critical_system_reminder` 提示）；各技能按「server+工具名」
  描述能力即可，勿写死 `mcp__<server>__<tool>` 直调串。sim/model3d/drawing 未实测，暂不启用；
- 组长侧：**最终不保留任何末端 MCP**（用户裁决：收尾阶段用户级清零）。过渡期可用根
  settings 的 `mcp.excluded` 屏蔽（§1.1）。组长需要文献/仿真信息一律经组员获取；
- license 纪律：comsol MCP 依赖 COMSOL 单许可证，sim 组员 charter 保留 mcp ensure
  监督、闲时释放纪律；
- Phase 0 实锤：`--strict-mcp-config` 完全屏蔽用户级 server 的行为。

---

## 9. 强制层

- **pod-guard**（PreToolUse matcher `Write|Edit|NotebookEdit`，`orchestration/guards/`
  共用脚本）：按 §2 矩阵执行两层结构——自维护层（bench/outbox/AGENTS.md/自建
  rules/自建 skills）放行；只读层（charter.md/部署技能/settings/mcp.json/inbox）与
  pod 外路径（白名单目录除外）exit 2 + stderr 注入。保护名单由 settings 接线时经
  环境变量注入（`PYSCI_TASK_DIRS` + `PYSCI_READONLY_PATHS`）；`orch skills sync` 只
  触碰 manifest 部署名，**不删除成员自建技能**；
- **delivery-gate**（Stop hook）：交付格式强制（§3.4）**＋组员自维护层预算硬闸四条**
  （§3.1/§13 裁决，2026-10-10）：AGENTS.md ≤8192B；自建 rules 禁 `always_on`
  （槽位专属 charter）；自建 rules/技能 description 合计各 ≤8192B；自建技能每个
  SKILL.md ≤8192B——违规 exit 2 + 精确整改指令（文件、实测字节、上限、动作）；
- **harness-budget**（pre-commit，真本侧）：`orchestration/skills/**` 的 SKILL.md 与
  references、根 `.qoder/rules/` 每文件 ≤8192B 硬拦——管住组长与 devops 的真本编辑，
  与 delivery-gate 同阈值常量；
- **doctor 预算巡检**：`pysci-dev doctor` 全域审计三档（全量注入/常驻暴露/按需），
  超限 ✗、>90% ⚠；
- 权限模式：用户级默认 yolo（用户裁决），orch 不显式传参；hooks 在 yolo 下照常执行
  （v1 实测，Phase 0 以当前参数名复测）；
- 诚实条款：guard 只拦 Write/Edit/NotebookEdit，Bash 重定向理论上可绕过——设计目标是
  把失效模式从「遗忘/失误」升级为「刻意违规」，后者经 transcript 审计发现（devops
  巡检职责）；
- 组长侧不装硬 guard（用户在场的会话靠规程 + 报告义务 + 护栏授权红线）。
- **恢复面三闸（收官并入）**：
  - **WIP-salvage 自动抢救**：drain 获锁回收孤儿条目时**先抢救再清理**——把前任 worker
    worktree 内未提交成果快照留档并自动开续作单（授权 devops 在遗留面上续完、不重做），
    然后才 `worktree remove` 残留（worktree 名 = 条目 id，只清孤儿名不误删活跃面）；
  - **sid 在途闸**：派发前对目标会话做**判活 + argv 双闸**——同一 sid 已有在途会话时
    人工派发 rc=2 拒、批次派发自动换新 sid 并标 `sid_new_reason=prev_alive`；同理「一个
    worktree 一个写者」（并发写者即停手上报），防双写互踩；
  - **渠道级故障识别（CHANNEL-STOP）**：`channel_error` 判据 = 固定文案 ∧ 会话档末轮
    故障形状**合取**（单看文案会被复盘它的任务书正文误报）；命中 → drain 全局停止（种子
    复位 pending、释放锁），watch 打 `CHANNEL-STOP` 事件（§4.5）——恢复动作是**等渠道
    恢复**，与额度类的充值/等待重置分列两事件。
  - **交集成因停链（抢救面的例外闸，backlog 20261013-102152-devops）**：blocked 带
    `tags="merge_dirty_face"` **或**正文含 `pysci-dev worktree merge` 的 `[✗] 交集 N 项` 行 →
    认定成因在 **main 的工作区脏面**：续作单不立（下一跳改变不了 main，实测三连跳树哈希
    逐字节相同=空转），进度照抢救、面与分支保住，待清文件进条目 `merge_intersect` 并渲染到
    `orch status`；处置=组长清面后 `pysci-dev worktree merge <面>`。判定细则见
    `skills/orchestration/references/tier-a-tags.md`。

---

## 10. 端到端工作流

```
用户需求（可能含多环节）
→ 组长拆解 → orch plan new <计划文件>（环节粗描述、成员、白名单、review 旗标、模型角色）
→ orch plan run：
   ① [决策] 组长书写**当前环节**任务书（延迟书写，§4.2）→ [自动] 派发组员
      （任务书+审批回复+交接物路径注入；会话选择按 §3.2 默认策略呈报组长确认）
      → [自动等待] 组员执行期间组长经后台任务机制休眠，完成时事件驱动唤醒（§4.5）
   ② [自动] delivery-gate 保证格式 → orch 解析 result|blocked 互斥分支
   ③ [自动] 成员声明的机械验收：FAIL→自动回派（≤3 次）
   ④ [自动/例外] review:true → orch review → FAIL 回派+复审一次 → 二次 FAIL
      呈组长【决策】仲裁
   ⑤ [自动] infra_suggestion 非空 → 呈组长【决策】approve/reject
   ⑥ [暂停] 交接处呈组长：本环节摘要 + 下一环节原文 recall +【决策】
      一键继续 / plan amend / 中止
   ⑦ [blocked 时]【决策】补充重派 / orch consult 副组长 / 上报用户
→ 计划完成 → orch stats 自动呈现本计划统计 → 组长向用户交付汇总
  （成果路径 + 统计 + 未决事项 + 组长自己的改进建议）
```

组长实际决策点：计划拆解与修改、会话归属确认、交接确认、INFRA 审批、仲裁升级、
最终汇报。其余全部由 orch 自动完成或自动提示。**计划外事项的正确姿势 = plan amend
或单步 adhoc 计划**，不是手动执行底层命令（§4.1）。

---

## 11. CLI 调用范式（全员统一）

实测（2026-10-07）：三种范式同速可用（裸 exe 路径 / `uv run --no-sync pysci-X` /
`uv run --no-sync`，0.6~0.75 s）；裸关键字此前失败仅因 `.venv/Scripts` 不在 PATH。

**已实施（用户授权的全局改动）**：`~/.bashrc` 追加一行
`export PATH="$PATH:/d/XXXIIIGGG/projects/pySci/pySciWS/.venv/Scripts"`
（Git Bash 自动补建了 `~/.bash_profile` 加载链）——**新开的 Qoder 会话**（含 headless
pod）中裸 `pysci-X` 直接可用；已实测新登录 shell 解析成功。

配套纪律（写入 basic.md §3、charter 与技能真本——**四面同口径**）：
- 首选裸 `pysci-X <子命令>`；命令不存在时 fallback `uv run --no-sync pysci-X`（**裸名只试一次，
  禁止反复重试**——历史重试的根因即裸名失败循环与 PowerShell 陷阱）；
- orch/dev 内部子进程调用统一 `uv run --no-sync pysci-X`（不依赖会话环境快照，最稳健）；
- **改 `pyproject.toml`（含新增 console script/依赖）后必须先 `pysci-dev venv-sync`**：不带
  `--no-sync` 的裸 `uv run` 会先隐式 sync，Windows 上撞运行中 exe 的锁并**先删光其余入口**，
  造成工具面集体失联（2026-10-12 实测；细则与自救配方 `orchestration/skills/devops/references/uv-venv-sync.md`）；
- 组员 Bash 即 Git Bash，避免 PowerShell（basic.md §3 的引号/编码陷阱）。

---

## 12. 阶段计划

### Phase 0 —— 验证冲刺（**已完成**，2026-10-07/09，沙盒 `orchestration/tmp-p0` 已清理；结果如下）
- [x] 泄漏面：rules/AGENTS.md 向上注入 pod（是）；skills/项目级 MCP/settings 不泄漏
      （否）；pod 嵌套目录继承信任、自身配置正常加载（标记实测）
- [x] 裸命令 PATH 方案：~/.bashrc 追加后新登录 shell 可用（实测）
- [x] 自动压缩：本会话 jsonl 含 `isCompactSummary`×2 与数十 compaction 标记——摘要式
      延续、非截断，运行时已实际发生；envelope 另提供 `usage.context_usage_ratio`
      可做归档提示阈值
- [x] 会话池全操作：`--session-id/--name/--resume/--fork-session/--list-sessions/
      --delete-session` 全部可用；**坑**：list/delete 是跨项目聚合视图且按序号删除
      （序号漂移）→ orch 会话池索引以 registry.json 为权威，prune 须先按 sid 匹配
      序号，匹配失败仅做 registry 标记不删文件
- [x] yolo 用户级默认在 headless 生效：无权限旗标执行 Bash，`permission_denials` 为空
- [x] envelope 字段清单：`result / session_id / num_turns / is_error / stop_reason /
      duration_ms / duration_api_ms / total_credits / total_cost_usd / usage(含
      context_usage_ratio、cache 字段) / permission_denials / modelUsage`
- [x] `--mcp-config + --strict-mcp-config`：pod 会话仅见自有 server（工具可枚举=已
      连接），用户级 5 个全屏蔽；根 settings `mcp.excluded` 可精确 block 指定 server
- [x] delivery-gate（Stop hook）：无标签收尾被 exit 2 拦截、stderr 注入改正指令、
      成员补标签后放行（turns 1→2）；防循环 `stop_hook_active` 生效；yolo 下照常执行
- [x] pod-guard（PreToolUse）：只读层写拦截（stderr 原样透传给成员）、白名单区放行；
      JSON 解析失败时 fail-open（诚实条款一致）
- [x] 组员自维护层三形态：always_on 注入 / model_decision 只注描述、正文按需 /
      glob 规则在读取匹配文件后加载；自建 skill 正常发现
- [x] pod cwd 下 `uv run` 向上解析根项目、共享 `.venv`（`pysci.__file__` 指向主树）
- [x] `--add-dir` 语义：**yolo 默认下不是硬边界**（外部 cwd 无 add-dir 亦可读写项目
      文件）→ 路径纪律的唯一硬防线是 pod-guard；「副组长全项目访问」无需 add-dir
- [x] `-m` 档位传参：`-m Qwen3.8-Flash` headless 可用；`--list-models` 含
      Qwen3.8-Max/Flash（双档映射成立）
- [x] 后台长任务：130 s 与 **3600 s**（BG-OK-3600）后台 Bash 任务均存活并事件驱动
      通知，等待期零轮询；TUI 空闲下通知正常送达
- [x] 等待-恢复链路：多跳任务中途按 PID 精确击杀 → 会话 jsonl 增量落盘 →
      `--resume` 精确还原中断现场（已完成步骤定位 + 暗号记忆无损）
- [ ] 长会话成本曲线：resume 成本随跳数增长——转 Phase 1 运行期持续观察（envelope
      的 usage/cache 字段直接入台账）

### Phase 1 —— orch 骨架 + figure 纵切面（**基本完成**，2026-10-09）
- [x] `src/pysci/skills/orchestration/`：registry / dispatch / envelope 解析 / 交付解析 /
      会话池封装 / 台账（credits+耗时）/ 下一步提示（plan 与 approve 后置 Phase 2）；
      ruff 全绿；`pysci-orch` 入口注册（pyproject）
- [x] `orchestration/pods/figure/`：charter 第一版（交付协议/自检清单/CLI 范式/轻量
      笔记原则）、AGENTS.md 种子、pod-guard + delivery-gate 接线（guards 正式版单测
      8/8）、mcp.json（空集）、bench/inbox/outbox
- [x] 技能真本迁移第一份：scientific-plotting → `orchestration/skills/` + manifest +
      sync（部署/一致性校验通过；根技能列表热更新确认移出）
- [x] orchestration 技能真本第一版（组长操作知识）+ 部署到根 `.qoder/skills/`
- [x] paths.py 新增 ORCHESTRATION_ROOT / ORCH_STATE_ROOT / PODS_ROOT 锚点
- [x] 冒烟派发全环路：任务书落 inbox → 新会话（Flash 档位路由生效）→ charter 加载 →
      bench 可写 / charter 只读拦截（组员侧逐字回报）→ 交付协议解析 → 台账/会话池登记
- [x] 真实图件任务全环路（fig0_orch_smoke）：首跳 blocked（白名单与两层结构冲突，
      触发改进循环首次实战，见 §2 脚注）→ 审批+扩权续派 → 正规管线 build →
      视觉自检 → **figure-audit 机械验收 √ PASS** → 台账/审批回复附送/归档全链路自动
- [x] credits/耗时基线登记：冒烟跳 7 轮/48 s；fig0 阻塞跳 23 轮/251 s；收尾跳
      15 轮/137 s；会话 ctx 三跳后 52%；**观察**：envelope `total_credits` 恒为 0
      （疑似该账号/模型不上报计费；成本统计暂以耗时+轮数为准，持续观察）
- 验收：✅ 达成——真实任务全环路无人工兜底（blocked 分支的组长决策属设计内环节）

### Phase 2 —— 计划驱动 + 改进循环 + devops（**完成**，2026-10-09）
- [x] orch plan new/run/amend/drop/list/adhoc：状态机全生命周期实测（登记/延迟书写
      强制/对账/中止审计）；带派发的完整 run 与 dispatch 同源（dispatch 已实战）
- [x] orch approve/reject/suggestions/backlog/consult/stats：命令面实装 + 冒烟通过
      （consult 在 deputy 缺位时优雅降级）；approve 全链路待首个自然建议到来时实战
- [x] `orchestration/pods/devops/`：charter（worktree/git 纪律/巡检职责）、devops 技能
      真本+部署、`src/pysci/skills/devops/` + `pysci-dev`（doctor/sync/worktree）；
      registry 注册（max 档 + extra_dirs=项目根）
- [x] worktree venv 策略**实测定型：PYTHONPATH 覆盖法**——
      `PYTHONPATH=<wt>/src uv run --no-sync pytest <wt>/tests`（复用主 .venv，worktree 代码优先；1452 项收集 7.5 s，
      全量 45 s）；worktree 内裸 uv run 不可用（无环境）；仅依赖变更时才 `uv sync
      --project <wt>`（重）。已写入 devops 技能
- [x] **首个完整改进循环实战**：backlog 队首（figures new 路径打印缺陷）→ orch
      dispatch devops --from-backlog（任务书自动生成）→ devops worktree 实施 +
      回归测试新增 + 全量 1416 passed → 合并 commit（cz 规范，未 push）→ 交付含
      backlog id → orch 自动销账。实测：31 轮 / 855 s / Max 档 / ctx 29%
- [x] orch archive/sessions/prune + 归档前蒸馏跳：实测通过——蒸馏跳中成员**自主整合
      升华**了此前任务经验进 AGENTS.md（STYLE.yaml 缺陷规避、数据管线约定），
      「越来越聪明」机制成立；归档后池默认正确回退工作会话
- [ ] 模型双档路由实装（registry 映射 + `-m` 传参）

### Phase 3 —— reviewer + 副组长（**完成**，2026-10-09）
- [x] `orchestration/pods/reviewer/`：charter（VERDICT 协议/独立性纪律/审查流程）、
      `rubrics/generic.md` **最小占位版 v0.1**（用户裁决：跑通流程优先，正式评审体系
      ——科学正确性/创新性/美观性 + 专属技能与程序基础设施——待框架稳定后用户与
      组长深度共议）、文献 MCP（arxiv/zotero/paper-search）、registry（max 档 +
      `readonly_extra: ["rubrics"]` 只读保护——成员级只读层扩展机制首用）
- [x] `orch review`：VERDICT 解析（单测过）+ FAIL 自动回派 + 复审一次 + 升级仲裁
      状态机；审查记录持久化 `state/reviews/`
- [x] **review PASS 链路实战**（reviewer 首战，fig0_orch_smoke）：独立复跑 audit CLI、
      手算抽查数据自洽性、Read 图件视觉判读——占位 rubric 下仍展现审查素养；
      7 轮/106 s。reviewer 并提出高质量建议（review 任务书应嵌原生产任务目标陈述，
      否则 G2 依赖生产者自述、与独立性纪律冲突）→ **approve 全链路实战通过**
      （建议→审批→backlog 第 6 位→回复入队附送）
- [ ] FAIL→回派→复审链路：代码已实现，待首个自然 FAIL 实战（占位 rubric 判定从宽，
      人为构造 FAIL 成本不划算——Phase 3 验收以 PASS 链 + 状态机代码审查为准）
- [x] `orchestration/pods/deputy/`：全 7 技能部署（**9 技能真本迁移提前全部完成**，
      根 `.qoder/skills/` 只剩 orchestration 部署副本）、平级协作条款 charter
      （拒绝权流程/异议义务/全项目视野）、extra_dirs=项目根、MCP 不配 comsol
      （license 纪律，仿真走 CLI 或改派 sim）
- [x] **consult 实战**（deputy 首战）：异议义务成立——副组长按 charter 先校正了任务书
      两处前提错误（根技能已清空、README Phase 3 未登记），再以实证给出
      「**脚手架批量建、战备逐个验收**」的 Phase 4 策略（引 figure pod 首任务即
      blocked 的先例：charter/白名单冲突只有实战暴露）。组长采纳。
- [x] 跨域双路径对比演练 → **延后至 Phase 4 末**（全员就绪后成本对比才有意义；
      用户裁决跑通流程优先）

### Phase 4 —— 全员迁移 + 组长 harness 瘦身（**基本完成**，2026-10-10）
- [x] 六专职 pod **脚手架批量建**（devops worktree 任务：sim/writing/theory/lit/
      drawing/model3d，charter 含各域自检清单，MCP 归属落位，manifest targets 追加 +
      sync；doctor 十 pod 全绿；34457f5 合并）
- [x] **lit pod 战备验收**（首个战备抽查）：MCP 隔离会话级实锤（只见 arxiv/zotero，
      无 comsol/blender）、charter 加载、arxiv 实检索成功；成员自主把 429 限流经验
      蒸馏入 AGENTS.md。**其余五 pod 战备验收随各自首个真实任务进行**（预期暴露各域
      特有冲突，参照 figure/lit 首战模式处理）
- [x] 根 `.qoder/skills/` 清空（Phase 3 提前完成）
- [x] **用户级 MCP 清零**（用户授权任何时机执行；备份存 state/backups/，gitignored）：
      根侧新会话实测零末端 MCP；lit pod strict 配置不受影响。组长本 TUI 会话的 MCP
      工具随下次重启消失。paper-search-mcp 预先存在的连接失败入 backlog
      （20261010-paper-search-mcp-down）
- [x] basic.md 重写：§3 PowerShell 细节压缩为「Git Bash 为唯一工作 shell + PS 历史
      纪律四行摘要」（完整实测结论存 git 历史）；CLI 裸命令范式入公约数；§1/§2/§6
      原文未动
- [x] orch 移交 devops 维护：自本登记起，orch/pysci-dev 代码改动一律走 backlog →
      devops worktree 实施（组长不再直接改 orch 代码；bootstrap 例外条款作废）
- [x] devops 巡检例程：pysci-dev doctor 已上线（Phase 2）；巡检节奏 = 组长按需派发
      + 每个 Phase 收尾必跑；transcript 抽查入 devops 技能清单
- [ ] 跨域双路径对比演练 → **延后至首个真实跨域科研任务**（届时 naturally 产生
      「副组长单体 vs 计划接力」对比数据，stats 记录）
- [ ] （可选加固）组长侧 permission deny：禁 Write/Edit 于生产目录——暂缓（组长
      仍需写 orchestration/state 与本文档；deny 规则需精细豁免，收益/复杂度比待评估）

### 自改进体系 E0–E7（原 REFACTOR，**收官**：2026-10-11 定稿 → 2026-10-13 全链完成并入本文）

逐阶段成功判据与实施实录见 git 历史中的原设计文档；下表是收官状态（机制落点见括注）：

- [x] **E0 纪律与原料层**：`success_criteria` 必填、`kind` 结构化、reject=override 强制
      留痕、决策日志回灌、deputy/reviewer 降档（§5.3/§6）；
- [x] **E1 Tier A 全链**：failure-tags 受控词表、`<blocked tags>` 解析、`orch stats`
      by_failure、阈值自动立案（§5.4）；
- [x] **E2 minimum 档 + harness 套件骨架**：`orch eval run`、`state/evals/harness/` 首批案、
      baseline.json（§5.5）；
- [x] **E3 仲裁团**：直连三席、双序对局多数解析、jury-brief「仅涉自身基建直通」规则、
      Key 泄漏哨兵（§5.3 J 通道）；
- [x] **E4 bencher**：pod/FIFO 非常驻、基线捕获、副本 A/B、择机查看型通知（§5.5）；
- [x] **E5 决策流硬化**：判决包四态闸、inspector 全流程（自动晋升/双队列迟滞/全局开关/
      设计审查 rubric）、E5b 交付后核验闭环与缺陷×blocked 例外通道（§5.3/§6）；
- [x] **E6 twin + mirror + 回退链**：`orch taste` 冷启动、盲审/口述分流/偏差报告绿色
      通道、`escalate=twin` 自动终裁、`pysci-mirror` 首批全景项（§3.1/§4.1/§6）；
- [x] **E7 twin 上线**：代理裁决接入原「呈用户」升级点（护栏级审批/升级仲裁/Phase 验收），
      最终交付验收仍呈用户本人；2026-10-13 PREFERENCE v2 经用户确认转正、twin 全量（§1/§13）；
- [x] **收尾：融合与退役**——生效机制并入本文（§1 角色表、§3.1、§4.1、§4.5、§5.3–5.5、
      §6、§9、§13），原设计文档删除、仅存 git 历史（即本轮收官单）。

**实施期夜航要点（审计留痕，原委记录要点）**：
- E0 首战四缺陷：registry lost-update（drain 陈旧快照回写抹配置）、watch QUEUE-COMPLETE
  脱靶吞账、锁面 released 墓碑、`.done` 收尾硬证据否决判活——全部修复并有回归（§4.5/§6
  同源纪律）；
- 2026-10-12 00:04 E1+E2 组包 `run_failed(tool_use)`：E1 半成品经抢救单续作、E2 重排队
  ——该次处置直接促成 §9 的 **WIP-salvage 先例**；
- 委托期（10-11 23:2x → 10-12 07:00）到期撤销，期内自主决定全量可追溯于决策日志
  （`kind=delegation`）；**后续愿景（用户裁决 2026-10-12）**：把「永不阻塞」做成受硬机制
  保证的项目特性——除最终交付外系统不因等待用户停摆，代理授权点全量留痕、事后可追溯；
- 2026-10-13 收官审查：ctx 水位闸绕过判决通道案 → 回退该闸 + **修复形状分流**（§5.3）+
  **后台决策审计面** `pysci-mirror decisions`（§6）——三项均经用户裁决并落码。

---

## 13. 裁决记录

| 日期 | 裁决 | 裁决人 |
|---|---|---|
| 2026-10-05 | v1 时期裁决（代码事务免审批、pod 联邦、量表用户审定等），有效部分已并入正文；原文见 git 历史 | 用户/组长 |
| 2026-10-07 | 编排主力 = headless 星形拓扑；组长严禁末端工作（例外：用户点名亲做）；本项目内组长决定权高于用户（建议权/无强制否决） | 用户 |
| 2026-10-07 | 硬 QA 门取消 → 可选 reviewer；客观检查下沉；机械验收改**成员声明式**，不得阻碍灵活任务，组长可抽检 | 用户 |
| 2026-10-07 | pods 从属 orchestration/；组员 MCP 下放 pod、用户级最终清零、MCP 自配权不下放；组长零末端 MCP | 用户 |
| 2026-10-07 | 优先复用原生机制；pod=完整项目；会话管理复用原生（orch 仅薄封装）；默认策略=继续最近活跃会话，组长确认归属 | 用户 |
| 2026-10-07 | 副组长近平级：拒绝权（附理由留痕）+ 异议义务；组长最终裁定权 | 用户 |
| 2026-10-07 | 交付协议：result/blocked 互斥 + XML 标签块；delivery-gate Stop hook 强制格式，解析失败通知组员改正 | 用户 |
| 2026-10-07 | infra_suggestion 从宽受理：缺陷类附证据，难用类描述场景即可；采纳由组长裁定 | 用户 |
| 2026-10-07 | 组员 harness 两层结构：只读层（charter/部署技能/settings/mcp）+ 自维护层（AGENTS.md/自建 rules/自建 skills）；轻量笔记原则，不自建代码工具 | 用户 |
| 2026-10-07 | devops 不使用 orch；cwd=自身 pod、worktree 为作业对象 | 用户 |
| 2026-10-07 | 改进循环覆盖全员；组长自提→用户审批，不设 orch 硬保护（软性规程约束） | 用户 |
| 2026-10-07 | 一成员一技能一 CLI（除 deputy/reviewer）：9 技能真本统一 orchestration/skills/，manifest 部署到 pod 与项目根；orch 实现迁至 src/pysci/skills/orchestration/（pysci-orch）；devops 有 pysci-dev | 用户 |
| 2026-10-07 | plan-first：组长一切工作经 plan 系统；底层命令仅降级/调试用；计划外事项 = amend 或 adhoc 单步计划 | 用户 |
| 2026-10-07 | plan 系统起步阶段仅串行；并行分支为未来扩展（格式不堵死） | 用户 |
| 2026-10-07 | 任务书延迟书写：计划阶段只写环节粗描述，推进到该环节时才强制任务书 | 用户 |
| 2026-10-07 | 等待模型：长任务派发走后台 Bash 通道（事件驱动唤醒，零轮询）；前台工具调用 600 s 上限不支持小时级阻塞；pid 状态 + --recover 恢复；cron 看门狗为可选兜底 | 用户要求调研 + Agent 实测（130 s 后台任务突破前台默认超时并成功通知） |
| 2026-10-07 | 统计系统：每跳 credits+耗时；计划级/成员级/历史级三层占比；orch stats 入口 | 用户 |
| 2026-10-07 | 模型双档抽象（max/flash）：组长/deputy/reviewer/devops=max，专职组员=flash；orch 内路由型号；计划环节可覆盖 | 用户 |
| 2026-10-07 | yolo 为用户级默认，orch 权限参数降为可选 | 用户 |
| 2026-10-07 | 技能真本 git 跟踪、副本 gitignored、sync 再生；星形拓扑；reviewer 复审一次后升级；降级手册；官方 Workflows 不用于组员接力（理念由 plan-runner 吸收） | 用户（批准 Agent 提案） |
| 2026-10-07 | CLI 范式：~/.bashrc PATH 持久化（用户授权的全局改动）+ 裸命令首选/uv run fallback/禁重试循环；orch 内部用 uv run | 用户授权 + Agent 实测 |
| 2026-10-05~07 | 实测事实登记：resume 接力、headless Teams、Agent 工具单层、exe 直调、泄漏面矩阵（§1.1）、CLI 三范式同速、自动记忆仅交互式、自动压缩为运行时行为（文档级） | Agent 实测/查证 |
| 2026-10-09 | **Phase 0 验证冲刺完成**（16 项全过，详见 §12）：envelope 字段全清单（含 context_usage_ratio/permission_denials/total_credits/duration_ms）；yolo 默认在 headless 生效；delivery-gate 与 pod-guard 全链路拦截实锤；strict-mcp-config 与 mcp.excluded 双通道实锤；自维护层三形态加载实锤；--add-dir 在 yolo 下非硬边界（路径纪律唯一硬防线=pod-guard）；list/delete-session 聚合视图与序号漂移坑（orch 池索引自管）；自动压缩 jsonl 实证（isCompactSummary）；后台任务 3600 s 存活+事件通知；kill-resume 精确恢复中断现场 | Agent 实测 |
| 2026-10-09 | 任务级授权例外（组员白名单可扩至研究线代码镜像 src/pysci/research/<线>/） | 用户追认 |
| 2026-10-09 | credits 恒 0 根因 = 用户级默认模型为 BYOK（不计入 Qoder 订阅计费）；BYOK 成本统计方案入 backlog（20261009-byok-cost-accounting），不阻塞主线 | 用户说明 + 组长登记 |
| 2026-10-09 | **Phase 2 完成**：plan 状态机、改进循环命令化、devops pod + pysci-dev、worktree venv 策略定型（PYTHONPATH 覆盖法）、首个完整改进循环实战（devops worktree 修复合并 + 自动销账）、归档蒸馏实测（成员自主整合升华经验进 AGENTS.md） | Agent 实测 |
| 2026-10-09 | reviewer rubric 现阶段为**最小占位版**（跑通流程优先；正式评审体系——科学正确性/创新性/美观性及专属技能基础设施——待框架稳定后用户与组长深度共议） | 用户裁决 |
| 2026-10-09 | **Phase 3 完成**：reviewer/deputy pod 实战验收；review PASS 链 + approve 全链 + consult 全链通过；9 技能真本迁移提前完成（根技能列表仅剩 orchestration）；readonly_extra 成员级只读扩展机制上线 | Agent 实测 |
| 2026-10-09 | Phase 4 策略采纳副组长咨询意见：**脚手架批量建、战备逐个验收**；文档滞后教训入状态条款（阶段完成当轮登记） | 组长裁决（consult 输入） |
| 2026-10-10 | **Phase 4 基本完成**：六 pod 脚手架（devops worktree，十 pod doctor 全绿）、lit 战备验收（MCP 隔离会话级实锤 + 成员自主蒸馏经验）、用户级 MCP 清零（用户全权授权，备份留档）、basic.md 重写（PS 细节压缩、Git Bash 唯一 shell、裸命令范式入公约数）、**orch 移交 devops 维护（bootstrap 例外作废，orch 改动一律走 backlog）** | Agent 实测 + 用户授权 |
| 2026-10-10 | **credit 耗尽事故与模型渠道裁决**：registry 曾把双档映射到内置 Qwen3.8-Max/Flash（订阅计费）而 envelope 低报 credits=0，drain 首战 88 分钟烧穿订阅额度 → 全员派发连败。用户裁决渠道策略：**max=BYOK Qwen-3.8-Max（不传 -m，走用户级默认）；flash=内置 Qwen3.8-Flash（限时免费）；flash_fallback=BYOK flash（渠道待开通，代码先行）**。registry models 改动自此属护栏级（须经用户）。配套改进入 backlog：额度失败识别 + drain 全局停止（quota-awareness）、flash 回退机制（flash-fallback） | 用户裁决 + Agent 复盘 |
| 2026-10-10 | **渠道策略修订 + devops 降档**：BYOK 双模型就位后 max/flash_fallback 改为显式 UUID 绑定（982ae403/ed6f5996，经 refresh-models 自愈机制维持）；三通道均实测可用。devops 默认档 max→flash（末端实现属性 + 成本；升档通道保留，台账观察 2~3 周）。推理强度双档默认「中」，不做全局 high，按项标注机制入 backlog | 用户裁决 + Agent 实测 |
| 2026-10-10 | **BYOK 计费=估算做相对排名，放弃绝对精度**：平台网关无 usage/balance 路由、网页仪表盘仅总览、Qoder 双渠道 token 计数全 0（均实测）——用户判断为平台产品策略（阻碍 API 转售/法律风险考量），不对抗；est_tokens（jsonl 字符增量，双渠道同法）入台账/stats，缓存修正不做（排名影响中性）。阿里云 BSS 账户级路径不评估（凭据过重） | 用户裁决 + Agent 调研 |
| 2026-10-10 | **drain 观察闭环硬化**：`orch watch` 阻塞式事件等待器（QUEUE-COMPLETE/NEEDS-LEADER/QUOTA-STOP/CRASH），组长经后台 Bash 挂载、退出即原生通知唤醒；唤醒类命令 [NEXT] 代码注入挂载指令。结构上限登记：唤醒注册只能由组长会话工具发起，外部代码无法代按——硬化到「判定在代码+提示在代码」，残余软点与体系最小保证同级 | 用户理念 + Agent 设计 |
| 2026-10-10 | **修正 2026-10-09「credits 恒 0 根因=BYOK」判断**：实测 14 跳台账证明 BYOK 下 credits **依模型上报**——max 档（Qwen3.8-Max）正常计量（devops/reviewer/deputy 跳 13~1142 credits），flash 档（Qwen3.8-Flash）恒 0（figure/lit 跳）；usage tokens 两档均 0、唯 context_usage_ratio 恒有效。台账新增 `model` 字段 + `stats` 标注成本覆盖率（`format_credits`），使部分计量数据可解释。完整 token/账单核算仍按用户 2026-10-09 裁决留待未来（backlog 20261009-byok-cost-accounting 据此销账） | devops 实测（backlog 20261009-byok-cost-accounting） |
| 2026-10-10 | **review 任务书嵌入目标陈述**（G2 独立性）：`orch review` 增 `--goal-from <原生产任务书>`/`--goal <文本>`，把组长撰写的目标陈述摘录嵌入审查任务书并登记进 review 记录（`goal_source`/`goal_excerpt`），使 reviewer 的 G2「与任务书目标一致」据组长原意独立核验、不再依赖生产者 notes.md 自述（消除与 charter 独立性纪律的张力）；缺省告警不中断。采纳 reviewer 建议 20261009-231455-reviewer | 组长采纳 + devops 实施（backlog 20261009-231455-reviewer） |
| 2026-10-10 | **Harness 预算三层制 + 执法 + 身份锚定**：全量注入档每文件 ≤8 KB（charter/AGENTS.md）、常驻暴露档每类 description 合计 ≤8 KB、按需档单文件 ≤8 KB 总量不限；组员自建 rules 必须条件式（always_on 槽位只留 charter）。执法=delivery-gate 预算校验（组员侧）+ pre-commit（真本侧）+ doctor（巡检），条款入 basic.md §1、§3.1。既有 10 个超限 references 文件严格拆分一劳永逸（用户裁决「严格执行」）。插件 description 税与 MCP lazyLoad 列实测探索项。leader-only.md 增「身份锚定」条款加固新会话组长自我认知。MCP schema 不计入知识预算（属工具面）。全部基建改动入 backlog 六项（refs-split-8kb / budget-delivery-gate / budget-precommit / budget-doctor / plugin-tax-probe / mcp-lazyload-probe）交 devops | 用户三点裁决 + 组长实施 |
| 2026-10-11 | **REFACTOR v1.0 定稿：评价驱动的自改进体系**（七轮迭代，用户终审通过；E0–E7 分阶段实施——2026-10-13 收官，生效机制已并入本文，见 §12 收官段）。核心裁决：① 目标优先级=客观优质>用户审美（审美为 tiebreaker/廉价替代）；② 不新增抽象层级、允许新增要素，改进管道自覆盖（豁口：inspector 不自巡其 CLI；twin 不自改权重）；③ 硬代码优先（教训：consult 软选项从不被使用）；④ 判决包连带决策：审批强制选 J（实施前仲裁：SiliconFlow 直连、非 Qwen 家族、阻塞≈1min、双序多数、材料撰写者侧组装）/M（基线捕获+实施后 bencher 副本 A/B 实测）/J+M（实测覆盖仲裁）/不选=override 留痕；「仅涉自身基建直通」原则移入 jury-brief（组长主观豁免取消）；⑤ 交付后核验零新裁决点、只实测不再仲裁、未改善自动立案纠正；⑥ 新成员 inspector（巡检找茬，「与用户视角相同但知识不同」，设计审查 rubric=社区优先/反冗余/LLM 直觉友好；双队列迟滞节流+全局开关；发现不经组长自行判决；无 override 权）、bencher（harness 实测，FIFO 非常驻，套件与基线区完全写权）、twin（审美孪生：冷启动无记忆、PREFERENCE.md 置 pod 内只读；客观通道模棱两可时回退终裁不给平局；吸收语料后免阈值直接上线，代理原需用户裁决事项——先决策后问责，上线/下线宣告权归用户；最终交付验收不代理）；⑦ 决策形式统一二选一，多选拆分为依序二元对局；⑧ Tier A 失败标签硬性接入（阈值自动立案，不指望组长有空）；⑨ 一成员一技能一 CLI，共享逻辑做库不做新 CLI（pysci-jury 撤销）；CLI 人体工学范本=Qoder CLI/commitizen；⑩ mirror 控制台（status 全景=用户唯一关注入口+注册纪律；review 盲审即训练→偏差报告绿色通道；journal 机械分流：偏好陈述→语料/行动指令→强制派发不得 override）；⑪ deputy/reviewer 降档 flash、新增 minimum 压测档（型号待选）；⑫ 每阶段自带可证伪成功判据，组件连续两 Phase 无改变决策的读数即退役 | 用户（七轮裁决）+ 组长起草 |

| 2026-10-12 | **twin 初步上线宣告 + 渠道修订（用户裁决）**：① 用户审阅同意 PREFERENCE.md v1，指令优先吸收 prompts_storage「1近期」文件最新段后**初步上线**——「宁可粗糙但在用，先上线再说」（例外清单仅存的上线宣告权已交付，E7 接线单 #73）；② 仲裁渠道修订：三席官方直调 DEEPSEEK/GLM/KIMI（减平票）、SiliconFlow 降回退且回退必须案卷/status/mirror 三面告警、minimum 选型火山方舟 doubao-seed-2.0-mini（Key 载体=项目根 `.env` 非 OS 环境变量；需通用 Key/模型开通时用户指示联系）；③ basic.md §3 经用户授权修订：fallback 改 `uv run --no-sync`、改 pyproject 必先 `pysci-dev venv-sync`（隐式 sync 撞锁删入口事故固化）；④ 断点恢复实测通过（重启演练：孤儿回收/重派/黑匣子全绿）；⑤ 连环触顶根治立案：drain 组包不看任务尺寸（#72 --solo）、bencher A/B 空壳 done 无判决（#70 done 语义收紧）、E5a 续作含里程碑提交规程（#71） | 用户（四项）+ 组长立案 |
| 2026-10-13 | **自改进体系实施收官（E0–E7 全链实证）**：各阶段成功判据逐项落地并经实战验证，生效机制全部并入本文（§1/§3.1/§4.1/§4.5/§5.3–5.5/§6/§9），原设计文档删除、仅存 git 历史；收官单=backlog 20261013-072000-leader | 用户终审授权 + 组长立案 |
| 2026-10-13 | **PREFERENCE v2 生效 + twin 全量上线**：用户审阅确认 v2 转正（33 条语料蒸馏增量+4 修订入宪章），twin 判断通道由初步上线转全量；盲审/回测动线不变，下线权归用户 | 用户 |
| 2026-10-13 | **修复形状分流 + 后台决策审计面（用户裁决）**：① 回退 ctx 水位闸（fe68a40 改了策略面却绕过判决通道、压缩防溢出使宣称收益无效——重启须经判决包并以真实账单/缓存读数为依据）；② 修复形状分流：inspector mechanical 免判决仅限 restore，change 无判决记录 → 任务书要求 devops 交付 blocked 上报（登记字段/任务书注入/charter 纪律三落点）；③ `mirror status` 注册「决策」项 + `pysci-mirror decisions` 决策链全貌（只读拼装、免判决+change 置顶） | 用户 |
| 2026-10-13 | **路由 v3 谷时隐性 DeepSeek**（backlog 20261013-093000-leader）：DeepSeek 官方价分峰谷（峰 2 倍）→ 谷时自动把 qwen/mimo 预设的 **main** 席链换成 DeepSeek，`preset` 行不动（不新增预设）；判据 chinesecalendar（周末全天=含调休上班的周末、法定节假日全天、工作日其余时段；覆盖外年份/库缺席保守不切换并告警）；入谷即切、入峰提前 1 小时、**只影响新派发**（在途单跳台账 model 不变）。终裁修正：pro 席保持原预设升档席、仲裁团固定 GLM+KIMI+Mimo（矩阵 qwen/mimo/deepseek/隐性态四态定格）、**一律按量付费、TokenPlan 路径整个不做**（个人版条款禁 API 自动化调用）、缺 Key 席双面告警减员运行（平票 twin 终裁兜底）、被评模型同族回退席不上场；观测面 = `orch status` 路由段 + mirror 概览「路由」项 + 决策日志 `kind=routing`。附带：minimum 席链改 BYOK Mimo-V2.6-Flash（火山豆包调查关闭，P-25） | 用户设计 + 终裁 v4 |
| 2026-10-13 | **批准单单一源纪律机械化**（backlog 20261013-071759-devops）：批准可被追加段写成自相矛盾而无人能发现（`20261012-164601-leader` 的 `taste_enabled` 实案）。新增 `dispatch.criteria_conflicts` 只判**同一配置键的相反布尔值**（精度优先：义务/禁止反差不判，误报=噪声=没人读），approve 告警 + 派发时把矛盾与取舍规则嵌进任务书；单一源登记=`superseded` 且点名该键。全语料 78 份建议实测：1 处真阳、0 误报 | 组长采纳 + devops 实施 |
| 2026-10-13 | **交集类失败停链**（backlog 20261013-102152-devops）：词表增 `merge_dirty_face`，drain 认此成因→抢救照做、续作单不立，待清清单进 `merge_intersect` 并渲染 `orch status`（§9 恢复面例外闸）；任务书尾部教组员贴清单。实测三连跳树哈希逐字节相同=续作链空转 | 组长采纳 + devops 实施 |
| 2026-10-13 | **lint 幻影抗体扩面到 13 charter**（backlog 20261013-085045-devops）：「harness 注入的 lint/告警含 `file:line` 先对盘核验」此前只在 8 面在册，其余 5 名组员同样会被平台幻影误导；快照测试钉 13 charter + 2 技能面（清单长度与文件数双锁） | 组长批准（`229a238`）+ devops 实施 |

## 14. 已知代价与风险

- **orch 是体系咽喉**：plan-first 使组长对 orch 依赖更深——§4.3 红线（fail-open +
  降级手册 + state 可读格式）不可妥协；orch 自身的改进走 backlog 循环；
- **组员自维护层的治理**：写坏记忆会持续污染该成员后续会话；自建内容膨胀会推高每跳
  上下文税。缓解：charter 编写纪律（AGENTS.md 控体量、笔记分类进 rules/skills 按需
  加载）+ git 可回滚（入库部分）+ devops 巡检（体量/质量抽查）+ 台账成本曲线暴露
  异常增长；
- **flash 档位质量风险**：专职组员默认 flash，复杂环节可能力不从心。缓解：计划环节
  可升档 max；blocked/返工率入台账，数据驱动调整成员默认档位；
- **声明式机械验收的滥用面**：一律 check="none" 会架空验收。缓解：台账透明 + 抽检 +
  reviewer 可见验收状态；Phase 1 观察声明行为再决定是否加软约束；
- **resume 成本随会话增长**：原生自动压缩兜底上下文窗口，但摘要有细节损失——重要
  约定及时蒸馏进自维护层，不依赖会话历史；归档阈值由台账数据定；
- **basic.md 泄漏税**：注入全员，保持精简（其「只增不减是失败信号」条款在此更重要）；
  根目录不建 AGENTS.md（建了即注入全员）；
- **reviewer 主观性**：rubric 用户审定 + VERDICT 证据义务 + 组长仲裁逃生门；
- **yolo + guard 组合的失败面**：Bash 绕过（诚实条款 §9）；transcript 审计是唯一事后
  手段；
- **PATH 持久化的边界**：仅覆盖 Git Bash 新会话；PowerShell 手动使用需另配 profile
  （可选）；旧会话快照不含新 PATH（fallback uv run 兜底）。

---

## 附录 A：降级手动操作手册（orch 故障时）

组长手动派发模板（Git Bash；**必须用原生 exe**，`.cmd` 包装器剥引号）：

```bash
EXE="$HOME/.qoder-cn/bin/qoderclicn/qoderclicn.exe"
POD="orchestration/pods/figure"
# 新会话（续会话：把 --session-id 换成 --resume <sid>；sid 查 --list-sessions）：
"$EXE" --cwd "$POD" -p "$(cat $POD/inbox/task-XXX.md)

--- 交付协议：最终回复须含 <result> 或 <blocked>（互斥），可附 <infra_suggestion>；
产物用 <artifact check=...> 声明。" \
  --session-id "$(uv run --no-sync python -c 'import uuid;print(uuid.uuid4())')" \
  --name "任务slug" -m <型号> --max-turns 50 -o json > /tmp/envelope.json 2>&1
uv run --no-sync python -c "import json;print(json.load(open('/tmp/envelope.json'))['result'])"
```

审批回复附送：把 `orchestration/state/replies/<member>/*.md` 内容拼进 -p 文本。
机械验收手动运行：按 check 类型对应各 CLI 审计子命令（如 `uv run --no-sync pysci-figures audit
<产物>`）。台账手工补记（credits/耗时见 envelope 字段，Phase 0 确认字段名）。
