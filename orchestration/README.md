# pySci 多 Agent 编排顶层设计

**状态**：v2.2 设计定稿；**Phase 0–4 已基本完成**（§12：验证冲刺、orch 骨架+figure
纵切面、plan/改进循环/devops、reviewer+副组长、六 pod 脚手架+MCP 清零+basic.md 重写
均实战验收）。**体系进入常态运行**：剩余事项 = 各 pod 随首个真实任务逐个战备验收、
跨域双路径对比（随首个真实跨域任务）、可选加固项。本文档是编排体系的**唯一权威**：
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
| **devops** | `orchestration/pods/devops/` + headless | 会话池 | 维护全部基础设施：orch/dev CLI、全体 pod 只读层、技能真本与部署、guards、验收脚本；处理 backlog（FIFO）；git 提交与推送 | **不使用 orch**（组长专属）；cwd=自身 pod、worktree 为作业对象（§7）；护栏级修改经组长转呈用户授权 |

**副组长平级协作条款（用户裁决）**：副组长与组长接近平级，不是从属关系。
- 副组长**有权拒绝**派发任务（须附理由，典型情形：组长对复杂度估计误判）；拒绝经
  orch 记录，组长应修改任务书重派，或行使最终裁定权坚持原派（裁定理由留痕台账）；
- 组长咨询时，副组长**有责任补充不同视野与建议**，不鼓励无条件附和；
- 硬性决策上组长保留最终裁定权——本条款目的是扩展组长视野，不是制造对抗。

组员 id：`sim` `figure` `writing` `theory` `lit` `drawing` `model3d`（专职 ×7，一一对应
7 大技能）；`deputy` `reviewer` `devops`（特殊组员）。

**「一成员一技能一 CLI」组织模式（用户裁决）**：除副组长与 reviewer 外，每个成员有
专属技能真本与专属 CLI——7 专职 ↔ 现有 7 个 `pysci-*` CLI；组长 ↔ `orchestration`
技能 ↔ `pysci-orch`；devops ↔ `devops` 技能 ↔ `pysci-dev`。全部 9 个技能真本统一存
`orchestration/skills/`（§2），按 manifest 部署（部署目标含项目根与各 pod）。

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
  手段）；`mcp.lazyLoad` 可减 MCP 首轮开销。

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
│   │   ├── registry.json            #     成员注册表：pod 路径、模型档位、参数模板、会话池索引
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
ledger 流水为可再生物不入库。

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

### 3.2 会话池（原生能力 + orch 薄封装）

原生能力齐备（分储/resume/continue/fork/name/list/delete），orch 只做面向 LLM 的
封装，**组长永不接触 sid**：

- **默认策略 = 继续最近活跃会话**（用户裁决）。orch 指导派发时显式列出该成员最近
  3~5 个会话（名称/摘要/跳数/最后活跃），并告知「默认继续最近一次会话」；组长判断
  本次任务归属：属于最近会话→一键继续；属于更早会话→显式指定；全新任务→显式选择
  新开；
- 每次派发以 `--name <任务slug>` 命名，会话池自描述；
- **归档**：`orch archive`：先派一跳「把本会话可复用经验蒸馏进你的自维护层」，确认后
  标记归档；归档会话仍可 resume（历史有价值时），不出现在默认列表；
- **清理**：`orch sessions <id> --prune`（复用 `--delete-session`）；
- 上下文膨胀由**原生自动压缩**兜底（摘要式延续，非截断；§1.1）；orch 台账记录每跳
  credits 与耗时，成本曲线异常时提示组长归档/新开——不自研压缩。

### 3.3 派发机制

`orch dispatch <id> --task <file|--text> [--session …] [--plan <id> --step <n>]`
（日常经 plan 系统调用，见 §4.2；直接 dispatch 属底层命令，§4.1）：

1. 任务书写入 `pods/<id>/inbox/task-<ts>.md`（自包含：目标、输入路径、交付要求、
   白名单目录、验收声明要求、所属计划环节、模型档位）；落盘前经 `state/taskbook-lint.json`
   轻量 lint（正则扫已知错误 CLI 范式，命中仅**告警不阻断**，规则表 devops 可增补）；
2. 组装：`<exe> --cwd orchestration/pods/<id> -p <注入文本> --resume <sid>|
   --session-id <新sid> --name <slug>`；注入文本 = 任务书路径 + 待附送审批回复（§6）+
   交付协议提醒；
3. 参数模板存 registry：模型档位（§3.5）、推理强度（§3.5 按项标注，可选 `effort`）、
   `--max-turns`、`--mcp-config .qoder/mcp.json
   --strict-mcp-config`（§8）；权限依赖用户级 yolo 默认（用户已设），orch 不显式传参
   （registry 留覆盖项备用）；
4. 解析 `-o json` envelope：结果文本、credits、耗时、num_turns、session_id → 台账；
   另测本跳会话 jsonl 字符增量 → `delta_chars`/`est_tokens`（相对用量尺子，§4.4）；
5. delivery-gate hook 已在成员侧保证格式合规（§3.4）→ orch 解析交付块（第二道防线：
   解析失败时呈组长原始文本）→ 执行声明的机械验收（§5.1）→ 按计划推进 → **在工具
   回复中直接给出下一步**（决策选项，或已自动完成的机械接力说明）。

失败处理：超时/非零退出 → 自动重试一次（同会话 resume，注入错误上下文）；再失败 →
呈组长决策（改任务书重派 / `orch consult` 副组长 / 上报用户）。**例外**：额度类失败
（envelope/stderr 含『credit usage limit』类文案）→ 不做**同渠道**重试；若该档配置了
`<tier>_fallback`（如 flash→BYOK flash）则**换档重试一次**，仍耗尽 → `kind=quota_exhausted`
（drain 据此全局停止），[NEXT] 提示检查 registry models 渠道绑定 / 等待额度重置。

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

### 3.5 模型双档路由（用户裁决，2026-10-10 渠道策略修订）

- registry 维护档位抽象：组长只见「max/flash」两档，orch 内部路由具体渠道；
- **渠道策略（用户裁决，源于 credit 耗尽事故）**：
  - `max` = **不传 `-m`**，走用户级默认的 **BYOK Qwen-3.8-Max**（Token Plan 额度大且
    实惠；**禁用内置同名模型**——烧订阅 credit 且 envelope 低报为 0，2026-10-10 实战
    烧穿额度）；
  - `flash` = 内置 `Qwen3.8-Flash`（**限时免费**，零额度消耗）；
  - `flash_fallback` = BYOK Qwen-3.8-Flash 的 modelID（免费期结束后的回退；渠道待
    用户开通）。**回退逻辑已实现**（backlog 20261010-flash-fallback）：do_dispatch
    遇 quota_exhausted 且该档配了非空 `<tier>_fallback`（异于当前型号）→ 换备用渠道
    重试一次；仍耗尽才归类 quota_exhausted 触发 drain 全局停止（见 §3.3 失败处理）；
- **模型映射自动刷新**（用户裁决 2026-10-10）：BYOK 模型的人类名→UUID 目录只来自
  `--list-models`（账户级；settings.json 仅存当前激活 UUID、无法监听），故 orch 在
  **drain 启动时**跑 `--list-models` 按名称模式（`DEFAULT_MODEL_PATTERNS`：Qwen-3.8-Max→max、
  Qwen-3.8-Flash→flash_fallback、内置无 UUID 的 Qwen3.8-Flash→flash；registry.model_patterns
  可覆盖）刷新 registry.models，重配 BYOK 后下次 drain 自愈、零 daemon；手动 `orch
  refresh-models` 同逻辑。**未命中档保持现值不清空**（宁旧勿空）；刷新只更新既有档的
  UUID/名，**不改变哪个名对应哪个档**——那仍是护栏级（下方教训登记），改名须更新模式表；
- 默认档位：组长（TUI）/deputy/reviewer = **max**；7 专职组员 + **devops** = **flash**
  （devops 于 2026-10-10 经用户裁决降档：其工作为「规格已审定的末端实现」，Flash+强
  harness 匹配；架构敏感项由组长派发时显式 `--model-tier max` 升档；降档后以台账
  数据观察 2~3 周失败率/返工率，质量下滑即回调）；计划环节可覆盖（`steps[].model`）；
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
  轮数/失败率/返工率对比，**数据驱动再调策略**；
- **教训登记**：模型渠道 = 成本渠道，registry 的 models 映射改动属护栏级（影响全体
  成员计费），须经用户；额度类失败的系统性识别与 drain 全局停止**已实现**（backlog
  20261010-orch-quota-awareness，见 §3.3 失败处理与 §6）。

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
| `orch plan amend <id>` | 修改未执行环节（增删改、review 旗标、模型档位） | 更新后计划 |
| `orch approve/reject <reply-id> --note` | 审批 INFRA_SUGGESTION | 通过→入 backlog＋自动唤醒 devops＋watch 挂载指令；审批回复自动附送给提议人 |
| `orch watch [--timeout 4h]` | 阻塞等待后台 drain 事件（§4.5）：以 Bash run_in_background 启动 | **单行**事件（QUEUE-COMPLETE / NEEDS-LEADER / QUOTA-STOP / CRASH）或 TIMEOUT；退出即唤醒组长会话 |
| `orch consult <问题>` | 快捷咨询副组长（平级协作条款生效） | 副组长意见（含异议） |
| `orch review <产物>` | 派发 reviewer | VERDICT；FAIL→自动回派返工+复审一次，再 FAIL 升级仲裁 |
| `orch stats [--plan <id>]` | 统计（§4.4） | 任务/成员/历史三级耗时与 credits |

**底层命令面**（降级手册与 devops 调试用；组长规程明确**不鼓励**直接使用，防止形成
绕过 plan 的习惯）：`dispatch`、`sessions`、`archive`、`check`（抽检机械验收）、
`ledger`、`backlog`、`skills sync`。orch 瘫痪时全部可按附录 A 手动等价执行。

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
    model: flash            # 可覆盖默认档位（§3.5）
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
  中承担更重角色），为 harness 改进与模型档位调整提供数据。

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
  恒 0 档位（内置 flash）唯一可用的用量指标。
- **不做缓存命中修正**（本地不可得，对排名影响近似中性）。
- 增量基准优先取上一跳记账水位：崩溃跳/手动 resume 造成的「记账外增长」计入同会话
  下一跳，总量守恒（漏记会扭曲成员间排名）；水位高于实测起点只可能是文件被重写
  （压缩/清理），此时以实测起点为准。

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
  - `QUEUE-COMPLETE <run>`——新 `.done` 出现且锁已不再跑该 run（drain 在 finally 里先释放
    锁再写摘要，故二者同批可见）
  - `NEEDS-LEADER <id>`——backlog 中**新出现**的 needs_leader 条目
  - `QUOTA-STOP <run>`——额度类全局停止（`.done.stopped`，或 run 日志的 `[Q]` 痕迹——后者
    早一拍，不必等摘要落盘）
  - `CRASH <run>`——锁 pid 已死且该 run 无 `.done`
- **用法**：组长以 Bash `run_in_background` 启动，进程退出即原生完成通知唤醒本会话；一次
  watch 报一批事件即退，后续按需重挂（输出自带重挂命令）。`approve`（经 `wake_devops`）与
  `dispatch --from-backlog` 的输出**固定追加挂载指令行**，不靠组长翻文档。
- **边界**：watch 随组长会话消亡，**不影响 detached drain**；基线快照与心跳落
  `state/watch.json`（gitignore：纯运行态）。漏挂/会话中断期间的账由下次 `orch status` 的
  watch 段**一次性补账**（报完即推进基线，重复运行不刷屏；watch 仍在挂载时 status 不抢账，
  免双份唤醒）。同刻多事件按 CRASH > QUOTA-STOP > NEEDS-LEADER > QUEUE-COMPLETE 取主事件、
  其余以「同时」附注。**弃 Monitor 工具方案**（会话侧注册更重，且同样要组长发起）。

---

## 5. 质量保证：两层

### 5.1 机械验收（成员声明式，零 LLM）

组员工作形态多样（仿真组员可能只答疑问，绘图组员可能只画草图），**是否触发由组员在
交付中声明**（§3.4 `check` 属性），机械验收不得阻碍灵活任务：

- `check="<type>"` → orch 自动运行对应确定性脚本（注册于 registry checks 表；如图件→
  figures CLI 合规审计）；FAIL→自动回派返工（≤3 次，同指纹连败升级组长）；
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
  · 额度类失败（quota_exhausted）→ **全局停止**：种子复位 pending（非 needs_leader）、
    组包与剩余条目保持 pending、释放锁、.done 注明系统性故障；渠道恢复后下次唤醒自动续消化
→ run 日志 + .done 摘要落 state/devops-runs/<ts>.{log,done}；orch status 展示 worker
  态（running/idle/stale，其中 stale 再分**崩溃**〔pid 死且缺该 run 的 .done〕vs**干净完成
  后锁残留**〔.done 存在〕）、最近 run 摘要（含停止方式：自然跑完/协作停止/额度耗尽/异常中断）、
  needs_leader 提醒（组长据此重派/改任务书/上报）
→ 组长的等待面（代码化，§4.5）：唤醒/采纳/交办三处输出固定附 `orch watch` 挂载指令；
  watch 后台阻塞轮询 state 文件与锁，命中 QUEUE-COMPLETE / NEEDS-LEADER / QUOTA-STOP /
  CRASH 即单行退出唤醒会话；watch 随会话消亡、不影响 drain，漏掉的账由下次 `orch status`
  的 watch 段一次性补账
→ 涉及发起人 pod 的改动完成后，组长可立即重派该成员验证
```

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
worktree 或仓库根为 cwd，因此不继承组长 harness。orch 是组长专属；devops 不用：
backlog 条目经任务书注入，完成经 outbox 交付。

- worktree 事实（实测）：`data/` 磁盘 1.4 GB 但 git 仅跟踪 16 MB；worktree 共享
  `.git` 对象库，每份仅检出 ≈30 MB——容量顾虑不成立。worktree 不含未跟踪重数据，
  需要时以绝对路径只读引用主树；
- 作业流：领任务 → `git worktree add` → 实施 → 完整测试 → 合并回 main → commit+push
  （用户已授权为其职责）→ 交付；
- worktree 内 venv 策略（editable 安装指向主树 src）：候选 `uv run --project <worktree>`
  或 worktree 内 `uv sync`——Phase 2 实测定型，写入 devops 技能；
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
- **delivery-gate**（Stop hook）：交付格式强制（§3.4）；
- 权限模式：用户级默认 yolo（用户裁决），orch 不显式传参；hooks 在 yolo 下照常执行
  （v1 实测，Phase 0 以当前参数名复测）；
- 诚实条款：guard 只拦 Write/Edit/NotebookEdit，Bash 重定向理论上可绕过——设计目标是
  把失效模式从「遗忘/失误」升级为「刻意违规」，后者经 transcript 审计发现（devops
  巡检职责）；
- 组长侧不装硬 guard（用户在场的会话靠规程 + 报告义务 + 护栏授权红线）。

---

## 10. 端到端工作流

```
用户需求（可能含多环节）
→ 组长拆解 → orch plan new <计划文件>（环节粗描述、成员、白名单、review 旗标、模型档位）
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

实测（2026-10-07）：三种范式同速可用（裸 exe 路径 / `uv run pysci-X` /
`uv run --no-sync`，0.6~0.75 s）；裸关键字此前失败仅因 `.venv/Scripts` 不在 PATH。

**已实施（用户授权的全局改动）**：`~/.bashrc` 追加一行
`export PATH="$PATH:/d/XXXIIIGGG/projects/pySci/pySciWS/.venv/Scripts"`
（Git Bash 自动补建了 `~/.bash_profile` 加载链）——**新开的 Qoder 会话**（含 headless
pod）中裸 `pysci-X` 直接可用；已实测新登录 shell 解析成功。

配套纪律（写入 charter 与组长技能）：
- 首选裸 `pysci-X <子命令>`；命令不存在时 fallback `uv run pysci-X`（**裸名只试一次，
  禁止反复重试**——历史重试的根因即裸名失败循环与 PowerShell 陷阱）；
- orch/dev 内部子进程调用统一用 `uv run pysci-X`（不依赖会话环境快照，最稳健）；
- 组员 Bash 即 Git Bash，避免 PowerShell（basic.md §3 的引号/编码陷阱）；
- 旧会话环境快照不含新 PATH——迁移期（Phase 1 前）以 `uv run` 为准。

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
- [x] worktree venv 策略**实测定型：PYTHONPATH 覆盖法**——`PYTHONPATH=<wt>/src uv run
      --no-sync pytest <wt>/tests`（复用主 .venv，worktree 代码优先；1452 项收集 7.5 s，
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
  --session-id "$(uv run python -c 'import uuid;print(uuid.uuid4())')" \
  --name "任务slug" -m <型号> --max-turns 50 -o json > /tmp/envelope.json 2>&1
uv run python -c "import json;print(json.load(open('/tmp/envelope.json'))['result'])"
```

审批回复附送：把 `orchestration/state/replies/<member>/*.md` 内容拼进 -p 文本。
机械验收手动运行：按 check 类型对应各 CLI 审计子命令（如 `uv run pysci-figures audit
<产物>`）。台账手工补记（credits/耗时见 envelope 字段，Phase 0 确认字段名）。
