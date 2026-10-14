# REFACTOR.md — `pysci` 交互式控制台 + 全员 FIFO 队列化（施工蓝图）

> 状态：**已批准，施工中**（2026-10-14 用户批准全部五项裁决，④⑤ 明确同意）。
> 施工计划：`orchestration/state/plans/refactor-20261014.plan.yaml`（M1–M7 逐环节推进）。
> 实施完成后裁决并入 `orchestration/README.md` §12/§13，本文件退役（REFACTOR 历史惯例）。

## 0. 目标

两部分（用户提案 2026-10-14）：

1. **`pysci` 交互式控制台**：`pysci-mirror` 改名 `pysci`，裸命令进入交互式 TUI 菜单；
   命令式子命令退役为 **LLM 专用通道**（不再考虑人类便利性）。
2. **全员 FIFO 队列化**：所有可派发组员改异步队列处理（队列非空拉起 worker、空后销毁），
   统一事件/唤醒体系；组长侧新增**任务书 FIFO**（用户经 TUI 投递大任务，多 plan 并行推进）。

核心语义转变：组长从「跑 plan 的人」变成「调度员」——派发入队即返回，完成事件驱动推进。

## 1. 现状基线（2026-10-14 实查）

- **mirror.py**（`src/pysci/skills/orchestration/tools/mirror.py`）：纯 print 命令式骨架，
  12 个 STATUS_ITEMS 一行一项；`journal` 有编号分流菜单（唯一切交互面）；无任何 TUI。
  数据源：registry.json / PREFERENCE.md / evals/taste / mirror-queue.jsonl / jury-queue /
  channel-events.jsonl / inspectd 决策 / backlog 延期 / mirror-pending.json。
- **队列原语已通用**：`workq.py`（JSON 文件 FIFO、`pending→in_progress→done|failed`、
  `take_next`、孤儿回收、keep 20），注释明言「共享逻辑做库，不做第二套实现」。
- **现役队列三套**：devops `backlog.json`（workflow/drain，**组包** ≤4、`devops.lock`、
  临时 drain 进程队空即退）；bencher `state/bencher/queue.json`（benchd，run/capture/ab/check，
  完成「择机查看」不唤醒）；inspector（inspectd，queue-first 自循环）。——「只有 devops 和
  bencher」的说法不准确，inspector 已是第三套。
- **普通派发 = 进程级阻塞**：`orch plan run`（plans.py）→ `dispatch.do_dispatch` →
  `runner.run_headless`（Popen + communicate 同步等完）；plan 内环节串行。组长不阻塞仅因
  `plan run` 挂后台 Bash，**跨 plan 并行不是一等能力**。
- **watch/wake**：`watch.py` 事件等待器 + `.qoder/hooks/watch-arm.sh`（每回合重臂
  `orch watch`，命中 exit 2 唤醒组长）；现事件 5 种：QUEUE-COMPLETE / NEEDS-LEADER /
  QUOTA-STOP / CHANNEL-STOP / CRASH。方向：worker 从不主动唤醒，唤醒全靠 watch。
- **限流避让半接线**：`backpressure.py` 活跃会话注册表写侧（`note_session`/`heartbeat`）
  **全仓零调用者**，注册表实际为空；延期（`defer_until` 指数退避）只在 devops 链生效。
- **任务书现状**：只有派发产物级任务书（`pods/<member>/inbox/task-*.md`，dispatch 落盘前
  lint）与 backlog.json（基建小项）；**组长级任务书 FIFO 不存在**，是新物。
- **编制**（orchestration/pods/ 13 席）：7 专职生产组员 sim / writing / figure / theory /
  lit / drawing / model3d + deputy；支撑席 reviewer / devops / bencher / inspector / twin。

## 2. 第一部分：`pysci` 交互式控制台

### 2.1 命名与入口（〔取舍〕⑤：硬改名，不留别名）

- console script：`pysci = pysci.skills.orchestration.tools.mirror:main`（或 tui 门面，
  见 2.5）；**删除 `pysci-mirror`**，全仓引用清扫（README、charter、技能、hooks、注释）。
  pyproject 处「加前缀 pysci- 避免与全局工具重名」的注释同步改为记录本次例外及其理由
  （ADR 写在决策发生的文件里）。
- 入口语义：
  - `pysci`（tty）→ 交互式 TUI；
  - `pysci`（非 tty）→ 等价 `pysci status` 静态输出（防 agent 会话误入交互）；
  - `pysci status|journal|decisions …`（任意环境）→ 静态命令式输出，**LLM 专用通道**，
    输出格式向机器可读优化（稳定的 `[key] value` 行、退出码语义化）。
- `status` 参数不弃用但降格：只作为 LLM 通道存在，人类不再需要它。

### 2.2 交互模型

- **主菜单**：每模块一行——标题（左）＋ **更新徽标**（右），不再显示简介。
  - 更新徽标 = 「自上次进入查看以来有值得关注的更新」。真源 `state/mirror-seen.json`：
    `{module: cursor}`（cursor 为各数据源的水位：时间戳或最大 id）。各模块实现
    `updates_since(cursor) -> 计数/布尔`；进入该模块详情视图即写入新水位。
  - 徽标样式示例：`● NEW`（有更新）/ 数字（更新条数）/ 灰点（无）。
- **选择**：↑↓ 切换；选中项高亮（加粗 + 彩色 + 边框/前缀装饰，细节由实现定）并在其下方
  **展开 3–5 行概览板**（复用现 status 读数函数裁剪，只放最关键的读数）。
- **Enter** → 全屏详情视图：整个 CLI 仅展示该模块的详细信息；Esc/← 返回主菜单。
  详情面板内的具体功能**留后续单**，但宗旨固定：**该模块全部工作入口在此交互完成**，
  不要求用户翻文件夹。
- 现有 `journal` 编号分流菜单迁移进 TUI（「口述」模块动作），保留偏好/指令二分法理；
  `decisions` 全貌迁移进「决策审计」详情视图。
- **非 tty 回退**：任何交互面在非 tty 下输出静态读数 + 用法提示，永不阻塞等输入。

### 2.3 技术选型（〔取舍〕④）

- **主选 Textual**（社区标准，Textualize/rich 作者出品；§2 自研 vs 社区：不手搓 TUI 框架）。
- **M1 必做 spike**：在用户实际终端（Git Bash/mintty、Windows Terminal、PowerShell）实测
  键盘事件与渲染；不兼容则**退 rich + prompt_toolkit**。spike 结论写回本文件。
- 依赖入 `pyproject.toml` 时在旁注释 ADR（选了什么、弃了什么、为什么）。
- 美观开放目标：主题色、装饰线、徽标、展开动画朝「精美」努力，但不为美观阻塞功能。

**M1 spike 结论（2026-10-14，devops 实施）：Textual 定案，退路未启用。**

- 环境：Windows 11 / Python 3.13.15。安装：`uv pip install textual` → **`textual==8.2.8`**
  （带入 linkify-it-py 2.2.0、mdit-py-plugins 0.6.1）；已入 `pyproject.toml` dependencies 并附 ADR 注释。
- **复现命令 + 实测输出**：
  ```
  $ uv run --no-sync python -c "import textual; print(textual.__version__)"
  8.2.8
  $ uv run --no-sync pytest tests/skills/orchestration/test_tui_spike.py -q
  6 passed
  ```
- 骨架落点：`src/pysci/skills/orchestration/tools/tui/app.py`——`pysci` 入口 `main()` 按 §2.1 分派
  （子命令→静态通道 / tty→TUI / 非 tty→`status`），`modules()` 复用 `mirror` 纯读数函数（§2.5）。
  测试 `tests/skills/orchestration/test_tui_spike.py` 用 **Textual Pilot** 无头驱动 ↑↓/Enter/Esc
  （断言选中态与视图切换）、启动→渲染（七行菜单）→干净退出冒烟、入口分派（非 tty→status 逐字等价）。
- **实机终端残余风险（Pilot 覆盖不到，如实标注、不宣称「全部兼容」）**：本环节的交互验证全在
  **Pilot 无头驱动**下完成——它证明的是**交互逻辑**，**不等于**真实终端的渲染与键盘事件可用。
  Git Bash/mintty、Windows Terminal、PowerShell 5.1 下的实机验证本环节**未做**（devops 会话非 tty），
  列为残余风险，需用户在实机跑一次骨架（`pysci` 交互：↑↓/Enter/Esc）复核；若发现不兼容，按上条
  退路改 rich+prompt_toolkit 并改写本结论。

### 2.4 模块清单（初版）与旧概览项归并

| # | 模块 | 归并的旧 STATUS_ITEMS | 详情视图工作入口（后续单细化） |
|---|------|----------------------|------------------------------|
| 1 | 任务书 | （新） | 大任务列表 / 新增多行输入 / 批复返工（§3.4） |
| 2 | 计划与派发 | （新） | active plan 清单、环节进度、多 plan 并行面 |
| 3 | 队列与运行 | 队列、延期 | 各成员队列水位、worker 状态、延期到期 |
| 4 | 盲审与审查 | 盲审 | twin 待盲审案件、reviewer 环节附属面 |
| 5 | 系统与路由 | 开关、路由、仲裁、回退 | taste/inspector 开关、路由预设、席位健康、回退事件 |
| 6 | 决策审计 | 决策（+ 趋势占位） | inspector 决策链全貌（现 cmd_decisions 迁入） |
| 7 | 待决与验收 | 待决、验收 | mirror-pending 增删、待验收交付批复 |

宪章项并入「系统与路由」或「盲审与审查」（twin 宪章）——实现时择一，不单列。
E6 占位项（overrides/stats/偏差报告）在对应模块内继续如实标注「占位」。
**注册纪律不变**：新增「需用户关注事项类型」必须加进模块读数（原 STATUS_ITEMS 登记表语义）。

### 2.5 代码布局与迁移

- `mirror.py` 拆**数据层/表现层**：读数函数（`_line_*`、`charter_state`、`pending_items`…）
  保留为纯函数供 LLM 通道与 TUI **共享**（同一能力不留两份实现）；TUI 层新建
  `tools/tui/`（`app.py` 门面 + `modules/<模块>.py`），入口 `main()` 按 2.1 分派。
- 迁移原则：已有的精美交互面（journal 编号菜单）**搬进** TUI；其余新建，无历史包袱。

## 3. 第二部分：全员 FIFO 队列化 + 统一唤醒

### 3.1 队列层

- **统一基于 workq 原语**，不造第二套队列实现。既有队列保留落点与各自条目 schema
  （devops 的 effort/batch/defer 字段、bencher 的 kind/cases 字段），状态机统一
  `pending→in_progress→done|failed`。新成员队列落 `state/queues/<member>.json`。
- **新队列覆盖**：7 专职 + deputy（+ reviewer，见 3.2 附属语义）。inspector 维持自循环
  队列；twin 是判决服务不入队。
- **worker 生命周期**（推广 devops drain 模式）：入队且无存活 worker → `wake_<member>`
  拉起分离进程（per-member 锁、孤儿回收、**队空写 .done 退出销毁**）；非常驻。
- **组包**：仅 devops 保留（基建小项打包效率实证好）；生产组员各 job 相互独立，
  **一律不组包**（用户指示）。
- **job 单元**：普通环节 = 1 job = 1 通知；带 review 的环节 = **环节 + 其 review 合并为
  1 个通知单元**（review 是附属步骤，由 reviewer 队列执行，宿主环节交付→review 完成后
  统一发一次事件）。

### 3.2 统一事件/唤醒层

watch.py 扩展事件类型（优先序沿现制）：

| 事件 | 触发 | 唤醒策略 |
|------|------|---------|
| JOB-DONE | 7 专职 / deputy 每件完成 | **每件唤醒组长**（及时接续下一份工作） |
| STEP-REVIEW-DONE | 环节+附属 review 合并完成 | 合并单元唤醒一次 |
| QUEUE-EMPTY | devops / bencher 队列消化完 | 队列空唤醒一次（② bencher 归此类） |
| TASKBOOK-NEW | 用户 TUI 投递任务书 | 立即唤醒组长 |
| NEEDS-LEADER / QUOTA-STOP / CHANNEL-STOP / CRASH | 沿现制 | 沿现制 |

- **合并唤醒（防唤醒风暴，硬规格）**：唤醒是**边沿触发**——组长每次醒来经
  `orch events drain` 一次排空积压事件，再逐 plan 推进；多 worker 同时完工只折合一次
  回合。事件流落 `state/events.jsonl`（channel-events 同族，追加式）。
- inspector 沿现制：自循环巡检，仅 NEEDS-LEADER 类唤醒组长。

### 3.3 限流避让融合（硬验收项）

- **接线 backpressure 写侧**：worker 拉起即 `note_session`（active-sessions.json）、
  运行中周期 `heartbeat`、完成注销；>10min 无心跳判死回收（沿现阈值）。
- **出队前查延期**：队列管理器释放 job 前过 backpressure 延期纯函数；bandwidth 类失败
  回队挂 `defer_until`（沿 `workflow.backlog_defer` 语义推广到全部队列），到期前不被取。
- 验收：有 worker 时 `active-sessions.json` 非空；构造 bandwidth 失败可见全队列条目
  延期重取；`pysci`「队列与运行」模块可见延期读数（沿现 `_line_defer`）。

### 3.4 组长任务书 FIFO（用户裁决 ① 落点 `data/taskbooks/`）

- **文件**：`data/taskbooks/<id>.taskbook.md`（元数据头：ts / title / status / source=
  mirror-tui|llm ＋ 用户纯文本正文）＋ `<id>.state.json`（状态机）。
  id 形如 `tb-YYYYMMDD-HHMMSS-<slug>`；写入 data/ 前过 `assert_within_data`。
- **状态机**：`new → planning → active → delivered → accepted | rejected`；
  rejected **带用户批语回 new**（返工重拆解）；state.json 记 `plan_id` 链接。
- **TUI「任务书」模块**（§2.4 #1）：
  - 列表：现有大任务（active/planning）、已完成待批复（delivered）、历史（accepted）；
  - 新增：多行纯文本编辑器，程序化组装时间等元数据并落盘；落盘即 TASKBOOK-NEW 唤醒组长；
  - 批复：accept / reject+批语（交互输入批语，写回 state.json）。
- **组长处理**：`orch taskbooks list|show`（程序化面，LLM 通道）→ 拆解为 `orch plan` →
  推进至交付 → 用户批复。plan 完成联动 delivered，用户 accept/reject 联动终态。
- **多 plan 并行**：组长同时持有多个 active plan；每次唤醒逐 plan advance。同一成员的
  队列天然串行（FIFO），不同成员即并行——两个大任务不共用成员就真并行（用户模型成立）。
- **既有行动指令车道**（mirror-queue.jsonl，强制派 devops、不得 override）**保持独立**，
  与任务书的合并留后续单，不在本次范围。

### 3.5 plan 推进语义改造

- 现 `orch plan run` = 串行阻塞推进（Popen+communicate 等完才下一环节）；改为
  **`orch plan advance <id>`**：把下一环节入队即返回，完成事件触发下一次 advance。
- `plan run` 保留为 LLM 通道兼容形态（内部=循环 advance+等待），人类/组长正道走事件驱动。
- 既有语义全保留：环节交接暂停、任务书延迟书写、机械验收、失败分类回退。

## 4. 实施阶段（每阶段走 backlog → devops/对应组员，验收标准写进任务书）

| 阶段 | 内容 | 依赖 |
|------|------|------|
| M0 | 本蓝图用户审阅、裁决定稿 | — |
| M1 | TUI spike：Textual 在用户终端实测；裸 `pysci` 入口 + 非 tty 回退 | M0 |
| M2 | TUI 主菜单 + 更新水位（mirror-seen.json）+ 详情视图框架 + 模块归并迁移 | M1 |
| M3 | 任务书模块 + `data/taskbooks/` 落盘/状态机 + TASKBOOK-NEW 唤醒 | M2 |
| M4 | 队列层：7 专职+deputy+reviewer FIFO、worker 生命周期（workq 统一） | M0 |
| M5 | 事件/唤醒层统一（JOB-DONE/STEP-REVIEW-DONE/QUEUE-EMPTY、events drain 合并唤醒）+ plan advance 事件驱动 + 多 plan 并行 | M4 |
| M6 | backpressure 写侧接线 + 限流融合验收（§3.3） | M5 |
| M7 | 命令式面转 LLM 通道 + `pysci-mirror` 引用清扫 + charter/README 同步 + 本文件裁决并入 §12/§13 | M2+M6 |

M2/M3 与 M4/M5 可**分线并行**（不同成员承接，恰好用上新队列能力）；M1 spike 未过则先换
选型再进 M2，不带病推进。

## 5. 风险与开放项

- **Textual × mintty 兼容**：M1 spike 裁决；退路 rich+prompt_toolkit 已定。
- **唤醒频率 vs 组长回合成本**：每件唤醒是用户明确要求；合并唤醒兜底，实测后若仍吵，
  可在后续单调「小 job 合并窗口」，不动本次裁决。
- **改名清扫面**：README/charter/技能/hooks 中 `pysci-mirror` 引用须全数更新（M7 验收：
  全仓 grep 零命中）。
- **data/taskbooks/ 入 git**：任务书含用户意图文本，项目内即视为可管理（§1 权限模型）；
  需要外置保密的由用户备份到项目外。

## 6. 裁决记录（2026-10-14；全部经用户批准，④⑤ 用户明确同意）

- ① 任务书落点 = `data/taskbooks/`（用户裁决；入 git、用户可见可编辑）。
- ② bencher 唤醒 = 队列空时唤醒一次，与 devops 同类（用户裁决）。
- ③ 施工节奏 = 本蓝图定稿并经用户审阅后开工（用户裁决）。
- ④〔取舍〕TUI 选型 Textual 主选、rich+prompt_toolkit 兜底，M1 spike 实测定案（组长）。
- ⑤〔取舍〕`pysci-mirror` 硬改名不留别名；命令式子命令保留为 LLM 通道（组长）。
