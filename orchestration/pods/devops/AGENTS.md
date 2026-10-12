# devops 组员长期记忆

> 本文件由 devops 组员自维护（每次会话启动自动加载）。只记稳定事实与约定。

## 身份

- pySci 编排体系 devops 组员（基础设施维护专员），charter 见 `.qoder/rules/charter.md`。
- 专属工具：`pysci-dev`（doctor/sync/worktree）；实现于 `src/pysci/skills/devops/`。
- 作业方式：worktree 隔离 + cz commit + commit 后 `timeout 25 git push origin main`
  （best-effort，失败忽略、不重试、不设代理；用户裁决 2026-10-10，超时硬要求防挂起阻塞 drain）。

## 经验（工作中积累）

- **worktree 内跑测试/CLI 的环境选法（2026-10-12 定型，backlog 20261012-023430-devops）**：
  `PYTHONPATH="<wt>/src" uv run --no-sync pytest "<wt>/tests" -q` 复用主 `.venv`，**前提
  =cwd 在 worktree 之外**——cd 进去时 uv 就地新建空 `<wt>/.venv`，缺 pyyaml 即 import 崩
  （曾被误记成「此法必崩」）；配方与兜底法见技能 `references/worktree-merge.md` §1。副作用：
  `pysci.paths.PROJECT_ROOT` 就近找 `pyproject.toml`，故 state/PODS 指向 **worktree** 副本——
  跑 sync/drain 会改其 `state/*.json`（backlog.json 被跟踪！），实验后
  `git checkout -- orchestration/state/…` 还原。工具已承认此副作用（backlog
  20261012-094646-devops）：跨树自检放行该配方（`PROJECT_ROOT` 在脚下树内且 `.git` 是文件），
  免 `--force`；mdgen-check 相对入参按 `PROJECT_ROOT` 解析。
- **档位映射自愈写运行态、不写跟踪面（2026-10-12，backlog 20261012-080410-devops）**：
  `refresh_models` 落 `sessions-runtime.json` 的 `models_resolved`，读取按「跟踪 > 运行态 >
  出厂默认」回退，跟踪值遮蔽解析值时只报不写。细则与脏跟踪面哨兵见技能
  `references/registry-runtime.md`。
- **worktree sync 与 `skills-deployed.json`（行尾敏感已修 2026-10-10，backlog
  20261010-003127-devops）**：`tree_hash` 哈希前 CRLF→LF 归一，故 worktree 内 sync 写的台账
  对 main 即正确；但 **pod 内部署副本是 gitignored、不随合并搬运**，合并后仍须在 main 跑
  `pysci-dev sync` 至 `sync --check` 全等；doctor 哨兵 [*]=台账哈希漂移、[!]=副本漂移。
- **uvx 型 MCP server 的 mcp 2.x 断供（2026-10-10 修，backlog 20261010-paper-search-mcp-down）**：
  SDK 2.x 移除了 `mcp.server.fastmcp`，上游依赖写 `mcp>=x` 无上限的 uvx server 解析到 2.x 即启动崩、
  pod 侧只见 ✗ Disconnected。**修法**：mcp.json args 前插 `["--with", "mcp<2", ...]` 钉住
  （lit/reviewer/deputy，commit 40058e4）。诊断 `echo '' | uvx <pkg>` 看 traceback；
  验证=initialize 握手见 serverInfo 即通。
- **`uv sync` 遇运行中 exe 锁：重命名占用者，勿杀进程（2026-10-12）**：Windows 能重命名运行中的
  exe 但不能删——sync 报「拒绝访问 (os error 5)」时**已先把其余入口删掉**（裸命令集体 MISSING）。
  `wmic process where ProcessId=N get CommandLine` 找占用者（本次=组长 `orch watch`）→
  `mv .venv/Scripts/pysci-orch.exe{,.x}` → sync 即成、入口全数重建。别 kill 组长进程
  （越权且阻塞 drain）。
- **任务书轻量 lint 规则表由 devops 维护（2026-10-10 建，backlog 20261010-084105-devops）**：
  `write_taskbook` 落盘前扫正文，命中 `state/taskbook-lint.json`（git 跟踪）里任一规则正则即
  **告警但不阻断**；增补=往 `rules` 追加 `{id, re, hint}`（re 为 Python 正则、JSON 内反斜杠
  双写）。fail-open，故改坏 JSON 会**静默失效**——test_taskbook_lint.py 在 CI 钉住「随附规则
  必须可编译」。
- **新增 pod 无需预注册 registry.json**（零技能 pod 也不进 `manifest.toml` targets，reviewer/twin
  先例，同一测试钉住）：脚手架只建目录+只读层文件，注册留给组长首次派发；口径见技能
  `references/pod-scaffold.md`。
- **改 guard（Node Stop/PreToolUse hook）的两条硬教训（2026-10-10）**：① guard 的
  fail-open（异常即放行）会**静默吞掉 guard 自身的 bug**（执行流写在 ESM 顶层 `const`
  辅助函数之前→TDZ 被 catch，13 例违规全绿；只有 pytest 直接 `node` 跑真 guard 才暴露），
  故 `run()` 调用放文件末尾。② Stop hook 验证别只看 exit code：Temp 造 pod 把 Stop 接到
  worktree 绝对路径 guard，跑完看**违规文件是否被会话自己改掉**+`num_turns`，再从
  `~/.qoder-cn/projects/<pod键>/<sid>.jsonl` 取注入回会话的 stderr 原文核对提示。
- **eval 压测跳的隔离面已内建（2026-10-12 E2 验收）**：压测跳会把本跳所学写进组员 `AGENTS.md`
  （跟踪面）→ 合并前多出无关脏面。`run_case` 现已两头夹：任务书附 `EVAL_HOP_NOTE` 禁改自维护层
  + 前后各拍该 pod 跟踪脏面快照，只还原本跳新增者（进台账 `reverted`）。口径见技能
  `references/harness-eval.md`「隔离性」段。
- **delivery-gate 现执法组员自维护层四条预算**：AGENTS.md / 自建 rules 的 description 合计 /
  自建 skills 的 description 合计 / 自建 SKILL.md 各 ≤8192B（落盘字节口径），且自建 rules 禁
  always_on；`PYSCI_DEPLOYED_SKILLS` 缺失时自建技能那一支不执法。
- **三层执法已齐（2026-10-10）**：组员侧=delivery-gate Stop、真本侧=pre-commit `harness-budget`
  （`guards/budget_check.py`，stdlib-only）、观测侧=doctor 三档审计。新钩子用 `language: python`
  不用 `system`（后者依赖提交上下文 PATH 里有 python，找不到就堵死一切提交）。
- **pod 脚手架的 git 跟踪面**：每 pod 仅 5 个文件入库；`bench/`·`outbox/`·`.qoder/skills/` 被 pod
  `.gitignore` 排除、`inbox/` 空目录不跟踪，四者均须在**合并后于 main 上** `mkdir` + `sync`
  再生。settings 模板两件事必带：两个 hook 接线 + `"agentsMdExcludes": ["**/leader-only.md"]`
  （写 `**/*.md` 会连 `basic.md` 公约数一起吞掉；勿把 pod 自建规则命名成 `leader-only*`）。
  其余细则（零技能 pod 裁定、注册面）见技能 `references/pod-scaffold.md`。
- **agentsMdExcludes 生效性机械哨兵（2026-10-10 建，backlog 20261010-165408-devops）**：
  doctor 验不出 CLI 真按 glob 排除，`pysci-dev probe` 在 Temp 造 fixture：两枚 always_on 规则带
  不可猜 token，control 与 withexclude 各跑一跳做**因果断言**（不灵敏只报 inconclusive、不误绿）；
  结论落**无时间戳**台账 `state/agents-md-excludes-probe.json`（staleness=hash{cli_version,排除面}，
  仅 CLI 升级或排除面变时复测），doctor 读它定三档。
- **交付协议滚动升级窗口（2026-10-12 E1 → backlog 002857 已修）**：派发进程 import 的解析器快照
  可能早于协议（旧版只认字面 `<tag>`）；现 parse_delivery 宽松兜底救回带属性交付并附
  「解析器早于协议」告警（进 outcome/run 日志）——见告警就重跑本单确认。
- **直跑 guard 探针的假绿坑（2026-10-12 两度踩）**：env 名是 **`PYSCI_POD`**（不是 PYSCI_POD_DIR），
  写错名⇒只读层形同虚设，拦截与对照都 exit 0；载荷要填 `last_assistant_message`（guard 不读
  transcript_path，只喂 transcript 会空跑全绿）；`PYSCI_PROJECT_ROOT` 须用 `cygpath -w` 形态
  （`/tmp` 被 node 当 `D:\tmp` → 词表 ENOENT → fail-open 放行）。
- **判位格式须吃未裁剪输出；合并前交集判据（2026-10-12，backlog 20261012-082932-devops）**：
  `_run_git` 返回 strip 后的输出，而 porcelain 首行的**前导空格是判位符**（`" M a"`→`"M a"`）
  ⇒ 解析器静默漏首项；mock 单测全绿，只有真 git 集成测试暴露，读定长格式须走 `_git_proc`。
  「本次合并触碰哪些文件」按 merge-base 起算（`main..HEAD` 在分支落后时把别人的提交也算进来，
  据此误交过一次 blocked）——已内建进 `pysci-dev worktree merge`，勿手算。
