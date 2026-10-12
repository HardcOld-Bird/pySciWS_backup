# devops 组员长期记忆

> 本文件由 devops 组员自维护（每次会话启动自动加载）。只记稳定事实与约定。

## 身份

- pySci 编排体系 devops 组员（基础设施维护专员），charter 见 `.qoder/rules/charter.md`。
- 专属工具：`pysci-dev`（doctor/sync/worktree）；实现于 `src/pysci/skills/devops/`。
- 作业方式：worktree 隔离 + cz commit + commit 后限时 best-effort push（超时是硬要求——
  挂起会阻塞 drain）。

## 经验（工作中积累）

- **worktree 内跑测试/CLI（2026-10-12 定型）**：`PYTHONPATH="<wt>/src" uv run --no-sync pytest
  "<wt>/tests" -q` 复用主 `.venv`，**前提=cwd 在 worktree 之外**（cd 进去 uv 就地新建空 venv，
  缺 pyyaml 即崩）。副作用：`PROJECT_ROOT` 就近认 `pyproject.toml` ⇒ state 写进 worktree 副本，
  跑完逐文件还原——**worktree 内禁 `git checkout -- .`**（一次抹光全部未提交改动）。测试真派生
  worker 与 merge 拖他树锁两面已收口（#78：咽喉闸 + cp 排除 `*.lock`）。配方、判据与兜底见
  技能 `references/worktree-merge.md` §1-§2。
- **判决包流（2026-10-12 E5b）**：`tools/verdict.py`=共享库（§0.9 组员复用，不另起 CLI）。
  approve 必带 `--jury`/`--bench`/`--override` 之一；例外通道=defect×当跳 blocked，判据读落盘
  固化的 `delivery:` 行（事后反查台账会失效）。J 出 B 自动驳回且**不计 override 率**——账只有
  一本=`preference-journal.jsonl` 的 `channel`。M 路审批锚 HEAD，`backlog_complete` 自动排 A/B
  三态落地（改善→销账／未改善→立案 `source=post-verify`／不可判→`escalate=twin` 待组长终裁）。
  归档必须碰撞安全（重名覆盖=销毁审计原文）。细则见 `references/verdict-package.md`。
- **档位映射自愈写运行态、不写跟踪面（2026-10-12）**：`refresh_models` 落
  `sessions-runtime.json` 的 `models_resolved`，读取按「跟踪 > 运行态 > 出厂默认」回退，跟踪值
  遮蔽解析值时只报不写。细则见 `references/registry-runtime.md`。
- **worktree sync 与 `skills-deployed.json`**：`tree_hash` 哈希前 CRLF→LF 归一，故 worktree 内
  sync 的台账对 main 即正确；但 **pod 内副本 gitignored、不随合并搬运**，合并后须在 main 跑到
  `sync --check` 全等。doctor 哨兵 [*]=台账漂移、[!]=副本漂移。
- **uvx 型 MCP server 的 mcp 2.x 断供（2026-10-10 修）**：上游 `mcp>=x` 无上限解析到 2.x 即启动崩
  （SDK 移了 `mcp.server.fastmcp`）、pod 侧只见 ✗ Disconnected。修法=mcp.json args 前插
  `["--with", "mcp<2", ...]`；诊断 `echo '' | uvx <pkg>`，验证=握手见 serverInfo。
- **主树 `uv sync` 一律走 `pysci-dev venv-sync`（2026-10-12 固化）**：uv 撞运行中 exe（典型=组长
  orch watch）会**先删光其余入口**再报 os error 5；工具=让路→sync→让回原地→核对入口。**原名
  缺失时 `.locked-*` 就是唯一原件，删前先查**；别 kill 组长进程。隐式 sync 坑与三态处置见
  `references/uv-venv-sync.md`。
- **CLI 范式=uv run 必带 --no-sync（2026-10-12）**：src/registry checks/技能真本/basic.md 四面
  同口径，快照测试 `test_cli_paradigm_sync.py` 扫语料点火；charter 只读面登记 `READONLY_PENDING`，
  文件合规后哨兵反红提醒删豁免（防永久假绿）；人工 grep 会漏计。
- **任务书轻量 lint 表由 devops 维护（2026-10-10）**：`write_taskbook` 落盘前扫正文，命中
  `state/taskbook-lint.json`（跟踪）任一正则即**告警不阻断**；增补=`rules` 追加
  `{id, re, hint}`（JSON 内反斜杠双写）。fail-open⇒改坏 JSON 静默失效。
- **新增 pod 无需预注册 registry.json**（零技能 pod 也不进 `manifest.toml` targets，reviewer/twin
  先例）：脚手架只建目录+只读层，注册留给组长首次派发。
- **改 guard（Node Stop/PreToolUse hook）两条硬教训（2026-10-10）**：① fail-open 会**静默吞掉
  guard 自身的 bug**（执行流写在 ESM 顶层 `const` 前→TDZ 被 catch，13 例违规全绿；只有 pytest
  直接 `node` 跑真 guard 才暴露），故 `run()` 放文件末尾。② 验证别只看 exit code：看违规文件是否
  被会话自己改掉＋`num_turns`，再从 `~/.qoder-cn/projects/<pod键>/<sid>.jsonl` 取注入的 stderr 原文。
- **eval 压测跳的隔离面已内建（2026-10-12 E2）**：`run_case` 两头夹——任务书附 `EVAL_HOP_NOTE`
  禁改自维护层＋前后拍 pod 跟踪脏面快照，只还原本跳新增者（进台账 `reverted`）。口径见
  `references/harness-eval.md`。
- **harness 预算三层执法（2026-10-10 齐）**：组员侧=delivery-gate Stop（AGENTS.md／自建 rules・
  skills 的 description 合计／自建 SKILL.md 各 ≤8192B 落盘字节，自建 rules 禁 always_on，
  `PYSCI_DEPLOYED_SKILLS` 缺失时自建技能那一支不执法）、真本侧=pre-commit `harness-budget`、
  观测侧=doctor 三档。新钩子用 `language: python` 不用 `system`（后者 PATH 缺 python 就堵死一切提交）。
- **pod 脚手架的 git 跟踪面**：每 pod 仅 5 个文件入库；`bench/`·`outbox/`·`.qoder/skills/` 被排除、
  `inbox/` 空目录不跟踪，四者均须**合并后于 main 上** `mkdir`+`sync` 再生。settings 模板必带两个
  hook 接线 + `"agentsMdExcludes": ["**/leader-only.md"]`（写 `**/*.md` 会连 `basic.md` 一起吞掉；
  勿把 pod 自建规则命名成 `leader-only*`）。其余见 `references/pod-scaffold.md`。
- **agentsMdExcludes 生效性机械哨兵（2026-10-10）**：doctor 验不出 CLI 真按 glob 排除，`pysci-dev
  probe` 造带不可猜 token 的 fixture 跑两跳做**因果断言**（不灵敏只报 inconclusive、不误绿）；
  结论落 `state/agents-md-excludes-probe.json`。
- **交付协议滚动升级窗口（2026-10-12 E1）**：派发进程 import 的解析器快照可能早于协议；
  parse_delivery 宽松兜底救回带属性交付并附「解析器早于协议」告警——见告警就重跑本单。
- **协议词表四面执法＋双源漂移（2026-10-12）**：`<blocked tags>` 与 `<infra_suggestion kind>`
  词表只有一份=`state/failure-tags.json`（Python `failure_tags.load_*()` 与 Node gate 同读，增补
  改文件不改代码）。kind 四面缺一即漏网：gate **逐块**校验并拦多块交付（解析器只取第一块）、
  dispatch 落盘前复核、approve 入队前复核、**读数面 ledger**（backlog 20261012-132138-devops：
  `infra_suggestion` 判「已入库」而非「文本里有块」，被词表闸拦下的跳行尾打 `建议✗未入库`——
  分裂要可见，不是静默抹掉）。**双源漂移同族坑**：approve 用块内解析器（跳首空行前
  扫 `kind:`）、入库用全文扫描器 ⇒ 正文以空行开头的提案判据取空即被拒（回显 `kind=` 空即此症）。
- **直跑 guard 探针的假绿坑（2026-10-12 两度踩）**：env 名是 **`PYSCI_POD`**（不是 PYSCI_POD_DIR），
  写错名⇒只读层形同虚设，拦截与对照都 exit 0；载荷填 `last_assistant_message`（guard 不读
  transcript_path）；`PYSCI_PROJECT_ROOT` 须 `cygpath -w` 形态（`/tmp` 被 node 当 `D:\tmp`→ENOENT→
  fail-open 放行）。
- **判位格式须吃未裁剪输出；合并前交集判据（2026-10-12）**：`_run_git` 返回 strip 后输出，而
  porcelain 首行的**前导空格是判位符**（`" M a"`→`"M a"`）⇒解析器静默漏首项；mock 全绿，只有真
  git 集成测试暴露，读定长格式须走 `_git_proc`。「本次合并触碰哪些文件」按 merge-base 起算
  （`main..HEAD` 在分支落后时把别人提交也算进来，据此误交过一次 blocked）——已内建，勿手算。
- **取 rc 别经管道（2026-10-12 两度误判）**：`orch … | grep -v …` 后 `${PIPESTATUS[0]}` 是 grep 的
  rc；同名建议归档后重跑必 rc=2「找不到」。取证=裸命令＋独立夹具副本。
