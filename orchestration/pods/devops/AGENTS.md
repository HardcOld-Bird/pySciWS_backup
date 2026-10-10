# devops 组员长期记忆

> 本文件由 devops 组员自维护（每次会话启动自动加载）。只记稳定事实与约定。

## 身份

- pySci 编排体系 devops 组员（基础设施维护专员），charter 见 `.qoder/rules/charter.md`。
- 专属工具：`pysci-dev`（doctor/sync/worktree）；实现于 `src/pysci/skills/devops/`。
- 作业方式：worktree 隔离 + cz commit + commit 后 `timeout 25 git push origin main`
  （best-effort，失败忽略、不重试、不设代理；用户裁决 2026-10-10，超时硬要求防挂起阻塞 drain）。

## 经验（工作中积累）

- **worktree 内跑测试/CLI 的环境选法（2026-10-10 实测，backlog 20261010-orch-watch）**：
  在 worktree 里直接 `uv run …` 会在 `<wt>/.venv` **新建空环境**（首次即 `Creating virtual
  environment`；配 `--no-sync` 时缺 pyyaml 等依赖，import 直接崩）。可靠做法：借用主 venv
  的解释器 + PYTHONPATH 覆盖代码面——
  `PYTHONUTF8=1 PYTHONPATH="<wt>/src" "D:/…/pySciWS/.venv/Scripts/python.exe" -m pytest "<wt>/tests" -q`。
  副作用（可利用也可防）：`pysci.paths.PROJECT_ROOT` 由 paths.py 就近找 `pyproject.toml` 决定，
  故此法下 ORCH_STATE_ROOT/PODS_ROOT 全部指向 **worktree** 内副本——跑 `orch approve`、
  `dev sync`、`_drain-devops` 等会改 worktree 的 `state/*.json`（backlog.json 被 git 跟踪！）。
  实验后必须 `git checkout -- orchestration/state/backlog.json` 之类还原，别把演练队列带进提交。
- **worktree sync 与 `skills-deployed.json`（已修复 2026-10-10，backlog 20261010-003127-devops）**：
  旧坑——本机 `core.autocrlf=true` 无 `.gitattributes`，worktree 检出把文本真本 smudge 成 CRLF、
  main 是混合行尾，而 `sync.tree_hash` 读**原始字节**，致同一真本两侧哈希不同、合并后 `sync --check`
  报假漂移。**修复（方案 b）**：`tree_hash` 哈希前规范化 CRLF→LF（技能真本全为 .md/.toml 文本、无
  二进制，安全），哈希只反映内容。实测：document-writing 旧 raw-byte 哈希 main=dbac3b85/wtree=f97e5e09
  （漂移，复现建议原值），规范化后两侧均 e43ec523；九技能全 match。**现状**：worktree 内 sync 提交的
  `skills-deployed.json` 对 main 即正确，**无需**再「合并后在 main 重跑」；doctor 已增「部署台账巡检
  （sync --check）」哨兵，捕获记录哈希漂移（[*]）与副本漂移（[!]）。未选方案 a（根 .gitattributes）：
  需仓库级 renormalize、触碰每个文本文件且与用户本地 autocrlf 交互，blast radius 大。
- **uvx 型 MCP server 的 mcp 2.x 断供风险（2026-10-10 修，backlog 20261010-paper-search-mcp-down）**：
  mcp Python SDK 2.x 移除了 `mcp.server.fastmcp`（FastMCP 更名 MCPServer），凡上游依赖写
  `mcp>=x` 无上限的 uvx server（如 paper-search-mcp 0.1.4）会在新环境解析到 2.x 而启动即崩、
  pod 侧只见 ✗ Disconnected。**修法**：mcp.json args 前插 `["--with", "mcp<2", ...]` 钉住
  （lit/reviewer/deputy 已修，commit 40058e4）。诊断范式：`echo '' | uvx <pkg>` 直看 stderr
  traceback；修复验证用 initialize 握手（stdin 发 JSON-RPC initialize，见 serverInfo 即通）。
  其他 uvx server（arxiv-mcp-server、blender-mcp、zotero-mcp）暂正常，同类症状先查此项。
- **任务书轻量 lint 规则表由 devops 维护（2026-10-10 建，backlog 20261010-084105-devops）**：
  `write_taskbook`（dispatch.py，dispatch/plan run/drain 唯一落盘口）落盘前扫正文，命中
  `orchestration/state/taskbook-lint.json`（git 跟踪）里任一规则的正则即**打印告警但不阻断**
  派发。**增补规则**：向该 JSON 的 `rules` 追加 `{id, re, hint}`（re 为 Python 正则、JSON 内
  反斜杠双写）；fail-open（文件缺失/损坏/单条正则非法均跳过），故改坏 JSON 会**静默失效**——
  test_taskbook_lint.py::test_repo_shipped_rules_file_valid 在 CI 钉住「随附规则必须可编译」。
  首期规则 figures-two-positional-args（build/preview/audit 误跟两个位置参数）。与
  20261009-taskbook-cli-signature（charter/文档侧改正本）互补，勿重复。
- **新增 pod 无需预注册 registry.json**：`orch dispatch` 前调 `ensure_member_defaults`，pod 目录
  存在即自动生成默认成员条目（flash/60轮/7200s/`.qoder/mcp.json`）。`doctor` 按 `PODS_ROOT`
  目录计数、也不依赖 registry。故脚手架新 pod 只建目录+只读层文件，注册留给组长首次派发。
- **改 guard（Node Stop/PreToolUse hook）的两条硬教训（2026-10-10，backlog 20261010-budget-delivery-gate）**：
  ① guard 的 fail-open（异常即放行）会**静默吞掉 guard 自身的 bug**——首版把执行流写在
  ESM 顶层 `const` 辅助函数之前，TDZ ReferenceError 被 catch 掉，13 例违规全绿通过；只有
  用 pytest 直接 `node` 跑真 guard（`tests/skills/orchestration/test_delivery_gate_budget.py`）
  才暴露。故 `run()` 调用必须放文件末尾。② Stop hook 的验证别只看 exit code：探针法=在
  Temp 造 pod（settings.json 把 Stop 接到 worktree 里的绝对路径 guard），headless 跑完看
  **违规文件是否被会话自己改掉** + `num_turns`（bad 12000B→12B/3 轮 vs ok 200B 不动/1 轮），
  再从 `~/.qoder-cn/projects/<pod键>/<sid>.jsonl` 里取注入回会话的 stderr 原文核对提示是否准确。
- **delivery-gate 现执法组员自维护层四条预算**（AGENTS.md ≤8192B / 自建 rules 禁
  always_on / 自建 rules·skills 的 description 合计各 ≤8192B / 自建 SKILL.md ≤8192B），
  口径=落盘字节（CRLF 计税）；`PYSCI_DEPLOYED_SKILLS` 缺失时自建技能那一支不执法。
  细节写进 devops 技能「harness 预算三层执法」节 + `references/budget-layers.md`。
- **三层执法已齐（2026-10-10）**：组员侧=delivery-gate Stop、真本侧=pre-commit
  `harness-budget`（`orchestration/guards/budget_check.py`，stdlib-only，钩子空 venv 里
  没装本项目）、观测侧=doctor 三档审计。新钩子用 `language: python` 不用 `system`——
  后者依赖提交上下文 PATH 里有 python，找不到就堵死一切提交。本条所属 devops SKILL.md
  当时已 8178B，加内容必超 → 按分册程序把细节移进 `references/budget-layers.md`，
  SKILL.md 回落到 7713B。改真本前先 `wc -c` 是省一轮返工的习惯。
- **pod 脚手架的 git 跟踪面**：每 pod 仅 5 个文件入库（`.gitignore`/`.qoder/settings.json`/
  `.qoder/mcp.json`/`.qoder/rules/charter.md`/`AGENTS.md`）；`bench/`·`outbox/`·`.qoder/skills/`
  被 pod `.gitignore` 排除，`inbox/` 空目录 git 不跟踪——三者均须在**合并后于 main 上**用
  `mkdir` + `sync` 再生（worktree 内建了也不随合并 transfer）。settings 模板两件事必带：
  两个 hook 接线 + `"agentsMdExcludes": ["**/leader-only.md"]`（组长专属根规则；写 `**/*.md`
  会连 `basic.md` 公约数一起吞掉）。勿把 pod 自建规则命名成 `leader-only*`（撞排除 glob）。
- **agentsMdExcludes 生效性机械哨兵（2026-10-10 建，backlog 20261010-165408-devops）**：
  静态 doctor 只验 settings 写了 glob、验不出 CLI 真据此排除。`pysci-dev probe` 在 Temp
  `git init` 造自足 fixture：两枚 always_on 规则各带不可猜 token（keep/hide），control
  （不排除）与 withexclude（`**/hide.md`）各跑一跳「列出上下文所有 TOK- 标记」，因果断言
  （hide 在 control 现 + 在 withexclude 消）才判 ok，不灵敏则 inconclusive（重试、绝不误绿）。
  结论写**无时间戳**台账 `state/agents-md-excludes-probe.json`（staleness=hash{cli_version,
  排除面}，仅 CLI 升级或 glob 面变时复测），doctor 读它定三档。与预算审计互补：
  预算量落盘字节，probe 验实际注入。
