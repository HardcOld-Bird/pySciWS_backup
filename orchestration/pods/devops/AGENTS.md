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
- **pod 脚手架的 git 跟踪面**：每 pod 仅 5 个文件入库（`.gitignore`/`.qoder/settings.json`/
  `.qoder/mcp.json`/`.qoder/rules/charter.md`/`AGENTS.md`）；`bench/`·`outbox/`·`.qoder/skills/`
  被 pod `.gitignore` 排除，`inbox/` 空目录 git 不跟踪——三者均须在**合并后于 main 上**用
  `mkdir` + `sync` 再生（worktree 内建了也不随合并 transfer）。settings 模板两件事必带：
  两个 hook 接线 + `"agentsMdExcludes": ["**/leader-only.md"]`（组长专属根规则；写 `**/*.md`
  会连 `basic.md` 公约数一起吞掉）。勿把 pod 自建规则命名成 `leader-only*`（撞排除 glob）。
