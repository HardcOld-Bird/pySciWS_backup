# devops 经验底账（事故经过与判据）

`AGENTS.md` 是每跳全文注入档，只留约束；本文件是**按需档**——动到对应面之前先读这里。
「为什么是这样」比「是什么」更容易失传，所以每条都带实测出处（日期/backlog 号）。

## 1. worktree 内跑测试/CLI（2026-10-12 定型）

`PYTHONPATH="<wt>/src" uv run --no-sync pytest "<wt>/tests" -q` 复用主 `.venv`，**前提=cwd
在 worktree 之外**——cd 进去 uv 会就地新建空 venv，缺 pyyaml 即崩。副作用：`PROJECT_ROOT`
就近认 `pyproject.toml` ⇒ state 写进 worktree 副本，跑完逐文件还原；**worktree 内禁
`git checkout -- .`**（一次抹光全部未提交改动）。测试真派生 worker 与 merge 拖他树锁两面
已收口（backlog 20261012-191512-devops：`_spawn_detached` 咽喉闸 + cp 排除 `*.lock`）。
配方与兜底见 `worktree-merge.md` §1-§2。

## 2. 判决包流（2026-10-12 E5b）

`tools/verdict.py`=共享库（§0.9 组员复用，不另起 CLI）。approve 必带
`--jury`/`--bench`/`--override` 之一；例外通道=defect×当跳 blocked，判据读落盘固化的
`delivery:` 行（事后反查台账会失效）。J 出 B 自动驳回且**不计 override 率**——账只有一本
=`preference-journal.jsonl` 的 `channel`。M 路审批锚 HEAD，`backlog_complete` 自动排 A/B
三态落地（改善→销账／未改善→立案 `source=post-verify`／不可判→`escalate=twin` 待组长终裁）。
归档必须碰撞安全（重名覆盖=销毁审计原文）。细则见 `verdict-package.md`（orchestration 技能）。

## 3. 档位映射与运行态面（2026-10-12）

`refresh_models` 落 `sessions-runtime.json` 的 `models_resolved`，读取按「跟踪 > 运行态 >
出厂默认」回退；跟踪值遮蔽解析值时**只报不写**。worktree 内 sync 的 `skills-deployed.json`
对 main 即正确（`tree_hash` 哈希前 CRLF→LF 归一），但 **pod 内副本 gitignored、不随合并
搬运**，合并后须在 main 跑到 `sync --check` 全等。doctor 哨兵 [*]=台账漂移、[!]=副本漂移。

## 4. uvx 型 MCP server 的 mcp 2.x 断供（2026-10-10 修）

上游 `mcp>=x` 无上限解析到 2.x 即启动崩（SDK 移了 `mcp.server.fastmcp`），pod 侧只见
✗ Disconnected。修法=mcp.json args 前插 `["--with", "mcp<2", ...]`；诊断 `echo '' | uvx
<pkg>`，验证=握手见 serverInfo。

## 5. 主树同步入口一律 `pysci-dev venv-sync`（2026-10-12 固化）

uv 撞运行中 exe（典型=组长 orch watch）会**先删光其余入口**再报 os error 5；工具=让路→
sync→让回原地→核对入口。**原名缺失时 `.locked-*` 就是唯一原件，删前先查**；别 kill 组长
进程。隐式 sync 坑与三态处置见 `uv-venv-sync.md`。

## 6. CLI 范式=uv run 必带 --no-sync（2026-10-12）

src/registry checks/技能真本/basic.md 四面同口径，快照测试 `test_cli_paradigm_sync.py`
扫语料点火；charter 只读面登记 `READONLY_PENDING`，文件合规后哨兵反红提醒删豁免（防永久
假绿）；人工 grep 会漏计。

## 7. 任务书轻量 lint 表（2026-10-10）

`write_taskbook` 落盘前扫正文，命中 `state/taskbook-lint.json`（跟踪）任一正则即**告警不
阻断**；增补=`rules` 追加 `{id, re, hint}`（JSON 内反斜杠双写）。fail-open⇒改坏 JSON 静默
失效。新增 pod 无需预注册 registry.json（零技能 pod 也不进 `manifest.toml` targets，
reviewer/twin 先例）：脚手架只建目录+只读层，注册留给组长首次派发。

## 8. guard（Node Stop/PreToolUse hook）三条硬教训（2026-10-10/12）

① fail-open 会**静默吞掉 guard 自己的 bug**：执行流写在 ESM 顶层 `const` 前→TDZ 被 catch，
13 例违规全绿；只有 pytest 直接 `node` 跑真 guard 才暴露，故 `run()` 放文件末尾。
② 验证别只看 exit code：看违规文件是否被会话自己改掉＋`num_turns`，再从
`~/.qoder-cn/projects/<pod键>/<sid>.jsonl` 取注入的 stderr 原文。
③ 直跑 guard 探针的假绿坑（两度踩）：env 名是 **`PYSCI_POD`**（不是 PYSCI_POD_DIR），写错
名⇒只读层形同虚设，拦截与对照都 exit 0；载荷填 `last_assistant_message`（guard 不读
transcript_path）；`PYSCI_PROJECT_ROOT` 须 `cygpath -w` 形态（`/tmp` 被 node 当 `D:\tmp`→
ENOENT→fail-open 放行）。

## 9. 预算三层与 harness 面（2026-10-10/12）

组员侧=delivery-gate Stop（AGENTS.md／自建 rules·skills 的 description 合计／自建 SKILL.md
各 ≤8192B **落盘字节**，自建 rules 禁 always_on，`PYSCI_DEPLOYED_SKILLS` 缺失时自建技能那一支
不执法）、真本侧=pre-commit `harness-budget`、观测侧=doctor 三档。新钩子用 `language: python`
不用 `system`（后者 PATH 缺 python 就堵死一切提交）。

**钩子计税口径的缺口（本分册自曝）**：`harness-budget` 量的是**索引字节**（LF），而 basic.md §1
的口径是**落盘字节**（CRLF 计税）——110 行的文件能过钩子却在 main 上超 8192B（本文件初稿
8084→检出 8194 即此坑）。写完长 references 后须在合并**之后**于 main 复核
`find orchestration/skills -name "*.md" -size +8192c`，别只信钩子绿。

脚手架 5 文件入库面、agentsMdExcludes 写法坑与 probe 因果断言见 `pod-scaffold.md`；分册次序与
豁免见 `budget-layers.md`；eval 压测跳的 `EVAL_HOP_NOTE` 与脏面还原见 `harness-eval.md`。

## 10. 交付协议两面（2026-10-12 E1 + 132138）

**滚动升级窗口**：派发进程 import 的解析器快照可能早于协议，parse_delivery 宽松兜底救回
带属性交付并附「解析器早于协议」告警——见告警就重跑本单。
**词表单一源**：`<blocked tags>` 与 `<infra_suggestion kind>` 只有一份=
`state/failure-tags.json`（Python `failure_tags.load_*()` 与 Node gate 同读，增补改文件不改
代码）。kind 四面缺一即漏网：gate **逐块**校验并拦多块交付（解析器只取第一块）、dispatch
落盘前复核、approve 入队前复核、**读数面 ledger**（判「已入库」而非「文本里有块」，被词表
闸拦下的跳行尾打 `建议✗未入库`——分裂要可见，不是静默抹掉）。**双源漂移同族坑**：approve
用块内解析器（跳首空行前扫 `kind:`）、入库用全文扫描器 ⇒ 正文以空行开头的提案判据取空即被
拒（回显 `kind=` 空即此症）。

## 11. shell/git 读数的三个静默坑（2026-10-12）

- **判位格式须吃未裁剪输出**：`_run_git` 返回 strip 后输出，而 porcelain 首行的**前导空格
  是判位符**（`" M a"`→`"M a"`）⇒解析器静默漏首项；mock 全绿，只有真 git 集成测试暴露，
  读定长格式须走 `_git_proc`。
- **合并交集按 merge-base 起算**：`main..HEAD` 在分支落后时把别人的提交也算进来（据此误交
  过一次 blocked）——已由 `pysci-dev worktree merge` 内建，勿手算。
- **取 rc 别经管道**：`orch … | grep -v …` 后 `${PIPESTATUS[0]}` 是 grep 的 rc；同名建议归档
  后重跑必 rc=2「找不到」。取证=裸命令＋独立夹具副本。

## 12. 上下文膨胀的正确读数（2026-10-12）

est_tokens 量的是**吞吐**（本跳往会话档追加多少字符），不是上下文占用：devops 78 跳实测与
`num_turns` 相关 **+0.81**、与 `ctx_ratio` 相关 **−0.11**。故巡检读侧两支分开
（`check_cost_curve`=单跳工作量／`check_context_pressure`=末跳 ctx_ratio）。组员侧唯一自有
杠杆是压薄自维护层（AGENTS.md 曾贴 8192B 顶，本文件即那次蒸馏的落点）。

## 13. 可注入测试缝经模块属性调用

orch 工具里「测试会 patch 的注入缝」（`_sleep`/`_spawn_detached` 等）禁 `from .drain import`
值绑定，须写 `drain._sleep(...)`——绑定会让 monkeypatch 打不到调用点 → worker 真睡/挂死
（`test_worker_seams.py` 钉住）。
