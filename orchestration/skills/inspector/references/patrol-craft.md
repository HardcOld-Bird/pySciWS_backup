# 巡检取证手册（菜单项配方 + 假读数坑）

SKILL.md 只留决策面；本文放**怎么取证**与**哪些读数是假的**。

## 1. 各菜单项的取证配方

| 类别 | 起手命令 | 看什么 |
|---|---|---|
| `doctor-sweep` | `pysci-dev doctor` | 每条 [!] 是否可复现；复现不了的别报（可能是本机脏树所致，见 §2 首条） |
| `transcript-audit` | 取 `~/.qoder-cn/projects/<pod键>/<sid>.jsonl`，grep `pod-guard` | 拦截记录之后有没有「换工具/换路径」把同一件事做成的痕迹 |
| `self-maint-audit` | 读 pod 的 `AGENTS.md` + `.qoder/rules/*.md` | 文中承诺的动作在代码里是否还存在；`wc -c` 是否触 8192 |
| `code-walk` | 选一个 `src/pysci/**.py` 读全文 | 静默 `except Exception: pass`、`if False else` 式兼容垫片、fail-open 吞 guard 自身 bug |
| `pipeline-self` | 读 `tools/{orch,drain,benchd,inspectd,verdict,jury}.py` 与 guards | **文档承诺但代码没实现**的条款（最坏的一类漂移） |
| `intent-drift` | README/charter 摘一句 → 写成命令 | 实际输出与宣称不符即 mechanical；符合与否需判断则 judgment |
| `design-rubric` | `rubrics/design-review.md` 三维 | 见 §3 |
| `bench-probe` | `orch bench-queue --all` + `pysci-bench report`（只读） | 缺口提案，不自己跑压测 |

## 2. 假读数坑（先排除这些，再报发现）

- **worktree 里的 `PROJECT_ROOT` 就近认 `pyproject.toml`**：在 worktree 内跑 CLI/测试，
  state 会写到 worktree 副本。据此看到的「state 漂移」多半是环境现象，不是缺陷。
- **Git Bash 把 `/` 前缀参数转成路径**：`tasklist /FI`、`findstr /C` 会静默假阴性。
  判断进程存活只经 `pysci-orch status` 一类代码面。
- **经管道取 rc**：`cmd | grep -v …` 之后 `${PIPESTATUS[0]}` 才是 cmd 的 rc；
  直接 `$?` 会读到 grep 的。
- **fail-open 的守卫会静默放行**：guard 自己有 bug 时「全绿」是假绿。要验拦截面，
  直跑 guard 时 env 名是 `PYSCI_POD`（不是 `PYSCI_POD_DIR`），载荷填
  `last_assistant_message`（guard 不读 `transcript_path`）。
- **归档重名**：同名建议归档后重跑命令会得到 rc=2「找不到」——换一份干净副本再取。
- **他人未提交的脏面**：`git status` 里不属于本任务的东西是开发者现场，别当成体系缺陷。

## 3. 设计审查三维（rubric 的操作化）

判据要能落到一条命令或一处代码位置，否则只能算感受：

- **社区优先**：该模块做的事有没有维护活跃的社区工具可做？给得出包名/仓库与
  版本活跃度（判断类）；
- **反冗余**：同一语义在两处实现。本项目的高发形态：解析器两份（块内 vs 全文扫描）、
  词表多处硬编码、阈值默认值同时写在代码与文档里。给得出两处文件+行号；
- **LLM 直觉友好**：使用一个模块需要的**隐式知识**有多少？典型反模式：参数名与真实
  语义不符、必须记住的顺序依赖、失败时不给 [NEXT]。以「一个新会话第一次调用它会不会
  猜错」为判据，并给出猜错的具体位置。

此类发现属判断类，必须 `submit --jury`。

## 4. 案卷可判性（写不好会得 inconclusive）

`verdict.assemble_case` 用你的 `--title`/`--symptom`/`--expect` 组两选项。两个选项
**必须内容可区分**：如果 A/B 读起来是同义反复，双序对局会给出互相矛盾票 →
`inconclusive` → `escalate=twin` 挂起。写「实施 X（收益：可测的 Y）」对
「维持现状（代价：Z 继续存在）」，别写「改 / 不改」。

## 5. 什么不该报（噪音比漏报更贵）

- 单次观察到的性能抖动、无复现命令的「感觉变慢了」；
- 已明确登记在 backlog/queue 的在途工作（先 `finding list` 与 `orch status` 查重）；
- 别人会话的临时脏面、`.venv`/`__pycache__` 一类的可再生物；
- 「建议加更多文档」——除非你能指出哪一条不读就会做错事（basic.md 的收录门槛）。

发现数量不是绩效。**一轮零发现是合法交付**。
