# harness 预算三层执法与分册处置（devops 技能分册）

> basic.md §1 的三档上限由三层机制执法；本册记各层的实现位置、判据口径与维护注意点。
> 决策与调用面（改完要跑什么）留在 `SKILL.md`，需要动某一层时再读这里。

## 0. 计量口径（三层共用）

一律按**落盘字节**：`wc -c` / `find -size +8192c` / `Path.stat().st_size` /
guard 的 `fs.statSync().size` 全部一致。Windows 本机 `core.autocrlf=true` 且无
`.gitattributes`，检出为 CRLF，故**每个换行也多计 1 字节**——这才是 harness 真正
注入的税。判据是 `> 8192`（等于不拦），与 `find -size +8192c` 严格一致。

三档：全量注入档（charter/AGENTS.md/SKILL.md/根 rules，每文件 ≤8192B）、常驻暴露档
（条件式 rules 与技能 `description` 行，每类合计 ≤8192B）、按需档（`references/*.md`
等，单文件 ≤8192B、总量不限）。

## 1. 组员侧硬闸：`orchestration/guards/delivery-gate.mjs`（Stop hook）

执法**自维护层**四条：

- ① pod `AGENTS.md` ≤8192B；
- ② `.qoder/rules/*.md` 中非 `charter.md` 者不得 `trigger: always_on`（大小写、
  `"always_on"` 引号形式、`always-on` 连字符形式都判；`charter.md` 豁免——它就是该
  槽位的合法持有者）；
- ③ 自建 rules 的 `description` 行合计 ≤8192B；自建 skills 另立一本账同样 ≤8192B
  （两本账各自判，不合并、不连坐）；
- ④ 每个自建 `SKILL.md` ≤8192B。

`description` 字节 = 含 `description: ` 前缀那一行的落盘字节，折行的块标量（下一行有
缩进）一并计入。

三条设计约束：

- **格式与预算合并成一轮上报**。若格式不过就先 exit 2，`stop_hook_active` 会让下一次
  Stop 无条件放行，预算违规必然漏过一次拦截。
- **部署副本豁免**：`PYSCI_DEPLOYED_SKILLS`（runner 注入，`;` 分隔，与 pod-guard 同一套）
  里的技能目录不算自建；**该变量缺失时自建技能那一支整体不执法**——分不清自建与副本时，
  让组员为 devops 真本的尺寸买单是错的。
- **fail-open 会吞掉 guard 自己的 bug**：预算段包在 try 里，异常即静默放行。首版把执行流
  写在 ESM 顶层 `const size/mdIn/…` 之前（函数声明提升，`const` 有 TDZ），
  `ReferenceError` 被 catch 掉 → 13 例应拦的全绿。教训两条：`run()` 调用必须放**文件末尾**；
  验证 Stop hook 不能只看 exit code——探针法=Temp 造 pod（settings.json 把 Stop 接到
  绝对路径 guard），headless 跑完看**违规文件是否被会话自己改掉** + `num_turns`
  （bad 12000B→12B/3 轮 vs ok 200B 不动/1 轮），再从
  `~/.qoder-cn/projects/<pod键>/<sid>.jsonl` 取注入回会话的 stderr 原文核对提示是否准确。

测试：`tests/skills/orchestration/test_delivery_gate_budget.py`（pytest 用 `node` 跑真
guard，含「真实十 pod 现状全过」哨兵）。

## 2. 真本侧硬闸：`.pre-commit-config.yaml` 的 `harness-budget`

管**组长与 devops 自己**改真本时的文档膨胀，拦三类路径每文件 ≤8192B：
`orchestration/skills/*/SKILL.md`、`orchestration/skills/*/references/*.md`、
`.qoder/rules/basic.md` 与 `leader-only.md`。实现是 stdlib-only 的
`orchestration/guards/budget_check.py`，超限打印 文件+实测字节+上限+档位+整改动作。

- **`language: python` 而非 `system`**：钩子解释器由 pre-commit 自身的 venv 提供，
  不依赖提交上下文（TUI/IDE/CI）的 PATH 里有 python——`language: system` 一旦在某个
  上下文里找不到解释器就会堵死**一切**提交。
- **脚本不 import `pysci`**：钩子跑在空 venv 里，装的是 pre-commit 的依赖而非本项目；
  所以范围判定用路径正则自足。上限值与档位集合和 `budget.py` 的一致性由
  `tests/skills/devops/test_budget_precommit.py` 钉住（含 `files:` 正则与脚本 `TIERS`
  两层判据必须同进同退——不一致就会留下「超限但根本不进钩子」的死角）。
- 不执法 pod 只读层与组员自维护层（前者走组长流程，后者归 delivery-gate）。

## 3. 巡检面：`pysci-dev doctor` 的预算三档审计

`src/pysci/skills/devops/tools/budget.py` 全量扫描三档（真本 + 根 rules + 各 pod
charter/AGENTS.md/自建 rules·skills），逐条 `[!]` 并计入退出码 2；另有
`test_repo_state_within_budget` 在 CI 上钉住「仓库现状全绿」。它是唯一的**观测**层：
硬闸只在交付/提交那一刻起作用，中途死掉的会话留下的自维护层膨胀要靠 doctor 看见。

## 4. 真本 references 超 8KB 的分册处置

按 `##` 主题拆成 `<stem>-0N-<slug>.md` 分册（`0N` 零填充防重名，slug 只留 ASCII，
前言类分册名用 `overview`），**逐字搬运**：机械做法是围栏感知切段 → >上限者再按
`### ` 切 → 贪心装箱（正文预算留头块余量）→ 每册加 5 行头块（册号、含哪些小节、
回指索引）；原文件名留作**分册索引**（列出各册所含小节与实测尺寸，标明「只读你需要
的那一册」）。所属 SKILL.md 的索引行标 ``*`` 并在节首加一行
「（`*`=分册索引，按需读单册）」——SKILL.md 自身也须 ≤8192B，超限就压缩措辞而不是
删信息。

验证两步：① 分册剥掉 5 行头块后按序拼接，与原文件比「全部内容行逐条有序相等」；
② `find orchestration/skills -name '*.md' -size +8192c` 输出为空。

**EOF 窗口的机械治理**（backlog 20261010-191922-devops）：pre-commit 的
`trailing-whitespace` 与 `end-of-file-fixer` 会在**首次提交**削掉文件末尾空白而改写
文件、中止提交——此时 ① 校验的是**生成前**字节，钩子改后即过期。做法：生成/改写
后立即跑 `pysci-dev mdgen-check <paths…>`，工具把这两支钩子提前做完（`pre-commit run
<hook_id> --files …`，只跑改写类、不越权触发 ruff/budget），逐文件打印 pre/post
sha256；① 的内容等价性断言以 **post_sha256**（钩子终态）为基准。首提即过、无验证
过期窗口；`--dry-run` 只报告不改文件。
