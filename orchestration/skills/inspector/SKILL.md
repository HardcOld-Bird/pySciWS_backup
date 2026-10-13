---
name: inspector
description: inspector 组员的巡检规程：pysci-inspect 命令面、两类发现的登记与分流口径、巡检菜单项怎么做、闸门读数含义与收尾清单。inspector pod 会话中被派发巡检或核查任务时使用。
---

# inspector 巡检规程

角色边界与权限面见 charter（`.qoder/rules/charter.md`）；本文放**操作知识**。
专属 CLI：`pysci-inspect`。全项目只读——**你只登记发现，不修任何东西**。

## 命令面

```bash
pysci-inspect finding add --class mechanical|judgment --title '一句话现象' \
    [--symptom T] [--repro '命令'] [--evidence '实际输出'] [--expect '期望输出'] \
    [--kind defect|usability] [--criteria '可证伪判据'] [--tag 标签]
pysci-inspect finding list [--status pending|promoted|rejected|queued|escalated]
pysci-inspect promote <finding-id>            # 仅机械证据类
pysci-inspect submit <finding-id> --jury      # 判断类唯一出口
pysci-inspect gate                            # 读当前闸门（只读）
# 会话记录审计工具链（transcript-audit 用；返回值即阅读面，落盘只是缓存）
pysci-inspect transcript list [--all] [--limit N]        # 优先级头部（缺省 ≤3）
pysci-inspect transcript scan <sid> [--turns a-b] [--verbose] [--save]
pysci-inspect transcript fetch --anchor <sid>.jsonl:L<a>-L<b>   # 或 --sid X --grep 模式
```

`transcript` 三件套把「随机盲读」换成**摘要先行、锚点深挖**：`list` 按优先级选靶
（leader 会话 > 组员；新 > 旧），`scan` 出每回合一行索引（含**行锚**），`fetch` 按锚取
原文。findings 的 `--repro` **直接引行锚**——比转述可复现。`scan` 会把该会话记成已检查
（三态游标 `state/inspector/transcript-cursor.json`），可检判据见 references/patrol-craft.md。

`pysci-inspect` **没有 `--override`**，判断类也没有免判决通道：这不是纪律条款，是参数表
里根本没有这一项。别去试、也别指望组长替你自裁——`orch reject --override` 是组长的
通道，账记在组长头上。

## 两类发现的分流口径（先分类，再写）

- **mechanical**：能给出一条别人重跑就能得到同样输出的命令。三件套 `--repro` /
  `--evidence` / `--expect` 缺任一项就不成立（basic.md §2：没证据的发现等于印象）。
  这类**免判决**直接晋升 backlog，所以标准要比印象更硬。
- **judgment**：需要判断的那类（设计是否冗余、命名是否反直觉、文档承诺与实现是否一致）。
  必经 `submit --jury` 仲裁。案卷由 `verdict.assemble_case` 自动组，你只需把
  `--title`/`--symptom`/`--expect` 写清楚到「两选项的差异能被读出来」的程度。

两类都必须 `--criteria`：一句可证伪的成功判据（「X 命令输出含 Y」「文件 ≤8192B」），
不是「更好」「更稳」。缺判据按 §6 总闸被判 `rejected`——补上判据可复活，重跑没有用。

## 巡检菜单项怎么做

菜单在引擎侧，任务书里告诉你类别 id。八项的通用动作形状：**取证 → 判类 → 登记 → 分流**。

- `doctor-sweep`：`pysci-dev doctor` 全域，逐条核对 [!] 与 [✗] 是否能复现；
- `transcript-audit`：走**工具链**——`pysci-inspect transcript list` 取优先级头部（leader
  会话先、新先，每轮 ≤3）→ `scan <sid> --verbose` 拿简版索引定位疑点 → 疑点按锚
  `fetch --anchor <sid>.jsonl:L<a>-L<b>` 取原文（旧形态是随机 grep 盲读）；看 pod-guard
  拦截记录与其后的绕行迹象；
- `self-maint-audit`：随机 pod 的 `AGENTS.md`/自建 rules，对照代码看它还在讲什么已死的事；
- `code-walk`：`src/pysci/` 挑一模块，找静默 except、兼容垫片、fail-open 吞掉自身 bug；
- `pipeline-self`（§0.2 自覆盖）：查「文档承诺 vs 代码实现」是否一致；
  **豁口**：`pysci-inspect` 自身不在范围内；
- `intent-drift`：挑 README/charter 一段宣称，用可执行命令核验；
- `design-rubric`：照 `rubrics/design-review.md` 三维审一个模块（判断类）；
- `bench-probe`：读 bench 队列与台账找测量缺口，**提案**而非自己跑压测。

细节与常见反例见 `references/patrol-craft.md`。

## 闸门读数（`pysci-inspect gate`）

- `闸门=可派跳`：正常做巡检、登记与分流；
- `闸门=只读`（让路）：生产侧正忙，**只做零成本取证**，不新增晋升；把要说的写进交付正文，
  下一轮再做；
- `产出=paused`（迟滞）：backlog/bench 队列积压超阈，**停止产出**，等回落。

闸门的放行与否由引擎在一处决定，你改不了也不该改——它存在的意义就是在你不想停的时候停。

## 收尾清单

1. 每条发现都已 `finding add` 落台账（口头汇报不算）；
2. 机械类已 `promote`（或等引擎自动晋升），判断类已 `submit --jury`；
3. `<result>` 报告：类别 id + 登记了哪些 finding + 各自分流结果 + 无发现时明写「本轮无发现」；
4. 不改任何文件（包括你自己的 `AGENTS.md` 之外的东西）；想记经验只写 `AGENTS.md`。
