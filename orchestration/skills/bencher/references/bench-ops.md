# 副本机制与队列运行细节（按需读）

SKILL.md 放决策与命令面；本文放**为什么这样设计**与**坏了往哪查**。

## 1. 组长侧队列面（你不驱动它，但要知道它怎么调到你了）

`orch`（组长专属）的 bencher 面，实现在 `src/pysci/skills/orchestration/tools/benchd.py`：

| 命令 | 作用 |
|---|---|
| `orch bench-add <run\|capture\|ab\|check> [--cases …] [--start]` | 入队一个测量任务 |
| `orch bench-up [--dry]` | 拉起后台 worker（非空即消化，空即自毁） |
| `orch bench-queue [--all]` | 列队列（默认只看在途与待查看） |
| `orch bench-seen <id>` | 提请人确认已查看，从待查看面摘掉 |
| `orch bench-stop` | 协作停止（条目间隙干净退出，不 kill） |
| `orch status` | `== bencher worker ==` 段：锁态/计数/待查看结果 |

- 队列文件 `state/bencher/queue.json`，锁 `state/bencher.lock`，run 日志与 `.done`
  落 `state/bencher-runs/`，取消信号 `state/bencher.cancel`。
- **FIFO 单 worker**：互斥由锁保证，所以获锁一刻仍在 `in_progress` 的条目必是上一任被
  杀死时留下的孤儿 → `reset_orphans()` 就地复位为 pending。stale 判据与 devops drain
  共用 `WorkerPaths`（`drain.py`），不另写一套事故硬化的锁逻辑。
- 完成项**不即时唤醒组长**（REFACTOR §5.1-B 择机查看）：终态条目留在「待查看」面，
  watch 事件 `BENCH-DONE <run>` 属可选挂载，默认不挂。
- **系统性故障**（`quota_exhausted` 额度耗尽 / `channel_error` 渠道级故障）：全局停止、剩余
  条目保持 pending，`.done` 附 `stopped=<kind>` 与分列说明（两类处置相同、恢复动作不同）。

## 2. 副本重锚的因果链（改这里前先读）

`pysci.paths._find_project_root` 从 `pysci/paths.py` 的 `__file__` 就近上溯，找同时具备
`pyproject.toml` 与 `.python-version` 的目录。于是：

```
PYTHONPATH=<replica>/src  →  import pysci 命中副本源码  →  __file__ 在副本里
                          →  PROJECT_ROOT = 副本根  →  state/ pods/ guards/ 技能真本 全重锚
```

- **cwd 必须留在主树**：cd 进副本会让 uv 就地新建空 `.venv`，缺依赖即崩
  （devops AGENTS.md 已记录的坑）。
- 副本没有 `.venv` → 组员裸 `pysci-X` 失效。对策：`runner.build_env` 读
  `PYSCI_VENV_SCRIPTS` 环境变量决定 PATH 里的脚本目录；`_replica_call` 把它指向**主树**
  venv，PYTHONPATH 提供**副本**代码——「脚本来自主 venv、代码来自副本」正是 A/B 要的形状。
- `.env` 是 gitignored 且由 literature 配置 loader 从 `PROJECT_ROOT/.env` 读 → 副本必须
  复制一份（用后即弃）。
- pods 的 gitignored 运行目录（`bench/ outbox/ inbox/ .qoder/skills/`）在副本里不存在 →
  `_replica-prep` 重建并跑技能 sync（副本自己的真本 → 副本 pods）。

## 3. 归档还原为什么必须**先于** sync

`capture` 归档的是 gitignored 的组员面，其中含**部署技能副本**。若先 sync 后还原，还原会把
基线时的旧副本压回副本树，A/B 就测成了「旧真本 + 旧副本 vs 新真本 + 旧副本」——差异被归档
吃掉一半。正确顺序：还原归档 → 再跑副本自己的 sync，让新真本覆盖部署副本。

## 4. 读数怎么从副本回到主树台账

`_hop` 在副本语义下跑一批案，把行 JSON 夹在 `ROWS_BEGIN`/`ROWS_END` 标记带里吐到 stdout，
父进程 `rows_from_stdout` 取出后追加进主树 `state/ledger/evals.jsonl`。

- 为何用标记带而不是「取最后一行」：组员正文里本来就可能有多行 JSON，裸取末行会把交付
  正文当读数。
- 为何父进程代写台账而不是副本自写：`PROJECT_ROOT` 在副本里，副本自写会把读数写进**副本**
  的 gitignored 面，随 worktree 一起被销毁。
- 单案崩溃不外抛：`_hop` 逐案 try，失败案吐 `verdict=fail` + 原因行，标记带必须吐得出来。

## 5. 污染闸的判据边界

`HARNESS_FACE`（`replica.py`）= orchestration/devops/bencher 工具源码、guards、技能真本、
`state/evals`、`state/failure-tags.json`、pods、`pyproject.toml`。

- **只算 `ref_a..ref_b` 之间、且排除两端本身**的提交：两端是「被测改动」，中间才是混杂因子。
- 不在名单里的（论文、scripts、data）不影响测量口径，算进污染就是把自己的注意力浪费在噪声上。
- 反过来，**新增会改变测量的面时必须同步扩充这个元组**，否则污染闸会漏报假阴性——
  静默给出错误因果比停手更糟。

## 6. 与 devops 的职责分界

- devops：`tests/` 与 src 正确性（代码跑不跑得对）。
- bencher：harness 案卷与基线（成员在真实任务上表现可不可测、比上轮好没好）。
- `orch eval run` 保留为**组长抽检面**（`runner=leader`）；套件所有权在 E4 之后属 bencher：
  案卷增删、基线重取、判决口径都由你负责，组长只读与抽检。

## 7. 常见故障

| 症状 | 先查 |
|---|---|
| 副本跳「未返回读数」 | `PYSCI_VENV_SCRIPTS` 是否指向存在的目录；副本 `orchestration/state/evals/harness/` 里有没有案内案（旧 ref 可能还没这个案） |
| A/B 永远 inconclusive | 两 ref 之间确实有别的 harness 提交——换相邻提交对，或把混杂提交并入被测面 |
| 台账出现 `runner` 缺失的行 | 那是移交前（E2）的老行，读法按 `leader` 回退，不是缺陷 |
| 主树多出脏面 | 归档还原/`.env` 复制是否误写主树；副本机制只许写副本与 `state/evals/**` |
