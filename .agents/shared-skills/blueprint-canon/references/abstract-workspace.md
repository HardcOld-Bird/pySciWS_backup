# 工作区布局与边界

> **所属**：PySci 蓝图 · 设计摘要
> **上位**：[设计摘要（总纲）](abstract-core.md)
> **效力**：与实现、旧计划冲突时，以蓝图为准；路径细节以 `src/pysci/paths.py` 为准。

## 顶层结构

```text
pySciWS2/
├── AGENTS.md / AGENTS.leader.md / AGENTS.meta.md
├── .agents/
│   ├── shared-skills/blueprint-canon/   # worker/leader 共享的只读蓝图
│   ├── skills/                          # worker：pysci-infra + 待重建的科研技能
│   ├── leader-skills/                   # leader：只规划/派发/验收，无生产技能
│   └── meta-skills/                     # meta：蓝图导航、harness-dev、eval-runner、repo-recon
├── src/pysci/
│   ├── paths.py                         # 资产与支撑目录的权威索引
│   ├── toolchain.py                     # 项目内工具链绝对路径
│   ├── skills/                          # 技能后端（随技能重建增加）
│   ├── support/                         # eval runner、通用薄工具
│   └── research/<line>/                 # 研究线（本地保留，不入库）
├── scripts/
│   ├── research/<line>/                 # 研究脚本（本地）
│   └── ...                              # 技能/MCP 安装脚本（随技能重建增加）
├── data/
│   ├── skills/<name>/                   # 技能级模板/配方/知识（按需重建）
│   ├── research/<n>_<line>/             # 研究资产（本地）
│   └── 用户的inbox/                      # 用户专属区
├── sub_root/
│   ├── deps_management/                 # toolchain.json + deps_manager + deps + cache
│   ├── tests/                           # 第 1 层：可机械断言
│   ├── evals/                           # 第 2 层：可基准测试
│   └── others/plans/                    # 计划与过程产物
├── pyproject.toml / .python-version
├── .gitignore / .gitattributes
└── .venv/                               # uv 项目环境（gitignore）
```

## 各层职责

### `src/pysci/`

**只定义**，不直接对外提供一次性脚本。`paths.py` 是资产与支撑目录的权威索引：每个 `*_ROOT` 常量的注释写明用途与规范落盘位置；与 README、计划冲突时以 `paths.py` 为准。

- `paths.py`：项目根、资产根、技能资产根、研究线目录、`assert_within_data()`。
- `toolchain.py`：项目内 git/node/uv/Python/uv tool 的绝对路径解析；程序化 subprocess 不依赖 PATH 上的 `.cmd`。
- `skills/<name>/`：技能后端；每个技能一个子包，代码在其 `tools/` 下。
- `support/`：跨技能薄工具（eval runner、通用进程/预算工具等）。
- `research/<line>/`：研究线代码；本地保留，不入库。

### `scripts/`

**只使用**，不定义复杂类或长函数。研究线脚本进 `scripts/research/<line>/`；技能/MCP 安装脚本按技能重建需要添加。所有产物写入 `data/` 前必须经 `pysci.paths.assert_within_data(path)` 断言。

### `data/`

与 `src/pysci/` 平行的资产树：

- `data/skills/<name>/` ⟷ `src/pysci/skills/<name>/`：技能级模板、配方、知识、缓存。
- `data/research/<n>_<line>/` ⟷ `src/pysci/research/<line>/`：研究资产与产物；研究线目录名保留数字序号（Python 包路径不能以数字开头）。
- `data/用户的inbox/`：用户专属区；Agent 可增删改，但删除须用户同意；通用基建不得依赖其中文件。

`.gitignore` 对 `data/` 采用白名单制：只跟踪 `data/skills` 的骨架面与 `data/用户的inbox/pysci_settings.toml`；其余本地保留。

### `sub_root/`

- `deps_management/`：依赖管理系统；`toolchain.json`、`deps_manager/` 入库；`deps/`、`cache/`、`dsh_home/` 不入库。
- `tests/`：第 1 层测试；`artifacts/` 不入库。
- `evals/`：第 2 层黄金任务与台账；`runs/` 不入库，`tasks/`、`reports/improvement-ledger.md` 入库。
- `others/plans/`：计划与过程产物；完成后归档到 `archived/`。

### `.agents/`

- `shared-skills/blueprint-canon/`：蓝图正典，worker 与 leader 共享，只读；人类维护。
- `skills/`：worker 技能；`pysci-infra` 在奠基阶段交付，7 个科研技能由项目 DSH 逐项重建。
- `leader-skills/`：leader 的组织、验收技能；不含生产技能。
- `meta-skills/`：meta 的蓝图导航、harness-dev、eval-runner、repo-recon。

## 代码树 / 资产树镜像

`src/pysci/skills/<name>/` ⟷ `data/skills/<name>/`；`src/pysci/research/<line>/` ⟷ `data/research/<n>_<line>/`。镜像关系让代码与资产各自演进，但语义一一对应；不要把一个技能的资产散落到另一个技能或仓库根。

## 入库边界

**入库**：框架代码、技能文档、支撑系统代码与清单、测试/评估定义、最小资产骨架、蓝图与计划。

**不入库**：`.venv/`、`sub_root/deps_management/{deps,cache}`、`dsh_home/`、`.env`、凭据、日志、测试/评估产物、研究线内容、图片等大文件、缓存目录。

## 命名约定

- 目录/文件：snake_case（`deps_manager`、`scientific_plotting`）。
- DSH 技能名与技能目录：kebab-case（`scientific-plotting`、`pysci-infra`）。
- Python 包：snake_case；研究线资产目录保留数字序号前缀。
- 仅当生态有硬性要求时才例外（npm 包名、工具约定）。
