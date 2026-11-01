# pySciWS · pysci

个人科研工作区：把科研全流程中可自动化的环节（文献调研、理论计算、有限元仿真、科研绘图、论文写作）
交给 LLM Agent 驱动，人只保留真实物理世界的实验部分。仓库只含**核心骨架**，
可直接开展任意新研究线；具体研究内容与使用痕迹留在本地、不入库（见文末「版本控制范围」）。

代码组织为单一 **src-layout 安装包 `pysci`**，通过 [uv](https://docs.astral.sh/uv/) 以 editable
方式装入唯一的 `.venv`——**无需任何 `sys.path` hack**，直接 `import pysci...`、
`uv run --no-sync python scripts/...` 即可运行。非代码资产放在与代码树镜像的平行资产树 `data/`。

> **本文件是项目结构的事实描述，可能随重构过时。**
> 资产目录布局的**权威索引是 [`src/pysci/paths.py`](src/pysci/paths.py)**——每个 `*_ROOT` 常量的
> 注释逐条写明了该目录的用途与规范落盘位置，且它与代码同步演进。二者冲突时以 `paths.py` 为准。
>
> **开发规范见 [`.qoder/rules/basic.md`](.qoder/rules/basic.md)**（项目唯一的常驻规则文件，
> 含 shell/编码陷阱、已裁决的架构选择、代码与产物落盘约定）。

---

## 环境准备

要求 Python **3.13.15**（由 `.python-version` 与 `pyproject.toml` 的 `requires-python = "==3.13.15"` 锁定）。

```powershell
sub_root\deps_management\deps_manager\init-deps.cmd   # 工具链自包含引导（git/node/uv/Python + .venv）
```

工具链（git/node/uv/Python）vendored 于 `sub_root/deps_management/`（`toolchain.json` 版本真源，
含 url/sha256 可复现下载），**不依赖任何系统级安装**；`init-deps` 会一并创建 `.venv`、
安装依赖并以 editable 方式装入 pysci。此后：

```bash
uv sync                          # 增量同步依赖（uv 经 vendored/共享 shim 解析）
uv pip install -e ".[imaging]"   # 可选 extra：ai_drawing 的进阶位图算子（scikit-image + opencv-headless，纯 CPU）
```

> `nidaqmx`（NI 数据采集）是**主依赖**而非 extra，`uv sync` 即安装；包本身跨平台可装，
> 但只有 Windows + NI-DAQmx 驱动下才能真正采集。无驱动时相关用例由 `conftest.py` 守卫自动跳过收集。

### 非 pip 的外部依赖

| 用途 | 依赖 | 说明 |
|---|---|---|
| LaTeX 编译 | TeX Live | 须在系统 PATH 中（`latexmk` / `chktex`） |
| Office ↔ PDF 转换 | LibreOffice | 须在系统 PATH 中 |
| 有限元仿真 | COMSOL Multiphysics + COMSOL Server | 经 `mph` 库交互；单一 license，注意并发占用 |
| PDF → Markdown | MinerU 云端 | `.env` 配 `MINERU_TOKEN`；本地兜底 `pymupdf4llm`（快但丢公式） |
| 文献库 / 检索 MCP | zotero-mcp、paper-search-mcp、comsol MCP | 安装脚本见 `scripts/*_mcp/setup_*.ps1` |

所有 API 密钥与本地路径集中在根目录 `.env`（**不入版本控制**）。

---

## 目录结构

```
pySciWS/
├── src/pysci/                          # 唯一安装包（src-layout）
│   ├── paths.py                        # 路径锚点与产物落盘规范（本结构的权威索引）
│   ├── research/<line>/                # 研究线（自建）：theory/ · experiment/ · article/figures/ …
│   └── skills/                         # LLM 技能后端（每个技能一个子包，代码在其 tools/ 下）
│       ├── literature_research/        #   文献检索 / 抽取 / 笔记 / Zotero / RAG
│       ├── comsol_simulation/          #   COMSOL 建模、求解、导出、渲染、MCP 服务
│       ├── document_writing/           #   LaTeX / PPTX / DOCX / PDF 渲染与转换
│       ├── scientific_plotting/        #   出版级图件管线、期刊风格、合规审计
│       ├── theoretical_computation/    #   符号计算、本征值、参数空间、拓扑特征
│       └── ai_drawing/                 #   火山方舟（即梦 Seedream）图像生成与位图处理
├── scripts/                            # 可执行脚本（只使用 pysci，不定义复杂类/长函数）
│   ├── research/<line>/                #   各研究线的驱动脚本（experiment/ · simulation/ · theory/）
│   └── {comsol,paper_search,zotero}_mcp/  # 社区 MCP 的安装、注册与连通性自检套件
├── tests/                              # pytest（testpaths = tests）
│   ├── test_paths.py / test_toolchain.py
│   ├── research/<line>/                #   各研究线测试
│   └── skills/<6 个技能各一目录>/
├── data/                               # 平行资产树（非代码资产，与 src/pysci/ 镜像，不入包）
│   ├── research/<n>_<line>/            #   ⟷ pysci.research.<line>（article/theory/simulation/…）
│   ├── skills/<6 个技能各一目录>/       #   技能级资产：templates / recipes / knowledge …
│   └── 用户的inbox/                     #   用户专属区（pysci_settings.toml 为用户自管配置真源）
├── orchestration/                      # 多 Agent 编排体系（README.md 为唯一权威）
└── .qoder/
    ├── rules/basic.md                  # 常驻规则（每会话全文注入）
    └── skills/<6 个技能>/SKILL.md       # 技能说明书（Agent 的使用入口）
```

> **代码树 / 数据树镜像**：`src/pysci/research/<name>/` ⟷ `data/research/<n>_<name>/`、
> `src/pysci/skills/<name>/` ⟷ `data/skills/<name>/`。Python 包路径不能以数字开头，
> 故资产目录保留数字序号：如 `data/research/1_example` 对应包 `pysci.research.example`。

### 开展新研究线

1. 建包 `src/pysci/research/<name>/`（theory/experiment/article 按需）；
2. 建资产目录 `data/research/<n>_<name>/`（`research_asset_dir("<name>")` 按 `*_<name>`
   通配自动定位）；
3. 脚本进 `scripts/research/<name>/`、测试进 `tests/research/<name>/`——三者均在
   `.gitignore` 的研究树规则内，**本地保留、不上传**。

### 路径解析与产物落盘

统一通过 `pysci.paths` 定位，**禁止** `sys.path.insert(...)` 与脆弱的 `Path(__file__).parents[N]` 硬编码：

```python
from pysci.paths import assert_within_data, research_asset_dir, research_fig_dir

asset_dir = research_asset_dir("example")  # data/research/<n>_example
fig_dir = research_fig_dir("example", slug="fig1")  # …/article/figures/fig1
assert_within_data(fig_dir / "out.pdf")  # 写入前断言落在 data/ 内（防散落护栏）
```

技能级资产（模板、缓存、配方）落在各 `*_ROOT`（`LITERATURE_ROOT` / `COMSOL_ROOT` / `PLOTTING_ROOT` /
`THEORY_ROOT` / `AI_DRAWING_ROOT` / `DOCWRITING_ROOT`）；**具体研究线的产物一律落在研究资产目录内**。
完整清单与规范见 `paths.py` 的常量注释。

---

## 运行方式

```bash
# 研究脚本（已 editable 安装，直接 import pysci）
uv run --no-sync python scripts/research/<line>/<...>.py

# 技能 CLI：6 个 console script（uv sync 后生成于 .venv/Scripts）
uv run --no-sync pysci-research   --help      # 文献调研        ⟷ literature_research
uv run --no-sync pysci-simulation --help      # COMSOL 仿真     ⟷ comsol_simulation
uv run --no-sync pysci-compose    --help      # 文档写作        ⟷ document_writing
uv run --no-sync pysci-figures    --help      # 科研绘图        ⟷ scientific_plotting
uv run --no-sync pysci-theory     --help      # 理论计算        ⟷ theoretical_computation
uv run --no-sync pysci-imagine    --help      # AI 绘图         ⟷ ai_drawing

# 等价兜底调用（入口映射见 pyproject.toml 的 [project.scripts]）
uv run --no-sync python -m pysci.skills.literature_research.tools.research doctor
```

> **Windows PowerShell 注意**：调用任何输出中文的命令前，须先在同一 shell 执行
> `[Console]::OutputEncoding = [System.Text.Encoding]::UTF8`，否则 stdout 一旦被管道就会乱码，
> 拿不到任何状态信息。原因与其余 PowerShell 陷阱见 [`.qoder/rules/basic.md`](.qoder/rules/basic.md) §3。

---

## 测试

```bash
uv run --no-sync pytest                     # 全量（硬件用例默认排除）
uv run --no-sync pytest -m "not hardware"   # 显式排除硬件用例（addopts 已默认如此）
uv run --no-sync pytest -m hardware         # 只跑硬件用例（需 NI 设备 / COMSOL Server）
uv run --no-sync pytest --cov               # 附带覆盖率（source = src/pysci）
```

- `testpaths = ["tests"]`，`--strict-markers`；markers：`hardware`（需外部设备）、`slow`。
- 各研究线若有**手动 COMSOL 脚本**（模块级直接连接 COMSOL Server），在该目录 `conftest.py`
  用 `collect_ignore` 精确排除，避免被 pytest 收集；需要时先启动 COMSOL Server 再直接运行。
- 无 `nidaqmx` 时，实验类测试目录的 `conftest.py` 以 `collect_ignore_glob` 跳过收集，
  不会导致 `uv run --no-sync pytest` 失败。

---

## 代码质量与提交

- **ruff**：`line-length = 88`，`target-version = py313`，isort `known-first-party = ["pysci"]`；
  `data/`、`.venv/` 等已在 `[tool.ruff].exclude` 中排除。研究树按子树豁免**纯风格类**
  规则（见 `[tool.ruff.lint.per-file-ignores]`，物理符号命名等），正确性规则（F401 / F811 / E9）
  与 `paths.py` 仍全树严格。
  ```bash
  uv run ruff check .
  uv run ruff format .
  ```
- **pre-commit**：首次克隆后执行 `uv run pre-commit install`。钩子为 trailing-whitespace /
  end-of-file-fixer / check-yaml / check-toml / check-added-large-files（`--maxkb=2000`）/
  ruff-check `--fix` / ruff-format / commitizen（commit-msg 阶段）。
- **提交信息**：Conventional Commits，可用 `cz commit` 或 `git commit -m "type(scope): 描述"`。
  跳过钩子：`SKIP=commitizen git commit -m "..."`。

---

## 版本控制范围

骨架修剪原则（2026-10-30 裁决）：**研究内容（研究线代码/数据/测试）与一切使用痕迹
只在本地保留、永不上传**；仓库只含可开展任意新研究的核心骨架。`.gitignore` 的
「骨架修剪」一节是唯一判据，要点：

- 研究树 `src/pysci/research/`、`scripts/research/`、`tests/research/`、`data/research/`
  整体不入库；
- `data/` 白名单制：只上传 `data/skills/` 骨架面（templates/recipes/knowledge/docs）
  与 `data/用户的inbox/pysci_settings.toml`（用户自管配置真源）；
- `orchestration/state/` 只保留最小配置面（registry、失败词表、lint 规则、部署台账、
  backlog 空种子），任务史与台账流水不入库；
- 组员 pod 只保留初始 harnessing（AGENTS.md/charter、settings/mcp 配置面），工作文件
  与自建 rules/rubrics 不入库；
- `.env`（凭据）、`sub_root/`（本机工具链）、`*.log`、图像文件、缓存目录一律不入库。
