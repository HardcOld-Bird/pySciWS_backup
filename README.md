# pySciWS · pysci

个人科研工作区，面向**物理声学 / 声学超构表面 / 非厄密物理**（例外点 EP、BIC、CPA、拓扑）。
目标是把科研全流程中可自动化的环节（文献调研、理论计算、有限元仿真、科研绘图、论文写作）
交给 LLM Agent 驱动，人只保留真实物理世界的实验部分。

代码组织为单一 **src-layout 安装包 `pysci`**，通过 [uv](https://docs.astral.sh/uv/) 以 editable
方式装入唯一的 `.venv`——**无需任何 `sys.path` hack**，直接 `import pysci...`、
`uv run python scripts/...` 即可运行。非代码资产放在与代码树镜像的平行资产树 `data/`。

> **本文件是项目结构的事实描述，可能随重构过时。**
> 资产目录布局的**权威索引是 [`src/pysci/paths.py`](src/pysci/paths.py)**——每个 `*_ROOT` 常量的
> 注释逐条写明了该目录的用途与规范落盘位置，且它与代码同步演进。二者冲突时以 `paths.py` 为准。
>
> **开发规范见 [`.qoder/rules/basic.md`](.qoder/rules/basic.md)**（项目唯一的常驻规则文件，
> 含 PowerShell 编码陷阱、已裁决的架构选择、代码与产物落盘约定）。

---

## 环境准备

要求 Python **3.13.15**（由 `.python-version` 与 `pyproject.toml` 的 `requires-python = "==3.13.15"` 锁定）。

```bash
uv sync                          # 创建 .venv、安装依赖，并以 editable 方式装入 pysci
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
│   ├── research/gain_ep/               # 研究线：增益管槽例外点
│   │   ├── theory/                     #   理论 / 仿真：sim、theory、cmt_reflection_s_matrix、
│   │   │                               #     transfer_function_analysis
│   │   ├── experiment/                 #   实验平台（原 sweeper400）：analyze / calib / config /
│   │   │                               #     gui / measure / move / sim / use + logger
│   │   └── article/figures/            #   论文插图的生产模块（ep_eigval_3d / ep_phase_wind /
│   │                                   #     ep_riemann_3d / fig0_demo_smoke）
│   └── skills/                         # LLM 技能后端（每个技能一个子包，代码在其 tools/ 下）
│       ├── literature_research/        #   文献检索 / 抽取 / 笔记 / Zotero / RAG
│       ├── comsol_simulation/          #   COMSOL 建模、求解、导出、渲染、MCP 服务
│       ├── document_writing/           #   LaTeX / PPTX / DOCX / PDF 渲染与转换
│       ├── scientific_plotting/        #   出版级图件管线、期刊风格、合规审计
│       ├── theoretical_computation/    #   符号计算、本征值、参数空间、拓扑特征
│       └── ai_drawing/                 #   火山方舟（即梦 Seedream）图像生成与位图处理
├── scripts/                            # 可执行脚本（只使用 pysci，不定义复杂类/长函数）
│   ├── research/gain_ep/               #   experiment/ · simulation/ · theory/
│   └── {comsol,paper_search,zotero}_mcp/  # 社区 MCP 的安装、注册与连通性自检套件
├── tests/                              # pytest（testpaths = tests）
│   ├── test_paths.py
│   ├── research/gain_ep/{experiment,theory}/
│   └── skills/<6 个技能各一目录>/
├── data/                               # 平行资产树（非代码资产，与 src/pysci/ 镜像，不入包）
│   ├── research/1_gain_ep/             #   ⟷ pysci.research.gain_ep
│   │   ├── article/                    #     exp_report / figures / old_ppt
│   │   ├── theory/                     #     理论笔记与推导（按 slug 分子目录）
│   │   ├── simulation/                 #     .mph 模型（old_mphs）、refs、validation
│   │   ├── experiment/                 #     实验数据与参考资料
│   │   └── storage/                    #     可再生产物（calib / sim / …）
│   └── skills/<6 个技能各一目录>/       #   技能级资产：templates / recipes / cache / runs …
└── .qoder/
    ├── rules/basic.md                  # 唯一常驻规则（每会话全文注入）
    └── skills/<6 个技能>/SKILL.md       # 技能说明书（Agent 的使用入口）
```

> **代码树 / 数据树镜像**：`src/pysci/research/<name>/` ⟷ `data/research/<n>_<name>/`、
> `src/pysci/skills/<name>/` ⟷ `data/skills/<name>/`。Python 包路径不能以数字开头，
> 故资产目录保留数字序号：`data/research/1_gain_ep` 对应包 `pysci.research.gain_ep`。

### 路径解析与产物落盘

统一通过 `pysci.paths` 定位，**禁止** `sys.path.insert(...)` 与脆弱的 `Path(__file__).parents[N]` 硬编码：

```python
from pysci.paths import assert_within_data, research_asset_dir, research_fig_dir

asset_dir = research_asset_dir("gain_ep")  # data/research/1_gain_ep
fig_dir = research_fig_dir("gain_ep", slug="fig1")  # …/article/figures/fig1
assert_within_data(fig_dir / "out.pdf")  # 写入前断言落在 data/ 内（防散落护栏）
```

技能级资产（模板、缓存、配方）落在各 `*_ROOT`（`LITERATURE_ROOT` / `COMSOL_ROOT` / `PLOTTING_ROOT` /
`THEORY_ROOT` / `AI_DRAWING_ROOT` / `DOCWRITING_ROOT`）；**具体研究线的产物一律落在研究资产目录内**。
完整清单与规范见 `paths.py` 的常量注释。

---

## 运行方式

```bash
# 研究脚本（已 editable 安装，直接 import pysci）
uv run python scripts/research/gain_ep/theory/相位绕数参数空间对比.py
uv run python scripts/research/gain_ep/simulation/runner_beam_scatter.py

# 技能 CLI：6 个 console script（uv sync 后生成于 .venv/Scripts）
uv run pysci-research    --help      # 文献调研        ⟷ literature_research
uv run pysci-simulation  --help      # COMSOL 仿真     ⟷ comsol_simulation
uv run pysci-compose     --help      # 文档写作        ⟷ document_writing
uv run pysci-figures     --help      # 科研绘图        ⟷ scientific_plotting
uv run pysci-theory      --help      # 理论计算        ⟷ theoretical_computation
uv run pysci-imagine     --help      # AI 绘图         ⟷ ai_drawing

# 等价兜底调用（入口映射见 pyproject.toml 的 [project.scripts]）
uv run python -m pysci.skills.literature_research.tools.research doctor
```

> **Windows PowerShell 注意**：调用任何输出中文的命令前，须先在同一 shell 执行
> `[Console]::OutputEncoding = [System.Text.Encoding]::UTF8`，否则 stdout 一旦被管道就会乱码，
> 拿不到任何状态信息。原因与其余 PowerShell 陷阱见 [`.qoder/rules/basic.md`](.qoder/rules/basic.md) §3。

---

## 测试

```bash
uv run pytest                     # 实测：1450 passed, 2 skipped, 6 deselected（约 22 s）
uv run pytest -m "not hardware"   # 显式排除硬件用例（addopts 已默认如此）
uv run pytest -m hardware         # 只跑硬件用例（需 NI 设备 / COMSOL Server）
uv run pytest --cov               # 附带覆盖率（source = src/pysci）
```

- `testpaths = ["tests"]`，`--strict-markers`；markers：`hardware`（需外部设备）、`slow`。
- `tests/research/gain_ep/theory/` 下的 `test_*.py` 是**手动 COMSOL 脚本**（模块级直接连接
  COMSOL Server），已由该目录 `conftest.py` 的 `collect_ignore` 精确排除，不被 pytest 收集；
  需要时先启动 COMSOL Server 再直接 `python tests/research/gain_ep/theory/<script>.py`。
- 无 `nidaqmx` 时，`tests/research/gain_ep/experiment/conftest.py` 会以 `collect_ignore_glob`
  跳过整个目录的收集，不会导致 `uv run pytest` 失败。

---

## 代码质量与提交

- **ruff**：`line-length = 88`，`target-version = py313`，isort `known-first-party = ["pysci"]`；
  `data/`、`.venv/` 等已在 `[tool.ruff].exclude` 中排除。迁移进来的既有代码按子树豁免**纯风格类**
  规则（见 `[tool.ruff.lint.per-file-ignores]`），正确性规则（F401 / F811 / E9）与 `paths.py` 仍全树严格。
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

`.gitignore` 排除以下**可再生或超大**内容（其余研究资产按需跟踪）：

- `data/research/**/simulation/`、`data/research/**/experiment/`、`data/research/**/storage/`
  ——含 `.mph` 模型与实验/仿真输出；
- 图管线中间产物与超大矢量交付物：`figures/*/raw/`、`figures/*/panels/`、`figures/*/out/*.eps`、
  `figures/*/out/*.svg`、`figures/*/cache_*.npz`（`pdf` / `png` 仍跟踪）；
- `.env`（凭据）、`__pycache__/`、`.idea/`、`.venv/`、`*.egg-info/`。

文献数据区的缓存治理由 `data/skills/literature_research/.gitignore` 自管
（`cache/*` 全忽略，唯独 `cache/extracted/` 纳入跟踪——那是用 MinerU 配额换来的转换全文）。
