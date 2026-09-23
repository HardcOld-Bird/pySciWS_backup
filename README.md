# pySciWS · pysci

个人科研计算工作区。以单一 **src-layout 包 `pysci`** 组织多条研究线的理论 / 仿真 / 实验代码，
并配套 LLM 文献检索 skill 后端。项目通过 [uv](https://docs.astral.sh/uv/) 以 editable 方式安装进
唯一的 `.venv`，**无需任何 `sys.path` hack**——直接 `import pysci...`、`python scripts/...`
或 `python -m pysci...` 即可运行。

## 环境准备

要求 Python **3.13.15**（由 `.python-version` 与 `pyproject.toml` 的 `requires-python` 锁定）。

```bash
uv sync                      # 创建 .venv、安装依赖，并以 editable 方式装入 pysci
uv sync --extra experiment   # 额外安装 gain_ep 实验平台依赖（nidaqmx；仅 Windows + NI-DAQmx 驱动可用）
```

## 目录结构

三层语义 + 平行资产树：

```
pySciWS/
├── src/pysci/                      # 唯一安装包（src-layout）
│   ├── paths.py                    # 路径锚点：PROJECT_ROOT / ASSET_ROOT / LITERATURE_ROOT / research_asset_dir()
│   ├── common/                     # 共享工具层（可复用的理论储备与基础工具）
│   │   ├── matrix.py               #   2×2 复矩阵本征系统可视化
│   │   ├── numerical.py            #   sympy 表达式 → numpy 数值函数
│   │   ├── space_curve.py          #   空间曲线 / 复方程可视化
│   │   ├── dtypes.py               #   参数空间等数据类型
│   │   └── plotting.py             #   matplotlib 中文字体 / 绘图配置
│   ├── research/gain_ep/           # 研究线：增益管槽例外点（gain_ep）
│   │   ├── theory/                 #   理论 / 仿真：sim.py（COMSOL 接口）、theory.py（扫参与绘图）
│   │   └── experiment/             #   实验平台（原 sweeper400）：
│   │                               #     analyze / calib / config / gui / measure / move / sim / use
│   └── skills/literature_research/ # LLM 文献检索 skill 后端（代码；数据在 data/skills/literature_research/）
├── scripts/                        # 可执行脚本（非包代码）
│   ├── foundations/{cpa,ep_bic,topo_bic,matrix}/   # 共享工具层的演示 / 验证脚本
│   └── gain_ep/{theory,experiment}/                # gain_ep 研究线脚本
├── tests/                          # pytest 测试（testpaths = tests）
│   └── gain_ep/{theory,experiment}/
├── data/                           # 平行资产树——数据树（与 src/pysci/ 代码树镜像；不入包）
│   ├── research/1_gain_ep/         #   ⟷ src/pysci/research/gain_ep/（.mph 与 storage 不入版本控制）
│   │   ├── theory/                 #     理论笔记 / PDF
│   │   ├── experiment/mphs/        #     COMSOL .mph 模型
│   │   ├── storage/                #     仿真 / 校准输出（可再生）
│   │   └── 参考资料/               #     gain_ep 相关参考资料（如 TEP）
│   └── skills/literature_research/ #   ⟷ src/pysci/skills/literature_research/（文献数据区）
│       └── papers / shortlists / reviews / templates / cache / INDEX.md
└── 参考资料（手动）/               # foundations 各主题参考资料（CPA / EP & BIC / Topo & BIC）
```

> **代码树 / 数据树镜像**：`src/pysci/`（代码）与 `data/`（非代码资产）一一镜像——
> `src/pysci/research/<name>/` ⟷ `data/research/<n>_<name>/`、`src/pysci/skills/literature_research/`
> ⟷ `data/skills/literature_research/`。Python 包路径不能以数字开头，故资产目录保留数字序号：
> `data/research/1_gain_ep` 对应包 `pysci.research.gain_ep`。代码统一通过
> `pysci.paths.research_asset_dir("gain_ep")` 与 `LITERATURE_ROOT` 定位资产，避免脆弱的 `parents[N]` 硬编码。

## 运行方式

```bash
# 运行脚本（已 editable 安装，直接 import pysci，无需 path hack）
uv run python scripts/foundations/ep_bic/1-EP-BIC-Hamiltonian.py
uv run python scripts/gain_ep/theory/传递函数分析.py

# 以模块方式运行文献检索 skill（数据区在 data/skills/literature_research/）
uv run python -m pysci.skills.literature_research.tools.research doctor

# 测试
uv run pytest                     # 默认：无 nidaqmx / 无 COMSOL Server 时，硬件与仿真用例由 conftest 守卫自动跳过收集
uv run pytest -m "not hardware"   # 显式排除硬件相关用例
uv run pytest --cov               # 附带覆盖率统计（source = src/pysci）
```

> `tests/gain_ep/theory/` 下的 `test_*.py` 是**手动 COMSOL 脚本**（模块级直接连接 COMSOL Server），
> 已由该目录 `conftest.py` 的 `collect_ignore` 排除，不被 pytest 收集；需要时先启动 COMSOL Server
> 再 `python tests/gain_ep/theory/<script>.py` 直接运行。

## 代码质量与提交

- **ruff**：`line-length = 88`，`target-version = py313`，isort `known-first-party = ["pysci"]`；
  资产树 / 缓存目录已在 `[tool.ruff].exclude` 中排除。
  ```bash
  uv run ruff check .
  uv run ruff format .
  ```
- **pre-commit**：提交时自动运行（trailing-whitespace / end-of-file-fixer / check-toml /
  check-added-large-files / commitizen）。首次克隆后执行 `uv run pre-commit install`。
- **提交信息**：遵循 Conventional Commits，可用 `cz commit` 或直接
  `git commit -m "type(scope): 描述"`。
