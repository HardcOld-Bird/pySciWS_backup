---
trigger: manual
alwaysApply: false
---

# pySci开发规范（COMSOL仿真）

## 项目概述

- 本项目是一个**物理学理论探索项目**，专注于波动声学领域的符号计算和数值模拟。项目采用Python生态系统，以SymPy为核心进行符号推导，结合NumPy/SciPy进行数值计算，使用Matplotlib/PyVista进行可视化。
- 对于有限元仿真，使用mph库通过 COMSOL Multiphysics 软件进行仿真研究和理论计算验证。
- 代码组织为单一 **src-layout 安装包 `pysci`**（通过 uv 以 editable 方式装入唯一 `.venv`），按“共享工具层 / 研究层 / skill 后端”分层；非代码资产（.mph、storage、笔记、参考资料）放在平行资产树 `my_research/`。
- 无 CI/CD 需求，强调代码清晰性和可维护性。

## 目录结构与职责

```
pySciWS/
├── src/pysci/                  # 唯一安装包（editable 安装，直接 import pysci，无需 sys.path hack）
│   ├── paths.py                # 路径锚点：PROJECT_ROOT / ASSET_ROOT / research_asset_dir()
│   ├── common/                 # 共享工具层：matrix / numerical / space_curve / dtypes / plotting
│   ├── research/<name>/        # 研究层：每条研究线一个子包（当前仅 gain_ep）
│   │   ├── theory/             #   理论 / 仿真代码
│   │   └── experiment/         #   实验平台代码
│   └── skills/                 # LLM skill 后端（literature_research）
├── scripts/                    # 执行脚本（使用 pysci 中的代码，避免复杂定义）
│   ├── foundations/{cpa,ep_bic,topo_bic,matrix}/   # 共享工具层演示 / 验证
│   └── gain_ep/{theory,experiment}/                # gain_ep 研究线脚本
├── tests/                      # pytest 测试（testpaths = tests）
├── literature/                 # 文献检索 skill 的数据区
├── my_research/<n>_<name>/     # 平行资产树（非代码资产，不入包）
│   ├── theory/                 #   理论笔记 / PDF
│   ├── experiment/mphs/        #   COMSOL 仿真文件 (.mph)
│   ├── storage/                #   中间文件、数据、绘图输出（可再生，不入版本控制）
│   └── 参考资料/               #   该研究线的理论推导、文献资料
└── 参考资料（手动）/           # foundations 各主题参考资料（CPA / EP & BIC / Topo & BIC）
```

### 核心原则

- **src/pysci/common**: 定义跨研究线可复用的类和函数，包含完整的 docstring
- **src/pysci/research/<name>**: 某条研究线专属的理论 / 仿真 / 实验代码
- **scripts**: 使用 pysci 中的代码执行具体逻辑，避免定义复杂函数和类
- **my_research/<n>_<name>/storage**: 按研究线组织，存储数据文件和图表（可再生，不入版本控制）
- **包名与资产目录**: Python 包路径不能以数字开头，故资产目录 `my_research/1_gain_ep` 对应包路径 `pysci.research.gain_ep`（数字序号只保留在资产树目录名中）

## 路径解析（重要）

- **禁止** `sys.path.insert(...)` hack 与脆弱的 `Path(__file__).parents[N]` 层级硬编码。
- 资产路径统一通过 `pysci.paths` 定位：

```python
from pysci.paths import research_asset_dir

# gain_ep 研究资产根（= my_research/1_gain_ep）
asset_dir = research_asset_dir("gain_ep")
mph_file = asset_dir / "experiment" / "mphs" / "gainEP_basic.mph"
save_dir = asset_dir / "storage" / "data"
```

## 代码规范

### 1. 文件组织

**pysci 中的模块示例**：
```python
"""模块简短描述"""


def function_name(param: type) -> return_type:
    """Google-style docstring."""
    # 实现
```

**scripts 中的脚本示例**：
```python
"""脚本用途的简短描述。

该脚本通过 [方法] 实现 [目标]。
"""

import numpy as np
from pysci.common.numerical import function_name

# %% 使用 IPython 单元格分隔符组织代码
# ============================================================================
# 1. 第一部分标题
# ============================================================================

# 代码实现

# %%
# ============================================================================
# 2. 第二部分标题
# ============================================================================

# 代码实现
```

### 2. COMSOL 仿真工作流

**标准流程**：
1. 在 COMSOL GUI 中创建和调试模型，保存到 `my_research/<n>_<name>/experiment/mphs/`（包含多个版本，除mph外均可作为文本文件读取，方便你查看仿真文件细节）
2. 在 `scripts/<layer>/<name>/` 创建 Python 脚本自动化仿真
3. 使用 MPh 库连接 COMSOL Server（文档：https://mph.readthedocs.io/en/stable/）
4. 提取数据保存到 `research_asset_dir(name) / "storage" / "data"`
5. 可视化结果保存到 `research_asset_dir(name) / "storage" / "plots"`

**MPh 使用要点**：
- 使用 `mph.start(cores=1)` 连接 COMSOL Server

### 3. 数据和可视化

- **数据存储**: 保存到 `research_asset_dir(name) / "storage" / "data"`；使用 `numpy.save()` 保存为 `.npy` 格式
- **图表输出**: 保存到 `research_asset_dir(name) / "storage" / "plots"`
- **Matplotlib中文显示**: 在使用matplotlib时，建议在脚本开头使用项目通用配置以优化中文字符显示：
```python
# 导入项目通用的matplotlib配置，并应用
from pysci.common.plotting import setup_chinese_fonts

setup_chinese_fonts()
```

### 4. 参考资料

- foundations 各主题存储在根 `参考资料（手动）/<主题>/`；研究线专属资料存储在 `my_research/<n>_<name>/参考资料/`
- 使用 Markdown 格式，包含完整的数学推导和公式
- 使用 LaTeX 语法编写公式

## 开发实践

### 必须遵守

1. **不创建非必要文件**：
   - ❌ 不创建说明文档（README.md 等）
   - ❌ 不创建演示脚本（demo/example 等）
   - ✅ 通过清晰的 docstring 和注释说明代码用法

2. **代码质量**：
   - 所有函数/类必须包含完整的 docstring（Google 风格）
   - 使用类型注解（type hints）
   - 变量命名清晰，必要时使用希腊字母（如 `ωᵣ`, `Zᵢ`）

3. **依赖管理**：
   - 环境由 **uv** 管理，依赖声明在 `pyproject.toml`（`requires-python = "==3.13.15"`）
   - 同步环境：`uv sync`；运行脚本：`uv run python <script>`
   - 核心依赖：Python 3.13.15, mph, numpy, matplotlib, pandas, sympy, scipy, pyvista, pydantic
   - ❌ 不要自行安装/更新package，请通知我手动处理

### 可以忽略

1. **CI/CD**：无需考虑持续集成/部署
2. **版本控制细节**：专注于代码本身

> 注：`tests/` 现已纳入 pytest（`testpaths = tests`）。experiment 硬件用例用 `hardware` marker；
> 无 nidaqmx / COMSOL Server 时由 conftest 守卫自动跳过收集，不会导致 `uv run pytest` 失败。

## 环境信息

- **操作系统**: Windows 11
- **Shell**: PowerShell（注意：不支持 `&&` 操作符，使用 `;` 或分开执行）
- **IDE**: PyCharm
- **Python**: 3.13.15
- **环境管理**: uv（唯一 `.venv`，已从 conda 迁移；conda 已卸载）

## 典型工作流示例

### 符号计算验证（参考 `scripts/foundations/cpa/1-CPA-laser.py`）

1. 使用 sympy 定义符号和参数
2. 构建数学模型（矩阵、方程等）
3. 从 `pysci.common` 导入可复用函数进行数值化
4. 可视化结果

### COMSOL 仿真自动化（参考 `src/pysci/research/gain_ep/theory/sim.py`）

1. 连接 COMSOL Server
2. 加载 `.mph` 模型（`research_asset_dir("gain_ep") / "experiment" / "mphs" / ...`）
3. 运行仿真（参数扫描等）
4. 使用 `model.evaluate()` 提取数据
5. 保存数据到 `research_asset_dir("gain_ep") / "storage"`
6. 可视化分析

## 关键提醒

- **简洁优先**：代码应简洁明了，避免过度工程化
- **文档在代码中**：docstring 是唯一的文档，必须写清楚
- **快速迭代**：优先实现功能，保持代码整洁即可
- **存储规范**：所有输出文件必须保存到对应研究线的 `my_research/<n>_<name>/storage/` 子目录

---
