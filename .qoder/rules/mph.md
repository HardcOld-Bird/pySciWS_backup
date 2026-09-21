---
trigger: manual
alwaysApply: false
---

# pySci开发规范（COMSOL仿真）

## 项目概述

- 本项目是一个**物理学理论探索项目**，专注于波动声学领域的符号计算和数值模拟。项目采用Python生态系统，以SymPy为核心进行符号推导，结合NumPy/SciPy进行数值计算，使用Matplotlib/PyVista进行可视化。
- 对于有限元仿真，使用mph库通过 COMSOL Multiphysics 软件进行仿真研究和理论计算验证。
- 无 CI/CD 需求，强调代码清晰性和可维护性。

## 目录结构与职责

```
pySciWS/
├── mphs/              # COMSOL 仿真文件 (.mph)
├── src/               # 可复用代码（类和函数定义）
├── scripts/           # 执行脚本（使用 src 中的代码，避免复杂定义）
├── storage/           # 中间文件、数据、绘图输出
├── 参考资料/          # 理论推导、文献资料（Markdown 格式）
├── config/            # 全局配置（如 matplotlib 配置）
└── tests/             # 仅在有需要时使用，无需主动维护
```

### 核心原则

- **src**: 定义可复用的类和函数，包含完整的 docstring
- **scripts**: 使用 src 中的代码执行具体逻辑，避免定义复杂函数和类
- **storage**: 按项目子目录组织（如 `storage/gainEP/`），存储数据文件和图表

## 代码规范

### 1. 文件组织

**src 中的模块示例**：
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
from src.module import function_name

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
1. 在 COMSOL GUI 中创建和调试模型，保存到 `mphs/项目名/`（包含多个版本，除mph外均可作为文本文件读取，方便你查看仿真文件细节）
2. 在 `scripts/项目名/` 创建 Python 脚本自动化仿真
3. 使用 MPh 库连接 COMSOL Server（文档：https://mph.readthedocs.io/en/stable/）
4. 提取数据保存到 `storage/项目名/`
5. 可视化结果保存到 `storage/项目名/plots/`

**MPh 使用要点**：
- 使用 `mph.start(cores=1)` 连接 COMSOL Server

### 3. 数据和可视化

- **数据存储**: 保存到 `storage/项目名/data/`；使用 `numpy.save()` 保存为 `.npy` 格式
- **图表输出**: 保存到 `storage/项目名/plots/`
- **Matplotlib中文显示**: 在使用matplotlib时，建议在脚本开头使用项目config以优化中文字符显示：
```python
# 导入项目通用的matplotlib配置，并应用
from config.matplotlib_config import setup_chinese_fonts
setup_chinese_fonts()
```

### 4. 参考资料

- 存储在 `参考资料/项目名/` 目录
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
   - 环境定义在 `environment.yml`（conda环境）
   - 核心依赖：Python 3.13, mph, numpy, matplotlib, pandas, sympy, scipy, pyvista, pydantic
   - ❌ 不要自行安装/更新package，请通知我手动处理

### 可以忽略

1. **测试套件**：无需主动维护 `tests/` 目录
2. **CI/CD**：无需考虑持续集成/部署
3. **版本控制细节**：专注于代码本身

## 环境信息

- **操作系统**: Windows 11
- **Shell**: PowerShell（注意：不支持 `&&` 操作符，使用 `;` 或分开执行）
- **IDE**: PyCharm
- **Python**: 3.13.9
- **Conda 环境**: sci

## 典型工作流示例

### 符号计算验证（参考 `scripts/CPA/1-CPA-laser.py`）

1. 使用 sympy 定义符号和参数
2. 构建数学模型（矩阵、方程等）
3. 从 src 导入可复用函数进行数值化
4. 可视化结果

### COMSOL 仿真自动化（参考 `scripts/gainEP/run_basic_simulation.py`）

1. 连接 COMSOL Server
2. 加载 `.mph` 模型
3. 运行仿真（参数扫描等）
4. 使用 `model.evaluate()` 提取数据
5. 保存数据到 `storage/`
6. 可视化分析

## 关键提醒

- **简洁优先**：代码应简洁明了，避免过度工程化
- **文档在代码中**：docstring 是唯一的文档，必须写清楚
- **快速迭代**：优先实现功能，保持代码整洁即可
- **存储规范**：所有输出文件必须保存到 `storage/` 对应子目录

---
