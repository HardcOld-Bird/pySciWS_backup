---
trigger: manual
alwaysApply: false
---

# pySci开发规范（符号计算）

## 1. 项目概述

### 1.1 项目性质
- 本项目是一个**物理学理论探索项目**，专注于波动声学领域的符号计算和数值模拟。项目采用Python生态系统，以SymPy为核心进行符号推导，结合NumPy/SciPy进行数值计算，使用Matplotlib/PyVista进行可视化。
- 对于有限元仿真，使用mph库通过 COMSOL Multiphysics 软件进行仿真研究和理论计算验证。
- 代码组织为单一 **src-layout 安装包 `pysci`**（uv 以 editable 方式装入唯一 `.venv`，直接 `import pysci...`，无需任何 `sys.path` hack）；非代码资产放在平行资产树 `data/research/`。
- 无 CI/CD 需求，强调代码清晰性和可维护性。

### 1.2 技术栈
- **Python版本**: 3.13.15（由 `.python-version` 与 `pyproject.toml` 的 `requires-python` 锁定）
- **核心依赖**: sympy, numpy, scipy, matplotlib, pyvista, pydantic, mph 等（详见 `pyproject.toml`）
- **开发工具**: ruff (代码质量), Pylance/Pyright (静态检查), pytest (测试)
- **环境管理**: uv（唯一 `.venv`，editable 安装 `pysci`；已从 conda 迁移，conda 已卸载。同步 `uv sync`，运行 `uv run python ...`）
- **开发环境**: Windows 11, PowerShell, PyCharm

---

## 2. 项目架构与目录结构

### 2.1 核心分层架构
项目遵循**理论-实现-应用**的分层思想，实现层统一收敛到安装包 `pysci`：

```
参考资料（手动）/     ← 理论层：foundations 各主题的学术论文和理论文档
    └── [主题名]/         （研究线专属资料在 data/research/<n>_<name>/参考资料/）
        └── *.md
src/pysci/            ← 实现层：唯一安装包
    ├── paths.py          （路径锚点：PROJECT_ROOT / ASSET_ROOT / LITERATURE_ROOT / research_asset_dir()）
    ├── common/           （通用科学计算库：dtypes / numerical / space_curve / matrix / plotting）
    ├── research/<name>/  （研究线专属：theory / experiment）
    └── skills/           （LLM skill 后端）
scripts/              ← 应用层：具体研究脚本
    ├── foundations/{cpa,ep_bic,topo_bic,matrix}/
    └── <name>/{theory,experiment}/
```

> **包名与资产目录**：Python 包路径不能以数字开头，故资产目录 `data/research/1_gain_ep`
> 对应包路径 `pysci.research.gain_ep`。资产路径统一通过 `pysci.paths.research_asset_dir(name)`
> 定位，禁止脆弱的 `Path(__file__).parents[N]` 层级硬编码。

### 2.2 目录详细说明

#### 2.2.1 `scripts/` - 研究脚本目录
**用途**: 存放具体研究主题的探索性代码

**组织方式**:
- 顶层按层级分组：`foundations/`（共享工具层演示，如 `cpa/`, `ep_bic/`, `topo_bic/`, `matrix/`）与 `<研究线名>/`（如 `gain_ep/{theory,experiment}/`）
- foundations 子文件夹名称与根 `参考资料（手动）/` 中的对应主题保持一致

**代码风格**:
- **必须使用** `# %%` 单元格分隔符（Jupyter-style cells），便于交互式探索
- **编写原则**: "只使用，不定义" - 仅调用 `pysci` 中的函数/类，避免冗长的本地定义
- **标准代码结构**:
  ```python
  """模块文档字符串：简要说明脚本目的"""

  # 导入标准库
  import [...]

  # 导入项目库
  from pysci.common[...] import [...]

  # %%
  # ============================================================================
  # 1. 定义符号和参数
  # ============================================================================
  # [符号定义代码]

  # %%
  # ============================================================================
  # 2. 构建符号模型
  # ============================================================================
  # [符号计算代码]

  # %%
  # ============================================================================
  # 3. 数值计算和可视化
  # ============================================================================
  # [数值化和绘图代码]
  ```

**关键要求**:
- 每个单元格应有清晰的注释分隔符（`# ===...===`）和标题
- 逻辑分块：符号定义 → 符号推导 → 数值计算 → 可视化
- 保持代码简洁，复杂逻辑应封装到 `pysci` 中

**变量命名规范**:
- **充分利用Unicode**: 科学计算中应尽可能使用Unicode字符提高可读性（数字/符号上下角标除外）
- **参照理论文献**: 变量名应与对应参考资料中的原始文献保持一致（数字/符号上下角标除外）
- **示例**:
  - ✅ 推荐: `ωᵣ`, `γᵢⁱⁿᵗ`, `Δω`, `κ̃`, `ρ̄` (使用希腊字母、**字母**上下角标、附加符号)
  - ❌ 避免: `omega_r`, `gamma_i_int`, `delta_omega`, `kappa_tilde`, `rho_bar`
- **优势**: 在复杂科学计算中，高可读性的变量名能显著提升开发和迭代效率
- **注意**: 由于python语法限制，数字上下角标（Unicode分类：Number, Other）和符号上下角标（Unicode分类：Symbol, Math）不能作为变量名的一部分（如`ω₁`, `ω₂`, `ω₊`, `ω₋`会报错）。对于数字上下角标，请使用普通数字代替（`ω1`, `ω2`）；对于符号上下角标，请使用类似的合法字符代替（比如`ω十`, `ω一`——这是因为`+`和`-`同样是非法的）。

#### 2.2.2 `src/pysci/common/` - 共享工具层
**用途**: 存放跨研究线可复用的通用科学计算功能

**编写原则**: "只定义，不使用" - 仅定义函数/类，不包含具体应用逻辑

**文件组织**（导入路径 `pysci.common.<module>`）:
- `dtypes.py`: 自定义数据类型（如 `ParamSpace3D`）
- `numerical.py`: 数值计算工具（如 `get_numpy_func`）
- `space_curve.py`: 空间曲线相关的可视化工具（如 `vis_complex_equation`）
- `matrix.py`: 复矩阵本征系统可视化；`plotting.py`: matplotlib 中文字体 / 绘图配置
- **新增文件**: 根据功能领域创建，命名应清晰反映用途（如 `matrix_utils.py`, `acoustic_models.py`）；研究线专属代码放入 `src/pysci/research/<name>/`

**代码规范**:
- **必须使用** Google风格的docstring
- **必须提供** 完整的类型注解（参数、返回值）
- **示例**: 参考 `src/pysci/common/numerical.py` 中的 `get_numpy_func` 函数
  ```python
  def get_numpy_func(expr: Expr, param_space: ParamSpace3D) -> Callable:
      """将 sympy 表达式转换为高效的numpy数值函数。

      Args:
          expr: sympy 表达式。
          param_space: 三维参数空间定义，包含参数符号、范围和分辨率。

      Returns:
          numpy数值函数。
      """
      # 实现代码
  ```

**何时添加新功能到 `pysci.common`**:
- 功能在多个脚本 / 研究线中重复使用
- 功能具有通用性，不局限于特定研究主题
- 功能逻辑复杂，封装后可提高 `scripts/` 的可读性

#### 2.2.3 理论文档目录（`参考资料（手动）/` 与 `data/research/<n>_<name>/参考资料/`）
**用途**: 存放学术论文、理论推导、技术参考文档

**组织方式**:
- foundations 各主题资料在根 `参考资料（手动）/`，目录结构**镜像** `scripts/foundations/`（每个主题一个文件夹）
- 研究线专属资料在 `data/research/<n>_<name>/参考资料/`（如 `data/research/1_gain_ep/参考资料/TEP`）
- 文件格式：主要为Markdown（`.md`）

**内容示例**: 参考 `参考资料（手动）/CPA/CPAL-传递矩阵法.md`
- 包含完整的数学推导（LaTeX公式）
- 提供物理背景和理论解释
- 作为 `scripts/` 中代码实现的理论依据

**LLM使用指南**:
- 在实现新功能前，**优先查阅**对应主题的参考资料
- 确保代码实现与理论文档中的公式/符号保持一致
- 理解物理意义，避免盲目编码

#### 2.2.4 `tests/` - 测试目录
**规则**:
- 已纳入 pytest（`pyproject.toml` 中 `testpaths = ["tests"]`，`--strict-markers`）
- 按 `tests/gain_ep/{theory,experiment}/` 等分区组织，与 `src/pysci` 结构对应
- experiment 硬件相关用例打 `hardware` marker；无 nidaqmx / COMSOL Server 时由各级 `conftest.py` 守卫自动跳过收集，`uv run pytest` 不会因此失败
- 运行：`uv run pytest`；排除硬件用例：`uv run pytest -m "not hardware"`
- 探索性理论脚本可临时验证后删除，但迁移而来的正式测试应保留维护

---

## 3. 代码规范

### 3.1 文档字符串（Docstring）
**强制要求**: 所有 `pysci` 中的函数/类必须使用Google风格的docstring

### 3.2 类型注解
**强制要求**: 所有 `pysci` 中的函数必须提供完整类型注解

### 3.3 代码质量
- **格式化**: 遵循Ruff配置（`pyproject.toml`，`line-length = 88`，isort `known-first-party = ["pysci"]`）

### 3.4 简洁性原则
- **避免过度检查**: 非必要情况下，不编写冗长的输入验证
- **信任类型系统**: 依赖Pylance/Pyright的静态检查
- **专注核心逻辑**: 代码应清晰表达物理/数学意图

### 3.5 可视化配置
**Matplotlib中文显示**: 在使用matplotlib时，建议在脚本开头使用项目通用配置以优化中文字符显示：
```python
# 导入项目通用的matplotlib配置，并应用
from pysci.common.plotting import setup_chinese_fonts

setup_chinese_fonts()
```

---

## 4. LLM工作指南

### 4.1 任务执行流程
1. **理解理论**: 查阅对应主题参考资料（`参考资料（手动）/[主题]/` 或 `data/research/<n>_<name>/参考资料/`）中的相关文档
2. **检查现有代码**: 使用`codebase-retrieval`查找已有实现
3. **确认依赖**: 验证需要使用的函数/类的签名
4. **编写代码**: 遵循项目架构和代码规范
5. **测试验证**: 必要时创建临时测试，验证后删除

### 4.2 典型工作模式

**符号计算脚本** (`scripts/`):
- 查阅参考资料 → 定义SymPy符号 → 构建符号表达式 → 数值化 → 可视化

**通用工具开发** (`pysci.common`):
- 在 `scripts/` 中编写原型 → 识别可复用部分 → 在 `pysci.common` 中实现（含文档和类型注解） → 在 `scripts/` 中调用测试

### 4.3 关键约束
**禁止**:
- ❌ 在 `scripts/` 中定义复杂类/长函数（应移至 `pysci`）
- ❌ 在 `pysci.common` 中编写具体应用逻辑（应在 `scripts/` 中）
- ❌ 使用 `sys.path.insert(...)` hack 或 `Path(__file__).parents[N]` 硬编码定位资产（用 `pysci.paths`）
- ❌ 手动编辑依赖配置文件（使用 uv 管理）
- ❌ 创建未经请求的文档文件

**必须**:
- ✅ 理论驱动：代码实现基于参考资料
- ✅ 分层清晰：`scripts/` 使用，`pysci` 定义，参考资料指导
- ✅ 文档完备：`pysci` 中的公共接口必须有文档和类型注解
- ✅ 保持 `scripts/foundations/` 和 `参考资料（手动）/` 的目录结构同步
- ✅ 使用并行工具调用提高效率

---
