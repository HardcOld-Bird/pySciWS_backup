---
trigger: always_on
alwaysApply: true
---

# pySci开发规范（通用）

## 1. 项目概述

### 1.1 项目性质
- 本项目是一个**物理学理论探索项目**，专注于物理声学领域的文献阅读/调研、文献复现、符号计算、数值模拟、论文写作和科研绘图等。
- 项目采用Python生态系统，目前使用uv进行依赖管理，并工作于项目虚拟环境"pysci"（所有依赖安装在其中）。建议在运行代码前激活虚拟环境，或者使用`uv run`来自动使用虚拟环境。
- 无 CI/CD 需求，强调代码清晰性和可维护性。

### 1.2 技术栈
- **Python版本**: 3.13.15
- **核心依赖**: sympy, numpy, scipy, matplotlib, scienceplots, ultraplot, pyvista, mph 等（详见 `pyproject.toml`）
- **开发工具**: ruff (代码质量), Pylance/Pyright (静态检查), pytest (测试)
- **环境管理**: uv
- **开发环境**: Windows 11, PowerShell, PyCharm
