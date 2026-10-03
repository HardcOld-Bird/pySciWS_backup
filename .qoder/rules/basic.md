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

## 2. PowerShell 调用约定（跨技能生效）

适用于全部 6 个 `pysci-*` console script（`pysci-research` / `pysci-imagine` /
`pysci-simulation` / `pysci-compose` / `pysci-figures` / `pysci-theory`）以及任何输出中文的
Python 命令。

### 2.1 中文输出必须先设 UTF-8 控制台编码（关键）

调用任何 `pysci-*` CLI **之前**，先在同一个 shell 里执行：

```powershell
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
```

原因（已实测）：项目环境已设 `PYTHONUTF8=1`，Python 端 `sys.stdout.encoding` 恒为 `utf-8`，
因此**直接输出到终端时不会乱码**；但只要 stdout **被管道**（`| Select-Object`、`| Out-String`、
`2>&1 |`、重定向到文件等），PowerShell 会用 `[Console]::OutputEncoding`（本机默认 GBK/936）
去解码这些 UTF-8 字节，中文全部变成 `鈥?` / `妯″潡鏍?` 一类乱码，Agent 由此拿不到任何状态信息。

**两条代码侧自愈路径均已实测无效，不要再尝试**：

- `sys.stdout.reconfigure(encoding="utf-8")` —— stdout 已经是 utf-8，是空操作；
- 子进程内 `ctypes.windll.kernel32.SetConsoleOutputCP(65001)` —— 管道场景下同样乱码
  （PowerShell 在启动子进程前已缓存了解码器）。

唯一可行的修复位置在 **PowerShell 侧**。另注意：同一 shell 会被后续命令复用，前面设过的编码
会一直生效；反过来，若要复现乱码基线，须显式改回
`[Console]::OutputEncoding = [System.Text.Encoding]::GetEncoding(936)`。

### 2.2 其余约定

- **多词参数一律用单引号**：`research search 'acoustic exceptional point'`。双引号会被剥离，
  导致参数被拆散或直接 ParserError。
- **链式命令用 `;`，绝不用 `&&`**：PowerShell 不支持 `&&` 作为语句分隔符。
- **命令行内避免带引号的 `-f` 格式化字符串与 `"$()"` 插值**：引号同样会被剥离。需要拼接表格时
  改用 `[pscustomobject]@{A = 1; B = 2} | Format-Table -AutoSize | Out-String -Width 200`。
- **`Select-String` 结果需要路径时取 `Path` 属性而非 `Filename`**：多个同名文件（如各技能下的
  `SKILL.md`）用 `Filename` 无法区分。
- **丢弃 stderr 写 `2>$null`，绝不要转义成 ``2>`$null``**：反引号会让 PowerShell 把 `$null`
  当成**字面量文件名**，于是在当前目录创建一个名为 `$null` 的文件（实测留下几百字节的
  stderr 垃圾，并被 git 当作未跟踪文件混进 `git status`）。要合并进 stdout 则用 `2>&1`。
  读取或清理这类文件必须用 `-LiteralPath`（`Get-Content -LiteralPath '.\$null'`），否则
  `$null` 会在路径里再次被展开成空串，命令静默作用到错误的目标上。
