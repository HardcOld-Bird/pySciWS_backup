---
trigger: always_on
alwaysApply: true
---

# pySci 常驻开发规范

本文件是项目**唯一**的常驻规则，每个会话全文注入，每个字都在向所有未来会话征税。

**收录门槛**：一条内容能进这里，当且仅当「不读它就会做出错误或有破坏性的行为」。
可按需查阅的一律不进——**资产布局与产物目录的权威索引是 `src/pysci/paths.py`**
（模块 docstring 与各常量注释逐条写明了每个目录的用途），依赖与选型理由看 `pyproject.toml`
注释，技能用法看各 `SKILL.md`。**新增条目前先自问：能否写成一句可机械判定的话？**
写不成，说明它是偏好而非规则，不该占用这里。

> 本文件的「实测」结论均于 2026-10-04 在 Windows PowerShell **5.1.22621** 下逐条复核。
> 换 shell 版本或换机器后，这些结论需要重新验证。

---

## 1. 已裁决事项（非显然，无新证据勿重新提议）

### 1.1 权限模型：Agent 对项目内一切有完全执行权

**本项目的设计本意就是让 Agent 全权管理其内容** —— 若某项内容不希望被 Agent 管理，
用户就不会把它放进项目目录。因此：

- **`src/`、`data/`、`scripts/`、`tests/` 下的全部源码、数据与目录组织，Agent 拥有完全的
  修改、重构与删除权，无需事先征求同意。** 这包括删除项目内的任何数据。
  用户自行负责把不希望被删的东西备份到**项目外**；放在项目内即视为可管理。
- 架构设计、目录组织、实现取舍，**Agent 有权直接执行，且没有预判用户是否会反对的责任**。
  用户只在认为确有必要时主动介入并提出异议，且不会自行改代码 —— 一律与 Agent 商议后
  由 Agent 执行。**主动提出意见是用户的责任；对绝大多数更改，不要询问。**
- **唯一需要斟酌的例外：`data/research/`** —— 这是用户与 Agent 的工作交接区，双方都有编辑权。
  在此目录下**删除**内容时，**倾向于先问一句**（非强制要求，用户仍自行负责备份）；
  该目录下的新增与修改不受此限。
- 另一例外见 §6：本规则文件自身的裁决须由用户授权修改。

### 1.2 自研 vs 社区

- **默认用社区工具 / MCP**；自研只写社区工具表达不了的那层胶水。
- **同一能力绝不保留两份实现**：发现第三方依赖与自研代码功能重叠，必须在**同一个提交**里
  删掉一个 —— 不允许「先都留着」。
- 删除某层时，在 `pyproject.toml` 对应依赖旁注明「选了什么、弃了什么、为什么」。
  **这是本项目 ADR 的既定形式：决策记录写在决策发生的那个文件里。**

### 1.3 文档规模预算

- 每个 `SKILL.md` **≤ 8 KB**。超出的细节移到同目录 `references/*.md`，
  `SKILL.md` 只留决策与调用面，外加一行「遇到 X 时读 references/x.md」的索引。
- **理由**：文档越长 → 越会漏读条款 → 基于漏读做出错误行为 → 需要更多文档去澄清。
- **若某技能确实需要 > 8 KB 才能被正确驱动，那不是文档问题，而是该技能设计得过于复杂、
  难以使用。** 正确反应是重新设计它的使用方式，或把它拆成两个技能 —— **不是放宽预算**。

---

## 2. 上报纪律

- 被要求「审查 / 检查 / 评估」时，**只报告有证据的问题**：复现命令 + 实际输出 + 期望输出。
- 未发现问题时**直接回答「无缺陷」**。这是合法的、被期望的输出，不是低努力表现。
- **未被要求、却主动想建议重构时，请说出来。** Agent 默认对重构是消极的，几乎不会无故提议；
  因此当摩擦严重到让你想提议重构时，这个冲动本身很可能说明问题已经真实存在。
  这类反思应当**说出来并附证据**，而不是自我压制（执行权见 §1.1）。
- 两个方案僵持时，**用可执行命令的实测输出裁决**，不要靠推理，也不要另开会话问第三个意见。

---

## 3. PowerShell 约定（跨技能生效）

适用于各 `pysci-*` console script（清单见 `pyproject.toml` 的 `[project.scripts]`，数目会增长）
以及任何输出中文的 Python 命令。

### 3.1 中文输出必须先设 UTF-8（否则 Agent 拿不到任何状态信息）

调用前先在同一 shell 执行：

```powershell
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
```

**实测**：环境已设 `PYTHONUTF8=1`，Python 端 `sys.stdout.encoding` 恒为 `utf-8`；但只要 stdout
**被管道**（`| Out-String`、`2>&1 |`、重定向到文件），PowerShell 会用 `[Console]::OutputEncoding`
（本机默认 GBK/936）去解码这些 UTF-8 字节，中文全部变成 `妯″潡鏍?` 一类乱码。
同一 shell 会被后续命令复用，设一次即持续生效；要复现乱码基线须显式改回 `GetEncoding(936)`。

**两条代码侧自愈路径已实测无效，不要再尝试**：

- `sys.stdout.reconfigure(encoding="utf-8")` —— stdout 本就是 utf-8，空操作；
- 子进程内 `ctypes.windll.kernel32.SetConsoleOutputCP(65001)` —— 实测该调用**确实成功**
  （CP 由 936 变为 65001），但管道输出**仍然乱码**：父进程 PowerShell 在启动子进程前
  已缓存了解码器，子进程改不动它。

唯一可行的修复位置在 **PowerShell 侧**。

### 3.2 其余（均实测）

- **`Get-Content` / `Set-Content` 在 PS 5.1 默认按 ANSI/GBK 编解码，不是 UTF-8**（与 §3.1 的控制台
  编码是两回事）。不加 `-Encoding` 时：读 UTF-8 中文文件得乱码；写出的是 GBK 字节
  （实测 `增益管槽` → `d4 f6 d2 e6 b9 dc b2 db`），会**静默损坏**项目里任何 UTF-8 文件。
  加 `-Encoding UTF8` 可修正，但 PS 5.1 会**附带 BOM**（`ef bb bf`）。
  → **编辑项目文件一律用文件编辑工具，不要用 shell 重定向。**
- **双引号会被剥离，一律用单引号。** 多词参数：`research search 'acoustic exceptional point'`。
  `Write-Output` 的字符串若用双引号且含括号，括号内容会被当命令（实测 `(no pipe)` → CommandNotFound）。
  **内层双引号同样会被剥离**：`python -c 'print("模块树")'` 实际传给 Python 的是 `print(模块树)`
  → NameError。需要内层引号时，**把代码写进临时 `.py` 文件再执行**，不要在命令行里拼。
  带引号的 `-f` 格式化串与 `"$()"` 插值同此理，一律避免；拼表格改用
  `[pscustomobject]@{A = 1; B = 2} | Format-Table -AutoSize | Out-String -Width 200`。
- **链式命令用 `;`，绝不用 `&&`**：PS 5.1 下 `a && b` 直接抛解析错误。
- **`Select-String` 需要路径时取 `Path` 属性而非 `Filename`**：实测同名文件（如各技能下的
  `SKILL.md`）的 `Filename` 完全相同、无法区分，`Path` 才是完整路径。
- **丢弃 stderr 写 `2>$null`，绝不要转义成 ``2>`$null``**：反引号会让 PowerShell 把 `$null`
  当成**字面量文件名**，实测在当前目录生成一个名为 `$null` 的垃圾文件，并被 git 当作未跟踪文件
  混进 `git status`。要合并进 stdout 则用 `2>&1`。清理它时路径**用单引号**（`'.\$null'`）
  或 `-LiteralPath`；用双引号会让 `$null` 展开成空串、路径退化为目录本身（实测报「访问被拒绝」）。

---

## 4. 网络代理约定（跨技能生效）

- **默认直连；访问期刊数据库 / 下载正版全文绝对禁止走代理**（走代理即丧失机构访问资格）。
- 访问国内常无法直连的站点（GitHub、Google、YouTube 等）失败时，用本机代理
  `http://127.0.0.1:7890` 重试（2026-10-05 实测可用），**仅限单命令级注入，禁止持久化**：
  `git -c http.proxy=http://127.0.0.1:7890 push ...`、`curl -x http://127.0.0.1:7890 ...`、
  bash 前缀 `https_proxy=http://127.0.0.1:7890 <cmd>`。
- ❌ 禁止 `git config --global/--local http.proxy`、`setx`、写 profile、`export`/`$env:` 残留。

---

## 5. 环境与代码约定

- **Python 3.13.15**（`.python-version` 与 `requires-python` 锁定），**uv** 管理唯一 `.venv`，
  `pysci` 以 editable 方式装入。运行统一用 `uv run python ...` / `uv run pytest`；
  增删依赖改 `pyproject.toml` 后 `uv sync`。
- ❌ 禁止 `sys.path.insert(...)` hack 与脆弱的 `Path(__file__).parents[N]` 层级硬编码。
  资产路径统一走 `pysci.paths`：`PROJECT_ROOT` / `DATA_ROOT` / `ASSET_ROOT` / `LITERATURE_ROOT` /
  `research_asset_dir(name)`。包路径不能以数字开头，故 `data/research/1_gain_ep`
  对应 `pysci.research.gain_ep`。
- **产物落盘**：写入前用 `pysci.paths.assert_within_data(path)` 断言路径落在 `data/` 内
  （防散落护栏）；论文插图 / 理论计算 / 效果图分别用 `research_fig_dir(name, slug=...)` /
  `research_theory_dir(...)` / `research_artwork_dir(...)` 定位，
  **不得散落到仓库根或代码树中**。
- **分层**：`src/pysci/` 下的包**只定义**，`scripts/` **只使用**（不在 `scripts/` 里定义复杂类
  或长函数）。`pysci` 中公共接口须有 Google 风格 docstring 与完整类型注解。
- ❌ **不创建未被要求的文件**：不写 README / 说明文档 / demo / example，用法写进 docstring 与注释。
- **动手前先查阅既有资料**：该研究线资产目录（`research_asset_dir(name)`）下已有参考资料与笔记，
  代码中的符号须与文献保持一致，不要重新推导一遍。
- **写脚本时**：用 `# %%` 单元格分块（符号定义 → 推导 → 数值化 → 可视化）；
  变量名**优先用 Unicode 并与文献一致**（✅ `ωᵣ`、`Δω`、`ρ` ❌ `omega_r`、`delta_omega`）。

---

## 6. 本文件自身的维护

- **本文件只增不减是失败信号。** 每次新增条目前，先检查能否合并进已有条目，或删除一条过时的。
- 修改 §1 的裁决**必须由用户明确授权**，Agent 不得自行改写护栏。
- 发现表述与仓库现状矛盾（如引用的目录、函数、路径已不存在）**直接修正措辞**，
  但不得借机改变裁决内容。**修正前必须实测确认，不得凭记忆断言。**
