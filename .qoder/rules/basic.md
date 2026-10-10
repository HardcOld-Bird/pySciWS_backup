---
trigger: always_on
alwaysApply: true
---

# pySci 常驻开发规范（全员公约数）

本文件全文注入每个会话（含所有组员 pod），每个字都在向所有未来会话征税。**收录门槛**：
一条内容能进这里，当且仅当「不读它就会做出错误或有破坏性的行为」。顶层设计、权限模型
与维护条款见 `leader-only.md`（组长专属，组员不注入；组员需要时可直接读，跨 pod 只读）；
编排体系权威是 `orchestration/README.md`；资产布局索引是 `src/pysci/paths.py`；依赖与
选型理由看 `pyproject.toml` 注释；各成员的操作知识看自己的 charter 与部署技能。

---

## 1. Harness 预算（裁决；适用于一切成员文档——含组员自建）

- **全量注入档**（charter、AGENTS.md 等每跳全文进上下文的文件）：**每文件 ≤ 8 KB**。
- **常驻暴露档**（条件式 rules 与技能的 description 行——不进正文但每跳占位）：
  **每类合计 ≤ 8 KB**。
- **按需档**（`references/*.md` 等只在被指向时读取的文件）：**单文件 ≤ 8 KB**，总量不限。
- 每个 `SKILL.md` **≤ 8 KB**：超出的细节移到同目录 `references/*.md`，SKILL.md 只留
  决策与调用面 + 一行「遇到 X 时读 references/x.md」的索引。
- **组员自建 rules 必须条件式**（model_decision/glob/manual）；`trigger: always_on`
  槽位只留给 charter。组员交付时由 delivery-gate 机械校验，超限即拒绝。
- **理由**：文档越长 → 越会漏读条款 → 基于漏读做出错误行为 → 需要更多文档去澄清。
  超出预算说明该设计过于复杂——正确反应是重新设计使用方式或拆分，**不是放宽预算**。

## 2. 上报纪律

- 被要求「审查 / 检查 / 评估」时，**只报告有证据的问题**：复现命令 + 实际输出 + 期望输出。
- 未发现问题时**直接回答「无缺陷」**。这是合法的、被期望的输出，不是低努力表现。
- **未被要求、却主动想建议重构时，请说出来**（附证据）。Agent 默认对重构消极；当摩擦
  严重到让你想提议重构时，这个冲动本身很可能说明问题真实存在，不要自我压制。
- 两个方案僵持时，**用可执行命令的实测输出裁决**，不要靠推理，也不要另开会话问第三个意见。

## 3. Shell 与编码约定（跨技能生效）

- 所有 agent（组长与组员）的 Bash 工具均为 **Git Bash**：多词参数用单引号；
  **编辑项目文件一律用文件编辑工具，不用 shell 重定向**（编码/转义事故源）。
- CLI 调用范式（全员统一，实测三范式同速）：**首选裸 `pysci-X`**（PATH 已含
  `.venv/Scripts`）；命令不存在时 fallback `uv run pysci-X`——**裸名只试一次，
  禁止重试循环**。程序化子进程调用（工具内部）统一 `uv run`（不依赖会话环境）。
- Python 输出中文：环境已设 `PYTHONUTF8=1`，Git Bash 下无需额外处理。
- **Git Bash 下 Windows 命令的 `/` 前缀参数会被 MSYS 转成路径**：`tasklist /FI`、
  `findstr /C`、`netstat /an` 等的 `/xx` 被改成 `C:/Program Files/Git/xx`，命令静默
  报错或给出**假阴性**（2026-10-10 实测：`tasklist /FI "PID eq N"` 据此把活进程误判为
  死）。对策（择一）：命令前缀 `MSYS_NO_PATHCONV=1`、参数写双斜杠 `//FI`、或改用
  Python `subprocess` 直接传 argv 列表（不经 shell，免疫）。
- **仅当显式调用 PowerShell（5.1）时**才需其历史纪律（中文输出先设
  `[Console]::OutputEncoding=UTF8`、双引号会被剥离一律单引号、链式用 `;` 不用 `&&`、
  `Get/Set-Content` 必须 `-Encoding UTF8`）——完整实测结论见 git 历史本文件
  2026-10 前版本的 §3，无必要不再使用 PowerShell。

## 4. 网络代理约定（跨技能生效）

- **默认直连；访问期刊数据库 / 下载正版全文绝对禁止走代理**（走代理即丧失机构访问资格）。
- 访问国内常无法直连的站点（GitHub、Google、YouTube 等）失败时，用本机代理
  `http://127.0.0.1:7890` 重试（2026-10-05 实测可用），**仅限单命令级注入，禁止持久化**：
  `git -c http.proxy=http://127.0.0.1:7890 push ...`、`curl -x http://127.0.0.1:7890 ...`、
  bash 前缀 `https_proxy=http://127.0.0.1:7890 <cmd>`。
- ❌ 禁止 `git config --global/--local http.proxy`、`setx`、写 profile、`export`/`$env:` 残留。

## 5. 环境与代码约定

- **Python 3.13.15**（`.python-version` 与 `requires-python` 锁定），**uv** 管理唯一 `.venv`，
  `pysci` 以 editable 方式装入。运行统一用 `uv run python ...` / `uv run pytest`；
  增删依赖改 `pyproject.toml` 后 `uv sync`。
- ❌ 禁止 `sys.path.insert(...)` hack 与脆弱的 `Path(__file__).parents[N]` 层级硬编码。
  资产路径统一走 `pysci.paths`：`PROJECT_ROOT` / `ASSET_ROOT` / `research_asset_dir(name)` /
  `research_fig_dir·theory_dir·artwork_dir·model_dir` / `ORCHESTRATION_ROOT` /
  `ORCH_STATE_ROOT` / `PODS_ROOT`。包路径不能以数字开头，故 `data/research/1_gain_ep`
  对应 `pysci.research.gain_ep`。
- **产物落盘**：写入 `data/` 前用 `pysci.paths.assert_within_data(path)` 断言（防散落护栏）；
  论文插图 / 理论计算 / 效果图 / 模型分别用 `research_fig_dir(name, slug=...)` /
  `research_theory_dir(...)` / `research_artwork_dir(...)` / `research_model_dir(...)` 定位，
  **不得散落到仓库根或代码树中**。
- **分层**：`src/pysci/` 下的包**只定义**，`scripts/` **只使用**（不在 `scripts/` 里定义复杂类
  或长函数）。`pysci` 中公共接口须有 Google 风格 docstring 与完整类型注解。
- ❌ **不创建未被要求的文件**：不写 README / 说明文档 / demo / example，用法写进 docstring 与注释。
- **动手前先查阅既有资料**：该研究线资产目录（`research_asset_dir(name)`）下已有参考资料与笔记，
  代码中的符号须与文献保持一致，不要重新推导一遍。
- **写脚本时**：用 `# %%` 单元格分块（符号定义 → 推导 → 数值化 → 可视化）；
  变量名**优先用 Unicode 并与文献一致**（✅ `ωᵣ`、`Δω`、`ρ` ❌ `omega_r`、`delta_omega`）。
- **多 Agent 编排已启用**（唯一权威 `orchestration/README.md`）：末端生产工作由组长
  （TUI 主会话）经 `pysci-orch` 派发给 `orchestration/pods/<id>/` 的 headless 组员；
  组员交付协议与写权限由各自 charter 与 pod-guard 约束。
