# PySci 改进台账

> 第 2 层（可基准测试层）与第 3 层（主观层）的交接面。每次 harness 专项改进一条，按时间倒序。
> 规则：问题来源、改动、基准结果、成本、是否实装、回滚点、预审表。失败与无效改动同样记录。

## 条目

### 2026-11-14 / shim-01：共享 shim 路径按「盘根约定」定位

- **问题来源**：用户确认约定——shim 恒位于 `<工作区所在盘>:\XXXIIIGGG\common\bin`（如 `D:\XXXIIIGGG\common\bin`），与工作区在盘内的深度无关，用户会遵守该约定。而 `init-deps.ps1` 当时按 `$WsRoot` 上推两级再拼 `common\bin`，算得 `D:\XXXIIIGGG\projects\common\bin`，每次 init 都误报 `shared shim layer missing`（上一轮 `git-cfg-01` 遗留项 ②）。
- **改动**：
  - `sub_root/deps_management/deps_manager/init-deps.ps1`：`$shimDir = "$(Split-Path -Qualifier $WsRoot)\XXXIIIGGG\common\bin"`，删除按深度推导的 `$commonRoot`，注释写明约定「不要按工作区深度推导」；文件首部布局注释同步去掉机器绝对路径。
  - `sub_root/src/pysci/toolchain.py`：模块 docstring 的硬编码 `D:\XXXIIIGGG\common\bin` 改为盘根约定表述。
  - `sub_root/tests/test_toolchain.py`：新增 `test_shared_shim_lives_at_workspace_drive_root`——按 `WORKSPACE_ROOT.drive` 拼盘根 shim 路径，未部署则 skip，部署则断言 `_find_project.cmd`/`git`/`node`/`uv`/`dsh` 五个入口在位。
  - 文档：`pysci-infra` 目录锚点新增共享 shim 约定位置，依赖管理一节点明「盘根」。
- **基准结果**：`init-deps.ps1 -CheckOnly` → `shared shim layer present: D:\XXXIIIGGG\common\bin`（修复前同一命令为 `[warn] shared shim layer missing at D:\XXXIIIGGG\projects\common\bin`）；`sub_root\tests\tests.cmd all` → **29 passed**。
- **成本**：约 8 轮工具调用、~10 分钟墙钟；无 eval token 消耗。
- **是否实装**：是。
- **回滚点**：`63d7bf7`。
- **风险备忘**：约定假设工作区所在盘根有 `XXXIIIGGG\common\bin`；若工作区被放到另一块盘而 shim 不在该盘根，init 会如实报 missing——这是预期行为，不是误报。

### 2026-11-14 / git-cfg-01：init-deps 生成项目内 git 全局配置（identity + credential helper）

- **问题来源**：本次会话收尾推送时暴露：共享 shim `D:\XXXIIIGGG\common\bin\git.cmd` 把 `GIT_CONFIG_GLOBAL` 指向 `sub_root/deps_management/deps/git/config/gitconfig`，同时把 `GIT_CONFIG_SYSTEM` 指向 `<git>/mingw64/etc/gitconfig`，而 vendored git 的 tar.bz2 两者都不提供，`init-deps.ps1` 也不生成。后果：`git config --global` 直接 fatal；`git commit` 无身份；`git push` 无凭据，只能靠 `git -c credential.helper=manager` 临时绕过；用户要求把该缺口作为支撑系统修复补上。
- **改动**：
  - `sub_root/deps_management/deps_manager/init-deps.ps1`：新增 `Get-GitKey`（读单个键，避免 PowerShell 丢空参数导致 `--default ""` 报错）与 `Set-ProjectGitConfig`（在 vendored git 安装后调用）。生成 `<deps>/git/config/gitconfig`：首次写 ASCII 头注释；`user.name`/`user.email` **仅在该文件缺键时**从 `%USERPROFILE%\.gitconfig` 播种，已有值保留（重跑 init/update/rollback 不覆盖项目内自定义）；`credential.helper` 恒设 `manager`；用 `git credential-manager --version` 探测 GCM，失败则告警。`-CheckOnly` 只报告现状，不落盘。
  - `sub_root/tests/test_toolchain.py`：新增 `test_project_git_global_config_carries_identity_and_helper`（沿用 `_needs_vendored` 跳过未装依赖场景，解析生成文件断言三键）。
  - 文档：`pysci-infra` 技能——依赖管理补该文件说明，排障顺序新增 `git config --global --list` 一步。
- **基准结果**：
  - 修复前 `init-deps.ps1 -CheckOnly` → `[warn] git global config missing: ...deps\git\config\gitconfig`。
  - 修复后实跑（`-SkipDsh`）→ `git identity seeded from C:\Users\antar\.gitconfig: HardcOld-Bird <antarcticbird@163.com>`，文件落盘。
  - 消费面（走 shim）：`git config --global --list` → `user.name`/`user.email`/`credential.helper=manager`；`git var GIT_COMMITTER_IDENT` → `HardcOld-Bird <antarcticbird@163.com>`（仓库 `.git/config` 的本地 identity 已删除，证明走的是项目内全局配置）；`git push --dry-run origin main` → `Everything up-to-date`（未加 `-c credential.helper`，证明凭据链路自动可用）。
  - `sub_root\tests\tests.cmd all` → **28 passed**。
- **成本**：约 22 轮工具调用、~25 分钟墙钟；无 eval token 消耗。
- **是否实装**：是。
- **回滚点**：`05c5926`（`git checkout 05c5926 -- sub_root/deps_management/deps_manager/init-deps.ps1 sub_root/tests/test_toolchain.py .agents/skills/pysci-infra/SKILL.md`）。
- **风险备忘 / 待办（交用户）**：① 该修复依赖项目外 shim 的 `GIT_CONFIG_GLOBAL` 约定，若 shim 改路由此处需同步；② 同文件里共享 shim 探测路径的「按深度上推」已改为盘根约定，见 `shim-01`；③ 该生成文件在 `deps/` 内被 `.gitignore` 覆盖，不入库，换机重跑 `init-deps` 即重建。

### 2026-11-14 / layout-02：`.env` 下沉 `sub_root/`，删除未启用的 `.pre-commit-config.yaml`

- **问题来源**：用户追问「`.env` 与 `.pre-commit-config.yaml` 能否也进 `sub_root`，额外维护面多大」。先核实事实再决定，用户选择「移 `.env` + 删 pre-commit 配置」。
- **事实核查**：
  - DSH 本体不读 `.env`：`@deepseek-ai/*` 安装目录内无 `dotenv` / `loadEnvFile`，`dsh --help` 无 env 文件开关。全仓库唯一读者是 `evals/cli.py`。
  - pre-commit 配置当时**完全未生效**：`.git/hooks` 为空（从未 `pre-commit install`），且注释写「ruff 0.16.7 与 uv.lock 一致」而 lock 实为 0.17.5，已漂移；其 `ci:` 段指向 pre-commit.ci，而该服务要求配置在仓库根，搬进子目录会直接失效。
- **改动**：
  - `sub_root/src/pysci/support/evals/cli.py`：`env_file = WORKSPACE_ROOT / ".env"` → `PROJECT_ROOT / ".env"`。
  - 文档同步：`paths.py`（工作区根不再含 `.env`）、蓝图 `abstract-workspace.md`（树 +「两个根」+ `.env` 归属）、`philosophy-detailed.md` §3.1（根目录允许项收紧为 `.git/`、`.gitignore`、`.gitattributes`、`.agents/`、`AGENTS*.md`）、蓝图 `SKILL.md`、`AGENTS.md`（2.4 与 §五 密钥位置）、`pysci-infra`、`repo-recon`。
  - 删除 `.pre-commit-config.yaml`（`git rm` + 删文件）。
- **基准结果**：`_load_api_key()` → `True`（len=35，证明从 `sub_root/.env` 读取）；`pysci-eval ... example-smoke` → **returncode 0 / 3.66s / stdout `OK`**；`tests.cmd all` → **27 passed**；`git check-ignore sub_root/.env` → 命中 `.gitignore:16:.env`（仍不入库）。
- **成本**：约 12 轮工具调用、~15 分钟墙钟；1 次 headless smoke。
- **是否实装**：是。
- **回滚点**：`a3beda3`（`.env` 移回根、恢复 pre-commit 配置）。
- **风险备忘**：① 结论「DSH 不读 `.env`」成立于 0.2.1-rc.5；若上游将来在 workspace 根加 dotenv 加载，需重新评估。② 质量门现由 `tests.cmd all`（ruff + pytest + harness 契约）承担；若将来重新引入 pre-commit，配置应留在仓库根（工具约定），不要放进 `sub_root/`。③ `toolchain.py:224` 与 `tests/conftest.py:4` 的注释仍以 pre-commit 举 PATH 例子，不影响行为，未改。

### 2026-11-13 / layout-01：根目录收纳（Python 工程移入 `sub_root/`）

- **问题来源**：用户指示（本会话）：根目录要与 `mySimsWS2` 同形，只留工作区级元数据，凡可移动内容（`src`、`data` 等）全部收进 `sub_root/`。与蓝图 `philosophy-detailed.md` §3.1「根目录约定」（原来把 `pyproject.toml`、`.python-version`、`.venv/` 列为根的合法例外）冲突，已向用户报告并获批准改蓝图。
- **改动**：
  - 移动：`pyproject.toml`、`.python-version`、`uv.lock`、`src/`、`data/` → `sub_root/`；`.venv/` 删除后在 `sub_root/` 重建（旧 launcher 内嵌旧绝对路径，不可搬）。
  - 代码：`paths.py` 增加双锚点 `WORKSPACE_ROOT`（git/DSH 根）与 `PROJECT_ROOT`（= `sub_root/`），`TESTS_ROOT`/`EVALS_ROOT` 随之改；`toolchain.DEPS_ROOT`；`support/evals/cli.py` 的 dsh/dsh_home/node 路径、`.env` 位置与 `cwd`（→ 工作区根）；`__init__.py` 导出 `WORKSPACE_ROOT`。
  - 脚本：`tests.ps1`（Python 项目根 = 脚本父目录，harness 路径改 `tests/harness`）、`init-deps.ps1`（`$ProjRoot` 与新增 `$WsRoot`）、`update-deps.ps1`（git 操作用 `$WsRoot`）。
  - 配置：`sub_root/pyproject.toml` 的 `testpaths=["tests"]`；`.gitignore` 全部 root-anchored 模式加 `sub_root/` 前缀。
  - 测试：`test_blueprint.py`（`.agents`/`AGENTS*` 走 `WORKSPACE_ROOT`）、`test_presets.py`（`deps_management` 直挂 `PROJECT_ROOT`、`cwd=WORKSPACE_ROOT`）。
  - 文档：蓝图 `abstract-workspace.md`（新树 + “两个根”）、`abstract-core.md`、`abstract-support-systems.md`、`philosophy-detailed.md` §3.1（删除 `.venv` 例外条款）、`philosophy-community-first.md`、蓝图 `SKILL.md`；`AGENTS.md`（2.2 路径 + 新增 2.4 目录锚点）、`AGENTS.meta.md`（三不碰路径）；技能 `pysci-infra`（新增“目录锚点”节、`testpaths`、运行器路径）、`repo-recon`（新增“两个根”）。
- **基准结果**（判据为机械测试 + 冒烟）：
  - `sub_root\tests\tests.cmd all` → ruff 绿 + **27 passed**（rootdir 现为 `sub_root`）。
  - `dsh --profile web --dump-config` → exit 0、1431 行、无 patch 警告，三 preset 与 `selectedDefault: pysci-worker` 不变。
  - `dsh --dsh-router-info` → project 仍解析为工作区根，DSH_HOME 不变。
  - `update-deps.cmd status` → git/node/uv/python/dsh 全部 ok。
  - `pysci-eval run sub_root\evals\tasks\example-smoke` → **returncode 0，4.1s**，stdout `OK`（验证工作区根 cwd、`.env` 读取、dsh 绝对路径）。
  - 路径锚点冒烟：`PROJECT_ROOT=…\sub_root`、`WORKSPACE_ROOT=…\pySciWS2`，各 `*_ROOT` 全部 `is_dir()`。
- **成本**：约 35 轮工具调用、~40 分钟墙钟；无 eval token 消耗（仅 1 次 headless smoke）。
- **是否实装**：是（工作区未提交，HEAD 仍为 `a3beda3`）。
- **回滚点**：`git checkout -- .` 回到 `a3beda3`；`.venv` 由 `sub_root\deps_management\deps_manager\init-deps.cmd` 重建。
- **预审表**：不适用（结构迁移，判据是既有机械测试；无 harness 行为对比）。

### 2026-11-13 / env-01：workspace 写权限阻断 pwsh（验收中发现的阻断项）

- **问题来源**：P6 交接验收（`HANDOFF.md` §3）过程中，`pysci-worker` 与 `pysci-leader` 两个子代理分别独立复现 `pwsh` 启动即失败：`SetNamedSecurityInfoW failed (Win32 5): grantWrite(D:\XXXIIIGGG\projects\pySci\pySciWS2)`。完整记录见 `sub_root/evals/reports/p6-acceptance-20261113.md` §5。
- **改动**：非 harness 改动，属环境前置条件修复。用 `diagnose-windows-sandbox-acl` 技能单命令（一次授权、非提权）在 workspace 根补一条当前用户 FullControl 允许项；文件内容与所有者未变，未删任何拒绝项。
- **基准结果**：修复前 shell 完全不可用 → 修复后 `pwsh` 在 meta 与 worker 侧均正常；`tests.cmd all` 27 passed、`harness` 11 passed、`dsh --profile web --dump-config` 无警告。验收 A/B/C 三项均符合预期。
- **成本**：约 10 轮工具调用、约 10 分钟墙钟；含 1 次授权升级（脚本未签名需 `-ExecutionPolicy Bypass`，在第二轮才成功）。
- **是否实装**：是（环境修复已生效）。
- **回滚点**：`sub_root/evals/acl-recovery/acl-backup-c27ab7c5b09146a296f53dab043fad0b.json`（+.ps1，内含回滚命令）。
- **待办（交用户）**：①「pwsh 能否启动」应作为交接验收的机械项写入清单；② 该技能脚本受执行策略限制、文档未提示，建议反馈上游。

### 模板

- **日期 / id**：
- **问题来源**：会话、失败记录或用户反馈；附路径/会话 id。
- **改动**：改了哪个 preset / 技能 / AGENTS 段落 / 工具配置；diff 或路径。
- **基准结果**：任务 id、base vs delta、判据通过情况、人工结论。
- **成本**：token、墙钟、轮次。
- **是否实装**：是/否；否的话原因。
- **回滚点**：备份文件或 git commit。
- **预审表**：链接到本次对比的预审表；逐项初判 + 证据引用。
