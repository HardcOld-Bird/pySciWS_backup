---
name: pysci-infra
description: PySci 项目支撑系统操作面：项目内依赖管理（init/update/status/rollback/log）、分层测试入口（sub_root/tests）、eval 骨架（sub_root/evals）与项目 DSH 配置定位。需要装依赖、更新工具链、跑测试、定位 DSH/preset 配置或排查环境问题时使用。
---

# PySci Infra

项目支撑系统的操作面。设计目标见蓝图 `abstract-support-systems.md`；本技能只放“怎么用”。

## 目录锚点

- **工作区根**（git 根 = DSH workspace 根）：`AGENTS*.md`、`.agents/`、`sub_root/`。
- **Python 项目根**：`sub_root/`——`pyproject.toml`、`.python-version`、`uv.lock`、`.venv/`、`.env`、`src/pysci/`、`data/`、`tests/`、`evals/`、`deps_management/`。
- 代码里用 `pysci.paths`：`WORKSPACE_ROOT`（工作区根）与 `PROJECT_ROOT`（= `sub_root/`，Python 项目根）；不要手写 `Path(__file__).parents[N]`。
- uv / pytest / ruff 的命令都在 `sub_root/` 下执行（那里才是 pyproject 所在处）；`tests.cmd` 已代为切目录。

## 依赖管理

项目内工具链在 `sub_root/deps_management/`，版本真源是 `toolchain.json`。

```powershell
# 首次 / 换机
sub_root\deps_management\deps_manager\init-deps.cmd

# 只审计，不改
sub_root\deps_management\deps_manager\update-deps.cmd status

# 按 toolchain.json 重新对齐（安装缺失、重装 DSH）
sub_root\deps_management\deps_manager\update-deps.cmd update

# 依赖变更史 / 回退
sub_root\deps_management\deps_manager\update-deps.cmd log
sub_root\deps_management\deps_manager\update-deps.cmd rollback <ref>
```

- 工具安装物在 `deps/{git,node/runtime,uv,python}`；项目 DSH 在 `deps/node/modules`；`dsh_home` 也在那里。
- `init-deps` 还会生成项目内 **git 全局配置** `deps/git/config/gitconfig`：共享 shim 的 `GIT_CONFIG_GLOBAL` 指向它，vendored git 又不带 system config，所以 `user.name`/`user.email` 与 `credential.helper=manager` 只能待在这里。identity 仅首次从 `~/.gitconfig` 播种，之后保留文件里的现值（可手改，改后不必重跑）。
- **不要**把依赖装到系统/用户级；不要 `pip install --user`、`npm install -g`。
- 程序化调用用 `pysci.toolchain` 的绝对路径，不依赖 PATH 上的 `.cmd`；交互 shell 才走 `common\bin` shim。
- `.env` 只放本地密钥；不入库。

## 测试

第 1 层入口（P5 交付后）：

```powershell
sub_root\tests\tests.cmd all       # ruff + pytest + harness 契约
sub_root\tests\tests.cmd fast      # 快速子集
sub_root\tests\tests.cmd hardware  # 外部设备用例
```

- `testpaths = ["tests"]`（pytest rootdir 是 Python 项目根 `sub_root/`）；默认 `-m "not hardware"`。
- 产物在 `sub_root/tests/artifacts/`，不入库。
- 新技能必须在 `sub_root/tests/skills/<skill>/` 补机械断言，并跑 `all`。

## eval

第 2 层数据在 `sub_root/evals/`：

- `tasks/<id>/`：`prompt.md`、`task.yaml`、`checks/`。
- `runs/`：运行记录（不入库）。
- `reports/improvement-ledger.md`：改进台账。
- `overlays/`：harness 改进 patch。

运行器在 `sub_root/src/pysci/support/evals/`；优先 `dsh --profile headless`，需要细粒度事件时用 SDK。eval 会话的 cwd 是**工作区根**（`.env` 与 `.agents/` 所在处），不是 `sub_root/`。

## 项目 DSH 配置定位

- `dsh_home`：`sub_root/deps_management/deps/node/modules/dsh_home`
- web profile：`.../dsh_home/profiles/web/`
- preset 声明：`.../profiles/web/cordis.patch.yml`
- 生成片段：`.../profiles/web/pysci/{standard,cordis}.plugins.yml`（由 `deps_manager/scripts/extract_presets.mjs` 生成，不手改）
- 启动：项目根执行 `dsh web`；无头：`dsh --profile headless "<任务>"`。

## 排障顺序

1. `update-deps.cmd status` 看工具是否就位。
2. `git config --global --list` 应当有 `user.name`/`user.email`/`credential.helper=manager`；缺则重跑 `init-deps.cmd`（该文件在 `deps/` 内，不入库，随机器生成）。
3. `dsh --dsh-router-info` 看路由到的项目/DSH_HOME 是否正确。
4. 看 `dsh_home/profiles/web/cordis.patch.yml` 是否有 patch 警告（entry not found）。
5. 运行 `dsh --profile web --dump-config` 看组合树；不要在不清楚的情况下热改 preset。
6. 仍失败：保留命令、输出、版本，交给用户/leader。
