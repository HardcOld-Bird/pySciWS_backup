---
name: pysci-infra
description: PySci 项目支撑系统操作面：项目内依赖管理（init/update/status/rollback/log）、分层测试入口（sub_root/tests）、eval 骨架（sub_root/evals）与项目 DSH 配置定位。需要装依赖、更新工具链、跑测试、定位 DSH/preset 配置或排查环境问题时使用。
---

# PySci Infra

项目支撑系统的操作面。设计目标见蓝图 `abstract-support-systems.md`；本技能只放“怎么用”。

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

- `testpaths = ["sub_root/tests"]`；默认 `-m "not hardware"`。
- 产物在 `sub_root/tests/artifacts/`，不入库。
- 新技能必须在 `sub_root/tests/skills/<skill>/` 补机械断言，并跑 `all`。

## eval

第 2 层数据在 `sub_root/evals/`：

- `tasks/<id>/`：`prompt.md`、`task.yaml`、`checks/`。
- `runs/`：运行记录（不入库）。
- `reports/improvement-ledger.md`：改进台账。
- `overlays/`：harness 改进 patch。

运行器在 `src/pysci/support/evals/`；优先 `dsh --profile headless`，需要细粒度事件时用 SDK。

## 项目 DSH 配置定位

- `dsh_home`：`sub_root/deps_management/deps/node/modules/dsh_home`
- web profile：`.../dsh_home/profiles/web/`
- preset 声明：`.../profiles/web/cordis.patch.yml`
- 生成片段：`.../profiles/web/pysci/{standard,cordis}.plugins.yml`（由 `deps_manager/scripts/extract_presets.mjs` 生成，不手改）
- 启动：项目根执行 `dsh web`；无头：`dsh --profile headless "<任务>"`。

## 排障顺序

1. `update-deps.cmd status` 看工具是否就位。
2. `dsh --dsh-router-info` 看路由到的项目/DSH_HOME 是否正确。
3. 看 `dsh_home/profiles/web/cordis.patch.yml` 是否有 patch 警告（entry not found）。
4. 运行 `dsh --profile web --dump-config` 看组合树；不要在不清楚的情况下热改 preset。
5. 仍失败：保留命令、输出、版本，交给用户/leader。
