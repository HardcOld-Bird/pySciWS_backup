# 子代理动态模型路由薄插件（dsh-subagent-model-policy）· 实施计划

> **状态**：**已完成并归档（2026-11-16）**。包、配置点、日历数据、34 项单测、bundle 挂载、`dsh --profile web --dump-config` 组合验证、启动冒烟与 `tests.cmd all` 36 项回归全部通过；§7.2 派发级 live 探针留待项目 DSH 进入日常使用时执行（清单见 §7.2）。
> **来源**：复用 mySimsWS2 已验收的同名插件（其 DSH 同为 0.2.2-rc.1，代码可近乎原样移植）；本计划按 pySciWS2 语境改写，§编号与插件源码注释互引保持一致。
> **用户设计决策（2026-11-16 确认）**：① 直接复制 mySimsWS2 的插件模式（`modules/dsh_plugins/` + `link:` 挂载），不重新造轮子；② 缺省路由由 guard 拒绝、模型按拒绝文本重试（不做常驻上下文注入）；③ profile 全局生效；④ qwen 映射留占位并在 provider 缺失时 fail-loud；⑤ 未显式传路由时，仅当父路由恰好等于本次默认才放行；⑥ 插件放 `sub_root/deps_management/deps/node/modules/dsh_plugins/`；⑦ 用户配置点放 `sub_root`，格式 YAML；⑧ **初始 `plan: mimo`**，授权白名单三路由全开（mimo flash/pro + deepseek-flash）；⑨ QA 补机械断言并跑 `tests.cmd all`。
> **上位**：`AGENTS.md` §2.1 三段式；`blueprint-canon` 的社区优先与薄胶水。
> **非目标**：主会话模型路由、`workflow` 旁路、UI 面板、自动切换主模型。

## 1. 选型结论（已与用户确认）

- **底座**：自研薄插件（经 mySimsWS2 社区调研选型并验收；其调研结论：社区最接近的 `dsh-delegate-router` 只能当参考，`dsh-subagent-router` 语义相反）。本项目直接复用其实现，属「同生态已验证薄胶水」而非新自研。
- **复用**：内置 `subagent-model-selection-settings` 作**授权层**（会话级静态并集，GUI 管理）；`chinese-days` 的节假日 JSON 作**日历数据**（MIT，见插件 `data/SOURCE.md`）。
- **拦截**：`ctx.tools.guard`（同步、只拒不改）。v1 不注入常驻上下文；拒绝文本是唯一教学点。
- **范围**：profile 全局 host 插件，管所有 preset 的 `subagent`/`subagent_fork` 调用（worker/leader/meta）。
- **qwen**：映射留占位；`plan=qwen` 且 provider 未注册时**直接拒绝并说明**，不静默回退。

## 2. 机制接缝（已核实，2026-11-16，DSH 0.2.2-rc.1）

| 事实 | 证据 |
|---|---|
| guard 是同步硬拒，返回 `string` 即拒绝，`undefined` 放行 | `ToolsService.guard(guard)`（dsh-tools `lib/index.js` L649–659） |
| `subagent` 的 `provider/model` 参数只在工具行 `modelSelection: true` 时暴露 | dsh-tool-subagent `lib/index.js` L304–305、L351（新 schema：`modelSelection`/`modelDiscovery`，旧键 `modelSelectionSettings` 已废弃） |
| 工具参数声明是对象根 JSON Schema：参数名在 `properties` 下 | 判定用 `declaresRouteParameter()` 兼容两种形状 |
| 原生调用不带 `exec.schema`，判「是否暴露 provider」只能问注册表 `ctx.tools.get(name, agent, toolSet)` | dsh-tools `lib/index.js` L519（`get(name, scope, toolSet)`）；`definitions(scope, toolSet)` 供激活期巡检 |
| 会话授权集记录在 log-only 事件 `subagent/model-selection-policy`；子会话继承父策略 | dsh-tool-subagent `lib/index.js` L132–174（投影键 `subagentModelSelectionPolicy`、`stateVersion: 1`） |
| 插件可用 `ctx.sessionProjections` 折叠该事件按会话读授权集 | dsh-session-projection 注册契约（同 key 共享单元计数） |
| 父路由可从 `session.requestHeader().config` 或 `agent.options.provider/model` 读到 | index.js `parentRouteOf()` |
| `subagent_fork` 无 `provider/model` 参数，路由继承父会话 | 工具行 `provider: fork, toolName: subagent_fork` |
| 授权行 `subagent-model-selection-settings` 是 `.volatile()`，GUI 改动即时生效；策略在 `agent/created` 时盖章，只影响之后新建的顶层会话 | dsh-tool-subagent `lib/model-selection-settings.js`（`enabled`/`allowedModels`） |
| 时区口径：按 `Asia/Shanghai` 读本地时钟（`Intl`），不联网校时 | 前身 pySciWS `routing.py` |
| 峰段：工作日 `[9,12)`、`[14,18)`；周六日与法定节假日全天谷（调休上班的周六日仍按谷）；缺数据年份按峰保守并告警 | 同上 |
| 早切：仅谷→峰，提前 `leadtimeSeconds`（默认 600）；反向不早切 | policy.js `resolvePolicy()` |
| **DSH 升级（2026-11-16）**：0.2.1-rc.5 → 0.2.2-rc.1（npm `latest`），`extract_presets.mjs` 已重跑，profile 补丁旧键 `modelSelectionSettings` 已改新键，`dsh --profile web --dump-config` 通过 | `toolchain.json`、`modules/package.json`、`cordis.patch.yml.bak-20261116-pre-0.2.2` |

## 3. 判定表（v1）

`plan` 三值；峰段/谷段由 §2 的日历与时钟判定。`deepseek.flash` 默认取 `deepseek-account/deepseek-flash`（与主会话默认同源）。

| plan | 峰 | 谷 |
|---|---|---|
| mimo | 允许 `mimo.flash`、`mimo.pro`；默认 `mimo.flash` | 允许 `deepseek.flash` + `mimo.pro`；默认 `deepseek.flash` |
| qwen | 允许 `qwen.flash`、`qwen.pro`；默认 `qwen.flash` | 允许 `deepseek.flash` + `qwen.pro`；默认 `deepseek.flash` |
| deepseek | 允许 `deepseek.flash`；默认 `deepseek.flash` | 同峰 |

- **早切**：仅「当前谷、且 `now + leadtimeSeconds` 进峰」时按峰处理；反向不早切。默认 `leadtimeSeconds: 600`。
- **未知年份**：`unknownYear: peak` 默认；拒绝文本带「日历缺 <年> 年，按峰保守」。
- `basis` 必须可自证，例：`周四 10:12，北京时间；工作日峰段 09:00–12:00`。

## 4. Guard 决策表

| 情形 | 结果 |
|---|---|
| 工具实例无 `provider` 参数（`modelSelection: false`） | allow（fail-open；apply 时记一条 warning，不逐次刷屏） |
| 显式 `provider/model` 且在 allowed 集 | allow |
| 显式但不在 allowed 集 | deny |
| 未显式，且父路由**等于本次默认** | allow（继承即默认；不浪费一跳） |
| 未显式，其他情况 | deny（带默认与允许集，指示重试） |
| `subagent_fork`，`guardFork: true`，父路由 ∉ allowed | deny |
| `plan=qwen` 且 `qwen` 的 provider 未在 `llm.listProviders()` | deny（fail-loud，提示补 provider 与映射） |
| 会话没有 policy（旧会话） | allow（fail-open；每个旧会话记一次 warning） |
| 显式路由在 `plans` 允许集、但不在会话 policy | deny（提示更新白名单并开新会话） |
| `plans` 允许集与会话 policy 的交集为空 | deny（列出两侧路由） |
| 其他工具 | 不看 |

拒绝文本模板（可执行）：

```
subagent 路由被拒绝：plan=mimo，相位=峰（周四 10:12，北京时间；峰段 09:00–12:00）。
允许：xiaomi-token-plan-cn/mimo-v2.6-flash（默认）、xiaomi-token-plan-cn/mimo-v2.6-pro。
请带 provider/model 重试本次调用。
```

## 5. 文件与配置

### 5.1 目录

```
sub_root/subagent-model-policy.yaml                    # 用户配置点（入 git；改 plan 只动这里）

sub_root/deps_management/deps/node/modules/dsh_plugins/subagent-model-policy/
├─ package.json              # private、type module、dsh.bundle.patch、files
├─ cordis.patch.yml          # 只插入策略行；授权行由 GUI/profile 管理
├─ index.js                  # apply：解析配置、注册 session projection、注册 guard
├─ config.js                 # 默认值 + 严格校验（非法即 throw）；环境变量 PYSCI_SUBAGENT_MODEL_POLICY
├─ policy.js                 # 纯函数 resolvePolicy → { phase, allowed, default, basis, warnings }
├─ calendar.js               # 加载 data/chinese-days.json；周末/节假日/未知年份
├─ data/chinese-days.json    # 自 chinese-days 派生：只保留 holidays
├─ data/SOURCE.md            # 版本、抓取日、MIT、刷新命令（原地 ADR）
├─ scripts/refresh-holidays.mjs  # 从 jsDelivr 抓取并覆盖 data/；人工按年跑
├─ locale/en.json / zh.json  # meta.title / meta.description
├─ icon.svg
└─ test/policy.test.mjs / guard.test.mjs / config.test.mjs
```

运行时**不新增安装依赖**：YAML 解析用 DSH 自带的 `yaml`（`modules/node_modules/yaml`），`zod` 同理；插件位于 `modules/dsh_plugins/` 下，`link:` 安装时 Node 从祖先 `modules/node_modules` 解析。

### 5.2 `package.json` 要点

`name: dsh-subagent-model-policy`、`private`、`type: module`、`main: index.js`、`dsh.bundle.patch: ./cordis.patch.yml`（原样复制自 mySimsWS2）。

### 5.3 `cordis.patch.yml`（只插策略行）

```yaml
- insert:
    - id: subagent-model-policy
      name: 'dsh-subagent-model-policy'
      config: {}
```

授权行 `subagent-model-selection-settings` **不由插件插入**，写入 profile 的 `cordis.patch.yml`（GUI 后续改动会写回该处，单一真源）：

```yaml
- id: subagent-model-selection-settings
  name: "@deepseek-ai/dsh-tool-subagent/model-selection-settings"
  config:
    enabled: true
    allowedModels:
      - { provider: xiaomi-token-plan-cn, model: mimo-v2.6-flash }
      - { provider: xiaomi-token-plan-cn, model: mimo-v2.6-pro }
      - { provider: deepseek-account, model: deepseek-flash }
```

### 5.4 配置点（`sub_root/subagent-model-policy.yaml`）

用户改 `plan` 只动这一行；其余键有默认值。插件按 mtime 缓存热读，改完下一次派发生效（计划 §5.4「用户只在一个配置点切换套餐，改完即生效」）。

```yaml
plan: mimo
plans:
  mimo:
    flash: { provider: xiaomi-token-plan-cn, model: mimo-v2.6-flash }
    pro:   { provider: xiaomi-token-plan-cn, model: mimo-v2.6-pro }
  qwen:
    flash: { provider: qwen-token-plan-cn, model: qwen-flash }   # provider 落地前保持占位
    pro:   { provider: qwen-token-plan-cn, model: qwen-max }
  deepseek:
    flash: { provider: deepseek-account, model: deepseek-flash }
peak: [[9, 12], [14, 18]]
leadtimeSeconds: 600
unknownYear: peak
guardFork: true
forcePhase: auto
```

解析顺序：`PYSCI_SUBAGENT_MODEL_POLICY` 环境变量 → 插件目录上溯六级 → `DSH_HOME` 上溯五级；取第一个存在的候选。找不到 → `apply` 直接 throw（不静默用默认）。非法值一律 throw，不做回退（极简原则）。

### 5.5 授权契约（按会话读 policy，不读 profile 行）

- 授权集由 **GUI 管的 profile 行**持有；插件不读/不写该行。
- 插件注册 `sessionProjections` 单元（key `subagentModelSelectionPolicy`，与内置同键共享单元），折叠 `subagent/model-selection-policy`。
- **有效允许集 = `plans` 解析结果 ∩ 会话 policy**（不会「guard 放行、原生随后拒绝」）。
- **旧会话无 policy** → fail-open 并记一次 warning；新会话出生即带 policy，严格生效。
- 会话 policy 缺 `plans` 路由 → deny 并提示「更新白名单并开新会话」。

## 6. 实施步骤

1. ✅ 建包：从 mySimsWS2 复制插件目录，改 `config.js` 环境变量名为 `PYSCI_SUBAGENT_MODEL_POLICY`，微调注释指涉（计划路径、blueprint-canon 路径、「前身项目 pySciWS」）。
2. ✅ 写 `sub_root/subagent-model-policy.yaml`（§5.4，`plan: mimo`）。
3. ✅ 接线 profile：`profiles/web/package.json` 加 `link:` 依赖 + bundle 列表项；`cordis.patch.yml` 加授权行（§5.3）。
4. ✅ 验证：`node --check` 全绿；`node --test` 34 项全过；`dsh plugin --profile web install` 挂载 link（期间为项目 node runtime 补装 pnpm@10.34.9——DSH 的 profile 安装器需要它，mySims 同款布局）；`dsh --profile web --dump-config` 确认插件行与授权行组合通过。
5. ✅ QA：`sub_root/tests/harness/test_subagent_model_policy.py` 新增 7 项机械断言；`tests.cmd all` 36 项全过（含 `extract_presets --check`）。
6. 归档本计划到 `sub_root/others/plans/archived/`（§7.2 启动冒烟通过后执行）。

> **为什么必须重启才换模块代**：link 安装的 host 插件改源码不会热替换；安装本身热生效，但源码修正要 `dsh web` 重启才进进程（mySimsWS2 实测）。

## 7. 测试与验收

### 7.1 单元测试（`node --test`，不依赖 DSH 运行）

- 判定矩阵：3 plan × 峰/谷；边界时刻、周六、法定节假日、调休上班的周六、未知年份。
- 早切：08:55 + 600s → 峰；反向不早切；`leadtimeSeconds: 0` 关闭。
- guard：显式允许/拒绝、未显式且父=默认、无 `provider` 参数 fail-open、fork 父路由越界、qwen 无 provider、会话 policy 交集为空、配置损坏只拦委派工具。

### 7.2 安装后 live 验收

**启动冒烟（2026-11-16 已通过）**：`dsh web` 成功拉起（http://127.0.0.1:3081），无插件错误输出，profile 依赖树确认 `dsh-subagent-model-policy@link:` 挂载，`pnpm-lock.yaml` 落链；随即关闭。注意：`dsh web` 会自动打开默认浏览器，临时冒烟应加 `--no-open`。

**派发级探针（待项目 DSH 进入日常使用时执行）**：

1. 启动日志出现「子代理模型策略已启用：plan=mimo」（info 级日志目前不上屏，可在会话内派发验证）。
2. 新开顶层会话（带 policy）：`forcePhase: valley` 显式 deepseek 路由放行、mimo-flash 被拒；`forcePhase: peak` 反向各一例。
3. `plan: qwen` fail-loud 文案核对；旧会话 fail-open 核对；主会话模型请求不受影响。

### 7.3 回归

- `node --check` 全绿；`node --test` 全绿。
- `extract_presets.mjs --check` 无漂移；`dsh --profile web --dump-config` 组合树含插件行。
- `sub_root/tests/tests.cmd all`（ruff + pytest + harness 契约）。

## 8. QA 同步

- 本插件是 harness 改动；`sub_root/tests/harness/test_presets.py` 补机械断言：① profile `package.json` 已挂 `dsh-subagent-model-policy` bundle；② profile `cordis.patch.yml` 含 `subagent-model-selection-settings` 授权行且三条白名单齐全；③ `tool-subagent` 补丁使用新 schema 键（`modelSelection`/`modelDiscovery`，无旧键 `modelSelectionSettings`）。
- 跑 `tests.cmd all` 全量回归。
- eval（`sub_root/evals`）不引入子代模型选择的变更，不动。

## 9. 边界与开放项

**边界**：
- `workflow` 的 `agent(...)` 是另一条 spawn 路径，绕过 `subagent` 工具 guard；v1 文档化，不拦。
- `subagent_fork` 不能选模型，只能按父路由放行/拒绝；preset 给 fork 配 `agentOptions` 时 guard 看不到，已知盲区。
- v1 不注入每回合上下文；若观察到「被拒后不重试」，v1.1 再考虑「相位变更时注入一行」。
- 旧会话无 policy 时 fail-open，避免误伤存量会话。

**开放项**：
1. `leadtimeSeconds` 默认 600 是否合适（备选 0 / 3600）。
2. `guardFork` 默认 true 是否过严。
3. qwen 的 provider/model id（占位；provider 落地时补白名单与 `plans.qwen`）。
4. 节假日数据 2028 年发布后重跑 `scripts/refresh-holidays.mjs`。
5. mimo 套餐到期时把 `plan` 改为 `deepseek` 即可（无需改代码/白名单）；续费后改回。

## 10. 实施记录（2026-11-16）

**已完成并验证**

- DSH 升级：`toolchain.json` + `modules/package.json` 0.2.1-rc.5 → 0.2.2-rc.1（npm `latest`；`next` 通道为 rc.3 预发布，不取），`update-deps.cmd update` 安装 302 包变更；`extract_presets.mjs` 重跑；profile 补丁 3 处 `modelSelectionSettings` → `modelSelection`/`modelDiscovery`（备份 `cordis.patch.yml.bak-20261116-pre-0.2.2`）；`--dump-config` 通过。
- 插件包：自 mySimsWS2 复制 18 个文件；适配 3 处——`config.js`/`test/config.test.mjs` 环境变量改 `PYSCI_SUBAGENT_MODEL_POLICY`，`index.js`/`policy.js` 注释指涉改 pySci 路径（计划、blueprint-canon、前身项目表述）。
- 接线：profile `package.json` 加 `link:` 依赖 + bundle 项；`dsh plugin --profile web install` 挂链（为此给项目 node runtime 补装 pnpm@10.34.9，mySims 同款布局；DSH 的 profile 安装器依赖它）；`cordis.patch.yml` 加授权行（三路由全开）。
- 配置点：`sub_root/subagent-model-policy.yaml`，`plan: mimo`。
- 测试：`node --check` 4 文件全绿；`node --test` 34 项全过；`tests.cmd all` 36 项全过（新增 `tests/harness/test_subagent_model_policy.py` 7 项）。
- 启动冒烟：`dsh web` 拉起无错、bundle/lock 确认、随即关闭（§7.2）。

**环境事务（非插件本身）**

- 本会话前段遇到工作区写入被拦（`sub_root` 以下）：经 ACL 诊断修复 7 个目录的「修改权限」缺失（备份与还原命令在 `D:\XXXIIIGGG\projects\pySci\acl-recovery-pySciWS2\`）；后经会话切至完全权限解除限制。诊断报告两份 jsonl 同目录保存。
- `update-deps.cmd update` 尾段 `uv venv` 在 `.venv` 已存在时失败（init-deps.ps1 未带 `--clear` 也不跳过），DSH 安装部分不受影响；属既有缺陷，已报用户待决。

**偏差（相对 mySimsWS2 模板）**

1. 环境变量名 `MYSIMS_SUBAGENT_MODEL_POLICY` → `PYSCI_SUBAGENT_MODEL_POLICY`。
2. 初始配置 `plan: mimo`（mySims 因套餐到期用 `deepseek`）。
3. `node --test` 在完全权限下可直接用；QA 断言里仍按逐文件 `node <file>` 执行（受限沙盒下 `node --test` 的子进程管道会被挡，mySims 实测）。
4. 派发级 live 探针延后（项目 DSH 未进入日常使用），清单固化在 §7.2。

## 11. v1.1 修改记录（2026-11-16，派发级探针后）

**背景**：§7.2 派发级探针实测通过（峰时 deepseek 被拒、mimo 放行，见当日会话）；随后的误伤排查穷举了 harness 全部工具参数声明，发现 v1 的「暴露 `provider` 参数即参与选路」判定误伤两个只读工具：

1. `list_subagent_models`（全部 preset，`modelDiscovery: true`）——无参/只给 provider 的查询在父路由≠默认时被拒，且无法查看本相位允许集之外的模型。
2. `cordis_inspect_query`（仅 pysci-meta preset）——其 `provider` 是 Cordis Inspect Provider ID、无 `model` 参数，v1 判定下几乎永远被拒且文案不可执行，直接卡死 meta 的插件开发流程。

**方案 A（用户 2026-11-16 确认）**：选路判定改为「`provider`+`model` 成对声明」（`declaresRoutePair`，与 `readExplicitRoute` 口径一致），自动豁免语义碰撞类工具；另按稳定名豁免只读发现工具（`READ_ONLY_DISCOVERY_TOOLS`，现仅 `list_subagent_models`）。guard 单测增至 24 项（新增两个误伤回归用例），QA 补机械断言 `test_guard_scope_route_pair_and_discovery_exemption`。

**环境记录**：同日发现受限沙盒下 pytest 9 的 `tmp_path` 全部 EACCES——机制为 CPython 3.13.15 Windows 实现在**创建时**把 `os.mkdir(mode=0o700)` 综合成「仅属主」受保护权限表（继承阻断），受限令牌无法受益于属主权利；`os.mkdir` 默认模式与事后 `os.chmod` 均无害。已在 `sub_root/tests/conftest.py` 加垫片中和创建时 mode，`tests.cmd all` 受限下 37 项全绿（诊断报告：`D:\XXXIIIGGG\projects\pySci\acl-recovery-pySciWS2\acl-report-fd2f008d96934a43bdebd9b91474b6ed.jsonl`，诊断脚本判 NOT_THIS_CLASS——该场景不在其修复类别内，值得作为技能反馈上报）。

## 12. v1.2 修改记录（2026-11-18，用户拍板）

**背景**：`needsExplicitRoute` 原回显「父会话当前路由 X」。主会话常用 pro 级而子代理默认 flash 级，隐式派发恒落入该分支且被回显的恒为 pro 级路由；pro 又在允许集内，模型照抄即合法通过，恰好违背 flash 优先的成本偏好。该句相对 §4 模板本就是实现时追加的解释信息，非文档契约必需。

**改动**（用户 2026-11-18 确认）：

1. `needsExplicitRoute` 删除父路由回显，回归 §4 模板（保留「（默认 X）」提示）；`forkNotAllowed` 保留——那里父路由是被拒对象，语义无歧义。
2. 允许集/授权集列举保证 flash 级排在 pro/max 级之前：`policy.js` 新增 `flashFirst` 稳定排序（按模型名识别档位），`formatAllowed`/`formatPolicy` 应用；GUI 持有的会话授权集顺序不作数。
3. 拒绝文本增补档位选择规则教学行（`TIER_GUIDANCE`，每条路由拒绝均带）：优先 flash；仅当任务确实复杂（较多临场决策、影响深远）才用 pro；拿不准就选 flash；选用 pro 后必须事后向用户显式声明。放行路径不注入（维持零常驻 token 成本）。

**验证**：`node --check` 全绿；插件 node 三套件通过；`tests.cmd all` 全量回归通过（QA 增机械断言 `test_denial_text_contract_v12`）。link 安装需重启 `dsh web` 才生效。
