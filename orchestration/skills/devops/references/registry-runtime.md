# registry 配置面 vs 运行态（backlog 20261010-registry-runtime-split）

编排状态**分储两文件**，写代码/排查 dirty 前先认清哪份是谁的落点。

## 两份文件

- `orchestration/state/registry.json`（**git 跟踪**）= 低频配置面：成员 → pod 路径、
  max_turns/timeout_s/mcp_config/effort/extra_dirs/readonly_extra、exe、**presets**
  （预设 → main/pro 席位链，账号级 UUID 整链覆盖）、models/model_chains（只余独立的
  minimum）、model_fallback（粘滞窗参数）、jury（仲裁席定义）、checks、
  est_tokens_per_char。人手编辑、随代码评审演进；**模型档位 model_tier/max|flash 已退役**
  （2026-10-12 22:4x），预设名在 `data/用户的inbox/pysci_settings.toml`。
- `orchestration/state/sessions-runtime.json`（**gitignore**）= 高频运行态：会话池
  `members[*].sessions`（sid/name/hops/last_active/**chars_offset**/status），加上席位
  面三键 `models_resolved` / `model_seat_names` / `model_seat_fail`（见下两节）。每次派发
  一跳都改写它。

## 为什么分储（同构合并阻塞的根治）

sessions/hops/chars_offset 是**心跳**——每派发必变。留在跟踪的 registry.json 里，main
工作区就**永远 dirty**，任何碰它的交付都被合并前的交集检查（「main 有会被本次合并触碰的
未提交改动 → blocked」）拦下（163347/220711 两笔同形阻塞）。迁走后 registry.json 只承载
低频配置，碰它不再被心跳拦。

## 读写契约（`Registry` 内部，读写方照此调用）

- 记账一跳：`reg.member(id) → m.sessions → reg.write_sessions(m) → reg.save()`。
  `write_sessions()` **只写运行态**并声明「本进程拥有该成员的会话池」，不回写配置面
  （旧名 `write_member` 会顺手把内存 Member 的参数模板抄进 registry.json）。
- 机器改配置面**只剩声明这一条通路**：`reg.declare_config(checks={...})`
  （改顶层键，值=改动子集）与 `ensure_member_defaults()`（新增成员，仅当磁盘也无该成员才落）。
  档位映射自愈**不走这条路**，见下节。
- **`save()` 是读-改-写按键合并，不是整体回写**：落盘前重读两份文件，配置面以磁盘为
  权威——无声明则一个字节不写；有声明只把声明的键叠加上去。运行态只动声明过的成员，
  其会话池按 `sid` 合并（同 sid 本进程为准，磁盘独有 sid 保留）。
- **首次 `Registry.load()` 自动迁移**：配置内嵌 sessions 且运行态缺席时，抬高 sessions→
  运行态、剥离配置、原文备份到 `registry.presplit.json`（已存在不覆盖），两份原子落盘；
  幂等，且这是唯一被许可的全量写。
- `runtime_path`/`backup_path` 随 `Registry.path` 的父目录移动 → 测试 patch 一处
  `REGISTRY_PATH` 即两面隔离。`load()` 显式以模块级 `REGISTRY_PATH` 绑 `path`（非
  dataclass 默认值——后者在类定义时已绑真实路径，无视 monkeypatch）。

## 为什么配置面必须磁盘权威（backlog 20261011-220621-leader）

分储搬走了心跳，没搬走「整体回写内存副本」这个坏习惯。实发：drain worker 启动即
`Registry.load()`，期间 E0-C 合并把 deputy/reviewer 降档 flash，`git diff HEAD` 却显示
工作树**回退为 max、tier_note 消失**——长驻进程收尾把启动快照整个 `save()` 回去，后写
覆盖，用户已裁决的配置被无声撤销。

护栏形状：**任何进程都不得用陈旧全量副本覆盖 state 配置面**。故 save 前必重读磁盘、
只覆盖本进程负责的键；纯记账的派发路径根本不该产生配置文件写入（`git diff HEAD` 应为空
是 drain run 的事后判据）。

## 预设与角色：席位表三级读（080410-devops 方案 A 沿用 / 225500-leader 改键）

`main`/`pro` 的席位链**随预设**（`data/用户的inbox/pysci_settings.toml` `[model] preset` 选，见
`presets.py`）；`minimum` 独立不随预设。机器自愈只服务**人没写值**的键——故：

- **读取三级回退**：`Registry.model_chain(角色)` = 跟踪 `presets.<预设>.<role>`（整链覆盖）
  ← 出厂 `presets.PRESETS`；`minimum` 走 跟踪 `models` ← 运行态 `models_resolved` ←
  出厂 `DEFAULT_MODEL_CHAINS`。含 `""` 的席是人的裁决「该席不传 `-m`」，不是待填的空位；
  无 `presets` 块的旧部署读 `models.flash/max` 兼容（main←flash、pro←max）；
- **机器只写运行态**：`merge_runtime_models()` 落 `models_resolved`（值=整条链），跟踪面
  一字节不碰；`refresh_models()` 里不存在 `save()`/`declare_config()`。同文件另两键：
  `model_seat_names`（席 → 目录显示名，**presets 块的 UUID 也在这反查**，告警报人名）、
  `model_seat_fail`（降级粘滞窗，见下节）；
- **合并粒度=整条链**：机器不得往人排的次序里插席——逐席不等就整条遮蔽；某档真名**一个都
  没**命中目录时该档**保持现值**并记 `missing`（那不是解析结果，写成运行态等于替人裁决，
  且每轮 drain 无意义 churn）；
- **遮蔽要出声**：跟踪值与解析值不一致记 `res["shadowed"]`，drain 日志与
  `orch refresh-models` 都打印——否则「刷新了却没换档」只能靠猜。要换席改跟踪面
  `presets`/`models`（护栏级，经用户授权），或删该键让机器接管。

## 降级粘滞窗与事件流

坏席不会自己变好（BYOK 余额暂停充值），逐跳探测＝逐跳白烧。某席系统性失败即记失败时刻
写进 `model_seat_fail`（窗 = `model_fallback.skip_window_s`，默认 3600s；成功即
`note_seat_success` 清窗；全链都在窗内时强探**失败最早**=最先到期的一席，不停摆）。
降级/跳过/全链耗尽各追加一行到 `state/channel-events.jsonl`（**gitignore**，周期墙理由
同本文件开头）：mirror `status` 的「回退」项与 `verdict` 健康面读它算统计，永久审计不靠
它。窗内零探测但仍打「席 X 跳过中…窗口内不重试」（省跳数不能换成安静）；预设不可激活
（deepseek 过渡期）时 dispatch 拒发、drain 拒绝开工。告警文案、全局停止、台账 `model` 记
**实际所用席位**等行为面见 orchestration 技能 `drain-watch.md`。

## 脏跟踪面哨兵（同上）

`registry.dirty_tracked_state()` = `git status --porcelain --untracked-files=no --
orchestration/state`，返回相对路径列表；非仓库/git 缺失/超时/race≠0 一律 `[]`（fail-open，
哨兵失灵不该影响主流程）。消费面两处，都是 **advisory（不计失败）**：`orch status` 的
「跟踪 state 脏面」段 + `[NEXT]` 提示，`pysci-dev doctor` 的同名段。脏跟踪 state 常常正是
组长在飞的改动或本单自己要提交的文件，硬失败会把正常作业挡成事故；真正要盯的信号是
「无人操作却自己变脏」——那说明某个自愈路径又在写跟踪面了。

## 排查提示

- 见到 `orchestration/state/sessions-runtime.json` 出现在 `git status` → 不对，它应被
  gitignore；检查 ignore 规则是否生效。
- **运行态缺席警告**：配置面有成员却无内嵌 sessions、且运行态文件不存在 → stderr 显著
  警告（不反推数据）。典型触发=`--no-ff` 合并后没搬运行态；解法用 `pysci-dev worktree
  merge <id>`（cp + merge + 会话数不减自检）。首次安装与正常态都不响。
- 改 sessions 读写逻辑后须同步 `tests/skills/orchestration/conftest.py` 的 `iso_dispatch`
  （预置运行态文件）与 `read_runtime`/`write_runtime`——est_tokens 系列断言读的是运行态。
- 默认夹具把 `Registry.save` no-op 掉以防污染真实 state；要断言落盘（含按键合并语义）
  用 `enable_registry_save(monkeypatch)` 还原真实 save。
