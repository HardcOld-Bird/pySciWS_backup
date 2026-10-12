# registry 配置面 vs 运行态（backlog 20261010-registry-runtime-split）

编排状态**分储两文件**，写代码/排查 dirty 前先认清哪份是谁的落点。

## 两份文件

- `orchestration/state/registry.json`（**git 跟踪**）= 低频配置面：成员 → pod 路径、
  model_tier/max_turns/timeout_s/mcp_config/effort/extra_dirs/readonly_extra、exe、
  models、checks、est_tokens_per_char、model_patterns。人手编辑、随代码评审演进。
- `orchestration/state/sessions-runtime.json`（**gitignore**）= 高频运行态：会话池
  `members[*].sessions`（sid/name/hops/last_active/**chars_offset**/status）。每次派发
  一跳都改写它。

## 为什么分储（同构合并阻塞的根治）

sessions/hops/chars_offset 是**心跳**——每派发必变。留在 registry.json 里，main 工作区
的该文件就**永远 dirty**；任何改动 registry.json 的 worktree 交付，merge 前的 `git status`
检查（「main 有会被本次合并触碰的未提交改动 → 交付 blocked」）必被心跳拦下。163347 与
220711 是同一形状的两笔阻塞，故把运行态迁到 gitignore 文件，registry.json 只承载低频
配置——碰它的交付不再被心跳 dirty 拦。

## 读写契约（`Registry` 内部，读写方照此调用）

- 记账一跳：`reg.member(id) → m.sessions → reg.write_sessions(m) → reg.save()`。
  `write_sessions()` **只写运行态**并声明「本进程拥有该成员的会话池」；它不再回写配置面
  （旧名 `write_member` 会顺手把内存 Member 的 model_tier/pod/参数模板抄进
  registry.json——见下节事故）。
- 机器改配置面**只剩声明这一条通路**：`reg.declare_config(checks={...})`
  （改顶层键，值=改动子集）与 `ensure_member_defaults()`（新增成员，仅当磁盘也无该成员才落）。
  档位映射自愈**不走这条路**，见下节。
- **`save()` 是读-改-写按键合并，不是整体回写**：落盘前重读两份文件，配置面以磁盘为
  权威——无声明则一个字节不写；有声明只把声明的键叠加上去。运行态只动声明过的成员，
  其会话池按 `sid` 合并（同 sid 本进程为准，磁盘独有 sid 保留）。
- **首次 `Registry.load()` 自动迁移**：若配置内嵌 sessions 且运行态文件缺席，即抬高
  sessions→运行态、从配置剥离、备份原文到 `registry.presplit.json`（已存在则不覆盖），
  两份原子落盘。第二次 load 天然跳过（幂等）；这是唯一被许可的全量写（数据即本次 load
  刚读的磁盘内容，窗口毫秒级）。
- `runtime_path`/`backup_path` 随 `Registry.path` 的父目录移动 → 测试 patch 一处
  `REGISTRY_PATH` 即两面隔离。`load()` 显式以模块级 `REGISTRY_PATH` 绑 `path`（非
  dataclass 默认值——后者在类定义时已绑真实路径，无视 monkeypatch）。

## 为什么配置面必须磁盘权威（backlog 20261011-220621-leader）

分储只搬走了心跳，没搬走「整体回写内存副本」这个坏习惯。实测首战：21:05:57 drain worker
启动并 `Registry.load()`（deputy/reviewer=max）→ 21:42:41 E0-C 批次合并降档 flash → 22:04
`git diff HEAD -- registry.json` 显示工作树**回退为 max、tier_note 消失**。长驻进程收尾时
把启动快照整个 `save()` 回去，后写覆盖——用户已裁决的配置改动在运行态被无声撤销。

护栏形状：**任何进程都不得用陈旧全量副本覆盖 state 配置面**。故 save 前必重读磁盘、
只覆盖本进程负责的键；纯记账的派发路径根本不该产生配置文件写入（`git diff HEAD` 应为空
是 drain run 的事后判据）。

## 档位映射：三级回退与运行态面（backlog 20261012-080410-devops）

`models`（档位 → 具体型号）里混着两种来源：人写的覆盖与机器从 CLI 目录解析出的自愈值
（`refresh_models`，drain 启动时跑一次）。自愈值若写进跟踪的 `registry.json`，每轮 drain
都留下 ` M registry.json`——把「碰 registry 的交付」重新堵成心跳那样的周期墙；且
`models` 按 charter 属护栏级，机器静默改写本身就越权。故：

- **读取三级回退**：`Registry.models` = 出厂默认 ← 运行态 `models_resolved` ← 跟踪
  `data["models"]`，**跟踪值优先**（含 `""`——空值是人的裁决「该档不指定，交 CLI 默认」，
  不是待填的空位）；
- **机器只写运行态**：`merge_runtime_models()` 落 `sessions-runtime.json` 的
  `models_resolved` 键（gitignored），跟踪面一个字节不碰；`refresh_models()` 里
  `save()`/`declare_config()` 全不存在；
- **遮蔽要出声**：跟踪值与解析值不一致时该档记入 `res["shadowed"]`（人不写则不换），
  drain 日志与 `orch refresh-models` 都打印——否则「为什么我刷新了却没换档」只能靠猜。
  要换档就改 `registry.json` 的 `models.<档>`（护栏级，经用户授权），或删掉该键让机器接管。

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
- **(a) 运行态缺席警告**：`Registry.load()` 时若配置面有成员却无内嵌 sessions、且
  运行态文件不存在，即向 stderr 打显著警告（不自动反推数据）。典型触发=刚 `--no-ff`
  合并过含运行态改动的分支却没搬运行态文件（`--no-ff` 只带 git 面）。解法：用
  `pysci-dev worktree merge <id>`（机械化 cp + merge + 会话数不减自检）；首次安装（无成员）
  与本就在盘的正常态都不响。
- 改 sessions 读写逻辑后须同步 `tests/skills/orchestration/conftest.py` 的 `iso_dispatch`
  （预置运行态文件）与 `read_runtime`/`write_runtime`——est_tokens 系列断言读的是运行态。
- 默认夹具把 `Registry.save` no-op 掉以防污染真实 state；要断言落盘（含按键合并语义）
  用 `enable_registry_save(monkeypatch)` 还原真实 save。
