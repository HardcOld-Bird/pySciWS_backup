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

## 迁移机制（`Registry` 内部，读写方无感）

- 读写方只经 `reg.member(id) → m.sessions → reg.write_member(m) → reg.save()`；会话池
  在 `member()` 从运行态载入、`write_member()` 写回运行态，配置面永不含 `sessions`。
- **首次 `Registry.load()` 自动迁移**：若配置内嵌 sessions 且运行态文件缺席，即抬高
  sessions→运行态、从配置剥离、备份原文到 `registry.presplit.json`（已存在则不覆盖），
  两份原子落盘。第二次 load 天然跳过（幂等）。`save()` 恒**双写**（配置面防御性剥离任何
  残留 sessions + 运行态）。
- `runtime_path`/`backup_path` 随 `Registry.path` 的父目录移动 → 测试 patch 一处
  `REGISTRY_PATH` 即两面隔离。`load()` 显式以模块级 `REGISTRY_PATH` 绑 `path`（非
  dataclass 默认值——后者在类定义时已绑真实路径，无视 monkeypatch）。

## 排查提示

- 见到 `orchestration/state/sessions-runtime.json` 出现在 `git status` → 不对，它应被
  gitignore；检查 ignore 规则是否生效。
- 改 sessions 读写逻辑后须同步 `tests/skills/orchestration/conftest.py` 的 `iso_dispatch`
  （预置运行态文件）与 `read_runtime`/`write_runtime`——est_tokens 系列断言读的是运行态。
