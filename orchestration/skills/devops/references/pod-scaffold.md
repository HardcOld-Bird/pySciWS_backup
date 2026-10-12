# 新 pod 脚手架：入库面与 settings 模板（backlog 20261012-082932-devops 自 SKILL.md 移出）

## 1. git 跟踪面（每 pod 仅 5 个文件入库）

`.gitignore`、`.qoder/settings.json`、`.qoder/mcp.json`、`.qoder/rules/charter.md`、
`AGENTS.md`。其余皆**不入库**，须在**合并后于 main 上**再生：

| 路径 | 为什么不入库 | 合并后怎么补 |
|---|---|---|
| `bench/` `outbox/` | pod `.gitignore` 排除 | `mkdir` |
| `.qoder/skills/` | 部署副本（gitignored，不随合并搬运） | `pysci-dev sync` |
| `inbox/` | 空目录 git 本就不跟踪 | `mkdir` |

worktree 里建了这三样也不会随 `git merge` 转移到 main——这是「合并后 doctor 报漂移/缺目录」
的常见根因（副本台账机制见 `registry-runtime.md`）。

## 2. settings.json 模板必带两件事

1. **两个 hook 接线**：`Stop`=delivery-gate、`PreToolUse`=pod-guard，均以
   `node ../../guards/*.mjs` 形式指向仓库 guards（相对 pod cwd）；缺一个 → 交付无硬闸或
   只读层无执法；
2. **`"agentsMdExcludes": ["**/leader-only.md"]`**：排除组长专属根规则。缺则组员注入
   `leader-only.md`（越权面）；glob 写宽成 `**/*.md` 会连全员公约数 `basic.md` 一起吞掉——
   doctor 对这两种形态都判 ✗。

**命名坑**：勿把组员自建规则文件命名成 `leader-only*`，会撞上述排除 glob 而静默不注入。

## 3. 零技能 pod 不进 manifest.toml targets

只读层不挂技能的成员（reviewer、twin 先例）：不为其建 `.qoder/skills` 空副本，也不为凑
规格造未被要求的技能真本。该裁定由测试钉住（twin 见 `test_twin_intentionally_has_no_manifest_target`），
manifest 里任何 target 指向的 pod 必须真实存在（同测试的反向护栏）。

## 4. 注册面：无需预写 registry.json

`orch dispatch` 前调 `ensure_member_defaults`，pod 目录存在即生成默认成员条目
（flash / 60 轮 / 7200s / `.qoder/mcp.json`）；`doctor` 按 `PODS_ROOT` 目录计数，也不依赖
registry。脚手架只建目录+只读层，注册留给组长首次派发。
