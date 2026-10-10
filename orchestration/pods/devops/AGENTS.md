# devops 组员长期记忆

> 本文件由 devops 组员自维护（每次会话启动自动加载）。只记稳定事实与约定。

## 身份

- pySci 编排体系 devops 组员（基础设施维护专员），charter 见 `.qoder/rules/charter.md`。
- 专属工具：`pysci-dev`（doctor/sync/worktree）；实现于 `src/pysci/skills/devops/`。
- 作业方式：worktree 隔离 + cz commit + 禁 push。

## 经验（工作中积累）

- **worktree sync 与 `skills-deployed.json`（已修复 2026-10-10，backlog 20261010-003127-devops）**：
  旧坑——本机 `core.autocrlf=true` 无 `.gitattributes`，worktree 检出把文本真本 smudge 成 CRLF、
  main 是混合行尾，而 `sync.tree_hash` 读**原始字节**，致同一真本两侧哈希不同、合并后 `sync --check`
  报假漂移。**修复（方案 b）**：`tree_hash` 哈希前规范化 CRLF→LF（技能真本全为 .md/.toml 文本、无
  二进制，安全），哈希只反映内容。实测：document-writing 旧 raw-byte 哈希 main=dbac3b85/wtree=f97e5e09
  （漂移，复现建议原值），规范化后两侧均 e43ec523；九技能全 match。**现状**：worktree 内 sync 提交的
  `skills-deployed.json` 对 main 即正确，**无需**再「合并后在 main 重跑」；doctor 已增「部署台账巡检
  （sync --check）」哨兵，捕获记录哈希漂移（[*]）与副本漂移（[!]）。未选方案 a（根 .gitattributes）：
  需仓库级 renormalize、触碰每个文本文件且与用户本地 autocrlf 交互，blast radius 大。
- **新增 pod 无需预注册 registry.json**：`orch dispatch` 前调 `ensure_member_defaults`，pod 目录
  存在即自动生成默认成员条目（flash/60轮/7200s/`.qoder/mcp.json`）。`doctor` 按 `PODS_ROOT`
  目录计数、也不依赖 registry。故脚手架新 pod 只建目录+只读层文件，注册留给组长首次派发。
- **pod 脚手架的 git 跟踪面**：每 pod 仅 5 个文件入库（`.gitignore`/`.qoder/settings.json`/
  `.qoder/mcp.json`/`.qoder/rules/charter.md`/`AGENTS.md`）；`bench/`·`outbox/`·`.qoder/skills/`
  被 pod `.gitignore` 排除，`inbox/` 空目录 git 不跟踪——三者均须在**合并后于 main 上**用
  `mkdir` + `sync` 再生（worktree 内建了也不随合并 transfer）。
