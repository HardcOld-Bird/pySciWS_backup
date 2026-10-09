# devops 组员长期记忆

> 本文件由 devops 组员自维护（每次会话启动自动加载）。只记稳定事实与约定。

## 身份

- pySci 编排体系 devops 组员（基础设施维护专员），charter 见 `.qoder/rules/charter.md`。
- 专属工具：`pysci-dev`（doctor/sync/worktree）；实现于 `src/pysci/skills/devops/`。
- 作业方式：worktree 隔离 + cz commit + 禁 push。

## 经验（工作中积累）

- **worktree 内 `pysci-dev sync` 会污染 `skills-deployed.json`（重要）**：本机
  `core.autocrlf=true` 且无 `.gitattributes`，`git worktree add` 把全部文本真本 smudge 成
  CRLF，而 main 工作区是**混合行尾**（如 `document-writing/references/digest.md` 在 main 是
  LF、SKILL.md 是 CRLF）。`sync.tree_hash` 读**原始字节**，故含 LF 文件的技能在 worktree 与
  main 算出不同源哈希。后果：worktree sync 写进提交的哈希对 main 是错的，合并后 main 上
  `sync --check` 会报漂移。**正解**：worktree 内 sync 仅供 doctor 自检；`skills-deployed.json`
  的最终值必须在**合并回 main 后、在 main 上重跑 `pysci-dev sync`** 生成再提交（main 磁盘真值）。
  `doctor` 不受影响——它活算 source vs 部署副本哈希、不读该 json。
- **新增 pod 无需预注册 registry.json**：`orch dispatch` 前调 `ensure_member_defaults`，pod 目录
  存在即自动生成默认成员条目（flash/60轮/7200s/`.qoder/mcp.json`）。`doctor` 按 `PODS_ROOT`
  目录计数、也不依赖 registry。故脚手架新 pod 只建目录+只读层文件，注册留给组长首次派发。
- **pod 脚手架的 git 跟踪面**：每 pod 仅 5 个文件入库（`.gitignore`/`.qoder/settings.json`/
  `.qoder/mcp.json`/`.qoder/rules/charter.md`/`AGENTS.md`）；`bench/`·`outbox/`·`.qoder/skills/`
  被 pod `.gitignore` 排除，`inbox/` 空目录 git 不跟踪——三者均须在**合并后于 main 上**用
  `mkdir` + `sync` 再生（worktree 内建了也不随合并 transfer）。
