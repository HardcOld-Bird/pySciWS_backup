# 强制层（README §9 全量）

> 由 `orchestration/README.md §9` 迁入（细节块；**唯一权威仍是 README §9**）。

## 9. 强制层

- **pod-guard**（PreToolUse `Write|Edit|NotebookEdit`）：按 §2 矩阵两层执法（自维护层放行／只读层与
  pod 外 exit 2 + stderr 注入）；**不删除成员自建技能**。
- **delivery-gate**（Stop hook）：§3.4 格式强制 ＋ 自维护层**预算硬闸**（AGENTS.md ≤8192B；自建 rules
  禁 `always_on`；自建 rules/技能 description 合计各 ≤8192B；自建技能 SKILL.md ≤8192B）。
- **harness-budget**（pre-commit 真本侧）：`orchestration/skills/**` 的 SKILL.md/references 与根
  `.qoder/rules/` 每文件 ≤8192B 硬拦；`pysci-dev doctor` 三档巡检。
- **权限模式**：用户级默认 yolo（用户裁决）；hooks 在 yolo 下照常执行。
- **诚实条款**：guard 只拦 Write/Edit/NotebookEdit（Bash 重定向理论可绕）——目标是把失效模式从
  「遗忘/失误」升级为「刻意违规」，由 transcript 审计（devops 巡检）发现。
- **恢复面三闸**：WIP-salvage 自动抢救（回收孤儿前先快照留档+开续作单）／sid 在途闸（判活+argv 双闸
  防双写）／CHANNEL-STOP（渠道故障合取判据 → drain 全局停+出声）；交集成因停链（`merge_dirty_face`
  → 保进度、停链待清面）。细则见技能 `references/tier-a-tags.md`。
