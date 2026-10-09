---
name: orchestration
description: 组长专属的多 Agent 编排操作知识：经 pysci-orch 派发/续作组员任务、管理会话池、读取台账与统计、审批改进建议的规程与纪律。当用户提出任何需要末端生产的工作（仿真/绘图/写作/理论/文献/AI 作画/3D 建模）而需要派发给组员时，或询问编排体系状态时，使用本技能。组长自己严禁执行末端工作。
---

# 编排操作规程（组长技能 v1，Phase 1 范围）

权威设计：`orchestration/README.md`。本技能只放**操作面**；架构疑问读 README。

## 铁律

1. **严禁亲自执行末端工作**（生产/编辑代码/跑仿真绘图等），一律经 `pysci-orch`
   派发。唯一例外：用户明确要求你亲做。
2. **长任务必须后台运行**：`dispatch` 用 Bash 工具的 `run_in_background` 启动，
   完成通知自动唤醒你；等待期间**禁止 sleep-轮询循环**、禁止反复读输出文件。
3. 派发前**不需要**记忆任何 sid/型号/参数细节——orch 全部代管并在回复中给下一步。
4. 对 orch 输出的 `[NEXT]` 段照做即可；需要决策时它会列出选项。

## 命令面（Phase 1 已实装）

```
uv run pysci-orch status                    # 全景：成员/会话池/台账近况/队列
uv run pysci-orch dispatch <member> --task <file>      # 任务书文件派发
uv run pysci-orch dispatch <member> --text "<任务>"    # 正文直接派发
        [--session latest|new|<sid前缀>]    # 默认 latest=续最近活跃会话
        [--name <slug>] [--model-tier max|flash] [--dirs d1,d2]
        [--max-turns N] [--timeout S] [--no-checks]
uv run pysci-orch sessions <member> [--all] [--archive <sid> [--distill]] [--prune <sid>]
uv run pysci-orch ledger [--member M] [--days N] [--stats]
uv run pysci-orch sync [--check]            # 技能真本→部署副本
```

成员 id：`sim` `figure` `writing` `theory` `lit` `drawing` `model3d`（专职×7）；
`deputy` `reviewer` `devops`（Phase 2/3 接入）。当前已就绪的 pod 以 `orch status` 为准。

## 派发要领

- **任务书自包含**：目标、输入路径（绝对或项目相对）、交付要求、白名单目录
  （`--dirs`，注入 pod-guard）、验收预期。组员看不到你的会话上下文——一切经任务书传递；
- **会话选择**：同一工作的后续跳用默认 `latest`；全新且无关的工作用 `--session new`
  并给 `--name`；要续更早的特定会话先看 `orch sessions <member>` 再显式指定；
- **白名单最小化**：只给任务需要的 `data/research/...` 目录；
- **模型档位**：默认 flash（专职组员）；任务复杂/组员连续 blocked 时升 `--model-tier max`；
- 交接物：把上一环节产物路径写进下一环节任务书（Phase 2 起 plan 系统自动注入）。

## 交付判读

- `<result>` = 成功；`<blocked>` = 失败（含卡点描述）——orch 已解析并给出 [NEXT]；
- 机械验收 FAIL → 按 [NEXT] 回派返工（同会话），同指纹连败 3 次停止回派、升级决策；
- `<infra_suggestion>` 非空 → 审批（Phase 2 前手动）：
  值得采纳 → 编辑 `orchestration/state/backlog.json` 追加
  `{"id": "<ts>", "member": "<id>", "summary": "...", "status": "pending"}`；
  无论采纳与否，把回复写到 `orchestration/state/replies/<member>/<id>.md`
  （下次派发自动附送给组员，闭环告知）；
- `run_failed`（运行级失败）→ 按 [NEXT] 三选一（再试/查 jsonl/上报用户）。

## 归档与成本

- 一项工作彻底完结且会话不再需要 → `orch sessions <member> --archive <sid> --distill`
  （先派蒸馏跳写 AGENTS.md 再归档）；
- `orch ledger --stats` 看成员耗时/credits 占比；ctx 占用高（>60%）的会话提示归档。

## 降级

orch 瘫痪 → 照 `orchestration/README.md` 附录 A 手动模板派发（原生 exe 直调，
**勿用 .cmd 包装器**——剥引号），台账手工补记，并尽快派 devops 修复（Phase 2 前
向用户报告）。
