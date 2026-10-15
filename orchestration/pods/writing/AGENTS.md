# writing 组员长期记忆

> 本文件由 writing 组员自维护（每次会话启动自动加载）。只记「下次会话还要知道的
> 事实与约定」：LaTeX 编译链经验、各研究线行文/模板偏好、refs.bib 踩过的坑。保持精简；
> 主题多了以后可拆分到 `.qoder/rules/` 自建规则（model_decision/glob 触发按需加载）。

## 身份

- pySci 编排体系 writing 组员（科技文档写作专员），charter 见 `.qoder/rules/charter.md`。
- 工作区：`bench/` 草稿、`outbox/` 交付副本、正式产物落任务书白名单目录。
- 专业知识：部署技能 document-writing（Skill 工具调用）；CLI `pysci-compose`。

## 经验

- **白名单未注入时 pod-guard 会拦白名单目录（2026-10-15 实测）**：任务书 `## 白名单（写权限）`
  段是**声明面**，**不会自动注入** pod-guard 的 `PYSCI_TASK_DIRS`——后者只来自派发 `--dirs` /
  plan step `dirs`。`dirs` 为空时，对任务书白名单目录的 `Write/Edit` 会被拦（`EXIT=2`）。
  排查：`echo "$PYSCI_TASK_DIRS"`、看 `orchestration/state/queues/<member>.json` 的 `"dirs"`。
  **处置**：勿 Bash 绕行（README §9 定性为刻意违规）；把成品留 `bench/`，回 `<blocked
  tags="whitelist_conflict" reason="blocked_by_env">` 并提 `<infra_suggestion kind="defect">`，
  由组长补 `--dirs` 重派或 `cp` 代写。重派带 `--dirs` 后即可原地 `Write` 覆盖。
