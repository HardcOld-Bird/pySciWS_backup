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
- **标题框宽估算器偏窄（2026-10-18 实测）**：`compose slides geometry` 的
  `estimate_title_width_in` 只估**文字宽**，未计文本框默认左右内边距（合计 0.2 in）与
  `spc` 字距（页 3 为 300＝3 pt）。长拉丁标题（如 `·AI COMSOL·`）按 3 pt 字距实测需
  2.39 in，而估算仅 2.03 in；严守 ±10% 必折行。WPS 对 `Heiti SC Light` 的回退字体是
  Microsoft YaHei（可用 PDF 的 `fitz` 提取 span/字符 origin 精测）。二选一：超容差 hug 或
  微调字距；本轮取字距 3→1.5 pt 并留文字右余量 ~0.12 in。根本解法是修估算器（计入
  insets 与 spc、按回退字体标定拉丁系数）。
- **旧 W2「跨页标题一致性」与自适应框宽互斥**：其格式指纹含框宽 `cx`，要求三页同宽；
  在页 3 固定 1.39 in、页 2/6 各自适配时永远报 2 项「不一致：宽」。geometry 因此无法
  真正「零报」；交付时明确分解为「预设三项 0 报 + W2 旧闸 2 项（机制矛盾）」并提
  infra_suggestion，勿改 `src/` 绕过。
