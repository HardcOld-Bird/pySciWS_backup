# 任务书：审批口 success_criteria 形态解析修复（gate 漏读列表项形态）

## 缺陷（组长实测 2026-10-15）
- 复现：`pysci-orch approve 20261015-130838-writing`（建议文件
  `orchestration/state/suggestions/20261015-130838-writing.md` 里判据写在
  **列表项**行 `- success_criteria: …`）→ 被判「缺可证伪成功判据，拒绝入队」。
- 实际：文件里判据完整且可证伪；只是行形态是 `- success_criteria:`（列表项前缀），
  解析器只认行首裸 `success_criteria`，整条漏读 → 误拒。
- 影响：真实判据被当无判据拒绝，组长被迫 `--criteria` 手工补录绕过；若没细看会把
  好建议误毙。

## 修复要求
1. 判据解析接受常见书写形态：行首 `success_criteria:`、列表项
   `- success_criteria:`、`**success_criteria**:`（粗体）等合理变体（以
   建议文件实际书写习惯为准，先 grep state/suggestions/ 全量看形态分布再定）；
2. 误拒报错文案列出**接受的书写形态**与最小示例（让下一个人 10 秒自救）；
3. 同族顺检：`kind`/`source` 等元字段的解析是否存在同类「前缀即漏读」。

## 验收预期
1. 对 `20261015-130838-writing.md` 的原件形态重放解析 → 判据被正确读出（不再误拒）；
2. 三种形态（行首/列表项/粗体）各有回归用例；
3. 全量 pytest 绿。

## 白名单（写权限）
- src/pysci/skills/orchestration/tools/
- tests/
