targets: src/pysci/skills/orchestration/tools/drain.py; src/pysci/skills/orchestration/tools/benchd.py; src/pysci/skills/orchestration/tools/inspectd.py

# 案卷 e3live-disconnect-glm（E3-live 阳性用例：人为断开 glm 主席 → 回退席上场必须三面告警；正文取真实争议案 20261012-174114-devops）

## 问题

是否采纳这条基建改进提案并派 devops 实施？提案摘要：worker 派生时 stdout 与 run 日志
同文件双写者，任何 print 从偏移 0 覆写整段运行日志；建议派生侧收口（stdout/stderr →
DEVNULL 或分文件），三条派生路径同形。

## 选项 A

采纳（kind=defect）。提案全文：

**现象**：headless worker 的 stdout 被重定向进本 run 的 `.log`，而 `_log()` 也写同一文件。
任何一次 `print()` 都从偏移 0 **覆写整段运行日志**——让路/闸门/派跳过程当场全灭，且 mock
测不出（`_spawn_detached` 被桩时 stdout 不参与）。2026-10-12 实测引爆于 `cmd_inspect_worker`
的收尾 `print(summary)`，run 日志只剩摘要 JSON（已修：commit `0463ee2`，删 print + 加
stdout 静默回归）。

**为什么模式本身仍脆弱**：修复只消除了本次的 print，没有消除「一个文件两个写者、其中一个用
C 语义从 0 覆写」的结构。benchd 与 drain 的 worker 同形（`_spawn_detached` 重定向
stdout→run log），下一个 `print` 只是还没被写出来。建议派生侧收口：stdout/stderr → DEVNULL
（或 `<run>.out` 分文件），日志只由 `_log()` 以追加语义写；再钉一条断言「worker 进程 stdout
不是 run 日志 fd」。这属结构裁决、影响三条派生路径，故提案而非本跳擅自改。

## 选项 B

暂不采纳：维持现状——这套摩擦继续由组长和组员的记忆与文档吸收，不占 backlog、不派 devops、
不增加设施复杂度。已落地的局部修复（删掉那次 print + stdout 静默回归测试）覆盖了当前已知
引爆点；结构性收口要动 drain/benchd/inspectd 三条派生路径，收益是消除一类尚未再次发生的
覆写，代价是三条路径的派生语义变更本身也可能引入新的运行态问题。

## 证据

成功判据（做完后可机械验证的条款）：worker 派生时 stdout 不再指向运行日志文件（改 DEVNULL
或分文件），且有一条阳性测试证明「worker 内任何 print 都无法破坏 run 日志」；benchd/inspectd/drain
三条派生路径同形。
