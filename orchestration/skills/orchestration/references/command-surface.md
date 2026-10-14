# pysci-orch 命令全表（技能 SKILL.md 的展开）

SKILL.md「命令面（速查）」只给骨干；本文件是逐行语义与旗标。日常敲命令前先扫这一段。

```
pysci-orch status                        # 全景：计划/成员/队列/worker/台账
# —— 日常面 ——
pysci-orch plan new <计划.yaml> [--id X]  # 登记计划（格式见 README §4.2）
pysci-orch plan run <id> --text "<当前环节任务书>"   # 推进（交接处暂停）
pysci-orch plan advance <id> [--text/--task]  # 事件驱动：入队即返回（M5，不阻塞）
pysci-orch taskbooks list|show <id>|link <id> --plan <pid>|sync  # 用户大任务书
pysci-orch queues list | queues enqueue <member> --text "..." [--review]  # 成员 FIFO 队列
pysci-orch events drain | events list  # 统一事件流：排空积压（幂等）/水位
pysci-orch plan amend <id>               # 编辑 plan YAML 后校验对账
pysci-orch plan adhoc <member> --text "..." [--dirs d1,d2]  # 计划外单步
pysci-orch plan list | plan drop <id>
pysci-orch suggestions                   # 待审批改进建议
pysci-orch approve <建议id> (--jury | --bench <案id[,案id]> | --override "<理由>")
        [--note "..."] [--no-wake] [--criteria "<判据>"] [--solo]  # 四态必居其一
        # 例外=缺陷×该跳 blocked。判决包 verdict-package.md；大单 solo-batching.md
pysci-orch watch [--timeout 14400]
        # 等 drain 事件（drain-watch.md §2）；后台挂起，漏挂由 status 补账
pysci-orch reject <建议id> --override "<判决依据>"   # 否决：依据必填、留痕（J/M 判决的驳回不计）
pysci-orch consult "<问题>"               # 咨询副组长（平级：异议义务已注入）
pysci-orch consult-twin <案卷.md>  # 咨询 twin（未启用即拒 rc=2）
pysci-orch review <产物> --origin <member> [--rubric generic]
        [--goal-from <原生产任务书> | --goal "<目标陈述>"]  # --goal* 供 reviewer G2 独立核验
        # 审查链：VERDICT → FAIL 回派返工 → 复审一次 → 二次 FAIL 升级仲裁
pysci-orch jury <案卷.md> [--dry|--timeout S]  # Tier C 仲裁团（非 Qwen 双序对局，阻塞出裁决；
        # 不可达入队 rc=1；jury-panel.md/jury-channels.md）
pysci-orch eval run (--case <id>|--all) [--label L]
        # Tier B 压测**抽检面**（runner=leader）；套件本体归 bencher
pysci-orch bench-add <run|capture|ab|check> [--cases a,b] [--ref-a R --ref-b R]
        [--label L] [--tier minimum] [--brief T] [--start]  # 测量任务入队（不阻塞）
pysci-orch bench-up [--dry] | bench-queue [--all] | bench-seen <id> | bench-stop
        # bencher 生命周期（FIFO 非常驻）；判决进 status「待查看」不即时唤醒（§5.1-B）
        # 要领/案卷格式 drain-watch.md、harness-eval.md
pysci-orch taste <案卷.md> [--dry]  # twin 判断通道（§9 冷启动+宪章注入；未生效即拒，twin-taste-mirror.md）
pysci-orch inspector-up [--dry|--force] | inspector-status | inspector-patrol [--target 菜单id]
        [--start] | inspector-assign --brief T [--start] | inspector-on|off
        # 巡检引擎（§7）：拉起即内嵌 status/approve/dispatch（幂等；旋钮在代码/registry）。见
        # references/inspector-engine.md
pysci-orch stats [--plan id|--member m|--days N]
# —— 底层命令面（降级/调试）——
pysci-orch dispatch <member> (--task F|--text T|--from-backlog) [--session latest|new|<sid>] ...
pysci-orch sessions <member> [--all|--archive <sid> [--distill]|--prune <sid>]
pysci-orch ledger [--member --days --stats] | sync [--check]
```

成员 id：sim figure writing theory lit drawing model3d（专职）；deputy reviewer devops
bencher twin inspector（特殊）。就绪 pod 以 `orch status` 为准。

用户控制台 `pysci`：裸跑进 TUI（人类正道）；`pysci status|journal|decisions`＝LLM 静态通道。

**四态审批**：`approve` 必带 `--jury`/`--bench`/`--override` 之一；例外通道=缺陷×当跳
blocked。判决包结构与用量/override 率口径见 `verdict-package.md`。
