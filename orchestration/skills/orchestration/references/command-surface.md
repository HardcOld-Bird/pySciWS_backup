# pysci-orch 命令全表（技能 SKILL.md 的展开）

SKILL.md「命令面（速查）」只给骨干；本文件是逐行语义与旗标。

```
pysci-orch status                        # 全景：计划/成员/队列/worker/台账/夜航
# —— 日常面 ——
pysci-orch plan new <计划.yaml> [--id X]  # 登记计划（格式见 README §4.2）
#   └ id 缺省取**文件名 stem**（建议显式 --id）；YAML **复制**进 `state/plans/<id>.plan.yaml`
#     （源文件登记后可删）
pysci-orch plan run <id> --text "<当前环节任务书>" [--evidence <报告.md>]  # 推进（交接处暂停；成员含 devops）
pysci-orch plan advance <id> [--text/--task] [--evidence <报告.md>]  # 事件驱动：入队即返回（M5，不阻塞；成员含 devops）
#   └ --evidence：收尾 kind:self 组长环节（文件存在 + 标题段 lint 通过才置 done，PSEP_3 §2.1）
#   └ 两子命令旗标**各自独立**（照抄对方会被 argparse 拒，backlog 20261015-202557）：
#     run 旗标：--text --task --session --name --model-tier --effort --max-turns --dirs --evidence
#     advance 旗标：--text --task --session --model-tier --effort --dirs --dry --evidence
#     （advance **无** --name/--max-turns；run **无** --dry。--session 默认 latest，
#      **续作余量已尽必须 new**；--evidence 见上行）
pysci-orch plan rework <tb-id> [--mode full|no_inspector|rework_only] [--plan-id X] [--force]
        # 返工任务书 → 三档骨架（A/B/C）并登记（§4.2）；[NEXT] 给
        # 「taskbooks link → advance」两步（返工件不走 plan new）
pysci-orch next [--limit N]              # 跨任务路由：优先级→动作类别→序；置顶任务卡片 + 简表（PSEP_3 §3.2）
pysci-orch taskbooks list|show <id>|link <id> --plan <pid>|sync|report <id>  # 用户大任务书（report=全流程报告）
pysci-orch queues list | queues enqueue <member> --text "..." [--review]  # 成员 FIFO 队列（含 devops）
pysci-orch events drain | events list  # 统一事件流：排空积压（幂等）/水位
pysci-orch plan amend <id>               # 编辑 plan YAML 后校验对账
pysci-orch plan adhoc <member> --text "..." [--dirs d1,d2]  # 计划外单步：**同步阻塞到完成**
#   └ 内部走 `plan run`（等本计划全部在途收尾）⇒ **与 `plan advance` 的「入队即返回」相反**；
#     长任务**必须后台**（run_in_background），否则撞 Bash 工具超时（backlog 20261015-211200）
pysci-orch plan list | plan drop <id>
pysci-orch suggestions                   # 待审批改进建议（附质检标记，合规显示 []）
pysci-orch approve <建议id> (--jury | --bench <案id[,案id]> | --override "<理由>")
        [--note "..."] [--no-wake] [--criteria "<判据>"] [--solo]  # 四态必居其一
        # 例外=缺陷×当跳 blocked；判决包 verdict-package.md
#   └ 组长立单（**无命令**，手写即立案）：落点 `orchestration/state/suggestions/<ts>-<member>.md`
#     （`<ts>`=YYYYMMDD-HHMMSS；后缀是**来源**成员，组长代补时写**目标**成员——判断来源看正文）。
#     列表 `[member]` 列**即**该后缀 ⇒ 代补单显示**目标**成员。正文须含 `kind: defect|usability`
#     与 `success_criteria:` 段；实质正文（去空白/标题/`kind:|source:|fingerprint:` 行）≥40 字，
#     否则标 `[!] 空壳建议`，合规标 `[]`。流程：手写 → suggestions 自检 → approve 四态 → worker 销账
#   └ **重复建议单**：`resolve` **不**收建议单 id（作用域=backlog 条目）；既定
#     `reject <id> --override "<合并至哪单>"`——历史面显示「已否决」而事实是「内容经另单
#     采纳」，审计以 `--override` 正文为准（`--as-duplicate`／扩作用域属判决面，未做）
pysci-orch watch [--timeout 14400]
        # 等 drain 事件（drain-watch.md §2）；后台挂起，漏挂 status 补账
pysci-orch wake-claim [status] [--force [--session-id id]]
        # 唤醒权读数与应急面；归属由钩子按载荷 session_id 自动完成（README §4.5）
pysci-orch reject <建议id> --override "<判决依据>"   # 否决：依据必填、留痕（J/M 判决的驳回不计）
pysci-orch consult "<问题>"               # 咨询副组长（平级：异议义务已注入）
pysci-orch consult-twin <案卷.md>  # 咨询 twin（未启用即拒 rc=2）
pysci-orch review <产物> --origin <member> [--rubric generic]
        [--goal-from <原生产任务书> | --goal "<目标陈述>"]  # --goal* 供 G2 独立核验
        # 审查链：VERDICT → FAIL 回派返工 → 复审一次 → 二次 FAIL 升级仲裁
pysci-orch jury <案卷.md> [--dry|--timeout S]  # Tier C 仲裁团（非 Qwen 双序对局，阻塞出裁决；
        # 不可达入队 rc=1；jury-panel.md/jury-channels.md）
pysci-orch eval run (--case <id>|--all) [--label L]
        # Tier B 压测**抽检面**（runner=leader）；套件本体归 bencher
pysci-orch bench-add <run|capture|ab|check> [--cases a,b] [--ref-a R --ref-b R]
        [--label L] [--tier minimum] [--brief T] [--start]  # 测量任务入队（不阻塞）
pysci-orch bench-up [--dry] | bench-queue [--all] | bench-seen <id> | bench-stop
        # bencher 生命周期（FIFO 非常驻）；判决进 status「待查看」不即时唤醒（§5.1-B）；
        # 要领 drain-watch.md、harness-eval.md
pysci-orch taste <案卷.md> [--dry]  # twin 判断通道（未生效即拒；twin-taste-mirror.md）
pysci-orch inspector-up [--dry|--force] | inspector-status | inspector-patrol [--target 菜单id]
        [--start] | inspector-assign --brief T [--start] | inspector-on|inspector-off
        # 巡检引擎（§7）：内嵌 status/approve/dispatch；inspector-engine.md
pysci-orch stats [--plan id|--member m|--days N]
# —— 低频顶层命令 ——
pysci-orch requeue <条目id> [--note "…"]   # 回排：needs_leader → pending 并唤醒 drain（组长面正门）
pysci-orch resolve <条目id> --note "…"     # 销账：确认已达标的 needs_leader 条目 → done（--note 必填留痕）
#   └ 两者作用域**同**：backlog 条目且 **status=needs_leader**；**不收建议单 id**（见立单块）
pysci-orch watchdog start|stop|status       # 看门狗生命周期：start 幂等 / stop 广播纪元停全部 / status 报读数（含逻辑实例数）
pysci-orch drain-stop                       # 请求 devops worker 协作式停止（计划器 + devops FIFO worker 两处 cancel）
pysci-orch refresh-models                   # 按 --list-models 刷新 registry 模型映射（drain 启动亦自动执行一次）
pysci-orch twin-review [case] (--list|--confirm|--reject) [--note "…"]  # 用户复核 twin 裁决（否决入 override 账）
# —— 底层命令面（降级/调试）——
pysci-orch dispatch <member> (--task F|--text T|--from-backlog) [--session latest|new|<sid>] ...
pysci-orch sessions <member> [--all|--archive <sid> [--distill]|--prune <sid>]
pysci-orch ledger [--member --days --stats] | sync [--check]
```

成员 id：sim figure writing theory lit drawing model3d（专职）；deputy reviewer devops
bencher twin inspector（特殊）。就绪 pod 以 `orch status` 为准。

**队列成员**（上列三条路径可直派）＝ 7 专职 + deputy + reviewer + **devops**；非队列成员三闸均拒。
**devops 双通道并存**（plan 直派 `kind=produce` / backlog FIFO 组包 `kind=backlog-batch`，同队列
同 worker）；bencher/inspector 各有专用队列，**不经** `queues enqueue`。细则见 `dispatch-playbook.md`。

用户控制台 `pysci`：裸跑进 TUI（人类正道）；`pysci status|journal|decisions`＝LLM 静态通道。

**四态审批**：`approve` 必带 `--jury`/`--bench`/`--override` 之一；例外=缺陷×当跳 blocked。
判决包结构与用量/override 率口径见 `verdict-package.md`。
