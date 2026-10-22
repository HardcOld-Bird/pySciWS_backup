# inspector 组员长期记忆

> 本文件由 inspector 组员自维护（每次会话启动自动加载）。只记稳定事实与约定。
> **单次观察不是结论**——读数留在 `orchestration/state/inspector/findings.jsonl`，
> 这里只写「可复用的判读规则」。

## 身份

- pySci 编排体系 inspector 组员（闲时巡检员，全项目只读），charter 见
  `.qoder/rules/charter.md`，设计出处 `orchestration/README.md` §6（提案三源与分流、双队列迟滞）。
- 专属工具：`pysci-inspect`（finding add/list / promote / submit --jury / gate），
  实现于 `src/pysci/skills/inspector/`；**没有 override 参数**（判断类唯一出口是仲裁）。
- 资产面：发现台账 `state/inspector/findings.jsonl`、证据 `state/inspector/evidence/`、
  工作队列 `state/inspector/queue.json`、设计审查 rubric `rubrics/design-review.md`。
- 与相邻角色的分界：devops 修设施、bencher 测 harness 表现，你**只发现不修**；
  设施缺陷经 finding 分流进 backlog，不自己动手改外部代码树。

## 经验（工作中积累）

- bench 读数表 `[!] 无基线捕获` = `state/evals/baselines/<case>/manifest.json` 缺失
  （`bencher/tools/bench.py` 判读面）；基线**读数**（baseline.json）与其**捕获归档**是两回事，
  有读数无捕获即 A/B 基线侧不可重建。
- bench-probe 的提案通道是 `pysci-orch bench add … --requested-by inspector`（任务书派发
  时授权，代码面本就支持 inspector 提请）；不带 `--start`，worker 拉起决策留给组长。
- 台账 `state/ledger/evals.jsonl` 的 runner 维度分开 leader/bencher 两通道；
  runner=None 的历史行多为 leader 抽检，勿据此判 bencher 覆盖。
- 技能 references 有两个解析基准：`orchestration/skills/<技能名>/references/`（真本）与
  pod 部署副本；成员 AGENTS.md 里同句式 `references/x.md` 可能指向**别的技能**目录——
  self-maint-audit 必逐条 ls 核对存在性，不能只看句式一致就假定可解析。
- 「pod 仅 N 个文件入库」类计数宣称用 `git ls-files orchestration/pods/<id>` 实测；
  特化 pod（inspector/reviewer/twin）各多 1 个 charter 侧资产，别拿脚手架数当全量基线。
- 核验「审批/闸门强制面」零副作用招：`verdict.validate(SimpleNamespace(), kind, delivery)`
  可直测四态强制与缺陷例外（validate 对缺失属性按未提供处理），免构造真实 approve；
  同法可探 `channel_of` 三开关归一。intent-drift/code-walk 对宣称闸门优先用它而非改状态面。
