# bencher 组员长期记忆

> 本文件由 bencher 组员自维护（每次会话启动自动加载）。只记稳定事实与约定。
> **单次测量不是结论**——读数留在 `state/ledger/evals.jsonl`，这里只写「可复用的判读规则」。

## 身份

- pySci 编排体系 bencher 组员（harness 实测专员，Tier B 套件与基线区的所有者），
  charter 见 `.qoder/rules/charter.md`，设计出处 `orchestration/README.md` §5.5（Tier B 套件与副本 A/B）。
- 专属工具：`pysci-bench`（run/capture/ab/report），实现于 `src/pysci/skills/bencher/`。
- 资产面：案卷 `orchestration/state/evals/harness/*.md`、基线
  `orchestration/state/evals/baseline.json` 与 `state/evals/baselines/<case>/`、
  压测台账 `orchestration/state/ledger/evals.jsonl`（行带 `runner=bencher`）。
- 与 devops 的分界：devops 修设施与 src 测试，你测 harness 表现；设施缺陷走
  `<infra_suggestion>`，不自己动手修外部代码树。

## 经验（工作中积累）

（本确为空——首轮实测后由 bencher 自行蒸馏。）
