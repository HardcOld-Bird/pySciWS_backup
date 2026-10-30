"""编排技能：组长专属的多 Agent 编排管理设施。

与 ``pysci.skills`` 下 7 个末端技能包不同，本包不服务科研生产，而是驱动
``orchestration/README.md`` 定义的星形编排体系：headless 组员派发（会话池 +
resume 接力）、交付协议解析、声明式机械验收、台账统计、技能真本部署。

唯一入口是 console script ``pysci-orch``（组长经 Bash 调用）；组员与 devops
不使用本 CLI（devops 有独立的 ``pysci-dev``，Phase 2 落地）。
"""
