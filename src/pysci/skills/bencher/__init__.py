"""bencher 技能：harness 实测组员的专属工具集（``pysci-bench``）。

与 orchestration 工具面（组长专属 ``pysci-orch``）和 devops 工具面（``pysci-dev``）严格
三分（README §1「一成员一技能一 CLI」与共享逻辑做库）：本包**只做测量**——跑套件案、捕获基线、副本 A/B、出
二元判决。执行链一律**复用**共享库 :mod:`orchestration.tools.eval` /
:mod:`orchestration.tools.replica`，不在这里复制第二套派发或判据实现；也不实现任何
组长侧编排逻辑（队列与 worker 生命周期在 ``benchd``，由 ``pysci-orch bench-*`` 门面调用）。

设计依据 ``orchestration/README.md`` §5.5；作业规程见 bencher 技能 SKILL.md。
"""
