# 任务书 fig0-orch-smoke

（派发时间：2026-10-09T20:10:37+08:00）

编排体系 Phase 1 验收图件任务（真实生产）。

目标：在研究线 gain_ep 下产出一张单面板验收图 fig0_orch_smoke，主题为「编排体系冒烟：示例色散曲线」。合成数据即可（例如 ω(k)=c·k 的两条分支加一个避免交叉微扰），不需真实物理精度，但必须走完正规图管线并通过合规审计。

步骤要求：
1. 用 pysci-figures new gain_ep fig0_orch_smoke --style aps --width single 脚手架；
2. 在图目录内写数据与绘图脚本（遵守管线约定），build 出交付件与 _preview.png；
3. Read 预览图做视觉自检（布局/标注/字体无缺陷）；
4. 本地跑 pysci-figures audit <figdir> --panels 直到 PASS；
5. 交付：<result> 内附一句话图说明 + <artifact check="figure-audit">data/research/1_gain_ep/article/figures/fig0_orch_smoke</artifact>。

约束：所有正式产物只落在 data/research/1_gain_ep/article/figures/fig0_orch_smoke/ 内；草稿放 bench/。
