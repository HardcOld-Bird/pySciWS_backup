member: figure
tier: minimum
dirs: data/research/eval_scratch/fig-basic
max_turns: 14
timeout_s: 900
产物: data/research/eval_scratch/fig-basic/eval_band check=figure-audit
说明: E2 首案——最低成本档能否独立完成一次**出版级出图 + 自检**。判据=产物存在 +
  figure-audit 通过（0 error）。产物落在 gitignore 的 eval_scratch，跑完不留脏树。

## 任务书

为 harness 压测生产一张**单面板**出版级图，主题：非厄米二能级的能级劈裂

    ω± = ω̄ ± √(δ² + κ²)，取 ω̄ = 1、κ = 0.35、δ ∈ [-2, 2]（201 点）

**落盘位置（本任务白名单，勿写到别处）**：
`data/research/eval_scratch/fig-basic/eval_band/`

必须产出两个文件：

1. `STYLE.yaml`——内容照此（单栏 aps）：

   ```yaml
   style: aps
   width: single
   aspect: 0.618
   palette: okabe-ito
   ```

2. `eval_band.py`——在图目录内定义管线（`discover_pipeline` 认「目录名同名 .py」）：

   ```python
   import numpy as np
   from pysci.skills.scientific_plotting.tools import layout, palette


   def build_figure(style=None, research_dir=None, **kwargs):
       δ = np.linspace(-2.0, 2.0, 201)
       ωp = 1.0 + np.sqrt(δ**2 + 0.35**2)
       ωm = 1.0 - np.sqrt(δ**2 + 0.35**2)
       fig, ax = layout.grid(1, 1)
       ax.plot(δ, ωp, lw=1.4, color=palette.color(1), label=r"$\mathrm{Re}\,\omega_+$")
       ax.plot(δ, ωm, lw=1.4, color=palette.color(2), label=r"$\mathrm{Re}\,\omega_-$")
       ax.set_xlabel(r"$\delta$")
       ax.set_ylabel(r"$\omega$")
       ax.legend()
       return fig
   ```

然后**自己跑验收**并把输出贴进交付：

    pysci-figures build data/research/eval_scratch/fig-basic/eval_band
    pysci-figures audit data/research/eval_scratch/fig-basic/eval_band --panels

`audit` 必须 `verdict: PASS (0 error …)`。若报 width/fontsize 的 ERROR，改到通过为止，
不要交付未过的图。交付走 `<result>`，产物声明：

    <artifact check="figure-audit">data/research/eval_scratch/fig-basic/eval_band</artifact>

不要碰 `src/`、`orchestration/` 或任何白名单外的路径；不要 commit。
