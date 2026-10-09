# figure 组员长期记忆

> 本文件由 figure 组员自维护（每次会话启动自动加载）。只记「下次会话还要知道的
> 事实与约定」：研究线图件偏好、踩过的坑、与特定数据格式相关的经验。保持精简；
> 主题多了以后可拆分到 `.qoder/rules/` 自建规则（model_decision/glob 触发按需加载）。

## 身份

- pySci 编排体系 figure 组员（科研绘图专员），charter 见 `.qoder/rules/charter.md`。
- 工作区：`bench/` 草稿、`outbox/` 交付副本、正式产物落任务书白名单目录。

## 经验

- **图管线代码位与写权限冲突（重要）**：`pysci-figures new` 把管线 .py 落到
  `src/pysci/research/<线>/article/figures/<slug>.py`，而 `runner.discover_pipeline()` 是
  **src 优先**、图目录内的同名管线只是回退项。figure pod 对 `src/` 无写权限，所以任务书白名单
  必须同时给出该 src 图目录，否则 `build` 出来的永远是脚手架占位图。
- **`audit --panels` PASS 不代表图的内容对**：它只量宽度/字号/字体嵌入/角标/色盲可辨，
  脚手架 2×2 正弦占位图同样 PASS。判定前必须 `Read` 预览图核对内容。
- **Python 标识符里的 Unicode 陷阱**：下标数字与正负号（₀₁₂₊₋）和 U+2212 减号**不是**合法
  标识符/运算符（`SyntaxError: invalid character '₁'`）。可用写法：`ω0 ω1 ωp Δω ω̄ κ δ Ω`
  （希腊字母 + ASCII 数字合法）；运算符一律 `-`。
- **aps/single = 85 mm**，审计字号下限 = tick 字号 × 0.9（aps 下 6.3 pt，legend 固定 7 pt）；
  绘图代码里**不要**显式传 `fontsize=`，交给 style_context 继承即可。
- **CLI 用法**：从项目根运行，裸名 `pysci-figures` 本机可用（只试一次）；Git Bash 下多词参数
  用单引号；`build`/`audit` 的 figdir 用相对路径即可。
- **STYLE.yaml 须逐图目录放置（已知缺陷，组长已确认）**：根目录 `STYLE.yaml` 非逐键合并，
  缺本图专属文件时 `audit`/`build` 会用根默认值判定（gain_ep 根为 `width: double`，单栏图被按
  170 mm「假通过」）。devops 修复（backlog 20261009-audit-width-masked）前，正式交付务必确保
  figdir 内有本图专属 `STYLE.yaml`（`style/aspect/palette` 一并给出）或显式传 `--width`。
- **数据驱动管线的落法**：合成数据生成脚本放 pod `bench/`（草稿），产出的 CSV 落
  `<figdir>/data/`，CSV 的 `#` 注释行写模型与参数（`np.loadtxt(comments='#')` 默认跳过），
  管线只读 CSV、不再自己造数。
