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

## 研究线图件偏好（1_gain_ep）

- **beam_scatter 场图配色 = COMSOL Rainbow，0 值黄绿非白**（用户 2026-10-15 显式要求，
  组长 2026-10-18 裁决维持）：五段 `#1b3f9e`(深蓝) → `#35c6d9`(青) → `#b5e02a`(黄绿/零)
  → `#ffd21e`(黄) → `#d81e2a`(红)，配合对称 SymLogNorm（0 落 0.5）。不要改回 `RdBu_r`。

## 非管线图件（示意图 / 框图 / PPT 贴图）

白名单只给图目录（无 `src/` 写入权）时，把构建脚本放 `<figdir>/` 并**加下划线前缀**
（如 `_build_xxx.py`）——`discover_pipeline` 与 `audit` 的 `stray` 判定都跳过下划线 .py；
非下划线的唯一 .py 会被当成管线尝试构建（无 `build_figure` → `ERROR build` → audit FAIL）。

- **`figure-audit` 声明要指「图目录」而不是「文件」**（2026-10-16 实测）：`audit <png 路径>`
  → `[ERROR] build: 构建管线失败：FileNotFoundError('图目录不存在：…png')`，exit 1 → 交付被判 FAIL；
  `audit <图目录>`（且目录内无非下划线 .py，src 侧亦无同名管线）→ `INFO not-scaffold` +
  `verdict: N/A (0 error, 0 warn)`，exit 0 PASS。故非管线产物用
  `<artifact check="figure-audit">…/figures/<slug></artifact>` 声明**目录**。
- **非管线产物的真判据**（上述 `audit` 是 N/A，不算证据）：
  `pysci-figures inspect <png>`（尺寸/彩度/边缘水印）+
  `pysci-figures overlap <A> --white <W>`（零遮挡像素判据）。`W`＝同一次渲染出的**删文字版**
  （同一 `draw(texts=False)`，版面完全一致），且**先写 A 后写 W**（overlap 有 mtime 新鲜度守卫，
  否则 rc=2）。构建脚本可用 `--occlusion-dir` 一次导出两层。
- **线色亮度 < 60 才会被 overlap 当「图线真身」**（`LINE_LUM_MAX=60`，均值口径）：框图线条用
  `#33383E`（均值 56）可入蒙版；比 `#6F7880`（均值 120）更浅的灰**不入蒙版**，判据会变空转假 PASS。
- **PNG 用不透明白底**：透明底 PNG 的 RGB 通道是脏的，凡 `convert('RGB')` 的工具（inspect/overlap）
  都会读到黑背景 → 假读数。PPT 模板底是 `#F8F8F8`，白底贴上去（与 248 差 7/255）不可见。
- **PPT 模板气质**（`data/research/1_gain_ep/汇报PPT/PPT模板.pptx`，2026-10-16 实测）：16:9
  13.333×7.5 in，底色 `#F8F8F8`（非纯白，带纸纹），深炭块 `#404040`、细线冷灰，中文字族
  **等线 DengXian**（本机 `C:\Windows\Fonts\Deng.ttf`，mpl 名 `DengXian`；`Microsoft YaHei` 为备选）。
  看模板原貌：`pysci-compose slides render <pptx> --out <pod bench 目录>` 再 Read。
- **框图脚本自定版式、不走 `style_context`**：数据坐标系取 16×8（figure 9.6×4.8 in，1 单位 = 0.6 in），
  `ax.set_position([0,0,1,1])` + `aspect='equal'` → 图形不走形；字号按「主框 > 其余框 > 箭头注 > 行标」
  分级（如 18/14/10.5/9.5 pt），两行/两栏共用同一组线宽常量即满足「同线宽」。
