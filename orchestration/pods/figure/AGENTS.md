# figure 组员长期记忆

> 本文件由 figure 组员自维护（每次会话启动自动加载）。只记「下次会话还要知道的
> 事实与约定」：研究线图件偏好、踩过的坑、与特定数据格式相关的经验。保持精简；
> 主题多了以后可拆分到 `.qoder/rules/` 自建规则（model_decision/glob 触发按需加载）。

## 身份

- pySci 编排体系 figure 组员（科研绘图专员），charter 见 `.qoder/rules/charter.md`。
- 工作区：`bench/` 草稿、`outbox/` 交付副本、正式产物落任务书白名单目录。

## 经验

- **Pillow 存 GIF 会合并连续相同帧（2026-10-29 实测）**：`save_all` 把逐字节相同的相邻帧
  合并并把 duration 累加 → 344 帧变 60 帧（总时长不变但 `n_frames/duration` 的 fps 失真，
  不满足 12–15 fps 类验收）。对策：让每帧至少有一个小像素变化（本项目用角落时间读数
  `t = … s`），既防合并又给时间轴上下文。
- **GIF 调色板**：共享调色板（缩略图拼图 MEDIANCUT 后 `quantize(palette=…)`）会把深蓝
  `#123a63`/炭灰 `#33383E` 映射成灰；改逐帧 `quantize(colors=256, dither=NONE)` 颜色准确，
  900×700×344 帧仍只有 3.2 MB（Pillow 写 delta 子矩形）。
- **数据驱动动画的像素级自检**：`audit` 不适用时，取锚帧 PNG 在固定数据坐标列扫描深色
  像素（RGB 均值 <100），与 CSV 映射的轴像素比对，6 锚帧 maxerr <1 px 即可作证据；
  同时可扫几何壁线验证 x 平移后的轮廓。
- **mp4 无系统 ffmpeg 时**：`uv pip install --target <pod bench>/_libs imageio-ffmpeg`
  取自带 libx264 的 ffmpeg，`PYTHONPATH` 注入即可；脚本里按 `FFMPEG` env → PATH →
  `imageio_ffmpeg` 顺序查找，保持可复跑。

- **colorbar 端标取整 v2**（用户 2026-10-27）：vmax/端标默认走
  `field.ceil_label`（v2 阶梯 `d1.d2×10^k`：`d2∈{0,5}`，`d1=1` 另含 2/4/6/8，1.0 合法；
  向上取整，如 65.48→70、1.4655→1.5、2.11→2.5、0.6436→0.65；旧名 `ceil_one_sig` 仍可用），
  显式量程也应给取整值；**径向刻度 rmax 例外**（维持 `_nice_ceil` 现口径）。

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
  缺本图专属文件时 `audit`/`build` 会用根默认值判定（example 根为 `width: double`，单栏图被按
  170 mm「假通过」）。devops 修复（backlog 20261009-audit-width-masked）前，正式交付务必确保
  figdir 内有本图专属 `STYLE.yaml`（`style/aspect/palette` 一并给出）或显式传 `--width`。
- **数据驱动管线的落法**：合成数据生成脚本放 pod `bench/`（草稿），产出的 CSV 落
  `<figdir>/data/`，CSV 的 `#` 注释行写模型与参数（`np.loadtxt(comments='#')` 默认跳过），
  管线只读 CSV、不再自己造数。

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
  都会读到黑背景 → 假读数。贴进浅底幻灯片时白底即可（与 248 差 7/255 不可见）。
- **框图脚本自定版式、不走 `style_context`**：数据坐标系取 16×8（figure 9.6×4.8 in，1 单位 = 0.6 in），
  `ax.set_position([0,0,1,1])` + `aspect='equal'` → 图形不走形；字号按「主框 > 其余框 > 箭头注 > 行标」
  分级（如 18/14/10.5/9.5 pt），两行/两栏共用同一组线宽常量即满足「同线宽」。
