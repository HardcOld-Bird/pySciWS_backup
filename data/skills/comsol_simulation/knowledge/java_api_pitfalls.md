# COMSOL Java API 坑点速查（压力声学 / 通用建模）

本文件所有事实均在 **COMSOL 6.4 + mph/JPype** 实机验证过（2026-09，beam-scatter 专用 mph 任务）。
遇到「未知类型 / 未知属性 / 结果不对」时：**先用 `simulation inspect node --path …` 拿 ground
truth，或 `simulation docs search` 查手册**；不要手写一次性探针脚本（见 §6/§7）。

## 1. 类型名（create() 的类型字符串易错）

| 节点 | 正确类型名 | 常见错误写法 |
|---|---|---|
| 压力声学频域接口 | `PressureAcoustics` | PressureAcousticsFrequency |
| 极坐标绘图组 | `PolarGroup` | PolarPlotGroup / PlotGroupPolar |
| 2D 绘图组 / 表面图 | `PlotGroup2D` / `Surface` | — |
| 远场方向图特征 | `RadiationPattern`（挂在 PolarGroup 下） | — |
| 背景场 / 辐射边 / 远场积分 | `BackgroundPressureField` / `PlaneWaveRadiation` / `ExteriorFieldCalculation` | — |

频域行为来自 `PressureAcoustics` 的自动子特征 fpam1（FrequencyPressureAcousticsModel），
无需另行创建；参考模型另有 shb1(SoundHardBoundary)/init1/dcont1(Continuity) 默认特征。

## 2. 属性名（set() 的属性字符串易错）

- 材料属性组：`soundspeed`（**一个词**）与 `density`；soundspeed 的**内部变量名**是
  `comp1.matX.def.c`（见 §4 撞名）。
- study 的 Frequency 步频率列表：`plist`（+ `punit`），**不是** flist。
- BackgroundPressureField：`pamp`（幅值）、`dir`（3 向量）、`phi`、
  `PressureFieldType=PlaneWave`、`k_src=FromSpeedOfSound`、`c_mat` 须设 `from_mat`。
- 远场变量全名：`acpr.<efc tag>.pext`（如 `acpr.efc1.pext`）；pext 是 efc 特征局部变量，
  不能在域上做 Data export，只用于 RadiationPattern/派生表达式。
- Image 导出节点来源：`sourcetype=plotgroup` + `sourceobject=<pg tag>`（技能 export.py 用法）；
  输出文件名 `pngfilename`；尺寸 `size=manual` + `unit` + `width/height`。
- 组件级 Box 选择：`entitydim`（JPype 需 JInt 消歧）、`condition`、`xmin/xmax/ymin/ymax`；
  访问组件选择节点用 `comp.selection().get(tag)`。

## 3. 默认值陷阱（不报错但结果错）

- **新建 BackgroundPressureField 的域选择默认为空** → p_b=0、全场为 0。
  必须显式 `bpf.selection().all()`（或 named(...)）。
- **绘图组数据集绑定**：solve 前落盘的 mph、或 reload 后重跑 geom/mesh/study，绘图组会失去/
  从未建立 dataset 绑定 → 导出 PNG 空白（只剩轴框）。重求解后必须
  `result(pg).set("data", dset)`。`export image` 现已自动做轴框内部空白自校验并 warning。
- Array 开 `selresult=on` + `selresultshow=dom` 会自动创建组件级域选择
  `geom1_<arrtag>_dom`；材料/物理的域选择优先用这类 named selection（跨几何参数切换稳定）。
- 导出节点 tag 必须唯一：同 tag 重复 create 抛异常；用 `img_<输出 stem>` 或自动序号。
- 材料覆盖优先级：后创建者的 selection 覆盖先创建者（mat1 全域 + mat2 named 指定域是正确顺序）。

## 4. 命名冲突

- 全局参数名**不得**与材料/物理内部变量名冲突：全局参数 `c` 会与材料 soundspeed 内部变量
  `comp1.matX.def.c` 撞名 → solve 报「循环变量相关性」。声速全局参数用 `c0`。
- 增益/损耗虚部参数 `ci` 符号约定按各研究理论模型定义（如 gain_ep：ci_COMSOL = +ci_CMT，ci<0 增益）。

## 5. JPype / Python 侧陷阱

- Python int 传给 `set(String, int)` / `set(String, boolean)` 重载歧义 → `jpype.JInt(1)` 等包裹。
- ExportFeatureListClient 不可调用、无 `.feature()`；导出节点用 `create()` 返回值或 `exp.get(tag)`。
- COMSOL 导出 CSV 有多行 `%` 头：先过滤 `%` 行再跳列名行（见 postprocess.read_comsol_csv）。
- PowerShell：含 `|`/空格的 Select-String pattern 须整体引号；含空格/中文路径传本地进程时
  用 Python 内字面量而非 CLI 参数。

## 6. 诊断入口（替代一次性探针脚本）

- `simulation inspect node --mph M --path component(comp1).physics(acpr).feature(bpf1)`
  → 类型 / 标签 / 属性当前值 / 枚举允许值 / 选择各维实体数；`--methods` 附反射方法签名。
- `simulation inspect tree/params/inventory` → 结构级；`inspect java F.java` → GUI 导出摘要。
- **写入端**：`simulation node set --mph M --path <node> --set k=v [--set …]`（与 inspect node 对称）
  → 改任意节点属性零自定义代码；整数自动 JInt 包裹，单条失败汇总不中断。改完加 `--save O.mph` 落盘。
- `simulation post framebox --image P` → 渲染 PNG 轴框像素框（pixel↔data 映射/裁剪用）。
- 导出图空白 → 先看 `export image` 的 warning 与 `<png>.sidecar.json` 的 `interior.blank`，
  再查数据集绑定（§3）。

## 7. 卫生约定

- 一次性诊断脚本命名 `_probe_*.py`，**用完即删**，不留在仓库。
- builder/runner 脚本统一用 `pysci.paths.PROJECT_ROOT` 解析工作区根；
  禁用 `Path(__file__).parents[N]` 层级猜测（层级错会静默把产物写进 scripts/data/ 等 stray 目录）。
- 保存模型/产物后立即打印绝对路径核对。

## 8. 2D 轴限不暴露为可 set 属性（export --extent 对 2D 无效）

- `PlotGroup2D` 与 `component(comp1).view(view1)`（ModelView2D）**均无** xmin/xmax/ymin/ymax
  等轴限属性（已用 inspect node 逐属性确证）；GUI 的轴限靠内部 zoom 状态，Java API 不暴露。
  故 `export image --extent` 对 2D 绘图组只会得到「extent 仅部分生效：无」warning，**不要依赖**。
- 需要精确 pixel↔data 映射时，用 **auto-zoom 反演模型**：COMSOL 2D auto-zoom = 等纵横比、
  以几何包围盒为中心、按轴框像素框（sidecar.crop_box_px）长宽比在宽/高受限方向展开。
  几何 bbox 可由建模参数精确算出 → extent = f(几何 bbox, crop_box_px)，无需目视迭代。
  参考实现：`postprocess.comsol_auto_window(geom_bbox, crop_box)`；`export image --geom-bbox` 已内置
  该反演并把结果写入 sidecar 的 `extent_recovered`，下游 raster 直接可用。
- 极坐标图（PolarGroup）无矩形轴框，`post framebox` 返回 null 属正常；其轴系亦不接受 extent。

## 9. 常用节点属性速查表（cookbook，免重复自省）

未来任务先查此表；表中未覆盖或不确定时再用 `inspect node --methods` 现场自省（一次 JVM 往返）。
所有属性均可用 `simulation node set --path <node> --set k=v` 直接写入（见 §6）。

| 节点（--path 例） | 属性 | 取值/说明 |
|---|---|---|
| Surface `result(pg).feature(surf1)` | `rangecoloractive` | `on`/`off` — 开手动色标（否则逐 case 自动漂移） |
| | `rangecolormin` / `rangecolormax` | 字符串数值，色标下/上限（对称发散用 ∓MAX） |
| | `colorlegend` | `on`/`off` — 是否显示内置 colorbar（多图共享外部 colorbar 时置 off） |
| | `colorscalemode` | `linear`/`linearsymmetric`/`logarithmic` |
| PolarGroup `result(pg_far)` | `axislimits` | `on`/`off` — 开手动极径 |
| | `rmin` / `rmax` | 字符串数值，极径下/上限（多 case 统一 rmax 才能比相对能量） |
| RadiationPattern `result(pg).feature(radiationpattern1)` | `anglerestr` | `on`/`off` — 限制角域 |
| | `phimin` / `phirange` | 角域起点/跨度（度） |
| | `circle` | `unit`/`manual` — 极坐标圆归一 |
| Image 导出 `result().export(img)` | `sourcetype` | `plotgroup` |
| | `sourceobject` | 来源绘图组 tag |
| | `pngfilename` | 输出 PNG 绝对路径 |
| | `size` / `unit` / `width` / `height` | `size=manual` + `unit=pixel` + 像素宽高 |
| BackgroundPressureField `component(comp1).physics(acpr).feature(bpf1)` | `pamp` / `dir` / `phi` | 幅值 / 3 向量方向 / 相位 |
| | `PressureFieldType` | `PlaneWave` |
| | `k_src` / `c_mat` | `FromSpeedOfSound` / 设 `from_mat` |
| | `selection()` | 新建默认空 → 必须 `.all()` 或 `.named(...)`（见 §3） |
| Array `component(comp1).geom(geom1).feature(arr1)` | `selresult` | `on` — 自动建组件级域选择 `geom1_<arr>_dom`（跨参数切换稳定） |
| | `selresultshow` | `dom` |
| Box 选择 `component(comp1).selection(box1)` | `entitydim` | 几何维（JPype 需 `JInt` 消歧） |
| | `condition` | `inside`/`outside`/`intersects` |
| | `xmin`/`xmax`/`ymin`/`ymax` | 字符串数值，盒范围 |
| study Frequency `study(std1).feature(freq)` | `plist` | 频率列表（**不是** flist） |
| | `punit` | 频率单位（如 `Hz`） |
| 材料 `component(comp1).material(mat1).propertyGroup(def)` | `soundspeed` | 一个词；内部变量 `comp1.matX.def.c`（勿与全局参数 `c` 撞名，见 §4） |
| | `density` | 密度 |
