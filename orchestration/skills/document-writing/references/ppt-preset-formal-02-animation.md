# 学院正式汇报预设·分册 02：焦点动画与配色/字体

> 主册：`ppt-preset-formal.md`（定位、§0 速查表、§1–§3 版式纪律）。
> 本册保留 staging 详版的 §5 动画参数与 §6 配色/字体阶梯（w1 声明不得删减）。

## 5. 焦点块动画规格（核心，XML 实测）

### 5.1 分块：逻辑块，不是组合

`slide3/4` 的 `p:grpSp=0`；动画按形状 ID 逐个打靶，**不要真组合**。每块 = 1 个标签胶囊 +
1–2 个白色正文面板 + 可选主图。

| 页 | 块 | 内容 | 形状 ID（名称） |
|---|---|---|---|
| 3 | A | 非厄米理论 | 13 文本框12（正文）；9 文本框12（胶囊） |
| 3 | B | PT 对称性 | 11 文本框10；21 文本框12 |
| 3 | C | 研究对象 | 8 图片7；14 文本框13；19 文本框18；22/23/24 文本框12（研究对象/增益段/损耗段） |
| 4 | A | 散射矩阵 | 13 文本框12；9 文本框12 |
| 4 | B | 两种特殊点 | 11 文本框10；21 文本框12 |
| 4 | C | 单周期系统 | 8 图片7；19 文本框18；22/23 文本框12（研究过程/单周期系统） |
| 4 | D | 多周期系统 | 5 图片4（淡入）；14 文本框13；24 文本框12 |
| 4 | E | 无限周期系统 | 18 图片17（淡入）；15 文本框14；17 文本框12 |

### 5.2 状态与色值

| 状态 | 胶囊填充 | 面板/主图描边 | 文字 |
|---|---|---|---|
| 中性（动画前） | 渐变 `#C3CCD7 → #8898A7` | `#D6DCE5` | 胶囊白字；正文 `#595959` |
| 聚焦（当前块） | 纯色 `#8496B0` | `#8496B0` | 不变 |
| 失焦（非当前块） | 纯色 `#ADB9CA` | `#D6DCE4` | 不变 |

强调**只改胶囊填充与面板/主图描边**；正文文字颜色、面板白底、字号、透明度、位置均不变，
无移动/缩放/淡出。

### 5.3 效果、时长、缓动

| 动画 | preset | 实现 | 时长 | 延迟 | 节点关系 |
|---|---|---|---|---|---|
| 聚焦·填充 | ID=1，emph，sub=2 | `animClr fillcolor → #8496B0`；fill.type=solid；fill.on=true | 250 ms | 0 | withEffect |
| 聚焦·线条 | ID=7，emph，sub=2 | `animClr stroke.color → #8496B0`；stroke.on=true | 250 ms | 0 | withEffect |
| 失焦·填充 | 同上 | `animClr fillcolor → #ADB9CA`；fill.type=solid；fill.on=true | 1000 ms | 0 | withEffect |
| 失焦·线条 | 同上 | `animClr stroke.color → #D6DCE4`；stroke.on=true | 1000 ms | 0 | withEffect |
| 图片入场 | ID=10，entr，sub=0 | `animEffect transition="in" filter="fade"`；visibility 1 ms | 250 ms | 0 | clickEffect |
| 页面切换 | — | `<p:fade/>` | 350 ms | — | 页级 |

第 3 页所有颜色 effect 带 `decel=100000`（ease-out）；第 4 页同结构但无 `decel/accel`（线性）。
复用时建议统一 ease-out。

### 5.4 触发与顺序

- 第 1 块随页自动：首组 `stCondLst` 含 `evt="onBegin" tn=2`，effect 为 `withEffect`，页面
  进入即聚焦 A。
- 其后单击：第 2 组起只含 `delay="indefinite"`，首 effect 为 `clickEffect`；每次单击前进
  一块，不自动连播、不循环。
- 第 3 页：进入→A 深；单击 1→A 浅/B 深；单击 2→B 浅/C 深（2 次点击）。
- 第 4 页：进入→A 深；单击 1→A 浅/B 深；单击 2→B 浅/C 深；单击 3→C 浅/D 深（图片4 淡入）；
  单击 4→D 浅/E 深（图片17 淡入）（4 次点击）。
- 每次点击的「旧块变浅 1000 ms」与「新块变深 250 ms」均为 delay=0、withEffect，同时开始；
  `p:seq nextAc="seek"`，只能顺序前进。

### 5.5 XML 证据与复现

- `ppt/slides/slide3.xml`：`<p:timing>` 3 个顶层 `<p:par>`；`presetID=7/1`；`dur=250/1000`；
  `decel=100000`；目标与色值见 §5.1–5.3；`p:grpSp=0`，无 `p:bldLst`。
- `ppt/slides/slide4.xml`：5 个顶层 `<p:par>`；同色动画（无 `decel`）；第 4/5 组
  `presetID=10 entr` 的 `animEffect transition="in" filter="fade" dur=250`，目标 `spid=5/18`。
- 两页 `<p:transition p14:dur="350"><p:fade/></p:transition>`。
- 复现（PowerPoint/WPS 动画窗格）：每块加「强调·填充颜色」`#8496B0` 与「强调·线条颜色」
  `#8496B0`（与上一动画同时、0.25 s、delay 0）；加失焦版 `#ADB9CA`/`#D6DCE4`（1.0 s、
  delay 0、与下一块聚焦同时）；第 1 块自动，第 2 块起「单击时」；第 4 页 D/E 主图加
  「进入·淡入」0.25 s，与聚焦同时；页面切换 fade 0.35 s。
- **生成工具**：`pysci-compose slides focus-anim`（默认值即本节；用法见 `visual-qa.md`）。

## 6. 配色与字体阶梯

| 层级 | 字体/字号/字重 | 颜色 | 行距 |
|---|---|---|---|
| 封面主标题 | 汉仪大宋简 48 pt Regular | `#404040` | 默认 |
| 封面副标题 | 华文楷体 20 pt Regular | `#595959` | 150% |
| 节页 PART / 中文 / 英文 | 等线 24 pt Bold；汉仪大宋简 72 pt Regular；等线 12 pt Regular | 白；`#404040`；`#595959` | 默认/默认/150% |
| 内容页标题 | 汉仪大宋简 28 pt Regular | `#404040` | 默认 |
| 标签胶囊 | 汉仪大宋简 20 pt Bold | 白色 | 默认 |
| 正文 | 思源宋体 CN SemiBold 16 pt | `#595959` | 150% |
| 页底引用 | 思源宋体 CN SemiBold 10 pt | `#595959` | 单倍 |
| 结尾“谢谢” | 汉仪大宋简 66 pt Regular | `#404040` | 默认 |
| 结尾引用 | 思源宋体 CN SemiBold 10 pt | `#808080` | 150% |
| 封面/结尾胶囊 | 汉仪大宋简，未显式 sz（约 18 pt） | 白色 | 默认 |

配色（页面硬编码）：`#C3CCD7`/`#8898A7`（渐变）、`#BEC6D1`（竖条）、`#8496B0`（焦点）、
`#ADB9CA`（失焦填充）、`#D6DCE4`/`#D6DCE5`（描边）、`#FFFFFF`、`#404040`、`#595959`、
`#808080`；主题 `dk2=#44546A`。字体需本机安装 `汉仪大宋简/思源宋体 CN SemiBold/华文楷体`，
缺失会回退并改变折行。
