# 导出与验收清单（③绘图 reference 3/3）

> 何时读：第 5–7 步（导出/audit/预览闭环）出问题或交付前终检时。
> 机械实现：`export.py` / `audit.py` / `field_panel.ceil_label`；本册是口径与清单。

## 图脚本契约（形态约定，替代旧 runner）

- 每图一个小脚本：`src/pysci/research/<name>/article/figures/<slug>.py`，暴露
  `build_figure() -> Figure`（可选参数自定）；产物落
  `data/research/<n>_<name>/article/figures/<slug>/out/`（`pysci.paths.research_fig_dir`）。
- 脚本内：`style_context` 铺样式 → 画 → `export_figure`；**src 代码 / data 产物分离**，
  迭代日志写 `notes.md`（同图目录）。CLI 只 audit/查询，不代跑图脚本（旧 figures
  全集壳不保留）。

## 格式矩阵与导出口径

| 格式 | 用途 | 字体契约（audit 判定） | 备注 |
|---|---|---|---|
| PDF | 投稿主交付 | `/FontFile2`（fonttype=42） | 元数据去时间戳（确定性） |
| EPS | 期刊 legacy 要求 | `/sfnts`（Type 42；Type3 位图=硬伤） | 同上 |
| SVG | Illustrator/Inkscape 末修 | 保留 `<text>` 节点 | `svg.fonttype=none` |
| PNG | 预览/嵌入 | 位图不适用 | 出版 600dpi；预览 `_preview.png` 150dpi |

- **确定性元数据**：PDF `CreationDate=None`、SVG `Date=None`——同一图两次导出字节
  稳定，diff 可读（换机/重跑不产生假变更）。
- 导出期强制 `pdf.fonttype=42 / ps.fonttype=42 / svg.fonttype=none`（rc_context）。
- 产物路径有防 stray 护栏（`pysci.paths.assert_within_data`，默认开）。

## 轻量 audit 五项（`pysci-figures audit` / `audit.audit_figure`）

| 检查 | 判据 | fail 修法 |
|---|---|---|
| width | 成品实测宽 vs 契约 ±2mm | 查 tight bbox 裁切；以名义 figsize 冒充实测=假绿 |
| fontsize | 非空文本 ≥5pt | 移位/改版式，不缩字号硬挤 |
| font-embedding | PDF FontFile2 / EPS /sfnts / SVG `<text>` | 契约层样式 + 重导出 |
| panel-labels | `(a)` 形标号个数 == 面板轴数（可见 + 携带数据；排除 colorbar 轴与 ultraplot 图例/caption 自动轴） | 逐面板打标，自动轴不打标 |
| color-cycle | prop_cycle == 家风循环（Okabe-Ito 去黄） | 铺 `house.mplstyle`；改色板过色盲核验 |

**判定口径**：pass / fail / **na（依据不足）**——na 不连坐 FAIL（工具链不可用不判死）；
报告有任何 fail 即 FAIL。假绿防护：宽度拿**实测**（PNG 由 dpi 元数据换算 mm），
不拿自述值；`savefig.bbox` 模式 PNG 不记录，勿反推版面常量。

## 零遮挡铁律（硬伤级四条 + 近而不交；出图前逐条自查）

1. 文字/刻度值不得压主图图线——处理 = **移位且不破坏语义**（如沿射线外移刻度），
   不是缩字号硬挤。
2. 图体与标题保持间距（`ax.set_title(pad=...)` 或布局留白），标题不贴子图边。
3. colorbar 标题↔色标、色标↔子图保持边距（label `pad` / 布局留白），三者不顶死。
4. 任何元素重叠 = 硬伤，无例外（含标题压 colorbar）。
5. **近而不交**：不该相交的线/框保持距离——贴边/临界间距（<10px @600dpi）＝缺陷，
   与重叠同级。

自查手段：预览闭环 read_image 逐子图放大判读；旧像素判据（occlusion/ink_gap）已砍，
**以视觉验收兜底**。

## 标注语义四判据（规则文字；机械版已砍）

任务书给出可计数语义（如"每侧 2 个框、外框>内框、虚线相连"）时，画前先写逐要素
对照（参考图对应物 → 物理含义 → 产物元素 → 元素间关系 → 处置：照搬/适配/偏离+理由），
出图后按四判据人工核对：

- **counts**：每类标注原语数量与任务书一致；
- **contain**：包含/嵌套关系与边距符合描述；
- **sizes**：相对大小关系（外框>内框等）成立；
- **connectors**：连接线端点落在被连元素上、线型符合描述。

处置分支不得替换：欲换处置（含技术判断否决显式指令）→ 以疑问清单上报裁决，
不得静默替换。

## colorbar 端标取整 v2（规则文字；`field_panel.ceil_label` 已接线）

合法端标 `d1.d2×10^k`：`d1` 取 1–9 任意，`d2∈{0,5}`，`d1=1` 时另含 `d2∈{2,4,6,8}`；
第三位起全为 0。**向上取整绝不裁峰**：65.48→70、1.4655→1.5、1.8277→2.0、2.11→2.5、
1.1968→1.2、0.6436→0.65、0.98→1.0、9.7→10、0.151→0.16。负端标对称处理
`-ceil_label(abs(x))`；**径向刻度 rmax 不适用**本口径。

## 预览闭环（第 7 步，交付前必做）

1. `export_figure` 自动产 `<base>_preview.png`（150dpi）；
2. **read_image 目检**（视觉策略 2.4：视觉内容交付前必须 VLM 查看）：整体版式 →
   逐子图放大判读（零遮挡铁律）→ 公式/刻度可读性 → 色图/图例语义；
3. 有伤 → 回第 2 步改图脚本重跑；无伤 → `pysci-figures audit` 过五项后交付
   （present 声明交付件）。

## 冒烟验收（真实产物目检，计划 §5）

APS 双栏 170mm 或 Nature 双栏 183mm 实产出一图：矢量三件 + PNG 全出、PDF/EPS 字体
嵌入过 audit、含公式标注、面板标号齐全、色盲安全循环生效——目检 + audit 双过即验收。
