# 出版级出图契约（COMSOL 直出栅格 + sidecar）

> 与③域导出契约同风格：**尺寸/DPI/字体/背景/品牌全部契约化**，纯函数拼装、机械断言
> 锁死（`tests/skills/comsol_simulation/test_export.py`）。本册是「COMSOL Image 导出
> 节点」控制面的真源表；③域 matplotlib/ultraplot 导出契约见
> `.agents/skills/scientific-plotting/references/export-acceptance.md`。

## 1. 尺寸/DPI/背景契约

- 尺寸契约：`export.pixels_from_size(width_mm=…, height_mm=…, dpi=…)`——
  `px = round(mm/25.4*dpi)`；出版 DPI 默认 **600**（期刊印刷口径），预览 150。
- 图像输出 PNG（`imagetype='png'`）；**字体不嵌入**（栅格即像素）——含文字的可编辑
  交付走③域矢量链（fonttype=42 嵌入），COMSOL 直出只管「看得准」。
- 背景 `background='color'`（白底 `customcolor=[1,1,1]`）/ `'transparent'`；
  `antialias='on'`；`lockratio='off'`（宽高都给定时锁死像素尺寸）。

## 2. 品牌剥离与 clean（像素级控制）

- **品牌剥离只走原生开关**（imgcheck 像素擦除不建，用户裁决 2026-11-17）：
  **先开** `options1d/2d/3d='on'`（optional components 总开关，logo*d 仅在其开启时
  受控），**再关** `logo1d/2d/3d='off'`——顺序不可换，1d/2d/3d 是各自独立属性全设。
- **clean**（下游排版用）= `title1d/2d/3d='off'` + `legend1d/2d/3d='off'`（导出节点侧）
  + 子特征 `colorlegend='off'` + 绘图组 `titletype='none'`。
- 色标钉死（防逐 case 漂移）：子特征 `rangecoloractive='on'` + `rangecolormin/max`；
  极坐标：`axislimits='on'` + `rmin/rmax`（多 case 统一 rmax 才能比相对能量）。

## 3. Image 导出节点属性真源表

mph 官方 demo `create_capacitor.py` 全量属性 + COMSOL 6.4 手册（薄胶水
`export.build_export_plan` 用**候选名首中**防御不同版本拼写差异）：

| 用途 | 属性 | 取值 |
|---|---|---|
| 来源 | `sourcetype` / `sourceobject` | `plotgroup` / **绘图组 tag**（非显示名） |
| 输出 | `filename`（mph）/ `pngfilename`（手册） | 绝对路径 |
| 尺寸 | `size` / `unit` / `width` / `height` | `manualweb` / `px` / 像素宽 / 像素高 |
| 分辨率 | `resolution` | **DPI 数值**（默认 96，出版 600） |
| 比例 | `lockratio` | `off`（双尺寸锁定） |
| 背景 | `background` / `customcolor` | `color`/`transparent` / `[1,1,1]` |
| 抗锯齿 | `antialias` | `on` |
| 标题/图例 | `title1d/2d/3d`、`legend1d/2d/3d` | clean 时 `off` |
| 品牌 | `options1d/2d/3d`、`logo1d/2d/3d` | options=`on` 后 logo=`off` |

其他常用：`zoomextents`、`fontsize`、`axes1d/2d`、`showgrid`、`grid`、
`axisorientation`、`target='file'`、`qualitylevel`、`imagetype`。

## 4. sidecar 契约（`<png>.sidecar.json`，schema `pysci.comsol.sidecar/1`）

一句话：**下游不目视迭代即可把数据坐标画回像素图上**。像素约定：原点**左上**、
x 向右、y 向下；`crop_box_px=[x0,y0,x1,y1]`（轴框矩形，顶左→底右）；数据 y 向上。

```json
{
  "schema": "pysci.comsol.sidecar/1",
  "image": {"path": "...", "width_px": 709, "height_px": 709},
  "crop_box_px": [0, 0, 709, 709],
  "extent": [0, 1, 0, 1],
  "transform": {"sx": 709, "sy": 709, "ox": 0, "oy": 709},
  "source": {"plot_group": "pg1", "geom_bbox": [0, 1, 0, 1]},
  "export": {"width_mm": 120, "dpi": 150, "clean": true, "brand_stripped": true},
  "interior": {"blank": false, "flat_frac": 0.31}
}
```

- **变换**：`px = ox + sx*x`；`py = oy - sy*y`（sy>0 显式表达 y 翻转）。
  `crop_box_px=None` → 轴框=整幅图（clean 导出无 colorbar/标题时成立）。
- **extent**：显式给定或由 `comsol_auto_window(geom_bbox, crop_box_px)` 反演
  （COMSOL 2D auto-zoom：等纵横比、几何 bbox 中心、按 crop 长宽比展开——
  2D 轴限不暴露为属性，见坑点册 §6）。
- **interior.blank**：绘图区中心区（60%）主色占比 ≥0.995 判 blank（防「数据集未绑
  → 空白 PNG」复发；轴框/刻度混入采样会误判，故只采中心）。
- ③域 truthful colorbar 数据通路：COMSOL `export('Data')`/`evaluate` → CSV/VTK →
  ③域重绘；栅格直用则本 sidecar 给精确像素↔数据映射。

## 5. 消费端用法（叠加；衔接③域）

```python
from pysci.skills.comsol_simulation import sidecar as sc

data = sc.read_sidecar("out.sidecar.json")
sc.imshow_sidecar(ax, data)                      # imshow(extent=…, origin='upper')
x, y = sc.sidecar_transform(data).to_data(400, 250)   # 像素 → 数据坐标
sc.overlay_rect_pixel(ax, data, center_px=(400, 250), width_px=80, height_px=40,
                      angle_deg=30, edgecolor="k")    # 像素空间旋转矩形（锚点旋转修正）
sc.overlay_point_pixel(ax, data, 400, 250, marker="x")
```

叠加矩形在**像素空间**旋转后逐角映射到数据坐标画 Polygon——绕开 matplotlib
Rectangle 绕角点旋转 + 非等纵横比形变的坑（③域移交项 #9 的消费端形态）。

## 6. 验收清单

1. 机械：出图参数拼装/变换数学单测恒跑；真实冒烟 `-m hardware` 实跑（尺寸契约
   `width_px == round(mm/25.4*dpi)`、`logo2d=off` 在 applied、`interior.blank=False`）。
2. 视觉（策略 2.4）：**read_image 目检产物 PNG**——场是否真画出来、坐标范围与
   sidecar extent 一致、无品牌 logo、clean 生效；「能导出 ≠ 是对的」。
3. 交付物三件：PNG + `<png>.sidecar.json` + （重绘线）CSV/VTK 数据导出。
