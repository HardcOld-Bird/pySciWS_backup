# 可视化坑点配方（跨研究通用教训）

来源：gain_ep 研究 Phase A（3D 谱曲线/相位绕数）与 Phase B（COMSOL 场图合成）的实机迭代教训。

## 1. 发散曲线：nan 遮断优于饱和裁剪

奇点邻域发散的曲线（如 EP 附近本征值实/虚部、绕数密度）不要用 clip 饱和到轴限——
饱和段会画成贴轴的假平台，被误读为物理特征。正确做法：把超出显示窗口的采样点
置为 `nan` 让 matplotlib 断线（`np.where(np.abs(y) > ylim, np.nan, y)`）。

## 2. 过采样导致虚假弯曲 + 轴限匹配

奇点附近过密采样会把数值噪声放大成肉眼可见的"钩形/弯曲"假结构；采样密度应与
显示窗口匹配。先定**感兴趣有限窗口**（轴限），再按窗口尺度选采样步长；轴限本身
也要与理论关注范围匹配，避免发散尾部压缩主体特征。

## 3. 栅格叠图：数据坐标优于 image-fraction 目视标定

在外部渲染栅格（COMSOL PNG 等）上叠标注时，**不要**用 image-fraction 坐标目视标定：
渲染含边距/colorbar、`imshow` 的 `origin` 易翻（`origin="lower"` 会上下翻转栅格）、
坐标漂移需多轮目视迭代。受支持路径：

1. comsol 技能 `export image` → PNG + sidecar（`crop_box_px` 轴框像素框）。
   注意：COMSOL 2D 轴限**不暴露**为可 set 属性，`--extent` 对 2D 不生效（见 comsol
   knowledge/java_api_pitfalls.md §8）；数据窗口用 **auto-zoom 反演**：等纵横比、几何 bbox
   居中、按 crop_box 长宽比展开 → extent = f(几何 bbox, crop_box_px)。
2. `figures raster-panel --image … --sidecar … --overlay spec.json`（或脚本内调
   `raster.read_raster` + `raster.apply_overlays`）：裁轴框内部后 `imshow(extent=数据窗口)`，
   叠加原语（rotbox/panel/dashed/text）全用数据坐标。

## 4. matplotlib Rectangle 绕锚点旋转，非绕中心

`Rectangle((x, y), w, h, angle=θ)` 的旋转中心是**锚点 (x,y)=未旋转的左下角**，不是矩形中心。
若直接传 `(cx-w/2, cy-h/2)` 作锚点，旋转后真中心会偏离 (cx,cy)，且偏移随 θ 变化（左右不对称）。
要令 (cx,cy) 为旋转后真中心，须反解锚点：
`ax = cx - (w/2·cosθ - h/2·sinθ)`，`ay = cy - (w/2·sinθ + h/2·cosθ)`。
raster.apply_overlays 已内置此修正。

## 5. 极坐标/方向图半空间问题

硬墙对称面（y=0 无限刚壁）的远场只在上半平面有物理意义：RadiationPattern 开
`anglerestr` 限 0–180°，否则下半平面画出阴影区伪瓣。
