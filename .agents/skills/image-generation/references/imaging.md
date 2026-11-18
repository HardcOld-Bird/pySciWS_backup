# Imaging：measure 连通域量化 / align 亚像素配准（⑦域）

> 薄胶水实现：`pysci.skills.image_generation.imaging`。
> ADR（2026-11-17，用户批准）：旧 pySciWS 基于 scikit-image（regionprops /
> phase_cross_correlation）；重建为 **scipy.ndimage + numpy.fft 单轨**——scipy 已是
> 核心依赖，为两个算子不拉 skimage/opencv 全家桶（极简）。**语义与 skimage 兼容**
> （字段与约定一致），skimage/opencv 到位可无痛切换。裁决注记：measure/align **留⑦域**
> （服务生成图/实验图量化与图层/组图/叠加配准；mpl 数据图用不上像素测量/配准）。

## 一、measure_regions：连通域量化

```python
regions = imaging.measure_regions("fig.png", min_area=50)
```

- **二值化来源**（优先级）：`mask`（>127 前景）> `thresh` 固定阈值 > Otsu 自动阈值；
- **连通性**：8 连通（skimage `connectivity=2` 同款）；`min_area` 去噪点；
- 输出字段（面积降序）：

| 字段 | 口径 |
|---|---|
| `area_px` | 像素面积 |
| `centroid` | (x, y) 质心，0.01 px |
| `bbox_xyxy` | 外接框 (min_col, min_row, max_col, max_row)，max 端开区间（skimage 同款） |
| `axis_major/minor` | 长/短轴 = **4·√λ**（λ=二阶中心矩协方差特征值；regionprops 直径口径） |
| `aspect` | major/minor |
| `eccentricity` | √(1 − λ2/λ1)（0=圆，→1=细长） |

- 用途：配 scale bar 换算物理面积/尺寸；统计生成图元素相对占比；
  核对生成图几何比例（AI 爱画错比例——measure 是机械核对面）。

## 二、align：相位相关亚像素配准

```python
res = imaging.align("layer.png", "base.png", dest="layer_aligned.png")
# res: {"shift_yx": [dy, dx], "shift_xy": [dx, dy], "rms_error": float, "out": path}
```

- **约定**：`shift` 是"把 src 平移 shift 后与 ref 重合"
  （= `skimage.registration.phase_cross_correlation(ref, src)` 同款），
  应用即 `scipy.ndimage.shift(src, shift)`；给 `dest` 自动应用并写出；
- **算法**：FFT 相位相关（功率归一化 + **相位门限**：低于峰值功率 1e-3 的 bin
  其相位是插值/数值噪声，不参与）→ 整数粗峰 → Guizar-Sicairos 上采样 DFT
  ±1px 邻域细化（`upsample=10` → 0.1px 栅格）；
- 要求两图**同尺寸**（先 `postprocess.crop_resize` 统一）；
- 用途：图层拆分后归位、组图/系列图对齐、生成图与③域图/COMSOL 场图叠加前对齐
  （叠加走③域 `imshow(extent=...)` 文档化用法）。

### 实测精度（2026-11-17 固定样例，tests/skills/image_generation/test_imaging.py）

| 输入 | 误差 |
|---|---|
| 整数平移（样条重采样=任意实图） | 0.00 px |
| 分数平移（精确傅立叶平移） | ≤0.05 px（=1/upsample 栅格量化） |
| 分数平移（样条重采样） | ≤0.2 px（带限图欠约束，见下） |

### 已知限制（与 skimage 同款，非实现缺陷）

- **极带限图**（大片平滑渐变/色块，几乎无纹理）亚像素估计天然欠约束——相位信息
  集中在低频，0.1px 级定位不可靠。真实生成图/实验图是宽带纹理，适用；
- 相位相关只解**平移**（旋转/缩放/透视不在此列；透视拉正等旧算子已收缩）；
- 大图一次性 FFT：O(N log N) 内存 ~16 bytes/px，4K 图 ~130MB 可用，8K 谨慎。
