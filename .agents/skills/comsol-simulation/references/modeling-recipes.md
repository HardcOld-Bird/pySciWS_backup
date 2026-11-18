# 建模配方（mph/COMSOL-MCP 双入口速查）

> 属性名/类型名真源：mph 官方 demo `create_capacitor.py`（COMSOL 6.x 全量 Image 导出
> 属性表）+ COMSOL 6.4 手册 + 旧项目 2026-09/2026-11-17 实机验证。凡本册未覆盖的
> 类型/属性名：先 `mph.inspect(node)` 或 COMSOL-MCP `physics_list_features` 现场自省，
> 不要猜名。

## 1. 双入口分工

| 任务 | 首选入口 |
|---|---|
| 通用建模/求解/导出（agent 对话式驱动） | COMSOL-MCP 工具（geometry 16/physics 28/mesh 4/study 8） |
| 脚本化/批处理/研究树可复现流水线 | mph 脚本（本册口径） |
| 差异化（出版级出图/sidecar/探测/重绑/验证） | `pysci.skills.comsol_simulation` 薄胶水 |

## 2. mph 建模骨架（官方 demo 同款 API）

```python
from pysci.skills.comsol_simulation import session
client = session.start_client("6.4.0.293", cores=2)   # 占证！进程退出放证
model = client.create("my_model")

# 几何（2D 序列；create(维数, name=...)）
geometry = (model/'geometries').create(2, name='geometry')
rect = geometry.create('Rectangle', name='domain')
rect.property('size', ['1', '1'])
model.build(geometry)

# 材料（属性组按 tag 'def' 走 Java 侧——显示名随界面语言本地化）
materials = model/'materials'
air = materials.create('Common', name='air')
pg = air.java.propertyGroup('def')
pg.set('density', '1.2')
pg.set('soundspeed', '343')          # 一个词（不是 sound_speed）

# 物理（类型名 + 几何序列；physics.create(类型, 几何, name=...)）
physics = model/'physics'
acpr = physics.create('PressureAcoustics', geometry, name='acoustics')

# 网格（**显式建序列**，否则求解静默空解）
(model/'meshes').create(geometry, name='mesh')

# 研究（study 节点下 create 研究步）
study = (model/'studies').create(name='study')
step = study.create('Frequency', name='frequency')
step.property('plist', '1000')       # 频率列表属性名 plist（不是 flist）
step.property('punit', 'Hz')

model.solve('study')                  # 按**名**求解（tag 查找不可靠）
```

节点寻址一律 `model/'分组'/'名字'`（mph 隐藏 tag，名字唯一）；分组名：`parameters`/
`functions`/`components`/`geometries`/`selections`/`physics`/`materials`/`meshes`/
`studies`/`solutions`/`datasets`/`tables`/`evaluations`/`plots`/`exports`。
写属性 `node.property('k', v)`；Java 逃生 `node.java.…`。

## 3. 声学最小模型要点（冒烟同款，2026-11-17 实跑通过）

- 材料**必须**给 `density` + `soundspeed`，缺则求解**静默**出空解（见坑点册 §1）。
- `BackgroundPressureField`（背景平面波）配方四件套：
  `bpf.java.selection().all()`（默认空选择 → 全场 0）、`pamp`、`dir=['1','0','0']`、
  `PressureFieldType='PlaneWave'` + `k_src='FromSpeedOfSound'` + `c_mat='from_mat'`
  （后三者缺省为未定值 → 残差 NaN）。
- 频率研究 1kHz、λ=0.343m；1m×1m 域约 3λ——平面波 + 刚性壁干涉条纹即冒烟基准图。
- 网格口径：COMSOL 默认**二阶元**，每波长 ≥3 即可（线性元才要 ≥6）——
  `session.check_mesh_quality(h, λ, max_per_wavelength=3)`。

## 4. 常用属性速查（免重复自省）

| 节点 | 属性 | 取值/说明 |
|---|---|---|
| Surface 子特征 | `rangecoloractive` | `on`/`off` 开手动色标（防逐 case 漂移） |
| | `rangecolormin` / `rangecolormax` | 字符串数值（对称发散 ∓MAX） |
| | `colorlegend` | `off` = 隐藏内置 colorbar（共享外部 colorbar 时） |
| | `colorscalemode` | `linear`/`linearsymmetric`/`logarithmic` |
| PolarGroup | `axislimits` + `rmin`/`rmax` | 手动极径（多 case 统一 rmax 才能比相对能量） |
| RadiationPattern | `anglerestr` + `phimin`/`phirange` | 角域限制（度） |
| Image 导出 | 见 [export-contract.md](export-contract.md) 真源表 | 尺寸/DPI/品牌/clean |
| 材料 def 组 | `soundspeed` / `density` | 内部变量 `comp1.matX.def.c`（勿与全局参数 `c` 撞名） |
| study Frequency 步 | `plist` / `punit` | 频率列表（**不是** flist） |
| Box 选择 | `entitydim`/`condition`/`xmin…ymax` | `entitydim` 是 int，JPype 需 `JInt` 消歧 |

## 5. 命名/材料符号口径（研究联动）

- 全局参数名**不得**与材料/物理内部变量撞名：全局参数 `c` 与材料 soundspeed 内部
  `comp1.matX.def.c` 冲突 → 循环变量报错；声速全局参数用 `c0`。
- 槽介质复声速 `soundspeed = 343[m/s]*(crN + ciN*i)`（COMSOL e^{+iωt}）：
  **`ciN<0 = 增益`、`ciN>0 = 损耗`**，与④域 CMT 模块坐标 `ci` 同号同义。
  判符号用这条等式，不要"跑一版看会不会爆"试符号（反接即伪迹）。
- Array 开 `selresult=on` + `selresultshow=dom` 自动建组件级域选择
  `geom1_<arr>_dom`；材料/物理选域优先用这类 named selection（跨几何参数稳定）。
- 材料覆盖优先级：后创建者的 selection 覆盖先创建者（全域能 + 指定域的顺序）。

## 6. 参数扫掠两形态

- **研究内 Parametric 步**：`step.property('pname', ['d'])` + `plistarr` + `punit`；
  COMSOL 自动 outer 索引，`model.outer('<dataset 名>')` 读参数值。
- **Python 侧循环**（推荐批量/优化）：进程池每 worker 一个 `mph.start(cores=1)`，
  `model.parameter('d', '...')` + `model.solve()` 逐点跑（mph 官方 worker_pool demo 形态）。
  单进程单 Client 限制 → 并行靠**多进程**，不是多线程。
