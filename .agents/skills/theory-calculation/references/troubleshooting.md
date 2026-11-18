# 故障手册（troubleshooting）

> 旧项目"When something breaks"要点 + 本轮选型补充；按症状索引。

## simplify 大式超时/挂死

SymPy `sp.simplify` 在大型表达式上组合爆炸。对策（按序）：
1. 先 `sp.expand` / `sp.cancel` / `sp.together` / `sp.collect` 等定向化简，不做全量 simplify；
2. 只对关键子式 simplify，其余保持展开态传递；
3. 传 `ratio=1.7` 之类保守参数或干脆跳过化简——数值验证不依赖化简形态。
   （旧项目 simplify strategy 路由器已砍：选择权在物理脚本，不做自动路由。）

## lambdify 产出 NaN

表达式含 `sqrt`/`log`/`**` 复分支时，NumPy 实数分支在负输入处给 NaN。
对策：数值化时投复分支——`np.sqrt(x + 0j)`、`np.log(x + 0j)`，或整体
`f = sp.lambdify(..., "numpy")` 后对输入数组 `x.astype(complex)`。
另查：除零/溢出（用 `divergence_safety.mask_divergent` 遮断后再看图）、
0 维数组当标量（`float(...)` 包一层）。

## 本征值排序看起来"乱跳"

每点独立排序会在交叉处产生伪影。用相位连续性分支追踪
（[ep-topology-recipes.md](ep-topology-recipes.md) §2），实二支可退化为 min/max。

## 数值大矩阵/刚性 ODE 很慢

SymPy 不是干这个的：`numpy.linalg.eig`（稠密）/`scipy.sparse.linalg.eigs`（稀疏）
接手；刚性 ODE 用 `solve_ivp(method="BDF"/"Radau")`；超纲见
[scipy-boundaries.md](scipy-boundaries.md)。

## 高精度结果与 float64 对不上

mpmath `workdps(50)` 复核关键数值；注意 `evalf` 的 dps 是**十进制位数**不是二进制位；
`lambdify` 结果是 float64——高精度链路全程 mpmath/SymPy `evalf`，勿混用。

## pint 报量纲不一致

SymPy 符号不带单位：量纲检查只能在 pint 数值侧做。把关键中间量先赋 `Quantity`
再换算/相加；符号推导里的单位自行在文档/符号名标注（如 `omega_Hz`）。

## Windows 终端中文乱码

PowerShell 输出中文时按 `.agents` 工作约定设置代码页/编码（旧项目 basic 规则 §3）；
脚本内写文件一律显式 `encoding="utf-8"`。

## 表达式打印出来不是想要的 LaTeX

打印器设置已被 `latex_export` 固定（确定性）；要不同形态（如不折叠短分式）用
`expr_to_latex(..., fold_short_frac=False)`；矩阵定界用 `matrix_to_latex(env=...)`。
需要 `equation`/`align` 等环境时由⑤域模板包裹，胶水只出环境体。
