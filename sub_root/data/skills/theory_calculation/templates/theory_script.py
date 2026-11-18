# 理论计算脚手架模板（④理论计算域）。
#
# 用法（调用形态＝纯 uv run python，无 CLI，用户裁决 2026-11-17）：
#   1. 复制本文件到 src/pysci/research/<name>/theory/<slug>.py；
#   2. 按七步工作流填入物理内容：符号建模→化简/方程→数值验证→高精度→量纲检查→
#      LaTeX 导出→⑤域排版（pysci-doc latex build / formula）；
#   3. 运行：sub_root> uv run python src/pysci/research/<name>/theory/<slug>.py
#
# 本模板在仓库内直接运行即是一条最小冒烟链（符号求导→数值验证→LaTeX 导出→落盘）。
from __future__ import annotations

import sympy as sp

from pysci.skills.theory_calculation import latex_export, session

# ── 1. 符号建模（SymPy 直调，零胶水）────────────────────────────────
x = sp.symbols("x", real=True)
expr = sp.sin(x) * sp.exp(-x)

# ── 2. 化简/方程（大式 simplify 超时见 references/troubleshooting.md）──
d_expr = sp.diff(expr, x)

# ── 3. 数值验证（SciPy/NumPy 直调）──────────────────────────────────
import numpy as np  # noqa: E402

f_num = sp.lambdify(x, d_expr, modules="numpy")  # NaN 排查见 +0j 技巧
xs = np.linspace(0.0, 5.0, 6)
ys = f_num(xs)

# ── 4/5. 高精度（mpmath.workdps）/ 量纲（pint Quantity）────────────
# 数值关键点高精度复核与单位检查按需插入；两者均为社区库直调，无胶水。

# ── 6. LaTeX 导出（归一化，确定性输出）──────────────────────────────
latex = latex_export.expr_to_latex(d_expr)
print("d/dx:", latex)

# ── 7. 落盘 + ⑤域排版（把 latex 贴进 amsmath/MD 公式）───────────────
sess = session.ensure_session("example", "demo")  # data/research/example/theory/demo/
session.save_results(sess, {"x": xs, "dy": ys}, name="profile")
session.append_note(sess, f"求导结果：${latex}$")
print("session:", sess)
