"""自定义绘图函数"""

import matplotlib.pyplot as plt
import numpy as np
from sympy import lambdify

# 配置matplotlib中文字体支持
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "SimSun",
    "Microsoft JhengHei",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


def plot_zero_contours(
    func,
    param_x,
    param_y,
    x_range,
    y_range,
    fixed_params,
    resolution=500,
    title=None,
    levels=[0],
    colors="blue",
    linewidths=2,
):
    """
    绘制函数在二维参数空间中的零等值线。

    Parameters
    ----------
    func : sympy expression
        要绘制的符号函数
    param_x : sympy symbol
        x轴对应的参数符号
    param_y : sympy symbol
        y轴对应的参数符号
    x_range : tuple
        x轴范围 (x_min, x_max)
    y_range : tuple
        y轴范围 (y_min, y_max)
    fixed_params : dict
        固定参数的值，格式为 {symbol: value}
    resolution : int, optional
        网格分辨率，默认500
    title : str, optional
        图表标题
    levels : list, optional
        等值线的值，默认为[0]
    colors : str or list, optional
        等值线颜色，默认为'blue'
    linewidths : float or list, optional
        等值线宽度，默认为2

    Returns
    -------
    fig, ax : matplotlib figure and axes
        返回图形对象和坐标轴对象
    """
    # 获取所有参数符号
    all_params = [param_x, param_y] + list(fixed_params.keys())

    # 确定lambdify的参数顺序：先是x和y参数，然后是其他固定参数
    lambda_params = [param_x, param_y]
    fixed_values = []

    for param in all_params:
        if param not in [param_x, param_y]:
            if param not in fixed_params:
                raise ValueError(f"参数 {param} 必须在 fixed_params 中指定值")
            lambda_params.append(param)
            fixed_values.append(fixed_params[param])

    # 使用lambdify转换为数值函数
    func_numeric = lambdify(lambda_params, func, modules=["numpy"])

    # 创建二维网格
    x = np.linspace(x_range[0], x_range[1], resolution)
    y = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(x, y)

    # 计算函数值
    # 注意：X和Y是网格，fixed_values是标量
    Z = func_numeric(X, Y, *fixed_values)

    # 处理复数结果（取实部）
    if np.iscomplexobj(Z):
        print("警告：函数返回复数值，仅绘制实部")
        Z = np.real(Z)

    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 8))

    # 绘制零等值线
    contour = ax.contour(X, Y, Z, levels=levels, colors=colors, linewidths=linewidths)
    ax.clabel(contour, inline=True, fontsize=10)

    # 设置标签和标题
    ax.set_xlabel(str(param_x), fontsize=12)
    ax.set_ylabel(str(param_y), fontsize=12)
    if title:
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title(f"{func} = 0 的等值线", fontsize=14)

    ax.grid(True, alpha=0.3)
    ax.set_aspect("auto")

    plt.tight_layout()

    return fig, ax
