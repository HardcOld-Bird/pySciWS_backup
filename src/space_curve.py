"""空间曲线可视化函数"""

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import pyvista as pv
from numpy.typing import NDArray

from src.my_dtypes import ParamSpace3D

if TYPE_CHECKING:
    pass


def vis_complex_equation(
    numpy_func: Callable,
    param_space: ParamSpace3D,
    plot: bool = False,
) -> tuple[pv.PolyData, pv.PolyData, pv.PolyData]:
    """可视化复变恒等式的零点集（在三维参数空间中的空间曲线）。

    该函数通过计算复变表达式在三维参数空间中的零点集来可视化空间曲线。
    零点集定义为同时满足 Re(expr) = 0 和 Im(expr) = 0 的点集。

    Args:
        numpy_func: numpy数值函数。
        param_space: 三维参数空间定义，包含参数符号、范围和分辨率。
        plot: 是否绘制图像，默认为 False。

    Returns:
        包含三个元素的元组：
            - curve: 零点集的交线（两个等值面的交集）。
            - first_face: 实部为0的等值面。
            - second_face: 虚部为0的等值面。

    Examples:
        >>> numpy_func = get_numpy_func(expr, param_space)  # noqa
        >>> curve, first_face, second_face = vis_complex_equation(numpy_func, param_space)  # noqa
    """
    # 计算参数空间的长度，备用
    x_len: float = param_space.x_range[1] - param_space.x_range[0]
    y_len: float = param_space.y_range[1] - param_space.y_range[0]
    z_len: float = param_space.z_range[1] - param_space.z_range[0]

    # 创建三维网格
    x: NDArray[np.floating] = np.linspace(
        param_space.x_range[0],
        param_space.x_range[1],
        param_space.x_resolution,
    )
    y: NDArray[np.floating] = np.linspace(
        param_space.y_range[0],
        param_space.y_range[1],
        param_space.y_resolution,
    )
    z: NDArray[np.floating] = np.linspace(
        param_space.z_range[0],
        param_space.z_range[1],
        param_space.z_resolution,
    )

    # 使用 meshgrid 创建三维坐标数组
    coord_x_mat: NDArray[np.floating]
    coord_y_mat: NDArray[np.floating]
    coord_z_mat: NDArray[np.floating]
    coord_x_mat, coord_y_mat, coord_z_mat = np.meshgrid(x, y, z, indexing="ij")

    # 计算表达式的数值结果（复数）
    # 使用 numpy 的通用函数进行高效计算
    values: NDArray[np.complexfloating] = numpy_func(
        coord_x_mat, coord_y_mat, coord_z_mat
    )

    # 提取实部和虚部
    real_part: NDArray[np.floating] = np.real(values)
    imag_part: NDArray[np.floating] = np.imag(values)

    # 创建 PyVista 的 ImageData（结构化网格）
    grid: pv.ImageData = pv.ImageData()
    grid.dimensions = (
        param_space.x_resolution,
        param_space.y_resolution,
        param_space.z_resolution,
    )
    grid.origin = (
        param_space.x_range[0],
        param_space.y_range[0],
        param_space.z_range[0],
    )
    grid.spacing = (
        x_len / (param_space.x_resolution - 1),
        y_len / (param_space.y_resolution - 1),
        z_len / (param_space.z_resolution - 1),
    )

    # 将实部和虚部添加为标量场
    grid["real"] = real_part.flatten(order="F")  # Fortran order for VTK
    grid["imag"] = imag_part.flatten(order="F")

    # 使用 contour 滤波器提取实部为0的等值面
    real_contour: pv.PolyData = grid.contour(isosurfaces=[0], scalars="real")

    # 使用 contour 滤波器提取虚部为0的等值面
    imag_contour: pv.PolyData = grid.contour(isosurfaces=[0], scalars="imag")

    # 计算两个等值面的交线
    curve: pv.PolyData
    first_face: pv.PolyData
    second_face: pv.PolyData
    curve, first_face, second_face = real_contour.intersection(
        imag_contour, split_first=False, split_second=False
    )

    # 如果需要绘图
    if plot:
        plotter: pv.Plotter = pv.Plotter()

        # 使用参数空间的范围作为边界框，而不是curve的边界
        # 这样可以显示curve在参数空间中的真实位置
        max_len = max(x_len, y_len, z_len)

        # 设置缩放比例，使得最长的轴为1，其他轴按比例缩放
        # 这样可以避免视图平面法向量平行的警告
        plotter.set_scale(
            xscale=max_len / x_len if x_len != 0 else 1,
            yscale=max_len / y_len if y_len != 0 else 1,
            zscale=max_len / z_len if z_len != 0 else 1,
        )

        # 添加curve
        plotter.add_mesh(
            curve,
            color="red",
            line_width=3,
            label="零点集",
        )

        # 创建一个不可见的边界框，用于固定显示范围为参数空间
        # 这样即使curve很小或偏在一角，也能看到完整的参数空间
        bounds_box = pv.Box(
            bounds=(
                param_space.x_range[0],
                param_space.x_range[1],
                param_space.y_range[0],
                param_space.y_range[1],
                param_space.z_range[0],
                param_space.z_range[1],
            )
        )
        plotter.add_mesh(bounds_box, opacity=0.0)  # 完全透明，仅用于设置边界

        # 添加带有真实参数值的坐标轴
        plotter.show_bounds(
            grid="back",
            location="outer",
            ticks="both",
            xtitle=str(param_space.x),
            ytitle=str(param_space.y),
            ztitle=str(param_space.z),
            font_size=10,
            color="black",
        )

        plotter.add_legend()
        plotter.add_text(
            f"复变恒等式零点集\n参数: {param_space.x}, {param_space.y}, {param_space.z}",
            position="upper_left",
            font_size=10,
        )
        plotter.show()

    return curve, first_face, second_face
