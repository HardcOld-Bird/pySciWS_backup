"""自定义数据类型"""

from dataclasses import dataclass, field
from typing import Any

from sympy import Symbol


@dataclass(frozen=True)
class ParamSpace3D:
    """三维参数空间定义。

    用于定义符号计算中的三维参数空间，包含三个参数的符号、取值范围、
    分辨率以及可选的固定参数。主要用于数值化和可视化。

    Attributes:
        x: 第一个参数的符号。
        x_range: 第一个参数的取值范围 (最小值, 最大值)。
        y: 第二个参数的符号。
        y_range: 第二个参数的取值范围 (最小值, 最大值)。
        z: 第三个参数的符号。
        z_range: 第三个参数的取值范围 (最小值, 最大值)。
        x_resolution: 第一个参数的分辨率（网格点数），默认为 50。
        y_resolution: 第二个参数的分辨率（网格点数），默认为 50。
        z_resolution: 第三个参数的分辨率（网格点数），默认为 50。
        fixed_param: 固定参数字典，键为符号，值为数值，默认为空字典。

    Examples:
        >>> from sympy import symbols
        >>> x, y, z, a = symbols('x y z a')
        >>> param_space = ParamSpace3D(
        ...     x=x, x_range=(0, 1),
        ...     y=y, y_range=(0, 2),
        ...     z=z, z_range=(0, 3),
        ...     fixed_param={a: 1.0}
        ... )
    """

    x: Symbol
    x_range: tuple[float, float]
    y: Symbol
    y_range: tuple[float, float]
    z: Symbol
    z_range: tuple[float, float]
    x_resolution: int = 50
    y_resolution: int = 50
    z_resolution: int = 50
    fixed_param: dict[Symbol, Any] = field(default_factory=dict)
