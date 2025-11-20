"""自定义数据类型"""

from dataclasses import dataclass
from typing import Any

from sympy import Symbol


@dataclass(frozen=True)
class ParamSpace3D:
    """三维参数空间"""

    x: Symbol
    x_range: tuple[float, float]
    y: Symbol
    y_range: tuple[float, float]
    z: Symbol
    z_range: tuple[float, float]
    x_resolution: int = 50
    y_resolution: int = 50
    z_resolution: int = 50
    fixed_param: dict[Symbol, Any] = None
