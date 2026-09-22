"""gain_ep 理论层测试收集守卫。

本目录下的 ``test_10in16out_function.py`` / ``test_gain_ep_functions.py`` 并非 pytest
单元测试，而是**在模块级直接连接 COMSOL Server 跑仿真**的手动脚本（无 ``test_*`` 函数、
无断言、无 ``__main__`` 守卫）。pytest 收集时会 import 它们，从而在导入阶段即触发 COMSOL
连接——在没有运行 COMSOL Server 的环境下会报错甚至挂起，拖垮整轮 ``uv run pytest``。

因此默认忽略这两个文件的收集。需要手动运行（且已启动 COMSOL Server）时，直接执行::

    python tests/gain_ep/theory/test_gain_ep_functions.py
    python tests/gain_ep/theory/test_10in16out_function.py

日后在本目录新增**真正的** pytest 理论单元测试（安全 import + mock COMSOL）时不受此守卫
影响——``collect_ignore`` 只精确排除上述两个脚本，而非整个目录。
"""

# 模块级连接 COMSOL 的手动脚本，非 pytest 测试，默认不收集（详见本文件 docstring）。
collect_ignore = [
    "test_10in16out_function.py",
    "test_gain_ep_functions.py",
]
