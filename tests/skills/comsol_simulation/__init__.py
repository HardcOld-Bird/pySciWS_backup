"""comsol_simulation 技能测试包。

- 纯 Python 单测（config 发现 / docs FTS5 索引 / postprocess 数学与 CSV / build 注册表）
  不依赖 COMSOL，随 ``uv run pytest -m "not hardware"`` 运行。
- hardware 冒烟（活体 COMSOL 闭环）标记 ``hardware`` + ``slow``，仅在 ``-m hardware`` 时运行。
"""
