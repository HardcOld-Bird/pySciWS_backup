"""ai_drawing 技能测试包。

- 纯 Python 单测（config 目录/脱敏、ark_client 请求体护栏与响应解析、postprocess Pillow
  操作、ledger 记账/查询/元数据内嵌、bridge 抽色/脚手架、imagine CLI 接线、paths 锚点）
  不联网、不需方舟 key，随 ``uv run pytest -m "not hardware"`` 始终运行。
- 真实云端出图（会花钱）标记 ``hardware`` + ``slow``，经 ``conftest.py`` 的 ``ark_key``
  fixture 在无 key 时 ``pytest.skip``；依赖 optional extra 的位图算子用例在 ``test_imaging.py``
  里按库分别 ``skipif``（对齐 comsol conftest，默认全绿）。
"""
