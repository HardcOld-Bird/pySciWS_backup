"""④理论计算域薄胶水（skill：``theory-calculation``）。

社区四件套 SymPy / SciPy / mpmath / pint（+NumPy 基座）承载全部计算能力，
本包只留最小自研面（≤250 行，选型与裁决见
``sub_root/others/community_research/2026-11-17/domain-04-theory-selection.md``）：

- :mod:`latex_export`：SymPy 打印机输出归一化（结果→LaTeX，衔接⑤域排版链）；
- :mod:`divergence_safety`：发散安全原语（奇点感知采样 / 发散遮断 / 安全绘图）；
- :mod:`session`：会话脚手架与持久化杂件（results/ plots/ notes.md 目录约定）。

调用形态＝纯 ``uv run python <脚本>`` + 模板（theory CLI 全集已按裁决砍除）。
"""
