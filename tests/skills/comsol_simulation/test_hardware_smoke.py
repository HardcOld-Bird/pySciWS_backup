"""hardware 冒烟：一条 COMSOL 会话跑通闭环（对齐计划的 Phase 0 门槛）。

加载 pySweep 模型 → dump_tree → introspect 活体 Java 节点（活体 javadoc）→ 幂等改一参 →
求解（默认 study）→ 导出 PNG（绘图组）+ CSV（数据集）→ 读回校验产物非空。

需要 mph + 可用 COMSOL 安装 + 冒烟模型；任一缺失由 conftest 的 fixture 自动 skip。
标记 hardware + slow：仅在 ``uv run pytest -m hardware`` 时运行（会启动 JVM、占 license）。
"""

from __future__ import annotations

import pytest

pytest.importorskip("mph", reason="需要 mph（COMSOL Java 桥）")

from pysci.skills.comsol_simulation.tools import (  # noqa: E402
    build,
    export,
    inspect,
    postprocess as pp,
    run,
)

pytestmark = [pytest.mark.hardware, pytest.mark.slow]

#: Phase 0 已验证的绘图组 tag（其底层数据集由 export_data 自动解析）
_PLOTGROUP = "pg10"


def test_closed_loop_smoke(comsol_client, smoke_mph, tmp_path):
    model = comsol_client.load(str(smoke_mph))

    # 1) 读结构 + 参数清单
    tree = inspect.dump_tree(model)
    assert "Model Tree" in tree
    params = inspect.list_parameters(model)
    assert isinstance(params, dict)

    # 2) introspect 活体 Java 节点 = 活体 javadoc（类名 + 方法签名）
    info = inspect.introspect(model.java)
    assert info["class"]
    assert len(info["methods"]) > 0
    assert inspect.format_introspection(info)

    # 3) 幂等改一参：验证参数写入路径可用且不破坏模型
    if params:
        name = next(iter(params))
        build.set_parameters(model, {name: params[name]["value"]})

    # 4) 求解（默认 study；对齐 simulator.py 里验证过的 model.solve()）
    sr = run.solve(model)
    assert sr.ok, sr.report()
    assert sr.elapsed >= 0.0

    # 5) 导出 PNG（COMSOL 原生渲染）——必须远大于空图基线（~14KB）
    png = tmp_path / "pg10.png"
    ir = export.export_image(model, _PLOTGROUP, png)
    assert ir.ok, ir.report()
    assert png.exists() and png.stat().st_size > 15_000

    # 6) 导出 CSV（数据集坐标）并读回解析
    csv = tmp_path / "pg10.csv"
    dr = export.export_data(model, _PLOTGROUP, csv)
    assert dr.ok, dr.report()
    table = pp.read_comsol_csv(csv)
    assert table.shape[0] > 0
    assert "X" in table.columns and "Y" in table.columns
