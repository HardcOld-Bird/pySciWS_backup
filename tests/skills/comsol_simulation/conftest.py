"""comsol_simulation 测试的共享 fixture 与 hardware 守卫。

约定（对齐 ``tests/gain_ep/experiment/conftest.py``，但**不**整目录忽略收集）：
- 纯 Python 单测不依赖 COMSOL/mph，始终运行——因此这里不能用 ``collect_ignore_glob``。
- 需要活体 COMSOL 的 hardware 冒烟用例通过 ``comsol_client`` / ``smoke_mph`` fixture 守卫：
  mph 不可导入、未发现 COMSOL 安装、或冒烟模型缺失时 ``pytest.skip``（而非 collection error）。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pysci.paths import PROJECT_ROOT
from pysci.skills.comsol_simulation.tools import session

#: Phase 0 冒烟模型（相对项目根）。缺失则 hardware 用例 skip。
SMOKE_MPH_RELPATH = Path(
    "data/research/1_gain_ep/simulation/4 pySweep增益系数探究用模型备份.mph"
)


def comsol_available() -> bool:
    """mph 可导入且发现了可用 COMSOL 安装（不启动 JVM、不占 license）。"""
    return session.is_available()


@pytest.fixture(scope="session")
def smoke_mph() -> Path:
    """冒烟模型绝对路径；环境或模型缺失时 skip。"""
    if not comsol_available():
        pytest.skip("需要 mph + 可用 COMSOL 安装（hardware）")
    p = PROJECT_ROOT / SMOKE_MPH_RELPATH
    if not p.exists():
        pytest.skip(f"冒烟模型缺失：{p}")
    return p


@pytest.fixture(scope="session")
def comsol_client():
    """会话级 standalone COMSOL 客户端（一次 JVM 冷启动，供 hardware 用例共享）。"""
    if not comsol_available():
        pytest.skip("需要 mph + 可用 COMSOL 安装（hardware）")
    with session.session(mode="standalone", cores=2) as client:
        yield client
