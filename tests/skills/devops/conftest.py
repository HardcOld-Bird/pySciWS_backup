"""devops 测试夹具：把 doctor 的 git 观测面钉成「干净」。

``cmd_doctor`` 的脏跟踪面哨兵（backlog 20261012-080410-devops 方案 C）会真跑
``git status -- orchestration/state``——不隔离则断言踩在开发者工作区的瞬时状态上。
要测脏面呈现的用例自行 monkeypatch 覆写。
"""

from __future__ import annotations

import pytest

from pysci.skills.devops.tools import dev


@pytest.fixture(autouse=True)
def _clean_tracked_state(monkeypatch):
    monkeypatch.setattr(dev, "dirty_tracked_state", lambda **kw: [])
