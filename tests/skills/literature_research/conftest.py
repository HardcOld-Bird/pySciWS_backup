"""literature_research 测试套件的共享夹具。

只放**跨文件都需要**的隔离措施；单个测试文件专用的夹具留在各自文件里。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pysci.skills.literature_research.tools import journal_metrics


@pytest.fixture(autouse=True)
def isolated_scimago_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """把 SCImago 索引指向一个必然不存在的临时路径，并返回该路径。

    ``data/skills/literature_research/data/scimago_index.json`` 是 **git 跟踪**的数据资产
    （约 1.4 MB）。用户一旦构建过它就常驻磁盘，于是所有涉及 ``scimago_quartile`` 的断言
    都会随「这台机器建没建索引」而改变结果——测试必须与用户的本地数据状态无关。

    默认状态是「索引不存在」，这同时也是生产环境里最常见的状态（降级路径）。需要真实
    索引的测试可以：

    - 用 ``journal_metrics.build_scimago_index(csv, out_path=...)`` 显式写到别处；或
    - 再次 ``monkeypatch.setattr(journal_metrics, "index_path", lambda: my_path)``
      覆盖本夹具（同一 ``monkeypatch`` 实例，后设置的生效）。
    """
    fake = tmp_path / "_no_such_scimago_index.json"
    monkeypatch.setattr(journal_metrics, "index_path", lambda: fake)
    journal_metrics._CACHE.clear()
    yield fake
    journal_metrics._CACHE.clear()
