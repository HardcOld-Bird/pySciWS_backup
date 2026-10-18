"""literature_research 测试套件的共享夹具。

只放**跨文件都需要**的隔离措施；单个测试文件专用的夹具留在各自文件里。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pysci.skills.literature_research.tools import journal_metrics


@pytest.fixture(autouse=True)
def offline_curl_direct(monkeypatch: pytest.MonkeyPatch) -> None:
    """默认关闭 curl 直链（测试必须离线）。

    ``read`` 会先试出版商直链（curl 优先）；不隔离就会在涉及直链的用例里真联网。
    需要验证 curl 分支的用例用模块级保存的原函数覆盖回本桩（见 test_read_fetch_order）。
    """
    from pysci.skills.literature_research.tools import research

    monkeypatch.setattr(research, "_curl_download_pdf", lambda *a, **k: None)


@pytest.fixture(autouse=True)
def offline_oa_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """默认关闭 OA 回退的外部查询（S2）：测试必须离线、确定。

    ``research.read`` 在抓取失败后会调用 ``oa_fallback.try_oa`` → Semantic Scholar。
    不隔离就会在「抽取失败」类用例里真联网取回全文，把 rc=1 断言翻成 0（实测）。
    需要验证回退链的用例自行 monkeypatch ``semantic_scholar``／``_download_pdf`` 等
    （后设置的 monkeypatch 覆盖本夹具）。
    """
    from pysci.skills.literature_research.tools import oa_fallback

    monkeypatch.setattr(oa_fallback, "semantic_scholar", lambda *a, **k: {})


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
