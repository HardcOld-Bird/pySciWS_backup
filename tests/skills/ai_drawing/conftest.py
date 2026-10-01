"""ai_drawing 测试的共享 fixture。

约定：
- 纯 Python 单测不联网、不需要火山方舟 key，始终运行——因此这里**不**用
  ``collect_ignore_glob`` 整目录忽略收集。云端调用的请求体构造、护栏校验、响应解析
  全部走纯函数（``ark_client.build_payload`` / ``_parse_response`` 等），离线即可覆盖。
- 只有**真实出图**（会花钱）的用例才需要 key，通过 ``ark_key`` fixture 守卫：未配置时
  ``pytest.skip``（而非 collection error）。这类用例另标 ``hardware`` + ``slow``，
  默认 ``-m "not hardware"`` 下不收集。
- 依赖 optional extra ``imaging``（scikit-image + OpenCV）的用例，在 ``test_imaging.py`` 里
  按**库分别**用 ``pytest.mark.skipif`` 守卫（只装了其中一个时，另一侧的用例仍能跑）。
"""

from __future__ import annotations

import pytest

from pysci.skills.ai_drawing.tools.config import settings


@pytest.fixture(scope="session")
def ark_key() -> str:
    """火山方舟 API Key；未配置时 skip。

    ``.env`` 的 ``ARK_API_KEY`` 既是门槛也是实际调用凭据（重构后不再有中间层代管密钥）。
    只有 ``-m hardware`` 的真实出图用例才会用到本 fixture。
    """
    if not settings.ark_ready:
        pytest.skip("需要 ARK_API_KEY（真实云端生成会消耗额度；hardware）")
    return settings.ark_api_key or ""
