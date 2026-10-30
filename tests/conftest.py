"""pytest 会话级环境：把 vendored 工具目录前插 PATH。

自包含改造（2026-10-25）后系统不再有全局 git/node/uv：测试与第三方工具
（pre-commit 等）里裸 ``node``/``git``/``bash`` 的 subprocess 调用靠 PATH 解析
**真 .exe**（``.cmd`` shim 对 CreateProcess 不可见，解析即 FileNotFoundError）。
前插 ``sub_root/dependencies`` 下的真实工具目录即全量恢复，无须逐测试改写。
"""

from __future__ import annotations

import os

from pysci.toolchain import prepend_toolchain_path

os.environ["PATH"] = prepend_toolchain_path()
