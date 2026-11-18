"""pytest 会话级环境垫片：Windows 受限令牌下的 pytest 临时目录兼容。

机制（2026-11-16 一行最小复现确认）：
- pytest 9 出于安全加固，用 `Path.mkdir(mode=0o700)` 创建它的全部临时目录
  （`_pytest/tmpdir.py`、`_pytest/pathlib.py` 的 `make_numbered_dir`）。
- 本机 CPython 3.13.15 的 Windows 实现会在**创建时**把 `mode=0o700` 综合成
  「仅属主」的受保护权限表（继承被阻断）。DSH 沙盒的受限令牌无法受益于
  属主权利，于是刚创建的目录对创建进程自己就完全不可访问——所有 `tmp_path`
  夹具、pytest 自身的 basetemp 扫描与收尾清理一律 EACCES。
- 对照实验：`os.mkdir(p)`（默认 0o777）与事后 `os.chmod(p, 0o700)` 均无害，
  只有创建时的 mode 有害。因此这里只中和 `os.mkdir` 的 mode，让目录走正常
  继承（沙盒授权项可继承进去），受限与不受限运行都恢复正常。

只在 win32 生效；极简口径：QA 套件内不测试「mode 必须生效」语义，故不保留回退。
"""

from __future__ import annotations

import os
import sys

if sys.platform == "win32":
    _os_mkdir = os.mkdir

    def _mkdir_neutralize_mode(path, mode=0o777, *, dir_fd=None):
        """以继承权限创建目录：忽略 mode（见模块 docstring 的机制说明）。"""
        return _os_mkdir(path, 0o777, dir_fd=dir_fd)

    os.mkdir = _mkdir_neutralize_mode
