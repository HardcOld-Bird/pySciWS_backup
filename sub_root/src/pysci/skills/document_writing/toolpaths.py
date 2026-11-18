"""pandoc / TeX 工具链路径常量（⑤写作域）。

ADR（2026-11-17，原地记录，定稿）——为什么不入 ``toolchain.json``：
``toolchain.json`` 由 deps_manager 按**固定五条目**消费（git/node/uv/python/dsh，
见 ``init-deps.ps1`` 的 ``Install-Tool`` 调用与 ``update-deps.ps1`` 的 ``Row`` 渲染），
schema 无自定义条目消费者：硬塞 pandoc/TeX 条目不会被安装、校验或在 status 中显示，
只是死数据。且两者不满足 ``{version,url,sha256}`` 归档安装模型——pandoc 是手工解包的
自包含 exe，TeX Live 是目录 junction（→ ``D:\\XiGPrograms\\tex``，物理迁移待用户关闭
占用进程，见 ``others/plans/domains-01-05-text-chain.md`` 步骤 3），无单一归档可校验。
故**不登记**；本模块即 pandoc/TeX Live 的路径与版本注释真源（toolchain.json 继续做
git/node/uv/python/dsh 五件套的版本真源）。禁止在业务模块手写 ``Path(__file__).parents[N]``
或散落路径拼接。

来源（既有落盘位置，2026-11-17 核实）：

- pandoc：官方 zip 解包自包含 exe，GPL-2.0，
  ``deps/tools/pandoc/pandoc-3.14/pandoc.exe``。
- TeX：``deps/tex`` 是目录 **junction** → ``D:\\XiGPrograms\\tex``（物理迁移待用户
  关闭占用进程，见 ``others/plans/domains-01-05-text-chain.md`` 步骤 3）；TeX Live
  2026 工具在 ``deps/tex/base/texlive/2026/bin/windows/``（latexmk/chktex/pdflatex/
  xelatex/lualatex 均已实证可用）。

解析口径与 ``pysci.toolchain`` 一致：fail-open——落盘缺失时回退裸命令名（交 PATH
解析），不在解析层硬失败；缺失状态由 ``pysci-doc doctor`` 暴露。
"""

from __future__ import annotations

from pathlib import Path

from pysci.toolchain import DEPS_ROOT

#: pandoc 可执行文件（vendored 自包含 exe；GPL-2.0）。
PANDOC_EXE: Path = DEPS_ROOT / "tools" / "pandoc" / "pandoc-3.14" / "pandoc.exe"

#: TeX Live 2026 bin 目录（经 junction ``deps/tex``；见模块 docstring）。
TEXLIVE_BIN: Path = DEPS_ROOT / "tex" / "base" / "texlive" / "2026" / "bin" / "windows"


def _exe(path: Path, fallback: str) -> str:
    """落盘返回绝对路径字符串，缺失回退裸命令名（fail-open，见模块 docstring）。"""
    return str(path) if path.is_file() else fallback


def pandoc_exe() -> str:
    """pandoc 可执行文件：vendored 绝对路径，缺失回退 ``"pandoc"``。"""
    return _exe(PANDOC_EXE, "pandoc")


def tex_exe(name: str) -> str:
    """TeX Live 工具可执行文件（``latexmk`` / ``chktex`` / ``pdflatex`` …）。

    Args:
        name: 工具裸名（不含 ``.exe``）。

    Returns:
        绝对路径字符串；落盘缺失回退裸命令名（交 PATH 解析）。
    """
    return _exe(TEXLIVE_BIN / f"{name}.exe", name)


def latexmk_exe() -> str:
    """latexmk（编译驱动），缺失回退 ``"latexmk"``。"""
    return tex_exe("latexmk")


def chktex_exe() -> str:
    """chktex（LaTeX 静态检查），缺失回退 ``"chktex"``。"""
    return tex_exe("chktex")
