"""pandoc 3.14 subprocess 包装（薄层，≈50 行预算）。

ADR（2026-11-17）：**不引** pypandoc / pypandoc-binary（差②，且只是 pandoc 的 pip
壳）——直接 subprocess 调 vendored ``pandoc.exe``（路径见 ``toolpaths``）。本模块只做
参数拼装与调用；能力语义以 pandoc 手册为准（互转枢纽、docx/xlsx/pptx reader、
``--citeproc`` 引用）。pandoc **不能**增量追加既有 Office 文件——追加/编辑走 DSH
office 技能（office-docx / office-pptx），见技能文档任务路由表。
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from pysci.skills.document_writing.toolpaths import pandoc_exe


class PandocError(RuntimeError):
    """pandoc 调用失败（非零退出码），携带 stdout/stderr 便于定位。"""


def build_command(
    input_file: str | Path,
    *,
    to: str = "docx",
    output: str | Path | None = None,
    reference_doc: str | Path | None = None,
    slide_level: int | None = None,
    toc: bool = False,
    number_sections: bool = False,
    citeproc: bool = False,
    bib: str | Path | None = None,
    csl: str | Path | None = None,
    resource_path: str | Path | None = None,
    standalone: bool = False,
) -> list[str]:
    """拼装 pandoc 命令行（纯函数，不执行）。

    Args:
        input_file: 输入文档（MD 或 pandoc 支持的任何 reader 格式）。
        to: 目标格式（``docx`` / ``pptx`` / ``latex`` / ``html`` / ``odt`` …）。
        output: 输出文件；None 时由 pandoc 写 stdout（调用方自行捕获）。
        reference_doc: ``--reference-doc`` 家风样式模板（docx/pptx）。
        slide_level: ``--slide-level``（pptx 分页层级）。
        toc: 是否 ``--toc``。
        number_sections: 是否 ``--number-sections``。
        citeproc: 是否 ``--citeproc``（引文排版；需配 bib/csl）。
        bib: 参考文献库（refs.bib）；传入即自动加 ``--bibliography``。
        csl: CSL 样式文件；传入即自动加 ``--csl``。
        resource_path: ``--resource-path``（图片等资源检索路径）。
        standalone: 是否 ``--standalone``。

    Returns:
        可直接交给 :func:`subprocess.run` 的参数列表（argv）。

    Raises:
        ValueError: citeproc 与 bib 同时缺失（bib 是 citeproc 的数据源）这类
            明显不自洽的参数组合。
    """
    if citeproc and bib is None:
        raise ValueError("citeproc 需要 --bib 提供参考文献库（refs.bib）")
    cmd = [pandoc_exe(), str(input_file), "-t", to]
    if output is not None:
        cmd += ["-o", str(output)]
    if reference_doc is not None:
        cmd += ["--reference-doc", str(reference_doc)]
    if slide_level is not None:
        cmd += ["--slide-level", str(slide_level)]
    if toc:
        cmd.append("--toc")
    if number_sections:
        cmd.append("--number-sections")
    if bib is not None:
        cmd += ["--bibliography", str(bib)]
    if csl is not None:
        cmd += ["--csl", str(csl)]
    if citeproc:
        cmd.append("--citeproc")
    if resource_path is not None:
        cmd += ["--resource-path", str(resource_path)]
    if standalone:
        cmd.append("--standalone")
    return cmd


def convert(
    input_file: str | Path,
    *,
    to: str = "docx",
    output: str | Path | None = None,
    timeout: float = 300.0,
    **kwargs: object,
) -> Path | None:
    """执行一次 pandoc 转换。

    Args:
        input_file: 输入文档。
        to: 目标格式。
        output: 输出文件；None 时写 stdout 且返回 None。
        timeout: 超时秒数。
        **kwargs: 透传给 :func:`build_command` 的其余参数。

    Returns:
        输出文件路径（:class:`Path`）；``output=None`` 时返回 None。

    Raises:
        PandocError: pandoc 非零退出或超时。
    """
    cmd = build_command(input_file, to=to, output=output, **kwargs)  # type: ignore[arg-type]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise PandocError(f"pandoc 超时（{timeout}s）：{' '.join(cmd)}") from exc
    if result.returncode != 0:
        raise PandocError(
            f"pandoc 失败（exit {result.returncode}）：{' '.join(cmd)}\n{result.stderr}"
        )
    return Path(output) if output is not None else None
