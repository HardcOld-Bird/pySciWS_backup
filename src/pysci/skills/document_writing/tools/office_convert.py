"""office_convert —— LibreOffice headless 格式转换（pptx/docx/odt ↔ pdf 等）。

用于「必须交付 pdf 的 pptx/docx」或「把 office 文件转成可阅读格式」的场景。
LibreOffice 非 pip 依赖，需另行安装；缺失时给出清晰指引而非裸报错。
阅读/提取/LaTeX 编译均**不**依赖本模块。
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from .config import settings


class LibreOfficeNotInstalled(RuntimeError):
    """未检测到 LibreOffice（soffice）。"""

    def __str__(self) -> str:  # noqa: D105
        return (
            "未检测到 LibreOffice（soffice）。\n"
            "格式转换（pptx/docx → pdf 等）需要 LibreOffice headless。\n"
            "安装：https://www.libreoffice.org/download/download-libreoffice/ \n"
            "安装后重开终端跑 `compose doctor` 复核。\n"
            "提示：阅读/提取（read/slides/docx）与 LaTeX 编译不依赖 LibreOffice，可照常使用。"
        )


def convert(
    src: str | Path,
    to: str,
    *,
    out_dir: str | Path | None = None,
    timeout: int = 300,
) -> Path:
    """把 src 转换为 to 格式（如 pdf/docx/pptx/html），返回输出文件路径。

    Args:
        src: 输入文件（pptx/docx/odt/xlsx/...）。
        to: 目标扩展名（不含点），如 "pdf"。
        out_dir: 输出目录；默认与 src 同目录。
        timeout: 转换超时秒数。
    """
    src = Path(src).resolve()
    if not src.exists():
        raise FileNotFoundError(f"输入文件不存在：{src}")
    soffice = settings.find_libreoffice()
    if not soffice:
        raise LibreOfficeNotInstalled()

    out_dir = Path(out_dir) if out_dir else src.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        soffice,
        "--headless",
        "--norestore",
        "--convert-to",
        to,
        "--outdir",
        str(out_dir),
        str(src),
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"转换超时（>{timeout}s）：{src.name}") from None
    except FileNotFoundError as e:
        raise LibreOfficeNotInstalled() from e

    expected = out_dir / f"{src.stem}.{to}"
    if proc.returncode != 0 or not expected.exists():
        raise RuntimeError(
            f"转换失败（rc={proc.returncode}）：{(proc.stdout or '') + (proc.stderr or '')}".strip()
        )
    return expected
