"""office_convert —— office 格式转换（**WPS wpscli 优先，LibreOffice 兜底**）。

用户明示「本机无 MS Office，WPS 为主力」——WPS 渲染≈MS Office 口径，故转换链默认先试
新版 WPS 的 ``wpscli.exe``（插件式，``ppt2pdf``/``word2pdf``），不可用或转换失败再回落
LibreOffice headless（backlog 20261017-105200）。阅读/提取/LaTeX 编译**不**依赖本模块。
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .config import settings

#: wpscli 子命令：目标格式 × 源后缀。
_WPS_SUB = {
    "pdf": {"pptx": "ppt2pdf", "ppt": "ppt2pdf", "docx": "word2pdf", "doc": "word2pdf"},
}


class LibreOfficeNotInstalled(RuntimeError):
    """WPS wpscli 与 LibreOffice（soffice）均未检测到。"""

    def __str__(self) -> str:  # noqa: D105
        return (
            "未检测到可用的 office 转换后端（wpscli / soffice）。\n"
            "格式转换（pptx/docx → pdf 等）需要 WPS 12.1+ 的 clitool/wpscli.exe，或 LibreOffice headless。\n"
            "安装后重开终端跑 `compose doctor` 复核。\n"
            "提示：阅读/提取（read/slides/docx）与 LaTeX 编译不依赖它们，可照常使用。"
        )


def _stamp(p: Path) -> int | None:
    try:
        return p.stat().st_mtime_ns if p.exists() else None
    except OSError:
        return None


def _run(cmd: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )


def _try_wpscli(
    src: Path,
    to: str,
    out_dir: Path,
    expected: Path,
    timeout: int,
    before: int | None,
    errors: list[str],
) -> bool:
    """试 wpscli；成功返回 True。它**没有 --outdir**：源目录产出后按需搬到 out_dir。"""
    exe = settings.find_wpscli()
    sub = _WPS_SUB.get(to.lower(), {}).get(src.suffix.lower().lstrip("."))
    if not exe:
        errors.append("wpscli 未安装")
        return False
    if not sub:
        errors.append(f"wpscli 不支持 {src.suffix}→{to}")
        return False
    try:
        proc = _run([exe, sub, str(src)], timeout)
    except subprocess.TimeoutExpired:
        errors.append(f"wpscli 超时（>{timeout}s）")
        return False
    except OSError as e:
        errors.append(f"wpscli 启动失败：{e}")
        return False
    cands = (
        [expected]
        if expected == src.parent / expected.name
        else [
            expected,
            src.parent / expected.name,
        ]
    )
    produced = next((c for c in cands if c.exists() and _stamp(c) != before), None)
    if proc.returncode != 0 or produced is None:
        errors.append(f"wpscli rc={proc.returncode} 未产出 {expected.name}")
        return False
    if produced != expected:
        shutil.move(str(produced), str(expected))
    return True


def _try_soffice(
    src: Path,
    to: str,
    out_dir: Path,
    expected: Path,
    timeout: int,
    before: int | None,
    errors: list[str],
) -> bool:
    soffice = settings.find_libreoffice()
    if not soffice:
        errors.append("LibreOffice 未安装")
        return False
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
        proc = _run(cmd, timeout)
    except subprocess.TimeoutExpired:
        errors.append(f"LibreOffice 超时（>{timeout}s）")
        return False
    except FileNotFoundError:
        errors.append("soffice 不可执行")
        return False
    except OSError as e:
        errors.append(f"soffice 启动失败：{e}")
        return False
    if proc.returncode != 0 or not expected.exists() or _stamp(expected) == before:
        errors.append(f"LibreOffice rc={proc.returncode} 未产出 {expected.name}")
        return False
    return True


def convert(
    src: str | Path,
    to: str,
    *,
    out_dir: str | Path | None = None,
    timeout: int = 300,
) -> Path:
    """（兼容入口）转换并返回输出路径；后端选择见 :func:`convert_ex`。"""
    return convert_ex(src, to, out_dir=out_dir, timeout=timeout)[0]


def convert_ex(
    src: str | Path,
    to: str,
    *,
    out_dir: str | Path | None = None,
    timeout: int = 300,
    backend: str = "auto",
) -> tuple[Path, str]:
    """转换 → ``(输出路径, 后端名)``；后端名 ∈ ``{"wpscli", "libreoffice"}``。

    ``backend="auto"``（缺省）：先 **wpscli**（用户所见口径），失败回落 LibreOffice；
    ``"wpscli"``/``"libreoffice"`` 则只用指定后端。调用方应把后端名**打印出来**（注明口径）。
    """
    src = Path(src).resolve()
    if not src.exists():
        raise FileNotFoundError(f"输入文件不存在：{src}")
    out_dir = Path(out_dir) if out_dir else src.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    expected = out_dir / f"{src.stem}.{to}"
    before = _stamp(expected)
    order = {
        "wpscli": ("wpscli",),
        "libreoffice": ("libreoffice",),
    }.get(backend, ("wpscli", "libreoffice"))
    errors: list[str] = []
    for name in order:
        ok = (
            _try_wpscli(src, to, out_dir, expected, timeout, before, errors)
            if name == "wpscli"
            else _try_soffice(src, to, out_dir, expected, timeout, before, errors)
        )
        if ok:
            return expected, name
    if not settings.find_wpscli() and not settings.find_libreoffice():
        raise LibreOfficeNotInstalled()
    raise RuntimeError("转换失败：" + "；".join(errors))
