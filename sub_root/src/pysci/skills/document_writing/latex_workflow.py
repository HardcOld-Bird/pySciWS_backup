"""LaTeX 工作流胶水：latexmk 调用 + 日志解析（file:line 定位）+ chktex 集成。

ADR（2026-11-17）：期刊投稿（revtex4-2 等）只能 LaTeX，此线**无可替代**（用户裁决
"LaTeX 单轨、Typst 不引入"）；本模块是选型预算项 11（≈150-300 行），只做接缝：
驱动 latexmk、把 TeX 日志归一成 ``file:line`` 错误报告、包一层 chktex——社区无对应
薄件（latexmk 自身不产结构化诊断）。

日志解析口径（2026-11-17 对 TeX Live 2026 + latexmk 4.88 实证）：

- **file-line-error 风格**（本模块统一加 ``-file-line-error``）：
  ``./chapters/intro.tex:12: Undefined control sequence.``；
- **经典风格**：``! LaTeX Error: ...`` 配随后的 ``l.<n>`` 指针行；
- **警告**：``LaTeX Warning: ...`` / ``Package xxx Warning: ...``，行号取其后续
  行里的 ``on input line <n>``；
- **文件归属**：TeX 日志用括号标记文件开闭（``(./file.tex`` 开、``)`` 闭），
  用栈追踪当前文件。已知限制：文件名含空格/括号的极端情况解析不准
  （TeX 官方工具同此限制）。
"""

from __future__ import annotations

import contextlib
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from pysci.skills.document_writing.toolpaths import chktex_exe, latexmk_exe

# file-line-error 风格：`./file.tex:12: message`
_FILELINE_ERR = re.compile(
    r"^(?P<file>\.?/?[^()\s:][^:\s]*\.[A-Za-z]+):(?P<line>\d+):\s*(?P<msg>.+)$"
)
# 经典风格：`! message`（随后 l.<n> 行给行号）
_BANG_ERR = re.compile(r"^!\s*(?P<msg>.+)$")
_L_CONTEXT = re.compile(r"^l\.(?P<line>\d+)")
# 警告：`LaTeX Warning: ...` / `Package x Warning: ...` / `Class x Warning: ...`
_WARN = re.compile(
    r"^(?:(?:Package|Class)\s+\S+\s+|LaTeX\s+)?Warning:\s*(?P<msg>.+)$", re.I
)
_WARN_LINE = re.compile(r"on input line (?P<line>\d+)")
# chktex 定制格式 `%f:%l:%c: %m`（允许 Windows 盘符前缀）
_CHKTEX_LINE = re.compile(
    r"^(?P<file>(?:[A-Za-z]:)?[^:\s]+):(?P<line>\d+):(?P<col>\d+):\s*(?P<msg>.+)$"
)


@dataclass(frozen=True)
class LogIssue:
    """一条结构化诊断（来自 TeX 日志或 chktex）。"""

    severity: str  # "error" | "warning"
    message: str
    file: str | None = None
    line: int | None = None


@dataclass(frozen=True)
class BuildResult:
    """latexmk 一次构建的结果。"""

    ok: bool
    pdf: Path | None
    issues: tuple[LogIssue, ...]
    log_text: str
    returncode: int
    stdout: str = field(default="", repr=False)


class LatexError(RuntimeError):
    """工具链不可用 / 超时等基础设施错误（区别于文档本身的错误）。"""


def _track_files(text: str) -> list[str | None]:
    """逐字符扫描括号文件栈，返回每个字符偏移对应的"当前文件"（栈顶）。

    Returns:
        与 ``text`` 等长的列表，元素为该位置的当前文件（可为 None）。
    """
    stack: list[str] = []
    current: list[str | None] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch == "(":
            j = i + 1
            while j < n and text[j] not in "() \t\r\n":
                j += 1
            token = text[i + 1 : j]
            if token:
                stack.append(token)
        elif ch == ")" and stack:
            stack.pop()
        current.append(stack[-1] if stack else None)
        i += 1
    return current


def parse_log(log_text: str) -> list[LogIssue]:
    """把 TeX 日志解析成结构化诊断列表（纯函数，机械可测）。

    Args:
        log_text: ``<jobname>.log`` 全文（或 latexmk stdout，可含杂质行）。

    Returns:
        :class:`LogIssue` 列表：error 在前（按出现顺序），warning 随后；
        ``l.<n>`` 指针行的行号回填给其前最近的 ``!`` 错误。
    """
    file_at = _track_files(log_text)
    errors: list[LogIssue] = []
    warnings: list[LogIssue] = []
    lines = log_text.splitlines()
    offsets: list[int] = []
    acc = 0
    for line in lines:
        offsets.append(acc)
        acc += len(line) + 1
    pending_index: int | None = None
    for index, line in enumerate(lines):
        stripped = line.strip()
        base = min(offsets[index], len(file_at) - 1) if file_at else 0
        here = file_at[base] if file_at else None
        match = _FILELINE_ERR.match(stripped)
        if match:
            errors.append(
                LogIssue(
                    severity="error",
                    message=match.group("msg").strip(),
                    file=match.group("file"),
                    line=int(match.group("line")),
                )
            )
            pending_index = len(errors) - 1
            continue
        match = _BANG_ERR.match(stripped)
        if match:
            errors.append(
                LogIssue(severity="error", message=match.group("msg").strip(), file=here)
            )
            pending_index = len(errors) - 1
            continue
        match = _L_CONTEXT.match(stripped)
        if match and pending_index is not None:
            old = errors[pending_index]
            errors[pending_index] = LogIssue(
                severity=old.severity,
                message=old.message,
                file=old.file,
                line=int(match.group("line")),
            )
            pending_index = None
            continue
        match = _WARN.match(stripped)
        if match:
            line_no: int | None = None
            found = _WARN_LINE.search(" ".join(lines[index : index + 3]))
            if found:
                line_no = int(found.group("line"))
            warnings.append(
                LogIssue(
                    severity="warning",
                    message=match.group("msg").strip(),
                    file=here,
                    line=line_no,
                )
            )
    return errors + warnings


def parse_chktex_output(text: str) -> list[LogIssue]:
    """解析 chktex 输出（本模块统一以 ``%f:%l:%c: %m`` 格式调用）。

    Args:
        text: chktex stdout（每行 ``file:line:col: message``）。

    Returns:
        全部为 warning 的 :class:`LogIssue` 列表。
    """
    issues: list[LogIssue] = []
    for line in text.splitlines():
        match = _CHKTEX_LINE.match(line.strip())
        if match:
            issues.append(
                LogIssue(
                    severity="warning",
                    message=match.group("msg").strip(),
                    file=match.group("file"),
                    line=int(match.group("line")),
                )
            )
    return issues


def lint_command(src: str | Path) -> list[str]:
    """拼装 chktex 命令行（纯函数，不执行）。

    注意：``-f`` 格式串的记录分隔必须是**真换行符**（Python 源码 ``"\\n"`` 转义
    出来的单字符）；chktex 不解释字面的反斜杠+n 两字符序列——2026-11-17 冒烟
    实证：字面反斜杠+n 会让多条 warning 被并成一条、后续 ``file:line:col`` 被吞
    进 message。
    """
    src = Path(src)
    return [chktex_exe(), "-q", "-f", "%f:%l:%c: %m\n", src.name]


def lint(tex_path: str | Path, *, timeout: float = 120.0) -> list[LogIssue]:
    """对单个 .tex 跑 chktex 静态检查。

    Args:
        tex_path: TeX 源文件（在该文件所在目录执行）。
        timeout: 超时秒数。

    Returns:
        warning 列表（chktex 只产 warning；其非零退出码不代表失败）。

    Raises:
        LatexError: chktex 不可用或超时。
    """
    src = Path(tex_path)
    cmd = lint_command(src)
    try:
        result = subprocess.run(
            cmd,
            cwd=src.parent,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise LatexError(f"chktex 无法执行：{' '.join(cmd)}（{exc}）") from exc
    return parse_chktex_output(result.stdout)


def build(
    tex_path: str | Path,
    *,
    engine: str = "pdflatex",
    run_lint: bool = True,
    timeout: float = 600.0,
) -> BuildResult:
    """latexmk 编译 + 日志解析（+ 可选 chktex）。

    Args:
        tex_path: 主 .tex 文件（在该文件所在目录编译，PDF 落在源文件旁）。
        engine: ``pdflatex`` / ``xelatex`` / ``lualatex``（映射 latexmk 同名选项）。
        run_lint: 是否附带 chktex 检查（结果并入 issues，均为 warning）。
        timeout: 超时秒数。

    Returns:
        :class:`BuildResult`：``ok`` 为真当且仅当 latexmk 零退出、PDF 产出、
        且日志解析无 error。

    Raises:
        LatexError: engine 非法、latexmk 不可用或超时。
    """
    if engine not in ("pdflatex", "xelatex", "lualatex"):
        raise LatexError(f"不支持的引擎：{engine}（可选 pdflatex/xelatex/lualatex）")
    src = Path(tex_path)
    flag = "-pdf" if engine == "pdflatex" else f"-{engine}"
    cmd = [latexmk_exe(), flag, "-interaction=nonstopmode", "-file-line-error", src.name]
    try:
        proc = subprocess.run(
            cmd,
            cwd=src.parent,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise LatexError(f"latexmk 无法执行：{' '.join(cmd)}（{exc}）") from exc

    log_path = src.with_suffix(".log")
    log_text = (
        log_path.read_text(encoding="utf-8", errors="replace")
        if log_path.is_file()
        else ""
    )
    issues = list(parse_log(log_text))
    if run_lint:
        # chktex 缺失不阻断编译；由 doctor 暴露
        with contextlib.suppress(LatexError):
            issues += lint(src)
    pdf = src.with_suffix(".pdf")
    has_error = any(issue.severity == "error" for issue in issues)
    ok = proc.returncode == 0 and pdf.is_file() and not has_error
    return BuildResult(
        ok=ok,
        pdf=pdf if pdf.is_file() else None,
        issues=tuple(issues),
        log_text=log_text,
        returncode=proc.returncode,
        stdout=proc.stdout,
    )
