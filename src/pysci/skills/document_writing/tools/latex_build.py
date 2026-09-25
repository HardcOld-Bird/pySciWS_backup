"""latex_build —— latexmk 编译驱动 + .log 报错解析 + chktex 语法检查。

LLM 撰写 .tex 后，本模块负责：
1. 调 latexmk 编译（自动处理 bibtex/biber 多趟），产物写入 build/ 保持项目整洁；
2. **解析 .log**，把错误/警告定位到「文件:行号: 信息」，回传给 LLM 精准修复；
3. 可选 chktex 静态检查。

TeX 未安装时抛 TeXNotInstalled，附清晰安装指引（不静默失败）。
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from .config import ENGINES, settings

# 引擎 → latexmk 标志
_ENGINE_FLAG = {
    "pdflatex": "-pdf",
    "xelatex": "-xelatex",
    "lualatex": "-lualatex",
}


class TeXNotInstalled(RuntimeError):
    """未检测到 TeX 发行版时抛出，附安装指引。"""

    def __init__(self) -> None:
        super().__init__(
            "未检测到 TeX 发行版（latexmk / pdflatex / xelatex）。\n"
            "请安装 TeX Live（完整版，推荐）并加入 PATH：\n"
            "  1. 下载 install-tl-windows.exe（TUNA 镜像）：\n"
            "     https://mirrors.tuna.tsinghua.edu.cn/CTAN/systems/texlive/tlnet/install-tl-windows.exe\n"
            "  2. 安装时把 repository 改为 TUNA：\n"
            "     https://mirrors.tuna.tsinghua.edu.cn/CTAN/systems/texlive/tlnet\n"
            "  3. 选 scheme-full，装到 C:\\texlive\\<year>\n"
            "  4. 确认 PATH 含 C:\\texlive\\<year>\\bin\\windows，重开终端\n"
            "  5. 运行 `compose doctor` 复核。"
        )


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------
@dataclass
class TexIssue:
    """一条编译问题（错误/警告/溢出盒）。"""

    kind: str  # error / warning / box
    message: str
    file: str = ""
    line: int | None = None

    def fmt(self) -> str:
        loc = self.file or "?"
        if self.line:
            loc = f"{loc}:{self.line}"
        return f"[{self.kind}] {loc}: {self.message}"


@dataclass
class BuildResult:
    ok: bool
    pdf_path: Path | None
    engine: str
    log_path: Path | None
    returncode: int
    errors: list[TexIssue] = field(default_factory=list)
    warnings: list[TexIssue] = field(default_factory=list)
    stdout_tail: str = ""

    def summary(self) -> str:
        head = "✓ 编译成功" if self.ok else "✗ 编译失败"
        lines = [
            f"{head}（engine={self.engine}, rc={self.returncode}）",
            f"  PDF : {self.pdf_path or '（未生成）'}",
            f"  log : {self.log_path or '（无）'}",
        ]
        if self.errors:
            lines.append(f"  错误 {len(self.errors)} 条：")
            lines.extend(f"    - {e.fmt()}" for e in self.errors[:30])
        if self.warnings:
            lines.append(f"  警告 {len(self.warnings)} 条（前 15）：")
            lines.extend(f"    - {w.fmt()}" for w in self.warnings[:15])
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# .log 解析（纯函数，可单测）
# ---------------------------------------------------------------------------
# file-line-error 格式：./main.tex:42: Undefined control sequence.
_RE_FILELINE_ERR = re.compile(r"^(.+?):(\d+):\s*(.+)$")
# 经典格式：! LaTeX Error: ...
_RE_BANG_ERR = re.compile(r"^!\s*(?:LaTeX|Package|Class)?\s*(?:Error)?:?\s*(.+)$")
# 紧随其后的行号提示：l.42
_RE_LNUM = re.compile(r"^l\.(\d+)")
# 警告：LaTeX Warning: ... / Package X Warning: ...（消息取到行尾，行号单独再抽）
_RE_WARNING = re.compile(
    r"(?:LaTeX|Package\s+\w+|Class\s+\w+)\s+Warning:\s*(.+)",
    re.IGNORECASE,
)
_RE_INPUT_LINE = re.compile(r"on input line\s+(\d+)", re.IGNORECASE)
# 溢出盒：Overfull \hbox (10.0pt too wide) in paragraph at lines 20--25
_RE_BOX = re.compile(
    r"^(Overfull|Underfull)\s+\\[hv]box\s+\((.+?)\).*?at lines\s+(\d+)--(\d+)"
)


def parse_log(log_text: str) -> tuple[list[TexIssue], list[TexIssue]]:
    """解析 .log 文本 → (errors, warnings)。

    兼容三种报错风格：file-line-error（`f:l: msg`）、经典（`! msg` + `l.N`）、
    以及 Overfull/Underfull 盒警告。
    """
    errors: list[TexIssue] = []
    warnings: list[TexIssue] = []
    lines = log_text.splitlines()

    for i, raw in enumerate(lines):
        line = raw.rstrip()
        if not line:
            continue

        # 1) file-line-error 错误
        m = _RE_FILELINE_ERR.match(line)
        if m and not line.startswith("!"):
            fpath, lnum, msg = m.group(1), int(m.group(2)), m.group(3).strip()
            # 排除误报：形如 "l.42" 或非 tex 路径的普通输出
            if msg and not msg.startswith("l."):
                errors.append(TexIssue("error", msg, file=fpath, line=lnum))
                continue

        # 2) 经典 ! 错误
        m = _RE_BANG_ERR.match(line)
        if m:
            msg = m.group(1).strip()
            lnum = None
            # 向下找最近的 l.N 行号提示
            for j in range(i + 1, min(i + 6, len(lines))):
                lm = _RE_LNUM.match(lines[j].strip())
                if lm:
                    lnum = int(lm.group(1))
                    break
            errors.append(TexIssue("error", msg or "TeX error", line=lnum))
            continue

        # 3) Overfull/Underfull 盒
        m = _RE_BOX.search(line)
        if m:
            warnings.append(
                TexIssue(
                    "box",
                    f"{m.group(1)} {m.group(2)} (lines {m.group(3)}--{m.group(4)})",
                    line=int(m.group(3)),
                )
            )
            continue

        # 4) 一般警告
        m = _RE_WARNING.search(line)
        if m:
            msg = m.group(1).strip()
            lm = _RE_INPUT_LINE.search(msg)
            lnum = int(lm.group(1)) if lm else None
            # 过滤无信息量的重复警告
            if msg and "There were undefined references" not in msg:
                warnings.append(TexIssue("warning", msg, line=lnum))

    return errors, warnings


def _dedupe(issues: list[TexIssue]) -> list[TexIssue]:
    seen: set[tuple] = set()
    out: list[TexIssue] = []
    for it in issues:
        key = (it.kind, it.file, it.line, it.message)
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out


# ---------------------------------------------------------------------------
# 编译
# ---------------------------------------------------------------------------
def _resolve_engine(engine: str | None) -> tuple[str, list[str]]:
    """返回 (engine_name, latexmk_flags)。engine='auto'/None 时不强制引擎，
    交给项目 latexmkrc / latexmk 默认决定。"""
    engines = settings.find_engines()
    name = (engine or settings.default_latex_engine or "auto").lower()
    if name in ("auto", ""):
        # 探测实际可用引擎仅用于报告，不强制传标志
        present = next((e for e in ENGINES if engines.get(e)), "pdflatex(latexmk默认)")
        return present, []
    if name not in _ENGINE_FLAG:
        raise ValueError(f"未知引擎：{engine}（可选 auto/{'/'.join(ENGINES)}）")
    if not engines.get(name):
        raise TeXNotInstalled()
    return name, [_ENGINE_FLAG[name]]


def build(
    main_tex: str | Path,
    *,
    engine: str | None = None,
    out_dir: str | Path | None = None,
    bib: bool = True,
    timeout: int = 600,
    extra_args: list[str] | None = None,
) -> BuildResult:
    """用 latexmk 编译主 .tex，解析日志，返回 BuildResult。

    Args:
        main_tex: 主文件（含 \\documentclass）。
        engine: auto / pdflatex / xelatex / lualatex。
        out_dir: 构建产物目录；默认 <main_dir>/build（保持项目整洁）。
        bib: 是否允许 latexmk 自动跑 bibtex/biber（False → -bibtex-）。
        timeout: 编译超时秒数。
        extra_args: 追加给 latexmk 的参数（如需 --shell-escape）。
    """
    main_tex = Path(main_tex).resolve()
    if not main_tex.exists():
        raise FileNotFoundError(f"主文件不存在：{main_tex}")

    tools = settings.find_tex_tools()
    latexmk = tools.get("latexmk")
    if not latexmk:
        raise TeXNotInstalled()

    engine_name, engine_flags = _resolve_engine(engine)
    if out_dir is None:
        out_dir = main_tex.parent / "build"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        latexmk,
        *engine_flags,
        "-interaction=nonstopmode",
        "-file-line-error",
        "-synctex=1",
        f"-outdir={out_dir}",
    ]
    if not bib:
        cmd.append("-bibtex-")
    if extra_args:
        cmd.extend(extra_args)
    cmd.append(str(main_tex))

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(main_tex.parent),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            env=settings.augmented_env(),
        )
        rc = proc.returncode
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
    except subprocess.TimeoutExpired:
        return BuildResult(
            ok=False,
            pdf_path=None,
            engine=engine_name,
            log_path=None,
            returncode=-1,
            errors=[TexIssue("error", f"编译超时（>{timeout}s）；可能卡在交互或缺包")],
            stdout_tail="",
        )
    except FileNotFoundError as e:
        raise TeXNotInstalled() from e

    log_path = out_dir / f"{main_tex.stem}.log"
    errors: list[TexIssue] = []
    warnings: list[TexIssue] = []
    if log_path.exists():
        errors, warnings = parse_log(log_path.read_text(encoding="utf-8", errors="replace"))
        errors, warnings = _dedupe(errors), _dedupe(warnings)

    pdf_path = out_dir / f"{main_tex.stem}.pdf"
    ok = pdf_path.exists() and rc == 0 and not errors

    tail = "\n".join((stdout + ("\n" + stderr if stderr else "")).splitlines()[-25:])
    return BuildResult(
        ok=ok,
        pdf_path=pdf_path if pdf_path.exists() else None,
        engine=engine_name,
        log_path=log_path if log_path.exists() else None,
        returncode=rc,
        errors=errors,
        warnings=warnings,
        stdout_tail=tail,
    )


def clean(main_tex: str | Path, *, out_dir: str | Path | None = None) -> None:
    """latexmk -C 清理构建产物。"""
    main_tex = Path(main_tex).resolve()
    latexmk = settings.find_tex_tools().get("latexmk")
    if not latexmk:
        raise TeXNotInstalled()
    if out_dir is None:
        out_dir = main_tex.parent / "build"
    subprocess.run(
        [latexmk, "-C", f"-outdir={out_dir}", str(main_tex)],
        cwd=str(main_tex.parent),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=settings.augmented_env(),
    )


def lint(main_tex: str | Path, *, timeout: int = 120) -> list[TexIssue]:
    """chktex 静态检查；未装 chktex 时返回空列表（非致命）。"""
    main_tex = Path(main_tex).resolve()
    chktex = settings.find_tex_tools().get("chktex")
    if not chktex:
        return []
    try:
        proc = subprocess.run(
            [chktex, "-q", "-f", "%f:%l:%c:%t:%m\n", str(main_tex)],
            cwd=str(main_tex.parent),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            env=settings.augmented_env(),
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return []
    issues: list[TexIssue] = []
    for line in (proc.stdout or "").splitlines():
        # chktex 格式 %f:%l:%c:%t:%m；用 maxsplit=4 保证消息内的冒号不被截断
        parts = line.split(":", 4)
        if len(parts) == 5:
            fname, line_no, _col, kind_raw, msg = parts
            try:
                ln = int(line_no)
            except ValueError:
                ln = None
            kind = "error" if kind_raw.strip().lower() == "error" else "warning"
            issues.append(TexIssue(kind, msg.strip(), file=fname, line=ln))
    return issues
