"""API Key 泄漏哨兵（REFACTOR §14、§4 Key 安全；E3-prep backlog 20261011-233003-leader）。

**为什么需要它**：SiliconFlow / 方舟这类 Key 只能由用户本人提供且永不入库，但「永不
入库」是一条纪律，不是一道机制——把 Key 粘进 ``.env`` 之外的地方（配置示例、任务书、
测试正文、AGENTS.md 笔记）就会随提交进 git，而 Key 一旦进过一次历史就等同泄露。
本模块把这条纪律变成**每次 doctor 都跑的机械检查**。

口径：**只扫 git 跟踪的文件**（``git ls-files``）——工作区里未跟踪的 ``.env`` 正是
Key 该待的地方（项目根 ``.env`` 已 gitignore），扫它会天天误报。命中后按模式二次确认
并给出「文件:行号 + 脱敏片段」，**绝不打印疑似 Key 全文**。

豁免：``tests/**``（阳性用例素材——哨兵自己的假 Key 也得写在测试里，见
``tests/skills/devops/test_keyscan.py``）与超大/二进制文件。豁免是**降敏**不是降效：
真实泄露几乎只会发生在配置、文档与代码里。
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from pysci.paths import PROJECT_ROOT

#: 疑似密钥模式（id → 正则）。宁可多报再由人判读，也不要漏报——漏报的代价是 Key 进历史。
#: 键名模式**故意不加词首 ``\b``**：真实泄漏最常见的形态是带前缀的环境变量名
#: （``SILICONFLOW_API_KEY = "sk-…"``），词首边界会让它整体漏报；代价是 ``risk-xxx``
#: 一类噪音，由 ``\bsk-``（模式一）和人工判读消化。
PATTERNS: dict[str, re.Pattern[str]] = {
    "sk-前缀密钥": re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}"),
    "键名=长随机值": re.compile(
        r"(?i)(api[_-]?key|apikey|secret|access[_-]?token|auth[_-]?token|password)"
        r"(\s*[:=]\s*['\"]?)([A-Za-z0-9_\-]{20,})"
    ),
    "Bearer 长令牌": re.compile(r"(?i)\bbearer\s+[A-Za-z0-9_\-\.]{20,}"),
}

#: 豁免面（相对项目根的 posix 前缀）。tests 下假 Key 是素材，不是泄露。
EXCLUDE_PREFIXES: tuple[str, ...] = ("tests/",)

#: 单文件大小上限（字节）：超了不扫——密钥串不会藏在万行生成物里，而扫它要付时间。
MAX_FILE_BYTES = 1_000_000


@dataclass
class Finding:
    """一处疑似泄漏。``excerpt`` 已脱敏（只留匹配起点前后的上下文，不含完整密钥）。"""

    path: str
    line_no: int
    pattern_id: str
    excerpt: str


def tracked_files(root: Path | None = None) -> list[str]:
    """git 跟踪文件清单（相对根、posix 分隔）。

    Raises:
        RuntimeError: 不在 git 仓库里 / git 不可用 / 超时——由调用方降级成 advisory。
    """
    proc = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=str(root or PROJECT_ROOT),
        capture_output=True,
        timeout=120,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"git ls-files rc={proc.returncode}")
    names = proc.stdout.decode("utf-8", errors="replace").split("\0")
    return [n for n in names if n.strip()]


def excluded(rel: str) -> bool:
    """是否落在豁免面（``tests/**`` 等）。"""
    return any(rel.startswith(p) for p in EXCLUDE_PREFIXES)


def _excerpt(line: str, match: re.Match[str]) -> str:
    """匹配点上下文（前后各 12 字），并把疑似密钥本体换成掩码——报告自身不得带毒。"""
    secret = match.group(0)
    masked = f"{secret[:4]}…{len(secret)}B…"
    start = max(0, match.start() - 12)
    end = min(len(line), match.end() + 12)
    body = line[start:end].replace(secret, masked)
    return ("…" if start > 0 else "") + body.strip() + ("…" if end < len(line) else "")


def scan_lines(text: str, *, path: str) -> list[Finding]:
    """对一段文本逐模式找命中（纯函数，供 :func:`scan_file` 与单测共用）。"""
    out: list[Finding] = []
    for i, line in enumerate(text.splitlines(), start=1):
        for pid, rx in PATTERNS.items():
            if m := rx.search(line):
                out.append(
                    Finding(
                        path=path, line_no=i, pattern_id=pid, excerpt=_excerpt(line, m)
                    )
                )
                break
    return out


def scan_file(rel: str, *, root: Path | None = None) -> list[Finding]:
    """扫一个跟踪文件；二进制/超大/读不动都静默跳过（哨兵不许把 doctor 拖崩）。"""
    p = (root or PROJECT_ROOT) / rel
    try:
        if p.stat().st_size > MAX_FILE_BYTES:
            return []
        raw = p.read_bytes()
    except OSError:
        return []
    if b"\0" in raw[:8000]:
        return []
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return []
    return scan_lines(text, path=rel)


def scan_files(files: list[str], *, root: Path | None = None) -> list[Finding]:
    """扫一批相对路径（豁免面之外）。"""
    out: list[Finding] = []
    for rel in files:
        if not excluded(rel):
            out.extend(scan_file(rel, root=root))
    return out


def scan(root: Path | None = None) -> tuple[list[Finding], str]:
    """全仓扫描：``(findings, error)``——``error`` 非空表示未能完成扫描（advisory）。"""
    try:
        files = tracked_files(root)
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
        return [], f"git 不可用，跳过扫描（{type(exc).__name__}: {exc}）"
    return scan_files(files, root=root), ""


def report(findings: list[Finding], error: str = "") -> list[str]:
    """doctor 用的人类可读行（√/✗/[*] 三档，与其余哨兵同风格）。"""
    if error:
        return [f"  [*] {error}"]
    if not findings:
        return ["  [√] 跟踪文件无疑似 API Key（tests/ 豁免：阳性用例素材）"]
    lines = [
        f"  [✗] {len(findings)} 处疑似 API Key 泄漏（Key 只能由用户本人提供，永不入库）："
    ]
    lines += [
        f"      {f.path}:{f.line_no} [{f.pattern_id}] {f.excerpt}"
        for f in findings[:20]
    ]
    if len(findings) > 20:
        lines.append(f"      …（其余 {len(findings) - 20} 处略）")
    lines.append(
        "      → 处置：删掉明文、改为环境变量注入；已提交过的须请组长重写历史/吊销 Key"
    )
    return lines
