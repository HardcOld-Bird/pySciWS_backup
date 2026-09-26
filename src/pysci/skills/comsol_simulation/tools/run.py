"""求解、参数扫描、批处理与诊断捕获。

这是 Agent「调试能力」的核心：COMSOL 的报错常常埋在 JPype 异常链与求解日志里，本模块把
它们**结构化**地捞出来，让我能读懂失败原因并迭代。

- :func:`solve` — 求解（可选先 clear），捕获耗时、JPype 异常链 + Java 栈、mph/COMSOL 日志尾部、求解后 problems。
- :func:`solve_param_scan` — Python 侧驱动的参数扫描（改参→求解→收集），适合模型未内置 parametric 的情形。
- :func:`run_batch` — 用 ``comsolbatch.exe`` 跑 detached 批处理（重活/无需交互会话时），捕获 stdout/stderr。
- :func:`save_model` — 保存 .mph 检查点。

live COMSOL 调用一律防御式包裹；日志经 Python logging 的 "mph" logger 捕获（mph 会把 COMSOL
求解器消息转发到这里）。
"""

from __future__ import annotations

import logging
import subprocess
import time
import traceback
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import settings
from .inspect import _jmodel, _safe


class SolveError(RuntimeError):
    """求解失败（保留结构化诊断）。"""


# ---------------------------------------------------------------------------
# 日志捕获（mph 把 COMSOL 消息转发到 "mph" logger）
# ---------------------------------------------------------------------------
class _LogCapture(logging.Handler):
    """把指定 logger 的记录缓存到内存 list，供求解后回读尾部。"""

    def __init__(self, maxlen: int = 400) -> None:
        super().__init__()
        self.maxlen = maxlen
        self.records: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.records.append(self.format(record))
            if len(self.records) > self.maxlen:
                self.records = self.records[-self.maxlen :]
        except Exception:  # noqa: BLE001
            pass

    def tail(self, n: int = 60) -> str:
        return "\n".join(self.records[-n:])


def _capture_logs(logger_names: Sequence[str] = ("mph",)) -> tuple[_LogCapture, list[tuple[logging.Logger, int]]]:
    """在给定 logger 上挂一个捕获 handler，返回 (handler, 原级别列表)。"""
    handler = _LogCapture()
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    originals: list[tuple[logging.Logger, int]] = []
    for name in logger_names:
        lg = logging.getLogger(name)
        originals.append((lg, lg.level))
        lg.addHandler(handler)
        if lg.level > logging.DEBUG or lg.level == logging.NOTSET:
            lg.setLevel(logging.DEBUG)
    return handler, originals


def _restore_logs(handler: _LogCapture, originals: list[tuple[logging.Logger, int]]) -> None:
    for lg, lvl in originals:
        try:
            lg.removeHandler(handler)
            lg.setLevel(lvl)
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------------------
# 诊断提取
# ---------------------------------------------------------------------------
def _extract_java_stack(exc: BaseException) -> str:
    """从 JPype 异常里尽力提取 Java 侧栈/消息（COMSOL 真正的报错常在这里）。"""
    parts: list[str] = []
    # JPype 的 JException 常带 stacktrace() / .message()
    for attr in ("stacktrace", "message"):
        fn = getattr(exc, attr, None)
        if callable(fn):
            val = _safe(fn, default=None)
            if val:
                parts.append(f"[{attr}] {val}")
    return "\n".join(parts)


def _scan_problems(model: Any) -> list[str]:
    """求解后扫描模型的 problems 节点（警告/错误），返回可读文本列表。"""
    jm = _jmodel(model)
    out: list[str] = []
    problems = _safe(getattr(jm, "problems", None), default=None) if hasattr(jm, "problems") else None
    # model.java.problems() 可能返回一个 Problem 序列
    try:
        if problems is not None and callable(problems):
            problems = problems()
    except Exception:  # noqa: BLE001
        problems = None
    if problems is None:
        return out
    # 尽力枚举
    for acc in ("problems", "getAll", "tags"):
        items = _safe(getattr(problems, acc, lambda: None), default=None)
        if items:
            try:
                for it in items:
                    out.append(str(it))
            except Exception:  # noqa: BLE001
                pass
            break
    return out


# ---------------------------------------------------------------------------
# 求解结果
# ---------------------------------------------------------------------------
@dataclass
class SolveResult:
    ok: bool
    elapsed: float
    study: str | None = None
    error: str | None = None
    problems: list[str] = field(default_factory=list)
    log_tail: str = ""

    def report(self) -> str:
        lines = [
            f"solve ok={self.ok} elapsed={self.elapsed:.1f}s study={self.study or '(default)'}",
        ]
        if self.error:
            lines.append("--- error ---")
            lines.append(self.error)
        if self.problems:
            lines.append(f"--- problems ({len(self.problems)}) ---")
            lines += [f"  {p}" for p in self.problems[:40]]
        if self.log_tail:
            lines.append("--- log tail ---")
            lines.append(self.log_tail)
        return "\n".join(lines)


def _run_study_java(jm: Any, study: str) -> None:
    """按 tag 直驱 Java study/sol 节点（绕过 mph 的节点**名**导航）。

    mph 的 ``model.solve(tag)`` 走 ``self/'studies'/tag`` 的节点名查找，对 Java tag 常常
    查不到（报 LookupError）。这里直接用 COMSOL Java API：先试 ``study(tag).run()``，
    再回退 ``sol(tag).runAll()``。
    """
    try:
        jm.study(study).run()
        return
    except Exception:  # noqa: BLE001 - 不是 study tag，尝试 sol
        pass
    try:
        jm.sol(study).runAll()
        return
    except Exception as e:  # noqa: BLE001
        raise SolveError(f"无法运行 study/sol '{study}': {e}") from e


def solve(model: Any, study: str | None = None, *, clear: bool = False) -> SolveResult:
    """求解模型（默认 study）。捕获耗时、异常链、日志尾部与 problems。

    Args:
        model: mph.Model（或裸 Java model）。
        study: study/solver tag（如 "std1"）；None → 用 mph 默认（model.solve() 求解全部）。
        clear: 求解前是否 ``model.clear()``（清旧解，参数扫描重跑时常用）。
    """
    t0 = time.time()
    handler, originals = _capture_logs()
    error: str | None = None
    ok = True
    try:
        if clear and hasattr(model, "clear"):
            _safe(model.clear)
        if hasattr(model, "solve"):
            if study:
                _run_study_java(_jmodel(model), study)
            else:
                model.solve()
        else:
            # 裸 Java model：走 study().run() 或 sol().runAll()
            jm = _jmodel(model)
            if study:
                _run_study_java(jm, study)
            else:
                jm.sol().runAll()
    except Exception as e:  # noqa: BLE001
        ok = False
        java_stack = _extract_java_stack(e)
        error = f"{type(e).__name__}: {e}"
        if java_stack:
            error += f"\n{java_stack}"
        error += "\n--- python traceback ---\n" + traceback.format_exc()
    finally:
        elapsed = time.time() - t0
        log_tail = handler.tail()
        _restore_logs(handler, originals)

    problems = _scan_problems(model) if ok else []
    return SolveResult(
        ok=ok, elapsed=elapsed, study=study, error=error, problems=problems, log_tail=log_tail
    )


def solve_param_scan(
    model: Any,
    param_name: str,
    values: Sequence[Any],
    collect: Callable[[Any, Any], Any],
    *,
    study: str | None = None,
    clear: bool = True,
    on_error: str = "raise",
) -> list[tuple[Any, Any, SolveResult]]:
    """Python 侧驱动的参数扫描：逐个设 ``param_name``→求解→``collect(model, value)``。

    适合模型未内置 parametric sweep 的情形。返回 ``[(value, collected, SolveResult), ...]``。

    Args:
        collect: ``collect(model, value) -> Any``，从求解后的模型提取所需数据。
        on_error: ``"raise"``（默认，首个失败即抛）| ``"skip"``（记录失败继续）。
    """
    jm = _jmodel(model)
    p = jm.param()
    results: list[tuple[Any, Any, SolveResult]] = []
    for v in values:
        p.set(param_name, str(v))
        sr = solve(model, study=study, clear=clear)
        if not sr.ok:
            if on_error == "raise":
                raise SolveError(
                    f"扫描 {param_name}={v} 求解失败：\n{sr.error}"
                )
            results.append((v, None, sr))
            continue
        collected = collect(model, v)
        results.append((v, collected, sr))
    return results


# ---------------------------------------------------------------------------
# comsolbatch 批处理（detached，重活）
# ---------------------------------------------------------------------------
@dataclass
class BatchResult:
    ok: bool
    returncode: int
    elapsed: float
    output_file: Path | None
    stdout: str
    stderr: str


def run_batch(
    input_mph: str | Path,
    output_mph: str | Path | None = None,
    *,
    study: str | None = None,
    cores: int | None = None,
    tempdir: str | Path | None = None,
    timeout: float | None = None,
    extra_args: Sequence[str] = (),
) -> BatchResult:
    """用 ``comsolbatch.exe`` 跑批处理求解（不占交互会话，适合重活/长任务）。

    Args:
        input_mph: 输入 .mph。
        output_mph: 输出 .mph（默认在 runs/ 下同名加 ``_out``）。
        study: ``-study`` tag。
        cores: ``-np`` 线程数（夹到 config 上限）。
        tempdir: ``-tmpdir`` 磁盘临时目录（默认 config 的 tempdir）。
        timeout: 秒；None = 不限。
        extra_args: 透传给 comsolbatch 的额外参数。
    """
    exe = settings.install.batch_exe
    if not exe or not Path(exe).exists():
        raise SolveError(f"未找到 comsolbatch.exe（发现结果：{exe!r}）")
    in_path = Path(input_mph)
    if not in_path.exists():
        raise SolveError(f"输入模型不存在：{in_path}")
    out_path = Path(output_mph) if output_mph else (settings.runs_dir / f"{in_path.stem}_out.mph")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = cores if cores is not None else settings.comsol_max_cores
    n = max(1, min(int(n), max(1, settings.comsol_max_cores)))
    tmp = Path(tempdir) if tempdir else settings.tempdir
    tmp.mkdir(parents=True, exist_ok=True)

    args = [
        str(exe),
        "-inputfile", str(in_path),
        "-outputfile", str(out_path),
        "-np", str(n),
        "-tmpdir", str(tmp),
    ]
    if study:
        args += ["-study", study]
    args += list(extra_args)

    t0 = time.time()
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    proc = subprocess.run(  # noqa: S603
        args,
        capture_output=True,
        text=True,
        timeout=timeout,
        creationflags=creationflags,
    )
    elapsed = time.time() - t0
    return BatchResult(
        ok=(proc.returncode == 0 and out_path.exists()),
        returncode=proc.returncode,
        elapsed=elapsed,
        output_file=out_path if out_path.exists() else None,
        stdout=proc.stdout or "",
        stderr=proc.stderr or "",
    )


# ---------------------------------------------------------------------------
# 保存
# ---------------------------------------------------------------------------
def save_model(model: Any, path: str | Path, *, client: Any = None) -> Path:
    """保存 .mph 检查点。优先 ``client.save(model, path)``，回退 ``model.save(path)``。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if client is not None and hasattr(client, "save"):
        client.save(model, str(p))
        return p
    if hasattr(model, "save"):
        model.save(str(p))
        return p
    jm = _jmodel(model)
    jm.save(str(p))
    return p
