"""声明式机械验收（README §5.1）：按组员交付中的 check 声明运行确定性脚本。

零 LLM、orch 自动执行；check 类型注册于 registry.json 的 checks 表；未注册类型
按 unrouted 放行并记录（门禁故障不阻塞生产）。脚本所有权在 devops，组员禁改。
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from pysci.paths import PROJECT_ROOT

from .delivery import Artifact
from .registry import Registry


def _path_arg(p: Path, spec: dict[str, Any]) -> str:
    """``{path}`` 的取值（backlog 20261015-132143-devops 判据①）。

    默认＝产物路径本身（同 ``figure-audit`` 的形状）。注册项可给 ``path_default``：
    当产物**不是本 check 吃得下的形态**时回落到它——`orch-tests` 收的是 **pytest 目标**
    （``tests/`` 下的路径或 ``test_*.py``），而交付常声明源码产物（如 ``plans.py``）；
    若把源码路径喂给 ``pytest``，它会**零收集并以退出码 5 判 fail** ——那正是本单要消灭的
    「假 FAIL」。故：是 pytest 目标 → 收窄到该目标（自指套件）；否则 → 回落到整目录默认面。
    """
    fallback = str(spec.get("path_default") or "")
    if not fallback:
        return str(p)
    posix = p.as_posix()
    is_test_target = ("/tests/" in posix or posix.startswith("tests/")) or (
        p.suffix == ".py" and p.name.startswith("test_")
    )
    return str(p) if is_test_target else fallback


#: 单次验收命令的墙钟上限（秒）——验收必须是轻量确定性检查。
CHECK_TIMEOUT_S = 600


def run_checks(
    artifacts: list[Artifact], registry: Registry, *, member_pod: Path
) -> list[dict[str, Any]]:
    """对交付声明的产物逐条运行机械验收。

    Args:
        artifacts: 解析出的产物声明列表。
        registry: 注册表（checks 命令模板来源）。
        member_pod: 成员 pod 绝对路径（相对产物路径的解析基准之一）。

    Returns:
        每条产物的验收结果字典列表：
        ``{path, check, verdict: pass|fail|unrouted|skipped, output_tail, returncode}``。
        ``returncode`` 是判据真源（``None``＝命令没能跑起来）；``output_tail`` 只作**人读**
        摘要，其首位固定是 ``[check exit=N]`` 锚（backlog 20261015-132939-fix3）。
    """
    results: list[dict[str, Any]] = []
    for art in artifacts:
        if art.check in ("none", ""):
            results.append(
                {
                    "path": art.path,
                    "check": art.check or "none",
                    "verdict": "skipped",
                    "reason": art.reason,
                    "output_tail": "",
                }
            )
            continue
        spec = registry.checks.get(art.check)
        if spec is None:
            results.append(
                {
                    "path": art.path,
                    "check": art.check,
                    "verdict": "unrouted",
                    "output_tail": "",
                }
            )
            continue
        # 产物路径解析：绝对路径原样；相对路径先按项目根、再按 pod 解析
        p = Path(art.path)
        if not p.is_absolute():
            cand_root = PROJECT_ROOT / p
            p = cand_root if cand_root.exists() else member_pod / p
        cmd = [str(part).replace("{path}", _path_arg(p, spec)) for part in spec["cmd"]]
        try:
            from . import procs as _procs

            proc = _procs.run_hidden(  # 隐藏窗口：验收链常由无控制台的 worker 触发
                cmd,
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=CHECK_TIMEOUT_S,
            )
            verdict = "pass" if proc.returncode == 0 else "fail"
            # 摘要把**退出码写在最前**（backlog 20261015-132939-fix3）：判据本来就只认
            # 退出码（下一行的 verdict），把口径摆在摘要首位，读 tail 的人不会再拿尾部
            # 文本当判据（实案 A.a5 的误读土壤）。
            tail = (
                f"[check exit={proc.returncode}] "
                + (proc.stdout or proc.stderr or "")[-600:]
            )
            dirty_face: list[str] = []
            if verdict == "fail":
                # 污染降级（backlog 20261015-130555-devops / -132143-devops，判据②）：
                # 现场有**与本交付无关**的在飞脏面时，判据 FAIL 不能算数——`orch-tests` 跑
                # 整个编排队列、其红多半来自别人正在编辑的跟踪面（实案 A.a5：已完成的交付被
                # 判 FAIL → code=2 → drain 抢救 + 另起续作单，白烧一跳）。降级为 **advisory**：
                # 仍然出声（tail 首位标注 + `dirty_face` 字段留证），但**不计入 failed_checks**
                # → 交付的 `code` 不受影响、不触发抢救与续作。
                dirty_face = _unrelated_dirty_paths(p)
                if dirty_face:
                    verdict = "advisory"
                    tail = (
                        f"[check advisory: 现场有无关脏面 {len(dirty_face)} 项"
                        f"（{', '.join(dirty_face[:4])}{'…' if len(dirty_face) > 4 else ''}）"
                        f"——FAIL 不判交付] " + tail
                    )
        except subprocess.TimeoutExpired:
            verdict, tail, proc = "fail", f"验收命令超时（>{CHECK_TIMEOUT_S}s）", None
        except OSError as exc:
            verdict, tail, proc = "fail", f"验收命令无法执行：{exc}", None
        results.append(
            {
                "path": str(p),
                "check": art.check,
                "verdict": verdict,
                "output_tail": tail,
                # 判据真源＝退出码（None=未能执行）：台账/报告据此复核，不必从 tail 里猜。
                "returncode": proc.returncode if proc is not None else None,
                **({"dirty_face": dirty_face} if dirty_face else {}),
            }
        )
    return results


def _unrelated_dirty_paths(artifact: Path) -> list[str]:
    """工作区里**与本次产物无关**的脏路径（跟踪面）；读不到即空表（fail-open）。

    「无关」＝不在本产物路径之下（产物自身可以合法地还没提交）。判据只取 git 侧的
    ``status --porcelain``（跟踪面改动 + 未跟踪），不猜语义。
    """
    from . import drain as _drain

    try:
        rc, out = _drain._run_git("status", "--porcelain", "--untracked-files=all")
    except Exception:  # noqa: BLE001 - 探活面 fail-open（探不到就不降级）
        return []
    if rc != 0:
        return []
    try:
        rel = artifact.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except (OSError, ValueError):
        rel = ""
    skip = ("__pycache__/", ".pytest_cache/")
    dirty: list[str] = []
    for ln in out.splitlines():
        p = ln[3:].strip().strip('"').replace("\\", "/")
        if not p or any(s in p for s in skip):
            continue
        if rel and (
            p == rel
            or p.startswith(rel.rstrip("/") + "/")
            or rel.startswith(p.rstrip("/") + "/")
        ):
            continue  # 产物自身/其父目录：属于本交付，不算「无关」
        dirty.append(p)
    return dirty
