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
        ``{path, check, verdict: pass|fail|unrouted|skipped, output_tail}``。
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
        cmd = [str(part).replace("{path}", str(p)) for part in spec["cmd"]]
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=CHECK_TIMEOUT_S,
            )
            verdict = "pass" if proc.returncode == 0 else "fail"
            tail = (proc.stdout or proc.stderr or "")[-600:]
        except subprocess.TimeoutExpired:
            verdict, tail = "fail", f"验收命令超时（>{CHECK_TIMEOUT_S}s）"
        except OSError as exc:
            verdict, tail = "fail", f"验收命令无法执行：{exc}"
        results.append(
            {
                "path": str(p),
                "check": art.check,
                "verdict": verdict,
                "output_tail": tail,
            }
        )
    return results
