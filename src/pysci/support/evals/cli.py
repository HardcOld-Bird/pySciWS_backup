"""pysci-eval: run one eval task through the project headless DSH.

Usage:
    pysci-eval run sub_root/evals/tasks/example-smoke
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from pysci.paths import EVALS_ROOT, PROJECT_ROOT


def _dsh_cmd() -> Path:
    return (
        PROJECT_ROOT
        / "sub_root"
        / "deps_management"
        / "deps"
        / "node"
        / "modules"
        / "node_modules"
        / ".bin"
        / "dsh.cmd"
    )


def _dsh_home() -> Path:
    return (
        PROJECT_ROOT
        / "sub_root"
        / "deps_management"
        / "deps"
        / "node"
        / "modules"
        / "dsh_home"
    )


def _node_dir() -> Path:
    runtime = (
        PROJECT_ROOT
        / "sub_root"
        / "deps_management"
        / "deps"
        / "node"
        / "runtime"
    )
    version = (runtime / "current.txt").read_text(encoding="utf-8").strip()
    return runtime / version


def _load_api_key() -> str | None:
    key = os.environ.get("DEEPSEEK_API_KEY")
    if key:
        return key
    env_file = PROJECT_ROOT / ".env"
    if not env_file.is_file():
        return None
    for line in env_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("DEEPSEEK_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def run_task(task_dir: Path, timeout_min: int = 10) -> int:
    prompt_file = task_dir / "prompt.md"
    if not prompt_file.is_file():
        print(f"missing prompt.md: {task_dir}", file=sys.stderr)
        return 2
    dsh = _dsh_cmd()
    if not dsh.is_file():
        print(f"missing project dsh: {dsh}", file=sys.stderr)
        return 2
    prompt = prompt_file.read_text(encoding="utf-8")
    task_id = task_dir.name
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = EVALS_ROOT / "runs" / f"{task_id}-{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)

    env = os.environ.copy()
    env["DSH_HOME"] = str(_dsh_home())
    env["PATH"] = str(_node_dir()) + os.pathsep + env.get("PATH", "")
    key = _load_api_key()
    if key:
        env["DEEPSEEK_API_KEY"] = key

    started = time.time()
    try:
        result = subprocess.run(
            [str(dsh), "--profile", "headless", prompt],
            cwd=PROJECT_ROOT,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_min * 60,
        )
        stdout, stderr, code = result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        code = -1
    elapsed = round(time.time() - started, 3)

    (run_dir / "stdout.txt").write_text(stdout, encoding="utf-8")
    (run_dir / "stderr.txt").write_text(stderr, encoding="utf-8")
    record = {
        "task": task_id,
        "run_dir": str(run_dir),
        "started_at": stamp,
        "elapsed_seconds": elapsed,
        "returncode": code,
        "prompt": prompt,
        "note": "MVP: headless profile default composition; preset-aware A/B is project-DSH work.",
    }
    (run_dir / "result.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"run: {run_dir}")
    print(f"returncode: {code}  elapsed: {elapsed}s")
    return 0 if code == 0 else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pysci-eval")
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="run one task directory")
    run.add_argument("task_dir", type=Path)
    run.add_argument("--timeout-min", type=int, default=10)
    args = parser.parse_args(argv)
    if args.cmd == "run":
        return run_task(args.task_dir.resolve(), args.timeout_min)
    return 2
