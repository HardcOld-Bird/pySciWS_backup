"""空终端根治：钩子脚本与 hook 命令的 spawn 隐藏口径钉子（wake-bind-20261018 s1）。

现象：看门狗 `_rearm` 以 `DETACHED_PROCESS|CREATE_NO_WINDOW` 拉起 bash（**完全无控制台**），
bash 内裸 spawn `python.exe`/`pysci-orch.exe`（console 子系统）→ Windows 分配**新的可见
控制台**，标题=解释器路径；看门狗反复重臂即反复闪窗。

本文件钉两件事：
1. 三个钩子脚本（watch-arm/night-gate/night-policy）的非注释行**不得**出现 console 启动器，
   且必须走 `pythonw.exe`（GUI 子系统，永不分配控制台）；
2. 原有的 stdout/stderr 重定向与命令替换语义逐条保留（pythonw 下未重定向才会丢写）。

`PostToolUse` 的 turnbudget hook 同理：win32 下命令必须解析到 `pythonw.exe`。
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from pysci.paths import PROJECT_ROOT
from pysci.skills.orchestration.tools import turnbudget

HOOKS = PROJECT_ROOT / ".qoder" / "hooks"
SCRIPTS = ("watch-arm.sh", "night-gate.sh", "night-policy.sh")


def _code_lines(path: Path) -> list[str]:
    """去掉整行注释后的脚本正文（注释里会**解释**被淘汰的 python.exe，不参与执行判据）。"""
    lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.lstrip().startswith("#"):
            continue
        lines.append(line)
    return lines


def test_hook_scripts_never_invoke_console_launchers():
    """非注释行不得出现 python.exe / pysci-orch.exe / pysci-dev.exe（会闪窗的启动器）。"""
    for name in SCRIPTS:
        code = "\n".join(_code_lines(HOOKS / name))
        for bad in ("python.exe", "pysci-orch.exe", "pysci-dev.exe", "pysci-dev "):
            assert bad not in code, (
                f"{name} 仍在裸 spawn {bad}（无控制台下会开可见终端）"
            )


def test_hook_scripts_use_pythonw_and_keep_redirections():
    watch = "\n".join(_code_lines(HOOKS / "watch-arm.sh"))
    assert 'PYW="$ROOT/.venv/Scripts/pythonw.exe"' in watch
    # night 差分 / 归属判定 / 看门狗自启 / watch 挂载四条路径的重定向语义不变
    assert 'ndiff=$(printf \'%s\' "$PAYLOAD" | PYTHONPATH="$ROOT/src" "$PYW"' in watch
    assert '2>>"$LOG")' in watch
    assert "orch wake-claim --check" in watch
    assert "orch watchdog start 2>&1" in watch
    assert 'orch watch --timeout "$CYCLE" 2>&1' in watch

    gate = "\n".join(_code_lines(HOOKS / "night-gate.sh"))
    assert '"$PYW" -m pysci.skills.orchestration.tools.night' in gate
    assert (
        '--gate --settings "$ROOT/data/用户的inbox/pysci_settings.toml" <<<"$PAYLOAD"'
        in gate
    )
    assert "exit 0" in gate and "[night-gate fail-open]" in gate

    policy = "\n".join(_code_lines(HOOKS / "night-policy.sh"))
    assert '"$PYW" -m pysci.skills.orchestration.tools.night' in policy
    assert '2>/dev/null <<<"$PAYLOAD" || exit 0' in policy


def test_orch_module_entrypoint_exists_for_pythonw():
    """钩子改 `pythonw -m ...tools.orch` 后，模块必须有 __main__ 守卫（缺则静默无动作）。"""
    orch_py = (
        PROJECT_ROOT
        / "src"
        / "pysci"
        / "skills"
        / "orchestration"
        / "tools"
        / "orch.py"
    )
    assert re.search(r'if __name__ == "__main__":', orch_py.read_text(encoding="utf-8"))


def test_turnbudget_hook_resolves_to_pythonw_on_win32():
    """PostToolUse hook 同样不得在无控制台父进程下开窗。"""
    settings = json.loads(turnbudget.hook_settings(60, since=0.0))
    command = settings["hooks"]["PostToolUse"][0]["hooks"][0]["command"]
    assert "turnbudget.py" in command and "--budget 60" in command
    if sys.platform == "win32":
        assert "pythonw.exe" in command, (
            "win32 下 hook 解释器须为 pythonw.exe（GUI 子系统不分配控制台）"
        )
    else:
        assert "pythonw" not in command, "POSIX 无 pythonw 概念，保持 sys.executable"
