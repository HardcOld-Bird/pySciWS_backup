#!/bin/bash
# .qoder/hooks/night-gate.sh — 夜航模式 AskUserQuestion 机械拦截（PreToolUse）
#
# PSEP_2 §3.3a（实施单 task-20261014-psep2-n2）：matcher=AskUserQuestion；读
# data/pysci_settings.toml 的 [night] enabled（**每回合现读，热切换**）。
#   ON  → **exit 2**（harness 拦截语义）并在 stderr 注出「转 twin 终裁」指令——
#         提问工具本身成为不可能，不依赖模型自觉；
#   OFF → exit 0 放行（零行为变化）。
# 判定与文案单点在 src/pysci/…/tools/night.py（--gate）。
#
# 安全：**fail-open**——读开关出错/环境缺失/gate 返回非预期码一律**放行（exit 0）**并
# 在 stderr 留 `[night-gate fail-open]` 痕（不能因配置损坏就废掉提问工具；与
# watch-arm.sh / night-policy.sh 同目录同风格，不塌任何主链路）。
#
# 2026-10-18（wake-bind-20261018 s1-noconsole）：钩子执行体在**无控制台**父进程下也可能是
# pythonw/DETACHED——裸 spawn `python.exe` 会新开可见控制台。统一改 **pythonw.exe**
# （GUI 子系统，永不分配控制台）；stdin 走 here-string、stderr 由 harness 捕获，语义不变。
PAYLOAD=$(cat)
# 入口 cwd 在 cd 之前取样（pod 直聊会话判据；载荷带 cwd 时优先用载荷）。
ENTRY_CWD="$(pwd -W 2>/dev/null || pwd)"
export PYSCI_HOOK_CWD="$ENTRY_CWD"
ROOT="${QODER_PROJECT_DIR:-/d/XXXIIIGGG/projects/pySci/pySciWS}"
cd "$ROOT" || exit 0
PYW="$ROOT/.venv/Scripts/pythonw.exe"
if [ ! -x "$PYW" ]; then
  echo "[night-gate fail-open] 环境缺失（pythonw 不可执行）→ 放行" >&2
  exit 0
fi
PYTHONPATH="$ROOT/src" "$PYW" -m pysci.skills.orchestration.tools.night \
  --gate --settings "$ROOT/data/pysci_settings.toml" <<<"$PAYLOAD"
rc=$?
case "$rc" in
  0|2) exit "$rc" ;;
  *)
    echo "[night-gate fail-open] gate 异常 rc=$rc → 放行" >&2
    exit 0
    ;;
esac
