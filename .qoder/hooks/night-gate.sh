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
PAYLOAD=$(cat)
ROOT="${QODER_PROJECT_DIR:-/d/XXXIIIGGG/projects/pySci/pySciWS}"
cd "$ROOT" || exit 0
PY="$ROOT/.venv/Scripts/python.exe"
if [ ! -x "$PY" ]; then
  echo "[night-gate fail-open] 环境缺失（python 不可执行）→ 放行" >&2
  exit 0
fi
PYTHONPATH="$ROOT/src" "$PY" -m pysci.skills.orchestration.tools.night \
  --gate --settings "$ROOT/data/pysci_settings.toml" <<<"$PAYLOAD"
rc=$?
case "$rc" in
  0|2) exit "$rc" ;;
  *)
    echo "[night-gate fail-open] gate 异常 rc=$rc → 放行" >&2
    exit 0
    ;;
esac
