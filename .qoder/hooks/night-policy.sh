#!/bin/bash
# .qoder/hooks/night-policy.sh — 夜航模式政策注入（SessionStart + PostCompact）
#
# PSEP_2 §3.2（实施单 task-20261014-psep2-n1）：读 data/pysci_settings.toml 的
# [night] enabled（**每回合现读，热切换**——用户改一行即生效，无需重启会话）。
#   ON  → 经既有钩子体系 hookSpecificOutput.additionalContext 注入政策全文（含 twin
#         授权书；判定与文本单点在 src/pysci/…/tools/night.py）；
#   OFF → 零注入零噪声（脚本无输出）。
# 触发点选 SessionStart + PostCompact（非每回合 Stop）：逐回合注入烧 token，而上下文
# 压缩正是政策最易丢的时刻——压缩后补一针即覆盖真风险。
#
# 安全：**一切异常 fail-open（exit 0、零输出）**——与 watch-arm.sh 同目录同风格，
# 绝不塌任何主链路（watch-arm 的 token 竞争 / wake-claim 均不经过本脚本）。
PAYLOAD=$(cat)
ROOT="${QODER_PROJECT_DIR:-/d/XXXIIIGGG/projects/pySci/pySciWS}"
cd "$ROOT" || exit 0
PY="$ROOT/.venv/Scripts/python.exe"
[ -x "$PY" ] || exit 0
PYTHONPATH="$ROOT/src" "$PY" -m pysci.skills.orchestration.tools.night \
  --settings "$ROOT/data/pysci_settings.toml" 2>/dev/null <<<"$PAYLOAD" || exit 0
exit 0
