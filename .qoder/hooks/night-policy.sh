#!/bin/bash
# .qoder/hooks/night-policy.sh — 夜航模式政策注入（SessionStart + PostCompact）
#
# PSEP_2 §3.2（实施单 task-20261014-psep2-n1）：读 data/用户的inbox/pysci_settings.toml 的
# [night] enabled（**每回合现读，热切换**——用户改一行即生效，无需重启会话）。
#   ON  → 经既有钩子体系 hookSpecificOutput.additionalContext 注入政策全文（含 twin
#         授权书；判定与文本单点在 src/pysci/…/tools/night.py）；
#   OFF → 零注入零噪声（脚本无输出）。
# 触发点选 SessionStart + PostCompact（非每回合 Stop）：逐回合注入烧 token，而上下文
# 压缩正是政策最易丢的时刻——压缩后补一针即覆盖真风险。
#
# 安全：**一切异常 fail-open（exit 0、零输出）**——与 watch-arm.sh 同目录同风格，
# 绝不塌任何主链路（watch-arm 的 token 竞争 / wake-claim 均不经过本脚本）。
#
# 2026-10-18（wake-bind-20261018 s1-noconsole）：钩子执行体可能是**无控制台**父进程
# （pythonw/DETACHED/harness 钩子执行体）——裸 spawn `python.exe` 会新开可见控制台。
# 统一改 **pythonw.exe**；stdout 由 harness 捕获（additionalContext JSON），不丢写。
PAYLOAD=$(cat)
# 入口 cwd 在 cd 之前取样（pod 直聊会话判据；载荷带 cwd 时优先用载荷）。
ENTRY_CWD="$(pwd -W 2>/dev/null || pwd)"
export PYSCI_HOOK_CWD="$ENTRY_CWD"
ROOT="${QODER_PROJECT_DIR:-/d/XXXIIIGGG/projects/pySci/pySciWS}"
cd "$ROOT" || exit 0
PYW="$ROOT/.venv/Scripts/pythonw.exe"
[ -x "$PYW" ] || exit 0
PYTHONPATH="$ROOT/src" "$PYW" -m pysci.skills.orchestration.tools.night \
  --settings "$ROOT/data/用户的inbox/pysci_settings.toml" 2>/dev/null <<<"$PAYLOAD" || exit 0
exit 0
