#!/bin/bash
# .qoder/hooks/night-gate.sh — 夜航模式 AskUserQuestion 机械拦截（PreToolUse）
#
# PSEP_2 §3.3a（实施单 task-20261014-psep2-n2）：matcher=AskUserQuestion；读
# data/用户的inbox/pysci_settings.toml 的 [night] enabled（**每回合现读，热切换**）。
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
#
# 2026-10-25（自包含改造 Phase 0）：CRLF 行尾曾把 fail-open 破坏成 fail-block（`exit 0\r`
# 致 bash 非零退出，PreToolUse 视为拦截）——本文件起强制 LF（.gitattributes 钉死）。
# 另路径处理改**双方言**（MSYS/WSL 皆可跑）：ROOT 由 $0 推导，传给 Windows python 的
# 一律 Windows 形态（cygpath → wslpath → 原样兜底）。
PAYLOAD=$(cat)
win_path() {
  if command -v cygpath >/dev/null 2>&1; then cygpath -m "$1"
  elif command -v wslpath >/dev/null 2>&1; then wslpath -w "$1" 2>/dev/null || printf '%s' "$1"
  else printf '%s' "$1"; fi
}
# 入口 cwd 在 cd 之前取样（pod 直聊会话判据；载荷带 cwd 时优先用载荷）。
ENTRY_CWD="$(pwd -W 2>/dev/null || wslpath -w "$(pwd)" 2>/dev/null || pwd)"
export PYSCI_HOOK_CWD="$ENTRY_CWD"
if [ -n "${QODER_PROJECT_DIR:-}" ]; then
  ROOT=$(cygpath -u "$QODER_PROJECT_DIR" 2>/dev/null || wslpath -u "$QODER_PROJECT_DIR" 2>/dev/null || printf '%s' "$QODER_PROJECT_DIR")
else
  ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
fi
cd "$ROOT" || exit 0
ROOTW=$(win_path "$ROOT")
PYW="$ROOT/.venv/Scripts/pythonw.exe"
if [ ! -x "$PYW" ]; then
  echo "[night-gate fail-open] 环境缺失（pythonw 不可执行）→ 放行" >&2
  exit 0
fi
PYTHONPATH="$ROOTW/src" "$PYW" -m pysci.skills.orchestration.tools.night \
  --gate --settings "$ROOTW/data/用户的inbox/pysci_settings.toml" <<<"$PAYLOAD"
rc=$?
case "$rc" in
  0|2) exit "$rc" ;;
  *)
    echo "[night-gate fail-open] gate 异常 rc=$rc → 放行" >&2
    exit 0
    ;;
esac
