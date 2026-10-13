#!/bin/bash
# .qoder/hooks/watch-arm.sh — Stop + asyncRewake：回合结束自动接管 drain 事件等待器
#
# v2（2026-10-13，缺陷修复）。实测教训（README §4.5）：
#   v1 的「已有活跃 watch 则放行」守卫，在**会话重启 / 中途回合**下会误伤——旧那任 hook 把
#   watch 挂住，新会话的 hook 被守卫挡成 no-op（exit 0），于是**当前会话根本没有在听的 hook**；
#   等旧 hook 命中事件时 exit 2，harness 只认**最近一任**的唤醒，早已唤不动重启后的会话
#   （实测 2026-10-13 20:57 丢失 QUEUE-COMPLETE 唤醒）。
#   v2 改为**每回合接管**：本 hook 以 token 抢占"最新在听者"身份并挂 watch；旧 hook 在下一个
#   周期发现自己被取代即让位退出（避免多 watcher 叠加）。唤醒权始终归当前会话最近一任。
#
# 安全：任何异常一律 exit 0（观察面塌不了主链路）；token 文件缺失时自抢（不误判为被取代）。
cat >/dev/null
ROOT="${QODER_PROJECT_DIR:-/d/XXXIIIGGG/projects/pySci/pySciWS}"
cd "$ROOT" || exit 0
PY="$ROOT/.venv/Scripts/python.exe"; ORCH="$ROOT/.venv/Scripts/pysci-orch.exe"
{ [ -x "$PY" ] && [ -x "$ORCH" ]; } || exit 0
TOKEN="$ROOT/.qoder/hooks/watch-holder.token"
LOG="$ROOT/.qoder/hooks/watch-arm.log"
log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*" >> "$LOG" 2>/dev/null; }
MY="$$-$(date +%s)"
printf '%s\n' "$MY" > "$TOKEN" 2>/dev/null || exit 0
log "接任 watch（token=$MY）"

CYCLE=120
while :; do
  cur=$(cat "$TOKEN" 2>/dev/null)
  if [ -z "$cur" ]; then printf '%s\n' "$MY" > "$TOKEN" 2>/dev/null; cur="$MY"; fi
  [ "$cur" = "$MY" ] || { log "被取代 → 让位退出"; exit 0; }
  out=$("$ORCH" watch --timeout "$CYCLE" 2>&1); rc=$?
  [ "$rc" -ne 0 ] && { log "watch rc=$rc → 退出"; exit 0; }
  case "$out" in
    *TIMEOUT*) : ;;
    *) log "命中事件 → 唤醒组长"; printf '%s\n' "$out" >&2; exit 2 ;;
  esac
done
