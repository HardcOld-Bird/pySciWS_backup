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
# v3（2026-10-14，日志完整化；autonomy-fix a1）：2026-10-14 02:23–07:40 五小时唤醒真空，取证
#   结论=watcher 被**外部无声杀死**（本脚本所有**正常**退出路径都留痕，零留痕只能是外部终止，
#   详见 README §4.5）。本轮把剩余盲区补齐：环境缺失 / token 写失败 / token 自抢分支都留痕；
#   另加 TERM/INT 陷阱留痕（SIGKILL 无法拦截——由看门狗按 watch 心跳判死重臂，`orch watchdog`）。
#
# v4（2026-10-14，看门狗自启动；backlog 20261014-104738）：看门狗此前只由手工 `orch watchdog
#   start` 拉起——冷启动/重启后无人敲它，则「watcher 被杀即重臂」这条链在最需要它的时刻（冷启
#   动后）空转。本轮在每回合**幂等**自启动看门狗（已在跑则秒返回；仅在实际拉起/失败时留痕）。
#
# v5（2026-10-14，唤醒权认领制；wake-claim-20261014）：用户主会话工作时另开次要会话处理琐事，
#   而 v2 的「最新在听者胜」让次要会话一有回合就抢走唤醒权，drain/事件的 exit-2 被投到次要
#   会话（污染其上下文，正是用户要避免的；凌晨「被取代却无接任」的谜之日志疑为同源）。本轮
#   加**认领闸**：`pysci-orch wake-claim` 认领后，**只有 owner 会话的钩子**抢 token/唤醒，
#   非 owner 钩子直接退出——不监听、不抢、**不 exit 2**。事件不会丢：持久水位 + 迟到唤醒由
#   owner 下次挂载补（a1 语义），**不存在「转投其他会话」语义**。未认领 = legacy 行为不变。
#   判据在 `wake-claim --check`（两端同源的会话身份；取证见 README §4.5 与 watch.py）。
#
# 安全：任何异常一律 exit 0（观察面塌不了主链路）；token 文件缺失时自抢（不误判为被取代）。
cat >/dev/null
ROOT="${QODER_PROJECT_DIR:-/d/XXXIIIGGG/projects/pySci/pySciWS}"
cd "$ROOT" || exit 0
LOG="$ROOT/.qoder/hooks/watch-arm.log"
log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*" >> "$LOG" 2>/dev/null; }
PY="$ROOT/.venv/Scripts/python.exe"; ORCH="$ROOT/.venv/Scripts/pysci-orch.exe"
{ [ -x "$PY" ] && [ -x "$ORCH" ]; } || { log "环境缺失（python/orch 不可执行）→ 退出"; exit 0; }

# 一次性会话身份取证（v5）：把本钩子可见的区分性变量记进日志，供 README §4.5 固化判据。
SEEN="$ROOT/.qoder/hooks/session-env.seen"
if [ ! -f "$SEEN" ]; then
  log "会话身份取证（一次性）：WT_SESSION=${WT_SESSION:-<缺>} QODER_PROJECT_DIR=${QODER_PROJECT_DIR:-<缺>} PYSCI_POD=${PYSCI_POD:-<缺>}"
  { : > "$SEEN"; } 2>/dev/null
fi

# 唤醒权认领闸（v5）：0=owner/legacy 可监听；3=非 owner（不抢 token、不唤醒）；其余=fail-open。
# 检查本身失败（旧二进制/环境坏）不得让唤醒链塌掉 → 按 legacy 放行并留痕。
"$ORCH" wake-claim --check >/dev/null 2>&1; wcr=$?
case "$wcr" in
  3) log "非 owner 会话 → 不监听不抢（唤醒权在 owner：pysci-orch wake-claim status 可查）"; exit 0 ;;
  0) : ;;
  *) log "wake-claim --check rc=$wcr → fail-open 按 legacy 监听" ;;
esac

# 看门狗自启动（v4）：幂等——在场则秒返回，不在场则拉起分离监督进程（冷启动后即在位）。
wd=$("$ORCH" watchdog start 2>&1); wdrc=$?
case "$wd" in
  *已后台启动*) log "看门狗不在场 → 自启动（$wd）" ;;
  *) [ "$wdrc" -ne 0 ] && log "watchdog 自启动失败（rc=$wdrc）：$wd" ;;
esac

TOKEN="$ROOT/.qoder/hooks/watch-holder.token"
MY="$$-$(date +%s)"
{ printf '%s\n' "$MY" > "$TOKEN" && log "接任 watch（token=$MY）"; } || { log "token 写失败 → 退出"; exit 0; }
trap 'log "收到 TERM/INT（token=$MY）→ 退出"; exit 0' TERM INT

CYCLE=120
while :; do
  cur=$(cat "$TOKEN" 2>/dev/null)
  if [ -z "$cur" ]; then
    printf '%s\n' "$MY" > "$TOKEN" 2>/dev/null; cur="$MY"
    log "token 缺失 → 自抢（token=$MY）"
  fi
  [ "$cur" = "$MY" ] || { log "被取代（现持 $cur）→ 让位退出"; exit 0; }
  out=$("$ORCH" watch --timeout "$CYCLE" 2>&1); rc=$?
  [ "$rc" -ne 0 ] && { log "watch rc=$rc → 退出"; exit 0; }
  case "$out" in
    *TIMEOUT*) : ;;
    *) log "命中事件 → 唤醒组长"; printf '%s\n' "$out" >&2; exit 2 ;;
  esac
done
