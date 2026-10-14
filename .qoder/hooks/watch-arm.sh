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
# v6（2026-10-14 晚，唤醒权自动归属；wake-owner-20261014，用户设计定稿「机制优于纪律」）：
#   v5 的 env 取证 + 手工 `wake-claim` 认领是「硬保证退化成软步骤」，已退役。现行**读取并
#   解析 stdin 载荷**（取代原先的整包丢弃）取 session_id / hook_event_name →
#   `wake-claim --check` 判定：
#     rc 0 = claim（无 owner 自动认领，首会话即主会话）/ owner（续心跳）/ takeover（owner
#           stale：心跳 >30min 且其 watch 已死 → 自动接管）/ fail-open（空载荷=看门狗重臂、
#           解析失败、无 session_id → **走旧 token 语义**并留痕）→ 照常监听；
#     rc 3 = 非 owner 且未 stale → 不监听、不抢 token、**绝不 exit 2**（退出留痕；事件留
#           持久水位，owner 下次挂载迟到唤醒补上——**不存在「转投其他会话语义**）；
#     rc 4 = SessionEnd → 释放检查完毕直接退出（会话正在结束，不监听）。
#   owner==我时把 session_id export 给 `orch watch`（落 watch.json 的 armed_by）——「owner
#   自己的 watch」才挡接管，看门狗重臂的无主替身不冒充。归属真源 orchestration/state/
#   wake-owner.json；`pysci-orch wake-claim status` 查看，`--force` 应急强制迁移（非常规）。
#
# 安全：任何异常一律 exit 0（观察面塌不了主链路）；token 文件缺失时自抢（不误判为被取代）。
PAYLOAD=$(cat)
ROOT="${QODER_PROJECT_DIR:-/d/XXXIIIGGG/projects/pySci/pySciWS}"
cd "$ROOT" || exit 0
LOG="$ROOT/.qoder/hooks/watch-arm.log"
log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*" >> "$LOG" 2>/dev/null; }

# 载荷解析（v6）：sed 取顶层 session_id / hook_event_name；缺席/解析不出 → 留痕并 fail-open。
# 顺手剥掉继承的 owner 身份：无载荷路径（看门狗重臂的无主替身 / 上游 shell 恰好带着
# PYSCI_WAKE_OWNER）若漏剥，替身 watch 会盖上别人的 armed_by、冒充「owner 自己的 watch」
# 把 stale 接管永久挡死（实测 2026-10-14 17:53 重臂现场抓到；watchdog._rearm 另有同源剥离）。
unset PYSCI_WAKE_OWNER
SID=$(printf '%s' "$PAYLOAD" | sed -n 's/.*"session_id"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' | head -1)
HEV=$(printf '%s' "$PAYLOAD" | sed -n 's/.*"hook_event_name"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' | head -1)
if [ -n "$SID" ]; then
  log "载荷解析 ok（session_id=$SID event=${HEV:-<缺>}）"
elif [ -n "$PAYLOAD" ]; then
  log "载荷解析失败/无 session_id → fail-open 走旧 token 语义"
else
  log "空载荷（看门狗重臂/无 stdin）→ fail-open 走旧 token 语义"
fi

PY="$ROOT/.venv/Scripts/python.exe"; ORCH="$ROOT/.venv/Scripts/pysci-orch.exe"
{ [ -x "$PY" ] && [ -x "$ORCH" ]; } || { log "环境缺失（python/orch 不可执行）→ 退出"; exit 0; }

# 会话结束（SessionEnd 钩子）：显式释放 owner 后退出——正在结束的会话不挂 watch。
if [ "$HEV" = "SessionEnd" ]; then
  if [ -n "$SID" ]; then
    dec=$(printf '%s' "$PAYLOAD" | "$ORCH" wake-claim --check 2>>"$LOG"); log "SessionEnd：${dec:-<无输出>}"
  else
    log "SessionEnd（无 session_id，跳过释放）"
  fi
  exit 0
fi

# 唤醒权自动归属闸（v6）：判定单一实现在 `wake-claim --check`（README §4.5）。
# rc 0=放行 / 3=非 owner 不抢 / 4=SessionEnd（上方已拦）；未知 rc=检查本身坏了 →
# fail-open 按旧 token 语义监听并留痕（唤醒链不塌）。
if [ -n "$SID" ]; then
  dec=$(printf '%s' "$PAYLOAD" | "$ORCH" wake-claim --check 2>>"$LOG"); wcr=$?
  case "$wcr" in
    0) log "归属判定：$dec" ;;
    3) log "归属判定：$dec → 不监听不抢，退出"; exit 0 ;;
    4) log "归属判定：$dec → 退出"; exit 0 ;;
    *) log "wake-claim --check rc=$wcr（${dec:-<无输出>}）→ fail-open 按旧 token 语义监听" ;;
  esac
  export PYSCI_WAKE_OWNER="$SID"
else
  log "无 session_id 可判定归属 → fail-open 按旧 token 语义监听"
fi

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
