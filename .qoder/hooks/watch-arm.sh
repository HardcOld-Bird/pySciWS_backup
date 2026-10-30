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
# v7（2026-10-15 晚，夜航热生效；adhoc-night-text）：回合边界顺带比对 `[night] enabled` 与
#   本会话上次读数（night.py `--diff`，读数存 watch.json 的 night_seen）——有跳变即用本钩子
#   既有的 rewake 通道（stderr + exit 2）在**下一轮**注入差分（关→开=授权书全文、开→关=刹车
#   差分），无变化零输出。开关**只归用户手动**（不实现自拨回/时限到期）。
#
# 安全：任何异常一律 exit 0（观察面塌不了主链路）；token 文件缺失时自抢（不误判为被取代）。
#
# v8（2026-10-18，空终端根治；wake-bind-20261018 s1-noconsole）：本脚本由 `watchdog._rearm`
#   经 `DETACHED_PROCESS|CREATE_NO_WINDOW` 拉起——**父进程完全没有控制台**。此时裸 spawn
#   console 子系统的 `python.exe` / `pysci-orch.exe` 会让 Windows 分配**新的可见控制台**
#   （标题=解释器路径），看门狗反复重臂即反复闪窗。所有 venv 调用改走 **pythonw.exe**
#   （GUI 子系统，永不分配控制台）；`orch` 改 `-m ...tools.orch`（有 __main__ 守卫）。
#   每处调用的 stdout/stderr 重定向与命令替换语义保持不变（pythonw 下未重定向才会丢写）。
#
# v9（2026-10-25，自包含改造 Phase 0）：强制 LF 行尾 + 双方言路径归一（见 night-gate.sh
#   同日注）：ROOT 由 $0 推导、传给 Windows python 的一律 Windows 形态（cygpath → wslpath
#   → 原样兜底）。CRLF 曾把所有 exit 路径变 bash 致命错误，fail-open 名存实亡。
#
# v10（2026-10-27，wake-repair：唤醒通道/token 竞争/coreutils 三面修复）。实测死循环
#   （复现沙箱三轮 rc=2 空转，与生产 2026-10-27 事故同签名）：看门狗重臂的补位实例**无
#   harness 附着**，命中事件「stderr + exit 2」无人接收（喊给空气）→ 退出 → 看门狗再臂 →
#   再命中，事件永远滞留、组长永远不醒；且化石 PATH（缺 MSYS usr\bin）令 date/cat/sed 全部
#   command not found——日志时间戳 []、token `$$-`、cat 失败使 token 每轮误判「缺失」自抢，
#   把**活的会话 watch 顶下线**（生产 10:19:53 实测「被取代→让位退出」）。三面修复：
#   ① coreutils 自愈：$BASH 所在目录前插 PATH（须在 `PAYLOAD=$(cat)` 之前——cat 也靠它）；
#   ② token 竞争：`<pid>-<epoch>`，持有者每轮刷新 epoch=心跳、让位判定按 pid；补位实例
#      （无 SID）接管前查持有者心跳，新鲜即「会话 watch 在岗」不顶替退出（会话附着实例
#      维持 v2「每回合接管」，不走此闸）；
#   ③ 补位实例挂 `orch watch --guardian`（看护模式）：命中只登记不退出不发唤醒、基线不
#      推进（owner 下次挂载迟到唤醒），心跳不断则看门狗不再重臂空转；命中分支对无 SID
#      实例绝不 exit 2（无接收方）。
# coreutils 自愈（v10①）：$BASH 即本 bash 绝对路径，其目录就是 coreutils 所在。
case ":$PATH:" in
  *":${BASH%/*}:"*) : ;;
  *) PATH="${BASH%/*}:$PATH"; export PATH ;;
esac
PAYLOAD=$(cat)
win_path() {
  if command -v cygpath >/dev/null 2>&1; then cygpath -m "$1"
  elif command -v wslpath >/dev/null 2>&1; then wslpath -w "$1" 2>/dev/null || printf '%s' "$1"
  else printf '%s' "$1"; fi
}
# 入口 cwd 必须在 `cd "$ROOT"` **之前**取样（下一行就会抹掉）：归属闸的 pod 判据读它
# （载荷带 cwd 时优先用载荷；这里是无 cwd 载荷的兜底）。输出 Windows 形态，
# 避免 POSIX 路径被 Python 误读成相对路径。
ENTRY_CWD="$(pwd -W 2>/dev/null || wslpath -w "$(pwd)" 2>/dev/null || pwd)"
export PYSCI_HOOK_CWD="$ENTRY_CWD"
if [ -n "${QODER_PROJECT_DIR:-}" ]; then
  ROOT=$(cygpath -u "$QODER_PROJECT_DIR" 2>/dev/null || wslpath -u "$QODER_PROJECT_DIR" 2>/dev/null || printf '%s' "$QODER_PROJECT_DIR")
else
  ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
fi
cd "$ROOT" || exit 0
ROOTW=$(win_path "$ROOT")
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

PYW="$ROOT/.venv/Scripts/pythonw.exe"
[ -x "$PYW" ] || { log "环境缺失（pythonw 不可执行）→ 退出"; exit 0; }
orch() { PYTHONPATH="$ROOTW/src" "$PYW" -m pysci.skills.orchestration.tools.orch "$@"; }

# 会话结束（SessionEnd 钩子）：显式释放 owner 后退出——正在结束的会话不挂 watch。
if [ "$HEV" = "SessionEnd" ]; then
  if [ -n "$SID" ]; then
    dec=$(printf '%s' "$PAYLOAD" | orch wake-claim --check 2>>"$LOG"); log "SessionEnd：${dec:-<无输出>}"
  else
    log "SessionEnd（无 session_id，跳过释放）"
  fi
  exit 0
fi

# 唤醒权自动归属闸（v6）：判定单一实现在 `wake-claim --check`（README §4.5）。
# rc 0=放行 / 3=非 owner 不抢 / 4=SessionEnd（上方已拦）；未知 rc=检查本身坏了 →
# fail-open 按旧 token 语义监听并留痕（唤醒链不塌）。
if [ -n "$SID" ]; then
  dec=$(printf '%s' "$PAYLOAD" | orch wake-claim --check 2>>"$LOG"); wcr=$?
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

# 夜航差分注入（v7，2026-10-15 热生效单 adhoc-night-text）：回合边界比对 `[night] enabled`
#   与本会话上次读数（night.py `--diff`，读数寄存在 watch.json 的 night_seen）。
#   有跳变 → **走本钩子既有 rewake 通道**（stderr + exit 2）在**下一轮**注入差分：
#     关→开 = 授权书全文（同 SessionStart 注入文本）；开→关 = 刹车差分。
#   无变化 → 零输出零打扰；判定与文案单点在 night.py，本脚本只做通道。
#   **归属闸之后**（2026-10-18 wake-bind s2）：非 owner 会话（rc 3/4）在上一段已退出，
#   夜航差分只可能作用在 owner/认领会话上——子会话不再被夜航差分自动 wake。
#   fail-open：读不出/异常一律零输出继续走原有链路。
if [ -n "$SID" ]; then
  ndiff=$(printf '%s' "$PAYLOAD" | PYTHONPATH="$ROOTW/src" "$PYW" -m pysci.skills.orchestration.tools.night \
    --diff --session "$SID" --settings "$ROOTW/data/用户的inbox/pysci_settings.toml" 2>>"$LOG")
  ndrc=$?
  if [ -n "$ndiff" ]; then
    log "夜航差分注入（session=$SID）→ exit 2（rewake）"
    printf '%s\n' "$ndiff" >&2
    exit 2
  fi
  [ "$ndrc" -eq 0 ] || log "夜航差分 rc=$ndrc → fail-open 继续监听"
fi

# 看门狗自启动（v4）：幂等——在场则秒返回，不在场则拉起分离监督进程（冷启动后即在位）。
wd=$(orch watchdog start 2>&1); wdrc=$?
case "$wd" in
  *已后台启动*) log "看门狗不在场 → 自启动（$wd）" ;;
  *) [ "$wdrc" -ne 0 ] && log "watchdog 自启动失败（rc=$wdrc）：$wd" ;;
esac

TOKEN="$ROOT/.qoder/hooks/watch-holder.token"
# token 语义（v10②）：`<pid>-<epoch>`——持有者每轮刷新 epoch（=心跳），让位判定按 **pid**。
TOKEN_FRESH_S=300
# 补位实例（无 SID：看门狗重臂/无 stdin）接管前先查持有者心跳：新鲜（≤TOKEN_FRESH_S）即
# 会话 watch 在岗 → 不顶替退出（2026-10-27 实测：补位实例曾把活的会话 watch 顶下线，此后
# 只剩 exit 2 无人接收的替身）。会话附着实例维持 v2「每回合接管」，不走此闸。
if [ -z "$SID" ]; then
  cur=$(cat "$TOKEN" 2>/dev/null)
  cepoch=${cur#*-}
  if [ -n "$cur" ] && [ "$cepoch" -gt 0 ] 2>/dev/null && [ $(( $(date +%s) - cepoch )) -le "$TOKEN_FRESH_S" ]; then
    log "token 持有者心跳新鲜（$cur）→ 会话 watch 在岗，补位不接管，退出"
    exit 0
  fi
fi
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
  [ "${cur%%-*}" = "$$" ] || { log "被取代（现持 $cur）→ 让位退出"; exit 0; }
  printf '%s\n' "$$-$(date +%s)" > "$TOKEN" 2>/dev/null  # 刷新持有心跳（epoch）
  if [ -n "$SID" ]; then
    out=$(orch watch --timeout "$CYCLE" 2>&1)
  else
    out=$(orch watch --guardian --timeout "$CYCLE" 2>&1)  # v10③：看护模式，命中不退出
  fi
  rc=$?
  [ "$rc" -ne 0 ] && { log "watch rc=$rc → 退出"; exit 0; }
  case "$out" in
    *TIMEOUT*)
      # 看护登记行与 TIMEOUT 同帧时只进日志（否则看护命中在排障面完全静默——
      # 「零命中=没听见」的判读会把「听见了但叫不醒」误诊成聋）。
      note=$(printf '%s\n' "$out" | grep -v 'TIMEOUT' | grep -v '自动重挂')
      [ -n "$note" ] && log "看护登记：$note"
      ;;
    *)
      if [ -n "$SID" ]; then
        log "命中事件 → 唤醒组长"; printf '%s\n' "$out" >&2; exit 2
      else
        # v10③：补位实例 exit 2 无人接收（实测喊给空气）——只留痕退出，绝不假唤醒。
        log "命中事件但无会话附着（看门狗补位）→ 不发唤醒（exit 2 无接收方）"; log "$out"; exit 0
      fi ;;
  esac
done
