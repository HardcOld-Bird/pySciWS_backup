#!/bin/bash
# run-guard.sh <guard.mjs> [args...] — stable guard entry (self-contained toolchain).
# Runs orchestration/guards/<guard.mjs> with the vendored node (sub_root/deps_management/deps/node),
# so pod hooks never resolve bare "node" (system node removal silently disabled guards).
# stdin/stdout/stderr passthrough; exit code reported as-is.
# FAIL-OPEN: missing node/script -> exit 0 with a trace on stderr (guards are an
# observation layer and must not take down the main pipeline).
# 2026-10-26：取代 run-hook/run-guard 的 .cmd 形态——`cmd /c <反斜杠路径>` 在钩子执行
#   语境（MSYS bash 单次解析）下 `/c` 被路径转换、反斜杠被剥离，cmd 落入交互态输出
#   垃圾横幅且护栏静默失效（实测）；正斜杠 .sh 直跑形态对一切层间转义免疫。
set -u
GUARDS_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$GUARDS_DIR/../.." && pwd)"
NODE_HOME="$ROOT/sub_root/deps_management/deps/node"
NODE=""
if [ -f "$NODE_HOME/current.txt" ]; then
  NODE_VER=$(tr -d '\r\n' < "$NODE_HOME/current.txt")
  [ -f "$NODE_HOME/$NODE_VER/node.exe" ] && NODE="$NODE_HOME/$NODE_VER/node.exe"
fi
if [ -z "$NODE" ]; then
  echo "[run-guard fail-open] node not found under $NODE_HOME - run sub_root/deps_management/deps_manager/init-deps.cmd" >&2
  exit 0
fi
if [ ! -f "$GUARDS_DIR/${1:-}" ]; then
  echo "[run-guard fail-open] guard script not found: $GUARDS_DIR/${1:-}" >&2
  exit 0
fi
exec "$NODE" "$GUARDS_DIR/$1" "${@:2}"
