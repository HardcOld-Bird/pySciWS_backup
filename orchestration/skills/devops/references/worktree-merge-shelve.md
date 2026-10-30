# 自动搁置失败的两级兜底（Tier A 20261021-174433）

> 本册是 `worktree-merge.md` §2 的细则分册；主册只留一行索引（basic.md §1 预算）。

## 背景

`merge_dirty_face` 是 devops 的历史高频失败标签：14 天 / 20 跳滑窗累计 7 次（阈值 3，
Tier A 自动立案）。逐条回看，7 次全部发生在**自动搁置**能力（backlog 20261017-051318，
`pysci-dev worktree merge` 默认 `--shelve`）上线**之前**；上线后默认路径是
「交集搁置 → 合并 → 三路应回」，台账最后一条事件为 **2026-10-16**，此后未再发生。

## 残余失败形状与兜底

自动搁置仍可能因下面两类原因返回 rc=2：

1. **pathspec 搁置失败**——`git stash push -u -- <paths>` 对未跟踪目录、已被并发
   改写的路径或并发 `index.lock` 报错；
2. **sha 漂移**——并发写者推入新 stash，`stash@{0}` 不再是自己刚建的那条。

对应两级兜底（`dev._shelve_paths`）：

- **第一级**：pathspec 搁置失败 → 回退 `git stash push -m <msg>`（**整面 tracked**，
  不带 `-u`）。只收 tracked 修改；未跟踪的运行态由 `_cp_ignored_state` 正常覆盖——
  若把未跟踪文件也 stash，应回时会与 cp 后的文件冲突（实测
  `could not restore untracked files from stash`）。
- **第二级**：stash sha 按**唯一 message**（`pysci-merge-shelve…`）在
  `git stash list` 中定位，找不到才退 `stash@{0}`；合并后按该 sha 精确 apply/drop。
- 两级都失败 → 按旧行为 rc=2（工作区不动、worktree 与分支保留供抢救）。

## 判据

- 回归测试：`test_shelve_falls_back_to_stash_all_on_pathspec_failure`（pathspec 失败 →
  整面 tracked 搁置，合并成功且 main 在飞改动完整应回）、
  `test_find_stash_by_message_ignores_stash0_drift`（按 message 定位）、
  `test_shelve_fails_both_paths_returns_rc2`（两级失败仍 rc=2）。
- 指标：`pysci-orch stats` 的 `by_failure` 段中
  `(merge_dirty_face, devops)` 在下一个 14 天 / 20 跳滑窗内计数 < 3。
- 组员侧仍遇到 rc=2 时照旧按 `<blocked tags="merge_dirty_face">` 交付并原样贴交集清单；
  本兜底只减少「搁置失败」这一支，不改变「真冲突/被 --no-shelve 拒绝」的语义。
