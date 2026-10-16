# 任务书：DEBT 清零——9 处低频裸 spawn 站点并入隐藏口径

## 背景
空终端风暴根治（adhoc-storm-fix，`844f05d`）建立了唯一 spawn 口径
`procs.run_hidden`/`popen_hidden`（win32 恒 `CREATE_NO_WINDOW`），并留了 **DEBT 清单**
（登记在回归用例里，新裸 spawn 即红）：9 处低频站点尚未并入——
`sync.py`、`registry.py`×2、`replica.py`×2、`sessions.py`×2、`verdict.py`、`jury.py`、
`eval.py`、`runner.py`（+ `drain.py` 余 1 处）。以用例 DEBT 清单为准（逐条带说明）。

这些站点在**无控制台的分离进程**（队列 worker/看门狗/巡更）里被调时同样会闪空窗，
只是频度低——风暴修复的判据③完整形态即本单。

## 修复要求
1. DEBT 清单内全部站点并入 `run_hidden`/`popen_hidden` 口径（POSIX 行为零变化）；
2. **清空用例里的 DEBT 清单**（该用例转为「除 procs.py 外全树无裸 spawn」的硬闸）；
3. 顺检全树 `subprocess.run`/`Popen` 无漏网（grep 核对，含本单未列的新增站点）。

## 验收预期
1. DEBT 清单为空，硬闸用例全绿；
2. 全量 pytest 绿（哨兵零写穿）；
3. `grep -rn "subprocess.run(\|Popen(" src/pysci/skills/orchestration/tools/*.py` 除
   procs.py 口径实现外零命中（或逐条注明豁免理由）。

## 白名单（写权限）
- src/pysci/skills/orchestration/tools/
- tests/
