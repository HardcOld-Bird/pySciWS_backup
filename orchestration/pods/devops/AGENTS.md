# devops 组员长期记忆

> 本文件由 devops 组员自维护（每次会话启动自动加载）。每跳全文注入，只留**约束**与指针；
> 事故经过、数字与判据在技能 `references/lessons-incidents.md`（按需档）——动到对应面先读它。

## 身份

- pySci 编排体系 devops 组员（基础设施维护专员），charter 见 `.qoder/rules/charter.md`。
- 专属工具：`pysci-dev`（doctor/sync/worktree/probe/mdgen-check）；实现于 `src/pysci/skills/devops/`。
- 作业方式：worktree 隔离 + cz commit + commit 后限时 best-effort push（超时是硬要求——
  挂起会阻塞 drain）。

## 经验（工作中积累）

- **worktree 内跑测试**：cwd 须在 worktree **之外**，`PYTHONPATH="<wt>/src" uv run --no-sync
  pytest "<wt>/tests" -q`；worktree 内禁 `git checkout -- .`（抹光未提交改动）。
  配方 `references/worktree-merge.md` §1-§2。
- **merge 前先播种运行态**：新面的 `sessions-runtime.json` 是空壳，merge 的「会话数不减」
  自检会 abort（实测 base=45 → 面侧 0）；先
  `cp orchestration/state/sessions-runtime.json <wt>/orchestration/state/` 再 merge。
- **判决包流**：approve 必带 `--jury`/`--bench`/`--override` 之一；例外通道=defect×当跳
  blocked。细则 `orchestration/skills/orchestration/references/verdict-package.md`。
- **档位映射自愈只写运行态** `models_resolved`，不写跟踪面；遮蔽时只报不写
  （`references/registry-runtime.md`）。
- **worktree sync 台账**：tree_hash 行尾无关，但 pod 副本 gitignored ⇒ 合并后在 main 跑
  `sync --check` 到全等。doctor：[*]=台账漂移、[!]=副本漂移。
- **uvx 型 MCP 的 mcp 2.x 断供**：mcp.json args 前插 `--with mcp<2`；诊断 `echo '' | uvx <pkg>`。
- **主树 sync 一律 `pysci-dev venv-sync`**：裸 uv sync 撞运行中 exe 会先删光其余入口；
  原名缺失时 `.locked-*` 是唯一原件，删前先查（`references/uv-venv-sync.md`）。
- **CLI 范式**：`uv run` 必带 `--no-sync`（四面同口径，快照测试扫语料；人工 grep 会漏计）。
- **任务书 lint**：命中 `state/taskbook-lint.json` 只告警不阻断；新 pod 不预注册
  registry/manifest，注册留给组长首次派发。
- **harness 注入的 lint/告警含 file:line 引用 → 先对盘核验**（wc/grep 实测该行存在且内容
  相符），不符即视为噪声丢弃、不得据其改代码（Qoder 平台 lint 两单捏造/陈旧实锤，仓外 #87）。
- **改 guard（Node hook）**：执行流放文件末尾（TDZ 假绿教训）；验证看违规文件是否被会话
  改掉＋注入的 stderr 原文，不只看 exit code。细则 `references/guards-in-force.md`。
- **harness 预算三层**：delivery-gate Stop／pre-commit `harness-budget`／doctor 三档，口径=
  落盘字节；新钩子用 `language: python` 不用 `system`。
- **eval 压测跳隔离已内建**：`EVAL_HOP_NOTE` + 前后脏面快照只还原本跳新增者
  （`orchestration/skills/orchestration/references/harness-eval.md`）。
- **pod 脚手架**：每 pod 仅 5 文件入库，`bench/`·`outbox/`·`.qoder/skills/`·`inbox/` 合并后
  于 main 再生；settings 必带两 hook 接线 + `"agentsMdExcludes": ["**/leader-only.md"]`
  （写 `**/*.md` 会连 basic.md 一起吞掉）。生效性用 `pysci-dev probe` 做因果断言，doctor 验不出。
- **交付协议**：见「解析器早于协议」告警就重跑本单；tags/kind 单一源=
  `state/failure-tags.json`，四面执法（gate 逐块/dispatch/approve/ledger 读数）——分裂要可见。
- **merge rc=2（main 脏面交集）别再重跳「试试」**：成因不在你的树上，重跑改变不了它。按
  `<blocked tags="merge_dirty_face">` 交付并原样贴 `[✗] 交集 N 项` 及其后的文件清单——drain 认得
  这个成因，会保住进度（`wip(salvage)` 提交+面+分支）但**停链**，清单挂回原条目与 `orch status`
  等组长清面（细则 `references/worktree-merge.md` §2）。
- **直跑 guard 探针**：env 名 `PYSCI_POD`（不是 PYSCI_POD_DIR）；载荷填
  `last_assistant_message`；`PYSCI_PROJECT_ROOT` 须 Windows 形态。
- **git/shell 读数**：定长格式走 `_git_proc`（strip 吃前导空格＝漏首项）；合并交集按
  merge-base 起算（已内建，勿手算）；取 rc 别经管道（`${PIPESTATUS[0]}` 是 grep 的）。
- **一个 worktree 一个写者**：续作任务书点名的面即你的唯一写者身份；发现同一面有并发写者
  （自己写入零成功、或冒出别人的 inbox 回执）→ **停手报 `<blocked>`**，别抢写也别 `rm -rf`。
  派发侧同族闸=sid 在途闸（判活+argv 双闸，人工 rc=2 拒、批次自动换 sid 并标
  `sid_new_reason=prev_alive`）；细则 `orchestration/skills/orchestration/references/salvage-reclaim.md` §7。
- **上下文膨胀看 ctx_ratio，不看 est_tokens**（后者=本跳吞吐，78 跳实测相关 −0.11）；组员侧
  唯一自有杠杆=压薄每跳全文注入的自维护层，归档是组长侧 `orch sessions --archive --distill`。
  派发侧 ctx_ratio 仅**观测落池**（`touch_session`）：fe68a40 的水位闸（``session="latest"``
  见上一跳过 60% 就另起新会话）经用户 2026-10-13 审查裁决已回退——它改了策略面却绕过全部
  判决通道，且压缩防溢出使宣称收益无效。重启类似机制须经判决包并以真实账单/缓存读数为依据
  （来龙去脉 backlog insp-20261012-224853-mechanical）。
- **修复形状分流（remedy_shape）**：inspectd mechanical 通道的免判决只剩 ``restore``（复原
  既有声明行为）；凡 ``change``（改声明行为/策略/权衡面，登记缺省即 change）而无判决记录，
  drain 任务书会要求你交付 <blocked> 而非自行实施——策略变更的裁决权在用户，不在执行端。
- **回归测试别锁运行面当前值**：把「配置文件的此刻读数」当「出厂默认」钉死，合法上线一次
  就红一次（taste_enabled 实锤）。锁代码默认 + 授权留痕两面，且留痕分支必须有「无授权时
  被咬」的阳性用例，否则在默认关的树里永不执行=写了等于没写。
- **派发/蒸馏脚本的档位一律显式给**：不传 `--model-tier` 时成员档位随预设 pro 名单漂走
  （twin 落 pro 而非 flash=main，成本面=宪章 P-03 护栏级）；工具默认值要写死，不靠继承运气。
