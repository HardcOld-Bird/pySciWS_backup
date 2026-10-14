# 技能部署与 pysci-dev 命令行（devops SKILL.md 的展开）

SKILL.md 只给红线与速查；本文件给逐项语义、旗标与部署细则。

## 技能部署（真本 → 副本）

- 真本：`orchestration/skills/<name>/`（git 跟踪）；manifest：同目录 `manifest.toml`；
- 改完真本执行 `pysci-dev sync`（与 pysci-orch sync 同引擎）；`sync --check` 查漂移；
- `tree_hash` **行尾无关**（哈希前 CRLF→LF）：worktree 内 sync 提交的
  `skills-deployed.json` 对 main 即正确，无需合并后重跑；
- 红线：只碰 manifest 声明的部署名；**不删除成员自建技能**；不手改部署副本。
- **真本 references 超 8KB 的处置**：按 `##` 主题拆 `<stem>-0N-<slug>.md` 分册，所属
  SKILL.md 索引行标 ``*``；搬运纪律与次序（含 pre-commit 先改写文件的坑）见
  `budget-layers.md` §4。
- **机械生成/改写 markdown 的 EOF 窗口**：`pysci-dev mdgen-check <paths…>` 提前把
  pre-commit（trailing-ws + end-of-file-fixer）改写**做完**并打印 pre/post sha256；
  等价性断言基准=**post_sha256**（钩子终态），非生成前字节。首提必过。

## pysci-dev 命令面

```
pysci-dev doctor [--pod <id>]   # pod 巡检：hooks/leader 排除/charter/AGENTS 体量/积压
                                 # + 部署台账（sync --check）+ 预算三档（≤8192B/文件）
                                 # + API Key 泄漏哨兵（tests/ 豁免）+ 跟踪 state 脏面
                                 #   （advisory：脏=组员合并前检查会拦的东西）
pysci-dev sync [--check]        # 技能部署
pysci-dev worktree add|merge|remove|list|clean [<name>]  # merge 成功自动摘面(--keep 留；clean 清遗留)
pysci-dev push [--timeout S]    # 推 origin/main（跨平台超时 + 成败留痕；best-effort）
pysci-dev mdgen-check <paths…>  # 提前做完 pre-commit 对 md 的改写
pysci-dev doc-faces <paths…|--changed>  # 动的源文件→文档面（改语义别忘对盘；不阻断）
pysci-dev venv-sync             # 主树同步入口/依赖（裸 uv sync 删光入口，见 uv-venv-sync.md）
pysci-dev probe                 # hook 接线因果断言（doctor 验不出）
pysci-dev skilltax              # 平台技能税关停核查
```

`push` 失败属**常态**（代理常态关、校园网仅偶尔直连）：best-effort、不重试、不设代理、
不为它上报/焦虑/加告警面。
