# plotting 分册 2/2

> 含小节：CLI argument forms (avoid the `new` vs `build` mix-up)
> 原 `plotting.md` 超 8KB 预算拆分（2026-10-10，backlog 20261010-refs-split-8kb）；分册索引见同目录 `plotting.md`，按需只读所需分册。

## CLI argument forms (avoid the `new` vs `build` mix-up)

The subcommands do **not** share one argument shape — conflating them is the single most common
CLI mistake (it yields `pysci-figures: error: unrecognized arguments`):

| Subcommand | Positional args | Example |
|---|---|---|
| `new` | `<research> <slug>` (a *pair*) | `figures new gain_ep fig1_ep_band` |
| `build` / `preview` / `audit` | one `<figdir>` *path* | `figures build 'data/research/1_gain_ep/article/figures/fig1_ep_band'` |
| `list` | `<research>` | `figures list gain_ep` |

`new` **creates** the two sides (code module `src/pysci/research/<name>/article/figures/<slug>.py`
+ data dir `data/research/<n>_<name>/article/figures/<slug>/`) and prints the exact `build '<figdir>'`
command to run next. Everything that **operates on** an existing figure takes the single data-side
`<figdir>` path — never the `<research> <slug>` pair. Quote the path in Git Bash. When in doubt,
`figures build -h` prints the accepted form and an example.
