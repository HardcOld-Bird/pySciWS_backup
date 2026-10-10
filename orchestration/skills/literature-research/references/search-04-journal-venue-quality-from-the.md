# search 分册 4/5

> 含小节：`journal` — venue quality from the free layers；`library` — query the user's Zotero
> 原 `search.md` 超 8KB 预算拆分（2026-10-10，backlog 20261010-refs-split-8kb）；分册索引见同目录 `search.md`，按需只读所需分册。

## `journal` — venue quality from the free layers

```
research journal lookup <issn> [--json]
research journal build-scimago --csv <path> [--year 2024]
research journal status [--json]
```

Answers "how good is this venue?" **with no paid credential**, by putting the free layers side by
side. The official JIF / JCR quartile / JCI / ESI can only come from the **WoS Journals API** (applied
for, not yet granted; the WoS *Starter* API does not carry them, and Elsevier/Scopus need paid keys),
so `lookup` prints four numbered sections and is explicit about the fourth:

1. **SCImago SJR** — quartile / SJR value / h-index / index version year, from the local index, matched **by ISSN exactly**.
2. **OpenAlex `listed_in` → `journal_tier`** — the expert-panel grade, the exact tags it was derived from, and the full `listed_in` list. Zero extra requests.
3. **OpenAlex citation metrics** — `2yr_mean_citedness` (the `jif` estimate) and the journal `h_index`.
4. **Official JCR** — `jcr_quartile` / official JIF / JCI / ESI: printed as *未接入* rather than left blank, so an unavailable number is never mistaken for a measured zero.

`lookup` also states **why** a layer is empty, because the two causes call for different fixes:
*索引未建* → run `build-scimago`; *该 ISSN 不在索引里* → the venue genuinely has no SJR record. Same
for OpenAlex: *未收录该 ISSN，或网络不可达*. The `jif` value is rendered at **2 decimals** in the human
output to match what a note's frontmatter stores (`round(x, 2)`); `--json` keeps full precision, since
that output is for programs and dropping digits there would be a loss with no upside.

`lookup --json` keys: `issn`, `issn_normalized`, `scimago` (`{issn, quartile, sjr, h_index,
sjr_year}` or `null`), `openalex` (the raw source fields — `display_name`, `issn`, `issn_l`,
`publisher`, `country_code`, `h_index`, `works_count`, `2yr_mean_citedness`, `listed_in`), `journal`,
`jif`, `journal_h_index`, `listed_in`, `journal_tier`, `journal_tier_basis`. A missing ISSN exits **2**.

### `build-scimago` — one-time local index

The official CSV must be **downloaded by hand** from `https://www.scimagojr.com/journalrank.php` (the
site returns 403 to programmatic requests and needs form/JS interaction, so no auto-fetch is
implemented):

```
research journal build-scimago --csv <path-to-csv> [--year 2024]
```

Only `Issn` / `SJR` / `SJR Best Quartile` / `H index` are kept — enough for quartile judgment, and it
compresses a ~11 MB CSV into a ~1.4 MB JSON that git can carry. Output:
`data/skills/literature_research/data/scimago_index.json` (**tracked**); `*.csv` under that tree is
git-ignored, so the raw download may sit beside the index or be deleted — the index is self-sufficient.
Multi-valued `Issn` cells produce **one key per ISSN** pointing at the same record; the 2025 export
quotes them and separates with a comma, unhyphenated (`"10797114, 00319007"`), which is why that
build yielded **53,404** keys from a 32,194-line CSV. The version year is inferred from the
header (`Total Docs. (2025)`) or the file name, and `--year` overrides it — it is never invented.
ISSNs are normalized by stripping hyphens, so `0031-9007` and `00319007` both match.

> **This one path is deliberately *not* silent.** Everywhere else in the module a missing input
> degrades quietly, but a failed build that quietly wrote an empty index would leave
> `scimago_quartile` blank in *every* note with no visible cause — far harder to diagnose than an
> error. So: missing `--csv` → exit **2**; unreadable, empty, or column-less CSV → exit **1** with the
> reason on stderr. After a successful build, record the download URL / version year / download date
> in `data/skills/literature_research/data/SOURCE.md` (the attribution line is printed for you), and
> refresh yearly — see `references/maintenance.md`.

### `status`

Reports `exists` / `readable` / `path` / `sjr_year` / `n_entries` / `built_at` / `source_csv` /
`attribution` / `download_url` (also as `--json`). Exit **0** when the index is simply absent — that
is a supported state: `scimago_quartile` stays empty while `journal_tier` and `jif` are unaffected —
and exit **1** when the file exists but cannot be parsed (rebuild it). `research doctor` prints the
same summary under its journal-metrics section.

---

## `library` — query the user's Zotero

These subcommands are a thin differentiated facade over the community **zotero-mcp**'s
`zotero-cli --json`; they exit with a clear install notice when `zotero-cli` isn't on PATH. For
deeper library work — reading full text / **PDF page images**, merging duplicates, Scite
retraction alerts — drive the registered `zotero` MCP directly.

```
research library ping
research library list [--limit N] [--type journalArticle]
research library search --query '<terms>' [--limit N]
research library get --key <ITEMKEY>
```
- `ping` — connectivity + backend (local `zotero.sqlite` preferred; falls back to Web API).
- `list` — recent items, optionally filtered by `itemType`. **Best-effort, not a real enumeration**:
  `zotero-cli` has no "list everything" command, so this degrades to an *empty-keyword* search capped
  at 100 items upstream. It says so on stderr every time, and if it returns 0 items it says that too
  — a local library that demonstrably has papers can legitimately yield nothing here, and printing
  "（无条目）" would read as "your library is empty", which is simply false. **When you need to know
  whether a specific paper is in the library, use `search --query`, not `list`.**
- `search` — metadata search (title/author/tag). Full-text search needs the local backend.
- `get` — one item's full JSON by key. **stdout is a data channel**: on success it carries the item
  JSON and *nothing else*. Three outcomes are kept apart, because their next actions are opposite:

  | Situation | Where | Exit |
  |---|---|---|
  | item found | item JSON on **stdout** | 0 |
  | bridge answered `ok:true` but with no data (the key genuinely isn't there) | `（未找到）` on **stdout** | 0 |
  | `zotero-cli` errored — then `ping` arbitrates: bridge up ⇒ “probably not in this library”; bridge down ⇒ “Zotero unreachable, this is *not* a not-found” | reason on **stderr** | 1 |

  `zotero-cli` reports `ok:false` for *both* “can't connect” and “no such key”, and the error text
  alone cannot tell them apart, so `cmd_library` calls `zb.ping()` as the judge — only on the failure
  path, so the happy path pays no extra subprocess. Collapsing the two into one “not found” used to
  disguise “Zotero desktop isn't running” as “your library doesn't have this paper”. Relatedly,
  `zotero_cli.get_item()` **re-raises** instead of printing the error itself: it used to print one
  Chinese sentence to *stdout*, which turned the JSON channel into invalid JSON for every caller.

`list` and `search` failures (Zotero not running, local API not authorized — both ordinary states,
not exceptions) degrade to one stderr line + exit **1**; `search` without `--query` and `get` without
`--key` exit **2** — usage errors, which never touch the bridge at all.

Use `library` to check whether a paper is **already in the user's Zotero** before `add`-ing it
(avoids duplicates), and to find the `zotero_key` that links a note to its library entry.

---
