# search 分册 3/5

> 含小节：`citegraph` — snowball the citation graph
> 原 `search.md` 超 8KB 预算拆分（2026-10-10，backlog 20261010-refs-split-8kb）；分册索引见同目录 `search.md`，按需只读所需分册。

## `citegraph` — snowball the citation graph

```
research citegraph <doi｜arxiv-id｜openalex-id>
                   [--direction backward｜forward｜both] [--limit N] [--year Y]
                   [--sort relevance｜date｜citations] [--save] [--purpose '<goal>'] [--json]
```

Rolls the citation graph around one seed paper through **OpenAlex**: `backward` = what it cites (its
reference list), `forward` = what cites it. This is the snowballing step — run it on a seed you trust,
then feed the interesting hits back into `get` / `read` / `add`.

| Flag | Default | Effect |
|---|---|---|
| `--direction` | `both` | `backward` needs only the reference list; `forward` needs a real OpenAlex id. |
| `--limit` | `25` | **Per direction.** `0` = unlimited. |
| `--year` | — | **forward only** — `2020-2026` / `2020-` / `2020`. |
| `--sort` | `citations` | `relevance`｜`date`｜`citations`. Defaults to citation-descending so the first screenful is the most influential, not the most recent. |
| `--save` | — | Write a snapshot to `shortlists/` (same template as `search --save`), so a snowball round is as reproducible as a query. |
| `--purpose` | — | Recorded in the snapshot. |
| `--json` | — | Emit both directions **separately** (shape below). |

Behavior worth knowing before you rely on it:

- **`forward` requires an OpenAlex id.** An arXiv id or DOI is first resolved through OpenAlex; if
  that fails, `--direction forward` exits **2** with the reason, while `both` degrades to
  backward-only and keeps going. Citation graphs exist only in OpenAlex — there is no fallback source.
- **Sorting happens where it is cheapest.** `forward` is sorted *and* truncated **server-side**
  (OpenAlex `sort=`), so `--limit 25` fetches 25 records, not 200. `backward` comes back from a bulk
  id lookup in no guaranteed order, so it is fetched in full, sorted **client-side**, then truncated
  — otherwise `--sort citations --limit 20` would silently mean "the best 20 of an arbitrary first 50".
- **`both` de-duplicates across directions.** A paper cannot both cite and be cited by the seed, but
  OpenAlex's citation data carries a few dirty bidirectional records. Overlaps are dropped using the
  same key as `search` (DOI → arXiv id → title slug); `backward` is inserted first, so it wins.
- **Progress lines go to stderr** (`[citegraph] backward：参考文献 42 条，OpenAlex 取回 39 条…`), keeping
  stdout clean for `--json`. A shortfall is normal — those are merged or unindexed records — and is
  reported, not treated as an error.
- **Exit codes:** `2` when the seed cannot be resolved to an OpenAlex record at all, or `forward` was
  requested without an id; `0` otherwise, *including* "both directions empty". A network failure in
  the graph layer degrades to an empty list plus one line, never a traceback.

`--json` shape — the two directions are **not** merged into one array, because whether a hit *cites
the seed* or *is cited by it* changes what you do with it next:

```json
{
  "id": "<as given>", "openalex_id": "W…", "title": "…",
  "year": 2018, "cited_by_count": 230,
  "backward": {"referenced_works_count": 42, "resolved_count": 39, "rows": []},
  "forward":  {"total_citing": 187, "returned_count": 25, "rows": []}
}
```

Each `rows` entry is the same 13-key object `search --json` emits (all tagged `source: "openalex"`).
`total_citing` is the server's own count, so `returned_count < total_citing` tells you to raise
`--limit` rather than concluding the forward set is small.

---
