# maintenance 分册 11/11

> 含小节：7. Extending the system；8. Verifying changes
> 原 `maintenance.md` 超 8KB 预算拆分（2026-10-10，backlog 20261010-refs-split-8kb）；分册索引见同目录 `maintenance.md`，按需只读所需分册。

## 7. Extending the system

- **New data source**: add `tools/<source>_client.py` exposing `search_*()` and `get_*()` that
  return records convertible to the unified work-dict contract, plus a
  `<source>_to_note_frontmatter()`. Wire it into `cmd_search` (and optionally an enricher like
  `enrich_openalex_work`). Keep failures non-fatal.
- **New output type**: add a template under `templates/` and a `cmd_*` in `research.py` that fills it
  via `notes.render_note` / `notes.dump_frontmatter` + `_fill_placeholders`. The hand-rolled
  `_dump_yaml` / `_yaml_scalar` / `_slugify` / `_note_filename` / `_fmt_bytes` helpers are **gone** —
  all YAML and filename logic now lives in the `notes` leaf, so a new writer must go through it too
  (otherwise the byte-identical frontmatter rendering that `index --fix` and the merge path rely on
  stops being idempotent).
- **New PDF backend**: extend `pdf_extract.available_backends()` + `_pick_backend()` + `extract_pdf()`.

## 8. Verifying changes

```
.venv\Scripts\python.exe -m py_compile src/pysci/skills/literature_research/tools/<file>.py
.venv\Scripts\python.exe -m pysci.skills.literature_research.tools.research doctor
```
Then smoke-test the affected command (`search`/`read`/`get`/`add`/`citecheck`/`citegraph`/`library`/
`journal`/`review`/`index`/`rag`/`cache`/`ingest`). `doctor`
is the fastest way to confirm config, sources, backends, Playwright, Zotero, and the RAG layer are all wired
up. For `rag` specifically: `research rag status` (offline) then a small `research rag index --path <one .md>`
+ `research rag search '<q>' -k 3` is the cheapest end-to-end check (uses real SiliconFlow embedding).
