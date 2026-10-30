# read 分册 3/6

> 含小节：How notes are written: the merge contract；`add` — put a paper in the library
> 原 `read.md` 超 8KB 预算拆分（2026-10-10，backlog 20261010-refs-split-8kb）；分册索引见同目录 `read.md`，按需只读所需分册。

## How notes are written: the merge contract

`read` and `add` both write through the same `_merge_note()`, which reports one of five actions. The
distinction matters: the previous writer was all-or-nothing, so on a note that already existed it
**returned without writing anything** — which is how `local_pdf_path` and `extracted_md_path` went
missing from every note built by the normal `add`-then-`read` workflow, and with them the
`cache prune --keep-referenced` protection that reads those fields (three now — WP-H added
`extracted_html_path`; `maintenance.md` §5a documents the second, note-independent source).

| Action | When | What happens to the file |
|---|---|---|
| `created` | no note yet | written from `templates/paper_note.md` |
| `merged` | note exists and the new data fills at least one **empty** key | frontmatter re-rendered; **body preserved byte for byte**; one audit line appended under `## Changelog` |
| `unchanged` | note exists, nothing empty left to fill | **not touched at all** — mtime preserved, so `cache prune`'s LRU ordering isn't disturbed |
| `overwritten` | `--overwrite` | rebuilt from the template; the body is reset |
| `blocked` | the derived filename is new **and several** existing notes carry the same `doi` / `openalex_id` / `arxiv_id` | **nothing is written anywhere** — the guard refuses to guess which one to merge into; resolve the duplicates by hand |

The rules that make this safe to run repeatedly:

- **Only empty keys are filled.** Missing, `null`, `""` and `[]` all count as empty. Your `my_rating`,
  `status`, `related_to_my_work` and every prose section are never replaced by a machine value, so
  re-running cannot downgrade a note you have already evaluated.
- **The body is byte-identical** after a `merged` write. The only body edit is the appended Changelog
  line (`- 2026-09-30: 补齐字段 local_pdf_path, extracted_md_path`), which makes the change auditable
  rather than silent.
- **Unparseable frontmatter is never rewritten.** A hand-written or corrupted note is reported and
  left alone (`unchanged`); pass `--overwrite` explicitly if you truly want the template back.
- **A drifted filename no longer spawns a duplicate.** The target path is derived from the *freshly
  fetched* `short_title`, which drifts whenever upstream retitles a work. When that path does not
  exist, the write is redirected to whichever existing note carries the same `doi` / `openalex_id` /
  `arxiv_id`, and both filenames are printed so you can `git mv` if you want the canonical name.
  `--overwrite` is **downgraded to a merge** on that path: the target is a file you never named, and
  rebuilding it from the template would silently erase hand-written prose. A note with none of the
  three identifiers cannot be recognized and still gets created (`maintenance.md` §5f).
- The report separates `merged` (naming the keys it filled) from `unchanged` ("nothing to fill, file
  untouched"), so neither can be misread as a failed write. `blocked` gets its own wording for the
  same reason — it also writes nothing, but because the guard refused to decide.

> **One thing merging cannot do: correct a wrong value.** It fills blanks; it never overwrites. Notes
> written before a field's semantics were fixed therefore keep the old value — re-running `read` will
> not repair it, and `index --fix` normalizes *shape* (field order, block style, missing keys) while
> likewise preserving existing values. Fix such a value by hand — but **recompute** it through the
> project's own funnel instead of typing bibliographic data; `maintenance.md` §5e records the
> whitelist recipe used to repair the three pre-WP-D notes (37 fields, body untouched).

---

## `add` — put a paper in the library

```
research add <doi｜arxiv-id｜openalex-id> [--tags t1,t2] [--overwrite]
             [--verify｜--no-verify] [--allow-fail]
```
Resolves + enriches metadata, **creates a Zotero item** when `zotero-cli` (community zotero-mcp) is
installed — it normalizes the frontmatter to BibTeX and delegates the write to `zotero-cli add
bibtex` — then writes the returned `zotero_key` + `zotero_uri` into a new `papers/` note skeleton,
and does **not** fetch full text. If `zotero-cli` is unavailable, `add` degrades to a skeleton-only
note (no library write). Use `add` to build the library quickly; use `read` when you intend to
read the paper. Check `library search` first to avoid duplicating an existing Zotero entry. The note
is written under the same [merge contract](#how-notes-are-written-the-merge-contract), so `add`-ing a
paper you later `read` fills in the paths instead of discarding them.

Before writing, `add` runs the **citation integrity gate** (`citecheck` below): a three-source
**FAIL** blocks the Zotero write + note. `--no-verify` skips the check entirely; `--allow-fail`
writes even on FAIL.

> `--allow-fail` was formerly spelled `--force`, which still works as a deprecated alias. The rename
> *is* the point: this flag opens a **safety gate**, and sharing a name with an ordinary cache-refresh
> switch made it far too easy to reach for without realizing what it did.

---
