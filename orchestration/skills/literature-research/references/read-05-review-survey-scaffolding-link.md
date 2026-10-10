# read 分册 5/6

> 含小节：`review` — survey scaffolding, link integrity, count sync
> 原 `read.md` 超 8KB 预算拆分（2026-10-10，backlog 20261010-refs-split-8kb）；分册索引见同目录 `read.md`，按需只读所需分册。

## `review` — survey scaffolding, link integrity, count sync

```
research review new <topic> [--purpose '<goal>'] [--time-window '<window>']
                    [--from-shortlist <f1> <f2> …]
research review status [<file>]
research review sync   [<file>] [--dry-run]
```

**There is deliberately no survey generator.** §1–§5 of `templates/review_note.md` (field overview,
key threads, timeline narrative, contested claims, interface with your own work) are the LLM's core
judgment; a template-filler would turn live judgment into dead boilerplate. `review` does only the
three *mechanical* jobs around it.

### `new`

Instantiates `templates/review_note.md` into `reviews/{YYYY-MM}_{topic_slug}_survey.md`, filling the
frontmatter's `topic` / `topic_slug` / `created_date` / `last_updated` / `author` / `status: draft` /
`time_window`, and substituting `--purpose` into the body's 综述目标 line — **only** that line, and if
the template has been edited so the line cannot be found it says so instead of silently dropping your
purpose. An existing file is never overwritten (exit **2**).

`--from-shortlist` aggregates `sources_used` and `query_strings` by **reading the snapshots'
frontmatter** instead of asking you to retype the search strings: a retyped query always drifts, while
the snapshot holds the one that was actually submitted. It accepts full paths or a filename fragment
matching uniquely under `shortlists/`; a snapshot that cannot be found or parsed is reported and
skipped, and the scaffold is still produced.

`inclusion_criteria` and `exclusion_criteria` are left **empty on purpose** — they are a
methodological judgment, not metadata. `new` then stops and prints the next step: write §1–§5, list
the notes you cover as `[[file-name]]` in `papers_reviewed`, and run `review sync`.

### `status`

Read-only audit of one file, or of every `reviews/*.md` when no target is given:

- every `[[wiki-link]]` in `papers_reviewed` must resolve to an existing `papers/*.md`. Aliases
  (`[[x|Zhang 2026]]`) and anchors (`[[x#§2]]`) are handled, and a bare stem with no brackets is
  accepted because hand-written surveys use it;
- `papers_total_count` / `papers_read_count` are compared against reality — the number of resolvable
  links, and of those how many have `status != "unread"` — with any drift printed as
  `papers_read_count 3 → 应为 7`.

**Exit 1** on a broken link, and also on a file that *could not be audited* (unparseable frontmatter):
a survey that was never actually checked passing a gate silently is more dangerous than an error.
Exit **2** if the named file doesn't exist. An empty `reviews/` is a normal state — one line, exit 0.
This is the same CI-friendly contract as `index --check` and `citecheck`.

### `sync`

Recomputes the canonical `papers_reviewed` list, both counts and `last_updated`, then rewrites **only
the frontmatter** — the body stays byte-identical. It appends **no** Changelog line: that log records
a survey's intellectual progress (drafting, revising, adding evidence), and a mechanical count refresh
that can happen many times a day would drown the entries worth reading.

`unchanged` leaves the file untouched (mtime preserved), exactly like the note merge contract.
`--dry-run` reports which keys *would* be rewritten and writes nothing. Broken links are reported on
stderr but **kept** — `sync` fixes counts, it does not guess at your intent. A file that cannot be
audited is skipped with exit **1** rather than rewritten.

---
