# digest 分册 2/2

> 含小节：6. Status ledger & batch cadence (the key to resumability)；7. Realistic assessment of sub-agent parallelism (don't fantasize)；8. Finish (after all interpretations are filled and `-
> 原 `digest.md` 超 8KB 预算拆分（2026-10-10，backlog 20261010-refs-split-8kb）；分册索引见同目录 `digest.md`，按需只读所需分册。

## 6. Status ledger & batch cadence (the key to resumability)

- It is **recommended** to maintain a status-ledger md in the workspace (e.g. `STATUS.md`; name it
  yourself, just don't collide with tool outputs), recording:
  - **Progress table**: per part "interpreted / total figure slides" + a global *next up* (the next
    target Slide);
  - **Glossary / recurring motifs**: a one-line authoritative interpretation + first-occurrence Slide,
    to keep wording consistent across pages and turns;
  - **Decisions & notes**: anchor methodology, pitfalls and fixes, batch cadence — for fast recovery
    after a session switch.
  This makes resumption **lossless** after context compression or a new session — more robust than
  relying only on `progress.json`'s `done_slides`.
- **Batching**: during the validation phase, 3–5 figure slides/turn (confirm interpretation quality,
  anchors, and term consistency all look right); once stable, run as many as possible per turn (limited
  by context / vision tokens, empirically ~**8–12 figure slides/turn**). **Pure-text slides** (no
  images) need no vision and can be batched many at a time at near-zero cost — they only carry
  `要点/正文` prose and have **no `_(待填)_` slot**, so usually keep the faithful extraction as-is; do
  not rewrite the author's original prose.
- **Not self-triggering**: each turn still needs one user message to start; "automation" = do more
  within a turn + commit the ledger, so resuming is frictionless.
- Always commit the status ledger before ending a turn, guaranteeing lossless resumption after a
  session switch / compression.

## 7. Realistic assessment of sub-agent parallelism (don't fantasize)

- Generic sub-agents (e.g. Browser / CodeReview) **cannot read a local render PNG and write a
  structured md interpretation**, so the core "look-at-figure → interpret" step **cannot** be
  outsourced to parallel sub-agents.
- The only real "parallelism" is **batching within one turn**: Read several renders at once + one
  SearchReplace with several replacements (see §3).

## 8. Finish (after all interpretations are filled and `--lint` broken=0)

- Append a **narrative review** md (e.g. `NARRATIVE.md`): condense the whole deck into a through-line
  long-form (problem → method/theory → platform/implementation → key results → conclusion/outlook),
  linking back to each part and key figure.
- Back-fill index.md's "图像语义清单" (image semantic index): tag high-frequency/key images with a
  semantic label + occurrence pages, for future retrieval and reuse.
- Run `--lint` once more to confirm broken=0, and verify real remaining placeholders == 0.
