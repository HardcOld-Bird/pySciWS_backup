# Sourcing non-academic web pages (取证规范)

How to source facts from **web pages that are not academic sources** — vendor launch pages,
product docs, changelogs (e.g. an OpenAI / Anthropic post you quote in a report). Academic
literature and paywalled journal full text have their own paths: [sources.md](sources.md) and
[maintenance.md](maintenance.md) §4.

Every claim that reaches a report carries **one** of two provenance labels — this file exists so
nobody has to re-invent the distinction each time:

- **[直取]** — fetched directly from the primary/official page (its HTML / PDF / text), live.
- **[二手互证]** — the primary page was unreachable, so the fact is cross-checked against **≥2
  independent authoritative secondary sources that agree**. Name them, and lower the confidence.

Never present a `[二手互证]` fact as if it were `[直取]`.

## Decision path when a page won't fetch

1. **Try direct first** (WebFetch / `curl` / plain GET). Content returned → `[直取]`.
2. On **403 / bot-block** or timeout, retry **once** through the local proxy
   `http://127.0.0.1:7890` — *single-command injection only*. The rules live in
   `.qoder/rules/basic.md` §4 (default direct; proxy never persisted; **never for journal
   databases / licensed full text**). `curl` exit 7 = the proxy isn't running → do **not**
   loop-retry; go to step 3.
3. Still 403, or proxy down → switch to the **alternative-evidence regime** below and label the
   result `[二手互证]`.

## Two failure classes seen in practice

- **Anti-scraping 403.** `openai.com` / `developers.openai.com` return 403 to WebFetch, `curl`
  and Playwright alike (bot filtering), proxy or not. The page is not gone — only the automated
  path is closed.
- **Region gating.** `docs.anthropic.com` redirects to `claude.com/app-unavailable-in-region`.
  A geo restriction, not a missing resource — the content usually exists on a surface that is
  reachable (below).

## Alternative-evidence regime (acceptable `[二手互证]` sources)

Roughly by authority; each must be **dated and named** in the report:

1. **Official content on a reachable official surface.** The same vendor's GitHub org — release
   notes, tagged changelog, README, committed docs (`github.com/<org>/<repo>`) — stays *primary*
   even when the marketing page 403s. If you fetched it directly, label it by surface, e.g.
   `[直取 github release vX]`.
2. **Dated web-archive snapshot** of the blocked page (e.g. `web.archive.org`). Label
   `[二手互证 存档快照 <date>]` — a snapshot is frozen evidence, not the live page.
3. **≥2 independent authoritative secondary sources that agree** (established tech press,
   standards bodies, the vendor's own community/forum). Two independent outlets is the minimum
   bar; a single blog is not evidence.
4. **Not acceptable:** SEO content farms, scraped mirrors of unknown provenance, LLM-generated
   summaries, or a lone social post.

State the source set inline, e.g.
`[二手互证: Anthropic docs 存档 2026-09 + TechCrunch 2026-09-15]`.

## Recording the 口径 in a report

- One line per contested fact, carrying the label and its sources; a short 取证 appendix listing
  every `[二手互证]` item is enough.
- If a fact can't even reach the `[二手互证]` bar, say so and mark it **未取证** — do not assert it.
- Proxy state matters only at fetch time; don't raise it as a caveat unless it changed the outcome.

## Not this file

Paywalled academic full text, institutional access, Cloudflare on publisher sites: `read` /
`browser_fetch` — [maintenance.md](maintenance.md) §4. Academic source limits and where
journal-quality numbers come from: [sources.md](sources.md).
