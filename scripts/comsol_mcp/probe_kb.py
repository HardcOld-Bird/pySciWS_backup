#!/usr/bin/env python
"""Report the TRUE state of the community COMSOL MCP's RAG knowledge base.

Why this exists
---------------
Upstream's own status path is broken, and it is still broken at pin 0f6b2c58:
``scripts/build_knowledge_base.py --status`` constructs a ``VectorRetriever`` and calls
``get_stats()`` WITHOUT calling ``initialize()`` first, and ``get_stats()`` returns
``{"initialized": False, "count": 0}`` whenever ``self._collection is None``. Since
``__init__`` sets ``_collection = None``, ``--status`` prints ``Documents: 0`` even for a
fully populated index. Anything that trusts it -- including this project's
``setup_comsol_mcp.ps1 -StatusOnly`` used to -- reports a lie.

The MCP's ``pdf_search`` tool is NOT affected: ``VectorRetriever.search()`` lazily calls
``initialize()`` itself, so retrieval works at runtime. Only the status reporting is wrong.

A second, scale-dependent upstream bug (found 2026-10-01 after the index grew from 3669 to
35923 chunks): ``get_stats()`` collects its module list with an *unbounded*
``collection.get(include=["metadatas"])``, and Chroma binds one SQL variable per row, so past
SQLite's 32766-variable ceiling it raises ``InternalError: too many SQL variables``. Upstream
swallows it with a bare ``except Exception: pass``, so ``pdf_search_status`` does not fail -- it
quietly reports ``count: 35923`` alongside ``modules: []`` / ``module_count: 0``. Read that zero
as "upstream could not enumerate", NOT as "nothing is indexed". ``pdf_search`` (bounded by
``n_results``) and ``pdf_list_modules`` (filesystem-based, never touches Chroma) are both fine.
This probe pages instead, so it reports the true module list at any index size.

Usage
-----
Run it with the COMMUNITY venv's python (this project's venv has no chromadb)::

    <repo_dir>\\.venv\\Scripts\\python.exe scripts\\comsol_mcp\\probe_kb.py
    <repo_dir>\\.venv\\Scripts\\python.exe scripts\\comsol_mcp\\probe_kb.py --search 'background pressure field'

The default path only reads Chroma metadata: it loads NO embedding model, starts NO JVM
and takes NO COMSOL license, and finishes in ~2s. ``--search`` additionally proves
retrieval quality; that does load the SentenceTransformer model (~10s, ~0.5 GB RAM,
needs the HF cache or network) -- still license-free.

Exit codes: 0 OK, 2 db_dir missing, 3 collection unreadable, 4 collection empty,
5 embedding/init failure under --search, 6 --search returned no hits.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

# Defaults match setup_comsol_mcp.ps1's -RepoDir / -PdfDir so the probe can be run bare.
DEFAULT_REPO_DIR = Path(r"D:\XXXIIIGGG\projects\pySci\COMSOL_Multiphysics_MCP")
DEFAULT_PDF_DIR = Path(r"D:\XiGPrograms\comsol\6.4\base\doc\pdf")
COLLECTION = "comsol_docs"
# Chroma binds one SQL variable per row in an unbounded get(), and SQLite caps that at 32766.
# Measured on the real index: page=20000 works, page=40000 raises "too many SQL variables".
# 5000 stays far under the ceiling and still needs only ~8 round trips for 35923 chunks.
PAGE_SIZE = 5000


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report the real document count of the community MCP's RAG index.",
    )
    parser.add_argument(
        "--repo-dir",
        type=Path,
        default=DEFAULT_REPO_DIR,
        help="community MCP clone (default: %(default)s)",
    )
    parser.add_argument(
        "--pdf-dir",
        type=Path,
        default=DEFAULT_PDF_DIR,
        help="source PDFs; only needed for --search (default: %(default)s)",
    )
    parser.add_argument(
        "--db-dir",
        type=Path,
        default=None,
        help="Chroma dir (default: <repo-dir>/knowledge_base, which is what the server reads)",
    )
    parser.add_argument(
        "--search",
        default=None,
        metavar="QUERY",
        help="also run one retrieval to prove the index is queryable (loads the model)",
    )
    parser.add_argument(
        "--results",
        type=int,
        default=3,
        help="hits to show with --search (default: %(default)s)",
    )
    return parser.parse_args(argv)


def read_counts(db_dir: Path) -> tuple[int | None, list[str]]:
    """Read the chunk count and module list straight from Chroma, without an embedding model.

    Returns (None, []) when the collection cannot be read at all.
    """
    # Imported BY NAME: chromadb only exists in the COMMUNITY venv, so a static import here would
    # be a permanent unresolved reference under this project's interpreter.
    chromadb = importlib.import_module("chromadb")

    client = chromadb.PersistentClient(path=str(db_dir))
    # list_collections() yields Collection objects on newer chromadb, plain names on older.
    names = [getattr(c, "name", c) for c in client.list_collections()]
    print(f"collections   = {names}")
    if COLLECTION not in names:
        return None, []

    # No embedding_function on purpose: count()/get() need no model, so this stays fast
    # and offline. Passing one would download/load SentenceTransformer for nothing.
    collection = client.get_collection(COLLECTION)
    count = collection.count()

    # Paged on purpose -- see PAGE_SIZE. An unbounded get() is what makes upstream's
    # get_stats() lose its module list once the index passes SQLite's variable ceiling.
    metas: list = []
    offset = 0
    while True:
        batch = collection.get(
            include=["metadatas"], limit=PAGE_SIZE, offset=offset
        ).get("metadatas") or []
        if not batch:
            break
        metas.extend(batch)
        offset += len(batch)
        if len(batch) < PAGE_SIZE:
            break
    if len(metas) != count:
        print(f"PAGING_NOTE   : read {len(metas)} metadatas but count() says {count}")

    modules = sorted({m["module"] for m in metas if m and m.get("module")})
    return count, modules


def run_search(repo_dir: Path, db_dir: Path, pdf_dir: Path, query: str, n: int) -> int:
    """Prove retrieval works, via the same code path the MCP's pdf_search uses."""
    # Imported BY NAME at runtime, never statically: `src` here is the COMMUNITY repo's src/
    # (main() puts it on sys.path), but this project has a src/ of its own, so any static
    # `from src...` resolves to the wrong tree and shows up as a bogus unresolved reference.
    retriever_cls = importlib.import_module("src.knowledge.retriever").VectorRetriever

    retriever = retriever_cls(str(pdf_dir), str(db_dir))
    if not retriever.initialize():
        print(
            "INIT_FAILED   : initialize() returned False (embedding model unavailable?)"
        )
        return 5

    hits = retriever.search(query, n_results=n)
    print(f"query         = {query!r}")
    print(f"hits          = {len(hits)}")
    for hit in hits:
        src = Path(hit.source).name if hit.source else "?"
        print(
            f"  - score={hit.score:.3f} module={hit.module} page={hit.page} src={src}"
        )
        print(f"    {' '.join(hit.text[:140].split())}")
    return 0 if hits else 6


def main(argv: list[str] | None = None) -> int:
    # Windows consoles default to the ANSI code page (GBK on zh-CN), which mangles the UTF-8 manual
    # text this probe prints into unreadable mojibake. Force UTF-8 -- the same trap the installer
    # documents for reading .ps1/.json under PowerShell 5.1, fixed on the writer side this time.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args(argv)
    db_dir = args.db_dir or (args.repo_dir / "knowledge_base")
    print(f"repo_dir      = {args.repo_dir}")
    print(f"db_dir        = {db_dir}")

    if not db_dir.exists():
        print(
            "KB_MISSING    : db_dir does not exist -- run the installer's RAG step (7/8)."
        )
        return 2

    # The probe imports upstream's package for --search, so its root must be importable.
    sys.path.insert(0, str(args.repo_dir))

    try:
        count, modules = read_counts(db_dir)
    except Exception as exc:  # noqa: BLE001 - a status probe must never raise
        print(f"CHROMA_ERROR  : {type(exc).__name__}: {exc}")
        return 3

    if count is None:
        print(f"KB_UNREADABLE : collection {COLLECTION!r} not found in {db_dir}.")
        return 3

    print(f"documents     = {count}")
    print(f"module_count  = {len(modules)}")
    print(f"modules       = {modules}")

    if count == 0:
        print("KB_EMPTY      : collection exists but holds no chunks -- rebuild it.")
        return 4
    if len(modules) < 10:
        # The installer's -RagLimit truncates the build; a handful of modules means the
        # index only covers the first few PDFs. Not fatal, but pdf_search will miss most
        # of the manuals (COMSOL_Multiphysics alone ships 14 PDFs).
        print(
            "COVERAGE_NOTE : few modules indexed -- the build was likely limited by "
            "-RagLimit. Rebuild without it for full manual coverage."
        )

    if args.search:
        code = run_search(
            args.repo_dir, db_dir, args.pdf_dir, args.search, args.results
        )
        if code != 0:
            return code

    print("KB_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
