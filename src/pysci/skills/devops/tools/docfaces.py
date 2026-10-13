"""源文件 → **文档面**映射（backlog 20261013-184842-devops）。

**为什么需要它**：`producer_family` / `jury_panel` 这类**相对化**改造只扫了代码——实测
`jury-channels.md` / `jury-panel.md` 仍把拒庭判据写成「任何席位（**含回退席**）含 `qwen`」，
而同文件族的 `jury.py` 注释早就预警「同族回退席不得上场……会把『该降级』的案子误判成
『拒开庭』」。**同一份设施里注释说不能这样、文档说就这样**；实测 `approve 131505 --jury`
证明注释对、代码与文档错——一次语义变更改了代码三处、文档两处没跟。

README §6 已有 `intent-drift` 巡检项，但它是**抽样式**的（随机挑一段宣称核验），不是
**改动驱动**的：谁改了 jury 语义，没人提醒他去看 jury 的文档面。本模块就是那声提醒。

**设计取向：表小、手维护、只提示**。不做自动推断（按 import/关键词猜文档面必然误报，
久了就没人看）；命中即打印、不阻断；没命中就静默。表里的路径有 :func:`misconfigured`
兜底——写错的映射等于把人对盘到错文件，比没有映射更坏，故有单测钉住「表里每个路径都
真实存在」。
"""

from __future__ import annotations

from pathlib import Path

from pysci.paths import PROJECT_ROOT

#: 技能真本根（文档面都在这底下）。
_SKILLS = "orchestration/skills"
_ORCH_REF = f"{_SKILLS}/orchestration/references"
_INSP_REF = f"{_SKILLS}/inspector/references"
_DEVOPS_REF = f"{_SKILLS}/devops/references"

#: ``源文件路径**后缀** → 文档面``。键用后缀（而非完整路径）是为了在 worktree 里也命中：
#: devops 的改动总发生在 ``<repo>/.qoder/worktrees/<id>/src/...``，绝对/相对路径都对不齐，
#: 而尾部 ``orchestration/tools/jury.py`` 两边一致。后缀取自**包内相对路径**，别取太短
#: （``jury.py`` 会误命中任何同名文件）。
DOC_FACES: dict[str, tuple[str, ...]] = {
    "orchestration/tools/jury.py": (
        f"{_ORCH_REF}/jury-channels.md",
        f"{_ORCH_REF}/jury-panel.md",
    ),
    "orchestration/tools/dispatch.py": (f"{_ORCH_REF}/drain-watch.md",),
    "orchestration/tools/drain.py": (
        f"{_ORCH_REF}/drain-watch.md",
        f"{_ORCH_REF}/salvage-reclaim.md",
        f"{_ORCH_REF}/tier-a-tags.md",
    ),
    "orchestration/tools/workflow.py": (
        f"{_ORCH_REF}/drain-watch.md",
        f"{_ORCH_REF}/tier-a-tags.md",
    ),
    "orchestration/tools/inspectd.py": (
        f"{_SKILLS}/inspector/SKILL.md",
        f"{_INSP_REF}/patrol-craft.md",
    ),
    "orchestration/tools/routing.py": (f"{_ORCH_REF}/drain-watch.md",),
    "devops/tools/dev.py": (
        f"{_DEVOPS_REF}/worktree-merge.md",
        f"{_DEVOPS_REF}/uv-venv-sync.md",
        f"{_DEVOPS_REF}/registry-runtime.md",
    ),
}


def faces_for(path: str | Path) -> tuple[str, ...]:
    """该源文件对应的文档面（无映射 → 空元组）。

    按**后缀**匹配（见 :data:`DOC_FACES` 注释）；分隔符先归一到 ``/``，免得 Windows 的
    反斜杠把匹配整得看运气。
    """
    norm = str(path).replace("\\", "/")
    for suffix, faces in DOC_FACES.items():
        if norm.endswith(suffix):
            return faces
    return ()


def misconfigured(*, root: Path | None = None) -> list[str]:
    """表里**不存在**的文档面路径（写错的映射要把人对盘到错文件，必须能被发现）。

    Returns:
        缺失路径列表（按字典序；空表＝表是干净的）。``root`` 缺省取仓库根。
    """
    base = PROJECT_ROOT if root is None else root
    missing = {f for faces in DOC_FACES.values() for f in faces}
    return sorted(f for f in missing if not (base / f).is_file())


def report(paths: list[str] | list[Path]) -> list[str]:
    """把待对盘的行拼出来（**命中才有行**，未映射文件零输出）。

    一行形如 ``[doc-faces] <你动的文件> → 请对盘：<文档面1>、<文档面2>``。
    """
    lines: list[str] = []
    for raw in paths:
        faces = faces_for(raw)
        if not faces:
            continue
        lines.append(f"[doc-faces] {raw} → 请对盘：{'、'.join(faces)}")
    return lines
