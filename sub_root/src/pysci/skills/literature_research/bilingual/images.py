"""图片回补：md 引用名与落盘文件名闭合校验（prepare 的机械闸之一）。

回补源（新架构衔接，2026-11-17）：①域 read 提取（``mineru_cloud.extract_markdown``
产物）与源 md 同目录的 ``images/`` 按名复制——MinerU 结果 zip 自带 images/，提取落盘
时一并留存即命中；名字对不上时 fallback **PyMuPDF 从源 PDF 按页序抽图顺序配对**
（旧实现同款先例，配对结果告警）。引用名不闭合抛 :class:`ImageBackfillError`，
submit 据 ``manifest.images_closed`` 拒绝出稿。

源 PDF 反解（旧三级 cache 反解已随旧耦合移除）：显式 ``--pdf`` → 与源 md 同目录同名
``<stem>.pdf`` / ``<stem 去 _fulltext>.pdf``（①域 ``papers/<key>`` 同名约定）。
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path


class ImageBackfillError(RuntimeError):
    """图片回补失败（缺图且无法配对）——prepare 默认据此报错。"""


def resolve_source_pdf(source_md: Path, explicit: str | Path | None = None) -> Path | None:
    """解析源 PDF：显式指定 → 同目录同名（含去 ``_fulltext`` 后缀）→ None。

    Args:
        source_md: 源 markdown 路径。
        explicit: ``--pdf`` 手工指定。

    Returns:
        源 PDF 路径；都命中不到返回 ``None``（调用方按需报错）。

    Raises:
        FileNotFoundError: 显式指定的文件不存在。
    """
    if explicit is not None:
        p = Path(explicit).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"--pdf 指定的文件不存在：{p}")
        return p.resolve()
    stem = source_md.stem
    bare = re.sub(r"_fulltext$", "", stem)
    for name in (f"{stem}.pdf", f"{bare}.pdf"):
        cand = source_md.parent / name
        if cand.is_file():
            return cand.resolve()
    return None


def pymupdf_page_images(pdf_path: Path) -> list[bytes]:
    """PyMuPDF 按页序抽嵌入图（顺序配对兜底的数据源）。"""
    import fitz  # type: ignore

    out: list[bytes] = []
    with fitz.open(str(pdf_path)) as doc:
        for page in doc:
            for info in page.get_images(full=True):
                xref = info[0]
                base = doc.extract_image(xref)
                out.append(base["image"])
    return out


def backfill_images(
    workdir: Path,
    refs: list[str],
    source_pdf: Path | None,
    *,
    no_images: bool,
    source_md: Path | None = None,
    provider: Callable[[Path], dict[str, bytes]] | None = None,
) -> None:
    """把 md 引用的图回补到 ``workdir/images/``，核对引用名与落盘文件名闭合。

    顺序：同目录 ``images/`` 按名复制（①域提取产物）→ ``provider``（可插拔按名抽取，
    如 MinerU 云端重抽）→ PyMuPDF 按页序顺序配对（告警）。缺图抛
    :class:`ImageBackfillError`；``no_images=True`` 直接跳过（submit 会拒绝出稿）。

    Args:
        workdir: 工作区目录（``bilingual/<stem>/``）。
        refs: md 内图片引用名清单（如 ``images/xxx.jpg``）。
        source_pdf: 源 PDF 路径；``None`` 且需要兜底时报错。
        no_images: 跳过回补（测试/调试用）。
        source_md: 源 markdown 路径（找同目录 ``images/`` 用）。
        provider: 按名抽取器 ``PDF → {图名: 字节}``（可选）。

    Raises:
        ImageBackfillError: 缺图且无法配对，或缺源 PDF。
    """
    if no_images or not refs:
        return
    img_dir = workdir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    want = {r.rsplit("/", 1)[-1] for r in refs}
    saved: dict[str, bytes] = {}
    if source_md is not None:
        sibling = source_md.parent / "images"
        if sibling.is_dir():
            for name in want:
                f = sibling / name
                if f.is_file():
                    saved[name] = f.read_bytes()
    if provider is not None and source_pdf is not None:
        for name, data in provider(source_pdf).items():
            saved.setdefault(name, data)
    missing = sorted(want - set(saved))
    if missing:
        if source_pdf is None:
            raise ImageBackfillError(
                "缺源 PDF，无法回补图片（先走①域 read 提取，或用 --pdf 指定）："
                f"{missing}"
            )
        # fallback：按页序顺序配对（旧 extract_figs.py 同款先例），配对结果告警。
        page_imgs = pymupdf_page_images(source_pdf)
        if len(page_imgs) < len(missing):
            raise ImageBackfillError(
                f"缺图 {missing}：PyMuPDF 抽图数（{len(page_imgs)}）不足配对"
            )
        for k, name in enumerate(missing):
            saved[name] = page_imgs[k]
        print(
            f"[bilingual] 告警：图名与提取件不闭合，已用 PyMuPDF 按页序顺序配对 {missing}",
            flush=True,
        )
    for name, data in saved.items():
        (img_dir / name).write_bytes(data)
    still = sorted(want - {p.name for p in img_dir.iterdir() if p.is_file()})
    if still:
        raise ImageBackfillError(f"缺图 {still}：回补后引用名仍不闭合")


def images_closed(workdir: Path, refs: list[str]) -> bool:
    """引用名与 ``workdir/images/`` 落盘文件名是否闭合（manifest.images_closed 用）。"""
    img_dir = workdir / "images"
    return all((img_dir / r.rsplit("/", 1)[-1]).is_file() for r in refs)
