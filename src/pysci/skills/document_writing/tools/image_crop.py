"""图像裁剪：置入 PPT 前去掉刊头/图注/冗余边条（``compose crop-img``）。

用户批语（tb-5 返工第 2 轮）要求「置入前裁剪」不再靠一次性脚本。本模块只做几何裁切：

- 四边可按**像素**或**比例**给（同一边二选一）；
- 快捷语义：``--strip-top``＝去顶部条带（刊头），``--strip-caption``＝去底部图注；
- 默认输出 ``<stem>_cropped<ext>``，不覆盖源文件；显式 ``--out`` 可指定替换件。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CropMargins:
    """四边裁掉的像素数。"""

    top: int = 0
    bottom: int = 0
    left: int = 0
    right: int = 0

    def box(self, width: int, height: int) -> tuple[int, int, int, int]:
        return (self.left, self.top, width - self.right, height - self.bottom)


@dataclass
class CropResult:
    """一次裁剪的读数。"""

    src: Path
    dest: Path | None
    original: tuple[int, int]
    cropped: tuple[int, int]
    margins: CropMargins
    dry_run: bool = False


def resolve_margins(
    width: int,
    height: int,
    *,
    top: int = 0,
    bottom: int = 0,
    left: int = 0,
    right: int = 0,
    top_ratio: float | None = None,
    bottom_ratio: float | None = None,
    left_ratio: float | None = None,
    right_ratio: float | None = None,
) -> CropMargins:
    """把像素/比例统一成像素边距并校验（比例按对应边长四舍五入）。

    Raises:
        ValueError: 同一边同时给像素与比例、比例越界、负值或裁后无剩余。
    """
    out: dict[str, int] = {}
    sides = (
        ("top", top, top_ratio, height),
        ("bottom", bottom, bottom_ratio, height),
        ("left", left, left_ratio, width),
        ("right", right, right_ratio, width),
    )
    for name, px, ratio, dim in sides:
        if ratio is not None:
            if px:
                raise ValueError(f"{name}: 像素与比例只能给一个")
            if not 0 <= float(ratio) < 1:
                raise ValueError(f"{name}-ratio 须在 [0,1)：{ratio!r}")
            px = int(round(float(ratio) * dim))
        if int(px) < 0:
            raise ValueError(f"{name} 不能为负：{px!r}")
        out[name] = int(px)
    margins = CropMargins(**out)
    if margins.left + margins.right >= width:
        raise ValueError(
            f"左右共裁 {margins.left + margins.right}px ≥ 宽 {width}px——裁后无剩余"
        )
    if margins.top + margins.bottom >= height:
        raise ValueError(
            f"上下共裁 {margins.top + margins.bottom}px ≥ 高 {height}px——裁后无剩余"
        )
    return margins


def crop_image(
    src: str | Path,
    *,
    out: str | Path | None = None,
    quality: int = 95,
    dry_run: bool = False,
    **margins: Any,
) -> CropResult:
    """按四边像素/比例裁图；``dry_run`` 只算尺寸不落盘。

    Args:
        src: 源图片（PNG/JPEG/WebP/TIFF…）。
        out: 输出路径；缺省 ``<stem>_cropped<ext>``（可与源同路径以原地替换）。
        quality: JPEG/WebP 质量（1–100）。
        dry_run: 只返回读数，不写文件。
        **margins: ``top/bottom/left/right`` 或 ``*_ratio``。

    Raises:
        FileNotFoundError: 源不存在。
        ValueError: 边距非法/裁后无剩余。
    """
    from PIL import Image

    path = Path(src)
    if not path.exists():
        raise FileNotFoundError(f"图片不存在：{path}")
    with Image.open(path) as img:
        width, height = img.size
        m = resolve_margins(width, height, **margins)
        box = m.box(width, height)
        cropped_size = (box[2] - box[0], box[3] - box[1])
        if dry_run:
            return CropResult(path, None, (width, height), cropped_size, m, True)
        dest = Path(out) if out else path.with_name(f"{path.stem}_cropped{path.suffix}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        cropped = img.crop(box)
        fmt = (img.format or "").upper() or None
        save_kw: dict[str, Any] = {}
        if fmt in ("JPEG", "JPG"):
            if cropped.mode not in ("RGB", "L"):
                cropped = cropped.convert("RGB")
            save_kw["quality"] = max(1, min(100, int(quality)))
        elif fmt == "WEBP":
            save_kw["quality"] = max(1, min(100, int(quality)))
        cropped.save(dest, format=fmt, **save_kw)
    return CropResult(path, dest, (width, height), cropped_size, m)
