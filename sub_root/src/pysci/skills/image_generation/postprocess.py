"""本地轻后处理：rembg 透明底兜底 + Pillow 裁剪/缩放 + alpha 合规验证。

透明底策略（详见 ``.agents/skills/image-generation/references/transparent-background.md``）：
**云直出优先**（方舟 Seedream 5.5 ``background="transparent"`` 直出 RGBA PNG），
**rembg 兜底**（任意图抠图 / 副通道无透明参数 / 玻璃类不透光时）。云直出与 rembg 产物
都必须过 :func:`verify_alpha`（alpha 最高值须 255，防"251–254 差一点不透明"的假透明）。

rembg 为**惰性导入**的可选依赖（26k★，MIT，ONNX U2Net 族，纯 CPU）：未安装时
:func:`remove_background` 抛带安装提示的 :class:`PostprocessError`，其余 Pillow 功能
不受影响。首次运行会下载模型权重（自包含缓存约定见 deps_management）。

用法::

    from pysci.skills.image_generation import postprocess

    postprocess.remove_background("art.png", "art_cutout.png")
    postprocess.verify_alpha("art_cutout.png")   # {"ok": True, "alpha_max": 255, ...}
    postprocess.crop_resize("art.png", "thumb.png", size=(512, 512))
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from pysci.paths import PROJECT_ROOT


class PostprocessError(RuntimeError):
    """后处理失败（rembg 缺失、参数非法、产物不合规等）。"""


#: rembg 安装提示（惰性导入失败时给可执行命令）。
INSTALL_HINT: str = (
    "透明底兜底需要 rembg：并行依赖批次安装（uv add rembg）后重试；"
    "云直出透明底（方舟 5.5 background=transparent）不受影响。"
)

#: rembg 模型缓存根（自包含：权重 vendor 在 deps_management，不落用户主目录）。
REMBG_CACHE_DIR: Path = PROJECT_ROOT / "deps_management" / "deps" / "cache" / "rembg"


def _ensure_model_env() -> None:
    """模型缓存 env 契约：**import rembg 之前**设置（rembg 只在 import/建 session 时读）。

    - ``U2NET_HOME`` → ``deps_management/deps/cache/rembg``（rembg 的
      ``<home>/models/<name>`` 布局正好落在预下载的 ``models/u2net/u2net.onnx``）；
    - ``HF_HOME`` / ``HF_HUB_DISABLE_XET`` → 其余 HF 系模型（如 wtpsplit）同缓存根。

    已设的环境变量不覆盖（用户显式指定优先）。
    """
    os.environ.setdefault("U2NET_HOME", str(REMBG_CACHE_DIR))
    os.environ.setdefault(
        "HF_HOME", str(PROJECT_ROOT / "deps_management" / "deps" / "cache" / "huggingface")
    )
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


def remove_background(
    src: str | Path,
    dest: str | Path,
    *,
    model: str = "u2net",
    alpha_matting: bool = False,
) -> Path:
    """用 rembg 把 ``src`` 抠成透明底 PNG 写到 ``dest``（任意图→RGBA）。

    Args:
        src: 源图路径（生成图或实拍图）。
        dest: 目标路径（建议 .png——alpha 需要 PNG 容器）。
        model: rembg 模型名（u2net / u2netp / isnet-general-use …）。
        alpha_matting: 边缘精修（慢，抠发丝/半透明边缘时开）。

    Returns:
        写出的目标路径。

    Raises:
        PostprocessError: rembg 未安装 / 源图缺失 / 产物不合规。
    """
    _ensure_model_env()  # import 前设 env（rembg 只在 import 时读）
    try:
        from rembg import new_session, remove  # type: ignore
    except ImportError as e:
        raise PostprocessError(f"缺少 rembg：{INSTALL_HINT}") from e

    p = Path(src).expanduser()
    if not p.is_file():
        raise PostprocessError(f"源图不存在：{p}")
    out = Path(dest).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)

    session = new_session(model)
    with Image.open(p) as im:
        result = remove(im, session=session, alpha_matting=bool(alpha_matting))
    if not isinstance(result, Image.Image):
        result = Image.open(_bytes_io(result))
    tmp = out.with_name(out.name + ".tmp")
    try:
        result.convert("RGBA").save(tmp, format="PNG")
        os.replace(tmp, out)  # 原子落盘：失败不留截断目标
    finally:
        with contextlib.suppress(OSError):
            tmp.unlink()
    return out


def _bytes_io(data: bytes) -> Any:
    """把 rembg 返回的字节流包成文件对象（rembg 对 bytes 输入返回 bytes）。"""
    import io

    return io.BytesIO(data)


def verify_alpha(path: str | Path) -> dict[str, Any]:
    """验证透明底产物合规：RGBA、alpha 有透明也有不透明（max==255）。

    云直出/抠图产物的常见坑是 alpha 最高值 251–254（"差一点不透明"）——叠到深色
    底上会露白边。本函数给出量化结论，由调用方决定重抠还是接受。

    Returns:
        ``{"ok": bool, "mode": str, "alpha_max": int, "alpha_min": int,
        "transparent_px": int, "reason": str}``。
    """
    p = Path(path).expanduser()
    if not p.is_file():
        raise PostprocessError(f"图像不存在：{p}")
    with Image.open(p) as im:
        mode = im.mode
        arr = np.asarray(im.convert("RGBA"))
    alpha = arr[:, :, 3]
    alpha_max = int(alpha.max())
    alpha_min = int(alpha.min())
    transparent_px = int((alpha < 255).sum())
    ok = mode in ("RGBA", "LA") and alpha_max == 255 and transparent_px > 0
    if mode not in ("RGBA", "LA"):
        reason = f"模式 {mode} 无 alpha 通道（非透明底产物）"
    elif alpha_max != 255:
        reason = f"alpha 最高值 {alpha_max} < 255（假透明，叠深色底会露白边）"
    elif transparent_px == 0:
        reason = "alpha 全不透明（抠图失败或本就不透明）"
    else:
        reason = "ok"
    return {
        "ok": ok,
        "mode": mode,
        "alpha_max": alpha_max,
        "alpha_min": alpha_min,
        "transparent_px": transparent_px,
        "reason": reason,
    }


def crop_resize(
    src: str | Path,
    dest: str | Path,
    *,
    crop: tuple[int, int, int, int] | None = None,
    size: tuple[int, int] | None = None,
    overwrite: bool = False,
) -> Path:
    """Pillow 裁剪/缩放（固定顺序：crop → resize）并原子落盘。

    Args:
        src: 源图路径（先完整解码，坏图在动 dest 之前失败）。
        dest: 目标路径（后缀决定格式）。
        crop: 像素裁剪框 ``(x0, y0, x1, y1)``。
        size: 目标尺寸 ``(w, h)``（LANCZOS）；None 保持原尺寸。
        overwrite: 允许覆盖已存在的 dest（默认拒绝，防误毁已计费产物）。

    Returns:
        写出的目标路径。

    Raises:
        PostprocessError: 目标已存在而未允许覆盖 / crop 越界。
    """
    src_p = Path(src).expanduser()
    if not src_p.is_file():
        raise PostprocessError(f"源图不存在：{src_p}")
    out = Path(dest).expanduser()
    if out.exists() and not overwrite:
        raise PostprocessError(
            f"目标已存在：{out}——默认拒绝覆盖（已计费产物防误毁）；确认后传 overwrite=True"
        )
    with Image.open(src_p) as im:
        im.load()  # 源坏/不完整在动 dest 之前失败
        img = im.copy()
    if crop is not None:
        x0, y0, x1, y1 = (int(v) for v in crop)
        if not (0 <= x0 < x1 <= img.width and 0 <= y0 < y1 <= img.height):
            raise PostprocessError(
                f"crop {crop} 越界：图像尺寸 {(img.width, img.height)}"
            )
        img = img.crop((x0, y0, x1, y1))
    if size is not None:
        img = img.resize((int(size[0]), int(size[1])), Image.LANCZOS)
    out.parent.mkdir(parents=True, exist_ok=True)
    fmt = _format_for(out)
    if fmt == "JPEG" and img.mode in ("RGBA", "P", "LA"):
        img = img.convert("RGB")  # JPEG 无 alpha
    tmp = out.with_name(out.name + ".tmp")
    try:
        img.save(tmp, format=fmt)  # 显式格式：.tmp 后缀推断不出容器
        os.replace(tmp, out)  # 原子落盘
    finally:
        with contextlib.suppress(OSError):
            tmp.unlink()
    return out


#: 目标后缀 → Pillow 保存格式。
_FORMAT_BY_SUFFIX = {
    ".png": "PNG",
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
    ".webp": "WEBP",
    ".bmp": "BMP",
    ".tif": "TIFF",
    ".tiff": "TIFF",
}


def _format_for(dest: Path) -> str:
    """由目标后缀推断 Pillow 保存格式（未知后缀报错，不猜）。"""
    fmt = _FORMAT_BY_SUFFIX.get(dest.suffix.lower())
    if fmt is None:
        raise PostprocessError(
            f"无法从后缀 {dest.suffix!r} 推断输出格式；支持 {sorted(_FORMAT_BY_SUFFIX)}"
        )
    return fmt
