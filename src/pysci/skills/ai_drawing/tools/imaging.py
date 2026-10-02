"""本地位图算子层：AI 素材 → 科研图 的像素级加工。

这一层回答"复杂需求/位图处理能力"——它**不需要 GPU**，也**不需要 ComfyUI 节点图**：
通用像素算子的正确社区基础设施是 scikit-image 与 OpenCV，二者都比 ComfyUI 的图像
节点（为扩散 pipeline 设计的 ``IMAGE`` tensor，``[B,H,W,C]`` float 0–1）更专业、更活跃。

依赖策略：**惰性导入 + optional extra**。未安装时相关函数抛 :class:`ImagingError`
并给出确切的安装命令，而不是在 import 期崩溃::

    uv pip install -e ".[imaging]"      # scikit-image + opencv-python-headless

与 :mod:`postprocess` 的分工：
- ``postprocess`` — 几何/格式基础操作（crop/resize/rotate/pad/mode/调色板/contact sheet），
  纯 Pillow，**核心依赖**，任何环境都可用；
- ``imaging``     — 需要科学图像算法的进阶算子（泊松融合、算法 inpaint、抠图、形态学、
  连通域测量、配准、多图层 alpha 合成），**可选依赖**。

统一约定：模块内部一律 **RGB** 顺序（PIL / scikit-image 语义）。OpenCV 用 BGR，
故所有进出 cv2 的数组都在本模块内转换，调用方无需关心。

典型流水线（云端拆分 → 本地合成）::

    imagine edit --prompt "..." --layers out/          # 方舟图层拆分，得 ≤16 张带 alpha 的 PNG
    imagine img fuse --src out/layer_03.png --base 实拍底图.jpg --out fused.png
    imagine img measure --src fused.png                # 连通域面积/质心，配 scale bar
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

#: 安装提示（惰性导入失败时原样打印，省掉一轮查文档）。
INSTALL_HINT: str = (
    '本功能需要可选依赖：uv pip install -e ".[imaging]"'
    "（scikit-image + opencv-python-headless）"
)


class ImagingError(RuntimeError):
    """位图算子调用失败（依赖缺失、参数非法、mask 尺寸不匹配等）。"""


# ---------------------------------------------------------------------------
# 依赖探测（惰性导入）
# ---------------------------------------------------------------------------
def available() -> dict[str, str | None]:
    """探测可选依赖版本；``None`` 表示未安装。供 ``imagine doctor`` 展示。"""
    out: dict[str, str | None] = {}
    for name, mod in (("scikit-image", "skimage"), ("opencv", "cv2")):
        try:
            m = __import__(mod)
            out[name] = str(getattr(m, "__version__", "ok"))
        except Exception:  # noqa: BLE001
            out[name] = None
    return out


def _cv2() -> Any:
    """惰性导入 OpenCV；缺失时抛带安装命令的 :class:`ImagingError`。

    scikit-image 侧不走此封装，直接在各函数内 ``from skimage... import``，
    因为 skimage 用 lazy-loader，顶层 import 本身就很廉价。
    """
    try:
        import cv2  # type: ignore

        return cv2
    except ImportError as e:
        raise ImagingError(f"缺少 OpenCV：{INSTALL_HINT}") from e


# ---------------------------------------------------------------------------
# I/O 辅助
# ---------------------------------------------------------------------------
def _load_rgb(path: str | Path, *, mode: str = "RGB") -> np.ndarray:
    """读图为 uint8 numpy 数组（``RGB`` 或 ``RGBA`` 或 ``L``）。"""
    p = Path(path).expanduser()
    if not p.is_file():
        raise ImagingError(f"图像不存在：{p}")
    with Image.open(p) as im:
        return np.asarray(im.convert(mode), dtype=np.uint8)


def _save(arr: np.ndarray, dest: str | Path, *, quality: int = 95) -> Path:
    """把 uint8 数组写盘；后缀决定格式，jpg 用 quality 压缩。"""
    d = Path(dest).expanduser()
    d.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(arr)
    kwargs: dict[str, Any] = {}
    if d.suffix.lower() in (".jpg", ".jpeg"):
        kwargs["quality"] = int(quality)
        if img.mode == "RGBA":
            img = img.convert("RGB")  # JPEG 无 alpha 通道
    img.save(d, **kwargs)
    return d


def _to_bgr(arr_rgb: np.ndarray) -> np.ndarray:
    return arr_rgb[:, :, ::-1].copy() if arr_rgb.ndim == 3 else arr_rgb


def _from_bgr(arr_bgr: np.ndarray) -> np.ndarray:
    return arr_bgr[:, :, ::-1].copy() if arr_bgr.ndim == 3 else arr_bgr


def _parse_points(spec: str | Sequence[Any] | None) -> np.ndarray | None:
    """解析点列：``"x1,y1;x2,y2;x3,y3;x4,y4"`` 或已有序列 → float32 (N,2)。"""
    if spec is None:
        return None
    if isinstance(spec, str):
        pairs = [p for p in spec.replace("|", ";").split(";") if p.strip()]
        pts = [[float(v) for v in p.split(",")] for p in pairs]
    else:
        pts = [list(map(float, pt)) for pt in spec]
    arr = np.asarray(pts, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ImagingError(f"点列格式应为 (N,2)，实得 {arr.shape}")
    return arr


# ---------------------------------------------------------------------------
# 图层：alpha 合成 / 通道分离（消费方舟「图层拆分」的 ≤16 层输出）
# ---------------------------------------------------------------------------
def split_alpha(
    src: str | Path, dest_dir: str | Path, *, stem: str | None = None
) -> dict[str, Path]:
    """把带 alpha 的图拆成 ``_rgb.png`` + ``_alpha.png``（灰度）。

    方舟图层拆分返回的每层都带透明通道；分离后 alpha 可直接当 mask 用于
    :func:`fuse` / :func:`inpaint`。
    """
    rgba = _load_rgb(src, mode="RGBA")
    d = Path(dest_dir).expanduser()
    d.mkdir(parents=True, exist_ok=True)
    base = stem or Path(src).stem
    rgb_p = _save(rgba[:, :, :3], d / f"{base}_rgb.png")
    alpha_p = _save(rgba[:, :, 3], d / f"{base}_alpha.png")
    return {"rgb": rgb_p, "alpha": alpha_p}


def composite(
    base: str | Path,
    layers: Iterable[str | Path],
    dest: str | Path,
    *,
    opacities: Sequence[float] | None = None,
    positions: Sequence[tuple[int, int]] | None = None,
) -> Path:
    """按顺序 alpha 合成多张图层到底图之上。

    - ``opacities``：每层不透明度 0–1（默认全 1）；
    - ``positions``：每层左上角偏移 ``(x, y)``（默认 ``(0, 0)``，即整幅对齐）。

    各层尺寸可与底图不同：超出部分被裁掉，不足部分留空。
    """
    layer_list = [Path(p).expanduser() for p in layers]
    if not layer_list:
        raise ImagingError("composite 需要至少一张图层")
    ops = list(opacities) if opacities else [1.0] * len(layer_list)
    poss = list(positions) if positions else [(0, 0)] * len(layer_list)
    if not (len(ops) == len(layer_list) == len(poss)):
        raise ImagingError(
            f"图层 {len(layer_list)} 张，但 opacities={len(ops)} / positions={len(poss)} 数量不匹配"
        )

    canvas = Image.open(Path(base).expanduser()).convert("RGBA")
    for p, op, (ox, oy) in zip(layer_list, ops, poss, strict=True):
        with Image.open(p) as im:
            layer = im.convert("RGBA")
        if not 0.0 <= float(op) <= 1.0:
            raise ImagingError(f"opacity 必须在 0–1，实得 {op}")
        if float(op) < 1.0:
            r, g, b, a = layer.split()
            a = a.point(lambda v: int(v * float(op)))
            layer = Image.merge("RGBA", (r, g, b, a))
        tmp = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        tmp.paste(layer, (int(ox), int(oy)))
        canvas = Image.alpha_composite(canvas, tmp)

    out = Path(dest).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    return out


# ---------------------------------------------------------------------------
# 泊松融合 / 算法 inpaint（OpenCV 独有，均**不需要 GPU**）
# ---------------------------------------------------------------------------
def fuse(
    src: str | Path,
    base: str | Path,
    dest: str | Path,
    *,
    center: tuple[int, int] | None = None,
    mask: str | Path | None = None,
    mode: str = "mixed",
) -> Path:
    """把 ``src`` 泊松融合进 ``base``（``cv2.seamlessClone``）。

    用途：将 AI 生成素材无缝嵌入实拍照片或 COMSOL 场图，边界色调自动过渡——
    这是本项目"Matplotlib 叠加 COMSOL 场图与实验图"合成规范的自然延伸。

    **关键前提**：泊松融合迁移的是**梯度**，不是绝对颜色。均匀色素材融进均匀底图会
    完全"消失"（方程无源项，解恒等于边界值）；要贴一块纯色/整幅图请用
    :func:`composite`，而不是本函数。

    - ``center``：src 在 base 中的落点（像素，默认 base 中心）；
    - ``mask``：融合区域（灰度图，非零为融合区）。默认取 src 的 alpha 通道；
      src 无 alpha 时用全白矩形（整块融合）；
    - ``mode``：``mixed``（默认）取 max(|∇src|,|∇dst|)——底图纹理会**透出来**，适合
      素材本身平坦而底图有细节；``normal`` 只用 |∇src|——底图纹理被抹成边界值的调和
      过渡。二者仅在**底图有纹理**时可区分（均匀底图上输出完全相同）。
    """
    cv2 = _cv2()
    src_rgba = _load_rgb(src, mode="RGBA")
    base_rgb = _load_rgb(base, mode="RGB")
    h, w = src_rgba.shape[:2]
    bh, bw = base_rgb.shape[:2]

    if mask is not None:
        m = _load_rgb(mask, mode="L")
        if m.shape[:2] != (h, w):
            raise ImagingError(f"mask 尺寸 {m.shape[:2]} 与 src {(h, w)} 不一致")
        mask_arr = (m > 127).astype(np.uint8) * 255
    elif src_rgba[:, :, 3].max() > 0:
        mask_arr = (src_rgba[:, :, 3] > 127).astype(np.uint8) * 255
    else:
        mask_arr = np.full((h, w), 255, dtype=np.uint8)
    if mask_arr.max() == 0:
        raise ImagingError("融合区域为空（mask 全黑）")

    cx, cy = center if center else (bw // 2, bh // 2)
    if not (w <= bw and h <= bh):
        raise ImagingError(f"src {w}x{h} 大于 base {bw}x{bh}，无法融合（请先缩小 src）")

    flag = cv2.MIXED_CLONE if str(mode).lower() == "mixed" else cv2.NORMAL_CLONE
    try:
        out_bgr = cv2.seamlessClone(
            _to_bgr(src_rgba[:, :, :3]),
            _to_bgr(base_rgb),
            mask_arr,
            (int(cx), int(cy)),
            flag,
        )
    except cv2.error as e:
        raise ImagingError(f"seamlessClone 失败：{e}") from e
    return _save(_from_bgr(out_bgr), dest)


def inpaint(
    src: str | Path,
    dest: str | Path,
    *,
    mask: str | Path | None = None,
    radius: float = 3.0,
    method: str = "telea",
) -> Path:
    """算法级 inpaint（``cv2.inpaint``），抹除瑕疵、水印残留、杂物。

    与扩散 inpaint 不同：这是 Telea / Navier-Stokes 经典算法，**不需要 GPU、不需要
    模型权重**，毫秒级完成，适合小面积修补。``mask`` 非零处为待修补区域；
    未给 mask 时报错（避免误抹全图）。
    """
    cv2 = _cv2()
    if mask is None:
        raise ImagingError("inpaint 必须提供 --mask（非零区域为待修补区）")
    rgb = _load_rgb(src, mode="RGB")
    m = _load_rgb(mask, mode="L")
    if m.shape[:2] != rgb.shape[:2]:
        raise ImagingError(f"mask 尺寸 {m.shape[:2]} 与 src {rgb.shape[:2]} 不一致")
    mask_arr = (m > 127).astype(np.uint8)
    if mask_arr.max() == 0:
        raise ImagingError("mask 全黑，无待修补区域")
    flag = (
        cv2.INPAINT_NS if str(method).lower() in ("ns", "navier") else cv2.INPAINT_TELEA
    )
    out_bgr = cv2.inpaint(_to_bgr(rgb), mask_arr, float(radius), flag)
    return _save(_from_bgr(out_bgr), dest)


def make_mask(
    src: str | Path,
    dest: str | Path,
    *,
    method: str = "otsu",
    thresh: int | None = None,
    invert: bool = False,
    blur: int = 0,
    rect: Sequence[int] | None = None,
    iterations: int = 5,
) -> Path:
    """生成二值 mask（供 :func:`fuse` / :func:`inpaint` 使用）。

    - ``method="otsu"``   自动阈值（``cv2.threshold`` + ``THRESH_OTSU``），适合主体/背景对比明显；
    - ``method="manual"`` 固定阈值，需给 ``thresh``（0–255）；
    - ``method="canny"``  边缘检测后闭运算成轮廓（``cv2.Canny`` + ``dilate``）；
    - ``method="grabcut"``交互式抠图（``cv2.grabCut``），需给 ``rect=[x,y,w,h]`` 框住主体，
      对"从复杂背景抠出实验样品/装置"很有用；
    - ``blur`` > 0 时先高斯模糊抑噪；``invert`` 反转前景/背景。
    """
    cv2 = _cv2()
    gray = _load_rgb(src, mode="L")
    if blur and int(blur) > 0:
        k = int(blur) * 2 + 1
        gray = cv2.GaussianBlur(gray, (k, k), 0)

    meth = str(method).lower()
    if meth == "otsu":
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif meth == "manual":
        if thresh is None:
            raise ImagingError('method="manual" 需要 --thresh（0-255）')
        _, binary = cv2.threshold(gray, int(thresh), 255, cv2.THRESH_BINARY)
    elif meth == "canny":
        lo = int(thresh) if thresh else 50
        edges = cv2.Canny(gray, lo, lo * 2)
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.dilate(edges, kernel, iterations=2)
    elif meth == "grabcut":
        if not rect or len(rect) != 4:
            raise ImagingError('method="grabcut" 需要 --rect x,y,w,h（框住主体）')
        h, w = gray.shape[:2]
        x, y, rw, rh = (int(v) for v in rect)
        if x + rw >= w or y + rh >= h:
            raise ImagingError(f"rect {rect} 超出图像范围 {w}x{h}")
        bgr = _to_bgr(_load_rgb(src, mode="RGB"))
        gcmask = np.zeros(gray.shape[:2], np.uint8)
        cv2.grabCut(
            bgr,
            gcmask,
            (x, y, rw, rh),
            None,
            None,
            max(1, int(iterations)),
            cv2.GC_INIT_WITH_RECT,
        )
        binary = np.where(
            (gcmask == cv2.GC_FGD) | (gcmask == cv2.GC_PR_FGD), 255, 0
        ).astype(np.uint8)
    else:
        raise ImagingError(
            f"未知 method={method!r}；支持 otsu / manual / canny / grabcut"
        )

    if invert:
        binary = cv2.bitwise_not(binary)
    return _save(binary, dest)


# ---------------------------------------------------------------------------
# 形态学 / 几何校正（scikit-image / OpenCV）
# ---------------------------------------------------------------------------
def morphology(
    src: str | Path,
    dest: str | Path,
    *,
    op: str = "open",
    radius: int = 2,
    iterations: int = 1,
) -> Path:
    """形态学开/闭/腐蚀/膨胀运算（``skimage.morphology``）。

    ``open`` 去小噪点（先腐蚀后膨胀），``close`` 填小孔洞（先膨胀后腐蚀），
    对清理 AI 生成图的碎斑、二值 mask 的毛刺都常用。作用在灰度图上。
    """
    from skimage import morphology as skmorph  # type: ignore

    funcs = {
        "open": skmorph.opening,
        "close": skmorph.closing,
        "erode": skmorph.erosion,
        "dilate": skmorph.dilation,
    }
    key = str(op).lower()
    if key not in funcs:
        raise ImagingError(f"未知 op={op!r}；支持 {sorted(funcs)}")
    gray = _load_rgb(src, mode="L")
    r = max(1, int(radius))
    selem = skmorph.disk(r)
    out = gray
    for _ in range(max(1, int(iterations))):
        out = funcs[key](out, selem)
    return _save(np.asarray(out, dtype=np.uint8), dest)


def perspective(
    src: str | Path,
    dest: str | Path,
    *,
    src_pts: str | Sequence[Any],
    dst_pts: str | Sequence[Any] | None = None,
    size: tuple[int, int] | None = None,
    rotate: float | None = None,
) -> Path:
    """几何校正：透视变换（把斜拍的实验照片拉正）或纯旋转。

    - ``src_pts``：原图上 4 个角点，``"x1,y1;x2,y2;x3,y3;x4,y4"``（顺时针）；
    - ``dst_pts``：目标角点，默认取 ``size``（或原图尺寸）的矩形四角 → 即"拉正成矩形"；
    - ``rotate``：给了角度（度，逆时针）则只做旋转，忽略点列。
    """
    cv2 = _cv2()
    rgb = _load_rgb(src, mode="RGB")
    h, w = rgb.shape[:2]

    if rotate is not None:
        center = (w / 2.0, h / 2.0)
        m = cv2.getRotationMatrix2D(center, float(rotate), 1.0)
        out = cv2.warpAffine(
            _to_bgr(rgb),
            m,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return _save(_from_bgr(out), dest)

    sp = _parse_points(src_pts)
    if sp is None or sp.shape[0] != 4:
        raise ImagingError("透视变换需要恰好 4 个 src 点")
    dp = _parse_points(dst_pts)
    if dp is None:
        ow, oh = size if size else (w, h)
        dp = np.array(
            [[0, 0], [ow - 1, 0], [ow - 1, oh - 1], [0, oh - 1]], dtype=np.float32
        )
    if dp.shape[0] != 4:
        raise ImagingError("透视变换需要恰好 4 个 dst 点")
    ow, oh = (
        size
        if size
        else (int(round(dp[:, 0].max())) + 1, int(round(dp[:, 1].max())) + 1)
    )
    m = cv2.getPerspectiveTransform(sp, dp)
    out = cv2.warpPerspective(_to_bgr(rgb), m, (ow, oh), flags=cv2.INTER_CUBIC)
    return _save(_from_bgr(out), dest)


# ---------------------------------------------------------------------------
# 测量 / 配准（scikit-image，科研量化）
# ---------------------------------------------------------------------------
def measure_regions(
    src: str | Path,
    *,
    min_area: int = 50,
    thresh: int | None = None,
    mask: str | Path | None = None,
) -> list[dict[str, Any]]:
    """连通域测量：面积、质心、外接框、长轴/短轴、离心率（``skimage.measure``）。

    科研用途：配 scale bar 后可把像素面积换算成物理面积；统计 AI 示意图中
    各元件的相对占比；核对生成图是否符合真实几何比例。

    二值化来源：给了 ``mask`` 用 mask；给了 ``thresh`` 用固定阈值；否则 Otsu 自动阈值。
    """
    from skimage import measure as skmeasure  # type: ignore

    if mask is not None:
        binary = _load_rgb(mask, mode="L") > 127
    else:
        gray = _load_rgb(src, mode="L")
        if thresh is not None:
            binary = gray > int(thresh)
        else:
            try:
                from skimage.filters import threshold_otsu  # type: ignore

                binary = gray > threshold_otsu(gray)
            except ValueError:
                binary = gray > 127  # 常量图无法 Otsu，退回中值

    labeled = skmeasure.label(binary, connectivity=2)
    out: list[dict[str, Any]] = []
    for rp in skmeasure.regionprops(labeled):
        if rp.area < int(min_area):
            continue
        cy, cx = rp.centroid
        minor = float(rp.axis_minor_length or 0.0)
        major = float(rp.axis_major_length or 0.0)
        min_row, min_col, max_row, max_col = rp.bbox
        out.append(
            {
                "label": int(rp.label),
                "area_px": int(rp.area),
                "centroid": (round(float(cx), 2), round(float(cy), 2)),
                "bbox_xyxy": (int(min_col), int(min_row), int(max_col), int(max_row)),
                "axis_major": round(major, 2),
                "axis_minor": round(minor, 2),
                "aspect": round(major / minor, 3) if minor > 0 else None,
                "eccentricity": round(float(rp.eccentricity), 4),
                "solidity": round(float(rp.solidity), 4),
            }
        )
    return sorted(out, key=lambda d: d["area_px"], reverse=True)


def align(
    src: str | Path,
    ref: str | Path,
    dest: str | Path | None = None,
    *,
    upsample: int = 10,
) -> dict[str, Any]:
    """相位相关配准：求 ``src`` 相对 ``ref`` 的亚像素平移量（``skimage.registration``）。

    用途：把 AI 生成图 / 实拍图与 COMSOL 场图对齐后再叠加。给了 ``dest`` 则
    应用该平移并把 src 对齐后写出。要求两图尺寸一致（不一致先 ``adjust --resize``）。
    """
    from skimage.registration import phase_cross_correlation  # type: ignore

    a = _load_rgb(src, mode="L").astype(np.float64)
    b = _load_rgb(ref, mode="L").astype(np.float64)
    if a.shape != b.shape:
        raise ImagingError(
            f"配准要求两图尺寸一致：src {a.shape} vs ref {b.shape}（先用 adjust --resize 统一）"
        )
    shift, error, _phasediff = phase_cross_correlation(
        b, a, upsample_factor=max(1, int(upsample))
    )
    result: dict[str, Any] = {
        "shift_yx": [round(float(v), 4) for v in shift],
        "shift_xy": [round(float(shift[1]), 4), round(float(shift[0]), 4)],
        "rms_error": round(float(error), 6) if error is not None else None,
    }
    if dest is not None:
        from scipy import ndimage as ndi  # type: ignore

        rgb = _load_rgb(src, mode="RGB").astype(np.float64)
        moved = ndi.shift(
            rgb, (shift[0], shift[1], 0), order=1, mode="constant", cval=0.0
        )
        result["out"] = str(_save(np.clip(moved, 0, 255).astype(np.uint8), dest))
    return result
