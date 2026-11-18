"""真实出图冒烟脚本（hardware 标记，**当前挂起**：ARK_API_KEY 已过期，2026-11-17）。

状态：⑦域静态链（mock）已恒跑全绿；本脚本是**唯一**的真实云出图验收面，
按用户裁决"真实出图冒烟留待 key 更新后补做"暂不执行。

TODO（key 更新后跑）：
1. 更新项目根 ``sub_root/.env`` 的 ``ARK_API_KEY``（方舟控制台重新创建）与
   ``SILICONFLOW_API_KEY``（如需验副通道）；
2. 在本目录执行 ``python smoke_real.py``（或
   ``python -m pytest smoke_real.py -m hardware -s``）；
3. 逐项核对（每次真实计费，1K 单张最小档；失败勿反复重试烧钱）：
   - 方舟文生图：出图 → watermark 无水印 → 账本 + PNG tEXt 溯源；
   - 方舟 5.5 透明底：``background="transparent"`` 直出 → ``verify_alpha`` 过（alpha max=255）；
   - 副通道（可选）：硅基流动 Qwen-Image 出图 → url 1h 内立即下载；
   - measure/align 在真实生成图上冒烟（量化 + 配准）；
4. 把结果与踩坑记回 ``sub_root/others/plans/domains-06-07-3d-imagegen.md`` 验收节
   （5.5 参数/价目核验项一并勾销）。

成本护栏提醒：真实出图 = 真实计费；本脚本固定 1K/单张，产出落临时目录。
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from pysci.skills.image_generation import imaging, ledger, postprocess, providers

pytestmark = pytest.mark.hardware  # 外部服务 + 真实计费：默认不跑（tests.cmd all 跳过）


@pytest.mark.hardware
def test_real_ark_t2i_smoke():
    """方舟 1K 单张文生图：出图 → 无水印 → 账本 + PNG tEXt 溯源。"""
    out_dir = Path(tempfile.mkdtemp(prefix="imagegen_smoke_"))
    resp = providers.generate(
        "ark",
        "a minimal flat vector schematic of a red circle on white background",
        model="doubao-seedream-5-5",
        size="1K",
        n=1,
    )
    assert resp.ok_images, f"出图失败：{resp.failed_images}"
    paths = providers.save_images(resp, out_dir, stem="smoke_t2i")
    meta = {"prompt": resp.request.get("prompt", ""), "model": resp.model,
            "provider": "ark", "ts": ""}
    entry = ledger.record(prompt=meta["prompt"], model=resp.model, provider="ark",
                          out=paths[0], extra={"url_ttl": providers.guardrails.url_ttl_note("ark")})
    meta["ts"] = entry.ts
    if paths[0].suffix == ".png":
        assert ledger.embed_metadata(paths[0], meta), "PNG tEXt 溯源写入失败"
    print(f"[smoke] ark t2i -> {paths[0]}")


@pytest.mark.hardware
def test_real_ark_transparent_smoke():
    """方舟 5.5 透明底直出：background=transparent → verify_alpha 过（alpha max=255）。"""
    out_dir = Path(tempfile.mkdtemp(prefix="imagegen_smoke_"))
    resp = providers.generate(
        "ark",
        "a simple flat sticker of a blue hexagon, isolated on transparent background",
        model="doubao-seedream-5-5",
        transparent=True,
    )
    assert resp.ok_images, f"出图失败：{resp.failed_images}"
    paths = providers.save_images(resp, out_dir, stem="smoke_transparent")
    report = postprocess.verify_alpha(paths[0])
    assert report["ok"], f"透明底不合规：{report}"
    print(f"[smoke] transparent -> {paths[0]} alpha_max={report['alpha_max']}")


@pytest.mark.hardware
def test_real_siliconflow_t2i_smoke():
    """硅基流动副通道：1 张文生图 → url 立即下载（1h 失效）。"""
    out_dir = Path(tempfile.mkdtemp(prefix="imagegen_smoke_"))
    resp = providers.generate(
        "siliconflow",
        "a minimal flat illustration of an island with a lighthouse",
        model="Qwen/Qwen-Image",
        size="1328x1328",
        n=1,
        seed=7,  # 副通道 seed 全档生效，可复现
    )
    assert resp.ok_images, f"出图失败：{resp.failed_images}"
    paths = providers.save_images(resp, out_dir, stem="smoke_sf")
    ledger.record(prompt="smoke sf", model=resp.model, provider="siliconflow",
                  out=paths[0], seed=resp.seed)
    print(f"[smoke] siliconflow -> {paths[0]}")


@pytest.mark.hardware
def test_real_measure_align_on_generated():
    """measure/align 真图冒烟：透明底产物连通域量化 + 与底图相位配准。"""
    out_dir = Path(tempfile.mkdtemp(prefix="imagegen_smoke_"))
    resp = providers.generate(
        "ark", "a flat black square centered on pure white background",
        model="doubao-seedream-5-5", size="1K",
    )
    paths = providers.save_images(resp, out_dir, stem="smoke_measure")
    regions = imaging.measure_regions(paths[0], min_area=100)
    assert regions, "measure 未检出任何区域"
    res = imaging.align(paths[0], paths[0])  # 自配准：shift≈0
    assert abs(res["shift_yx"][0]) < 0.5 and abs(res["shift_yx"][1]) < 0.5
    print(f"[smoke] measure regions={len(regions)} self-align shift={res['shift_yx']}")


def main() -> int:
    """手动运行全部冒烟（pytest 基建之外的裸跑入口）。"""
    failures = 0
    for fn in (
        test_real_ark_t2i_smoke,
        test_real_ark_transparent_smoke,
        test_real_siliconflow_t2i_smoke,
        test_real_measure_align_on_generated,
    ):
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as e:  # noqa: BLE001 - 冒烟脚本：报告所有失败
            failures += 1
            print(f"FAIL {fn.__name__}: {e}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
