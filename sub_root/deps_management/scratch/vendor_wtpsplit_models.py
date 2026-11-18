"""Vendor wtpsplit SaT 小模型 + XLM-R tokenizer 至项目内（步骤1·任务二，2026-11-17）。

产物（安装物可重建，不入仓；原地 ADR 与重建命令见同目录 SOURCE.md）：
  deps_management/deps/models/wtpsplit/sat-3l/
      config.json, model_optimized.onnx          # segment-any-text/sat-3l（MIT，ONNX 推理路径）
  deps_management/deps/models/wtpsplit/xlm-roberta-base-tokenizer/
      tokenizer.json, tokenizer_config.json, sentencepiece.bpe.model
                                                 # FacebookAI/xlm-roberta-base（MIT）

sat-3l 为 CPU 友好的最小 SaT 变体（3 层，仅 CPU/可用内存 ~5GB 设备裁决）；
wtpsplit.SaT 本地目录加载只读 model_optimized.onnx + config.json，tokenizer 由
tokenizer_name_or_path 指到本地目录以实现全自包含（默认会去 HF 拉
facebookAI/xlm-roberta-base）。下载顺序（ADR）：HF 直连 → 代理 http://127.0.0.1:12450
→ hf-mirror/ModelScope 镜像。

重建：sub_root/.venv/Scripts/python.exe deps_management/scratch/vendor_wtpsplit_models.py

vendor fixup：重建后须补写 xlm-roberta-base-tokenizer/special_tokens_map.json（见 main()）——
上游仓库根无此文件，transformers 5.x TokenizersBackend 不从 tokenizer.json 推断
cls/sep/pad 角色，缺它则 wtpsplit extract() 因 cls_token_id=None 崩溃。
"""

import json
import os
import shutil
import time
from pathlib import Path

# ADR（2026-11-17）：本机 ~/.cache/huggingface/xet/ 写入报 EACCES（os error 5），
# 且 xet 传输层对大文件直接 I/O 失败——关 Xet 走经典 HTTPS 传输；
# HF 缓存一并重定向进项目内 deps/cache/huggingface（自包含，与 UV_CACHE_DIR 惯例一致）。
SUB_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HOME", str(SUB_ROOT / "deps_management" / "deps" / "cache" / "huggingface"))

from huggingface_hub import hf_hub_download  # noqa: E402

MODELS_ROOT = SUB_ROOT / "deps_management" / "deps" / "models" / "wtpsplit"

JOBS = [
    ("segment-any-text/sat-3l", "sat-3l", ["config.json", "model_optimized.onnx"]),
    (
        "FacebookAI/xlm-roberta-base",
        "xlm-roberta-base-tokenizer",
        ["tokenizer.json", "tokenizer_config.json", "sentencepiece.bpe.model"],
    ),
]


def main() -> None:
    for repo, local_dir, files in JOBS:
        d = MODELS_ROOT / local_dir
        d.mkdir(parents=True, exist_ok=True)
        for f in files:
            t0 = time.perf_counter()
            p = Path(hf_hub_download(repo_id=repo, filename=f, local_dir=str(d)))
            print(
                f"{repo}/{f} -> {p} "
                f"({p.stat().st_size / 1e6:.1f} MB, {time.perf_counter() - t0:.1f}s)"
            )
        # 清理 hf_hub_download 的 local_dir 元数据缓存，保持 vendor 目录干净
        meta = d / ".cache"
        if meta.exists():
            shutil.rmtree(meta)
    # vendor fixup（ADR 见模块 docstring）：补 special token 角色映射，
    # 取值与 tokenizer.json 的 added_tokens（<s>=0,<pad>=1,</s>=2,<unk>=3,<mask>=250001）一致。
    stm = MODELS_ROOT / "xlm-roberta-base-tokenizer" / "special_tokens_map.json"
    stm.write_text(
        json.dumps(
            {
                "bos_token": "<s>",
                "eos_token": "</s>",
                "cls_token": "<s>",
                "sep_token": "</s>",
                "unk_token": "<unk>",
                "pad_token": "<pad>",
                "mask_token": "<mask>",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"fixup -> {stm}")
    total = sum(f.stat().st_size for f in MODELS_ROOT.rglob("*") if f.is_file())
    print(f"TOTAL {total / 1e6:.1f} MB at {MODELS_ROOT}")


if __name__ == "__main__":
    main()
