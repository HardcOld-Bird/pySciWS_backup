"""wtpsplit SaT 本地模型句切冒烟（步骤1·任务二，2026-11-17）。

验证点：
1. SaT 以**本地模型路径** + 本地 tokenizer 完成句切（HF_HUB_OFFLINE=1 证明全自包含，
   无任何联网回退）；
2. 输入为中英混排真实小样（含 Dr. 缩写、括号术语、中英文标点、省略号）；
3. 记录加载/句切耗时与进程峰值 RAM（psutil peak_wset）。

运行：sub_root/.venv/Scripts/python.exe deps_management/scratch/sat_sentence_smoke.py
"""

import os
import time
from pathlib import Path

# 离线证明：先于 wtpsplit/transformers import 设置
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
# 与 vendor 脚本一致：缓存入项目内 + 关 Xet（本机 ~/.cache/huggingface/xet EACCES）
SUB_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault(
    "HF_HOME", str(SUB_ROOT / "deps_management" / "deps" / "cache" / "huggingface")
)

import psutil  # noqa: E402
from wtpsplit import SaT  # noqa: E402

MODELS = SUB_ROOT / "deps_management" / "deps" / "models" / "wtpsplit"
MODEL_DIR = MODELS / "sat-3l"
TOK_DIR = MODELS / "xlm-roberta-base-tokenizer"

TEXT = (
    "机器学习（Machine Learning）是人工智能的一个分支，它使计算机能够从数据中学习。"
    "Dr. Smith visited Beijing in 2024 and discussed AI safety with Prof. Wang. "
    "深度学习（deep learning）在图像识别（image recognition）领域取得了突破性进展！"
    "What about NLP? 自然语言处理（Natural Language Processing, NLP）包括分词、句法分析和语义理解等任务……"
    "例如，Transformer 架构于 2017 年被提出，它对后续研究影响深远。"
    "数值实验表明，当参数 ε→0 时误差以 O(h²) 速度衰减。"
)

proc = psutil.Process()
rss0 = proc.memory_info().rss

t0 = time.perf_counter()
sat = SaT(
    str(MODEL_DIR),
    tokenizer_name_or_path=str(TOK_DIR),
    ort_providers=["CPUExecutionProvider"],
)
t_load = time.perf_counter() - t0
rss1 = proc.memory_info().rss

t1 = time.perf_counter()
sents = sat.split(TEXT)
t_split = time.perf_counter() - t1

peak = proc.memory_info().peak_wset
print(f"model_dir   : {MODEL_DIR}")
print(f"tokenizer   : {TOK_DIR}")
print(f"t_load      : {t_load:.2f}s   rss +{(rss1 - rss0) / 1e6:.0f} MB")
print(f"t_split     : {t_split:.2f}s   (n={len(sents)}, {len(TEXT)} chars)")
print(f"rss_end     : {proc.memory_info().rss / 1e6:.0f} MB")
print(f"peak_wset   : {peak / 1e6:.0f} MB")
joined = "".join(sents)
print(f"lossless    : {joined == TEXT}")
print("--- segments ---")
for i, s in enumerate(sents, 1):
    print(f"{i:2d}. {s!r}")
