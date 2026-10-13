# %% 为价表 77 行模型挂上第三方综合性能分（Artificial Analysis Intelligence Index）
# 口径：AA Intelligence Index v4.3.x（10 项评测合成：AA-Briefcase、GDPval-AA、AutomationBench-AA、
# Terminal-Bench 4.0、SciCode、HLE、GDP.pdf、CritPt、AA-Omniscience、AA-LCR）。
# 抓取来源：DataLearner 对 AA 榜单的镜像（全榜 254 行，2026-10 快照）；AA 官方前端为 JS 渲染，
# 直接抓取只能拿到个别行。镜像标注「数据仅供参考，以官方来源为准」。
# 关键纪律：分数只在同一版本/快照内可比——旧文章里 Kimi K3=57、GLM-5.2=51、MiniMax M3=44 等
# 均为 v4.0/v4.1 口径，与下表 v4.3.x 数值不可直接比较，故单独标「异源参照」。
import sys
from pathlib import Path

import pandas as pd

from pysci.paths import assert_within_data

OUT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT_DIR))
from build_pricing_tables import PAYG  # noqa: E402

SNAPSHOT = "AA Intelligence Index v4.3.x（DataLearner 镜像，2026-10 快照，同榜可比）"

# (提供商, 模型) → (AA 模型名, 分数, 可比性, 备注)
# 可比性：同榜=本快照内直接可比；异源=取自其他版本/文章，仅供参照；未评测/不适用=无分
PERF: dict[tuple[str, str], tuple[str, float | None, str, str]] = {
    ("阿里云百炼", "qwen3.8-max"): ("Qwen3.8 Max (0902)", 45, "同榜", "阿里旗舰"),
    ("千问AI平台", "qwen3.8-max"): ("Qwen3.8 Max (0902)", 45, "同榜", "同源同模型"),
    ("阿里云百炼", "qwen3.8-flash"): ("Qwen3.8-Flash-Next", 40, "同榜", "flash 名对应关系待官方确认"),
    ("千问AI平台", "qwen3.8-flash"): ("Qwen3.8-Flash-Next", 40, "同榜", "同上"),
    ("阿里云百炼", "qwen3.7-max"): ("—", None, "未评测", "本快照未收录上一代 max"),
    ("阿里云百炼", "qwen3.7-plus"): ("Qwen3.7-Plus", 25, "同榜", ""),
    ("阿里云百炼", "qwen3.7-flash"): ("—", None, "未评测", "本快照未收录"),
    ("阿里云百炼", "ZHIPU/GLM-5.3（智谱直供）"): ("GLM-5.3 (max)", 45, "同榜", "同模型，渠道不同"),
    ("阿里云百炼", "kimi-k3"): ("Kimi K3 (max)", 44, "同榜", ""),
    ("阿里云百炼", "kimi-k2.7-code"): ("Kimi K2.7 Code", 26, "同榜", ""),
    ("阿里云百炼", "kimi-k2.5"): ("—", None, "未评测", "本快照未收录"),
    ("阿里云百炼", "vanchin/deepseek-v4-pro（万擎直供）"): ("DeepSeek-V4-Pro (max)", 36, "同榜", "同模型，第三方直供"),
    ("阿里云百炼", "vanchin/deepseek-v4.1-flash（万擎直供）"): ("DeepSeek V4.1 Flash (max)", 39, "同榜", "同模型，第三方直供"),
    ("阿里云百炼", "siliconflow/deepseek-v3.2（硅基流动直供）"): ("—", None, "未评测", "V3.2 未进本快照"),

    ("智谱", "GLM-5.3"): ("GLM-5.3 (max)", 45, "同榜", "低推理档 GLM-5.3 (low)=34"),
    ("智谱", "GLM-5.3-Flash"): ("GLM-5.3-Flash", 42, "同榜", ""),
    ("智谱", "GLM-5.3-FlashX"): ("—", None, "未评测", "本快照未单列，应略低于 42"),
    ("智谱", "GLM-5.2"): ("GLM-5.2 (max)", 34, "异源", "取自镜像 2026-09-13 榜单；旧文章 v4.1 口径报 51，不可与 45 并列比较"),
    ("智谱", "GLM-5.1（输入<32K / ≥32K）"): ("—", None, "未评测", "本快照未收录"),
    ("智谱", "GLM-5-Turbo（输入<32K）"): ("—", None, "未评测", "本快照未收录"),
    ("智谱", "GLM-5（输入<32K）"): ("—", None, "未评测", "本快照未收录"),
    ("智谱", "GLM-4.7（输入<32K，输出<0.2K）"): ("—", None, "未评测", "上一代，本快照未收录"),

    ("Kimi", "kimi-k3"): ("Kimi K3 (max)", 44, "同榜", "低推理档 Kimi K3 (low)=30"),
    ("Kimi", "kimi-k2.7-code"): ("Kimi K2.7 Code", 26, "同榜", ""),
    ("Kimi", "kimi-k2.7-code-highspeed"): ("Kimi K2.7 Code", 26, "同榜", "同模型极速档，性能视为等同"),
    ("Kimi", "kimi-k2.6"): ("Kimi K2.6", 43, "异源", "取自 v4.0/v4.1 文章口径；本快照未收录，与 44/45 不可直接比较"),
    ("Kimi", "kimi-k2.5"): ("—", None, "未评测", "本快照未收录"),

    ("MiniMax", "MiniMax-M3（输入≤512K）"): ("MiniMax M3", 29, "同榜", "镜像 09-13 报 30、旧文 v4.1 报 44，版本漂移明显"),
    ("MiniMax", "MiniMax-M2.7"): ("—", None, "未评测", "本快照未收录"),
    ("MiniMax", "MiniMax-M2.7-highspeed"): ("—", None, "未评测", "本快照未收录"),
    ("MiniMax", "MiniMax-M2.5 / M2.1 / M2"): ("—", None, "未评测", "AA 有 M2.5 条目但本快照未抓到分数"),
    ("MiniMax", "MiniMax-M3.1-Flash-Preview"): ("—", None, "未评测", "预览模型，AA 尚未评测"),

    ("DeepSeek", "deepseek-flash（V4.1-Flash）闲时"): ("DeepSeek V4.1 Flash (max)", 39, "同榜", "非推理档=25"),
    ("DeepSeek", "deepseek-v4-pro 闲时"): ("DeepSeek-V4-Pro (max)", 36, "同榜", "高推理档；(high)=30"),

    ("小米MiMo", "MiMo-V2.6-Pro"): ("MiMo-V2.6-Pro (Xiaomi)", 46, "同榜", "本快照国产最高分之一"),
    ("小米MiMo", "MiMo-V2.6-Flash"): ("MiMo-V2.6-Flash (Xiaomi)", 38, "同榜", ""),
    ("小米MiMo", "MiMo-V2.6-Pro-UltraSpeed"): ("MiMo-V2.6-Pro", 46, "同榜", "同模型极速档"),
    ("小米MiMo", "MiMo-V2.5（输入<256K）"): ("—", None, "未评测", "本快照未收录"),
    ("小米MiMo", "MiMo-V2.5-Pro（输入<256K）"): ("MiMo-V2.5-Pro", 54, "异源", "v4.0 口径（当时排名第 10）；与 v4.3.x 不可比"),

    # 火山方舟：AA 未评测豆包 Seed 2.x 全系——本维度对自研模型失效，只能靠单项 benchmark 或实测
    ("火山方舟", "doubao-seed-evolving（输入≤1024K）"): ("—", None, "未评测", "AA 无此条目"),
    ("火山方舟", "doubao-seed-2.1-pro（输入≤1024K）"): ("—", None, "未评测", "AA 无此条目；第三方报 Terminal-Bench 2.1=71.0（GPT-5.5=73.8），口径不可比"),
    ("火山方舟", "doubao-seed-2.1-turbo（输入≤256K）"): ("—", None, "未评测", "AA 无此条目"),
    ("火山方舟", "doubao-seed-2.1-lite（输入≤1024K）"): ("—", None, "未评测", "AA 无此条目"),
    ("火山方舟", "doubao-seed-2.0-pro（≤32K / 32-128K / 128-256K）"): ("—", None, "未评测", "AA 无此条目"),
    ("火山方舟", "doubao-seed-2.0-lite（≤32K）"): ("—", None, "未评测", "AA 无此条目"),
    ("火山方舟", "doubao-seed-2.0-mini（≤32K）"): ("—", None, "未评测", "AA 无此条目；唯一相关条目 Doubao Seed Code=17 系另一型号"),
    ("火山方舟", "doubao-seed-character（≤32K）"): ("—", None, "未评测", "角色专用模型，AA 不评"),
    ("火山方舟", "glm-5.3-flash（方舟渠道）"): ("GLM-5.3-Flash", 42, "同榜", "同模型，渠道不同"),
    ("火山方舟", "glm-5.2（方舟渠道）"): ("GLM-5.2 (max)", 34, "异源", "同智谱行口径"),
    ("火山方舟", "deepseek-v4.1-flash（方舟渠道）闲时"): ("DeepSeek V4.1 Flash (max)", 39, "同榜", ""),
    ("火山方舟", "deepseek-v4-pro 正式版（方舟渠道）"): ("DeepSeek-V4-Pro (max)", 36, "同榜", ""),
    ("火山方舟", "deepseek-v4-flash 正式版（方舟渠道）"): ("—", None, "未评测", "V4-Flash 正式版未进本快照；同族 V4.1-Flash=39"),

    ("SiliconFlow", "GLM-5.3-Flash"): ("GLM-5.3-Flash", 42, "同榜", "同模型，渠道不同"),
    ("SiliconFlow", "GLM-5.3 / GLM-5.2"): ("GLM-5.3 (max)", 45, "同榜", "取该行的 GLM-5.3 身份"),
    ("SiliconFlow", "GLM-5.1 (Pro)"): ("—", None, "未评测", "本快照未收录"),
    ("SiliconFlow", "GLM-4.5V / GLM-4.5-Air"): ("—", None, "未评测", "上一代，本快照未收录"),
    ("SiliconFlow", "Hy4 preview（混元）"): ("—", None, "未评测", "本快照仅列 Hy3=25"),
    ("SiliconFlow", "Hunyuan-A13B-Instruct"): ("—", None, "未评测", "本快照未收录"),
    ("SiliconFlow", "DeepSeek-V4-Flash"): ("—", None, "未评测", "同方舟 V4-Flash 行"),
    ("SiliconFlow", "DeepSeek-V4-Pro"): ("DeepSeek-V4-Pro (max)", 36, "同榜", ""),
    ("SiliconFlow", "DeepSeek-V3.2 / V3.2 (Pro)"): ("—", None, "未评测", "本快照未收录"),
    ("SiliconFlow", "LongCat-2.0（美团）"): ("LongCat 2.0", 19, "同榜", ""),
    ("SiliconFlow", "Kimi-K2.7-Code"): ("Kimi K2.7 Code", 26, "同榜", ""),
    ("SiliconFlow", "Kimi-K2.6 (Pro)"): ("Kimi K2.6", 43, "异源", "v4.0/v4.1 口径"),
    ("SiliconFlow", "Qwen3.8-27B（开源版）"): ("Qwen3.8-27B (xhigh)", 34, "同榜", "medium=28、low=26、非推理=20；与商业版 3.8-max/flash 非同模型"),
    ("SiliconFlow", "Qwen3.5-35B-A3B（输入<128K）"): ("—", None, "未评测", "本快照仅有 Qwen3.6-35B-A3B=18"),
    ("SiliconFlow", "Qwen3.5-122B-A10B（输入<128K）"): ("Qwen3.5 122B A10B", 18, "同榜", "推理档 16、非推理档 18"),
    ("SiliconFlow", "0.7/2.8 档（Qwen3-VL-30B-A3B、Qwen3-Omni-30B-A3B（Instruct/Thinking/Captioner）、Qwen3-Coder-30B-A3B、Qwen3-30B-A3B-Instruct-2507）"):
        ("Qwen3-Omni-30B-A3B", 10, "同榜", "该档多为 30B-A3B 开源小模型，AA 条目 6–10 分区间"),
    ("SiliconFlow", "Qwen3-VL-32B-Instruct / Qwen3-32B"): ("—", None, "未评测", "本快照未收录"),
    ("SiliconFlow", "Qwen3-VL-32B-Thinking"): ("—", None, "未评测", "本快照未收录"),
    ("SiliconFlow", "Qwen3-VL-8B-Thinking"): ("—", None, "未评测", "本快照未收录"),
    ("SiliconFlow", "Qwen3-VL-Embedding / Reranker-8B"): ("—", None, "不适用", "AA 另有 embedding/reranker 专榜"),
    ("SiliconFlow", "Qwen3-Embedding/Reranker 8B·4B·0.6B"): ("—", None, "不适用", "同上"),
    ("SiliconFlow", "Qwen2.5-72B-Instruct（含 128K 版）"): ("—", None, "未评测", "上一代开源模型"),
    ("SiliconFlow", "Step-3.5-Flash / Ling-flash-2.0 / Ling-mini-2.0 / Seed-OSS-36B / Xing4.0-29B"):
        ("—", None, "未评测", "本快照有 Step 5 Preview=44、Ring-flash-2.0=7，但与该行合并型号非一一对应"),
    ("SiliconFlow", "bge-m3 / bge-reranker-v2-m3 / bge-large-zh·en-v1.5"): ("—", None, "不适用", "embedding/reranker，不入智能指数"),
    ("SiliconFlow", "Z-Image-Turbo / Wan2.2-I2V（图像/视频）"): ("—", None, "不适用", "图像/视频生成，不入智能指数"),
}

# %% 组装
rows = []
for p in PAYG:
    key = (p["提供商"], p["模型"])
    aa_name, score, cmp_, note = PERF.get(key, ("—", None, "未匹配", "检查映射"))
    rows.append(dict(核实日期="2026-10-12", 提供商=p["提供商"], 模型=p["模型"],
                     AA模型名=aa_name, 智能指数=score, 可比性=cmp_, 备注=note))
perf = pd.DataFrame(rows)
missing = [k for k in {(p["提供商"], p["模型"]) for p in PAYG} if k not in PERF]
assert not missing, f"映射缺项：{missing}"

# %% 并上按量模拟月价 → 每分成本（同榜行才计算）
sim = pd.read_csv(OUT_DIR / "模拟月成本_按量.csv")
base = sim.groupby(["提供商", "模型"])["月总成本"].min().rename("按量模拟月价")
perf = perf.merge(base, on=["提供商", "模型"], how="left")
same = perf["可比性"].eq("同榜")
perf.loc[same, "每智能指点月成本"] = (perf.loc[same, "按量模拟月价"] / perf.loc[same, "智能指数"]).round(1)

perf.to_csv(OUT_DIR / "模型性能指数.csv", index=False, encoding="utf-8-sig")

# %% 输出：同榜可比集合，按分数降序
board = perf[same].sort_values(["智能指数", "按量模拟月价"], ascending=[False, True])
print(SNAPSHOT)
print(board[["提供商", "模型", "智能指数", "按量模拟月价", "每智能指点月成本"]].to_string(index=False))
print(f"\n覆盖：同榜 {same.sum()} 行 / 异源参照 {perf['可比性'].eq('异源').sum()} 行 / "
      f"未评测 {perf['可比性'].eq('未评测').sum()} 行 / 不适用 {perf['可比性'].eq('不适用').sum()} 行，共 {len(perf)} 行")
print(f"独立型号分数（同榜）：{board.drop_duplicates('AA模型名')[['AA模型名','智能指数']].shape[0]} 个")
