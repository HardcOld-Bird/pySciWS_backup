# %% 选型·步骤三：DeepSeek 系全渠道成本排名（按量 / 订阅等效同榜）
# 与 Qwen 表同源同口径：月价一律从 模拟月成本_按量.csv / 模拟月成本_订阅等效.csv 读取，
# 性能分从 模型性能指数.csv 读取，本脚本不重复登记价目。
# 用量口径：未缓存输入 101.573M / 缓存输入 1553.231M / 输出 15.785M（月化）。
# 峰谷分时（DeepSeek 特有）已在按量模拟中按 24h 均匀假设折入（闲时基准 ×1.5）。
from pathlib import Path

import pandas as pd

from pysci.paths import assert_within_data

OUT_DIR = Path(__file__).resolve().parent
assert_within_data(OUT_DIR)
import sys

sys.path.insert(0, str(OUT_DIR))
from build_usage_table import USAGE  # noqa: E402

ANCHOR = 40.0  # 主力锚点 qwen3.8-flash 的 AA 同榜分，断层线取 锚点−10
_u = pd.DataFrame(USAGE, columns=["日期", "未缓存输入", "缓存输入", "输出"])
_m = 30 / len(_u)
U_NC, U_C, U_O = (float(_u[c].sum()) * _m / 1e6 for c in ["未缓存输入", "缓存输入", "输出"])  # 月化百万 tokens

sim = pd.read_csv(OUT_DIR / "模拟月成本_按量.csv")
plan = pd.read_csv(OUT_DIR / "模拟月成本_订阅等效.csv")
perf = pd.read_csv(OUT_DIR / "模型性能指数.csv")

score = {(r["提供商"], r["模型"]): (r["智能指数"], r["可比性"], r["备注"]) for _, r in perf.iterrows()}
equiv = plan.set_index(["提供商", "模型"])[["订阅等效月价", "备注"]]

# (提供商, 型号) 全量清单：原厂 2 / 方舟 5 / SiliconFlow 3 / 百炼第三方直供 3
DEEPSEEK_ROWS = [
    ("DeepSeek", "deepseek-flash（V4.1-Flash）闲时", "原厂 V4.1-Flash，峰谷均摊后 1.5/0.03/6.0"),
    ("DeepSeek", "deepseek-v4-pro 闲时", "原厂 V4-Pro，峰谷均摊后 6.75/0.225/20.25"),
    ("火山方舟", "deepseek-v4.1-flash（方舟渠道）闲时", "方舟渠道，与原厂峰谷同构、价同"),
    ("火山方舟", "deepseek-v4-flash 正式版（方舟渠道）", "方舟常规档，3.0/0.10/9.0"),
    ("火山方舟", "deepseek-v4-flash 正式版（方舟渠道）（低优/批量）", "低优/批量档，输入输出 5 折、缓存价不变"),
    ("火山方舟", "deepseek-v4-pro 正式版（方舟渠道）", "方舟常规档 9.0/0.30/27.0，高于原厂闲时"),
    ("火山方舟", "deepseek-v4-pro 正式版（方舟渠道）（低优/批量）", "低优/批量 4.5/0.30/13.5≈原厂闲时价"),
    ("SiliconFlow", "DeepSeek-V4-Flash", "2–8 时五折，均匀分布后有效 ×0.667"),
    ("SiliconFlow", "DeepSeek-V4-Pro", "SF 渠道 12/1.0/24，明显高于原厂"),
    ("SiliconFlow", "DeepSeek-V3.2 / V3.2 (Pro)", "上一代 V3.2，本快照 AA 未收录"),
    ("阿里云百炼", "vanchin/deepseek-v4.1-flash（万擎直供）", "第三方直供、无缓存价→缓存按输入全价"),
    ("阿里云百炼", "vanchin/deepseek-v4-pro（万擎直供）", "第三方直供、无缓存价，价格最劣渠道"),
    ("阿里云百炼", "siliconflow/deepseek-v3.2（硅基流动直供）", "第三方直供、无缓存价"),
]

rows = []
for prov, model, note in DEEPSEEK_ROWS:
    s, cmp_, _ = score.get((prov, model), (None, "未评测", ""))
    src = sim.loc[(sim["提供商"] == prov) & (sim["模型"] == model)].iloc[0]
    payg_cost = float(src["月总成本"])
    drop = f"AA {s:.0f} 分，与 3.8-flash 断层（差 {ANCHOR - s:.0f}）" if pd.notna(s) and ANCHOR - s >= 10 else ""
    rows.append(dict(提供商=prov, 型号=model, 模式="按量", 月估价=round(payg_cost, 1),
                     AA分数=s if pd.notna(s) else None, 分数口径=cmp_,
                     每智能指点月成本=round(payg_cost / s, 1) if pd.notna(s) else None,
                     入榜=("否" if drop else "是"), 剔除理由=drop, 备注=note))
    # 分时计价杠杆：峰谷型（闲时基准价）与时段折扣型（五折档价）各列一条「全量挪入低价时段」口径
    span = str(src["计价口径"])
    if "峰谷" in span or "时段均匀" in span:
        # 峰谷行的表列价即闲时基准；SF 分时行的表列价是常规时段价，低价时段为其 5 折
        factor = 0.5 if "时段均匀" in span else 1.0
        off = factor * (float(src["输入单价"]) * U_NC + float(src["缓存单价"]) * U_C + float(src["输出单价"]) * U_O)
        rows.append(dict(提供商=prov, 型号=model, 模式="按量·全量挪入低价时段", 月估价=round(off, 1),
                         AA分数=s if pd.notna(s) else None, 分数口径=cmp_,
                         每智能指点月成本=round(off / s, 1) if pd.notna(s) else None,
                         入榜=("否" if drop else "是"), 剔除理由=drop,
                         备注=f"全部负载压进低价时段后 {src['计价口径']}；表列价 {src['输入单价']}/{src['缓存单价']}/{src['输出单价']}"
                              f"{'（低价时段为其 5 折）' if factor == 0.5 else '（即闲时基准价）'}"))
    if (prov, model) in equiv.index:
        eq, enote = equiv.loc[(prov, model), ["订阅等效月价", "备注"]]
        if pd.notna(eq):
            rows.append(dict(提供商=prov, 型号=model, 模式="订阅等效", 月估价=round(float(eq), 1),
                             AA分数=s if pd.notna(s) else None, 分数口径=cmp_,
                             每智能指点月成本=round(float(eq) / s, 1) if pd.notna(s) else None,
                             入榜=("否" if drop else "是"), 剔除理由=drop,
                             备注=("方舟 Coding Plan：k>1 为溢价，封顶 200 元/月｜" if prov == "火山方舟" else "百炼 Token Plan｜")
                                  + str(enote).split("｜")[-1]))

df = pd.DataFrame(rows)
df["排序键"] = df["月估价"].fillna(float("inf"))
df = df.sort_values(["入榜", "排序键"], ascending=[True, True]).reset_index(drop=True)
df.insert(0, "排名", range(1, len(df) + 1))
df = df.drop(columns="排序键")
df.insert(1, "核实日期", "2026-10-12")
df.to_csv(OUT_DIR / "选型_DeepSeek_成本排名.csv", index=False, encoding="utf-8-sig")

live = df[df["入榜"] == "是"]
print(live[["排名", "提供商", "型号", "模式", "月估价", "AA分数", "每智能指点月成本", "备注"]].to_string(index=False))
print(f"\n入榜 {len(live)} 项 / 剔除 {len(df) - len(live)} 项，已写入 选型_DeepSeek_成本排名.csv")
