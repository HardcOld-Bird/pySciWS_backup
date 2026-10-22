# %% 模拟月成本：个人 15 天用量 × 各提供商按量刊例价（2026-10-12，组长代算）
# 假定（与用户确认前采用的缺省，均可在 CSV 的「计价口径」列追溯）：
#  1. 月化 = 15 天实测合计 × 30/15；结构比例（未缓存/缓存/输出 = 6.09%/93.02%/0.94%）保持不变。
#  2. 分时段/峰谷计费 → 用量均匀分布 24h 取时均有效价；「全天24h峰谷对半」的翻倍类（DeepSeek 原厂、
#     方舟 deepseek 系）按厂商规则峰谷各半。
#  3. 无缓存价或表标「-」的模型 → 缓存 token 按输入全价计费（保守口径）。
#  4. 阶梯计费取低档（多数 Agent 单次请求 <32K/512K），高档以「（高档）」后缀并行另算一行。
#  5. 价格取刊例价，限时折扣不叠加（qwen3.8-max 缓存用隐式命中 1.5，显式 1.0 记入口径注）。
from pathlib import Path
import sys
import pandas as pd
from pysci.paths import assert_within_data

OUT_DIR = Path(__file__).resolve().parent
assert_within_data(OUT_DIR)
sys.path.insert(0, str(OUT_DIR))
from build_pricing_tables import PAYG
from build_usage_table import USAGE

usage = pd.DataFrame(USAGE, columns=["日期", "未缓存输入", "缓存输入", "输出"])
MONTH = 30 / len(usage)
U_nc = usage["未缓存输入"].sum() * MONTH / 1e6  # 百万 tokens
U_c = usage["缓存输入"].sum() * MONTH / 1e6
U_o = usage["输出"].sum() * MONTH / 1e6
print(f"月化用量（百万tokens）：未缓存输入 {U_nc:.1f} / 缓存输入 {U_c:.1f} / 输出 {U_o:.1f}\n")

# 峰谷对半（全天恰好一半时间高峰，价 ×1.5）
PEAK_BAL = {("DeepSeek", "deepseek-flash（V4.1-Flash）闲时"),
            ("DeepSeek", "deepseek-v4-pro 闲时"),
            ("火山方舟", "deepseek-v4.1-flash（方舟渠道）闲时")}
# 时段覆盖比例：主列价所占小时数 → 有效倍率 = 占比 + (1-占比)×折扣
TIME_FRACTION = {  # (主列占时, 另一时段倍率)
    ("SiliconFlow", "DeepSeek-V4-Flash"): (8 / 24, 0.5),    # 2-8点 5折：1.5/0.15/4.5
    ("SiliconFlow", "Hunyuan-A13B-Instruct"): (9 / 24, 0.8),  # 9-18点 1.0/4.0；其余 0.8/3.2（×0.8）
}
# 显式另算行：(提供商, 原模型名, 新档位名, 输入, 缓存, 输出, 口径注)
EXTRA_ROWS = [
    ("智谱", "GLM-5.1（输入<32K / ≥32K）", "（高档）", 8.0, 2.0, 28.0, "输入≥32K 档"),
    ("智谱", "GLM-5-Turbo（输入<32K）", "（高档）", 7.0, 1.8, 26.0, "输入≥32K 档"),
    ("智谱", "GLM-5（输入<32K）", "（高档）", 6.0, 1.5, 22.0, "输入≥32K 档"),
    ("智谱", "GLM-4.7（输入<32K，输出<0.2K）", "（高档）", 4.0, 0.8, 16.0, "输入[32K,200K) 档"),
    ("MiniMax", "MiniMax-M3（输入≤512K）", "（>512K档）", 4.20, 0.84, 16.80, "整次请求>512K 取高档"),
    ("火山方舟", "deepseek-v4-pro 正式版（方舟渠道）", "（低优/批量）", 4.5, 0.30, 13.5, "低优/批量渠道"),
    ("火山方舟", "deepseek-v4-flash 正式版（方舟渠道）", "（低优/批量）", 1.5, 0.10, 4.5, "低优/批量渠道"),
    # 火山方舟自研豆包系的阶梯档：整次请求按输入 token 取档，跨 32K/128K 即跳档（缓存价与输出价同步翻倍）
    ("火山方舟", "doubao-seed-2.0-mini（≤32K）", "（32-128K档）", 0.4, 0.08, 4.0, "整次请求输入∈(32K,128K]"),
    ("火山方舟", "doubao-seed-2.0-mini（≤32K）", "（>128K档）", 0.8, 0.16, 8.0, "整次请求输入>128K"),
    ("火山方舟", "doubao-seed-2.0-lite（≤32K）", "（32-128K档）", 0.9, 0.18, 5.4, "整次请求输入∈(32K,128K]"),
    ("火山方舟", "doubao-seed-2.0-lite（≤32K）", "（>128K档）", 1.8, 0.36, 10.8, "整次请求输入>128K"),
    ("火山方舟", "doubao-seed-2.0-pro（≤32K / 32-128K / 128-256K）", "（32-128K档）", 4.8, 0.96, 24.0, "整次请求输入∈(32K,128K]"),
    ("火山方舟", "doubao-seed-2.0-pro（≤32K / 32-128K / 128-256K）", "（>128K档）", 9.6, 1.92, 48.0, "整次请求输入>128K"),
]

records = []
for row in PAYG:
    key = (row["提供商"], row["模型"])
    if row["输入未命中"] is None:
        continue
    p_in, p_c, p_o = row["输入未命中"], row["缓存命中"], row["输出"]
    notes = []
    mult = 1.0
    if key in PEAK_BAL:
        mult = 1.5
        notes.append("峰谷均匀→×1.5")
    if key in TIME_FRACTION:
        frac, other = TIME_FRACTION[key]
        mult = frac + (1 - frac) * other
        notes.append(f"时段均匀→有效×{mult:.3f}")
    cache_note = ""
    if p_c is None:
        c_eff_in = (p_in or 0) + (p_o or 0) * 0  # 缓存按输入全价
        cost_c = U_c * (p_in or 0) * mult
        cache_note = "无缓存价→缓存按输入全价"
    else:
        cost_c = U_c * p_c * mult
    cost_nc = U_nc * p_in * mult
    cost_o = U_o * (p_o if p_o is not None else 0) * mult
    if mult != 1.0:
        cost_c *= 1  # mult 已分别乘入各分项
    records.append(dict(
        提供商=row["提供商"], 模型=row["模型"],
        输入单价=p_in, 缓存单价=(p_c if p_c is not None else p_in), 输出单价=p_o,
        月成本_未缓存输入=round(cost_nc, 1), 月成本_缓存=round(cost_c, 1),
        月成本_输出=round(cost_o, 1), 月总成本=round(cost_nc + cost_c + cost_o, 1),
        计价口径="；".join(notes + ([cache_note] if cache_note else [])) or
                 "刊例低档；缓存用表列命中价" + ("（显式命中更低，见价目表）" if "显式" in str(row.get("其他维度", "")) else ""),
    ))

for prov, base_name, tag, i, c, o, note in EXTRA_ROWS:
    src = next(r for r in records if r["提供商"] == prov and r["模型"] == base_name)
    cost_nc, cost_c, cost_o = U_nc * i, U_c * c, U_o * o
    records.append(dict(src, 模型=base_name.replace("（输入<32K / ≥32K）", "") + tag,
                        输入单价=i, 缓存单价=c, 输出单价=o,
                        月成本_未缓存输入=round(cost_nc, 1), 月成本_缓存=round(cost_c, 1),
                        月成本_输出=round(cost_o, 1), 月总成本=round(cost_nc + cost_c + cost_o, 1),
                        计价口径=note))

df = pd.DataFrame(records).drop_duplicates(subset=["提供商", "模型"])
df = df.sort_values("月总成本").reset_index(drop=True)
df.insert(0, "成本排名", df.index + 1)
df.insert(1, "核实日期", "2026-10-12")
df.to_csv(OUT_DIR / "模拟月成本_按量.csv", index=False, encoding="utf-8-sig")

print(df[["成本排名", "提供商", "模型", "月总成本", "计价口径"]].to_string(index=False))
print(f"\n共 {len(df)} 行，已写入 {OUT_DIR / '模拟月成本_按量.csv'}")
