# %% 选型第一步：主力候选 qwen3.8-flash 跨渠道×跨模式月估价排名（用量口径同前）
# qwen3.8-flash 为阿里系专有，实际存在渠道 = 阿里云百炼 / 千问AI平台（同源同价）；
# 另列 SiliconFlow「Qwen3.8-27B 开源版」作最近似开源替代参考（非同模型，能力不等价）。
# 模式维度：按量（隐式/显式缓存）、按量 Batch 半价、订阅等效（k=0.1212，多份制）。
from pathlib import Path
import pandas as pd
from pysci.paths import assert_within_data

OUT_DIR = Path(__file__).resolve().parent
assert_within_data(OUT_DIR)

sim = pd.read_csv(OUT_DIR / "模拟月成本_按量.csv").set_index(["提供商", "模型"])["月总成本"].to_dict()
sub = pd.read_csv(OUT_DIR / "模拟月成本_订阅等效.csv")
K_WEN = float(sub[sub.模型 == "qwen3.8-flash"]["k"].iloc[0])
CAP = 499.0

U_nc, U_c, U_o = 101.573, 1553.231, 15.785  # 月化百万 tokens（build_cost_simulation 同口径）

rows = []
def add(prov, mode, i, c, o, note, equiv=None, n=None):
    cost = U_nc * i + U_c * c + U_o * o
    rows.append(dict(提供商=prov, 模式=mode, 输入=i, 缓存=c, 输出=o,
                     月估价=round(cost, 1) if equiv is None else round(equiv, 1),
                     需购份数=n, 备注=note))

for prov in ("阿里云百炼", "千问AI平台"):
    add(prov, "按量·隐式缓存", 0.8, 0.1, 2.7, "官方价目 0.8/0.1/2.7")
    add(prov, "按量·Batch 半价", 0.4, 0.4, 1.35, "离线批处理，缓存不适用→按未命中价折半；须可异步",
        )
add("SiliconFlow", "按量（近似替代 Qwen3.8-27B 开源版）", 3.0, 3.0, 12.0,
    "无缓存价→缓存按输入全价（保守）；开源权重版，与商用 3.8-flash 能力不等价")

# 订阅等效（仅阿里系有）
for prov in ("阿里云百炼", "千问AI平台"):
    payg_cost = sim[(prov, "qwen3.8-flash")]
    equiv = payg_cost * K_WEN
    add(prov, f"订阅 Token Plan Pro 等效（k={K_WEN}）", 0.8 * K_WEN, 0.1 * K_WEN, 2.7 * K_WEN,
        "线性外推，乐观口径（Credits↔人民币线性）", equiv=equiv, n=max(1, -(-equiv // CAP)))

df = pd.DataFrame(rows).sort_values("月估价").reset_index(drop=True)
df.insert(0, "排名", df.index + 1)
df.insert(1, "核实日期", "2026-10-12")
df.to_csv(OUT_DIR / "选型_flash_月估价排名.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.max_colwidth", 52)
print(df[["排名", "提供商", "模式", "月估价", "需购份数", "备注"]].to_string(index=False))
print(f"\n月化用量：未缓存 {U_nc}M / 缓存 {U_c}M / 输出 {U_o}M")
print(f"已写入 {OUT_DIR / '选型_flash_月估价排名.csv'}")
