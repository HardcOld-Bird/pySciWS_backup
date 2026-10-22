# %% 压测方案成本：MiMo Max 订阅（已购）vs 其他可购模型（2026-10-13，组长代算）
# 问题：压测（Token 消耗比例＝日常 Agent 情景比例）下，是否有比「最高档 MiMo Token Plan 里的
#       mimo-v2.6-flash」更低成本的模型？不考虑免费模型（限速不可接受）。
# 订阅事实（用户实测＋官方扣减表）：
#   MiMo Token Plan Max 年付 6,959.04 元/年 → 月均 579.92 元，820 亿 Credits/月。
#   官方扣减表（per token，非 per M）：
#     mimo-v2.6-flash  缓存命中 2 / 未命中 100 / 输出 200
#     mimo-v2.6-pro    缓存命中 2.5 / 未命中 300 / 输出 600
#   V2.5 与 V2.6 共用额度；非高峰 0.8x 折扣（本文不计，保守口径）。
# 两个口径必须分开（否则会把沉没成本当成选择依据）：
#   单位成本＝按量等价（元/M tokens），跨方案可比；
#   现金支出＝订阅按已消耗 Credits 分摊 / 固定费套餐按档位月费 / 按量按实际消耗。
# 单一事实来源：用量比例与各模型单价读自生成的 CSV；本文件只存扣减系数、订阅额度与档位。
from pathlib import Path
import pandas as pd
from pysci.paths import assert_within_data

OUT_DIR = Path(__file__).resolve().parent
assert_within_data(OUT_DIR)

U_VOL = 1670.6      # 个人月化总量（百万 tokens，15 天实测×30/15）
STRESS_SHARE = 0.25  # 压测份额（组长定义 50/25/25）
U_STRESS = round(U_VOL * STRESS_SHARE, 1)
AG = {"nc": 0.06086, "c": 0.92970, "o": 0.00945}   # 实测 Agent 结构（未缓存/缓存/输出）

# %% MiMo Max 订阅的 Credits 换算（per token 扣减 → 单位 Credits/M tokens）
CRED = {"mimo-v2.6-flash": {"nc": 100.0, "c": 2.0, "o": 200.0},
        "mimo-v2.6-pro": {"nc": 300.0, "c": 2.5, "o": 600.0}}
MAX_PRICE = 579.92          # 元/月（年付 6,959.04÷12）
MAX_CREDITS = 820e8         # Credits/月

cred_per_M = {m: (AG["nc"] * d["nc"] + AG["c"] * d["c"] + AG["o"] * d["o"]) * 1e6  # per token → per M tokens
              for m, d in CRED.items()}
CRED_YUAN = MAX_PRICE / MAX_CREDITS                        # 元/Credit
mimo_plan_unit = {m: v * CRED_YUAN for m, v in cred_per_M.items()}  # 元/M tokens
mimo_full_cred = U_VOL * cred_per_M["mimo-v2.6-flash"] / 1e8       # 全量需 Credits（亿/月）

print("== MiMo Max 订阅换算 ==")
print(f"  Max：{MAX_PRICE} 元/月 → {MAX_CREDITS/1e8:.0f} 亿 Credits/月；单价 {CRED_YUAN:.3e} 元/Credit")
for m, v in cred_per_M.items():
    print(f"  {m:<16} 本比例消耗 {v:.4g} Credits/M → 订阅单位价 {mimo_plan_unit[m]:.4f} 元/M")
print(f"  按此比例，全量 {U_VOL:.0f}M 需 {mimo_full_cred:.1f} 亿 Credits＝额度的 {mimo_full_cred*1e8/MAX_CREDITS:.1%}；"
      f"压测份额 {U_STRESS:.0f}M 需 {U_STRESS * cred_per_M['mimo-v2.6-flash'] / 1e8:.1f} 亿 Credits＝"
      f"{U_STRESS * cred_per_M['mimo-v2.6-flash'] / MAX_CREDITS:.1%}")

# %% 竞争方案（不考虑免费）。固定费套餐必须检查容量是否装得下 U_STRESS。
FIXED = [  # (提供方, 方案, 单位成本来源, 月费, 容量_M, 容量口径)
    ("火山方舟", "Coding Plan Lite（限时）", 0.0396, 9.9, 250.0, "实测约 250M tokens/月，按次数计量"),
    ("火山方舟", "Coding Plan Pro（限时）", 0.0396, 49.9, 1249.0, "实测约 1,249M tokens/月"),
    ("火山方舟", "Coding Plan Lite（刊例）", 0.0396, 40.0, 250.0, "同上"),
    ("火山方舟", "Agent Plan Small（doubao-2.0-mini AFP）", 0.0500, 40.0, 20000.0, "2 万 AFP；容量为 AFP 折算 token 数，见注"),
]
PAYG = [  # (提供方, 模型, 单位成本 元/M)  ← 读自压测表（agent 口径）
    ("小米MiMo", "MiMo-V2.6-Flash（按量）", 0.0983),
    ("火山方舟", "doubao-seed-2.0-mini（≤32K 按量）", 0.0682),
    ("火山方舟", "doubao-seed-2.0-mini（32-128K 档）", 0.1365),
    ("火山方舟", "doubao-seed-2.0-lite（≤32K 按量）", 0.1821),
    ("DeepSeek", "deepseek-flash 闲时（按量）", 0.1171),
    ("智谱", "GLM-5.3-Flash（限时五折按量）", 0.1445),
]

rows = []
base = mimo_plan_unit["mimo-v2.6-flash"]
rows.append(dict(方案="MiMo Max 订阅（已购）· mimo-v2.6-flash", 类型="订阅已购",
                 单位成本=round(base, 4), 比订阅贵倍=1.00, 压测月量M=U_STRESS,
                 现金支出=round(base * U_STRESS, 1), 装得下="是（5% 额度）",
                 需新开充值点="否（沉没）", 备注="分摊口径；订阅本身已是既付成本，压测边际现金支出≈0"))
rows.append(dict(方案="MiMo Max 订阅·mimo-v2.6-flash（非高峰 0.8x）", 类型="订阅已购",
                 单位成本=round(base * 0.8, 4), 比订阅贵倍=round(0.8, 3), 压测月量M=U_STRESS,
                 现金支出=round(base * 0.8 * U_STRESS, 1), 装得下="是（额度仅需 4%）",
                 需新开充值点="否（沉没）",
                 备注="官方「非高峰 0.8x 消耗」规则；压测本身无交互时延压力 → 把重跑排到非高峰即免费拿 20%"))
for prov, name, unit, fee, cap, capnote in FIXED:
    fits = cap >= U_STRESS
    rows.append(dict(方案=f"{prov}·{name}", 类型="固定费套餐",
                     单位成本=unit, 比订阅贵倍=round(unit / base, 3),
                     压测月量M=U_STRESS, 现金支出=fee if fits else None,
                     装得下=("是" if fits else f"否（容量 {cap:g}M < {U_STRESS}M）"),
                     需新开充值点="是", 备注=capnote))
for prov, name, unit in PAYG:
    rows.append(dict(方案=f"{prov}·{name}", 类型="按量",
                     单位成本=unit, 比订阅贵倍=round(unit / base, 3),
                     压测月量M=U_STRESS, 现金支出=round(unit * U_STRESS, 1),
                     装得下="是（无上限）", 需新开充值点=("否（已按需使用）" if prov == "小米MiMo" else "是"),
                     备注="无最低消费"))
df = pd.DataFrame(rows).sort_values("单位成本").reset_index(drop=True)
df.insert(0, "排名", range(1, len(df) + 1))
df.insert(1, "核实日期", "2026-10-13")
df.to_csv(OUT_DIR / "选型_压测_订阅vs竞品.csv", index=False, encoding="utf-8-sig")

print(f"\n== 压测 {U_STRESS}M tokens/月（占个人总量 25%）==")
print(df[["排名", "方案", "类型", "单位成本", "比订阅贵倍", "现金支出", "装得下", "需新开充值点"]]
      .to_string(index=False, max_colwidth=38))
print("\n注：方舟 Agent Plan 的容量是 AFP 抵扣值折算（doubao-2.0-mini 系数 0.25），公式无缓存项 →"
      "对 93% 缓存的负载实为偏保守估计；Coding Plan「实测容量」是第三方经验均值，非承诺值。")
print(f"输出：选型_压测_订阅vs竞品.csv")
