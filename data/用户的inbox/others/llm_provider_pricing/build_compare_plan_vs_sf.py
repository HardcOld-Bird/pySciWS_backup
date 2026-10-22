# %% 对比：百炼/千问 Token Plan 上的「非 Qwen 系」型号订阅等效价 vs SiliconFlow 按量价
# 场景 A（本表主口径）：k 乘在百炼自己的刊例价上——即假定 Token Plan 按百炼直供标价扣 Credits。
# 场景 B（乐观上界）：k 乘在该型号的原厂/SF 价上——假定 Token Plan 对第三方直供按原厂价折算，
#   不吸收百炼的直供溢价。两场景差距 = 百炼直供相对原厂的渠道溢价倍数。
# 前提存疑项：Token Plan 覆盖模型写的是「deepseek-v4、kimi、glm」，未说明是阿里托管版还是
#   vanchin/ZHIPU 直供版；若直供版根本不在池内，则本表两列都不成立，只能原厂渠道购买。
from pathlib import Path

import pandas as pd

from pysci.paths import assert_within_data

OUT_DIR = Path(__file__).resolve().parent
assert_within_data(OUT_DIR)

sim = pd.read_csv(OUT_DIR / "模拟月成本_按量.csv")
plan = pd.read_csv(OUT_DIR / "模拟月成本_订阅等效.csv")
COST = sim.groupby(["提供商", "模型"])["月总成本"].min().to_dict()
k_ali = float(plan.loc[(plan["提供商"] == "阿里云百炼") & (plan["模型"] == "qwen3.8-max"), "k"].iloc[0])

# (百炼行, SF 对应行, 原厂/最廉渠道行, 说明)
PAIRS = [
    (("阿里云百炼", "ZHIPU/GLM-5.3（智谱直供）"), ("SiliconFlow", "GLM-5.3 / GLM-5.2"),
     ("智谱", "GLM-5.3"), "百炼直供与智谱/SF 同价，无渠道溢价"),
    (("阿里云百炼", "kimi-k3"), None, ("Kimi", "kimi-k3"),
     "SF 未上架 K3；百炼直供无缓存价，故标价 3.5× 原厂"),
    (("阿里云百炼", "kimi-k2.7-code"), ("SiliconFlow", "Kimi-K2.7-Code"),
     ("Kimi", "kimi-k2.7-code"), "百炼直供无缓存价，SF 有 1.3"),
    (("阿里云百炼", "kimi-k2.5"), None, ("Kimi", "kimi-k2.5"),
     "SF 未上架 K2.5；百炼直供无缓存价"),
    (("阿里云百炼", "vanchin/deepseek-v4-pro（万擎直供）"), ("SiliconFlow", "DeepSeek-V4-Pro"),
     ("DeepSeek", "deepseek-v4-pro 闲时"), "万擎直供 12/–/24 且无缓存价，原厂闲时 4.5/0.15/13.5"),
    (("阿里云百炼", "vanchin/deepseek-v4.1-flash（万擎直供）"), ("SiliconFlow", "DeepSeek-V4-Flash"),
     ("DeepSeek", "deepseek-flash（V4.1-Flash）闲时"), "万擎直供 2/–/8 无缓存价，原厂闲时 1.0/0.02/4.0"),
    (("阿里云百炼", "siliconflow/deepseek-v3.2（硅基流动直供）"),
     ("SiliconFlow", "DeepSeek-V3.2 / V3.2 (Pro)"), None,
     "同一模型的 SF 自营价 4.0/0.4/6.0，百炼转售反而标价更低（2.0/–/3.0）"),
]

rows = []
for bailian, sf, native, note in PAIRS:
    bl_cost = COST[bailian]
    sf_cost = COST[sf] if sf else None
    nv_cost = COST[native] if native else None
    ref = min(x for x in (sf_cost, nv_cost) if x is not None)
    rows.append(dict(
        型号=bailian[1],
        百炼按量=round(bl_cost, 1),
        订阅等效A_乘百炼价=round(bl_cost * k_ali, 1),
        SF按量=round(sf_cost, 1) if sf_cost else None,
        原厂按量=round(nv_cost, 1) if nv_cost else None,
        订阅等效B_乘原厂价=round(ref * k_ali, 1),
        A比SF=("A 更贵" if sf_cost and bl_cost * k_ali > sf_cost else "A 更便宜" if sf_cost else "SF 无此型号"),
        A比原厂=("A 比原厂按量贵" if bl_cost * k_ali > nv_cost else "A 比原厂按量便宜") if nv_cost else "—",
        B比SF=("B 更便宜" if sf_cost and ref * k_ali < sf_cost else "B 更贵" if sf_cost else "SF 无此型号"),
        渠道溢价倍数=round(bl_cost / ref, 1),
        备注=note,
    ))

df = pd.DataFrame(rows)
df.insert(0, "核实日期", "2026-10-12")
df.insert(1, "k(阿里云系)", k_ali)
df.to_csv(OUT_DIR / "对比_非Qwen_订阅等效_vs_SF按量.csv", index=False, encoding="utf-8-sig")
print(df.drop(columns=["k(阿里云系)", "备注"]).to_string(index=False))
print()
print(df[["型号", "备注"]].to_string(index=False))
