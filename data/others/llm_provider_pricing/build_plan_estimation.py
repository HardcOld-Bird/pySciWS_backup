# %% 订阅等效按量价模拟（用户建模：渠道统一「套餐定价系数 k」；等效价 = 按量模拟价 × k，封顶套餐价）
# k 锚定方式（透明度递减，各渠道单一 k 存于 KS）：
#   实测锚点（千问/百炼 Pro 499：qwen3.8-max 15 天用量恰耗 180,000 Credits → 月需 2 份 → k=998/4117）
#   公式锚定（智谱 Max 1078：积分系数(6.9/1.7/24)/按量价(8/2/28) 结构均匀 → k=0.247）
#   逐模型公式（MiMo Pro 329：池 380亿 Credits÷模型扣减率 → 每模型独立 k，KS 记均值）
#   代表模型锚定池（Kimi Allegro 699 池1600M→锚 k2.6；方舟 Pro 200 池1249M→锚 doubao-mini；
#     MiniMax Build 469（12x）→锚 M3 且池≈本用量：三者 k≈套餐价/该锚模型按量价）
# 覆盖修正：等效价>套餐价 → 记套餐价并标注「池不足，需+按量补足」；k 为需求侧折扣，池撑不满贵模型时失真。
from pathlib import Path
import sys
import pandas as pd
from pysci.paths import assert_within_data

OUT_DIR = Path(__file__).resolve().parent
assert_within_data(OUT_DIR)
sys.path.insert(0, str(OUT_DIR))
from build_usage_table import USAGE

payg = pd.read_csv(OUT_DIR / "模拟月成本_按量.csv")
sim = payg.set_index(["提供商", "模型"])["月总成本"].to_dict()

u = pd.DataFrame(USAGE, columns=["日期", "未缓存输入", "缓存输入", "输出"])
MONTH = 30 / len(u)
nc, cch, out = (u[c].sum() * MONTH / 1e6 for c in ["未缓存输入", "缓存输入", "输出"])
A_M = nc + cch + out

# 实测锚点修正（2026-10-12）：15 天用量耗完一份 Pro 的 180,000 Credits，故月用量 = 2 份 Pro = 998 元；
# k = 998 / 4117 = 0.2424。旧值 499/4117 隐含「一份覆盖整月」，与锚点矛盾，会把阿里云系低估一倍。
K_WEN = 2 * 499 / sim[("阿里云百炼", "qwen3.8-max")]                # 0.2424（实测锚点：月需 2 份 Pro）
K_ZH = 1078 / sim[("智谱", "GLM-5.3")]                              # 0.2472（公式锚定，结构均匀）
K_KIMI = 699 / sim[("Kimi", "kimi-k2.6")]                           # 0.2501（池1600M≈锚模型月需量，自洽）
K_ARK = 200 / sim[("火山方舟", "doubao-seed-2.0-mini（≤32K）")]      # 1.7544（池1249M=代表模型口径，封顶制）
K_MM = 469 / sim[("MiniMax", "MiniMax-M3（输入≤512K）")]             # 0.4698（Build 12x≈本用量，自参照）
# MiMo：扣减表仅公布 V2.5 系（缓存2/未命中100/输出200 Credits/token）。本用量 V2.5 月耗
# = 101.6e6×100+1553.2e6×2+15.8e6×200 = 164.4 亿 Credits；Pro 池 380 亿 → 可支撑 2.31 个月
# → 月等效价 = 329/2.31 = 142.4 元 → k = 142.4/1375.2（V2.5 按量模拟价）
MIMO_CRED = (nc * 100 + cch * 2 + out * 200) / 100  # 亿 Credits
K_MIMO = (329 * MIMO_CRED / 380) / sim[("小米MiMo", "MiMo-V2.5（输入<256K）")]

KS = {  # 提供商: (k, 套餐价, 锚定说明)
    "阿里云百炼": (K_WEN, 499, "实测锚点：15 天用量耗完一份 Pro 的 180,000 Credits，月需 2 份=998（刊例 600×2=1200 时 k=0.2915）"),
    "千问AI平台": (K_WEN, 499, "同百炼"),
    "智谱": (K_ZH, 1078, "公式锚定 GLM-5.3；积分公式跨模型近似均匀；周限速使贵模型失真"),
    "Kimi": (K_KIMI, 699, "池 1600M 锚定 k2.6；池计量口径（全/非缓存）不明，k2.6 两口径下自洽"),
    "MiniMax": (K_MM, 469, "自参照：Build 12x 假设≈本用量（置信低）"),
    "小米MiMo": (K_MIMO, 329, f"扣减表锚定 V2.5（本用量 {MIMO_CRED:.1f} 亿 Credits/月 vs Pro 池 380 亿，撑 2.31 月）；V2.6 系无扣减表按同 k 外推"),
    "火山方舟": (K_ARK, 200, "池 1249M 锚定 doubao-mini（官方实测参考点）→ 封顶制：多数模型等效=200"),
}

# 多份购买修正（用户 2026-10-12 决策）：Token Plan 类额度池可经多账户购买 N 份解除单池上限，
#   订阅等效月价 = cost×k（线性，不封顶），N = max(1, ceil(cost×k/套餐价))。
#   例：qwen3.8-max 需 2 份 Pro ≈998 元。N>1 时按 token 计对便宜模型可能高估份数（备注标明）。
# Coding Plan 类（智谱/Kimi/方舟/MiniMax）限速或难以多账户切换 → 不修正，封顶信息保留在备注。
MULTIBUY = {"阿里云百炼", "千问AI平台", "小米MiMo"}  # Token Plan 池制，允许多份

recs = []
for (prov, model), cost in sim.items():
    if prov not in KS:
        recs.append(dict(提供商=prov, 模型=model, 按量模拟月价=cost, k=None, 订阅等效月价=None,
                         需购份数N=None, 备注="无订阅套餐（按量渠道）"))
        continue
    k, cap, how = KS[prov]
    equiv = cost * k
    if prov in MULTIBUY:
        n = max(1, -(-equiv // cap))  # ceil
        note = how + (f"｜多份购买：N={n:.0f} 份（cost×k={equiv:.0f}>套餐 {cap}）；按量贵的模型以 token 计 N 或更低" if n > 1 else "")
        recs.append(dict(提供商=prov, 模型=model, 按量模拟月价=cost, k=round(k, 4),
                         订阅等效月价=round(equiv, 1), 需购份数N=int(n), 备注=note))
    else:
        note = how + (f"｜单池上限 {cap} 元等效，cost×k={equiv:.0f} 超出——Coding Plan 限速制不修正，"
                      f"实际为套餐跑满+超池按量补足" if equiv > cap else "")
        recs.append(dict(提供商=prov, 模型=model, 按量模拟月价=cost, k=round(k, 4),
                         订阅等效月价=round(min(equiv, cap), 1) if equiv > cap else round(equiv, 1),
                         需购份数N=None, 备注=note))

df = pd.DataFrame(recs).sort_values("订阅等效月价", na_position="last").reset_index(drop=True)
df.insert(0, "排名", df.index + 1)
df.insert(1, "核实日期", "2026-10-12")
df.to_csv(OUT_DIR / "模拟月成本_订阅等效.csv", index=False, encoding="utf-8-sig")

print(f"月化口径A = {A_M:,.1f}M tokens\n")
print("渠道套餐定价系数 k（=订阅等效价/按量模拟价）：")
for prov, (k, cap, how) in KS.items():
    print(f"  {prov:10s} k={k:.4f}  套餐 {cap} 元｜{how}")
print()
pd.set_option("display.max_colwidth", 46)
print(df.head(18)[["排名", "提供商", "模型", "按量模拟月价", "k", "订阅等效月价"]].to_string(index=False))
print(f"\n共 {len(df)} 行，已写入 {OUT_DIR / '模拟月成本_订阅等效.csv'}")
