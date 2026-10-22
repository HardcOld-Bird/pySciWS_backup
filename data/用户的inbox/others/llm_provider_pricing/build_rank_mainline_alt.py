# %% 主力模型替代选型（排除 Qwen，只用各家官方渠道＋自家模型，2026-10-12，组长代算）
# 用例性质：主力 Agent 负载——单模型、大量、高缓存命中（实测结构 6.1%/93.0%/0.9%），
#   份额＝个人月化总量 1,670.6M 的 50% ＝ 835.3M tokens/月。
# 排序：按**主力单位价（元/百万 tokens，agent 口径）从低到高**＝推荐优先级序（组长口述「价格降序」按此理解；
#   若要价格由高到低，把 sort_values 的 ascending 改成 False 即可）。
# 性能口径（组长补充）：不明显弱于 qwen3.8-flash（AA 40）与 deepseek-v4.1-flash（AA 39）即可，**允许更强**；
#   更强的型号因价格自然靠后，仍纳入统计作参考。AA 缺失的型号不剔除、只标注（遵早先「评分仅做参考」指示）。
# 订阅档另算：容量、限速与工具白名单条款决定「能不能当主力」，故与按量分列，并显式列出排除原因。
# 单一事实来源：按量单价、AA 分、上下文与套餐月费一律读自生成的 CSV；本文件只存选型判断与折算口径。
from pathlib import Path
import pandas as pd
from pysci.paths import assert_within_data

OUT_DIR = Path(__file__).resolve().parent
assert_within_data(OUT_DIR)

BASE = {"qwen3.8-flash": 40.0, "deepseek-v4.1-flash": 39.0}   # 性能基准（组长给定的两款）
FLOOR = 34.0        # 「不明显弱于」下界：AA < 34 视为明显弱，单列而不进推荐
VOL = 835.3         # 主力月量（百万 tokens）
AG = (0.0609, 0.9302, 0.0094)       # 实测主力负载结构（缓存命中率 93.9%）
LOW_HIT = (0.4953, 0.4953, 0.0094)  # 命中率腰斩到 50% 的同一负载（压测/harness 改动会移动它 → 稳健性检验）

cat = pd.read_csv(OUT_DIR / "按量付费_模型价目.csv")
perf = pd.read_csv(OUT_DIR / "模型性能指数.csv")
sub = pd.read_csv(OUT_DIR / "订阅套餐_对比.csv")
sc = perf.set_index(["提供商", "模型"])["智能指数"]
cmpv = perf.set_index(["提供商", "模型"])["可比性"]

# %% 候选（提供商, 模型, 家族, 该渠道是否为「自家模型」白名单, 说明）
CANDS = [
    ("小米MiMo", "MiMo-V2.6-Flash", "MiMo"), ("小米MiMo", "MiMo-V2.6-Pro", "MiMo"),
    ("DeepSeek", "deepseek-flash（V4.1-Flash）闲时", "DeepSeek"),
    ("DeepSeek", "deepseek-v4-pro 闲时", "DeepSeek"),
    ("智谱", "GLM-5.3-Flash", "GLM"), ("智谱", "GLM-5.3-Flash 限时五折", "GLM"),
    ("智谱", "GLM-5.3-FlashX", "GLM"), ("智谱", "GLM-5.3", "GLM"), ("智谱", "GLM-5.2", "GLM"),
    ("Kimi", "kimi-k2.7-code", "Kimi"), ("Kimi", "kimi-k2.6", "Kimi"), ("Kimi", "kimi-k3", "Kimi"),
    ("MiniMax", "MiniMax-M2.5 / M2.1 / M2", "MiniMax"), ("MiniMax", "MiniMax-M3（输入≤512K）", "MiniMax"),
    ("阿里云百炼", "qwen3.8-flash", "Qwen（基准，已排除）"), ("阿里云百炼", "qwen3.8-max", "Qwen（基准，已排除）"),
]
# 智谱限时五折价在价目表的「其他维度」里，这里显式成行（判断性数据，不改价目表结构）
SALE = {("智谱", "GLM-5.3-Flash 限时五折"): (0.4, 0.115, 1.4)}


def unit_price(prov, model):
    if (prov, model) in SALE:
        i, c, o = SALE[(prov, model)]
        return round(AG[0] * i + AG[1] * c + AG[2] * o, 4), i, c, o
    r = cat[(cat["提供商"] == prov) & (cat["模型"] == model)]
    if not len(r):
        return None, None, None, None
    r = r.iloc[0]
    i, c, o = r["输入未命中"], r["缓存命中"], r["输出"]
    if pd.isna(i):
        return None, i, c, o
    return round(AG[0] * i + AG[1] * (i if pd.isna(c) else c) + AG[2] * (0 if pd.isna(o) else o), 4), i, c, o


def mix3(i, c, o, sh):
    a, b, e = sh
    return round(a * i + b * (i if c is None or pd.isna(c) else c) + e * (0 if o is None or pd.isna(o) else o), 4)


rows = []
for prov, model, fam in CANDS:
    up, i, c, o = unit_price(prov, model)
    if up is None:
        rows.append(dict(提供商=prov, 型号=model, 家族=fam, 渠道="官方按量", 主力单位价=None,
                         备注="无按量价（仅订阅可用）"))
        continue
    aa = sc.get((prov, model.replace(" 限时五折", "")))
    cp = cmpv.get((prov, model.replace(" 限时五折", "")), "无行")
    low = mix3(i, c, o, LOW_HIT)          # 命中率腰斩后的单位价
    rows.append(dict(
        提供商=prov, 型号=model, 家族=fam, 渠道="官方按量", 主力单位价=up,
        主力月费=round(up * VOL, 0), 输入=i, 缓存=(i if pd.isna(c) else c), 输出=o,
        智能指数=aa, 可比性=cp,
        相对基准=(f"{aa - BASE['qwen3.8-flash']:+.0f} 对 flash" if pd.notna(aa) else "未评测"),
        命中率腰斩单价=low, 命中率敏感度=round(low / up, 1),
        缓存价占比=(None if pd.isna(c) else round(c / i, 3)),
    ))

# %% 订阅档可行性判定（容量能否托住 835M/月、限速、条款）
PLANS = [
    # (提供商, 档位关键词, 家族, 容量说明, 需购份数估算, 判定)
    ("小米MiMo", "Lite", "MiMo", "41 亿 Credits/月。V2.6 扣减表未公布；按「沿用 API 定价」同比例推估 V2.6-Flash "
     "全尺消耗 ≈58.6 亿 Credits → 主力半尺 ≈29.3 亿 ≤ 41 亿，Lite 可托住主力，实付 39 元 vs 按量 82 元＝2.1× 折扣",
     "1 份", "最便宜的主力方案，但依赖两桩核实：① 池是否随 V2.5→V2.6 更名；② V2.6 扣减系数是否同比例"),
    ("Kimi", "Allegro", "Kimi", "约 1,600M tokens/月（699 元）＝0.437 元/M；池含 K3（AA 44）", "1 份", "唯一能用上旗舰 K3 的低价路径，但需预约、长期限购"),
    ("MiniMax", "Build", "MiniMax", "≈本尺 M3 用量（12× Go）＝469 元；主力半尺 M3 按量 ≈250 元 → 几乎不打折",
     "1 份", "无价格优势（M3 已永久五折）"),
    ("智谱", "Pro", "GLM", "约 2.9–5.8 亿 GLM-5.3 tokens/周（95%+ 缓存），12,000 积分/5h 限速",
     "1 份（538 元）＝0.64 元/M，对按量 4.1× 折扣", "排除：官方条款「仅限指定工具」＋5 小时限速对主力高并发是硬节流"),
    ("阿里云百炼", "Lite", "Qwen", "11,500 Credits/月 ≈ 主力 flash 需求 12,200 Credits（差 6%）",
     "Essential 25,500（限时 79 元）", "基准参照行（Qwen 已排除）"),
]
plan_rows = []
for prov, tier, fam, cap, n, verdict in PLANS:
    r = sub[(sub["提供商"] == prov) & (sub["档位"].astype(str).str.contains(tier))]
    fee = float(r["月付刊例"].iloc[0]) if len(r) and pd.notna(r["月付刊例"].iloc[0]) else None
    if len(r) and pd.notna(r["限时价"].iloc[0]):
        fee = min(fee, float(r["限时价"].iloc[0]))
    plan_rows.append(dict(提供商=prov, 档位=tier, 家族=fam, 月费=fee, 容量与折算=cap, 需购=n, 判定=verdict))

df = pd.DataFrame(rows)
live = df[df["主力单位价"].notna() & (df["家族"].str.contains("已排除") == False)].sort_values("主力单位价").reset_index(drop=True)
weak = df[(df["智能指数"] < FLOOR) & df["智能指数"].notna()]
live.insert(0, "排名", range(1, len(live) + 1))
live.insert(1, "核实日期", "2026-10-12")
pd.DataFrame(plan_rows).to_csv(OUT_DIR / "选型_主力_订阅档可行性.csv", index=False, encoding="utf-8-sig")
live.to_csv(OUT_DIR / "选型_主力_非Qwen替代.csv", index=False, encoding="utf-8-sig")

print(f"主力月量 {VOL:.0f}M tokens（个人月化总量 1,670.6M 的 50%）；基准：qwen3.8-flash AA 40、deepseek-v4.1-flash AA 39\n")
cols = ["排名", "家族", "提供商", "型号", "主力单位价", "主力月费", "智能指数", "可比性", "相对基准",
        "命中率腰斩单价", "命中率敏感度"]
print(live[cols].to_string(index=False, max_colwidth=28))
print("\n== 稳健性：命中率腰斩到 50% 后的排序（压测/harness 改动会真的移动这个值）==")
print(live.sort_values("命中率腰斩单价")[["家族", "型号", "主力单位价", "命中率腰斩单价", "命中率敏感度", "智能指数"]]
      .head(8).to_string(index=False, max_colwidth=28))
print(f"\n基准对照：qwen3.8-flash 0.1671 元/M → {round(0.1671 * VOL)} 元/月（Qwen 已排除，仅作价格锚）")
print(f"明显弱于基准（AA<{FLOOR:.0f}，不入推荐）：",
      "; ".join(f"{r['家族']}/{r['型号']} AA {r['智能指数']:.0f}" for _, r in weak.iterrows()) or "无")
print("\n== 订阅档判定 ==")
print(pd.DataFrame(plan_rows)[["提供商", "档位", "家族", "月费", "需购", "判定"]].to_string(index=False, max_colwidth=34))
for _, r in pd.DataFrame(plan_rows).iterrows():
    print(f"  · {r['提供商']}/{r['档位']}：{r['容量与折算']}")
