# %% 多模型仲裁团选型：家族×档级矩阵 + 三口径单位成本 + 成员组合月费（2026-10-12，组长代算）
# 用例性质（组长定义）：并行调用**多家族**性能较强的模型做纯 Chat 咨询（非 Agent），冷启动、短会话。
#   → 与主力/压测反向：这里要的是「强」，且成本由「家族数 × 全价未命中输入」决定，而不是由渠道折扣决定。
# 三口径混合单价（元/百万 tokens，同一价目表、不同负载结构）：
#   agent 口径  0.061/0.930/0.009（实测个人 15 天月化结构，缓存命中 93.9%）
#   冷会话口径  0.65/0.15/0.20（仲裁团缺省：每家族都吃一遍完整 prompt 的未命中输入，短会话只有少量前缀复用）
#   零缓存口径  0.80/0.00/0.20（敏感性上界：完全无前缀复用 → 深缓存折扣型号在此口径失去全部优势）
# 无缓存价的型号 → 缓存 token 按输入全价计（与 模拟月成本_按量.csv 同规则）。
# 单一事实来源：价目/上下文/套餐月费一律读自生成的 CSV；本文件只存判断性数据（家族归属、AFP 系数、负载结构）。
from pathlib import Path
import pandas as pd
from pysci.paths import assert_within_data

OUT_DIR = Path(__file__).resolve().parent
assert_within_data(OUT_DIR)

PROFILES = {"agent口径": (0.0609, 0.9302, 0.0094), "冷会话": (0.65, 0.15, 0.20), "零缓存": (0.80, 0.00, 0.20)}
VOL = 417.65  # 仲裁团月用量（百万 tokens）＝个人月化总量 1670.6 的 25%（组长给定配比 50/25/25）
POOL_DISC = 39 / 131.5  # 千问/百炼 Token Plan 折扣倍率：Lite 39 元 ≈ 131.5 元按量（Credits↔人民币线性假设）

cat = pd.read_csv(OUT_DIR / "按量付费_模型价目.csv")
perf = pd.read_csv(OUT_DIR / "模型性能指数.csv")
u = pd.read_csv(OUT_DIR / "个人每日Token用量.csv")
U_M = float((u[["未缓存输入", "缓存输入", "输出"]].sum() * (30 / len(u)) / 1e6).sum())

cat = cat[cat["输入未命中"].notna()].copy()
for name, (a, b, c) in PROFILES.items():
    cat[name] = (a * cat["输入未命中"] + b * cat["缓存命中"].fillna(cat["输入未命中"])
                 + c * cat["输出"].fillna(0)).round(4)

# %% 家族归属（按型号厂商，不按渠道）；火山方舟 Agent Plan 的 AFP 抵扣系数（官方表，输入＝输出系数）
FAMILY = [("Qwen", ("qwen",)), ("GLM", ("glm",)), ("Kimi", ("kimi",)), ("DeepSeek", ("deepseek",)),
          ("MiniMax", ("minimax",)), ("MiMo", ("mimo",)), ("豆包", ("doubao",)), ("LongCat", ("longcat",)),
          ("混元", ("hunyuan", "hy4")), ("Step", ("step",)), ("Ling/Ring", ("ling", "ring"))]
AFP = {  # 方舟 Agent Plan 官方抵扣系数（输入, 输出）；缓存命中 token 不单列 → 全输入按输入系数计（须实测）
    "doubao-seed-2.0-mini": (0.25, 2.5), "doubao-seed-2.0-lite": (0.5, 0.5),
    "doubao-seed-2.1-lite": (0.5, 4.5), "doubao-seed-2.1-turbo": (2.5, 2.5), "doubao-seed-2.1-pro": (2.5, 2.5),
    "doubao-seed-evolving": (2.5, 2.5), "glm-5.3-flash": (0.5, 0.5), "glm-5.3": (4.5, 4.5), "glm-5.2": (4.5, 4.5),
    "deepseek-v4-flash": (0.5, 0.5), "deepseek-v4.1-flash": (2.5, 2.5), "deepseek-v4-pro": (5.5, 5.5),
    "kimi-k2.7-code": (4.5, 4.5), "kimi-k3": (10.0, 10.0), "MiniMax-M3": (2.5, 2.5), "Auto": (0.5, 0.5),
}
AFP_YUAN_PER_POINT = 40 / 20_000  # 0.002 元/AFP（Small 40 元＝2 万 AFP；Medium/Large/Max 同价 5/12.5/25 倍量）
AFP_TIERS = [("Small", 40, 20_000), ("Medium", 200, 100_000), ("Large", 500, 250_000), ("Max", 1000, 500_000)]


def afp_unit(coefs, in_share, out_share):
    """Agent Plan 元/百万 tokens。AFP 公式无缓存项 → 缓存命中部分并入「输入」计。"""
    ci, co = coefs
    return (in_share * ci + out_share * co) * 100 * AFP_YUAN_PER_POINT


def afp_key(model):
    m = model.lower().replace("（方舟渠道）", "")
    hits = [(k, v) for k, v in AFP.items() if k.lower() in m]
    return max(hits, key=lambda kv: len(kv[0]))[1] if hits else None


def family_of(model):
    m = model.lower()
    for fam, keys in FAMILY:
        if any(k in m for k in keys):
            return fam
    return "其他"


sc = perf.set_index(["提供商", "模型"])[["智能指数", "可比性"]]
idx = pd.MultiIndex.from_frame(cat[["提供商", "模型"]])
cat["智能指数"] = sc["智能指数"].reindex(idx).tolist()
cat["可比性"] = sc["可比性"].reindex(idx).tolist()
cat["家族"] = cat["模型"].map(family_of)
cat["档级"] = ["旗舰" if s >= 40 else "近旗舰" if s >= 30 else ("弱" if pd.notna(s) else "未评测")
              for s in cat["智能指数"]]

# %% 家族矩阵落盘（每家族按冷会话单价升序；同型号多渠道 → 各行并列，渠道溢价就此暴露）
mat = cat[["家族", "提供商", "模型", "智能指数", "可比性", "档级", "上下文",
           "agent口径", "冷会话", "零缓存"]].sort_values(["家族", "冷会话"]).reset_index(drop=True)
mat.insert(0, "核实日期", "2026-10-12")
mat["每智能点·M"] = (mat["冷会话"] / mat["智能指数"]).round(3)
mat.to_csv(OUT_DIR / "选型_仲裁团_家族矩阵.csv", index=False, encoding="utf-8-sig")

# %% 成员组合：每人一条 (家族, 提供商, 模型, 计量方式)；计量＝按量 / 千问池 / 方舟AFP
def unit(prov, model, how, profile="冷会话"):
    """(元/百万 tokens, 家族归属渠道)。AFP 口径与价目表无关，只取决于系数与负载的输入/输出份额。"""
    a, b, c = PROFILES[profile]
    if how == "方舟AFP":
        coef = afp_key(model)
        if coef is None:
            return None
        return afp_unit(coef, a + b, c)  # 缓存命中并入输入份额计
    r = mat[(mat["提供商"] == prov) & (mat["模型"] == model)]
    if not len(r):
        return None
    p = float(r[profile].iloc[0])
    return p * POOL_DISC if how == "千问池" else p


COMBOS = [
    ("3 席·池内三家族（奇数＝天然无平票）",
     [("GLM", "智谱", "GLM-5.3", "千问池"), ("Kimi", "Kimi", "kimi-k2.7-code", "千问池"),
      ("DeepSeek", "DeepSeek", "deepseek-v4-pro 闲时", "千问池")]),
    ("4 席（＋MiMo 按量，偶数→可能平票）",
     [("GLM", "智谱", "GLM-5.3", "千问池"), ("Kimi", "Kimi", "kimi-k2.7-code", "千问池"),
      ("DeepSeek", "DeepSeek", "deepseek-v4-pro 闲时", "千问池"),
      ("MiMo", "小米MiMo", "MiMo-V2.6-Pro", "按量")]),
    ("4 席·池三家族＋豆包 AFP（家族最多且奇数→5 见下）",
     [("GLM", "智谱", "GLM-5.3", "千问池"), ("Kimi", "Kimi", "kimi-k2.7-code", "千问池"),
      ("DeepSeek", "DeepSeek", "deepseek-v4-pro 闲时", "千问池"),
      ("豆包", "火山方舟", "doubao-seed-2.1-pro（输入≤1024K）", "方舟AFP")]),
    ("5 席·池三＋豆包 AFP＋MiMo 按量",
     [("GLM", "智谱", "GLM-5.3", "千问池"), ("Kimi", "Kimi", "kimi-k2.7-code", "千问池"),
      ("DeepSeek", "DeepSeek", "deepseek-v4-pro 闲时", "千问池"),
      ("豆包", "火山方舟", "doubao-seed-2.1-pro（输入≤1024K）", "方舟AFP"),
      ("MiMo", "小米MiMo", "MiMo-V2.6-Pro", "按量")]),
    ("3 席·方舟 Agent Plan 独家（GLM＋Kimi＋豆包 全走 AFP）",
     [("GLM", "智谱", "GLM-5.3", "方舟AFP"), ("Kimi", "Kimi", "kimi-k3", "方舟AFP"),
      ("豆包", "火山方舟", "doubao-seed-2.1-pro（输入≤1024K）", "方舟AFP")]),
    ("5 席·近旗舰全员（成本侧对照）",
     [("GLM", "智谱", "GLM-5.3-Flash", "千问池"), ("Kimi", "Kimi", "kimi-k2.7-code", "千问池"),
      ("DeepSeek", "DeepSeek", "deepseek-flash（V4.1-Flash）闲时", "千问池"),
      ("MiMo", "小米MiMo", "MiMo-V2.6-Flash", "按量"),
      ("豆包", "火山方舟", "doubao-seed-2.1-turbo（输入≤256K）", "方舟AFP")]),
    ("5 席·全按量（不开池、只充值 5 个点的对照）",
     [("GLM", "智谱", "GLM-5.3", "按量"), ("Kimi", "Kimi", "kimi-k2.7-code", "按量"),
      ("DeepSeek", "DeepSeek", "deepseek-v4-pro 闲时", "按量"),
      ("MiMo", "小米MiMo", "MiMo-V2.6-Pro", "按量"),
      ("豆包", "火山方舟", "doubao-seed-2.1-pro（输入≤1024K）", "按量")]),
]
for prof in ("冷会话", "零缓存", "agent口径"):
    print(f"\n== 组合月费（{prof}，元/月，每成员均分 {VOL:.0f}M）==")
    out = []
    for name, members in COMBOS:
        per = VOL / len(members)
        pp = [(f, unit(p, m, h, prof), h) for f, p, m, h in members]
        total = sum((x or 0) * per for _, x, _ in pp)
        need = sum(x * per / AFP_YUAN_PER_POINT for _, x, h in pp if x and h == "方舟AFP")
        tier = next((f"{tn}（{pr:g} 元）" for tn, pr, cap in AFP_TIERS if need <= cap), "超 Max，需加购") if need else "—"
        out.append(dict(组合=name, 成员=len(members), 月费=round(total, 0),
                        需AFP=round(need) if need else "—", AFP落档=tier,
                        成员单价=" ".join(f"{f}:{x:.2f}" if x else f"{f}:无行" for f, x, _ in pp)))
    print(pd.DataFrame(out).to_string(index=False, max_colwidth=58))

# %% 打印家族矩阵
print("\n== 各家族最强型号（AA≥34，每家族头名；同榜优先，异源单列以免误并口径）==")
strong = mat[mat["智能指数"] >= 34].sort_values(["可比性", "智能指数"], ascending=[True, False]).groupby("家族").head(1)
print(strong[["家族", "提供商", "模型", "智能指数", "可比性", "agent口径", "冷会话", "零缓存", "每智能点·M"]]
      .to_string(index=False, max_colwidth=32))
print("\n== 未评测但可用的家族旗舰（AA 无条目 → 进团前须自测校准）==")
print(mat[(mat["智能指数"].isna()) & (mat["模型"].str.contains("pro|evolving|2.1", case=False))]
      [["家族", "提供商", "模型", "上下文", "agent口径", "冷会话"]].sort_values("冷会话")
      .head(8).to_string(index=False, max_colwidth=34))
print("\n== 方舟 Agent Plan AFP 口径（元/M；AFP 无缓存项→冷会话份额 0.80 输入/0.20 输出；对照同型号按量冷会话价）==")
for k, (ci, co) in sorted(AFP.items(), key=lambda kv: kv[1][0]):
    r = mat[mat["模型"].str.lower().str.contains(k.lower(), regex=False)]
    cmp_ = f"{float(r['冷会话'].min()):.2f}" if len(r) and r["冷会话"].notna().any() else "—"
    print(f"  {k:<20} 系数 输入{ci}/输出{co} → {afp_unit((ci, co), 0.80, 0.20):.3f} 元/M（按量冷会话 {cmp_}）")
print(f"\n矩阵 {len(mat)} 行已写入 选型_仲裁团_家族矩阵.csv；个人月化总量 {U_M:.1f}M，仲裁份额按 25% 取 {VOL:.0f}M")
