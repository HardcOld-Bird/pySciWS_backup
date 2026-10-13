# %% SF 统一调价 vs 各家官网直调：AA 高分旗舰的同型号价差（2026-10-12，组长代算）
# 问题：SiliconFlow 作为聚合方「统一调和」各家价格，与我们分别走各家官网直调相比，差距多大？
# 方法：只比**同型号**（否则是在比不同的模型），对每一对给出三口径混合单价与倍率：
#   agent口径 0.061/0.930/0.009（实测个人月化结构，缓存命中 93.9%）
#   冷会话口径 0.65/0.15/0.20（仲裁团类负载）
#   零缓存口径 0.80/0.00/0.20（上界）
#   无缓存价的型号 → 缓存 token 按输入全价计（与 模拟月成本_按量 同规则）。
# 另列原厂**限时价**对照：官方常做限时五折/闲时价，而 SF 贴的是刊例价 → 刊例同价不等于实付同价。
# 单一事实来源：所有单价读自 按量付费_模型价目.csv，本文件只存配对关系与限时价（价目表「其他维度」里的原文摘要）。
from pathlib import Path
import pandas as pd
from pysci.paths import assert_within_data

OUT_DIR = Path(__file__).resolve().parent
assert_within_data(OUT_DIR)

PROFILES = {"agent口径": (0.0609, 0.9302, 0.0094), "冷会话": (0.65, 0.15, 0.20), "零缓存": (0.80, 0.00, 0.20)}
AA = {"GLM-5.3": 45, "GLM-5.3-Flash": 42, "Kimi K2.6": 43, "Kimi K2.7 Code": 26, "DeepSeek-V4-Pro": 36,
      "DeepSeek V4.1 Flash": 39, "Qwen3.8-27B": 34}

cat = pd.read_csv(OUT_DIR / "按量付费_模型价目.csv")
cat = cat[cat["输入未命中"].notna()].set_index(["提供商", "模型"])


def mix(prov, model, prof):
    """按 (提供商,模型) 行算三口径混合单价；行不存在 → None。"""
    if (prov, model) not in cat.index:
        return None
    r = cat.loc[(prov, model)]
    a, b, c = PROFILES[prof]
    pin, pc, po = r["输入未命中"], r["缓存命中"], r["输出"]
    return round(a * pin + b * (pin if pd.isna(pc) else pc) + c * (0 if pd.isna(po) else po), 4)


# (家族/型号, SF 行模型名, 原厂提供商, 原厂行模型名, 原厂限时/闲时价 (输入,缓存,输出) 或 None, 备注)
PAIRS = [
    ("GLM-5.3", "GLM-5.3 / GLM-5.2", "智谱", "GLM-5.3", None, "SF 声明贴齐智谱官方价；刊例完全一致"),
    ("GLM-5.3-Flash", "GLM-5.3-Flash", "智谱", "GLM-5.3-Flash", (0.4, 0.115, 1.4), "智谱限时五折 0.4/0.115/1.4 → SF 贴的是刊例价"),
    ("Kimi K2.6", "Kimi-K2.6 (Pro)", "Kimi", "kimi-k2.6", None, "同价 6.5/1.1/27；AA 43 系异源口径"),
    ("Kimi K2.7 Code", "Kimi-K2.7-Code", "Kimi", "kimi-k2.7-code", None, "同价 6.5/1.3/27"),
    ("DeepSeek-V4-Pro", "DeepSeek-V4-Pro", "DeepSeek", "deepseek-v4-pro 闲时", None, "原厂闲时 4.5/0.15/13.5，高峰翻倍 9.0/0.3/27.0"),
    ("DeepSeek V4.1 Flash", "DeepSeek-V4-Flash", "DeepSeek", "deepseek-flash（V4.1-Flash）闲时", None,
     "原厂闲时 1.0/0.02/4.0；SF 分时段仅 2-8 点 5 折 1.5/0.15/4.5"),
    ("Qwen3.8-27B（开源权重）", "Qwen3.8-27B（开源版）", "阿里云百炼", "qwen3.8-flash", None,
     "非同型号：SF 只有开源 27B，商业 flash/max 未上架；此处比的是「想要 Qwen 时 SF 能不能顶替」"),
]
rows = []
for fam, sf_model, off_prov, off_model, sale, note in PAIRS:
    rec = dict(家族型号=fam, 智能指数=AA.get(fam.split("（")[0], ""), SF行=sf_model, 原厂=f"{off_prov} / {off_model}")
    for prof in PROFILES:
        sf, off = mix("SiliconFlow", sf_model, prof), mix(off_prov, off_model, prof)
        rec[f"SF_{prof}"], rec[f"原厂_{prof}"] = sf, off
        rec[f"倍率_{prof}"] = round(sf / off, 2) if sf and off else None
    if sale:
        a, b, c = PROFILES["agent口径"]
        v = round(a * sale[0] + b * sale[1] + c * sale[2], 4)
        rec["原厂限时_agent"] = v
        rec["倍率_SF对限时"] = round(mix("SiliconFlow", sf_model, "agent口径") / v, 2)
    rec["备注"] = note
    rows.append(rec)
df = pd.DataFrame(rows)
df.insert(0, "核实日期", "2026-10-12")
df.to_csv(OUT_DIR / "对比_SF统一调价_vs_官网直调.csv", index=False, encoding="utf-8-sig")

print("== 同型号价差（元/百万 tokens）==")
print(df[["家族型号", "智能指数", "SF_agent口径", "原厂_agent口径", "倍率_agent口径",
          "SF_冷会话", "原厂_冷会话", "倍率_冷会话", "SF_零缓存", "原厂_零缓存", "倍率_零缓存"]]
      .to_string(index=False, max_colwidth=30))
print("\n== 原厂限时/闲时价对照（agent 口径）==")
print(df[df["原厂限时_agent"].notna()][["家族型号", "SF_agent口径", "原厂限时_agent", "倍率_SF对限时"]]
      .to_string(index=False))

# %% 覆盖缺口：AA≥34 的同榜型号里，SF 到底有没有上架（价差之外更要紧的问题）
perf = pd.read_csv(OUT_DIR / "模型性能指数.csv")
hi = perf[(perf["智能指数"] >= 34) & (perf["可比性"] == "同榜")].drop_duplicates("模型")
KEYS = {"GLM": "glm", "Kimi": "kimi", "Qwen": "qwen", "DeepSeek": "deepseek", "MiMo": "mimo",
        "豆包": "seed", "MiniMax": "minimax"}
sf_models = sorted(str(m).lower() for m in cat.loc["SiliconFlow"].index.get_level_values(0))
print(f"\n== 覆盖缺口（AA≥34 同榜型号 {len(hi)} 个，按家族看 SF 是否上架）==")
for fam, key in KEYS.items():
    ms = [f"{m}={s:.0f}" for m, s in zip(hi["模型"], hi["智能指数"]) if key in str(m).lower()]
    on_sf = [m for m in sf_models if key in m]
    print(f"  {fam:<8} SF{'有' if on_sf else '无'}（命中 {len(on_sf)} 行）｜AA≥34 同榜：{ms or '—'}")
