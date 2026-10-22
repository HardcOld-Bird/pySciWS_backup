# %% 选型·步骤二：Qwen 系全型号成本排名（按量 / Batch / 订阅等效三模式并列同榜）
# 用量口径：月化 未缓存输入 101.573M、缓存输入 1553.231M、输出 15.785M（个人 15 天实测 × 30/15）。
# 入榜规则：性能仅作参考、不设硬门限；仅剔除与 3.8-flash（AA v4.3 同榜 40 分）存在断层
# （分差 ≥ 10）或体量明显过小（3B/10B 级激活、embedding/图像）的型号，剔除项连同理由一并落表。
from pathlib import Path

import pandas as pd

from pysci.paths import assert_within_data

OUT_DIR = Path(__file__).resolve().parent
assert_within_data(OUT_DIR)

U_NC, U_C, U_O = 101.573, 1553.231, 15.785  # 百万 tokens / 月
ANCHOR_FLASH = 40.0  # qwen3.8-flash 的 AA 同榜分
ALI_PLAN_CAP = 499.0  # Token Plan 个人版 Pro 限时月价

perf = pd.read_csv(OUT_DIR / "模型性能指数.csv")
score = {(r["提供商"], r["模型"]): (r["智能指数"], r["可比性"]) for _, r in perf.iterrows()}
plan = pd.read_csv(OUT_DIR / "模拟月成本_订阅等效.csv")
k_ali = float(plan.loc[(plan["提供商"] == "阿里云百炼") & (plan["模型"] == "qwen3.8-max"), "k"].iloc[0])

# (提供商, 型号, 备注)；月价一律从 模拟月成本_按量.csv 取，此处不重复价目（单一数据源）
PAYG_QWEN = [
    ("阿里云百炼", "qwen3.8-max", "隐式缓存命中 1.5；限时5折 6.0/18.0"),
    ("阿里云百炼", "qwen3.8-flash", "隐式缓存命中 0.1"),
    ("阿里云百炼", "qwen3.7-max", "上一代旗舰，无隐式缓存价；AA 本快照未收录"),
    ("阿里云百炼", "qwen3.7-plus", "上一代 plus 档，无隐式缓存价"),
    ("阿里云百炼", "qwen3.7-flash", "上一代 flash，无隐式缓存价"),
    ("千问AI平台", "qwen3.8-max", "与百炼同源同价"),
    ("千问AI平台", "qwen3.8-flash", "与百炼同源同价"),
    ("SiliconFlow", "Qwen3.8-27B（开源版）", "开源权重 27B；SF 无缓存价→缓存按输入全价（保守）"),
    ("SiliconFlow", "Qwen3.5-35B-A3B（输入<128K）", "激活 3B 小体量"),
    ("SiliconFlow", "Qwen3.5-122B-A10B（输入<128K）", "激活 10B 小体量"),
    ("SiliconFlow", "Qwen2.5-72B-Instruct（含 128K 版）", "上上代 72B，AA 本快照未收录"),
]

sim = pd.read_csv(OUT_DIR / "模拟月成本_按量.csv")
PAYG_COST = sim.groupby(["提供商", "模型"])["月总成本"].min().to_dict()

# 无 AA 分但按体量直接剔除（用户规则：体量很小的 mini 千问不入排名）
SMALL_VOLUME = {
    ("SiliconFlow", "Qwen3.5-35B-A3B（输入<128K）"),
    ("SiliconFlow", "Qwen3.5-122B-A10B（输入<128K）"),
}


def drop_reason(prov: str, model: str) -> str | None:
    s, _ = score.get((prov, model), (None, "未评测"))
    if pd.notna(s) and ANCHOR_FLASH - s >= 10:
        return f"AA {s:.0f} 分，与 3.8-flash 断层（差 {ANCHOR_FLASH - s:.0f}）"
    if (prov, model) in SMALL_VOLUME:
        return "激活 3B/10B 小体量，同族同榜分仅 16–18，断层"
    return None


rows = []


def add(prov, model, mode, cost, note):
    s, cmp_ = score.get((prov, model), (None, "未评测"))
    reason = drop_reason(prov, model)
    rows.append(dict(提供商=prov, 型号=model, 模式=mode, 月估价=round(cost, 1),
                     AA分数=s if pd.notna(s) else None, 分数口径=cmp_,
                     入榜=("否" if reason else "是"), 剔除理由=reason or "", 备注=note))


# %% 按量
for prov, model, note in PAYG_QWEN:
    add(prov, model, "按量", PAYG_COST[(prov, model)], note)

# %% 30B-A3B / Omni / Coder 合并档与专用模型（整档剔除，仍落表留痕）
for model, cost, note in [
    ("0.7/2.8 档（Qwen3-VL/Omni/Coder-30B-A3B、30B-A3B-Instruct-2507）",
     U_NC * 0.7 + U_C * 0.7 + U_O * 2.8, "3B 级激活；AA 同族条目 6–10 分"),
    ("Qwen3-VL-32B-Instruct / Qwen3-32B", U_NC * 1.0 + U_C * 1.0 + U_O * 4.0, "AA 未收录的上一代 32B"),
    ("Qwen3-VL-8B-Thinking / Embedding·Reranker / bge 系列", float("nan"), "8B 级或向量模型，任务类型不符"),
]:
    rows.append(dict(提供商="SiliconFlow", 型号=model, 模式="按量", 月估价=round(cost, 1),
                     AA分数=None, 分数口径="未评测", 入榜="否",
                     剔除理由="小体量或非生成任务，断层/不适用", 备注=note))

# %% Batch（无缓存折扣：缓存 token 按折半未命中价计——对本高缓存负载反优化，列作对照）
add("阿里云百炼", "qwen3.8-flash", "Batch", U_NC * 0.4 + U_C * 0.4 + U_O * 1.35,
    "Batch 半价 0.4/1.35，无缓存折扣")
add("阿里云百炼", "qwen3.8-max", "Batch", U_NC * 6.0 + U_C * 6.0 + U_O * 18.0,
    "Batch File 6.0/18.0（5折），无缓存折扣")

# %% 订阅等效（Token Plan：按量月价 × k，允许多账户叠加，非整数份）
for prov, model, _ in PAYG_QWEN:
    if prov not in ("阿里云百炼", "千问AI平台"):
        continue
    equiv = PAYG_COST[(prov, model)] * k_ali
    n = max(1, -(-equiv // ALI_PLAN_CAP))
    add(prov, model, "订阅等效", equiv, f"按量×k({k_ali:.4f})；需购 {n:g} 份 Token Plan Pro（{ALI_PLAN_CAP:g}/月）")

df = pd.DataFrame(rows)
df["排序键"] = df["月估价"].fillna(float("inf"))
df = df.sort_values(["入榜", "排序键"], ascending=[True, True]).reset_index(drop=True)
df.insert(0, "排名", range(1, len(df) + 1))
df = df.drop(columns="排序键")
df.insert(1, "核实日期", "2026-10-12")
df.to_csv(OUT_DIR / "选型_Qwen_成本排名.csv", index=False, encoding="utf-8-sig")

live = df[df["入榜"] == "是"]
print(live.to_string(index=False))
print(f"\n入榜 {len(live)} 项 / 剔除 {len(df) - len(live)} 项，已写入 选型_Qwen_成本排名.csv")
