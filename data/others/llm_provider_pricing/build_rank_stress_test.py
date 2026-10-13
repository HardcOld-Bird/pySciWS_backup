# %% 弱模型 harnessing 压力测试选型：全库「最实惠」升序排名（2026-10-12，组长代算）
# 用例性质：压测不产出内容，只反复暴露 harnessing 缺陷 → 目标是「同样的缺陷暴露量，花钱最少」，
#   且同一次改进要能被便宜地重跑。与主力选型反向：性能不是目标，「弱」是优点（越早暴露缺陷越有用），
#   唯一硬约束是该模型还能驱动 agent 工作模式（API 层 tools 可用 + 上下文撑得住 + 不陷入不可收敛循环）。
# 指标口径（压测量由我们自己定，故排序用与量无关的单位成本，而非本尺月价）：
#   混合单价 = 该形态下每百万 token 实付元数（本负载结构 93.9% 缓存命中，可移植到任意压测量）；
#   本尺月价 = 按量混合单价 × 1670.6M（个人月用量，仅作品级参照）；
#   起购固定费 = 该型号所在套餐的最低档月费（按量付费无最低消费 → 0）；
#   单位成本比 = 按量混合单价 ÷ 套餐混合单价（>1 表示套餐单位成本更低）；
#   保本量 = 起购固定费 ÷ 该型号按量单价（月用量越过它，套餐才开始省钱）；
#   池顶量 = 该档额度的上界（M tokens；超过它须加购或转按量）。买套餐的有效区间＝保本量 < 量 ≤ 池顶量。
# 门限形态（用户指示：不严格落实门限，无评分模型保留在视野内）：不设剔除式性能门限，逐行标
#   「压测档位」：A 可稳定驱动 agent（作下限参照）／B 能驱动但高缺陷率（正是要压的标的）／
#   C 边界（须先用小任务子集试跑）／D 不能驱动 agent（无 tools／专用模态）或本用例无理由。
# 单一事实来源：按量价、上下文、套餐档位月费与额度一律读自生成的 CSV，本文件不重复录入价格。
#   本文件只存判断性数据：压测判据、免费档事实、套餐覆盖范围与额度折算口径。
from pathlib import Path
import pandas as pd
from pysci.paths import assert_within_data

OUT_DIR = Path(__file__).resolve().parent
assert_within_data(OUT_DIR)

payg = pd.read_csv(OUT_DIR / "模拟月成本_按量.csv")
perf = pd.read_csv(OUT_DIR / "模型性能指数.csv")
cat = pd.read_csv(OUT_DIR / "按量付费_模型价目.csv")
sub = pd.read_csv(OUT_DIR / "订阅套餐_对比.csv")
u = pd.read_csv(OUT_DIR / "个人每日Token用量.csv")
U_M = float((u[["未缓存输入", "缓存输入", "输出"]].sum() * (30 / len(u)) / 1e6).sum())  # 百万 tokens/月

cost = cost_payg = payg.groupby(["提供商", "模型"])["月总成本"].min()
score = perf.set_index(["提供商", "模型"])[["智能指数", "可比性"]]
ctx = cat.drop_duplicates(["提供商", "模型"]).set_index(["提供商", "模型"])["上下文"]

# 冷会话口径单价（元/M）：仲裁团等「冷启动短会话」负载＝未缓存 0.65／缓存 0.15／输出 0.20，
# 无缓存价的型号按输入全计（与 模拟月成本_按量 同规则）。与 agent 口径并列，供跨用例读同一张表。
CH = (0.65, 0.15, 0.20)
_c = cat[cat["输入未命中"].notna()]
cold = {(r.提供商, r.模型): round(CH[0] * r.输入未命中 + CH[1] * (r.缓存命中 if pd.notna(r.缓存命中) else r.输入未命中)
                                + CH[2] * (r.输出 if pd.notna(r.输出) else 0), 4) for r in _c.itertuples()}

# %% 免费档（0 元；价目表仅在「其他维度」注明，故此处显式成行）
FREENOTE = ("免费档 RPM/TPM 与并发显著低于付费，第三方实践明确建议「压测、高并发业务不要用免费版」，"
            "而本用例正是多路并发重跑 → 免费档只适合串行小批量试错")
FREE = [
    ("SiliconFlow", "GLM-4-9B（免费档）", "价目表 SF「GLM-4-9B 免费」注明", "B",
     "SF 官方 Function Calling 文档列明 THUDM/GLM-4-9B-0414 支持 tools → 可驱动 agent；9B 体量，"
     "漏调工具/参数幻觉/死循环风险高，正是压测标的。上下文与限流待核"),
    ("SiliconFlow", "Qwen3-8B（免费档）", "价目表 SF「Qwen3-8B 免费」注明", "B",
     "Qwen3 系原生 tool calling（官方推荐 Qwen-Agent 封装）。AgenticQwen 论文实测底座 Qwen3-8B 在 "
     "τ²-Bench/BFCL-V4 多轮平均 23.8（30B-A3B 底座 36.2）→ 能调工具但极易失稳；原生上下文约 40K"),
    ("SiliconFlow", "Qwen3.5-4B（免费档）", "价目表 SF「Qwen3.5-4B 免费」注明", "B",
     "AA 本快照 Qwen3.5 4B=13；官方定位「轻量级 Agent 基座」（ScreenSpot Pro ≈ 上代 Qwen3-VL-30B-A3B）；"
     "文本 agent 长循环稳定性未测"),
    ("智谱", "GLM-4.7-Flash（免费档）", "价目表智谱 GLM-4.7 行「GLM-4.7-Flash 免费」注明", "B",
     "GLM-4.7 系原生工具调用；上一代 flash；AA 本快照未收录；免费档限流"),
    ("SiliconFlow", "GLM-Z1-9B（免费档）", "价目表 SF「GLM-Z1-9B 免费」注明", "C",
     "Z1 思考系 9B；SF tools 列表含 GLM-Z1-32B，9B 变体支持情况待核"),
    ("SiliconFlow", "Qwen2.5-7B-Instruct（Free 档）", "价目表 SF「7B(Free) 免费」注明", "C",
     "SF 官方 tools 列表含 Qwen2.5 系；上代、上下文小，长 agent 循环预期撑不住"),
    ("SiliconFlow", "Xing4.0-29B（免费档）", "价目表 SF「Xing4.0-29B 免费」注明", "C",
     "AA 未评测、tools 支持未确认，型号信息薄，须先小任务试跑"),
    ("SiliconFlow", "Hunyuan-MT-7B（免费档）", "价目表 SF「Hunyuan-MT-7B 免费」注明", "D",
     "翻译专用模型，不承担 agent 语义"),
    ("SiliconFlow", "DeepSeek-OCR / PaddleOCR-VL-1.5（免费档）", "价目表 SF 注明免费", "D",
     "OCR 专用，无通用工具调用"),
]

# %% 逐行压测判据 (提供商, 模型) → (档位, 判据)
# 依据：AA 智能指数与可比性；AgenticQwen 论文（8B/30B-A3B 底座工具调用实测分、约 40K 原生上下文上限）；
# 独立 tool-calling 全尺寸扫描（Qwen3.5 系列 15 场景 × 12 工具：仅 27B dense 全过，35B 挂 2、122B 挂 1、
# 397B 挂 2，小模型典型病症＝死循环）；各渠道官方 tools 支持面（百炼/千问「所有通用文本生成模型均支持
# function calling」；SF 只有部分型号，DeepSeek R1/V3 曾因不支持而对 functions 参数返回 400）。
FEAS = {
    ("阿里云百炼", "qwen3.8-flash"): ("A", "AA 40；1M；百炼 tools 全支持＋Qoder 亲和加分。非弱模型，"
        "列作「改进有效性」下限参照（harness 变体在它身上应几乎不触发缺陷）"),
    ("千问AI平台", "qwen3.8-flash"): ("A", "同上，同源同价"),
    ("阿里云百炼", "qwen3.7-flash"): ("A", "无 AA 分；0.3/1.2 且无缓存价（本尺下被 93% 缓存负载放大）；"
     "Qwen3.7 代 tools 全支持＋Qoder 亲和；比 3.8 老一代 → 缺陷暴露率天然更高，正合本用例"),
    ("阿里云百炼", "qwen3.7-plus"): ("A", "AA 25；1M；tools 全支持；Qwen 系加分"),
    ("阿里云百炼", "qwen3.8-max"): ("A", "AA 45；本用例过强过贵，仅列作上界对照"),
    ("千问AI平台", "qwen3.8-max"): ("A", "同上"),
    ("小米MiMo", "MiMo-V2.6-Flash"): ("A", "AA 38；1M；全库最低非零按量单价，但 MiMo 无 Qoder 亲和加分且性能偏强 "
        "→ 作下限参照而非压测标的"),
    ("小米MiMo", "MiMo-V2.6-Pro"): ("A", "AA 46（本快照国产最高分之一）；作下限参照"),
    ("小米MiMo", "MiMo-V2.6-Pro-UltraSpeed"): ("A", "同 Pro 极速档，只贵不弱，无本用例价值"),
    ("小米MiMo", "MiMo-V2.5（输入<256K）"): ("B", "AA 未收录；1M；旧代工具调用格式与 harness 预期漂移的风险本身即压测信号"),
    ("小米MiMo", "MiMo-V2.5-Pro（输入<256K）"): ("A", "AA 54 系异源 v4.0 口径，不可与同榜并列；性能过强，不入压测标的"),
    ("智谱", "GLM-5.3-Flash"): ("A", "AA 42；1M；原生工具调用；作下限参照"),
    ("火山方舟", "glm-5.3-flash（方舟渠道）"): ("A", "同模型不同渠道，与智谱同价"),
    ("SiliconFlow", "GLM-5.3-Flash"): ("A", "同模型不同渠道，贴齐官方价"),
    ("智谱", "GLM-5.3-FlashX"): ("B", "AA 本快照未单列（应略低于 Flash 42）；输出 7.0 元/M 高于 Flash → 便宜不是它的卖点"),
    ("智谱", "GLM-4.7（输入<32K，输出<0.2K）"): ("B", "上一代未评测；≤32K 且输出<0.2K 的窄档计价 → agent 输出越过 "
        "0.2K 即跳档，压测会自然暴露这条计费边界"),
    ("智谱", "GLM-5（输入<32K）"): ("C", "未评测；GLM-5 代际在 Qoder 上的工具格式未验"),
    ("智谱", "GLM-5-Turbo（输入<32K）"): ("C", "同上，且比 GLM-5 贵"),
    ("智谱", "GLM-5.1（输入<32K / ≥32K）"): ("C", "未评测；32K 阶梯会截断长上下文 agent 循环"),
    ("智谱", "GLM-5.2"): ("C", "AA 34 系异源口径（镜像 09-13），不可与 45/42 并列"),
    ("火山方舟", "doubao-seed-2.0-mini（≤32K）"): ("B", "AA 无此条目（唯一相关 Doubao Seed Code=17 系另一型号）；256K；"
        "≤32K 阶梯 → 越过 32K 整次请求跳档翻倍，正是压测会撞的边界；方舟自研无第三方 agent 分 → 可行性只能自测"),
    ("火山方舟", "doubao-seed-2.0-lite（≤32K）"): ("B", "同上；未评测；256K"),
    ("火山方舟", "doubao-seed-2.0-pro（≤32K / 32-128K / 128-256K）"): ("C", "未评测；三档阶梯，长上下文直跳 9.6/1.92/48 高价档"),
    ("火山方舟", "doubao-seed-2.1-lite（输入≤1024K）"): ("B", "未评测；音频输入另计；方舟 tools 面须按模型卡片核对"),
    ("火山方舟", "doubao-seed-2.1-turbo（输入≤256K）"): ("C", "未评测；比 2.1-lite 贵 4 倍，无本用例理由"),
    ("火山方舟", "doubao-seed-2.1-pro（输入≤1024K）"): ("C", "未评测；旗舰定位"),
    ("火山方舟", "doubao-seed-evolving（输入≤1024K）"): ("C", "未评测；自演进实验型号，接口稳定性未知"),
    ("火山方舟", "deepseek-v4-flash 正式版（方舟渠道）"): ("B", "同族 V4.1-Flash=39，此型号未进本快照；"
        "方舟缓存 0.1 → 本尺下比原厂闲时贵"),
    ("火山方舟", "deepseek-v4-pro 正式版（方舟渠道）"): ("C", "AA 36；方舟价高于原厂闲时价，本用例无理由走该渠道"),
    ("火山方舟", "deepseek-v4.1-flash（方舟渠道）闲时"): ("B", "与原厂同构峰谷价（AA 39）；走方舟的唯一理由是搭 Coding Plan 的"
        "次数池，而该池又要求「同型号在 Coding Plan 覆盖内」→ 单看按量无理由"),
    ("火山方舟", "glm-5.2（方舟渠道）"): ("C", "AA 异源 34（镜像 09-13 口径）；本用例过强过贵"),
    ("DeepSeek", "deepseek-flash（V4.1-Flash）闲时"): ("A", "AA 39（非推理档 25）；1M；DeepSeek 原生 tools 的 JSON "
        "稳定性口碑好 → 削到非推理档可得「便宜且弱」；另有 33% 峰谷杠杆"),
    ("DeepSeek", "deepseek-v4-pro 闲时"): ("C", "AA 36；本用例过强过贵"),
    ("SiliconFlow", "DeepSeek-V4-Flash"): ("B", "同族 V4.1-Flash=39；SF 侧 DeepSeek 系 tools 曾长期缺位"
        "（R1/V3 一度不支持、传 functions 返回 400）→ 上架型号必须核卡片"),
    ("SiliconFlow", "DeepSeek-V4-Pro"): ("C", "AA 36；SF 价明显高于原厂"),
    ("SiliconFlow", "DeepSeek-V3.2 / V3.2 (Pro)"): ("B", "AA 未收录（上代）；SF 官方 tools 列表含 V3 系；"
        "SF 缓存 0.4 → 本尺 1122 元，贵于新代 flash"),
    ("MiniMax", "MiniMax-M2.5 / M2.1 / M2"): ("B", "AA 有 M2.5 条目但本快照未抓到分；200K；历史模型、工具调用格式老 "
        "→ 缺陷暴露面广；经百炼直连时 MiniMax 系有特殊渠道参数要求"),
    ("MiniMax", "MiniMax-M2.7"): ("B", "未评测；200K；M2 系较新一档，同上判据"),
    ("MiniMax", "MiniMax-M2.7-highspeed"): ("A", "同模型极速档，性能等同价翻倍 → 本用例无理由"),
    ("MiniMax", "MiniMax-M3（输入≤512K）"): ("A", "AA 29 同榜但版本漂移明显（09-13 报 30、旧文 v4.1 报 44）；1M；"
        "「永久五折」后价"),
    ("Kimi", "kimi-k2.7-code"): ("B", "AA 26 同榜；256K；代码向但总分低 → 缺陷暴露率高；官方要求思考模式每轮 assistant "
        "保留 reasoning_content 字段（harness 适配成本本身即压测对象）"),
    ("SiliconFlow", "Kimi-K2.7-Code"): ("B", "同模型不同渠道，贴齐官方价"),
    ("Kimi", "kimi-k2.7-code-highspeed"): ("A", "同模型极速档，性能等同价翻倍 → 本用例无理由"),
    ("Kimi", "kimi-k2.5"): ("C", "AA 本快照未收录；上下文未标"),
    ("Kimi", "kimi-k2.6"): ("C", "AA 43 系异源口径；价高于 k2.7-code"),
    ("Kimi", "kimi-k3"): ("C", "AA 44；本用例过强过贵"),
    ("阿里云百炼", "kimi-k2.7-code"): ("B", "同模型第三方直供，渠道溢价 3.6× → 本用例无渠道理由"),
    ("阿里云百炼", "kimi-k3"): ("C", "AA 44；溢价 5.2×"),
    ("阿里云百炼", "kimi-k2.5"): ("C", "未评测；溢价高"),
    ("阿里云百炼", "ZHIPU/GLM-5.3（智谱直供）"): ("C", "AA 45；过强过贵"),
    ("阿里云百炼", "vanchin/deepseek-v4.1-flash（万擎直供）"): ("C", "AA 39；渠道溢价 11.7×，本用例排除该渠道"),
    ("阿里云百炼", "vanchin/deepseek-v4-pro（万擎直供）"): ("D", "渠道溢价 14.9× 且型号过强——此处 D 指选型排除，非能力排除"),
    ("阿里云百炼", "siliconflow/deepseek-v3.2（硅基流动直供）"): ("C", "未评测；渠道溢价"),
    ("SiliconFlow", "Qwen3.5-35B-A3B（输入<128K）"): ("B", "同族同榜 AA 16–18；独立 tool-calling 全尺寸扫描中 35B 档挂 "
        "2/15 场景（典型病症＝死循环）→ 正是要暴露的缺陷面；Qwen 系有 harness 亲和加分；SF 无缓存价 → 本尺下被 93% 缓存"
        "负载放大，但绝对单价仍是全库最低一档"),
    ("SiliconFlow", "Qwen3.5-122B-A10B（输入<128K）"): ("B", "AA 18；扫描中 122B 挂 1 场景；无缓存价 → 单价翻倍"),
    ("SiliconFlow", "Qwen3.8-27B（开源版）"): ("A", "AA 34（xhigh；medium 28/low 26/非推理 20）；第三方独立扫描称其 "
        "agentic 维度反超同级 DeepSeek V4 Pro → 开源版里最强的下限参照"),
    ("SiliconFlow", "0.7/2.8 档（Qwen3-VL-30B-A3B、Qwen3-Omni-30B-A3B（Instruct/Thinking/Captioner）、"
                     "Qwen3-Coder-30B-A3B、Qwen3-30B-A3B-Instruct-2507）"): ("C", "AA 该档 6–10；30B-A3B 激活 3B，"
        "AgenticQwen 论文底座实测 τ²/BFCL 平均 36.2（能调工具、易失稳）、原生上下文约 40K → Qoder 单轮上下文常超此限；"
        "合并行含 Coder 变体，拆开后 Qwen3-Coder-30B-A3B 可单独升 B"),
    ("SiliconFlow", "Qwen3-VL-8B-Thinking"): ("C", "AA 未收录；8B＋VL，多模态非本用例；上下文小"),
    ("SiliconFlow", "Qwen3-VL-32B-Instruct / Qwen3-32B"): ("C", "上代未收录；无缓存价；VL 分支非本用例"),
    ("SiliconFlow", "Qwen3-VL-32B-Thinking"): ("C", "同上，Thinking 输出更贵"),
    ("SiliconFlow", "GLM-4.5V / GLM-4.5-Air"): ("C", "AA 有条目但本快照未抓到分；GLM-4.5-Air 系 30B-A3B 上代，"
        "SF 侧 tools 支持待核"),
    ("SiliconFlow", "LongCat-2.0（美团）"): ("B", "AA 19 同榜；全库同价位里「有分且最弱」的一档，缺陷暴露率高；"
        "SF 渠道 tools 需核卡片"),
    ("SiliconFlow", "Step-3.5-Flash / Ling-flash-2.0 / Ling-mini-2.0 / Seed-OSS-36B / Xing4.0-29B"): ("C",
        "合并行，与 AA 条目（Step 5 Preview=44、Ring-flash-2.0=7）非一一对应 → 拆名单型号前无法判档"),
    ("SiliconFlow", "Hunyuan-A13B-Instruct"): ("C", "AA 未收录；13B 上代混元，工具调用格式老；分时段仅 ×0.8"),
    ("SiliconFlow", "Hy4 preview（混元）"): ("C", "AA 本快照仅列 Hy3=25；preview 版本不稳定"),
    ("SiliconFlow", "Qwen2.5-72B-Instruct（含 128K 版）"): ("C", "上代开源；无缓存价 → 本尺 6899 元；72B 不弱且贵，"
        "本用例无理由"),
}

DROP = {
    ("SiliconFlow", "Qwen3-VL-Embedding / Reranker-8B"): "embedding/reranker，无 agent 语义",
    ("SiliconFlow", "Qwen3-Embedding/Reranker 8B·4B·0.6B"): "同上",
    ("SiliconFlow", "bge-m3 / bge-reranker-v2-m3 / bge-large-zh·en-v1.5"): "同上",
    ("SiliconFlow", "Z-Image-Turbo / Wan2.2-I2V（图像/视频）"): "图像/视频生成，非 token 计价",
    ("MiniMax", "MiniMax-M3.1-Flash-Preview"): "仅订阅可用、不支持按量，预览模型未评测",
    ("火山方舟", "doubao-seed-character（≤32K）"): "角色对话专用，AA 不评；agent 工具循环未证实",
    ("阿里云百炼", "qwen3.7-max"): "AA 45 上代旗舰，本用例过强过贵（排除非能力判断）",
    ("小米MiMo", "MiMo-V2.5（输入<256K）"): "官方 2026-10-09 下线通知：10-14 自动转 V2.6-Flash、10-21 名称失效 → 不可采购",
    ("小米MiMo", "MiMo-V2.5-Pro（输入<256K）"): "同上（转 V2.6-Pro）。且 MiMo 全线最弱档＝V2.6-Flash AA 38，与主力同档 → "
        "本家族无「够弱」的压测选手，其开源 9B 蒸馏（MiMo-V2.6-Distill-Qwen-9B）需自托管、非按量 API",
}

# %% 套餐最低可购档：月费与额度读自 订阅套餐_对比.csv；覆盖范围取自该表「覆盖模型」列
# 不在覆盖内的型号，其「订阅等效价」属不可购数字 → 本表不采信，并在末尾显式列出（对既有表的修正）。
TIER = {  # 提供商: (最低档名, 覆盖关键词)
    "阿里云百炼": ("Lite", ("qwen", "deepseek", "kimi", "glm", "minimax")),
    "千问AI平台": ("Lite", ("qwen",)),
    "小米MiMo": ("Lite", ("V2.5",)),
    "MiniMax": ("Go", ("M3.1", "H3", "图像", "音频")),
    "智谱": ("Lite", ("GLM-5.3", "GLM-5-Turbo", "GLM-4.7")),
    "Kimi": ("Andante", ("K3", "K2.7", "K2.8")),
    "火山方舟": ("Lite", ("doubao-seed-2.0", "doubao-seed-2.1", "deepseek-v4", "glm-5.2", "kimi-k2.7", "minimax-m3")),
}
ALIAS = {"千问AI平台": "阿里云百炼"}  # 订阅表未为千问单列数值（档名「与百炼同构」）→ 读取同档的百炼数值
_L = sub.set_index(["提供商", "档位"])


def _cell(prov, col):
    p = ALIAS.get(prov, prov)
    return _L.loc[(p, TIER[p][0]), col]


LIST = {p: float(_cell(p, "月付刊例")) for p in TIER}
SALE = {p: (None if pd.isna(_cell(p, "限时价")) else float(_cell(p, "限时价"))) for p in TIER}
ENTRY = {p: min(v for v in (LIST[p], SALE[p]) if v is not None) for p in TIER}
QUOTA = {p: str(_cell(p, "额度")) for p in TIER}
COVERS = {  # 覆盖说明（原文摘要，判据用）
    "阿里云百炼": "Token Plan 覆盖 qwen3.8/3.7、deepseek-v4、kimi、glm、MiniMax-M2.5 等（Lite 11,500 Credits/月；"
                "无 API 生产自动化权限——压测若走 API 调用则此条是硬伤，须核实）",
    "千问AI平台": "同百炼",
    "小米MiMo": "MiMo Token Plan 覆盖列明「V2.5 系列 8 款」，而 V2.5/V2.5-Pro 名称 2026-10-21 失效并自动转 V2.6 "
                "按新版计价 → 池覆盖须随迁核实；若随迁至 V2.6-Flash，池内单价反而可能优于本表按量口径",
    "MiniMax": "M Plan Go 档列明 M3.1 Flash Preview/图像/音频；M3/M2.7 是否在池内需核",
    "智谱": "GLM Coding Plan 覆盖 GLM-5.3/Turbo/4.7；Lite 是 2,000 积分/5h＋10,000 积分/周的**限速**而非额度池，"
            "且官方条款「仅限官方指定工具」（Qoder 可否接入须核实）；非高峰按 50% 扣",
    "Kimi": "Kimi Coding Plan 覆盖 K3/K2.7 Code/K2.8 Preview 等；Andante 约 28M tokens/月；需预约、长期限购",
    "火山方舟": "Coding Plan Lite 1,200 次/5h＋18,000 次/月（实测约 250M tokens/月），覆盖 Doubao-Seed-2.0/2.1、"
              "DeepSeek-V4、GLM-5.2、Kimi-K2.7、MiniMax-M3（K3 暂不支持）；按「请求次数」计量 → 成本单位天然就是 agent 轮次。"
              "另有 Agent Plan（AFP 抵扣制，官方系数表见 build_rank_arbitration.py）：doubao-seed-2.0-mini 在压测负载口径下 "
              "≈0.054 元/M，比按量 0.068 便宜 21%，但 Small 是 40 元固定费 → 保本量 587M > 本用例 418M，压测仍应按量。"
              "两个须核实项：① Coding/Agent Plan 的适用工具列（OpenCode/OpenClaw/TRAE/Hermes/Codex CLI/Cline/Cursor 等）"
              "**未点名 Qoder**，须核条款；② 额度规则两个月动过三次（GLM-5.2 曾缩至 1/4）→ 容量是动态变量，不是固定参数",
}


def covered(prov, model):
    return any(k.lower() in model.lower() for k in TIER[prov][1])


# 最低档额度的单位成本折算。池的计量单位分两类：
#   ① 元按量等值池（Credits/相对量）→ 容量先折成「元按量」，折扣倍率 = 容量元按量 ÷ 起购月费；
#      百炼/千问 Credits↔人民币线性（假定，阿里不开放消耗账单）→ 倍率与型号无关；
#      MiMo 扣减表按型号逐 token 定价 → 容量须按该型号的 Credits 消耗折算；
#   ② tokens/次数池 → 容量是 token 数，单位成本与型号无关（便宜型号在②下毫无优势，在①下优势巨大）。
LOAD = (u[["未缓存输入", "缓存输入", "输出"]].sum() * (30 / len(u))).tolist()  # tokens/月
CRED_YUAN = float(cost[("阿里云百炼", "qwen3.8-max")]) / 360_000  # 元按量/Credit（实测：15 天用量＝1 份 Pro）
MIMO_DEDUCT = {  # 官方扣减表 Credits/token（未命中, 缓存, 输出）；V2.5 系 ≥256K 档翻倍
    "MiMo-V2.5（输入<256K）": (100.0, 2.0, 200.0),
    "MiMo-V2.5-Pro（输入<256K）": (300.0, 2.5, 600.0),
}


def _pool_bailian(model):
    return 11_500 * CRED_YUAN, "Credits↔人民币线性（假定）→ 折 11,500 Credit × 元/Credit"


def _pool_mimo(model):
    cred = sum(t * d for t, d in zip(LOAD, MIMO_DEDUCT[model]))
    return (41e8 / cred) * float(cost[("小米MiMo", model)]), f"本尺消耗 {cred / 1e8:.1f} 亿 Credits，Lite 41 亿＝{41e8 / cred:.1%} 尺"


POOL_YUAN = {  # 提供商 → (型号 → 容量_元按量, 折算口径)
    "阿里云百炼": _pool_bailian,
    "千问AI平台": _pool_bailian,
    "小米MiMo": _pool_mimo,
    "MiniMax": lambda m: (float(cost[("MiniMax", "MiniMax-M3（输入≤512K）")]) / 12, "Go＝Build 基准量 1/12，Build≈本尺 M3"),
}
POOL_TOKENS = {"Kimi": 28.0, "火山方舟": 250.0}  # M tokens/月（方舟另有 18,000 次/月）
PLAN_BLOCK = {  # 套餐存在、但对本用例不可购/不可用的硬障碍（覆盖内的型号也不发行）
    "智谱": "GLM Coding Plan 官方条款「仅限官方指定工具」＋Lite 是限速（2,000 积分/5h）而非额度池；"
            "Qoder 是否在其许可工具列表内须核实——未确认前按不可购处理，不发「套餐最低档」行",
}


def plan_unit_cost(prov, model, payg_mix):
    """返回 (套餐混合单价 元/M, 池顶量 M tokens, 容量说明)。不可折算/不可购 → None。"""
    if prov in PLAN_BLOCK:
        return None, None, PLAN_BLOCK[prov]
    if prov in POOL_TOKENS:
        cap = POOL_TOKENS[prov]
        return ENTRY[prov] / cap, cap, f"{QUOTA[prov]}；按 tokens 计 → {cap:g}M/月容量，与型号无关"
    if prov in POOL_YUAN:
        cap_yuan, how = POOL_YUAN[prov](model)
        return payg_mix * ENTRY[prov] / cap_yuan, cap_yuan / payg_mix, f"{QUOTA[prov]}；容量折 {cap_yuan:.1f} 元按量（{how}）"
    return None, None, "无套餐"


VARIANT_SUFFIX = ("（低优/批量）", "（高档）", "（>512K档）", "（32-128K档）", "（>128K档）")  # 模拟月成本_按量.csv 里的并行计价变体名后缀


def judge(k, mode):
    if k in FREE_TIER:
        t, b, _ = FREE_TIER[k]
        return t, b
    if k in DROP:
        return "D", DROP[k]
    if k in FEAS:
        return FEAS[k]
    for s in VARIANT_SUFFIX:  # 阶梯/低优等计价变体 → 沿用基础型号判据（同模型，只是价档不同）
        if k[1].endswith(s) and (k[0], k[1][: -len(s)]) in FEAS:
            t, b = FEAS[(k[0], k[1][: -len(s)])]
            return t, f"同基础型号判据（{s}）：" + b
    return "C", "未逐行判定（型号未拆或候选带之外）→ 须先小任务试跑"


# %% 组表：按量 / 套餐最低档 / 免费 三种可购形态各成一行
QWEN = lambda m: "Qwen" in m or m.startswith("qwen")
FREE_TIER = {(prov, model): (tier, basis, src) for prov, model, src, tier, basis in FREE}
rows = []
nosale = {}
for (prov, model), p in cost_payg.items():
    mix = float(p) / U_M
    rows.append(dict(提供商=prov, 模型=model, 形态="按量", 混合单价=round(mix, 4), 起购固定费=0.0,
                     保本量M=float("nan"), 池顶量M=float("nan"), 单位成本比=1.0,
                     本尺月价=round(float(p), 1), 容量与口径="按量无最低消费、无上限"))
    if prov in TIER and covered(prov, model):
        uc, top, cap_txt = plan_unit_cost(prov, model, mix)
        if uc is not None:
            rows.append(dict(提供商=prov, 模型=model, 形态="套餐最低档", 混合单价=round(uc, 4),
                             起购固定费=ENTRY[prov], 保本量M=round(ENTRY[prov] / mix, 1), 池顶量M=round(top, 1),
                             单位成本比=round(mix / uc, 2), 本尺月价=round(float(p), 1),
                             容量与口径=f"{TIER[prov][0]} 档 刊例{LIST[prov]:g}"
                                        + (f"/限时{SALE[prov]:g}" if SALE[prov] else "") + f"：{cap_txt}"))
        else:
            nosale.setdefault(f"{prov} / {model}", cap_txt)
for (prov, model), (tier, basis, src) in FREE_TIER.items():
    rows.append(dict(提供商=prov, 模型=model, 形态="免费档", 混合单价=0.0, 起购固定费=0.0,
                     保本量M=float("nan"), 池顶量M=float("nan"), 单位成本比=float("nan"),
                     本尺月价=0.0, 容量与口径=f"{src}（限流）"))

df = pd.DataFrame(rows)
key = list(zip(df["提供商"], df["模型"]))
df["上下文"] = [str(ctx.get(k, "未核")) for k in key]
df["智能指数"] = [score["智能指数"].get(k) for k in key]
df["可比性"] = [score["可比性"].get(k, "无行") for k in key]
df["冷会话单价"] = [cold.get(k) if m == "按量" else None for k, m in zip(key, df["形态"])]
jd = [judge(k, m) for k, m in zip(key, df["形态"])]
df["压测档位"] = [x[0] for x in jd]
df["判据"] = [x[1] for x in jd]
df.loc[df["提供商"].isin(TIER) & (df["形态"] == "套餐最低档"), "判据"] += "；覆盖：" + df["提供商"].map(COVERS)
df.loc[df["形态"] == "免费档", "判据"] += "；" + FREENOTE
df["家族加分"] = ["＋Qwen 亲和加分" if QWEN(m) and t != "D" else ""
                  for (_, m), t in zip(key, df["压测档位"])]

# %% 落盘：入榜（A/B/C，按混合单价升序）；D 档单列备查
excl = df[df["压测档位"] == "D"].sort_values("混合单价").copy()
live = df[df["压测档位"] != "D"].sort_values(["混合单价", "压测档位", "起购固定费"]).reset_index(drop=True)
live.insert(0, "排名", range(1, len(live) + 1))
live.insert(1, "核实日期", "2026-10-12")
cols = ["排名", "核实日期", "提供商", "模型", "形态", "混合单价", "冷会话单价", "起购固定费", "保本量M", "池顶量M", "单位成本比",
        "本尺月价", "上下文", "智能指数", "可比性", "压测档位", "家族加分", "容量与口径", "判据"]
live[cols].to_csv(OUT_DIR / "选型_弱模型压测_成本排名.csv", index=False, encoding="utf-8-sig")

plan_tbl = pd.read_csv(OUT_DIR / "模拟月成本_订阅等效.csv")
bad = [(p, m, v) for p, m, v in zip(plan_tbl["提供商"], plan_tbl["模型"], plan_tbl["订阅等效月价"])
       if p in TIER and not covered(p, m)]

# %% 输出
print(f"月化用量合计 {U_M:.1f} 百万 tokens。混合单价＝元/百万 tokens（本负载结构，与压测量无关）\n")
show = ["排名", "提供商", "模型", "形态", "混合单价", "起购固定费", "保本量M", "池顶量M", "单位成本比", "压测档位", "家族加分"]
print(live[live["混合单价"] <= 0.35][show].to_string(index=False, max_colwidth=30))
gap = live[(live["判据"].str.startswith("未逐行")) & (live["混合单价"] <= 0.35)]
print(f"\n[自检] 便宜带（≤0.35 元/M）内未逐行判定的行数 = {len(gap)}")
if len(gap):
    print(gap[["提供商", "模型", "形态", "混合单价"]].to_string(index=False))
print(f"\n入榜 {len(live)} 行／剔除 {len(excl)} 行；档位分布 {live['压测档位'].value_counts().to_dict()}")
print("\n各渠道最低可购档（该渠道套餐里单位成本最廉的型号；单位成本比>1 即套餐每 token 比按量便宜）：")
one = live[live["形态"] == "套餐最低档"].sort_values(["混合单价", "单位成本比"]).groupby("提供商").head(1)
print(one[["提供商", "模型", "起购固定费", "混合单价", "保本量M", "池顶量M", "单位成本比"]]
      .to_string(index=False, max_colwidth=46))
print("\nD 档剔除：")
print(excl[["提供商", "模型", "形态", "判据"]].drop_duplicates(["提供商", "模型"])
      .to_string(index=False, max_colwidth=44))
print(f"\n套餐不覆盖 → 其「订阅等效价」不可购（修正既有 模拟月成本_订阅等效.csv 共 {len(bad)} 行）：")
seen = set()
for p, m, v in bad:
    if m not in seen:
        seen.add(m)
        print(f"  {p} / {m}（等效 {v:g} 元）——{COVERS[p]}")
print(f"\n套餐覆盖但在本用例下不可用（共 {len(nosale)} 行，故不发「套餐最低档」）：")
for k, why in sorted(nosale.items())[:12]:
    print(f"  {k}——{why}")
if len(nosale) > 12:
    print(f"  …其余 {len(nosale) - 12} 行同因")
