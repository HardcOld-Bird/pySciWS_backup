"""档位链的同源断言（backlog 20261012-173500-leader 渠道互换 → 20261012-211500-leader 链化）。

本文件的血缘：173500 那跳把 flash 首席从内置换成 BYOK，漏改「名→档」规则表就会被下一次
``refresh-models`` 刷回旧渠道（#74 教训）。211500 那跳把每档升级成**有序回退链**
（Mimo 主力 → BYOK Qwen 回退 → 内置兜底），同源要求随之变成**三处**：

1. 跟踪面 ``orchestration/state/registry.json`` 的 ``models``（席位 UUID 链）；
2. 代码链模板 :data:`registry.DEFAULT_MODEL_CHAINS`（「哪个名属于哪个档、谁在前」的唯一
   机器可读口径）；
3. README §3.5 渠道策略文案。

漏改任一侧的实发症状都钉在这里：只改 registry 不改模板 ⇒ 自愈把解析链报成 ``shadowed``
（人值被点名，永不生效）；只改模板不改 registry ⇒ 下一次 drain 直接按新链顶掉裁决渠道。
反向用例（``test_pre_chain_tracking_face_is_named_as_drift``）用**链化前**的单值跟踪面验证
断言确有牙。

链化前的文件名叫 ``test_flash_channel_swap.py``——README §3.5 的教训登记按这个名字索引本
组断言，故保留文件名，内容口径已升级为链。
"""

from __future__ import annotations

import json
import re

import pytest

from pysci.skills.orchestration.tools import presets, registry

# --list-models 实测格式（2026-10-12 22:xx 真实账户目录，含用户新增的 Mimo 自定义模型）
MIMO_PRO_UUID = "99127e47-626d-421f-b0cb-02ffdbf1ada7"
MIMO_FLASH_UUID = "017036af-b630-498a-bfea-883a20c67e33"
BYOK_MAX_UUID = "982ae403-bc5e-4f74-996d-04a9cd66f933"
BYOK_FLASH_UUID = "ed6f5996-0992-479f-86a8-3f29ce0e794c"
BUILTIN_FLASH = "Qwen3.8-Flash"

CATALOG = (
    "MODEL\n"
    f"Mimo-V2.6-Pro ({MIMO_PRO_UUID})\n"
    f"Mimo-V2.6-Flash ({MIMO_FLASH_UUID})\n"
    f"Qwen-3.8-Max ({BYOK_MAX_UUID})\n"
    f"Qwen-3.8-Flash ({BYOK_FLASH_UUID})\n"
    f"{BUILTIN_FLASH}\n"
    "Qwen3.8-Max\n"  # 内置同名模型：永不入链（2026-10-10 烧穿订阅 credit 的教训）
)

#: 链化前的跟踪面（单值写法 + 已废的 ``flash_fallback`` 键），反向用例用它模拟旧配置。
PRE_CHAIN_MODELS = {
    "max": BYOK_MAX_UUID,
    "flash": BYOK_FLASH_UUID,
    "flash_fallback": BUILTIN_FLASH,
    "minimum": BUILTIN_FLASH,
}

#: 导入期锚定真本配置面（夹具会把 ``registry.REGISTRY_PATH`` 搬到 tmp）。
REAL_REGISTRY_PATH = registry.REGISTRY_PATH


@pytest.fixture
def iso_reg(tmp_path, monkeypatch):
    reg_file = tmp_path / "registry.json"
    monkeypatch.setattr(registry, "REGISTRY_PATH", reg_file)
    return reg_file, tmp_path / registry.RUNTIME_NAME


def _tracked_face() -> dict:
    """真本 registry.json 的 models + model_chains（读侧实际生效的跟踪配置）。

    路径在**导入期**捕获：夹具把 ``registry.REGISTRY_PATH`` 搬到 tmp（那是 refresh_models
    的写面），而这里要读的是随 main 入库的那份配置。
    """
    data = json.loads(REAL_REGISTRY_PATH.read_text(encoding="utf-8"))
    return {
        "models": data.get("models", {}),
        "model_chains": data.get("model_chains", {}),
    }


def _seed(reg_file, models: dict, chains: dict | None = None) -> bytes:
    data = {"version": 1, "exe": None, "models": models, "members": {}, "checks": {}}
    if chains is not None:
        data["model_chains"] = chains
    reg_file.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return reg_file.read_bytes()


def _fake_output(monkeypatch, text=CATALOG):
    monkeypatch.setattr(registry, "list_models_output", lambda *, timeout_s=60: text)


def _runtime_face(runtime_file):
    face = json.loads(runtime_file.read_text(encoding="utf-8"))
    return face.get(registry.RUNTIME_MODELS_KEY, {})


def _doc_corpus() -> str:
    """文档语料（README v3.0 拆分后口径）：references/legacy-readme-v2.2.md（v2.2 原文留档）
    + README.md（v3.0 正文）+ references/*.md（细节块）——doc-face 断言据此判「文档面声明 X」。
    """
    root = registry.PROJECT_ROOT / "orchestration"
    parts: list[str] = []
    legacy = root / "references" / "legacy-readme-v2.2.md"
    if legacy.exists():
        parts.append(legacy.read_text(encoding="utf-8"))
    parts.append((root / "README.md").read_text(encoding="utf-8"))
    refs = root / "references"
    if refs.is_dir():
        for f in sorted(refs.glob("*.md")):
            if f.name != "legacy-readme-v2.2.md":
                parts.append(f.read_text(encoding="utf-8"))
    return (chr(10) * 2).join(parts)


def _readme_clause() -> str:
    text = _doc_corpus()
    m = re.search(r"^### 3\.5 .*?(?=^### |\Z)", text, re.S | re.M)
    assert m, "README 找不到 §3.5（文案同源断言的锚点丢了）"
    return re.sub(r"\s+", " ", m.group(0))


# ---------------------------------------------------------------------------
# 1. 跟踪面：三档三席，排序正确（success_criteria 第一条）
# ---------------------------------------------------------------------------
def test_tracked_registry_is_ordered_chains():
    """档位退役后的跟踪面（2026-10-12 22:4x）：models 只留独立的 minimum，
    main/pro 席位表落 ``presets`` 块，成员条目不再带 ``model_tier``。"""
    data = json.loads(REAL_REGISTRY_PATH.read_text(encoding="utf-8"))
    models = data.get("models", {})
    assert set(models) == {"minimum"}, f"max/flash 档概念已退役：{sorted(models)}"
    assert models["minimum"] == [MIMO_FLASH_UUID, BUILTIN_FLASH], (
        "压测最低成本档独立不随预设：Qwen BYOK 不进链（按量额度宝贵）"
    )
    assert "flash_fallback" not in models, "旧回退键已并入链，不该再留第二张表"
    face = data.get("presets", {})
    assert {"qwen", "mimo"} <= set(face), "账号级 BYOK 两预设齐备（deepseek 见下）"
    # 席链断言**只锁形态不锁此刻读数**（backlog 20261013-143233-devops 缺陷 2）：
    # 钉死整串会把**授权的配置变更**误报成代码回归——路由 v3 在 main/pro 链里插了
    # DeepSeek 席（`987f80c`）之后，本测试就在 main 上恒红。中间席可有授权覆盖，
    # 首尾两端才是出厂不变量。
    assert _chain_shape_ok(face["mimo"]["main"], MIMO_FLASH_UUID, BUILTIN_FLASH), (
        f"mimo.main 形态非法：{face['mimo']['main']}"
    )
    assert _chain_shape_ok(face["mimo"]["pro"], MIMO_PRO_UUID, BUILTIN_FLASH), (
        f"mimo.pro 形态非法：{face['mimo']['pro']}"
    )
    assert _chain_shape_ok(face["qwen"]["main"], BYOK_FLASH_UUID, BUILTIN_FLASH), (
        f"qwen.main 形态非法：{face['qwen']['main']}"
    )
    assert _chain_shape_ok(face["qwen"]["pro"], BYOK_MAX_UUID, BUILTIN_FLASH), (
        f"qwen.pro 形态非法：{face['qwen']['pro']}"
    )
    assert "deepseek" not in face, (
        "deepseek 席已回填（UUID 首席+内置同名回退，2026-10-13 用户供 Key）落代码"
        "出厂表——跟踪面不再留空链占位（留了=整链覆盖置空、预设反而不可激活）"
    )
    assert all("model_tier" not in m for m in data.get("members", {}).values()), (
        "成员条目的档位字段随档位模式一起退役"
    )


def _chain_shape_ok(chain: list[str], head: str, tail: str) -> bool:
    """跟踪席链的**授权形态**：首席=预设出厂首席、末席=内置兜底、无重复。

    为什么不锁「此刻读数」：见用例内注释（教训「回归测试别锁运行面当前值」）。
    """
    return (
        bool(chain)
        and chain[0] == head
        and chain[-1] == tail
        and len(set(chain)) == len(chain)
    )


def test_chain_shape_predicate_bites_on_unauthorized_face():
    """**阳性对照**：形态判据必须能咬住非授权面——否则它只是恒真的装饰。

    缺了这一支，`_chain_shape_ok` 写成 `return True` 也全绿（「阳性用例不红=没测到」）。
    """
    assert _chain_shape_ok(
        [MIMO_FLASH_UUID, "1a54cc57-e614-4521-80cc-447de31dc896", BUILTIN_FLASH],
        MIMO_FLASH_UUID,
        BUILTIN_FLASH,
    ), "中间插授权席（路由 v3 的 DeepSeek）应判合规"
    assert not _chain_shape_ok(
        ["foreign-id", BUILTIN_FLASH], MIMO_FLASH_UUID, BUILTIN_FLASH
    )
    assert not _chain_shape_ok([MIMO_FLASH_UUID, "x"], MIMO_FLASH_UUID, BUILTIN_FLASH)
    assert not _chain_shape_ok(
        [MIMO_FLASH_UUID, MIMO_FLASH_UUID, BUILTIN_FLASH],
        MIMO_FLASH_UUID,
        BUILTIN_FLASH,
    )
    assert not _chain_shape_ok([], MIMO_FLASH_UUID, BUILTIN_FLASH)


# ---------------------------------------------------------------------------
# 2. 三处同源：跟踪链 ↔ 链模板（自愈既不改写、也不该报漂移）
# ---------------------------------------------------------------------------
def test_tracked_values_agree_with_chain_template(iso_reg, monkeypatch):
    """以真 registry.json 的 models/model_chains 为跟踪面跑一次自愈 → 四个读数皆空。

    这是「``refresh-models`` 后不被刷回」的阳性判据：模板旧 ⇒ 解析链顶到人值上报
    ``shadowed``；registry 旧 ⇒ 同样报 ``shadowed``；账户缺名 ⇒ 报 ``missing``。
    """
    reg_file, runtime_file = iso_reg
    face = _tracked_face()
    before = _seed(reg_file, face["models"], chains=face["model_chains"])
    _fake_output(monkeypatch)

    res = registry.refresh_models()

    assert reg_file.read_bytes() == before, "自愈绝不写 git 跟踪的配置面"
    assert res["changed"] == {}, f"跟踪链已同源，自愈不该改任何档：{res['changed']}"
    assert res["missing"] == [], f"链模板里的名都在账户目录：{res['missing']}"
    assert res["shadowed"] == {}, (
        f"跟踪值与链模板已同源，不该有漂移被点名：{res['shadowed']}"
    )
    assert res["conflicts"] == []
    assert not runtime_file.exists() or _runtime_face(runtime_file) == {}, (
        "三档都在跟踪面 → 没有档交给机器，运行态面不该多出值"
    )


def test_mimo_names_resolve_to_their_tiers(iso_reg, monkeypatch):
    """名→档 规则表的阳性断言（规格 2）：Mimo 两名各归其档，不被自愈塞错地方。"""
    reg_file, runtime_file = iso_reg
    _seed(reg_file, {})  # 跟踪面空缺 = 把档交给机器
    _fake_output(monkeypatch)

    registry.refresh_models()

    assert _runtime_face(runtime_file) == {
        "minimum": [MIMO_FLASH_UUID, BUILTIN_FLASH],
    }, "档位退役：刷新只代管 minimum 模板"
    rules = registry.model_tier_rules()
    assert rules["Mimo-V2.6-Flash"] == {"minimum"}
    assert rules["Qwen3.8-Flash"] == {"minimum"}
    assert "Mimo-V2.6-Pro" not in rules, (
        "pro 席归预设表管（presets.PRESETS），不进刷新规则表"
    )
    assert "Qwen3.8-Max" not in rules, (
        "内置同名 Qwen3.8-Max 永不入链：它烧订阅 credit 且 envelope 低报为 0"
        "（2026-10-10 实战烧穿额度）"
    )
    # 规格 2 的正名断言换了落点：Mimo 两名的归属改由预设表钉住
    assert presets.PRESETS["mimo"].pro[0] == "Mimo-V2.6-Pro"
    assert presets.PRESETS["mimo"].main[0] == "Mimo-V2.6-Flash"
    assert presets.PRESETS["qwen"].main[0] == "Qwen-3.8-Flash"


def test_pre_chain_tracking_face_is_named_as_drift(iso_reg, monkeypatch):
    """反向阳性：跟踪面还停在链化前的单值写法 → 三档全被报成漂移（断言确有牙）。"""
    reg_file, _ = iso_reg
    _seed(reg_file, PRE_CHAIN_MODELS)
    _fake_output(monkeypatch)

    res = registry.refresh_models()

    assert set(res["shadowed"]) == {"minimum"}, (
        "刷新面只剩 minimum；旧 max/flash 键不进刷新、不被代管"
    )
    assert res["shadowed"]["minimum"] == {
        "tracked": BUILTIN_FLASH,
        "resolved": [MIMO_FLASH_UUID, BUILTIN_FLASH],
    }
    data = json.loads(reg_file.read_text(encoding="utf-8"))
    assert data["models"]["max"] == BYOK_MAX_UUID
    assert data["models"]["flash"] == BYOK_FLASH_UUID, (
        "旧档键的跟踪值一字不改地留在配置面（刷新只读不代写）"
    )


# ---------------------------------------------------------------------------
# 3. 旧写兼容读（规格 1 的向后兼容半边）
# ---------------------------------------------------------------------------
def test_legacy_single_value_and_fallback_key_still_read(iso_reg):
    """无 ``presets`` 块的旧跟踪面照样读：单值＝一席链、``<tier>_fallback`` 并入链尾、
    旧档名 flash/max 归一到 main/pro；没配的 role 落**预设出厂表**（预设总给得出席）。"""
    reg_file, _ = iso_reg
    _seed(reg_file, {"flash": "solo-flash", "flash_fallback": "legacy-backup"})

    reg = registry.Registry.load()

    assert reg.model_chain("flash") == ["solo-flash", "legacy-backup"]
    assert reg.model_chain("main") == ["solo-flash", "legacy-backup"], "flash=main 别名"
    assert reg.models["main"] == "solo-flash", "首席平读面不变（老消费方不受惊）"
    assert "flash" not in reg.models, "旧档键归一进角色后不再双写"
    assert reg.model_chain("max") == list(presets.active().chain("pro")), (
        "跟踪面没写的 role → 预设出厂表（pro 席链）；出厂面不猜账号 UUID、"
        "但预设总给得出席——空串只留给预设不可激活时的兜底"
    )


def test_legacy_fallback_already_in_chain_not_duplicated(iso_reg):
    """回退键与链内某席同名 → 保序去重（同一席在链里出现两次只会白烧一跳）。"""
    reg_file, _ = iso_reg
    _seed(reg_file, {"flash": ["a-seat", "b-seat"], "flash_fallback": "b-seat"})

    assert registry.Registry.load().model_chain("flash") == ["a-seat", "b-seat"]


# ---------------------------------------------------------------------------
# 4. README §3.5 与代码链模板同源（文案不许领先或落后于代码）
# ---------------------------------------------------------------------------
def test_readme_clause_matches_preset_tables():
    """README §3.5 与代码预设表/链模板同源：三预设构成串、席位名、minimum 独立链。"""
    clause = _readme_clause()

    # 仲裁团构成串与 presets.PRESETS[*].jury 逐预设同源
    for name, table in presets.PRESETS.items():
        want = f"{name}→{'+'.join(table.jury)}"
        assert want in clause, (
            f"README §3.5 缺「{want}」——构成串与 presets.PRESETS 不同源"
        )

    # 预设席位名逐个落文档（改出厂表不改 README 时这里变红）
    for name in ("Qwen-3.8-Flash", "Qwen-3.8-Max", "Mimo-V2.6-Flash", "Mimo-V2.6-Pro"):
        assert name in clause, f"README §3.5 缺席位名 {name}"

    # minimum 独立链：名字与 DEFAULT_MODEL_CHAINS 同源
    min_chain = registry.DEFAULT_MODEL_CHAINS["minimum"]
    assert all(seat in clause for seat in min_chain), (
        f"README §3.5 缺 minimum 链席位 {min_chain}"
    )
    assert "独立、不随预设走" in clause, "minimum 与预设解耦的语义必须写明"
    assert "data/pysci_settings.toml" in clause, "真源文件名必须逐字可检索"
    assert "--model-tier pro|main" in clause, "角色覆盖旗标的合法值写进文档"


def test_readme_documents_fallback_and_alert():
    """规格 6 的观察面锚点：文案要讲清「回退触发」与「回退必告警」，缺一就是文档落后的信号。"""
    clause = _readme_clause()

    for anchor in (
        "有序回退链",
        "回退触发",
        "启动即失败",
        "全局停止",
        "回退必告警",
        "channel-events.jsonl",
        "降级粘滞窗",
    ):
        assert anchor in clause, f"README §3.5 缺「{anchor}」——机制改了文案没跟上"
    assert "档位回退" in clause, (
        "调用者面告警文案（规格 4a）须逐字留在文档里，避免各处自立口径"
    )
