"""flash 渠道互换（backlog 20261012-173500-leader，用户裁决 2026-10-12）。

flash 档**永久**改指 BYOK ``Qwen-3.8-Flash``，内置 ``Qwen3.8-Flash``（限时免费但限速）
降为 ``flash_fallback`` 兜底位。互换必须同时落在三处并**同源**：

1. 跟踪面 ``orchestration/state/registry.json`` 的 ``models``；
2. 代码模式表 :data:`registry.DEFAULT_MODEL_PATTERNS`（「名→档」的唯一机器可读口径）；
3. README §3.5 渠道策略文案。

漏改模式表会怎样：``refresh_models`` 每个 drain 都跑，跟踪面**缺**的档按模式表补进运行态
面，跟踪面**写了**的档一旦被解析值顶到就报 ``shadowed``（漂移被点名，永不生效的配置）。
两组用例分别钉这两条通路；反向用例用「互换前的模式表」验证断言确实有牙。

quota 换档回退逻辑（backlog 20261010-flash-fallback）**不动**，见
``test_flash_fallback.py``——互换只是让 ``<tier>_fallback`` 的自然语义变成
「BYOK 按量耗尽 → 内置免费兜底」。
"""

from __future__ import annotations

import json
import re

import pytest

from pysci.skills.orchestration.tools import registry

# --list-models 实测格式（同 test_model_refresh.CATALOG；三行名分别是内置、BYOK flash、BYOK max）
CATALOG = (
    "MODEL\n"
    "Qwen3.8-Flash\n"
    "Qwen-3.8-Flash (ed6f5996-0992-479f-86a8-3f29ce0e794c)\n"
    "Qwen-3.8-Max (982ae403-bc5e-4f74-996d-04a9cd66f933)\n"
)
BUILTIN_FLASH = "Qwen3.8-Flash"
BYOK_FLASH_NAME = "Qwen-3.8-Flash"
BYOK_FLASH_UUID = "ed6f5996-0992-479f-86a8-3f29ce0e794c"
MAX_UUID = "982ae403-bc5e-4f74-996d-04a9cd66f933"

#: 导入期捕获的真实跟踪配置面（下面夹具会把 ``registry.REGISTRY_PATH`` 搬到 tmp）
REAL_REGISTRY_PATH = registry.REGISTRY_PATH

PRE_SWAP_PATTERNS = {
    "max": "Qwen-3.8-Max",
    "flash": BUILTIN_FLASH,
    "flash_fallback": BYOK_FLASH_NAME,
    "minimum": BUILTIN_FLASH,
}

#: 导入期锚定真本配置面（夹具会搬走 ``registry.REGISTRY_PATH``）。
REAL_REGISTRY_PATH = registry.REGISTRY_PATH


@pytest.fixture
def iso_reg(tmp_path, monkeypatch):
    reg_file = tmp_path / "registry.json"
    monkeypatch.setattr(registry, "REGISTRY_PATH", reg_file)
    return reg_file, tmp_path / registry.RUNTIME_NAME


def _seed(reg_file, models: dict, patterns: dict | None = None) -> bytes:
    data = {"version": 1, "exe": None, "models": models, "members": {}, "checks": {}}
    if patterns is not None:
        data["model_patterns"] = patterns
    reg_file.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return reg_file.read_bytes()


def _tracked_models() -> dict[str, str]:
    """真本 registry.json 的 models（读侧实际生效的跟踪值）。

    路径在**导入期**捕获：本文件的夹具把 ``registry.REGISTRY_PATH`` 搬到 tmp（那是
    refresh_models 的写面），而这里要读的是随 main 入库的那份配置。
    """
    data = json.loads(REAL_REGISTRY_PATH.read_text(encoding="utf-8"))
    return data.get("models", {})


def _fake_output(monkeypatch, text=CATALOG):
    monkeypatch.setattr(registry, "list_models_output", lambda *, timeout_s=60: text)


def _runtime_face(runtime_file):
    face = json.loads(runtime_file.read_text(encoding="utf-8"))
    return face.get(registry.RUNTIME_MODELS_KEY, {})


# ---------------------------------------------------------------------------
# 1. 跟踪面：值确实互换了
# ---------------------------------------------------------------------------
def test_tracked_registry_models_are_swapped():
    models = _tracked_models()
    assert models["flash"] == BYOK_FLASH_UUID, (
        "flash 主渠道=BYOK（用户裁决 2026-10-12）"
    )
    assert models["flash_fallback"] == BUILTIN_FLASH, "内置免费渠道降为兜底位"
    assert models["max"] == MAX_UUID and models["minimum"] == BUILTIN_FLASH


# ---------------------------------------------------------------------------
# 2. 三处同源：跟踪值 ↔ 模式表（自愈既不改写、也不该报漂移）
# ---------------------------------------------------------------------------
def test_tracked_values_agree_with_pattern_table(iso_reg, monkeypatch):
    """以真 registry.json 的 models 为跟踪面跑一次自愈 → changed/shadowed 皆空。

    这是「drain 启动后映射不被刷回」的阳性判据：任何一侧单独回退（改了 registry
    没改模式表，或反之）都会让本用例变红——模式表旧 ⇒ 解析值顶到人值上报 shadowed，
    registry 旧 ⇒ 同样报 shadowed，且被缺档通路（下条用例）直接刷回。
    """
    reg_file, runtime_file = iso_reg
    before = _seed(reg_file, _tracked_models())
    _fake_output(monkeypatch)

    res = registry.refresh_models()

    assert reg_file.read_bytes() == before, "自愈绝不写 git 跟踪的配置面"
    assert res["changed"] == {} and res["missing"] == []
    assert res["shadowed"] == {}, (
        f"跟踪值与模式表已同源，不该有漂移被点名：{res['shadowed']}"
    )
    assert not runtime_file.exists() or _runtime_face(runtime_file) == {}, (
        "四档都在跟踪面 → 没有档交给机器，运行态面不该多出值"
    )


def test_machine_face_resolves_byok_to_flash_tier(iso_reg, monkeypatch):
    """跟踪面把档交还机器（删键）时，自愈补的必须是互换后的新对应关系。"""
    reg_file, runtime_file = iso_reg
    _seed(reg_file, {})
    _fake_output(monkeypatch)

    registry.refresh_models()

    assert _runtime_face(runtime_file) == {
        "max": MAX_UUID,
        "flash": BYOK_FLASH_UUID,
        "flash_fallback": BUILTIN_FLASH,
        "minimum": BUILTIN_FLASH,
    }


def test_pre_swap_patterns_would_flip_the_mapping(iso_reg, monkeypatch):
    """反向阳性：用互换前的模式表（``model_patterns`` 覆盖模拟），跟踪面立即被报成漂移。

    证明上面两条断言真有牙——「只改 registry.json 不动模式表」不是纸面担忧。
    """
    reg_file, _ = iso_reg
    _seed(reg_file, _tracked_models(), patterns=PRE_SWAP_PATTERNS)
    _fake_output(monkeypatch)

    res = registry.refresh_models()

    assert res["shadowed"] == {
        "flash": {"tracked": BYOK_FLASH_UUID, "resolved": BUILTIN_FLASH},
        "flash_fallback": {"tracked": BUILTIN_FLASH, "resolved": BYOK_FLASH_UUID},
    }


# ---------------------------------------------------------------------------
# 3. README §3.5 与代码模式表同源（文案不许领先或落后于代码）
# ---------------------------------------------------------------------------
def test_readme_pattern_clause_matches_code_table():
    text = (registry.PROJECT_ROOT / "orchestration" / "README.md").read_text(
        encoding="utf-8"
    )
    m = re.search(r"^### 3\.5 .*?(?=^### |\Z)", text, re.S | re.M)
    assert m, "README 找不到 §3.5（文案同源断言的锚点丢了）"
    clause = re.sub(r"\s+", " ", m.group(0))

    documented: set[tuple[str, str]] = set()
    for name, tier, extra in re.findall(
        r"(Qwen3\.8-Flash|Qwen-3\.8-Flash|Qwen-3\.8-Max)→([a-z_]+)(?: 与 ([a-z_]+))?",
        clause,
    ):
        documented.add((tier, name))
        if extra:
            documented.add((extra, name))
    expected = set(registry.DEFAULT_MODEL_PATTERNS.items())
    assert documented == expected, (
        f"README §3.5 的模式表描述与 DEFAULT_MODEL_PATTERNS 不同源：只文档有="
        f"{documented - expected}，只代码有={expected - documented}"
    )
