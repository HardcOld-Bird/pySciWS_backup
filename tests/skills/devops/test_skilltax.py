"""平台注入税工具 skilltax 的纯函数回归（backlog 20261010-plugin-tax-probe）。

锁定 doctor「平台技能税关停核查」哨兵所依赖的解析/判定/落配置逻辑：
- ``classify``/``parse_listing``：清单附件正文 → 逐条技能（含来源归类与折行描述）；
- ``recommended_disabled``：应关平台技能 = 内置 11 + 插件 5 = 16 项，减去 KEEP_ON_PODS；
- ``disabled_gap``/``apply_disabled``：缺口检测与幂等落配置，且不误伤自有技能/其他键，
  settings 缺失或非法 JSON 时抛可读错误（不静默新建只读层文件）。
以 tmp_path 造 pod，不触碰真实 PODS_ROOT。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pysci.skills.devops.tools import skilltax


def _mkpod(root: Path, settings: dict | None, *, own: list[str] | None = None) -> Path:
    """在 tmp_path 下造一个 pod：写 settings.json（None 则不建文件），并建自有技能目录。"""
    if settings is not None:
        f = root / ".qoder" / "settings.json"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(settings, ensure_ascii=False), encoding="utf-8")
    for name in own or []:
        d = root / ".qoder" / "skills" / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "SKILL.md").write_text(f"name: {name}\n", encoding="utf-8")
    return root


def test_classify_by_source():
    assert skilltax.classify("qoder-sites:sites-building") == "插件"
    assert skilltax.classify("loop") == "平台"
    assert skilltax.classify("security-scan") == "平台"
    assert skilltax.classify("theoretical-computation") == "自有"


def test_parse_listing_folds_wrapped_desc():
    content = "- alpha: first desc\n- beta: second\n  continued desc"
    lines = skilltax.parse_listing(content)
    got = {sk.name: sk for sk in lines}
    assert set(got) == {"alpha", "beta"}
    assert got["alpha"].desc == "first desc"
    assert got["alpha"].source == "自有"
    # 折行描述并入前一条，不是新起一条
    assert got["beta"].desc == "second\n  continued desc"
    assert got["alpha"].bytes == len(b"- alpha: first desc") + 1


def test_recommended_disabled_is_all_platform_plus_plugin():
    keep = set(skilltax.KEEP_ON_PODS)
    want = sorted(
        (set(skilltax.PLATFORM_SKILLS) | set(skilltax.PLATFORM_PLUGIN_SKILLS)) - keep
    )
    got = skilltax.recommended_disabled()
    assert got == want
    assert "security-scan" in got
    assert "qoder-create-plugin:create-plugin" in got
    # 内置 11 + 插件 5 = 16，与 docstring/README §1.1 登记一致
    assert len(skilltax.PLATFORM_SKILLS) == 11
    assert len(skilltax.PLATFORM_PLUGIN_SKILLS) == 5
    assert len(got) == 16
    assert got == want


def test_gap_then_apply_is_idempotent_and_preserves_other_keys(tmp_path):
    pod = _mkpod(
        tmp_path / "theory",
        {
            "hooks": {
                "Stop": [{"hooks": [{"type": "command", "command": "node x.mjs"}]}]
            }
        },
        own=["theoretical-computation"],
    )
    # 未配置 → 缺口即全量 16 项
    assert skilltax.disabled_gap(pod) == skilltax.recommended_disabled()

    assert skilltax.apply_disabled(pod) is True
    data = skilltax.settings_of(pod)
    assert data["skills"]["disabled"] == skilltax.recommended_disabled()
    # 其他键（hooks）原样保留
    assert data["hooks"]["Stop"][0]["hooks"][0]["command"] == "node x.mjs"
    # 自有技能不被关停（不在 disabled 列表里）
    assert "theoretical-computation" not in data["skills"]["disabled"]
    # 幂等：再 apply 不改写
    assert skilltax.apply_disabled(pod) is False
    assert skilltax.disabled_gap(pod) == []


def test_apply_disabled_fails_loudly_on_missing_or_invalid(tmp_path):
    missing = tmp_path / "novcs"
    missing.mkdir()
    with pytest.raises(FileNotFoundError):
        skilltax.apply_disabled(missing)

    bad = _mkpod(tmp_path / "bad", None)
    (bad / ".qoder").mkdir(parents=True, exist_ok=True)
    (bad / ".qoder" / "settings.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError):
        skilltax.apply_disabled(bad)
    # 非法 JSON 时 settings_of 降级空 dict，缺口=全量而非误判已关停
    assert skilltax.disabled_gap(bad) == skilltax.recommended_disabled()
