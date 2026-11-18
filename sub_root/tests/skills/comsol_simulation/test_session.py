"""会话/domain-check 机械断言：版本串归一（纯）+ tasklist 解析（纯）+ 四问判定（纯）。"""

from __future__ import annotations

import pytest

from pysci.skills.comsol_simulation import session

# ---------------------------------------------------------------------------
# 版本串归一（mph discovery 短名坑）
# ---------------------------------------------------------------------------


def test_normalize_version_full_string_to_short_name():
    assert session.normalize_version("6.4.0.293") == "6.4"


def test_normalize_version_keeps_short_name():
    assert session.normalize_version("6.4") == "6.4"


def test_normalize_version_three_part():
    assert session.normalize_version("6.3.2") == "6.3"


def test_normalize_version_strips_whitespace():
    assert session.normalize_version("  6.4.0.293  ") == "6.4"


def test_normalize_version_rejects_garbage():
    with pytest.raises(ValueError, match="无法解析"):
        session.normalize_version("latest")


# ---------------------------------------------------------------------------
# 占证探测（jvm.dll 判据；_jpype.pyd 误报源排除）
# ---------------------------------------------------------------------------

_TASKLIST_SAMPLE = """\
信息: 没有运行的任务匹配指定标准。

Image Name                     PID Session Name        Session#    Modules
========================= ======== ================ =========== ==========
java.exe                       8260 Console                    1    jvm.dll
python.exe                     9124 Console                    1    _jpype.pyd
comsolmphserver.exe            7332 Console                    1    jvm.dll
"""


def test_parse_tasklist_modules_extracts_by_module_suffix():
    procs = session.parse_tasklist_modules(_TASKLIST_SAMPLE)
    assert procs["java.exe"] == {"jvm.dll"}
    assert procs["python.exe"] == {"_jpype.pyd"}
    assert procs["comsolmphserver.exe"] == {"jvm.dll"}


def test_parse_tasklist_modules_ignores_headers_and_na():
    text = "Image Name PID Modules\npython.exe 1 N/A\n"
    assert session.parse_tasklist_modules(text) == {}


def test_license_holders_excludes_jpype_only_python():
    procs = session.parse_tasklist_modules(_TASKLIST_SAMPLE)
    holders = session.license_holders(procs)
    assert holders == ["comsolmphserver.exe", "java.exe"]
    assert "python.exe" not in holders  # _jpype.pyd ≠ JVM 启动（误报源）


def test_license_holders_empty_when_no_jvm():
    assert session.license_holders({"python.exe": {"_jpype.pyd"}}) == []


# ---------------------------------------------------------------------------
# domain-check：域几何声明四问机械检查
# ---------------------------------------------------------------------------

_VALID_DECL = """\
# 任务书

## 域几何声明

- **兼容性**：目标场区（示例菱形区）完整位于 R=0.40m 域内，最小边距 0.06m；
  标注区位于域外画布内，距域边 0.03m。
- **覆盖率**：超表面 8/8 周期覆盖入射面（100%，孔径 0.32m）；无截断。
- **边界适配**：外场弧边界用 PerfectlyMatchedBoundary（禁用 PML）；bpf 仅选空气域
  （域 1），24 个散射体域均不在选择内。
- **激励口径**：高斯波束腰 0.03m、结构孔径 0.08m；实验换能器口径 0.05m。
  工作点 (cr, ci)=(-0.02, 0.073)，最近异常点 (cr*, ci*)=(-0.018, 0.071)，
  偏移 Δcr=0.002、Δci=0.002；域版本=v1，寻点版本=v1；寻点判据见
  theory/谱奇点探索小结.md。
"""


def test_declaration_full_pass():
    rep = session.check_declaration(_VALID_DECL)
    assert rep.ok
    assert rep.missing == [] and rep.fuzzy == []
    assert rep.soft_warnings == []
    assert "兼容性" in rep.answers


def test_declaration_missing_section_fails():
    rep = session.check_declaration("# 无声明节\n")
    assert not rep.ok
    assert set(rep.missing) == set(session.DECLARATION_KEYS)


def test_declaration_missing_key_fails():
    text = _VALID_DECL.replace("- **覆盖率**", "- **删掉的覆盖率**")
    rep = session.check_declaration(text)
    assert not rep.ok
    assert rep.missing == ["覆盖率"]


def test_declaration_fuzzy_answer_fails():
    text = _VALID_DECL.replace("最小边距 0.06m", "待补")
    rep = session.check_declaration(text)
    assert not rep.ok
    assert "兼容性" in rep.fuzzy


def test_declaration_answer_without_number_fails():
    text = (
        "## 域几何声明\n"
        "- **兼容性**：域足够大——已检查；\n"
        "- **覆盖率**：8/8 周期 100%；\n"
        "- **边界适配**：PMB 边界，仅选域 1；\n"
        "- **激励口径**：波束腰 0.03m、孔径 0.08m。工作点 (cr, ci)=(-0.02, 0.07)，"
        "最近异常点 (cr*, ci*)=(0, 0.07)，偏移 Δcr=0.02；寻点判据见 theory 小结。\n"
    )
    rep = session.check_declaration(text)
    assert not rep.ok
    assert "兼容性" in rep.fuzzy


def test_declaration_em_dash_in_numeric_answer_ok():
    # 破折号出现在有数值的答案正文里不得误伤（只按整答判「—」敷衍）
    text = _VALID_DECL.replace("无截断。", "无截断——孔径 0.32m 满覆盖。")
    rep = session.check_declaration(text)
    assert rep.ok


def test_declaration_excitation_needs_ep_subquestion():
    text = _VALID_DECL.replace("工作点 (cr, ci)", "残余项 (cr, ci)")
    rep = session.check_declaration(text)
    assert not rep.ok
    assert any("激励口径" in f for f in rep.fuzzy)


def test_declaration_version_mismatch_soft_warns():
    text = _VALID_DECL.replace("寻点版本=v1", "寻点版本=v2")
    rep = session.check_declaration(text)
    assert rep.ok  # 软告警不判 FAIL
    assert any("不一致" in w for w in rep.soft_warnings)


# ---------------------------------------------------------------------------
# 模型域/单位/网格质量断言
# ---------------------------------------------------------------------------


def test_check_units_same_unit_passes():
    assert session.check_units("m", "m").ok


def test_check_units_mm_vs_m_fails():
    r = session.check_units("mm", "m")
    assert not r.ok
    assert r.value == pytest.approx(1e-3)


def test_check_units_unknown_unit_fails():
    assert not session.check_units("parsec", "m").ok


def test_check_domain_extent_inside_with_margin_passes():
    r = session.check_domain_extent((-1, 1, -1, 1), (-0.5, 0.5, -0.5, 0.5), min_margin=0.4)
    assert r.ok
    assert r.value == pytest.approx(0.5)


def test_check_domain_extent_outside_fails():
    r = session.check_domain_extent((-1, 1, -1, 1), (-1.5, 0.5, -0.5, 0.5), min_margin=0.0)
    assert not r.ok
    assert r.value == pytest.approx(-0.5)


def test_check_mesh_quality_six_per_wavelength_passes():
    r = session.check_mesh_quality(min_element=0.01, wavelength=0.06)
    assert r.ok
    assert r.value == pytest.approx(6.0)


def test_check_mesh_quality_too_coarse_fails():
    assert not session.check_mesh_quality(0.05, 0.06).ok


def test_check_mesh_quality_rejects_nonpositive():
    with pytest.raises(ValueError):
        session.check_mesh_quality(0.0, 0.06)
