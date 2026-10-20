"""「域几何声明」机械检查：节存在 + 四问键齐全 + 每问带数值。"""

from __future__ import annotations

import json

from pysci.skills.comsol_simulation.tools import domain, simulation

GOOD = """# 仿真交付

## 域几何声明

- **兼容性**：目标场区（31cm 菱形）完整位于 R=0.40m 域内，最小边距 0.06m。
- **覆盖率**：超表面 8/8 周期覆盖入射面（100%，孔径 0.32m）。
- **边界适配**：外场弧用 PerfectlyMatchedBoundary（禁用 PML）；bpf 仅选空气域（域 1）。
- **激励口径**：高斯波束腰 0.03m、孔径 0.08m；实验换能器口径 0.05m，互证偏差 0.02m。
  工作点 (cr, ci)=(-0.02, 0.073)，最近 EP (cr*, ci*)=(-0.018, 0.071)，
  偏移 Δcr=0.002、Δci=0.002；寻点判据见 theory/增益EP与谱奇点理论探索小结.md。

## 其他
- 与声明无关的正文。
"""

#: 旧版四问（无工作点-EP 偏移子问）——须 FAIL（tb-3 R5 M4）。
GOOD_NO_EP = GOOD.replace(
    "  工作点 (cr, ci)=(-0.02, 0.073)，最近 EP (cr*, ci*)=(-0.018, 0.071)，\n"
    "  偏移 Δcr=0.002、Δci=0.002；寻点判据见 theory/增益EP与谱奇点理论探索小结.md。\n",
    "",
)


def test_complete_declaration_passes():
    rep = domain.check_domain_declaration(GOOD)
    assert rep["ok"], rep
    assert rep["section"] and all(q["numeric"] for q in rep["questions"].values())


def _with_excitation(value: str) -> str:
    return GOOD_NO_EP.replace(
        "- **激励口径**：高斯波束腰 0.03m、孔径 0.08m；实验换能器口径 0.05m，互证偏差 0.02m。",
        f"- **激励口径**：{value}",
    )


def test_missing_ep_subquestion_fails():
    """四问齐全但激励口径缺 EP 子问 → FAIL（tb-3 R5 M4）。"""
    rep = domain.check_domain_declaration(GOOD_NO_EP)
    assert not rep["ok"]
    assert rep["excitation_ep"]["ok"] is False
    assert set(rep["excitation_ep"]["missing"]) == {
        "工作点(cr,ci)",
        "最近EP偏移",
        "寻点判据出处",
    }
    assert "激励口径缺" in rep["detail"]


def test_ep_subquestion_requires_offset():
    rep = domain.check_domain_declaration(
        _with_excitation("工作点 (cr, ci)=(-0.02, 0.073)；寻点见 theory/x.md。")
    )
    assert not rep["ok"]
    assert rep["excitation_ep"]["missing"] == ["最近EP偏移"]


def test_ep_subquestion_requires_source():
    rep = domain.check_domain_declaration(
        _with_excitation("工作点 (cr, ci)=(-0.02, 0.073)，偏移 Δ=0.002。")
    )
    assert not rep["ok"]
    assert rep["excitation_ep"]["missing"] == ["寻点判据出处"]


def test_ep_subquestion_accepts_cr_ci_form():
    rep = domain.check_domain_declaration(
        _with_excitation(
            "高斯波束腰 0.03m；cr=-0.02、ci=0.073，最近 EP 偏移 0.002；"
            "判据见 theory/x.md。"
        )
    )
    assert rep["ok"], rep


def test_missing_section_fails():
    rep = domain.check_domain_declaration("# 仿真交付\n\n没有声明节。\n")
    assert not rep["ok"] and not rep["section"]
    assert rep["missing"] == list(domain.KEYS)


def test_missing_key_fails():
    text = GOOD.replace(
        "- **覆盖率**：超表面 8/8 周期覆盖入射面（100%，孔径 0.32m）。\n", ""
    )
    rep = domain.check_domain_declaration(text)
    assert not rep["ok"] and rep["missing"] == ["覆盖率"]


def test_wrapped_bullet_continuation_is_joined():
    """四问值跨行（缩进续行）时，数值在续行上也要能识别。"""
    text = GOOD.replace(
        "- **边界适配**：外场弧用 PerfectlyMatchedBoundary（禁用 PML）；bpf 仅选空气域（域 1）。",
        "- **边界适配**：外场弧用 PerfectlyMatchedBoundary（禁用 PML）；bpf 仅选空气域\n  （域 1），24 个管槽域均不在选择内。",
    )
    rep = domain.check_domain_declaration(text)
    assert rep["ok"], rep
    assert "24" in rep["questions"]["边界适配"]["value"]


def test_placeholder_and_no_number_fail():
    text = GOOD.replace(
        "- **覆盖率**：超表面 8/8 周期覆盖入射面（100%，孔径 0.32m）。",
        "- **覆盖率**：待补。",
    )
    rep = domain.check_domain_declaration(text)
    assert not rep["ok"] and "覆盖率" in rep["non_numeric"]

    text2 = GOOD.replace(
        "- **激励口径**：高斯波束腰 0.03m、孔径 0.08m；实验换能器口径 0.05m，互证偏差 0.02m。\n"
        "  工作点 (cr, ci)=(-0.02, 0.073)，最近 EP (cr*, ci*)=(-0.018, 0.071)，\n"
        "  偏移 Δcr=0.002、Δci=0.002；寻点判据见 theory/增益EP与谱奇点理论探索小结.md。",
        "- **激励口径**：已按实验口径核对，合理。",
    )
    rep2 = domain.check_domain_declaration(text2)
    assert not rep2["ok"] and "激励口径" in rep2["non_numeric"]


def test_reference_documents_ep_subquestion():
    from pysci.paths import ORCHESTRATION_ROOT

    doc = (
        ORCHESTRATION_ROOT
        / "skills"
        / "comsol-simulation"
        / "references"
        / "domain-declaration.md"
    )
    text = doc.read_text(encoding="utf-8")
    assert "工作点-异常点偏移" in text
    assert "(cr, ci)" in text and "寻点判据" in text


def test_cli_pass_fail_and_json(tmp_path, capsys):
    good = tmp_path / "good.md"
    good.write_text(GOOD, encoding="utf-8")
    assert simulation.main(["domain-check", str(good)]) == 0
    out = capsys.readouterr().out
    assert "PASS" in out and "[√] 兼容性" in out

    bad = tmp_path / "bad.md"
    bad.write_text("# 无声明\n", encoding="utf-8")
    assert simulation.main(["domain-check", str(bad)]) == 1
    capsys.readouterr()

    assert simulation.main(["domain-check", str(good), "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["ok"] is True and data["section"] is True
    assert data["excitation_ep"]["ok"] is True
