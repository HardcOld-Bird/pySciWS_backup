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

## 其他
- 与声明无关的正文。
"""


def test_complete_declaration_passes():
    rep = domain.check_domain_declaration(GOOD)
    assert rep["ok"], rep
    assert rep["section"] and all(q["numeric"] for q in rep["questions"].values())


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


def test_placeholder_and_no_number_fail():
    text = GOOD.replace(
        "- **覆盖率**：超表面 8/8 周期覆盖入射面（100%，孔径 0.32m）。",
        "- **覆盖率**：待补。",
    )
    rep = domain.check_domain_declaration(text)
    assert not rep["ok"] and "覆盖率" in rep["non_numeric"]

    text2 = GOOD.replace(
        "- **激励口径**：高斯波束腰 0.03m、孔径 0.08m；实验换能器口径 0.05m，互证偏差 0.02m。",
        "- **激励口径**：已按实验口径核对，合理。",
    )
    rep2 = domain.check_domain_declaration(text2)
    assert not rep2["ok"] and "激励口径" in rep2["non_numeric"]


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
