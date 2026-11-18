"""CLI 薄路由机械断言：路由表匹配 / doctor 数据 / refs 端到端 / 参数不自洽报错。"""

from __future__ import annotations

import json

import pytest

from pysci.skills.document_writing import cli


def test_route_office_editing_goes_to_dsh_office():
    rules = cli.match_routes("编辑 pptx 追加一页")
    assert [r.target for r in rules] == ["dsh-office"]
    assert "office-pptx" in rules[0].entry


def test_route_md_to_docx_goes_to_pandoc():
    rules = cli.match_routes("把 md 转 docx")
    assert [r.target for r in rules] == ["pandoc"]


def test_route_latex_goes_to_latex_workflow():
    rules = cli.match_routes("latex 编译论文出 PDF")
    assert [r.target for r in rules] == ["latex"]


def test_route_formula_goes_to_glue():
    rules = cli.match_routes("生成透明 png 公式")
    assert [r.target for r in rules] == ["glue"]
    assert "formula" in rules[0].entry


def test_route_citation_mentions_citeproc():
    rules = cli.match_routes("插入引用并排参考文献")
    assert rules and "citeproc" in rules[0].entry


def test_route_no_match_returns_empty():
    assert cli.match_routes("量子引力综述") == []


def test_route_command_exit_codes(capsys):
    assert cli.main(["route", "编辑 pptx"]) == 0
    assert "dsh-office" in capsys.readouterr().out
    assert cli.main(["route", "量子引力"]) == 1


def test_doctor_report_shape():
    report = cli.doctor_report()
    names = [name for name, _, _ in report]
    assert names == ["pandoc", "latexmk", "chktex", "pymupdf", "pillow"]
    assert all(isinstance(ok, bool) and isinstance(detail, str) for _, ok, detail in report)


def test_refs_command_end_to_end(tmp_path, capsys):
    items = [
        {
            "type": "article-journal",
            "title": "A Study",
            "author": [{"family": "Li", "given": "Ming"}],
            "issued": {"date-parts": [[2022]]},
        }
    ]
    src = tmp_path / "items.json"
    src.write_text(json.dumps(items), encoding="utf-8")
    out = tmp_path / "refs.bib"
    assert cli.main(["refs", str(src), "-o", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    assert "@article{li2022a," in text
    assert "1 条" in capsys.readouterr().out


def test_refs_command_bad_input_exits_1(tmp_path):
    src = tmp_path / "bad.json"
    src.write_text(json.dumps({"oops": 1}), encoding="utf-8")
    assert cli.main(["refs", str(src), "-o", str(tmp_path / "r.bib")]) == 1


def test_convert_citeproc_without_bib_exits_1(tmp_path, capsys):
    src = tmp_path / "doc.md"
    src.write_text("x", encoding="utf-8")
    code = cli.main(["convert", str(src), "--citeproc"])
    assert code == 1
    assert "citeproc" in capsys.readouterr().err


def test_latex_build_unknown_engine_exits_2(tmp_path):
    tex = tmp_path / "x.tex"
    tex.write_text("x", encoding="utf-8")
    with pytest.raises(SystemExit) as excinfo:  # argparse choices 拒绝 → SystemExit(2)
        cli.main(["latex", "build", str(tex), "--engine", "typst"])
    assert excinfo.value.code == 2
