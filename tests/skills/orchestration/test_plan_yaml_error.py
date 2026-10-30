"""坏 plan YAML 的可读报错（backlog 20261015-202556-devops）。

现场：`plan new` 登记含缩进错误的 plan YAML 时直接抛 ``yaml.scanner.ScannerError`` +
10 余行 traceback，且文件名显示成 ``<unicode string>``（``safe_load(text)`` 不带路径）——
组长手写 plan 缩进错一格就被糊脸，看不出**哪个文件的哪一行**。

修法：``plans.PlanYamlError``（继承 ``yaml.YAMLError``，兼容既有读面）+ ``parse_plan_yaml``
带 file:line:col；``plan new`` 内层带真实源文件路径报错，其余 plan 子命令由
``orch._plan_router`` 统一收口。
"""

from __future__ import annotations

import argparse

import pytest
import yaml

from pysci.skills.orchestration.tools import orch, plans

#: 缩进错位（`dirs:` 比同层兄弟多两格 → 被当成上一行标量的续行）。
BAD_YAML = """id: x
goal: demo
steps:
  - id: a1
    member: devops
    brief: do something
      dirs: src
    effort: low
"""

GOOD_YAML = """id: x
goal: demo
steps:
  - id: a1
    member: devops
    brief: do something
    dirs: src
"""


def _new_args(src) -> argparse.Namespace:
    return argparse.Namespace(file=str(src), id="x", force=False)


def test_plan_new_bad_yaml_single_line_no_traceback(qface, tmp_path, capsys):
    """坏 YAML：单行可读错误（含真实 file:line:col），rc=2，**无 traceback**。"""
    bad = tmp_path / "x.plan-src.yaml"
    bad.write_text(BAD_YAML, encoding="utf-8")

    rc = plans.cmd_plan_new(_new_args(bad))
    out = capsys.readouterr().out

    assert rc == 2
    assert "YAML 解析失败" in out
    assert (
        f"{bad}:7:11" in out
    )  # 真实源文件路径 + 行列（safe_load 只会给 <unicode string>）
    assert "Traceback" not in out and "ScannerError" not in out
    assert out.strip().splitlines()[-1].count(":") >= 2  # 末行为单行错误


def test_plan_new_good_yaml_unchanged(qface, tmp_path, capsys):
    """合法 YAML 行为不变：登记成功 rc=0。"""
    good = tmp_path / "x.plan-src.yaml"
    good.write_text(GOOD_YAML, encoding="utf-8")

    rc = plans.cmd_plan_new(_new_args(good))
    out = capsys.readouterr().out
    assert rc == 0 and "计划已登记" in out


def test_plan_amend_bad_yaml_friendly_via_router(qface, capsys):
    """`plan amend` 对同样坏 YAML 走路由收口：单行错误、rc=2、无 traceback。"""
    pid = "tb-probe"
    (qface / f"{pid}.plan.yaml").write_text(BAD_YAML, encoding="utf-8")

    rc = orch.main(["plan", "amend", pid])
    out = capsys.readouterr().out

    assert rc == 2
    assert "YAML 解析失败" in out and "Traceback" not in out


def test_plan_yaml_error_is_yaml_yaerror_subclass():
    """兼容性：继承 ``yaml.YAMLError``——既有「接住 yaml.YAMLError」的读面不受影响。"""
    err = plans.PlanYamlError("p.yaml", yaml.YAMLError("boom"))
    assert isinstance(err, yaml.YAMLError)
    assert "YAML 解析失败" in err.render()


def test_parse_plan_yaml_raises_typed_error(tmp_path):
    bad = tmp_path / "p.yaml"
    bad.write_text(BAD_YAML, encoding="utf-8")
    with pytest.raises(plans.PlanYamlError) as ei:
        plans.parse_plan_yaml(bad)
    assert ei.value.line == 7 and ei.value.col == 11
    assert str(bad) in ei.value.render()
