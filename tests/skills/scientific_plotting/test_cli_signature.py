"""build/preview/audit 的 figdir-only 签名范式回归测试。

背景（backlog 20261009-taskbook-cli-signature）：任务书曾把 build 写成
``build gain_ep fig0_orch_smoke``（照抄 new 的 ``<research> <slug>`` 签名），组员照抄报
``unrecognized arguments``。此处锁定：

- build/preview/audit 只收**单一 figdir 位置参数**，多给一个位置参数 → argparse 退出码 2；
- 正确形态 ``build '<figdir>'`` 可解析为 Path；
- 三者的 -h 帮助显式点明 figdir 形态、并与 new 的 ``<research> <slug>`` 对比（含示例 epilog）；
- new 仍收 ``<research> <slug>`` 一对参数（签名未被误改）。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from pysci.skills.scientific_plotting.tools import figures

FIGDIR = "data/research/1_gain_ep/article/figures/fig1_ep_band"


# ---------------------------------------------------------------------------
# 错误范式（new 的 <research> <slug> 用到 build 上）必须失败
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cmd", ["build", "preview", "audit"])
def test_research_slug_form_is_rejected(cmd):
    """照抄 new 签名 → argparse unrecognized arguments → SystemExit(2)。"""
    with pytest.raises(SystemExit) as ei:
        figures.main([cmd, "gain_ep", "fig0_orch_smoke"])
    assert ei.value.code == 2


# ---------------------------------------------------------------------------
# 正确范式：单一 figdir 路径可解析
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cmd", ["build", "preview", "audit"])
def test_figdir_form_parses(cmd):
    args = figures.build_parser().parse_args([cmd, FIGDIR])
    assert isinstance(args.figdir, Path)
    assert args.figdir == Path(FIGDIR)


def test_new_still_takes_research_slug_pair():
    """new 的签名不变：一对 <research> <slug>。"""
    args = figures.build_parser().parse_args(["new", "gain_ep", "fig1_ep_band"])
    assert args.research == "gain_ep"
    assert args.slug == "fig1_ep_band"


# ---------------------------------------------------------------------------
# 帮助文本自文档化：figdir 形态 + 与 new 的对比 + 示例
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cmd", ["build", "preview", "audit"])
def test_help_documents_figdir_paradigm(cmd, capsys):
    with pytest.raises(SystemExit):
        figures.main([cmd, "--help"])
    out = capsys.readouterr().out
    # 位置参数以 <figdir> 元变量呈现，help 点明「单一位置参数」且对比 new
    assert "<figdir>" in out
    assert "单一位置参数" in out
    assert "new" in out
    # epilog 给出可直接照抄的正确示例
    assert "data/research/1_gain_ep/article/figures/" in out


def test_new_help_contrasts_with_figdir_commands(capsys):
    with pytest.raises(SystemExit):
        figures.main(["new", "--help"])
    out = capsys.readouterr().out
    assert "<research> <slug>" in out
    assert "build/preview/audit" in out
    assert "<figdir>" in out


def test_figdir_action_metavar_and_help_wired():
    """直接检查 build 子解析器的 figdir 动作：metavar 与 help 均已强化。"""
    parser = figures.build_parser()
    build_p = parser._subparsers._group_actions[0].choices["build"]
    action = next(a for a in build_p._actions if a.dest == "figdir")
    assert action.metavar == "<figdir>"
    assert "单一位置参数" in action.help
    assert isinstance(build_p.formatter_class, type)
    assert build_p.formatter_class is argparse.RawDescriptionHelpFormatter
    assert "data/research/1_gain_ep/article/figures/" in build_p.epilog
