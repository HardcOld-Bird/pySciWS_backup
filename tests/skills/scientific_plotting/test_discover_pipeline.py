"""discover_pipeline 双侧同名冲突可见化与 prefer 覆盖的回归测试。

背景（backlog 20261009-discover-pipeline-fake-green）：src 脚手架占位管线曾静默压制
figdir 内组员定制的同名管线，build 渲染占位图却显示成功（假绿）。此处断言：

- 双侧同名并存 → 默认选 src 且打印 WARNING 指明实际选用者/被忽略者；
- ``--pipeline-in-figdir`` / STYLE.yaml ``pipeline:`` → 改选 figdir；
- 单侧命中 → 无告警；偏好侧缺失 → 回退另一侧并告警；
- ``RunResult.report`` 显示完整相对路径（不再只有 ``.name``）；
- ``_resolve_prefer`` 校验未知取值（边界输入）。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pysci.skills.scientific_plotting.tools import figures, runner


@pytest.fixture
def pipeline_roots(tmp_path, monkeypatch):
    """重定向 src 代码侧根到 tmp，并给出含 ``research`` 段的数据侧 figures 根。

    数据侧路径必须含 ``research/<n>_<name>/`` 段，``_research_from_figdir`` 才能解析出
    ``(research_name, slug)``；src 侧经 monkeypatch ``figures_code_root`` 指到 tmp。
    """
    data_figures = tmp_path / "data" / "research" / "1_gain_ep" / "article" / "figures"
    src_figures = (
        tmp_path / "src" / "pysci" / "research" / "gain_ep" / "article" / "figures"
    )
    data_figures.mkdir(parents=True)
    src_figures.mkdir(parents=True)
    monkeypatch.setattr(runner, "figures_code_root", lambda research: src_figures)
    return {"data": data_figures, "src": src_figures, "tmp": tmp_path}


def _make_fig(roots, slug, *, src=None, figdir=None):
    """建一个 figdir（含 out/）；按需写入 src 侧 / figdir 侧同名管线（内容为可辨识标记）。"""
    fd = roots["data"] / slug
    (fd / "out").mkdir(parents=True, exist_ok=True)
    if src is not None:
        (roots["src"] / f"{slug}.py").write_text(f"# {src}\n", encoding="utf-8")
    if figdir is not None:
        (fd / f"{slug}.py").write_text(f"# {figdir}\n", encoding="utf-8")
    return fd


# ---------------------------------------------------------------------------
# 冲突可见化（假绿防护核心）
# ---------------------------------------------------------------------------
def test_collision_prefers_src_and_warns(pipeline_roots, capsys):
    fd = _make_fig(
        pipeline_roots, "fig0_smoke", src="SRC-PLACEHOLDER", figdir="FIGDIR-REAL"
    )
    src_pipe = pipeline_roots["src"] / "fig0_smoke.py"
    fig_pipe = fd / "fig0_smoke.py"

    got = runner.discover_pipeline(fd)

    assert got == src_pipe  # 默认仍 src 优先（不破坏既有约定）
    err = capsys.readouterr().err
    assert "WARNING" in err
    assert "双侧并存" in err
    # 关键：实际选用者与被忽略者的**完整路径**都出现在告警里（消除静默）
    assert str(src_pipe) in err
    assert str(fig_pipe) in err
    assert "--pipeline-in-figdir" in err


def test_collision_prefer_figdir_selects_figdir(pipeline_roots, capsys):
    fd = _make_fig(pipeline_roots, "fig0_smoke", src="SRC", figdir="FIG")
    got = runner.discover_pipeline(fd, prefer="figdir")
    assert got == fd / "fig0_smoke.py"
    assert "已选用 figdir 侧" in capsys.readouterr().err


def test_collision_prefer_src_explicit(pipeline_roots):
    fd = _make_fig(pipeline_roots, "fig0_smoke", src="SRC", figdir="FIG")
    got = runner.discover_pipeline(fd, prefer="src")
    assert got == pipeline_roots["src"] / "fig0_smoke.py"


def test_collision_detects_priority_name_entry(pipeline_roots, capsys):
    """figdir 侧入口名为 fig.py（优先名）时也应识别为冲突。"""
    fd = _make_fig(pipeline_roots, "fig5", src="SRC")
    (fd / "fig.py").write_text("# FIG\n", encoding="utf-8")

    assert runner.discover_pipeline(fd) == pipeline_roots["src"] / "fig5.py"
    assert "双侧并存" in capsys.readouterr().err
    assert runner.discover_pipeline(fd, prefer="figdir") == fd / "fig.py"


def test_lone_figdir_helper_py_is_not_a_collision(pipeline_roots, capsys):
    """src 管线 + figdir 内一个非入口名的辅助 .py：不算冲突（strict 判定），不告警。"""
    fd = _make_fig(pipeline_roots, "fig6", src="SRC")
    (fd / "helper.py").write_text("# 辅助，不是管线\n", encoding="utf-8")
    got = runner.discover_pipeline(fd)
    assert got == pipeline_roots["src"] / "fig6.py"
    assert "WARNING" not in capsys.readouterr().err


# ---------------------------------------------------------------------------
# 单侧命中：无告警
# ---------------------------------------------------------------------------
def test_only_src_no_warning(pipeline_roots, capsys):
    fd = _make_fig(pipeline_roots, "fig1", src="SRC")
    assert runner.discover_pipeline(fd) == pipeline_roots["src"] / "fig1.py"
    assert "WARNING" not in capsys.readouterr().err


def test_only_figdir_no_warning(pipeline_roots, capsys):
    fd = _make_fig(pipeline_roots, "fig2", figdir="FIG")
    assert runner.discover_pipeline(fd) == fd / "fig2.py"
    assert "WARNING" not in capsys.readouterr().err


# ---------------------------------------------------------------------------
# 偏好侧缺失 → 回退另一侧并告警
# ---------------------------------------------------------------------------
def test_prefer_figdir_missing_falls_back_to_src(pipeline_roots, capsys):
    fd = _make_fig(pipeline_roots, "fig3", src="SRC")
    got = runner.discover_pipeline(fd, prefer="figdir")
    assert got == pipeline_roots["src"] / "fig3.py"
    assert "回退 src 侧" in capsys.readouterr().err


def test_prefer_src_missing_falls_back_to_figdir(pipeline_roots, capsys):
    fd = _make_fig(pipeline_roots, "fig3b", figdir="FIG")
    got = runner.discover_pipeline(fd, prefer="src")
    assert got == fd / "fig3b.py"
    assert "回退 figdir 侧" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# warn=False 静默（list 批量场景）
# ---------------------------------------------------------------------------
def test_warn_false_suppresses_collision_warning(pipeline_roots, capsys):
    fd = _make_fig(pipeline_roots, "fig4", src="SRC", figdir="FIG")
    got = runner.discover_pipeline(fd, warn=False)
    assert got == pipeline_roots["src"] / "fig4.py"
    assert capsys.readouterr().err == ""


# ---------------------------------------------------------------------------
# 错误路径（保留原有区分性信息）
# ---------------------------------------------------------------------------
def test_neither_side_raises_no_pipeline(pipeline_roots):
    fd = pipeline_roots["data"] / "fig7"
    (fd / "out").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="没有找到管线"):
        runner.discover_pipeline(fd)


def test_missing_figdir_raises(pipeline_roots):
    fd = pipeline_roots["data"] / "does_not_exist"
    with pytest.raises(FileNotFoundError, match="图目录不存在"):
        runner.discover_pipeline(fd)


def test_multiple_py_ambiguous_raises(pipeline_roots):
    fd = pipeline_roots["data"] / "fig8"
    (fd / "out").mkdir(parents=True)
    (fd / "a.py").write_text("# a\n", encoding="utf-8")
    (fd / "b.py").write_text("# b\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="多个 .py"):
        runner.discover_pipeline(fd)


# ---------------------------------------------------------------------------
# _resolve_prefer 边界校验
# ---------------------------------------------------------------------------
def test_resolve_prefer_precedence_and_validation(capsys):
    # CLI > STYLE.yaml
    assert runner._resolve_prefer("figdir", {"pipeline": "src"}) == "figdir"
    # 归一化大小写/空白
    assert runner._resolve_prefer(None, {"pipeline": " SRC "}) == "src"
    # 缺省 → None（自动）
    assert runner._resolve_prefer(None, {}) is None
    # 未知取值 → 告警并忽略
    assert runner._resolve_prefer(None, {"pipeline": "bogus"}) is None
    assert "未知的 pipeline 偏好" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# build_figure_dir / CLI 贯通 prefer
# ---------------------------------------------------------------------------
def test_build_honors_style_yaml_pipeline_key(pipeline_roots, monkeypatch):
    """STYLE.yaml 的 pipeline: 键经 _resolve_prefer 传入 discover_pipeline。"""
    fd = _make_fig(pipeline_roots, "fig9", src="SRC", figdir="FIG")
    (fd / "STYLE.yaml").write_text("pipeline: figdir\n", encoding="utf-8")

    captured: dict = {}

    class _Sentinel(Exception):
        pass

    def _spy(figdir, *, prefer=None, warn=True):
        captured["prefer"] = prefer
        raise _Sentinel  # 短路：build_figure_dir 的宽 except 会吞掉

    monkeypatch.setattr(runner, "discover_pipeline", _spy)
    monkeypatch.setattr(runner, "assert_within_data", lambda *a, **k: None)

    res = runner.build_figure_dir(fd)
    assert captured["prefer"] == "figdir"
    assert res.error is not None  # 被 _Sentinel 短路，结构化捕获


def test_build_cli_flag_passes_prefer_figdir(pipeline_roots, monkeypatch):
    fd = _make_fig(pipeline_roots, "fig10", src="SRC")
    captured: dict = {}

    def _spy(figdir, **kwargs):
        captured.clear()
        captured.update(kwargs)
        return runner.RunResult(
            figdir=Path(figdir), pipeline=Path(figdir), stem="x", error="spy"
        )

    monkeypatch.setattr(runner, "build_figure_dir", _spy)

    assert figures.main(["build", str(fd), "--pipeline-in-figdir"]) == 1
    assert captured.get("prefer") == "figdir"

    assert figures.main(["build", str(fd)]) == 1
    assert captured.get("prefer") is None


def test_preview_cli_flag_passes_prefer_figdir(pipeline_roots, monkeypatch):
    fd = _make_fig(pipeline_roots, "fig11", src="SRC")
    captured: dict = {}

    def _spy(figdir, **kwargs):
        captured.clear()
        captured.update(kwargs)
        return runner.RunResult(
            figdir=Path(figdir), pipeline=Path(figdir), stem="x", error="spy"
        )

    monkeypatch.setattr(runner, "preview_figure_dir", _spy)

    assert figures.main(["preview", str(fd), "--pipeline-in-figdir"]) == 1
    assert captured.get("prefer") == "figdir"


# ---------------------------------------------------------------------------
# RunResult.report 显示完整路径（不再只有 .name）
# ---------------------------------------------------------------------------
def test_report_shows_full_pipeline_path(pipeline_roots):
    fd = _make_fig(pipeline_roots, "figR", src="SRC")
    src_pipe = pipeline_roots["src"] / "figR.py"
    res = runner.RunResult(figdir=fd, pipeline=src_pipe, stem="figR")
    rep = res.report()
    # tmp 不在 PROJECT_ROOT 下 → _rel_to_root 回退绝对路径；关键是完整路径可见
    assert str(src_pipe) in rep
    assert "pipeline :" in rep
