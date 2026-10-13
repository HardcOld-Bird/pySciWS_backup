"""源文件 → 文档面对盘（backlog 20261013-184842-devops）。

起因是一处**代码与文档同错**：jury 的同族拒庭判据被相对化（`producer_family`）后，代码改了
三处、`jury-channels.md`/`jury-panel.md` 两处没跟，文档还替旧行为背书（「含回退席…含 qwen」）。
README §6 的 `intent-drift` 是**抽样式**的巡检，抓不到「谁改了 X 没看 X 的文档」——本组钉住
那声**改动驱动**的提示。
"""

from __future__ import annotations

from pysci.skills.devops.tools import dev, docfaces


def test_faces_for_matches_by_suffix_in_worktree():
    """键是**后缀**：devops 的改动总发生在 worktree 里，绝对/相对前缀都对不齐。"""
    abs_wt = "/d/repo/.qoder/worktrees/20261013-184842-devops/src/pysci/skills/orchestration/tools/jury.py"
    rel = "src/pysci/skills/orchestration/tools/jury.py"
    win = r"src\pysci\skills\orchestration\tools\jury.py"
    for p in (abs_wt, rel, win):
        faces = docfaces.faces_for(p)
        assert any(f.endswith("jury-channels.md") for f in faces), p
        assert any(f.endswith("jury-panel.md") for f in faces), p


def test_unmapped_file_is_silent():
    """未映射文件**零输出**（照抄自家「没命中就静默」口径，也才有可组合性）。"""
    assert docfaces.faces_for("README.md") == ()
    assert (
        docfaces.faces_for("src/pysci/skills/orchestration/tools/transcript.py") == ()
    )
    assert docfaces.report(["README.md", "setup.py"]) == []


def test_report_names_source_and_faces():
    lines = docfaces.report(["src/pysci/skills/orchestration/tools/dispatch.py"])
    assert len(lines) == 1 and "[doc-faces]" in lines[0]
    assert "dispatch.py" in lines[0] and "drain-watch.md" in lines[0]


def test_mapping_table_paths_all_exist():
    """**表不能腐烂**：写错的映射会把人对盘到错文件，比没有映射更坏。"""
    assert docfaces.misconfigured() == [], (
        f"映射表含不存在的文档面：{docfaces.misconfigured()}"
    )


def test_cli_prints_faces_and_stays_silent_for_unmapped(capsys):
    """success_criteria 原文的钉法：`tools/jury.py` → 输出含 `jury-channels.md`；非映射零输出。"""
    rc = dev.main(["doc-faces", "src/pysci/skills/orchestration/tools/jury.py"])
    out = capsys.readouterr().out
    assert rc == 0 and "jury-channels.md" in out

    rc = dev.main(["doc-faces", "README.md"])
    assert rc == 0
    assert capsys.readouterr().out == "", "非映射内文件不许有输出"
