"""write_taskbook 落盘任务书必须以恰好一个换行结尾（backlog 20261010-orch-taskbook-newline）。

裸文本任务书会触发 pre-commit end-of-file-fixer，把含任务书的提交撞回并改写文件。
"""

from __future__ import annotations

from pysci.skills.orchestration.tools.dispatch import write_taskbook


def _write(pod, *, text):
    return write_taskbook(pod, text=text, task_file=None, slug="t")


def test_text_without_trailing_newline_gets_one(tmp_path):
    dest = _write(tmp_path, text="摘要：做某事")
    raw = dest.read_text(encoding="utf-8")
    assert raw.endswith("\n")
    assert not raw.endswith("\n\n")


def test_text_with_trailing_newline_not_doubled(tmp_path):
    dest = _write(tmp_path, text="摘要：做某事\n")
    raw = dest.read_text(encoding="utf-8")
    assert raw.endswith("做某事\n")
    assert not raw.endswith("\n\n")


def test_text_with_multiple_trailing_newlines_collapsed(tmp_path):
    dest = _write(tmp_path, text="正文\n\n\n")
    assert dest.read_text(encoding="utf-8").endswith("正文\n")


def test_empty_text_still_ends_with_newline(tmp_path):
    dest = _write(tmp_path, text=None)
    assert dest.read_text(encoding="utf-8").endswith("\n")


def test_task_file_body_normalized(tmp_path):
    src = tmp_path / "src-task.md"
    src.write_text("目标：X", encoding="utf-8")
    dest = write_taskbook(
        tmp_path / "pod", text=None, task_file=str(src), slug="from-file"
    )
    raw = dest.read_text(encoding="utf-8")
    assert raw.endswith("目标：X\n")
    assert raw.startswith("# 任务书 from-file\n")
