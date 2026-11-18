"""prepare/status/submit/check-work 管线回归（旧 test_bilingual_pipeline.py 期望值平移）。

tmp_path 全隔离（conftest.ws）；图片回补与渲染打桩（网络回补/浏览器打印不在默认套件）。
submit 重点锁「英文侧逐字还原（=源文档切片）」、「未译/占位拒出稿」、「check-work 护栏
（pod-guard 等价物）」与「TM 回填（升级 --reuse）」。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from pysci import paths
from pysci.skills.literature_research.bilingual import cli, images, manifest

from .conftest import fake_translations, make_source

#: 真回补函数原件（ws 垫片打桩前的引用；缺源/同名回补用例恢复它）。
REAL_BACKFILL = images.backfill_images

MINI_MD = """# Introduction

This is the first sentence. It has a second one.

![x](images/fig1.jpg)

Fig. 1. The experimental setup (a).

$$
E = mc^2
$$

<table>
<tr><td>1</td></tr>
</table>

Table 1. The measured data.

More text here. Second sentence here?

## References

[1] A. Author et al., Title, J. 2020.
"""


def fake_render(md_path: Path, pdf_path: Path) -> Path:
    """渲染打桩：落一个假 PDF（接线验证，不碰浏览器）。"""
    Path(pdf_path).write_bytes(b"%PDF-1.4 fake")
    return Path(pdf_path)


# ---------------------------------------------------------------------------
# prepare
# ---------------------------------------------------------------------------
def test_prepare_writes_work_and_manifest(ws: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """prepare：work.md 行格式（ID<TAB>英文）、manifest 单元/透传块齐全、切片自检过。"""
    src = make_source(ws, MINI_MD)
    assert cli.cmd_prepare(str(src)) == 0
    workdir = ws / "bilingual" / "mini"
    work_md = (workdir / "work.md").read_text(encoding="utf-8").split("\n")
    assert work_md[0].startswith("# 翻译规则：")
    id_lines = [ln for ln in work_md if ln and not ln.startswith("#")]
    assert id_lines[0] == "H0001\tIntroduction"
    assert "S0001\tThis is the first sentence." in id_lines
    assert all(manifest.ID_PREFIX_RE.match(ln) for ln in id_lines)

    man = json.loads((workdir / "manifest.json").read_text(encoding="utf-8"))
    assert man["source_md"] == str(src)
    assert [u["id"] for u in man["units"]][0] == "H0001"
    assert {u["type"] for u in man["units"]} == {"H", "S", "C", "T"}
    assert [b["kind"] for b in man["blocks"]] == ["image", "formula", "table", "reference"]
    assert man["images"] == ["images/fig1.jpg"]
    assert man["images_closed"] is True
    out = capsys.readouterr().out
    assert "翻译指引" in out and "check-work" in out


def test_prepare_missing_source_reports_hint(
    ws: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """源 md 不存在：报「先走①域 read 提取」并 rc=2。"""
    assert cli.cmd_prepare("no_such_doc") == 2
    assert "提取" in capsys.readouterr().out


def test_prepare_no_images_skips_backfill_and_flags_manifest(ws: Path) -> None:
    """--no-images：跳过回补（images/ 不建），manifest 记 no_images=True 供 submit 拒稿。"""
    src = make_source(ws, MINI_MD)
    assert cli.cmd_prepare(str(src), no_images=True) == 0
    workdir = ws / "bilingual" / "mini"
    assert not (workdir / "images").exists()
    man = json.loads((workdir / "manifest.json").read_text(encoding="utf-8"))
    assert man["no_images"] is True


def test_prepare_accepts_unspaced_panel_numbers(ws: Path) -> None:
    """数字紧接单字母面板号（``Fig. 3(a).``）不再被「切片自检失败」拒绝（游标逐字守恒）。"""
    src = make_source(
        ws,
        md=(
            "# Results\n\n"
            "As shown in Fig. 3(a). What we see here is clear. "
            "Figs. 3(b)–(d). More details follow.\n"
        ),
        name="panel.md",
    )
    assert cli.cmd_prepare(str(src)) == 0
    work_md = (ws / "bilingual" / "panel" / "work.md").read_text(encoding="utf-8")
    en_lines = [ln.split("\t", 1)[1] for ln in work_md.split("\n") if "\t" in ln]
    assert "As shown in Fig. 3(a)." in en_lines
    assert "Figs. 3(b)–(d)." in en_lines


def test_prepare_missing_pdf_reports_hint(
    ws: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """真回补路径缺源 PDF：rc=2 并提示 --pdf（工作区无 manifest，不留半成品契约）。"""
    monkeypatch.setattr(images, "backfill_images", REAL_BACKFILL)
    src = make_source(ws, MINI_MD)
    assert cli.cmd_prepare(str(src)) == 2
    out = capsys.readouterr().out
    assert "--pdf" in out and "图片回补失败" in out
    assert not (ws / "bilingual" / "mini" / "manifest.json").exists()


def test_prepare_serves_sibling_images_without_pdf(ws: Path) -> None:
    """①域衔接：源 md 同目录 images/ 按名复制回补（无源 PDF 也闭合）。"""
    src = make_source(ws, MINI_MD)
    (src.parent / "images").mkdir()
    (src.parent / "images" / "fig1.jpg").write_bytes(b"real-img")
    workdir = ws / "bilingual" / "mini"
    workdir.mkdir(parents=True, exist_ok=True)
    REAL_BACKFILL(workdir, ["images/fig1.jpg"], None, no_images=False, source_md=src)
    assert (workdir / "images" / "fig1.jpg").read_bytes() == b"real-img"


# ---------------------------------------------------------------------------
# status 对账
# ---------------------------------------------------------------------------
def test_status_counts_and_lists_untranslated(
    ws: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """status：缺 ID/内容==en 记未译，含 CJK 记已译，其余存疑；输出计数与未译 ID。"""
    src = make_source(ws, MINI_MD)
    assert cli.cmd_prepare(str(src)) == 0
    workdir = ws / "bilingual" / "mini"
    fake_translations(workdir)
    path = workdir / "work.md"
    text = path.read_text(encoding="utf-8")
    # S0002 改回英文原文＝未译；S0003 改成无 CJK 的异文＝存疑；其余保持假翻译。
    text = text.replace("S0002\t中译 S0002", "S0002\tIt has a second one.")
    text = text.replace("S0003\t中译 S0003", "S0003\tDoubtful latin text.")
    path.write_text(text, encoding="utf-8")

    assert cli.cmd_status("mini") == 0
    out = capsys.readouterr().out
    assert "未译 1" in out
    assert "存疑 1" in out
    assert "未译 ID：S0002" in out


# ---------------------------------------------------------------------------
# submit：英文侧逐字还原 + 形态契约
# ---------------------------------------------------------------------------
def test_submit_english_side_is_source_slices(ws: Path) -> None:
    """submit：英文侧从 manifest 重嵌入＝源文档切片（diff 为空），且落 inbox 副本。"""
    src = make_source(ws, MINI_MD)
    assert cli.cmd_prepare(str(src)) == 0
    workdir = ws / "bilingual" / "mini"
    fake_translations(workdir)
    assert cli.cmd_submit("mini", render_fn=fake_render) == 0

    text = (workdir / "bilingual.md").read_text(encoding="utf-8")
    man = json.loads((workdir / "manifest.json").read_text(encoding="utf-8"))
    italic = [ln[1:-1] for ln in text.split("\n") if ln.startswith("*") and ln.endswith("*")]
    source_slices = [u["en"] for u in man["units"]]
    assert italic == source_slices  # 英文侧逐字还原，diff 为空
    flat_src = " ".join(src.read_text(encoding="utf-8").split())
    assert all(" ".join(s.split()) in flat_src for s in source_slices)
    inbox = paths.INBOX_ROOT / "bilingual" / "mini"
    assert (inbox / "中译 H0001.md").read_text(encoding="utf-8") == text


def test_submit_format_contract(ws: Path) -> None:
    """输出格式契约逐条锁死（renderer 输入契约）：句对/斜体英文/---/图片紧跟图注/透传原样。"""
    src = make_source(ws, MINI_MD)
    assert cli.cmd_prepare(str(src)) == 0
    workdir = ws / "bilingual" / "mini"
    fake_translations(workdir)
    assert cli.cmd_submit("mini", render_fn=fake_render) == 0
    text = (workdir / "bilingual.md").read_text(encoding="utf-8")
    assert text == (
        "# 中译 H0001\n"
        "*Introduction*\n"
        "---\n"
        "中译 S0001\n"
        "*This is the first sentence.*\n"
        "\n"
        "中译 S0002\n"
        "*It has a second one.*\n"
        "---\n"
        "![x](images/fig1.jpg)\n"
        "中译 C0001\n"
        "*Fig. 1.*\n"
        "\n"
        "中译 C0002\n"
        "*The experimental setup (a).*\n"
        "---\n"
        "$$\n"
        "E = mc^2\n"
        "$$\n"
        "---\n"
        "<table>\n"
        "<tr><td>1</td></tr>\n"
        "</table>\n"
        "---\n"
        "中译 T0001\n"
        "*Table 1.*\n"
        "\n"
        "中译 T0002\n"
        "*The measured data.*\n"
        "---\n"
        "中译 S0003\n"
        "*More text here.*\n"
        "\n"
        "中译 S0004\n"
        "*Second sentence here?*\n"
        "---\n"
        "## 中译 H0002\n"
        "*References*\n"
        "---\n"
        "[1] A. Author et al., Title, J. 2020.\n"
    )


def test_submit_refuses_untranslated(
    ws: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """未译（行在但内容==英文原文）：报错列出并拒绝出稿（rc=1，不产 bilingual.md）。"""
    src = make_source(ws, MINI_MD)
    assert cli.cmd_prepare(str(src)) == 0
    workdir = ws / "bilingual" / "mini"
    fake_translations(workdir)
    path = workdir / "work.md"
    path.write_text(
        path.read_text(encoding="utf-8").replace("S0002\t中译 S0002", "S0002\tIt has a second one."),
        encoding="utf-8",
    )
    assert cli.cmd_submit("mini", render_fn=fake_render) == 1
    assert "拒绝出稿" in capsys.readouterr().out
    assert not (workdir / "bilingual.md").exists()


def test_submit_refuses_no_images(ws: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """prepare 带 --no-images 出来的稿：submit 拒绝出稿（rc=2）。"""
    src = make_source(ws, MINI_MD)
    assert cli.cmd_prepare(str(src), no_images=True) == 0
    fake_translations(ws / "bilingual" / "mini")
    assert cli.cmd_submit("mini", render_fn=fake_render) == 2
    assert "拒绝出稿" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# submit 串联 render + 交付副本（md/pdf 成对、只增不删）
# ---------------------------------------------------------------------------
def test_submit_renders_pdf_and_copies_md_and_pdf_to_inbox(
    ws: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """submit 自动串联 render，md+pdf 一并复制用户 inbox。"""
    src = make_source(ws, MINI_MD)
    assert cli.cmd_prepare(str(src)) == 0
    fake_translations(ws / "bilingual" / "mini")
    assert cli.cmd_submit("mini", render_fn=fake_render) == 0
    workdir = ws / "bilingual" / "mini"
    assert (workdir / "bilingual.pdf").is_file()
    inbox = paths.INBOX_ROOT / "bilingual" / "mini"
    assert (inbox / "中译 H0001.md").is_file()
    assert (inbox / "中译 H0001.pdf").is_file()
    assert "PDF 已复制到用户 inbox" in capsys.readouterr().out


def test_submit_render_failure_warns_but_md_delivers(
    ws: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """PDF 渲染失败=警告不阻塞 md 交付（stderr 报因），inbox 只有 md 无 pdf。"""

    def boom(_md: Path, _pdf: Path) -> Path:
        raise RuntimeError("no chrome")

    src = make_source(ws, MINI_MD)
    assert cli.cmd_prepare(str(src)) == 0
    fake_translations(ws / "bilingual" / "mini")
    assert cli.cmd_submit("mini", render_fn=boom) == 0
    inbox = paths.INBOX_ROOT / "bilingual" / "mini"
    assert (inbox / "中译 H0001.md").is_file()
    assert not (inbox / "中译 H0001.pdf").exists()
    err = capsys.readouterr().err
    assert "警告" in err and "PDF 渲染失败" in err


def test_submit_custom_out_dir_lands_there(
    ws: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """--out-dir 自定义交付目录：md+pdf 落此处（目录不存在则建），默认 inbox 不再落。"""
    src = make_source(ws, MINI_MD)
    assert cli.cmd_prepare(str(src)) == 0
    fake_translations(ws / "bilingual" / "mini")
    custom = paths.INBOX_ROOT / "research" / "1_example" / "bilingual"
    assert cli.cmd_submit("mini", out_dir=str(custom), render_fn=fake_render) == 0
    assert (custom / "中译 H0001.md").is_file()
    assert (custom / "中译 H0001.pdf").is_file()
    assert not (paths.INBOX_ROOT / "bilingual" / "mini").exists()
    assert "已复制到自定义交付目录" in capsys.readouterr().out


def test_submit_custom_out_dir_in_data(ws: Path) -> None:
    """--out-dir 落 data/（Agent 管理区）同样合法：交付双根护栏两侧行为一致。"""
    src = make_source(ws, MINI_MD)
    assert cli.cmd_prepare(str(src)) == 0
    fake_translations(ws / "bilingual" / "mini")
    custom = paths.DATA_ROOT / "research" / "1_example" / "bilingual"
    assert cli.cmd_submit("mini", out_dir=str(custom), render_fn=fake_render) == 0
    assert (custom / "中译 H0001.md").is_file()
    assert not (paths.INBOX_ROOT / "bilingual" / "mini").exists()


def test_submit_rejects_out_dir_outside_data(
    ws: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """--out-dir 越出 data/ 根：防 stray 护栏拒绝（rc=2），零副作用不落任何副本。"""
    src = make_source(ws, MINI_MD)
    assert cli.cmd_prepare(str(src)) == 0
    fake_translations(ws / "bilingual" / "mini")
    outside = paths.DATA_ROOT.parent / "somewhere-else"
    assert cli.cmd_submit("mini", out_dir=str(outside), render_fn=fake_render) == 2
    assert not outside.exists(), "被拒时不得创建越界目录"
    assert not (paths.INBOX_ROOT / "bilingual" / "mini").exists()
    assert "--out-dir 拒绝" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# render 独立重试入口
# ---------------------------------------------------------------------------
def test_cmd_render_missing_bilingual_md(ws: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """render 而无出稿：rc=2 并提示先 submit。"""
    assert cli.cmd_render("mini", render_fn=fake_render) == 2
    assert "先跑 bilingual submit" in capsys.readouterr().out


def test_cmd_render_writes_pdf(ws: Path) -> None:
    """render 有出稿时调用渲染层并落 PDF（打桩验证接线）。"""
    src = make_source(ws, MINI_MD)
    assert cli.cmd_prepare(str(src)) == 0
    fake_translations(ws / "bilingual" / "mini")
    assert cli.cmd_submit("mini", render_fn=fake_render) == 0
    pdf = ws / "bilingual" / "mini" / "bilingual.pdf"
    pdf.unlink()
    assert cli.cmd_render("mini", render_fn=fake_render) == 0
    assert pdf.is_file()


# ---------------------------------------------------------------------------
# 标题命名 / 摘要标注（2026-10-23 裁决、2026-11-03 修订）
# ---------------------------------------------------------------------------
def test_output_basename_sanitize_truncate_fallback() -> None:
    """命名：非法字符清洗、超长截断、无标题回退首单元、空回退 bilingual。"""
    man = {
        "units": [
            {"id": "H0001", "type": "H", "order": 0, "en": "Title"},
            {"id": "S0001", "type": "S", "order": 1, "en": "Body"},
        ]
    }
    base = manifest.output_basename(man, {"H0001": "a/b: c*?" + "长" * 80 + "|t"})
    assert not any(ch in base for ch in '/\\:*?"<>|')
    assert len(base) <= manifest._NAME_MAX
    man2 = {"units": [{"id": "S0001", "type": "S", "order": 0, "en": "Body"}]}
    assert manifest.output_basename(man2, {"S0001": "首句中文"}) == "首句中文"
    assert manifest.output_basename(man2, {}) == "bilingual"


ABS_MD = """Article

# A Study of Example Systems

https://doi.org/10.1000/xyz1

Received: 1 January 2026

Alice Smith $^{1}$ , Bob Jones $^{2,3}$

This is the abstract first sentence. It continues in the same abstract paragraph.

Body paragraph one lives here. It has two sentences.

## Results

Result sentence one here.
"""


def test_prepare_marks_abstract_in_manifest_and_work(
    ws: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """摘要识别：元数据/作者行后首个段落全组置 role=abstract，work.md 注释行可辨。"""
    src = make_source(ws, ABS_MD, name="abs.md")
    assert cli.cmd_prepare(str(src)) == 0
    workdir = ws / "bilingual" / "abs"
    man = json.loads((workdir / "manifest.json").read_text(encoding="utf-8"))
    assert {u["en"] for u in man["units"] if u.get("role") == "abstract"} == {
        "This is the abstract first sentence.",
        "It continues in the same abstract paragraph.",
    }
    work = (workdir / "work.md").read_text(encoding="utf-8")
    assert "# ↓摘要段" in work and "# ↑摘要段结束" in work
    assert "摘要段已标注" in capsys.readouterr().out


def test_submit_shows_cn_markers_only(ws: Path) -> None:
    """出稿只插中文小标题（`## 摘要`/`## 正文`），无管线发明英文行（2026-11-03 裁决）。"""
    src = make_source(ws, ABS_MD, name="abs.md")
    assert cli.cmd_prepare(str(src)) == 0
    fake_translations(ws / "bilingual" / "abs")
    assert cli.cmd_submit("abs", render_fn=fake_render) == 0
    text = (ws / "bilingual" / "abs" / "bilingual.md").read_text(encoding="utf-8")
    assert "## 摘要" in text and "## 正文" in text
    assert "*Abstract*" not in text and "*Main text*" not in text
    assert "## 摘要" in text.splitlines() and "## 正文" in text.splitlines()


# ---------------------------------------------------------------------------
# TM 回填（--reuse 升级；fake_tm 把存储层换离线替身，决策层照常受测）
# ---------------------------------------------------------------------------
def test_prepare_reuse_backfills_and_keeps_unmatched(
    ws: Path, fake_tm: None, capsys: pytest.CaptureFixture[str]
) -> None:
    """--reuse 沉淀 TM 并回填旧译文；未匹配保持未译（submit 拒绝，符合契约）。"""
    import shutil

    src = make_source(ws, ABS_MD, name="abs.md")
    assert cli.cmd_prepare(str(src)) == 0
    workdir = ws / "bilingual" / "abs"
    man = json.loads((workdir / "manifest.json").read_text(encoding="utf-8"))
    pick = {
        u["id"]
        for u in man["units"]
        if u["en"] in ("A Study of Example Systems", "This is the abstract first sentence.")
    }
    old = ws / "old"
    old.mkdir()
    shutil.copy2(workdir / "manifest.json", old / "manifest.json")
    rows = (workdir / "work.md").read_text(encoding="utf-8").split("\n")
    old_rows = [
        f"{ln.partition(chr(9))[0]}\t旧译 {ln.partition(chr(9))[0]}"
        if ln.partition("\t")[1] and ln.partition("\t")[0] in pick
        else ln
        for ln in rows
    ]
    (old / "work.md").write_text("\n".join(old_rows), encoding="utf-8")

    assert cli.cmd_prepare(str(src), reuse=str(old)) == 0
    assert "--reuse 回填 2/" in capsys.readouterr().out
    rows2 = (workdir / "work.md").read_text(encoding="utf-8")
    assert sum(1 for ln in rows2.split("\n") if "\t旧译 " in ln) == 2
    assert cli.cmd_submit("abs", render_fn=fake_render) == 1  # 未匹配保持未译 → 拒绝出稿
    assert "拒绝出稿" in capsys.readouterr().out


def test_reuse_skips_placeholder_translations_with_warning(
    ws: Path, fake_tm: None, capsys: pytest.CaptureFixture[str]
) -> None:
    """--reuse：占位译文告警跳过不回填（防夹具扩散），真实译文照常回填。"""
    import shutil

    src = make_source(ws, MINI_MD)
    assert cli.cmd_prepare(str(src)) == 0
    workdir = ws / "bilingual" / "mini"
    old = ws / "old2"
    old.mkdir()
    shutil.copy2(workdir / "manifest.json", old / "manifest.json")
    rows = (workdir / "work.md").read_text(encoding="utf-8").split("\n")
    out_rows = []
    for ln in rows:
        uid, sep, _tail = ln.partition("\t")
        if sep and uid == "H0001":
            out_rows.append(f"{uid}\t真译标题文本")
        elif sep:
            out_rows.append(f"{uid}\t假译：{uid}")
        else:
            out_rows.append(ln)
    (old / "work.md").write_text("\n".join(out_rows), encoding="utf-8")

    assert cli.cmd_prepare(str(src), reuse=str(old)) == 0
    out = capsys.readouterr().out
    assert "--reuse 回填 1/" in out and "跳过占位译文" in out
    rows2 = (workdir / "work.md").read_text(encoding="utf-8")
    assert "H0001\t真译标题文本" in rows2
    assert "假译" not in rows2, "占位译文不得回填进新工作区"


# ---------------------------------------------------------------------------
# 防覆写闸（backlog 20261023-163700-devops）
# ---------------------------------------------------------------------------
def test_prepare_refuses_overwrite_of_translated_workspace(
    ws: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """默认路径：含译文工作区再 prepare → 拒绝（rc=2），旧译文原样保全。"""
    src = make_source(ws, MINI_MD)
    assert cli.cmd_prepare(str(src)) == 0
    workdir = ws / "bilingual" / "mini"
    fake_translations(workdir)
    before = (workdir / "work.md").read_text(encoding="utf-8")

    assert cli.cmd_prepare(str(src)) == 2
    out = capsys.readouterr().out
    assert "拒绝覆写" in out and "--force" in out
    assert (workdir / "work.md").read_text(encoding="utf-8") == before, "译文不得被动一字"
    assert not list(workdir.glob("*.bak-*")), "拒绝路径不得产生备份（没覆写就没备份）"


def test_prepare_force_overwrites_with_backup(
    ws: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """--force 路径：放行覆写并自动备份旧 work.md/manifest（*.bak-<ts>）。"""
    src = make_source(ws, MINI_MD)
    assert cli.cmd_prepare(str(src)) == 0
    workdir = ws / "bilingual" / "mini"
    fake_translations(workdir)
    before = (workdir / "work.md").read_text(encoding="utf-8")

    assert cli.cmd_prepare(str(src), force=True) == 0
    out = capsys.readouterr().out
    assert "--force" in out and "已备份为 *.bak-" in out
    baks = sorted(workdir.glob("work.md.bak-*"))
    assert len(baks) == 1 and baks[0].read_text(encoding="utf-8") == before
    assert list(workdir.glob("manifest.json.bak-*"))
    fresh = (workdir / "work.md").read_text(encoding="utf-8")
    assert "中译 H0001" not in fresh and "H0001\tIntroduction" in fresh


# ---------------------------------------------------------------------------
# 占位假译（backlog 20261023-162903-lit：夹具不计已译、不回填、拒出稿）
# ---------------------------------------------------------------------------
def test_placeholder_classification_rules() -> None:
    """固定前缀启发式 + 「仅差固定前缀」等值判据；既有测试夹具形态不误伤。"""
    from pysci.skills.literature_research.bilingual import reconcile

    assert reconcile.is_placeholder_translation("假译：Ziling Liu $^{1,6}$", "Ziling Liu $^{1,6}$")
    assert reconcile.is_placeholder_translation("假译：旧译-S0001-0", "Any English here")
    assert reconcile.is_placeholder_translation("旧译-H0001-1", "Title")
    assert reconcile.is_placeholder_translation("中译 Introduction", "Introduction")
    assert not reconcile.is_placeholder_translation("中译 H0001", "Introduction")
    assert not reconcile.is_placeholder_translation("文章", "Article")
    assert not reconcile.is_placeholder_translation("原胞与能带", "")


def test_status_and_submit_treat_placeholder_as_untranslated(
    ws: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """占位译文不计已译：status 显式提示，submit 拒绝出稿（夹具不得成交付物）。"""
    src = make_source(ws, MINI_MD)
    assert cli.cmd_prepare(str(src)) == 0
    workdir = ws / "bilingual" / "mini"
    path = workdir / "work.md"
    lines = path.read_text(encoding="utf-8").split("\n")
    for k, ln in enumerate(lines):
        uid, sep, _tail = ln.partition("\t")
        if sep and re.match(r"^[A-Z]\d{4}\t", ln):
            lines[k] = f"{uid}\t假译：{uid}"
    path.write_text("\n".join(lines), encoding="utf-8")

    assert cli.cmd_status("mini") == 0
    out = capsys.readouterr().out
    assert "已译 0" in out and "占位译文" in out and "不计已译" in out
    assert cli.cmd_submit("mini", render_fn=fake_render) == 1
    out = capsys.readouterr().out
    assert "拒绝出稿" in out and "占位译文" in out


# ---------------------------------------------------------------------------
# check-work 护栏（pod-guard 等价物：submit 前强制 + 显式 override）
# ---------------------------------------------------------------------------
def test_check_work_passes_on_intact_workspace(ws: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """结构完好：check-work 通过（rc=0）。"""
    src = make_source(ws, MINI_MD)
    assert cli.cmd_prepare(str(src)) == 0
    assert cli.cmd_check_work("mini") == 0
    assert "check-work 通过" in capsys.readouterr().out


def test_check_work_blocks_edited_ids(ws: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """改行首 ID（pod-guard 时代的硬拦语义）：check-work 报 ID 前缀序列不一致（rc=2）。"""
    src = make_source(ws, MINI_MD)
    assert cli.cmd_prepare(str(src)) == 0
    path = ws / "bilingual" / "mini" / "work.md"
    path.write_text(
        path.read_text(encoding="utf-8").replace("S0002\t", "S0099\t"), encoding="utf-8"
    )
    assert cli.cmd_check_work("mini") == 2
    out = capsys.readouterr().out
    assert "ID 前缀序列" in out


def test_check_work_blocks_deleted_and_stray_lines(
    ws: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """删行/游离行（缺行首 ID）都被拦（行数与 ID 行结构两口径）。"""
    src = make_source(ws, MINI_MD)
    assert cli.cmd_prepare(str(src)) == 0
    path = ws / "bilingual" / "mini" / "work.md"
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace("S0002\tIt has a second one.\n", ""), encoding="utf-8")
    assert cli.cmd_check_work("mini") == 2
    assert "行数不等" in capsys.readouterr().out

    path.write_text(text + "游离一行\n", encoding="utf-8")
    assert cli.cmd_check_work("mini") == 2
    assert "缺行首 ID" in capsys.readouterr().out


def test_submit_enforces_check_work_with_explicit_override(
    ws: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """submit 前强制 check-work：结构破损拒出稿（rc=2）；--skip-check-work 显式 override。"""
    src = make_source(ws, MINI_MD)
    assert cli.cmd_prepare(str(src)) == 0
    workdir = ws / "bilingual" / "mini"
    fake_translations(workdir)
    path = workdir / "work.md"
    path.write_text(
        path.read_text(encoding="utf-8").replace("S0002\t中译 S0002", "S0099\t中译 S0002"),
        encoding="utf-8",
    )
    assert cli.cmd_submit("mini", render_fn=fake_render) == 2
    out = capsys.readouterr().out
    assert "check-work 未过" in out and "--skip-check-work" in out
    assert not (workdir / "bilingual.md").exists()
    # override 后继续走对账：S0002 缺行 → 未译拒出稿（rc=1），护栏不静默吞。
    assert cli.cmd_submit("mini", render_fn=fake_render, skip_check_work=True) == 1
    assert "override" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# reorder-suggest 转接（reorder.py 归并行者，此处只锁 CLI 接线）+ quality 闸
# ---------------------------------------------------------------------------
def test_reorder_suggest_delegates_to_reorder_module(ws: Path) -> None:
    """reorder-suggest：解析源与 PDF 后转接 reorder.suggest_reorder（注入打桩验接线）。"""
    captured: dict[str, object] = {}

    def fake_suggest(md: Path, pdf: Path, out_dir: Path) -> str:
        captured["md"], captured["pdf"], captured["out"] = md, pdf, out_dir
        return "ok"

    src = make_source(ws, MINI_MD)
    pdf = src.parent / "mini.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    assert cli.cmd_reorder_suggest(str(src), suggest_fn=fake_suggest) == 0
    assert captured["md"] == src and captured["pdf"] == pdf
    assert captured["out"] == ws / "bilingual" / "mini"


def test_reorder_suggest_requires_pdf(ws: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """缺源 PDF：rc=2 并提示 --pdf。"""
    src = make_source(ws, MINI_MD)
    assert cli.cmd_reorder_suggest(str(src), suggest_fn=lambda *_: "x") == 2
    assert "--pdf" in capsys.readouterr().out


def test_quality_glossary_gate_flags_fail(ws: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """quality --glossary：术语缺规范译名 → fail 级发现（rc=1）。"""
    import json as _json

    src = make_source(ws, MINI_MD)
    assert cli.cmd_prepare(str(src)) == 0
    fake_translations(ws / "bilingual" / "mini")
    glossary = ws / "glossary.json"
    glossary.write_text(_json.dumps({"sentence": "语句"}, ensure_ascii=False), encoding="utf-8")
    assert cli.cmd_quality("mini", glossary=glossary) == 1
    assert "term" in capsys.readouterr().out


def test_quality_back_check_degrades_without_key(
    ws: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """quality --back-check 无 DEEPL_AUTH_KEY：跳过回译校验并提示（rc=0 不阻塞）。"""
    monkeypatch.delenv("DEEPL_AUTH_KEY", raising=False)
    monkeypatch.delenv("DEEPL_API_KEY", raising=False)
    src = make_source(ws, MINI_MD)
    assert cli.cmd_prepare(str(src)) == 0
    fake_translations(ws / "bilingual" / "mini")
    assert cli.cmd_quality("mini", back_check=True) == 0
    assert "跳过回译校验" in capsys.readouterr().out
