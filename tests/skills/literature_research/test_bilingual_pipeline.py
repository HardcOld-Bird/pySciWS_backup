"""bilingual prepare/status/submit 管线回归（任务书 20261023-001431 §三「prepare/submit」面）。

tmp_path 全隔离：settings/工作区/inbox 都重定向，图片回补打桩（网络回补按 §三
单独手工验证，不进默认套件）。submit 重点锁「英文侧逐字还原（=源文档切片）」与
「未译/缺 ID 拒绝出稿」。
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from pysci.skills.literature_research.tools import bilingual as bl

#: 真回补函数原件（ws fixture 会打桩；需要真缺源路径的用例覆盖回它）。
REAL_BACKFILL = bl.backfill_images

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


@pytest.fixture
def ws(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """把 bilingual 工作区/inbox/缓存全重定向进 tmp_path；回补/渲染打桩（离线确定）。"""
    import pysci.paths as paths
    from pysci.skills.literature_research.tools import bilingual_render as brr
    from pysci.skills.literature_research.tools import cache_manager as cm

    data_root = tmp_path / "data"
    module_dir = data_root / "skills" / "literature_research"
    (module_dir / "cache" / "extracted").mkdir(parents=True)
    (module_dir / "cache" / "pdfs").mkdir(parents=True)
    (module_dir / "cache" / "html_fulltext").mkdir(parents=True)
    monkeypatch.setattr(paths, "DATA_ROOT", data_root)
    monkeypatch.setattr(
        bl,
        "settings",
        dataclasses.replace(
            bl.settings,
            module_dir=module_dir,
            cache_extracted=module_dir / "cache" / "extracted",
            cache_pdfs=module_dir / "cache" / "pdfs",
            cache_html_fulltext=module_dir / "cache" / "html_fulltext",
            mineru_token=None,
        ),
    )
    monkeypatch.setattr(cm, "settings", bl.settings)
    monkeypatch.setattr(bl, "INBOX_ROOT", data_root / "用户的inbox" / "bilingual")

    def fake_backfill(
        workdir: Path, refs: list[str], source_pdf: Path | None, *, no_images: bool
    ) -> None:
        if no_images or not refs:
            return
        img_dir = workdir / "images"
        img_dir.mkdir(parents=True, exist_ok=True)
        for r in refs:
            (img_dir / r.rsplit("/", 1)[-1]).write_bytes(b"fake-img")

    monkeypatch.setattr(bl, "backfill_images", fake_backfill)

    def fake_render(md_path: Path, pdf_path: Path, *, keep_html: bool = True) -> Path:
        Path(pdf_path).write_bytes(b"%PDF-1.4 fake")
        return Path(pdf_path)

    monkeypatch.setattr(brr, "render_pdf", fake_render)
    return module_dir


def make_source(module_dir: Path, md: str = MINI_MD, name: str = "mini.md") -> Path:
    """把合成 fixture 落成源 minerU markdown，返回路径。"""
    src = module_dir / "cache" / "extracted" / name
    src.write_text(md, encoding="utf-8")
    return src


def fake_translations(workdir: Path) -> dict[str, str]:
    """把 work.md 每个 ID 的英文行换成带 CJK 的假翻译（模拟翻译组员）。"""
    path = workdir / "work.md"
    out: dict[str, str] = {}
    lines = path.read_text(encoding="utf-8").split("\n")
    for k, line in enumerate(lines):
        uid, sep, _tail = line.partition("\t")
        if sep and bl.ID_PREFIX_RE.match(line):
            out[uid] = f"中译 {uid}"
            lines[k] = f"{uid}\t{out[uid]}"
    path.write_text("\n".join(lines), encoding="utf-8")
    return out


def test_prepare_writes_work_and_manifest(
    ws: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """prepare：work.md 行格式（ID<TAB>英文）、manifest 单元/透传块齐全、切片自检过。"""
    src = make_source(ws)
    rc = bl.cmd_prepare(str(src))
    assert rc == 0
    workdir = ws / "bilingual" / "mini"
    work_md = (workdir / "work.md").read_text(encoding="utf-8").split("\n")
    assert work_md[0].startswith("# 翻译规则：")
    id_lines = [ln for ln in work_md if ln and not ln.startswith("#")]
    assert id_lines[0] == "H0001\tIntroduction"
    assert "S0001\tThis is the first sentence." in id_lines
    assert all(bl.ID_PREFIX_RE.match(ln) for ln in id_lines)

    manifest = json.loads((workdir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["source_md"] == str(src)
    assert [u["id"] for u in manifest["units"]][0] == "H0001"
    assert {u["type"] for u in manifest["units"]} == {"H", "S", "C", "T"}
    assert [b["kind"] for b in manifest["blocks"]] == [
        "image",
        "formula",
        "table",
        "reference",
    ]
    assert manifest["images"] == ["images/fig1.jpg"]
    assert manifest["images_closed"] is True
    assert "翻译指引" in capsys.readouterr().out


def test_prepare_missing_source_reports_hint(
    ws: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """源 md 不存在：报「先走既有 pdf_extract/read 抽取」并 rc=2。"""
    rc = bl.cmd_prepare("no_such_doc")
    assert rc == 2
    assert "pdf_extract" in capsys.readouterr().out


def test_prepare_no_images_skips_backfill_and_flags_manifest(ws: Path) -> None:
    """--no-images：跳过回补（images/ 不建），manifest 记 no_images=True 供 submit 拒稿。"""
    src = make_source(ws)
    rc = bl.cmd_prepare(str(src), no_images=True)
    assert rc == 0
    workdir = ws / "bilingual" / "mini"
    assert not (workdir / "images").exists()
    manifest = json.loads((workdir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["no_images"] is True


def test_prepare_accepts_unspaced_panel_numbers(ws: Path) -> None:
    """数字紧接单字母面板号（``Fig. 3(a).``/``Figs. 3(b)–(d).``）不再被「切片自检失败」
    拒绝（backlog 20261029-121943-lit：pysbd 在 ``(`` 前误断句、回并硬插空格致错位）。"""
    src = make_source(
        ws,
        md=(
            "# Results\n\n"
            "As shown in Fig. 3(a). What we see here is clear. "
            "Figs. 3(b)–(d). More details follow.\n"
        ),
        name="panel.md",
    )
    assert bl.cmd_prepare(str(src)) == 0
    workdir = ws / "bilingual" / "panel"
    work_md = (workdir / "work.md").read_text(encoding="utf-8").split("\n")
    en_lines = [ln.split("\t", 1)[1] for ln in work_md if ln and "\t" in ln]
    assert "As shown in Fig. 3(a)." in en_lines
    assert "Figs. 3(b)–(d)." in en_lines


def test_status_counts_and_lists_untranslated(
    ws: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """status：缺 ID/内容==en 记未译，含 CJK 记已译，其余存疑；输出计数与未译 ID。"""
    src = make_source(ws)
    assert bl.cmd_prepare(str(src)) == 0
    workdir = ws / "bilingual" / "mini"
    fake_translations(workdir)
    path = workdir / "work.md"
    text = path.read_text(encoding="utf-8")
    # S0002 改回英文原文＝未译；S0003 改成无 CJK 的异文＝存疑；其余保持假翻译。
    text = text.replace("S0002\t中译 S0002", "S0002\tIt has a second one.")
    text = text.replace("S0003\t中译 S0003", "S0003\tDoubtful latin text.")
    path.write_text(text, encoding="utf-8")

    assert bl.cmd_status("mini") == 0
    out = capsys.readouterr().out
    assert "未译 1" in out
    assert "存疑 1" in out
    assert "未译 ID：S0002" in out


def test_submit_english_side_is_source_slices(ws: Path) -> None:
    """submit：英文侧从 manifest 重嵌入＝源文档切片（diff 为空），且落 inbox 副本。"""
    src = make_source(ws)
    assert bl.cmd_prepare(str(src)) == 0
    workdir = ws / "bilingual" / "mini"
    fake_translations(workdir)
    assert bl.cmd_submit("mini") == 0

    text = (workdir / "bilingual.md").read_text(encoding="utf-8")
    manifest = json.loads((workdir / "manifest.json").read_text(encoding="utf-8"))
    italic = [
        ln[1:-1] for ln in text.split("\n") if ln.startswith("*") and ln.endswith("*")
    ]
    source_slices = [u["en"] for u in manifest["units"]]
    assert italic == source_slices  # 英文侧逐字还原，diff 为空
    # 且每个切片逐字存在于源文档（空白归一后）——英文与原文逐字一致。
    flat_src = " ".join(src.read_text(encoding="utf-8").split())
    assert all(" ".join(s.split()) in flat_src for s in source_slices)
    assert (bl.INBOX_ROOT / "mini" / "中译 H0001.md").read_text(
        encoding="utf-8"
    ) == text


def test_submit_format_contract(ws: Path) -> None:
    """输出格式契约逐条锁死（环节 2 渲染依赖）：句对/斜体英文/---/图片紧跟图注/透传原样。"""
    src = make_source(ws)
    assert bl.cmd_prepare(str(src)) == 0
    workdir = ws / "bilingual" / "mini"
    fake_translations(workdir)
    assert bl.cmd_submit("mini") == 0
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
    """未译/缺 ID：报错列出并拒绝出稿（rc=1，不产 bilingual.md）。"""
    src = make_source(ws)
    assert bl.cmd_prepare(str(src)) == 0
    workdir = ws / "bilingual" / "mini"
    fake_translations(workdir)
    path = workdir / "work.md"
    path.write_text(
        path.read_text(encoding="utf-8").replace("S0002\t中译 S0002\n", ""),
        encoding="utf-8",
    )
    assert bl.cmd_submit("mini") == 1
    assert "拒绝出稿" in capsys.readouterr().out
    assert not (workdir / "bilingual.md").exists()


def test_submit_refuses_no_images(ws: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """prepare 带 --no-images 出来的稿：submit 拒绝出稿。"""
    src = make_source(ws)
    assert bl.cmd_prepare(str(src), no_images=True) == 0
    fake_translations(ws / "bilingual" / "mini")
    assert bl.cmd_submit("mini") == 2
    assert "拒绝出稿" in capsys.readouterr().out


def test_backfill_skip_on_no_images(tmp_path: Path) -> None:
    """no_images=True 时回补直接跳过（不触网、不建 images/）。"""
    bl.backfill_images(tmp_path, ["images/a.jpg"], None, no_images=True)
    assert not (tmp_path / "images").exists()


# ---------------------------------------------------------------------------
# submit 串联 render + inbox 双文件复制（任务书 20261023-011225 §二.4）
# ---------------------------------------------------------------------------
def test_submit_renders_pdf_and_copies_md_and_pdf_to_inbox(
    ws: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """submit 自动串联 render，md+pdf 一并复制用户 inbox（只增不删）。"""
    src = make_source(ws)
    assert bl.cmd_prepare(str(src)) == 0
    fake_translations(ws / "bilingual" / "mini")
    assert bl.cmd_submit("mini") == 0
    workdir = ws / "bilingual" / "mini"
    assert (workdir / "bilingual.pdf").is_file()
    inbox = bl.INBOX_ROOT / "mini"
    assert (inbox / "中译 H0001.md").is_file()
    assert (inbox / "中译 H0001.pdf").is_file()
    assert "PDF 已复制到用户 inbox" in capsys.readouterr().out


def test_submit_render_failure_warns_but_md_delivers(
    ws: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """PDF 渲染失败=警告不阻塞 md 交付（stderr 报因），inbox 只有 md 无 pdf。"""
    from pysci.skills.literature_research.tools import bilingual_render as brr

    def boom(*_a: object, **_k: object) -> Path:
        raise RuntimeError("no chrome")

    monkeypatch.setattr(brr, "render_pdf", boom)
    src = make_source(ws)
    assert bl.cmd_prepare(str(src)) == 0
    fake_translations(ws / "bilingual" / "mini")
    assert bl.cmd_submit("mini") == 0
    inbox = bl.INBOX_ROOT / "mini"
    assert (inbox / "中译 H0001.md").is_file()
    assert not (inbox / "中译 H0001.pdf").exists()
    err = capsys.readouterr().err
    assert "警告" in err and "PDF 渲染失败" in err


# ---------------------------------------------------------------------------
# submit 交付目录可自定义（--out-dir；adhoc-20261023-103449 用户点名返修）
# ---------------------------------------------------------------------------
def test_submit_custom_out_dir_lands_there(
    ws: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """--out-dir 自定义交付目录：md+pdf 落此处（目录不存在则建），默认 inbox 不再落。"""
    src = make_source(ws)
    assert bl.cmd_prepare(str(src)) == 0
    fake_translations(ws / "bilingual" / "mini")
    custom = ws.parents[1] / "用户的inbox" / "research" / "1_example" / "bilingual"
    assert bl.cmd_submit("mini", out_dir=str(custom)) == 0
    assert (custom / "中译 H0001.md").is_file()
    assert (custom / "中译 H0001.pdf").is_file()
    assert not (bl.INBOX_ROOT / "mini").exists(), "提供 --out-dir 时不应再落默认位置"
    assert "已复制到自定义交付目录" in capsys.readouterr().out


def test_submit_without_out_dir_falls_back_to_default(
    ws: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """不提供 --out-dir：行为与现状完全一致，落 data/用户的inbox/bilingual/<stem>/。"""
    src = make_source(ws)
    assert bl.cmd_prepare(str(src)) == 0
    fake_translations(ws / "bilingual" / "mini")
    assert bl.cmd_submit("mini") == 0
    inbox = bl.INBOX_ROOT / "mini"
    assert (inbox / "中译 H0001.md").is_file()
    assert (inbox / "中译 H0001.pdf").is_file()
    assert "已复制到用户 inbox" in capsys.readouterr().out


def test_submit_rejects_out_dir_outside_data(
    ws: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """--out-dir 越出 data/ 根：防 stray 护栏拒绝（rc=2），零副作用不落任何副本。"""
    src = make_source(ws)
    assert bl.cmd_prepare(str(src)) == 0
    fake_translations(ws / "bilingual" / "mini")
    outside = ws.parents[2] / "somewhere-else"
    assert bl.cmd_submit("mini", out_dir=str(outside)) == 2
    assert not outside.exists(), "被拒时不得创建越界目录"
    assert not (bl.INBOX_ROOT / "mini").exists()
    assert "--out-dir 拒绝" in capsys.readouterr().err


def test_cmd_render_missing_bilingual_md(
    ws: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """render 而无出稿：rc=2 并提示先 submit。"""
    assert bl.cmd_render("mini") == 2
    assert "先跑 bilingual submit" in capsys.readouterr().out


def test_cmd_render_writes_pdf(ws: Path) -> None:
    """render 有出稿时调用渲染层并落 PDF（打桩验证接线）。"""
    src = make_source(ws)
    assert bl.cmd_prepare(str(src)) == 0
    fake_translations(ws / "bilingual" / "mini")
    assert bl.cmd_submit("mini") == 0
    pdf = ws / "bilingual" / "mini" / "bilingual.pdf"
    pdf.unlink()
    assert bl.cmd_render("mini") == 0
    assert pdf.is_file()


# ---------------------------------------------------------------------------
# 源 PDF 反解升级（backlog 20261023-010824-devops）
# ---------------------------------------------------------------------------
DOI_MD = """# Paper Title

DOI: 10.1038/ncomms11110

Some text here.

---
![x](images/a.jpg)
"""


def test_extract_pdf_identifiers() -> None:
    """md 头部提取 DOI 与 arXiv id；头部窗口外（参考文献区）不抓。"""
    idents = bl.extract_pdf_identifiers(
        "# T\n\nDOI: 10.1038/ncomms11110\n\narXiv:1803.04110v2\n"
    )
    assert "10.1038/ncomms11110" in idents
    assert "1803.04110" in idents
    far = "x" * 5000 + "\nDOI: 10.9999/late.123\n"
    assert bl.extract_pdf_identifiers(far) == []


def test_find_source_pdf_resolves_doi_named_pdf(ws: Path) -> None:
    """DOI 命名的缓存 PDF（read/browser_fetch 面）：由 md 头部 DOI 反解到。"""
    pdf = ws / "cache" / "pdfs" / "10-1038-ncomms11110.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    src = make_source(ws, md=DOI_MD, name="accessing-ep_fulltext.md")
    assert bl._find_source_pdf("accessing-ep_fulltext", source_md=src) == pdf


def test_find_source_pdf_resolves_arxiv_named_pdf(ws: Path) -> None:
    """arXiv id 命名的缓存 PDF：同样反解到。"""
    pdf = ws / "cache" / "pdfs" / "1803.04110.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    src = make_source(
        ws, md="# T\n\narXiv:1803.04110\n", name="arxiv-paper_fulltext.md"
    )
    assert bl._find_source_pdf("arxiv-paper_fulltext", source_md=src) == pdf


def test_find_source_pdf_keeps_hash_convention(ws: Path) -> None:
    """既有 pdf_extract 命名（<pdfstem>_<hash12>.<backend>）不受影响。"""
    pdf = ws / "cache" / "pdfs" / "MyPaper.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    assert bl._find_source_pdf("MyPaper_0123456789ab.mineru-cloud") == pdf


def test_prepare_pdf_override_is_used(
    ws: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--pdf 指定后回补使用该 PDF（打桩验证传入的就是指定文件）。"""
    captured: dict[str, object] = {}

    def fake_backfill(
        workdir: Path, refs: list[str], source_pdf: Path | None, *, no_images: bool
    ) -> None:
        captured["pdf"] = source_pdf

    monkeypatch.setattr(bl, "backfill_images", fake_backfill)
    src = make_source(ws, md=DOI_MD, name="accessing-ep_fulltext.md")
    pdf = ws / "cache" / "pdfs" / "manual.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    assert bl.cmd_prepare(str(src), pdf=str(pdf)) == 0
    assert captured["pdf"] == pdf


def test_prepare_missing_pdf_lists_candidates(
    ws: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """缺源且反解不到：rc=2，列出 cache/pdfs 候选并提示 --pdf。"""
    monkeypatch.setattr(bl, "backfill_images", REAL_BACKFILL)
    cand = ws / "cache" / "pdfs" / "exceptional-points-review.pdf"
    cand.write_bytes(b"%PDF-1.4")
    src = make_source(ws, md=DOI_MD, name="accessing-exceptional-points_fulltext.md")
    assert bl.cmd_prepare(str(src)) == 2
    out = capsys.readouterr().out
    assert "--pdf" in out
    assert "exceptional-points-review.pdf" in out
    assert not (
        ws / "bilingual" / "accessing-exceptional-points_fulltext" / "manifest.json"
    ).exists()


# ---------------------------------------------------------------------------
# 标题命名 / 摘要标注 / 译文复用（bilingual-orderfix p2-2，2026-10-23 用户裁决）
# ---------------------------------------------------------------------------
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


def test_output_basename_sanitize_truncate_fallback() -> None:
    """命名：非法字符清洗、超长截断（保扩展名）、无标题回退首单元、空回退 bilingual。"""
    man = {
        "units": [
            {"id": "H0001", "type": "H", "order": 0, "en": "Title"},
            {"id": "S0001", "type": "S", "order": 1, "en": "Body"},
        ]
    }
    base = bl.output_basename(man, {"H0001": "a/b: c*?" + "长" * 80 + "|t"})
    assert not any(ch in base for ch in '/\\:*?"<>|')
    assert len(base) <= bl._NAME_MAX
    man2 = {"units": [{"id": "S0001", "type": "S", "order": 0, "en": "Body"}]}
    assert bl.output_basename(man2, {"S0001": "首句中文"}) == "首句中文"
    assert bl.output_basename(man2, {}) == "bilingual"


def test_prepare_marks_abstract_in_manifest_and_work(
    ws: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """摘要识别：元数据/作者行后首个段落全组置 role=abstract，work.md 注释行可辨。"""
    src = make_source(ws, md=ABS_MD, name="abs.md")
    assert bl.cmd_prepare(str(src)) == 0
    workdir = ws / "bilingual" / "abs"
    man = json.loads((workdir / "manifest.json").read_text(encoding="utf-8"))
    assert {u["en"] for u in man["units"] if u.get("role") == "abstract"} == {
        "This is the abstract first sentence.",
        "It continues in the same abstract paragraph.",
    }
    work = (workdir / "work.md").read_text(encoding="utf-8")
    assert "# ↓摘要段" in work and "# ↑摘要段结束" in work
    assert "摘要段已标注" in capsys.readouterr().out


def test_submit_and_render_show_cn_markers_only(
    ws: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """出稿只插中文小标题（`## 摘要`/`## 正文`），无管线发明英文行（2026-11-03
    用户裁决覆盖 10-23 旧裁决：删 `*Abstract*`/`*Main text*`，不做换行兜底）。"""
    from pysci.skills.literature_research.tools import bilingual_render as brr

    src = make_source(ws, md=ABS_MD, name="abs.md")
    assert bl.cmd_prepare(str(src)) == 0
    fake_translations(ws / "bilingual" / "abs")
    assert bl.cmd_submit("abs") == 0
    text = (ws / "bilingual" / "abs" / "bilingual.md").read_text(encoding="utf-8")
    assert "## 摘要" in text and "## 正文" in text
    assert "*Abstract*" not in text and "*Main text*" not in text
    assert "## 摘要" in text.splitlines() and "## 正文" in text.splitlines()
    pandoc_md = brr.to_pandoc_md(text)
    assert "## 摘要" in pandoc_md and "## 正文" in pandoc_md
    assert "Abstract" not in pandoc_md and "Main text" not in pandoc_md


def test_prepare_reuse_backfills_and_keeps_unmatched(
    ws: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """--reuse 按英文原文回填旧译文；未匹配保持未译（submit 拒绝，符合契约）。"""
    import shutil

    src = make_source(ws, md=ABS_MD, name="abs.md")
    assert bl.cmd_prepare(str(src)) == 0
    workdir = ws / "bilingual" / "abs"
    man = json.loads((workdir / "manifest.json").read_text(encoding="utf-8"))
    pick = {
        u["id"]
        for u in man["units"]
        if u["en"]
        in ("A Study of Example Systems", "This is the abstract first sentence.")
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

    assert bl.cmd_prepare(str(src), reuse=str(old)) == 0
    assert "--reuse 回填 2/" in capsys.readouterr().out
    rows2 = (workdir / "work.md").read_text(encoding="utf-8")
    assert sum(1 for ln in rows2.split("\n") if "\t旧译 " in ln) == 2
    assert bl.cmd_submit("abs") == 1  # 未匹配保持未译 → 拒绝出稿
    assert "拒绝出稿" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# 防覆写闸（backlog 20261023-163700-devops：含译文工作区再 prepare）
# ---------------------------------------------------------------------------
def test_prepare_refuses_overwrite_of_translated_workspace(
    ws: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """默认路径：含译文工作区再 prepare → 拒绝（rc=2），旧译文原样保全。"""
    src = make_source(ws)
    assert bl.cmd_prepare(str(src)) == 0
    workdir = ws / "bilingual" / "mini"
    fake_translations(workdir)
    before = (workdir / "work.md").read_text(encoding="utf-8")

    assert bl.cmd_prepare(str(src)) == 2
    out = capsys.readouterr().out
    assert "拒绝覆写" in out and "--force" in out
    assert (workdir / "work.md").read_text(encoding="utf-8") == before, (
        "译文不得被动一字"
    )
    assert not list(workdir.glob("*.bak-*")), "拒绝路径不得产生备份（没覆写就没备份）"


def test_prepare_force_overwrites_with_backup(
    ws: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """--force 路径：放行覆写并自动备份旧 work.md/manifest（*.bak-<ts>）。"""
    src = make_source(ws)
    assert bl.cmd_prepare(str(src)) == 0
    workdir = ws / "bilingual" / "mini"
    fake_translations(workdir)
    before = (workdir / "work.md").read_text(encoding="utf-8")

    assert bl.cmd_prepare(str(src), force=True) == 0
    out = capsys.readouterr().out
    assert "--force" in out and "已备份为 *.bak-" in out
    baks = sorted(workdir.glob("work.md.bak-*"))
    assert len(baks) == 1 and baks[0].read_text(encoding="utf-8") == before
    assert list(workdir.glob("manifest.json.bak-*"))
    fresh = (workdir / "work.md").read_text(encoding="utf-8")
    assert "中译 H0001" not in fresh and "H0001\tIntroduction" in fresh


# ---------------------------------------------------------------------------
# 占位译文识别（backlog 20261023-162903-lit：夹具不计已译、不回填）
# ---------------------------------------------------------------------------
def test_placeholder_classification_rules() -> None:
    """固定前缀启发式 + 「仅差固定前缀」等值判据；既有测试夹具形态不误伤。"""
    assert bl.is_placeholder_translation(
        "假译：Ziling Liu $^{1,6}$", "Ziling Liu $^{1,6}$"
    )
    assert bl.is_placeholder_translation("假译：旧译-S0001-0", "Any English here")
    assert bl.is_placeholder_translation("旧译-H0001-1", "Title")
    assert bl.is_placeholder_translation("中译 Introduction", "Introduction")
    assert not bl.is_placeholder_translation("中译 H0001", "Introduction")
    assert not bl.is_placeholder_translation("文章", "Article")
    assert not bl.is_placeholder_translation("原胞与能带", "")


def test_status_and_submit_treat_placeholder_as_untranslated(
    ws: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """占位译文不计已译：status 显式提示，submit 拒绝出稿（夹具不得成交付物）。"""
    src = make_source(ws)
    assert bl.cmd_prepare(str(src)) == 0
    workdir = ws / "bilingual" / "mini"
    path = workdir / "work.md"
    lines = path.read_text(encoding="utf-8").split("\n")
    for k, ln in enumerate(lines):
        uid, sep, _tail = ln.partition("\t")
        if sep and bl.ID_PREFIX_RE.match(ln):
            lines[k] = f"{uid}\t假译：{uid}"
    path.write_text("\n".join(lines), encoding="utf-8")

    assert bl.cmd_status("mini") == 0
    out = capsys.readouterr().out
    assert "已译 0" in out and "占位译文" in out and "不计已译" in out
    assert bl.cmd_submit("mini") == 1
    out = capsys.readouterr().out
    assert "拒绝出稿" in out and "占位译文" in out


def test_reuse_skips_placeholder_translations_with_warning(
    ws: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """--reuse：占位译文告警跳过不回填（防夹具扩散），真实译文照常回填。"""
    import shutil

    src = make_source(ws)
    assert bl.cmd_prepare(str(src)) == 0
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

    assert bl.cmd_prepare(str(src), reuse=str(old)) == 0
    out = capsys.readouterr().out
    assert "--reuse 回填 1/" in out and "跳过占位译文" in out
    rows2 = (workdir / "work.md").read_text(encoding="utf-8")
    assert "H0001\t真译标题文本" in rows2
    assert "假译" not in rows2, "占位译文不得回填进新工作区"
