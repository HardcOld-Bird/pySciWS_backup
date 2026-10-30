"""worktree add 播种 gitignored 测试资产（backlog 20261023-014957-devops）。

动机：全量测试依赖 gitignore 的本地资产（COMSOL 模板 ``.mph`` 等）——新 worktree 只带
git 跟踪面，「合并前自跑全量」必假红、每次人工从主树补种。播种器按
``orchestration/worktree-assets.txt`` 从主树机械复制，幂等且 fail-open。
"""

from __future__ import annotations

from pathlib import Path

from pysci.skills.devops.tools import dev


def _cfg(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "worktree-assets.txt"
    p.write_text(text, encoding="utf-8")
    return p


def test_load_skips_comments_and_blanks(tmp_path):
    cfg = _cfg(tmp_path, "# 头注释\n\n  data/a.bin  \n# 尾注释\ndata/b.bin\n")
    assert dev._load_worktree_assets(cfg) == ["data/a.bin", "data/b.bin"]


def test_load_missing_config_is_empty(tmp_path):
    """清单缺席不视为错误：返回空表（add 面据此静默跳过）。"""
    assert dev._load_worktree_assets(tmp_path / "absent.txt") == []


def test_seed_copies_exact_and_glob_entries(tmp_path):
    main = tmp_path / "main"
    wt = tmp_path / "wt"
    exact = (
        main
        / "data"
        / "research"
        / "1_example"
        / "simulation"
        / "12 示例模板_J3模板.mph"
    )
    exact.parent.mkdir(parents=True)
    exact.write_bytes(b"J3")
    other = exact.parent / "12 示例模板_J4模板.mph"
    other.write_bytes(b"J4")
    cfg = _cfg(
        tmp_path,
        "# COMSOL 模板（中文名+空格）\n"
        "data/research/1_example/simulation/12 示例模板_J3模板.mph\n"
        "data/research/1_example/simulation/12 示例模板_J*模板.mph\n",
    )

    lines = dev._seed_test_assets_from_main(wt, main_root=main, config=cfg)

    assert lines and lines[0].startswith("[√] 已播种 gitignored 测试资产 2 个")
    assert (wt / exact.relative_to(main)).read_bytes() == b"J3"
    assert (wt / other.relative_to(main)).read_bytes() == b"J4"


def test_seed_reports_missing_source_entry(tmp_path):
    """主树缺该资产 → 记一条 [i]，别的条目照常播种，不抛。"""
    main = tmp_path / "main"
    wt = tmp_path / "wt"
    present = main / "data" / "x.bin"
    present.parent.mkdir(parents=True)
    present.write_bytes(b"X")
    cfg = _cfg(tmp_path, "data/x.bin\ndata/absent.bin\n")

    lines = dev._seed_test_assets_from_main(wt, main_root=main, config=cfg)

    assert any("主树缺失" in ln and "data/absent.bin" in ln for ln in lines)
    assert any("worktree-assets.txt" in ln for ln in lines), (
        "陈旧条目的告警须指路播种名单（资产改名/搬迁后的手工同步）"
    )
    assert (wt / "data" / "x.bin").read_bytes() == b"X"


def test_seed_is_idempotent_and_never_overwrites(tmp_path):
    """已存在的目标不覆盖；重复调用报「已就位」。"""
    main = tmp_path / "main"
    wt = tmp_path / "wt"
    src = main / "data" / "x.bin"
    src.parent.mkdir(parents=True)
    src.write_bytes(b"MAIN")
    dst = wt / "data" / "x.bin"
    dst.parent.mkdir(parents=True)
    dst.write_bytes(b"WT-OWN")
    cfg = _cfg(tmp_path, "data/x.bin\n")

    lines = dev._seed_test_assets_from_main(wt, main_root=main, config=cfg)

    assert lines == ["[=] 测试资产已就位（无需播种）"]
    assert dst.read_bytes() == b"WT-OWN"


def test_seed_with_no_matching_glob_reports_line(tmp_path):
    main = tmp_path / "main"
    main.mkdir()
    cfg = _cfg(tmp_path, "data/research/**/missing_*.mph\n")

    lines = dev._seed_test_assets_from_main(tmp_path / "wt", main_root=main, config=cfg)

    assert len(lines) == 1
    assert lines[0].startswith(
        "[i] 测试资产主树缺失，跳过播种：data/research/**/missing_*.mph"
    )
    assert "worktree-assets.txt" in lines[0], "告警须指路播种名单（改名/搬迁同步）"
