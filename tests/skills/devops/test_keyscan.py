"""API Key 泄漏哨兵阳性测试（E3-prep，backlog 20261011-233003-leader）。

本文件里的密钥串全是**假素材**——这正是 ``keyscan.EXCLUDE_PREFIXES`` 豁免 ``tests/``
的原因：哨兵自己的用例若被自己扫中，doctor 就永远不绿。豁免只降敏不降效：真实泄漏
几乎只会发生在配置、文档与代码里。
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from pysci.skills.devops.tools import keyscan

FAKE_SK = "sk-" + "A1b2C3d4E5f6G7h8I9j0"  # 20+ 位，够命中 sk- 模式
DOC_TEXT = f"""\
# 配置说明
base_url: https://api.siliconflow.cn/v1
api_key: {FAKE_SK}Kk
"""


def _git(tmp_path: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
        timeout=60,
    )


def _repo(tmp_path: Path, files: dict[str, str]) -> Path:
    """造一个真 git 仓库并把文件写进 index（``git ls-files`` 看得见）。"""
    _git(tmp_path, "init", "-q")
    for rel, text in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        _git(tmp_path, "add", "-f", "--", rel)
    return tmp_path


# ---------------------------------------------------------------------------
# 命中口径
# ---------------------------------------------------------------------------
def test_detects_sk_prefixed_secret_with_line_number():
    findings = keyscan.scan_lines(DOC_TEXT, path="docs/jury.md")
    assert len(findings) == 1
    f = findings[0]
    assert f.path == "docs/jury.md"
    assert f.line_no == 3
    assert f.pattern_id == "sk-前缀密钥"


def test_detects_prefixed_env_var_assignment():
    """最常见泄漏形态：带服务前缀的环境变量名（旧词首 ``\\b`` 会让它整体漏报）。"""
    text = 'SILICONFLOW_API_KEY = "xR7tY2uI8oP3aS6dF9gH0jK2lZ4xC"\n'
    got = keyscan.scan_lines(text, path="src/a.py")
    assert [f.pattern_id for f in got] == ["键名=长随机值"]


def test_detects_key_name_assignment_and_bearer_token():
    text = (
        "client_secret: 9m2Qk4Xv7Zp1Rt5Yw3Eu8Io0AsDfGhJk\n"
        "header: Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6\n"
        "普通说明文字，不含密钥。\n"
    )
    got = {f.pattern_id for f in keyscan.scan_lines(text, path="src/a.py")}
    assert got == {"键名=长随机值", "Bearer 长令牌"}


def test_clean_text_produces_no_findings():
    benign = (
        "Key 从环境变量 SILICONFLOW_API_KEY 注入，永不入库。\n"
        "timeout: 25  # 秒\n"
        "sk-短不构成命中；risk-A1b2C3d4E5f6G7h8I9j0 也不是密钥\n"
        "password 字段一律留空，改由运行态注入\n"
    )
    assert keyscan.scan_lines(benign, path="README.md") == []


def test_report_never_echoes_full_secret():
    findings = keyscan.scan_lines(DOC_TEXT, path="docs/jury.md")
    joined = "\n".join(keyscan.report(findings))
    assert FAKE_SK not in joined, "报告自身不得带毒：疑似密钥必须被掩码"
    assert "sk-A…" in joined and "[✗]" in joined and "永不入库" in joined


# ---------------------------------------------------------------------------
# 豁免面与非文本文件
# ---------------------------------------------------------------------------
def test_tests_directory_is_exempt(tmp_path: Path):
    """哨兵的假 Key 写在 tests/ 下是素材，不该被自己抓住（同款内容在 src/ 必报）。"""
    repo = _repo(
        tmp_path,
        {
            "tests/skills/devops/test_keyscan.py": DOC_TEXT,
            "src/settings.py": DOC_TEXT,
        },
    )
    findings, _ = keyscan.scan(repo)
    assert [f.path for f in findings] == ["src/settings.py"]


def test_binary_and_oversize_files_are_skipped(monkeypatch, tmp_path: Path):
    (tmp_path / "blob.bin").write_bytes(b"\x00\x01" + FAKE_SK.encode() + b"\x02")
    (tmp_path / "big.txt").write_text(DOC_TEXT, encoding="utf-8")
    monkeypatch.setattr(keyscan, "MAX_FILE_BYTES", 10)
    files = ["blob.bin", "big.txt"]
    assert keyscan.scan_files(files, root=tmp_path) == []


def test_missing_file_is_tolerated(tmp_path: Path):
    """跟踪清单与磁盘不一致（如刚删未提交）不许把 doctor 拖崩。"""
    assert keyscan.scan_file("gone/nowhere.txt", root=tmp_path) == []


# ---------------------------------------------------------------------------
# 仓库级三档：✗ 泄漏 / √ 干净 / [*] 未能扫描
# ---------------------------------------------------------------------------
def test_scan_finds_leak_in_tracked_file_and_reports_x(tmp_path: Path):
    repo = _repo(tmp_path, {"src/settings.py": DOC_TEXT, "README.md": "干净文件\n"})
    findings, error = keyscan.scan(repo)
    assert error == ""
    assert [f.path for f in findings] == ["src/settings.py"]
    assert keyscan.report(findings, error)[0].startswith("  [✗] 1 处疑似")


def test_scan_clean_repo_reports_green(tmp_path: Path):
    repo = _repo(
        tmp_path,
        {
            "src/ok.py": "API key 只走环境变量，永不入库。\n",
            "tests/fake.py": DOC_TEXT,  # 豁免面内的素材不算泄露
        },
    )
    findings, error = keyscan.scan(repo)
    assert (findings, error) == ([], "")
    assert keyscan.report(findings, error)[0].startswith("  [√]")


def test_untracked_dotenv_is_not_scanned(tmp_path: Path):
    """.env 正是 Key 该待的地方（已 gitignore）——扫它会天天误报。"""
    repo = _repo(tmp_path, {"src/ok.py": "print(1)\n"})
    (repo / ".env").write_text(f"SILICONFLOW_API_KEY={FAKE_SK}Kk\n", encoding="utf-8")
    findings, error = keyscan.scan(repo)
    assert (findings, error) == ([], "")


def test_no_git_degrades_to_advisory(tmp_path: Path):
    """不在仓库里（或 git 不可用）→ advisory 行，不 raise、不假装绿。"""
    repo = tmp_path / "plain"
    repo.mkdir()
    (repo / "leak.txt").write_text(DOC_TEXT, encoding="utf-8")
    findings, error = keyscan.scan(repo)
    assert findings == [] and error
    assert keyscan.report(findings, error)[0].startswith("  [*]")
