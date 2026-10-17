"""pre-commit 侧 harness 预算硬闸（backlog 20261010-budget-precommit）。

被锁定的三件事：
1. **执法面**：三类路径（真本 ``SKILL.md`` / 真本 ``references/*.md`` / 根
   ``basic.md``·``leader-only.md``）每文件 ≤8192B，超限的报错文本必须含文件、实测字节、
   上限与整改动作——组长与 devops 自己改真本时被拦，不靠事后 doctor 发现。
2. **配置与脚本不漂移**：``.pre-commit-config.yaml`` 的 ``files:`` 正则决定哪些文件会
   走到钩子，脚本内的 ``TIERS`` 决定哪些文件真的被量。两层判据若不一致，就会留下
   「超限但根本不进钩子」的死角（或反过来把不相干的文件也拦下）。这里用同一批路径
   形状分别过两层，断言判定一致。
3. **上限只有一个真源**：与 ``pysci.skills.devops.tools.budget.LIMIT``（doctor 与
   预算审计用的那一份）相等，避免 doctor 报绿而钩子拦人。

脚本是 stdlib-only（钩子跑在 pre-commit 自建的空 venv 里），故测试用
``sys.executable`` 直接跑它，而不是 import 进来——测的就是钩子实际执行的那份代码。
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from pysci.paths import PROJECT_ROOT
from pysci.skills.devops.tools import budget

SCRIPT = PROJECT_ROOT / "orchestration" / "guards" / "budget_check.py"
CONFIG = PROJECT_ROOT / ".pre-commit-config.yaml"

IN_SCOPE = (
    "orchestration/skills/literature-research/SKILL.md",
    "orchestration/skills/devops/SKILL.md",
    "orchestration/skills/literature-research/references/maintenance-04-cache.md",
    ".qoder/rules/basic.md",
    ".qoder/rules/leader-only.md",
)
OUT_SCOPE = (
    "orchestration/README.md",
    "orchestration/guards/delivery-gate.mjs",
    "orchestration/pods/lit/AGENTS.md",  # 组员自维护层：归 delivery-gate 执法，不由此拦
    "orchestration/pods/lit/.qoder/rules/charter.md",  # 只读层：改它走组长流程
    "data/research/1_gain_ep/notes.md",
    "src/pysci/skills/devops/tools/budget.py",
    "README.md",
)


def run_script(cwd: Path, *paths: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *paths],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=str(cwd),
    )


def _load_guard():
    """按文件路径加载钩子脚本（它是 stdlib-only 的独立文件，不在包内）。"""
    import importlib.util

    spec = importlib.util.spec_from_file_location("budget_check", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_GUARD = _load_guard()


def hook_files_regex() -> str:
    """取 .pre-commit-config.yaml 里 harness-budget 钩子的 files 正则。"""
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    for repo in cfg["repos"]:
        for hook in repo.get("hooks", []):
            if hook.get("id") == "harness-budget":
                assert hook["language"] == "python", (
                    "钩子须用 pre-commit 自带解释器，不依赖 PATH"
                )
                assert hook["entry"].endswith("budget_check.py")
                return str(hook["files"])
    raise AssertionError("harness-budget 钩子未注册")


def _tier(path: str) -> str | None:
    """脚本自己的判据（按文件路径加载那份模块，不走 sys.path hack）。"""
    hit = _GUARD.tier_for(path)
    return hit[0] if hit else None


# ===========================================================================
#  1. 执法面与报错文本
# ===========================================================================
def test_three_shapes_are_classified():
    assert "全量注入" in _tier("orchestration/skills/x/SKILL.md")
    assert "按需" in _tier("orchestration/skills/x/references/y.md")
    assert "全量注入" in _tier(".qoder/rules/basic.md")
    assert "全量注入" in _tier(".qoder/rules/leader-only.md")


def test_out_of_scope_paths_are_ignored():
    for p in OUT_SCOPE:
        assert _tier(p) is None, p


@pytest.mark.parametrize(
    "rel",
    [
        "orchestration/skills/lit/SKILL.md",
        "orchestration/skills/lit/references/big.md",
        ".qoder/rules/basic.md",
    ],
)
def test_oversized_in_scope_file_blocks_and_explains(tmp_path: Path, rel: str):
    """三类路径都要拦，且报错含 文件 / 实测字节 / 上限 / 动作（任务书硬要求）。"""
    f = tmp_path / rel
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text("x" * (budget.LIMIT + 123), encoding="utf-8", newline="\n")

    proc = run_script(tmp_path, rel)

    assert proc.returncode == 1
    out = proc.stdout
    assert rel in out
    assert str(budget.LIMIT + 123) in out  # 实测字节
    assert str(budget.LIMIT) in out  # 上限
    assert "超出 123B" in out
    assert "动作：" in out


def test_file_at_the_limit_passes(tmp_path: Path):
    """边界与 `find -size +8192c` 一致：判据是 > LIMIT，等于不拦。"""
    rel = "orchestration/skills/lit/SKILL.md"
    f = tmp_path / rel
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text("x" * budget.LIMIT, encoding="utf-8", newline="\n")

    proc = run_script(tmp_path, rel)

    assert proc.returncode == 0, proc.stdout


def test_lf_file_is_measured_as_crlf_equivalent(tmp_path: Path):
    """LF 检出预演（backlog 20261017-125557）：worktree 里脚本写出的 LF 文件合并到 main
    后每行 +1B；钩子须按 CRLF 等价计量，否则「worktree 绿 / main 红」。"""
    rel = "orchestration/skills/lit/references/lf-big.md"
    f = tmp_path / rel
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_bytes(b"x\n" * 4000)  # 落盘 8000B → CRLF 等价 12000B
    assert _GUARD.crlf_equivalent_size(f) == 12000

    proc = run_script(tmp_path, rel)

    assert proc.returncode == 1
    assert "12000" in proc.stdout


def test_lf_file_at_crlf_limit_passes(tmp_path: Path):
    """CRLF 等价恰 8192B 的 LF 文件通过；同等价 8193B 即拦（判据 > LIMIT）。"""
    rel = "orchestration/skills/lit/SKILL.md"
    f = tmp_path / rel
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_bytes(b"x\n" * 1000 + b"x" * 5192)  # 7192 落盘 + 1000 裸 LF = 8192
    assert _GUARD.crlf_equivalent_size(f) == budget.LIMIT
    assert run_script(tmp_path, rel).returncode == 0

    f.write_bytes(b"x\n" * 1000 + b"x" * 5193)  # 等效 8193
    proc = run_script(tmp_path, rel)
    assert proc.returncode == 1
    assert "8193" in proc.stdout


def test_crlf_metric_has_one_source(tmp_path: Path):
    """钩子与 doctor 侧的 CRLF 等价读数必须逐字节相等（LF/CRLF/混合三态）。"""
    f = tmp_path / "mixed.md"
    f.write_bytes(b"a\r\nb\nc")
    assert _GUARD.crlf_equivalent_size(f) == budget.crlf_equivalent_size(f) == 7


def test_near_limit_warns_but_still_passes(tmp_path: Path):
    """余量 >95%（未超限）：打印 ⚠ 余量读数但**不拦截**（rc 仍 0）。"""
    rel = "orchestration/skills/lit/references/tight.md"
    f = tmp_path / rel
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text("x" * (_GUARD.WARN_AT + 1), encoding="utf-8", newline="\n")

    proc = run_script(tmp_path, rel)

    assert proc.returncode == 0, proc.stdout
    assert "tight.md" in proc.stdout and "余量" in proc.stdout
    assert "⚠" in proc.stdout


def test_warn_threshold_has_one_source():
    """钩子与 doctor 的预警起点必须相等（同 LIMIT 一样单一真源）。"""
    assert _GUARD.WARN_AT == budget.WARN_AT
    assert _GUARD.WARN_RATIO == budget.WARN_RATIO


def test_headroom_threshold_is_512_and_shared():
    """文档族余量闸 = 每文件 ≥512B 可用余量（backlog 20261016-030200）。

    阈值同一真源：钩子与 doctor 都是 ``WARN_HEADROOM = 512``，起点 ``LIMIT - 512``。
    钉死 512 而非只钉相等——留出的是**可用余量**，改小即等于变相挤压存量。
    """
    assert _GUARD.WARN_HEADROOM == budget.WARN_HEADROOM == 512
    assert _GUARD.WARN_AT == budget.LIMIT - 512
    assert budget.WARN_AT == budget.LIMIT - 512


def test_headroom_warning_names_512_and_split_hint(tmp_path: Path):
    """余量预警正文必须点名 512B 并指向 basic.md §1 的「拆分/重设计」出路。"""
    rel = "orchestration/skills/lit/references/tight.md"
    f = tmp_path / rel
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text("x" * (_GUARD.WARN_AT + 1), encoding="utf-8", newline="\n")

    out = run_script(tmp_path, rel).stdout

    assert "< 512B" in out  # 警告行点名余量阈值
    assert "拆分或重设计" in out  # 出路指向 §1，而非放宽预算
    assert "references 分册" in out


def test_mixed_batch_reports_only_offenders(tmp_path: Path):
    ok = "orchestration/skills/lit/SKILL.md"
    bad = "orchestration/skills/lit/references/big.md"
    for rel, n in ((ok, 100), (bad, budget.LIMIT + 1)):
        f = tmp_path / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("x" * n, encoding="utf-8", newline="\n")

    proc = run_script(tmp_path, ok, bad)

    assert proc.returncode == 1
    assert bad in proc.stdout and ok not in proc.stdout


def test_staged_deletion_does_not_break_the_hook(tmp_path: Path):
    """删除/重命名时 pre-commit 可能传来不存在的路径：无内容可注入，不算违规。"""
    proc = run_script(tmp_path, "orchestration/skills/lit/references/gone.md")
    assert proc.returncode == 0, proc.stdout


def test_no_files_means_pass(tmp_path: Path):
    assert run_script(tmp_path).returncode == 0


# ===========================================================================
#  2. 配置与脚本判据一致（防「超限但根本不进钩子」的死角）
# ===========================================================================
@pytest.mark.parametrize("rel", IN_SCOPE)
def test_config_files_regex_covers_every_in_scope_shape(rel: str):
    assert re.search(hook_files_regex(), rel), rel


@pytest.mark.parametrize("rel", OUT_SCOPE)
def test_config_files_regex_leaves_out_of_scope_alone(rel: str):
    assert not re.search(hook_files_regex(), rel), rel


@pytest.mark.parametrize("rel", IN_SCOPE + OUT_SCOPE)
def test_config_and_script_agree_on_every_path(rel: str):
    """两层判据必须同进同退：正则命中 ⇔ 脚本分类命中。"""
    assert bool(re.search(hook_files_regex(), rel)) is (_tier(rel) is not None), rel


# ===========================================================================
#  3. 真源一致与真实仓库现状
# ===========================================================================
def test_limit_has_one_source():
    """钩子与 doctor 的上限必须相等，否则 doctor 报绿而钩子拦人。"""
    assert _GUARD.LIMIT == budget.LIMIT


def test_repo_tracked_docs_pass_the_hook():
    """全仓现状（任务书「移除 → 通过」的机械版）：所有在范围内的 git 跟踪文件都过。

    钩子按 **CRLF 等价字节**计量（backlog 20261017-125557）——LF 检出（worktree 里脚本
    写出的文件）与 main 检出读同一判据，本用例在两种检出下都应绿。
    """
    tracked = subprocess.run(
        ["git", "ls-files", "-z"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=str(PROJECT_ROOT),
    ).stdout.split("\0")
    in_scope = [p for p in tracked if _tier(str(p)) is not None]
    assert len(in_scope) > 20, in_scope  # 真在扫东西，不是空跑

    proc = run_script(PROJECT_ROOT, *in_scope)

    assert proc.returncode == 0, proc.stdout
