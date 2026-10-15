"""pod 的 ``.qoder/skills`` 面：部署产物语义 + 「不得按目录存在性断言」的机械闸。

背景（backlog id=20261015-185941-devops，原建议 20261015-185941-devops）：曾出现
**无领域技能 pod**（twin / reviewer）也带一个**空的** `.qoder/skills/`，令「目录不存在」
型断言在全套跑时误红。写入者**定位结论：不在本仓可复现面**——
① 全树 grep：`mkdir` 该路径的只有 `pysci-dev sync`（且只按 manifest targets 落）；
② 全新 worktree 里 13 个 pod **一个都没有**该目录（真本是部署产物，不随检出/合并搬运）；
③ 全套 3467 例在全新 worktree 跑完（0.2s 轮询探针 + `skills` 占位文件绊线两种手法）
   **零目录被创建**；main 上恰好只有 manifest 的 11 个 pod 有（sync 落的）。
故「改为不写 twin/reviewer」**没有可改的代码站点**；按组长取舍采**低成本那一支**：
把该行为**文档化**（`references/pod-scaffold.md` §3）+ 用例面**禁止「不存在」型断言**
（本文件的闸 + 把现存那处改成「有目录则不得有内容」）。
"""

from __future__ import annotations

import re
from pathlib import Path

from pysci.paths import PODS_ROOT, PROJECT_ROOT
from pysci.skills.orchestration.tools import sync

#: 本闸只扫用例面（判据是「测试怎么写断言」，不是产品代码怎么写）。
_TESTS_ROOT = PROJECT_ROOT / "tests"


def _manifest_skill_dirs() -> set[Path]:
    """manifest 声明的部署父目录（``<pod>/.qoder/skills``）集合，路径已 resolve。"""
    out: set[Path] = set()
    for spec in sync.load_manifest():
        for target in spec.targets:
            out.add(target.resolve())
    return out


def test_pods_outside_manifest_deploy_no_skill_content():
    """manifest 之外的 pod 不得**被部署**技能——判据是「无内容」，不是「无目录」。

    空目录可能由外部/平台初始化留下（本仓代码路径不写它，见模块 docstring 的三条证据），
    故按目录存在性断言会把「部署漂移」与「外部残留」混为一谈，并在全套跑时误红。
    """
    declared = _manifest_skill_dirs()
    offenders: list[str] = []
    for pod in sorted(p for p in PODS_ROOT.iterdir() if p.is_dir()):
        skills = pod / ".qoder" / "skills"
        if not skills.is_dir():
            continue
        if skills.resolve() in declared:
            continue
        deployed = sorted(p.name for p in skills.iterdir())
        if deployed:
            offenders.append(f"{pod.name}: {deployed}")
    assert offenders == [], (
        f"下列 pod 未在 manifest 里声明却挂了技能内容（部署漂移或误落）：{offenders}"
    )


def test_no_pod_skills_dir_is_occupied_but_undeclared():
    """逆命题同样成立：挂了内容的 skills 目录必须在 manifest 里被声明（防「悄悄部署」）。"""
    declared = _manifest_skill_dirs()
    occupied = {
        (pod / ".qoder" / "skills").resolve()
        for pod in PODS_ROOT.iterdir()
        if (pod / ".qoder" / "skills").is_dir()
        and any((pod / ".qoder" / "skills").iterdir())
    }
    assert occupied <= declared, (
        f"未声明的技能部署面：{sorted(map(str, occupied - declared))}"
    )


# ---------------------------------------------------------------------------
# 机械闸：禁止「不存在」型断言锁 pod 技能面（写死一条纪律，别靠人记）
# ---------------------------------------------------------------------------
#: 脆弱形态：把「部署面此刻读数（目录不在）」当「出厂默认（永不被建）」钉死。
_NEGATIVE_EXISTS = (
    re.compile(r"\.exists\(\)\s+is\s+False"),
    re.compile(r"assert\s+not\s+[\w\.\(\)\[\]\"'/\\ ]*\.exists\(\)"),
)


def test_no_negative_existence_assertion_on_pod_skills():
    """用例里**不得**出现「pod skills 目录不存在」型断言（改断「有目录则无内容」）。

    为什么禁：该目录是部署产物、且可能被外部初始化留下空目录——存在性不是部署判据，
    按它断言会在别的树/别的时刻误红（本单的起因）。判定式见 ``_NEGATIVE_EXISTS``。
    """
    offenders: list[str] = []
    for path in sorted(_TESTS_ROOT.rglob("*.py")):
        if "__pycache__" in path.parts or path.name == Path(__file__).name:
            continue  # 自身除外：本文件的正则字面量天然含这些片段
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if "skills" not in line:
                continue
            if any(rx.search(line) for rx in _NEGATIVE_EXISTS):
                where = path.relative_to(PROJECT_ROOT).as_posix()
                offenders.append(f"{where}:{n}: {line.strip()}")
    assert offenders == [], (
        "禁止对 pod 的 `.qoder/skills` 做「不存在」型断言（它是部署产物，空目录可能是外部"
        "残留）——请改断「若目录在，则其中不得有部署内容」：\n  "
        + "\n  ".join(offenders)
    )
