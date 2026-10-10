"""技能唯一真本 → 部署副本的同步（README §2：真本 git 跟踪、副本 gitignored 再生）。

manifest：``orchestration/skills/manifest.toml``（tomllib 解析，Python 3.13 标准库）。
部署 = 按名整树复制；哈希（sha256，逐文件路径+内容）记录于
``orchestration/state/skills-deployed.json``，``--check`` 检测源漂移与副本漂移。
红线：只触碰 manifest 声明的部署名，**绝不删除成员自建技能**。
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tomllib
from dataclasses import dataclass
from pathlib import Path

from pysci.paths import ORCH_STATE_ROOT, ORCHESTRATION_ROOT, PROJECT_ROOT

MANIFEST_PATH: Path = ORCHESTRATION_ROOT / "skills" / "manifest.toml"
DEPLOYED_PATH: Path = ORCH_STATE_ROOT / "skills-deployed.json"


@dataclass
class SkillSpec:
    """manifest 中一个技能的部署规格。"""

    name: str
    source: Path
    targets: list[Path]


def load_manifest() -> list[SkillSpec]:
    """读取 manifest.toml；不存在时返回空表（sync 会提示先建 manifest）。

    manifest 格式::

        version = 1
        [[skills]]
        name = "scientific-plotting"
        source = "orchestration/skills/scientific-plotting"   # 相对项目根
        targets = ["orchestration/pods/figure/.qoder/skills"]  # 部署父目录
    """
    if not MANIFEST_PATH.exists():
        return []
    data = tomllib.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    specs: list[SkillSpec] = []
    for entry in data.get("skills", []):
        specs.append(
            SkillSpec(
                name=entry["name"],
                source=PROJECT_ROOT / entry["source"],
                targets=[PROJECT_ROOT / t for t in entry.get("targets", [])],
            )
        )
    return specs


def tree_hash(root: Path) -> str:
    """计算目录树的内容哈希（逐文件：相对路径 + 规范化字节内容，排序后级联 sha256）。

    行尾无关（ADR 2026-10-10，backlog 20261010-003127-devops）：哈希前把 CRLF→LF。
    本机 ``core.autocrlf=true`` 且无 ``.gitattributes``，``git worktree add`` 把文本真本
    smudge 成 CRLF，而 main 工作区是混合行尾；旧实现读**原始字节**，致同一真本在
    worktree/main 算出不同哈希、合并后 ``skills-deployed.json`` 报假漂移。技能真本全为
    文本（.md/.toml，无二进制），故规范化安全。选此方案而非根 ``.gitattributes``：后者需
    仓库级 renormalize（触碰每个文本文件、与用户本地 autocrlf 交互），blast radius 大；
    本改动外科式、只影响哈希计算，worktree 内 sync 提交的哈希对 main 即正确。

    Args:
        root: 目标目录（不存在时返回 ``"MISSING"`` 标记）。

    Returns:
        十六进制 sha256 摘要。
    """
    h = hashlib.sha256()
    if not root.exists():
        return "MISSING"
    for p in sorted(root.rglob("*")):
        if p.is_file():
            h.update(str(p.relative_to(root)).replace("\\", "/").encode("utf-8"))
            h.update(p.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()


def _load_deployed() -> dict:
    if DEPLOYED_PATH.exists():
        return json.loads(DEPLOYED_PATH.read_text(encoding="utf-8"))
    return {"version": 1, "skills": {}}


def _save_deployed(data: dict) -> None:
    DEPLOYED_PATH.parent.mkdir(parents=True, exist_ok=True)
    DEPLOYED_PATH.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def sync(check_only: bool = False) -> list[str]:
    """执行同步或漂移检查。

    Args:
        check_only: True 只报告不写盘。

    Returns:
        人类可读的报告行列表（供 orch 门面直接打印）。
    """
    specs = load_manifest()
    if not specs:
        return [f"[!] manifest 不存在或为空：{MANIFEST_PATH.relative_to(PROJECT_ROOT)}"]
    deployed = _load_deployed()
    report: list[str] = []
    changed = False
    for spec in specs:
        src_hash = tree_hash(spec.source)
        rec = deployed["skills"].setdefault(
            spec.name, {"source_hash": "", "targets": {}}
        )
        if src_hash != rec.get("source_hash"):
            report.append(
                f"[*] 真本变更：{spec.name}（{rec.get('source_hash', '无记录')[:8]} → {src_hash[:8]}）"
            )
        for tgt_parent in spec.targets:
            tgt = tgt_parent / spec.name
            tgt_hash = tree_hash(tgt)
            key = str(tgt.relative_to(PROJECT_ROOT)).replace("\\", "/")
            if tgt_hash == src_hash:
                report.append(f"[=] {spec.name} → {key}（一致）")
                rec["targets"][key] = src_hash
                continue
            if check_only:
                report.append(
                    f"[!] 漂移：{spec.name} → {key}（副本 {tgt_hash[:8]} ≠ 真本 {src_hash[:8]}）"
                )
                continue
            if tgt.exists():
                shutil.rmtree(tgt)
            tgt_parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(spec.source, tgt)
            rec["targets"][key] = src_hash
            report.append(f"[+] 已部署：{spec.name} → {key}")
            changed = True
        if not check_only:
            rec["source_hash"] = src_hash
            changed = True
    if changed and not check_only:
        _save_deployed(deployed)
    return report
