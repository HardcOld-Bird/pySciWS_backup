"""交付物冻结回脚本——产物↔脚本指纹对账（⑥域红线之一）。

**绑定规则**：交付物只认"版本化脚本重跑"的产物。每次交付把**脚本指纹 + 产物指纹**
冻结成账本（JSON）；事后对账（:func:`reconcile`）即可发现"手改模型/手改渲染图"——
产物漂移（ARTIFACT_DRIFT）或脚本漂移（SCRIPT_DRIFT）都意味着复现链已断，必须回脚本
重跑而不是续用手改件。交互探索（blender-mcp）成果同样须先冻结回脚本再出交付。

``[model3d]`` 标记行协议（旧域形态保留）：bpy 脚本向 stdout 打
``print(f"[model3d] <key>: <value>")``（如 ``stl:`` / ``glb:`` / ``render:`` /
``faces:`` / ``volume_mm3:``），胶水用 :func:`parse_markers` 收集产物清单与度量。

路径语义：账本内记录相对**账本所在目录**的 POSIX 相对路径（可整体搬移会话目录），
解析后逐一重算 sha256。写账本默认带 ``data/`` 防散落护栏（测试沙箱等豁免场景传
``guard=False``，原地 ADR 见模块 docstring——护栏语义沿用 :func:`pysci.paths.assert_within_data`）。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from pysci.paths import assert_within_data

#: 账本 schema 版本。
LEDGER_SCHEMA = "modeling3d.ledger/1"

#: 标记行协议前缀（旧域形态保留，勿改）。
MARKER_PREFIX = "[model3d]"

_MARKER_RE = re.compile(r"^\[model3d\]\s*(?P<key>[A-Za-z0-9_]+)\s*:\s*(?P<value>.*)$")


def parse_markers(output: str) -> dict[str, str]:
    """解析脚本 stdout 里的 ``[model3d] key: value`` 标记行。

    Args:
        output: 脚本运行的全部 stdout/stderr 文本。

    Returns:
        标记字典（同名键后出现者覆盖先出现者；值为原文字符串）。
    """
    markers: dict[str, str] = {}
    for line in output.splitlines():
        m = _MARKER_RE.match(line.strip())
        if m:
            markers[m.group("key")] = m.group("value").strip()
    return markers


def fingerprint(path: Path | str) -> str:
    """文件内容 sha256（十六进制）。"""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_record(
    script: Path | str,
    artifacts: list[Path | str],
    *,
    ledger_dir: Path | str,
    meta: dict[str, str] | None = None,
) -> dict:
    """构建冻结记录（纯读文件，不落盘）。

    Args:
        script: 生成交付物的 bpy 脚本（模型真源）。
        artifacts: 交付物列表（STL/GLB/OBJ/PNG…）。
        ledger_dir: 账本将存放的目录（路径相对其记录）。
        meta: 可选元数据（如 slug、blender 版本）。

    Returns:
        可 json 序列化的记录字典。

    Raises:
        FileNotFoundError: 脚本或产物不存在。
    """
    base = Path(ledger_dir).resolve()
    script_p = Path(script).resolve()
    if not script_p.is_file():
        raise FileNotFoundError(f"脚本不存在：{script_p}")

    def _rel(p: Path) -> str:
        return Path(os.path.relpath(p, base)).as_posix()

    record: dict = {
        "schema": LEDGER_SCHEMA,
        "created": datetime.now().astimezone().isoformat(timespec="seconds"),
        "script": {"path": _rel(script_p), "sha256": fingerprint(script_p)},
        "artifacts": [],
        "meta": dict(meta or {}),
    }
    for art in artifacts:
        art_p = Path(art).resolve()
        if not art_p.is_file():
            raise FileNotFoundError(f"交付物不存在：{art_p}")
        record["artifacts"].append(
            {
                "path": _rel(art_p),
                "sha256": fingerprint(art_p),
                "size": art_p.stat().st_size,
            }
        )
    return record


def freeze(
    script: Path | str,
    artifacts: list[Path | str],
    out: Path | str,
    *,
    meta: dict[str, str] | None = None,
    guard: bool = True,
) -> Path:
    """冻结一次交付：写账本 JSON（产物↔脚本指纹对账的基准）。

    Args:
        script: 生成交付物的 bpy 脚本。
        artifacts: 交付物列表。
        out: 账本 JSON 路径（建议 ``<会话>/ledger/<slug>.ledger.json``）。
        meta: 可选元数据。
        guard: True（默认）时账本路径必须在 ``data/`` 根内；测试沙箱/联调
            场景传 False（豁免面，勿在交付路径上使用）。

    Returns:
        账本落盘路径。

    Raises:
        ValueError: ``guard=True`` 且账本路径在 ``data/`` 根外。
        FileNotFoundError: 脚本或产物不存在。
    """
    out_p = Path(out)
    if guard:
        out_p = assert_within_data(out_p, what="交付冻结账本")
    out_p.parent.mkdir(parents=True, exist_ok=True)
    record = build_record(script, artifacts, ledger_dir=out_p.parent, meta=meta)
    out_p.write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return out_p


def load_record(path: Path | str) -> dict:
    """读取账本 JSON（原样返回字典）。"""
    return json.loads(Path(path).read_text(encoding="utf-8"))


@dataclass
class ReconcileReport:
    """对账结果：每项状态 ``ok`` / ``drift`` / ``missing``。"""

    ok: bool
    script_status: str
    artifact_status: dict[str, str] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)

    def format(self) -> str:
        """人类可读对账报告（中文）。"""
        lines = ["=== 交付冻结对账 ===", f"  script    : {self.script_status}"]
        for p, s in self.artifact_status.items():
            lines.append(f"  artifact  : {s}  {p}")
        lines.append("  RESULT    : 一致 ✅" if self.ok else "  RESULT    : 漂移 ❌")
        lines.extend(f"    - {f_}" for f_ in self.failures)
        return "\n".join(lines)


def reconcile(record: dict, *, base_dir: Path | str) -> ReconcileReport:
    """按账本重算指纹对账：脚本或产物被手改/缺失即暴露。

    防"手改模型丢失复现"：任何 ``drift``/``missing`` 都意味着该交付已不可复现，
    正确处置是回脚本重跑并重新冻结，而不是接受手改件。

    Args:
        record: :func:`load_record` / :func:`build_record` 的记录。
        base_dir: 账本所在目录（记录内相对路径的解析基准）。

    Returns:
        :class:`ReconcileReport`。
    """
    base = Path(base_dir).resolve()
    failures: list[str] = []
    script = record.get("script", {})
    script_p = base / script.get("path", "")
    if not script_p.is_file():
        script_status = "missing"
        failures.append(f"脚本缺失：{script_p}——复现链断裂，回版本库/会话找回脚本")
    elif fingerprint(script_p) != script.get("sha256"):
        script_status = "drift"
        failures.append(f"脚本已漂移：{script_p}——冻结后被改过，需确认改动已入脚本并重新冻结")
    else:
        script_status = "ok"

    artifact_status: dict[str, str] = {}
    for art in record.get("artifacts", []):
        rel = art.get("path", "")
        art_p = base / rel
        if not art_p.is_file():
            artifact_status[rel] = "missing"
            failures.append(f"交付物缺失：{art_p}")
        elif fingerprint(art_p) != art.get("sha256"):
            artifact_status[rel] = "drift"
            failures.append(
                f"交付物已手改：{art_p}——与冻结指纹不符，禁止交付；回脚本重跑复现"
            )
        else:
            artifact_status[rel] = "ok"
    return ReconcileReport(
        ok=not failures,
        script_status=script_status,
        artifact_status=artifact_status,
        failures=failures,
    )
