"""可选 extra（``[imaging]``）的**存活策略**必须在工具面点明（backlog 20261015-203158）。

现场：`imagine doctor` 只写 `(not installed, optional — uv pip install -e ".[imaging]")`，
不说它会消失；按文档装上后，下一次 `pysci-dev venv-sync`（uv sync 同步到默认依赖集、
prune lock 之外的包）**静默**移除 → 使用者反复经历「装了又没了」。

策略（本单定夺，属文档面而非改依赖）：extra = **按需临时安装**，要长期可用须纳入
`pyproject.toml` 默认依赖。故下钉两处工具面 + 一处真本册。
"""

from __future__ import annotations

from pysci.paths import ORCHESTRATION_ROOT
from pysci.skills.ai_drawing.tools import imagine, imaging

#: 无任何可选依赖时的 installed 表（deterministic，不看本机实际装没装）。
_MISSING = {"scikit-image": None, "opencv": None}
_PRESENT = {"scikit-image": "0.25.0", "opencv": "4.11.0"}

_VENV_SYNC_DOC = (
    ORCHESTRATION_ROOT / "skills" / "devops" / "references" / "uv-venv-sync.md"
)


def test_doctor_warns_extra_is_not_durable(capsys, monkeypatch):
    """可选依赖缺失时，doctor 必须点明「装上会被下一次 venv-sync 清除」。"""
    monkeypatch.setattr(imagine._img, "available", lambda: dict(_MISSING))

    assert imagine.main(["doctor"]) == 0
    out = capsys.readouterr().out

    assert "可选 extra" in out
    assert "venv-sync" in out and "清除" in out
    assert "[imaging]" in out  # 安装命令仍在（可执行出路）


def test_doctor_quiet_when_optional_present(capsys, monkeypatch):
    """可选依赖齐备时不打该告警（不制造噪声）。"""
    monkeypatch.setattr(imagine._img, "available", lambda: dict(_PRESENT))

    assert imagine.main(["doctor"]) == 0
    assert "可选 extra" not in capsys.readouterr().out


def test_install_hint_carries_durability_caveat():
    """运行期缺失报错也带该告警——且保留既有安装命令（既有断言依赖它）。"""
    hint = imaging.INSTALL_HINT
    assert 'uv pip install -e ".[imaging]"' in hint
    assert "不在默认依赖集" in hint and "venv-sync" in hint


def test_venv_sync_reference_documents_extra_policy():
    """真本册给出权威策略：uv sync 会移除 lock 之外的包（含 extra）。"""
    md = _VENV_SYNC_DOC.read_text(encoding="utf-8")
    assert "存活策略" in md
    assert "extra" in md and "被移除" in md
    assert "纳入" in md and "默认依赖" in md
