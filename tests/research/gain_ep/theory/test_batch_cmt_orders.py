"""batch_cmt_orders：参考端口栅格 + 正号提取 + 复数 JSON（backlog 20261021-195032）。"""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mph", reason="batch_cmt_orders 顶层 import mph")

from pysci.paths import PROJECT_ROOT, research_asset_dir  # noqa: E402
from pysci.research.gain_ep.theory.cmt_reflection_s_matrix import (  # noqa: E402
    MetagratingParams,
    solve_s_matrix,
)

D = 0.15
K = 2.0 * np.pi * 3430.0 / 343.0


def _gain_ep_root() -> Path:
    root = research_asset_dir("gain_ep")
    if (root / "simulation" / "cmt_batch.json").exists():
        return root
    common = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "--git-common-dir"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return Path(common).resolve().parent / "data" / "research" / "1_gain_ep"


def _script():
    path = (
        PROJECT_ROOT
        / "scripts"
        / "research"
        / "gain_ep"
        / "simulation"
        / "batch_cmt_orders.py"
    )
    spec = importlib.util.spec_from_file_location("batch_cmt_orders", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_reference_ports_match_cmt():
    mod = _script()
    ports = mod.reference_ports(3430.0, 30.0, D)
    ref = solve_s_matrix(
        MetagratingParams(f0=3430.0, c0=343.0, theta_i=30.0, period=D)
    ).ports
    assert [p.order for p in ports] == [p.order for p in ref] == [0, -1, -2]
    assert np.allclose([p.kx for p in ports], [p.kx for p in ref])


def test_positive_sign_extracts_mode():
    """扫描场 e^{-ikx x} 用 e^{+ikx x} 提取 → 归一化 1（旧负号会漏阶）。"""
    mod = _script()
    port = mod.reference_ports(3430.0, 30.0, D)[0]
    x = np.linspace(0.0, 2.0 * D, 4001)
    ps = np.exp(-1j * port.kx * x)
    coeffs = mod.extract_coefficients(x, ps, [port], D, k=K)
    assert coeffs[str(port.order)]["abs"] == pytest.approx(1.0, abs=1e-6)
    assert {"re", "im", "abs", "phase_rad", "kx_over_k"} <= set(coeffs[str(port.order)])
    old = np.trapezoid(ps * np.exp(-1j * port.kx * x), x) / (2.0 * D)
    assert abs(old) < 0.01, "负号提取应漏掉该传播阶"


def test_json_3port_L_matches_cmt_and_has_complex():
    doc = json.loads(
        (_gain_ep_root() / "simulation" / "cmt_batch.json").read_text(encoding="utf-8")
    )
    entry = next(e for e in doc if e["theta_deg"] == pytest.approx(30.0))
    left = entry["sides"]["L"]
    # t5 CMT：order -2/-1/0 = 0.88/0.63/0.22（t4 旧值 ~1e-3）
    assert left["-2"]["abs"] == pytest.approx(0.88, rel=0.02)
    assert left["-1"]["abs"] == pytest.approx(0.63, rel=0.02)
    assert left["0"]["abs"] == pytest.approx(0.22, rel=0.02)
    for coeff in left.values():
        assert {"re", "im", "abs", "phase_rad"} <= set(coeff)


def test_json_45deg_labels_align_with_cmt():
    doc = json.loads(
        (_gain_ep_root() / "simulation" / "cmt_batch.json").read_text(encoding="utf-8")
    )
    entry = next(e for e in doc if e["theta_deg"] == pytest.approx(45.0))
    left = entry["sides"]["L"]
    # CMT anchor +45 列（ports [0,-1]）：[0.0226, 0.9997]——旧脚本标签翻转
    assert left["0"]["abs"] == pytest.approx(0.0226, rel=0.05)
    assert left["-1"]["abs"] == pytest.approx(0.9997, rel=0.01)
