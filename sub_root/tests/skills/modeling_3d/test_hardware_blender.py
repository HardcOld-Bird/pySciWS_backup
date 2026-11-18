"""Blender 真实冒烟（-m hardware）：立方体→Cycles CPU 低采样渲染→GLB 导出→验收门→冻结。

跑前纪律：确认**单实例**（内存 ~5GB 禁并行）；Blender 未 vendor 时 skip（并行代理
可能仍在落盘）。产物落 ``tests/artifacts/modeling_3d/``（不入库）。
用 ``tests.cmd hardware`` 单独跑；``tests.cmd all`` 默认不含（addopts -m "not hardware"）。
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from pysci.paths import TESTS_ROOT
from pysci.skills.modeling_3d import acceptance, doctor, ledger, scaffold

pytestmark = pytest.mark.hardware

OUT_ROOT = TESTS_ROOT / "artifacts" / "modeling_3d"
RUN_TIMEOUT_S = 600


def _run_blender(exe: Path, script: Path) -> str:
    proc = subprocess.run(
        [str(exe), "-b", "-P", str(script)],
        capture_output=True,
        text=True,
        timeout=RUN_TIMEOUT_S,
    )
    assert proc.returncode == 0, (
        f"blender -b -P 失败（rc={proc.returncode}）\n"
        f"stdout:\n{proc.stdout[-2000:]}\nstderr:\n{proc.stderr[-2000:]}"
    )
    return proc.stdout + "\n" + proc.stderr


def test_cube_render_export_gate_freeze():
    # 0. 单实例红线：已有 Blender 在跑就跳过（禁并行）
    pids = doctor.blender_processes()
    if pids:
        pytest.skip(f"已有 Blender 实例（PID {pids}）：内存预算禁并行，稍后重跑")
    exe = doctor.find_blender()
    if exe is None:
        pytest.skip(f"Blender 未 vendor（{doctor.BLENDER_VENDOR_DIR}）或未设 {doctor.BLENDER_ENV}")
    version = doctor.blender_version(exe)
    assert version, "blender --version 无输出，vendor 可能损坏"

    session = OUT_ROOT / "cube_hw"
    shutil.rmtree(session, ignore_errors=True)
    (session / "stl").mkdir(parents=True)
    (session / "renders").mkdir(parents=True)
    (session / "scripts").mkdir(parents=True)

    # 1. 建模：part 脚手架（20mm 立方体）→ Blender 无头跑 → STL/OBJ/GLB
    part_script = scaffold.write_scaffold(
        scaffold.render_scaffold(
            "part", "cube20", session, size_mm=(20.0, 20.0, 20.0)
        ),
        session / "scripts" / "cube20_part.py",
        guard=False,
    )
    out = _run_blender(exe, part_script)
    markers = ledger.parse_markers(out)
    assert "stl" in markers and "glb" in markers and "faces" in markers, (
        f"标记行协议缺失：{markers}\n{out[-1500:]}"
    )
    stl = session / "stl" / "cube20.stl"
    glb = session / "renders" / "cube20.glb"
    assert stl.is_file() and glb.is_file() and glb.stat().st_size > 0

    # 2. 验收门（打印判据）：流形/法线/尺寸/体积
    metrics = acceptance.stl_metrics(stl)
    gate = acceptance.judge_print(
        metrics,
        expect_bbox_mm=(20.0, 20.0, 20.0),
        expect_volume_mm3=8000.0,
    )
    assert gate.ok, gate.format()
    assert metrics.faces == 12 and metrics.watertight

    # 3. 渲染：scene 脚手架导入 STL → Cycles CPU 低采样单帧 PNG
    scene_script = scaffold.write_scaffold(
        scaffold.render_scaffold("scene", "cube20", session, mesh=stl),
        session / "scripts" / "cube20_scene.py",
        guard=False,
    )
    _run_blender(exe, scene_script)
    scene_text = scene_script.read_text(encoding="utf-8")
    assert 'scene.cycles.device = "CPU"' in scene_text  # 实跑配置即脚本配置
    png = session / "renders" / "cube20.png"
    assert png.is_file() and png.stat().st_size > 0

    # 4. 渲染判据 + 交付冻结（产物↔脚本指纹）→ 对账一致
    render_gate = acceptance.judge_render(
        acceptance.metrics_from_markers(ledger.parse_markers(out))
    )
    assert render_gate.ok, render_gate.format()
    ledger_out = ledger.freeze(
        part_script,
        [stl, session / "stl" / "cube20.obj", glb],
        session / "ledger" / "cube20.ledger.json",
        meta={"blender": version or ""},
        guard=False,
    )
    report = ledger.reconcile(ledger.load_record(ledger_out), base_dir=ledger_out.parent)
    assert report.ok, report.format()

    # 5. 防手改红线：篡改 STL 后对账必须暴露 drift
    with stl.open("ab") as f:
        f.write(b"\0")
    tampered = ledger.reconcile(ledger.load_record(ledger_out), base_dir=ledger_out.parent)
    assert not tampered.ok
    assert any(
        k.endswith("cube20.stl") and v == "drift"
        for k, v in tampered.artifact_status.items()
    )
