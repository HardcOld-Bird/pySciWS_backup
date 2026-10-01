"""End-to-end smoke test for the community COMSOL MCP over path C.

Covers exactly the two items ``scripts/comsol_mcp/UPSTREAM.lock.json`` lists under
``NOT_yet_verified``:

1. Can a **Background Pressure Field** -- the node the gain_ep study needs -- be created through the
   generic ``physics_configure_acoustic_boundary``? Its knowledge tool
   (``physics_get_acoustic_boundary_conditions``) lists only 8 common boundary conditions and never
   mentions BPF, but the tool promises "unknown types are passed through", and upstream
   ``_create_boundary_features`` really does hand ``condition_type`` straight to
   ``physics.create(tag, type, dim)`` with no whitelist.
2. Does the ``study_solve`` -> ``results_evaluate`` chain actually run?

Two non-obvious requirements, both read out of the upstream source *before* running (guessing them
costs one license-holding round trip each), plus two constraints that only a real run reveals:

* BPF is a **domain** feature, while ``_get_boundary_dimension`` defaults to ``getSDim() - 1``, i.e.
  *boundaries*. So ``boundary_dimension=2`` has to be passed explicitly in a 2D model.
* ``_create_boundary_features`` **refuses** an empty selection ("Missing boundary selection or
  selection name") and never calls ``selection().all()``. That is worth knowing in the good sense:
  the silent zero-field trap recorded in our own notes -- BPF created with an empty selection solves
  cleanly and returns ``p == 0`` everywhere -- cannot happen through this path; it errors out.
* ``study_create`` must precede ``mesh_create``: the physics-controlled mesh derives its maximum
  element size from the study frequency, and with no study yet COMSOL refuses outright.
* A material is required before ``mesh_create`` too, since that same sizing needs the sound speed
  ``c`` -- and this is where the from-scratch chain runs out of road (see ``BUILD_STEPS``).

The run has two phases, because no single model can serve both purposes:

* **A -- build from scratch** as far as the Background Pressure Field. That is the differentiator
  check.
* **B -- solve and evaluate** a real research model that already carries materials, mesh and study.

The verdict deliberately does not rest on ``success: true`` alone. Phase B's model is driven by a
background field with ``pamp = 1`` and has no other source, so a non-zero ``acpr.p_t`` is the
evidence that the solve really produced a field rather than merely reporting that it ran.

**This starts COMSOL and takes the single license.** Check ``uv run pysci-simulation license``
first. ``comsol_disconnect`` at the end clears the models but **cannot** release the license:
JPype never unloads a started JVM, so ``jvm.dll`` stays mapped into the server process. Free it with
``uv run pysci-simulation mcp stop`` before using the CLI or the GUI.

Run under the *community* venv's python (it owns ``mcp`` / ``httpx``)::

    <REPO>\\.venv\\Scripts\\python.exe scripts\\comsol_mcp\\smoke_mcp_chain.py [port] [model.mph]

Exit codes: 0 = both phases passed; 1 = phase A failed (the BPF could not be built);
2 = phase B failed, or it ran but the field came out identically zero.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import re
import sys
from pathlib import Path

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
URL = f"http://127.0.0.1:{PORT}/sse"
MODEL = "smoke_bpf"
DOMAIN = 0.1  # m; with c0 = 343 m/s and f = 3430 Hz the wavelength equals the domain size
FREQ = "3430"

#: Phase B loads a real model instead of building one; override with argv[2]. The default is the
#: smallest and most recent of the gain_ep set -- a 2D half-circle beam-scattering control case that
#: already carries materials, mesh, a Frequency study, a BPF and a far-field PolarGroup.
_DEFAULT_MPH = Path("data/research/1_gain_ep/simulation/6 半圆波束散射对照.mph")
REAL_MPH = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parents[2] / _DEFAULT_MPH

#: The step under test. Everything else exists only to give it a solvable model.
BPF_ARGS = {
    "physics_name": "acpr",
    "boundary_condition": "BackgroundPressureField",
    "boundary_dimension": 2,  # domain feature; upstream would otherwise pick getSDim()-1
    "boundary_selection": [1],
    "properties": {"PressureFieldType": "PlaneWave", "pamp": "1"},
}

#: Phase A -- build from scratch as far as the node the gain_ep study actually needs.
#:
#: It stops before ``mesh_create`` on purpose. Two measured facts make meshing from scratch
#: impossible with this MCP alone:
#:
#: * ``physics_set_material`` creates an **empty** ``Common`` material node. It answers
#:   ``success: true`` yet warns "Material node has no physical properties": upstream only runs
#:   ``material().create(tag, "Common")`` + ``label()`` and never loads COMSOL's built-in library,
#:   and no other tool can fill in ``c`` / ``rho`` afterwards.
#: * The physics-controlled mesh sizes its elements from the wavelength, so without ``c`` even
#:   ``mesh_create`` fails -- "the material property 'c' required by Pressure Acoustics 1 is not
#:   defined" (measured 2026-10-01).
BUILD_STEPS: list[tuple[str, dict]] = [
    ("comsol_start", {"cores": 2}),
    ("model_create", {"name": MODEL}),
    # Creates the geometry sequence too, and that is what fixes the space dimension at 2D.
    ("model_create_component", {"component_name": "comp1", "space_dimension": 2}),
    ("geometry_add_rectangle", {"component_name": "comp1", "size": [DOMAIN, DOMAIN]}),
    ("geometry_build", {"component_name": "comp1"}),
    ("physics_add_pressure_acoustics", {"component_name": "comp1", "physics_tag": "acpr"}),
    ("physics_set_material", {"physics_name": "acpr", "material_name": "Air"}),
    ("physics_configure_acoustic_boundary", BPF_ARGS),
    # ``study_create`` must precede ``mesh_create``: the physics-controlled mesh derives its maximum
    # element size from the study frequency, and with no study in the model yet COMSOL refuses
    # outright -- "the mesh is not used in any study" (measured 2026-10-01).
    #
    # The frequency property is ``plist``, NOT ``freq`` -- the tool's own docstring suggests
    # ``{"freq": "..."}`` for Frequency studies and COMSOL rejects it with "unknown property"
    # (measured 2026-10-01). The unit defaults to Hz, so a bare number is enough.
    ("study_create", {"study_type": "Frequency", "step_properties": {"plist": FREQ}}),
]

#: Phase B -- the solve/evaluate half, on a model whose materials, mesh and study are already real.
SOLVE_STEPS: list[tuple[str, dict]] = [
    ("model_load", {"file_path": str(REAL_MPH)}),
    ("study_solve", {"wait": True, "timeout": 900}),
    ("results_evaluate", {"expression": "acpr.p_t"}),
]


def _community_import(name: str):
    """Import a module that only exists in the **community** venv.

    Same reasoning as ``probe_mcp_http.py``: this file lives in the project tree and is type-checked
    against the project interpreter, where ``mcp`` / ``httpx`` are absent. Going through
    ``importlib`` keeps the IDE quiet and turns ModuleNotFoundError into an actionable message.
    """
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"cannot import {name!r}: {exc}\n"
            "This smoke test must run under the COMMUNITY venv's python (it owns mcp/httpx):\n"
            f"  <REPO_DIR>\\.venv\\Scripts\\python.exe scripts\\comsol_mcp\\smoke_mcp_chain.py {PORT}"
        ) from exc


def _flat(payload) -> str:
    return " ".join(json.dumps(payload, ensure_ascii=False).split())[:300]


def _peak(payload) -> float:
    """Largest magnitude among the numbers in an ``results_evaluate`` payload.

    Deliberately crude: the payload shape depends on the expression and the dataset, and the only
    question here is "is the field identically zero or not". ``true``/``false`` are not numbers in
    JSON, so they cannot leak in.
    """
    found = [float(x) for x in re.findall(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?", json.dumps(payload))]
    return max((abs(v) for v in found), default=0.0)


async def _call(session, name: str, args: dict) -> tuple[bool, dict]:  # noqa: ANN001
    """Call one tool, print a one-line verdict, and hand back the parsed payload."""
    result = await session.call_tool(name, args)
    raw = result.content[0].text if result.content else ""
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        payload = {"success": False, "error": raw[:300]}
    # failed_count has to be checked on its own: the boundary tools report a bad selection through
    # it while still answering success=false only at the top level.
    ok = payload.get("success", True) is not False and not payload.get("failed_count")
    print(f"[{'PASS' if ok else 'FAIL'}] {name:36} {_flat(payload)}")
    return ok, payload


async def _chain(session, steps: list[tuple[str, dict]]) -> dict[str, dict]:  # noqa: ANN001
    """Run steps in order, stopping at the first failure. Returns ``{tool_name: payload}``."""
    got: dict[str, dict] = {}
    for name, args in steps:
        ok, payload = await _call(session, name, args)
        got[name] = payload
        if not ok:
            break
    return got


async def run() -> int:
    mcp = _community_import("mcp")
    sse_client = _community_import("mcp.client.sse").sse_client
    print(f"TARGET        : {URL}")
    print(f"PHASE B MODEL : {REAL_MPH}")
    # sse_read_timeout has to outlive the solve: the reply arrives as one event when it finishes.
    async with sse_client(URL, timeout=30, sse_read_timeout=1800) as (read, write):
        async with mcp.ClientSession(read, write) as session:
            info = await session.initialize()
            print(f"INIT_OK       : server={info.serverInfo.name!r} version={info.serverInfo.version!r}")

            print("--- Phase A: build from scratch up to the Background Pressure Field ---")
            built = await _chain(session, BUILD_STEPS)
            bpf = built.get("physics_configure_acoustic_boundary", {})
            # Upstream echoes unknown types back; their presence proves there is no whitelist.
            print(f"       custom_condition_types = {bpf.get('custom_condition_types')}")
            a_ok = len(built) == len(BUILD_STEPS) and "BackgroundPressureField" in (
                bpf.get("custom_condition_types") or []
            )
            verdict = "BPF built through the generic boundary tool" if a_ok else "BPF NOT built"
            print(f"PHASE A       : {'PASS' if a_ok else 'FAIL'} -- {verdict}")
            if not a_ok:
                await session.call_tool("comsol_disconnect", {})
                return 1

            print("--- Phase B: solve + evaluate on a real research model ---")
            solved = await _chain(session, SOLVE_STEPS)
            peak = _peak(solved.get("results_evaluate", {}))
            print(f"PEAK |acpr.p_t| : {peak:.6g}")
            b_ok = len(solved) == len(SOLVE_STEPS) and peak > 0.0
            verdict = "solve + evaluate produced a non-zero field" if b_ok else "no usable field"
            print(f"PHASE B       : {'PASS' if b_ok else 'FAIL'} -- {verdict}")
            await session.call_tool("comsol_disconnect", {})

    print("NOTE          : jvm.dll stays mapped -- run `pysci-simulation mcp stop` to free the license")
    return 0 if b_ok else 2


if __name__ == "__main__":
    sys.exit(asyncio.run(run()))
