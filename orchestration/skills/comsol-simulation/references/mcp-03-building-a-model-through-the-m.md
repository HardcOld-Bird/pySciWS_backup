# mcp 分册 3/3

> 含小节：Building a model through the MCP — five measured constraints
> 原 `mcp.md` 超 8KB 预算拆分（2026-10-10，backlog 20261010-refs-split-8kb）；分册索引见同目录 `mcp.md`，按需只读所需分册。

## Building a model through the MCP — five measured constraints

Every item below was measured end-to-end on 2026-10-01 by `scripts/comsol_mcp/smoke_mcp_chain.py`, and
each cost a license-holding round trip to learn. Read them before driving the MCP by hand.

1. **`study_create` must precede `mesh_create`.** The physics-controlled mesh derives its maximum element
   size from the study frequency; with no study in the model yet COMSOL refuses outright — *"the mesh is
   not used in any study"*.
2. **A Frequency step's frequency property is `plist`, not `freq`.** Upstream's own `study_create`
   docstring suggests `{"freq": "…"}` and COMSOL rejects it as an *unknown property*. The tool reports
   that in `property_errors` while still answering `success: true`, so **check that field**. Units default
   to Hz, so a bare number is enough.
3. **Domain features need an explicit `boundary_dimension`.** `physics_configure_acoustic_boundary`
   defaults to `getSDim() - 1`, i.e. *boundaries*, which is wrong for a domain-level node. A **Background
   Pressure Field** — the node the gain_ep study needs — is built in 2D with `boundary_dimension: 2` plus
   an explicit `boundary_selection`: upstream passes the type straight to `physics.create(tag, type, dim)`
   with no whitelist and echoes it back in `custom_condition_types`. Its knowledge tool
   `physics_get_acoustic_boundary_conditions` lists only 8 common boundary conditions and never mentions
   BPF, so a miss there proves nothing — and `simulation inspect node --path
   component(comp1).physics(acpr).feature(bpf1)` is still how you get the real answer (52 properties,
   allowed enums such as `PressureFieldType: PlaneWave|CylindricalWave|UserDefined`, selection counts).
4. **Upstream refuses an empty selection** (`Missing boundary selection or selection name.`) and never
   calls `selection().all()`. Worth knowing in the good sense: the silent zero-field trap in our own
   pitfalls notes — a BPF on an empty selection solves cleanly and returns `p == 0` everywhere — cannot
   happen through this path; it errors out instead.
5. **`physics_set_material` is an empty shell, so a model built from scratch cannot be meshed.** It only
   runs `material().create(tag, "Common")` + `label(name)` and never loads COMSOL's built-in library, so
   it answers `success: true` right next to the warning *"Material node has no physical properties"*, and
   no other tool can fill in `c` / `rho` afterwards. The physics-controlled mesh needs the sound speed, so
   `mesh_create` then fails with *"the material property `c` required by Pressure Acoustics 1 is not
   defined"*. **Let the MCP drive an existing `.mph`** (`model_load` → `param_set` → `study_solve` →
   `results_evaluate`) and create genuinely new models with a `build` recipe, which runs the Java API
   directly and can attach a real material — that split is exactly what the smoke test's two phases do.

> **Path C's coexistence promise is measured, not assumed.** With the MCP server running, a CLI
> `inspect tree` started its own JVM and read a model successfully (2026-10-01): the two really do share
> the box, because an *idle* server holds no license. Once the server has started COMSOL it holds the one
> license, so `mcp stop` before a CLI solve or opening the GUI.
