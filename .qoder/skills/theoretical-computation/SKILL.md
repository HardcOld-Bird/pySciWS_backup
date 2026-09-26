---
name: theoretical-computation
description: Perform symbolic and numerical theoretical computations (CAS via sympy, eigenvalue analysis, parameter space exploration, topological feature detection) with visual feedback. Drives a unified `theory` CLI that scaffolds computation sessions, executes scripts with output capture, and exports exploration plots as PNG for the agent to Read and iterate. Use when the user asks to derive/verify a theoretical model, compute eigenvalues/dispersion, explore a parameter space, find EP/BIC/singularities, or visualize a mathematical relationship.
---

# Theoretical Computation

One CLI drives the whole theory workflow: **scaffold → compute → visualize → Read → iterate**.

The backend lives in `src/pysci/skills/theoretical_computation/` (cas / numerical / eigen /
topology / visualize / session + a `theory` facade). **Treat it as a black box** and drive
everything through the `theory` CLI below. Only open the code when maintaining it.

The key design split: **this skill owns the reusable computation toolkit** (CAS engine,
numerical pipelines, eigenvalue analysis, topology exploration, lightweight plotting);
**each research's specific theoretical models live as computation scripts** under
`src/pysci/research/<name>/theory/`, with results exported to `data/research/<n>_<name>/theory/<slug>/`.

## Invocation

Run from the **project root**. The skill installs a console script `pysci-theory`:

```
uv run pysci-theory <command> [options]
```

Below, `theory …` is shorthand for `uv run pysci-theory …`. (Fallback if the script isn't
installed: `uv run --no-sync python -m pysci.skills.theoretical_computation.tools.theory …`.)

> **PowerShell rule (critical):** wrap multi-word arguments in **single quotes**; use `;`
> (never `&&`) to chain commands.

## Capability boundary (read this first)

**Available now:**
- CAS engine: symbolic matrix construction, simplification, series expansion, LaTeX output,
  equation solving, discriminant/EP-condition computation.
- Numerical pipeline: N-dimensional parameter space definition, lambdify (sympy→numpy),
  grid evaluation, adaptive 1D sampling.
- Eigenvalue analysis: symbolic/numeric eigensystems, EP detection (eigenvalue + eigenvector
  coalescence), branch tracking (avoid crossing artifacts), complex-plane trajectories.
- Topology exploration: complex zero-set extraction (isosurface intersection), scalar
  isosurfaces, singularity detection, critical point classification (Morse index),
  winding number computation.
- Visualization: lightweight matplotlib 2D plots (real/complex/band), pyvista 3D offscreen
  rendering (surfaces/zero-sets/band structures), PNG export for agent visual feedback.
- Session management: scaffold computation scripts, persist results (.npz), save exploration
  plots, maintain computation logs (notes.md).

**Not yet (v2+):** persistent homology / topological data analysis; symbolic regression;
interactive plotly HTML export; automatic phase-diagram generation; COMSOL cross-validation loop.

## Commands at a glance

| Command | Use when | Key output |
|---|---|---|
| `doctor` | Session start, or anything seems broken | dependency versions + backend status |
| `new <research> <slug>` | Starting a new computation | scaffolded script + data session dir |
| `run <script> [--session <name>]` | Executing a computation | stdout + results + PNG plots |
| `plot <result_file>` | Quick-look at saved data | PNG for agent Read |
| `list <research>` | Seeing existing sessions | session inventory |

Run `theory <command> -h` for the full option list.

## Quick start: zero → visual result

```
- [ ] 1. theory doctor                                       # confirm backends
- [ ] 2. theory new gain_ep ep_band_analysis                 # scaffold
- [ ] 3. Edit src/pysci/research/gain_ep/theory/ep_band_analysis.py  # fill in physics
- [ ] 4. theory run 'src/pysci/research/gain_ep/theory/ep_band_analysis.py' --session ep_band_analysis --research gain_ep
- [ ] 5. Read the printed plots/*.png                        # VISUAL CHECK — iterate on step 3-4
- [ ] 6. When satisfied, hand off data to scientific_plotting for publication figures
```

**Always Read the exported PNG before declaring done** — a computation can run without error
yet produce physically meaningless results. The plot is your eyes.

## The computation session

Each session has two halves:

**Code** (in `src/pysci/research/<name>/theory/<slug>.py`):
```python
def main(session_dir: Path | None = None) -> None:
    # 1. Define symbolic model (sympy)
    # 2. Numerical evaluation (ParamSpace + lambdify + grid)
    # 3. Analysis (eigenvalues / zero-sets / topology)
    # 4. Visualize + export (PNG to session_dir/plots/)
```

**Data** (in `data/research/<n>_<name>/theory/<slug>/`):
```
<slug>/
├── results/       # .npz / .csv numerical outputs
├── plots/         # exploration PNGs (agent visual feedback)
└── notes.md       # computation log (timestamped entries)
```

## Module API quick reference

When writing computation scripts, import from the tools package:

```python
from pysci.skills.theoretical_computation.tools import cas, numerical, eigen, topology, visualize
from pysci.skills.theoretical_computation.tools.session import ensure_session, save_results, save_plot
from pysci.skills.theoretical_computation.tools.config import settings
```

| Module | Key functions |
|---|---|
| `cas` | `symbolic_matrix`, `simplify_expr`, `series_expand`, `expr_to_latex`, `solve_system`, `discriminant_2x2` |
| `numerical` | `ParamAxis`, `ParamSpace`, `lambdify_expr`, `evaluate_on_grid`, `make_meshgrid`, `adaptive_sample_1d` |
| `eigen` | `eigensystem_symbolic`, `eigensystem_numeric`, `detect_ep`, `track_branches`, `ep_condition_symbolic` |
| `topology` | `find_zero_set`, `find_isosurface`, `find_singularities`, `find_critical_points`, `compute_winding_number` |
| `visualize` | `quick_plot_2d`, `quick_plot_complex`, `quick_plot_complex_plane`, `quick_plot_3d_surface`, `quick_plot_zero_set_3d`, `export_exploration` |
| `session` | `ensure_session`, `save_results`, `save_plot`, `write_log`, `list_sessions` |

## Visual closed loop

All `visualize.*` functions produce matplotlib Figures or pyvista Plotters in **offscreen/Agg
mode**. Pass `out_path=` or call `export_exploration(fig, path)` to save a PNG, then **Read it**
to see the result. This is the same render→Read pattern as the comsol/figures skills.

For 3D pyvista renders, the CLI forces `pv.OFF_SCREEN = True` — no window pops up.

## Output locations

| Path | Contents |
|---|---|
| `src/pysci/research/<name>/theory/<slug>.py` | computation script (code) |
| `data/research/<n>_<name>/theory/<slug>/results/` | numerical outputs (.npz/.csv) |
| `data/research/<n>_<name>/theory/<slug>/plots/` | exploration PNGs |
| `data/research/<n>_<name>/theory/<slug>/notes.md` | agent computation log |
| `data/skills/theoretical_computation/templates/` | scaffold templates |
| `data/skills/theoretical_computation/recipes/` | reusable computation recipes |
| `data/skills/theoretical_computation/cache/` | transient caches (git-ignored) |

## When something breaks

1. Run `theory doctor` — reports sympy/numpy/scipy/matplotlib/pyvista versions and backend.
2. If pyvista 3D render fails: check `THEORY_PYVISTA_OFF_SCREEN=1` in .env; some GPU drivers
   block offscreen rendering — fall back to matplotlib 2D projections.
3. If lambdify produces NaN: the expression likely has branch-cut issues with complex sqrt;
   use `+0j` casting or manual complex handling (see recipes).
4. If a computation script hangs: sympy `simplify` on large expressions can be very slow;
   try `strategy="expand"` or skip simplification.

## Relationship to other skills

- **scientific_plotting**: theory produces exploration PNGs; when a result is ready for
  publication, create a figure pipeline via `pysci-figures new` and load the .npz data.
- **comsol_simulation**: theory predicts (e.g., EP location); COMSOL validates. Cross-check
  by comparing exported data arrays.
- **literature_research**: theory reproduces paper models. Use `pysci-research` to fetch
  the source paper, then scaffold a computation to verify equations.
