# Module API reference — what to import in a computation script

The imports and the per-module function inventory for
`pysci.skills.theoretical_computation.tools`. `SKILL.md` keeps only the rule that a
computation script imports from this package; look here for the exact symbol names when
filling in a scaffolded `<slug>.py`.

When writing computation scripts, import from the tools package:

```python
from pysci.skills.theoretical_computation.tools import (
    cas,
    numerical,
    eigen,
    topology,
    visualize,
)
from pysci.skills.theoretical_computation.tools.session import (
    ensure_session,
    save_results,
    save_plot,
)
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
