# CI test models

These STL files are for CI testing only. They let the regression suite run on a
clean checkout (GitLab CI, GitHub Actions). Do not put customer, production or
working models here.

This folder is the only place CAD files may be committed. `.gitignore` ignores
every other `*.stl`, `*.step` and `*.stp`, and `scripts/ci/check.sh models`
fails the pipeline if one is tracked anywhere else. Only `.stl` is allowed here.

Tests load them through `tests/helpers/ci_models.py` (`ci_model("block_2inch.stl")`).

| File | Geometry (inches) | Used by |
|---|---|---|
| `block_2inch.stl` | 2 x 2 x 2 cube | scenario, porosity case pack, mold, thermal tests |
| `small_block.stl` | 1 x 1 x 1 cube (chill, core, riser stand-in) | scenario, masking, thermal tests |
| `cylinder.stl` | 2 dia x 3 cylinder (riser) | scenario and mold tests |
| `thick_thin_test.stl` | thick and thin sections, 3 x 3 x 5 envelope | scenario and k-d tree tests |
| `l_bracket.stl` | L bracket, 4 x 2 x 3.5 envelope | desktop orientation smoke test |
| `cast.stl` | 15.25 x 6 x 2.5 plate | riser symmetry, pipe and sleeve tests |
| `riser.stl` | 4 dia x 6 cylinder on the plate (z 2.5 to 8.5) | riser symmetry, pipe and sleeve tests |

Results in `tests/data/porosity_reference_targets.json` and pinned numbers in
the tests were measured on these exact files. Changing one changes those numbers.
