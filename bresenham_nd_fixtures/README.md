# N-D Bresenham reference fixtures

Generated against `_bresenham_nd` in the `bresenham-nd` worktree.

## Other implementations (not only ITK)

| Source | Dims | Notes |
| --- | --- | --- |
| **raster_geometry** `bresenham_line` | N-D | Pure Python. Pip package currently broken on NumPy 2 (`np.float_`); algorithm vendored for this run. |
| **line_drawing** (Rust) `Bresenham3d` | 3-D only | Cargo crate 1.0.1. No generic N-D API. |
| **ITK** `BresenhamLine` | N-D | Gold imaging reference. Python wheels do not wrap this class; fixtures use a port of `itkBresenhamLine.hxx` from ITK **v5.4.0**. Index→Index goes through a **normalised float direction**. |
| ActiveState recipe 578112 | N-D | Skipped: `nslope` + `np.rint` (DDA-style), not integer Bresenham. |
| encukou `bresenham` | 2-D only | Not N-D. |

So there **are** other N-D options: `raster_geometry` is the practical Python one. Rust covers 3-D only. ITK remains the main imaging-library reference.

## Agreement (same corpora as the JSON)

On a ±3 2-D box (2401 pairs), random 3-D/4-D samples:

| | 2-D | 3-D | 4-D |
| --- | ---: | ---: | ---: |
| ours vs `skimage.draw.line` / `_line` | **100%** | — | — |
| ours vs ITK port | 71% | 77% | 64% |
| ours vs raster_geometry | 71% | 57% | 43% |
| ITK vs raster_geometry | 90% | 80% | 79% |
| raster_geometry vs Rust Bresenham3d | — | **100%** | — |

Disagreements are **tie-breaking**, not length: all keep Chebyshev length `max(|Δ|) + 1`. Example `(0,0,0)→(2,4,8)`: ours steps the middle axes earlier than Rust/raster; both sequences have 9 points.

`_bresenham_nd` is locked to scikit-image’s 2-D Bresenham, not to ITK.

## Files

- `nd_bresenham_references.json` — all refs side by side (`ours`, `skimage_line`, `itk`, `raster_geometry`, `rust_line_drawing` where applicable) plus agreement table.
- `itk_bresenham_line.json` — ITK port pixels only (external imaging reference).
- `ours_bresenham_nd.json` — `_bresenham_nd` pixels (regression oracle for this worktree).

Regenerate with the comparison script in the agent session, or re-run the ITK/raster/Rust drivers against the same starts/stops listed in the JSON.
