# Coordinate port plan for SK2-API

Companion to `coordinate_review.md` (what we want) and
`skimage2_port_background.md` (how the port works).

This document gives a staged plan to move `src/_skimage2` to the array ("ij")
coordinate convention.  Each stage leaves `src/_skimage2`, `src/skimage`, and
both test trees in a testable state.

All file references are to `src/_skimage2` unless stated differently.


## 1. Goal

In SK2-API, a coordinate of length D always indexes the first D axes of the
image array, in order.  No function reverses the order of the first two
coordinates.  No parameter, return value, or docstring uses `x`, `y`, `row`, or
`column` to name an image axis.

SK1-API (`src/skimage`) must keep its present behaviour, bit for bit.


## 2. What the code does now

The good news: the code that reverses coordinates is small and it is in one
place.  The transform classes are already dimension-agnostic.  They multiply an
`(N, D)` array of coordinates by a matrix.  They do not know about images.  The
reversal happens only where transforms meet an image, and in the named
constructor parameters.

| Where | Line | What it does |
| --- | --- | --- |
| `transform/_warps.py` | 699 | `warp_coords` builds the grid as `np.indices((cols, rows))`, so `coord_map` receives `(col, row)` pairs. |
| `transform/_warps_cy.pyx` | 180 | `_warp_fast` calls `transform_func(tfc, tfr, ...)`, so the matrix acts on `(col, row)`. |
| `transform/_geometric.py` | 2672 | `matrix_transform` documents its input as "x, y coordinates". |
| `transform/_warps.py` | 554 | `swirl` takes `center` as `(column, row)`. |
| `transform/_warps.py` | 368 | `rotate` takes `center` as `(cols, rows)`. |

Note that `warp` already uses array order for the N-D path, where
`inverse_map` is an array of coordinates.  Only the 2-D callable path reverses.
This is the inconsistency that SKIP-4 gives as its first example.

`warp_polar` is already correct: its `center` is `(row, col)`
(`transform/_warps.py:1139`).

Three other facts control the cost of the work:

* `doc/examples` uses only SK1-API (327 `from skimage` imports, 0 `skimage2`).
  A change in SK2-API does not touch the gallery until we port the gallery.
* `tests/skimage` and `tests/skimage2` are near-duplicates today.  A flip makes
  them diverge.  That divergence is the useful signal, not a problem.
* `skimage` shims share their docstrings with `_skimage2` through
  `adapt_doctests`, which rewrites import names only.  A shim whose *behaviour*
  differs from `_skimage2` must own its docstring.  `skimage/feature/corner.py`
  (`corner_peaks`) is the model to copy.


## 3. Three groups of work

Sort every affected name into one of three groups.  The groups have very
different risk, and they must not be mixed in one pull request.

### Group A — names and signatures, no numbers change

The code is already in array order.  Only parameters, return names, and prose
are wrong.

* `draw`: `r0, c0, r1, c1` -> `i0, j0, i1, j1`; `rr, cc` -> `ii, jj`
  (`draw/draw.py`, `draw/_draw.pyx`, `draw/_polygon2mask.py`,
  `draw/_random_shapes.py`).
* `feature`: keypoints and blobs are already `(row, col)`
  (`feature/util.py:38`, `feature/orb.py:63`, `feature/sift.py:161`,
  `feature/blob.py:103`, `feature/brief.py:166`, `feature/censure.py:179`).
* `measure`: `find_contours`, `regionprops` prose, `profile_line`, `moments`.
* `graph`, `segmentation`, `registration`, `morphology`, `util`: prose only.

Risk: low.  A reviewer can check these by reading.  A silent numeric change is
not possible if the diff touches no arithmetic.

### Group B — the numbers change

These reverse coordinates today.  About 30 names, of which 16 are the transform
cluster.

Transform cluster (must move together).  Section 10 covers the breaking
change that `warp` forces on users:

1. `warp` (2-D callable path)
2. `warp_coords`
3. `_warp_fast`
4. `matrix_transform`
5. `estimate_transform`
6. `ProjectiveTransform`
7. `AffineTransform` (`scale`, `shear`, `rotation`, `translation`)
8. `EuclideanTransform` (`rotation`, `translation`)
9. `SimilarityTransform` (`scale`, `rotation`, `translation`)
10. `PolynomialTransform` (the `x`/`y` formula in its docstring)
11. `PiecewiseAffineTransform`
12. `ThinPlateSplineTransform` (`_thin_plate_splines.py:107`)
13. `FundamentalMatrixTransform`
14. `EssentialMatrixTransform`
15. `rotate` (`center`)
16. `swirl` (`center`)

Separable, one pull request each:

17. `hough_circle_peaks` — returns `accum, cx, cy, rad`
    (`transform/hough_transform.py:350`)
18. `probabilistic_hough_line` — returns `((x0, y0), (x1, y1))`
    (`transform/hough_transform.py:279`)
19. `hough_line`, `hough_line_peaks` — the angle refers to the display axes
20. `hough_ellipse` — fields `(accumulator, yc, xc, a, b, orientation)`
    (`transform/hough_transform.py:148`)
21. `CircleModel` — `params` is `(xc, yc, r)`
22. `EllipseModel` — `params` is `(xc, yc, a, b, theta)`
23. `points_in_poly` (`measure/pnpoly.py:52`)
24. `grid_points_in_poly`
25. `LineModelND.predict_x` / `predict_y` (rename only; the model is N-D and
    convention free)
26. `structure_tensor(order=)` (`feature/corner.py:72`)
27. `hessian_matrix(order=)` and `_hessian_matrix_with_gaussian`
28. `filters.rank.*` — `shift_x`, `shift_y`
    (`filters/rank/generic.py:109`, shared by about 20 functions)

### Group C — angles

An angle has no meaning until the coordinate frame is fixed.  Every angle
changes meaning when the frame changes, even when no line of code changes.
Treat angles as their own group and do them after Group B.

Two separate things move here: the **sign**, per decision D2, and for `rotate`
the **units**, per decision D6. They are independent, they affect the same
scalar argument, and both are silent.

* `rotate(angle=)` — sign **and** units: degrees today, radians after D6
* `swirl(rotation=)` — sign only; already radians, but undocumented
* `AffineTransform(rotation=)`, `EuclideanTransform`, `SimilarityTransform`
* `regionprops.orientation` (`measure/_regionprops.py:679`)
* `draw.ellipse(rotation=)`, `draw.ellipse_perimeter(orientation=)`
* `EllipseModel` `theta`
* `hough_line` `theta`, `hough_ellipse` `orientation`
* `filters.gabor(theta=)`, `gabor_kernel(theta=)`
* `radon(theta=)`, `iradon`, `iradon_sart`, `order_angles_golden_ratio` — in
  degrees today, radians after D6
* `warp_polar`


## 4. Decisions to make before any code changes

These are API design decisions.  They are not implementation details.  Make
them first, write them down, then implement.  Each one blocks a later stage.

### D1. Letters for the axes — DECIDED: coordinates become tuples

**Decision: fixed-arity functions take coordinate tuples, not one scalar per
axis.**  Letters for the third and higher axes are then not needed, and the
arity problem disappears with them.

This finishes a conversion that `draw` has already half done.  Today the
module holds both idioms, and two direct siblings disagree:

    disk(center, radius)                 # tuple
    circle_perimeter(r, c, radius)       # scalars

`disk`, `rectangle`, `rectangle_perimeter`, `set_color`, `polygon2mask` and
`line_nd` already take tuples or coordinate arrays.  The list below is the
remainder.

| Now | Becomes | Group |
| --- | --- | --- |
| `line(r0, c0, r1, c1)` | `line(start, stop)` | A |
| `line_aa(r0, c0, r1, c1)` | `line_aa(start, stop)` | A |
| `bezier_curve(r0, c0, r1, c1, r2, c2, ...)` | `bezier_curve(p0, p1, p2, ...)` | A |
| `circle_perimeter(r, c, radius, ...)` | `circle_perimeter(center, radius, ...)` | A |
| `circle_perimeter_aa(r, c, radius, shape)` | `circle_perimeter_aa(center, radius, shape)` | A |
| `ellipse(r, c, r_radius, c_radius, ...)` | `ellipse(center, radii, ...)` | A |
| `ellipse_perimeter(r, c, r_radius, c_radius, ...)` | `ellipse_perimeter(center, radii, ...)` | A |
| `ellipsoid(a, b, c, ...)` | `ellipsoid(radii, ...)` | **B** |
| `ellipsoid_stats(a, b, c)` | `ellipsoid_stats(radii)` | **B** |
| `haar_like_feature(ii, r, c, width, height, ...)` | `haar_like_feature(ii, start, extent, ...)` | **B** |
| `draw_haar_like_feature(im, r, c, width, height, ...)` | `draw_haar_like_feature(im, start, extent, ...)` | **B** |
| `multiblock_lbp(ii, r, c, width, height)` | `multiblock_lbp(ii, start, extent)` | **B** |
| `draw_multiblock_lbp(im, r, c, width, height, ...)` | `draw_multiblock_lbp(im, start, extent, ...)` | **B** |
| `_shared/_geometry.py::polygon_clip(rp, cp, r0, c0, r1, c1)` | private; same shape | A |

Four of these are Group B, not Group A, because a number changes as well as a
name.  `ellipsoid(a, b, c)` documents its semi-axes as x, y and z
(`draw/draw3d.py:14`).  The four haar and LBP functions take `(r, c)` in array
order and `(width, height)` in imaging order **in the same call**; folding the
pair into an `extent` in array order reverses it.  Keep those four out of the
Group A pull requests, by rule 4 of Section 6.

Return values are unchanged.  `line` keeps returning `(ii, jj)`, a tuple of
index arrays, because that is what indexes an array: ``img[ii, jj]``.  This
decision is about arguments.

**Three questions the decision forces.**

1. `polygon(r, c, shape)` and `polygon_perimeter(r, c, shape, clip)` take two
   *arrays*, not two scalars, so they sit outside the decision as worded.  But
   `polygon2mask(image_shape, polygon)` already takes an `(N, 2)` array, so the
   pair disagrees the same way `disk` and `circle_perimeter` do.
   Recommendation: convert them too, to `polygon(coords, shape)`.  Confirm.
2. `line(start, stop)` and `line_nd(start, stop)` become signature-identical,
   and they are different algorithms.  `line` is integer Bresenham; `line_nd`
   samples the segment with `np.linspace` and rounds each axis on its own.
   Measured over every integer endpoint pair in a 13x13 box, with
   `endpoint=True` so the point counts match, **21% of pairs give different
   pixels**.  Their symmetries are complementary and neither has both: `line`
   is translation-invariant but not reversal-symmetric; `line_nd` is
   reversal-symmetric but not translation-invariant.  Both are
   diagonal-connected in 2-D, so `line_nd`'s "ndim-connected" does not
   distinguish them.  See Section 12.
   Recommendation: keep both for this port, cross-reference them, and record
   "merge them" as a separate proposal.  Merging changes output for a fifth of
   all lines.
3. The user guide must still state that axis order in prose is positional, and
   that "first axis" means the first *spatial* axis when a `channel_axis` is
   given.  Dropping the letters does not drop that obligation.

**Cost.**  Under the rejected option (a), a shim could rewrite parameter names
in the inherited docstring mechanically.  Under (b) the Parameters section
changes shape, so each of the 13 shims must carry a hand-written docstring, as
in Section 5.4.  Budget for that in Stage 1.

### D2. The sign of a rotation — DECIDED: follow the algebra

Let `P` be the matrix that swaps the first two coordinates.  For a homogeneous
2-D transform, `H_ij = P H_xy P`.  For a rotation this gives:

    P R(theta) P == R(-theta)

**Decision: a positive angle always turns axis 0 toward axis 1.**  The algebra
governs; no constructor hides a sign.

Consequences, all confirmed by measurement (see Section 10.1):

* Today, in `(x, y)`, `R(theta)` turns *coordinates* clockwise on a
  `plt.imshow` display, because `y` points down the screen.
* After the flip, in `(i, j)`, the same `R(theta)` turns coordinates
  anticlockwise on the display.
* `warp` uses its map as the *inverse* map, so the *picture* always turns the
  opposite way to the coordinates.  Today a positive rotation gives an
  anticlockwise picture; after the flip it gives a clockwise picture.

So any code of the form `warp(image, SomeTransform(rotation=theta))` turns the
picture the other way after the port.  This is the intended result of the
decision, not an oversight.

**Open sub-decision: `rotate` and `swirl`.**  `rotate(image, angle)` and
`swirl(image, rotation=...)` take an angle but no coordinate, apart from
`center`.  Their angle is documented as a *visual* direction
(`transform/_warps.py:369`: "counter-clockwise"), and the sign is already
hidden today by the inverse-map trick, not by the parameter.  So they have a
free choice inside decision D2:

* (i) Reimplement as `warp(image, Sim(rotation=theta))`.  The picture reverses.
  Keeps the undocumented identity `rotate(theta) ~ warp(image,
  Sim(rotation=theta))`.
* (ii) Reimplement as `warp(image, Sim(rotation=-theta))`.  `rotate(image, 30)`
  looks exactly as it does today.  Breaks that identity.

Recommendation: (ii).  `rotate` has by far the widest user base in the module,
its promise is visual and can be kept honestly, and reversing it silently buys
no conceptual gain.  Record the choice either way, because it is the difference
between a wide silent change and a narrow one.

### D3. `shear` and `scale` in `AffineTransform`

`scale` is a simple swap: `(sx, sy)` becomes `(s_axis0, s_axis1)`.  `shear`
needs a definition: which axis shears toward which.  Write the definition down
before the flip.

### D4. What replaces `order='rc'|'xy'`

`structure_tensor` and `hessian_matrix` are the only functions with an explicit
switch today.  In SK2-API there is one convention, so the parameter should go.
Confirm that we remove it rather than keep it as `order='ij'`.

### D5. `shift_x` / `shift_y` in `filters.rank`

Replace with one `shift` tuple in array order, or with `shift_axis0` /
`shift_axis1`.  A tuple is closer to the rest of the API.


### D6. Angle units — DECIDED: radians

**Decision: `transform.rotate` and `transform.swirl` take radians.**

Most of the library already does. Measured across `src/_skimage2`, angles are
documented in radians in `measure/fit.py`, `filters/_gabor.py`,
`feature/orb.py`, `draw/draw.py`, the Hough modules, `feature/texture.py` and
the transform constructors. Only two APIs use degrees:

| Function | Parameter | Units today |
| --- | --- | --- |
| `transform.rotate` | `angle` | **degrees** (`transform/_warps.py:363`) |
| `transform.swirl` | `rotation` | **radians already**, undocumented |
| `radon`, `iradon`, `iradon_sart`, `order_angles_golden_ratio` | `theta` | **degrees** (`transform/radon_transform.py:27, 209, 332, 407`) |
| everything else carrying an angle | — | radians |

So the two named functions need different work.

**`rotate` changes units.** `rotate(image, 30)` becomes
`rotate(image, np.pi / 6)`. Anyone who does not update gets a rotation 57.3
times too large, from a call that stays perfectly valid. Together with the
direction question of D2, this makes `rotate` the single most dangerous
signature in the port: two independent silent changes to one scalar argument.

**`swirl` changes nothing numerically.** Its `rotation` is already in radians,
which the docstring never states. Measured: `rotation=2*pi` differs from
`rotation=0` by 4e-14, and `rotation=360` differs by 9.8e-01, so a full turn is
a no-op only in radians. The work here is to document the existing behaviour,
not to change it.

**A tripwire worth considering.** Unlike the coordinate flip, this one has a
usable heuristic. Radian rotations are almost always within a few multiples of
`2 * pi`, while a stale degrees call is typically 15, 30, 45, 90 or 180. A
warning on `abs(angle) > 2 * pi` — "angle is in radians; did you mean
`np.deg2rad(30)`?" — would catch nearly every unported call. It is a heuristic
and would occasionally fire on a legitimate large rotation, so it should be a
warning and never an error. Decide whether to ship it.

The same test is far more reliable for the `radon` family, because `theta`
there is an array spanning most of a half turn. Any stale call passes values up
to 180 where the new maximum is `pi`, so a check on `max(abs(theta)) > 2 * pi`
catches essentially every unported call with no realistic false positive.

**The `radon` family converts too.** `radon`, `iradon`, `iradon_sart` and
`order_angles_golden_ratio` all take `theta` in degrees today, and all move to
radians. The change reaches past the signatures, into three places that
hard-code 180:

| Where | What | Becomes |
| --- | --- | --- |
| `radon_transform.py:62` | `theta = np.arange(180)`, the `radon` default | `np.linspace(0, np.pi, 180, endpoint=False)` |
| `radon_transform.py:259` | `iradon` default, `np.linspace(0, 180, n, endpoint=False)` | the same span in radians |
| `radon_transform.py:490` | `iradon_sart` default, likewise | the same span in radians |
| `radon_transform.py:102` | `np.deg2rad(theta)` inside `radon` | delete |
| `radon_transform.py:308` | `np.deg2rad(theta)` inside `iradon` | delete |
| `radon_transform.py:357` | `interval = 180` in `order_angles_golden_ratio` | `np.pi` |
| `_radon_transform.pyx:35` | `theta = theta / 180. * M_PI` in `bilinear_ray_sum` | delete |
| `_radon_transform.pyx:122` | the same, in `bilinear_ray_update` | delete |

Two of those are easy to miss. The Cython takes **degrees** and converts on
entry, so `iradon_sart` hands it degrees with no Python-level `deg2rad` to grep
for. And `order_angles_golden_ratio` holds no conversion at all: its
golden-ratio ordering is modulo a half turn, spelled `180`, which has to become
`np.pi` or the ordering silently degrades.

The defaults change representation but not meaning, so a caller who never
passed `theta` sees no difference.

## 5. Mechanism

### 5.1 The main rule

`_skimage2` flips.  The `skimage` shim adapts back.

This keeps the project policy: all real work is in `_skimage2`, and `skimage`
is a thin layer on top.  It also keeps `tests/skimage` unchanged, which is the
proof that SK1-API did not move.

The `skimage` shim gets:

* an explicit `def` with the SK1 signature and the SK1 parameter names,
* its own docstring (see 5.4),
* a `@ski2_migration_decorator(...)` message, which also writes the migration
  guide entry (`doc/tools/write_migration_guide.py`),
* the coordinate adapter.

`skimage/feature/corner.py::corner_peaks` is the working example of this shape.

### 5.2 The adapter for the transform cluster

For anything built on a homogeneous matrix, the adapter is exact and is one
line:

    H_xy = P @ H_ij @ P

with `P = [[0, 1, 0], [1, 0, 0], [0, 0, 1]]`.  The same form works for the
fundamental and essential matrices, because `P` is its own transpose and its
own inverse, so `F_xy = P F_ij P` as well.

For transforms without a matrix (`PolynomialTransform`,
`PiecewiseAffineTransform`, `ThinPlateSplineTransform`) the adapter reverses
the two coordinate columns on input and on output.

Put `_flip_coords` and `_flip_matrix` in `transform/_geometric.py` as private
helpers, and use them from both sides.

### 5.3 One exception: a private mode flag in `warp`

`warp` chooses a fast Cython path when `inverse_map` is a homography
(`transform/_warps.py:965`).  A shim that wraps `inverse_map` in a Python
callable would silently lose that path and make SK1-API much slower.

So `warp` and `warp_coords` should keep a private, temporary keyword — for
example `_coords='ij'|'xy'` — used only by the `skimage` shim.  Mark it clearly
as temporary, and add it to `TODO.txt` for removal when SK1-API goes.

Use this exception only here.  Everywhere else, adapt in the shim.

### 5.4 Docstrings

Today a shim that only re-exports shares the `_skimage2` docstring, and
`adapt_doctests` rewrites the import lines.  It does not rewrite numbers.

Therefore: **any function in Group B or Group C must get a full, separate
docstring in its `skimage` shim.**  Otherwise SK1 users read SK2 examples that
give the wrong answer.

This is the largest hidden cost in the plan.  Budget for about 30 docstrings.
Make it a line in the checklist for every flipped function (Section 8).

### 5.5 Renames in Group A

Group A changes parameter names, which is visible in SK2-API but must not be
visible in SK1-API.  Most of these parameters are used positionally
(`draw.line(r0, c0, r1, c1)`), so the shim is a two-line pass-through.

Add a small `rename_params` helper to `skimage/_migration.py` for the keyword
cases, so that we do not write the same wrapper 40 times.  It should map SK1
names to SK2 names silently, or with `PendingSkimage2Change` when the name is
one that users type often.


## 6. Rules that keep the tree testable

These rules make each pull request easy to review and hard to get wrong.

1. **`tests/skimage` is read-only.**  If a pull request in this programme
   changes a file under `tests/skimage`, that is a defect until proven
   otherwise, and the pull request description must explain it.  This is the
   single best guard against a silent SK1 change.
2. **`tests/skimage2` changes in the same pull request as the flip.**  The
   updated test is the specification of the new behaviour.  A reviewer reads
   the test diff to see exactly which numbers moved.
3. **Add a mirror test for every flipped name.**  In `tests/skimage`, assert
   that the SK1 result equals the flipped SK2 result:

       assert_allclose(ski.f(xy_in), flip(ski2.f(flip(xy_in))))

   These tests are cheap, they catch adapter errors directly, and they are
   deleted with the shims.
4. **One group per pull request.**  Never mix a rename (Group A) with a flip
   (Group B) in one pull request.  A rename diff must contain no arithmetic.
5. **Run the `_skimage2` doctests.**  `pytest --doctest-plus --pyargs _skimage2`
   is part of the acceptance for every stage, because the flip changes printed
   output in examples.
6. **Keep `test_public_skimage_api` green.**  The frozen SK1 API snapshot must
   not move.


## 7. Stages

Each numbered item is one pull request unless stated differently.

### Stage 0 — decisions and scaffolding

0.1 Write the convention document.  Record D1 to D5 with the reasoning.  A
    short SKIP, or a page under `doc/source/user_guide/`, so that later pull
    requests can cite one place.  **Blocking.**
0.2 Add `_flip_coords` and `_flip_matrix` to `transform/_geometric.py`, with
    tests.  No caller yet.
0.3 Add `rename_params` to `skimage/_migration.py`, with tests.
0.4 Optional: add `tools/check_coord_names.py`, which fails when a module on a
    "done" list uses `x`, `y`, `row`, `col`, `rr`, `cc` as a coordinate name.
    Start with an empty done list and add modules as stages land.

### Stage 1 — Group A renames

No numbers change in any of these.

1.1 `draw`: fold the scalar coordinates of `line`, `line_aa`,
    `bezier_curve`, `circle_perimeter`, `circle_perimeter_aa`, `ellipse` and
    `ellipse_perimeter` into tuples, per decision D1.  Signatures change;
    results do not.  The largest and most visible change in Stage 1.
1.2 `feature` keypoints, blobs, and descriptors.
1.3 `measure` prose: `find_contours`, `regionprops`, `moments`, `profile_line`.
1.4 `graph`, `segmentation`, `registration`, `morphology`, `util` prose.
1.5 `CONTRIBUTING.rst` and `doc/source/user_guide/numpy_images.rst`, which
    currently teach `plane, row, column` (see `planes.md` for the full list of
    places).

### Stage 2 — remove the existing convention switch

2.1 Remove `order='xy'` from `structure_tensor`, `hessian_matrix`, and
    `_hessian_matrix_with_gaussian`.  Keep `order=` in the SK1 shim.

    Self-contained, and a good place to prove the shim pattern of Section 5.1
    before the large stages.  It is **not** as simple as it looks: the two
    functions need different shims, and one of them exposes a bug.  See
    Section 14.

### Stage 3 — isolated flips

Any order.  Each is independent of the others.

3.1 `measure.points_in_poly` and `grid_points_in_poly`.
3.2 `measure.CircleModel` and `EllipseModel`; rename
    `LineModelND.predict_x` / `predict_y`.
3.3 `transform.hough_circle_peaks` and `probabilistic_hough_line`.
3.4 `transform.hough_ellipse` field order and names.
3.5 `filters.rank` `shift_x` / `shift_y` (per D5).
3.6 `draw.ellipsoid` and `ellipsoid_stats`: fold `a, b, c` into `radii` and
    reverse them, since they are documented as x, y, z.
3.8 `draw.line_nd`: round half up instead of half to even, per Section 13.3.
3.9 `draw.line` and `line_aa`: normalise the endpoint order, per Section 13.3.
3.7 `haar_like_feature`, `draw_haar_like_feature`, `multiblock_lbp` and
    `draw_multiblock_lbp`: fold `r, c` into `start` and `width, height` into an
    array-order `extent`.  These four mix both conventions in one call today,
    so the tuple change and the flip cannot be separated.

### Stage 4 — the transform cluster

This is the one part that cannot be split by function.  Split it by kind of
change instead.

4.1 **Refactor only, no behaviour change.**  Route every place that reverses
    coordinates through the Stage 0 helpers.  Correct the docstrings that
    already disagree with the code — `warp_coords` says `(row, col)` while the
    code builds `(col, row)` (`transform/_warps.py:650`, and the comment at `698` contradicts the code at `699`).  Add tests
    that pin the present 2-D against N-D difference, so that the flip shows up
    clearly in the next diff.
4.2 **The flip.**  Change `warp_coords` to build `(i, j)`, change `_warp_fast`
    to call `transform_func(tfr, tfc, ...)`, change the named constructor
    parameters, and change `rotate` and `swirl` centres.  Add the private
    `_coords` keyword of Section 5.3 and the SK1 adapters.  Update
    `tests/skimage2/transform/test_geometric.py` and `test_warps.py`
    (about 2600 lines together).
4.3 **The rest of the transform classes.**  `PolynomialTransform` formula,
    `PiecewiseAffineTransform`, `ThinPlateSplineTransform`,
    `FundamentalMatrixTransform`, `EssentialMatrixTransform`.
4.4 **In-tree users.**  `registration/_optical_flow.py`,
    `transform/radon_transform.py`, `filters/_window.py`, `measure/fit.py`.

Stage 4.2 is the highest-risk pull request in the programme.  It should have
the mirror tests of rule 6.3 written *before* the flip, in Stage 4.1, so that
they fail loudly if the adapter is wrong.

### Stage 5 — angles

Only after Stage 4 lands, because an angle has no meaning until the frame is
fixed.  Apply decision D2.

5.1 `rotate`: the direction, per D2, and degrees to radians, per D6. Also the
    `rotation` parameter of the matrix transforms. One migration guide entry
    must cover both changes, because a reader who fixes only one is still
    wrong.
5.1a `swirl`: document that `rotation` is in radians, and decide the direction
    question of D2. No numeric change.
5.2 `regionprops.orientation`.
5.3 `draw.ellipse` and `draw.ellipse_perimeter`; `EllipseModel.theta`.
5.4 `hough_line`, `hough_ellipse`, `warp_polar`.
5.4a The `radon` family: `radon`, `iradon`, `iradon_sart` and
    `order_angles_golden_ratio` from degrees to radians, including the three
    defaults, the two `np.deg2rad` calls, the `interval = 180` in the angle
    ordering, and the two `theta / 180. * M_PI` conversions inside
    `_radon_transform.pyx`. See the table in D6.
5.5 `filters.gabor` and `gabor_kernel`.

### Stage 6 — close out

6.1 Turn on `tools/check_coord_names.py` for the whole `_skimage2` tree.
6.2 Review the migration guide as a whole document, not as a set of fragments.
6.3 Port `doc/examples` to SK2-API.  This is a large job on its own, and it is
    not blocked by, nor blocking, the stages above.
6.4 Record the removal of the private `_coords` keyword in `TODO.txt`.


## 8. Checklist for one flipped function

Use this for every name in Group B and Group C.

- [ ] `_skimage2` implementation flipped.
- [ ] `_skimage2` docstring rewritten: parameters, returns, prose, and the
      doctest output values.
- [ ] `tests/skimage2` updated, with the expected values changed by hand, not
      by copying the new output.
- [ ] `skimage` shim written with the SK1 signature and SK1 parameter names.
- [ ] `skimage` shim has its own docstring with the SK1 values.
- [ ] `@ski2_migration_decorator` message written, and it reads well both as a
      warning and as a migration guide entry.
- [ ] Mirror test added in `tests/skimage`.
- [ ] `tests/skimage` otherwise unchanged.
- [ ] `pytest --doctest-plus --pyargs _skimage2` passes.
- [ ] `pytest --doctest-plus --pyargs skimage` passes.
- [ ] Any in-tree caller in `_skimage2` updated.


## 9. Risks

**Silent numeric change for users who port to SK2-API.**  See Section 10.

**Silent numeric change in SK1-API.**  The largest risk.  Controlled by rule
6.1 (`tests/skimage` is read-only) and by the mirror tests.

**Symmetric test data.**  Many existing tests use square images and symmetric
shapes, which pass under either convention.  Before Stage 4.2, check the
transform tests for symmetry and make the fixtures asymmetric where they are
not.  A test that cannot fail is worse than no test.

**Angle sign missed.**  A coordinate flip changes every angle in the library,
including angles in functions that no pull request touched.  Stage 5 exists to
make this explicit, but the risk is that a function is forgotten.  The Group C
list in Section 3 is the register; keep it up to date.

**Only one of the two `rotate` changes applied.**  Its `angle` moves in sign
(D2) and in units (D6). A user who reads the release notes and fixes only the
units still gets a picture rotated the wrong way, and one who fixes only the
sign is out by a factor of 57.3. Neither raises. The two must be described
together, in one migration entry and one release note, never as separate
bullets.

**Lost fast paths.**  A shim that wraps a transform in a plain callable makes
SK1 `warp` much slower.  Section 5.3 avoids this.  Add a benchmark check to
Stage 4.2.

**Docstring drift.**  Once a shim owns its docstring, the two copies can
diverge over time.  Accept this: the shims are temporary, and SK1-API is
frozen, so its docstrings should not need to change.


## 10. The `warp` breaking change

`warp` is where the convention lives, so `warp` is where users break.  The same
call, with the same arguments, gives a different picture before and after the
port, and it does not raise.  This section analyses who breaks and proposes how
to answer it.

### 10.1 Measured confirmation of D2

Two checks against the built tree, using an asymmetric test image (a bar in the
upper-left quadrant of a 41x41 array).

Direction today:

    original bearing        115.56 deg
    rotate(+30) bearing     145.52 deg    -> +30.0, anticlockwise on screen

Sense after the flip.  `rotate_ij(theta)` applies `R(theta)` to `(i, j)` column
vectors and uses it as the inverse map, which is what SK2 `warp` will do with
`SimilarityTransform(rotation=theta)`:

    rotate(+30) == rotate_ij(+30)     False
    rotate(+30) == rotate_ij(-30)     True

So the picture turns the other way, and the numeric bridge between the two
worlds is exactly `theta -> -theta`, as `P R(theta) P == R(-theta)` predicts.

### 10.2 Which inputs break, and which we can detect

`warp(image, inverse_map)` accepts five kinds of thing.  They do not carry the
same risk.

| Input kind | Broken by the flip? | Can we detect it? |
| --- | --- | --- |
| SK2 transform object, or its `.inverse` | Yes | Only when the object came from `skimage` (SK1) |
| Bare `(3, 3)` ndarray | Yes | No, while we still accept it |
| Plain callable | Yes | No |
| `(D, ...)` coordinate array | **No** | n/a — already array order |
| `map_args` for the callable | Yes, indirectly | No |

Counts from the 20 `warp` calls in `doc/examples`: 17 pass a transform object
or its `.inverse`, 2 pass a plain callable (`plot_swirl.py`,
`plot_tps_deformation.py`), 1 passes a coordinate array
(`plot_opticalflow.py`) and is unaffected.

Field data agrees, and adds one thing the gallery misses.  A survey of public
GitHub code (method in Section 10.7) gives, per repository:

| Form | Repositories | 95% interval | Broken by the flip? |
| --- | --- | --- | --- |
| Transform object or `.inverse` | 74.8% | 68.4 - 80.2 | Yes, silently |
| Coordinate array | 14.1% | 10.0 - 19.5 | **No** |
| Bare `(3, 3)` matrix | 10.2% | 6.8 - 15.1 | Yes, silently |
| Plain callable | 9.2% | 6.0 - 14.0 | Yes, silently |

Percentages are of 206 repositories and sum above 100, because a repository may
use more than one form.  By call site (n = 369) the split is 72.4, 10.8, 9.8
and 7.0 percent.

Two conclusions change the weight of the proposals in Section 10.5:

* The bare-matrix form is **not** negligible, although the gallery has no
  example of it.  About one repository in ten passes a raw `(3, 3)` array,
  usually as `np.linalg.inv(H)` or `np.vstack([rot_mat, [0, 0, 1]])`.  Refusing
  it (M2) therefore converts a real, silent failure into a loud one.
* About one repository in seven uses the coordinate-array form, which the flip
  does not touch at all.

The survey also found user code that already performs the conjugation of
Section 5.2 by hand.  Two unrelated repositories build the matrix they pass to
`warp` as `T[[1, 0, 2], :][:, [1, 0, 2]].T`, one of them commented "inverte
righe e colonne".  That is `P T P` written out, in the wild, to bridge array
order and imaging order.  It is direct evidence that the present convention
makes users do the work that this port removes.

### 10.3 The part we cannot fix

There is no mechanism that catches a user who ports a correct SK1 program to
SK2-API by changing only the import.  The arguments stay valid; only their
meaning moves.

That is by design, and it is the reason SKIP-4 chose a new package instead of a
major version.  Importing `skimage2` is a deliberate act, and the migration
guide is the contract for what that act means.  So the goal is not to make the
change impossible to get wrong.  The goal is to make every *reachable* failure
loud, and to make the remaining work mechanical.

### 10.4 One piece of good news

For a large class of real code the port **removes** a coordinate swap rather
than adding one.

Every point source in scikit-image already returns array order:
`corner_peaks`, `ORB.keypoints`, `blob_log`, `find_contours`,
`regionprops.centroid`, `peak_local_max`.  So today, code that feeds detected
points into a transform and then warps an image must reverse something
somewhere.

`doc/examples/transform/plot_matching.py` shows this in the gallery.  It fits an
`AffineTransform` to `corner_peaks` output, which is `(row, col)`, then prints
the ground-truth transform, which was applied to the image in `(x, y)`:

    f'Scale: ({tform.scale[1]:.4f}, {tform.scale[0]:.4f}), '
    f'Translation: ({tform.translation[1]:.4f}, {tform.translation[0]:.4f}), '
    f'Rotation: {-tform.rotation:.4f}'

The swapped indices and the `-tform.rotation` are `P H P` written out by hand in
a user-facing example.  After the port all of it deletes, and the example reads
straight.

This gives the migration guide its headline: **if your code reverses
coordinates between a detector and a transform, delete the reversal.**  It also
gives a good grep target — `[:, ::-1]` or `[::-1]` near a transform or `warp`
call.

### 10.5 Proposed mitigations

**M1. Distinct types, and cross-package rejection.  Recommended; do it.**

After Stage 4.2, `skimage.transform.AffineTransform` and
`skimage2.transform.AffineTransform` are different classes.  Use that:

* `skimage2.transform.warp` raises `TypeError` on an SK1 transform object.
* `skimage.transform.warp` raises `TypeError` on an SK2 transform object.

Each message names the class it got, the class it wanted, and the guide.  This
catches every partly-ported file, which is the most common real state of a
codebase in migration.  Cost: a few lines and a message.  Highest value per
line in this section.

**M2. Refuse a bare `(3, 3)` ndarray in SK2.  Recommended; do it.**

Today `warp` guesses that a `(3, 3)` array is a homography.  Its own docstring
admits the collision: "a `(3, 3)` matrix is interpreted as a homogeneous
transformation matrix, so you cannot interpolate values from a 3-D input, if
the output is of shape `(3,)`" (`transform/_warps.py:823`).  In SK2, require
`ProjectiveTransform(matrix)`.  A silent case becomes a loud one, and an
acknowledged wart goes away.

**M3. Give callables their own keyword.  Recommended, but it is a scope
decision.**

Keep `inverse_map` positional for transform objects and coordinate arrays.  Add
a keyword-only `coord_func=` for callables.  Then `warp(image, my_func)` — the
exact form used by both affected gallery examples — raises:

    TypeError: `inverse_map` does not accept a callable.  Pass it as
    `coord_func=`.  Note that `coord_func` now receives coordinates in array
    (i, j) order, not (x, y).  See <migration link>.

This is the only proposal that makes the fully-silent callable case loud.  It
also stands on its own design merit: `inverse_map` currently means four
different things, which is why its docstring needs six paragraphs.

It goes beyond a pure coordinate change, so it needs a separate decision.  If
it is rejected, the callable case stays silent and the guide is the only
defence.

**M4. An exact conversion recipe, plus a converter in the old package.
Recommended; do it.**

The guide must give a mechanical rule per input kind, not prose:

| You have | Write |
| --- | --- |
| matrix `H` | `P @ H @ P`, `P = [[0,1,0],[1,0,0],[0,0,1]]` |
| `translation=(tx, ty)` | `translation=(ty, tx)` |
| `scale=(sx, sy)` | `scale=(sy, sx)` |
| `rotation=theta` | `rotation=-theta` |
| `shear=...` | per decision D3 |
| `src`, `dst` point arrays | drop any `[:, ::-1]` |
| a callable `f(xy)` | swap the two columns in and out, or delete an existing swap |

Ship `skimage.transform.to_skimage2(tform)` as an exact converter.  Put it in
`skimage`, the package that is going away, so it never enters SK2-API and it
disappears with the shims.

**M5. A migration checker.  Useful, not sufficient.**

A `tools/` script, or a set of `ruff` rules, that flags `warp(` calls,
transform constructors using `rotation=`, `translation=`, `shear=` or `scale=`,
and `[:, ::-1]` near either.  It cannot be complete, because the change is
semantic.  It is still cheap and it points people at the right lines.

**M6. Tell users their tests will not catch it.  Do it, in the guide.**

Symmetric images, square arrays, and angles of 90 or 180 degrees all pass under
either convention.  The guide must say plainly: use an asymmetric fixture, and
look at one output picture.  The same warning applies to our own test suite —
see the "Symmetric test data" risk in Section 9.

### 10.6 What this adds to the stages

* Stage 0: add the D2 sub-decision on `rotate` and `swirl`, and the scope
  decision on M3.
* Stage 4.1: audit the transform tests for symmetric fixtures, and make them
  asymmetric before the flip.
* Stage 4.2: implement M1 and M2 in the same pull request as the flip, so no
  released state has the silent form.
* Stage 4.2: write the M4 recipe as the migration guide entry for `warp`.  This
  is the single most important guide entry in the port.
* Stage 4.3 or a separate pull request: implement M3, if accepted.
* Stage 6: M5, the checker.

### 10.7 Method for the field survey

Recorded so the numbers in Section 10.2 can be challenged or repeated.

1. **Corpus.**  GitHub code search API, two queries: `"from skimage.transform
   import warp" language:python` (1788 files reported) and `"transform.warp"
   language:python` (4704).  Five pages of 100 from each, merged and
   deduplicated: 886 distinct files in 709 repositories.
2. **Remove copies of scikit-image (29% of hits).**  Files owned by the
   `scikit-image` organisation, files inside a `skimage/<subpackage>/` source
   tree (vendored copies and ports such as `cupyimg`), files under
   `site-packages` or similar, and copies of our own gallery examples.  Left
   627 files in 547 repositories.
3. **Fetch.**  458 of the 627 files retrieved; the rest were stale index
   entries returning 404.  The loss should be unrelated to calling style.
4. **Extract.**  Paren-balanced parsing of every `warp(` call, taking the map
   argument either positionally or from `inverse_map=`.  483 call sites.
5. **Classify.**  By the map expression, resolving bare names through their
   assignment in the same file.  The 86 that stayed unresolved were read by
   hand.
6. **Remove two further contamination classes found during the work.**
   Duplicated vendored files, by content hash, so a widely copied module counts
   once (90 sites; TensorLayer's `prepro.py` appears in about 15 repositories
   and micasense's `imageutils.py` in three).  Library pass-throughs, where the
   map argument is the enclosing wrapper's own parameter and so carries no
   information (11 sites).
7. **Base for the estimate.**  369 call sites in 206 repositories.  13 sites
   (3.4%) stayed indeterminate and are excluded.

**Validation.**  18 call sites drawn at random and checked by hand: 18 correct.
Independent whole-index co-occurrence counts are consistent with the sample and
none contradicts it — files importing `warp` that also contain `map_args`
number 268 of 1788 (15.0%, an upper bound on the callable form, against 9.2%
measured); `np.linalg.inv` 453 (25.3%, upper bound on the matrix form, against
10.2%); `np.indices` 249 (13.9%, upper bound on the coordinate-array form,
against 14.1%, so that one may be slightly overestimated).

**Limits, in order of size.**

* GitHub code search ranks by relevance, not at random.  The 886 sampled files
  are not a random sample of the population, and probably lean toward
  well-known repositories.  This is the largest source of error, and it is
  larger than the quoted intervals, which describe sampling noise only.
* Only default branches of public, indexed repositories.  No private or
  industrial code, which is where the maintained downstream mostly lives.
* The API's `total_count` is approximate.
* The queries miss `from skimage import transform as tf` followed by
  `tf.warp(...)` when the file contains neither literal string.
* Public GitHub over-represents student work, tutorials and paper code.  Read
  the numbers as the shape of usage, not as a precise census.


## 11. N-D warping, and the shape of the `warp` API

A later requirement: accept a `(4, 4)` matrix for a 3-D image, and more
generally support N-D warping.  This section reports what `warp` does in N-D
today, and what that means for the input kinds of Section 10.2.

### 11.1 `warp` is 2-D only, and in 3-D it fails silently

Measured against this tree, with a single bright voxel in an ``(8, 9, 10)``
volume and a pure translation of one step along axis 0:

| Call | Result |
| --- | --- |
| `warp(vol, AffineTransform(matrix=M4), order=1)` | **all-zero array, no error** |
| `warp(vol, AffineTransform(matrix=M4), order=0)` | `ValueError` from `_apply_homogeneous` |
| `warp(vol, M4)` where `M4` is `(4, 4)` | `RuntimeError: invalid shape for coordinate array` |
| `warp(vol, coords)` with `coords` of shape `(3, 8, 9, 10)` | correct |
| `scipy.ndimage.affine_transform(vol, M4)` | correct |

The cause: `warp` reads `image.ndim == 3` as "2-D with channels", loops over
the last axis, and hands the `(4, 4)` matrix to `_warp_fast`, which reads the
first nine of its sixteen values as a `(3, 3)` homography.  Every output
coordinate then lands outside the image, so the result is empty.

So N-D support is not a new input kind to slot in.  It does not exist, and its
absence is currently mishandled rather than reported.  **This is a live bug,
independent of the port.  Confirm it against a release and file it separately.**

### 11.2 Matrices and coordinate arrays are not actually ambiguous

The premise that a `(3, 3)` array must be guessed at does not hold.  For a
spatial dimensionality `S`:

* a homogeneous matrix has shape `(S + 1, S + 1)`;
* a coordinate array has `shape[0] == S`.

The leading axis always differs by one, at every dimensionality.  Today's
ambiguity comes from the hard-coded test `inverse_map.shape == (3, 3)`, which
never looks at the image.  The caveat in the docstring — "you cannot
interpolate values from a 3-D input, if the output is of shape `(3,)`"
(`transform/_warps.py:823`) — describes that defect, not an inherent conflict.

The genuine ambiguity is `S` itself, because `warp` has no `channel_axis` and
so guesses that a 3-dimensional array is 2-D plus channels.  That guess is what
breaks 3-D.  Adding `channel_axis`, which `warp_polar` and most of skimage2
already have, settles `S` and removes the guess.

### 11.3 The flip is what unblocks N-D

`warp_coords` is 2-D only *because of* the imaging convention.  Its
`np.indices((cols, rows))`, its `.swapaxes(1, 2)` and its two `_stackcopy`
calls exist only to undo the reversal.  In array order the whole body is three
lines and carries no dimensionality assumption:

```python
def warp_coords(coord_map, shape, dtype=np.float64):
    coords = np.indices(shape, dtype=dtype).reshape(len(shape), -1).T
    return coord_map(coords).T.reshape((len(shape),) + tuple(shape))
```

Measured: this reproduces `scipy.ndimage.affine_transform` exactly in 3-D, and
works unchanged in 2-D and 4-D.  N-D warping is therefore a *payoff* of the
coordinate port, not a complication for it.

### 11.4 Options for the input kinds

**Option A — one `warp`, dispatch on type, not on shape.  Recommended.**

    warp(image, inverse_map, *, channel_axis=None, ...)

`inverse_map` is a transform object **or** a coordinate array.  Those are
distinguished by type, so no shape is ever sniffed.  A bare matrix is refused,
with a message naming the fix; `ProjectiveTransform(matrix=M)` already accepts
any dimensionality, so `(4, 4)` and beyond work through it.  Callables move to
`warp_coords`, which returns a coordinate array to feed back to `warp`.

* Gives M2 and M3 of Section 10.5 for free: bare matrices and callables both
  fail loudly.
* Leaves the coordinate-array users alone.  They are 14% of repositories and
  the flip does not affect them; breaking them buys nothing.
* A positional array that is square with shape `(S + 1, S + 1)` should get the
  specific "this looks like a homogeneous matrix" error, not a generic one.
* Costs: matrix users add one wrapper call, and callable users lose `map_args`
  and use `functools.partial` instead.

**Option B — the proposed split: `warp` for objects and matrices,
`warp_coordinates` for arrays and callables.**

* Symmetric, and it follows `scipy.ndimage`.
* Makes callables loud.
* But it keeps accepting bare matrices, so the 10% of repositories that pass
  one keep flipping silently.  M2 is given up.
* It breaks the 14% coordinate-array users for a reason unrelated to
  coordinates.
* `warp_coordinates` sits one character from the existing `warp_coords`, which
  takes a callable and returns coordinates.  If this option is taken, the two
  must be merged or renamed.

**Option C — copy `scipy.ndimage` fully: three entry points for matrix,
coordinate array and callable.**  Conceptually cleanest, most churn, and it
mostly re-implements `scipy.ndimage` without adding to it.

If the ergonomics of `(4, 4)` matter, Option A can take an explicit
`warp(image, matrix=M)` keyword.  That keeps the convenience and still breaks
the old positional `warp(image, H)` loudly, which is the point.

### 11.5 Effect on the stages

* Stage 0: add the Option A / B / C choice to the decision list.
* Stage 4.1: add `channel_axis` to `warp`, and replace the `== (3, 3)` test
  with one that compares against `S`.  Behaviour-preserving for 2-D.
* Stage 4.2: the three-line array-order `warp_coords` above replaces the
  present body.  N-D support arrives with the flip rather than after it.
* Stage 4.2: fold M2 and M3 into whichever option is chosen.
* Separate from this plan: file the 3-D silent-zero bug of Section 11.1.


## 12. `line` and `line_nd` are different algorithms

Recorded because decision D1 makes their signatures identical, which invites
the assumption that one generalises the other.  It does not.

### 12.1 `line`: integer Bresenham

`draw/_draw.pyx::_line`.  Integer arithmetic only.  It takes the axis with the
larger absolute delta as the driving axis, swapping the roles of the two axes
when the line is steep, and steps one pixel along that axis per iteration.  An
integer error term `d = 2*dr - dc` decides when the minor axis also steps:
`d` loses `2*dc` on each minor step and gains `2*dr` on each major step.  The
final point is written as the literal `(r1, c1)`, so both endpoints are always
present and the output holds exactly `max(|dr|, |dc|) + 1` pixels.

### 12.2 `line_nd`: sample, then round each axis

`draw/draw_nd.py`.  It computes `npoints = ceil(max(|stop - start|))`, samples
the segment at that many equally spaced parameters with `np.linspace`, then
rounds each axis independently.  Rounding is `np.round`, which is half-to-even,
guarded by `_round_safe`: that guard swaps in `np.floor` when the first
coordinate is exactly `.5` and the step is exactly 1, to stop half-to-even
opening a two-pixel gap.  `endpoint` is False by default, so the stop point is
excluded unless asked for.

### 12.3 Measured differences

| | `line` | `line_nd` |
| --- | --- | --- |
| Dimensions | 2 | N |
| Input | integer only; `TypeError` on float | float or integer |
| Stop point | always included | excluded unless `endpoint=True` |
| Point count | `max(abs) + 1` | `ceil(max(abs))`, plus 1 with `endpoint` |
| Arithmetic | exact integer error term | float sampling, then per-axis rounding |
| Connectivity | diagonal-connected | diagonal-connected — the same |
| Exact half-way tie | always steps early | half-to-even, so it depends on the parity of the coordinate |
| Translation invariant | **yes** (0 failures in 8019 cases) | **no** (5184 failures) |
| Reversal symmetric | **no** (2000 failures) | **yes** (0 failures) |
| Agreement, integer ends, `endpoint=True` | 79% identical, **21% different** | |

The disagreement is entirely about where the minor axis steps when the true
line passes exactly between two pixels:

    line(0, 0, 1, 4)                             rows [0, 0, 1, 1, 1]
    line_nd((0, 0), (1, 4), endpoint=True)       rows [0, 0, 0, 1, 1]

At column 2 the exact row is 0.5.  Bresenham's `d >= 0` test steps early.
`np.round` sends 0.5 to 0, so `line_nd` steps late.  Move the same line down
one row and `line_nd` reverses itself, because `np.round` sends 1.5 to 2.  That
parity dependence is why it is not translation-invariant.

### 12.4 What to do

Neither function is a generalisation of the other, and neither dominates:
`line` has the symmetry you want when translating a shape, `line_nd` the
symmetry you want when drawing a path in either direction.

Keep both in this port.  Give each a "See Also" that states the difference in
one line, so the identical signatures do not mislead.  Merging them, or making
`line_nd` translation-invariant by replacing `_round_safe` with a rounding rule
that does not depend on parity, are separate proposals; the first changes
output for a fifth of all lines and the second for the tie cases.


## 13. How our line drawing compares to other libraries

Measured against Pillow 12.2.0 and OpenCV 5.0.0, over all 6561 integer
endpoint pairs in a 9x9 box, with `endpoint=True` so the point counts match.

### 13.1 Agreement

| | sk.line | sk.line_nd | opencv 8-conn | opencv 4-conn | pillow |
| --- | --- | --- | --- | --- | --- |
| **sk.line** | 100% | 80.6% | 84.8% | 21.0% | **100%** |
| **sk.line_nd** | 80.6% | 100% | 80.6% | 21.0% | 80.6% |
| **opencv 8-conn** | 84.8% | 80.6% | 100% | 21.0% | 84.8% |
| **pillow** | **100%** | 80.6% | 84.8% | 21.0% | 100% |

`draw.line` is pixel-identical to Pillow's `ImageDraw.line`.  Both are textbook
Bresenham.  OpenCV's 8-connected line equals our Bresenham run in **one of the
two endpoint orders** in 100% of cases, so it is the same rasteriser with the
endpoint order normalised first.  OpenCV's 4-connected line is a different
connectivity, which we do not offer at all.

### 13.2 Symmetries

| | Reversal symmetric | Translation invariant |
| --- | --- | --- |
| `sk.line` | 69.5% | **100%** |
| `pillow` | 69.5% | **100%** |
| `sk.line_nd` | **100%** | 69.5% |
| `opencv` (both connectivities) | **100%** | **100%** |

Our two functions each hold one property and lose the other.  OpenCV holds
both.  Both properties are attainable together, so neither of our functions is
at a local optimum.

### 13.3 Two changes this suggests

**Change 1: replace half-to-even rounding in `line_nd`.  This is a defect.**

`line_nd` rounds with `np.round`, which is half-to-even, so at an exact
half-way tie the pixel chosen depends on the parity of the absolute
coordinate.  Drawing the same shape at row 0 and at row 1 gives different
shapes.  No comparator library behaves this way, and the existing
`_round_safe` guard is itself an admission that the rule misbehaves — it
patches one case rather than fixing the rule.

Rounding half up, `np.floor(x + 0.5)`, is translation-invariant by
construction, since `floor(x + t + 0.5) == floor(x + 0.5) + t` for integer `t`.
Measured: it gives **100% on both symmetries** and removes the need for
`_round_safe`.  It changes output for **19.4%** of endpoint pairs.

**Change 2: normalise the endpoint order in `line`.**

Sorting the two endpoints before rasterising gives **100% on both
symmetries**, matching OpenCV's guarantees.  It changes output for **15.2%** of
endpoint pairs.

The case is weaker than for change 1: current behaviour is textbook Bresenham
and matches Pillow exactly, so it is defensible as it stands.  Against that, a
drawing function whose output depends on which end you named first is
surprising, the fix is one line, and skimage2 is the place to take it.  Note
that any rule that depends only on the unordered pair works; OpenCV's own
choice differs from `sorted()` in 15.2% of cases, and matching OpenCV
bit-for-bit is not a requirement.

`line_aa` should get the same treatment as `line`, for consistency.  Not
measured here.

### 13.4 What not to do

Do not try to make `line` and `line_nd` produce the same pixels.  With both
changes applied they agree on 92.4% of pairs, up from 80.6%, and the residual
7.6% is inherent: an exact integer error term and a sampled-then-rounded line
choose different pixels at ties.  They are different algorithms with different
guarantees, and Section 12.4 already recommends keeping both.

### 13.5 Staging

Both changes alter results, so by rule 4 of Section 6 they are Group B and must
not ride along with the tuple-signature work of Stage 1.1.  Add them as
Stage 3.8 and 3.9.  Each needs a migration guide entry with a worked example,
because a user who compares old and new output will see a fifth of their lines
move by one pixel.

A separate proposal, out of scope here: OpenCV offers a 4-connected line and we
offer none.  A `connectivity` argument on `line` and `line_nd` would close a
real gap.

### 13.6 Limits of this comparison

Integer endpoints only, lines no longer than about 12 pixels, in a single 9x9
box of start points, so translation invariance was tested against one shift of
three pixels rather than exhaustively.  Anti-aliased variants (`line_aa`,
OpenCV `LINE_AA`, Pillow's) were not compared.  Pillow 12.2.0 and OpenCV
5.0.0; both algorithms are long-stable, but the versions are recorded in case
that changes.


## 14. Shimming `order='xy'` for `structure_tensor` and `hessian_matrix`

For a 2-D image, `order='xy'` reverses the axis order before the element list
is built, so the result is the reverse of the `order='rc'` result — in exact
arithmetic.  In floating point the two functions behave differently, and only
one of them can be shimmed by reversing the list.

### 14.1 Measured behaviour of a reversal shim

Sixty random asymmetric images, shapes 12x12 to 40x40, sigma in [0.3, 3.0],
comparing today's `order='xy'` output against `order='rc'` reversed:

| Function | Images that differ | Worst absolute difference |
| --- | --- | --- |
| `structure_tensor` | 0 / 60 | 0 |
| `hessian_matrix`, `use_gaussian_derivatives=False` | 15 / 60 | 1.1e-16 |
| `hessian_matrix`, `use_gaussian_derivatives=True` | **60 / 60** | **4.9e-02** |

`structure_tensor` is exact, and provably so: the only element that could
differ is the mixed one, `gaussian(d1 * d0)` against `gaussian(d0 * d1)`, and
IEEE multiplication is commutative, so `gaussian` receives bit-identical
input.

The `np.gradient` path differs only by rounding.  `np.gradient` along axis 0
then axis 1 computes `((a - b) - (c - d)) / 4` where the other order computes
`((a - c) - (b - d)) / 4`; equal in exact arithmetic, one ulp apart in
floating point.

The Gaussian-derivative path is different in kind.

### 14.2 A bug, not rounding

`_hessian_matrix_with_gaussian` builds the mixed element by two successive
`ndi.gaussian_filter` calls.  The cause is **not** that a 2-D Gaussian is
implemented as two 1-D passes.  Measured, a single `gaussian_filter` call is
bit-identical to padding once and cropping, for a plain Gaussian and for a
first-derivative Gaussian, in every boundary mode:

| Composition | Error against pad-once |
| --- | --- |
| one pass on axis 0 | 0 |
| axis 0, then axis 1 (different axes) | 0 |
| axis 0, axis 1, axis 0 (axis 0 twice) | 3.0e-02 |
| axis 1 twice | 7.6e-02 |

The rule is that **a boundary extension along one axis commutes exactly with
filtering along a different axis, but not with filtering along the same axis.**
A single `gaussian_filter` call touches each axis once, so it is exact.
`_hessian_matrix_with_gaussian` composes two calls, so every axis is filtered
twice, and the second call re-extends an array that the first has already
smoothed along that axis.

`mode` says how to continue the **image**.  Applying the same rule to a
smoothed, differentiated intermediate does not give that intermediate's true
continuation, and differentiating along axis 0 then axis 1 re-extends a
different array from the reverse order.  So the two do not agree at the
border.

Measured on a random 50x60 image, difference between `order='xy'` and
`order='rc'` reversed, for the mixed element:

| sigma | Max absolute | Relative |
| --- | --- | --- |
| 0.3 | 3.1e-12 | 1.7e-05 |
| 0.5 | 6.3e-04 | 2.1e-02 |
| 1.0 | 4.6e-02 | **3.6e-01** |
| 1.5 | 2.1e-02 | **5.0e-01** |
| 3.0 | 3.1e-03 | 3.5e-01 |

Restricting to pixels more than one kernel radius from the edge drops the
difference to 4e-17.  So the disagreement is entirely a boundary effect, it
reaches half the signal, and it means that **today `hessian_matrix` returns a
different mixed second partial depending on which `order` you ask for**.  That
is a defect in released scikit-image, independent of this port.  File it.

Removing `order=` makes the inconsistency unreachable, which is a small extra
reason to do it.  It does not remove the border error itself.

**A fix for the mixed element.**  It can be computed in a single call, with the
full sigma and `order=[1, 1]`, which touches each axis once and so is exact:

| Mixed element | Max error against pad-once | Interior |
| --- | --- | --- |
| two composed calls (current) | 2.1e-02 | 0 |
| one call, `order=[1, 1]` | 2.1e-06 | 1.7e-06 |

The border error falls by four orders of magnitude, and the ordering ambiguity
cannot arise at all, because `order=[1, 1]` has no left or right to choose.
The residual 1e-06 everywhere is the difference between two Gaussians of width
`sigma / sqrt(2)` composed and one of width `sigma` — the same operator,
different sampled kernels.

The code comment that justifies the two-pass scheme concerns scipy's
**second**-order Gaussian derivatives, which is the diagonal elements.  The
mixed element uses only a first derivative along each axis, so a single call
does not reintroduce that problem.  The diagonals still need two passes along
one axis and keep a border deviation, but they have no ordering ambiguity and
so do not break the invariant.

### 14.3 The shims

`structure_tensor` — reverse the list.  Exact.

```python
def structure_tensor(image, sigma=1, mode='constant', cval=0, order='rc'):
    _check_order(order, np.asarray(image).ndim)   # shim owns the guards now
    elems = ski2_structure_tensor(image, sigma=sigma, mode=mode, cval=cval)
    return elems[::-1] if order == 'xy' else elems
```

`hessian_matrix` — run the whole computation in the transposed frame, so that
padding happens in the same frame as before, then transpose back.

```python
def hessian_matrix(image, sigma=1, mode='constant', cval=0, order='rc',
                   use_gaussian_derivatives=None):
    image = np.asarray(image)
    _check_order(order, image.ndim)
    if order == 'rc':
        return ski2_hessian_matrix(image, sigma=sigma, mode=mode, cval=cval,
                                   use_gaussian_derivatives=use_gaussian_derivatives)
    # A sequence sigma is per axis, so it reverses with the axes.
    if not np.isscalar(sigma):
        sigma = tuple(sigma)[::-1]
    elems = ski2_hessian_matrix(image.T, sigma=sigma, mode=mode, cval=cval,
                                use_gaussian_derivatives=use_gaussian_derivatives)
    return [h.T for h in elems]
```

The transposed shim matches today's `order='xy'` to 5e-17 in the interior and,
unlike the reversal, is also right at the border.

**A decision this forces.**  The transposed shim gives the *correct* mixed
partial, which is not what SK1 returns today at the border.  Either accept that
SK1 output changes there and call it a bug fix in the release notes, or keep a
private xy path inside `_skimage2` purely to reproduce the old numbers, as
Section 5.3 does for `warp`.  Recommendation: take the bug fix and say so.  The
old values are wrong by up to half the signal.

### 14.4 Tests the shim needs

**The fixture decides everything.**  The existing
`test_hessian_matrix_order` cannot detect a wrong shim.  It uses a 5x5 array
with one spike at the centre, which is symmetric under transpose, at
`sigma=0.1`, where `truncate=100` drives the kernel to underflow.  It passes
for both `use_gaussian_derivatives` values whatever the shim does.  Every new
test must use a **non-square, asymmetric, seeded random image**, **sigma >= 1**,
and must look at the **border**, not only the interior.

1. **Characterisation.**  Before removing `order=`, freeze today's `order='xy'`
   output for a small parameter matrix into an `.npz` under
   `tests/skimage/feature/`.  After the change the shim must reproduce it.
   This is the only test that pins "SK1 behaviour preserved", because after the
   removal there is no other source of truth.  Compare with
   `assert_allclose(rtol=1e-12)`, never `assert_array_equal`: measured
   differences of 1e-16 arise from `np.gradient` association order and would
   make exact comparison flaky across platforms.  Record the border values of
   the Gaussian path as the *corrected* ones, per the decision in 14.3.
2. **Sequence sigma.**  `sigma=(1.0, 2.5)` on a non-square image, `order='xy'`.
   This is the single test most likely to catch a wrong transposed shim:
   forgetting to reverse `sigma` does not raise, it changes results by 7% to
   17%.
3. **Guards, now owned by the shim.**  `order='xy'` with 3-D input raises
   `ValueError` with the old message; an unrecognised `order` raises
   `ValueError` with the old message.  Both functions.  `_skimage2` no longer
   raises these.
4. **Structure.**  Return type is a `list`; length is `(n**2 + n) / 2`; element
   shapes match the input, so a stray transpose fails loudly on a non-square
   image; `float32` input gives `float32` elements, and the transpose does not
   upcast.
5. **N-D pass-through.**  For 3-D and 4-D input, `order='rc'` and the default
   give identical results, element for element.
6. **Warning.**  `use_gaussian_derivatives=None` still raises exactly one
   `FutureWarning` with the old message, and its `stacklevel` still points at
   the caller.  The shim adds a frame; use the existing
   `_shared/utils.py::_warning_stacklevel` machinery.
7. **Invariance of the consumers.**  `structure_tensor_eigenvalues` and
   `hessian_matrix_eigvals` return the same values for `rc` and `xy` element
   order, because reversing the elements conjugates the tensor by a swap and
   eigenvalues are invariant.  A cheap property test.
8. **Internal callers.**  `corner_harris`, `corner_shi_tomasi`,
   `corner_foerstner`, `shape_index` and `feature/censure.py` all pass
   `order='rc'` today.  Dropping the argument at those call sites is
   mechanical; their existing tests are the net.  Confirm they are untouched.
