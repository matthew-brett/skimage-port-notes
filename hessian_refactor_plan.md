# Refactoring plan: `structure_tensor` and `hessian_matrix`

Companion to `on_hessian.md`, which shows the measurements and the pictures.
Separate from `coordinate_port_plan.md` because this is a correctness fix, not
a coordinate change — but Section 8 explains where the two meet.

All measurements are reproducible from the notebook.

## 1. Summary

`structure_tensor` is correct and needs only the `order` removal.

`hessian_matrix` has a border defect. Values within one Gaussian-kernel radius
of the image edge are wrong by about a third, in **all three** elements. The
defect reaches `frangi`, `sato`, `meijering` and `hessian`, and it is the cause
of the `order='xy'` inconsistency reported in `coordinate_port_plan.md`
Section 14.

The path affected is `use_gaussian_derivatives=True`, which the code's own
`FutureWarning` says will become the default.

## 2. What is wrong, precisely

`mode` documents how to continue the **image** past its edge.
`_hessian_matrix_with_gaussian` composes two `ndi.gaussian_filter` calls, so
each axis is filtered twice, and the second call re-applies the image's
extension rule to an intermediate that the first call has already smoothed
along that axis.

The governing rule, measured:

| Composition | Error against padding once |
| --- | --- |
| one pass on axis 0 | 0 |
| axis 0, then axis 1 (different axes) | 0 |
| axis 0, axis 1, axis 0 (axis 0 twice) | 3.5e-02 |
| axis 1 twice (same axis twice) | 7.8e-02 |

A boundary extension along one axis commutes exactly with filtering along a
*different* axis, and not with filtering along the *same* axis. A single
`gaussian_filter` call touches each axis once and is therefore bit-identical to
the padded 2-D convolution — **separability is not the problem.** Composition
is.

## 3. Size of the defect

Against a reference that pads once by the rule `mode` names, filters with a
wide margin and crops (in which construction both `order` values agree to
1e-17):

| Element | Max absolute | Relative | Interior |
| --- | --- | --- | --- |
| `Hrr` diagonal | 3.2e-02 | 34.0% | 0 |
| `Hrc` mixed | 1.7e-02 | 31.5% | 0 |
| `Hcc` diagonal | 2.8e-02 | 36.7% | 0 |

The interior is exact. The error stops exactly one kernel radius from the edge.

End to end, on `data.camera()` downsampled to 256x256:

| Filter | Max difference | Share of range | 30 px in |
| --- | --- | --- | --- |
| `frangi` | 4.8e-02 | 8.2% | 2.0e-08 |
| `sato` | 4.5e-02 | 18.6% | 2.4e-04 |
| `meijering` | 2.5e-01 | 25.0% | 4.0e-02 |
| `hessian` | 1.0e+00 | **100%** | 3.0e-12 |

`hessian` inverts completely at the border. `meijering` also moves in the
interior, because it normalises by a global maximum that the border artefact
distorts — so for that one filter the defect is not confined to the edge.

## 4. What is *not* wrong

* `structure_tensor`, in any respect. Its `order` parameter is a pure
  relabelling: `xy` is bit-identical to `rc` reversed, in 60 of 60 fuzzed
  images, and provably so, because the only element that could differ is
  `gaussian(d1 * d0)` against `gaussian(d0 * d1)` and IEEE multiplication
  commutes.
* `hessian_matrix` with `use_gaussian_derivatives=False`. Its `order` values
  agree to 1.1e-16 — `np.gradient` association order, nothing more.
* The separable implementation of the Gaussian.

## 5. Preferred fix

**Pad once inside `_hessian_matrix_with_gaussian`.** Extend the image by the
rule `mode` names, run the existing two-pass scheme on the extended array, and
crop. Every element then matches the reference exactly, and the `order`
question disappears because both orders give the same answer.

**Fold in the one-call mixed element.** `ndi.gaussian_filter(image, sigma,
order=[1, 1])` touches each axis once, so it is exact by construction, has no
left-or-right to choose, and costs one filter call less:

| Mixed element | Max error against reference |
| --- | --- |
| two composed calls (current) | 1.7e-02 |
| one call, `order=[1, 1]` | 1.9e-06 |

The code comment that justifies the two-pass scheme concerns scipy's
**second**-order Gaussian derivatives, which is the diagonal elements. The
mixed element uses only a first derivative along each axis, so this change does
not reintroduce that problem.

### Cost

Timed on 256x256:

| sigma | Pad width | Current | Padded | Ratio |
| --- | --- | --- | --- | --- |
| 0.5 | 71 | 23.6 ms | 56.8 ms | 2.4x |
| 1.0 | 143 | 80.2 ms | 297.8 ms | 3.7x |
| 1.5 | 17 | 9.2 ms | 8.9 ms | 1.0x |
| 3.0 | 35 | 13.4 ms | 17.6 ms | 1.3x |

Free above `sigma = 1`. Below it the function sets `truncate=100` to fight
aliasing, which makes the pad absurd. **That hack needs its own review**, and
the pad width should be derived from the effective support rather than from
`truncate`. Treat it as a blocking sub-task for the `sigma < 1` case only.

### Rejected alternatives

* **Fix only the mixed element.** Cheap and removes the `order` inconsistency,
  but leaves both diagonals wrong by a third, so the ridge filters keep their
  border artefact. The inconsistency is the symptom; the border error is the
  disease.
* **Document the difference and narrow the test.** Defensible — see
  `coordinate_port_plan.md` Section 14 — but it leaves `hessian` inverting at
  the border and ships a known-wrong number.

## 6. Stages

Each is one pull request.

1. **Characterisation tests only, no behaviour change.** Freeze the current
   output of both functions for a parameter matrix into an `.npz`. Add the
   asymmetric non-square fixtures of Section 7. Nothing changes yet; this is
   the net for everything after.
2. **Mixed element in one call.** Small, self-contained, strictly better, and
   it removes the `order` inconsistency on its own.
3. **Pad once**, for `sigma >= 1` only, with the `sigma < 1` path unchanged and
   a `TODO.txt` entry. Update the frozen values from stage 1 and record the
   change in the release notes.
4. **Revisit `truncate=100`,** then extend padding to `sigma < 1`.
5. **Remove `order=`** from `_skimage2`, per `coordinate_port_plan.md`
   Stage 2.1. After stages 2 and 3 the shim is a plain list reversal for both
   functions, because the two orders now agree exactly — which is much simpler
   than the transposed shim that Section 14.3 of that plan currently
   prescribes.

Stage 5 gets materially easier if stages 2 and 3 land first. That ordering is
the main reason to treat this as its own piece of work rather than folding it
into the coordinate port.

## 7. Tests

**The fixture decides everything.** The existing `test_hessian_matrix_order`
cannot detect any of this: a 5x5 array with one spike at the centre is
symmetric under transpose, and at `sigma=0.1` the `truncate=100` kernel
underflows. It passes whatever the implementation does.

Every new test needs a **non-square, asymmetric, seeded random image**,
**sigma >= 1**, and must look at the **border**.

1. **Reference agreement.** For each element, `hessian_matrix` equals the
   pad-once reference to `rtol=1e-12`, at the border as well as the interior.
   This is the test that would have caught the defect.
2. **Order invariance.** `order='xy'` equals `order='rc'` reversed, exactly,
   for both `use_gaussian_derivatives` values. Currently fails for `True`.
3. **Sequence sigma.** `sigma=(1.0, 2.5)` on a non-square image, both orders.
   Catches axis-order slips that a scalar sigma hides.
4. **Interior unchanged.** Pin that the fix moves nothing more than one kernel
   radius from the edge — this is the promise made to users.
5. **Ridge filters.** `frangi`, `sato`, `meijering`, `hessian` agree with their
   padded-then-cropped equivalents. Note `meijering` needs a looser interior
   tolerance until its global normalisation is looked at separately.
6. **`structure_tensor` unchanged.** Bit-exact against the stage 1 freeze, both
   orders, all dtypes.
7. **Guards and warning.** The `ValueError` for 3-D with `order='xy'`, the
   `ValueError` for an unrecognised order, and exactly one `FutureWarning` for
   `use_gaussian_derivatives=None` with `stacklevel` pointing at the caller.

## 8. Relationship to the coordinate port

This work is independent of the coordinate convention and can proceed in
parallel. Two points of contact:

* `coordinate_port_plan.md` Section 14 prescribes a **transposed** shim for
  `hessian_matrix`, because reversal does not reproduce today's `order='xy'`.
  After stages 2 and 3 here, reversal does reproduce it, and that shim
  collapses to `elems[::-1]` — the same as `structure_tensor`. Doing this work
  first makes the port simpler.
* Section 14 frames the `order` disagreement as the defect. This plan shows it
  is a symptom: the border error affects the diagonals just as much, and they
  have no ordering ambiguity to expose them.

## 9. Open decisions

1. **Is the border change acceptable in `skimage` (v1), or `skimage2` only?**
   The values are wrong by up to a third, but they are the values users have
   today. Recommendation: fix in `skimage2`, release-note it, and leave v1
   alone — consistent with the project's rule that SK1 does not move.
2. **`meijering`'s global normalisation.** It spreads a border artefact across
   the whole image. Out of scope here; worth its own issue.
3. **`truncate=100` for `sigma < 1`.** Blocks padding for small sigma and is
   independently suspicious. Needs its own measurement.
4. **File the defect upstream first?** It exists in released scikit-image, and
   a bug report gives the fix somewhere to point.
