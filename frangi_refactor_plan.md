---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.5
kernelspec:
  name: python3
  display_name: Python 3 (ipykernel)
  language: python
---

# Refactoring plan: `frangi`

Companion to `on_frangi.md`, which catalogues the defects, and to
`frangi_testing.md`, which works out the tests and runs them against each
stage. Numbers are from `on_frangi.md` unless marked *(measured for this
plan)*.

Read with `hessian_refactor_plan.md`. One of the six defects below belongs to
`hessian_matrix` and not to `frangi`. Section 6 says what the `hessian_matrix`
work does and does not change here.

## 1. Summary

`frangi` has six defects. Four are in the filter, one is inherited from
`hessian_matrix`, and one is a docstring error.

| | defect | size | fixed in |
| --- | --- | --- | --- |
| D1 | `black_ridges` is inert on an ideal ridge | a wrong-polarity ridge scores 0.8647, the filter's single-scale ceiling, instead of 0 | `frangi` |
| D2 | **the σ² scale normalisation was lost in 0.20.0** | only the smallest σ ever wins; [#7711](https://github.com/scikit-image/scikit-image/issues/7711) | `frangi` |
| D3 | the output depends on the **order** of `sigmas` | worst permutation difference 0.9944 on a `[0, 1]` output | `frangi` |
| D4 | `gamma=None` is a whole-image statistic | brightening one distant pixel changes 95.7% of the output | documentation |
| D5 | a bright rim at the image border | 76.95% of the output range at the edge | `hessian_matrix` |
| D6 | `alpha` is documented and inert in 2-D | exactly 0 difference for any `alpha` | docstring |

**D2 is a regression, not a missing feature.** That changes the plan: it makes
the fix a one-line revert with precedent rather than a new convention to be
argued for, and it removes the exponent question entirely (Section 9).

**The critical ordering constraint is D3 before D2.** D3 is invisible today and
becomes a 0.33 error the moment D2 is fixed. Section 4 measures it.

## 1a. What the paper says

`library/frangi1998multiscale_filter.pdf`. Every stage below is a restoration
of something in it; none of them adds anything the paper does not have.

**The σ² is the paper's own normalisation, at the paper's own default.**
Equation (2) defines differentiation in scale space as

> ∂/∂x L(**x**, s) = s^γ L(**x**) * ∂/∂x G(**x**, s)

and the text under it reads:

> "The parameter γ was introduced by Lindeberg [9] to define a family of
> normalized derivatives. This normalization is particularly important for a
> fair comparison of the response of differential operators at multiple scales.
> **When no scale is preferred γ should be set to unity.**"

The Hessian is two derivatives, so it carries `s^{2γ}`, which at γ = 1 is `s²`.
Stage B is not a new convention: it is equation (2) at the value the paper
names.

**And the paper says what it is for.** Section 3.2:

> "By determining the vesselness of the MRA image at multiple scales we obtain
> separate images depicting vessels of various widths. ... The rightmost image
> shows how we can combine these multiscale measurements by using a scale
> selection procedure (**recall that we work with normalized derivatives**),
> eventually yielding a display of **both the small and large vessels**."

#7711's report — that only the smallest scale survives — is that sentence
negated. The paper's figure 4 is the bar picture of `frangi_testing.md` in
medical form.

**The threshold `c` sits outside the scale loop.** Equation (14) is

> V_o(γ) = max over s_min ≤ s ≤ s_max of V_o(s, γ)

`c` is a constant in equations (13) and (15); `s` is the only thing that varies
under the maximum. Stage A puts `c` where the paper has it. The maximum is also
over a *range* of scales, "chosen so that they will cover the range of vessel
widths" — an unordered set, with no counterpart for D3's dependence on which
element comes first.

**The sign condition is a branch, not a divide.** Equation (15) is literally

> V_o(s) = 0 if λ₂ > 0, else exp(−R_B²/2β²)(1 − exp(−S²/2c²))

and section 2 says "we shall look for structures whose λ₂ and λ₃ are both
simultaneously negative". D1's clipped denominator is an implementation of that
branch that fails when λ₁ is exactly zero; the sign test is the branch.

**Amplitude invariance is a stated property.** Page 133:

> "The two geometric ratios we introduced so far are grey-level invariant
> (i.e., they remain constant under intensity re-scalings)."

The `1e-10` clip breaks a property the paper asserts. That is the strongest
argument for the D1 repair, and it is not an appeal to taste.

**`alpha` is absent from the 2-D formula.** Equation (15) is given separately
from (13) and contains no R_A and no α. scikit-image evaluates the 3-D form at
R_A = ∞ instead, which is numerically the same and is why D6 is a docstring
fix rather than a code one.

**The paper is not satisfied with `c` either.** Immediately after proposing it:

> "The value of the threshold c depends on the grey-scale range of the image
> and half the value of the maximum Hessian norm has proven to work in most
> cases. However, **future research will be directed towards automating this
> threshold selection.** We expect that this threshold can be **fixed for a
> given application** where images are routinely acquired according to a
> standard protocol."

So D4 is faithful to the paper, the paper flags it as unfinished, and ITK's
choice — `c` as a parameter the caller sets — is the one the paper anticipates.
That is the whole of the argument in Section 10, decision 2.

**The paper says nothing about D5.** The boundary rule and the discretisation
of the Gaussian derivatives are `hessian_matrix`'s business, not the paper's.

### γ = 1 against γ = 3/4: not a disagreement about accuracy

Frangi cites Lindeberg for the γ family and sets γ = 1. Lindeberg
(`library/lindeberg1998edge_ridge.pdf`, §5.6) writes:

> "Clearly, 0 < γ < 3/2 is a necessary condition to give a local maximum over
> scales. Moreover, γ = 1 corresponds to t_{Mγ−norm} = 2 t₀. **If we want the
> selected scale level to reflect the width of the ridge** such that
> t_{Mγ−norm} = t₀, then we should select γ = 3/4."

That is conditional, and the condition is a different goal from `frangi`'s.
The two exponents each make something invariant, and they are not the same
something *(measured for this plan, Gaussian ridge, peak of `S·σ^p` over a
fine σ grid)*:

| ridge width | γ = 3/4 peak σ | γ = 3/4 peak value | γ = 1 peak σ | γ = 1 peak value |
| --- | --- | --- | --- | --- |
| 2 | 2.0 | 0.25000 | 2.8 | 0.38487 |
| 3 | 3.0 | 0.20412 | 4.2 | 0.38487 |
| 5 | 5.0 | 0.15811 | 7.0 | 0.38487 |
| 8 | 8.0 | 0.12500 | 11.4 | 0.38489 |
| 12 | 12.0 | 0.10206 | 17.0 | 0.38490 |

γ = 3/4 makes the **selected scale** equal the width, and the peak response
then decays as `w^{-1/2}`. γ = 1 makes the **peak response** independent of the
width, and the selected scale is then `w√2`. Lindeberg's sentence asks for the
first; `frangi` is a detector and wants the second.

The consequence is direct. #7711 is the complaint that wide vessels score less
than narrow ones. γ = 3/4 does not fix that — it only softens it *(measured
for this plan, on the bar picture of `frangi_testing.md`)*:

| exponent | w = 1 | w = 2 | w = 4 | w = 8 | spread |
| --- | --- | --- | --- | --- | --- |
| none, as shipped | 0.8647 | 0.4148 | 0.0535 | 0.0039 | 220x |
| `σ^1.5`, γ = 3/4 | 0.8647 | 0.6558 | 0.4133 | 0.2340 | 3.7x |
| `σ^2`, γ = 1 | 0.8238 | 0.8647 | 0.8646 | 0.8592 | 1.0x |

So γ = 1 is not merely the cited default and the in-repo convention: it is the
exponent that answers the reported defect, and γ = 3/4 is the exponent that
leaves a quarter of it in place. There is no accuracy argument on the other
side to weigh.

Where γ = 3/4 does belong is a width *estimator* — reading a vessel's calibre
off the scale that won. Nothing in `frangi` exposes the winning scale, so the
question does not arise here; it would arise for a future `ridge_width`.

## 2. What is wrong, precisely

### D1 — the clipped denominator

The filter rejects a wrong-polarity ridge by arranging for a division to
explode:

```python
(lambda2,) = np.maximum(eigvals[1:], 1e-10)   # the clip
r_b = abs(lambda1) / lambda2
```

For a bright ridge read as a dark one, `lambda2` is negative and is clipped to
`1e-10`, so `r_b` is meant to be huge and blobness to underflow. It only works
when `lambda1` is far from zero. On an ideal ridge `lambda1` is **exactly**
zero, so `r_b` is 0, blobness is 1, and the wrong-polarity ridge scores the
largest value a single-scale `frangi` can produce.

The threshold is sharp: blobness stays above 0.5 while `|lambda1| < 5.89e-11`,
and above 0.01 while `|lambda1| < 1.52e-10`. On a photograph almost nothing is
that flat — a census over `camera` at five scales found the leak on one pixel
at σ = 9 for the 0.5 threshold and two for 0.01. On synthetic ridges of width
1, 2, 4 and 8 it fires every time, and `black_ridges=True` and `False` return
**identical arrays**.

The paper has an explicit zero branch here. The clipped divide is
scikit-image's implementation of it, not Frangi's derivation.

**`1e-10` is an absolute constant in a filter that is otherwise all ratios.**
With `gamma=None` every term in `frangi` is either a ratio of eigenvalues or is
referred to `s.max()`, so scaling the image should change nothing. The clip
breaks that *(measured for this plan)*:

| image × | max difference from ×1 | as % of the output range |
| --- | --- | --- |
| 1e+2 | 4.4e-16 | 0.00% |
| 1e-2 | 1.7e-06 | 0.00% |
| 1e-4 | 1.3e-02 | 2.30% |
| 1e-5 | 9.7e-02 | 16.67% |
| 1e-6 | 6.0e-01 | 103.20% |

This is the same defect as the `eigval_tol` floor in the proposed `jerman`
filter, described in `jerman_vesselness.md` §3.1. The sign test removes it,
because it removes the clip.

### D2 — the σ² scale normalisation, lost in 0.20.0

This is [#7711](https://github.com/scikit-image/scikit-image/issues/7711), and
the reporter's diagnosis is correct. The history:

| version | what `frangi` did |
| --- | --- |
| 0.19.3 | called `compute_hessian_eigenvalues`, which applied `hessian_elements = [(sigma ** 2) * e for e in hessian_elements]` under the comment `# Correct for scale` — **unconditionally** |
| 0.20.0 | `meijering`, `sato` and `frangi` rewritten to call `hessian_matrix` directly. `sato` kept the σ². `frangi` and `meijering` did not. |
| 0.22 | the now-unused `compute_hessian_eigenvalues` deleted (#7162) |

**Only `frangi` regressed.** `sato` still multiplies by σ² today, in the same
module:

```python
vals = sigma**2 * np.prod(np.maximum(eigvals, 0), 0) ** (1 / len(eigvals))
```

and `meijering` cannot tell the difference, because it divides each scale by
that scale's own maximum, which cancels any per-scale positive factor exactly —
measured at 2.2e-16 on `camera` and 3.3e-16 on `coins` *(measured for this
plan)*.

The consequence is the one the issue names. For a Gaussian ridge of width `w`
at scale σ the Hessian norm falls as `S ∝ w (w² + σ²)^(-3/2)` — correlation
0.999996 against the measured values, worst deviation 0.5%, against 46% to
302% for the plausible alternatives. That is strictly decreasing in σ for every
`w`, so without the σ² the response always peaks at the smallest σ offered,
whatever the structure's width. Reproducing the issue's own picture — dark bars
of four widths, σ chosen from `(1, 2, 3, 4, 6, 8, 10, 12)` *(measured for this
plan)*:

| bar width | ideal σ (`w√2`) | as shipped | with σ² restored |
| --- | --- | --- | --- |
| 1 | 1.41 | 1 | 2 |
| 2 | 2.83 | 1 | 3 |
| 4 | 5.66 | 1 | 6 |
| 8 | 11.31 | 1 | 10 |

On `camera`, of the 10.1% of pixels with `V > 0.01`, σ = 1 wins 100%.

### D3 — the order of `sigmas` changes the answer

`gamma=None` is resolved inside the scale loop, on the first iteration only, so
it is frozen at whatever `sigmas[0]` happens to be:

| `sigmas[0]` | resolved γ |
| --- | --- |
| 1 | 0.184388 — the value the docstring describes |
| 3 | 0.017605 |
| 5 | 0.004892 |
| 7 | 0.002263 |
| 9 | 0.001418 |

A factor of 130 across the default `sigmas`. Six permutations of `(1, 3, 5)`
give three distinct outputs, keyed by `sigmas[0]`, and the total response spans
a factor of 28. Over the corpus:

| image | mean ratio, reversed over ascending | top 1% Jaccard |
| --- | --- | --- |
| `camera` | 41.1 | 0.071 |
| `retina` | 18.5 | 0.042 |
| `coins` | 25.7 | 0.031 |

The second column is the point. This is not a rescaling a downstream threshold
absorbs: the strongest-responding 1% of pixels under the two orderings barely
overlap. `sigmas` is documented as a set of scales and nothing says the order
matters. `sato` and `meijering` show exactly 0 under the same test.

### D4 — `gamma=None` is not local

`s.max()` is a maximum over every pixel, so one bright speck anywhere sets the
contrast reference for the whole image:

| filter | brighten one distant pixel | crop the surroundings |
| --- | --- | --- |
| `meijering` | 85.21% | 6.79% |
| `sato` | 0.00% | 0.00% |
| `frangi`, `gamma=None` | 95.70% | 0.00% |
| `frangi`, `gamma=15` | 0.00% | 0.00% |

Fixing D3 does not fix this; hoisting γ makes it a maximum over more pixels,
not fewer. Recomputing γ per scale is worse in kind: far field 82.55%, and it
imports `meijering`'s defect into `frangi`.

This one is faithful to the paper. Frangi describes `c` as depending on the
image grey-scale range and recommends half the maximum Hessian norm. ITK makes
it a parameter the caller sets and never derives it from the image.

### D5 — the border rim

Against a once-extended reference:

| Hessian | at the edge | beyond 1 px | beyond 3 px | beyond 10 px |
| --- | --- | --- | --- | --- |
| as shipped | 76.95% | 6.74% | 0.35% | 0.00% |
| one-pass, corrected | 0.00% | 0.00% | 0.00% | 0.00% |

Not a `frangi` defect. It is `hessian_matrix` applying the boundary rule twice,
and it disappears with the fix in `hessian_refactor_plan.md`.

### D6 — `alpha` in 2-D

The plate-sensitivity factor `1 - exp(-R_a²/2 alpha²)` reduces to 1 in 2-D
because `R_a` is infinite there. Measured difference against `alpha=0.5`:
exactly 0 in 2-D for `alpha` of 0.1, 2 and 5; 0.28 to 0.54 in 3-D. The
docstring does not say so.

## 3. What is *not* wrong

* The 3-D branch's use of `alpha`. D6 is documentation only.
* `beta` and the blobness term. `R_b` is a ratio, so no scale question touches
  it.
* The eigenvalue ordering and the `np.take_along_axis` sort.
* `gamma` when the caller passes a number. D3 and D4 are both properties of
  `gamma=None`; an explicit γ makes `frangi` order-independent and local, and
  the far-field probe then reads 0.00%.
* `sato`. It kept its σ² and shows 0 under the ordering test.
* `meijering`, as regards D2. Its per-scale normalisation makes σ² inert.

## 4. The two repairs that close #7711

The issue needs one line. That line is unsafe until a second, separate change
has landed, and the second change is a measured no-op. So this is two pull
requests, in this order.

### Stage A — hoist γ out of the scale loop (D3)

Compute the parts at every scale, resolve γ from all of them, then fuse. The
same number of Hessian evaluations as today.

```python
cache = [parts(image, sigma, mode=mode, cval=cval) for sigma in sigmas]
if gamma is None:
    gamma = max(norm.max() for _, norm in cache) / 2 or 1.0
return fuse(cache, gamma).max(0)
```

**This is bit-identical to today's output for any ascending `sigmas`** — not
close, identical, on all three corpus images *(measured for this plan)*:

| image | max difference, frozen γ vs hoisted γ | bit-identical |
| --- | --- | --- |
| `camera` | 0.0 | yes |
| `retina` | 0.0 | yes |
| `coins` | 0.0 | yes |

It is identical because `S` falls with σ, so the maximum over all scales sits
at `min(sigmas)`, which for a sorted list is also `sigmas[0]`. So stage A ships
as a pure refactor for the common case, and as a bug fix for unsorted lists.

It also makes the code match its own docstring, which already promises "half
the maximum Hessian norm".

### Stage B — restore the σ² (D2, #7711)

```python
eigvals = eigvals * sigma**2          # 0.19.3's "# Correct for scale"
```

One line, in the loop, before `S` is formed. It restores 0.19.3, matches `sato`
in the same module, and matches Frangi's paper, ITK, DIPlib and Jerman's
MATLAB. Section 2's bar table is the acceptance test.

### Why B must not precede A

Stage A is a no-op *because* `S` peaks at the smallest σ. The σ² destroys that
property, and then the frozen γ is taken from a scale that is no longer the
maximum. The robust statement is the ordering spread — worst difference over
the six permutations of `(1, 3, 5)` *(measured for this plan, against the
released `hessian_matrix`)*:

| image | as shipped | stage A | **stage B alone** | A + B |
| --- | --- | --- | --- | --- |
| `camera` | 0.9942 | 0.0000 | **0.2938** | 0.0000 |
| `retina` | 0.9891 | 0.0000 | **0.0678** | 0.0000 |
| `coins` | 0.9867 | 0.0000 | **0.0480** | 0.0000 |

Stage A drives it to exactly zero. Stage B on its own brings it back. Every
image shows it, on both Hessians.

Whether the **sorted** case is also wrong is image-dependent, and that is worth
stating separately because it is easy to over-claim. With σ² and the released
Hessian, `coins` moves its largest `S` to σ = 7 and its sorted output differs
from the hoisted one by 0.0719; `camera` and `retina` do not move. With the
one-pass Hessian of `hessian_refactor_plan.md`, `retina` moves to σ = 5 and
differs by 0.3264. So the sorted case breaks on some images, and which ones
depends on the build — which is a reason to fix it, not a reason to argue about
the corpus.

### What stage B costs

By design, everything *(measured for this plan)*:

| image | max difference | mean response ratio |
| --- | --- | --- |
| `camera` | 0.5233 | 3.98x |
| `retina` | 0.8422 | 3.19x |
| `coins` | 0.8106 | 6.28x |

`hessian()` calls `frangi` with an explicit `gamma=15`, so stage A does not
reach it and stage B moves it by 1e-4. No timing change from either stage: the
same Hessian evaluations, one extra multiply.

## 5. The remaining repairs

* **D1** — replace the clipped divide with an explicit sign test, keeping the
  division away from the masked pixels:

  ```python
  right_sign = lambda2 > 0
  r_b = np.abs(lambda1) / np.where(right_sign, lambda2, 1.0)
  blobness = np.where(right_sign, np.exp(-r_b**2 / (2 * beta**2)), 0.0)
  ```

  This is what ITK's `HessianToObjectnessMeasureImageFilter` does, and what the
  proposed `jerman` filter in
  [#8074](https://github.com/scikit-image/scikit-image/pull/8074) does. It also
  removes the amplitude dependence measured in Section 2. Independent of
  stages A and B; changes photographs by about 3e-10 and synthetic ridges
  completely.
* **D4** — no code repair. Document that `gamma=None` reads the whole image,
  and recommend an explicit `gamma` where that matters.
* **D6** — one docstring sentence.

## 6. What not to do

* **Do not restore the σ² without hoisting γ.** Section 4 measures the cost:
  0.33 on `retina`, on a sorted `sigmas`.
* **Do not recompute γ at every scale.** It removes the ordering dependence and
  replaces it with `meijering`'s: far field 82.55%, and the winning σ changes
  for 19.9% of far pixels when one distant pixel is brightened, against 0.0%
  for the hoisted γ.
* **Do not sort `sigmas` inside the function.** That hides D3 rather than
  fixing it, and silently reorders an argument the caller supplied.
* **Do not add σ² to `meijering` "for consistency".** It is exactly inert
  there, so it would be a change that reads as meaningful and is not. If
  `meijering`'s normalisation is revisited, revisit this with it.
* **Do not read D1 as harmless because photographs are barely affected.** It is
  a property of exact ridges, which is what the filter is for.
* **Do not treat D5 as a `frangi` bug.** Fixing it in `hessian_matrix` fixes it
  for `sato`, `meijering` and `hessian` at the same time.

## 7. Relationship to the `hessian_matrix` work

The one-pass moment-corrected `hessian_matrix` moves two things `frangi`
depends on: the border values, and the gain below σ of about 1. Measured
against both Hessians, on the same `frangi` code:

| defect | shipped Hessian | one-pass Hessian | verdict |
| --- | --- | --- | --- |
| D1, wrong-polarity ridge | 0.864665 | 0.864665 | unchanged |
| D1, `lambda1` at a ridge centre | exactly 0 | exactly 0 | unchanged |
| D3, worst permutation difference | 0.9942 | 0.9944 | unchanged |
| D5, border rim | 76.95% | 0.00% | fixed there |

**D1, D3, D4 and D6 are independent of it.** So are stages A and B: the σ²
restore is a revert to 0.19.3 behaviour, and the default `sigmas=range(1, 10,
2)` never goes below σ = 1.

What should wait is the *claim*. A scale-selecting `frangi` on the shipped
Hessian selects wrongly below σ = 1, because that operator reports 8% of a
quadratic's curvature at σ = 0.5 and 0.2% at σ = 0.4. Choosing σ from
`(0.3, 0.4, 0.5, 0.7, 1.0, 1.4, 2.0, 3.0, 4.0, 6.0, 8.0)` with σ²
*(measured for this plan)*:

| ridge width | correct σ (`w√2`) | shipped Hessian | one-pass Hessian |
| --- | --- | --- | --- |
| 0.5 | 0.71 | 1.0 | 0.7 |
| 0.8 | 1.13 | 1.4 | 1.0 |
| 1.2 | 1.70 | 2.0 | 2.0 |
| 4.0 | 5.66 | 6.0 | 6.0 |

They agree from width 1.2 up. So ship stages A and B when they are ready, and
say in the release note that scale selection is reliable for σ ≥ 1 until the
`hessian_matrix` fix lands. Do not hold a two-year-old regression behind a
larger refactor.

### One collision to expect

The `hessian_matrix` branch adds a guard to `frangi`:

```python
noise_floor = 100 * np.finfo(image.dtype).eps * np.abs(image).max()
...
if gamma is None:
    gamma = s.max() / 2
    if gamma <= noise_floor:      # was: if gamma == 0
        gamma = 1
```

It is needed because the corrected kernel returns about 1e-16 on a flat image
where the two-pass scheme returned exactly 0, and the `== 0` test relied on
that exact zero. Those are the lines stage A moves. The guard survives the
hoist unchanged, applied to the resolved γ; whichever lands first, the other is
rebased onto it.

**No eigenvalue-level repair removes the need for that guard.** D1's sign test
does not — flat-image maximum *(measured for this plan)*:

| constant image | as shipped | D1 sign test | `noise_floor` guard |
| --- | --- | --- | --- |
| 0.4 | 0.864665 | 0.000000 | 0.000000 |
| 1.0 | 0.864665 | 0.117020 | 0.000000 |
| 1/3 | 0.864665 | 0.117020 | 0.000000 |
| π | 0.864665 | 0.117020 | 0.000000 |

The `0.4` row is a coincidence — it is the one constant tried where the
rounding noise lands with `lambda2 <= 0` everywhere, so the sign test rejects
it. Elsewhere D1 takes 0.8647 down to 0.1170, which is `exp(-2) × (1 - exp(-2))`
and still not zero.

The reason is structural. With `gamma=None`, γ is derived from `S`, so
`1 - exp(-S²/2γ²)` is exactly `1 - e⁻²` at the maximum pixel **whatever the
magnitude of `S`**. The structuredness term is scale-free by construction and
cannot distinguish a faint image from rounding noise. The only absolute
reference available is the image's own magnitude, which is what the guard uses.
The one change that does dissolve the guard is removing `gamma=None` — see
Section 9.

## 8. Stages

Each is one pull request.

1. **Characterisation tests only, no behaviour change.** Freeze the current
   output for a parameter matrix. Include a reversed `sigmas` case, so the
   frozen values *record* D3 rather than hide it, with a comment saying the
   values are wrong and why.
2. **Stage A — hoist γ (D3).** Bit-identical for ascending `sigmas`. Release-note
   as a bug fix: unsorted `sigmas` now gives the same answer as sorted.
3. **Stage B — restore σ² (D2).** Closes #7711. An announced behaviour change,
   pointing at 0.19.3 and at `sato`.
4. **D1 — explicit sign test; D6 — docstring.** Independent of 2 and 3; may be
   done in parallel or first.
5. **D4 — documentation.** Say that `gamma=None` reads the whole image, that it
   is Frangi's published heuristic, and that an explicit `gamma` is the local
   alternative. Cross-reference `meijering`'s per-scale normalisation as the
   thing this deliberately is *not*.

Only one ordering is mandatory: **2 before 3**.

## 9. Tests

The existing tests cannot detect any of D1 to D4. `test_2d_null_matrix` and
`test_3d_null_matrix` use constant images, where every quantity is zero;
`test_3d_cropped_camera_image` compares `frangi` against itself.

1. **Order invariance** (stage 2). `frangi(image, sigmas=s)` equals
   `frangi(image, sigmas=tuple(reversed(s)))` exactly, for a non-square seeded
   random image. Fails today by 0.9944. This is the test that would have caught
   D3.
2. **Scale selection** (stage 3). A dark Gaussian bar of width `w` scores
   highest at σ nearest `w√2`, over a σ grid that brackets it. Section 2's
   table is the fixture. Keep the grid at σ ≥ 1 until the `hessian_matrix` fix
   lands, then extend it down and assert the small-`w` rows of Section 7.
3. **σ² parity with `sato`** (stage 3). Both filters normalise the same way.
   A regression test that they agree about which σ wins on the same bar is
   cheap and pins the module's one convention.
4. **Polarity** (stage 4). A bright Gaussian ridge with `black_ridges=True`
   scores 0 at its centre, and the same ridge with `black_ridges=False` does
   not. Fails today: both return 0.8647. Use an exact `exp(-d²/2w²)` profile,
   because D1 is triggered by `lambda1` being exactly zero and a rasterised
   ridge does not have that.
5. **Amplitude invariance** (stage 4). `frangi(image * a)` equals
   `frangi(image)` for `a` from 1e-6 to 1e+4. Fails today by 17% at ×1e-5 and by more than the whole output range at ×1e-6.
6. **γ matches the docstring** (stage 2). With `gamma=None`, the resolved value
   equals half the largest Hessian norm over **all** scales.
7. **`alpha` in 2-D** (stage 4). Assert what the docstring will then say:
   `alpha` changes nothing in 2-D and something in 3-D.
8. **Locality, as a documented limitation** (stage 5). With an explicit
   `gamma`, brightening a pixel more than one kernel radius away changes
   nothing; with `gamma=None` it changes a great deal. Assert both, so the
   documented behaviour is pinned rather than merely described.

Every one needs a **non-square, asymmetric** image and σ ≥ 1, for the reasons
`hessian_refactor_plan.md` Section 7 gives.

## 9a. Optional regression testing against DIPlib

`on_frangi.md` §9.1 cross-validates the repaired filter against DIPlib and ITK
by hand.  This section is the plan for making the DIPlib half of that a test
that runs in CI.  All measurements are *(measured for this plan)*, on the PR
build in the `frangi-fixes` worktree, `camera` decimated to 160².

DIPlib rather than ITK: it is 21.6 MB installed against ITK's 327 MB, its
largest wheel is 9 MB, its licence is Apache-2.0, and its API needs no
workaround for an output type that silently truncates to `int16`.

### The equivalence to assert

`dip.FrangiVesselness` is single-scale, takes `c` as a parameter, and does
**not** normalise across scale.  `frangi` multiplies the eigenvalues by
`sigma ** 2`, which cancels out of the blobness (a ratio) and survives only in
`S`.  So the matched pair is

```python
frangi(image, sigmas=[s], beta=b, gamma=c, black_ridges=False)
dip.FrangiVesselness(image, sigmas=[s], parameters=[b, c / s**2],
                     polarity='white')
```

In 3-D `parameters` takes three elements, `[alpha, beta, c]`; in 2-D it takes
**two**, dropping `alpha` exactly as eq. (15) drops the plate factor.  That is
worth a test on its own account, as an outside confirmation of D6.

### Three constraints the test must respect

**The border cannot be matched, so crop.**  None of DIPlib's boundary
conditions reproduces `scipy`'s `'reflect'`.  Whole-frame against first-six-rows
disagreement, σ = 3:

| `boundaryCondition` | whole frame | first 6 rows |
| --- | --- | --- |
| default / `mirror` | 0.512 | 0.355 |
| `periodic` | 0.565 | 0.557 |
| `zero order` | 0.644 | 0.644 |
| `first order` | 0.972 | 0.972 |

Crop at least the larger of the two kernel reaches.  A 40-pixel margin covers
σ ≤ 4 for `scipy` at `truncate=8` and any σ for DIPlib at its truncation of 3.

**Stay at σ ≥ 2.**  DIPlib truncates its Gaussian at 3σ where `scipy` uses 8σ,
and `FrangiVesselness` exposes no truncation knob (only `dip.Hessian` does).
The Hessians therefore differ by:

| σ | Hessian disagreement |
| --- | --- |
| 1 | **22.0%** |
| 2 | 0.40% |
| 3 | 0.61% |

σ = 1 is unusable, and the cause is DIPlib's truncation, not our code.

**Assert on a percentile, not a maximum.**  The disagreement is a handful of
pixels where an eigenvalue sign is near-degenerate; the median is zero.  A
maximum is therefore unstable — 0.4% at σ = 2 but 7.9% at σ = 6 — while the
fraction of agreeing pixels is flat:

| σ | repaired | with `sigma ** 2` removed |
| --- | --- | --- |
| 2 | 100.00% | 69.58% |
| 3 | 100.00% | 68.84% |
| 4 | 100.00% | 69.27% |
| 6 | 99.97% | 67.92% |
| 8 | 99.80% | 66.36% |

"at least 99.5% of interior pixels agree to within 1% of the range" passes with
room at every σ from 2 to 8 and fails by thirty points on the regression.

### What it catches, and what it does not

| defect | caught on a photograph? |
| --- | --- |
| D2, missing `sigma ** 2` | **yes** — 78% to 99.6% disagreement against 0.4% to 7.9% |
| D1, clipped divide | **no** — identical to the repaired filter, 0.4% / 0.9% / 7.9% |
| D3, `sigmas` ordering | no — single-scale comparison cannot see it |

D1 needs its own fixture, and there it is decisive.  On an ideal bright ridge
searched for dark ridges, at σ = 3:

| | ours | DIPlib |
| --- | --- | --- |
| sign test | 0.000000 | 0.000000 |
| clipped divide | **1.000000** | 0.000000 |

So the DIPlib suite is two tests, not one: an interior agreement test on a
photograph for D2, and a polarity test on an exact ridge for D1.  D3 stays with
the in-repo tests, which already cover it.

### The tests

1. **`test_frangi_matches_diplib`** — parametrised over σ in (2, 3, 4).  Build
   the matched pair above, crop 40 pixels, assert at least 99.5% of pixels
   agree to within 1% of the range.
2. **`test_frangi_matches_diplib_polarity`** — an exact `exp(-d**2/2w**2)`
   ridge, wrong polarity, assert both return 0 at its centre.  Use an exact
   profile: §9.3 of `frangi_testing.md` shows a rasterised turned bar hides
   this defect.
3. **`test_frangi_matches_diplib_3d`** — a 3-D volume, `parameters` of
   `[alpha, beta, c]`, same percentile assertion.  This is the only test in the
   repo that would exercise the plate factor against an outside reference.

### Packaging

The repo already has the pattern, in `tests/skimage2/restoration/test_j_invariant.py`:

```python
DIPLIB_NOT_INSTALLED = importlib.util.find_spec("diplib") is None
xfail_without_diplib = pytest.mark.xfail(
    DIPLIB_NOT_INSTALLED, reason="optional dependency DIPlib is not installed"
)
```

Add `diplib` to `requirements/optional.txt` beside `PyWavelets`.

**Two coverage gaps to accept knowingly.**  DIPlib 3.6.1 publishes wheels for
`macosx_14_0_arm64`, `macosx_15_0_x86_64`, `manylinux2014_x86_64` and
`win_amd64`, with Python tags cp311 to cp314.  This repo requires Python ≥ 3.12
and runs CI on `ubuntu-24.04-arm`.  So the tests would xfail on the arm Linux
runner, and on 3.15 once that is in the matrix.  That is tolerable behind
`xfail`, but it means the comparison runs on a subset of CI and cannot be
relied on as the only guard for anything.

There is also an import-order defect worth a comment in the test file: `diplib`
registers an IPython input hook on import and fails if IPython is already
imported.  Importing `IPython.terminal.pt_inputhooks` first is the shim; under
plain `pytest` it does not arise.

### What not to do

* **Do not assert a maximum difference.**  It is 0.4% at σ = 2 and 7.9% at
  σ = 6 for a correct implementation.
* **Do not test at σ = 1.**  22% of the disagreement there is DIPlib's
  truncation.
* **Do not compare whole frames.**  No boundary condition matches; the border
  disagrees by half the range whatever the implementation does.
* **Do not treat this as a replacement for the in-repo tests.**  It cannot see
  D3 at all, does not see D1 on natural images, and xfails on part of CI.  It
  is corroboration by an outside implementation, not a substitute for testing
  the properties directly.
* **Do not use SimpleITK for this.**  `ObjectnessMeasure` has no `sigma`, does
  not smooth internally, and rejects a Hessian built by hand, so the scale
  cannot be controlled.  Pre-smoothing and letting it difference internally
  disagrees by about 10%.  It is also 191 MB.

## 10. Open decisions

1. ~~The exponent for D2.~~ **Resolved by the paper and by the history.**
   Equation (2) with "γ should be set to unity" gives `s^{2γ} = s²`; 0.19.3
   used σ², `sato` uses σ² today, and ITK, DIPlib and Jerman's MATLAB agree.
   Section 1a shows that Lindeberg's γ = 3/4 is a conditional recommendation
   for a different goal, and that it would leave part of #7711 unfixed. The decision does **not** need co-deciding with
   `meijering`, because σ² is inert there.
2. **`gamma=None` at all.** ITK requires the caller to supply it. Keeping the
   heuristic is faithful to the paper and is what users have; removing it would
   make the filter local and order-independent by construction, would make D3
   and D4 both disappear, and would take the `noise_floor` guard of Section 7
   with them. Recommendation: keep it, document it, and do not pretend it is
   local.
3. ~~SK1 or skimage2?~~ **Decided: both.** `skimage.filters.frangi` is
   `_skimage2`'s, so it inherits, and it is left to inherit rather than frozen
   behind a migration wrapper. The argument is that none of these is a new
   behaviour: D2 is a revert to 0.19.3 and to `sato`, D3 is a bug fix by any
   reading, and D1 restores the grey-level invariance the paper states. That
   is the opposite of the call made for `hessian_matrix`, where the one-pass
   kernel *is* a new behaviour and `skimage` is frozen against it — the two
   cases differ in kind, not in appetite for churn.

   Consequence for the labels: the change ships in the 1.x line, so the
   ``versionchanged`` entries read 0.27 and not 2.0, matching the 0.20 entry
   already in that docstring.
4. **Say so on #7711 first.** The issue has a correct diagnosis and a wrong
   implicit assumption — that the fix is one line. Post the stage A / stage B
   split and the `retina` measurement, so the ordering constraint is on the
   record before anyone sends the one-line patch.
