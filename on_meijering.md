---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.1
kernelspec:
  name: python3
  display_name: Python 3 (ipykernel)
  language: python
---

# On meijering's normalisation

`on_hessian.md` finds a border defect in `hessian_matrix` and proposes fixes A
and C. Applied together they take `frangi`, `sato` and `hessian` to exact
agreement with a once-extended reference. `meijering` does not follow: it drops
from 26.3% to 3.3%, and the residue is its *interior* figure, unchanged.

This notebook is about that residue. It is a second defect, in `meijering`
alone, that the border fix exposes rather than causes — and the obvious repair
makes the filter worse, which is why it is worth a notebook rather than a patch.

Everything here assumes the Hessian is already fixed. Coordinates are in array
order throughout.

```{code-cell} ipython3
import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
```

```{code-cell} ipython3
import skimage as ski
from skimage.filters import meijering, sato, frangi, hessian
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
```

```{code-cell} ipython3
# Slots 1 to 3 of the reference categorical palette, validated all-pairs.
C_ONE, C_TWO, C_THREE = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#dedcd5"
SEQ = LinearSegmentedColormap.from_list("seq", ["#f7f7f4", C_ONE])

plt.rcParams.update(
    {"figure.dpi": 110, "font.size": 9, "axes.titlesize": 9,
     "axes.titlecolor": MUTED, "figure.facecolor": "white"}
)


def bare(ax, title=None):
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title:
        ax.set_title(title)
    return ax


def recede(ax, title=None):
    ax.tick_params(labelsize=8, colors=MUTED)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    if title:
        ax.set_title(title)
    return ax


PHOTO = ski.util.img_as_float(ski.data.camera())[::2, ::2]
SIGMAS = (1, 3, 5)
RIDGE = dict(sigmas=SIGMAS, mode="nearest")
```

## 1. One line, and only in `meijering`

`filters/ridges.py` holds four ridge filters that share a shape: sweep `sigmas`,
build the Hessian at each, reduce its eigenvalues to a score, and keep the
pixel-wise maximum across scales. Three of them do exactly that. `meijering`
inserts one more step, at `ridges.py:96`:

```python
vals = np.maximum(vals, 0)
# Normalize to max = 1 (unless everything is already zero).
max_val = vals.max()
if max_val > 0:
    vals /= max_val
filtered_max = np.maximum(filtered_max, vals)
```

Each scale is divided by **its own maximum over the whole image** before the
cross-scale maximum is taken. That is the only per-scale normalisation in the
module: `sato`, `frangi` and `hessian` have nothing like it.

The consequence is that the value at a pixel stops depending only on the
neighbourhood of that pixel.

```{code-cell} ipython3
def far_field_change(f, image, spot=(0, 0), value=10.0, keep=100):
    """How much the response changes far away when one distant pixel changes."""
    edited = image.copy()
    edited[spot] = value
    before, after = f(image, **RIDGE), f(edited, **RIDGE)
    far = (slice(keep, None), slice(keep, None))
    return (np.abs(before[far] - after[far]).max()
            / max(np.abs(before[far]).max(), 1e-12))


print("one pixel brightened at (0, 0); change measured 100 px away")
for name, f in (("meijering", meijering), ("sato", sato), ("frangi", frangi)):
    print(f"   {name:<10}{far_field_change(f, PHOTO):>10.2%}")
```

`sato` is unaffected, as a local filter should be. `meijering` moves by 87%.

`frangi` moves too, and for a different reason worth naming so it is not
mistaken for this one: its `gamma` defaults to `s.max() / 2`, computed per
scale from the Hessian norms of the whole image. That is also a global
statistic, but it does not disturb the comparison *between* scales, which is
what the rest of this notebook is about.

The cleaner probe is cropping, because it changes the image's extent without
touching any shared pixel.

```{code-cell} ipython3
def crop_change(f, image, size=120, margin=20):
    """Same pixels, smaller surroundings: how much does the shared interior move?"""
    whole = f(image, **RIDGE)[:size, :size]
    part = f(image[:size, :size], **RIDGE)
    inner = (slice(margin, -margin),) * 2
    return (np.abs(whole[inner] - part[inner]).max()
            / max(np.abs(whole[inner]).max(), 1e-12))


print("crop to 120x120; change in the shared interior, away from the new border")
for name, f in (("meijering", meijering), ("sato", sato), ("frangi", frangi)):
    print(f"   {name:<10}{crop_change(f, PHOTO):>10.2%}")
```

Only `meijering` moves. That is the 3.3% residue of `on_hessian.md` section 9,
seen without a Hessian anywhere in the argument: a once-extended reference is a
larger image, and a larger image has a different maximum.

## 2. Why the line is there

Delete it and the filter gets worse, which is the reason this is a notebook and
not a one-line patch.

A reimplementation with the normalisation switchable makes the comparison
possible. It reproduces the shipped filter when asked to.

```{code-cell} ipython3
from scipy import linalg


def meijering_core(image, sigmas, how="maxnorm", alpha=None, black_ridges=True,
                   mode="nearest"):
    """`meijering`, transcribed, with the per-scale normalisation switchable.

    how = "maxnorm"  divide each scale by its own global maximum, as shipped
          "none"     no per-scale normalisation
          "gamma"    multiply by sigma**2, the scale-normalised second derivative
    """
    image = image.astype(float, copy=False)
    if not black_ridges:                        # as shipped: negate the image,
        image = -image                          # not the response
    if alpha is None:
        alpha = 1 / (image.ndim + 1)
    mtx = linalg.circulant([1, *[alpha] * (image.ndim - 1)]).astype(image.dtype)

    out = np.zeros_like(image)
    per_scale = {}
    for sigma in sigmas:
        eigs = hessian_matrix_eigvals(
            hessian_matrix(image, sigma, mode=mode, use_gaussian_derivatives=True))
        vals = np.tensordot(mtx, eigs, 1)
        vals = np.take_along_axis(vals, abs(vals).argmax(0)[None], 0).squeeze(0)
        vals = np.maximum(vals, 0)
        if how == "maxnorm":
            peak = vals.max()
            if peak > 0:
                vals = vals / peak
        elif how == "gamma":
            vals = vals * sigma**2
        per_scale[sigma] = vals
        out = np.maximum(out, vals)
    return out, per_scale


for black in (True, False):
    mine, _ = meijering_core(PHOTO, SIGMAS, black_ridges=black)
    theirs = meijering(PHOTO, black_ridges=black, **RIDGE)
    print(f"reimplementation vs shipped, black_ridges={black}: "
          f"max |difference| {np.abs(mine - theirs).max():.2e}")
```

Now a test image with two ridges of very different width, far enough apart not
to interact.

```{code-cell} ipython3
N = 240
rows, cols = np.indices((N, N), dtype=float)
NARROW, WIDE = 1.5, 6.0
ridges = (np.exp(-((cols - 60) ** 2) / (2 * NARROW**2))
          + np.exp(-((cols - 170) ** 2) / (2 * WIDE**2)))
SCAN = (1, 2, 3, 4, 6, 8)

fig, axes = plt.subplots(1, 2, figsize=(8.6, 2.6),
                         gridspec_kw={"width_ratios": [1, 2]})
bare(axes[0], "two ridges, widths 1.5 and 6")
axes[0].imshow(ridges, cmap=SEQ)
axes[1].plot(ridges[N // 2], color=INK, lw=1.6)
recede(axes[1], "profile across the middle row")
axes[1].set_xlabel("column", fontsize=8, color=MUTED)
fig.tight_layout()
```

A multiscale ridge filter has one job on this image beyond finding the ridges:
report both of them, and report each most strongly at the scale that matches
it. That is what the sweep over `sigmas` is *for*.

```{code-cell} ipython3
print(f"{'variant':<10}{'sigma winning at':>20}{'':>8}{'peak response':>22}")
print(f"{'':<10}{'narrow':>12}{'wide':>8}{'':>8}{'narrow':>11}{'wide':>11}")
for how in ("maxnorm", "none", "gamma"):
    out, per_scale = meijering_core(ridges, SCAN, how=how,
                                    black_ridges=False)
    pick = lambda col: max(SCAN, key=lambda s: per_scale[s][N // 2, col])
    narrow_peak = out[N // 2, 50:70].max()
    wide_peak = out[N // 2, 160:180].max()
    print(f"{how:<10}{pick(60):>12}{pick(170):>8}{'':>8}"
          f"{narrow_peak:>11.4f}{wide_peak:>11.4f}")
```

Deleting the line is the "none" row, and it fails the job. Scale 1 wins at both
ridges, and the wide ridge comes back ten times weaker than the narrow one,
because raw Hessian eigenvalues shrink as `sigma` grows: a second derivative of
a smoothed image carries a factor that falls off with scale, so without
compensation the smallest scale always wins.

So the division is not decoration. It is **scale normalisation**, done by an
empirical statistic. It puts every scale on a common footing, which is exactly
what makes the cross-scale maximum meaningful.

```{code-cell} ipython3
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.0), sharey=False)
for ax, how, title in zip(axes, ("maxnorm", "none", "gamma"),
                          ("as shipped: divide by the scale's own max",
                           "line deleted: no normalisation",
                           "multiply by sigma**2")):
    _, per_scale = meijering_core(ridges, SCAN, how=how,
                                  black_ridges=False)
    for sigma, colour in zip((1, 3, 8), (C_ONE, C_TWO, C_THREE)):
        ax.plot(per_scale[sigma][N // 2], color=colour, lw=1.6,
                label=f"sigma = {sigma}")
    for centre in (60, 170):
        ax.axvline(centre, color=GRID, lw=1, zorder=0)
    recede(ax, title)
    ax.set_xlabel("column", fontsize=8, color=MUTED)
axes[0].legend(frameon=False, fontsize=7)
fig.suptitle("response at three scales; the narrow ridge is at 60, the wide at 170",
             y=1.03)
fig.tight_layout()
```

The middle panel is the argument against the naive fix, drawn: with no
normalisation, `sigma = 1` towers over the others everywhere and the wide ridge
barely registers. The left and right panels both put the scales on comparable
footing; they differ in what they use to do it.

## 3. The principled version of the same idea

Scale normalisation is a solved problem, and the solution predates the filter.
Lindeberg's γ-normalised derivative multiplies a derivative of order `n` by
`sigma**(n * gamma)`; for a second derivative with `gamma = 1` that is
`sigma**2`. It compensates for exactly the falloff the middle panel shows, and
it does so from `sigma` alone — no image statistic, so nothing outside the
neighbourhood enters.

Measured, it does the job the empirical version does:

```{code-cell} ipython3
print("does the filter stay local?")
print(f"{'variant':<10}{'one distant pixel':>20}{'crop':>10}")
for how in ("maxnorm", "gamma"):
    variant = (lambda im, _how=how, **kw: meijering_core(im, SIGMAS, how=_how)[0])
    print(f"{how:<10}{far_field_change(variant, PHOTO):>20.2%}"
          f"{crop_change(variant, PHOTO):>10.2%}")
```

`gamma` is local by construction: the same pixels give the same answer whatever
surrounds them. It is not bit-exact under cropping — 0.03% remains, from the
Hessian's own boundary handling at the new edge — but that is three orders of
magnitude below the 16% the empirical version shows, and it is a border effect
rather than a global one.

```{code-cell} ipython3
fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.2))
whole = {how: meijering_core(PHOTO, SIGMAS, how=how)[0]
         for how in ("maxnorm", "gamma")}
part = {how: meijering_core(PHOTO[:120, :120], SIGMAS, how=how)[0]
        for how in ("maxnorm", "gamma")}
for ax, how in zip(axes[:2], ("maxnorm", "gamma")):
    gap = np.abs(whole[how][:120, :120] - part[how])
    im = ax.imshow(gap, cmap=SEQ)
    bare(ax, f"{how}: |whole - cropped|")
    fig.colorbar(im, ax=ax, fraction=0.046)
bare(axes[2], "the crop, for reference")
axes[2].imshow(PHOTO[:120, :120], cmap="gray")
fig.suptitle("cropping the image should not change the pixels that remain", y=1.02)
fig.tight_layout()
```

## 4. Why this is not simply a patch

The obvious conclusion — replace the division with `sigma**2` — is the change
that was already made, questioned, and reversed. The history is worth knowing
before proposing it again.

[#5561, "Meijering scaling issue"](https://github.com/scikit-image/scikit-image/issues/5561),
reported that the implementation of the day applied a `sigma**2` scaling to the
Hessian eigenvalues that the reporter could not reconcile with the paper's
appendix: "this would be correct if the Hessian matrix missed a factor
`sigma**2` also, but I cannot find such a source of error". The filters were
then rewritten in
[#6446](https://github.com/scikit-image/scikit-image/issues/6446), and the
version in the tree today divides by the per-scale maximum instead.

So the two candidates have each been shipped, and each has an objection on the
record:

| | scale normalisation | objection |
| --- | --- | --- |
| before #6446 | `* sigma**2` | not derivable from the paper, per #5561 |
| today | `/ vals.max()` | not local; the value at a pixel depends on the whole image |

The objections are not of the same kind, which is the useful observation. The
first is about whether a particular constant matches a particular paper — a
question with an answer, in the appendix of
[Meijering et al. (2004)](https://doi.org/10.1002/cyto.a.20022), which this
notebook has not consulted. The second is about a property no ridge filter
should lack, and it can be demonstrated without reading anything.

A third possibility should be on the table when someone does open the paper: the
γ for ridge detection is not obviously 1. Lindeberg's own work on ridge
detection argues for `gamma = 3/4`, which would make the factor `sigma**1.5`
rather than `sigma**2`. If the paper does not settle the exponent, that is the
literature to reconcile against, and it may be that #5561's objection was right
about `sigma**2` and still compatible with some other power.

```{code-cell} ipython3
print("which sigma wins at each ridge, for a few exponents")
print(f"{'factor':>12}{'narrow (true 1.5)':>20}{'wide (true 6.0)':>18}")
for power in (0.0, 1.0, 1.5, 2.0, 2.5):
    _, per_scale = meijering_core(ridges, SCAN, how="none",
                                  black_ridges=False)
    scaled = {s: v * s**power for s, v in per_scale.items()}
    pick = lambda col: max(SCAN, key=lambda s: scaled[s][N // 2, col])
    print(f"{'sigma**' + str(power):>12}{pick(60):>20}{pick(170):>18}")
```

The exponent is not a free choice that only affects magnitudes: it decides which
scale is reported for a given feature, so it decides the filter's answer.

On this pair of ridges, `sigma**1.5` — Lindeberg's `gamma = 3/4` for ridge
detection — lands closest to the truth, reporting 2 and 6 for ridges of width
1.5 and 6.0, where `sigma**2` overshoots the wide one at 8. One synthetic image
is not evidence for an exponent, and the scan is here to show that the choice is
consequential rather than to settle it. But it does mean the two candidates on
the record are not the only ones worth putting to the paper.

## 5. What to do

**The defect is real and is worth an issue.** `meijering` is the only ridge
filter in `skimage` whose output at a pixel depends on pixels arbitrarily far
away. Cropping an image changes the answer for the part that remains by 16% on
the `camera` photograph; brightening one corner pixel changes it by 87% a
hundred pixels away. Neither is defensible for a filter, and neither is
documented.

**The fix is not obvious and should not be guessed.** Deleting the line makes
the filter monotonically prefer the smallest `sigma`, which is worse than what
it does now. Replacing it with `sigma**2` restores locality and plausible scale
selection, but that is the code #5561 objected to, and this notebook has not
read the appendix that would settle it.

**Recommended order.** Read the appendix of Meijering et al. and establish what
normalisation the method specifies. Then either implement that, or — if the
paper leaves it open — choose a γ-normalisation on scale-space grounds and say
in the docstring which one and why. Either outcome is local, which is the
property the current code lacks.

**Relation to fixes A and C.** They are independent and both are needed. A and C
correct the Hessian this filter consumes; nothing in them touches the
normalisation, which is why `meijering` alone keeps a residue after they land.
Conversely a normalisation fix does not repair the border. The one place they
meet is `truncate = 100`, and its origin is the same thread as this defect:
in #5561 the reporter concluded that the accuracy gain he had been chasing "is
just down to the use of the large truncation value of 100", and that it is
"relevant to use a (much) larger truncation value once sigma values become
small". `on_hessian.md` Fix C shows that a corrected kernel needs no such
margin, so fixing the kernel retires the hack that this thread introduced.

Measured with scikit-image from this working tree, on the 256x256 `camera`
photograph and a 240x240 synthetic pair of ridges.

```{code-cell} ipython3
print(f"scikit-image {ski.__version__}")
```
