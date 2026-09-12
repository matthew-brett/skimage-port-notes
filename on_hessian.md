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

# On the Hessian and the structure tensor

`skimage.feature.structure_tensor` and `skimage.feature.hessian_matrix` both
take an `order` parameter, documented as choosing between two labellings of the
same tensor. For one of them that is true. For the other it is not, and
following the difference leads to a border defect that reaches a third of the
signal and shows up in `frangi`, `sato`, `meijering` and `hessian`.

This notebook explains what each function computes, why they differ, what the
correct answer is, and what a fix changes.

Coordinates are in array order throughout: the first index runs down the
picture, the second runs right.

```{code-cell} ipython3
import warnings

import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

warnings.simplefilter("ignore")  # the use_gaussian_derivatives FutureWarning
```

```{code-cell} ipython3
import skimage as ski
from skimage.feature import structure_tensor, hessian_matrix
from skimage.filters import frangi, sato, meijering, hessian
```

```{code-cell} ipython3
# Comparator.
import cv2
```

```{code-cell} ipython3
# Slots 1 to 3 of the reference categorical palette, validated all-pairs.
C_ONE = "#2a78d6"
C_TWO = "#eb6834"
C_THREE = "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#dedcd5"

# Sequential: one hue, light to dark. Diverging: two hues, neutral midpoint.
SEQ = LinearSegmentedColormap.from_list("seq", ["#f7f7f4", C_ONE])
DIV = LinearSegmentedColormap.from_list("div", [C_ONE, "#f2f1ec", C_TWO])

plt.rcParams.update(
    {"figure.dpi": 110, "font.size": 9, "axes.titlesize": 9,
     "axes.titlecolor": MUTED, "figure.facecolor": "white"}
)


def show(ax, data, title=None, cmap=SEQ, diverging=False, vmax=None):
    """Draw an array with recessive axes; diverging data is centred on zero."""
    if diverging:
        m = vmax if vmax is not None else np.abs(data).max() or 1.0
        im = ax.imshow(data, cmap=DIV, norm=TwoSlopeNorm(0, -m, m))
    else:
        im = ax.imshow(data, cmap=cmap, vmin=0, vmax=vmax)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title is not None:
        ax.set_title(title)
    return im


rng = np.random.default_rng(7)
IMG = rng.random((60, 70))       # asymmetric and non-square, so axis order shows
SIGMA, MODE = 1.5, "nearest"
```

## 1. What a Hessian is, and why an image needs a scale

The [Hessian matrix](https://en.wikipedia.org/wiki/Hessian_matrix) collects the
second partial derivatives of a function. For a 2-D image it has three distinct
entries, because mixed partials are equal:

```
    H = | Hrr  Hrc |
        | Hrc  Hcc |
```

Its eigenvalues describe the local curvature. Two large negative eigenvalues
mark a bright blob; one large and one near zero mark a ridge. That is the basis
of [ridge detection](https://en.wikipedia.org/wiki/Ridge_detection), which is
what `frangi`, `sato` and `meijering` do, and of the determinant-of-Hessian
[blob detector](https://en.wikipedia.org/wiki/Blob_detection).

An image is a grid of samples, not a differentiable function, and differencing
neighbouring pixels amplifies noise without limit. The standard remedy is to
fix a **scale**: convolve with a Gaussian of width `sigma`, and differentiate
that instead. This is the basis of
[scale space](https://en.wikipedia.org/wiki/Scale_space). So the Hessian of an
image is always the Hessian *at a scale*, and `sigma` is part of the
definition rather than a tuning knob.

```{code-cell} ipython3
# A Gaussian blob, and the three distinct entries of its Hessian.
BLOB_N, BLOB_S = 121, 4.0
centre = BLOB_N // 2
bi, bj = np.indices((BLOB_N, BLOB_N), dtype=float)
radius2 = (bi - centre) ** 2 + (bj - centre) ** 2
blob = np.exp(-radius2 / (2 * BLOB_S**2)) / (2 * np.pi * BLOB_S**2)

entries = hessian_matrix(blob, sigma=2.0, mode="nearest",
                         use_gaussian_derivatives=True)

fig, axes = plt.subplots(1, 4, figsize=(10.0, 2.6))
show(axes[0], blob[30:-30, 30:-30], "the blob")
for ax, name, entry in zip(axes[1:], ("Hrr", "Hrc", "Hcc"), entries):
    show(ax, entry[30:-30, 30:-30], name, diverging=True)
fig.suptitle("a blob and its second derivatives at sigma = 2", y=1.04)
fig.tight_layout()
```

`Hrr` is negative in a band across the middle, where the blob curves downwards
along the row axis, and positive in the two lobes above and below it where that
curvature reverses. `Hcc` is the same pattern turned through a right angle,
because it differentiates along the other axis. `Hrc` is the saddle term: four
lobes of alternating sign, vanishing along both axes.

+++

## 2. The two ways `hessian_matrix` computes it

Convolution and differentiation commute, so for a Gaussian `G` and an image
`f`:

```
    d/dx (G * f)  ==  (dG/dx) * f
```

Either side is a valid route to a smoothed derivative, and
`use_gaussian_derivatives` chooses between them.

**`use_gaussian_derivatives=False`** takes the left-hand side. It smooths with
`gaussian`, then calls `np.gradient` twice. The derivative is an ordinary
finite difference, taken on the blurred image.

**`use_gaussian_derivatives=True`** takes the right-hand side. It convolves
with derivatives of the Gaussian directly, through the `order` argument of
[`scipy.ndimage.gaussian_filter`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html).
It applies two successive **first**-order passes rather than one second-order
pass, and each pass uses `sigma / sqrt(2)`, because two Gaussians of that width
compose to a single Gaussian of width `sigma`.

```{code-cell} ipython3
single = ndi.gaussian_filter(blob, 2.0, mode="nearest")
halved = ndi.gaussian_filter(
    ndi.gaussian_filter(blob, 2.0 / np.sqrt(2), mode="nearest"),
    2.0 / np.sqrt(2), mode="nearest",
)
print(f"one pass at sigma vs two at sigma/sqrt(2): "
      f"max difference {np.abs(single - halved).max():.2e}")
```

That composition is why the two-pass scheme is allowed at all. It is also, as
section 7 shows, where the border problem comes from.

+++

## 3. Which of the two is more accurate

The choice is not arbitrary, and it can be measured against a known answer. A
Gaussian blob has an analytic Hessian, and a Gaussian smoothed by a Gaussian is
another Gaussian with `t**2 = s**2 + sigma**2`. So the *smoothed* Hessian is
analytic too.

```{code-cell} ipython3
def analytic_hessian(sigma):
    """The exact Hessian of the blob after smoothing at `sigma`."""
    t2 = BLOB_S**2 + sigma**2
    g = np.exp(-radius2 / (2 * t2)) / (2 * np.pi * t2)
    di, dj = bi - centre, bj - centre
    return [g * (di**2 / t2**2 - 1 / t2),
            g * (di * dj / t2**2),
            g * (dj**2 / t2**2 - 1 / t2)]


INTERIOR = slice(30, -30)


def relative_error(computed, sigma):
    """Worst error over the three entries, away from the border."""
    truth = analytic_hessian(sigma)
    scale = max(np.abs(e).max() for e in truth)
    return max(np.abs(c[INTERIOR, INTERIOR] - e[INTERIOR, INTERIOR]).max()
               for c, e in zip(computed, truth)) / scale
```

```{code-cell} ipython3
sigmas = [0.6, 0.8, 0.9, 1.0, 1.5, 2.0, 3.0, 5.0]
finite, gaussian_deriv = [], []
for sigma in sigmas:
    finite.append(relative_error(
        hessian_matrix(blob, sigma=sigma, mode="nearest",
                       use_gaussian_derivatives=False), sigma))
    gaussian_deriv.append(relative_error(
        hessian_matrix(blob, sigma=sigma, mode="nearest",
                       use_gaussian_derivatives=True), sigma))

print(f"{'sigma':>6}{'finite differences':>21}{'gaussian derivatives':>23}")
for sigma, a, b in zip(sigmas, finite, gaussian_deriv):
    print(f"{sigma:>6}{a:>20.2%}{b:>23.2%}")
```

Above `sigma = 1` the Gaussian-derivative route is exact to the precision of
the test — the measured error is zero, which is why the curve below rests on an
imposed floor — while finite differences stay wrong by a few percent however much the
image is smoothed. Below about `sigma = 0.85` they swap places, and the
Gaussian route degrades sharply: its kernels are then narrower than a pixel and
badly sampled. That is the aliasing the docstring warns about when it advises
against a `sigma` much less than 1, and it is what the `truncate = 100` line in
the source is defending against.

Two things about this crossover are revisited later. The aliasing is repairable
in the kernel rather than inherent to the method, which is Fix C; and the
crossover itself moves with the test image, because `blob` is very smooth and
that flatters finite differences more than a textured image would.

+++

### How other libraries compute it

OpenCV has no Hessian function, but it has the parts: `GaussianBlur` for the
scale and `Sobel` for the derivatives. One catch is that `cv2.Sobel` returns
**unnormalised** kernels, so a direct comparison measures a scale factor rather
than an error. `cv2.getDerivKernels` shows the difference and supplies the
normalised form.

```{code-cell} ipython3
for dx, dy in ((2, 0), (1, 1)):
    raw = cv2.getDerivKernels(dx, dy, 3, normalize=False)
    normalised = cv2.getDerivKernels(dx, dy, 3, normalize=True)
    print(f"d{dx}{dy}  raw {raw[0].ravel()} x {raw[1].ravel()}"
          f"   normalised {normalised[0].ravel()} x {normalised[1].ravel()}")
```

```{code-cell} ipython3
def opencv_hessian(image, sigma):
    """Blur at `sigma`, then take normalised Sobel second derivatives."""
    size = int(6 * sigma) | 1
    smoothed = cv2.GaussianBlur(image, (size, size), sigma,
                                borderType=cv2.BORDER_REPLICATE)
    out = []
    for dy, dx in ((2, 0), (1, 1), (0, 2)):
        kx, ky = cv2.getDerivKernels(dx, dy, 3, normalize=True)
        out.append(cv2.sepFilter2D(smoothed, cv2.CV_64F, kx, ky,
                                   borderType=cv2.BORDER_REPLICATE))
    return out


opencv = [relative_error(opencv_hessian(blob, s), s) for s in sigmas]
print(f"{'sigma':>6}{'finite':>10}{'gaussian':>11}{'opencv Sobel':>15}")
for sigma, a, b, d in zip(sigmas, finite, gaussian_deriv, opencv):
    print(f"{sigma:>6}{a:>9.2%}{b:>11.2%}{d:>15.2%}")
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(6.2, 3.0))
for values, colour, label in ((finite, C_TWO, "finite differences"),
                              (gaussian_deriv, C_ONE, "gaussian derivatives"),
                              (opencv, C_THREE, "opencv Sobel")):
    # A log axis cannot show an exact zero, so the curve rests on a floor.
    ax.semilogy(sigmas, np.maximum(values, 1e-6), color=colour, linewidth=2,
                marker="o", markersize=4, label=label)
ax.set_xlabel("sigma", fontsize=8, color=MUTED)
ax.set_ylabel("worst relative error, floored at 1e-6", fontsize=8, color=MUTED)
ax.tick_params(labelsize=8, colors=MUTED)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_color(GRID)
ax.legend(frameon=False, fontsize=8)
ax.set_title("accuracy against an analytic Hessian")
fig.tight_layout()
```

Pillow has no equivalent: `ImageFilter` offers fixed 3 by 3 kernels, with no
scale parameter and no second derivatives, so it does not enter this
comparison.

The ordering is stable: Gaussian derivatives above `sigma = 1`, then OpenCV's
normalised Sobel, then our finite differences. This is the case for changing
the default, and it is separate from the border defect that the rest of this
notebook is about.

+++

## 4. What `order` is supposed to mean

Both functions return the upper-diagonal elements of a symmetric 2x2 tensor.
With `order='rc'` they are named `(Arr, Arc, Acc)`; with `order='xy'` they are
named `(Axx, Axy, Ayy)`. Since `x` is the column axis and `y` the row axis,
`Axx` is `Acc`, `Ayy` is `Arr`, and `Axy` is `Arc`. So the documented contract
is that `order` reverses the list and changes nothing else.

The test suite says so too, in `tests/skimage/feature/test_corner.py`:

```python
# verify results are equivalent, just reversed in order
assert_array_equal(Hxy, Hrc)
```

+++

## 5. `structure_tensor` keeps that contract exactly

```{code-cell} ipython3
st_xy = structure_tensor(IMG, sigma=SIGMA, mode=MODE, order="xy")
st_rc = structure_tensor(IMG, sigma=SIGMA, mode=MODE, order="rc")
print("xy equals reversed rc, bit for bit:",
      all(np.array_equal(a, b) for a, b in zip(st_xy, st_rc[::-1])))
```

This is not luck. `structure_tensor` forms each element as a pointwise product
of two derivative images and then smooths it. The only element that could
differ is the mixed one, `gaussian(d1 * d0)` against `gaussian(d0 * d1)`, and
IEEE multiplication is commutative — so `gaussian` receives bit-identical
input.

## 6. `hessian_matrix` does not

```{code-cell} ipython3
def hess(order, ugd):
    return hessian_matrix(IMG, sigma=SIGMA, mode=MODE, order=order,
                          use_gaussian_derivatives=ugd)

for ugd in (False, True):
    xy, rc = hess("xy", ugd), hess("rc", ugd)[::-1]
    worst = max(float(np.max(np.abs(a - b))) for a, b in zip(xy, rc))
    print(f"use_gaussian_derivatives={ugd!s:<5} "
          f"identical={all(np.array_equal(a, b) for a, b in zip(xy, rc))!s:<5} "
          f"max abs difference {worst:.3e}")
```

The `np.gradient` path agrees to rounding. The Gaussian-derivative path does
not — and that path is the one `frangi`, `sato`, `meijering` and `hessian` all
use, and the one the `FutureWarning` says will become the default.

```{code-cell} ipython3
xy = hess("xy", True)
rc = hess("rc", True)[::-1]

fig, axes = plt.subplots(1, 3, figsize=(9.0, 2.8))
show(axes[0], xy[1], "order='xy', mixed element", diverging=True)
show(axes[1], rc[1], "order='rc' reversed, mixed element", diverging=True)
im = show(axes[2], np.abs(xy[1] - rc[1]), "absolute difference")
fig.colorbar(im, ax=axes[2], fraction=0.046)
fig.suptitle("the same element, asked for two ways", y=1.02)
fig.tight_layout()
```

The difference is a ring around the border. Everything inside is identical.

## 7. Why: extensions commute across axes, not within one

`mode` says how to continue the image past its edge. Each call to
`ndi.gaussian_filter` applies that rule itself. The question is whether
composing calls preserves it.

```{code-cell} ipython3
P = 40
g1 = lambda a, ax: ndi.gaussian_filter1d(a, SIGMA, axis=ax, mode=MODE)

def error_vs_pad_once(f):
    return np.max(np.abs(f(IMG) - f(np.pad(IMG, P, mode="edge"))[P:-P, P:-P]))

print(f"{'composition':<48}{'error vs pad-once':>18}")
for label, f in [
    ("one pass on axis 0", lambda a: g1(a, 0)),
    ("axis 0, then axis 1   (different axes)", lambda a: g1(g1(a, 0), 1)),
    ("axis 0, axis 1, axis 0   (axis 0 twice)", lambda a: g1(g1(g1(a, 0), 1), 0)),
    ("axis 1, then axis 1   (same axis twice)", lambda a: g1(g1(a, 1), 1)),
]:
    print(f"{label:<48}{error_vs_pad_once(f):>18.2e}")
```

A boundary extension along one axis commutes **exactly** with filtering along a
*different* axis, and not at all with filtering along the *same* axis.

Here is what the second call gets wrong. The first pass produces a derivative
field. Outside the image that field keeps varying for about a kernel radius
before it settles. `mode='nearest'` on the *cropped* field asserts instead that
it is constant from the edge onwards.

```{code-cell} ipython3
ss = SIGMA / np.sqrt(2)
kw0 = dict(mode=MODE, truncate=8)
row = 30

true_field = ndi.gaussian_filter(np.pad(IMG, P, mode="edge"), ss, order=[1, 0], **kw0)
have = ndi.gaussian_filter(IMG, ss, order=[1, 0], **kw0)

span = 24
outside = np.arange(-span, 0)
inside = np.arange(0, span)
fig, ax = plt.subplots(figsize=(6.4, 2.8))
ax.plot(np.concatenate([outside, inside]),
        true_field[P + row, P - span:P + span], color=C_ONE, lw=2.6,
        label="what the field actually does")
ax.plot(outside, np.full(span, have[row, 0]), color=C_TWO, lw=2, ls="--",
        label="what mode='nearest' assumes outside")
ax.plot(inside, have[row, :span], color=C_TWO, lw=1.2)
ax.axvline(0, color=MUTED, lw=1)
ax.annotate("image edge", (0, ax.get_ylim()[1]), xytext=(4, -10),
            textcoords="offset points", fontsize=8, color=MUTED, va="top")
ax.set_xlabel("distance from the left edge, in pixels", fontsize=8, color=MUTED)
ax.set_ylabel("first-pass derivative", fontsize=8, color=MUTED)
ax.tick_params(labelsize=8, colors=MUTED)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(GRID)
ax.legend(frameon=False, fontsize=8, loc="lower right")
ax.set_title("the second call re-extends an intermediate, and gets it wrong")
fig.tight_layout()
```

The gap between the two curves outside the edge is what the second call
integrates over, and it is why the border band is wrong.

A single `gaussian_filter` call touches each axis once, so it is exact — a
separable 2-D Gaussian is bit-identical to the padded 2-D convolution.
Separability is innocent.

+++

### The two calls

`_hessian_matrix_with_gaussian` never asks for a second derivative directly. It
asks for a **first** derivative twice, in two separate calls to
`ndi.gaussian_filter`, each at `sigma / sqrt(2)`:

1. **One call per axis, on the image.** `order=[1, 0]` gives `d/dr` of the
   smoothed image and `order=[0, 1]` gives `d/dc`. For a 2-D image that is two
   calls, producing two gradient fields.
2. **One call per element, on a gradient field.** Each Hessian entry takes one
   of those fields and differentiates it again along one axis. `Hrr` is `d/dr`
   applied to the `d/dr` field, `Hcc` is `d/dc` applied to the `d/dc` field,
   and `Hrc` is `d/dc` applied to the `d/dr` field.

Written out, with nothing hidden:

```{code-cell} ipython3
scaled = SIGMA / np.sqrt(2)
filter_kwargs = dict(mode=MODE, truncate=8)

# Call 1: a first derivative along each axis, over the whole image.
grad_r = ndi.gaussian_filter(IMG, scaled, order=[1, 0], **filter_kwargs)
grad_c = ndi.gaussian_filter(IMG, scaled, order=[0, 1], **filter_kwargs)

# Call 2: a second first derivative, applied to one of those fields.
by_hand = [
    ndi.gaussian_filter(grad_r, scaled, order=[1, 0], **filter_kwargs),   # Hrr
    ndi.gaussian_filter(grad_r, scaled, order=[0, 1], **filter_kwargs),   # Hrc
    ndi.gaussian_filter(grad_c, scaled, order=[0, 1], **filter_kwargs),   # Hcc
]

shipped = hessian_matrix(IMG, sigma=SIGMA, mode=MODE,
                         use_gaussian_derivatives=True)
for name, mine, theirs in zip(("Hrr", "Hrc", "Hcc"), by_hand, shipped):
    print(f"{name}: reproduces hessian_matrix exactly: "
          f"{np.array_equal(mine, theirs)}")
```

Each of those calls is itself two one-dimensional passes, so one 2-D Hessian
entry costs four passes arranged as two calls. The passes are not the problem;
the **calls** are. A boundary extension is applied per call, so every axis is
extended twice, and on the second occasion it is extended from an array the
first call has already smoothed along that axis. Cropping threw information
away; re-padding invents a replacement.

That also explains the ordering difference. `Hrc` above runs `d/dr` and then
`d/dc`. Asking for `order='xy'` runs `d/dc` and then `d/dr`, which re-extends a
different intermediate, so the two disagree wherever the extension is felt.

## 8. There is a right answer, and no element has it

Pad once by the rule `mode` names, filter with a wide margin, then crop. In
that construction the two orders agree to 1e-17, so it defines the quantity
both are approximating.

```{code-cell} ipython3
def reference(image, sigma, mode=MODE, npmode="edge"):
    """The Hessian of the smoothed, once-extended image."""
    trunc = 8 if sigma > 1 else 100
    pad = 2 * int(trunc * sigma / np.sqrt(2) + 0.5) + 1
    big = np.pad(image, pad, mode=npmode)
    H = hessian_matrix(big, sigma=sigma, mode=mode, use_gaussian_derivatives=True)
    return [h[pad:-pad, pad:-pad] for h in H]


ref = reference(IMG, SIGMA)
cur = hessian_matrix(IMG, sigma=SIGMA, mode=MODE, use_gaussian_derivatives=True)

names = ["Hrr (diagonal)", "Hrc (mixed)", "Hcc (diagonal)"]
fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.8))
for ax, name, c, r in zip(axes, names, cur, ref):
    d = np.abs(c - r)
    im = show(ax, d, f"{name}\nmax {d.max():.1e}, {d.max()/np.abs(r).max():.0%} of range")
    fig.colorbar(im, ax=ax, fraction=0.046)
fig.suptitle("every element is wrong at the border, not just the mixed one", y=1.04)
fig.tight_layout()
```

```{code-cell} ipython3
for name, c, r in zip(names, cur, ref):
    d = np.abs(c - r)
    print(f"{name:<16} max {d.max():.3e}  relative {d.max()/np.abs(r).max():6.1%}"
          f"   interior {d[20:-20, 20:-20].max():.2e}")
```

The interior is exact. The defect is confined to a border band, and inside that
band it reaches about a third of the signal.

```{code-cell} ipython3
radius = int(8 * SIGMA / np.sqrt(2) + 0.5)
profile = np.abs(cur[1] - ref[1]).max(axis=0)

fig, ax = plt.subplots(figsize=(6.4, 2.6))
ax.semilogy(np.arange(len(profile)), np.maximum(profile, 1e-18), color=C_ONE, lw=2)
for edge in (radius, len(profile) - 1 - radius):
    ax.axvline(edge, color=C_TWO, lw=1.5, ls="--")
ax.annotate("one kernel radius", (radius, profile.max()), xytext=(6, -4),
            textcoords="offset points", fontsize=8, color=C_TWO)
ax.set_xlabel("column", fontsize=8, color=MUTED)
ax.set_ylabel("max error in that column", fontsize=8, color=MUTED)
ax.tick_params(labelsize=8, colors=MUTED)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(GRID)
ax.set_title("the error stops exactly one kernel radius in")
fig.tight_layout()
```

## 9. It reaches the ridge filters

`frangi`, `sato`, `meijering` and `hessian` all call `hessian_matrix` with
`use_gaussian_derivatives=True`.

Two details matter for making the comparison mean anything. The ridge filters
default to `mode='reflect'`, so the once-extended reference has to be built with
the matching `numpy` rule, not with `edge`; and their default `sigmas` start at
1, where `hessian_matrix` raises `truncate` to 100, so the margin has to exceed
one hundred pixels rather than the forty a `truncate = 8` kernel would need.
Both are easy to get wrong, and either one on its own produces a difference that
has nothing to do with the defect.

That `truncate = 100` is the first of several appearances. It is a guard against
aliasing at small `sigma`, and it turns up again as a cost in Fix B, as the
reason Fix A is slow at `sigma = 0.5`, and finally in Fix C, which removes the
reason for it.

```{code-cell} ipython3
photo = ski.util.img_as_float(ski.data.camera())[::2, ::2]
PAD = 110                                  # > truncate 100 * sigma 1
RIDGE = dict(mode="nearest")               # matches np.pad(..., mode='edge')
padded = np.pad(photo, PAD, mode="edge")

RIDGE_FILTERS = [("frangi", frangi), ("sato", sato),
                 ("meijering", meijering), ("hessian", hessian)]

fig, axes = plt.subplots(3, 4, figsize=(10.5, 7.6))
for col, (name, f) in enumerate(RIDGE_FILTERS):
    direct = f(photo, **RIDGE)
    ref_out = f(padded, **RIDGE)[PAD:-PAD, PAD:-PAD]
    diff = np.abs(direct - ref_out)
    show(axes[0, col], direct, f"{name}, as computed")
    show(axes[1, col], ref_out, "with the border handled once")
    im = show(axes[2, col], diff,
              f"difference, {diff.max()/max(np.abs(ref_out).max(), 1e-12):.0%} of range")
    fig.colorbar(im, ax=axes[2, col], fraction=0.046)
fig.suptitle("the border defect, end to end", y=1.01)
fig.tight_layout()
```

```{code-cell} ipython3
def ridge_border_error(label):
    """Each ridge filter against itself on a once-extended image."""
    print(f"{label}")
    for name, f in RIDGE_FILTERS:
        a = f(photo, **RIDGE)
        b = f(padded, **RIDGE)[PAD:-PAD, PAD:-PAD]
        d = np.abs(a - b)
        print(f"   {name:<10} max {d.max():.3e} "
              f"({d.max()/max(np.abs(b).max(), 1e-12):6.1%} of range)"
              f"   30 px in: {d[30:-30, 30:-30].max():.2e}")


ridge_border_error("as shipped")
```

`hessian` flips completely at the border. `meijering` also moves in the
interior, because it normalises by a global maximum that the border artefact
distorts.

## 10. Three candidate fixes

+++

### Fix A: compute every element in one call

Section 7 named the cause exactly: a boundary extension is applied per **call**,
so any element built from two calls extends some axis twice. The remedy follows
from the cause. Ask `scipy.ndimage.gaussian_filter` for the whole derivative in
one call, at the full `sigma`, using the `order` vector to say which axes to
differentiate and how many times:

| element | `order` | axes touched |
| --- | --- | --- |
| `Hrr` | `[2, 0]` | axis 0 once, at second order |
| `Hrc` | `[1, 1]` | each axis once, at first order |
| `Hcc` | `[0, 2]` | axis 1 once, at second order |

```{code-cell} ipython3
def one_call(image, sigma, mode=MODE, order="rc"):
    """Every Hessian element from a single `gaussian_filter` call, at full sigma."""
    trunc = 8 if sigma > 1 else 100
    kw = dict(mode=mode, truncate=trunc)
    orders = ([2, 0], [1, 1], [0, 2])
    if order == "xy":
        orders = orders[::-1]
    return [ndi.gaussian_filter(image, sigma, order=o, **kw) for o in orders]
```

The mixed element `[1, 1]` is the easy half of this, and it is uncontroversial:
it differentiates each axis once, so no axis is touched twice. The diagonals are
the question, because `order=2` is what the code comment in `corner.py` steers
away from, citing scipy's second-order Gaussian derivatives. That claim is
testable, so test it rather than assume it.

**Every element becomes exact.** Against a pad-once reference built the same
way, the difference is not merely small — it is zero to the last bit.

```{code-cell} ipython3
def padded_reference(f, image, sigma, npmode="edge", **kw):
    """Run `f` on a generously extended image, then crop back."""
    trunc = 8 if sigma > 1 else 100
    pad = 2 * int(trunc * sigma + 0.5) + 1
    big = np.pad(image, pad, mode=npmode)
    return [e[pad:-pad, pad:-pad] for e in f(big, sigma, **kw)]


one = one_call(IMG, SIGMA)
one_ref = padded_reference(one_call, IMG, SIGMA)
one_scale = max(np.abs(e).max() for e in one_ref)

print(f"{'element':<18}{'two calls, as shipped':>24}{'one call':>12}")
for name, c, o, r, orf in zip(names, cur, one, ref, one_ref):
    print(f"{name:<18}{np.abs(c - r).max() / np.abs(r).max():>23.1%}"
          f"{np.abs(o - orf).max() / one_scale:>12.1e}")
```

**The `order` ambiguity disappears rather than shrinking.** `[2, 0]` and
`[0, 2]` are each other's mirror and `[1, 1]` is its own, so relabelling the
axes permutes the three results and changes nothing else.

```{code-cell} ipython3
def order_disagreement(f):
    """Relative disagreement between order='rc' and order='xy'."""
    rc, xy = f(order="rc"), f(order="xy")
    sc = max(np.abs(e).max() for e in rc)
    return max(np.abs(a - b).max() for a, b in zip(rc, xy[::-1])) / sc


shipped = lambda order: hessian_matrix(IMG, sigma=SIGMA, mode=MODE, order=order,
                                       use_gaussian_derivatives=True)
print("order='rc' against order='xy', for the same quantity")
print(f"   two calls, as shipped : {order_disagreement(shipped):.2e}")
print(f"   one call              : "
      f"{order_disagreement(lambda order: one_call(IMG, SIGMA, order=order)):.2e}")
```

**In the interior it is more accurate, not less.** This is the measurement that
answers the code comment. Against the analytic Hessian of section 3, one call
beats two from about `sigma = 0.7` upwards, usually by an order of magnitude,
and from about `sigma = 0.75` it beats finite differences as well.

```{code-cell} ipython3
print(f"{'sigma':>6}{'finite diffs':>15}{'two calls':>12}{'one call':>12}{'best':>14}")
for sigma in (0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0, 1.5, 3.0):
    fd = relative_error(hessian_matrix(blob, sigma=sigma, mode="nearest",
                                       use_gaussian_derivatives=False), sigma)
    tc = relative_error(hessian_matrix(blob, sigma=sigma, mode="nearest",
                                       use_gaussian_derivatives=True), sigma)
    oc = relative_error(one_call(blob, sigma, mode="nearest"), sigma)
    best = min([(fd, "finite"), (tc, "two calls"), (oc, "one call")])[1]
    print(f"{sigma:>6}{fd:>14.2%}{tc:>12.2%}{oc:>12.2%}{best:>14}")
```

So the comment is right about the *direction* of the risk and wrong about where
it bites. Above about `sigma = 0.7` the extra pass costs more accuracy than the
harder kernel does, and the crossover against finite differences moves from
about `sigma = 0.85` down to about `sigma = 0.73`. Below `sigma = 0.65` one call
is far worse — 950% against 92% at `sigma = 0.5`.

That last figure deserves more than a shrug, because the two schemes do not fail
in the same way. Two tests separate them: a constant image, whose Hessian is
zero everywhere, and a quadratic ramp `f = r**2 / 2`, whose `Hrr` is 1
everywhere.

```{code-cell} ipython3
const = np.ones((40, 40))
ramp = np.repeat((np.arange(400.0)[:, None] - 200) ** 2 / 2, 8, axis=1)
middle = (200, 4)


def two_calls(image, sigma, mode=MODE):
    """The shipped scheme, written out."""
    trunc = 8 if sigma > 1 else 100
    kw = dict(mode=mode, truncate=trunc)
    scaled = sigma / np.sqrt(2)
    grad = ndi.gaussian_filter(image, scaled, order=[1, 0], **kw)
    return ndi.gaussian_filter(grad, scaled, order=[1, 0], **kw)


print(f"{'':7}{'constant, want 0':>28}{'quadratic ramp, want 1':>32}")
print(f"{'sigma':>7}{'two calls':>14}{'one call':>14}{'two calls':>16}{'one call':>16}")
for sigma in (0.5, 0.7, 1.0, 1.5):
    print(f"{sigma:>7}"
          f"{two_calls(const, sigma)[20, 20]:>14.4f}"
          f"{one_call(const, sigma)[0][20, 20]:>14.4f}"
          f"{two_calls(ramp, sigma)[middle]:>16.4f}"
          f"{one_call(ramp, sigma)[0][middle]:>16.4f}")
```

The shipped scheme suffers **gain collapse**. At `sigma = 0.5` it reports about
eight per cent of the true curvature — badly wrong in size, but the sign is
right, the structure is right, and it returns exactly zero on a constant. A
first-derivative kernel is odd, so it sums to zero by symmetry whatever the
sampling does to it.

One call has a **DC leak**, which is worse in kind. Its `order=2` kernel is
even, and nothing forces the sampled version to sum to zero. At `sigma = 0.5`
it sums to `-0.56`, so the operator responds to absolute brightness: a flat
bright region reads as strongly curved, and the reported curvature depends on
what you added to the image rather than on its shape.

```{code-cell} ipython3
def gaussian_taps(sigma, order, trunc):
    """The 1-D kernel `gaussian_filter` builds, from the explicit formula."""
    lw = int(trunc * sigma + 0.5)
    x = np.arange(-lw, lw + 1).astype(float)
    g = np.exp(-(x**2) / (2 * sigma**2))
    g /= g.sum()
    if order == 0:
        return x, g
    if order == 1:
        return x, g * (x / sigma**2)
    return x, g * ((x**2 - sigma**2) / sigma**4)


print("moments of the order=2 kernel; a second derivative needs sum 0 and "
      "sum k*x**2/2 = 1")
print(f"{'sigma':>7}{'sum':>14}{'sum k*x**2/2':>16}")
for sigma in (0.4, 0.5, 0.7, 1.0, 1.5):
    x, k = gaussian_taps(sigma, 2, 8 if sigma > 1 else 100)
    print(f"{sigma:>7}{k.sum():>14.2e}{(k * x**2).sum() / 2:>16.4f}")
```

Both moments are wrong together, and both are wrong for the same reason: at
`sigma = 0.5` the kernel varies faster than the sampling grid can follow, so the
discrete sum stops matching the integral it stands for. That is aliasing, named
precisely — and naming it that precisely is what makes it repairable, which is
Fix C.

**It is not slower.** Six one-dimensional passes replace ten, against kernels
`sqrt(2)` wider.

```{code-cell} ipython3
import time


def best_of(f, n=7):
    """Smallest wall-clock time over `n` runs, after one warm-up."""
    f()
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        f()
        times.append(time.perf_counter() - t0)
    return min(times)


print(f"{'sigma':>6}{'truncate':>10}{'two calls':>13}{'one call':>12}{'ratio':>8}")
for sigma in (0.5, 1.0, 1.5, 3.0, 6.0):
    t_two = best_of(lambda: hessian_matrix(photo, sigma=sigma, mode=MODE,
                                           use_gaussian_derivatives=True))
    t_one = best_of(lambda: one_call(photo, sigma))
    print(f"{sigma:>6}{8 if sigma > 1 else 100:>10}{t_two * 1e3:>10.1f} ms"
          f"{t_one * 1e3:>10.1f} ms{t_one / t_two:>7.2f}x")
```

It is faster from `sigma = 1` upwards, and slower at `sigma = 0.5` only because
the `truncate = 100` hack then makes the full-sigma kernel the wider one. That
hack is not a fixed cost of the method: Fix C shows it is unnecessary, and
removing it turns this row around.

**It generalises to N dimensions unchanged.** The `order` vector carries a 2 on
the diagonal and two 1s off it, whatever the dimension.

```{code-cell} ipython3
def one_call_nd(image, sigma, mode=MODE):
    """The same rule, written for any number of dimensions."""
    trunc = 8 if sigma > 1 else 100
    kw = dict(mode=mode, truncate=trunc)
    out = []
    for a in range(image.ndim):
        for b in range(a, image.ndim):
            orders = [0] * image.ndim
            orders[a] += 1
            orders[b] += 1
            out.append(ndi.gaussian_filter(image, sigma, order=orders, **kw))
    return out


vol = np.random.default_rng(3).random((24, 26, 28))
P3 = 40
crop = (slice(P3, -P3),) * 3
big3 = np.pad(vol, P3, mode="edge")

two_3d = hessian_matrix(vol, sigma=SIGMA, mode=MODE, use_gaussian_derivatives=True)
two_3d_ref = [e[crop] for e in hessian_matrix(big3, sigma=SIGMA, mode=MODE,
                                              use_gaussian_derivatives=True)]
one_3d = one_call_nd(vol, SIGMA)
one_3d_ref = [e[crop] for e in one_call_nd(big3, SIGMA)]
sc3 = max(np.abs(e).max() for e in one_3d_ref)

print(f"3-D volume, {len(two_3d)} elements, worst border error")
print(f"   two calls : "
      f"{max(np.abs(a - b).max() for a, b in zip(two_3d, two_3d_ref)) / sc3:.1%}")
print(f"   one call  : "
      f"{max(np.abs(a - b).max() for a, b in zip(one_3d, one_3d_ref)) / sc3:.1e}")
```

**End to end, three of the four ridge filters become exact.** The ridge filters
import `hessian_matrix` from the corner module when they are called, so
replacing it there runs the whole pipeline on Fix A.

```{code-cell} ipython3
import importlib
import sys
from unittest import mock

package = sys.modules[frangi.__module__].__package__.rsplit(".", 1)[0]
corner_mod = importlib.import_module(f"{package}.feature.corner")


def as_hessian_matrix(rule):
    """Wrap a Hessian rule in the `hessian_matrix` signature, ready to patch in."""
    def replacement(image, sigma=1, mode="reflect", cval=0, order="rc",
                    use_gaussian_derivatives=True):
        return rule(ski.util.img_as_float(image), sigma, mode=mode)
    return replacement


with mock.patch.object(corner_mod, "hessian_matrix", as_hessian_matrix(one_call_nd)):
    ridge_border_error("with Fix A")
```

`frangi`, `sato` and `hessian` go to zero exactly — not to a small residue.
`meijering` drops from 26.3% to 3.3%, and what is left is the *interior* figure,
unchanged: it normalises by a global maximum, and cropping a border band shifts
that maximum. That is a separate defect in `meijering`, which the border fix
exposes rather than causes.

+++

### Fix B: pad once inside the function

Extend the image by the rule `mode` names, run the existing scheme, crop. Every
element becomes exact. This was the obvious fix before Fix A was measured, and
it reaches the same place by paying for it: the second extension is still wrong,
so the padding exists to push it far enough away to be cropped off.

```{code-cell} ipython3
import time

# Timed on the 256x256 photograph, not the small test array: the padding cost
# is relative to image size, so a toy array badly overstates it.
print(f"{'sigma':>6}{'pad':>6}{'current':>12}{'padded':>10}{'cost':>8}")
for sigma in (0.5, 1.0, 1.5, 3.0):
    t0 = time.perf_counter()
    hessian_matrix(photo, sigma=sigma, mode=MODE, use_gaussian_derivatives=True)
    t1 = time.perf_counter()
    reference(photo, sigma)
    t2 = time.perf_counter()
    trunc = 8 if sigma > 1 else 100
    pad = 2 * int(trunc * sigma / np.sqrt(2) + 0.5) + 1
    print(f"{sigma:>6}{pad:>6}{(t1-t0)*1e3:>10.1f} ms{(t2-t1)*1e3:>8.1f} ms"
          f"{(t2-t1)/(t1-t0):>7.1f}x")
```

The cost is modest above `sigma = 1`. Below it the function sets
`truncate = 100` to fight aliasing, and the padding has to clear that, so it
becomes enormous — 143 pixels at `sigma = 1`. Fix C removes the reason for that
hack, which removes most of this cost with it.

+++

### Fix C: correct the kernel, so Fix A works at every scale

Fix A's only weakness is the small-`sigma` aliasing measured above, and that
weakness is stated as two failed moment conditions. A discrete second-derivative
operator has to annihilate constants and reproduce quadratics:

```
    sum(k)             ==  0        no response to a constant
    sum(k * x**2) / 2  ==  1        exact on f = x**2 / 2
```

Imposing both on the sampled kernel is the standard remedy for a badly sampled
derivative operator, and it takes two lines.

```{code-cell} ipython3
def corrected_taps(sigma, order, trunc=8):
    """Gaussian derivative taps with the discrete moments repaired."""
    x, g = gaussian_taps(sigma, 0, trunc)
    _, k = gaussian_taps(sigma, order, trunc)
    if order == 0:
        return x, g
    if order == 1:
        return x, k / (k * x).sum()             # exact on f = x
    k = k - k.sum() * g                         # annihilate constants
    return x, k / ((k * x**2).sum() / 2)        # exact on f = x**2 / 2
```

The first correction has to be spread **along the Gaussian**, not evenly. The
obvious alternative is to subtract `k.sum() / k.size` from every tap, which also
makes the kernel sum to zero. It is much worse, and the reason is the
`truncate = 100` hack: at `sigma = 0.7` the kernel has 141 taps and about four of
them carry the operator, so an even correction lays a wide, flat plateau under a
one-pixel derivative.

```{code-cell} ipython3
def flat_corrected_taps(sigma, order, trunc=8):
    """The naive variant: take the constant out of every tap equally."""
    x, k = gaussian_taps(sigma, order, trunc)
    if order != 2:
        return corrected_taps(sigma, order, trunc)
    k = k - k.sum() / k.size
    return x, k / ((k * x**2).sum() / 2)


def with_taps(taps_of, image, sigma, mode=MODE, trunc=8):
    """Build the three elements from whichever tap rule is passed."""
    taps = {o: taps_of(sigma, o, trunc)[1] for o in (0, 1, 2)}
    sep = lambda a, b: ndi.correlate1d(
        ndi.correlate1d(image, taps[a], axis=0, mode=mode), taps[b], axis=1, mode=mode)
    return [sep(2, 0), sep(1, 1), sep(0, 2)]


print(f"{'sigma':>7}{'taps':>7}{'corrected along g':>20}{'corrected flat':>17}")
for sigma in (0.5, 0.7, 1.0):
    trunc = 8 if sigma > 1 else 100          # the hack, as shipped
    n = 2 * int(trunc * sigma + 0.5) + 1
    along = relative_error(
        with_taps(corrected_taps, blob, sigma, mode="nearest", trunc=trunc), sigma)
    flat = relative_error(
        with_taps(flat_corrected_taps, blob, sigma, mode="nearest", trunc=trunc), sigma)
    print(f"{sigma:>7}{n:>7}{along:>19.2%}{flat:>17.2%}")
```

Both elements of the Hessian then come from one call each, exactly as in Fix A,
except that the taps are supplied rather than requested by name. Two 1-D passes
along *different* axes still commute with the boundary extension, so section 7's
guarantee is untouched.

```{code-cell} ipython3
def fix_c(image, sigma, mode=MODE, trunc=8, order="rc"):
    """Fix A, with the corrected kernels and no `truncate` hack."""
    taps = {o: corrected_taps(sigma, o, trunc)[1] for o in (0, 1, 2)}

    def sep(first, second):
        out = ndi.correlate1d(image, taps[first], axis=0, mode=mode)
        return ndi.correlate1d(out, taps[second], axis=1, mode=mode)

    out = [sep(2, 0), sep(1, 1), sep(0, 2)]
    return out[::-1] if order == "xy" else out
```

**It removes the small-`sigma` cost rather than trading it away.**

```{code-cell} ipython3
print(f"{'sigma':>6}{'finite':>10}{'two calls':>12}{'Fix A':>10}{'Fix C':>10}")
for sigma in (0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0, 1.5, 3.0):
    print(f"{sigma:>6}"
          f"{relative_error(hessian_matrix(blob, sigma=sigma, mode='nearest', use_gaussian_derivatives=False), sigma):>9.2%}"
          f"{relative_error(hessian_matrix(blob, sigma=sigma, mode='nearest', use_gaussian_derivatives=True), sigma):>12.2%}"
          f"{relative_error(one_call(blob, sigma, mode='nearest'), sigma):>10.2%}"
          f"{relative_error(fix_c(blob, sigma, mode='nearest'), sigma):>10.2%}")
```

`blob` is a very smooth test object, which flatters finite differences. A
sinusoid of known wavelength is a harder and more honest test, because the
smoothed answer is analytic at every frequency and the frequency is a knob.

```{code-cell} ipython3
N = 200
si, sj = np.indices((N, N), dtype=float)
INNER = (slice(20, -20),) * 2

print("interior error on f = cos(2*pi*r/L), at sigma = 0.5")
print(f"{'L, pixels':>11}{'finite':>10}{'two calls':>12}{'Fix A':>10}{'Fix C':>10}")
for L in (40, 20, 10, 8, 5, 4):          # each divides N, so mode='wrap' is exact
    w = 2 * np.pi / L
    wave = np.cos(w * si)
    truth = -(w**2) * np.exp(-(w**2) * 0.25 / 2) * np.cos(w * si)
    scale = (w**2) * np.exp(-(w**2) * 0.25 / 2)

    def miss(got):
        return np.abs(got[INNER] - truth[INNER]).max() / scale

    smooth = ndi.gaussian_filter(wave, 0.5, mode="wrap")
    fd = np.gradient(np.gradient(smooth)[0])[0]
    print(f"{L:>11}{miss(fd):>9.2%}"
          f"{miss(two_calls(wave, 0.5, mode='wrap')):>12.2%}"
          f"{miss(one_call(wave, 0.5, mode='wrap')[0]):>10.2%}"
          f"{miss(fix_c(wave, 0.5, mode='wrap')[0]):>10.2%}")
```

Finite differences run from 0.8% to 57% at a *fixed* `sigma`, purely with image
content. Their apparent advantage below `sigma = 0.85` is a property of the
smooth blob, not of the method, which is worth knowing before treating them as
the safe fallback at small scale.

**Everything Fix A won is kept.**

```{code-cell} ipython3
print(f"{'sigma':>6}{'border vs pad-once':>21}{'order rc vs xy':>17}"
      f"{'response to a constant':>25}")
for sigma in (0.4, 0.5, 0.7, 1.0, 1.5, 3.0):
    pad = 2 * int(8 * sigma + 0.5) + 1
    direct = fix_c(IMG, sigma)
    padded_ref = [e[pad:-pad, pad:-pad]
                  for e in fix_c(np.pad(IMG, pad, mode="edge"), sigma)]
    sc = max(np.abs(e).max() for e in padded_ref)
    swapped = fix_c(IMG, sigma, order="xy")[::-1]
    print(f"{sigma:>6}"
          f"{max(np.abs(a - b).max() for a, b in zip(direct, padded_ref)) / sc:>20.1e}"
          f"{max(np.abs(a - b).max() for a, b in zip(direct, swapped)) / sc:>17.1e}"
          f"{abs(fix_c(const, sigma)[0][20, 20]):>25.1e}")
```

**And `truncate = 100` becomes unnecessary.** That hack exists to fight the
aliasing by brute-force margin; correcting the kernel removes the reason for it,
and `truncate = 8` then gives the same answer as `truncate = 100` to the digit.

```{code-cell} ipython3
print(f"{'sigma':>6}{'Fix C, truncate=8':>20}{'Fix C, truncate=100':>22}")
for sigma in (0.4, 0.5, 0.7, 1.0):
    print(f"{sigma:>6}"
          f"{relative_error(fix_c(blob, sigma, mode='nearest', trunc=8), sigma):>19.2%}"
          f"{relative_error(fix_c(blob, sigma, mode='nearest', trunc=100), sigma):>22.2%}")

print()
print(f"{'sigma':>6}{'shipped':>12}{'Fix A':>10}{'Fix C':>10}")
for sigma in (0.5, 1.0, 1.5, 3.0):
    t_s = best_of(lambda: hessian_matrix(photo, sigma=sigma, mode=MODE,
                                         use_gaussian_derivatives=True))
    t_a = best_of(lambda: one_call(photo, sigma))
    t_c = best_of(lambda: fix_c(photo, sigma))
    print(f"{sigma:>6}{t_s * 1e3:>9.1f} ms{t_a * 1e3:>7.1f} ms{t_c * 1e3:>7.1f} ms")
```

#### The construction the literature recommends

Repairing a badly sampled derivative kernel is not new, and Lindeberg's
[discrete scale space](https://www.csc.kth.se/~tony/abstracts/Lin93-JMIV.html)
answers the same question a different way. Rather than sample the continuous
Gaussian and correct it, build the **discrete analogue** of the Gaussian,
`T(n; t) = exp(-t) * I_n(t)` for modified Bessel `I_n` and `t = sigma**2`, and
take derivatives with small-support central differences — `(-1/2, 0, 1/2)` and
`(1, -2, 1)`. A
[2023 survey](https://arxiv.org/abs/2311.11317) of the three discretisations
finds that sampled Gaussian derivatives "perform very poorly at very fine
scales" while the discrete analogue "performs substantially better", which is
this notebook's `sigma = 0.5` row stated from the other direction.

Both constructions satisfy the two moment conditions exactly, so neither can
leak DC.

```{code-cell} ipython3
from scipy.special import ive


def lindeberg_taps(sigma, order, trunc=8):
    """Discrete analogue of the Gaussian, then a small central difference."""
    lw = max(int(trunc * sigma + 0.5), 1)
    n = np.arange(-lw, lw + 1)
    T = ive(np.abs(n), sigma**2)          # exp(-t) * I_n(t)
    T = T / T.sum()
    if order == 0:
        return T
    step = np.array([-0.5, 0.0, 0.5]) if order == 1 else np.array([1.0, -2.0, 1.0])
    return np.convolve(T, step)


def lindeberg(image, sigma, mode=MODE, trunc=8):
    taps = {o: lindeberg_taps(sigma, o, trunc) for o in (0, 1, 2)}
    sep = lambda a, b: ndi.correlate1d(
        ndi.correlate1d(image, taps[a], axis=0, mode=mode), taps[b], axis=1, mode=mode)
    return [sep(2, 0), sep(1, 1), sep(0, 2)]


print(f"{'sigma':>7}{'Fix C sum':>12}{'Fix C m2':>11}"
      f"{'Lindeberg sum':>16}{'Lindeberg m2':>15}")
for sigma in (0.4, 0.5, 0.7, 1.0, 2.0):
    xd, kd = corrected_taps(sigma, 2)
    kl = lindeberg_taps(sigma, 2)
    xl = np.arange(-(kl.size // 2), kl.size // 2 + 1).astype(float)
    print(f"{sigma:>7}{kd.sum():>12.1e}{(kd * xd**2).sum() / 2:>11.4f}"
          f"{kl.sum():>16.1e}{(kl * xl**2).sum() / 2:>15.4f}")
```

They differ in what they converge to. `skimage` documents the Hessian of the
*continuous* image smoothed by a Gaussian, and against that target Fix C wins at
every scale tested.

```{code-cell} ipython3
print("interior error against the analytic smoothed Hessian")
print(f"{'sigma':>7}{'Fix C':>10}{'Lindeberg':>12}")
for sigma in (0.3, 0.5, 0.7, 1.0, 3.0):
    print(f"{sigma:>7}{relative_error(fix_c(blob, sigma, mode='nearest'), sigma):>9.2%}"
          f"{relative_error(lindeberg(blob, sigma, mode='nearest'), sigma):>11.2%}")
```

Lindeberg's error does not fall away with `sigma`: `(1, -2, 1)` has frequency
response `-4 * sin(w/2)**2` rather than `-w**2`, so it under-reports curvature
at high spatial frequency however much the image is smoothed first. That is the
same error family as the `np.gradient` path, milder but permanent — 1.40% at
`sigma = 1` where Fix C is already exact.

What the discrete analogue buys instead is exactness *within* the discrete
setting: an exact discrete semigroup, and guarantees about not creating new
extrema as scale increases. Those are real properties, and nothing measured here
probes them. The choice is therefore about the stated target. `hessian_matrix`
documents a continuous Gaussian at a scale, so Fix C matches what the docstring
promises; a library that redefined its scale space as discrete would prefer
Lindeberg's.

#### Does it still behave like a scale space?

The corrected kernel is no longer a sampled Gaussian derivative, so the question
has to be asked rather than assumed. Three properties matter, and Fix C holds or
improves all three.

**The response varies smoothly with `sigma`.** A multiscale filter sweeping
`sigma` and combining the responses needs the Hessian to vary smoothly with
scale, or it sees steps in its scale response that have nothing to do with the
image.

```{code-cell} ipython3
def scale_step(f, sigma, spread=0.01):
    """Relative change in H for a small relative change in sigma."""
    lo, hi = f(IMG, sigma * (1 - spread)), f(IMG, sigma * (1 + spread))
    w = (slice(20, -20),) * 2
    sc = max(np.abs(e[w]).max() for e in hi)
    return max(np.abs(a[w] - b[w]).max() for a, b in zip(lo, hi)) / sc


shipped_h = lambda image, sigma: hessian_matrix(image, sigma=sigma, mode=MODE,
                                                use_gaussian_derivatives=True)
print("change in H for a 2% change in sigma")
print(f"{'sigma':>7}{'shipped':>11}{'Fix A':>9}{'Fix C':>9}")
for sigma in (0.5, 0.7, 1.0, 1.5, 3.0):
    print(f"{sigma:>7}{scale_step(shipped_h, sigma):>10.2%}"
          f"{scale_step(one_call, sigma):>9.2%}{scale_step(fix_c, sigma):>9.2%}")
```

No step anywhere, and at `sigma = 0.5` Fix C is the *smoothest* of the three:
the shipped scheme moves 20% for a 2% change in scale, because its gain is
collapsing fast just there.

**The semigroup survives.** Smoothing twice at `sigma` should equal smoothing
once at `sigma * sqrt(2)`. Measured on the kernels themselves, so no boundary
enters:

```{code-cell} ipython3
def semigroup_error(order, sigma, corrected):
    taps = corrected_taps if corrected else (lambda s, o, t=8: gaussian_taps(s, o, t))
    narrow = taps(sigma, order)[1]
    smoother = taps(sigma, 0)[1]
    composed = np.convolve(narrow, smoother)
    wide = taps(sigma * np.sqrt(2), order)[1]
    n = max(composed.size, wide.size)
    fit = lambda v: np.pad(v, (n - v.size) // 2)
    return np.abs(fit(composed) - fit(wide)).max() / np.abs(wide).max()


print(f"{'sigma':>7}{'smoothing, scipy':>19}{'smoothing, Fix C':>19}"
      f"{'d2, scipy':>13}{'d2, Fix C':>13}")
for sigma in (0.4, 0.5, 0.7, 1.0, 2.0):
    print(f"{sigma:>7}{semigroup_error(0, sigma, False):>19.1e}"
          f"{semigroup_error(0, sigma, True):>19.1e}"
          f"{semigroup_error(2, sigma, False):>13.1e}"
          f"{semigroup_error(2, sigma, True):>13.1e}")
```

The smoothing kernel is untouched — the correction only rescales it, and scipy
already normalises it — so Fix C inherits whatever the sampled Gaussian does,
including its own breakdown below `sigma = 0.7`. The *derivative* semigroup
improves, sharply at `sigma = 0.4`.

**Scale selection still points at the right scale.** A blob of width `s` should
maximise the scale-normalised response `-sigma**2 * trace(H)` at `sigma = s`.
This is the property the ridge filters lean on when they sweep `sigmas`.

```{code-cell} ipython3
NB = 161
bc = NB // 2
bii, bjj = np.indices((NB, NB), dtype=float)
br2 = (bii - bc) ** 2 + (bjj - bc) ** 2
trial = np.geomspace(0.4, 8.0, 60)


def blob_of(width):
    return np.exp(-br2 / (2 * width**2)) / (2 * np.pi * width**2)


def scale_response(f, image, sigmas):
    """-sigma**2 * trace(H) at the centre, across scale."""
    return np.array([-(s**2) * (f(image, s)[0][bc, bc] + f(image, s)[2][bc, bc])
                     for s in sigmas])


print(f"{'blob width':>12}{'ideal':>8}{'shipped':>10}{'Fix A':>9}{'Fix C':>9}")
for width in (0.6, 0.8, 1.0, 2.0, 4.0):
    picks = [trial[int(np.argmax(scale_response(f, blob_of(width), trial)))]
             for f in (shipped_h, one_call, fix_c)]
    print(f"{width:>12}{width:>8.2f}{picks[0]:>10.2f}{picks[1]:>9.2f}{picks[2]:>9.2f}")
```

The table is the summary; the mechanism is clearer traced out.

```{code-cell} ipython3
WIDTHS = ((0.6, C_ONE), (1.5, C_TWO), (4.0, C_THREE))

fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), sharey=True)
for ax, (name, f) in zip(axes, (("shipped", shipped_h), ("Fix A", one_call),
                                ("Fix C", fix_c))):
    for width, colour in WIDTHS:
        resp = scale_response(f, blob_of(width), trial)
        resp = resp / np.abs(resp).max()
        ax.plot(trial, resp, color=colour, lw=1.9, label=f"width {width}")
        ax.axvline(width, color=colour, lw=0.9, ls=":")
        ax.plot([trial[int(np.argmax(resp))]], [resp.max()], "o", color=colour,
                markersize=6, markeredgecolor="white", zorder=5)
    ax.set_xscale("log")
    ax.set_title(name)
    ax.set_xlabel("sigma", fontsize=8, color=MUTED)
    ax.tick_params(labelsize=8, colors=MUTED)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
axes[0].set_ylabel("normalised response", fontsize=8, color=MUTED)
axes[0].legend(frameon=False, fontsize=7, loc="upper right")
fig.suptitle("dotted line: the blob's true width.   filled circle: where the "
             "method puts it", y=1.04)
fig.tight_layout()
```

The scale axis is 60 log-spaced points, so a peak is located to about two per
cent and the wide-blob rows land on 2.03 and 3.93 rather than exactly 2 and 4.

`shipped` peaks well to the right for the narrow blobs — 0.95 for a blob of
width 0.6 — and correctly for the wide ones. `Fix C` puts every circle on its
dotted line, to that resolution.

`Fix A` puts all three circles at `sigma = 0.4` — the smallest scale on the
axis, and it would put them at whatever smaller value the axis began from. The
curve has no peak of its own down there: the DC leak grows as `sigma` falls, so
the response climbs without limit and the genuine blob response survives only as
a secondary bump. Any code that sweeps scale and takes a maximum collapses onto
the shortest scale offered, whatever the image contains.

That is a silent failure with a sharp edge, and the edge is exactly where the
sweep starts.

```{code-cell} ipython3
fig, axes = plt.subplots(2, 3, figsize=(9.6, 6.4))
summary = {}
for row, sweep in enumerate([(1, 3, 5), (0.5, 1, 3, 5)]):
    got = {}
    for name, f in (("Fix A", one_call), ("Fix C", fix_c)):
        with mock.patch.object(corner_mod, "hessian_matrix", as_hessian_matrix(f)):
            got[name] = frangi(photo, sigmas=sweep, mode="nearest")
    summary[sweep] = got
    gap = np.abs(got["Fix A"] - got["Fix C"])
    # One scale for the pair, so a drop in response shows as a drop.
    together = max(got["Fix A"].max(), got["Fix C"].max())
    show(axes[row, 0], got["Fix A"], f"Fix A, sigmas={sweep}", vmax=together)
    show(axes[row, 1], got["Fix C"], f"Fix C, sigmas={sweep}", vmax=together)
    im = show(axes[row, 2], gap,
              f"difference, {gap.max() / max(got['Fix C'].max(), 1e-12):.0%} of range")
    fig.colorbar(im, ax=axes[row, 2], fraction=0.046)
fig.suptitle("frangi: the two agree until a small sigma enters the sweep", y=1.01)
fig.tight_layout()
```

```{code-cell} ipython3
print(f"{'sigmas':>16}{'':>9}{'peak response':>16}{'mean response':>16}")
for sweep, got in summary.items():
    for name in ("Fix A", "Fix C"):
        print(f"{str(sweep):>16}{name:>9}{got[name].max():>16.4f}"
              f"{got[name].mean():>16.5f}")
```

With the library's own sweep, starting at 1, the two agree to about `1e-06`.
Adding a single `sigma = 0.5` to that same sweep moves the output by 46% of its
range.

The direction of that move is the opposite of what "a small scale dominating"
would suggest. Fix A does not fill the picture with fine detail; it **erases**
what was there. The peak response falls from 0.589 to 0.364 and the mean falls
by a factor of four. A user adding a smaller `sigma` to catch narrower vessels
would lose the ones they already had.

The leak is visible directly on the photograph, without going through `frangi`
at all. A Hessian should describe shape, so its trace should have little to do
with how bright the pixel is; under Fix A at `sigma = 0.5` the two are almost
the same picture.

```{code-cell} ipython3
for name, f in (("Fix A", one_call), ("Fix C", fix_c)):
    H = f(photo, 0.5)
    trace = H[0] + H[2]
    print(f"{name}: corr(trace H, image) = "
          f"{np.corrcoef(trace.ravel(), photo.ravel())[0, 1]:+.3f}")
```

That is the DC leak on real data: `-0.88` says the reported curvature is mostly
a copy of the image intensity, `-0.23` says it is mostly shape. How that
propagates through `frangi`'s vesselness ratio to the fourfold drop above is not
traced here; the leak itself is enough to reject Fix A on its own.

So Fix A is safe today only because `frangi`, `sato`, `meijering` and `hessian`
all start their sweep at `sigma = 1`. That is an accident of a default, not a
guarantee: `sigmas` is an ordinary documented parameter and nothing warns the
caller who lowers it.

`blob_doh` is the one scale-sweeping function this does *not* reach. It selects
scale by the same kind of maximum, but it computes the determinant through
`_hessian_matrix_det`, a box-filter approximation over integral images, so it
never touches this code path at all.

+++

## 11. Preferred fix, and what it changes

Take **A and D together**. A removes the cause of the border defect — a boundary
extension applied per call, so one call per element — and D removes the only
price A charges for it.

| | A, one call | B, pad once | A + C |
| --- | --- | --- | --- |
| Border error | zero | zero | zero |
| `order='rc'` vs `'xy'` | identical | still differs | identical |
| Interior, `sigma >= 0.75` | better | unchanged | better |
| Interior, `sigma < 0.65` | far worse | unchanged | better |
| Response to a constant | `-0.56` at `sigma = 0.5` | zero | ~1e-16 |
| Scale selection | collapses to the smallest sigma | unchanged | correct |
| Smooth in `sigma` | yes | yes | yes |
| Cost at `sigma = 1` | 0.75x | 3.8x | **0.05x** |
| `truncate = 100` needed | yes | yes | no |

A on its own should not ship. Its scale-selection row is not a small-`sigma`
inconvenience: the DC leak grows as `sigma` falls, so a scale sweep maximises at
the smallest scale offered whatever the image is. It is invisible today only
because the ridge filters happen to start at `sigma = 1`.

B reaches the same border numbers by paying for them — it leaves the ordering
inconsistency in place, needs a 143-pixel padded copy at `sigma = 1`, and keeps
the `truncate = 100` hack that makes the padding that large.

That `truncate = 100` line is worth restating, because it is referred to at
every stage above and only Fix C settles it. The hack was fighting the aliasing
by brute-force margin, and it is the single largest cost in the function at
`sigma <= 1`: it is what makes Fix B's padding 143 pixels wide, and what makes
Fix A slower than the shipped code at `sigma = 0.5`. Correcting the kernel
removes the reason for it, which is why A + C is twenty times faster than the
shipped code at `sigma = 1` rather than merely a little faster.

Here is what users would see change.

```{code-cell} ipython3
fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.0))
before = frangi(photo, **RIDGE)
with mock.patch.object(corner_mod, "hessian_matrix", as_hessian_matrix(fix_c)):
    after = frangi(photo, **RIDGE)
show(axes[0], before, "frangi today")
show(axes[1], after, "frangi with A + C")
im = show(axes[2], np.abs(before - after), "what changes")
fig.colorbar(im, ax=axes[2], fraction=0.046)
fig.suptitle("the border band, and the interior at the smallest sigma", y=1.03)
fig.tight_layout()
```

The third panel is not confined to the edge, and it is worth being exact about
why. `frangi` sweeps `sigma = 1, 3, 5, 7, 9`. At `sigma = 3` and above the
interior is untouched; at `sigma = 1` it is not, because that is where the
shipped scheme still has interior error of its own.

```{code-cell} ipython3
print(f"{'sigma':>6}{'border':>10}{'interior, 20 px in':>21}"
      f"{'shipped error':>16}{'A + C error':>14}")
inner = (slice(20, -20),) * 2
for sigma in (0.5, 1.0, 1.5, 3.0):
    shipped_H = hessian_matrix(IMG, sigma=sigma, mode=MODE,
                               use_gaussian_derivatives=True)
    fixed_H = fix_c(IMG, sigma)
    sc = max(np.abs(e).max() for e in fixed_H)
    print(f"{sigma:>6}"
          f"{max(np.abs(a - b).max() for a, b in zip(shipped_H, fixed_H)) / sc:>9.2%}"
          f"{max(np.abs(a[inner] - b[inner]).max() for a, b in zip(shipped_H, fixed_H)) / sc:>20.2%}"
          f"{relative_error(hessian_matrix(blob, sigma=sigma, mode='nearest', use_gaussian_derivatives=True), sigma):>16.2%}"
          f"{relative_error(fix_c(blob, sigma, mode='nearest'), sigma):>14.2%}")
```

The last two columns say which way each move goes. At `sigma = 1.5` and above
the interior barely moves and only the border band changes. At `sigma = 1` and
below the interior does move, and it moves towards the analytic answer every
time: the shipped scheme is 92.27% wrong at `sigma = 0.5` where A + C is 0.67%.

```{code-cell} ipython3
print("summary of the change")
print(f"  {'border band, every element':<44}moves by up to "
      f"{max(np.abs(c - r).max() / np.abs(r).max() for c, r in zip(cur, ref)):.0%}")
print(f"  {'interior, sigma >= 1.5':<44}unchanged (< 0.1%)")
print(f"  {'interior, sigma <= 1':<44}moves, towards the analytic answer")
print(f"  {'order=xy against order=rc':<44}removed entirely")
print(f"  {'response to a constant image':<44}-0.56 -> 1e-16 at sigma = 0.5")
print(f"  {'scale selection':<44}unbiased from blob width 1 upwards")
print(f"  {'cost at sigma = 1':<44}about 0.05x")
print(f"  {'truncate = 100 hack':<44}no longer needed")
```

The border band changes by design, because it is currently wrong. The interior
changes at `sigma <= 1` are corrections rather than regressions, but they are
still changes users will see, so both belong in `skimage2` with a migration
note, not in a patch release.

`structure_tensor` needs none of it. It is correct as it stands, and its `order`
parameter is a pure relabelling, so the whole of this applies to
`hessian_matrix` alone.

`hessian_matrix_det` is the one caller that needs deciding separately, because
it has two paths and they fail differently.

```{code-cell} ipython3
from skimage.feature import hessian_matrix_det

print("hessian_matrix_det against its own once-extended reference")
print(f"{'sigma':>6}{'approximate=True':>20}{'approximate=False':>21}")
for sigma in (1.0, 2.0, 4.0):
    pad = 2 * int(8 * sigma + 0.5) + 1
    big = np.pad(photo, pad, mode="edge")
    row = []
    for approximate in (True, False):
        direct = hessian_matrix_det(photo, sigma=sigma, approximate=approximate)
        ref_det = hessian_matrix_det(big, sigma=sigma,
                                     approximate=approximate)[pad:-pad, pad:-pad]
        d = np.abs(direct - ref_det)
        row.append(d.max() / max(np.abs(ref_det).max(), 1e-12))
    print(f"{sigma:>6}{row[0]:>19.1%}{row[1]:>20.1%}")
```

`approximate=False` calls `hessian_matrix`, so it inherits everything above and
A + C repairs it along with the rest. `approximate=True` is a different
algorithm — box filters over integral images — with a border problem of its own,
larger still, that none of these fixes touches and that this notebook has not
diagnosed.

`blob_doh` always takes the approximate path, so it is untouched by A + C in
either direction.

## 12. Reproducing `order='xy'` exactly

Section 11 is about the number that *should* change.  This section is about the
number that should not: `order=` is slated for removal, and a caller who passes
`order='xy'` today needs some way to keep the answer they have.  There are
about forty-one files on GitHub calling `hessian_matrix(..., 'xy')`, so the
question is worth a precise answer rather than a migration note.

There are two obvious shims and only one of them works.

**Reversing the returned list** is what the name `order` suggests, and it is
what section 6 already measured failing.  It is worth seeing exactly *where* it
fails, because the headline number hides it.

```{code-cell} ipython3
names = ("Hxx", "Hxy", "Hyy")

print("order='xy' against order='rc' reversed, per element")
print(f"{'sigma':>8}{'element':>10}{'absolute':>12}{'of its own range':>19}")
for sigma in (1, 3, (3, 4)):
    got = hessian_matrix(IMG, sigma=sigma, mode=MODE, order="xy",
                         use_gaussian_derivatives=True)
    rev = hessian_matrix(IMG, sigma=sigma, mode=MODE, order="rc",
                         use_gaussian_derivatives=True)[::-1]
    for name, a, b in zip(names, got, rev):
        gap = float(np.max(np.abs(a - b)))
        print(f"{str(sigma):>8}{name:>10}{gap:>12.2e}"
              f"{gap / float(np.max(np.abs(a))):>18.1%}")
```

Both diagonals come back **exactly**: reversing the list simply swaps `Hrr` and
`Hcc`, and neither is affected. All the error lands on the mixed element, and
it is not a rounding difference — it is half the element or more. Section 7
says why: the mixed element is the only one built by differentiating along *two
different* axes in sequence, so it is the only one whose answer depends on
which axis went first.

How large depends on the image. `IMG` above is uniform random, so it carries
structure right up to the border where the defect lives. On the `coins`
photograph the same measurement gives 20% to 38% — smaller, and still far too
large for a shim.

**Transposing the frame** works, and section 7 says why that too. Run the whole
computation on the transposed image and transpose the results back. Every axis
is then extended in the same frame it was extended in before, so the operation
the old code performed is reproduced rather than approximated.

```{code-cell} ipython3
def as_xy(func, image, **kwargs):
    """Reproduce `order='xy'` from an implementation that only does 'rc'."""
    if "sigma" in kwargs and not np.isscalar(kwargs["sigma"]):
        # A sequence sigma is per axis, so it reverses with the axes.
        kwargs = dict(kwargs, sigma=tuple(kwargs["sigma"])[::-1])
    transposed = func(np.transpose(image), order="rc", **kwargs)
    return [np.transpose(h) for h in transposed]
```

It is exact, and it stays exact across the whole parameter space that matters —
both functions, both `use_gaussian_derivatives` settings, every boundary mode,
and an anisotropic `sigma`.

```{code-cell} ipython3
def shim_error(func, image, **kwargs):
    """Worst relative disagreement between real 'xy' and the transposed shim."""
    got = func(image, order="xy", **kwargs)
    shimmed = as_xy(func, image, **kwargs)
    scale = max(float(np.max(np.abs(a))) for a in got)
    return max(float(np.max(np.abs(a - b))) for a, b in zip(got, shimmed)) / scale


print(f"{'function':>18}{'setting':>26}{'relative error':>17}")
for ugd in (True, False):
    for mode in ("constant", "reflect", "nearest", "wrap", "mirror"):
        err = shim_error(hessian_matrix, IMG, sigma=(3, 4), mode=mode,
                         use_gaussian_derivatives=ugd)
        print(f"{'hessian_matrix':>18}{f'ugd={ugd}, mode={mode}':>26}{err:>17.1e}")
for mode in ("constant", "reflect", "nearest"):
    err = shim_error(structure_tensor, IMG, sigma=(3, 4), mode=mode)
    print(f"{'structure_tensor':>18}{f'mode={mode}':>26}{err:>17.1e}")
```

Floating-point noise throughout. Note the `sigma` reversal inside `as_xy`: a
sequence `sigma` is one value per axis, so transposing the image without
reversing it would silently apply the wrong width to each axis. That is the one
part of the shim that is easy to leave out and hard to notice, because it only
shows up for anisotropic smoothing.

For `structure_tensor`, and for `hessian_matrix` with
`use_gaussian_derivatives=False`, reversing the list is *also* exact — those
paths have no per-call boundary extension to get wrong, so they are already
transpose-equivariant. The transposed shim is the one that covers every case,
so it is the one to document.

**What this does not resolve.** The transposed shim reproduces what SK1 returns
today, border artefact included. Fixes A and C in section 10 change that border
deliberately, so a caller cannot have both the old numbers and the corrected
ones. The shim is for callers who need bit-compatibility during a migration; it
is not a recommendation to keep computing the old values.
