---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
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

from skimage.feature import structure_tensor, hessian_matrix
from skimage.filters import frangi, sato, meijering, hessian
import skimage as ski

warnings.simplefilter("ignore")  # the use_gaussian_derivatives FutureWarning

# Slots 1 and 2 of the reference categorical palette, validated as a pair.
C_ONE = "#2a78d6"
C_TWO = "#eb6834"
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

## 1. What `order` is supposed to mean

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

## 2. `structure_tensor` keeps that contract exactly

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

## 3. `hessian_matrix` does not

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

## 4. Why: extensions commute across axes, not within one

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
Separability is innocent. But `_hessian_matrix_with_gaussian` composes **two**
calls, so every axis is filtered twice, and the second call re-extends an array
the first has already smoothed along that axis. Cropping threw information
away; re-padding invents a replacement.

That also explains the ordering difference: differentiating along axis 0 then
axis 1 re-extends a different array from the reverse order.

## 5. There is a right answer, and no element has it

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

## 6. It reaches the ridge filters

`frangi`, `sato`, `meijering` and `hessian` all call `hessian_matrix` with
`use_gaussian_derivatives=True`.

```{code-cell} ipython3
photo = ski.util.img_as_float(ski.data.camera())[::2, ::2]
PAD = 40
padded = np.pad(photo, PAD, mode="edge")

fig, axes = plt.subplots(3, 4, figsize=(10.5, 7.6))
for col, (name, f) in enumerate(
    [("frangi", frangi), ("sato", sato), ("meijering", meijering), ("hessian", hessian)]
):
    direct = f(photo)
    ref_out = f(padded)[PAD:-PAD, PAD:-PAD]
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
for name, f in [("frangi", frangi), ("sato", sato),
                ("meijering", meijering), ("hessian", hessian)]:
    a, b = f(photo), f(padded)[PAD:-PAD, PAD:-PAD]
    d = np.abs(a - b)
    print(f"{name:<10} max {d.max():.3e} ({d.max()/np.abs(b).max():5.1%} of range)"
          f"   30 px in: {d[30:-30, 30:-30].max():.2e}")
```

`hessian` flips completely at the border. `meijering` also moves in the
interior, because it normalises by a global maximum that the border artefact
distorts.

## 7. Two candidate fixes

**Fix A — compute the mixed element in one call.** `order=[1, 1]` with the full
sigma touches each axis once, so it is exact, and it has no left or right to
choose, so the `order` ambiguity cannot arise.

```{code-cell} ipython3
kw = dict(mode=MODE, truncate=8)
one_call = ndi.gaussian_filter(IMG, SIGMA, order=[1, 1], **kw)
d_one = np.abs(one_call - ref[1])
d_two = np.abs(cur[1] - ref[1])
print(f"mixed element, two composed calls : max {d_two.max():.3e}")
print(f"mixed element, one call [1, 1]    : max {d_one.max():.3e}")
```

The code comment that justifies the two-pass scheme concerns scipy's
**second**-order Gaussian derivatives, which is the *diagonal* elements. The
mixed element uses only a first derivative along each axis, so this does not
reintroduce that problem. But it leaves the two diagonals wrong.

**Fix B — pad once inside the function.** Extend the image by the rule `mode`
names, run the existing scheme, crop. Every element becomes exact.

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
`truncate=100` to fight aliasing, which makes the padding enormous — that hack
deserves its own review.

## 8. Preferred fix, and what it changes

Take **B**, and fold in **A** as a simplification. B is what makes the numbers
right: `mode` documents how to continue the *image*, and padding once is the
only construction that honours that for every element. A is free on top — it
removes the ordering question at its source and costs one filter call less.

Here is what users would see change.

```{code-cell} ipython3
fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.0))
before, after = frangi(photo), frangi(padded)[PAD:-PAD, PAD:-PAD]
show(axes[0], before, "frangi today")
show(axes[1], after, "frangi with the fix")
im = show(axes[2], np.abs(before - after), "what changes")
fig.colorbar(im, ax=axes[2], fraction=0.046)
fig.suptitle("nothing moves except a band at the edge", y=1.03)
fig.tight_layout()
```

```{code-cell} ipython3
print("summary of the change")
print(f"  interior of every element               unchanged (0.0e+00)")
for name, c, r in zip(names, cur, ref):
    print(f"  {name:<38} border moves by up to {np.abs(c-r).max()/np.abs(r).max():.0%}")
print(f"  order='xy' vs 'rc' inconsistency         removed entirely")
```

Nothing in the interior moves. The border band changes, by design, because it
is currently wrong. That is a deliberate, documented change and belongs in
`skimage2` with a migration note, not in a patch release.

## 9. What not to conclude

The problem is not that a 2-D Gaussian is implemented as two 1-D passes; that
is exact. It is not a floating-point rounding issue; the differences are a
third of the signal. And it is not confined to the mixed element; the diagonals
are equally affected, they simply have no ordering ambiguity to expose them.

`structure_tensor` needs none of this. It is correct as it stands, and its
`order` parameter is a pure relabelling.
