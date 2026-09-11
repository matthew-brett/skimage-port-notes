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

# On the blob detectors

`skimage.feature` offers three blob detectors — `blob_dog`, `blob_log` and
`blob_doh` — with matching signatures and a shared output format. They are
presented as three routes to one answer, differing in speed.

They differ in more than speed. Measured against discs of known radius, one is
accurate to a few per cent, one reports radii a quarter too small, and one
reports them up to a third too large. This notebook explains where each
difference comes from, and which are defects rather than documented
approximations.

Coordinates are in array order throughout: the first index runs down the
picture, the second runs right.

```{code-cell} ipython3
import time

import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
```

```{code-cell} ipython3
import skimage as ski
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.transform import integral_image
from _skimage2.feature._hessian_det_appx import _hessian_matrix_det as box_det
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
SEQ = LinearSegmentedColormap.from_list("seq", ["#f7f7f4", C_ONE])

plt.rcParams.update(
    {"figure.dpi": 110, "font.size": 9, "axes.titlesize": 9,
     "axes.titlecolor": MUTED, "figure.facecolor": "white"}
)


def bare(ax, title=None):
    """Strip an axis down to the data."""
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title is not None:
        ax.set_title(title)
    return ax


def recede(ax, title=None):
    """Keep the ticks, but make the frame recede."""
    ax.tick_params(labelsize=8, colors=MUTED)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    if title is not None:
        ax.set_title(title)
    return ax
```

## 1. A test image with known answers

Four discs, well separated, with radii spanning the useful range.

```{code-cell} ipython3
N = 400
rows, cols = np.indices((N, N))
TRUTH = [(80, 80, 6), (80, 220, 10), (220, 80, 16), (220, 260, 24)]

discs = np.zeros((N, N))
for r0, c0, radius in TRUTH:
    discs[(rows - r0) ** 2 + (cols - c0) ** 2 <= radius**2] = 1.0

fig, ax = plt.subplots(figsize=(3.2, 3.2))
bare(ax, "four discs, radii 6, 10, 16 and 24")
ax.imshow(discs, cmap="gray")
fig.tight_layout()
```

## 2. What a blob detector does

Each detector builds a **scale-normalised response** — a filter applied at many
scales, scaled so that responses at different scales are comparable — and then
looks for maxima in the three-dimensional stack of position and scale. The
position of a maximum gives the blob's centre, and its scale gives the size.

That normalisation is the whole trick. A plain second derivative gets smaller as
`sigma` grows, so without it the smallest scale always wins. Multiplying by the
right power of `sigma` cancels that, and the power differs by filter: `sigma**2`
for a Laplacian, `sigma**4` for a Hessian determinant.

The three detectors differ in which filter they use.

| | filter | documented radius |
| --- | --- | --- |
| `blob_log` | Laplacian of Gaussian, `sigma**2 * div(grad(G)) * f` | `sqrt(2) * sigma` |
| `blob_dog` | difference of two Gaussians, approximating the above | `sqrt(2) * sigma` |
| `blob_doh` | determinant of the Hessian, by box filters | `sigma` |

The radius conventions really do differ, and they are documented that way. Every
comparison below uses each detector's own convention, taken from its docstring.

```{code-cell} ipython3
CONVENTION = {"blob_dog": np.sqrt(2), "blob_log": np.sqrt(2), "blob_doh": 1.0}
```

## 3. What each one reports

```{code-cell} ipython3
found = {
    "blob_dog": blob_dog(discs, min_sigma=2, max_sigma=30, threshold=0.05),
    "blob_log": blob_log(discs, min_sigma=2, max_sigma=30, num_sigma=30,
                         threshold=0.05),
    "blob_doh": blob_doh(discs, min_sigma=2, max_sigma=30, num_sigma=30,
                         threshold=0.005),
}


def matched(name, r0, c0, window=400):
    """The detection nearest a known centre, as a radius in pixels."""
    near = [b for b in found[name] if (b[0] - r0) ** 2 + (b[1] - c0) ** 2 < window]
    if not near:
        return np.nan
    best = min(near, key=lambda b: (b[0] - r0) ** 2 + (b[1] - c0) ** 2)
    return best[2] * CONVENTION[name]


print(f"{'true radius':>12}" + "".join(f"{n:>22}" for n in found))
for r0, c0, radius in TRUTH:
    line = f"{radius:>12}"
    for name in found:
        got = matched(name, r0, c0)
        line += f"{got:>13.1f} ({(got - radius) / radius:+.0%})"
    print(line)
print()
print("blobs found, against four present:",
      {n: len(b) for n, b in found.items()})
```

```{code-cell} ipython3
fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.4))
for ax, (name, blobs) in zip(axes, found.items()):
    bare(ax, name)
    ax.imshow(discs, cmap="gray")
    for r0, c0, radius in TRUTH:
        ax.add_patch(plt.Circle((c0, r0), radius, fill=False, color=C_THREE,
                                lw=1.2, ls=":"))
    for b in blobs:
        ax.add_patch(plt.Circle((b[1], b[0]), b[2] * CONVENTION[name],
                                fill=False, color=C_TWO, lw=1.4))
fig.suptitle("dotted green: the true disc.   orange: what the detector reports",
             y=1.03)
fig.tight_layout()
```

`blob_log` traces the discs closely. `blob_dog` sits consistently inside them and
`blob_doh` consistently outside. Neither of those is noise: both biases are
systematic and repeat at every radius.

## 4. `blob_dog`: a documented approximation

The difference of two Gaussians at `sigma` and `k * sigma` approximates the
scale-normalised Laplacian, and the approximation improves as `k` approaches 1.
`blob_dog` reports the *smaller* of the pair, so the reported `sigma` sits below
the scale the response actually peaked at.

```{code-cell} ipython3
print(f"{'sigma_ratio k':>15}{'reported radius for the r=16 disc':>36}")
for ratio in (1.6, 1.4, 1.2, 1.1, 1.05):
    got = blob_dog(discs, min_sigma=2, max_sigma=30, sigma_ratio=ratio,
                   threshold=0.05)
    near = [b for b in got if (b[0] - 220) ** 2 + (b[1] - 80) ** 2 < 400]
    radius = near[0][2] * np.sqrt(2) if near else np.nan
    print(f"{ratio:>15}{radius:>25.1f} ({(radius - 16) / 16:+.0%})")
```

The bias falls from -28% at the default to about -2%, which identifies it: it
is the `sigma_ratio` approximation, behaving as the theory says it should. It is
not monotone in `k`, because changing `sigma_ratio` also changes which discrete
scales get computed. `sigma_ratio`
defaults to 1.6, the value
[Lowe uses in SIFT](https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf), chosen to
trade accuracy for the number of scales that must be computed.

So `blob_dog`'s error is a parameter the caller controls, documented in the
signature. It is a deliberate approximation, not a defect. OpenCV's `SIFT`
makes the same trade for the same reason.

## 5. `blob_doh`: box filters over an integral image

`blob_doh` is the one that needs explaining. It never computes a Gaussian
derivative at all. It builds an **integral image**, in which every pixel holds
the sum of everything above and to the left, so the sum over any rectangle costs
four lookups regardless of its size. It then approximates each second derivative
with a small number of rectangles.

That is [SURF](https://doi.org/10.1007/11744023_32), and the point of it is that
the cost does not grow with `sigma`. The docstring says so directly: "Computation
of Determinant of Hessians is independent of the standard deviation."

The geometry is in `_hessian_det_appx_pythran.py`:

```
size = int(3 * sigma)       # the filter's full width
s3   = size // 3            # the lobe
w_i  = 1.0 / size / size    # normalise by area
...
dxx  = mid - 3 * side       # a (1, -2, 1) arrangement of boxes
out[r, c] = dxx * dyy - 0.81 * (dxy * dxy)
```

`mid` and `side` are rectangle sums, so `mid - 3 * side` is the box-filter
stand-in for a second derivative, and `0.81` is SURF's correction for the
diagonal term. Dividing by `size**2` is what makes the responses comparable
across scale, which is why nothing multiplies by `sigma**4` afterwards.

Here is the shape that replaces the second derivative of a Gaussian.

```{code-cell} ipython3
def box_profile(sigma):
    """The 1-D weight profile of `dxx`, from the source's own index arithmetic.

    `mid` spans `w = size` columns from `c - s2`; `side` spans `s3` columns
    from `c - s3 // 2`; and `dxx = mid - 3 * side`, normalised by `size ** 2`.
    """
    size = int(3 * sigma)
    s3, s2, w = size // 3, (size - 1) // 2, size
    span = size
    x = np.arange(-span, span + 1)
    profile = np.zeros_like(x, dtype=float)
    inside = lambda start, length: (x >= start) & (x < start + length)
    profile[inside(-s2, w)] += 1.0
    profile[inside(-(s3 // 2), s3)] -= 3.0
    return x, profile / size**2


def gaussian_d2(sigma, span):
    x = np.arange(-span, span + 1).astype(float)
    g = np.exp(-(x**2) / (2 * sigma**2))
    g /= g.sum()
    return x, g * ((x**2 - sigma**2) / sigma**4)


fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.6), sharey=False)
for ax, sigma in zip(axes, (2.0, 4.0, 8.0)):
    bx, bp = box_profile(sigma)
    gx, gp = gaussian_d2(sigma, int(3 * sigma))
    ax.plot(gx, gp / np.abs(gp).max(), color=C_ONE, lw=2, label="Gaussian d2")
    ax.step(bx, bp / np.abs(bp).max(), color=C_TWO, lw=1.8, where="mid",
            label="box filter")
    recede(ax, f"sigma = {sigma}")
    ax.axhline(0, color=GRID, lw=0.8)
axes[0].legend(frameon=False, fontsize=7)
fig.suptitle("what the box filter puts in place of a second derivative", y=1.04)
fig.tight_layout()
```

The box filter has the right gross shape — negative centre, positive flanks —
and the wrong everything else: hard edges, no tails, and a width fixed by
integer arithmetic rather than by `sigma`.

It is also **not centred**. `mid` starts at `c - s2` with `s2 = (size - 1) // 2`
and runs `size` columns; `side` starts at `c - s3 // 2` and runs `s3`. Where
`size` or `s3` is even, neither box can straddle the centre pixel, so the whole
filter sits half a pixel off. The panels above show it: at `sigma = 2` the
negative lobe spans roughly `-1.5` to `+0.5` rather than `-1` to `+1`.

A second-derivative filter that is not symmetric reports its maxima in the wrong
place, and that reaches the caller as a position error rather than a size error.

```{code-cell} ipython3
DETECTORS = {"blob_dog": lambda im: blob_dog(im, min_sigma=2, max_sigma=30,
                                             threshold=0.05),
             "blob_log": lambda im: blob_log(im, min_sigma=2, max_sigma=30,
                                             num_sigma=30, threshold=0.05),
             "blob_doh": lambda im: blob_doh(im, min_sigma=2, max_sigma=30,
                                             num_sigma=30, threshold=0.005)}

print("offset of the reported centre from the true centre, in pixels")
print(f"{'true radius':>12}" + "".join(f"{n:>20}" for n in DETECTORS))
for r0, c0, radius in TRUTH:
    line = f"{radius:>12}"
    for name, run in DETECTORS.items():
        near = [b for b in run(discs)
                if (b[0] - r0) ** 2 + (b[1] - c0) ** 2 < 400]
        if not near:
            line += f"{'missed':>20}"
            continue
        b = min(near, key=lambda b: (b[0] - r0) ** 2 + (b[1] - c0) ** 2)
        line += f"{f'{b[0] - r0:+.0f}, {b[1] - c0:+.0f}':>20}"
    print(line)
```

`blob_dog` and `blob_log` land on the centre exactly. `blob_doh` is one pixel up
and one pixel left, at every radius. Over random centres and radii the column
offset is `-1` every time and the row offset is `-1` almost every time.

```{code-cell} ipython3
rng = np.random.default_rng(0)
small_n = 300
sr, sc = np.indices((small_n, small_n))
offsets = []
for _ in range(12):
    r0, c0 = int(rng.integers(60, 240)), int(rng.integers(60, 240))
    radius = int(rng.integers(5, 20))
    one = np.zeros((small_n, small_n))
    one[(sr - r0) ** 2 + (sc - c0) ** 2 <= radius**2] = 1.0
    near = [b for b in blob_doh(one, min_sigma=2, max_sigma=30, num_sigma=30,
                                threshold=0.005)
            if (b[0] - r0) ** 2 + (b[1] - c0) ** 2 < 900]
    if near:
        b = min(near, key=lambda b: (b[0] - r0) ** 2 + (b[1] - c0) ** 2)
        offsets.append((b[0] - r0, b[1] - c0))

offsets = np.array(offsets)
print(f"{len(offsets)} blobs over random centres and radii")
print(f"   distinct offsets : {sorted(set(map(tuple, offsets)))}")
print(f"   mean offset      : {offsets.mean(axis=0)}")
```

A constant offset is the easiest kind of defect to fix and the easiest to miss,
because every blob moves together and the picture still looks right.

## 6. The scale axis is quantised

`size = int(3 * sigma)` is the whole dependence on `sigma`, and it is an
integer. Distinct scales therefore collapse onto the same filter.

```{code-cell} ipython3
photo = ski.util.img_as_float(ski.data.camera())[::2, ::2]
table = np.ascontiguousarray(integral_image(photo))
base = np.asarray(box_det(table, 3.0))

print(f"{'sigma':>8}{'int(3*sigma)':>14}{'identical to sigma = 3.0':>28}")
for sigma in (3.0, 3.2, 3.32, 3.34, 3.67, 4.0):
    same = np.array_equal(np.asarray(box_det(table, float(sigma))), base)
    print(f"{sigma:>8}{int(3 * sigma):>14}{str(same):>28}")
```

Asking for `sigma = 3.0`, `3.2` or `3.32` returns bit-identical planes. The
scale axis moves in steps of one third, so a `num_sigma` finer than that buys
duplicate work and nothing else — and the duplicates still count towards the
maximum search.

Below `sigma = 1` the lobe collapses to zero and the response vanishes
altogether.

```{code-cell} ipython3
small = np.exp(-((rows[:81, :81] - 40) ** 2 + (cols[:81, :81] - 40) ** 2) / 32)
small_table = np.ascontiguousarray(integral_image(small))
print(f"{'sigma':>7}{'size':>6}{'lobe':>6}{'centre response':>18}")
for sigma in (0.5, 0.9, 1.0, 2.0, 4.0):
    size = int(3 * sigma)
    got = np.asarray(box_det(small_table, float(sigma)))[40, 40]
    print(f"{sigma:>7}{size:>6}{size // 3:>6}{got:>18.3e}")
```

The docstring warns that the method "can't be used for detecting blobs of radius
less than 3px", which is this. It is documented, and it is a hard floor rather
than a gradual loss.

## 7. Scale selection, measured

The test that matters is whether the response peaks at the right scale. For
comparison, here is the same determinant computed exactly, with the corrected
Gaussian kernels of `on_hessian.md`.

```{code-cell} ipython3
def gaussian_taps(sigma, order, trunc=8):
    lw = int(trunc * sigma + 0.5)
    x = np.arange(-lw, lw + 1).astype(float)
    g = np.exp(-(x**2) / (2 * sigma**2))
    g /= g.sum()
    if order == 0:
        return x, g
    if order == 1:
        return x, g * (x / sigma**2)
    return x, g * ((x**2 - sigma**2) / sigma**4)


def corrected_taps(sigma, order, trunc=8):
    """Fix C of `on_hessian.md`: repair the discrete moments."""
    x, g = gaussian_taps(sigma, 0, trunc)
    _, k = gaussian_taps(sigma, order, trunc)
    if order == 0:
        return x, g
    if order == 1:
        return x, k / (k * x).sum()
    k = k - k.sum() * g
    return x, k / ((k * x**2).sum() / 2)


def exact_doh(image, sigma, mode="nearest", trunc=8):
    """Scale-normalised determinant of the Hessian, exact kernels."""
    taps = {o: corrected_taps(sigma, o, trunc)[1] for o in (0, 1, 2)}
    sep = lambda a, b: ndi.correlate1d(
        ndi.correlate1d(image, taps[a], axis=0, mode=mode), taps[b], axis=1, mode=mode)
    hrr, hrc, hcc = sep(2, 0), sep(1, 1), sep(0, 2)
    return (sigma**4) * (hrr * hcc - hrc**2)


def box_doh(image, sigma):
    """As `blob_doh` calls it: on the integral image."""
    return np.asarray(box_det(np.ascontiguousarray(integral_image(image)),
                              float(sigma)))
```

```{code-cell} ipython3
M = 201
mc = M // 2
mr2 = (np.indices((M, M))[0] - mc) ** 2 + (np.indices((M, M))[1] - mc) ** 2
trial = np.geomspace(1.0, 14.0, 50)


def gaussian_blob(width):
    return np.exp(-mr2 / (2 * width**2)) / (2 * np.pi * width**2)


print(f"{'blob width':>12}{'ideal':>8}{'box filters':>14}{'exact':>9}")
for width in (1.5, 2.0, 3.0, 5.0, 8.0):
    target = gaussian_blob(width)
    picks = [trial[int(np.argmax([f(target, s)[mc, mc] for s in trial]))]
             for f in (box_doh, exact_doh)]
    print(f"{width:>12}{width:>8.2f}{picks[0]:>14.2f}{picks[1]:>9.2f}")
```

```{code-cell} ipython3
fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2), sharey=True)
for ax, (name, f) in zip(axes, (("box filters", box_doh), ("exact", exact_doh))):
    for width, colour in ((1.5, C_ONE), (3.0, C_TWO), (8.0, C_THREE)):
        target = gaussian_blob(width)
        resp = np.array([f(target, s)[mc, mc] for s in trial])
        resp = resp / np.abs(resp).max()
        ax.plot(trial, resp, color=colour, lw=1.9, label=f"width {width}")
        ax.axvline(width, color=colour, lw=0.9, ls=":")
        ax.plot([trial[int(np.argmax(resp))]], [resp.max()], "o", color=colour,
                markersize=6, markeredgecolor="white", zorder=5)
    ax.set_xscale("log")
    ax.set_xlabel("sigma", fontsize=8, color=MUTED)
    recede(ax, name)
axes[0].set_ylabel("normalised response", fontsize=8, color=MUTED)
axes[0].legend(frameon=False, fontsize=7, loc="upper left")
fig.suptitle("dotted line: the blob's true width.   circle: where the method "
             "puts it", y=1.04)
fig.tight_layout()
```

The exact determinant lands on each dotted line. The box version reads high and
increasingly so, running into the top of the axis for the widest blob. That is
the `+5%` to `+33%` of section 3, seen at its source.

## 8. The border

```{code-cell} ipython3
print(f"{'sigma':>6}{'box filters':>14}{'exact':>10}")
for sigma in (2.0, 4.0, 8.0):
    pad = 2 * int(8 * sigma + 0.5) + 1
    big = np.pad(photo, pad, mode="edge")
    row = ""
    for f in (box_doh, exact_doh):
        reference = f(big, sigma)[pad:-pad, pad:-pad]
        gap = np.abs(f(photo, sigma) - reference)
        row += f"{gap.max() / max(np.abs(reference).max(), 1e-12):>13.1%}"
    print(f"{sigma:>6}{row}")
```

The exact route is exact at the border, for the reason `on_hessian.md` section 7
sets out: one filter call per element, so the boundary rule is applied once. The
box route has no boundary rule at all — `_integ` clamps rectangle corners to the
image, which silently shrinks the filter near an edge and changes what it
computes.

## 9. What the approximation buys

```{code-cell} ipython3
def best_of(f, n=5):
    f()
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        f()
        times.append(time.perf_counter() - t0)
    return min(times)


print(f"{'sigma':>6}{'box filters':>14}{'exact':>11}{'ratio':>9}")
for sigma in (2.0, 4.0, 8.0, 16.0):
    t_box = best_of(lambda: box_doh(photo, sigma))
    t_exact = best_of(lambda: exact_doh(photo, sigma))
    print(f"{sigma:>6}{t_box * 1e3:>11.2f} ms{t_exact * 1e3:>8.2f} ms"
          f"{t_exact / t_box:>8.1f}x")
```

This is the trade, and it is a real one. The box cost is flat: the same
rectangle count whatever `sigma` is. The exact cost grows linearly with `sigma`,
because the kernel does. At `sigma = 16` the gap is already fortyfold and it
keeps widening.

So replacing the box filters with exact kernels would fix the scale bias and the
border, and destroy the reason `blob_doh` exists. `blob_log` is already the
accurate Gaussian-based detector; a `blob_doh` that computed exact Gaussian
derivatives would be a slower `blob_log` under a different name.

## 10. Do the `hessian_matrix` fixes reach it?

No. `on_hessian.md` finds a border defect in `hessian_matrix` that reaches
`frangi`, `sato`, `meijering` and `hessian`, and proposes fixes A and C. None of
them touches `blob_doh`, which calls `_hessian_matrix_det` directly.

`hessian_matrix_det` is the function to watch, because it has two paths.

```{code-cell} ipython3
from skimage.feature import hessian_matrix_det

print(f"{'sigma':>6}{'approximate=True':>20}{'approximate=False':>21}")
for sigma in (1.0, 2.0, 4.0):
    pad = 2 * int(8 * sigma + 0.5) + 1
    big = np.pad(photo, pad, mode="edge")
    row = ""
    for approximate in (True, False):
        direct = hessian_matrix_det(photo, sigma=sigma, approximate=approximate)
        reference = hessian_matrix_det(big, sigma=sigma,
                                       approximate=approximate)[pad:-pad, pad:-pad]
        gap = np.abs(direct - reference)
        row += f"{gap.max() / max(np.abs(reference).max(), 1e-12):>19.1%}"
    print(f"{sigma:>6}{row}")
```

`approximate=False` routes through `hessian_matrix`, so it inherits the border
defect and fixes A and C repair it. `approximate=True` is the box route, with a
larger border problem of its own that no fix in `on_hessian.md` addresses.

## 11. How other libraries do it

OpenCV has no direct equivalent of this trio. `cv2.SimpleBlobDetector` is not a
scale-space method at all — it thresholds the image at a series of levels and
groups the connected components, so it finds regions rather than scales.

```{code-cell} ipython3
params = cv2.SimpleBlobDetector_Params()
params.filterByArea, params.minArea, params.maxArea = True, 50, 5000
params.filterByCircularity = params.filterByConvexity = False
params.filterByInertia = params.filterByColor = False
detector = cv2.SimpleBlobDetector_create(params)

keypoints = detector.detect((discs * 255).astype(np.uint8))
print(f"{'true radius':>12}{'SimpleBlobDetector':>22}")
for r0, c0, radius in TRUTH:
    near = [k for k in keypoints
            if (k.pt[1] - r0) ** 2 + (k.pt[0] - c0) ** 2 < 400]
    got = near[0].size / 2 if near else np.nan
    print(f"{radius:>12}{got:>13.1f} ({(got - radius) / radius:+.0%})")
```

It recovers the radii of solid discs well, because a disc is exactly what it
looks for. It has no scale-normalised response, so it cannot rank a blob's
strength across scale, and it does not generalise to blobs that are not
threshold-separable from their surroundings.

`cv2.SIFT` uses a difference of Gaussians, like `blob_dog`, with the same
`sigma_ratio` trade for the same reason. The genuine SURF implementation, which
is what `blob_doh` reproduces, sits in `opencv-contrib` behind a build flag and
is not available here.

## 12. What to do

**`blob_dog`.** Nothing. Its bias is the documented `sigma_ratio` approximation,
it shrinks when the caller asks for it to, and it matches what SIFT does.

**`blob_log`.** Nothing. It is the accurate detector of the three.

**`blob_doh`.** Three separate things, worth separating.

The **scale bias** of `+5%` to `+33%` is a calibration question. `size =
int(3 * sigma)` ties the filter width to `sigma` by a constant that does not
reproduce SURF's own relation between filter size and Gaussian scale. Fixing the
constant is cheap and changes no algorithm, but it changes output, so it belongs
in `skimage2` with a note.

The **quantised scale axis** is worth documenting rather than fixing. `num_sigma`
finer than one third silently computes duplicate planes, and a caller has no way
to know. A line in the docstring, or rounding `sigma_list` to the realisable
grid, would prevent the waste.

The **border** is the one to leave alone for now. It is larger than the
`hessian_matrix` defect, it has a different cause, and this notebook has
measured it without diagnosing it.

Replacing the box filters with exact kernels is not the answer to any of these.
It would fix the first and third and cost the property the function exists for.

Measured with scikit-image from this working tree, OpenCV 5.0.0, on 400x400
synthetic discs and the 256x256 `camera` photograph.
