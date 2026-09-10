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

# On lines

How `skimage.draw.line` and `skimage.draw.line_nd` turn a line segment into
pixels, why they disagree, how they compare with Pillow and OpenCV, and what we
could change.

Throughout, coordinates are in **array order**: the first number indexes the
first array axis, which runs *down* the picture, and the second indexes the
second axis, which runs *right*. Nothing here uses `x` and `y`.

```{code-cell} ipython3
import itertools

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from matplotlib.colors import ListedColormap
```

```{code-cell} ipython3
from skimage.draw import line, line_nd
```

```{code-cell} ipython3
# Comparators.
import cv2
from PIL import Image, ImageDraw
```

```{code-cell} ipython3
# Slots 1 and 2 of the reference categorical palette, validated as a pair:
# CVD dE 24.7, normal-vision dE 33.6, both well clear of the floors.
C_LINE = "#2a78d6"   # skimage.draw.line
C_ND = "#eb6834"     # skimage.draw.line_nd
C_BOTH = "#c9c8c1"   # pixels that both algorithms choose
C_OFF = "#f2f1ec"    # pixels neither chooses
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#dedcd5"

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.titlecolor": MUTED,
        "figure.facecolor": "white",
    }
)
```

## Drawing helpers

One helper draws an empty pixel grid, one fills pixels, one overlays the ideal
segment. Everything below is built from these three.

```{code-cell} ipython3
def pixel_axes(ax, shape, title=None):
    """Draw an empty pixel grid, axis 0 downwards and axis 1 rightwards."""
    n_i, n_j = shape
    ax.set_xlim(-0.5, n_j - 0.5)
    ax.set_ylim(n_i - 0.5, -0.5)
    ax.set_xticks(range(n_j))
    ax.set_yticks(range(n_i))
    ax.set_xticks(np.arange(n_j + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n_i + 1) - 0.5, minor=True)
    ax.grid(which="minor", color=GRID, linewidth=0.8)
    ax.tick_params(which="both", length=0, labelsize=7, colors=MUTED)
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title is not None:
        ax.set_title(title)
    return ax


def fill(ax, pixels, color, alpha=1.0):
    """Fill each ``(axis 0, axis 1)`` pixel in `pixels`."""
    for i, j in sorted(pixels):
        ax.add_patch(
            Rectangle(
                (j - 0.5, i - 0.5),
                1,
                1,
                facecolor=color,
                edgecolor="white",
                linewidth=1.0,
                alpha=alpha,
                zorder=1,
            )
        )
    return ax


def exact(ax, p, q, color=INK):
    """Overlay the ideal segment from `p` to `q`, with its endpoints."""
    ax.plot([p[1], q[1]], [p[0], q[0]], color=color, linewidth=1.4, zorder=3)
    ax.plot([p[1], q[1]], [p[0], q[0]], "o", color=color, markersize=4, zorder=4)
    return ax


def as_set(coords):
    """A tuple of index arrays, as a set of ``(axis 0, axis 1)`` pairs."""
    return set(zip(*(np.asarray(a).tolist() for a in coords)))


def sk_line(p, q):
    return as_set(line(p[0], p[1], q[0], q[1]))


def sk_nd(p, q):
    return as_set(line_nd(p, q, endpoint=True))
```

## 1. The problem

A segment runs between two pixel centres. Almost every pixel it crosses is
crossed only partly, so a rasteriser has to choose. Here is the segment from
`(0, 0)` to `(1, 4)`, with the pixels it passes through at all.

```{code-cell} ipython3
p, q = (0, 0), (1, 4)
shape = (3, 6)

fig, ax = plt.subplots(figsize=(3.6, 2.0))
pixel_axes(ax, shape)
fill(ax, {(0, j) for j in range(5)} | {(1, j) for j in range(5)}, C_OFF)
exact(ax, p, q)
ax.set_title("the segment (0, 0) to (1, 4) and the pixels it touches")
fig.tight_layout()
```

The segment descends one row over four columns, so it passes exactly halfway
between two pixel rows at column 2. That single tie is the source of nearly
every disagreement in this document.

+++

## 2. `line`: integer Bresenham

`skimage.draw.line` is classic Bresenham, implemented in Cython
(`draw/_draw.pyx::_line`). It uses integer arithmetic only: no floats, no
division, and no rounding function anywhere.

+++

### The terms

Five quantities do all the work.

- **`delta`** — how far there is to travel on each axis, `abs(stop - start)`.
- **`step`** — which way to travel on each axis, `+1` or `-1`.
- **major axis** — the axis with the larger `delta`. The line advances one
  pixel along it on every iteration without exception, which is why the output
  holds exactly `delta[major] + 1` pixels.
- **minor axis** — the other one. It advances on some iterations and not
  others. Choosing which is the whole of the algorithm.
- **`error`** — an integer carrying how far the true line has drifted from the
  minor coordinate currently being drawn. Its sign is the decision.

The Cython source calls the axes `r` and `c`, and physically swaps them when
the line is steep so that the driving axis is always `c`. Indexing the axes
rather than swapping them says the same thing with less bookkeeping.

+++

### The algorithm

```{code-cell} ipython3
def bresenham(start, stop):
    """Bresenham's line, in the same steps as `skimage.draw.line`."""
    start, stop = np.array(start), np.array(stop)
    delta = np.abs(stop - start)
    step = np.sign(stop - start)

    major = int(np.argmax(delta))    # the axis with further to travel
    minor = 1 - major

    # Positive when the true line has passed the midpoint between the current
    # minor pixel and the next one. Scaled by 2 * delta[major] to stay integer.
    error = 2 * delta[minor] - delta[major]

    at = start.copy()
    pixels = []
    for _ in range(delta[major]):
        pixels.append(tuple(at))
        if error >= 0:
            at[minor] += step[minor]
            error -= 2 * delta[major]
        at[major] += step[major]
        error += 2 * delta[minor]

    pixels.append(tuple(stop))       # the endpoint is written, never computed
    return pixels
```

The last line matters: the endpoint is assigned rather than arrived at, so both
ends are always present whatever the arithmetic did on the way.

+++

### It is the same algorithm

Prose about a reimplementation is worth little. Check it against the real
function, over every integer endpoint pair in a 15 by 15 box, comparing the
pixel *sequence* and not merely the set.

```{code-cell} ipython3
R = range(-7, 8)
all_pairs = [((a, b), (c, d)) for a, b, c, d in itertools.product(R, repeat=4)]


def sk_sequence(p, q):
    ii, jj = line(p[0], p[1], q[0], q[1])
    return list(zip(ii.tolist(), jj.tolist()))


matches = sum(bresenham(p, q) == sk_sequence(p, q) for p, q in all_pairs)
print(f"identical to skimage.draw.line on {matches}/{len(all_pairs)} pairs"
      f"  ({matches / len(all_pairs):.1%})")
```

+++

### What `error` measures

Two counters appear in the explanation. Both count **whole pixel steps already
taken**, one per axis, measured from the start pixel:

- `k` — steps already taken along the **major** axis. The loop takes exactly
  one of these per iteration, so `k` is also the iteration number and the
  number of pixels already emitted.
- `taken` — steps already taken along the **minor** axis: an integer count of
  rows, or of columns, whichever axis the minor one happens to be. It is a
  count of pixels, not a distance and not a fraction.

Here they are, alongside the error, for a single line:

```{code-cell} ipython3
def trace(start, stop):
    """The two counters and the error, at each decision."""
    start, stop = np.array(start), np.array(stop)
    delta = np.abs(stop - start)
    major = int(np.argmax(delta))
    minor = 1 - major
    d_major, d_minor = int(delta[major]), int(delta[minor])

    error, taken, rows = 2 * d_minor - d_major, 0, []
    for k in range(d_major):
        rows.append((k, taken, error))
        if error >= 0:
            error -= 2 * d_major
            taken += 1
        error += 2 * d_minor
    return rows


print(f"{'k: major steps done':>21}{'taken: minor steps done':>26}{'error':>8}")
for k, taken, err in trace((0, 0), (2, 7)):
    print(f"{k:>21}{taken:>26}{err:>8}")
```

`taken` only ever rises by one, and only on the iterations where `error` was
not negative.

Both counters are positive whichever way the line runs. The error arithmetic
uses `delta` alone, which holds absolute distances, and direction enters only
through `step`. So the decision sequence is identical in every octant, under
transposition, and under translation, which means the derivation below can be
read as though the line ran down and to the right.

```{code-cell} ipython3
base = trace((0, 0), (2, 7))
elsewhere = {
    "up and left, (-2, -7)": ((0, 0), (-2, -7)),
    "down and left, (2, -7)": ((0, 0), (2, -7)),
    "transposed, (7, 2)": ((0, 0), (7, 2)),
    "translated by (5, 5)": ((5, 5), (7, 12)),
}
for name, (start, stop) in elsewhere.items():
    print(f"{name:<24} same sequence as (2, 7): {trace(start, stop) == base}")
```

The test asks whether the minor axis should step *during this iteration* —
that is, by the time the major axis has reached `k + 1`. At that point the true
line lies `(k + 1) * delta[minor] / delta[major]` pixels from the start along
the minor axis. The two candidates for the minor coordinate are `taken` and
`taken + 1`, so what decides between them is the midpoint, `taken + 0.5`.

`error` is exactly that overshoot, multiplied by `2 * delta[major]`:

```
error == 2 * delta[major] * ((k + 1) * delta[minor] / delta[major] - (taken + 0.5))
```

The multiplier is the trick that removes the division. Scaling by a positive
constant cannot change a sign, so the integer `error` decides the same
question the fraction would have, and `error >= 0` means the line has passed
the midpoint and the minor axis must step.

That is a claim about every iteration of every line, so test it as one.

```{code-cell} ipython3
def error_matches_overshoot(start, stop):
    """Is `error` the scaled midpoint overshoot at every decision?"""
    start, stop = np.array(start), np.array(stop)
    delta = np.abs(stop - start)
    step = np.sign(stop - start)
    major = int(np.argmax(delta))
    minor = 1 - major
    d_major, d_minor = int(delta[major]), int(delta[minor])

    error, taken = 2 * d_minor - d_major, 0
    for k in range(d_major):
        overshoot = (k + 1) * d_minor / d_major - (taken + 0.5)
        if abs(error - 2 * d_major * overshoot) > 1e-9:
            return False
        if error >= 0:
            error -= 2 * d_major
            taken += 1
        error += 2 * d_minor
    return True


agree = sum(error_matches_overshoot(p, q) for p, q in all_pairs)
print(f"error equals the scaled overshoot on {agree}/{len(all_pairs)} pairs")
```

+++

### Why one minor step is always enough

The Cython source writes the decision as `while d >= 0`, not `if`. The two are
the same here: the minor axis never has further to travel than the major one,
so it can never need two steps in one iteration. Since that is what licenses
the `if` above, check it rather than assume it.

```{code-cell} ipython3
def never_steps_twice(start, stop):
    """Would a second pass of the `while` body ever be taken?"""
    start, stop = np.array(start), np.array(stop)
    delta = np.abs(stop - start)
    d_major, d_minor = int(delta.max()), int(delta.min())
    error = 2 * d_minor - d_major
    for _ in range(d_major):
        if error >= 0:
            error -= 2 * d_major
            if error >= 0:
                return False
        error += 2 * d_minor
    return True


single = sum(never_steps_twice(p, q) for p, q in all_pairs)
print(f"one minor step per major step suffices on {single}/{len(all_pairs)} pairs")
```

+++

And the line it draws:

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(3.6, 2.0))
pixel_axes(ax, shape)
fill(ax, sk_line(p, q), C_LINE)
exact(ax, p, q, color="white")
ax.set_title("line(0, 0, 1, 4)")
fig.tight_layout()
```

## 3. `line_nd`: sample, then round each axis

`skimage.draw.line_nd` (`draw/draw_nd.py`) works differently. It computes how
many points it needs, samples the segment at that many equally spaced
parameters with `np.linspace`, and then rounds **each axis independently**.

```
npoints = ceil(max(abs(stop - start)))
coords  = linspace(start, stop, npoints).T
coords  = round(coords)        # per axis, via _round_safe
```

The rounding function is `np.round`, which rounds a half to the nearest **even**
integer. `_round_safe` patches one case of that: when the first coordinate is
exactly `.5` and the step is exactly 1, it falls back to `np.floor`, to stop
half-to-even opening a two-pixel gap.

`endpoint` is `False` by default, so the stop point is left out unless asked
for. Every comparison below passes `endpoint=True` so the point counts match.

```{code-cell} ipython3
samples = np.linspace(p, q, 5, endpoint=True)

fig, ax = plt.subplots(figsize=(3.6, 2.0))
pixel_axes(ax, shape)
fill(ax, sk_nd(p, q), C_ND)
exact(ax, p, q, color="white")
ax.plot(samples[:, 1], samples[:, 0], "o", color=INK, markersize=5, zorder=5)
for i, j in samples:
    ax.annotate(
        f"{i:g}",
        (j, i),
        textcoords="offset points",
        xytext=(0, 9),
        ha="center",
        fontsize=7,
        color=INK,
    )
ax.set_title("line_nd((0, 0), (1, 4)) with its sample points, labelled by exact row")
fig.tight_layout()
```

The label on the middle sample is the whole story: the exact row there is
`0.5`, an exact tie.

```{code-cell} ipython3
print(f"{'column':>7}{'exact row':>11}{'line':>7}{'line_nd':>9}")
lr = dict(zip(*[a.tolist() for a in line(0, 0, 1, 4)][::-1]))
nr = dict(zip(*[a.tolist() for a in line_nd(p, q, endpoint=True)][::-1]))
for j in range(5):
    print(f"{j:>7}{0 + j * 0.25:>11.2f}{lr[j]:>7}{nr[j]:>9}")
```

## 4. The tie, side by side

```{code-cell} ipython3
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.1))
for ax, pix, color, name in (
    (axes[0], sk_line(p, q), C_LINE, "line"),
    (axes[1], sk_nd(p, q), C_ND, "line_nd"),
):
    pixel_axes(ax, shape, name)
    fill(ax, pix, color)
    exact(ax, p, q, color="white")
fig.suptitle("at the tie, line steps early and line_nd steps late", y=1.02)
fig.tight_layout()
```

Bresenham's `d >= 0` test resolves the tie by stepping **early**. `np.round`
sends `0.5` to `0`, so `line_nd` steps **late**.

+++

## 5. How often do they disagree?

Fix the start at the top-left corner and vary the end over a box. Each cell is
coloured by whether the two functions produce the same pixels for that endpoint.

```{code-cell} ipython3
n = 13
agree = np.zeros((n, n), dtype=int)
for i in range(n):
    for j in range(n):
        agree[i, j] = sk_line((0, 0), (i, j)) == sk_nd((0, 0), (i, j))

fig, ax = plt.subplots(figsize=(3.4, 3.4))
ax.imshow(agree, cmap=ListedColormap([C_ND, C_OFF]), vmin=0, vmax=1)
ax.set_xticks(range(0, n, 2))
ax.set_yticks(range(0, n, 2))
ax.tick_params(length=0, labelsize=7, colors=MUTED)
for spine in ax.spines.values():
    spine.set_visible(False)
ax.set_xlabel("end, axis 1", fontsize=8, color=MUTED)
ax.set_ylabel("end, axis 0", fontsize=8, color=MUTED)
ax.set_title("start fixed at (0, 0)")
fig.legend(
    handles=[
        Patch(facecolor=C_OFF, label="same pixels"),
        Patch(facecolor=C_ND, label="different pixels"),
    ],
    loc="lower center",
    ncols=2,
    frameon=False,
    fontsize=8,
)
fig.tight_layout(rect=(0, 0.08, 1, 1))
```

The disagreements are not scattered: they lie along the directions whose slope
puts a sample exactly on a half. Over **every** integer endpoint pair in a
13x13 box, not just those from one corner:

```{code-cell} ipython3
# Every ordered pair of endpoints in a 9x9 box, kept clear of the canvas edge
# so that Pillow and OpenCV have room to draw in section 7.
LO, HI = 4, 13
pts = list(itertools.product(range(LO, HI), repeat=2))
box = [(a, b) for a in pts for b in pts]

diff = sum(sk_line(a, b) != sk_nd(a, b) for a, b in box)
print(f"{len(box)} endpoint pairs, {diff} differ  ({diff / len(box):.1%})")

# The rate depends on the box: longer segments have more chances to hit a tie.
wide = range(-6, 7)
wide_pairs = [((a, b), (c, d)) for a, b, c, d in itertools.product(wide, repeat=4)]
wdiff = sum(sk_line(a, b) != sk_nd(a, b) for a, b in wide_pairs)
print(f"{len(wide_pairs)} pairs in a wider box, {wdiff} differ  ({wdiff / len(wide_pairs):.1%})")
```

## 6. Two symmetries, one each

A rasteriser can have two properties that users assume without thinking.

+++

### Translation invariance

Move the segment by a whole number of pixels and
the drawn shape should just move with it.

```{code-cell} ipython3
fig, axes = plt.subplots(2, 4, figsize=(9.6, 4.0))
for col, t in enumerate(range(4)):
    a, b = (t, 0), (1 + t, 4)
    for row, (f, color, name) in enumerate(
        ((sk_line, C_LINE, "line"), (sk_nd, C_ND, "line_nd"))
    ):
        ax = axes[row, col]
        pixel_axes(ax, (6, 6), f"{name}, shifted by {t}")
        fill(ax, f(a, b), color)
        exact(ax, a, b, color="white")
fig.suptitle(
    "shift the same segment down one row at a time: line keeps its shape, "
    "line_nd does not",
    y=1.01,
)
fig.tight_layout()
```

Look along the bottom row. The `line_nd` shape flips between stepping early and
stepping late as the segment moves, because `np.round` sends `0.5` to `0` but
`1.5` to `2` — half-to-even depends on the parity of the coordinate.

+++

### Reversal symmetry

Naming the ends in the other order should draw the same pixels.

```{code-cell} ipython3
fig, axes = plt.subplots(1, 4, figsize=(9.6, 2.1))
panels = [
    (sk_line, p, q, C_LINE, "line, start to stop"),
    (sk_line, q, p, C_LINE, "line, stop to start"),
    (sk_nd, p, q, C_ND, "line_nd, start to stop"),
    (sk_nd, q, p, C_ND, "line_nd, stop to start"),
]
for ax, (f, a, b, color, name) in zip(axes, panels):
    pixel_axes(ax, shape, name)
    fill(ax, f(a, b), color)
    exact(ax, a, b, color="white")
fig.suptitle("line changes when you swap the ends; line_nd does not", y=1.04)
fig.tight_layout()
```

Measured over the whole box:

```{code-cell} ipython3
def symmetry(f, pairs, kind, shift=3):
    ok = 0
    for a, b in pairs:
        if kind == "reversal":
            ok += f(a, b) == f(b, a)
        else:
            moved = {(i + shift, j + shift) for i, j in f(a, b)}
            at = (a[0] + shift, a[1] + shift), (b[0] + shift, b[1] + shift)
            ok += moved == f(*at)
    return ok / len(pairs)


print(f"{'':16}{'reversal':>12}{'translation':>14}")
for name, f in (("line", sk_line), ("line_nd", sk_nd)):
    print(
        f"{name:<16}{symmetry(f, box, 'reversal'):>11.1%}"
        f"{symmetry(f, box, 'translation'):>13.1%}"
    )
```

Each function holds one property and loses the other.

+++

## 7. Pillow and OpenCV

Two other widely used rasterisers, for comparison. Both take coordinates in
`(column, row)` order, so the wrappers below swap.

```{code-cell} ipython3
CANVAS = 24


def pil_line(a, b):
    im = Image.new("L", (CANVAS, CANVAS), 0)
    ImageDraw.Draw(im).line([(a[1], a[0]), (b[1], b[0])], fill=255, width=1)
    return set(map(tuple, np.argwhere(np.array(im))))


def cv_line(a, b, connectivity=8):
    arr = np.zeros((CANVAS, CANVAS), np.uint8)
    cv2.line(arr, (a[1], a[0]), (b[1], b[0]), 255, 1, lineType=connectivity)
    return {(int(i), int(j)) for i, j in np.argwhere(arr)}
```

```{code-cell} ipython3
off = (4, 4)  # keep the segment inside the canvas
pa, qa = (off[0] + p[0], off[1] + p[1]), (off[0] + q[0], off[1] + q[1])

panels = [("skimage line", sk_line(pa, qa), C_LINE), ("pillow", pil_line(pa, qa), C_BOTH)]
panels.append(("opencv, 8-connected", cv_line(pa, qa), C_BOTH))
panels.append(("skimage line_nd", sk_nd(pa, qa), C_ND))

fig, axes = plt.subplots(1, len(panels), figsize=(2.4 * len(panels), 2.2))
for ax, (name, pix, color) in zip(axes, panels):
    sub = {(i - off[0] + 0, j - off[1] + 0) for i, j in pix}
    pixel_axes(ax, shape, name)
    fill(ax, sub, color)
    exact(ax, p, q, color="white")
fig.suptitle("the same tie case in four rasterisers", y=1.04)
fig.tight_layout()
```

`line` and Pillow choose the same pixels. `line_nd` and OpenCV choose the same
pixels. Neither of our functions is unusual on this case — each has a peer.

+++

Across every integer endpoint pair in a 9x9 box:

```{code-cell} ipython3
fns = {"sk.line": sk_line, "sk.line_nd": sk_nd, "pillow": pil_line}
if cv2 is not None:
    fns["opencv8"] = cv_line
    fns["opencv4"] = lambda a, b: cv_line(a, b, 4)

drawn = {k: [f(a, b) for a, b in box] for k, f in fns.items()}
names = list(fns)
print(f"agreement over {len(box)} endpoint pairs")
print(f"{'':12}" + "".join(f"{n:>11}" for n in names))
for a in names:
    row = "".join(
        f"{sum(x == y for x, y in zip(drawn[a], drawn[b])) / len(box):>10.1%} "
        for b in names
    )
    print(f"{a:<12}{row}")
```

```{code-cell} ipython3
# Every eleventh pair: Pillow and OpenCV each build an image per call, so the
# full corpus is slow here. The skimage figures above use all 6561 pairs.
sample = box[::11]
print(f"{'':14}{'reversal':>12}{'translation':>14}")
for name, f in fns.items():
    print(
        f"{name:<14}{symmetry(f, sample, 'reversal'):>11.1%}"
        f"{symmetry(f, sample, 'translation'):>13.1%}"
    )
```

Two things fall out of those numbers.

`skimage.draw.line` is **pixel-identical to Pillow**, and OpenCV's 8-connected
line equals our Bresenham run in *one of the two endpoint orders* in every
case — it is the same rasteriser with the endpoint order normalised first.

More importantly, **OpenCV holds both symmetries at once**. So the trade-off
our two functions appear to make is not forced. Neither is at a local optimum.

OpenCV also offers a second rasteriser, chosen with `lineType`. It is a
different thing from the tie-breaking above, and it is the subject of the next
section.

## 8. The two connectivities

`lineType` selects between two rasterisers, `cv2.LINE_8` and `cv2.LINE_4`.
They are not two settings of one algorithm. They draw different pixel sets,
with different guarantees, for different jobs.

```{code-cell} ipython3
import scipy.ndimage as ndi

# Slot 3 of the reference categorical palette; validates all-pairs with the
# blue and orange already in use.
C_FOUR = "#1baf7a"

S4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
S8 = np.ones((3, 3), int)
```

+++

### What each one guarantees

An 8-connected line may step diagonally, so it needs one pixel per step of the
longer axis. A 4-connected line may not, so it needs one pixel per step of
*both* axes. That gives two exact formulas: a Chebyshev length and a Manhattan
length, each plus one for the starting pixel.

```{code-cell} ipython3
c8 = c4 = 0
for a, b in box:
    di, dj = abs(a[0] - b[0]), abs(a[1] - b[1])
    c8 += len(cv_line(a, b, 8)) == max(di, dj) + 1
    c4 += len(cv_line(a, b, 4)) == di + dj + 1
print(f"over {len(box)} endpoint pairs")
print(f"   8-connected count == max(|di|, |dj|) + 1 : {c8 / len(box):.1%}")
print(f"   4-connected count == |di| + |dj| + 1     : {c4 / len(box):.1%}")
```

Both hold exactly, so `lineType` fixes how many pixels you get before any
rounding decision is taken.

```{code-cell} ipython3
a, b = (4, 4), (7, 13)
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.4))
for ax, conn, color, name in (
    (axes[0], 8, C_BOTH, "LINE_8, 8-connected"),
    (axes[1], 4, C_FOUR, "LINE_4, 4-connected"),
):
    pixel_axes(ax, (5, 12), f"{name}  ({len(cv_line(a, b, conn))} pixels)")
    fill(ax, {(i - 3, j - 3) for i, j in cv_line(a, b, conn)}, color)
    exact(ax, (a[0] - 3, a[1] - 3), (b[0] - 3, b[1] - 3), color="white")
fig.suptitle("the same segment, drawn twice", y=1.04)
fig.tight_layout()
```

The difference is visible in the moves themselves. Walk each pixel set along
the driving axis and look at the step taken between consecutive pixels:

```{code-cell} ipython3
for conn in (8, 4):
    pix = sorted(cv_line(a, b, conn), key=lambda ij: ij[1])
    steps = sorted({(abs(p[0] - q[0]), abs(p[1] - q[1]))
                    for p, q in zip(pix, pix[1:])})
    print(f"LINE_{conn}: {len(pix):>3} pixels, steps {steps}")
```

`(1, 1)` is a diagonal move. Only the 8-connected line makes one; the
4-connected line replaces each with a `(1, 0)` and a `(0, 1)`, which is where
its extra pixels come from.

+++

### The pixel sets have different connectivity

The names describe a property of the drawn set, which is worth checking rather
than assuming. Label each line as a binary image, once with a 4-connected
structuring element and once with an 8-connected one. A line that is
"4-connected" should be a single component under the 4-connected element.

```{code-cell} ipython3
def as_mask(pixels, shape=(CANVAS, CANVAS)):
    """The pixel set as a boolean image."""
    m = np.zeros(shape, bool)
    for i, j in pixels:
        m[i, j] = True
    return m


print(f"{'':10}{'one component under S4':>26}{'under S8':>12}")
for conn in (8, 4):
    n4 = sum(ndi.label(as_mask(cv_line(a, b, conn)), structure=S4)[1] == 1
             for a, b in box)
    n8 = sum(ndi.label(as_mask(cv_line(a, b, conn)), structure=S8)[1] == 1
             for a, b in box)
    print(f"LINE_{conn:<5}{n4 / len(box):>25.1%}{n8 / len(box):>12.1%}")

axis_aligned = sum(a[0] == b[0] or a[1] == b[1] for a, b in box)
print(f"\npairs with no diagonal step at all: {axis_aligned / len(box):>10.1%}")
```

An 8-connected line falls into separate pieces under 4-connectivity. The
exceptions are exactly the axis-aligned segments, which take no diagonal step
and so are 4-connected by accident — the two rates above agree to the digit. A
4-connected line holds together under both. That is the whole difference,
stated as a property rather than as a name.

+++

### They are not nested

It is tempting to think the 4-connected line is the 8-connected one with corner
pixels added. It is not.

```{code-cell} ipython3
nested = sum(cv_line(a, b, 8) <= cv_line(a, b, 4) for a, b in box)
print(f"8-connected set contained in the 4-connected set: {nested / len(box):.1%}")

example = next((a, b) for a, b in box if not cv_line(a, b, 8) <= cv_line(a, b, 4))
s8, s4 = cv_line(*example, 8), cv_line(*example, 4)
print(f"\nfirst counter-example: {example[0]} to {example[1]}")
print(f"   only in LINE_8: {sorted(s8 - s4)}")
print(f"   only in LINE_4: {sorted(s4 - s8)}")
```

```{code-cell} ipython3
a, b = example
lo = (min(a[0], b[0]) - 1, min(a[1], b[1]) - 1)
span = (abs(a[0] - b[0]) + 3, abs(a[1] - b[1]) + 3)
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.2))
for ax, pix, color, name in (
    (axes[0], s8, C_BOTH, "LINE_8"), (axes[1], s4, C_FOUR, "LINE_4")
):
    pixel_axes(ax, span, name)
    fill(ax, {(i - lo[0], j - lo[1]) for i, j in pix}, color)
    exact(ax, (a[0] - lo[0], a[1] - lo[1]), (b[0] - lo[0], b[1] - lo[1]),
          color="white")
fig.suptitle("neither set contains the other", y=1.06)
fig.tight_layout()
```

The two are independent rasterisations of the same segment, and neither is
derived from the other. Each resolves awkward cases its own way. Which cases,
and by what rule, is not something this notebook settles.

+++

### Why a 4-connected line exists: it seals

The reason to pay for the extra pixels is that a 4-connected curve is a barrier
an 8-connected flood fill cannot cross. An 8-connected curve is not: a fill that
may move diagonally slips between two diagonally adjacent pixels.

```{code-cell} ipython3
def barrier(conn, shape=(28, 40), a=(2, 0), b=(25, 39)):
    """Draw a line across the array, then label what it leaves free."""
    arr = np.zeros(shape, np.uint8)
    cv2.line(arr, (a[1], a[0]), (b[1], b[0]), 255, 1, lineType=conn)
    labels, n = ndi.label(~(arr > 0), structure=S8)
    return arr > 0, labels, n


for conn in (8, 4):
    _, _, n = barrier(conn)
    verdict = "sealed" if n > 1 else "the fill leaks through"
    print(f"LINE_{conn}: the free area is {n} component(s)  -> {verdict}")
```

```{code-cell} ipython3
fig, axes = plt.subplots(2, 1, figsize=(5.4, 4.6))
for ax, conn in zip(axes, (8, 4)):
    m, labels, n = barrier(conn)
    ax.imshow(np.where(m, 0, labels),
              cmap=ListedColormap([C_FOUR, C_OFF, C_LINE]), vmin=0, vmax=2)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(f"LINE_{conn}: {n} free component(s)")
fig.legend(
    handles=[
        Patch(facecolor=C_FOUR, label="the line"),
        Patch(facecolor=C_OFF, label="one side"),
        Patch(facecolor=C_LINE, label="the other side"),
    ],
    loc="lower center", ncols=3, frameon=False, fontsize=8,
)
fig.suptitle("only the 4-connected line divides the array in two", y=1.0)
fig.tight_layout(rect=(0, 0.07, 1, 1))
```

The top panel is a single region: the fill has walked through the line. The
bottom panel is two. If you draw a boundary and then fill on one side of it,
that is the whole ballgame, and it is why the option exists.

+++

### What scikit-image would need

`skimage.draw` has no 4-connected line, and `line_nd`'s "ndim-connected"
guarantee is the diagonal one. Closing the gap needs no new rasteriser: take
the Bresenham line and insert a corner pixel at each diagonal step, choosing
whichever of the two candidate corners lies nearer the true segment.

```{code-cell} ipython3
def line_4(start, stop):
    """Bresenham, with a corner pixel inserted at each diagonal step."""
    ii, jj = line(start[0], start[1], stop[0], stop[1])
    origin = np.asarray(start, float)
    direction = np.asarray(stop, float) - origin

    def offset(pixel):
        v = np.asarray(pixel, float) - origin
        return abs(v[0] * direction[1] - v[1] * direction[0])

    out = [(int(ii[0]), int(jj[0]))]
    for i, j in zip(ii[1:].tolist(), jj[1:].tolist()):
        pi, pj = out[-1]
        if i != pi and j != pj:
            out.append(min(((pi, j), (i, pj)), key=offset))
        out.append((i, j))
    return out
```

```{code-cell} ipython3
conn_ok = count_ok = 0
for a, b in box:
    pix = line_4(a, b)
    conn_ok += ndi.label(as_mask(pix), structure=S4)[1] == 1
    count_ok += len(set(pix)) == abs(a[0] - b[0]) + abs(a[1] - b[1]) + 1
print(f"over {len(box)} endpoint pairs")
print(f"   4-connected           {conn_ok / len(box):.1%}")
print(f"   Manhattan pixel count {count_ok / len(box):.1%}")

arr = np.zeros((28, 40), np.uint8)
for i, j in line_4((2, 0), (25, 39)):
    arr[i, j] = 255
print(f"   seals the array       {ndi.label(~(arr > 0), structure=S8)[1] > 1}")
same = sum(set(line_4(a, b)) == cv_line(a, b, 4) for a, b in box)
print(f"   same pixels as LINE_4 {same / len(box):.1%}")
```

It meets both guarantees and it seals. Where it differs from OpenCV, at the
rate printed above, the difference is the corner chosen at each diagonal step —
a tie-breaking question of exactly the kind the rest of this notebook is about,
not a difference in what the two functions promise.

```{code-cell} ipython3
a, b = (4, 4), (7, 13)
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.4))
for ax, pix, color, name in (
    (axes[0], set(line_4(a, b)), C_ND, "candidate line_4"),
    (axes[1], cv_line(a, b, 4), C_FOUR, "opencv LINE_4"),
):
    pixel_axes(ax, (5, 12), name)
    fill(ax, {(i - 3, j - 3) for i, j in pix}, color)
    exact(ax, (a[0] - 3, a[1] - 3), (b[0] - 3, b[1] - 3), color="white")
fig.suptitle("same guarantees, different corners", y=1.04)
fig.tight_layout()
```

## 9. Two ways forward

+++

### Fix 1: round half up in `line_nd`

Half-to-even makes the drawn shape depend on the parity of an absolute
coordinate. Nothing wants that, and `_round_safe` is itself an admission that
the rule misbehaves — it patches one case instead of replacing the rule.

Rounding half up is translation-invariant by construction, because
`floor(x + t + 0.5) == floor(x + 0.5) + t` for whole `t`.

```{code-cell} ipython3
def line_nd_halfup(a, b):
    """`line_nd` with half-up rounding instead of half-to-even."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    npoints = int(np.ceil(np.max(np.abs(b - a)))) + 1
    coords = np.floor(np.linspace(a, b, npoints, endpoint=True).T + 0.5).astype(int)
    return set(zip(*(c.tolist() for c in coords)))


fig, axes = plt.subplots(2, 4, figsize=(9.6, 4.0))
for col, t in enumerate(range(4)):
    a, b = (t, 0), (1 + t, 4)
    for row, (f, name) in enumerate(
        ((sk_nd, "line_nd, now"), (line_nd_halfup, "line_nd, half up"))
    ):
        ax = axes[row, col]
        pixel_axes(ax, (6, 6), f"{name} (+{t})")
        fill(ax, f(a, b), C_ND)
        exact(ax, a, b, color="white")
fig.suptitle("half-up rounding makes line_nd keep its shape when it moves", y=1.01)
fig.tight_layout()
```

### Fix 2: normalise the endpoint order in `line`

Sorting the two endpoints before rasterising makes the result independent of
which end was named first. Any rule that depends only on the unordered pair
works; OpenCV uses a different one from `sorted`, and matching it exactly is
not a requirement.

```{code-cell} ipython3
def line_sorted(a, b):
    """`line` with the endpoint order normalised."""
    a, b = sorted([tuple(a), tuple(b)])
    return sk_line(a, b)


fig, axes = plt.subplots(1, 4, figsize=(9.6, 2.1))
panels = [
    (sk_line, p, q, "line, forward"),
    (sk_line, q, p, "line, backward"),
    (line_sorted, p, q, "sorted, forward"),
    (line_sorted, q, p, "sorted, backward"),
]
for ax, (f, a, b, name) in zip(axes, panels):
    pixel_axes(ax, shape, name)
    fill(ax, f(a, b), C_LINE)
    exact(ax, a, b, color="white")
fig.suptitle("normalising the endpoints makes line direction-independent", y=1.04)
fig.tight_layout()
```

### What each fix costs

```{code-cell} ipython3
changed_nd = sum(sk_nd(a, b) != line_nd_halfup(a, b) for a, b in box)
changed_ln = sum(sk_line(a, b) != line_sorted(a, b) for a, b in box)
agree_now = sum(sk_line(a, b) == sk_nd(a, b) for a, b in box)
agree_fix = sum(line_sorted(a, b) == line_nd_halfup(a, b) for a, b in box)
n_box = len(box)

print(f"line_nd output changes under half-up rounding : {changed_nd / n_box:6.1%}")
print(f"line output changes under sorted endpoints    : {changed_ln / n_box:6.1%}")
print(f"line and line_nd agree, now                   : {agree_now / n_box:6.1%}")
print(f"line and line_nd agree, both fixed            : {agree_fix / n_box:6.1%}")

for name, f in (("line, sorted", line_sorted), ("line_nd, half up", line_nd_halfup)):
    print(
        f"\n{name}: reversal {symmetry(f, box, 'reversal'):.1%},"
        f" translation {symmetry(f, box, 'translation'):.1%}"
    )
```

Both fixes reach 100% on both symmetries, matching OpenCV's guarantees.

The case for fix 1 is strong: parity-dependent rasterisation is a defect, and
no comparator library has it. The case for fix 2 is weaker — the current
behaviour is textbook Bresenham and matches Pillow exactly — but a drawing
function whose output depends on which end you named first is surprising, and
the change is one line.

Both change results, so both belong in `skimage2` with a migration note, not
in a patch release.

+++

## 10. What not to do

Do not try to make `line` and `line_nd` agree. Even with both fixes they still
differ on a small fraction of segments, and that residue is inherent: an exact
integer error term and a sampled-then-rounded line pick different pixels at
ties. They are different algorithms with different guarantees, and both are
worth keeping.

```{code-cell} ipython3
resid = [(a, b) for a, b in box if line_sorted(a, b) != line_nd_halfup(a, b)]
a, b = resid[0]
lo = (min(a[0], b[0]) - 1, min(a[1], b[1]) - 1)

fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.4))
span = (abs(a[0] - b[0]) + 3, abs(a[1] - b[1]) + 3)
for ax, (f, color, name) in zip(
    axes,
    ((line_sorted, C_LINE, "line, sorted"), (line_nd_halfup, C_ND, "line_nd, half up")),
):
    pixel_axes(ax, span, name)
    fill(ax, {(i - lo[0], j - lo[1]) for i, j in f(a, b)}, color)
    exact(ax, (a[0] - lo[0], a[1] - lo[1]), (b[0] - lo[0], b[1] - lo[1]), color="white")
fig.suptitle(f"both fixed, still different: {a} to {b}", y=1.04)
fig.tight_layout()
```

## Summary

| | `line` | `line_nd` | Pillow | OpenCV |
|---|---|---|---|---|
| Algorithm | integer Bresenham | sample, then round per axis | integer Bresenham | Bresenham, ends normalised |
| Dimensions | 2 | N | 2 | 2 |
| Input | integer only | float or integer | integer | integer (sub-pixel via `shift`) |
| Stop point | always included | excluded unless `endpoint=True` | included | included |
| Connectivity | diagonal | diagonal | diagonal | diagonal or 4-connected |
| Reversal symmetric | no | **yes** | no | **yes** |
| Translation invariant | **yes** | no | **yes** | **yes** |

Measured with Pillow 12.2.0 and OpenCV 5.0.0, over integer endpoints only, for
segments up to about twelve pixels long. Anti-aliased variants
(`line_aa`, Pillow's, OpenCV's `LINE_AA`) are not compared here; `line_aa`
would want the same endpoint treatment as `line`.
