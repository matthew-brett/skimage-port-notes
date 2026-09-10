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
(`draw/_draw.pyx::_line`) with integer arithmetic only.

It takes the axis with the larger absolute delta as the **driving axis**,
swapping the roles of the two axes when the line is steep, and steps one pixel
along that axis per iteration. An integer error term decides when the minor
axis steps as well:

```
d = 2 * dr - dc                # dr is the minor delta, dc the driving delta
for each step along the driving axis:
    emit the current pixel
    while d >= 0:              # dr <= dc, so this runs at most once
        minor += sign
        d -= 2 * dc
    driving += sign
    d += 2 * dr
```

The final point is written as the literal endpoint, so both ends are always
present and the output holds exactly `max(abs(delta)) + 1` pixels.

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
import itertools

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

OpenCV also offers a 4-connected line, which we do not have at all:

```{code-cell} ipython3
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.4))
a, b = (4, 4), (7, 13)
for ax, conn, name in ((axes[0], 8, "8-connected"), (axes[1], 4, "4-connected")):
    pixel_axes(ax, (5, 12), f"opencv, {name}")
    fill(ax, {(i - 3, j - 3) for i, j in cv_line(a, b, conn)}, C_BOTH)
    exact(ax, (a[0] - 3, a[1] - 3), (b[0] - 3, b[1] - 3), color="white")
fig.suptitle("a connectivity option scikit-image does not offer", y=1.04)
fig.tight_layout()
```

## 8. Two ways forward

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

## 9. What not to do

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
