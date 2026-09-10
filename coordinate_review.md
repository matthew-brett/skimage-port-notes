# Coordinate review

Partially distilled from AI summaries with:

* Gemini — 3
* Claude — Opus 4.5
* Cursor — auto

## Background

Let us imagine a 2D grayscale image.

To be more concrete, let's load such an image:

```{python}
import numpy as np
import skimage as ski

import matplotlib.pyplot as plt

img = ski.data.camerman()
```

Let us now imagine that we want to identify a pixel by its
coordinate.

A coordinate is a pair of numbers identifying the pixel.

For example, we may have a coordinate (5, 10).  To interpret
this coordinate, we need to know what the 5 and 10 refer to.
Put another way, we have to know what *coordinate axes* the
5 and 10 relate to.  The coordinate axes specify a *coordinate
system*.

There are two common coordinate systems in imaging.

## The array coordinate system

The first coordinate system will appear the most obvious to
experienced users of Numpy — is is the array coordinate
system, also know as the *matrix* or *linear algebra*
coordinate system.  We will also call this the "i, j" coordinate system, for reasons that should become clear.

This interprets the first number (here 5) as the *position* along the first axis of the array, and the second number (here 10) as a position along the second axis of the array.

In other words, the pixel at (5, 10), using the array coordinate system, is given by the Numpy operation:

```{python}
img[5, 10]
```

So, in this case, the coordinate system is given by the array
axes.

Here is the `cameraman` image displayed in Matplotlib.

```{python}
plt.imshow(img)
```

Notice that, in terms of the image we have just displayed, the
first axis is top to bottom, starting at 0, and the second
axis is left to right, starting at 0.

We call this the "ij" coordinate system on the basis that "i"
will always refer to the first array axis, and "j" to the
second.  We chose "i" and "j" to have no particular meaning in
terms of the way the image is displayed - to remind us that
this meaning is strictly in terms of the axes of the image
array.

## The imaging coordinate system

There is another common coordinate system in imaging, that we
will call the *imaging coordinate system*, or the "x, y"
coordinate system.

Very confusingly, the first axis in the imaging coordinate system corresponds to the *second* axis in the image array, and the second axis in the imaging coordinate system corresponds to the *first* axis in the image array.

Imagine I have some coordinate (11, 20).  If that is a coordinate in the imaging coordinate system, then the equivalent pixel in that coordinate system is given by:

```{python}
img[20, 11]
```

In terms of the display above, the first axis of the
coordinate system runs from left to right, starting at 0, and
the second axis in the coordinate system runs from top to
bottom, starting at 0.

Why is this *imaging* coordinate system popular?  Because it
matches, in part, the way we think of axes on a graph.  On
a graph, the first axis is *x* and runs from left to right,
and the second is *y* and it runs from bottom to top.  With
images, we often think of the first values in the image as
being at the *top* of the image (not the bottom), so, for the
imaging coordinate convention, it's most common to think of
the y (second) axis as running top to bottom (rather than the
y-axis of standard graphs, which run from bottom to top).

Notice that the labels "x" and "y" refer to the way that the
image is displayed on the screen.

Obviously, in order to know what pixel any coordinate refers
to, we need to know which coordinate system we are using.  The
coordinate (5, 10) means `img[5, 10]` if it's in array
coordinates, and `img[10, 5]` if it's in imaging coordinates.

## Row and columns

For a 2D image, there appears to be no ambiguity in the term
"row" or "column".  Both the "ij" and "xy" convention think of the row of an image as going left to right in the display, and therefore, in terms of the image array, the row at position 5 is given by:

```{python}
# Row at position 5.
img[5, :]
```

Columns in both conventions run top to bottom, so the column at position 10 is given by:

```{python}
# Column at position 10.
img[:, 10]
```

Thus, for a *2D grayscale array*, we can talk about *row, column* coordinates ("rc" coordinates), where the first axis gives row position, and the second axis give column position.  Of course, *for the 2D case* this is the same as the "ij" convention.

There are various instances in Scikit-image, of coordinate values referring to row and column axes.  `skimage.draw` has many such instances.

## Rows and columns in 3D

We've emphasized that the "rc" convention is the same as the "ij" convention for 2D.

Now consider a three-dimensional image:

```{python}
img_3d = ski.data.cells3d()
img_3d.shape
```


`row` and `column` are ambiguous; we should not use them.

However, there are many instances of row and column notation in the code.  We
should rewrite these.  As an example, see the `skimage.draw` functions.  Consider:

```python
def line(r0, c0, r1, c1):
    """Generate line pixel coordinates.

    Parameters
    ----------
    r0, c0 : int
        Starting position (row, column).
    r1, c1 : int
        End position (row, column).

    Returns
    -------
    rr, cc : (N,) ndarray of int
        Indices of pixels that belong to the line.
        May be used to directly index into an array, e.g.
        ``img[rr, cc] = 1``.

    ...
    """
```

This should probably be:

```python
def line(i0, j0, i1, j1):
    """Generate line pixel coordinates.

    Parameters
    ----------
    i0, j0 : int
        Starting position (first axis, second axis).
    i1, j1 : int
        End position (first axis, second axis).

    Returns
    -------
    ii, jj : (N,) ndarray of int
        Indices of pixels that belong to the line.
        May be used to directly index into an array, e.g.
        ``img[ii, jj] = 1``.

    ...
    """
```

Cursor identified multiple instances of row and column references in docstrings and variables.

## x and y in variables and docstrings

We should eliminate `x` and `y` in parameter names and
docstrings, because these are also ambiguous.

* `predict_x`, `predict_y`, `predict_xy` in transform classes.
* `x, y coordinates to transform` to `Coordinates to transform` in
  `_geometric.matrix_transform`. (PR #8319).

## xy vs ij in implementations

For skimage1 code, we should provide some way of specifying
coordinate convention.  We already have the examples of
`skimage.feature.corner.structure_tensor` and `hessian_matrix`
that have an `order=` parameter, that defaults to `rc` (meaning
ij).  However, we might prefer something like `coord_convention`
or similar.
