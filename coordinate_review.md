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

# Coordinate review

Partially distilled from AI summaries with:

* Gemini — 3
* Claude — Opus 4.5
* Cursor — auto

## Background

Let us imagine a 2D grayscale image.

To be more concrete, let's load such an image:

```{code-cell}
import numpy as np
import skimage as ski

import matplotlib.pyplot as plt

# Set default 2D image colormap.
plt.rc('image', cmap='gray')

img = ski.data.camera()
img.shape
```

Let us now imagine that we want to identify a pixel by its
coordinate.

A coordinate is a pair of numbers identifying the pixel.

For example, we may have a coordinate (5, 10).  To interpret this coordinate,
we need to know what the 5 and 10 refer to. Put another way, we have to know
what *coordinate axes* the 5 and 10 relate to.  The coordinate axes specify
a *coordinate system*.

There are two common coordinate systems in imaging.

## A note on terminology

When referring to an array, I will use upper-case letters as variables to
refer to the lengths of axes, and lower-case letters to refer to indices into these axes.

For example, say I have any array of two dimensions.  I could refer to the
shape of this array as `(I, J)`, if I'm labeling the first and second axis with
I and J, or `(R, C)` if labeling with R and C.  See below for discussions of
the array (or Numpy, or IJ) and row-column (RC) labeling conventions.  If I'm
referring to a variable containing an index into each those dimensions,
I will use `i`, and `j` (for array / IJ convention), or `r`, `c` (for RC
convention). Thus, for example, `j` is a variable that can take any value from
0 through J-1.

We'll use the lower case convention when we're referring to axis by letter.
Thus we'll talk about the `i` axis, rather than the `I` axis.

## The array coordinate system

The first coordinate system will appear the most obvious to experienced users
of Numpy — it is the array coordinate system, also known as the *matrix* or
*linear algebra* coordinate system.  We usually call this the IJ coordinate
system, for reasons that should become clear.

The array / IJ coordinate system interprets the first number (here 5) as the
*position* along the first axis of the array, and the second number (here 10)
as a position along the second axis of the array.

In other words, the pixel at (5, 10), using the array coordinate system, is
given by the Numpy operation:

```{code-cell}
img[5, 10]
```

So, in this case, the coordinate system is given by the array axes.

Here is the `cameraman` image displayed in Matplotlib.

```{code-cell}
plt.imshow(img)
```

Notice that, in terms of the image we have just displayed, the first axis is
top to bottom, starting at 0, and the second axis is left to right, starting
at 0.

We call this the IJ coordinate system on the basis that "i" will always
refer to the first array axis, and "j" to the second.  We chose "i" and "j" to
have no particular meaning in terms of the way the image is displayed - to
remind us that this meaning is strictly in terms of the axes of the image
array.

## The imaging coordinate system

There is another common coordinate system in imaging, that we will call the
*imaging coordinate system*, or the XY coordinate system.

Very confusingly, the first axis in the imaging coordinate system corresponds
to the *second* axis in the image array, and the second axis in the imaging
coordinate system corresponds to the *first* axis in the image array.

Imagine I have some coordinate (11, 20).  If that is a coordinate in the
imaging coordinate system, then the equivalent pixel in that coordinate system
is given by:

```{code-cell}
img[20, 11]
```

In terms of the display above, the first axis of the coordinate system runs
from left to right, starting at 0, and the second axis in the coordinate
system runs from top to bottom, starting at 0.

Why is this *imaging* coordinate system popular?  Because it matches, in part,
the way we think of axes on a graph.  On a graph, the first axis is *x* and
runs from left to right, and the second is *y* and it runs from bottom to top.
With images, we often think of the first values in the image as being at the
*top* of the image (not the bottom), so, for the imaging coordinate
convention, it's most common to think of the y (second) axis as running top to
bottom (rather than the y-axis of standard graphs, which run from bottom to
top).

Notice that the labels "x" and "y" refer to the way that the image is
displayed on the screen.

Obviously, in order to know what pixel any coordinate refers to, we need to
know which coordinate system we are using.  The coordinate (5, 10) means
`img[5, 10]` if it's in array coordinates, and `img[10, 5]` if it's in imaging
coordinates.

## Row and columns

For a 2D image, there appears to be no ambiguity in the term "row" or
"column".  Both the IJ and XY convention think of the row of an image as
going left to right in the display, and therefore, in terms of the image
array, the row at position 5 is given by:

```{code-cell}
# Row at position 5.
row_5 = img[5, :]
```

Columns in both conventions run top to bottom, so the column at position 10 is
given by:

```{code-cell}
# Column at position 10.
col_10 = img[:, 10]
```

Thus, for a *2D grayscale array*, we can talk about *row, column* coordinates
(RC coordinates), where the first axis gives row position, and the second axis
give column position.  Of course, *for the 2D case* this is the same as the IJ
convention.

There are various instances in Scikit-image, of coordinate values referring to
row and column axes.  `skimage.draw` has many such instances.  Here's the
docstring for `skimage.draw.line`:

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

## Rows and columns in 3D

We've emphasized that the RC convention is the same as the IJ convention for
2D.

Now consider an image with more than two dimensions.

```{code-cell}
img_4d = ski.data.cells3d()
img_4d.shape
```

In fact the first dimension tracks over a third spatial dimension, the second tracks two different things being imaged in the same sample — `img_4d[:, 0]` gives cell membranes, `img[:, 1]` gives cell nuclei.  The third and fourth dimensions are the rows and columns of the 2D images.

Let's first select out the nuclei, to give a 3D image of nuclei:

```{code-cell}
# Select images of nuclei
img_3d = img_4d[:, 1]
img_3d.shape
```

We're going to allow ourselves to sink into confusion for a few paragraphs, so
hold on tight.

The first axis is the third spatial axis of the image.   We might call this
axis — the "plane" axis.  We can think of `img_3d` as containing a series of
2D images stacked on top of each other, where the first axis selects the
plane, the second the "row" of the image, and third the "column" of the image.

For example, to show the middle plane:

```{code-cell}
middle_i = img_3d.shape[0] // 2
img_2d = img_3d[middle_i, :, :]
plt.imshow(img_2d)
```

Where's the confusion?  Well — for a 2D image, we were happy to call the first
array axis the "row" axis.  It's the row axis, because that axis represents
rows in the displayed image.  For a 2D image, this is also the "i" axis,
because it's the first axis.

However, now we're in 3D, the first axis is no longer the "row" axis, because
the row axis is the axis representing the rows in a 2D image.  So we now find
ourselves wanting to call the second axis (of `img_3d`) the "row" axis, if we
are thinking in terms of displayed 2D images.  The "row" axis is no longer the
same as the "i" axis — if anything it is the "j" axis, where the "column" axis
is the "k" axis.

This is to point out that "row" carries a meaning in terms of how we think of
the array, for 2D display.  In our example, the "row" axis might be the second
axis, but then again, it depends how you think of the 3D image in terms of 2D.
We could also think of `img_3d` as a stack of 2D images, where displayed rows
and columns are the first two dimensions, and the third is the last spatial
dimension:

```{code-cell}
middle_k = img_3d.shape[2] // 2
another_img_2d = img_3d[:, :, middle_k]
plt.imshow(another_img_2d)
```

So, "row" becomes ambiguous here — the axis we think of as corresponding to
rows, depends on how we think of the image display.  That is, "row" is
a *semantic* label for an axis, telling us what that axis means.

In contrast, when we use "i", or "j" or "k" to label an axis, we intend no
semantic meaning — we just refer to the first, second and third axis of an
array, regardless of the meaning of these axes.

## m and n are ambiguous labels

In various places in Skimage1, we refer to axes as `m` or `n`.  For example,
see the docstring listed in full further below for the `rank.equalize` filter:

```rest
    def equalize(image, footprint, out=None, mask=None, shift_x=0, shift_y=0, shift_z=0):
        ...
        image : ndarray of shape ([P,] M, N) and dtype (uint8 or uint16)
            Input image.
```

Here `M` and `N` appear to mean (semantic) "row" and "column", because `P`
should usually be taken to mean the plane axis.  However, elsewhere — for
example in the `watershed` function whose docstring we list further below — we
have:

```rest
    image : (M, N[, ...]) ndarray
        Data array where the lowest value points are labeled first.
```

where `M`, `N` appear to mean `i`, `j`.   Given the ambiguity, we suggest
dropping `M` and `N` as axis labels, in favor of `i` and `j`.

## Semantic and not-semantic labels

From the discussion above, we will call "row" and "column" and "plane"
*semantic* axis labels, because each tells about what that particular axis
means in terms of image display.

"i", "j" and "k" are non-semantic axis labels, and only refer to the first
second and third axes of the array.

## Convention and semantic labels

In general, in code, when we use "row" or "column" or "plane" as axis labels,
we would like to be able to predict which axes these refer to, in terms of the
image.  The only way we can do that, short of having an array structure where
we can explicitly label the axes (such as [xarray](https://docs.xarray.dev), is
*by convention*.  That is, we always make sure that, for example, the "plane"
axis is first, the "row" axis is second, and the "column" axis is third.

Conventions can differ in different fields.  For example, in my own (MB) field
of brain imaging, the convention is for 3D brain images to be row first, then
columns, then plane, so "plane", by convention, is the last axis, not the
first.

In fact, you can already see conventions for axis positions in play with 2D
color images.  Typically, we represent 2D color images with the three or four
color channels running along the third axis. That is a typical (by convention)
2D color image is 3D, with rows as the first dimension, columns as the second,
and color channel as the third.

## Skimage and row, column

I am going to argue that "row" and "column" should always be treated as
semantic labels, in that they refer to the displayed rows and columns of the
image, and the way you display an image is your choice, according to the
meaning of the axes.

But — there may be situations where the semantic meaning,  in practice, always corresponds to axes.  I would argue that the criterion should be:

> a) Does "row" always mean the first axis, in practice?
> b) Does "column" always mean the second axis, in practice?

If so, the use of "row" and "column" *may be* acceptable, and is a synonym for
"i" and "j" in the "i, j" convention.

Otherwise "row" and "column" are generally not acceptable, and need to be
rephrased in terms of "i, j, k ...".

Cursor (the AI agent) identified multiple instances of row and column
references in docstrings and variables.

## Other issues arising

### Color and `channel_axis`

While exploring coordinates, I found several instances where the color channel
is assumed to be the last.  We agreed on Zulip and meeting that we should always offer `channel_axis=` in the Skimage2 argument list for functions expecting a color channel axis, and should not assume it is the last.  This applies to:

* `set_color` in `draw/draw.py` (`image : ndarray of shape (M, N,
  C)`)
* `profile_line` in `measure/profile.py` (docstring: "multichannel (3D array,
  where the final axis contains the channel information)"; the assumption is
  the `if image.ndim == 3:` branch).
* `show_rag` in `graph/_rag.py` (`image : ndarray, shape (M, N[, 3])`).
* `rag_mean_color` in `graph/_rag.py`
  (`image : ndarray, shape(M, N[, ..., P], 3)`).
* `quickshift` in `segmentation/_quickshift.py`
* `active_contour` in `segmentation/active_contour_model.py`.
* `mark_boundaries` in `segmentation/boundaries.py`.

We need some agreed phrasing for the names / lengths of the channel axis — for
example for `(M, N, C)` above.  One suggestion in the meeting and on Zulip
was:

```rest
  image : 3D array, representing 2D image, with channel axis, e.g (I, J, C)
      Use the `channel_axis` parameter to specify the array axis corresponding to color channels.
```

See comments / plans for given functions in text below.

### Coordinate arguments as sequences

Sometimes we specify coordinates (row and column positions) as separate
arguments — one for row position, another for column position.   In other
places we use sequences to specify coordinates, with one sequence or array
argument containing both the row and column positions.   For an example of the second pattern (in `draw/draw.py`:

```python
def disk(center, radius, *, shape=None):
    """Generate coordinates of pixels within circle.

    Parameters
    ----------
    center : tuple
        Center coordinate of disk.
    radius : double
        Radius of disk.
    ...
    """
```

Other examples of the one-argument-for-both pattern:

* In `draw/draw.py`: `set_color(image, coords, ...)`, `def rectangle(start,
  end=None, ...)`, `def rectangle_perimeter(start, end=None, ...)`.
* In `draw/line_ne.py`: `def line_nd(start, stop, ...)`.

* Others found by Gemini:

  ```
  - skimage.measure.profile_line(image, src, dst, ...): Uses src and dst as coordinate pairs.
  - skimage.morphology.flood(image, seed_point, ...): Uses a seed_point tuple.
  ```

For an example of the first (separate argument) pattern, see the docstring of
`draw/draw.py::line`:

```python
def line(r0, c0, r1, c1):
    """Generate line pixel coordinates.

    Parameters
    ----------
    r0, c0 : int
        Starting position (row, column).
    r1, c1 : int
        End position (row, column).
    ...
    """
```

Each coordinate takes up two scalar input arguments (`r0` and `c0`, `r1, c1`).

We have the same pattern in (at least):

* `draw/draw.py::line_aa`
* `draw/draw.py::bezier_curve`:

  ```python
  def bezier_curve(r0, c0, r1, c1, r2, c2, weight, shape=None):`
      ...
  ```

* `draw/draw.py::polygon_perimeter`:

  ```python
  def polygon_perimeter(r, c, shape=None, clip=False):
      """Generate polygon perimeter coordinates.

      Parameters
      ----------
      r : (N,) ndarray
          Row coordinates of vertices of polygon.
      c : (N,) ndarray
          Column coordinates of vertices of polygon.
      ...
      """
  ```

* `draw/draw.py::polygon`: `r, c` arguments as above.

* `draw/draw.py::circle_perimeter`: `r`, `c` arguments as above.

* `draw/draw.py::circle_perimeter_aa`: `r`, `c` arguments as above.

* `draw/draw.py::ellipse_perimeter_aa`: `r`, `c` arguments as above.

* `draw/draw.py::ellipse`:

  ```python
  def ellipse(r, c, r_radius, c_radius, shape=None, rotation=0.0):
      """Generate coordinates of pixels within ellipse.

      Parameters
      ----------
      r, c : double
          Centre coordinate of ellipse.
      ...
      """
  ```

* `morphology/footprints.py::_cross`:

  ```python
  def _cross(r0, r1, dtype=np.uint8):
      """Cross-shaped structuring element of shape (r0, r1).

      Only the central row and column are ones.
      """
  ```

* Somewhat more ambiguously — `_shared/_geometry.py::polygon_clip`:

  ```python
  def polygon_clip(rp, cp, r0, c0, r1, c1):
      """Clip a polygon to the given bounding box.

      Parameters
      ----------
      rp, cp : (K,) ndarray of double
          Row and column coordinates of the polygon.
      (r0, c0), (r1, c1) : double
          Top-left and bottom-right coordinates of the bounding box.
      ...
      """
  ```

  It might be reasonable to change these to `poly_coords, top_left,
  bottom_right`.

* Others identified by Gemini:

  ```
  skimage.feature
   - multiblock_lbp(int_image, r, c, width, height): Computes the multi-block local binary pattern at a specific
     top-left corner (r, c).
   - draw_multiblock_lbp(image, r, c, width, height, lbp_code=0, ...): Visualizes a multi-block LBP at (r, c).
   - haar_like_feature(int_image, r, c, width, height, ...): Computes Haar-like features for a region starting at (r,
     c).
   - draw_haar_like_feature(image, r, c, width, height, ...): Visualizes Haar-like features starting at (r, c).

+++

  skimage.filters
   - LPIFilter2D(impulse_response, **filter_params): The impulse_response argument is a callable that must accept r
     and c as separate vectors representing row and column positions.
   - filter_forward, filter_inverse, and wiener: These functions also take an impulse_response callable with the same
     separate (r, c) requirement.
  ```

### Radians and degrees

Generally we give angles in radians.  There are six and a half exceptions:

* `transform/_warp.py::rotate` gives rotation angle in degrees.  Plan — see
  above.
* `transform/radon_transform.py::radon` gives projection angles `theta` in
  degrees.
* `transform/radon_transform.py::iradon`, and `iradon_sart` gives
  reconstruction angles `theta` in degrees.
* `transform/radon_transform.py::order_angles_golden_ratio` gives
  projection angles `theta` in degrees.
* `transform/_geometric.py::_euler_rotation_matrix` has a `degrees=False`
  keyword argument, allowing the user to use degrees instead of default
  radians.  A Github search of `/[ .]_euler_rotation_matrix\(/ AND NOT path:skimage` finds no-one using this with the `degrees=True` option.  Of course this is a private function, and it's a simple call-through to `scipy.spatial.transform.Rotation.from_euler`.

Proposal: standardize to radians throughout.  Remove `degrees=False` argument to `_euler_rotation_matrix` helper function.  Agreed on Zulip.

## Skimage and planes

Like "row" and "column", "plane" is a semantic label.  I am going to argue
that "plane" should not be used, as it is up the particular field or library,
which axis corresponds to the meaning "plane".   We should replace use of
"plane" in the library with "i, j, k" — i.e. use Numpy axis labels to make the
intention clear.   We should instead specify the plane axis, where necessary,
with a `plane_axis=` keyword argument.

## Policy

* We will prefer axis-order labels (i, j, ...) whenever possible.
* Any reference to "row" and "column" in docstrings or variable names, that can
  also be strictly interpreted as axis orders, should change to I, J... (number
  of elements along axis) or i, j ... (index onto axes).
* When we need the user to think about semantic meanings of axes, we should:
  * Explain that this is the case (and that this is not the default case),
  * Try to move any semantic labeling to keyword arguments, of form
    `channel_axis=` and `plane_axis=`.
  * Where this fails, make it clear in documentation that this is a special
    case.

## Plane plans

Here I explore the code-base for a) routines that handle 3D images, and which
b) may depend on knowing which of the three axes is the "plane" axis in a
semantic sense.

### Transpose invariance

One tool I'll use is looking for *transpose invariance* of the function.  If
the routine does not treat any axis as different from the others, then it may
also be true that the following operations will give the same or equivalent
output (`res` and `res_retp` are the same or equivalent) on a 3D image `img`:

1. Run the function `res = func(img)`.
2. Transpose the function with some axis `ordering` other than (0, 1, 2), run
   `res_t = func(img.transpose(ordering))`, retranspose `res_t` suitably (in
   fact with `np.argsort(ordering)`) to give `res_retp`.

See `slic_3d_axes.ipynb` for an example.

If that is not true, then the function may not be transpose invariant, and I'll need to investigate further.

### Plans and investigation by found 3D-capable routines

* `draw.draw3d` has `ellipsoid` with docstring:

  ```rest
  Parameters
  ----------
  a : float
      Length of semi-axis along x-axis.
  b : float
      Length of semi-axis along y-axis.
  c : float
      Length of semi-axis along z-axis.
  spacing : 3-tuple of floats
      Grid spacing in three spatial dimensions.
  levelset : bool
      If True, returns the level set for this ellipsoid (signed level
      set about zero, with positive denoting interior) as np.float64.
      False returns a binarized version of said level set.

  Returns
  -------
  ellipsoid : (M, N, P) array
      Ellipsoid centered in a correctly sized array for given `spacing`.
      Boolean dtype unless `levelset=True`, in which case a float array is
      returned with the level set above 0.0 representing the ellipsoid.
  ```

  Plan:

  ```rest
  Parameters
  ----------
  a : float
      Length of semi-axis along i-axis.
  b : float
      Length of semi-axis along j-axis.
  c : float
      Length of semi-axis along k-axis.
  spacing : 3-tuple of floats
      Grid spacing in three image dimensions.
  levelset : bool
      If True, returns the level set for this ellipsoid (signed level
      set about zero, with positive denoting interior) as np.float64.
      False returns a binarized version of said level set.

  Returns
  -------
  ellipsoid : (I, J, K) array
      Ellipsoid centered in a correctly sized array for given `spacing`.
      Boolean dtype unless `levelset=True`, in which case a float array is
      returned with the level set above 0.0 representing the ellipsoid.
  ```

  Similarly for similar arguments in `ellipsoid_stats` function in same
  module.

* `draw/draw.py`: `rectangle`.  `start` input argument is:

  ```rest
    start : tuple
        Origin point of the rectangle, e.g., ``([plane,] row, column)``.
    end : tuple
        End point of the rectangle ``([plane,] row, column)``.
        For a 2D matrix, the slice defined by the rectangle is
        ``[start:(end+1)]``.
        Either `end` or `extent` must be specified.
    extent : tuple
        The extent (size) of the drawn rectangle.  E.g.,
        ``([num_planes,] num_rows, num_cols)``.
        Either `end` or `extent` must be specified.
        A negative extent is valid, and will result in a rectangle
        going along the opposite direction. If extent is negative, the
        `start` point is not included.
  ```

  This does assume a semantic meaning to the first dimension, and to the second
  and third, which are "row" and "column".  But it doesn't appear a semantic
  meaning is needed.  One option is to change docstring to:

  ```
    start : tuple
        Origin point of the rectangle, e.g., ``(i, j, [k])``.
    end : tuple
        End point of the rectangle ``([i, j, [k])``.
        For a 2D matrix, the slice defined by the rectangle is
        ``[start:(end+1)]``.
        Either `end` or `extent` must be specified.
    extent : tuple
        The extent (size) of the drawn rectangle.  E.g.,
        ``(n_i, n_j[, n_k])``.
        Either `end` or `extent` must be specified.
        A negative extent is valid, and will result in a rectangle
        going along the opposite direction. If extent is negative, the
        `start` point is not included.
  ```

* `draw/draw.py`: `set_color`:

  ```rest
  """Set pixel color in the image at the given coordinates.

  Note that this function modifies the color of the image in-place.
  Coordinates that exceed the shape of the image will be ignored.

  Parameters
  ----------
  image : ndarray of shape (M, N, C)
      Image
  coords : tuple of (ndarray of shape (K,), ...)
      Row and column coordinates of pixels to be colored.
  color : ndarray of shape (C,)
      Color to be assigned to coordinates in the image.
  alpha : np.number or ndarray of dtype np.number and shape (K,)
      Alpha values used to blend color with image.  0 is transparent,
      1 is opaque.
  ```

  Plan:

  ```rest
  """Set pixel color in the image at the given coordinates.

  Note that this function modifies the color of the image in-place.
  Coordinates that exceed the shape of the image will be ignored.

  Parameters
  ----------
  image : 3D array, with channel axis, e.g. shape (I, J, C)
      2D image with channel axis specifying color.  Specify color channel axis
      with `channel_axis` (default is last axis).
  coords : tuple of (ndarray of shape (N,), ...)
      Row and column coordinates of pixels to be colored.
  color : ndarray of shape (C,)
      Color to be assigned to coordinates in the image.
  alpha : np.number or ndarray of dtype np.number and shape (N,)
      Alpha values used to blend color with image.  0 is transparent,
      1 is opaque.
  channel_axis : int or None, optional
      This parameter indicates which axis of the array corresponds to
      channels.
  ```

  Add `last_channel_axis` decorator and test.

* Various `filter.rank` functions calling `_core_3D` in
  `filters/rank/core_cy_3d.pyx`.  Maybe all in `rank.generic`, e.g
  `rank.equalize`, called with these inputs:

  ```rest
  def equalize(image, footprint, out=None, mask=None, shift_x=0, shift_y=0, shift_z=0):
      """Equalize image using local histogram.

      Parameters
      ----------
      image : ndarray of shape ([P,] M, N) and dtype (uint8 or uint16)
          Input image.
      footprint : ndarray
          The neighborhood expressed as an ndarray of 1's and 0's.
      out : ndarray of shape ([P,] M, N), same dtype as input `image`
          If None, a new array is allocated.
      mask : ndarray of dtype (int or float), optional
          Mask array that defines (>0) area of the image included in the local
          neighborhood. If None, the complete image is used (default).
      shift_x, shift_y, shift_z : int
          Offset added to the footprint center point. Shift is bounded to the
          footprint sizes (center must be inside the given footprint).

      Returns
      -------
      out : ([P,] M, N) ndarray, same dtype as `image`
          Output image.
  ```

  Input image can be 3D, and arguments `shift_x`, `shift_y`, and `shift_z`. But
  the `shift_` arguments appear, oddly, to be offsets in `i, j, k` respectively
  of the `footprint`.   There are specific 3D implementations in the
  `generic.py` filters, called as `generic_cy._<name>_3D`. For example, in
  `equalize`, this is `generic_cy._equalize_3D`.  These functions are
  implemented in `generic_cy.pyx`, and each end up calling the `_core_3D`
  function in `filters/rank/core_cy_3d.pyx`.  `plane` is used in Cython
  function `_core_3D` and functions called therefrom.  So the question is
  — does `_core_3D` treat the first dimension as special in some way?
  Testing suggests no — all filters with `shift_z` (allowing 3D) give the same
  result applied directly, or applied after transposing axes in all
  permutations, applying the filter, and retransposing.   Action: change
  docstrings accordingly to `(i, j[, k])` rather than `([P,] M, N)`.

  We have a problem for the `shift_` parameters.  Their meanings are different for 2D compared to 3D.  For 2D, `shift_x` means `shift_j`, `shift_y` means `shift_i`.  For 3D, `shift_x,y,z` mean `shift_i,j,k`.  So `x` and `y` mean different things for the 3D case.

  We should deprecate `shift_{x,y,z}`, in favor of a generic `shift` argument,
  where the elements are in `i, j[, k]` order, and make this keyword only.
  The question is what message to give those using Skimage1, for porting to
  Skimage2.

  To note, Github searches suggest it is very rare to use not-default values for any of the `shift_{x,y,z}` parameters.

  Here are the searches for keyword use, split into two because the full `|` search for all rank filter names returned too many results:

  ```
  /(autolevel|equalize|gradient|majority|maximum|mean|geometric_mean|subtract_mean|pop)\(.*?,.*?,.*shift_[xyz]\s*=/ AND NOT path:skimage AND (path:.py OR path:.ipynb)
  ```

  (Only [shift_z=1](https://github.com/NeuroDataDesign/mouselit/blob/dd11ee4a600f378c760cfb072027e331e00d70f2/ryan/3D%20Equalizer/Proof%20of%203D%20Equalization.ipynb#L240))

  and

  ```
  /(threshold|noise_filter|entropy|otsu|sum|median|minimum|modal|enhance_contrast)\(.*?,.*?,.*shift_[xyz]\s*=/ AND NOT path:skimage AND (path:.py OR path:.ipynb)
  ```

  (One non-default use of `shift_x`
  [here](https://github.com/geojames/CNN-Supervised-Classification/blob/594326487372ff06e45aa8a06913fdd9b2aa90fe/code/CnnSupervisedClassification.py#L402);
  [shift to upper left corner with x and
  y](https://github.com/giuliapusc/guessTheSketch/blob/9a3f1a1387d0beb4a254da9c4b58166a1398255b/analysis/entropy_analysis.py#L25)

  (Positional argument search rather harder, but I'll assume these are no more
  common than by keyword, given shift arguments start (`shict_x`) at position 4 (fifth argument)).

  Plan:

  In Skimage2, make everything after `image, footprint` keyword only.
  Remove `shift_{x,y,z}` arguments.  Add `shift` argument with default 0.
  Shifts should be either scalar, or of same number of dimensions of `image`
  and contain footprint shifts in `i, j, k` order.  If scalar they repeat to the required number of dimensions, so `shift=2` for a 3D image is equivalent to `shift=(2, 2, 2)`.

  Skimage1 wraps Skimage2 routine, and unpacks `shift_{x,y,z}` into `shift`
  accordingly.  Deprecation message on lines of (e.g. for `equalize`):

  ```rest
  For Skimage2 ``filters.rank.equalize``, all parameters after `footprint` are
  keyword only.  In Skimage2, specify footprint shifts with `shift` keyword
  argument.  `shift` can be scalar (in which case the argument applies to all
  axes of the footprint), or it should have the same number of elements as
  there are axes in `image`. If `shift` is a sequence, specify shift in `i,
  j[, k]` order. Note that, for 2D images, `shift_x` in Skimage1 corresponds
  to `j` (the second axis of the footprint) and `shift_y` corresponds to `i`
  (first axis of footprint). Thus a call with `shift_x=1, shift_y=2` for
  Skimage1 should become `shift=(2, 1)` in Skimage2, and a call with
  `shift_y=2` (and no `shift_x`) would be `shift=(2, 0)` in Skimage2. For 3D
  images, in Skimage1, `shift_x`, `shift_y`, and `shift_z` correspond to `i,
  j, k` (the first, second and third axes of the footprint).  Thus a call with
  `shift_x=1, shift_y=2, shift_z=3` in Skimage 1 would use `shift=(1, 2, 3)`
  in Skimage2. A call with `shift_y=2` (and no specification for `shift_x` or
  `shift_z`) corresponds, in Skimage2, to `shift=(0, 2, 0)`.
  ```

  Docstring becomes:

  ```rest
  def equalize(image, footprint, *, out=None, mask=None, shift=0):
      """Equalize image using local histogram.

      Parameters
      ----------
      image : ndarray of shape (I, J,[, K]) and dtype (uint8 or uint16)
          Input image.
      footprint : ndarray
          The neighborhood expressed as an ndarray of 1's and 0's.
      out : ndarray of shape (I, J,[, K]), same dtype as input `image`, optional
          If None, a new array is allocated.
      mask : ndarray of dtype (int or float), optional
          Mask array that defines (>0) area of the image included in the local
          neighborhood. If None, the complete image is used (default).
      shift : int or sequence, optional
          Offsets added to the footprint center point. Shift is bounded to
          the footprint sizes (center must be inside the given footprint).
          If scalar, offset applies to all axes of the `footprint`.  If it
          is a sequence, it should have the same number of elements as
          `footprint` has axes, and the values should be in axis order
          (first value applies to first axis, second to second, etc).

      Returns
      -------
      out : (I, J[, K]) ndarray, same dtype as `image`
          Output image.
  ```

* `filters/_gabor.py`: `(M, N)` to `(I, J)` throughout.
* `filters/_sparse.py`: — replace `(M, N[, ...], P)` with `(I, J[,
  ...], K)`, `(M, N)` with `(I, J)`.
* `filters/morphsnakes.py`: — replace `(L, M, N)`, `(M, N)` with
  `(I, J, K)` and `(I, J)` throughout.

* `feature/corner.py`: — replace `(M, N)` with and `(I, J)` throughout.
* `feature/_daisy.py`: — replace `(M, N)` with and `(I, J)` throughout.
* `feature/_daisy.py`: — replace `(M, N)` with and `(I, J)` throughout.
* `feature/haar.py`: — replace `(M, N)` with and `(I, J)` throughout.
* `feature/_haar.pyx`: — replace `(M, N)` with and `(I, J)` throughout.
* `feature/_hog.py::hog`:

  ```rest
  image : (M, N[, C]) ndarray
      Input image.
  ...
  channel_axis : int or None, optional
      If None, the image is assumed to be a grayscale (single channel) image.
      Otherwise, this parameter indicates which axis of the array corresponds
      to channels.
  ```

  Plan:

  ```rest
  Parameters
  ----------
  image : 2D or 3D ndarray, optional color axis, e.g. shape (I, J), (I, J, 3)
      Input image, 2D grayscale or 2D RGB.  If RGB, specify color channel axis explicitly with `channel_axis`.
  ...
  channel_axis : int or None, optional
      If ``None``, the image is assumed to be grayscale (single-channel).
      Otherwise, this parameter indicates which axis of the array corresponds
      to channels.
  ```

  Also see `denoise_bilateral`.

  And replace `(M, N)` with and `(I, J)` elsewhere.

* `feature/peak.py`: `_prominent_peaks`.  Docstring:

  ```rest
  Parameters
  ----------
  image : (M, N) ndarray
      Input image.
  min_xdistance : int
      Minimum distance separating features in the x dimension.
  min_ydistance : int
      Minimum distance separating features in the y dimension.
  threshold : float
      Minimum intensity of peaks. Default is `0.5 * max(image)`.
  num_peaks : int
      Maximum number of peaks. When the number of peaks exceeds `num_peaks`,
      return `num_peaks` coordinates based on peak intensity.

  Returns
  -------
  intensity, xcoords, ycoords : tuple of array
      Peak intensity values, x and y indices.
  ```

  Plan:

  ```rest
  Parameters
  ----------
  image : (I, J) ndarray
      Input image.
  min_distance : int or sequence
      Minimum distance separating features in each dimension. Can be scalar,
      in which case corresponds to ``(m, m)`` (where ``m`` is the value of
      `min_distance`).
  threshold : float
      Minimum intensity of peaks. Default is `0.5 * max(image)`.
  num_peaks : int
      Maximum number of peaks. When the number of peaks exceeds `num_peaks`,
      return `num_peaks` coordinates based on peak intensity.

  Returns
  -------
  intensity, i_coords, j_coords : tuple of array
      Peak intensity values, i and j indices.
  ```

  Docstring for Skimage1:

  ```rest
  There are two key differences between this routine (in Skimage1) and the equivalent routine in Skimage2.

  The first is that Skimage2 returns image coordinates in reversed order to that in Skimage1.  `Skimage2` returns the `intensity`, followed by the *column* coordinates in the array (`x_coords`) followed by the *row* cooordinates (`y_coords`).   In contrast, Skimage2 returns the intensity, followed by the *row* (first / 'i' axis) coordinates, followed by the *column* coordinates (second / 'j' axis).  This is for consistency with other Skimage2 routines.

  The second is that Skimage2 specifes minimum distances with `min_distance`
  keyword argument, instead of `min_distanc_x` and `min_distance_y` arguments.
  `min_distance` can be scalar (in which case the argument applies to both
  axes of the image), or it should be a sequence of length 2. If
  `min_distance` is a sequence, specify shift in `i, j` order.

  When porting Skimage1 code to Skimage 2, the following are equivalent:

  >>> import skimage as ski1
  >>> import skimage2 as ski2
  >>>
  >>> # Skimage1
  >>> intensity, col_coords, row_coords = ski1.feature._prominent_peaks(
  ...     img, min_distance_x=1, min_distance_y=2)
  >>> # Skimage2.  Note reversal of second and third return arguments,
  >>> # and min_distance specified in row, column (i, j) order.
  >>> intensity, row_coords, col_coords = ski2.feature._prominent_peaks(
  ...     img, min_distance=(2, 1)

  ```

  Having said that, searching Github, I could find [maybe one
  attempt](https://github.com/Danieloni1/nlp-final-project/blob/369b3d789e3a0915b46de0b61b3d5166a148c398/data/raw_data/python/hough_transform.py#L62)
  at importing and using the Skimage `_prominent_peaks` routine, and that likely broken.

* `feature/template.py::match_template`:

  ```rest
  Parameters
  ----------
  image : (M, N[, P]) array
      2-D or 3-D input image.
  template : (m, n[, p]) array
      Template to locate. It must be `(m <= M, n <= N[, p <= P])`.
  ```

  See `./ai_output/gemini_match_template_invariance.md` and
  `match_template_3d_axes.ipynb`.   Function appears to be, and can be shown
  to be, transpose invariant.  There is no special treatment of particular
  axes.

  Plan:

  ```rest
  Parameters
  ----------
  image : (I, J[, J]) array
      2-D or 3-D input image.
  template : (U, V[, W]) array
      Template to locate. It must be `(U <= I, V <= J[, W <= K])`.
  ```

* `feature/texture.py`: — replace `(M, N)` with and `(I, J)` throughout.

* `rag_mean_color` in `graph/_rag.py`.  Docstring:

  ```rest
  image : ndarray, shape(M, N[, ..., P], 3)
      Input image.
  labels : ndarray, shape(M, N[, ..., P])
      The labelled image. This should have one dimension less than
      `image`. If `image` has dimensions `(M, N, 3)` `labels` should have
      dimensions `(M, N)`.
  ```

  Plan:

  ```rest
  image : N-D ndarray
      Input image.  One or no axis can contain channel (color) information
      (see `channel_axis` argument).  For example the input image can have
      3 axes (I, J, C) where the last represents color (`channel_axis` default
      is -1).
  labels : N-D or N-1-D ndarray
      The labelled image. If one axis of `image` is a color axis, `labels`
      should have one dimension less than `image`, otherwise, `labels` should
      have dimensions matching `image`.  For example, if `image` has
      dimensions `(I, J, 3)` and `channel_axis=-1` (the default), then
      `labels` should have dimensions `(I, J)`.  If `image` has dimensions
      `(I, J)` and `channel_axis=None`, then labels should also have
      dimensions `(I, J)`.

  ...

  channel_axis : int or None, optional
      If None, the `image` is assumed to be a grayscale (single channel)
      image. Otherwise, this parameter indicates which axis of the array
      corresponds to channels.
  ```

  Add `channel_axis` and use `last_channel_axis` decorator.

* `show_rag` in `graph/_rag.py`.  Docstring:

  ```rest
  labels : ndarray, shape (M, N)
      The labelled image.
  rag : RAG
      The Region Adjacency Graph.
  image : ndarray, shape (M, N[, 3])
      Input image. If `colormap` is `None`, the image should be in RGB
      format.

  ...

  img_cmap : :py:class:`matplotlib.colors.Colormap`, optional
      Any matplotlib colormap with which the image is draw. If set to `None`
      the image is drawn as it is.
  ```

  Plan:

  ```
  labels : ndarray, shape (I, J)
      The labelled image.
  rag : RAG
      The Region Adjacency Graph.
  image : N-D ndarray
      Input image.  If `channel_axis` is not None (default is -1), then image
      will have an extra axis relative to `labels`, with axis positiion
      indicated by `channel_axis`. If `channel_axis` is None then `image`
      should be same shape as `labels`.  If `img_cmap` is `None` (default is
      `'bone'`), the image should be in RGB format with channel axis indicated
      by `channel_axis`.  For example, if `labels` is shape (I, J), and
      `channel_axis=-1` (the default), then `image` should be hape (I, J, C),
      where C is the length of the color axis (3 or 4).
  ...

  img_cmap : :py:class:`matplotlib.colors.Colormap`, optional
      Any matplotlib colormap with which the image is drawn. If set to `None`
      the image is drawn using `image` colors from the `channel_axis`.

  ...

  channel_axis : int or None, optional
      If None, the image is assumed to be a grayscale (single channel) image.
      Otherwise, this parameter indicates which axis of the array corresponds
      to channels.
  ```

  Use `last_channel_axis` decorator.

* `autolevel_percentile`, `gradient_percentile`, `mean_percentile`, in
  `filters/rank/_percentile.py`.  Docstrings have form:

  ```rest
  Parameters
  ----------
  image : ndarray of shape (M, N) and dtype (uint8 or uint16)
      Input image.
  footprint : ndarray of shape (m, n)
      The neighborhood expressed as a 2-D array of 1's and 0's.
  out : ndarray of shape (M, N) and dtype int
      If None, a new array is allocated.
  mask : ndarray
      Mask array that defines (>0) area of the image included in the local
      neighborhood. If None, the complete image is used (default).
  shift_x, shift_y : int
      Offset added to the footprint center point. Shift is bounded to the
      footprint sizes (center must be inside the given footprint).
  p0, p1 : float, optional, in interval [0, 1]
      Define the [p0, p1] percentile interval to be considered for computing
      the value.

  Returns
  -------
  out : ndarray of shape (M, N) and dtype int
      Output image.
  ```

  Plan:

  ```rest
  Parameters
  ----------
  image : ndarray of shape (I, J) and dtype (uint8 or uint16)
      Input image.
  footprint : ndarray of shape (U, V)
      The neighborhood expressed as a 2-D array of 1's and 0's.
  out : ndarray of shape (I, J) and dtype int
      If None, a new array is allocated.
  mask : ndarray
      Mask array that defines (>0) area of the image included in the local
      neighborhood. If None, the complete image is used (default).
  shift_x, shift_y : int
      Offset added to the footprint center point. Shift is bounded to the
      footprint sizes (center must be inside the given footprint).
  p0, p1 : float, optional, in interval [0, 1]
      Define the [p0, p1] percentile interval to be considered for computing
      the value.

  Returns
  -------
  out : ndarray of shape (I, J) and dtype int
      Output image.
  ```

* In `filters/lpi_filter.py`: several instances of `(M, N)`.  Can all
  be replaced with `(I, J)`.  Similarly, various instances of `\(r, c, ...)`.
  Can be replaced with `(i, j, ...)` with associated renaming of `r` and `c`
  in variables and docstrings.

* `filters/edges.py`: all instances of `(M, N)` can be
  replaced with `(I, J)`.

* `filters/ridges.py`.  See exploration in `ridges_3d_axes.ipynb`, and
  investigation by Gemini in `ai_output/gemini_ridge_invariance.md`.  Summary — apparent lack of transpose invariance is due to numerical instabilities in the axis order in which the Hessian is calculated; this can be fixed with a simple numerical stability change, noted in the notebook.  There is no fundamental meaning being attributed to a third axis.  Therefore:

  Plan:

  Replace `(M, N[, ...])` with `(I, J[, ...])`, `(M, N,[ P])` with `(I, J[,
  K])`

* `io/_io.py`: `imsave`. Docstring:

  ```rest
  Parameters
  ----------
  fname : str or pathlib.Path
      Target filename.
  arr : ndarray of shape (M, N[, C]), with C=3 or C=4
      Image data.
  check_contrast : bool, optional
      Check for low contrast and print warning (default: True).
  ```

  Plan:

  ```rest
  Parameters
  ----------
  fname : str or pathlib.Path
      Target filename.
  arr : 2 or 3 dimensional ndarray
      Image data. If a 2D array, this is a grayscale image.  If a 3D array,
      this is a color image, with one axis (specified by `channel_axis`,
      default -1) giving color information.  This axis, if present, should be length 3 or 4.
  check_contrast : bool, optional
      Check for low contrast and print warning (default: True).
  channel_axis : int or None, optional
      We ignore this parameter for 2D `arr`.  If `arr` is 3D, `channel_axis`
      indicates which axis of the array corresponds to color channels.  Before
      saving the image, the function moves the channel axis to be the final
      axis, for compatibility with the libraries we use to save the data.
  ```

  We will need to do something to allow raising errors if the image is 2D and
  `channel_axis` is not in (None, -1).  I guess we can just put another
  decorator to run before `channel_as_last_axis`.  Or we can specify
  `channel_arg_positions=1` to `channel_as_last_axis` and check whether there
  are in fact 3 dimensions in the `channel_as_last_axis` decorator.
  Specifically around current line 620, check if the selected `arg` does in
  fact have three or more dimensions, and raise an error if not — see
  <https://github.com/scikit-image/scikit-image/pull/8https://github.com/scikit-image/scikit-image/pull/808308>
  and [Zulip
  discussion](https://skimage.zulipchat.com/#narrow/channel/383139-core-devs/topic/Coordinate.20review/near/577704973)
  (and following).

* `measure/_colocalization.py`: — replace `(M, N)` with and `(I, J)`
  throughout.

* `measure/_ccomp.pyx:    # Handle first plane`.  This is in
  `scan3D`, called from `label_cython` in the same file.  Here the input array
  `input_` is first axis-reordered to order axes in ascending size, before
  calling `scan3D`, and the array is then axis-reordered to match input.
  `label_cython` gets called (named `clabel`) in `_label.py` `label` function.
  This implies that we can roll axes and run algorithm, then roll back to get
  the same result.  Tested, provisionally confirmed in `label_3d_axes.ipynb`
  — labels show 1 to 1 mapping when running label on rolled versions of binary
  image.

* `measure/_find_contours.py`: — replace `(M, N)` with and `(I, J)` throughout.
* `measure/_marching_cubes_lewiner.py::marching_cubes`:

  This routine does not appear to be transpose invariant.  See `./ai_output/gemini_marching_cubes_invariance.md` and `marching_cubes_3d_axes.ipynb` (also via Gemini).  The lack of invariance seems to be an order effect of the axes, and not special-casing of a particular (e.g. plane) axis.

  See [Zulip request for
  information](https://skimage.zulipchat.com/#narrow/channel/181448-development/topic/Coordinate.20review.20for.20skimage2/near/580425795)

  Replace `(M, N, P)` with and `(I, J, K)` throughout.

* `measure/_pnpoly.pyx`: — replace `(M, N)` with and `(I, J)` throughout.
  Rename `M` and `N` variables to `I` and `J`.
* `measure/_regionprops.py::regionprops`:

  ```rest
  label_image : (M, N[, P]) ndarray
      Label image. Labels with value 0 are ignored.
  ...
  intensity_image : (M, N[, P][, C]) ndarray, optional
      Intensity (input) image of same shape as label image, plus
      optionally an extra dimension for multichannel data. Currently,
      this extra channel dimension, if present, must be the last axis.
      Default is None.
  ```

  Gemini analysis in `ai_output/gemini_regionprops.md` and
  `regionprops_3d_axes.ipynb` suggests this routine is transpose invariant;
  there is no special treatment of a plane axis.

  Plan:

  ```rest
  label_image : (I, J[, K]) ndarray
      Label image. Labels with value 0 are ignored.
  ...
  intensity_image : (I, J[, K][, C]) ndarray, optional
      Intensity (input) image of same shape as label image, plus
      optionally an extra dimension for multichannel data.  Default is None.
      Specify the color channel axis with `channel_axis`. We ignore
      `channel_axis` if `intensity_image` has the same number of
      dimensions as `label_image`.
  ...
  channel_axis : int or None, optional
      This parameter indicates which axis of `intensity_image` corresponds to
      channels.  Ignored if `label_image` has the same number of axes as `intensity_image`.
  ```

  Apply `last_channel_axis` to `intensity_image`, and test.

* `measure/_regionprops.py::regionprops_table` - as for `regionprops` above.

* `measure/_regionprops_utils.py::euler_number`:

  ```rest
  image : (M, N[, P]) ndarray
      Input image. If image is not binary, all values greater than zero
      are considered as the object.
  ```

  Plan:

  ```rest
  image : (I, J[, K]) ndarray
      Input image. If image is not binary, all values greater than zero
      are considered as the object.
  ```

  Also: replace `(M, N)` with and `(I, J)` throughout.

* `measure/entropy.py`: — replace `(M, N)` with and `(I, J)` throughout.
* `measure/pnpoly.py`: — replace `(M, N)` with and `(I, J)` throughout.
* `measure/profile.py::profile_line`:

  ```rest
  image : ndarray, shape (M, N[, C])
      The image, either grayscale (2D array) or multichannel
      (3D array, where the final axis contains the channel
      information).
  src : array_like, shape (2,)
      The coordinates of the start point of the scan line.
  dst : array_like, shape (2,)
      The coordinates of the end point of the scan
      line. The destination point is *included* in the profile, in
      contrast to standard numpy indexing.
  ```

  `src` and `dst` are in `ij` form; see docstring example and `./line_profile_args.ipynb`.

  Plan:

  ```rest
  image : 2- or 3-D array, optional channel axis, shape e.g. (I, J), (I, J, C)
      The image, either grayscale (2D array) or multichannel (3D array).  See `channel_axis` to specify
      information).
  start : array_like, shape (2,)
      The coordinates of the start point of the scan line.  These are row, column (i, j) indices into `image`.
  end : array_like, shape (2,)
      The coordinates of the end point of the scan line (row, column / i, j).
      The destination point is *included* in the profile, in contrast to
      standard Numpy indexing.
  ...
  channel_axis : int or None, optional
      This parameter indicates which axis of the `image` array corresponds to
      channels.  Ignored if `image` is 2D.
  ```

  Add `last_channel_axis` with default of -1.

* `morphology/_skeletonize.py::skeletonize`:

  ```rest
  Parameters
  ----------
  image : (M, N[, P]) ndarray of bool or int
      The image containing the objects to be skeletonized. Each connected component
      in the image is reduced to a single-pixel wide skeleton. The image is binarized
      prior to thinning; thus, adjacent objects of different intensities are
      considered as one. Zero or ``False`` values represent the background, nonzero
      or ``True`` values -- foreground.

  ...

  Returns
  -------
  skeleton : (M, N[, P]) ndarray of bool
      The thinned image.
  ```

  The function is not transpose invariant, but this does not seem to be due to
  special treatment of particular axes.  See
  `./ai_output/gemini_skeletonize_invariance.md`.  [Asked on
  Zulip](https://skimage.zulipchat.com/#narrow/channel/181448-development/topic/Coordinate.20review.20for.20skimage2/near/580701060).

  Plan:

  ```rest
  Parameters
  ----------
  image : (I, J[, K]) ndarray of bool or int
      The image containing the objects to be skeletonized. Each connected component
      in the image is reduced to a single-pixel wide skeleton. The image is binarized
      prior to thinning; thus, adjacent objects of different intensities are
      considered as one. Zero or ``False`` values represent the background, nonzero
      or ``True`` values -- foreground.

  ...

  Returns
  -------
  skeleton : (I, J[, J]) ndarray of bool
      The thinned image.
  ```

  and replace `(M, N)` with and `(I, J)` throughout.

* `morphology/convex_hull.py`: — replace `(M, N)` with and `(I, J)`
  throughout.

* `registration/_optical_flow.py::optical_flow_tvl1` and `optical_flow_ilk`
  refer to `(M, N,[ P])` input and output arguments.  There is nothing to
  indicate these treat any dimension as special.   Adequately
  transpose-invariant (`./flow_3d_axes.ipynb`).  See summary in
  `ai_output/gemini_flow_invariance.md`.  [JNI
  agrees](https://skimage.zulipchat.com/#narrow/channel/383139-core-devs/topic/Coordinate.20review/near/580592594).

  Plan:

  Replace `(M, N[, P])` with `(I, J[, K])` throughout.

* `restoration/deconvolution.py::richardson_lucy` :

  ```rest
  def richardson_lucy(image, psf, num_iter=50, clip=True, filter_epsilon=None):
      """Richardson-Lucy deconvolution.

      Parameters
      ----------
      image : ([P, ]M, N) ndarray
         Input degraded image (can be n-dimensional). If you keep the
         default `clip=True` parameter, you may want to normalize
         the image so that its values fall in the [-1, 1] interval to avoid
         information loss.
      ...
  ```

  This ends up calling `scipy.signal.convolve` on the input array, so no
  semantic meaning to the axes, and docstring better put as:

  Plan:

  ```rest
      Parameters
      ----------
      image : N-D ndarray
  ```

* `restoration/deconvolution.py` : otherwise — replace `(M, N)` with and `(I,
  J)` throughout.

* `denoise_bilateral` in `restoration/_denoise.py` has the following:

  ```rest
  Parameters
  ----------
  image : ndarray, shape (M, N[, 3])
      Input image, 2D grayscale or RGB.
  ...
  channel_axis : int or None, optional
      If ``None``, the image is assumed to be grayscale (single-channel).
      Otherwise, this parameter indicates which axis of the array corresponds
      to channels.
  ```

  [Discussion on
  Zulip](https://skimage.zulipchat.com/#narrow/channel/383139-core-devs/topic/Channel.20axis/near/580405236)
  on desirability of having specify `channel_axis` for 2D color images.

  Plan:

  ```rest
  Parameters
  ----------
  image : 2D or 3D ndarray, optional color axis, e.g. shape (I, J), (I, J, 3)
      Input image, 2D grayscale or 2D RGB.  If RGB, specify color channel axis explicitly with `channel_axis`.
  ...
  channel_axis : int or None, optional
      If ``None``, the image is assumed to be grayscale (single-channel).
      Otherwise, this parameter indicates which axis of the array corresponds
      to channels.
  ```

* `denoise_wavelet` in `restoration/_denoise.py` has the following
  docstring:

  ```rest
  Perform wavelet denoising on an image.

  Parameters
  ----------
  image : ndarray (M[, N[, ...P]][, C]) of ints, uints or floats
      Input data to be denoised. `image` can be of any numeric type,
      but it is cast into an ndarray of floats for the computation
      of the denoised image.
  ...
  ```

  As for `watershed`, one can specify the `channel_axis`.
  `denoise_3d_axes.ipynb` shows that this denoising is transpose invariant, at
  least to fairly small tolerance.

  Plan:

  Rewrite docstring to:

  ```rest
  Perform wavelet denoising on an image.

  Parameters
  ----------
  image : array of 2, 3 or 4 dimensions of ints, uints or floats
      Input image to be denoised. Can be a 2D or 3D image, in grayscale
      (`image` has 2 or 3 dimensions) or multichannel (`image` has 3 or
      4 dimensions).  For multichannel images, use the `channel_axis`
      parameter to specify the channel axis. `image` can be of any numeric
      type, but it is cast into an ndarray of floats for the computation of
      the denoised image.
  ...
  ```

* `denoise_nl_means` in `restoration/non_local_means.py`.  Docstring
  is:

  ```rest
  image : 2D or 3D ndarray
      Input image to be denoised, which can be 2D or 3D, and grayscale
      or RGB (for 2D images only, see ``channel_axis`` parameter). There can
      be any number of channels (does not strictly have to be RGB).

  ...

  channel_axis : int or None, optional
      If None, the image is assumed to be a grayscale (single channel) image.
      Otherwise, this parameter indicates which axis of the array corresponds
      to channels.
  ```

  In fact, it seems that the routine will handle 3 or 4D images with a color
  channel (up to 5 dimensional array).

  `nl_means_3d_axes.ipynb` shows that the routine is transpose invariant;
  therefore there is no special meaning that needs to be attached to a "plane"
  axis.

  There are various uses of the term "plane" in `restoration/_nl_means_denoising.pyx`.  Plane seems to always mean "slice along first axis".

  Plan:

  Rewrite docstring as:

  ```rest
  image : array of 2 to 5 dimensions
      Input image. Can be a 2D, 3D or 4D image, in grayscale (`image` has 2,
      3 or 4 dimensions) or multichannel (`image` has 3, 4 or 5 dimensions).
      For multichannel images, use the `channel_axis` parameter to specify the
      channel axis.  There can be any number of channels (does not strictly have to be RGB).
  ...
  ```

  Replace uses of plane, row, column in Cython code with i, j, k.

* `segmentation/boundaries.py::mark_boundaries`:

  Routine marks boundaries in color, thus promoting a input grayscale
  image to color.

  ```rest
  Parameters
  ----------
  image : ndarray of shape (M, N[, 3])
      Grayscale or RGB image.
  label_img : ndarray of shape (M, N) and dtype int
      Label array where regions are marked by different integer values.
  ...

  Returns
  -------
  marked : ndarray of shape (M, N, 3) and dtype float
      An image in which the boundaries between labels are
      superimposed on the original image.
  ```

  Plan:

  ```rest
  Parameters
  ----------
  image : 2D or 3D ndarray, optional color channel e.g (I, J), (I, J, C)
      Grayscale or RGB image.   Indicate axis containing color channel
      with `channel_axis`.
  label_img : ndarray of shape (M, N) and dtype int
      Label array where regions are marked by different integer values.
  ...
  channel_axis : int or None, optional
      This parameter indicates which axis of the `image` array corresponds to
      color channels.  Ignored if `image` is 2D.

  Returns
  -------
  marked : 3D ndarray, eg shape (I, J, 3) and dtype float
      An image in which the boundaries between labels are superimposed
      on the original image.  If the input was grayscale (2D), output
      is color (3D), with channel axis last.  If input was color (3D) then output has same shape as input, with color channel indicated by the input `channel_axis` argument.
  ```

  Add `last_channel_axis` decorator, and rearrange axes of output image accordingly.

* `segmentation/_chan_vese.py`: all instances of `(M, N)` can be
  replaced with `(I, J)`.

* `segmentation/random_walker_segmentation.py::random_walker`.

  ```rest
  data : (M, N[, P][, C]) ndarray
      Image to be segmented in phases. Gray-level `data` can be two- or
      three-dimensional; multichannel data can be three- or four-
      dimensional with `channel_axis` specifying the dimension containing
      channels. Data spacing is assumed isotropic unless the `spacing`
      keyword argument is used.
  labels : (M, N[, P]) array of ints
      Array of seed markers labeled with different positive integers
      for different phases. Zero-labeled pixels are unlabeled pixels.
      Negative labels correspond to inactive pixels that are not taken
      into account (they are removed from the graph). If labels are not
      consecutive integers, the labels array will be transformed so that
      labels are consecutive. In the multichannel case, `labels` should have
      the same shape as a single channel of `data`, i.e. without the final
      dimension denoting channels.
  ...
  spacing : iterable of floats, optional
      Spacing between voxels in each spatial dimension. If `None`, then
      the spacing between pixels/voxels in each dimension is assumed 1.
  ...
  channel_axis : int or None, optional
      If None, the image is assumed to be a grayscale (single channel) image.
      Otherwise, this parameter indicates which axis of the array corresponds
      to channels.
  ```

  Random-walker is transpose invariant: see `./random_walker_3d_axes.ipynb`.

  Plan:

  ```rest
  data : 2-4D ndarray, optional channel axis, e.g (I, J), (I, J, K), (I, J, C)
      Image to be segmented in phases. Gray-level `data` can be two- or
      three-dimensional; multichannel data can be three- or four-
      dimensional with `channel_axis` specifying the dimension containing
      channels. Data spacing is assumed isotropic unless the `spacing`
      keyword argument is used.
  labels : (I, J[, K]) array of ints
      Array of seed markers labeled with different positive integers
      for different phases. Zero-labeled pixels are unlabeled pixels.
      Negative labels correspond to inactive pixels that are not taken
      into account (they are removed from the graph). If labels are not
      consecutive integers, the labels array will be transformed so that
      labels are consecutive. In the multichannel case, `labels` should have
      the same shape as a single channel of `data`, i.e. without the final
      dimension denoting channels.
  ...
  spacing : iterable of floats, optional
      Spacing between voxels in each spatial dimension. If `None`, then
      the spacing between pixels/voxels in each dimension is assumed 1.
  ...
  channel_axis : int or None, optional
      If None, the image is assumed to be a grayscale (single channel) image.
      Otherwise, this parameter indicates which axis of the array corresponds
      to channels.
  ```

* `segmentation/_watershed.py` `watershed` function:

  ```rest
  def watershed(
      image,
      markers=None,
      connectivity=1,
      offset=None,
      mask=None,
      compactness=0,
      watershed_line=False,
  ):
    """Find watershed basins in an image flooded from given markers.

    Parameters
    ----------
    image : (M, N[, ...]) ndarray
        Data array where the lowest value points are labeled first.
    markers : int, or (M, N[, ...]) ndarray of int, optional
        The desired number of basins, or an array marking the basins with the
        values to be assigned in the label matrix. Zero means not a marker. If
        None, the (default) markers are determined as the local minima of
        `image`. Specifically, the computation is equivalent to applying
        :func:`skimage.morphology.local_minima` onto `image`, followed by
        :func:`skimage.measure.label` onto the result (with the same given
        `connectivity`). Generally speaking, users are encouraged to pass
        markers explicitly.
    connectivity : int or ndarray, optional
        The neighborhood connectivity. An integer is interpreted as in
        ``scipy.ndimage.generate_binary_structure``, as the maximum number
        of orthogonal steps to reach a neighbor. An array is directly
        interpreted as a footprint (structuring element). Default value is 1.
        In 2D, 1 gives a 4-neighborhood while 2 gives an 8-neighborhood.
    offset : array_like of shape image.ndim, optional
        The coordinates of the center of the footprint.
    mask : (M, N[, ...]) ndarray of bools or 0's and 1's, optional
        Array of same shape as `image`. Only points at which mask == True
        will be labeled.
    compactness : float, optional
        Use compact watershed [1]_ with given compactness parameter.
        Higher values result in more regularly-shaped watershed basins.
    watershed_line : bool, optional
        If True, a one-pixel wide line separates the regions
        obtained by the watershed algorithm. The line has the label 0.
        Note that the method used for adding this line expects that
        marker regions are not adjacent; the watershed line may not catch
        borders between adjacent marker regions.

    Returns
    -------
    out : ndarray
        A labeled matrix of the same type and shape as `markers`.
  ```

  This looks like a generic N-D function.  However, tests of the type in
  `watershed_3d_axes.ipynb` show that the function is not transpose invariant,
  in the presence of tied values in the image.  This is expected, because, in
  the presence of ties, ties identified first in the neighborhood get
  processed first, and the axis ordering dictates the ordering of neighbors.
  However, if I make the image values unique, the routine is transpose
  invariant (see the notebook).

  Plan:

  Change `(M, N[, ...])` to `(I, J[, ...])`.  Consider adding note about ties
  and axis ordering in the code comments — it's probably too niche to put into
  the docstring.

* `segmentation/slic_superpixels.py`: `slic` function.  Docstring:

  ```rest
  Segments image using k-means clustering in Color-(x,y,z) space.

  Parameters
  ----------
  image : (M, N[, P][, C]) ndarray
      Input image. Can be 2D or 3D, and grayscale or multichannel
      (see `channel_axis` parameter).
      Input image must either be NaN-free or the NaN's must be masked out.
  ...
  channel_axis : int or None, optional
      If None, the image is assumed to be a grayscale (single channel) image.
      Otherwise, this parameter indicates which axis of the array corresponds
      to channels.
  ```

  Related, we have this in the called `segmentation/_slic.pyx:_slic_cython` docstring:

  ```rest
  Helper function for SLIC segmentation.

  Parameters
  ----------
  image_zyx : 4D array of np_floats, shape (Z, Y, X, C)
      The input image.
  mask : 3D array of bool, shape (Z, Y, X), optional
      The input mask.
  segments : 2D array of np_floats, shape (N, 3 + C)
      The initial centroids obtained by SLIC as [Z, Y, X, C...].

  ...

  Returns
  -------
  nearest_segments : 3D array of int, shape (Z, Y, X)
      The label field/superpixels found by SLIC.

  Notes
  -----
  The image is considered to be in (z, y, x) order, which can be
  surprising. More commonly, the order (x, y, z) is used. However,
  in 3D image analysis, 'z' is usually the "special" dimension, with,
  for example, a different effective resolution than the other two
  axes. Therefore, x and y are often processed together, or viewed as
  a cut-plane through the volume. So, if the order was (x, y, z) and
  we wanted to look at the 5th cut plane, we would write::

      my_z_plane = img3d[:, :, 5]

  but, assuming a C-contiguous array, this would grab a discontiguous
  slice of memory, which is bad for performance. In contrast, if we
  see the image as (z, y, x) ordered, we would do::

      my_z_plane = img3d[5]

  and get back a contiguous block of memory. This is better both for
  performance and for readability.
  ```

  Testing finds that `slic` is transpose invariant for the `cell3d` images,
  but not for the `brain` image.  Gemini asserts that this is due to ties in
  the k-means algorithm leading to slightly different regions within
  background — see `ai_outputs/slic_transpose_invariance.md`.  Indeed the
  notebook `slic_3d_axes.ipynb` confirms invariance for a version of the brain
  image without ties.  [JNI asserted invariance on
  Zulip](https://skimage.zulipchat.com/#narrow/channel/383139-core-devs/topic/Coordinate.20review/near/576963786).

  Note that the channel axis can be any axis, but the `channel_as_last_axis`
  decorator will transpose the channel axis to be the final axis before
  delegating to the function, and retranspose on output.

  Plan:

  Rewrite `slic`: docstring from the above, to:

  ```rest
  Parameters
  ----------
  image : array of 2, 3 or 4 dimensions
      Input image. Can be a 2D or 3D image, in grayscale (`image` has 2 or 3 dimensions) or multichannel (`image` has 3 or 4 dimensions).  For multichannel images, use the `channel_axis` parameter to specify the channel axis. Input image must either be NaN-free or the
      NaNs must be masked out.
  ...
  ```

  Rewrite `_slic_cython` docstring from the above, to:

  ```rest
  Helper function for SLIC segmentation.

  Parameters
  ----------
  image_ijkc : 4D array of np_floats, shape (I, J, K, C)
      The input image.
  mask : 3D array of bool, shape (I, J, K), optional
      The input mask.
  segments : 2D array of np_floats, shape (N, 3 + C)
      The initial centroids obtained by SLIC as [I, J, K, C...].

  ...

  Returns
  -------
  nearest_segments : 3D array of int, shape (I, J, K)
      The label field/superpixels found by SLIC.

  Notes
  -----
  The performance of this routine may depend on the memory ordering of the
  `image_ijkc`.   This routine was designed with the assumption that the array
  is C-contiguous, so that the last axis is the fastest-changing in memory,
  and the first axis is the slowest-changing in memory.  As for standard
  routines operating on Numpy arrays, this function will work with any memory
  ordering, but it may well be slower for memory orderings other than
  C-contiguous. Please benchmark your code if peformance of this routine is
  important to you.  Please also consider letting us know your results via our
  Github issues, so we have data with which to optimize this routine.
  ```

* `segmentation/thresholding.py`: all instances of `(M, N)` can be
  replaced with `(I, J)`.  All instances of `(M, N[, ...])` can be replaced by
  `(I, J[, ...])`.

* `segmentation/active_contour_model.py::active_contour`:

  ```rest
  Parameters
  ----------
  image : ndarray of shape (M, N[, 3])
      Input image.

  ...

  Examples
  --------
  >>> from skimage.draw import circle_perimeter
  >>> from skimage.filters import gaussian

  Create and smooth image:

  >>> img = np.zeros((100, 100))
  >>> rr, cc = circle_perimeter(35, 45, 25)
  >>> img[rr, cc] = 1
  >>> img = gaussian(img, sigma=2, preserve_range=False)

  Initialize spline:

  >>> s = np.linspace(0, 2*np.pi, 100)
  >>> init = 50 * np.array([np.sin(s), np.cos(s)]).T + 50

  Fit spline to image:

  >>> snake = active_contour(img, init, w_edge=0, w_line=1)  # doctest: +SKIP
  >>> dist = np.sqrt((45-snake[:, 0])**2 + (35-snake[:, 1])**2)  # doctest: +SKIP
  >>> int(np.mean(dist))  # doctest: +SKIP
  25
  ```

  Gemini (`./ai_output/gemini_summary_xy.md`) spotted that the doctest above
  mixes up row and column distance.  See `./active_contour_doctest_bug.ipynb`
  and fix below.

  Plan:

  ```rest
  Parameters
  ----------
  image : 2D/3D array, optional channel axis — e.g shape (I, J) or (I, J, C)
      Input image.  Use `channel_axis` argument to specify axis with color
      channel.
  ...
  channel_axis : int or None, optional
      This parameter indicates which axis of the `image` array corresponds to
      color channels.  Ignored if `image` is 2D.

  ...

  Examples
  --------
  >>> from skimage.draw import circle_perimeter
  >>> from skimage.filters import gaussian

  Create and smooth image:

  >>> img = np.zeros((100, 100))
  >>> rr, cc = circle_perimeter(35, 45, 25)
  >>> img[rr, cc] = 1
  >>> img = gaussian(img, sigma=2, preserve_range=False)

  Initialize spline:

  >>> s = np.linspace(0, 2*np.pi, 100)
  >>> init = 50 * np.array([np.sin(s), np.cos(s)]).T + 50

  Fit spline to image:

  First column of output `snake` is row (i) coordinate, second is column (j) coordinate.

  >>> snake = active_contour(img, init, w_edge=0, w_line=1)  # doctest: +SKIP

  Recall above that the circle center is at i=35, j=45, radius is 25.  Our
  calculated snake should follow the original circle:

  >>> dist = np.sqrt((35 - snake[:, 0]) ** 2 +
                     (45 - snake[:, 1]) ** 2)  # doctest: +SKIP
  >>> np.all(np.round(dist) == 25)
  np.True_
  ```

  Add `last_channel_axis` decorator.

* `segmentation/_quickshift.py`: `quickshift`:

  ```rest
  Parameters
  ----------
  image : ndarray of shape (M, N, C)
      Input image. The axis corresponding to color channels can be specified
      via the `channel_axis` argument.
  ```

  Plan:

  ```rest
  Parameters
  ----------
  image : 3D array, representing 2D image, with channel axis, e.g (I, J, C)
      Use the `channel_axis` parameter to specify the array axis corresponding to color channels.
  ```

* `segmentation/_quickshift_cy.pyx` - replace `(M, N, C)` and `(M,
  N)` with `(I, J, C)` and `(I, J)` throughout.
* `segmentation/_felzenszalb.py::felzenszalb`:

  ```rest
  Parameters
  ----------
  image : ndarray of shape (M, N[, 3])
      Input image.
  ...
  Returns
  -------
  segment_mask : ndarray of shape (M, N)
      Integer mask indicating segment labels.
  ```

  Plan:

  ```rest
  Parameters
  ----------
  image : 2- or 3-D array, with optional channel axis, e.g (I, J), or (I, J, C)
      Use the `channel_axis` parameter to specify the array axis corresponding to color channels, if any.
  ...
  Returns
  -------
  segment_mask : ndarray of shape (I, J)
      Integer mask indicating segment labels.
  ```

* `segmentation/_felzenszalb_cy.pyx` - replace `(M, N, C)` and `(M, N)` with
  `(I, J, C)` and `(I, J)` throughout.

* `util/compare.py`: — replace `(M, N)` with and `(I, J)` throughout.
* `util/unique.py::unique_rows`:

  ```rest
  Parameters
  ----------
  ar : ndarray, shape (M, N)
      The input array.

  Returns
  -------
  ar_out : ndarray, shape (P, N)
      A copy of the input array with repeated rows removed.

  ```

  Plan:

  ```rest
  Parameters
  ----------
  ar : ndarray, shape (I, J)
      The input array.

  Returns
  -------
  ar_out : ndarray, shape (U, J)
      A copy of the input array with repeated rows removed.

  ```

* See the "Checked" section in `plane.md` for other instances (via `grep`) of
  "plane" in the codebase, where reference does _not_ refer to semantic
  labeling of plane in array.
* See the "Checked" section in `mn.md` for other instances (via `grep`) of
  "M, N" in the codebase, where reference does _not_ refer to semantic labeling
  of plane in array.  `mn.md` started as the output of `rg "\(M\s*,\s*N"` on codebase.

## x and y in variables and docstrings

We have various instances of "x" and "y" in signatures, docstrings and code.

Most of the time, "x" means "column" or "j", and "y" means "row" or "i".

In those cases, we should have the following policy:

* Deprecate use of "x" and "y" (when they have meaning "column" and "row"),
  and prefer "j" and "i" (or maybe "column" and "row").  We should also either
  make these keyword-only, or change the default parameter order to prefer "i",
  "j" order rather than "column", "row" order.
* We may want to either:

  a) add an extra keyword argument similar to `coord_convention='xy' | 'ij'` or
  b) recommend `skimage2` version of same function, that uses "ij" by default.

  For compatibility, we will need either the keyword argument (as above) or some helper to explain how to achieve the "x, y" effect using the Skimage2 "i, j" implementation.

### Examples of "x", "y"

#### `predict_x`, `predict_y`, `predict_xy` in transform classes

The transform classes use "x" to mean the first column in an n by 2 coordinate
array, and "y" to mean the second.  And by convention, they are written as if
they expect the "x" coordinate to be image column coordinates, and "y" to be
image row coordinates.  However, there is no reason in the code that the first
column could not be image row coordinates, and the second be image column
coordinates.  So `predict_x` in fact means, predict coordinate in first
coordinate column, and so on.

We have various options, discussed on 10 March meeting.

*Preferred option* is to deprecate `predict_x`, `predict_`, in favor of using
`predict`. `predict_x` delegates to `predict(y, axis=1)[:, 0]`, `predict_y`
delegates to `predict(x, axis=0)[:, 1]`. We have `predict_xy` for `CircleModel`
and `EllipseModel`.  We will rename those to `predict_coords`. Rename
`predict(..., axis=` argument to `from_axis`.  There was some discussion of
whether to return 2D coordinates from e.g `predict(y, from_axis=1)` — therefore
returning (in this case) the input `y` as the second column of the output — as
is currently true, or to return only the generated column.  Note that `predict`
for `LineModelND` returns an N by D array, where D is the dimensionality of the
line.   Discussion centered on what use the user was likely to make of the
output — are they going to use it as coordinates (2D array for e.g. 2D line) or
as a column of coordinates (1D array for e.g. 2D line).  I (MB) argued that
changing the current API to return only the generated coordinate column(s)
would start to become confusing / inconvenient for the `LineModelND` case,
because the user will have to do some error-prone gymnastics to put the input
vector back into the output array.  Therefore I lean towards leaving the API
is-is.

Other options considered:

*  We could simply rename to reflect the fact that `x` and `y` correspond to
   the first and second coordinates columns, as in `predict_c0`, `predict_c1`,
   `predict_c01`, or `predict_0`, `predict_1`, `predict_01`.
*  Or we could commit to the interpretation of the coordinates, and rename to
   `predict_i`, `predict_j`, and `predict_ij`.

Github searches: `/\.predict_x\(/ AND skimage AND NOT path:skimage` suggests
there isn't much code out there using these functions.

#### Hough transforms

The public API comes from `transform/hough_transform.py`.  Cython implementation in `transform/_hough_transform.pyx`.

The key and difficult change is that the *Hough space* will change
from angles from the j (x) axis, clockwise, to angles from the
i (y) axis, counterclockwise.

See Gemini analysis in `ai_output/gemini_hough_analysis.md`.

* `hough_line`.   Notes should change from:

  ```rest
  Notes
  -----
  The origin is the top left corner of the original image.
  X and Y axis are horizontal and vertical edges respectively.
  The distance is the minimal algebraic distance from the origin
  to the detected line.
  The angle accuracy can be improved by decreasing the step size in
  the `theta` array.
  ```

  to:

  ```rest
  Notes
  -----
  The origin is the top left corner of the original image. i and j axes
  are vertical and horizontal edges respectively (first and second axes
  of the input `image` array).  Angles are anti-clockwise from the
  i axis (towards the j axis).

  The distance is the minimal algebraic distance from the origin to the
  detected line.  Distance is negative if the distance should be
  interpreted as having a negative i (first axis) direction.

  The angle accuracy can be improved by decreasing the step size in the
  `theta` array.
  ```

  Porting notes:

  ```rest
  The difference between this routine (in Skimage1) and the equivalent
  routine in Skimage2, is that the returned `hspace`, `angles` and
  `distances` values define lines relative to the i-axis (the first axis
  of the `image` array). Positive angles denote anti-clockwise rotation
  from that axis. In Skimage1, `angles` and `distances` are relative to
  the j-axis (the second axis of the `image` array), and positive angles
  denote clockwise rotation from that axis.

  To help porting, we have a utility routine, `hough_ij2ji`, that takes
  the `hspace`, `angles` and `distances` output from the Skimage2 *i, j*
  convention, and returns `hspace`, `angles` and `distance` for the
  Skimage1 *j, i* (column, row, *x, y*) convention.

  When porting Skimage1 code to Skimage 2, the following are equivalent:

  >>> import skimage as ski1
  >>> import skimage2 as ski2
  >>>
  >>> # Skimage1
  >>> hs1, angles1, dist1 = ski1.transform.hough_transform(img)
  >>> # Skimage2 equivalent to Skimage1 code.
  >>> hs2, angles2, dist2 = ski2.transform.hough_transform(img)
  >>> hs1_equiv, angles1_equiv, dist1_eqiv = ski2.transform.hough_ij2ji(
  ...     (hs2, angles2, dist2)

  ```

* `probabilistic_hough_line`: 

  ```rest
  Returns
  -------
  lines : list
    List of lines identified, lines in format ((x0, y0), (x1, y1)),
    indicating line start and end.

  ```

  Should be:

  ```rest
  Returns
  -------
  lines : list
    List of lines identified, lines in format ((i0, j0), (i1, j1)),
    indicating line start and end in terms of first and second axis of the `image` array.
  ```

  Porting notes:

  ```rest
  The difference between this routine (in Skimage1) and the equivalent
  routine in Skimage2, is that the returned `lines` are coordinates in
  Numpy convention, with the index on the first `image` axis first, and
  that on the second, second (see the docstring).  Put another way, the
  Skimage2 returned coordinates are in *i, j* order, whereas those in
  Skimage1 are in *j, i* (AKA column, row, or *x, y*) order.

  When porting Skimage1 code to Skimage 2, the following are equivalent:

  >>> import skimage as ski1
  >>> import skimage2 as ski2
  >>>
  >>> # Skimage1
  >>> lines1 = ski1.transform.probabilisitic_hough_transform(
  ...     img)
  >>> # Skimage2 equivalent to Skimage1 code.
  >>> lines2 = ski2.transform.probabilisitic_hough_transform(
  ...     img)
  >>> # Reverse coordinate order.
  >>> lines1_equiv = [((j0, i0), (j1, i1)) for ((i0, j0), (i1, j1)) in lines2]
  ```

* `hough_circle_peaks`.  This accepts the outputs of `hough_circle`, and
  returns coordinates, noted to be in x, y (i.e. (j, i) or (col, row))
  order.

  ```rest
  Parameters
  ----------
  hspaces : (M, N, P) array
      Hough spaces returned by the `hough_circle` function.
  radii : (M,) array
      Radii corresponding to Hough spaces.
  min_xdistance : int, optional
      Minimum distance separating centers in the x dimension.
  min_ydistance : int, optional
      Minimum distance separating centers in the y dimension.
  threshold : float, optional
      Minimum intensity of peaks in each Hough space.
      Default is `0.5 * max(hspace)`.
  ...
  Returns
  -------
  accum, cx, cy, rad : tuple of array
      Peak values in Hough space, x and y center coordinates and radii.
  ```

  Should be:

  ```rest
  Parameters
  ----------
  hspaces : (RI, I2R, J2R) array
      Hough spaces returned by the `hough_circle` function.
  radii : (RI,) array
      Radii corresponding to Hough spaces.
  \*
  min_distance : int or length 2 sequence, optional
      Minimum distance separating centers in the i, j dimensions.
  threshold : float, optional
      Minimum intensity of peaks in each Hough space.
      Default is `0.5 * max(hspace)`.
  ...
  Returns
  -------
  accum, ci, cj, rad : tuple of array
      Peak values in Hough space, i and j center coordinates and radii.
  ```

  Porting notes:

  ```rest
  TBA

  Changes:

  * Keyword only after 2
  * `min_distance` instead of `min_xdistance`, `min_ydistance`, and
    `min_distance` in i, j order.
  * Coordinates returned in i, j order not j, i order.
  ```

* `hough_ellipse`: meaning of returned `orientation` in Skimage1 is clockwise
  rotation from the *j* (second) axis (see `draw.py:ellipse_perimeter`.  In
  Skimage2 it will be anti-clockwise rotation from the *i* (first) axis.

  ```rest
  Returns
  -------
  result : ndarray with fields [(accumulator, yc, xc, a, b, orientation)].
      Where ``(yc, xc)`` is the center, ``(a, b)`` the major and minor
      axes, respectively. The `orientation` value follows the
      `skimage.draw.ellipse_perimeter` convention.

  ...

  Examples
  --------
  >>> from skimage.transform import hough_ellipse
  >>> from skimage.draw import ellipse_perimeter
  >>> img = np.zeros((25, 25), dtype=np.uint8)
  >>> rr, cc = ellipse_perimeter(10, 10, 6, 8)
  >>> img[cc, rr] = 1
  >>> result = hough_ellipse(img, threshold=8)
  >>> result.tolist()
  [(10, 10.0, 10.0, 8.0, 6.0, 0.0)]
  ```

  Claude noted the inconsistency of application of `rr`, `cc` above.

  Plan:

  ```rest
  Returns
  -------
  result : ndarray with fields [(accumulator, rc, cc, a, b, orientation)].
      Where ``(rc, cc)`` is the center, ``(a, b)`` the major and minor
      axes, respectively. The `orientation` value follows the
      `skimage.draw.ellipse_perimeter` convention, where positive values correspond to anti-clockwise rotation from the *i* axis (first image axis).

  ...

  Examples
  --------
  >>> from skimage.transform import hough_ellipse
  >>> from skimage.draw import ellipse_perimeter
  >>> img = np.zeros((25, 25), dtype=np.uint8)
  >>> rr, cc = ellipse_perimeter(11, 14, 6, 8)
  >>> img[rr, cc] = 1
  >>> result = hough_ellipse(img, threshold=8)
  >>> result.tolist()
  [(10, 11.0, 14.0, 8.0, 6.0, 0)]
  ```

  with associated implementation.

  Porting notes:

  ```rest
  There one difference in the returned output between this routine (in
  Skimage1) and the equivalent routine in Skimage2 — and that is the
  interpretation of the output `orientation` parameter.

  In Skimage2, `orientation` is the angle of the main axis of the ellipse,
  where the angle is anti-clockwise from the first (*i*) axis of an image
  array.

  In Skimage1, `orientation` was the angle, from the second (*j*) axis of the
  image array, clockwise.

  When porting Skimage1 code to Skimage 2, the following are equivalent:

  >>> import skimage as ski1
  >>> import skimage2 as ski2
  >>>
  >>> # Skimage1
  >>> accum,  = hough_ellipse(img, threshold=8)
  >>> # Skimage2 equivalent to Skimage1 code.
  >>> angle2 = np.pi / 2 - angle1  # Skimage2 equivalent angle.
  >>> rr2, cc2 = ski2.draw.ellipse_permeter(10, 10, 6, 8, orientation=angle2)
  ```

* `hough_line_peaks`: clarification in docstring.

  (Note — it operates on an accumulator (Hough space) array provided to it, so should be agnostic to the theta / rho interpretations of that accumulator.

  ```rest
  Examples
  --------
  >>> from skimage.transform import hough_line, hough_line_peaks
  >>> from skimage.draw import line
  >>> img = np.zeros((15, 15), dtype=bool)
  >>> rr, cc = line(0, 0, 14, 14)
  >>> img[rr, cc] = 1
  >>> rr, cc = line(0, 14, 14, 0)
  >>> img[cc, rr] = 1
  >>> hspace, angles, dists = hough_line(img)
  >>> hspace, angles, dists = hough_line_peaks(hspace, angles, dists)
  >>> len(angles)
  2
  ```

  Plan:

  ```rest
  Examples
  --------
  >>> from skimage.transform import hough_line, hough_line_peaks
  >>> from skimage.draw import line
  >>> img = np.zeros((15, 15), dtype=bool)
  >>> rr, cc = line(0, 0, 14, 14)  # Diagonal top-left -> bottom-right.
  >>> img[rr, cc] = 1
  >>> rr, cc = line(0, 14, 14, 0)  # Bottom-left -> top-right.
  >>> img[rr, cc] = 1
  >>> hspace, angles, dists = hough_line(img)
  >>> hspace, angles, dists = hough_line_peaks(hspace, angles, dists)
  >>> len(angles)
  2
  ```

**Docstrings**

Note multiple instances in both files of of `(M, N)`, replace with `(I, J)`.

Output specification for `H` in `hough_circle`: edit for M, N etc.

#### Draw module

* `draw/draw.py::ellipse_perimeter`: 

  ```rest
  Parameters
  ----------
  r, c : int
      Centre coordinate of ellipse.
  r_radius, c_radius : int
      Minor and major semi-axes. ``(r/r_radius)**2 + (c/c_radius)**2 = 1``.
  orientation : double, optional
      Major axis orientation in clockwise direction as radians.
  ```

  `orientation` here is a clockwise angle from the *j* (second) axis.

  Should be (will be):

  ```rest
  Parameters
  ----------
  r, c : int
      Centre coordinate of ellipse.
  r_radius, c_radius : int
      Minor and major semi-axes. ``(r/r_radius)**2 + (c/c_radius)**2 = 1``.
  orientation : double, optional
      Major axis orientation in anti-clockwise direction from first array
      axis` (*i* axis), as radians.
  ```

  Porting notes:

  ```rest
  The difference between this routine (in Skimage1) and the equivalent
  routine in Skimage2, is that the meaning (and therefore, value) for the input `orientation` parameter.

  In Skimage2, `orientation` is the angle of the main axis of the ellipse,
  where the angle is anti-clockwise from the first (*i*) axis of an image
  array.

  In Skimage1, `orientation` was the angle, from the second (*j*) axis of the
  image array, clockwise.

  When porting Skimage1 code to Skimage 2, the following are equivalent:

  >>> import skimage as ski1
  >>> import skimage2 as ski2
  >>>
  >>> # Skimage1
  >>> angle1 = np.pi / 8  # 22.5 degrees rotation.
  >>> rr1, cc1 = ski1.draw.ellipse_permeter(10, 10, 6, 8, orientation=angle1)
  >>> # Skimage2 equivalent to Skimage1 code.
  >>> angle2 = np.pi / 2 - angle1  # Skimage2 equivalent angle.
  >>> rr2, cc2 = ski2.draw.ellipse_permeter(10, 10, 6, 8, orientation=angle2)
  ```

#### Other x, y examples

* `skimage.feature.corner.structure_tensor` and `hessian_matrix` have an
  `order='rc' | 'xy'` input parameter that default to "rc" (meaning "ij").  Here's the parameter description from the docstring:

  ```rest
    order : {'rc', 'xy'}, optional
        NOTE: 'xy' is only an option for 2D images, higher dimensions must
        always use 'rc' order. This parameter allows for the use of reverse or
        forward order of the image axes in gradient computation. 'rc' indicates
        the use of the first axis initially (Arr, Arc, Acc), whilst 'xy'
        indicates the usage of the last axis initially (Axx, Axy, Ayy).
  ```

  There are currently 6 uses of this `'xy'` option on Github with search
  `/structure_tensor\(.*,.*["']xy["']/  AND NOT path:test_corner.py`
  (to avoid copies of the Skimage test suite).  Five are within `nematic` and
  `NematicTL` repos by `viciya`; one is in a project repo
  `tiagopetena/computer_vision`.  These could be resolved by suitable PRs.  But `/hessian_matrix\(.*,.*["']xy["']/  AND NOT path:test_corner.py` occurs in 41 files, so that would be some work to resolve with PRs.  In neither case is the `order='xy'` case easy to pull out with a helper function.

  See [Discussion on Zulip](https://skimage.zulipchat.com/#narrow/channel/181448-development/topic/Coordinate.20review.20for.20skimage2/near/580440609) and Gemini analysis in `./ai_output/gemini_hessian_structure_tensor.md`.

  Plan:

  Remove `order=` keyword for Skimage2 implementation.

  Make everything after `sigma` (the second argument) keyword-only.

  Porting docstring:

  ```rest
  Skimage2:

  * Makes all arguments after `sigma` keyword only.
  * Removes the `order=` argument, standardizing to the ij (row-column)
    ordering of the outputs.

  >>> import skimage as ski1
  >>> import skimage2 as ski2
  >>>
  >>> square = np.zeros((5, 5))
  >>> H1 = ski1.feature.hessian_matrix(square, sigma=0.1, order='xy')
  >>> # Reverse output element order to reproduce 'xy'.
  >>> # Note: this will differ from Skimage1 result by a small amount due
  >>> # to some precision differences in calculation.
  >>> H1_equiv = ski2.feature.hessian_matrix(square, sigma=0.1)[::-1]
  ```

  However, reversing arguments like this can give substantially different
  answers for `hessian_matrix` (but not `structure_tensor`); see
  `./error_hessian_structure_tensor.ipynb`.

  Alternatively (**check**), use porting function to do transformation:

  ```rest
  >>> H1 = ski1.feature.hessian_matrix(square, sigma=0.1, order='xy')
  >>> # Reverse recision differences in calculation.
  >>> H2 = ski2.feature.hessian_matrix(square, sigma=0.1)
  >>> H1_equiv = ski2.porting.rc_to_xy(H2)
  ```

  See AI analysis above for potential implementation of `rc_to_xy`.

* `transform/_geometric.py`: — replace `(M, N)` with and `(I, J)` throughout.

* `transform/radon_transform.py`: `iradon_sart`.  Replace (M, N) with (I, J),
  and edit rest of docstring to match.

* `transform/_radon_transform.pyx`: Replace (M, N) with (I, J) throughout.

* `transform/_warps.py`: `rescale`

  ```rest
  Parameters
  ----------
  image : (M, N[, ...][, C]) ndarray
      Input image.
  scale : {float, tuple of floats}
      Scale factors for spatial dimensions. Separate scale factors can be defined as
      (m, n[, ...]).
  ```

  Plan:

  ```rest
  Parameters
  ----------
  image : N-D array, with optionally, one channel axis — e.g (I, J, C)
      Use the `channel_axis` parameter to specify the array axis corresponding to color channels.  Default is no channel axis.
  scale : {float, tuple of floats}
      Scale factors for spatial dimensions. Separate scale factors can be defined as
      (i, j[, ...]).
  ```

* `transform/_warps.py`: `_stackcopy`:

  ```rest
  Copy b into each color layer of a, such that::

    a[:,:,0] = a[:,:,1] = ... = b

  Parameters
  ----------
  a : (M, N) or (M, N, P) ndarray
      Target array.
  b : (M, N)
      Source array.

  Notes
  -----
  Color images are stored as an ``(M, N, 3)`` or ``(M, N, 4)`` arrays.
  ```

  Plan:

  ```rest
  Copy b into each color layer of a, such that::

    a[:,:,0] = a[:,:,1] = ... = b

  Parameters
  ----------
  a : (I, J) or (I, J, C) ndarray
      Target array.
  b : (I, J)
      Source array.

  Notes
  -----
  Color images are stored as an ``(I, J, 3)`` or ``(I, J, 4)`` arrays.
  ```

* `transform/_warps.py`: `warp_polar`

  ```rest
  Parameters
  ----------
  image : (M, N[, C]) ndarray
      Input image. For multichannel images `channel_axis` has to be specified.
  ```

  Plan:

  ```rest
  Parameters
  ----------
  image : (M, N[, C]) ndarray
      Input image. For multichannel images `channel_axis` has to be specified.
  image : 2- or 3-D array, with optionally, one channel axis — e.g (I, J, C)
      Use the `channel_axis` parameter to specify the array axis corresponding to color channels.  Default is no channel axis.
  ```

* `transform/_warps.py`: `swirl`

  ```rest
  Parameters
  ----------
  image : ndarray
      Input image.
  center : (column, row) tuple or (2,) ndarray, optional
      Center coordinate of transformation.
  strength : float, optional
      The amount of swirling applied.
  radius : float, optional
      The extent of the swirl in pixels.  The effect dies out
      rapidly beyond `radius`.
  rotation : float, optional
      Additional rotation applied to the image.
  ```

  Plan:

  ```rest
  Parameters
  ----------
  image : ndarray
      Input image.
  center : (row, column) tuple or (2,) ndarray, optional
      Center coordinate of transformation.
  strength : float, optional
      The amount of swirling applied.
  radius : float, optional
      The extent of the swirl in pixels.  The effect dies out
      rapidly beyond `radius`.
  rotation : float, optional
      Additional anti-clockwise rotation applied to the image, in radians.
  ```

  with associated implementation.  Skimage porting docstring:

  ```rest
  This routine behaves differently in Skimage1 and Skimage2. Skimage2 specifes
  the center for rotation as a coordinate in i, j (row, column) order, whereas
  Skimage1 asked for the center in (column, row) order.

  When porting Skimage1 code to Skimage 2, the following are equivalent:

  >>> import numpy as np
  >>> import skimage as ski1
  >>> import skimage2 as ski2
  >>>
  >>> # Skimage1
  >>> s_img_1 = ski1.transform.swirl(img, (10, 5))
  >>> # Skimage2.  Note reversal of center coordinate ordering (col, row)
  >>> s_img_2 = ski2.transform.swirl(img, (5, 10))
  >>> # The Skimage 1 and 2 code are equivalent.
  >>> assert np.all(s_img_1 == s_img_2)
  ```

* `transform/_warps.py`: `rotate`

  ```rest
  """Rotate image by a certain angle around its center.

  Parameters
  ----------
  image : ndarray
      Input image.
  angle : float
      Rotation angle in degrees in counter-clockwise direction.
  resize : bool, optional
      Determine whether the shape of the output image will be automatically
      calculated, so the complete rotated image exactly fits. Default is
      False.
  center : iterable of length 2
      The rotation center. If ``center=None``, the image is rotated around
      its center, i.e. ``center=(cols / 2 - 0.5, rows / 2 - 0.5)``.  Please
      note that this parameter is (cols, rows), contrary to normal skimage
      ordering.
  ```

  Plan:

  ```rest
  Parameters
  ----------
  image : ndarray
      Input image.
  angle : float
      Rotation angle in radians in counter-clockwise direction.
  resize : bool, optional
      Determine whether the shape of the output image will be automatically
      calculated, so the complete rotated image exactly fits. Default is
      False.
  center : iterable of length 2
      The rotation center. If ``center=None``, the image is rotated around its
      center, i.e. ``center=(rows / 2 - 0.5, cols / 2 - 0.5)``.  Please note
      that this parameter is (rows, cols) — the standard skimage ordering.
  ```

  (with associated logic change in implementation).

  Add Skimage1 porting docstring:

  ```rest
  There are two key differences between this routine (in Skimage1) and the equivalent routine in Skimage2.

  The first is that Skimage2 expects *radians* for the `angle` argume to rotate.  Skimage1 expected *degrees*.

  The second is that Skimage2 specifes the center for rotation as a coordinate
  in i, j (row, column) order, whereas Skimage1 asked for the center in
  (column, row) order.

  When porting Skimage1 code to Skimage 2, the following are equivalent:

  >>> import numpy as np
  >>> import skimage as ski1
  >>> import skimage2 as ski2
  >>>
  >>> # Skimage1
  >>> r_img_1 = ski1.transform.rotate(img, np.pi / 8, (10, 5))
  >>> # Skimage2.  Note angle in degrees, and reversal of center coordinate
  >>> # ordering (col, row)
  >>> r_img_2 = ski2.transform.rotate(img, 22.5, (5, 10))
  >>> # The Skimage 1 and 2 code are equivalent.
  >>> assert np.all(r_img_1 == r_img_2)
  ```

* `transform/_warps_cy.pyx`: — replace `(M, N)` with and `(I, J)`.

## Documentation

Edit:

> CONTRIBUTING.rst:* Refer to array dimensions as (plane), row, column, not as
x, y, z. See :ref:`Coordinate conventions
<numpy-images-coordinate-conventions>` in the user guide for more information.

And edit that document (`doc/source/user_guide/numpy_images.rst`) for plane
conventions etc.

And change `When documenting array parameters, use ``image : ndarray of shape
(M, N)`` and then refer to ``M`` and ``N`` in the docstring, if necessary.` to
`When documenting array parameters, use ``image : ndarray of shape (I, J)``
and then refer to ``I`` and ``J`` in the docstring, if necessary.
