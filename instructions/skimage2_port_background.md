# Port

This directory houses many Git working-tree checkouts of the Scikit-image project:

Github repo: https://github.com/scikit-image/scikit-image
Docs: https://scikit-image.org

When I give paths, I refer to the paths in a working tree of a Scikit-image checkout.   Thus the `src` path is a path in the checked out repository.

## The refactor

We are currently working on a transition from the standard (existing, scikit-image 1) API — call this SK1-API — to a new API — call this the scikit-image 2 API, SK2-API.

SK1-API is enshrined in the `src/skimage` directory, and tested in the
`tests/skimage1` tree.

We are developing the SK2-API in `src/_skimage2`, but that development is by
no means complete.  The tests are in `tests/skimage2`.  The `src/skimage2`
directory does nothing but provide a tree to import the `src/_skimage2` with
a warning that the resulting API is not stable.

We have adopted the policy of doing all substantial code development in the `src/_skimage2` tree, then importing this code to `src/skimage`, using stub import files, and adapting code in the stubs, to implement the SK1-API in terms of the newer SK2-API.

## Coordinate systems

A major problem in this port is that we want to change the default
interpretation of coordinates in SK2-API.  Specifically, we want to standardize to the Numpy / Array convention for image coordinates, where the first coordinate indexes into the first axis of the image array, and the second into the second axis of the image array.

See the description in the notebook document `coordinate_review.md` for more details of this array convention, and the standard current SK1-API convention, that we call the "imaging" coordinate system, where the first coordinate refers to the second axis of the image array, and the second refers to the first axis of the image array.

## The problem

The problem is to make a refactoring plan, that allows us to proceed in a gradual and stately fashion, modifying SK2-API such that `src/_skimage2` is still testable, but we progress towards to desired changes of coordinate system, set out in `coordinate_review.md`.
