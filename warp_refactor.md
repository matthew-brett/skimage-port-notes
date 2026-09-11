# Warp refactor

This is an issue to plan the warp refactor.

The text is all me, no AI.  I did use AI to help analyze the problem.

## Proposal

In what follows, the image is N-D - thus N=2 for a 2-D image.

Warp has two positional (non-default) input arguments, currently named `image` and `inverse_map`.

This discussion is about the acceptable inputs for `inverse_map`. My proposal
is that `skimage2.tranform.warp` will accept only:

* scikit-image Transform instances OR
* An N+1 by N+1 homogenous transform matrix OR
* arrays of coordinates, as specified in the docstring.
