# Port notes

Working notes on the move from the scikit-image 1 API to the scikit-image 2
API, with an emphasis on coordinate conventions.

## Refactoring plans

* {doc}`coordinate_review` — what coordinate convention we want in SK2-API.
* {doc}`coordinate_port_plan` — staged plan to move `_skimage2` to array order.
* {doc}`hessian_refactor_plan` — correctness fix for `hessian_matrix` at the
  image border, and removal of `order` from `structure_tensor`.
* {doc}`frangi_refactor_plan` — six defects in `frangi`, which of them depend
  on the `hessian_matrix` fix, and the order to repair them in.
* {doc}`warp_refactor` — what `warp` should accept as `inverse_map`.
* {doc}`first_derivative_third_moment` — a missing moment condition in
  `_gaussian_derivative_kernel`, worth fixing independently of the Hessian
  refactor.

```{note}
Most of this material is AI-generated. See the repository `README.md` for what
that means for re-use.
```
