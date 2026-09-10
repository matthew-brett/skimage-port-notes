# Port notes

Working notes on the move from the scikit-image 1 API to the scikit-image 2
API, with an emphasis on coordinate conventions.

The notebooks diagnose one question each, and every number in them comes from
a cell that computes it.

* {doc}`on_lines` — how `draw.line` and `draw.line_nd` rasterise a segment,
  why they disagree, and how they compare with Pillow and OpenCV.
* {doc}`on_hessian` — why `hessian_matrix` returns a different mixed partial
  depending on the `order` argument, and what it costs at the image border.
* {doc}`on_warp_maps` — what each kind of argument to `warp`'s `inverse_map`
  means, and where the kinds disagree with each other.
* {doc}`rotate_description` — the direction and centre conventions of
  `transform.rotate`.

The plans and reviews that accompany these notebooks — `coordinate_review.md`,
`coordinate_port_plan.md`, `hessian_refactor_plan.md`, `warp_refactor.md` — are
plain Markdown rather than notebooks, and are not part of the book yet.

```{note}
Most of this material is AI-generated. See the repository `README.md` for what
that means for re-use.
```
