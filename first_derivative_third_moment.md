# Proposal: impose the third moment on the first-derivative kernel

`_gaussian_derivative_kernel` in `_skimage2/feature/corner.py` restores the
discrete moments of the sampled Gaussian derivative. For `order=2` it imposes
two conditions; for `order=1` it imposes one. The missing condition is
measurable, it is wrong by 270% at `sigma = 0.3`, and it is the whole of the
error in the Hessian's mixed element at small `sigma`.

This note proposes adding it. It is a self-contained change, it is a no-op
above `sigma` of about 0.8, and it does **not** depend on or block anything in
`hessian_refactor_plan.md`.

Measurements: `skimage-workbooks` notebook `on_shape_index.md` §12.

## 1. The missing condition

`ndi.correlate1d` at the origin returns `sum(k * x ** n)` for `f = x ** n`.
Smoothing `x ** 3` at scale `sigma` gives `x ** 3 + 3 * sigma ** 2 * x`, whose
first derivative at the origin is `3 * sigma ** 2`. So the first-derivative
kernel owes three conditions:

| condition | status in `_gaussian_derivative_kernel` |
| --- | --- |
| `sum(k) == 0` | holds, by the oddness of the kernel |
| `sum(k * x) == 1` | imposed, by the normalisation already there |
| `sum(k * x ** 3) == 3 * sigma ** 2` | **not imposed** |

Measured on the shipped kernel:

| sigma | taps | `sum(k * x ** 3)` | want `3 sigma²` | relative error |
| --- | --- | --- | --- | --- |
| 0.3 | 5 | 1.000001 | 0.270000 | 270.4% |
| 0.5 | 9 | 1.029461 | 0.750000 | 37.3% |
| 0.8 | 13 | 1.922350 | 1.920000 | 0.1% |
| 1.0 | 17 | 3.000008 | 3.000000 | 0.0% |
| 2.0 | 33 | 12.000000 | 12.000000 | 0.0% |

Below `sigma` of about 0.5 the sampled kernel has collapsed towards the plain
central difference `[-1/2, 0, +1/2]`, whose third moment is 1 whatever `sigma`
is, while the continuous operator asks for `3 * sigma ** 2`.

This is the same class of fault the `order=2` branch already repairs, and the
same remedy applies.

## 2. The change

Two odd basis functions carry the two conditions:

```python
radius = int(truncate * sigma + 0.5)
x = np.arange(-radius, radius + 1, dtype=float)
smoothing = np.exp(-(x ** 2) / (2 * sigma ** 2))
smoothing /= smoothing.sum()
basis = (smoothing * (x / sigma ** 2), smoothing * (x ** 3 / sigma ** 4))
M = np.array([[(b * x).sum() for b in basis],
              [(b * x ** 3).sum() for b in basis]])
weights = np.linalg.solve(M, np.array([1.0, 3.0 * sigma ** 2]))
kernel = weights[0] * basis[0] + weights[1] * basis[1]
```

The present `order=1` branch is the `weights = (c, 0)` special case.

## 3. What it buys

On a Gaussian blob of width 6, against the closed-form Hessian:

| sigma | mixed element, shipped | mixed element, with the third moment | improvement |
| --- | --- | --- | --- |
| 0.3 | 5.4e-03 | 1.1e-04 | 49x |
| 0.5 | 2.1e-03 | 4.6e-05 | 46x |
| 0.8 | 1.8e-05 | 1.4e-06 | 13x |
| 1.0 | 6.7e-08 | 1.2e-08 | 5.6x |

The diagonal elements are untouched: they use the `order=2` and `order=0`
kernels, and only the mixed element uses `order=1` twice.

Downstream, the eigenvalue pair turns less, which is what any ratio of
eigenvalues sees. The median shape index error on the same blob improves by 2.8
to 6.0 times between `sigma` 0.3 and 1.0.

## 4. What it costs

**Nothing above `sigma` 0.8.** The largest change to any tap is 0.12 at
`sigma = 0.3`, 3e-4 at 0.8, and 0 at 1.0. Every result at the scales the
library is usually used at is bit-unchanged.

**Conditioning.** `cond(M)` is 1.1e+07 at `sigma = 0.3`, 1.1e+02 at 0.5, and
order 10 to 100 above that. Double precision absorbs 1e7, but the trend is the
wrong way and `sigma` below 0.3 was not tested. A guard, or a fallback to the
present kernel below some `sigma`, needs considering before this ships.

**A worse tail.** The median shape index error improves, but the 99th
percentile gets worse: 3.2e-01 to 4.5e-01 at `sigma = 0.3` on the blob. The
mean therefore improves by only 1.2 to 2 times where the median improves by 3
to 6. Where those pixels are, and why, is not established.

## 5. What it does not do

It does not make the Gaussian-derivative shape index beat the finite-difference
one at small `sigma`. On the blob the crossover moves from `sigma` 0.70 to
0.65 only. `on_shape_index.md` §6 located the Gaussian-derivative shortfall in
the mixed element, and that was right, but the mixed element is not the whole
of the gap to finite differences. Any argument for a `sigma`-dependent route
switch in `shape_index` has to stand on its own, and `on_shape_index.md` §10
argues against it.

## 6. Recommendation

Worth doing, on its own merits, as a separate change from the Hessian refactor:

* it is one more condition on a kernel that already carries two, in the same
  function, with the same justification;
* it is a no-op at every `sigma` the library is usually used at;
* it makes the mixed element as well corrected as the diagonal ones, which is
  the asymmetry that `on_shape_index.md` §6 found.

Not worth doing yet without first settling the conditioning guard (§4) and
understanding the tail (§4). Neither is hard; neither is done.

```{note}
Most of this material is AI-generated. See the repository `README.md` for what
that means for re-use.
```
