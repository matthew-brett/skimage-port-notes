# Measured on hessian-refactor (camera[::2,::2], mode=nearest)

## Multi-scale ridge filters, sigmas=(1,3,5), black_ridges=True

| filter | whole | border 4px | interior 20px | % pixels \|Δ\| > 1% of peak |
| --- | --- | --- | --- | --- |
| frangi | 12.8% | 12.8% | 12.7% | 3.3% |
| sato | 10.5% | 10.5% | 10.4% | 6.8% |
| meijering | 26.3% | 21.7% | 20.9% | 6.0% |
| hessian | 100%* | 100%* | 100%* | 1.2% |

\*Hybrid `hessian` clamp (`<=0 → 1`) inflates peak relative error.

## Single-scale Frangi vesselness

| sigma | whole | border | interior |
| --- | --- | --- | --- |
| 0.5 | 83.0% | 51.5% | 83.0% |
| 1.0 | 12.8% | 12.8% | 12.7% |
| 1.5 | 4.7% | 4.7% | 0.8% |
| 3.0 | 9.7% | 9.7% | 0.0% |
