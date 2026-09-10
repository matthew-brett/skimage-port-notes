""" Score Hough transform
"""

from importlib import reload
from pathlib import Path
import sys

import numpy as np
from skimage.transform._hough_transform import _probabilistic_hough_line as _prob_hough_line

sys.path.append(str(Path() / 'tests' / 'skimage' / 'transform'))

import test_hough_transform as tht
reload(tht)


def precision_recall(lines, shape, threshold):
    out_img = tht.write_lines(lines, shape)
    theta = np.linspace(-np.pi / 2, np.pi / 2, 180, endpoint=False)
    out_lines = _prob_hough_line(
        out_img, threshold=threshold, line_gap=2, line_length=50,
        theta=theta
    )
    detected_img = tht.write_lines(out_lines, shape)
    out_tf, detected_tf = [img.astype(bool) for img in (out_img, detected_img)]
    tp = np.sum(detected_tf & out_tf)
    fp = np.sum(detected_tf & ~out_tf)
    return tp / (tp + fp), tp / np.sum(out_img)


n_iters = 10000
results = np.zeros((n_iters, 2))
shape = (256, 256)
line_length = 100
threshold = line_length // 2
n_lines = 20
for i in range(n_iters):
    lines = tht._gen_lines(shape, n_lines, line_length, margins=1)
    results[i, :] = precision_recall(lines, shape, threshold)

precision, recall = np.mean(results, axis=0)
print('Average over iterations (precision, recall)')
print(precision, recall)
