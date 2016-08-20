#!/usr/bin/env python3

from scipy.misc import imread
from scipy.interpolate import Rbf
from scipy.ndimage import gaussian_filter, binary_dilation, binary_erosion, label, \
    binary_closing, binary_propagation, binary_fill_holes
import numpy as np


def _load_grayscale(path):

    im = imread(path)
    if im.ndim == 3:
        im = im.mean(axis=-1)

    return im


def _dynamic_threshold(im, part=5):

    d1old = 0
    d2old = 0
    t = np.zeros_like(im) -1
    for d1 in np.linspace(0, im.shape[0], part + 1):

        if d1 == 0:
            d1old = d1
            continue

        for d2 in np.linspace(0, im.shape[1], part + 1):

            if d2 != 0:

                t[round((d1 + d1old) / 2), round((d2 + d2old) / 2)] = im[d1old: d1, d2old : d2].mean()

    y, x = np.where(t > -1)
    YI, XI = np.meshgrid(*(np.arange(v) for v in im.shape))
    rbf = Rbf(y, x, t[y, x])
    return rbf(YI, XI).T


def _dynamic_gauss_background_remove(im, sigma=101):

    return im - gaussian_filter(im, sigma=sigma)


def _label(im, c=0.3, iterations=2, minsize=30, structure=np.ones((5, 5)), max_worms=10):

    t = im > im.mean() + c * im.std()
    t = binary_dilation(binary_erosion(t, structure, iterations=iterations), structure, iterations=iterations)
    t = binary_fill_holes(t, structure)

    l, labels = label(t)
    c = np.zeros((labels + 1,))

    for i in range(labels + 1):

        if i == 0:
            continue

        c[i] = (l == i).sum()

        if c[i] < minsize:
            l[l == i] = 0
            c[i] = 0

    allowed = np.argsort(c)[-max_worms:]

    def filt(v):
        if v in allowed:
            return np.where(v == allowed)[0][0] + 1
        return 0

    return np.frompyfunc(filt, 1, 1)(l).astype(int)