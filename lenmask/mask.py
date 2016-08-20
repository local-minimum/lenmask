#!/usr/bin/env python3

from scipy.misc import imread
from scipy.ndimage import gaussian_filter, binary_dilation, binary_erosion, label, \
     binary_closing, binary_propagation, binary_fill_holes, distance_transform_cdt, cemter_of_mass
from scipy.signal import convolve2d

import numpy as np


def load_grayscale_image(path):

    im = imread(path)
    if im.ndim == 3:
        im = im.mean(axis=-1)

    return im


def clear_image(im, sigma=101):

    return im - gaussian_filter(im, sigma=sigma)


def _get_derivatives(img):
    kernel = np.array([[-1, -2, -1],
                       [0, 0, 0],
                       [1, 2, 1]])
    ix = convolve2d(img, kernel.T, 'same')
    iy = convolve2d(img, kernel, 'same')

    return ix, iy


def _edges_and_corners(im, kappa=0.1, sigma=1):

    ix, iy = _get_derivatives(im)
    ix2 = ix ** 2
    iy2 = iy ** 2
    sx2 = gaussian_filter(ix2, sigma)
    sy2 = gaussian_filter(iy2, sigma)
    sxy = gaussian_filter(ix + iy, sigma)
    h = np.array([[sx2, sxy], [sxy, sy2]])
    r = np.linalg.det(h.T).T - kappa * np.trace(h) ** 2
    return np.sqrt(ix2 + iy2), r


def _threshold_im(im, c=0.3):

    return im > im.mean() + c * im.std()


def _simplify_binary(im, iterations=2, structure=np.ones((5, 5))):
    # TODO: invert and label and fill in those that are not too big
    t = binary_dilation(binary_erosion(im, structure, iterations=iterations), structure, iterations=iterations)
    return binary_fill_holes(t, structure)


def _label(im, minsize=30, max_worms=10):

    l, labels = label(im)
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


def labeled(im, init_smoothing=5, edge_smoothing=3, seg_c=0.8):

    sim = gaussian_filter(im, sigma=init_smoothing)
    # TODO: Separate edges
    e, _ = _edges_and_corners(sim, sigma=edge_smoothing)
    t1 = _threshold_im(sim, seg_c)
    t2 = _threshold_im(e)
    t = _simplify_binary(t1 | t2)
    return _label(t)


def _distance_worm(im, size=3):

    k = np.ones((size, size)) / size **2
    return convolve2d(distance_transform_cdt(im), k, "same")


def _seed_walker(dworm):

    m = dworm == dworm.max()
    lim, l = label(m)
    best_i = 1
    best = (lim == 1).sum()

    for i in range(best_i + 1, l + 1):

        val = (lim == i).sum()
        if val > best:
            best_i = i

    m = lim == best_i
    cx, cy = cemter_of_mass(m)
    px, py = np.where(m)
    d = (px - cx) ** 2 - (py - cy) ** 2
    mind = d.argmin()

    origin = np.array((px[mind], py[mind]))
    slope, _ = np.polyfit(px, py, 1)

    v = np.array(1, slope)
    v /= v.sum()
    return origin, v, -v


def analyse(path, background_smoothing=101):

    im = load_grayscale_image(path)
    im = clear_image(im, sigma=background_smoothing)
    worms = labeled(im)