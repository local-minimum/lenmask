#!/usr/bin/env python3

from scipy.misc import imread
from scipy.ndimage import gaussian_filter, binary_dilation, binary_erosion, label, \
     binary_closing, binary_propagation, binary_fill_holes, distance_transform_edt, center_of_mass
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


def _edges(im, sigma=1):

    ix, iy = _get_derivatives(im)
    ix2 = ix ** 2
    iy2 = iy ** 2
    sx2 = gaussian_filter(ix2, sigma)
    sy2 = gaussian_filter(iy2, sigma)
    return np.sqrt(ix2 + iy2)


def _threshold_im(im, c=0.3):

    return im > im.mean() + c * im.std()


def _simplify_binary(im, iterations=2, structure=np.ones((5, 5))):
    # TODO: invert and label and fill in those that are not too big
    t = binary_dilation(binary_erosion(im, structure, iterations=iterations), structure, iterations=iterations)
    t = binary_fill_holes(t, structure)
    return t


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
    e = _edges(sim, sigma=edge_smoothing)
    t1 = _threshold_im(sim, seg_c)
    t2 = _threshold_im(e)
    t = _simplify_binary(t1 | t2)
    return _label(t)


def get_spine(binary_worm):

    dist_worm = _distance_worm(binary_worm)
    origin, v1, v2 = _seed_walker(dist_worm)
    step = 10
    path = [origin, np.round(origin + step * v1).astype(origin.dtype)]
    path = _walk(dist_worm, path)
    return np.array(_walk(dist_worm, path[::-1])).T


def _distance_worm(im, size=3):

    k = np.ones((size, size)) / size **2
    return convolve2d(distance_transform_edt(im), k, "same")


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
    cy, cx = center_of_mass(m)
    py, px = np.where(m)
    d = (px - cx) ** 2 - (py - cy) ** 2
    mind = d.argmin()

    origin = np.array((px[mind], py[mind]))
    slope, _ = np.polyfit(px, py, 1)

    v = np.array((1, slope))
    v /= np.sqrt((v ** 2).sum())
    return origin, v, -v


def _walk(im, path, step=10, minstep=3, kernel_half_size=11):

    kernel_size = 2 * kernel_half_size + 1

    x, y = path[-1]

    xmin = max(0, x - kernel_half_size)
    ymin = max(0, y - kernel_half_size)

    k = im[ymin: ymin + kernel_size, xmin: xmin + kernel_size]
    if k.any() == False:
        print("Outside worm")
        print(k)
        return path[:-1]

    newy, newx = (int(round(v)) for v in center_of_mass(k))

    newx += xmin
    newy += ymin

    pos = np.array((newx, newy))
    old_pos = path[-2]
    v = pos - old_pos
    l2 = (v ** 2).sum()
    if l2 == 0:
        print("Zero step")
        return path[:-1]

    l = np.sqrt(l2)
    if l < minstep:
        print("Small step")
        return path

    path[-1] = pos

    path.append(np.round(pos + v / l * step).astype(path[-1].dtype))

    return _walk(im, path, step, minstep, kernel_half_size)


def analyse(path, background_smoothing=101):

    im = load_grayscale_image(path)
    im = clear_image(im, sigma=background_smoothing)
    worms = labeled(im)
    worms_data = {}
    for id_worm in range(1, worms.max() + 1):
        worm = worms == worms.max()
        worm_path = get_spine(worm)
        worm_len = np.sqrt(np.sum(np.diff(worm_path) ** 2, axis=0)).sum()
        worms_data[id_worm] = {'ridge': worm_path,
                               'length': worm_len,
                               'area': worm.sum()}

    return worms_data, im, worms


def outline(binary_worm, edge_width=1):

    e = _edges(binary_worm) > 0
    if edge_width > 1:
        e = binary_dilation(e, iterations=edge_width - 1)
    return e