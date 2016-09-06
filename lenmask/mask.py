#!/usr/bin/env python3

from scipy.misc import imread
from scipy.ndimage import gaussian_filter, binary_dilation, binary_erosion, label, \
     binary_closing, binary_propagation, binary_fill_holes, distance_transform_edt, center_of_mass
from scipy.signal import convolve2d
from matplotlib import pyplot as plt
import numpy as np
import csv
from argparse import ArgumentParser

"""
Debug code

from matplotlib import pyplot as plt
plt.ion()
f = plt.gcf()
ax = f.add_subplot(2, 1, 1)
detailed_ax = f.add_subplot(2, 1, 2)
f.show()

import lenmask.mask as m
im = m.load_grayscale_image("")
cim = m.clear_image(im)
lim = m.labeled(cim)
i = m.get_spine(lim == lim.max(), ax=ax, detailed_ax=detailed_ax, step_wise=True)

val = next(i)

"""

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
    return np.sqrt(ix2 + iy2)


def _threshold_im(im, c=0.3):

    return im > im.mean() + c * im.std()


def _simplify_binary(im, iterations=2, structure=np.ones((5, 5)), hole_structure=np.ones((7, 7)), hole_size=0.01):

    t = binary_dilation(binary_erosion(im, structure, iterations=iterations), structure, iterations=2 * iterations)
    t = binary_fill_holes(t, hole_structure)

    bg = ~t

    holes, n = label(bg)
    counts = np.bincount(holes.ravel())
    refs = counts.argsort()[-2:]
    id_holes, = np.where(counts < (hole_size * refs.min()))
    for hole in id_holes:
        t[holes == hole] = False

    t = binary_fill_holes(t, hole_structure)

    return binary_erosion(t, structure, iterations=iterations)


def _label(im, minsize=500, max_worms=10):

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


def threshold(im, init_smoothing=5, edge_smoothing=3, seg_c=0.8):
    sim = gaussian_filter(im, sigma=init_smoothing)
    e = _edges(sim, sigma=edge_smoothing)
    t1 = _threshold_im(sim, seg_c)
    t2 = _threshold_im(e)
    return _simplify_binary(t1 | t2)


def labeled(im, init_smoothing=5, edge_smoothing=3, seg_c=0.8):
    t = threshold(im, init_smoothing, edge_smoothing, seg_c)
    return _label(t)


def get_spine(binary_worm, ax=None, detailed_ax=None, step_wise=False):

    dist_worm = _distance_worm(binary_worm)
    if ax is not None:
        ax.imshow(dist_worm, interpolation='nearest')
        if step_wise:
            yield  dist_worm

    origin, a1, a2 = _seed_walker2(dist_worm)
    path = [origin]
    if ax is not None:
        ax.plot(origin[0], origin[1], 'x', ms=20, mew=3, color='k')
        v1 = _angle_to_v2(a1) * 5 + origin
        v2 = _angle_to_v2(a2) * 5 + origin
        ax.plot([origin[0], v1[0]], [origin[1], v1[1]], '-', lw=3, color='c')
        ax.plot([origin[0], v2[0]], [origin[1], v2[1]], '-', lw=3, color='b')
        if step_wise:
            yield origin, a1, a2

    if ax is not None:
        path_line = None

    for a in (a1, a2):

        # print(a)

        for _, cur_a, best_a, local_kernel, vals, angles in _walk2(dist_worm, path, a, step_wise=step_wise):
            if ax is not None:
                x_data, y_data = np.array(path).T
                if path_line is None:
                    path_line, = ax.plot(x_data, y_data, color='w', marker='.', lw=1.5)
                else:
                    path_line.set_data(x_data, y_data)

                if not detailed_ax:
                    yield path, cur_a

            if detailed_ax:
                detailed_ax.cla()

                detailed_ax.imshow(local_kernel, interpolation='nearest')
                center = np.array([int((v - 1) / 2) for v in local_kernel.shape])
                v = center + _angle_to_v2(cur_a) * center[0] * 1.2
                vals = vals / max(vals)
                for i, (val, ang) in enumerate(zip(vals, angles)):
                    if i % 10 != 0 and ang != best_a:
                        continue
                    x, y = _get_pixel_vector(center[0], ang)
                    detailed_ax.plot(x, y, lw=1 + 3 * val, color='c' if ang == best_a else 'k')
                detailed_ax.plot([center[0], v[0]], [center[1], v[1]], '--', color='c', lw=2)

                if step_wise:
                    yield path, local_kernel, cur_a

        path = path[::-1]

    yield np.array(path).T


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


def _get_pixel_vector(h, a):

    y = np.sin(a) * h
    x = np.cos(a) * h
    v = np.array((np.linspace(h, h + x, h, endpoint=False),
                  np.linspace(h, h + y, h, endpoint=False)))
    v = np.round(v).astype(int)
    delta = (np.diff(v.T, axis=0) ** 2).sum(axis=1) > 0
    return v.T[1:][delta].T


def _eval_local_dist_transform(local_im, steps=360):

    h = np.ceil(np.sqrt(local_im.size)).astype(int)
    if h % 2 == 0:
        raise ValueError("Only odd sized slices")
    h = (h - 1) / 2
    angles = []
    values = []
    for a in np.linspace(0, 2 * np.pi, steps, endpoint=False):
        x, y = _get_pixel_vector(h, a)
        angles.append(a)
        values.append(np.power(np.prod(local_im[y, x].astype(float)), 1.0 / x.size))
    return angles, values


def _angle_dist(a, b):

    d = np.abs(a - b) % (2 * np.pi)
    d[d > np.pi] = 2 * np.pi - d[d > np.pi]
    return d


def _scaled_angle_value(angles, values, id_a=None, a=None, angle_dist_weight=2, exponent=1.6):

    if id_a is not None:
        filt = np.arange(angles.size) != id_a
        d = _angle_dist(angles, angles[id_a])[filt]
    else:
        filt = angles != a
        d = _angle_dist(angles, a)[filt]
    angles = angles[filt]
    values = values[filt]
    d = (np.pi - d)
    w = angle_dist_weight * (1. / (1. + np.power(np.e, -np.power(d, exponent)))) - 1
    #w = np.power(np.power(d, 1.0 / exponent) * angle_dist_weight + d * (1 - angle_dist_weight), root2)
    return values * w, angles


def _angle_to_v2(a):

    return np.array((np.cos(a), np.sin(a)))


def _seed_walker2(distance_worm, kernel_half_size=9, closeness_weight=-1):

    y, x = np.where(distance_worm == distance_worm.max())
    y = y[0]
    x = x[0]
    xmin = np.max((0, x - kernel_half_size))
    ymin = np.max((0, y - kernel_half_size))

    k = distance_worm[
        ymin: ymin + 2 * kernel_half_size + 1,
        xmin: xmin + 2 * kernel_half_size + 1]

    angles, values = _eval_local_dist_transform(k)
    angles = np.array(angles)
    values = np.array(values)

    best, = np.where(values == values.max())

    if best.size > 2:

        # TODO: This is one dim too many in the selection somehow
        err = np.abs(np.pi - np.subtract.outer(angles[best], angles[best]))
        pos = np.array(np.array(np.where(err == err.min())))

        # TODO: What does this even mean
        # pos = np.array([v for v in pos if (np.diff(v) > 0).all()])

        if pos.shape[0] > 1:
            a1 = angles[best[pos[np.random.randint(0, pos.shape[0])][0]]]
        else:
            a1 = angles[best[pos[0]][0]]
        v1, angles1 = _scaled_angle_value(angles, values, a1, angle_dist_weight=closeness_weight)
        a2 = np.where(angles == angles1[v1.argmax()])[0][0]

    elif best.size == 2:

        a1, a2 = best
        v1, angles1 = _scaled_angle_value(angles, values, a1, angle_dist_weight=closeness_weight)
        a1best = angles1[v1.argmax()]
        if angles[a2] != a1best:
            v2, angles2 = _scaled_angle_value(angles, values, a2, angle_dist_weight=closeness_weight)
            if v2.max() > v1.max():
                a1 = np.where(angles == angles2[v2.argmax()])[0][0]
            else:
                a2 = np.where(angles == angles1[v1.argmax()])[0][0]

    else:
        a1 = best[0]
        v1, angles1 = _scaled_angle_value(angles, values, a1, angle_dist_weight=closeness_weight)
        a2 = np.where(angles == angles1[v1.argmax()])[0][0]

    a1 = angles[a1]
    a2 = angles[a2]

    return np.array((x, y)), a1, a2


def _get_local_kernel(im, pos, kernel_half_size):

    kernel_size = 2 * kernel_half_size + 1

    x, y = pos

    xmin = int(round(max(0, x - kernel_half_size)))
    ymin = int(round(max(0, y - kernel_half_size)))

    return im[ymin: ymin + kernel_size, xmin: xmin + kernel_size], xmin, ymin


def _adjusted_guess(im, pos, kernel_half_size):

    k, xmin, ymin = _get_local_kernel(im, pos, kernel_half_size)

    if k.any() == False:
        None

    newy, newx = (int(round(v)) for v in center_of_mass(k))

    newx += xmin
    newy += ymin
    return newx, newy


def _duplicated_pos(pos1, pos2, minstep):

    v = pos1 - pos2
    l2 = (v ** 2).sum()
    if l2 == 0:
        print("Zero step")
        return True
    l = np.sqrt(l2)
    if l < minstep:
        print("Small step")
        return True
    return False


def _walk2(im, path, a, step=7, minstep=2, kernel_half_size=11, momentum=5.0, max_depth=5000, step_wise=False):

    for _ in range(max_depth):
        old_pos = path[-1]
        # pos = _adjusted_guess(im, old_pos + _angle_to_v2(a) * step, kernel_half_size)
        pos = old_pos + _angle_to_v2(a) * step
        if pos is None:
            break

        pos = np.array(pos)
        if _duplicated_pos(pos, old_pos, minstep):
            break
        elif len(path) > 1 and _duplicated_pos(pos, path[-2], minstep):
            break
        elif im[tuple(np.round(pos).astype(int)[::-1])] == 0:
            break

        path.append(pos)

        k, _, _ = _get_local_kernel(im, pos, kernel_half_size)

        if k.any() == False:
            break
        elif (k.shape[0] % 2) == 0 or (k.shape[1] % 2) == 0 or k.shape[0] != k.shape[1]:
            break

        angles, values = _eval_local_dist_transform(k)
        angles = np.array(angles)
        values = np.array(values)
        values, angles = _scaled_angle_value(angles, values, a=a)
        best_a = angles[values.argmax()]
        if step_wise:
            yield path, a, best_a, k, values, angles

        a = (momentum * a + best_a) / (1 + momentum)
        a %= 2 * np.pi


def _walk(im, path, step=10, minstep=3, kernel_half_size=11, max_depth=1000):

    for _ in range(max_depth):
        kernel_size = 2 * kernel_half_size + 1

        x, y = path[-1]

        xmin = max(0, x - kernel_half_size)
        ymin = max(0, y - kernel_half_size)

        k = im[ymin: ymin + kernel_size, xmin: xmin + kernel_size]
        if k.any() == False:
            print("Outside worm")
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

    return path


def analyse(path, background_smoothing=51, save=True):

    im = load_grayscale_image(path)
    im = clear_image(im, sigma=background_smoothing)
    if save:
        plt.imsave(path + ".clear.png", im)
    worms = labeled(im)
    if save:
        plt.imsave(path + ".worms.png", worms)
    worms_data = {}

    if save:
        f = plt.figure()
        ax = f.gca()
        ax.imshow(im, cmap=plt.cm.Greys)
        fh = open(path + ".data.csv", 'w')
        csv_writer = csv.writer(fh)
        csv_writer.writerow(["Worm", "Length", "Area", "X", "Y"])

    for id_worm in range(1, worms.max() + 1):
        worm = worms == id_worm
        for worm_path in get_spine(worm):
            pass

        if worm_path.size <= 2:
            continue

        worm_len = np.sqrt(np.sum(np.diff(worm_path) ** 2, axis=0)).sum()
        worms_data[id_worm] = {'ridge': worm_path,
                               'length': worm_len,
                               'area': worm.sum()}

        if save:
            ax.plot(worm_path[0], worm_path[1], lw=3, color="r")
            csv_writer.writerow([id_worm, worm_len, worm.sum(), worm_path[0].tolist(), worm_path[1].tolist()])

    if save:
        f.savefig(path + ".paths.png")
        fh.close()

    return worms_data, im, worms


def outline(binary_worm, edge_width=1):

    e = _edges(binary_worm) > 0
    if edge_width > 1:
        e = binary_dilation(e, iterations=edge_width - 1)
    return e

if  __name__ == "__main__":

    parser = ArgumentParser(
        "LenMask",
        description="""This program automatically detects worms and other crooked rectangles.""",
        epilog="""Remember to have fun

        /Martin""")

    parser.add_argument(
        '-b', '--background-smoothing', dest='bg_smoothing', type=int, default=51,
        help="""Size of the gaussian sigma by which the background is smoothed to estimate
        long range trends in background color as an adaptive threshold for the image.
        This value should not be set low (whatever that is) because it will then distort the
        stuff of interest.
        """
    )

    parser.add_argument(
        dest='filepath',
        help="""Path to file to analyse, supported filetypes will depend on your python installation,
        so if you get an error, try installing the newest version of the python package `pillow`
        (or its predecessor `PIL`).
        """
    )

    args = parser.parse_args()

    analyse(args.filepath, background_smoothing=args.bg_smoothing)
