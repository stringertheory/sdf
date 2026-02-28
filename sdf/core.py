from functools import partial
from multiprocessing.pool import ThreadPool
from skimage import measure

import multiprocessing
import itertools
import numpy as np
import time

from . import progress, stl

WORKERS = multiprocessing.cpu_count()
SAMPLES = 2 ** 22
BATCH_SIZE = 32

def _marching_cubes(volume, level=0):
    verts, faces, _, _ = measure.marching_cubes(volume, level)
    return verts[faces].reshape((-1, 3))

def _cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)

def _skip(sdf, job):
    X, Y, Z = job
    x0, x1 = X[0], X[-1]
    y0, y1 = Y[0], Y[-1]
    z0, z1 = Z[0], Z[-1]
    x = (x0 + x1) / 2
    y = (y0 + y1) / 2
    z = (z0 + z1) / 2
    corners = np.array(list(itertools.product((x0, x1), (y0, y1), (z0, z1))))
    all_pts = np.vstack([[[x, y, z]], corners])
    values = sdf(all_pts).reshape(-1)
    r = abs(values[0])
    d = np.linalg.norm(np.array((x-x0, y-y0, z-z0)))
    if r <= d:
        return False
    corner_vals = values[1:]
    same = np.all(corner_vals > 0) if corner_vals[0] > 0 else np.all(corner_vals < 0)
    return same

def _worker(sdf, job, step, sparse):
    X, Y, Z = job
    if sparse and _skip(sdf, job):
        return None
        # return _debug_triangles(X, Y, Z)
    P = _cartesian_product(X, Y, Z)
    shape = (len(X), len(Y), len(Z))
    volume = sdf(P).reshape(shape)
    try:
        points = _marching_cubes(volume)
    except Exception:
        return []
        # return _debug_triangles(X, Y, Z)
    scale = np.array([X[1] - X[0], Y[1] - Y[0], Z[1] - Z[0]])
    offset = np.array([X[0], Y[0], Z[0]])
    return points * scale + offset

def _estimate_bounds(sdf):
    s = 16
    x0 = y0 = z0 = -1e9
    x1 = y1 = z1 = 1e9
    prev = None
    for i in range(32):
        X = np.linspace(x0, x1, s)
        Y = np.linspace(y0, y1, s)
        Z = np.linspace(z0, z1, s)
        d = np.array([X[1] - X[0], Y[1] - Y[0], Z[1] - Z[0]])
        threshold = np.linalg.norm(d) / 2
        if threshold == prev:
            break
        prev = threshold
        P = _cartesian_product(X, Y, Z)
        volume = sdf(P).reshape((len(X), len(Y), len(Z)))
        where = np.argwhere(np.abs(volume) <= threshold)
        if len(where) == 0:
            break
        x1, y1, z1 = (x0, y0, z0) + where.max(axis=0) * d + d / 2
        x0, y0, z0 = (x0, y0, z0) + where.min(axis=0) * d - d / 2
    return ((x0, y0, z0), (x1, y1, z1))

def generate(
        sdf,
        step=None, bounds=None, samples=SAMPLES,
        workers=WORKERS, batch_size=BATCH_SIZE,
        verbose=True, sparse=True):

    start = time.time()

    if bounds is None:
        bounds = _estimate_bounds(sdf)
    (x0, y0, z0), (x1, y1, z1) = bounds

    if step is None and samples is not None:
        volume = (x1 - x0) * (y1 - y0) * (z1 - z0)
        step = (volume / samples) ** (1 / 3)

    try:
        dx, dy, dz = step
    except TypeError:
        dx = dy = dz = step

    if verbose:
        print('min %g, %g, %g' % (x0, y0, z0))
        print('max %g, %g, %g' % (x1, y1, z1))
        print('step %g, %g, %g' % (dx, dy, dz))

    X = np.arange(x0, x1, dx)
    Y = np.arange(y0, y1, dy)
    Z = np.arange(z0, z1, dz)

    s = batch_size
    Xs = [X[i:i+s+1] for i in range(0, len(X), s)]
    Ys = [Y[i:i+s+1] for i in range(0, len(Y), s)]
    Zs = [Z[i:i+s+1] for i in range(0, len(Z), s)]

    batches = list(itertools.product(Xs, Ys, Zs))
    num_batches = len(batches)
    num_samples = sum(len(xs) * len(ys) * len(zs)
        for xs, ys, zs in batches)

    if verbose:
        print('%d samples in %d batches with %d workers' %
            (num_samples, num_batches, workers))

    points = []
    skipped = empty = nonempty = 0
    bar = progress.Bar(num_batches, enabled=verbose)
    pool = ThreadPool(workers)
    f = partial(_worker, sdf, step=(dx, dy, dz), sparse=sparse)
    for result in pool.imap(f, batches):
        bar.increment(1)
        if result is None:
            skipped += 1
        elif len(result) == 0:
            empty += 1
        else:
            nonempty += 1
            points.append(result)
    bar.done()

    if verbose:
        print('%d skipped, %d empty, %d nonempty' % (skipped, empty, nonempty))

    points = np.concatenate(points) if points else np.empty((0, 3))

    if verbose:
        triangles = len(points) // 3
        seconds = time.time() - start
        print('%d triangles in %g seconds' % (triangles, seconds))

    return points

def save(path, *args, **kwargs):
    points = generate(*args, **kwargs)
    ext = path.lower()
    if ext.endswith('.stl'):
        stl.write_binary_stl(path, points)
    elif ext.endswith('.step') or ext.endswith('.stp'):
        from . import step as step_module
        step_module.write_step(path, points)
    else:
        mesh = _mesh(points)
        mesh.write(path)

def _mesh(points):
    import meshio
    points, cells = np.unique(points, axis=0, return_inverse=True)
    cells = [('triangle', cells.reshape((-1, 3)))]
    return meshio.Mesh(points, cells)

def _debug_triangles(X, Y, Z):
    x0, x1 = X[0], X[-1]
    y0, y1 = Y[0], Y[-1]
    z0, z1 = Z[0], Z[-1]

    p = 0.25
    x0, x1 = x0 + (x1 - x0) * p, x1 - (x1 - x0) * p
    y0, y1 = y0 + (y1 - y0) * p, y1 - (y1 - y0) * p
    z0, z1 = z0 + (z1 - z0) * p, z1 - (z1 - z0) * p

    v = [
        (x0, y0, z0),
        (x0, y0, z1),
        (x0, y1, z0),
        (x0, y1, z1),
        (x1, y0, z0),
        (x1, y0, z1),
        (x1, y1, z0),
        (x1, y1, z1),
    ]

    return [
        v[3], v[5], v[7],
        v[5], v[3], v[1],
        v[0], v[6], v[4],
        v[6], v[0], v[2],
        v[0], v[5], v[1],
        v[5], v[0], v[4],
        v[5], v[6], v[7],
        v[6], v[5], v[4],
        v[6], v[3], v[7],
        v[3], v[6], v[2],
        v[0], v[3], v[2],
        v[3], v[0], v[1],
    ]

def sample_slice(
        sdf, w=1024, h=1024,
        x=None, y=None, z=None, bounds=None):

    if bounds is None:
        bounds = _estimate_bounds(sdf)
    (x0, y0, z0), (x1, y1, z1) = bounds

    if x is not None:
        X = np.array([x])
        Y = np.linspace(y0, y1, w)
        Z = np.linspace(z0, z1, h)
        extent = (Z[0], Z[-1], Y[0], Y[-1])
        axes = 'ZY'
    elif y is not None:
        Y = np.array([y])
        X = np.linspace(x0, x1, w)
        Z = np.linspace(z0, z1, h)
        extent = (Z[0], Z[-1], X[0], X[-1])
        axes = 'ZX'
    elif z is not None:
        Z = np.array([z])
        X = np.linspace(x0, x1, w)
        Y = np.linspace(y0, y1, h)
        extent = (Y[0], Y[-1], X[0], X[-1])
        axes = 'YX'
    else:
        raise Exception('x, y, or z position must be specified')

    P = _cartesian_product(X, Y, Z)
    return sdf(P).reshape((w, h)), extent, axes

def bounds(sdf):
    return _estimate_bounds(sdf)

def volume(sdf, bounds=None, samples=100000):
    if bounds is None:
        bounds = _estimate_bounds(sdf)
    (x0, y0, z0), (x1, y1, z1) = bounds
    bbox_vol = (x1 - x0) * (y1 - y0) * (z1 - z0)
    rng = np.random.default_rng(42)
    pts = np.column_stack([
        rng.uniform(x0, x1, samples),
        rng.uniform(y0, y1, samples),
        rng.uniform(z0, z1, samples),
    ])
    inside = sdf(pts).reshape(-1) < 0
    return bbox_vol * np.sum(inside) / samples

def voxelize(sdf, step=None, bounds=None, samples=SAMPLES):
    if bounds is None:
        bounds = _estimate_bounds(sdf)
    (x0, y0, z0), (x1, y1, z1) = bounds

    if step is None:
        vol = (x1 - x0) * (y1 - y0) * (z1 - z0)
        step = (vol / samples) ** (1 / 3)

    try:
        dx, dy, dz = step
    except TypeError:
        dx = dy = dz = step

    X = np.arange(x0, x1, dx)
    Y = np.arange(y0, y1, dy)
    Z = np.arange(z0, z1, dz)
    P = _cartesian_product(X, Y, Z)
    volume_arr = sdf(P).reshape((len(X), len(Y), len(Z)))
    spacing = np.array([dx, dy, dz])
    offset = np.array([x0, y0, z0])
    return volume_arr, spacing, offset

def show_slice(*args, **kwargs):
    import matplotlib.pyplot as plt
    show_abs = kwargs.pop('abs', False)
    a, extent, axes = sample_slice(*args, **kwargs)
    if show_abs:
        a = np.abs(a)
    im = plt.imshow(a, extent=extent, origin='lower')
    plt.xlabel(axes[0])
    plt.ylabel(axes[1])
    plt.colorbar(im)
    plt.show()
