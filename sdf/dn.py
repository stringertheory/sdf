import itertools
import numpy as np

from . import ease as _ease

_min = np.minimum
_max = np.maximum

def union(a, *bs, k=None):
    def f(p):
        d1 = a(p)
        for b in bs:
            d2 = b(p)
            K = k or getattr(b, '_k', None)
            if K is None:
                d1 = _min(d1, d2)
            else:
                h = np.clip(0.5 + 0.5 * (d2 - d1) / K, 0, 1)
                m = d2 + (d1 - d2) * h
                d1 = m - K * h * (1 - h)
        return d1
    return f

def difference(a, *bs, k=None):
    def f(p):
        d1 = a(p)
        for b in bs:
            d2 = b(p)
            K = k or getattr(b, '_k', None)
            if K is None:
                d1 = _max(d1, -d2)
            else:
                h = np.clip(0.5 - 0.5 * (d2 + d1) / K, 0, 1)
                m = d1 + (-d2 - d1) * h
                d1 = m + K * h * (1 - h)
        return d1
    return f

def intersection(a, *bs, k=None):
    def f(p):
        d1 = a(p)
        for b in bs:
            d2 = b(p)
            K = k or getattr(b, '_k', None)
            if K is None:
                d1 = _max(d1, d2)
            else:
                h = np.clip(0.5 - 0.5 * (d2 - d1) / K, 0, 1)
                m = d2 + (d1 - d2) * h
                d1 = m + K * h * (1 - h)
        return d1
    return f

def blend(a, *bs, k=0.5):
    def f(p):
        d1 = a(p)
        for b in bs:
            d2 = b(p)
            K = k or getattr(b, '_k', None)
            d1 = K * d2 + (1 - K) * d1
        return d1
    return f

def negate(other):
    def f(p):
        return -other(p)
    return f

def dilate(other, r):
    def f(p):
        return other(p) - r
    return f

def erode(other, r):
    def f(p):
        return other(p) + r
    return f

def shell(other, thickness):
    def f(p):
        return np.abs(other(p)) - thickness / 2
    return f

def repeat(other, spacing, count=None, padding=0):
    count = np.array(count) if count is not None else None
    spacing = np.array(spacing)

    def neighbors(dim, padding, spacing):
        try:
            padding = [padding[i] for i in range(dim)]
        except Exception:
            padding = [padding] * dim
        try:
            spacing = [spacing[i] for i in range(dim)]
        except Exception:
            spacing = [spacing] * dim
        for i, s in enumerate(spacing):
            if s == 0:
                padding[i] = 0
        axes = [list(range(-p, p + 1)) for p in padding]
        return list(itertools.product(*axes))

    def f(p):
        q = np.divide(p, spacing, out=np.zeros_like(p), where=spacing != 0)
        if count is None:
            index = np.round(q)
        else:
            index = np.clip(np.round(q), -count, count)

        nbrs = neighbors(p.shape[-1], padding, spacing)
        n_nbrs = len(nbrs)
        n_pts = len(p)
        big_batch = np.concatenate([p - spacing * (index + n) for n in nbrs])
        big_result = other(big_batch)
        return big_result.reshape(n_nbrs, n_pts).min(axis=0)
    return f


# Arithmetic operations (from worbit/sdf)

def addition(a, b):
    def f(p):
        return a(p) + b(p)
    return f

def multiplication(a, b):
    def f(p):
        return a(p) * b(p)
    return f

def division(a, b):
    def f(p):
        return a(p) / b(p)
    return f

def morph(a, b, c, d):
    """Interpolate between SDFs a and b based on control field c over distance d."""
    def f(p):
        da = a(p)
        db = b(p)
        dc = c(p)
        t = np.clip(dc / d, 0, 1)
        return da + (db - da) * t
    return f


# Improved shell (from worbit/sdf, nobodyinperson/sdf)

def shell_sided(other, thickness, side="center"):
    if side == "center":
        return shell(other, thickness)
    elif side == "inner":
        def f(p):
            d = other(p)
            return _max(d, -(d + thickness))
        return f
    elif side == "outer":
        def f(p):
            d = other(p)
            return _max(-d, d - thickness)
        return f
    else:
        raise ValueError(f"side must be 'center', 'inner', or 'outer', got {side!r}")


# Mirror operations (from nobodyinperson/sdf)

def mirror(other, direction, at=0):
    direction = np.array(direction, dtype=float)
    direction = direction / np.linalg.norm(direction)
    def f(p):
        dot = np.sum(p * direction, axis=1, keepdims=True)
        reflected = p - 2 * (dot - at) * direction
        return other(reflected)
    return f

def mirror_copy(other, direction, at=0):
    direction = np.array(direction, dtype=float)
    direction = direction / np.linalg.norm(direction)
    def f(p):
        dot = np.sum(p * direction, axis=1, keepdims=True)
        reflected = p - 2 * (dot - at) * direction
        d1 = other(p)
        d2 = other(reflected)
        return _min(d1, d2)
    return f


# Stretch, shear, modulate_between (from nobodyinperson/sdf)

def stretch(other, a, b, symmetric=False, e=_ease.linear):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    ab = b - a
    length = np.linalg.norm(ab)
    direction = ab / length
    def f(p):
        t = np.dot(p - a, direction) / length
        if symmetric:
            t = np.abs(t - 0.5) * 2
        t = np.clip(t, 0, 1)
        offset = e(t).reshape(-1, 1) * ab
        return other(p - offset)
    return f

def shear(other, fix, grab, move, e=_ease.linear):
    fix = np.array(fix, dtype=float)
    grab = np.array(grab, dtype=float)
    move = np.array(move, dtype=float)
    fg = grab - fix
    length = np.linalg.norm(fg)
    direction = fg / length
    def f(p):
        t = np.clip(np.dot(p - fix, direction) / length, 0, 1)
        offset = e(t).reshape(-1, 1) * move
        return other(p - offset)
    return f

def modulate_between(other, a, b, e=_ease.in_out_cubic):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    ab = b - a
    length = np.linalg.norm(ab)
    direction = ab / length
    def f(p):
        t = np.clip(np.dot(p - a, direction) / length, 0, 1)
        scale = e(t).reshape(-1, 1)
        return other(p) * scale
    return f
