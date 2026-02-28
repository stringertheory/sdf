import functools
import numpy as np
# import operator

from . import core, dn, d2, ease

# Constants

ORIGIN = np.array((0, 0, 0))

X = np.array((1, 0, 0))
Y = np.array((0, 1, 0))
Z = np.array((0, 0, 1))

UP = Z

# SDF Class

_ops = {}

class SDF3:
    def __init__(self, f):
        self.f = f
    def __call__(self, p):
        return self.f(p).reshape((-1, 1))
    def __getattr__(self, name):
        if name in _ops:
            f = _ops[name]
            return functools.partial(f, self)
        return getattr(self.f, name)
        raise AttributeError
    def __or__(self, other):
        return union(self, other)
    def __and__(self, other):
        return intersection(self, other)
    def __sub__(self, other):
        return difference(self, other)
    def k(self, k=None):
        result = SDF3(self.f)
        result._k = k
        return result
    def generate(self, *args, **kwargs):
        return core.generate(self, *args, **kwargs)
    def save(self, path, *args, **kwargs):
        return core.save(path, self, *args, **kwargs)
    def show_slice(self, *args, **kwargs):
        return core.show_slice(self, *args, **kwargs)
    def bounds(self):
        return core.bounds(self)
    def volume(self, *args, **kwargs):
        return core.volume(self, *args, **kwargs)
    def voxelize(self, *args, **kwargs):
        return core.voxelize(self, *args, **kwargs)
    def show(self, **kwargs):
        from . import viewer
        return viewer.show(self, **kwargs)

def sdf3(f):
    def wrapper(*args, **kwargs):
        return SDF3(f(*args, **kwargs))
    return wrapper

def op3(f):
    def wrapper(*args, **kwargs):
        return SDF3(f(*args, **kwargs))
    _ops[f.__name__] = wrapper
    return wrapper

def op32(f):
    def wrapper(*args, **kwargs):
        return d2.SDF2(f(*args, **kwargs))
    _ops[f.__name__] = wrapper
    return wrapper

# Helpers

def _length(a):
    return np.sqrt(np.einsum('ij,ij->i', a, a))

def _normalize(a):
    return a / np.linalg.norm(a)

def _dot(a, b):
    return np.sum(a * b, axis=1)

def _vec(*arrs):
    return np.stack(arrs, axis=-1)

def _perpendicular(v):
    if v[1] == 0 and v[2] == 0:
        if v[0] == 0:
            raise ValueError('zero vector')
        else:
            return np.cross(v, [0, 1, 0])
    return np.cross(v, [1, 0, 0])

_min = np.minimum
_max = np.maximum

# Primitives

@sdf3
def sphere(radius=1, center=ORIGIN):
    def f(p):
        return _length(p - center) - radius
    return f

@sdf3
def plane(normal=UP, point=ORIGIN):
    normal = _normalize(normal)
    def f(p):
        return np.dot(point - p, normal)
    return f

@sdf3
def slab(x0=None, y0=None, z0=None, x1=None, y1=None, z1=None, k=None):
    fs = []
    if x0 is not None:
        fs.append(plane(X, (x0, 0, 0)))
    if x1 is not None:
        fs.append(plane(-X, (x1, 0, 0)))
    if y0 is not None:
        fs.append(plane(Y, (0, y0, 0)))
    if y1 is not None:
        fs.append(plane(-Y, (0, y1, 0)))
    if z0 is not None:
        fs.append(plane(Z, (0, 0, z0)))
    if z1 is not None:
        fs.append(plane(-Z, (0, 0, z1)))
    return intersection(*fs, k=k)

@sdf3
def box(size=1, center=ORIGIN, a=None, b=None):
    if a is not None and b is not None:
        a = np.array(a)
        b = np.array(b)
        size = b - a
        center = a + size / 2
        return box(size, center)
    size = np.array(size)
    def f(p):
        q = np.abs(p - center) - size / 2
        return _length(_max(q, 0)) + _min(np.amax(q, axis=1), 0)
    return f

@sdf3
def rounded_box(size, radius):
    size = np.array(size)
    def f(p):
        q = np.abs(p) - size / 2 + radius
        return _length(_max(q, 0)) + _min(np.amax(q, axis=1), 0) - radius
    return f

@sdf3
def wireframe_box(size, thickness):
    size = np.array(size)
    def g(a, b, c):
        return _length(_max(_vec(a, b, c), 0)) + _min(_max(a, _max(b, c)), 0)
    def f(p):
        p = np.abs(p) - size / 2 - thickness / 2
        q = np.abs(p + thickness / 2) - thickness / 2
        px, py, pz = p[:,0], p[:,1], p[:,2]
        qx, qy, qz = q[:,0], q[:,1], q[:,2]
        return _min(_min(g(px, qy, qz), g(qx, py, qz)), g(qx, qy, pz))
    return f

@sdf3
def torus(r1, r2):
    def f(p):
        xy = p[:,[0,1]]
        z = p[:,2]
        a = _length(xy) - r1
        b = _length(_vec(a, z)) - r2
        return b
    return f

@sdf3
def capsule(a, b, radius):
    a = np.array(a)
    b = np.array(b)
    def f(p):
        pa = p - a
        ba = b - a
        h = np.clip(np.dot(pa, ba) / np.dot(ba, ba), 0, 1).reshape((-1, 1))
        return _length(pa - np.multiply(ba, h)) - radius
    return f

@sdf3
def cylinder(radius):
    def f(p):
        return _length(p[:,[0,1]]) - radius;
    return f

@sdf3
def capped_cylinder(a, b, radius):
    a = np.array(a)
    b = np.array(b)
    def f(p):
        ba = b - a
        pa = p - a
        baba = np.dot(ba, ba)
        paba = np.dot(pa, ba).reshape((-1, 1))
        x = _length(pa * baba - ba * paba) - radius * baba
        y = np.abs(paba - baba * 0.5) - baba * 0.5
        x = x.reshape((-1, 1))
        y = y.reshape((-1, 1))
        x2 = x * x
        y2 = y * y * baba
        d = np.where(
            _max(x, y) < 0,
            -_min(x2, y2),
            np.where(x > 0, x2, 0) + np.where(y > 0, y2, 0))
        return np.sign(d) * np.sqrt(np.abs(d)) / baba
    return f

@sdf3
def rounded_cylinder(ra, rb, h):
    def f(p):
        d = _vec(
            _length(p[:,[0,1]]) - ra + rb,
            np.abs(p[:,2]) - h / 2 + rb)
        return (
            _min(_max(d[:,0], d[:,1]), 0) +
            _length(_max(d, 0)) - rb)
    return f

@sdf3
def capped_cone(a, b, ra, rb):
    a = np.array(a)
    b = np.array(b)
    def f(p):
        rba = rb - ra
        baba = np.dot(b - a, b - a)
        papa = _dot(p - a, p - a)
        paba = np.dot(p - a, b - a) / baba
        x = np.sqrt(papa - paba * paba * baba)
        cax = _max(0, x - np.where(paba < 0.5, ra, rb))
        cay = np.abs(paba - 0.5) - 0.5
        k = rba * rba + baba
        f = np.clip((rba * (x - ra) + paba * baba) / k, 0, 1)
        cbx = x - ra - f * rba
        cby = paba - f
        s = np.where(np.logical_and(cbx < 0, cay < 0), -1, 1)
        return s * np.sqrt(_min(
            cax * cax + cay * cay * baba,
            cbx * cbx + cby * cby * baba))
    return f

@sdf3
def rounded_cone(r1, r2, h):
    def f(p):
        q = _vec(_length(p[:,[0,1]]), p[:,2])
        b = (r1 - r2) / h
        a = np.sqrt(1 - b * b)
        k = np.dot(q, _vec(-b, a))
        c1 = _length(q) - r1
        c2 = _length(q - _vec(0, h)) - r2
        c3 = np.dot(q, _vec(a, b)) - r1
        return np.where(k < 0, c1, np.where(k > a * h, c2, c3))
    return f

@sdf3
def ellipsoid(size):
    size = np.array(size)
    def f(p):
        k0 = _length(p / size)
        k1 = _length(p / (size * size))
        return k0 * (k0 - 1) / k1
    return f

@sdf3
def pyramid(h):
    def f(p):
        a = np.abs(p[:,[0,1]]) - 0.5
        w = a[:,1] > a[:,0]
        a[w] = a[:,[1,0]][w]
        px = a[:,0]
        py = p[:,2]
        pz = a[:,1]
        m2 = h * h + 0.25
        qx = pz
        qy = h * py - 0.5 * px
        qz = h * px + 0.5 * py
        s = _max(-qx, 0)
        t = np.clip((qy - 0.5 * pz) / (m2 + 0.25), 0, 1)
        a = m2 * (qx + s) ** 2 + qy * qy
        b = m2 * (qx + 0.5 * t) ** 2 + (qy - m2 * t) ** 2
        d2 = np.where(
            _min(qy, -qx * m2 - qy * 0.5) > 0,
            0, _min(a, b))
        return np.sqrt((d2 + qz * qz) / m2) * np.sign(_max(qz, -py))
    return f

# Platonic Solids

@sdf3
def tetrahedron(r):
    def f(p):
        x = p[:,0]
        y = p[:,1]
        z = p[:,2]
        return (_max(np.abs(x + y) - z, np.abs(x - y) + z) - r) / np.sqrt(3)
    return f

@sdf3
def octahedron(r):
    def f(p):
        return (np.sum(np.abs(p), axis=1) - r) * np.tan(np.radians(30))
    return f

@sdf3
def dodecahedron(r):
    x, y, z = _normalize(((1 + np.sqrt(5)) / 2, 1, 0))
    def f(p):
        p = np.abs(p / r)
        a = np.dot(p, (x, y, z))
        b = np.dot(p, (z, x, y))
        c = np.dot(p, (y, z, x))
        q = (_max(_max(a, b), c) - x) * r
        return q
    return f

@sdf3
def icosahedron(r):
    r *= 0.8506507174597755
    x, y, z = _normalize(((np.sqrt(5) + 3) / 2, 1, 0))
    w = np.sqrt(3) / 3
    def f(p):
        p = np.abs(p / r)
        a = np.dot(p, (x, y, z))
        b = np.dot(p, (z, x, y))
        c = np.dot(p, (y, z, x))
        d = np.dot(p, (w, w, w)) - x
        return _max(_max(_max(a, b), c) - x, d) * r
    return f

# TPMS Lattice Primitives (from worbit/sdf)

@sdf3
def gyroid(w=1):
    k = 2 * np.pi / w
    def f(p):
        x, y, z = p[:,0], p[:,1], p[:,2]
        return (np.sin(k * x) * np.cos(k * y) +
                np.sin(k * y) * np.cos(k * z) +
                np.sin(k * z) * np.cos(k * x))
    return f

@sdf3
def schwartz_p(w=1):
    k = 2 * np.pi / w
    def f(p):
        x, y, z = p[:,0], p[:,1], p[:,2]
        return np.cos(k * x) + np.cos(k * y) + np.cos(k * z)
    return f

@sdf3
def diamond(w=1):
    k = 2 * np.pi / w
    def f(p):
        x, y, z = p[:,0], p[:,1], p[:,2]
        return (np.sin(k * x) * np.sin(k * y) * np.sin(k * z) +
                np.sin(k * x) * np.cos(k * y) * np.cos(k * z) +
                np.cos(k * x) * np.sin(k * y) * np.cos(k * z) +
                np.cos(k * x) * np.cos(k * y) * np.sin(k * z))
    return f

# Positioning

@op3
def translate(other, offset):
    def f(p):
        return other(p - offset)
    return f

@op3
def scale(other, factor):
    try:
        x, y, z = factor
    except TypeError:
        x = y = z = factor
    s = (x, y, z)
    m = min(x, min(y, z))
    def f(p):
        return other(p / s) * m
    return f

@op3
def rotate(other, angle, vector=Z):
    x, y, z = _normalize(vector)
    s = np.sin(angle)
    c = np.cos(angle)
    m = 1 - c
    matrix = np.array([
        [m*x*x + c, m*x*y + z*s, m*z*x - y*s],
        [m*x*y - z*s, m*y*y + c, m*y*z + x*s],
        [m*z*x + y*s, m*y*z - x*s, m*z*z + c],
    ]).T
    def f(p):
        return other(np.dot(p, matrix))
    return f

@op3
def rotate_to(other, a, b):
    a = _normalize(np.array(a))
    b = _normalize(np.array(b))
    dot = np.dot(b, a)
    if dot == 1:
        return other
    if dot == -1:
        return rotate(other, np.pi, _perpendicular(a))
    angle = np.arccos(dot)
    v = _normalize(np.cross(b, a))
    return rotate(other, angle, v)

@op3
def orient(other, axis):
    return rotate_to(other, UP, axis)

@op3
def circular_array(other, count, offset=0):
    other = other.translate(X * offset)
    da = 2 * np.pi / count
    def f(p):
        x = p[:,0]
        y = p[:,1]
        z = p[:,2]
        d = np.hypot(x, y)
        a = np.arctan2(y, x) % da
        d1 = other(_vec(np.cos(a - da) * d, np.sin(a - da) * d, z))
        d2 = other(_vec(np.cos(a) * d, np.sin(a) * d, z))
        return _min(d1, d2)
    return f

# Alterations

@op3
def elongate(other, size):
    def f(p):
        q = np.abs(p) - size
        x = q[:,0].reshape((-1, 1))
        y = q[:,1].reshape((-1, 1))
        z = q[:,2].reshape((-1, 1))
        w = _min(_max(x, _max(y, z)), 0)
        return other(_max(q, 0)) + w
    return f

@op3
def twist(other, k):
    def f(p):
        x = p[:,0]
        y = p[:,1]
        z = p[:,2]
        c = np.cos(k * z)
        s = np.sin(k * z)
        x2 = c * x - s * y
        y2 = s * x + c * y
        z2 = z
        return other(_vec(x2, y2, z2))
    return f

@op3
def bend(other, k):
    def f(p):
        x = p[:,0]
        y = p[:,1]
        z = p[:,2]
        c = np.cos(k * x)
        s = np.sin(k * x)
        x2 = c * x - s * y
        y2 = s * x + c * y
        z2 = z
        return other(_vec(x2, y2, z2))
    return f

@op3
def bend_linear(other, p0, p1, v, e=ease.linear):
    p0 = np.array(p0)
    p1 = np.array(p1)
    v = -np.array(v)
    ab = p1 - p0
    def f(p):
        t = np.clip(np.dot(p - p0, ab) / np.dot(ab, ab), 0, 1)
        t = e(t).reshape((-1, 1))
        return other(p + t * v)
    return f

@op3
def bend_radial(other, r0, r1, dz, e=ease.linear):
    def f(p):
        x = p[:,0]
        y = p[:,1]
        z = p[:,2]
        r = np.hypot(x, y)
        t = np.clip((r - r0) / (r1 - r0), 0, 1)
        z = z - dz * e(t)
        return other(_vec(x, y, z))
    return f

@op3
def transition_linear(f0, f1, p0=-Z, p1=Z, e=ease.linear):
    p0 = np.array(p0)
    p1 = np.array(p1)
    ab = p1 - p0
    def f(p):
        d1 = f0(p)
        d2 = f1(p)
        t = np.clip(np.dot(p - p0, ab) / np.dot(ab, ab), 0, 1)
        t = e(t).reshape((-1, 1))
        return t * d2 + (1 - t) * d1
    return f

@op3
def transition_radial(f0, f1, r0=0, r1=1, e=ease.linear):
    def f(p):
        d1 = f0(p)
        d2 = f1(p)
        r = np.hypot(p[:,0], p[:,1])
        t = np.clip((r - r0) / (r1 - r0), 0, 1)
        t = e(t).reshape((-1, 1))
        return t * d2 + (1 - t) * d1
    return f

@op3
def wrap_around(other, x0, x1, r=None, e=ease.linear):
    p0 = X * x0
    p1 = X * x1
    v = -Y
    if r is None:
        r = np.linalg.norm(p1 - p0) / (2 * np.pi)
    def f(p):
        x = p[:,0]
        y = p[:,1]
        z = p[:,2]
        d = np.hypot(x, y) - r
        d = d.reshape((-1, 1))
        a = np.arctan2(y, x)
        t = (a + np.pi) / (2 * np.pi)
        t = e(t).reshape((-1, 1))
        q = p0 + (p1 - p0) * t + v * d
        q[:,2] = z
        return other(q)
    return f

# Chamfer, twist_between, pieslice (from nobodyinperson/sdf)

@op3
def chamfer(other1, other2, size):
    def f(p):
        d1 = other1(p).reshape(-1)
        d2 = other2(p).reshape(-1)
        m = _max(d1, d2)
        # Chamfer: bevel the intersection edge. The diagonal term
        # grows faster than max(d1,d2) far from the surface, breaking
        # the Lipschitz-1 property. Clamp it to preserve bounds estimation.
        chamfer_term = (d1 + d2) * np.sqrt(0.5) + size
        return _max(m, np.minimum(chamfer_term, m + size))
    return f

@op3
def twist_between(other, a, b, e=ease.linear):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    ab = b - a
    length = np.linalg.norm(ab)
    direction = ab / length
    def f(p):
        t = np.clip(np.dot(p - a, direction) / length, 0, 1)
        angle = e(t) * 2 * np.pi
        x = p[:,0]
        y = p[:,1]
        z = p[:,2]
        c = np.cos(angle)
        s = np.sin(angle)
        x2 = c * x - s * y
        y2 = s * x + c * y
        return other(_vec(x2, y2, z))
    return f

@sdf3
def pieslice(angle, centered=False):
    half = angle / 2
    def f(p):
        x = p[:,0]
        y = p[:,1]
        a = np.arctan2(y, x)
        if centered:
            a = np.abs(a)
            return np.where(a < half, -1.0, 1.0)
        else:
            return np.where(np.logical_and(a >= 0, a <= angle), -1.0, 1.0)
    return f

# Capsule chain and bezier (from nobodyinperson/sdf)

def capsule_chain(points, radius=None, diameter=None, k=0):
    if radius is None and diameter is not None:
        radius = diameter / 2
    if radius is None:
        radius = 1
    points = [np.array(pt, dtype=float) for pt in points]
    segments = []
    for i in range(len(points) - 1):
        segments.append(capsule(points[i], points[i+1], radius))
    if len(segments) == 1:
        return segments[0]
    result = segments[0]
    for seg in segments[1:]:
        if k:
            result = union(result, seg, k=k)
        else:
            result = union(result, seg)
    return result

def bezier(p1, p2, p3, p4, radius=None, diameter=None, steps=20, k=None):
    if radius is None and diameter is not None:
        radius = diameter / 2
    if radius is None:
        radius = 1
    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)
    p3 = np.array(p3, dtype=float)
    p4 = np.array(p4, dtype=float)
    ts = np.linspace(0, 1, steps + 1)
    pts = []
    for t in ts:
        u = 1 - t
        pt = u*u*u*p1 + 3*u*u*t*p2 + 3*u*t*t*p3 + t*t*t*p4
        pts.append(pt)
    if callable(radius):
        # Easing-based variable radius
        segments = []
        for i in range(len(pts) - 1):
            t_mid = (ts[i] + ts[i+1]) / 2
            r = radius(np.array([t_mid])).item()
            segments.append(capsule(pts[i], pts[i+1], r))
        result = segments[0]
        for seg in segments[1:]:
            if k:
                result = union(result, seg, k=k)
            else:
                result = union(result, seg)
        return result
    else:
        return capsule_chain(pts, radius=radius, k=k or 0)

# Thread and Screw (from nobodyinperson/sdf)

def Thread(pitch=5, diameter=20, offset=1, left=False):
    """Infinite helical thread.

    Creates a solid cylinder with a helical V-groove cut into its surface.

    Args:
        pitch: axial distance per full revolution of the thread.
        diameter: outer diameter of the thread.
        offset: depth of the thread groove.
        left: if True, left-handed thread.
    """
    r = diameter / 2
    @sdf3
    def _thread():
        def f(p):
            x, y, z = p[:, 0], p[:, 1], p[:, 2]
            # Distance from z-axis
            rho = np.sqrt(x * x + y * y)
            # Helical angle: maps (z, angle) into a single phase
            angle = np.arctan2(y, x)
            phase = z * (2 * np.pi / pitch) - angle
            if left:
                phase = z * (2 * np.pi / pitch) + angle
            # Triangle wave in [0, 1] for the thread profile
            # This modulates the effective radius
            t = (phase / (2 * np.pi)) % 1.0
            # V-shaped groove: radius varies between r and r-offset
            groove = offset * (1 - 2 * np.abs(t - 0.5))
            effective_r = r - groove
            return rho - effective_r
        return f
    return _thread()

def Screw(length=40, head_shape=None, head_height=10, k_tip=10, k_head=0, **threadkwargs):
    thread = Thread(**threadkwargs)
    body = thread & slab(z0=0, z1=length)
    if head_shape is not None:
        head = head_shape & slab(z0=-head_height, z1=0)
        if k_head:
            return union(body, head, k=k_head)
        else:
            return body | head
    return body

# Skin operation (from pschou/py-sdf)

@op3
def skin(other, depth):
    def f(p):
        d = other(p).reshape(-1)
        return _max(d - depth, 0)
    return f

# 3D => 2D Operations

@op32
def slice(other):
    # TODO: support specifying a slice plane
    # TODO: probably a better way to do this
    s = slab(z0=-1e-9, z1=1e-9)
    a = other & s
    b = other.negate() & s
    def f(p):
        p = _vec(p[:,0], p[:,1], np.zeros(len(p)))
        A = a(p).reshape(-1)
        B = -b(p).reshape(-1)
        w = A <= 0
        A[w] = B[w]
        return A
    return f

# Common

union = op3(dn.union)
difference = op3(dn.difference)
intersection = op3(dn.intersection)
blend = op3(dn.blend)
negate = op3(dn.negate)
dilate = op3(dn.dilate)
erode = op3(dn.erode)
shell = op3(dn.shell)
repeat = op3(dn.repeat)
addition = op3(dn.addition)
multiplication = op3(dn.multiplication)
division = op3(dn.division)
morph = op3(dn.morph)
shell_sided = op3(dn.shell_sided)
mirror = op3(dn.mirror)
mirror_copy = op3(dn.mirror_copy)
stretch = op3(dn.stretch)
shear = op3(dn.shear)
modulate_between = op3(dn.modulate_between)
