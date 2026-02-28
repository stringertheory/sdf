import numpy as np


class Easing:
    """A callable easing function with composition and arithmetic support."""

    def __init__(self, f, name=None):
        self.f = f
        self.name = name or getattr(f, '__name__', 'easing')
        self.__name__ = self.name

    def __call__(self, t):
        return self.f(t)

    def __repr__(self):
        return f'Easing({self.name!r})'

    @staticmethod
    def function(f):
        return Easing(f, name=f.__name__)

    # Arithmetic operators — combine easing outputs

    def __add__(self, other):
        if isinstance(other, Easing):
            return Easing(lambda t, a=self, b=other: a(t) + b(t),
                          f'{self.name}+{other.name}')
        return Easing(lambda t, a=self, v=other: a(t) + v,
                      f'{self.name}+{other}')

    def __radd__(self, other):
        return Easing(lambda t, a=self, v=other: v + a(t),
                      f'{other}+{self.name}')

    def __sub__(self, other):
        if isinstance(other, Easing):
            return Easing(lambda t, a=self, b=other: a(t) - b(t),
                          f'{self.name}-{other.name}')
        return Easing(lambda t, a=self, v=other: a(t) - v,
                      f'{self.name}-{other}')

    def __rsub__(self, other):
        return Easing(lambda t, a=self, v=other: v - a(t),
                      f'{other}-{self.name}')

    def __mul__(self, other):
        if isinstance(other, Easing):
            return Easing(lambda t, a=self, b=other: a(t) * b(t),
                          f'{self.name}*{other.name}')
        return Easing(lambda t, a=self, v=other: a(t) * v,
                      f'{self.name}*{other}')

    def __rmul__(self, other):
        return Easing(lambda t, a=self, v=other: v * a(t),
                      f'{other}*{self.name}')

    def __truediv__(self, other):
        if isinstance(other, Easing):
            return Easing(lambda t, a=self, b=other: a(t) / b(t),
                          f'{self.name}/{other.name}')
        return Easing(lambda t, a=self, v=other: a(t) / v,
                      f'{self.name}/{other}')

    def __or__(self, other):
        """Transition: self for t<0.5, other for t>=0.5."""
        def f(t, a=self, b=other):
            ta = np.clip(t * 2, 0, 1)
            tb = np.clip(t * 2 - 1, 0, 1)
            return np.where(t < 0.5, a(ta), b(tb))
        return Easing(f, f'{self.name}|{other.name}')

    def __rshift__(self, amount):
        """Shift the easing by amount in the t domain."""
        def f(t, a=self, s=amount):
            return a(t - s)
        return Easing(f, f'{self.name}>>{amount}')

    def __getitem__(self, key):
        """Chain easings (list/tuple) or zoom into a slice."""
        if isinstance(key, slice):
            lo = key.start if key.start is not None else 0.0
            hi = key.stop if key.stop is not None else 1.0
            def f(t, a=self, lo=lo, hi=hi):
                return a(lo + t * (hi - lo))
            return Easing(f, f'{self.name}[{lo}:{hi}]')
        raise TypeError(f'Easing indices must be slices, not {type(key).__name__}')

    # Properties

    @property
    def reverse(self):
        def f(t, a=self):
            return a(1 - t)
        return Easing(f, f'{self.name}.reverse')

    @property
    def symmetric(self):
        def f(t, a=self):
            return np.where(t < 0.5, a(2 * t), a(2 * (1 - t)))
        return Easing(f, f'{self.name}.symmetric')

    @property
    def clipped(self):
        def f(t, a=self):
            return a(np.clip(t, 0, 1))
        return Easing(f, f'{self.name}.clipped')

    # Methods

    def between(self, lo, hi):
        """Scale output from [0,1] to [lo, hi]."""
        def f(t, a=self, lo=lo, hi=hi):
            return lo + a(t) * (hi - lo)
        return Easing(f, f'{self.name}.between({lo},{hi})')

    def append(self, other):
        """Run self for t<0.5, other for t>=0.5, both scaled to full range."""
        return self | other

    def prepend(self, other):
        """Run other for t<0.5, self for t>=0.5."""
        return other | self

    def clip(self, lo=0, hi=1):
        """Clip output values."""
        def f(t, a=self, lo=lo, hi=hi):
            return np.clip(a(t), lo, hi)
        return Easing(f, f'{self.name}.clip({lo},{hi})')

    def plot(self, n=200):
        import matplotlib.pyplot as plt
        t = np.linspace(0, 1, n)
        y = self(t)
        plt.plot(t, y, label=self.name)
        plt.legend()
        plt.title(self.name)
        plt.xlabel('t')
        plt.ylabel('f(t)')
        plt.show()

    @property
    def min(self):
        t = np.linspace(0, 1, 1000)
        return float(np.min(self(t)))

    @property
    def max(self):
        t = np.linspace(0, 1, 1000)
        return float(np.max(self(t)))

    @property
    def mean(self):
        t = np.linspace(0, 1, 1000)
        return float(np.mean(self(t)))


# ---------------------------------------------------------------------------
# Standard easing functions, all as Easing instances
# ---------------------------------------------------------------------------

@Easing.function
def linear(t):
    return t

@Easing.function
def smoothstep(t):
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)

@Easing.function
def in_quad(t):
    return t * t

@Easing.function
def out_quad(t):
    return -t * (t - 2)

@Easing.function
def in_out_quad(t):
    u = 2 * t - 1
    a = 2 * t * t
    b = -0.5 * (u * (u - 2) - 1)
    return np.where(t < 0.5, a, b)

@Easing.function
def in_cubic(t):
    return t * t * t

@Easing.function
def out_cubic(t):
    u = t - 1
    return u * u * u + 1

@Easing.function
def in_out_cubic(t):
    u = t * 2
    v = u - 2
    a = 0.5 * u * u * u
    b = 0.5 * (v * v * v + 2)
    return np.where(u < 1, a, b)

@Easing.function
def in_quart(t):
    return t * t * t * t

@Easing.function
def out_quart(t):
    u = t - 1
    return -(u * u * u * u - 1)

@Easing.function
def in_out_quart(t):
    u = t * 2
    v = u - 2
    a = 0.5 * u * u * u * u
    b = -0.5 * (v * v * v * v - 2)
    return np.where(u < 1, a, b)

@Easing.function
def in_quint(t):
    return t * t * t * t * t

@Easing.function
def out_quint(t):
    u = t - 1
    return u * u * u * u * u + 1

@Easing.function
def in_out_quint(t):
    u = t * 2
    v = u - 2
    a = 0.5 * u * u * u * u * u
    b = 0.5 * (v * v * v * v * v + 2)
    return np.where(u < 1, a, b)

@Easing.function
def in_sine(t):
    return -np.cos(t * np.pi / 2) + 1

@Easing.function
def out_sine(t):
    return np.sin(t * np.pi / 2)

@Easing.function
def in_out_sine(t):
    return -0.5 * (np.cos(np.pi * t) - 1)

@Easing.function
def in_expo(t):
    a = np.zeros(len(t))
    b = 2 ** (10 * (t - 1))
    return np.where(t == 0, a, b)

@Easing.function
def out_expo(t):
    a = np.zeros(len(t)) + 1
    b = 1 - 2 ** (-10 * t)
    return np.where(t == 1, a, b)

@Easing.function
def in_out_expo(t):
    zero = np.zeros(len(t))
    one = zero + 1
    a = 0.5 * 2 ** (20 * t - 10)
    b = 1 - 0.5 * 2 ** (-20 * t + 10)
    return np.where(t == 0, zero, np.where(t == 1, one, np.where(t < 0.5, a, b)))

@Easing.function
def in_circ(t):
    return -1 * (np.sqrt(np.clip(1 - t * t, 0, None)) - 1)

@Easing.function
def out_circ(t):
    u = t - 1
    return np.sqrt(np.clip(1 - u * u, 0, None))

@Easing.function
def in_out_circ(t):
    u = t * 2
    v = u - 2
    a = -0.5 * (np.sqrt(np.clip(1 - u * u, 0, None)) - 1)
    b = 0.5 * (np.sqrt(np.clip(1 - v * v, 0, None)) + 1)
    return np.where(u < 1, a, b)

@Easing.function
def in_elastic(t, k=0.5):
    u = t - 1
    return -1 * (2 ** (10 * u) * np.sin((u - k / 4) * (2 * np.pi) / k))

@Easing.function
def out_elastic(t, k=0.5):
    return 2 ** (-10 * t) * np.sin((t - k / 4) * (2 * np.pi / k)) + 1

@Easing.function
def in_out_elastic(t, k=0.5):
    u = t * 2
    v = u - 1
    a = -0.5 * (2 ** (10 * v) * np.sin((v - k / 4) * 2 * np.pi / k))
    b = 2 ** (-10 * v) * np.sin((v - k / 4) * 2 * np.pi / k) * 0.5 + 1
    return np.where(u < 1, a, b)

@Easing.function
def in_back(t):
    k = 1.70158
    return t * t * ((k + 1) * t - k)

@Easing.function
def out_back(t):
    k = 1.70158
    u = t - 1
    return u * u * ((k + 1) * u + k) + 1

@Easing.function
def in_out_back(t):
    k = 1.70158 * 1.525
    u = t * 2
    v = u - 2
    a = 0.5 * (u * u * ((k + 1) * u - k))
    b = 0.5 * (v * v * ((k + 1) * v + k) + 2)
    return np.where(u < 1, a, b)

@Easing.function
def in_bounce(t):
    return 1 - out_bounce(1 - t)

@Easing.function
def out_bounce(t):
    a = (121 * t * t) / 16
    b = (363 / 40 * t * t) - (99 / 10 * t) + 17 / 5
    c = (4356 / 361 * t * t) - (35442 / 1805 * t) + 16061 / 1805
    d = (54 / 5 * t * t) - (513 / 25 * t) + 268 / 25
    return np.where(
        t < 4 / 11, a, np.where(
        t < 8 / 11, b, np.where(
        t < 9 / 10, c, d)))

@Easing.function
def in_out_bounce(t):
    a = in_bounce(2 * t) * 0.5
    b = out_bounce(2 * t - 1) * 0.5 + 0.5
    return np.where(t < 0.5, a, b)

@Easing.function
def in_square(t):
    a = np.zeros(len(t))
    b = a + 1
    return np.where(t < 1, a, b)

@Easing.function
def out_square(t):
    a = np.zeros(len(t))
    b = a + 1
    return np.where(t > 0, b, a)

@Easing.function
def in_out_square(t):
    a = np.zeros(len(t))
    b = a + 1
    return np.where(t < 0.5, a, b)


# Convenience constants
def constant(x):
    """Create an easing that always returns x."""
    return Easing(lambda t, x=x: np.full_like(t, x, dtype=float), f'constant({x})')

zero = constant(0)
one = constant(1)


def _main():
    import matplotlib.pyplot as plt
    fs = [
        linear,
        in_quad, out_quad, in_out_quad,
        in_cubic, out_cubic, in_out_cubic,
        in_quart, out_quart, in_out_quart,
        in_quint, out_quint, in_out_quint,
        in_sine, out_sine, in_out_sine,
        in_expo, out_expo, in_out_expo,
        in_circ, out_circ, in_out_circ,
        in_elastic, out_elastic, in_out_elastic,
        in_back, out_back, in_out_back,
        in_bounce, out_bounce, in_out_bounce,
        in_square, out_square, in_out_square,
    ]
    x = np.linspace(0, 1, 1000)
    for f in fs:
        y = f(x)
        plt.plot(x, y, label=f.__name__)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    _main()
