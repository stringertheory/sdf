import numpy as np
import pytest

from sdf import ease
from sdf.ease import Easing


STANDARD_EASINGS = [
    ease.linear,
    ease.in_quad, ease.out_quad, ease.in_out_quad,
    ease.in_cubic, ease.out_cubic, ease.in_out_cubic,
    ease.in_quart, ease.out_quart, ease.in_out_quart,
    ease.in_quint, ease.out_quint, ease.in_out_quint,
    ease.in_sine, ease.out_sine, ease.in_out_sine,
    ease.in_circ, ease.out_circ, ease.in_out_circ,
]

MONOTONIC_EASINGS = STANDARD_EASINGS

ALL_EASINGS = STANDARD_EASINGS + [
    ease.in_expo, ease.out_expo, ease.in_out_expo,
    ease.in_elastic, ease.out_elastic, ease.in_out_elastic,
    ease.in_back, ease.out_back, ease.in_out_back,
    ease.in_bounce, ease.out_bounce, ease.in_out_bounce,
    ease.in_square, ease.out_square, ease.in_out_square,
]


class TestEasingEndpoints:
    @pytest.mark.parametrize("f", STANDARD_EASINGS, ids=lambda f: f.__name__)
    def test_f0_near_zero(self, f):
        t = np.array([0.0])
        assert abs(f(t).item()) < 1e-6, f"{f.__name__}(0) != 0"

    @pytest.mark.parametrize("f", STANDARD_EASINGS, ids=lambda f: f.__name__)
    def test_f1_near_one(self, f):
        t = np.array([1.0])
        assert abs(f(t).item() - 1.0) < 1e-6, f"{f.__name__}(1) != 1"


class TestEasingOutputShape:
    @pytest.mark.parametrize("f", ALL_EASINGS, ids=lambda f: f.__name__)
    def test_shape(self, f):
        t = np.linspace(0, 1, 100)
        result = f(t)
        assert result.shape == t.shape, f"{f.__name__} shape mismatch"


class TestMonotonicity:
    @pytest.mark.parametrize("f", MONOTONIC_EASINGS, ids=lambda f: f.__name__)
    def test_monotonic(self, f):
        t = np.linspace(0, 1, 200)
        y = f(t)
        diffs = np.diff(y)
        assert np.all(diffs >= -1e-10), f"{f.__name__} is not monotonically increasing"


class TestEasingIsInstance:
    def test_all_are_easing(self):
        for f in ALL_EASINGS:
            assert isinstance(f, Easing), f"{f} is not an Easing instance"


class TestEasingArithmetic:
    def test_add_easings(self):
        s = ease.linear + ease.in_quad
        t = np.array([0.5])
        expected = ease.linear(t) + ease.in_quad(t)
        np.testing.assert_allclose(s(t), expected)

    def test_mul_scalar(self):
        s = ease.linear * 2
        t = np.array([0.5])
        np.testing.assert_allclose(s(t), 1.0)

    def test_sub_easings(self):
        s = ease.linear - ease.in_quad
        t = np.array([0.5])
        expected = ease.linear(t) - ease.in_quad(t)
        np.testing.assert_allclose(s(t), expected)

    def test_div_scalar(self):
        s = ease.linear / 2
        t = np.array([0.5])
        np.testing.assert_allclose(s(t), 0.25)


class TestEasingComposition:
    def test_reverse(self):
        t = np.array([0.0, 0.5, 1.0])
        r = ease.in_quad.reverse
        np.testing.assert_allclose(r(t), ease.in_quad(1 - t))

    def test_symmetric(self):
        t = np.array([0.25, 0.75])
        s = ease.in_quad.symmetric
        # symmetric(0.25) == in_quad(0.5)
        np.testing.assert_allclose(s(np.array([0.25])), ease.in_quad(np.array([0.5])))

    def test_between(self):
        s = ease.linear.between(10, 20)
        t = np.array([0.0, 0.5, 1.0])
        np.testing.assert_allclose(s(t), [10, 15, 20])

    def test_transition_operator(self):
        s = ease.linear | ease.in_quad
        t = np.linspace(0, 1, 100)
        result = s(t)
        assert result.shape == t.shape

    def test_slice(self):
        s = ease.linear[0.25:0.75]
        t = np.array([0.0, 1.0])
        np.testing.assert_allclose(s(t), [0.25, 0.75])

    def test_clipped(self):
        s = ease.linear.clipped
        t = np.array([-0.5, 0.5, 1.5])
        result = s(t)
        np.testing.assert_allclose(result, [0.0, 0.5, 1.0])


class TestEasingConstants:
    def test_zero(self):
        t = np.linspace(0, 1, 10)
        np.testing.assert_allclose(ease.zero(t), 0.0)

    def test_one(self):
        t = np.linspace(0, 1, 10)
        np.testing.assert_allclose(ease.one(t), 1.0)

    def test_constant(self):
        c = ease.constant(0.5)
        t = np.linspace(0, 1, 10)
        np.testing.assert_allclose(c(t), 0.5)


class TestEasingProperties:
    def test_min(self):
        assert ease.linear.min == pytest.approx(0.0, abs=1e-3)

    def test_max(self):
        assert ease.linear.max == pytest.approx(1.0, abs=1e-3)

    def test_mean(self):
        assert ease.linear.mean == pytest.approx(0.5, abs=1e-3)

    def test_smoothstep(self):
        t = np.array([0.0, 0.5, 1.0])
        result = ease.smoothstep(t)
        np.testing.assert_allclose(result, [0.0, 0.5, 1.0])
