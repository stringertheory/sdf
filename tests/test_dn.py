import numpy as np
import pytest

import sdf
from .conftest import make_grid_3d, assert_sdf_callable


GRID = make_grid_3d(10)


class TestUnion:
    def test_min_semantics(self):
        a = sdf.sphere(1)
        b = sdf.sphere(1, center=(1, 0, 0))
        u = a | b
        pts = np.array([[0.0, 0, 0], [1.0, 0, 0], [0.5, 0, 0]])
        result_u = u(pts).reshape(-1)
        result_a = a(pts).reshape(-1)
        result_b = b(pts).reshape(-1)
        expected = np.minimum(result_a, result_b)
        np.testing.assert_allclose(result_u, expected, atol=1e-10)

    def test_output_shape(self):
        a = sdf.sphere(1)
        b = sdf.box(1)
        assert_sdf_callable(a | b, GRID)

    def test_smooth(self):
        a = sdf.sphere(1)
        b = sdf.sphere(1, center=(1.5, 0, 0))
        u_crisp = a | b
        u_smooth = sdf.d3.union(a, b, k=0.5)
        pts = np.array([[0.75, 0, 0]])
        # Smooth union should have smaller (more negative) values
        assert u_smooth(pts).item() <= u_crisp(pts).item() + 1e-10


class TestDifference:
    def test_max_neg_semantics(self):
        a = sdf.sphere(1)
        b = sdf.sphere(0.5)
        d = a - b
        pts = np.array([[0.0, 0, 0], [0.8, 0, 0]])
        result_d = d(pts).reshape(-1)
        result_a = a(pts).reshape(-1)
        result_b = b(pts).reshape(-1)
        expected = np.maximum(result_a, -result_b)
        np.testing.assert_allclose(result_d, expected, atol=1e-10)

    def test_output_shape(self):
        a = sdf.sphere(1)
        b = sdf.box(0.5)
        assert_sdf_callable(a - b, GRID)


class TestIntersection:
    def test_max_semantics(self):
        a = sdf.sphere(1)
        b = sdf.box(1)
        i = a & b
        pts = np.array([[0.0, 0, 0], [0.4, 0.4, 0.4]])
        result_i = i(pts).reshape(-1)
        result_a = a(pts).reshape(-1)
        result_b = b(pts).reshape(-1)
        expected = np.maximum(result_a, result_b)
        np.testing.assert_allclose(result_i, expected, atol=1e-10)

    def test_output_shape(self):
        a = sdf.sphere(1)
        b = sdf.box(1)
        assert_sdf_callable(a & b, GRID)


class TestBlend:
    def test_blend(self):
        a = sdf.sphere(1)
        b = sdf.sphere(1, center=(1, 0, 0))
        s = a.blend(b)
        assert_sdf_callable(s, GRID)


class TestNegate:
    def test_negate(self):
        a = sdf.sphere(1)
        n = a.negate()
        pts = np.array([[0.0, 0, 0]])
        assert a(pts).item() < 0
        assert n(pts).item() > 0


class TestDilateErode:
    def test_dilate(self):
        a = sdf.sphere(1)
        d = a.dilate(0.5)
        # Dilated sphere should be like sphere(1.5)
        pts = np.array([[1.2, 0.0, 0.0]])
        assert d(pts).item() < 0  # inside dilated

    def test_erode(self):
        a = sdf.sphere(1)
        e = a.erode(0.5)
        # Eroded sphere should be like sphere(0.5)
        pts = np.array([[0.8, 0.0, 0.0]])
        assert e(pts).item() > 0  # outside eroded


class TestShell:
    def test_shell(self):
        a = sdf.sphere(1)
        s = a.shell(0.1)
        # Origin should be outside (hollow)
        pts = np.array([[0.0, 0, 0]])
        assert s(pts).item() > 0
        # Surface should be inside shell
        pts = np.array([[1.0, 0, 0]])
        assert s(pts).item() < 0.1


class TestRepeat:
    def test_repeat(self):
        a = sdf.sphere(0.3)
        s = a.repeat(2)
        # Should have copies at grid positions
        assert_sdf_callable(s, GRID)
