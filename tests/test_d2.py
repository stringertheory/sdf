import numpy as np
import pytest

import sdf
from sdf import d2
from .conftest import (
    make_grid_2d, make_random_points_2d,
    assert_sdf_callable, assert_inside, assert_outside, assert_on_surface,
)


GRID = make_grid_2d(100)


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

class TestCircle:
    def test_origin_inside(self):
        s = sdf.circle(1)
        assert_inside(s, (0, 0))

    def test_outside(self):
        s = sdf.circle(1)
        assert_outside(s, (2, 0))

    def test_surface(self):
        s = sdf.circle(1)
        assert_on_surface(s, (1, 0))

    def test_output_shape(self):
        s = sdf.circle(1)
        assert_sdf_callable(s, GRID)


class TestLine:
    def test_along_normal_is_inside(self):
        # line(normal=UP) => dot(origin - p, UP) = -p_y
        # So y>0 is negative (inside), y<0 is positive (outside)
        s = sdf.d2.line()
        assert_inside(s, (0, 1))

    def test_against_normal_is_outside(self):
        s = sdf.d2.line()
        assert_outside(s, (0, -1))

    def test_output_shape(self):
        s = sdf.d2.line()
        assert_sdf_callable(s, GRID)


class TestSlab2D:
    def test_inside(self):
        s = sdf.d2.slab(x0=-1, x1=1)
        assert_inside(s, (0, 0))

    def test_outside(self):
        s = sdf.d2.slab(x0=-1, x1=1)
        assert_outside(s, (5, 0))

    def test_output_shape(self):
        s = sdf.d2.slab(x0=-1, x1=1)
        assert_sdf_callable(s, GRID)


class TestRectangle:
    def test_origin_inside(self):
        s = sdf.rectangle(2)
        assert_inside(s, (0, 0))

    def test_outside(self):
        s = sdf.rectangle(2)
        assert_outside(s, (2, 2))

    def test_surface(self):
        s = sdf.rectangle(2)
        assert_on_surface(s, (1, 0))

    def test_output_shape(self):
        s = sdf.rectangle(2)
        assert_sdf_callable(s, GRID)

    def test_a_b_form(self):
        s = sdf.rectangle(a=(0, 0), b=(2, 2))
        assert_inside(s, (1, 1))
        assert_outside(s, (-1, -1))


class TestRoundedRectangle:
    def test_origin_inside(self):
        s = sdf.rounded_rectangle(np.array([2.0, 2.0]), 0.1)
        assert_inside(s, (0, 0))

    def test_output_shape(self):
        s = sdf.rounded_rectangle(np.array([2.0, 2.0]), 0.1)
        assert_sdf_callable(s, GRID)


class TestEquilateralTriangle:
    def test_origin_inside(self):
        s = sdf.equilateral_triangle()
        assert_inside(s, (0, -0.2))

    def test_output_shape(self):
        s = sdf.equilateral_triangle()
        assert_sdf_callable(s, GRID)


class TestHexagon:
    def test_origin_inside(self):
        s = sdf.hexagon(1)
        assert_inside(s, (0, 0))

    def test_output_shape(self):
        s = sdf.hexagon(1)
        assert_sdf_callable(s, GRID)


class TestRoundedX:
    def test_output_shape(self):
        s = sdf.rounded_x(1, 0.1)
        assert_sdf_callable(s, GRID)


class TestPolygon:
    def test_square(self):
        pts = [(1, 1), (-1, 1), (-1, -1), (1, -1)]
        s = sdf.polygon(pts)
        assert_inside(s, (0, 0))
        assert_outside(s, (2, 2))

    def test_output_shape(self):
        pts = [(1, 1), (-1, 1), (-1, -1), (1, -1)]
        s = sdf.polygon(pts)
        assert_sdf_callable(s, GRID)


class TestVesica:
    def test_origin_inside(self):
        s = sdf.vesica(1, 0.5)
        assert_inside(s, (0, 0))

    def test_output_shape(self):
        s = sdf.vesica(1, 0.5)
        assert_sdf_callable(s, GRID)


# ---------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------

class TestTranslate2D:
    def test_translate(self, unit_circle):
        s = unit_circle.translate((2, 0))
        assert_inside(s, (2, 0))
        assert_outside(s, (0, 0))


class TestScale2D:
    def test_uniform(self, unit_circle):
        s = unit_circle.scale(2)
        assert_inside(s, (1.5, 0))


class TestRotate2D:
    def test_rotate(self, unit_rectangle):
        s = unit_rectangle.rotate(np.pi / 4)
        assert_inside(s, (0, 0))


class TestCircularArray2D:
    def test_array(self, unit_circle):
        s = unit_circle.circular_array(4)
        assert_inside(s, (0, 0))


class TestElongate2D:
    def test_elongate(self, unit_circle):
        s = unit_circle.elongate((1, 0))
        assert_inside(s, (1, 0))


# ---------------------------------------------------------------------------
# 2D → 3D
# ---------------------------------------------------------------------------

class TestExtrude:
    def test_extrude(self, unit_circle):
        s = unit_circle.extrude(2)
        pts_3d = np.array([[0, 0, 0], [0, 0, 0.5], [2, 0, 0]])
        assert_sdf_callable(s, pts_3d)
        assert_inside(s, (0, 0, 0))
        assert_outside(s, (2, 0, 0))


class TestExtrudeTo:
    def test_extrude_to(self, unit_circle, unit_rectangle):
        s = unit_circle.extrude_to(unit_rectangle, 2)
        pts_3d = np.array([[0, 0, 0], [0, 0, 0.5]])
        assert_sdf_callable(s, pts_3d)


class TestRevolve:
    def test_revolve(self, unit_circle):
        s = unit_circle.revolve(3)
        pts_3d = np.array([[3, 0, 0], [0, 0, 0]])
        assert_sdf_callable(s, pts_3d)
        assert_inside(s, (3, 0, 0))
