import numpy as np
import pytest

import sdf
from sdf import d3
from .conftest import (
    make_grid_3d, make_random_points_3d,
    assert_sdf_callable, assert_inside, assert_outside, assert_on_surface,
)


GRID = make_grid_3d(20)
RANDOM = make_random_points_3d(1000)


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

class TestSphere:
    def test_origin_inside(self):
        s = sdf.sphere(1)
        assert_inside(s, (0, 0, 0))

    def test_outside(self):
        s = sdf.sphere(1)
        assert_outside(s, (2, 0, 0))

    def test_surface(self):
        s = sdf.sphere(1)
        assert_on_surface(s, (1, 0, 0))
        assert_on_surface(s, (0, 1, 0))
        assert_on_surface(s, (0, 0, 1))

    def test_output_shape(self):
        s = sdf.sphere(1)
        assert_sdf_callable(s, GRID)

    def test_center(self):
        s = sdf.sphere(1, center=(1, 0, 0))
        assert_inside(s, (1, 0, 0))
        # At origin, distance to center is 1 = radius, so on surface
        assert_on_surface(s, (0, 0, 0))
        assert_outside(s, (-1, 0, 0))


class TestBox:
    def test_origin_inside(self):
        s = sdf.box(1)
        assert_inside(s, (0, 0, 0))

    def test_outside(self):
        s = sdf.box(1)
        assert_outside(s, (1, 1, 1))

    def test_surface(self):
        s = sdf.box(2)
        assert_on_surface(s, (1, 0, 0))

    def test_output_shape(self):
        s = sdf.box(1)
        assert_sdf_callable(s, GRID)

    def test_a_b_form(self):
        s = sdf.box(a=(0, 0, 0), b=(2, 2, 2))
        assert_inside(s, (1, 1, 1))
        assert_outside(s, (-1, -1, -1))


class TestRoundedBox:
    def test_origin_inside(self):
        s = sdf.rounded_box((2, 2, 2), 0.1)
        assert_inside(s, (0, 0, 0))

    def test_output_shape(self):
        s = sdf.rounded_box((2, 2, 2), 0.1)
        assert_sdf_callable(s, GRID)


class TestWireframeBox:
    def test_origin_outside(self):
        s = sdf.wireframe_box((2, 2, 2), 0.1)
        assert_outside(s, (0, 0, 0))

    def test_edge_inside(self):
        s = sdf.wireframe_box((2, 2, 2), 0.2)
        assert_inside(s, (1, 1, 0))

    def test_output_shape(self):
        s = sdf.wireframe_box((2, 2, 2), 0.1)
        assert_sdf_callable(s, GRID)


class TestTorus:
    def test_ring_inside(self):
        s = sdf.torus(1, 0.25)
        assert_inside(s, (1, 0, 0))

    def test_center_outside(self):
        s = sdf.torus(1, 0.25)
        assert_outside(s, (0, 0, 0))

    def test_output_shape(self):
        s = sdf.torus(1, 0.25)
        assert_sdf_callable(s, GRID)


class TestCapsule:
    def test_inside(self):
        s = sdf.capsule((0, 0, -1), (0, 0, 1), 0.5)
        assert_inside(s, (0, 0, 0))

    def test_end_inside(self):
        s = sdf.capsule((0, 0, -1), (0, 0, 1), 0.5)
        assert_inside(s, (0, 0, 1))

    def test_outside(self):
        s = sdf.capsule((0, 0, -1), (0, 0, 1), 0.5)
        assert_outside(s, (2, 0, 0))

    def test_output_shape(self):
        s = sdf.capsule((0, 0, -1), (0, 0, 1), 0.5)
        assert_sdf_callable(s, GRID)


class TestCylinder:
    def test_inside(self):
        s = sdf.cylinder(1)
        assert_inside(s, (0, 0, 0))

    def test_outside(self):
        s = sdf.cylinder(1)
        assert_outside(s, (2, 0, 0))

    def test_output_shape(self):
        s = sdf.cylinder(1)
        assert_sdf_callable(s, GRID)


class TestCappedCylinder:
    def test_inside(self):
        s = sdf.capped_cylinder((0, 0, -1), (0, 0, 1), 0.5)
        assert_inside(s, (0, 0, 0))

    def test_outside(self):
        s = sdf.capped_cylinder((0, 0, -1), (0, 0, 1), 0.5)
        assert_outside(s, (0, 0, 3))

    def test_output_shape(self):
        s = sdf.capped_cylinder((0, 0, -1), (0, 0, 1), 0.5)
        assert_sdf_callable(s, GRID)


class TestRoundedCylinder:
    def test_inside(self):
        s = sdf.rounded_cylinder(1, 0.1, 2)
        assert_inside(s, (0, 0, 0))

    def test_output_shape(self):
        s = sdf.rounded_cylinder(1, 0.1, 2)
        assert_sdf_callable(s, GRID)


class TestCappedCone:
    def test_inside(self):
        s = sdf.capped_cone((0, 0, -1), (0, 0, 1), 1, 0.5)
        assert_inside(s, (0, 0, 0))

    def test_output_shape(self):
        s = sdf.capped_cone((0, 0, -1), (0, 0, 1), 1, 0.5)
        assert_sdf_callable(s, GRID)


class TestRoundedCone:
    def test_inside(self):
        s = sdf.rounded_cone(1, 0.5, 2)
        assert_inside(s, (0, 0, 0.5))

    def test_output_shape(self):
        s = sdf.rounded_cone(1, 0.5, 2)
        assert_sdf_callable(s, GRID)


class TestEllipsoid:
    def test_inside(self):
        s = sdf.ellipsoid((1, 2, 3))
        # Origin gives 0/0=nan, test slightly offset
        assert_inside(s, (0.1, 0.1, 0.1))

    def test_outside(self):
        s = sdf.ellipsoid((1, 2, 3))
        assert_outside(s, (5, 5, 5))

    def test_output_shape(self):
        s = sdf.ellipsoid((1, 2, 3))
        assert_sdf_callable(s, GRID)


class TestPyramid:
    def test_inside(self):
        s = sdf.pyramid(1)
        assert_inside(s, (0, 0, 0.1))

    def test_output_shape(self):
        s = sdf.pyramid(1)
        assert_sdf_callable(s, GRID)


class TestPlane:
    def test_along_normal_inside(self):
        # plane(normal=UP) => dot(origin - p, UP) = -p_z
        # So z>0 is negative (inside), z<0 is positive (outside)
        s = sdf.plane()
        assert_inside(s, (0, 0, 1))

    def test_against_normal_outside(self):
        s = sdf.plane()
        assert_outside(s, (0, 0, -1))

    def test_output_shape(self):
        s = sdf.plane()
        assert_sdf_callable(s, GRID)


class TestSlab3D:
    def test_inside(self):
        s = sdf.d3.slab(z0=-1, z1=1)
        assert_inside(s, (0, 0, 0))

    def test_outside(self):
        s = sdf.d3.slab(z0=-1, z1=1)
        assert_outside(s, (0, 0, 5))

    def test_output_shape(self):
        s = sdf.d3.slab(z0=-1, z1=1)
        assert_sdf_callable(s, GRID)


# Platonic Solids

class TestTetrahedron:
    def test_origin_inside(self):
        s = sdf.tetrahedron(1)
        assert_inside(s, (0, 0, 0))

    def test_output_shape(self):
        s = sdf.tetrahedron(1)
        assert_sdf_callable(s, GRID)


class TestOctahedron:
    def test_origin_inside(self):
        s = sdf.octahedron(1)
        assert_inside(s, (0, 0, 0))

    def test_output_shape(self):
        s = sdf.octahedron(1)
        assert_sdf_callable(s, GRID)


class TestDodecahedron:
    def test_origin_inside(self):
        s = sdf.dodecahedron(1)
        assert_inside(s, (0, 0, 0))

    def test_output_shape(self):
        s = sdf.dodecahedron(1)
        assert_sdf_callable(s, GRID)


class TestIcosahedron:
    def test_origin_inside(self):
        s = sdf.icosahedron(1)
        assert_inside(s, (0, 0, 0))

    def test_output_shape(self):
        s = sdf.icosahedron(1)
        assert_sdf_callable(s, GRID)


# ---------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------

class TestTranslate3D:
    def test_translate(self, unit_sphere):
        s = unit_sphere.translate((2, 0, 0))
        assert_inside(s, (2, 0, 0))
        assert_outside(s, (0, 0, 0))

    def test_output_shape(self, unit_sphere):
        s = unit_sphere.translate((1, 0, 0))
        assert_sdf_callable(s, GRID)


class TestScale3D:
    def test_uniform(self, unit_sphere):
        s = unit_sphere.scale(2)
        assert_inside(s, (1.5, 0, 0))

    def test_non_uniform(self, unit_sphere):
        s = unit_sphere.scale((2, 1, 1))
        assert_inside(s, (1.5, 0, 0))
        assert_outside(s, (0, 1.5, 0))

    def test_output_shape(self, unit_sphere):
        s = unit_sphere.scale(2)
        assert_sdf_callable(s, GRID)


class TestRotate3D:
    def test_rotate_z(self, unit_box):
        s = unit_box.rotate(np.pi / 4)
        assert_inside(s, (0, 0, 0))

    def test_output_shape(self, unit_box):
        s = unit_box.rotate(np.pi / 4)
        assert_sdf_callable(s, GRID)


class TestRotateTo:
    def test_identity(self, unit_sphere):
        s = unit_sphere.rotate_to(sdf.d3.Z, sdf.d3.Z)
        assert_inside(s, (0, 0, 0))


class TestOrient:
    def test_orient(self, unit_sphere):
        s = unit_sphere.orient(sdf.d3.X)
        assert_inside(s, (0, 0, 0))


class TestCircularArray3D:
    def test_array(self, unit_sphere):
        s = unit_sphere.circular_array(4, offset=3)
        assert_inside(s, (3, 0, 0))
        assert_outside(s, (0, 0, 0))

    def test_output_shape(self, unit_sphere):
        s = unit_sphere.circular_array(4, offset=3)
        assert_sdf_callable(s, GRID)


class TestElongate3D:
    def test_elongate(self, unit_sphere):
        s = unit_sphere.elongate((1, 0, 0))
        assert_inside(s, (1, 0, 0))

    def test_output_shape(self, unit_sphere):
        s = unit_sphere.elongate((1, 0, 0))
        assert_sdf_callable(s, GRID)


class TestTwist:
    def test_callable(self, unit_box):
        s = unit_box.twist(0.5)
        assert_sdf_callable(s, GRID)


class TestBend:
    def test_callable(self, unit_box):
        s = unit_box.bend(0.5)
        assert_sdf_callable(s, GRID)


class TestBendLinear:
    def test_callable(self, unit_box):
        s = unit_box.bend_linear((0, 0, -1), (0, 0, 1), (0.5, 0, 0))
        assert_sdf_callable(s, GRID)


class TestBendRadial:
    def test_callable(self, unit_box):
        s = unit_box.bend_radial(0, 1, 0.5)
        assert_sdf_callable(s, GRID)


class TestTransitionLinear:
    def test_callable(self, unit_sphere, unit_box):
        s = unit_sphere.transition_linear(unit_box)
        assert_sdf_callable(s, GRID)


class TestTransitionRadial:
    def test_callable(self, unit_sphere, unit_box):
        s = unit_sphere.transition_radial(unit_box)
        assert_sdf_callable(s, GRID)


class TestWrapAround:
    def test_callable(self, unit_box):
        s = unit_box.wrap_around(-2, 2)
        assert_sdf_callable(s, GRID)


class TestSlice:
    def test_callable(self, unit_sphere):
        s = unit_sphere.slice()
        pts = np.array([[0.0, 0.0], [2.0, 0.0], [0.5, 0.5]])
        assert_sdf_callable(s, pts)


# ---------------------------------------------------------------------------
# Boolean operators
# ---------------------------------------------------------------------------

class TestBooleanOperators:
    def test_union_operator(self, unit_sphere, unit_box):
        s = unit_sphere | unit_box
        assert_inside(s, (0, 0, 0))
        assert_sdf_callable(s, GRID)

    def test_intersection_operator(self, unit_sphere, unit_box):
        s = unit_sphere & unit_box
        assert_inside(s, (0, 0, 0))
        assert_sdf_callable(s, GRID)

    def test_difference_operator(self, unit_sphere, unit_box):
        s = unit_sphere - unit_box
        assert_sdf_callable(s, GRID)

    def test_union_function(self):
        a = sdf.sphere(1)
        b = sdf.sphere(1, center=(1, 0, 0))
        s = sdf.d3.union(a, b)
        assert_inside(s, (0, 0, 0))
        assert_inside(s, (1, 0, 0))

    def test_difference_function(self):
        a = sdf.sphere(1)
        b = sdf.sphere(0.5)
        s = sdf.d3.difference(a, b)
        assert_outside(s, (0, 0, 0))

    def test_intersection_function(self):
        a = sdf.sphere(1)
        b = sdf.box(1)
        s = sdf.d3.intersection(a, b)
        assert_inside(s, (0, 0, 0))


class TestKBehavior:
    def test_k_returns_copy(self):
        """Updated behavior: .k() returns a new object (copy-on-write)."""
        s = sdf.sphere(1)
        result = s.k(0.5)
        assert result is not s
        assert result._k == 0.5
        assert not hasattr(s, '_k')
