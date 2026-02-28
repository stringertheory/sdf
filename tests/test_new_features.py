"""Tests for Phase 2 new features."""
import numpy as np
import pytest

import sdf
from sdf import d3, d2, ease
from .conftest import (
    make_grid_3d, make_grid_2d, make_random_points_3d,
    assert_sdf_callable, assert_inside, assert_outside, assert_on_surface,
)


GRID3 = make_grid_3d(10)
GRID2 = make_grid_2d(50)


# ---------------------------------------------------------------------------
# 2.2 TPMS Lattice Primitives
# ---------------------------------------------------------------------------

class TestGyroid:
    def test_callable(self):
        s = sdf.d3.gyroid(1)
        assert_sdf_callable(s, GRID3)

    def test_periodicity(self):
        s = sdf.d3.gyroid(1)
        pts = np.array([[0.0, 0.0, 0.0]])
        shifted = np.array([[1.0, 0.0, 0.0]])
        np.testing.assert_allclose(
            s(pts).reshape(-1), s(shifted).reshape(-1), atol=1e-10
        )


class TestSchwartzP:
    def test_callable(self):
        s = sdf.d3.schwartz_p(1)
        assert_sdf_callable(s, GRID3)


class TestDiamond:
    def test_callable(self):
        s = sdf.d3.diamond(1)
        assert_sdf_callable(s, GRID3)


# ---------------------------------------------------------------------------
# 2.3 Arithmetic & Morph Operations
# ---------------------------------------------------------------------------

class TestArithmeticOps:
    def test_addition(self):
        a = sdf.sphere(1)
        b = sdf.sphere(1, center=(1, 0, 0))
        s = a.addition(b)
        assert_sdf_callable(s, GRID3)

    def test_multiplication(self):
        a = sdf.sphere(1)
        b = sdf.sphere(1, center=(1, 0, 0))
        s = a.multiplication(b)
        assert_sdf_callable(s, GRID3)

    def test_division(self):
        a = sdf.sphere(1)
        b = sdf.sphere(2)
        s = a.division(b)
        assert_sdf_callable(s, GRID3)

    def test_morph(self):
        a = sdf.sphere(1)
        b = sdf.box(1)
        c = sdf.d3.plane()  # control field
        s = a.morph(b, c, 1.0)
        assert_sdf_callable(s, GRID3)


# ---------------------------------------------------------------------------
# 2.4 Improved Shell
# ---------------------------------------------------------------------------

class TestShellSided:
    def test_center(self):
        s = sdf.sphere(1).shell_sided(0.1, side="center")
        assert_sdf_callable(s, GRID3)

    def test_inner(self):
        s = sdf.sphere(1).shell_sided(0.1, side="inner")
        assert_sdf_callable(s, GRID3)

    def test_outer(self):
        s = sdf.sphere(1).shell_sided(0.1, side="outer")
        assert_sdf_callable(s, GRID3)

    def test_invalid_side(self):
        with pytest.raises(ValueError):
            sdf.sphere(1).shell_sided(0.1, side="invalid")


# ---------------------------------------------------------------------------
# 2.5 Mirror Operations
# ---------------------------------------------------------------------------

class TestMirror:
    def test_mirror_3d(self):
        s = sdf.sphere(1, center=(2, 0, 0))
        m = s.mirror((1, 0, 0))
        assert_inside(m, (-2, 0, 0))
        assert_outside(m, (2, 0, 0))

    def test_mirror_copy_3d(self):
        s = sdf.sphere(1, center=(2, 0, 0))
        m = s.mirror_copy((1, 0, 0))
        assert_inside(m, (2, 0, 0))
        assert_inside(m, (-2, 0, 0))


# ---------------------------------------------------------------------------
# 2.6 Stretch, Shear, Modulate Between
# ---------------------------------------------------------------------------

class TestStretch:
    def test_callable(self):
        s = sdf.sphere(1).stretch((0, 0, -1), (0, 0, 1))
        assert_sdf_callable(s, GRID3)


class TestShear:
    def test_callable(self):
        s = sdf.box(1).shear((0, 0, -1), (0, 0, 1), (1, 0, 0))
        assert_sdf_callable(s, GRID3)


class TestModulateBetween:
    def test_callable(self):
        s = sdf.sphere(1).modulate_between((0, 0, -1), (0, 0, 1))
        assert_sdf_callable(s, GRID3)


# ---------------------------------------------------------------------------
# 2.7 Chamfer, Twist Between, Pieslice
# ---------------------------------------------------------------------------

class TestChamfer:
    def test_callable(self):
        a = sdf.sphere(1)
        b = sdf.box(1)
        s = a.chamfer(b, 0.1)
        assert_sdf_callable(s, GRID3)


class TestTwistBetween:
    def test_callable(self):
        s = sdf.box(1).twist_between((0, 0, -1), (0, 0, 1))
        assert_sdf_callable(s, GRID3)


class TestPieslice:
    def test_callable(self):
        s = sdf.d3.pieslice(np.pi / 4)
        assert_sdf_callable(s, GRID3)

    def test_centered(self):
        s = sdf.d3.pieslice(np.pi / 4, centered=True)
        assert_sdf_callable(s, GRID3)


# ---------------------------------------------------------------------------
# 2.8 Capsule Chain + Bezier
# ---------------------------------------------------------------------------

class TestCapsuleChain:
    def test_callable(self):
        pts = [(0, 0, 0), (1, 0, 0), (1, 1, 0)]
        s = d3.capsule_chain(pts, radius=0.1)
        assert_sdf_callable(s, GRID3)

    def test_inside_segment(self):
        pts = [(0, 0, 0), (2, 0, 0)]
        s = d3.capsule_chain(pts, radius=0.5)
        assert_inside(s, (1, 0, 0))

    def test_with_diameter(self):
        pts = [(0, 0, 0), (2, 0, 0)]
        s = d3.capsule_chain(pts, diameter=1.0)
        assert_inside(s, (1, 0, 0))


class TestBezier:
    def test_callable(self):
        s = d3.bezier(
            (0, 0, 0), (1, 1, 0), (2, 1, 0), (3, 0, 0),
            radius=0.1, steps=10,
        )
        assert_sdf_callable(s, GRID3)

    def test_variable_radius(self):
        s = d3.bezier(
            (0, 0, 0), (1, 1, 0), (2, 1, 0), (3, 0, 0),
            radius=ease.linear.between(0.3, 0.1), steps=10,
        )
        assert_sdf_callable(s, GRID3)


# ---------------------------------------------------------------------------
# 2.9 Thread + Screw
# ---------------------------------------------------------------------------

class TestThread:
    def test_callable(self):
        s = d3.Thread(pitch=5, diameter=20, offset=1)
        assert_sdf_callable(s, GRID3)


class TestScrew:
    def test_callable(self):
        s = d3.Screw(length=10, pitch=2, diameter=5, offset=0.5)
        assert_sdf_callable(s, GRID3)


# ---------------------------------------------------------------------------
# 2.10 New 2D Primitives
# ---------------------------------------------------------------------------

class TestRoundedPolygon:
    def test_callable(self):
        pts = [(1, 1, 0.1), (-1, 1, 0.1), (-1, -1, 0.1), (1, -1, 0.1)]
        s = d2.rounded_polygon(pts)
        assert_sdf_callable(s, GRID2)

    def test_inside(self):
        pts = [(1, 1, 0.1), (-1, 1, 0.1), (-1, -1, 0.1), (1, -1, 0.1)]
        s = d2.rounded_polygon(pts)
        assert_inside(s, (0, 0))


class TestRoundedCog:
    def test_callable(self):
        s = d2.rounded_cog(2, 0.3, 8)
        assert_sdf_callable(s, GRID2)


# ---------------------------------------------------------------------------
# 2.11 New Extrusion Variants
# ---------------------------------------------------------------------------

class TestRoundedExtrude:
    def test_callable(self):
        c = sdf.circle(1)
        s = c.rounded_extrude(2, radius=0.2)
        pts_3d = make_grid_3d(10)
        assert_sdf_callable(s, pts_3d)


class TestTaperExtrude:
    def test_callable(self):
        c = sdf.circle(1)
        s = c.taper_extrude(2, slope=-0.5)
        pts_3d = make_grid_3d(10)
        assert_sdf_callable(s, pts_3d)


class TestScaleExtrude:
    def test_callable(self):
        c = sdf.circle(1)
        s = c.scale_extrude(2, top=0.5, bottom=1.0)
        pts_3d = make_grid_3d(10)
        assert_sdf_callable(s, pts_3d)


# ---------------------------------------------------------------------------
# 2.12 Skin
# ---------------------------------------------------------------------------

class TestSkin:
    def test_callable(self):
        s = sdf.sphere(1).skin(0.1)
        assert_sdf_callable(s, GRID3)

    def test_inside_becomes_zero(self):
        s = sdf.sphere(1).skin(0.1)
        # Deep interior should return 0 (max(sdf-depth, 0) where sdf << 0)
        pts = np.array([[0.0, 0, 0]])
        assert s(pts).item() == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# 2.13 Copy-on-write .k()
# ---------------------------------------------------------------------------

class TestKCopyOnWrite:
    def test_3d(self):
        s = sdf.sphere(1)
        s2 = s.k(0.5)
        assert s2 is not s
        assert s2._k == 0.5

    def test_2d(self):
        s = sdf.circle(1)
        s2 = s.k(0.5)
        assert s2 is not s
        assert s2._k == 0.5


# ---------------------------------------------------------------------------
# 2.14 Analysis Functions
# ---------------------------------------------------------------------------

class TestBoundsFunction:
    def test_sphere_bounds(self):
        s = sdf.sphere(1)
        (x0, y0, z0), (x1, y1, z1) = s.bounds()
        assert x0 < -0.5
        assert x1 > 0.5


class TestVolumeFunction:
    def test_sphere_volume(self):
        s = sdf.sphere(1)
        v = s.volume(samples=50000)
        expected = 4/3 * np.pi  # ~4.189
        assert abs(v - expected) < 1.0  # rough check


class TestVoxelize:
    def test_callable(self):
        s = sdf.sphere(1)
        vol, spacing, offset = s.voxelize(step=0.2)
        assert vol.ndim == 3
        assert spacing.shape == (3,)
        assert offset.shape == (3,)
