import numpy as np
import pytest
import tempfile
import os

import sdf
from sdf.core import _cartesian_product, _estimate_bounds, generate, save, sample_slice


class TestCartesianProduct:
    def test_shape(self):
        X = np.array([1, 2, 3])
        Y = np.array([4, 5])
        Z = np.array([6])
        result = _cartesian_product(X, Y, Z)
        assert result.shape == (6, 3)

    def test_content(self):
        X = np.array([0, 1])
        Y = np.array([0, 1])
        Z = np.array([0])
        result = _cartesian_product(X, Y, Z)
        assert result.shape == (4, 3)
        expected = {(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 0)}
        actual = {tuple(row) for row in result}
        assert actual == expected


class TestEstimateBounds:
    def test_sphere_bounds(self):
        s = sdf.sphere(1)
        (x0, y0, z0), (x1, y1, z1) = _estimate_bounds(s)
        # Should contain the surface
        assert x0 <= -0.9
        assert x1 >= 0.9
        assert y0 <= -0.9
        assert y1 >= 0.9
        assert z0 <= -0.9
        assert z1 >= 0.9
        # Should be reasonably tight (not huge)
        assert x0 >= -3
        assert x1 <= 3


class TestGenerate:
    def test_sphere_mesh(self):
        s = sdf.sphere(1)
        points = generate(s, step=0.1, verbose=False)
        if isinstance(points, list):
            points = np.array(points)
        assert len(points) > 0
        # Points should be near surface
        if len(points) > 0:
            pts = np.array(points).reshape(-1, 3)
            dists = np.linalg.norm(pts, axis=1)
            assert np.all(dists < 1.5)
            assert np.all(dists > 0.5)


class TestSave:
    def test_save_stl(self):
        s = sdf.sphere(1)
        with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as f:
            path = f.name
        try:
            save(path, s, step=0.2, verbose=False)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0
        finally:
            os.unlink(path)


class TestSampleSlice:
    def test_z_slice(self):
        s = sdf.sphere(1)
        result, extent, axes = sample_slice(s, w=32, h=32, z=0)
        assert result.shape == (32, 32)
        assert axes == 'YX'

    def test_x_slice(self):
        s = sdf.sphere(1)
        result, extent, axes = sample_slice(s, w=32, h=32, x=0)
        assert result.shape == (32, 32)
        assert axes == 'ZY'
