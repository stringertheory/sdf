import numpy as np
import pytest

import sdf


def make_grid_3d(n=20):
    lin = np.linspace(-2, 2, n)
    return np.array(np.meshgrid(lin, lin, lin)).T.reshape(-1, 3)


def make_grid_2d(n=100):
    lin = np.linspace(-2, 2, n)
    return np.array(np.meshgrid(lin, lin)).T.reshape(-1, 2)


def make_random_points_3d(n=1000, lo=-2, hi=2, seed=42):
    rng = np.random.default_rng(seed)
    return rng.uniform(lo, hi, (n, 3))


def make_random_points_2d(n=1000, lo=-2, hi=2, seed=42):
    rng = np.random.default_rng(seed)
    return rng.uniform(lo, hi, (n, 2))


def assert_sdf_callable(s, points):
    result = s(points)
    assert result.shape == (len(points), 1), (
        f"Expected shape ({len(points)}, 1), got {result.shape}"
    )


def assert_inside(s, point):
    point = np.array(point, dtype=float).reshape(1, -1)
    val = s(point).item()
    assert val < 0, f"Expected negative SDF at {point}, got {val}"


def assert_outside(s, point):
    point = np.array(point, dtype=float).reshape(1, -1)
    val = s(point).item()
    assert val > 0, f"Expected positive SDF at {point}, got {val}"


def assert_on_surface(s, point, tol=0.01):
    point = np.array(point, dtype=float).reshape(1, -1)
    val = s(point).item()
    assert abs(val) < tol, f"Expected ~0 SDF at {point}, got {val}"


@pytest.fixture
def unit_sphere():
    return sdf.sphere(1)


@pytest.fixture
def unit_box():
    return sdf.box(1)


@pytest.fixture
def unit_circle():
    return sdf.circle(1)


@pytest.fixture
def unit_rectangle():
    return sdf.rectangle(1)
