"""Interactive 3D preview using pyvista."""

import numpy as np

# Default sample count for .show() — much lower than generate()'s 2**22
# to give a fast preview. Users can override with samples= or step=.
SHOW_SAMPLES = 2 ** 18


def _import_pyvista():
    try:
        import pyvista as pv
        return pv
    except ImportError:
        raise ImportError(
            "pyvista is required for .show(). Install it with: pip install pyvista"
        )


def show(sdf_or_points, **generate_kwargs):
    """Show an SDF or mesh points in an interactive 3D viewer.

    Renders off-screen and returns an image array, which displays
    natively in marimo (via mo.image()) and Jupyter notebooks.

    Args:
        sdf_or_points: Either an SDF3 object or an Nx3 array of triangle vertices.
        **generate_kwargs: Keyword arguments passed to generate() if SDF3 is given.
            Defaults to samples=2**18 for a fast preview.

    Returns:
        Image as numpy array (H, W, 3), or None if no triangles.
    """
    pv = _import_pyvista()

    if hasattr(sdf_or_points, 'generate'):
        generate_kwargs.setdefault('verbose', False)
        generate_kwargs.setdefault('samples', SHOW_SAMPLES)
        points = sdf_or_points.generate(**generate_kwargs)
    else:
        points = np.asarray(sdf_or_points)

    points = np.asarray(points, dtype='float64').reshape(-1, 3)
    n_triangles = len(points) // 3

    if n_triangles == 0:
        print('No triangles to display.')
        return None

    # Deduplicate vertices for pyvista
    unique_verts, inverse = np.unique(points, axis=0, return_inverse=True)
    faces = inverse.reshape(-1, 3)

    # pyvista expects faces as [n_verts, v0, v1, v2, ...]
    pv_faces = np.column_stack([
        np.full(n_triangles, 3),
        faces,
    ]).ravel()

    mesh = pv.PolyData(unique_verts, pv_faces)
    mesh.compute_normals(inplace=True)

    pl = pv.Plotter(off_screen=True)
    pl.add_mesh(mesh, color='steelblue', smooth_shading=True)
    return pl.screenshot()
