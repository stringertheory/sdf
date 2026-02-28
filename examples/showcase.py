import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md("""
    # SDF Library Showcase

    This notebook demonstrates the SDF library's capabilities for generating
    3D meshes from signed distance functions. We'll build up from simple
    primitives to complex compositions, and preview everything with `.show()`.

    Drag to rotate, scroll to zoom, right-click to pan.
    """)
    return


@app.cell
def _():
    import sdf
    from sdf import (
        X, Y, Z, box, capped_cylinder, circle, cylinder,
        hexagon, sphere, union,
    )
    from sdf import ease
    import numpy as np
    return (X, Y, Z, box, capped_cylinder, circle, cylinder, ease, hexagon, np, sdf, sphere, union)


@app.cell
def _(mo):
    mo.md("""
    ## 1. Classic CSG Example

    The canonical [Constructive Solid Geometry](https://en.wikipedia.org/wiki/Constructive_solid_geometry)
    example: a sphere intersected with a box, minus three cylinders.
    """)
    return


@app.cell
def _(X, Y, Z, box, cylinder, sphere):
    # Classic CSG: sphere ∩ box - 3 cylinders
    csg = sphere(1) & box(1.5)
    _c = cylinder(0.5)
    csg -= _c.orient(X) | _c.orient(Y) | _c.orient(Z)
    csg.show()
    return


@app.cell
def _(mo):
    mo.md("""
    ## 2. Smooth Boolean Operations

    Using the `k` parameter for smooth blending between shapes.
    """)
    return


@app.cell
def _(box, sphere, union):
    # Smooth union of a sphere and a box
    smooth = union(sphere(1), box(1, center=(1.2, 0, 0)), k=0.3)
    smooth.show()
    return


@app.cell
def _(mo):
    mo.md("""
    ## 3. TPMS Lattice Structures

    Triply Periodic Minimal Surfaces are useful for lightweight
    lattice structures. Here we bound a gyroid with a sphere.
    """)
    return


@app.cell
def _(sdf, sphere):
    # Gyroid lattice bounded by a sphere
    lattice = sdf.d3.gyroid(w=2.0).shell(0.4) & sphere(3)
    lattice.show()
    return


@app.cell
def _(mo):
    mo.md("""
    ## 4. Bezier Curves with Variable Radius

    Using the `Easing` system for variable-width bezier curves.
    """)
    return


@app.cell
def _(ease, sdf):
    # Bezier curve that tapers from thick to thin
    curve = sdf.d3.bezier(
        (-2, 0, 0), (-1, 2, 0), (1, -2, 0), (2, 0, 0),
        radius=ease.linear.between(0.4, 0.1),
        steps=30,
    )
    curve.show()
    return


@app.cell
def _(mo):
    mo.md("""
    ## 5. Mirror Operations

    Create symmetric designs by mirroring shapes.
    """)
    return


@app.cell
def _(capped_cylinder, sphere, union):
    # Build half a shape, then mirror it
    arm = capped_cylinder((0, 0, 0), (2, 0, 0), 0.3)
    body = sphere(1)
    half = union(body, arm, k=0.2)
    mirrored = half.mirror_copy((1, 0, 0))
    # Add a hat
    hat = capped_cylinder((0, 0, 0.8), (0, 0, 1.5), 0.6)
    figure = union(mirrored, hat, k=0.15)
    figure.show()
    return


@app.cell
def _(mo):
    mo.md("""
    ## 6. Twist

    The `.twist()` operation rotates a shape progressively along the Z axis.
    Here a box is twisted into a helix-like column.
    """)
    return


@app.cell
def _(box):
    # A tall box twisted along the Z axis
    twisted = box((0.8, 0.8, 2)).twist(1.5)
    twisted.show()
    return


@app.cell
def _(mo):
    mo.md("""
    ## 7. Extrusion Variants

    Different ways to extrude 2D shapes into 3D.
    """)
    return


@app.cell
def _(circle, hexagon, union):
    # Regular extrude vs rounded extrude vs scale extrude
    _a = hexagon(1).extrude(1).translate((-3, 0, 0))
    _b = circle(1).rounded_extrude(1, radius=0.2)
    _c = circle(1).scale_extrude(1.5, top=0.3, bottom=1.0).translate((3, 0, 0))
    extrusions = union(_a, _b, _c)
    extrusions.show()
    return


@app.cell
def _(mo):
    mo.md("""
    ## 8. Thread

    Helical thread primitive for mechanical parts.
    """)
    return


@app.cell
def _(sdf):
    # Threaded rod section
    thread = sdf.d3.Thread(pitch=3, diameter=10, offset=0.8)
    thread = thread & sdf.d3.slab(z0=0, z1=20)
    thread.show()
    return


@app.cell
def _(mo):
    mo.md("""
    ## 9. Analysis: Volume Estimation

    Monte Carlo volume estimation for any SDF.
    """)
    return


@app.cell
def _(mo, np, sphere):
    s = sphere(1)
    estimated = s.volume(samples=100000)
    exact = 4 / 3 * np.pi
    mo.md(
        f"""
        **Unit sphere volume:**
        - Estimated: `{estimated:.3f}`
        - Exact: `{exact:.3f}`
        - Error: `{abs(estimated - exact):.3f}`
        """
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## 10. Easing System

    The new `Easing` class supports arithmetic, composition, and plotting.
    """)
    return


@app.cell
def _(ease):
    # Compose easings and plot
    composed = ease.in_quad | ease.out_quad  # transition at midpoint
    scaled = ease.in_out_cubic.between(0, 10)  # scale output to [0, 10]
    ease.in_out_cubic.plot()
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
