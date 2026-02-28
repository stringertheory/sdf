"""Visual regression tests for SDF mesh generation and rendering.

Run with:
    pytest -m visual                    # compare against references
    pytest -m visual --update-references  # regenerate reference PNGs
"""

import base64
import html as html_mod
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from textwrap import dedent
from typing import Callable, Optional

import numpy as np
import pytest

import sdf
from sdf import (
    X, Y, Z, pi,
    box, capsule, capped_cone, capped_cylinder, circle, cylinder,
    ellipsoid, hexagon, plane, rectangle, sphere, union,
)
from sdf import ease
from sdf.viewer import _generate_mesh

REFERENCE_DIR = Path(__file__).parent / "visual_references"
GALLERY_PATH = Path(__file__).parent / "visual_gallery.html"

# Image comparison threshold: mean absolute error per pixel channel (0-255 scale)
MAE_THRESHOLD = 2.0

# Mesh generation samples — low enough to be fast, high enough for recognizable shapes
SAMPLES = 2**18

# Max faces in viewer mesh data (keeps per-shape base64 under ~250KB)
MAX_VIEWER_FACES = 15000


# ---------------------------------------------------------------------------
# Script helper — wraps body code into a complete runnable script
# ---------------------------------------------------------------------------

def _script(body, ease=False):
    imports = "from sdf import *"
    if ease:
        imports += "\nfrom sdf import ease"
    return f"{imports}\n\n{dedent(body).strip()}\n\nf.save('out.stl')"


# ---------------------------------------------------------------------------
# Shape registry
# ---------------------------------------------------------------------------

@dataclass
class Shape:
    name: str
    code: str
    builder: Callable[[], "sdf.d3.SDF3"]
    generate_kwargs: dict = field(default_factory=dict)


# -- 3D Primitives --

def _build_sphere():
    return sphere(1)


def _build_box():
    return box(1)


def _build_rounded_box():
    return sdf.d3.rounded_box((1, 2, 3), 0.25)


def _build_wireframe_box():
    return sdf.d3.wireframe_box((1, 2, 3), 0.05)


def _build_torus():
    return sdf.d3.torus(1, 0.25)


def _build_capsule():
    return capsule(-Z, Z, 0.5)


def _build_capped_cylinder():
    return capped_cylinder(-Z, Z, 0.5)


def _build_rounded_cylinder():
    return sdf.d3.rounded_cylinder(0.5, 0.1, 2)


def _build_capped_cone():
    return capped_cone(-Z, Z, 1, 0.5)


def _build_rounded_cone():
    return sdf.d3.rounded_cone(0.75, 0.25, 2)


def _build_ellipsoid():
    return ellipsoid((1, 2, 3))


def _build_pyramid():
    return sdf.d3.pyramid(1)


# -- Platonic Solids --

def _build_tetrahedron():
    return sdf.d3.tetrahedron(1)


def _build_octahedron():
    return sdf.d3.octahedron(1)


def _build_dodecahedron():
    return sdf.d3.dodecahedron(1)


def _build_icosahedron():
    return sdf.d3.icosahedron(1)


# -- Infinite 3D Primitives (bounded) --

def _build_plane():
    return sphere() & plane()


def _build_slab():
    return sphere() & sdf.d3.slab(z0=-0.5, z1=0.5, x0=0)


def _build_cylinder_infinite():
    return sphere() - cylinder(0.5)


# -- Positioning --

def _build_translate():
    return sphere().translate((0, 0, 2))


def _build_scale():
    return sphere().scale(2)


def _build_rotate():
    return capped_cylinder(-Z, Z, 0.5).rotate(pi / 4, X)


def _build_orient():
    c = capped_cylinder(-Z, Z, 0.25)
    return c.orient(X) | c.orient(Y) | c.orient(Z)


# -- Boolean Operations --

def _build_boolean_union():
    a = box((3, 3, 0.5))
    b = sphere()
    return a | b


def _build_boolean_difference():
    a = box((3, 3, 0.5))
    b = sphere()
    return a - b


def _build_boolean_intersection():
    a = box((3, 3, 0.5))
    b = sphere()
    return a & b


def _build_csg():
    f = sphere(1) & box(1.5)
    c = cylinder(0.5)
    f -= c.orient(X) | c.orient(Y) | c.orient(Z)
    return f


def _build_smooth_union():
    a = box((3, 3, 0.5))
    b = sphere()
    return a | b.k(0.25)


def _build_smooth_difference():
    a = box((3, 3, 0.5))
    b = sphere()
    return a - b.k(0.25)


def _build_smooth_intersection():
    a = box((3, 3, 0.5))
    b = sphere()
    return a & b.k(0.25)


# -- Repetition --

def _build_repeat():
    return sphere().repeat(3, (1, 1, 0))


def _build_circular_array():
    return capped_cylinder(-Z, Z, 0.5).circular_array(8, 4)


# -- Miscellaneous --

def _build_blend():
    return sphere().blend(box())


def _build_dilate():
    f = sphere(1) & box(1.5)
    c = cylinder(0.5)
    f -= c.orient(X) | c.orient(Y) | c.orient(Z)
    return f.dilate(0.1)


def _build_erode():
    f = sphere(1) & box(1.5)
    c = cylinder(0.5)
    f -= c.orient(X) | c.orient(Y) | c.orient(Z)
    return f.erode(0.1)


def _build_shell():
    return sphere().shell(0.05) & plane(-Z)


def _build_elongate():
    f = sphere(1) & box(1.5)
    c = cylinder(0.5)
    f -= c.orient(X) | c.orient(Y) | c.orient(Z)
    return f.elongate((0.25, 0.5, 0.75))


def _build_twist():
    return box().twist(pi / 2)


def _build_bend():
    return box().bend(1)


def _build_bend_linear():
    return capsule(-Z * 2, Z * 2, 0.25).bend_linear(-Z, Z, X, ease.in_out_quad)


def _build_bend_radial():
    return box((5, 5, 0.25)).bend_radial(1, 2, -1, ease.in_out_quad)


def _build_transition_linear():
    return box().transition_linear(sphere(), e=ease.in_out_quad)


def _build_transition_radial():
    return box().transition_radial(sphere(), e=ease.in_out_quad)


# -- 2D to 3D Operations --

def _build_extrude():
    return hexagon(1).extrude(1)


def _build_extrude_to():
    return rectangle(2).extrude_to(circle(1), 2, ease.in_out_quad)


def _build_revolve():
    return hexagon(1).revolve(3)


def _build_rounded_extrude():
    return circle(1).rounded_extrude(2, radius=0.2)


def _build_taper_extrude():
    return circle(1).taper_extrude(2, slope=-0.5)


def _build_scale_extrude():
    return circle(1).scale_extrude(2, top=0.5, bottom=1.0)


# -- 3D to 2D Operations --

def _build_slice():
    f = sphere(1) & box(1.5)
    c = cylinder(0.5)
    f -= c.orient(X) | c.orient(Y) | c.orient(Z)
    return f.translate((0, 0, 0.55)).slice().extrude(0.1)


# -- TPMS Lattice Primitives --

def _build_gyroid():
    return sdf.d3.gyroid(w=2.0).shell(0.4) & sphere(3)


def _build_schwartz_p():
    return sdf.d3.schwartz_p(1) & box(2)


def _build_diamond():
    return sdf.d3.diamond(1) & box(2)


# -- Arithmetic Operations --

def _build_addition():
    return sphere().addition(box(1.5))


def _build_morph():
    return sphere().morph(box(), plane(), 2.0)


# -- Mirror Operations --

def _build_mirror():
    return sphere(1, center=(2, 0, 0)).mirror((1, 0, 0))


def _build_mirror_copy():
    return sphere(1, center=(2, 0, 0)).mirror_copy((1, 0, 0))


# -- Additional Transformations --

def _build_chamfer():
    return box(1.5).chamfer(sphere(), 0.3)


def _build_twist_between():
    return box().twist_between((0, 0, -1), (0, 0, 1), ease.in_out_quad)


def _build_stretch():
    return sphere().stretch((0, 0, -1), (0, 0, 1))


def _build_shear():
    return box().shear((0, 0, -1), (0, 0, 1), (1, 0, 0))


def _build_skin():
    return sphere().skin(0.1)


def _build_shell_sided():
    return sphere().shell_sided(0.1, side="inner") & plane(-Z)


# -- Fillets & Chamfers --

def _build_fillet():
    return box((2, 2, 2)) & sdf.d3.slab(z1=0.5).k(0.4)


def _build_chamfer_edges():
    return box((2, 2, 2)).chamfer(sdf.d3.slab(z1=0.5), 0.4)


# -- Capsule Chain and Bezier --

def _build_capsule_chain():
    return sdf.d3.capsule_chain(
        [(0, 0, 0), (1, 1, 0), (2, 0, 0), (3, 1, 0)], radius=0.1,
    )


def _build_bezier():
    return sdf.d3.bezier(
        (-2, 0, 0), (-1, 2, 0), (1, -2, 0), (2, 0, 0),
        radius=ease.linear.between(0.4, 0.1),
        steps=30,
    )


# -- Thread and Screw --

def _build_thread():
    thread = sdf.d3.Thread(pitch=3, diameter=10, offset=0.8)
    return thread & sdf.d3.slab(z0=0, z1=20)


def _build_screw():
    return sdf.d3.Screw(length=30, pitch=3, diameter=10, offset=0.5)


def _build_pieslice():
    return sphere() & sdf.d3.pieslice(pi / 2, centered=True)


# -- Compositions (from showcase) --

def _build_figure():
    arm = capped_cylinder((0, 0, 0), (2, 0, 0), 0.3)
    body = sphere(1)
    half = union(body, arm, k=0.2)
    mirrored = half.mirror_copy((1, 0, 0))
    hat = capped_cylinder((0, 0, 0.8), (0, 0, 1.5), 0.6)
    return union(mirrored, hat, k=0.15)


# CSG body used by dilate, erode, elongate, slice
_CSG_BODY = """\
f = sphere(1) & box(1.5)
c = cylinder(0.5)
f -= c.orient(X) | c.orient(Y) | c.orient(Z)"""

# Boolean body used by smooth_* and boolean_*
_BOOL_BODY = """\
a = box((3, 3, 0.5))
b = sphere()"""

SHAPES = [
    # 3D Primitives
    Shape("sphere", _script("f = sphere(1)"), _build_sphere),
    Shape("box", _script("f = box(1)"), _build_box),
    Shape("rounded_box", _script("f = rounded_box((1, 2, 3), 0.25)"), _build_rounded_box),
    Shape("wireframe_box", _script("f = wireframe_box((1, 2, 3), 0.05)"), _build_wireframe_box),
    Shape("torus", _script("f = torus(1, 0.25)"), _build_torus),
    Shape("capsule", _script("f = capsule(-Z, Z, 0.5)"), _build_capsule),
    Shape("capped_cylinder", _script("f = capped_cylinder(-Z, Z, 0.5)"), _build_capped_cylinder),
    Shape("rounded_cylinder", _script("f = rounded_cylinder(0.5, 0.1, 2)"), _build_rounded_cylinder),
    Shape("capped_cone", _script("f = capped_cone(-Z, Z, 1, 0.5)"), _build_capped_cone),
    Shape("rounded_cone", _script("f = rounded_cone(0.75, 0.25, 2)"), _build_rounded_cone),
    Shape("ellipsoid", _script("f = ellipsoid((1, 2, 3))"), _build_ellipsoid),
    Shape("pyramid", _script("f = pyramid(1)"), _build_pyramid),
    # Platonic Solids
    Shape("tetrahedron", _script("f = tetrahedron(1)"), _build_tetrahedron),
    Shape("octahedron", _script("f = octahedron(1)"), _build_octahedron),
    Shape("dodecahedron", _script("f = dodecahedron(1)"), _build_dodecahedron),
    Shape("icosahedron", _script("f = icosahedron(1)"), _build_icosahedron),
    # Infinite 3D Primitives (bounded)
    Shape("plane", _script("f = sphere() & plane()"), _build_plane),
    Shape("slab", _script("f = sphere() & slab(z0=-0.5, z1=0.5, x0=0)"), _build_slab),
    Shape("cylinder_infinite", _script("f = sphere() - cylinder(0.5)"), _build_cylinder_infinite),
    # Positioning
    Shape("translate", _script("f = sphere().translate((0, 0, 2))"), _build_translate),
    Shape("scale", _script("f = sphere().scale(2)"), _build_scale),
    Shape("rotate", _script("f = capped_cylinder(-Z, Z, 0.5).rotate(pi / 4, X)"), _build_rotate),
    Shape("orient", _script("""\
        c = capped_cylinder(-Z, Z, 0.25)
        f = c.orient(X) | c.orient(Y) | c.orient(Z)"""), _build_orient),
    # Boolean Operations
    Shape("boolean_union", _script(f"{_BOOL_BODY}\nf = a | b"), _build_boolean_union),
    Shape("boolean_difference", _script(f"{_BOOL_BODY}\nf = a - b"), _build_boolean_difference),
    Shape("boolean_intersection", _script(f"{_BOOL_BODY}\nf = a & b"), _build_boolean_intersection),
    Shape("csg", _script(f"{_CSG_BODY}"), _build_csg),
    Shape("smooth_union", _script(f"{_BOOL_BODY}\nf = a | b.k(0.25)"), _build_smooth_union),
    Shape("smooth_difference", _script(f"{_BOOL_BODY}\nf = a - b.k(0.25)"), _build_smooth_difference),
    Shape("smooth_intersection", _script(f"{_BOOL_BODY}\nf = a & b.k(0.25)"), _build_smooth_intersection),
    # Repetition
    Shape("repeat", _script("f = sphere().repeat(3, (1, 1, 0))"), _build_repeat),
    Shape("circular_array", _script("f = capped_cylinder(-Z, Z, 0.5).circular_array(8, 4)"), _build_circular_array),
    # Miscellaneous
    Shape("blend", _script("f = sphere().blend(box())"), _build_blend),
    Shape("dilate", _script(f"{_CSG_BODY}\nf = f.dilate(0.1)"), _build_dilate),
    Shape("erode", _script(f"{_CSG_BODY}\nf = f.erode(0.1)"), _build_erode),
    Shape("shell", _script("f = sphere().shell(0.05) & plane(-Z)"), _build_shell),
    Shape("elongate", _script(f"{_CSG_BODY}\nf = f.elongate((0.25, 0.5, 0.75))"), _build_elongate),
    Shape("twist", _script("f = box().twist(pi / 2)"), _build_twist),
    Shape("bend", _script("f = box().bend(1)"), _build_bend),
    Shape("bend_linear", _script(
        "f = capsule(-Z * 2, Z * 2, 0.25).bend_linear(-Z, Z, X, ease.in_out_quad)",
        ease=True), _build_bend_linear),
    Shape("bend_radial", _script(
        "f = box((5, 5, 0.25)).bend_radial(1, 2, -1, ease.in_out_quad)",
        ease=True), _build_bend_radial),
    Shape("transition_linear", _script(
        "f = box().transition_linear(sphere(), e=ease.in_out_quad)",
        ease=True), _build_transition_linear),
    Shape("transition_radial", _script(
        "f = box().transition_radial(sphere(), e=ease.in_out_quad)",
        ease=True), _build_transition_radial),
    # 2D to 3D Operations
    Shape("extrude", _script("f = hexagon(1).extrude(1)"), _build_extrude),
    Shape("extrude_to", _script(
        "f = rectangle(2).extrude_to(circle(1), 2, ease.in_out_quad)",
        ease=True), _build_extrude_to),
    Shape("revolve", _script("f = hexagon(1).revolve(3)"), _build_revolve),
    Shape("rounded_extrude", _script("f = circle(1).rounded_extrude(2, radius=0.2)"), _build_rounded_extrude),
    Shape("taper_extrude", _script("f = circle(1).taper_extrude(2, slope=-0.5)"), _build_taper_extrude),
    Shape("scale_extrude", _script("f = circle(1).scale_extrude(2, top=0.5, bottom=1.0)"), _build_scale_extrude),
    # 3D to 2D Operations
    Shape("slice", _script(f"""\
        {_CSG_BODY}
        f = f.translate((0, 0, 0.55)).slice().extrude(0.1)"""), _build_slice),
    # TPMS Lattice Primitives
    Shape("gyroid", _script("f = gyroid(w=2.0).shell(0.4) & sphere(3)"), _build_gyroid),
    Shape("schwartz_p", _script("f = schwartz_p(1) & box(2)"), _build_schwartz_p),
    Shape("diamond", _script("f = diamond(1) & box(2)"), _build_diamond),
    # Arithmetic Operations
    Shape("addition", _script("f = sphere().addition(box(1.5))"), _build_addition,
          generate_kwargs={"bounds": ((-2, -2, -2), (2, 2, 2))}),
    Shape("morph", _script("f = sphere().morph(box(), plane(), 2.0)"), _build_morph),
    # Mirror Operations
    Shape("mirror", _script("f = sphere(1, center=(2, 0, 0)).mirror((1, 0, 0))"), _build_mirror),
    Shape("mirror_copy", _script("f = sphere(1, center=(2, 0, 0)).mirror_copy((1, 0, 0))"), _build_mirror_copy),
    # Additional Transformations
    Shape("chamfer", _script("f = box(1.5).chamfer(sphere(), 0.3)"), _build_chamfer),
    Shape("twist_between", _script(
        "f = box().twist_between((0, 0, -1), (0, 0, 1), ease.in_out_quad)",
        ease=True), _build_twist_between),
    Shape("stretch", _script("f = sphere().stretch((0, 0, -1), (0, 0, 1))"), _build_stretch),
    Shape("shear", _script("f = box().shear((0, 0, -1), (0, 0, 1), (1, 0, 0))"), _build_shear),
    Shape("skin", _script("f = sphere().skin(0.1)"), _build_skin),
    Shape("shell_sided", _script(
        "f = sphere().shell_sided(0.1, side='inner') & plane(-Z)"), _build_shell_sided),
    # Fillets & Chamfers
    Shape("fillet", _script("""\
        f = box((2, 2, 2)) & slab(z1=0.5).k(0.4)"""), _build_fillet),
    Shape("chamfer_edges", _script("""\
        f = box((2, 2, 2)).chamfer(slab(z1=0.5), 0.4)"""), _build_chamfer_edges),
    # Capsule Chain and Bezier
    Shape("capsule_chain", _script("""\
        f = capsule_chain(
            [(0, 0, 0), (1, 1, 0), (2, 0, 0), (3, 1, 0)],
            radius=0.1,
        )"""), _build_capsule_chain),
    Shape("bezier", _script("""\
        f = bezier(
            (-2, 0, 0), (-1, 2, 0), (1, -2, 0), (2, 0, 0),
            radius=ease.linear.between(0.4, 0.1),
            steps=30,
        )""", ease=True), _build_bezier),
    # Thread and Screw
    Shape("thread", _script("""\
        thread = Thread(pitch=3, diameter=10, offset=0.8)
        f = thread & slab(z0=0, z1=20)"""), _build_thread),
    Shape("screw", _script("f = Screw(length=30, pitch=3, diameter=10, offset=0.5)"), _build_screw),
    Shape("pieslice", _script("f = sphere() & pieslice(pi / 2, centered=True)"), _build_pieslice),
    # Compositions
    Shape("figure", _script("""\
        arm = capped_cylinder((0, 0, 0), (2, 0, 0), 0.3)
        body = sphere(1)
        half = union(body, arm, k=0.2)
        mirrored = half.mirror_copy((1, 0, 0))
        hat = capped_cylinder((0, 0, 0.8), (0, 0, 1.5), 0.6)
        f = union(mirrored, hat, k=0.15)"""), _build_figure),
]


# ---------------------------------------------------------------------------
# Mesh subsampling for viewer
# ---------------------------------------------------------------------------

def _subsample_mesh(verts, faces, max_faces=MAX_VIEWER_FACES):
    """Decimate mesh to at most max_faces using pyvista, preserving surface continuity."""
    if len(faces) <= max_faces:
        return verts.astype(np.float32), faces.astype(np.int32)

    import pyvista as pv

    n = len(faces)
    pv_faces = np.column_stack([np.full(n, 3), faces]).ravel()
    mesh = pv.PolyData(verts, pv_faces)
    target_reduction = 1.0 - (max_faces / n)
    decimated = mesh.decimate(target_reduction)

    out_verts = np.asarray(decimated.points, dtype=np.float32)
    # pyvista faces: flat [3, v0, v1, v2, 3, v0, v1, v2, ...]
    out_faces = decimated.faces.reshape(-1, 4)[:, 1:].astype(np.int32)
    return out_verts, out_faces


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _render_shape(verts, faces, path, width=400, height=300):
    """Render mesh to PNG using pyvista off-screen."""
    import pyvista as pv

    n = len(faces)
    pv_faces = np.column_stack([np.full(n, 3), faces]).ravel()
    mesh = pv.PolyData(verts, pv_faces)
    mesh.compute_normals(inplace=True)

    pl = pv.Plotter(off_screen=True, window_size=[width, height])
    pl.add_mesh(mesh, color="steelblue", smooth_shading=True)
    pl.camera_position = "iso"
    pl.screenshot(str(path))
    pl.close()


def _load_image(path):
    """Load a PNG as a uint8 numpy array via PIL."""
    from PIL import Image
    return np.array(Image.open(path).convert("RGB"))


def _image_mae(a, b):
    """Mean absolute error between two uint8 image arrays."""
    return np.mean(np.abs(a.astype(float) - b.astype(float)))


# ---------------------------------------------------------------------------
# Gallery collection (session-scoped)
# ---------------------------------------------------------------------------

@dataclass
class GalleryEntry:
    name: str
    code: str
    actual_path: Path
    reference_path: Path
    passed: bool
    mae: float
    viewer_verts: Optional[np.ndarray] = field(default=None, repr=False)
    viewer_faces: Optional[np.ndarray] = field(default=None, repr=False)


_gallery_entries: list[GalleryEntry] = []


def _write_gallery(entries, output_path):
    """Write a self-contained HTML gallery with base64-inlined images and Three.js viewers."""
    def img_tag(path, label):
        if path.exists():
            b64 = base64.b64encode(path.read_bytes()).decode()
            return (
                f'<div class="img-cell">'
                f'<img src="data:image/png;base64,{b64}">'
                f'<span class="img-label">{label}</span></div>'
            )
        return f'<div class="img-cell"><span class="img-label">{label}: not found</span></div>'

    def mesh_json(entry):
        if entry.viewer_verts is None or entry.viewer_faces is None:
            return ""
        v_b64 = base64.b64encode(entry.viewer_verts.tobytes()).decode()
        f_b64 = base64.b64encode(entry.viewer_faces.tobytes()).decode()
        return (
            f'<script type="application/json" id="mesh-{entry.name}">'
            f'{{"v":"{v_b64}","f":"{f_b64}"}}</script>'
        )

    cards = []
    for e in entries:
        status = "PASS" if e.passed else "FAIL"
        badge_cls = "badge-pass" if e.passed else "badge-fail"
        escaped_code = html_mod.escape(e.code)
        cards.append(f"""
    <div class="shape-card">
      <h3>{e.name} <span class="badge {badge_cls}">{status}</span>
        <span class="mae">MAE: {e.mae:.2f}</span></h3>
      <div class="card-grid">
        <div class="code-panel">
          <button class="copy-btn" data-target="code-{e.name}">Copy</button>
          <pre><code id="code-{e.name}">{escaped_code}</code></pre>
        </div>
        <div class="images-panel">
          {img_tag(e.actual_path, "Rendered")}
          {img_tag(e.reference_path, "Reference")}
        </div>
        <div class="viewer-panel" data-shape="{e.name}">
          <div class="viewer-placeholder">Loading 3D viewer...</div>
        </div>
      </div>
      {mesh_json(e)}
    </div>""")

    passed = len([e for e in entries if e.passed])

    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>SDF Visual Regression Gallery</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      max-width: 1600px; margin: 0 auto; padding: 20px;
      background: #fafafa; color: #333;
    }}
    h1 {{ margin-bottom: 4px; }}
    .summary {{ color: #666; margin-bottom: 24px; }}
    .shape-card {{
      border: 1px solid #ddd; margin: 16px 0; padding: 16px;
      border-radius: 8px; background: #fff;
    }}
    .shape-card h3 {{
      margin: 0 0 12px 0; display: flex; align-items: center; gap: 8px;
    }}
    .badge {{
      font-size: 12px; padding: 2px 8px; border-radius: 4px;
      font-weight: 600; text-transform: uppercase;
    }}
    .badge-pass {{ background: #d4edda; color: #155724; }}
    .badge-fail {{ background: #f8d7da; color: #721c24; }}
    .mae {{ font-size: 12px; color: #888; font-weight: normal; margin-left: auto; }}
    .card-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr 2fr;
      gap: 12px;
      min-height: 300px;
    }}
    .code-panel {{
      position: relative;
      overflow: hidden;
    }}
    .code-panel pre {{
      background: #f5f5f5; padding: 12px; border-radius: 4px;
      overflow: auto; margin: 0; height: 100%;
      font-size: 12px; line-height: 1.4;
    }}
    .copy-btn {{
      position: absolute; top: 4px; right: 4px; z-index: 1;
      background: #fff; border: 1px solid #ccc; border-radius: 4px;
      padding: 2px 8px; font-size: 11px; cursor: pointer;
      opacity: 0; transition: opacity 0.15s;
    }}
    .code-panel:hover .copy-btn {{ opacity: 1; }}
    .copy-btn:hover {{ background: #e8e8e8; }}
    .images-panel {{
      display: flex; flex-direction: column; gap: 8px;
    }}
    .img-cell {{
      display: flex; flex-direction: column; align-items: center;
    }}
    .img-cell img {{
      max-width: 100%; border: 1px solid #eee; border-radius: 4px;
    }}
    .img-label {{
      font-size: 11px; color: #888; margin-top: 2px;
    }}
    .viewer-panel {{
      border: 1px solid #eee; border-radius: 4px;
      display: flex; align-items: center; justify-content: center;
      background: #f0f0f0; min-height: 300px; overflow: hidden;
    }}
    .viewer-panel canvas {{
      display: block; width: 100% !important; height: 100% !important;
    }}
    .viewer-placeholder {{
      color: #aaa; font-size: 13px;
    }}
    @media (max-width: 900px) {{
      .card-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <h1>SDF Visual Regression Gallery</h1>
  <p class="summary">{passed}/{len(entries)} passed</p>
  {"".join(cards)}

  <script type="module">
    import * as THREE from "https://esm.sh/three@0.170.0";
    import {{ OrbitControls }} from "https://esm.sh/three@0.170.0/addons/controls/OrbitControls.js";

    function b64ToFloat32(b64) {{
      const bin = atob(b64);
      const buf = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) buf[i] = bin.charCodeAt(i);
      return new Float32Array(buf.buffer);
    }}

    function b64ToInt32(b64) {{
      const bin = atob(b64);
      const buf = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) buf[i] = bin.charCodeAt(i);
      return new Int32Array(buf.buffer);
    }}

    // Track live viewer state per container so we can tear down on scroll-out
    const viewers = new Map();

    function initViewer(container) {{
      if (viewers.has(container)) return; // already live

      const name = container.dataset.shape;
      const el = document.getElementById("mesh-" + name);
      if (!el) {{
        container.innerHTML = '<div class="viewer-placeholder">No mesh data</div>';
        return;
      }}

      const data = JSON.parse(el.textContent);
      const verts = b64ToFloat32(data.v);
      const faces = b64ToInt32(data.f);

      container.innerHTML = "";

      const rect = container.getBoundingClientRect();
      const width = rect.width || 400;
      const height = rect.height || 300;

      const scene = new THREE.Scene();
      scene.background = new THREE.Color(0xf0f0f0);

      const camera = new THREE.PerspectiveCamera(45, width / height, 0.01, 1000);
      const renderer = new THREE.WebGLRenderer({{ antialias: true }});
      renderer.setSize(width, height);
      renderer.setPixelRatio(window.devicePixelRatio);
      container.appendChild(renderer.domElement);

      const controls = new OrbitControls(camera, renderer.domElement);
      controls.enableDamping = true;
      controls.dampingFactor = 0.1;

      // Lighting — matches sdf/viewer.py
      scene.add(new THREE.AmbientLight(0xffffff, 0.5));
      const d1 = new THREE.DirectionalLight(0xffffff, 0.8);
      d1.position.set(1, 2, 3);
      scene.add(d1);
      const d2 = new THREE.DirectionalLight(0xffffff, 0.3);
      d2.position.set(-2, -1, -1);
      scene.add(d2);

      const geom = new THREE.BufferGeometry();
      geom.setAttribute("position", new THREE.Float32BufferAttribute(verts, 3));
      geom.setIndex(Array.from(faces));
      geom.computeVertexNormals();

      const mat = new THREE.MeshStandardMaterial({{
        color: new THREE.Color("#4682b4"),
        metalness: 0.1, roughness: 0.6, side: THREE.DoubleSide,
      }});
      const mesh = new THREE.Mesh(geom, mat);
      scene.add(mesh);

      // Fit camera
      const box = new THREE.Box3().setFromObject(mesh);
      const center = box.getCenter(new THREE.Vector3());
      const size = box.getSize(new THREE.Vector3()).length();
      camera.position.copy(center);
      camera.position.x += size * 0.8;
      camera.position.y += size * 0.6;
      camera.position.z += size * 0.8;
      camera.lookAt(center);
      controls.target.copy(center);
      camera.near = size * 0.001;
      camera.far = size * 100;
      camera.updateProjectionMatrix();

      let animId;
      function animate() {{
        animId = requestAnimationFrame(animate);
        controls.update();
        renderer.render(scene, camera);
      }}
      animate();

      const ro = new ResizeObserver(() => {{
        const r = container.getBoundingClientRect();
        if (r.width > 0 && r.height > 0) {{
          renderer.setSize(r.width, r.height);
          camera.aspect = r.width / r.height;
          camera.updateProjectionMatrix();
        }}
      }});
      ro.observe(container);

      // Store everything needed for teardown
      viewers.set(container, {{ renderer, controls, scene, geom, mat, mesh, animId, ro }});
    }}

    function destroyViewer(container) {{
      const v = viewers.get(container);
      if (!v) return;
      cancelAnimationFrame(v.animId);
      v.ro.disconnect();
      v.controls.dispose();
      v.geom.dispose();
      v.mat.dispose();
      v.renderer.dispose();
      v.renderer.forceContextLoss();
      viewers.delete(container);
      container.innerHTML = '<div class="viewer-placeholder">Scroll to load 3D viewer</div>';
    }}

    // Observe both enter and exit — init on enter, dispose on exit
    const observer = new IntersectionObserver((entries) => {{
      entries.forEach(entry => {{
        if (entry.isIntersecting) {{
          initViewer(entry.target);
        }} else {{
          destroyViewer(entry.target);
        }}
      }});
    }}, {{ rootMargin: "200px" }});

    document.querySelectorAll(".viewer-panel").forEach(el => observer.observe(el));

    // Copy buttons
    document.querySelectorAll(".copy-btn").forEach(btn => {{
      btn.addEventListener("click", () => {{
        const code = document.getElementById(btn.dataset.target).textContent;
        navigator.clipboard.writeText(code).then(() => {{
          const orig = btn.textContent;
          btn.textContent = "Copied!";
          setTimeout(() => btn.textContent = orig, 1500);
        }});
      }});
    }});
  </script>
</body>
</html>"""
    output_path.write_text(html)


# ---------------------------------------------------------------------------
# Pytest fixtures and hooks
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def gallery_output_dir(tmp_path_factory):
    """Session-scoped temp dir for rendered images."""
    return tmp_path_factory.mktemp("visual_renders")


@pytest.fixture(autouse=True, scope="session")
def _write_gallery_on_teardown():
    """Write the HTML gallery after all visual tests complete."""
    yield
    if _gallery_entries:
        _write_gallery(_gallery_entries, GALLERY_PATH)


# ---------------------------------------------------------------------------
# Parametrized visual test
# ---------------------------------------------------------------------------

@pytest.mark.visual
@pytest.mark.parametrize("shape", SHAPES, ids=[s.name for s in SHAPES])
def test_visual(shape, request, gallery_output_dir):
    update_refs = request.config.getoption("--update-references")

    # 1. Build SDF and generate mesh
    sdf_obj = shape.builder()
    result = _generate_mesh(sdf_obj, samples=SAMPLES, **shape.generate_kwargs)
    assert result is not None, f"No mesh generated for {shape.name}"
    verts, faces = result

    # 2. Subsample mesh for viewer
    viewer_verts, viewer_faces = _subsample_mesh(verts, faces)

    # 3. Render to temp dir
    actual_path = gallery_output_dir / f"{shape.name}.png"
    _render_shape(verts, faces, actual_path)

    # 4. Reference path
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    ref_path = REFERENCE_DIR / f"{shape.name}.png"

    if update_refs:
        # Copy rendered image as new reference
        shutil.copy2(actual_path, ref_path)
        _gallery_entries.append(GalleryEntry(
            name=shape.name, code=shape.code,
            actual_path=actual_path, reference_path=ref_path,
            passed=True, mae=0.0,
            viewer_verts=viewer_verts, viewer_faces=viewer_faces,
        ))
        pytest.skip("Reference image updated")
        return

    # 5. Compare against reference
    assert ref_path.exists(), (
        f"Reference image not found: {ref_path}\n"
        f"Run: pytest -m visual --update-references"
    )

    actual_img = _load_image(actual_path)
    ref_img = _load_image(ref_path)
    mae = _image_mae(actual_img, ref_img)

    passed = mae <= MAE_THRESHOLD
    _gallery_entries.append(GalleryEntry(
        name=shape.name, code=shape.code,
        actual_path=actual_path, reference_path=ref_path,
        passed=passed, mae=mae,
        viewer_verts=viewer_verts, viewer_faces=viewer_faces,
    ))

    assert passed, (
        f"Visual regression for {shape.name}: MAE={mae:.2f} > {MAE_THRESHOLD}\n"
        f"Actual: {actual_path}\n"
        f"Reference: {ref_path}"
    )
