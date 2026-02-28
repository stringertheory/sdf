"""Interactive 3D preview using pyvista or anywidget+Three.js."""

import numpy as np

# Default sample count for .show() — much lower than generate()'s 2**22
# to give a fast preview. Users can override with samples= or step=.
SHOW_SAMPLES = 2 ** 18


def _generate_mesh(sdf_or_points, **generate_kwargs):
    """Generate or accept mesh data, returning (unique_verts, faces) or None."""
    if hasattr(sdf_or_points, 'generate'):
        generate_kwargs.setdefault('verbose', False)
        generate_kwargs.setdefault('samples', SHOW_SAMPLES)
        points = sdf_or_points.generate(**generate_kwargs)
    else:
        points = np.asarray(sdf_or_points)

    points = np.asarray(points, dtype='float64').reshape(-1, 3)
    n_triangles = len(points) // 3

    if n_triangles == 0:
        return None

    unique_verts, inverse = np.unique(points, axis=0, return_inverse=True)
    faces = inverse.reshape(-1, 3)
    return unique_verts, faces


def show(sdf_or_points, **generate_kwargs):
    """Show an SDF or mesh in an interactive 3D viewer.

    Returns a draggable Three.js widget (anywidget) if anywidget is installed,
    otherwise falls back to a static pyvista screenshot.

    Works in marimo notebooks, Jupyter, and scripts.

    Args:
        sdf_or_points: Either an SDF3 object or an Nx3 array of triangle vertices.
        **generate_kwargs: Keyword arguments passed to generate() if SDF3 is given.
            Defaults to samples=2**18 for a fast preview.

    Returns:
        An interactive MeshViewer widget, a static image array, or None.
    """
    result = _generate_mesh(sdf_or_points, **generate_kwargs)
    if result is None:
        print('No triangles to display.')
        return None

    verts, faces = result

    try:
        return MeshViewer(verts, faces)
    except ImportError:
        pass

    # Fallback: static pyvista screenshot
    try:
        import pyvista as pv
    except ImportError:
        raise ImportError(
            ".show() requires anywidget or pyvista. Install one with:\n"
            "  pip install anywidget   (interactive, recommended)\n"
            "  pip install pyvista     (static screenshot)"
        )

    n_triangles = len(faces)
    pv_faces = np.column_stack([np.full(n_triangles, 3), faces]).ravel()
    mesh = pv.PolyData(verts, pv_faces)
    mesh.compute_normals(inplace=True)

    pl = pv.Plotter(off_screen=True)
    pl.add_mesh(mesh, color='steelblue', smooth_shading=True)
    return pl.screenshot()


# ---------------------------------------------------------------------------
# Three.js interactive viewer via anywidget
# ---------------------------------------------------------------------------

_ESM = """
import * as THREE from "https://esm.sh/three@0.170.0";
import { OrbitControls } from "https://esm.sh/three@0.170.0/addons/controls/OrbitControls.js";

function render({ model, el }) {
  const width = model.get("width");
  const height = model.get("height");
  const color = model.get("color");

  // Scene setup
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xf0f0f0);

  const camera = new THREE.PerspectiveCamera(45, width / height, 0.01, 1000);
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(width, height);
  renderer.setPixelRatio(window.devicePixelRatio);
  el.appendChild(renderer.domElement);

  // Orbit controls
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.1;

  // Lighting
  const ambient = new THREE.AmbientLight(0xffffff, 0.5);
  scene.add(ambient);
  const dirLight1 = new THREE.DirectionalLight(0xffffff, 0.8);
  dirLight1.position.set(1, 2, 3);
  scene.add(dirLight1);
  const dirLight2 = new THREE.DirectionalLight(0xffffff, 0.3);
  dirLight2.position.set(-2, -1, -1);
  scene.add(dirLight2);

  function buildMesh() {
    const verts = model.get("vertices_flat");
    const idx = model.get("faces_flat");
    if (!verts || !idx || verts.length === 0) return null;

    const geom = new THREE.BufferGeometry();
    geom.setAttribute("position", new THREE.Float32BufferAttribute(verts, 3));
    geom.setIndex(Array.from(idx));
    geom.computeVertexNormals();

    const mat = new THREE.MeshStandardMaterial({
      color: new THREE.Color(color),
      metalness: 0.1,
      roughness: 0.6,
      side: THREE.DoubleSide,
    });
    return new THREE.Mesh(geom, mat);
  }

  const mesh = buildMesh();
  if (mesh) {
    scene.add(mesh);

    // Fit camera to object
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
  }

  // Render loop
  let animId;
  function animate() {
    animId = requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
  }
  animate();

  // Cleanup on widget removal
  return () => {
    cancelAnimationFrame(animId);
    renderer.dispose();
    controls.dispose();
  };
}

export default { render };
"""

_CSS = """
.mesh-viewer canvas {
  display: block;
  border-radius: 4px;
}
"""


def MeshViewer(vertices, faces, color='#4682b4', width=700, height=500):
    """Create an interactive 3D mesh viewer widget.

    Args:
        vertices: Nx3 array of vertex positions.
        faces: Mx3 array of triangle face indices.
        color: CSS color string for the mesh.
        width: Widget width in pixels.
        height: Widget height in pixels.

    Returns:
        An anywidget instance for interactive 3D viewing.

    Raises:
        ImportError: If anywidget is not installed.
    """
    import anywidget
    import traitlets

    _v = np.asarray(vertices, dtype='float32').ravel().tolist()
    _f = np.asarray(faces, dtype='int32').ravel().tolist()
    _c = str(color)
    _w = int(width)
    _h = int(height)

    class _MeshViewer(anywidget.AnyWidget):
        _esm = _ESM
        _css = _CSS
        vertices_flat = traitlets.List(_v).tag(sync=True)
        faces_flat = traitlets.List(_f).tag(sync=True)
        color = traitlets.Unicode(_c).tag(sync=True)
        width = traitlets.Int(_w).tag(sync=True)
        height = traitlets.Int(_h).tag(sync=True)

    return _MeshViewer()
