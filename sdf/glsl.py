"""Real-time GLSL sphere-tracing viewer via anywidget + Three.js."""

_VERT = """
void main() {
    gl_Position = vec4(position.xy, 0.0, 1.0);
}
"""

_FRAG = """
precision highp float;

uniform vec3 u_camPos;
uniform vec3 u_camRight;
uniform vec3 u_camUp;
uniform vec3 u_camForward;
uniform float u_fov;
uniform vec2 u_resolution;

uniform float u_width;
uniform float u_height;
uniform float u_depth;
uniform float u_rows;
uniform float u_cols;
uniform float u_wall_thickness;
uniform float u_wall_radius;
uniform float u_bottom_radius;
uniform float u_top_fillet;
uniform float u_divider_thickness;
uniform float u_row_divider_depth;
uniform float u_col_divider_depth;
uniform float u_divider_fillet;
uniform float u_lid_thickness;
uniform float u_lid_depth;
uniform float u_lid_radius;
uniform float u_show_lid;

// Three.js Y-up -> SDF Z-up
vec3 toSDF(vec3 p) { return vec3(p.x, -p.z, p.y); }

float sdRoundedBox(vec3 p, vec3 size, float r) {
    vec3 q = abs(p) - size * 0.5 + r;
    return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0) - r;
}

float opSmoothIntersection(float d1, float d2, float k) {
    float h = clamp(0.5 - 0.5 * (d2 - d1) / k, 0.0, 1.0);
    return mix(d2, d1, h) + k * h * (1.0 - h);
}

float opShell(float d, float thickness) {
    return abs(d) - thickness * 0.5;
}

float sdColDividers(vec3 p) {
    float spacing = u_width / u_cols;
    if (mod(u_cols, 2.0) > 0.5) p.x -= spacing * 0.5;
    float idx = floor(p.x / spacing + 0.5);
    float d = 1e10;
    for (int n = -1; n <= 1; n++) {
        vec3 q = p;
        q.x -= spacing * (idx + float(n));
        q.z -= u_col_divider_depth * 0.5;
        d = min(d, sdRoundedBox(q, vec3(u_divider_thickness, 1000.0, u_col_divider_depth), u_divider_fillet));
    }
    return d;
}

float sdRowDividers(vec3 p) {
    float spacing = u_height / u_rows;
    if (mod(u_rows, 2.0) > 0.5) p.y -= spacing * 0.5;
    float idx = floor(p.y / spacing + 0.5);
    float d = 1e10;
    for (int n = -1; n <= 1; n++) {
        vec3 q = p;
        q.y -= spacing * (idx + float(n));
        q.z -= u_row_divider_depth * 0.5;
        d = min(d, sdRoundedBox(q, vec3(1000.0, u_divider_thickness, u_row_divider_depth), u_divider_fillet));
    }
    return d;
}

float sdDividers(vec3 p) {
    return min(sdColDividers(p), sdRowDividers(p));
}

float sdOuterForm(vec3 p) {
    float d = sdRoundedBox(p, vec3(u_width - u_wall_thickness, u_height - u_wall_thickness, 1000.0), u_wall_radius);
    float slabZ0 = u_wall_thickness * 0.5 - p.z;
    d = opSmoothIntersection(d, slabZ0, u_bottom_radius);
    return d;
}

float sdBoxSDF(vec3 p) {
    float outer = sdOuterForm(p);
    float divs = max(sdDividers(p), outer);
    float shelled = opShell(outer, u_wall_thickness);
    float slabZ1 = p.z - u_depth;
    shelled = opSmoothIntersection(shelled, slabZ1, u_top_fillet);
    return min(shelled, divs);
}

float sdLidSDF(vec3 p) {
    float d = sdRoundedBox(p, vec3(u_width + u_wall_thickness, u_height + u_wall_thickness, 1000.0), u_wall_radius);
    float slabZ0 = u_wall_thickness * 0.5 - p.z;
    d = opSmoothIntersection(d, slabZ0, u_lid_radius);
    d = opShell(d, u_lid_thickness);
    float slabZ1 = p.z - u_lid_depth;
    d = opSmoothIntersection(d, slabZ1, u_top_fillet);
    return d;
}

float sdScene(vec3 p) {
    if (u_show_lid > 0.5) return sdLidSDF(p);
    return sdBoxSDF(p);
}

vec3 calcNormal(vec3 p) {
    const float h = 0.001;
    return normalize(vec3(
        sdScene(p + vec3(h,0,0)) - sdScene(p - vec3(h,0,0)),
        sdScene(p + vec3(0,h,0)) - sdScene(p - vec3(0,h,0)),
        sdScene(p + vec3(0,0,h)) - sdScene(p - vec3(0,0,h))
    ));
}

float calcAO(vec3 p, vec3 n) {
    float occ = 0.0;
    float sca = 1.0;
    for (int i = 0; i < 5; i++) {
        float h = 0.01 + 0.12 * float(i);
        float d = sdScene(p + h * n);
        occ += (h - d) * sca;
        sca *= 0.95;
    }
    return clamp(1.0 - 3.0 * occ, 0.0, 1.0);
}

void main() {
    vec2 uv = (gl_FragCoord.xy / u_resolution - 0.5) * 2.0;
    uv.x *= u_resolution.x / u_resolution.y;

    float fovScale = tan(u_fov * 0.5);
    vec3 rd3 = normalize(u_camForward + uv.x * fovScale * u_camRight + uv.y * fovScale * u_camUp);
    vec3 ro = toSDF(u_camPos);
    vec3 rd = toSDF(rd3);

    float t = 0.0;
    float d;
    for (int i = 0; i < 128; i++) {
        d = sdScene(ro + rd * t);
        if (abs(d) < 0.001) break;
        t += d;
        if (t > 200.0) break;
    }

    vec3 col;
    if (t < 200.0 && abs(d) < 0.001) {
        vec3 p = ro + rd * t;
        vec3 n = calcNormal(p);
        float ao = calcAO(p, n);

        vec3 L1 = normalize(vec3(1.0, 1.0, 2.0));
        vec3 L2 = normalize(vec3(-2.0, -1.0, 0.5));
        float diff = max(dot(n, L1), 0.0);
        float diff2 = max(dot(n, L2), 0.0);

        vec3 V = normalize(-rd);
        vec3 H = normalize(L1 + V);
        float spec = pow(max(dot(n, H), 0.0), 32.0);

        vec3 baseColor = vec3(0.275, 0.510, 0.706);
        col = baseColor * (0.15 + 0.65 * diff + 0.2 * diff2) * ao + vec3(0.4) * spec * ao;
    } else {
        col = vec3(0.941);
    }

    col = pow(col, vec3(1.0 / 2.2));
    gl_FragColor = vec4(col, 1.0);
}
"""

_ESM = (
    'const VERT = `' + _VERT + '`;\n'
    'const FRAG = `' + _FRAG + '`;\n'
    + r"""
import * as THREE from "https://esm.sh/three@0.170.0";
import { OrbitControls } from "https://esm.sh/three@0.170.0/addons/controls/OrbitControls.js";

const UNIFORM_NAMES = [
    "width", "height", "depth", "rows", "cols",
    "wall_thickness", "wall_radius", "bottom_radius", "top_fillet",
    "divider_thickness", "row_divider_depth", "col_divider_depth", "divider_fillet",
    "lid_thickness", "lid_depth", "lid_radius", "show_lid",
];

function render({ model, el }) {
    const w = model.get("width");
    const h = model.get("height");
    const u = model.get("uniforms");
    const dpr = window.devicePixelRatio || 1;

    const renderer = new THREE.WebGLRenderer({ antialias: false });
    renderer.setSize(w, h);
    renderer.setPixelRatio(dpr);
    el.appendChild(renderer.domElement);

    const orthoCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);

    const perspCamera = new THREE.PerspectiveCamera(45, w / h, 0.1, 1000);
    perspCamera.position.set(15, 8, 12);

    const controls = new OrbitControls(perspCamera, renderer.domElement);
    controls.target.set(0, 1, 0);
    controls.enableDamping = true;
    controls.dampingFactor = 0.1;
    controls.update();

    const shaderUniforms = {
        u_resolution: { value: new THREE.Vector2(w * dpr, h * dpr) },
        u_camPos:     { value: new THREE.Vector3() },
        u_camRight:   { value: new THREE.Vector3() },
        u_camUp:      { value: new THREE.Vector3() },
        u_camForward: { value: new THREE.Vector3() },
        u_fov:        { value: perspCamera.fov * Math.PI / 180 },
    };
    for (const name of UNIFORM_NAMES) {
        shaderUniforms["u_" + name] = { value: u[name] ?? 0.0 };
    }

    const material = new THREE.ShaderMaterial({
        uniforms: shaderUniforms,
        vertexShader: VERT,
        fragmentShader: FRAG,
    });

    const scene = new THREE.Scene();
    scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), material));

    model.on("change:uniforms", () => {
        const newU = model.get("uniforms");
        for (const name of UNIFORM_NAMES) {
            const key = "u_" + name;
            if (shaderUniforms[key] && newU[name] !== undefined) {
                shaderUniforms[key].value = newU[name];
            }
        }
    });

    let animId;
    function animate() {
        animId = requestAnimationFrame(animate);
        controls.update();

        perspCamera.updateMatrixWorld();
        const e = perspCamera.matrixWorld.elements;
        shaderUniforms.u_camRight.value.set(e[0], e[1], e[2]);
        shaderUniforms.u_camUp.value.set(e[4], e[5], e[6]);
        shaderUniforms.u_camForward.value.set(-e[8], -e[9], -e[10]);
        shaderUniforms.u_camPos.value.copy(perspCamera.position);

        renderer.render(scene, orthoCamera);
    }
    animate();

    return () => {
        cancelAnimationFrame(animId);
        renderer.dispose();
        controls.dispose();
    };
}

export default { render };
""")

_CSS = """
.shader-viewer canvas {
    display: block;
    border-radius: 4px;
}
"""


def ShaderViewer(uniforms, width=700, height=500):
    """Create a real-time GLSL sphere-tracing viewer for the customizable box.

    Args:
        uniforms: Dict of {name: float} parameter values (e.g. width, height,
            depth, rows, cols, wall_thickness, ...).
        width: Widget width in CSS pixels.
        height: Widget height in CSS pixels.

    Returns:
        An anywidget instance. Update ``viewer.uniforms = {...}`` to change
        parameters in real time without recreating the widget.
    """
    import anywidget
    import traitlets

    _u = dict(uniforms)
    _w = int(width)
    _h = int(height)

    class _ShaderViewer(anywidget.AnyWidget):
        _esm = _ESM
        _css = _CSS
        uniforms = traitlets.Dict(_u).tag(sync=True)
        width = traitlets.Int(_w).tag(sync=True)
        height = traitlets.Int(_h).tag(sync=True)

    return _ShaderViewer()
