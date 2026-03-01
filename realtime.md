# Real-Time Rendering for Interactive Parametric SDF Design

This document surveys approaches for making this SDF library render fast enough
for real-time parametric design — where slider changes produce visual feedback
in <100ms (ideally <16ms). It is a reference for future implementation work.

**Current state**: Slider changes rebuild the SDF closure and re-mesh via
marching cubes (~1-2s). The `glsl.py` module provides a hand-written GLSL
sphere-tracer for one specific model (customizable box). The goal is to
generalize this.

---

## Table of Contents

1. [Path A: GLSL Codegen from Expression Tree](#path-a-glsl-codegen-from-expression-tree)
2. [Path B: Fidget (JIT-compiled CPU evaluation)](#path-b-fidget-jit-compiled-cpu-evaluation)
3. [Path C: Taichi (Python → GPU JIT)](#path-c-taichi-python--gpu-jit)
4. [Path D: JAX-accelerated Evaluation](#path-d-jax-accelerated-evaluation)
5. [Prerequisite: Expression Tree / IR](#prerequisite-expression-tree--ir)
6. [Reference Implementations](#reference-implementations)
7. [Comparison Matrix](#comparison-matrix)
8. [Recommended Sequence](#recommended-sequence)

---

## Path A: GLSL Codegen from Expression Tree

**Approach**: Compile the SDF expression tree into a GLSL fragment shader.
Run sphere tracing entirely on the GPU via WebGL in an anywidget. Parameter
changes update WebGL uniforms — no shader recompile, no Python round-trip.

**Latency**: <16ms per frame (GPU-bound). This is what `glsl.py` already
demonstrates for the box, but hand-written rather than compiled.

### How it would work

1. Each `@sdf3` primitive gets a `glsl_geo()` method returning a GLSL
   expression string and a list of function-definition fragments.
2. Each `@op3` operation gets a `glsl_compose()` method that wraps its
   children's GLSL expressions (e.g., `min(a, b)` for union).
3. A `compile_glsl(sdf)` function walks the expression tree top-down,
   collects all fragments, deduplicates them, and assembles a complete
   fragment shader with a `float sdScene(vec3 p)` entry point.
4. Parameters marked as dynamic become `uniform float` declarations.
   Static parameters are baked as GLSL literals.
5. The shader is injected into a `ShaderViewer` anywidget (similar to the
   existing one in `glsl.py` but with a generated shader instead of a
   hard-coded one).

### GLSL for each primitive/operation

The built-in primitives map directly to well-known GLSL SDFs
(see [Inigo Quilez's reference](https://iquilezles.org/articles/distfunctions/)):

| Python primitive | GLSL function | Notes |
|------------------|---------------|-------|
| `sphere(r)` | `length(p) - r` | Trivial |
| `box(size)` | Standard `sdBox` | Exact SDF |
| `rounded_box(size, r)` | `sdRoundedBox` | Exact SDF |
| `torus(r1, r2)` | `sdTorus` | Exact SDF |
| `capsule(a, b, r)` | `sdCapsule` | Exact SDF |
| `cylinder(r)` | `length(p.xy) - r` | Infinite cylinder |
| `capped_cylinder` | `sdCappedCylinder` | Exact SDF |
| `slab(z0, z1, ...)` | Plane half-space intersections | Compiles to `max()` chain |

Operations:

| Python operation | GLSL | Notes |
|------------------|------|-------|
| `translate(offset)` | `p -= offset` | Domain shift |
| `rotate(angle, axis)` | `p = mat3(...) * p` | Pre-computed rotation matrix baked as literal |
| `scale(factor)` | `p /= s; return d * min(s)` | Non-uniform scale needs care |
| `union(a, b)` | `min(a, b)` | Hard union |
| `union(a, b, k=k)` | `opSmoothUnion(a, b, k)` | Smooth union from dn.py formula |
| `intersection(a, b, k=k)` | `opSmoothIntersection(a, b, k)` | From dn.py formula |
| `difference(a, b, k=k)` | `opSmoothDifference(a, b, k)` | From dn.py formula |
| `shell(t)` | `abs(d) - t/2` | Trivial |
| `repeat(spacing)` | Domain folding via `round()` | Need ±1 neighbor check |
| `mirror(dir)` | Reflect `p` across plane | Straightforward |
| `twist(k)` | Rotate `p.xy` by `k * p.z` | Domain warp |
| `bend(k)` | Rotate `p.xy` by `k * p.x` | Domain warp |
| `elongate(size)` | Standard elongation formula | Well-known |

### What cannot be compiled

- **Custom lambdas**: `SDF3(lambda p: ...)` has no inspectable structure.
  Must raise `NotImplementedError`.
- **Operations using numpy-specific patterns**: `np.where`, fancy indexing,
  `np.einsum` — these don't have GLSL equivalents. Most built-in primitives
  use only basic math that translates cleanly; the exceptions would need
  case-by-case handling.
- **`morph`, `modulate_between`**: These reference easing functions that
  are Python callables. Would need GLSL easing equivalents.
- **Very deep trees**: Shader compilation time grows with tree depth. GPU
  drivers may refuse shaders beyond ~10,000 instructions. Practical limit
  is roughly 50-100 CSG nodes before shader length becomes a concern.

### Dynamic uniforms

For real-time slider interaction, certain values must be uniforms rather
than constants. Two possible APIs:

```python
# Option 1: Explicit Param wrapper
from sdf import Param
r = Param(1.0, name="radius")
s = sphere(r).translate(X)
s.show_realtime()  # compiles shader with uniform u_radius

# Option 2: Compile with named parameters
s = sphere(1.0).translate(X)
shader = compile_glsl(s, params={"radius": s.children[0].radius})
viewer = ShaderViewer(shader, uniforms={"radius": 1.0})
```

Option 1 is cleaner but requires the Param object to propagate through
the expression tree. Option 2 keeps the SDF API unchanged but requires
the user to manually identify which values are dynamic.

### Prior art: sdfray

[sdfray](https://github.com/BenLand100/sdfray) by BenLand100 is the
closest existing implementation of this approach in Python. Key details:

**Architecture**: Each SDF primitive inherits from an `SDF` base class. The
`glsl()` method returns a 3-tuple `(geo_expr, prop_expr, fragments)`:
- `geo_expr`: GLSL expression evaluating to `float` (the distance)
- `prop_expr`: GLSL expression evaluating to `GeoInfo` (distance + surface)
- `fragments`: list of GLSL function definitions needed by the expressions

**Transform handling**: Translations and rotations are pre-composed
numerically in Python and baked as GLSL mat3/vec3 literals. No runtime
matrix math in the shader. This means transforms that depend on dynamic
parameters need the `Parameter` symbolic system.

**Fragment deduplication**: Same function definition (e.g., `float sphere(...)`)
collected from multiple tree nodes is deduplicated by string equality before
shader assembly.

**Shader assembly pipeline** (from `scene.py`):
1. Core uniforms + struct definitions (`GeoInfo`, `Property`)
2. Deduplicated function fragments from the SDF tree
3. Generated `float sdf(vec3 p) { return <expression>; }`
4. Generated `Property prop(vec3 p, ...) { return <expression>; }`
5. Ray marching infrastructure (gradient, next_surface)
6. Light functions
7. `void main()` entry point

Rendered via ModernGL (desktop OpenGL). Uses a fullscreen quad with a
triangle strip.

**Limitations**: Only 4 primitives (sphere, box, cylinder, plane). No smooth
blending. No repeat, twist, bend. Binary-only CSG tree. GPL-3.0 license.

**Lesson for this project**: The sdfray pattern proves the approach works.
The main gap is that sdfray has a tiny primitive set and no smooth ops.
This library has ~20 primitives and ~30 operations, so the codegen would
be more work but architecturally identical.

### Prior art: @thi.ng/shader-ast (TypeScript)

The most mature expression-tree-to-GLSL system found. Relevant patterns:

**Node types**: Every value is a `Term<T>` with a `tag` field (`"lit"`,
`"sym"`, `"op1"`, `"op2"`, `"call"`, `"fn"`, `"if"`, etc.) and a `type`
field (GLSL type). SDF primitives are `Func` nodes created via `defn()`:

```typescript
// sphere SDF — creates a Func AST node
export const sdfSphere = defn(F, "sdSphere", [V3, F], (p, r) => [
    ret(sub(length(p), r)),
]);
```

**Dual backend**: Same AST compiles to both GLSL (for GPU) and JavaScript
(for CPU), via `targetGLSL()` and `targetJS()` code generators that use
a visitor pattern over the AST tags.

**SDF composition**: Call SDF functions (creating `FnCall` nodes) and pass
results to combinators:

```typescript
d1 = sym(sdfSphere(pos, float(1.0)));
d2 = sym(sdfBox3(pos, vec3(1, 1, 1)));
result = sdfSmoothUnion(d1, d2, float(0.5));
```

**Topological ordering**: Function dependencies are tracked and sorted
automatically during code generation, so the emitted GLSL defines
functions before they're called.

### Effort estimate

- Expression tree IR (see [Prerequisite](#prerequisite-expression-tree--ir)): ~3-5 days
- GLSL codegen for ~15 core primitives: ~2 days
- GLSL codegen for ~15 core operations: ~2 days
- ShaderViewer generalization (camera, lighting, sphere tracing): ~1 day (mostly done in `glsl.py`)
- Dynamic uniform / Param system: ~1 day
- Testing (visual comparison of shader vs mesh renders): ~1-2 days
- **Total: ~10-16 days**

---

## Path B: Fidget (JIT-compiled CPU Evaluation)

**Approach**: Translate SDF expression trees into
[Fidget](https://github.com/mkeeter/fidget) tapes, which are JIT-compiled
to native SIMD code. Re-mesh via Fidget's Manifold Dual Contouring (faster
and higher-quality than marching cubes). Feed resulting triangles to the
existing `MeshViewer`.

**Latency**: ~50-200ms for re-mesh on parameter change (vs current ~1-2s).
Not true real-time, but fast enough to feel responsive.

### How Fidget works

Fidget's pipeline:
```
Math expression tree → DAG (CSE) → SSA tape → Register-allocated bytecode
                                                        |
                                                  +-----+------+
                                                  |            |
                                            VmFunction    JitFunction
                                           (portable)    (31x faster)
```

**Expression graphs**: Built from `x`, `y`, `z` variables and math operations.
Identical subexpressions are deduplicated into a DAG.

**Tape compilation**: The DAG is lowered to SSA (single static assignment)
form — a flat list of instructions. A register allocator maps unlimited SSA
registers to 256 physical registers.

**Tape simplification**: The key innovation. During interval evaluation,
Fidget traces which tape instructions actually contribute to the output
for a given spatial region. Instructions that don't contribute are pruned,
producing a simplified tape for that region. This is applied recursively
during octree subdivision.

**JIT compilation**: Hand-written aarch64 (NEON) and x86_64 (AVX2) code
generators produce native SIMD code from the tape. Processes 4 floats
(ARM) or 8 floats (x86) per instruction.

**Meshing**: Manifold Dual Contouring on an adaptive octree. Produces
watertight, manifold meshes with sharp feature preservation. Fewer
triangles than marching cubes for equivalent quality.

### Performance

From Fidget benchmarks on M1 Max:

| Method | 1024³ voxels | 2048³ voxels |
|--------|-------------|-------------|
| libfive (CPU) | 66.8 ms | 211 ms |
| MPR (GTX 1080 Ti GPU) | 22.6 ms | 60.6 ms |
| Fidget VM (CPU) | 61.7 ms | 184 ms |
| Fidget JIT (CPU) | 23.6 ms | 77.4 ms |

For a 7,867-expression model at 1024×1024:
- Brute-force bytecode: 5.8s → Brute-force JIT: 182ms (31x speedup)
- With interval pruning: 6ms (bytecode) → 4.6ms (JIT)

### Python bindings: fidgetpy

[fidgetpy](https://github.com/alexneufeld/fidgetpy) provides Python access.

**Installation**: Not on PyPI. Requires Rust toolchain + maturin:
```bash
git clone https://github.com/alexneufeld/fidgetpy.git
cd fidgetpy && maturin develop
```

**API** (two levels):

```python
# Low-level: symbolic Tree math
from fidgetpy.math import axes
from fidgetpy.types import Tree, Vec3

x, y, z = axes()
sphere = (x*x + y*y + z*z).sqrt() - 1.0
mesh = sphere.mesh(depth=6, cx=0, cy=0, cz=0, region_size=2.0)

# High-level: Shape objects with auto bounds
from fidgetpy.shapes import sphere, box, union, move
s = sphere(1.0)
b = move(box(2, 2, 2), 1.5, 0, 0)
combined = union(s, b)
mesh = combined.mesh(depth=6)
stl_bytes = mesh.to_stl()
```

**Supported primitives**: sphere, box, torus, cylinder, circle, rectangle.
**Supported operations**: union, intersection, difference, xor, move, expand,
extrude_z, revolve_z.

**What's missing** (compared to this library):
- No smooth blending (no `k` parameter) — would need to implement smooth
  min/max as tree expressions: `smin(a,b,k) = -ln(exp(-k*a) + exp(-k*b))/k`
- No rotate, scale, twist, bend, repeat, mirror, shell
- No batch point evaluation exposed to Python (single-point `eval()` only)
- No numpy array I/O

### Integration strategy

The most promising approach is **expression-tree transpiler**:

1. Add an expression tree to `SDF3` (see [Prerequisite](#prerequisite-expression-tree--ir))
2. Write a `to_fidget()` function that walks the tree and builds a Fidget
   `Tree` from the nodes
3. Implement missing operations as Fidget tree expressions:
   - `shell(d, t)` → `abs(d) - t/2`
   - `smooth_union(a, b, k)` → approximate via `exp`/`ln` (Fidget has both)
   - `rotate(angle, axis)` → `remap_xyz()` with trig expressions
   - `repeat(spacing)` → modular arithmetic via `floor`/`round`
4. At mesh time, if Fidget tree is available, use `tree.mesh()` instead of
   marching cubes
5. Feed the mesh triangles to `MeshViewer` as before

### Maturity assessment

- **Fidget core** (Rust): 1,269+ commits, actively maintained by Matt Keeter
  (author of libfive). v0.4.0 released September 2025. Last push March 2026.
  Self-described as "experimental".
- **fidgetpy**: 44 commits over ~2 weeks in April 2025 by a community
  contributor. Not actively maintained. Has bugs (typos in `intersection()`,
  `xor()`). No CI, no PyPI package.
- **Verdict**: Fidget core is solid but fidgetpy needs work. May need to
  fork fidgetpy or write minimal bindings directly.

### Effort estimate

- Expression tree IR: ~3-5 days (shared with Path A)
- fidgetpy bindings fixes/extensions: ~2-3 days
- Tree-to-Fidget transpiler: ~3-4 days
- Missing operations as Fidget expressions: ~2-3 days
- Integration with MeshViewer: ~1 day
- **Total: ~11-16 days** (with IR work shared with Path A)

---

## Path C: Taichi (Python → GPU JIT)

**Approach**: Rewrite SDF primitives as `@ti.func` functions that are
JIT-compiled to GPU kernels. A sphere-tracing `@ti.kernel` renders the
SDF, producing a numpy image array that is displayed in an anywidget.

**Latency**: ~30-120 FPS for sphere tracing on GPU (the rendering itself).
But the display pipeline adds overhead: GPU → numpy copy → image encode →
websocket → browser. Realistic interactive rate: ~10-30 FPS.

### How Taichi works

```python
import taichi as ti
ti.init(arch=ti.gpu)  # tries CUDA → Vulkan → Metal → OpenGL

@ti.func
def sd_sphere(p: ti.math.vec3, r: float) -> float:
    return p.norm() - r

@ti.func
def sd_box(p: ti.math.vec3, size: ti.math.vec3) -> float:
    q = abs(p) - size
    return ti.math.length(ti.math.max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0)

@ti.kernel
def render(color_buffer: ti.template(), cam_pos: ti.math.vec3, ...):
    for u, v in color_buffer:  # parallelized over all pixels
        ray = compute_ray(u, v, cam_pos, ...)
        d = ray_march(ray)
        color_buffer[u, v] = shade(d, ...)
```

All `@ti.func` are force-inlined at compile time — zero function-call
overhead. The resulting kernel is a single monolithic GPU program.

**Backend targets**: CUDA, Vulkan, Metal, OpenGL, OpenGL ES, x64 CPU, ARM64 CPU.

**Official SDF example**: `taichi/python/taichi/examples/rendering/sdf_renderer.py`
implements a full path tracer (sphere, box, cylinder, fractal modifier)
at 1280×720 with 6-bounce depth. A simpler single-pass sphere tracer
would be ~100-1000x cheaper per frame.

### Notebook integration

Taichi has **no native notebook widget support**. Two GUI systems exist
(legacy `ti.GUI` and GGUI/Vulkan) but both are desktop-only windows.

**Workaround**: Headless rendering + image streaming:
```python
# Render to numpy array
window = ti.ui.Window('SDF', (640, 360), show_window=False)
# ... render ...
img_np = window.get_image_buffer_as_numpy()  # HxWx4

# Stream to anywidget as PNG bytes
viewer.image_data = encode_png(img_np)
```

A Deepnote example achieves ~50 FPS with this pipeline using `ipycanvas`
and a threaded render loop. For marimo, a custom anywidget would work
similarly.

**Key limitation**: Every frame requires a GPU→CPU copy (`field.to_numpy()`)
plus image encoding plus websocket transfer. This adds ~10-30ms of latency
that pure WebGL (Path A) avoids entirely.

### Challenges

- **Static compilation**: All SDF composition must be known at compile time.
  `@ti.func` cannot be composed dynamically at runtime. Changing the SDF
  tree structure (adding/removing nodes) requires kernel recompilation
  (2-30 seconds cold, cached afterward). Changing numeric parameters
  (radius, position) does NOT require recompilation.
- **Exponential IR blowup**: If `f(i)` calls `f(i-1)` twice, inlining
  produces O(2^N) intermediate IR. Deep CSG trees with shared sub-SDFs
  can trigger this. Taichi's simplification passes reduce it, but
  compilation time can spike.
- **Heavy dependency**: 50-83 MB wheel (comparable to VTK). Requires
  GPU drivers.
- **No browser-side rendering**: All computation is server-side. Can't
  work in static notebook exports or serverless environments.

### Effort estimate

- Rewrite ~15 primitives as `@ti.func`: ~2 days
- Rewrite ~15 operations as `@ti.func`: ~2 days
- Sphere tracing kernel + shading: ~1-2 days
- anywidget with image streaming + mouse controls: ~2-3 days
- Integration with SDF3 API (auto-generate kernel from tree): ~3-4 days
- **Total: ~10-13 days**

---

## Path D: JAX-accelerated Evaluation

**Approach**: Use JAX's `jit` + `vmap` to accelerate the existing
numpy-based SDF evaluation on GPU/CPU. Keep marching cubes for meshing
but make each evaluation step dramatically faster.

**Latency**: ~100-500ms for re-mesh (better than current ~1-2s, worse
than Fidget). The main win is that existing SDF functions can be used
with minimal changes since JAX's API mirrors numpy.

### How it would work

```python
import jax
import jax.numpy as jnp

# Existing SDF functions mostly work with jnp instead of np:
def sphere_jax(p, radius=1.0):
    return jnp.sqrt(jnp.sum(p**2, axis=1)) - radius

# JIT compile + vectorize
fast_eval = jax.jit(sphere_jax)

# Evaluate on GPU
points = jnp.array(grid_points)  # Nx3
distances = fast_eval(points)     # N×1, runs on GPU
```

JAX's tracing mechanism captures the computation graph on first call,
compiles it via XLA to optimized GPU/CPU code, and caches the result.

### Advantages

- **Minimal API changes**: JAX's numpy compatibility means many existing
  SDF functions work with `jnp` drop-in. `np.minimum` → `jnp.minimum`,
  `np.abs` → `jnp.abs`, etc.
- **Automatic differentiation**: `jax.grad` computes exact SDF gradients
  (normals) analytically, avoiding the 6-eval finite-difference approximation.
- **No expression tree needed**: JAX traces through the existing Python
  closures — no IR refactor required.
- **GPU acceleration**: XLA compiles to CUDA/ROCm for GPU, or optimized
  CPU code.

### Challenges

- **Not all numpy patterns trace cleanly**: `np.where` with side effects,
  Python control flow (`if/else` on traced values), `np.clip` with
  Python scalars — these need `jax.lax` equivalents.
- **Dynamic shapes**: JAX requires fixed array shapes at trace time.
  The `repeat` operation with variable `padding` and the `neighbors`
  function with `itertools.product` would need reworking.
- **Still uses marching cubes**: The speedup is only in SDF evaluation,
  not in meshing. The marching cubes step itself has overhead.
- **Heavy dependency**: JAX + jaxlib is ~200-400MB.
- **No rendering acceleration**: Still generates a mesh, sends it to
  MeshViewer. No sphere tracing benefit.

### Effort estimate

- Audit SDF functions for JAX compatibility: ~2 days
- Write jnp-compatible versions of incompatible functions: ~2-3 days
- Integration (auto-detect JAX, use as accelerated backend): ~1-2 days
- Testing: ~1-2 days
- **Total: ~6-9 days**

---

## Prerequisite: Expression Tree / IR

Paths A and B both require an inspectable representation of SDF
compositions. Currently, SDFs are opaque closures — `sphere().translate(X)`
produces nested Python functions with no way to recover the structure.

### Design

Add an `SDFNode` tree that is built alongside the closure:

```python
class SDFNode:
    """Base class for SDF expression tree nodes."""
    pass

class PrimitiveNode(SDFNode):
    name: str            # "sphere", "rounded_box", etc.
    params: dict         # {"radius": 1.0, "center": [0,0,0]}

class OperationNode(SDFNode):
    name: str            # "translate", "union", "shell", etc.
    children: list       # [SDFNode, ...]
    params: dict         # {"offset": [1,0,0]}
```

### Integration with SDF3

```python
class SDF3:
    def __init__(self, f, node=None):
        self.f = f
        self.node = node  # None for custom lambdas

def sdf3(f):
    def wrapper(*args, **kwargs):
        closure = f(*args, **kwargs)
        node = PrimitiveNode(f.__name__, args, kwargs)
        return SDF3(closure, node)
    return wrapper

def op3(f):
    def wrapper(*args, **kwargs):
        closure = f(*args, **kwargs)
        # args[0] is the SDF being operated on
        children = [a.node for a in args if isinstance(a, SDF3)]
        node = OperationNode(f.__name__, children, ...)
        return SDF3(closure, node)
    _ops[f.__name__] = wrapper
    return wrapper
```

### Keeping representations in sync

Three proven strategies from other projects:

1. **Single AST, multiple backends** (@thi.ng/shader-ast, Curv):
   The AST is the source of truth. CPU eval and GLSL codegen both derive
   from it. Most robust, but requires rewriting SDF evaluation to walk
   the tree rather than call closures.

2. **Dual representation** (sdfray, this proposal):
   Keep the closure for CPU evaluation (backward compatible) and add a
   parallel node tree for compilation. Risk: the two can drift out of sync
   if a new primitive adds the closure but forgets the node.

3. **Tracing** (JAX):
   Run the closure with tracer objects that record operations. No node tree
   needed — the trace IS the IR. Elegant but requires all operations to
   be traceable (no Python control flow, no side effects).

**Recommendation**: Option 2 (dual representation) for this project.
It's the least disruptive — all existing tests and user code continue
to work unchanged. The node tree is additive. A `node=None` sentinel
on SDF3 gracefully degrades for custom lambdas.

### What to store on nodes

At minimum, enough information to reconstruct the SDF:
- Primitive name + constructor arguments
- Operation name + operation arguments
- References to child nodes
- The `_k` smooth blending parameter (if set)

For GLSL codegen, also useful:
- Whether each parameter is a Param (dynamic uniform) or a constant
- Precomputed transform matrices (as sdfray does)
- A structural hash for CSE (as libfive does)

### Effort estimate

- `SDFNode` class hierarchy: ~0.5 days
- Update `sdf3`, `op3`, `op2`, `op23` decorators: ~1 day
- Update all primitives in `d3.py`, `d2.py`: ~1 day
- Update all operations in `d3.py`, `d2.py`, `dn.py`: ~1-2 days
- Tests (verify node trees match expected structure): ~0.5-1 day
- **Total: ~3-5 days**

---

## Reference Implementations

### Codegen references

| Project | Language | What to study | Link |
|---------|----------|---------------|------|
| **sdfray** | Python | Recursive `glsl()` on SDF class hierarchy, fragment deduplication, transform pre-composition, ModernGL rendering | [github.com/BenLand100/sdfray](https://github.com/BenLand100/sdfray) |
| **@thi.ng/shader-ast** | TypeScript | Tagged AST nodes, `defn()` for SDF functions, `targetGLSL()` code generator, topological dependency ordering, dual GLSL+JS output | [github.com/thi-ng/umbrella/.../shader-ast](https://github.com/thi-ng/umbrella/tree/develop/packages/shader-ast) |
| **Curv** | C++ | Functional SDF language compiled to GLSL fragment shaders, SubCurv compiler, constant folding + CSE | [github.com/curv3d/curv](https://github.com/curv3d/curv) |

### Expression tree / IR references

| Project | Language | What to study | Link |
|---------|----------|---------------|------|
| **Fidget** | Rust | Expression graph → DAG → SSA tape → JIT, interval-based tape simplification, Manifold Dual Contouring | [github.com/mkeeter/fidget](https://github.com/mkeeter/fidget) |
| **libfive** | C++ | Expression tree with `TreeData` variant type, canonical-map CSE, affine accumulation, interval evaluators, Python bindings via C FFI | [github.com/libfive/libfive](https://github.com/libfive/libfive) |

### Interactive parametric CAD in notebooks

| Project | Stack | What to study | Link |
|---------|-------|---------------|------|
| **marimo-cad** | build123d + anywidget | Reactive parametric CAD in marimo with sliders, Three.js viewer, how they handle widget persistence across re-renders | [github.com/cemrehancavdar/marimo-cad](https://github.com/cemrehancavdar/marimo-cad) |

---

## Comparison Matrix

| | Path A: GLSL Codegen | Path B: Fidget | Path C: Taichi | Path D: JAX |
|---|---|---|---|---|
| **Latency** | <16ms (GPU sphere trace) | ~50-200ms (re-mesh) | ~30-100ms (GPU render + copy) | ~100-500ms (re-mesh) |
| **Requires IR?** | Yes | Yes | Partially (static @ti.func) | No (tracing) |
| **Dependency** | anywidget only (~0.5MB) | Rust toolchain + fidgetpy | taichi (~50-80MB) | jax + jaxlib (~200-400MB) |
| **Browser-native?** | Yes (WebGL) | No (CPU, sends mesh) | No (server-side render) | No (CPU/GPU, sends mesh) |
| **Works offline?** | Yes (all client-side) | No (needs Python) | No (needs Python + GPU) | No (needs Python) |
| **Custom lambdas?** | No | No | No | Yes (if traceable) |
| **Smooth blending?** | Yes (direct GLSL) | Needs workaround | Yes (@ti.func) | Yes (jnp) |
| **Effort** | ~10-16 days | ~11-16 days | ~10-13 days | ~6-9 days |
| **UX quality** | Best (instant, smooth orbit) | Good (slight lag on change) | Good (slight lag from streaming) | Adequate (noticeable lag) |

---

## Recommended Sequence

These paths are not mutually exclusive. A pragmatic approach:

### Phase 1: Expression tree IR (~3-5 days)

Add `SDFNode` to `SDF3`/`SDF2`. This unlocks both Path A and Path B and
is valuable on its own (enables tree inspection, serialization, etc.).
All existing tests and user code continue to work unchanged.

### Phase 2: GLSL codegen (Path A) (~7-11 days)

Build on the IR to compile SDFs to GLSL shaders. Start with the ~10 most
common primitives and operations — enough to cover the customizable box
and the README examples. Generalize the existing `ShaderViewer` anywidget
to accept compiled shaders.

This gives the best interactive UX and has the lightest dependency
footprint (just anywidget, which is already a dependency).

### Phase 3 (optional): Fidget backend (Path B) (~8-11 days)

For users who want fast meshing (STL export, 3D printing workflows)
rather than real-time preview, add Fidget as an optional backend. This
reuses the same IR from Phase 1.

### Skip for now: Taichi and JAX

Taichi adds a heavy dependency for a worse-than-WebGL interactive
experience. JAX gives a moderate speedup without solving the fundamental
latency problem. Both are better suited if this library's scope expands
to simulation or optimization, not just parametric design preview.
