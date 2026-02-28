"""ISO-10303-21 STEP file writer for triangle meshes.

Port from pschou/py-sdf. Converts triangle mesh (array of vertex triplets)
to STEP format with edge deduplication.
"""

import numpy as np
from datetime import datetime


def write_step(path, points, tol=0):
    """Write a triangle mesh as a STEP file.

    Args:
        path: Output file path (.step or .stp)
        points: Nx3 array of vertices (every 3 rows = 1 triangle)
        tol: Tolerance for vertex deduplication (0 = exact match)
    """
    points = np.array(points, dtype='float64').reshape((-1, 3, 3))
    n_triangles = len(points)

    if n_triangles == 0:
        with open(path, 'w') as fp:
            fp.write(_step_header())
            fp.write(_step_footer())
        return

    # Deduplicate vertices
    all_verts = points.reshape(-1, 3)
    if tol > 0:
        unique_verts, vert_map = _deduplicate(all_verts, tol)
    else:
        unique_verts, vert_map = np.unique(
            all_verts, axis=0, return_inverse=True
        )
    vert_map = vert_map.reshape(-1, 3)

    # Build edge set (deduplicated, directed)
    edges = {}  # (v1_idx, v2_idx) -> edge_id
    edge_list = []
    for tri_idx in range(n_triangles):
        for i in range(3):
            j = (i + 1) % 3
            v1 = int(vert_map[tri_idx, i])
            v2 = int(vert_map[tri_idx, j])
            key = (min(v1, v2), max(v1, v2))
            if key not in edges:
                edges[key] = len(edge_list)
                edge_list.append(key)

    with open(path, 'w') as fp:
        fp.write(_step_header())
        fp.write('DATA;\n')

        eid = 1  # entity counter

        # Write cartesian points
        point_ids = {}
        for i, v in enumerate(unique_verts):
            point_ids[i] = eid
            fp.write(f"#{eid}=CARTESIAN_POINT('',({v[0]:.6f},{v[1]:.6f},{v[2]:.6f}));\n")
            eid += 1

        # Write vertex points
        vertex_ids = {}
        for i in range(len(unique_verts)):
            vertex_ids[i] = eid
            fp.write(f"#{eid}=VERTEX_POINT('',#{point_ids[i]});\n")
            eid += 1

        # Write edges
        edge_ids = {}
        for idx, (v1, v2) in enumerate(edge_list):
            edge_ids[idx] = eid
            fp.write(f"#{eid}=EDGE_CURVE('',#{vertex_ids[v1]},#{vertex_ids[v2]},#{eid+1},.T.);\n")
            eid += 1
            # Line for the edge curve
            fp.write(f"#{eid}=LINE('',#{point_ids[v1]},#{eid+1});\n")
            eid += 1
            # Direction vector
            d = unique_verts[v2] - unique_verts[v1]
            norm = np.linalg.norm(d)
            if norm > 0:
                d = d / norm
            fp.write(f"#{eid}=VECTOR('',#{eid+1},{norm:.6f});\n")
            eid += 1
            fp.write(f"#{eid}=DIRECTION('',({d[0]:.6f},{d[1]:.6f},{d[2]:.6f}));\n")
            eid += 1

        # Write faces (triangles)
        face_ids = []
        for tri_idx in range(n_triangles):
            # Normal
            v0 = unique_verts[vert_map[tri_idx, 0]]
            v1 = unique_verts[vert_map[tri_idx, 1]]
            v2 = unique_verts[vert_map[tri_idx, 2]]
            normal = np.cross(v1 - v0, v2 - v0)
            norm_len = np.linalg.norm(normal)
            if norm_len > 0:
                normal = normal / norm_len

            # Plane surface
            plane_id = eid
            fp.write(f"#{eid}=PLANE('',#{eid+1});\n")
            eid += 1
            fp.write(f"#{eid}=AXIS2_PLACEMENT_3D('',#{point_ids[vert_map[tri_idx, 0]]},#{eid+1},#{eid+2});\n")
            eid += 1
            fp.write(f"#{eid}=DIRECTION('',({normal[0]:.6f},{normal[1]:.6f},{normal[2]:.6f}));\n")
            eid += 1
            ref_dir = v1 - v0
            ref_len = np.linalg.norm(ref_dir)
            if ref_len > 0:
                ref_dir = ref_dir / ref_len
            fp.write(f"#{eid}=DIRECTION('',({ref_dir[0]:.6f},{ref_dir[1]:.6f},{ref_dir[2]:.6f}));\n")
            eid += 1

            # Edge loops for the triangle
            oriented_edges = []
            for i in range(3):
                j = (i + 1) % 3
                va = int(vert_map[tri_idx, i])
                vb = int(vert_map[tri_idx, j])
                key = (min(va, vb), max(va, vb))
                edge_idx = edges[key]
                orientation = '.T.' if va <= vb else '.F.'
                fp.write(f"#{eid}=ORIENTED_EDGE('',*,*,#{edge_ids[edge_idx]},{orientation});\n")
                oriented_edges.append(eid)
                eid += 1

            loop_id = eid
            oe_refs = ','.join(f'#{oe}' for oe in oriented_edges)
            fp.write(f"#{eid}=EDGE_LOOP('',({oe_refs}));\n")
            eid += 1

            bound_id = eid
            fp.write(f"#{eid}=FACE_OUTER_BOUND('',#{loop_id},.T.);\n")
            eid += 1

            face_id = eid
            fp.write(f"#{eid}=ADVANCED_FACE('',({f'#{bound_id}'}),#{plane_id},.T.);\n")
            face_ids.append(face_id)
            eid += 1

        # Closed shell
        face_refs = ','.join(f'#{fid}' for fid in face_ids)
        fp.write(f"#{eid}=CLOSED_SHELL('',({face_refs}));\n")
        shell_id = eid
        eid += 1

        # Manifold solid
        fp.write(f"#{eid}=MANIFOLD_SOLID_BREP('',#{shell_id});\n")
        eid += 1

        fp.write('ENDSEC;\n')
        fp.write(_step_footer())


def _step_header():
    now = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
    return f"""ISO-10303-21;
HEADER;
FILE_DESCRIPTION(('SDF generated STEP'),'2;1');
FILE_NAME('sdf_output.step','{now}',('sdf'),(''),
  'sdf','sdf','');
FILE_SCHEMA(('AUTOMOTIVE_DESIGN'));
ENDSEC;
"""


def _step_footer():
    return 'END-ISO-10303-21;\n'


def _deduplicate(points, tol):
    """Deduplicate points within tolerance."""
    from scipy.spatial import cKDTree
    tree = cKDTree(points)
    groups = tree.query_ball_tree(tree, tol)
    mapping = np.arange(len(points))
    for group in groups:
        canonical = min(group)
        for idx in group:
            mapping[idx] = canonical
    unique_map, inverse = np.unique(mapping, return_inverse=True)
    unique_points = points[unique_map]
    return unique_points, inverse
