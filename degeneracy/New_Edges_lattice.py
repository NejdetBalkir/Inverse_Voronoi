import numpy as np

def New_Edges(new_vertices, new_vertices_index, minx, maxx, miny, maxy, vor, tol=1e-9):
    """
    Create boundary edges connecting new intersection vertices to the
    correct bounding-box corners.
    Works with globally reindexed vertices coming from the new New_Vertices.
    """

    n_old = len(vor.vertices)                     # original voronoi vertex count
    n_new = len(new_vertices)                     # number of intersection vertices
    base_corner = n_old + n_new                   # first id reserved for corners

    # corner coordinates (bottom-left, top-left, bottom-right, top-right)
    corners = {
        0: np.array([minx, miny]),
        1: np.array([minx, maxy]),
        2: np.array([maxx, miny]),
        3: np.array([maxx, maxy]),
    }
    corner_ids = {k: base_corner + k for k in corners}  # global ids

    # prepare data containers
    new_edges_storage = []
    boundary_points_count = [[0, 0, []], [1, 0, []], [2, 0, []], [3, 0, []]]

    # classify each new vertex
    for v_idx, (vx, vy) in enumerate(new_vertices):
        gid = new_vertices_index[v_idx]

        # detect which boundary this vertex lies on
        if abs(vx - minx) < tol:
            bnum = 0  # left
            corner_pair = (corner_ids[0], corner_ids[1])
        elif abs(vx - maxx) < tol:
            bnum = 1  # right
            corner_pair = (corner_ids[2], corner_ids[3])
        elif abs(vy - miny) < tol:
            bnum = 2  # bottom
            corner_pair = (corner_ids[0], corner_ids[2])
        elif abs(vy - maxy) < tol:
            bnum = 3  # top
            corner_pair = (corner_ids[1], corner_ids[3])
        else:
            continue  # not on any boundary

        # register vertex in its boundary
        boundary_points_count[bnum][1] += 1
        boundary_points_count[bnum][2].append(gid)

        # connect vertex to its *nearest* corner on that boundary
        c1, c2 = corners[corner_pair[0] - base_corner], corners[corner_pair[1] - base_corner]
        d1 = np.hypot(vx - c1[0], vy - c1[1])
        d2 = np.hypot(vx - c2[0], vy - c2[1])
        corner_id = corner_pair[0] if d1 < d2 else corner_pair[1]
        new_edges_storage.append([gid, corner_id])

    # always include all boundaries (even if no intersections)
    for k in range(4):
        if not boundary_points_count[k][2]:
            boundary_points_count[k] = [k, 0, "no intersection"]

    return boundary_points_count, new_edges_storage, new_vertices
