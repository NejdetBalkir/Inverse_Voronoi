import numpy as np

def ExplicitVoronoi(vor, neighbor_vertices_storage_updated, total_edges):
    """
    Build an explicit Voronoi dictionary after vertex merging/re-indexing.
    Each cell lists its edges, neighbor cells, and vertex indices.
    """

    region_of_point = vor.point_region
    point_of_region = {region_of_point[i]: i for i in range(len(vor.points))}

    explicit_voronoi = {}

    # ------------------------------------------------------------
    # Build neighbor and edge lists per seed (cell)
    # ------------------------------------------------------------
    for seed in range(len(vor.points)):
        seed_neighbors = []
        seed_edges = []

        seed_region = region_of_point[seed]

        for a_region, b_region, edge in neighbor_vertices_storage_updated:
            if a_region == seed_region:
                nb = point_of_region.get(b_region, None)
                if nb is not None:
                    seed_neighbors.append(nb)
                    seed_edges.append(edge)
            elif b_region == seed_region:
                nb = point_of_region.get(a_region, None)
                if nb is not None:
                    seed_neighbors.append(nb)
                    seed_edges.append(edge)

        sub_dict = {
            "seed number": seed,
            "seed coordinates": vor.points[seed].tolist(),
            "neighbors": seed_neighbors,
            "edges": seed_edges,
        }
        explicit_voronoi[f"sub_dict_{seed}"] = sub_dict

    # ------------------------------------------------------------
    # Add new boundary connections (for corners and box closure)
    # ------------------------------------------------------------
    number_of_old_vertices = len(vor.vertices)

    for key in explicit_voronoi:
        cell_edges = explicit_voronoi[key]["edges"]

        # find boundary vertices (those with index >= number_of_old_vertices)
        boundary_vertices = [
            idx
            for sub in cell_edges
            for idx in sub
            if idx >= number_of_old_vertices
        ]

        if not boundary_vertices:
            # this cell has no boundary intersections
            unique_vertices = {v for e in cell_edges for v in e}
            explicit_voronoi[key]["cell vertices"] = list(unique_vertices)
            continue

        # Attempt to pair boundary vertices that are spatially close
        used = set()
        for i in range(len(boundary_vertices)):
            v1 = boundary_vertices[i]
            if v1 in used:
                continue
            for j in range(i + 1, len(boundary_vertices)):
                v2 = boundary_vertices[j]
                if v2 in used:
                    continue
                v1_nei = []
                v2_nei = []
                for e0, e1 in total_edges:
                    if e0 == v1:
                        v1_nei.append(e1)
                    elif e1 == v1:
                        v1_nei.append(e0)
                    if e0 == v2:
                        v2_nei.append(e1)
                    elif e1 == v2:
                        v2_nei.append(e0)

                common = list(set(v1_nei) & set(v2_nei))
                if len(common) == 0:
                    explicit_voronoi[key]["edges"].append([v1, v2])
                else:
                    explicit_voronoi[key]["edges"].append([v1, common[0]])
                    explicit_voronoi[key]["edges"].append([v2, common[0]])
                used.add(v1)
                used.add(v2)
                break

        # finalize unique vertex list for this cell
        unique_vertices = set()
        for e in explicit_voronoi[key]["edges"]:
            unique_vertices.update(e)
        explicit_voronoi[key]["cell vertices"] = list(unique_vertices)

    return explicit_voronoi
