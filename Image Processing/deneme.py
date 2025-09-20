# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from skimage.morphology import skeletonize

# def skeletonize_image(img):
#     """Convert image to binary skeleton (1-pixel wide lines)."""
#     _, binary = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)
#     binary = binary // 255  # normalize to {0,1}
#     skeleton = skeletonize(binary).astype(np.uint8)
#     return skeleton

# def find_vertices(skel):
#     """Find true Voronoi vertices (junctions & endpoints)."""
#     vertices = []
#     h, w = skel.shape

#     def neighbors(y, x):
#         return [(yy, xx) for yy in range(y-1, y+2)
#                           for xx in range(x-1, x+2)
#                           if not (yy==y and xx==x)
#                           and 0 <= yy < h and 0 <= xx < w
#                           and skel[yy, xx] == 1]

#     for y in range(h):
#         for x in range(w):
#             if skel[y, x] == 1:
#                 neigh = neighbors(y, x)
#                 if len(neigh) != 2:  # endpoints (1) or junctions (>=3)
#                     vertices.append((x,y))
#     return vertices

# def build_graph(skel, vertices):
#     """Walk skeleton to connect vertices with edges."""
#     h, w = skel.shape
#     vert_set = set(vertices)
#     vertex_index = {v: i for i, v in enumerate(vertices)}
#     edges = []

#     visited = set()

#     def neighbors(y, x):
#         return [(yy, xx) for yy in range(y-1, y+2)
#                           for xx in range(x-1, x+2)
#                           if not (yy==y and xx==x)
#                           and 0 <= yy < h and 0 <= xx < w
#                           and skel[yy, xx] == 1]

#     for v in vertices:
#         if v not in vertex_index:
#             continue
#         vx, vy = v
#         for n in neighbors(vy, vx):
#             if (v, n) in visited or (n, v) in visited:
#                 continue
#             path = [v]
#             cur = n
#             prev = v
#             while cur not in vert_set:
#                 neigh = neighbors(cur[1], cur[0])
#                 neigh = [p for p in neigh if p != prev]
#                 if not neigh:
#                     break
#                 path.append(cur)
#                 prev, cur = cur, neigh[0]
#             if cur in vert_set:
#                 path.append(cur)
#                 v1, v2 = path[0], path[-1]
#                 e = tuple(sorted([vertex_index[v1], vertex_index[v2]]))
#                 if e not in edges:
#                     edges.append(e)
#             visited.add((v, n))
#     return edges

# def process_voronoi_image(image_path):
#     img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     if img is None:
#         raise FileNotFoundError(f"Cannot read image: {image_path}")

#     skel = skeletonize_image(img)
#     vertices = find_vertices(skel)
#     edges = build_graph(skel, vertices)

#     explicit_voronoi = {
#         "vertices": [list(v) for v in vertices],
#         "edges": [list(e) for e in edges],
#         "cells": {}  # optional
#     }

#     # Plot reconstruction
#     fig, ax = plt.subplots()
#     ax.imshow(img, cmap="gray")
#     verts = np.array(explicit_voronoi["vertices"])
#     ax.plot(verts[:,0], verts[:,1], "ro", markersize=4)
#     for e in explicit_voronoi["edges"]:
#         x = [verts[e[0],0], verts[e[1],0]]
#         y = [verts[e[0],1], verts[e[1],1]]
#         ax.plot(x, y, "g-", linewidth=2)
#     ax.set_title("Clean Explicit Voronoi")
#     plt.show()

#     return explicit_voronoi

# # Example
# if __name__ == "__main__":
#     explicit_voronoi = process_voronoi_image("C:\\Users\\user\\OneDrive\\PHD\\Repositories\\Inverse_Voronoi\\Image Processing\\voronoi_image.png")
#     # print("Vertices:", len(explicit_voronoi["vertices"]))
#     # print("Edges:", len(explicit_voronoi["edges"]))
#     print(explicit_voronoi)



import cv2
import numpy as np
import matplotlib.pyplot as plt

def process_voronoi_image(image_path):
    # Read grayscale image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    # Threshold to get black Voronoi lines
    _, binary = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)

    # Find contours (each cell)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Detect candidate vertices (junctions and endpoints)
    corners = cv2.goodFeaturesToTrack(binary, 500, 0.01, 5)
    corners = np.int0(corners).reshape(-1, 2)
    vertices = [list(v) for v in corners]

    explicit_voronoi = {}
    edge_set = set()

    for i, cnt in enumerate(contours):
        cnt = cnt.squeeze()
        if cnt.ndim != 2 or cnt.shape[0] < 3:
            continue

        # Snap polygon vertices to nearest detected corner
        cell_vertices = []
        for pt in cnt:
            dists = np.linalg.norm(np.array(vertices) - pt, axis=1)
            v_idx = int(np.argmin(dists))
            if v_idx not in cell_vertices:
                cell_vertices.append(v_idx)

        # Build edges
        edges = [[cell_vertices[j], cell_vertices[(j+1) % len(cell_vertices)]]
                 for j in range(len(cell_vertices))]
        for e in edges:
            edge_set.add(tuple(sorted(e)))

        # Approximate seed coordinate as polygon centroid
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
        else:
            cx, cy = np.mean(cnt, axis=0)

        explicit_voronoi[f"sub_dict_{i}"] = {
            "seed number": i,
            "seed coordinates": [float(cx), float(cy)],
            "neighbors": [],  # will be filled later
            "edges": edges,
            "cell vertices": cell_vertices
        }

    # Compute neighbors: cells sharing an edge are neighbors
    for i, ci in explicit_voronoi.items():
        for j, cj in explicit_voronoi.items():
            if i == j: 
                continue
            if any(set(e1) == set(e2) for e1 in ci["edges"] for e2 in cj["edges"]):
                if cj["seed number"] not in ci["neighbors"]:
                    ci["neighbors"].append(cj["seed number"])

    return explicit_voronoi, vertices, list(edge_set), img


def plot_explicit_voronoi(explicit_voronoi, vertices, edges, img):
    fig, ax = plt.subplots()
    ax.imshow(img, cmap="gray")

    verts = np.array(vertices)
    ax.plot(verts[:,0], verts[:,1], "ro", markersize=4, label="Vertices")

    for e in edges:
        x = [verts[e[0],0], verts[e[1],0]]
        y = [verts[e[0],1], verts[e[1],1]]
        ax.plot(x, y, "g-", linewidth=2)

    ax.set_title("Reconstructed Explicit Voronoi")
    ax.legend()
    plt.show()


# -------------------------------
# Example usage
# -------------------------------
if __name__ == "__main__":
    voronoi_dict, vertices, edges, img = process_voronoi_image("C:\\Users\\user\\OneDrive\\PHD\\Repositories\\Inverse_Voronoi\\Image Processing\\voronoi_image.png")

    # Print first cell as example
    first_key = list(voronoi_dict.keys())[0]
    print(first_key, ":", voronoi_dict[first_key])

    # Plot reconstruction
    plot_explicit_voronoi(voronoi_dict, vertices, edges, img)
