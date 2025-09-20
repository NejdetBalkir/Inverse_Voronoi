import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from collections import defaultdict
from scipy.ndimage import label

def extract_voronoi_graph(image_path, plot=True, merge_tol=8, vertex_tol=3):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    # --- skeletonization ---
    _, binary = cv2.threshold(img, 200, 1, cv2.THRESH_BINARY_INV)
    skel = skeletonize(binary).astype(np.uint8)
    h, w = skel.shape

    # 8-neighborhood
    NB8 = [(-1,-1),(-1,0),(-1,1),
           (0,-1),        (0,1),
           (1,-1),(1,0),(1,1)]

    def deg(x, y):
        s = 0
        for dx,dy in NB8:
            nx, ny = x+dx, y+dy
            if 0 <= nx < w and 0 <= ny < h and skel[ny, nx] > 0:
                s += 1
        return s

    # --- vertex detection ---
    raw_vertices = []
    for y in range(1, h-1):
        for x in range(1, w-1):
            if skel[y, x] == 0: 
                continue
            n = deg(x, y)
            if n >= 3 or n == 1:
                raw_vertices.append((x, y))
            if x in (0, w-1) or y in (0, h-1):
                raw_vertices.append((x, y))

    ys, xs = np.where(skel > 0)
    if len(xs) > 0:
        raw_vertices.extend([
            (int(xs.min()), int(ys.min())),
            (int(xs.max()), int(ys.max())),
            (int(xs.max()), int(ys.min())),
            (int(xs.min()), int(ys.max()))
        ])

    # kaba kümeleme
    uniq = []
    for v in raw_vertices:
        if not any(np.linalg.norm(np.array(v)-np.array(u)) < 1 for u in uniq):
            uniq.append(tuple(v))

    # merge yakın vertex’ler
    merged = []
    used = set()
    for i, vi in enumerate(uniq):
        if i in used: 
            continue
        cluster = [vi]
        for j, vj in enumerate(uniq):
            if j <= i or j in used: 
                continue
            if np.linalg.norm(np.array(vi) - np.array(vj)) < merge_tol:
                cluster.append(vj)
                used.add(j)
        cx = int(np.mean([p[0] for p in cluster]))
        cy = int(np.mean([p[1] for p in cluster]))
        merged.append((cx, cy))
        used.add(i)

    vertices = {i:v for i, v in enumerate(merged)}

    # --- skeleton tracing edges ---
    node_mask = np.zeros_like(skel, dtype=np.uint8)
    ys, xs = np.where(skel > 0)
    for x, y in zip(xs, ys):
        if deg(x, y) != 2:
            node_mask[y, x] = 1

    node_labeled, n_nodes = label(node_mask)

    node_centers = {}
    for nid in range(1, n_nodes+1):
        ny, nx = np.where(node_labeled == nid)
        cx = int(np.mean(nx))
        cy = int(np.mean(ny))
        node_centers[nid] = (cx, cy)

    def nearest_vertex_id(pt):
        px, py = pt
        d2 = [ (vx - px)**2 + (vy - py)**2 for (vx, vy) in vertices.values() ]
        return int(np.argmin(d2))

    node_to_vertex = {nid: nearest_vertex_id(c) for nid, c in node_centers.items()}

    visited_pixel_edges = set()
    skeleton_edges = set()

    def walk_from_node(nid):
        ny, nx = np.where(node_labeled == nid)
        starts = []
        for x, y in zip(nx, ny):
            for dx, dy in NB8:
                tx, ty = x+dx, y+dy
                if 0 <= tx < w and 0 <= ty < h and skel[ty, tx] > 0 and node_labeled[ty, tx] == 0:
                    starts.append((x, y, tx, ty))
        for sx, sy, tx, ty in starts:
            prev = (sx, sy)
            cur  = (tx, ty)
            while True:
                step = tuple(sorted((prev, cur)))
                if step in visited_pixel_edges:
                    break
                visited_pixel_edges.add(step)
                cx, cy = cur
                if node_labeled[cy, cx] > 0:
                    nid2 = int(node_labeled[cy, cx])
                    v1 = node_to_vertex[nid]
                    v2 = node_to_vertex[nid2]
                    if v1 != v2:
                        skeleton_edges.add(tuple(sorted((v1, v2))))
                    break
                neigh = []
                for dx, dy in NB8:
                    tx, ty = cx+dx, cy+dy
                    if 0 <= tx < w and 0 <= ty < h and skel[ty, tx] > 0 and (tx, ty) != prev:
                        neigh.append((tx, ty))
                if not neigh:
                    break
                prev, cur = cur, neigh[0]

    for nid in range(1, n_nodes+1):
        walk_from_node(nid)

    # --- pairwise ek edges ---
    labeled, ncomponents = label(skel)
    pairwise_edges = set()
    for comp_id in range(1, ncomponents+1):
        ys, xs = np.where(labeled == comp_id)
        pixels = list(zip(xs, ys))
        if not pixels:
            continue
        dmap = []
        comp_vertices = []
        for vid, vp in vertices.items():
            min_dist = min((vp[0]-px)**2+(vp[1]-py)**2 for px, py in pixels)
            dmap.append((min_dist, vid))
            if min_dist < vertex_tol**2:
                comp_vertices.append(vid)
        dmap.sort()
        closest_two = [dmap[0][1], dmap[1][1]] if len(dmap) >= 2 else []
        comp_vertices.extend(closest_two)
        comp_vertices = list(set(comp_vertices))
        if len(comp_vertices) >= 2:
            for i in range(len(comp_vertices)):
                for j in range(i+1, len(comp_vertices)):
                    pairwise_edges.add(tuple(sorted((comp_vertices[i], comp_vertices[j]))))

    # --- skeleton + sadece eksik pairwise ---
    edges = set(skeleton_edges)
    for e in pairwise_edges:
        if e not in edges:
            edges.add(e)

    connectivity = defaultdict(list)
    for a, b in edges:
        connectivity[a].append(b)
        connectivity[b].append(a)

    if plot:
        plt.imshow(img, cmap='gray')
        for vid, (x, y) in vertices.items():
            plt.scatter(x, y, c='red', s=40)
        for a, b in edges:
            (x1, y1) = vertices[a]
            (x2, y2) = vertices[b]
            plt.plot([x1, x2], [y1, y2], 'g-', lw=1.6)
        plt.title("Voronoi vertices + connectivity (skeleton + filtered pairwise)")
        plt.show()

    return vertices, connectivity

if __name__ == "__main__":
    image_path = "C:\\Users\\user\\OneDrive\\PHD\\Repositories\\Inverse_Voronoi\\Image Processing\\voronoi_image.png"
    vertices, connectivity = extract_voronoi_graph(image_path, merge_tol=5, vertex_tol=8)
    print("Extracted vertices:", vertices)
    print("Connectivity:", dict(connectivity))
