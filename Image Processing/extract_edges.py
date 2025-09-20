import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from collections import defaultdict
from scipy.spatial import distance
from scipy.ndimage import label

def extract_voronoi_graph(image_path, plot=True, merge_tol=8, vertex_tol=1):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    # skeletonization
    _, binary = cv2.threshold(img, 200, 1, cv2.THRESH_BINARY_INV)
    skeleton = skeletonize(binary).astype(np.uint8)
    h, w = skeleton.shape

    # 8-neighborhood
    NB8 = [(-1,-1),(-1,0),(-1,1),
           (0,-1),        (0,1),
           (1,-1),(1,0),(1,1)]

    # --- vertex detection ---
    raw_vertices = []
    for y in range(1,h-1):
        for x in range(1,w-1):
            if skeleton[y,x] > 0:
                n = sum(skeleton[y+dy, x+dx] > 0 for dy,dx in NB8)
                if n >= 3 or n == 1:
                    raw_vertices.append((x,y))
                if x in [0,w-1] or y in [0,h-1]:
                    raw_vertices.append((x,y))

    # add extreme skeleton points
    ys, xs = np.where(skeleton > 0)
    if len(xs) > 0:
        raw_vertices.extend([
            (xs.min(), ys.min()),
            (xs.max(), ys.max()),
            (xs.max(), ys.min()),
            (xs.min(), ys.max())
        ])

    # cluster close detections (ilk kaba filtreleme)
    criteria = 1
    unique_vertices = []
    for v in raw_vertices:
        if not any(np.linalg.norm(np.array(v)-np.array(u)) < criteria for u in unique_vertices):
            unique_vertices.append(tuple(v))

    # --- MERGE çok yakın vertex'ler ---
    merged = []
    used = set()
    for i, vi in enumerate(unique_vertices):
        if i in used:
            continue
        cluster = [vi]
        for j, vj in enumerate(unique_vertices):
            if j <= i or j in used:
                continue
            if np.linalg.norm(np.array(vi) - np.array(vj)) < merge_tol:
                cluster.append(vj)
                used.add(j)
        cx = int(np.mean([p[0] for p in cluster]))
        cy = int(np.mean([p[1] for p in cluster]))
        merged.append((cx, cy))
        used.add(i)

    vertices = {i:v for i,v in enumerate(merged)}

    # --- edge detection via connected components ---
    connectivity = defaultdict(list)
    labeled, ncomponents = label(skeleton)

    for comp_id in range(1, ncomponents+1):
        ys, xs = np.where(labeled == comp_id)
        pixels = list(zip(xs, ys))
        if not pixels:
            continue

        dmap = []
        comp_vertices = []
        for vid, vp in vertices.items():
            min_dist = min(distance.euclidean(p, vp) for p in pixels)
            dmap.append((min_dist, vid))
            if min_dist < vertex_tol:
                comp_vertices.append(vid)

        # en yakın 2 vertex'i de ekle
        dmap.sort()
        closest_two = [dmap[0][1], dmap[1][1]] if len(dmap) >= 2 else []
        comp_vertices.extend(closest_two)
        comp_vertices = list(set(comp_vertices))

        # burada düzeltme: eğer >= 2 vertex varsa hepsini pairwise bağla
        if len(comp_vertices) >= 2:
            for i in range(len(comp_vertices)):
                for j in range(i+1, len(comp_vertices)):
                    v1, v2 = comp_vertices[i], comp_vertices[j]
                    if v2 not in connectivity[v1]:
                        connectivity[v1].append(v2)
                    if v1 not in connectivity[v2]:
                        connectivity[v2].append(v1)

    # --- plot ---
    if plot:
        plt.imshow(img, cmap='gray')
        for vid,(x,y) in vertices.items():
            plt.scatter(x,y,c='red',s=40)
            for nb in connectivity[vid]:
                xn,yn = vertices[nb]
                plt.plot([x,xn],[y,yn],'g-',lw=1)
        plt.title("Voronoi vertices + connectivity (multi-vertex full pairwise)")
        plt.show()

    return vertices, connectivity

# Example usage
if __name__ == "__main__":
    image_path = "C:\\Users\\user\\OneDrive\\PHD\\Repositories\\Inverse_Voronoi\\Image Processing\\voronoi_image.png"
    vertices, connectivity = extract_voronoi_graph(image_path, merge_tol=5, vertex_tol=8)
    print("Extracted vertices:", vertices)
    print("Connectivity:", dict(connectivity))
