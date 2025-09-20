import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize

def skeletonize_image(img):
    """Convert image to binary skeleton (1-pixel wide lines)."""
    _, binary = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)
    binary = binary // 255
    skel = skeletonize(binary).astype(np.uint8)
    return skel

def get_neighbors(y, x, skel):
    """Return 8-neighbors of a pixel in skeleton."""
    h, w = skel.shape
    neigh = []
    for yy in range(y-1, y+2):
        for xx in range(x-1, x+2):
            if yy == y and xx == x:
                continue
            if 0 <= yy < h and 0 <= xx < w and skel[yy, xx] == 1:
                neigh.append((yy, xx))
    return neigh

def extract_graph(skel):
    """Extract vertices (junctions + boundary) and edges (straight lines between vertices)."""
    h, w = skel.shape
    vertices = []
    vertex_map = {}

    # Step 1: detect true vertices
    for y in range(h):
        for x in range(w):
            if skel[y, x] == 1:
                n = len(get_neighbors(y, x, skel))
                # Junction, endpoint, OR boundary pixel
                if n != 2:
                    vid = len(vertices)
                    vertices.append((x, y))
                    vertex_map[(y, x)] = vid
                # for the boundary box corners
                if x == 0 and y == 0:
                    vid = len(vertices)
                    vertices.append((x, y))
                    vertex_map[(y, x)] = vid
                if x == w-1 and y == 0:
                    vid = len(vertices)
                    vertices.append((x, y))
                    vertex_map[(y, x)] = vid
                if x == w-1 and y == h-1:
                    vid = len(vertices)
                    vertices.append((x, y))
                    vertex_map[(y, x)] = vid
                if x == 0 and y == h-1:
                    vid = len(vertices)
                    vertices.append((x, y))
                    vertex_map[(y, x)] = vid
                

    edges = []
    visited = set()

    # Step 2: walk edges
    for (y, x), vid in vertex_map.items():
        for ny, nx in get_neighbors(y, x, skel):
            if ((y, x), (ny, nx)) in visited or ((ny, nx), (y, x)) in visited:
                continue
            path = [(y, x), (ny, nx)]
            py, px = y, x
            cy, cx = ny, nx
            while (cy, cx) not in vertex_map:
                neigh = get_neighbors(cy, cx, skel)
                neigh = [p for p in neigh if p != (py, px)]
                if not neigh:
                    break
                py, px = cy, cx
                cy, cx = neigh[0]
                path.append((cy, cx))
            if (cy, cx) in vertex_map:
                v2 = vertex_map[(cy, cx)]
                edges.append([vid, v2])
            visited.add(((y, x), (ny, nx)))
    return vertices, edges


def process_voronoi_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read {image_path}")

    skel = skeletonize_image(img)
    vertices, edges = extract_graph(skel)

    explicit_voronoi = {
        "vertices": [list(v) for v in vertices],
        "edges": edges,
    }

    return explicit_voronoi, img

def add_boundary_intersections(explicit_voronoi, img_shape):
    """Add intersection points where Voronoi edges meet the bounding box."""
    h, w = img_shape
    verts = explicit_voronoi["vertices"]
    edges = explicit_voronoi["edges"]

    new_vertices = []
    new_edges = []

    for i, (x, y) in enumerate(verts):
        # Check if point lies exactly on border
        if x == 0 or x == w-1 or y == 0 or y == h-1:
            # Already stored, skip
            continue
        # Check each edge: if it crosses border, add intersection
        for e in edges:
            v1, v2 = verts[e[0]], verts[e[1]]
            (x1, y1), (x2, y2) = v1, v2
            # Check if segment crosses bounding box edges
            if (min(x1, x2) <= 0 <= max(x1, x2)) or (min(x1, x2) <= w-1 <= max(x1, x2)) or \
               (min(y1, y2) <= 0 <= max(y1, y2)) or (min(y1, y2) <= h-1 <= max(y1, y2)):
                # Approx intersection by clamping to border
                ix = min(max(0, int((x1+x2)/2)), w-1)
                iy = min(max(0, int((y1+y2)/2)), h-1)
                if [ix, iy] not in verts and [ix, iy] not in new_vertices:
                    new_vertices.append([ix, iy])
                    new_edges.append([e[0], len(verts) + len(new_vertices) - 1])
                    new_edges.append([e[1], len(verts) + len(new_vertices) - 1])

    # Update explicit_voronoi
    verts.extend(new_vertices)
    edges.extend(new_edges)
    explicit_voronoi["vertices"] = verts
    explicit_voronoi["edges"] = edges
    return explicit_voronoi

def plot_explicit_voronoi(explicit_voronoi, img):
    fig, ax = plt.subplots()
    ax.imshow(img, cmap="gray")

    verts = np.array(explicit_voronoi["vertices"])
    ax.plot(verts[:,0], verts[:,1], "ro", markersize=6, label="Vertices")

    for e in explicit_voronoi["edges"]:
        x = [verts[e[0],0], verts[e[1],0]]
        y = [verts[e[0],1], verts[e[1],1]]
        ax.plot(x, y, "g-", linewidth=2)

    ax.set_title("Clean Explicit Voronoi")
    ax.legend()
    plt.show()

    
def line_intersects_box(p1, p2, w, h):
    """Return intersection points of segment p1-p2 with image box [0,w-1]x[0,h-1]."""
    intersections = []
    (x1, y1), (x2, y2) = p1, p2

    # avoid division by zero
    dx, dy = x2 - x1, y2 - y1

    # Check each boundary line (x=0, x=w-1, y=0, y=h-1)
    if dx != 0:
        # left edge (x=0)
        t = -x1 / dx
        y = y1 + t * dy
        if 0 <= t <= 1 and 0 <= y <= h-1:
            intersections.append((0, int(round(y))))
        # right edge (x=w-1)
        t = (w-1 - x1) / dx
        y = y1 + t * dy
        if 0 <= t <= 1 and 0 <= y <= h-1:
            intersections.append((w-1, int(round(y))))
    if dy != 0:
        # top edge (y=0)
        t = -y1 / dy
        x = x1 + t * dx
        if 0 <= t <= 1 and 0 <= x <= w-1:
            intersections.append((int(round(x)), 0))
        # bottom edge (y=h-1)
        t = (h-1 - y1) / dy
        x = x1 + t * dx
        if 0 <= t <= 1 and 0 <= x <= w-1:
            intersections.append((int(round(x)), h-1))

    return intersections


def add_boundary_intersections(explicit_voronoi, img_shape):
    """Compute exact intersections with bounding box and add them as vertices."""
    h, w = img_shape
    verts = explicit_voronoi["vertices"]
    edges = explicit_voronoi["edges"]

    new_edges = []
    for e in edges:
        v1, v2 = verts[e[0]], verts[e[1]]
        inters = line_intersects_box(v1, v2, w, h)
        if inters:
            for pt in inters:
                if list(pt) not in verts:
                    verts.append(list(pt))
                vid = verts.index(list(pt))
                new_edges.append([e[0], vid])
                new_edges.append([e[1], vid])
        else:
            new_edges.append(e)

    explicit_voronoi["vertices"] = verts
    explicit_voronoi["edges"] = new_edges
    return explicit_voronoi

# Example
if __name__ == "__main__":
    explicit_voronoi, img = process_voronoi_image("C:\\Users\\user\\OneDrive\\PHD\\Repositories\\Inverse_Voronoi\\Image Processing\\voronoi_image.png")
    explicit_voronoi = add_boundary_intersections(explicit_voronoi, img.shape)
    print("Vertices:", len(explicit_voronoi["vertices"]))
    print("Edges:", len(explicit_voronoi["edges"]))
    plot_explicit_voronoi(explicit_voronoi, img)
