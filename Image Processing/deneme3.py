import cv2
import numpy as np
import matplotlib.pyplot as plt

def process_voronoi_image(image_path, epsilon=1.0):
    """Extract Voronoi vertices + edges from both inner and outer lines."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read {image_path}")

    # Invert so lines = white
    _, binary = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)

    # Find ALL contours (not just external!)
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    vertices = []
    edges = []
    explicit_voronoi = {}

    for i, cnt in enumerate(contours):
        if len(cnt) < 10:  # skip tiny noise
            continue

        # Polygon approximation
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        approx = approx.reshape(-1, 2)

        cell_vertices = []
        cell_edges = []

        for j in range(len(approx)):
            v = tuple(approx[j])
            if list(v) not in vertices:
                vertices.append(list(v))
            v_idx = vertices.index(list(v))
            cell_vertices.append(v_idx)

            # Edge between this and next
            v_next = tuple(approx[(j+1) % len(approx)])
            if list(v_next) not in vertices:
                vertices.append(list(v_next))
            v_next_idx = vertices.index(list(v_next))

            edge = [v_idx, v_next_idx]
            if edge not in edges and edge[::-1] not in edges:
                edges.append(edge)
            cell_edges.append(edge)

        # Use polygon centroid as "seed"
        M = cv2.moments(approx)
        if M["m00"] != 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
        else:
            cx, cy = np.mean(approx, axis=0)

        explicit_voronoi[f"sub_dict_{i}"] = {
            "seed number": i,
            "seed coordinates": [float(cx), float(cy)],
            "neighbors": [],
            "edges": cell_edges,
            "cell vertices": cell_vertices
        }

    # Fill neighbors: if two cells share an edge
    for i, ci in explicit_voronoi.items():
        for j, cj in explicit_voronoi.items():
            if i == j: 
                continue
            if any(set(e1) == set(e2) for e1 in ci["edges"] for e2 in cj["edges"]):
                ci["neighbors"].append(cj["seed number"])

    return explicit_voronoi, vertices, edges, img


def plot_explicit_voronoi(explicit_voronoi, vertices, edges, img):
    fig, ax = plt.subplots()
    ax.imshow(img, cmap="gray")

    verts = np.array(vertices)
    ax.plot(verts[:,0], verts[:,1], "ro", markersize=4, label="Vertices")

    for e in edges:
        x = [verts[e[0],0], verts[e[1],0]]
        y = [verts[e[0],1], verts[e[1],1]]
        ax.plot(x, y, "g-", linewidth=1.5)

    ax.set_title("Reconstructed Explicit Voronoi (with inner vertices)")
    ax.legend()
    plt.show()


# Example usage
if __name__ == "__main__":
    explicit_voronoi, vertices, edges, img = process_voronoi_image(
        "C:\\Users\\user\\OneDrive\\PHD\\Repositories\\Inverse_Voronoi\\Image Processing\\voronoi_image.png", epsilon=2.0
    )
    print("Vertices:", len(vertices))
    print("Edges:", len(edges))
    plot_explicit_voronoi(explicit_voronoi, vertices, edges, img)
