import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize

# ---------- load and preprocess ----------
img = cv2.imread("C:\\Users\\user\\OneDrive\\PHD\\Repositories\\Inverse_Voronoi\\Image Processing\\voronoi_image.png", cv2.IMREAD_GRAYSCALE)

# threshold to binary (invert so lines=1, background=0)
_, binary = cv2.threshold(img, 200, 1, cv2.THRESH_BINARY_INV)

# skeletonize (thin to 1 pixel width)
skeleton = skeletonize(binary).astype(np.uint8)

# ---------- find vertices ----------
NB8 = [(-1,-1),(-1,0),(-1,1),
       (0,-1),        (0,1),
       (1,-1),(1,0),(1,1)]

vertices = []
h, w = skeleton.shape

for y in range(1,h-1):
    for x in range(1,w-1):
        if skeleton[y,x] > 0:
            # count neighbors
            n = sum(skeleton[y+dy, x+dx] > 0 for dy,dx in NB8)

            # junction or endpoint
            if n >= 3 or n == 1:
                vertices.append((x,y))

            # boundary intersection (touches frame)
            if x == 0 or x == w-1 or y == 0 or y == h-1:
                vertices.append((x,y))

# ---------- add extreme skeleton points ----------
ys, xs = np.where(skeleton > 0)
if len(xs) > 0:
    minx_point = (xs.min(), ys.min())
    maxx_point = (xs.max(), ys.max())
    miny_point = (xs.max(), ys.min())
    maxy_point = (xs.min(), ys.max())
    forced_corners = [minx_point, maxx_point, miny_point, maxy_point]
    vertices.extend(forced_corners)

# ---------- cluster nearby points ----------
vertices = np.array(vertices)
criteria = 5  # px tolerance
unique_vertices = []
for v in vertices:
    if not any(np.linalg.norm(v - np.array(u)) < criteria for u in unique_vertices):
        unique_vertices.append(tuple(v))

print("Final Voronoi vertices:", unique_vertices)

# ---------- plot ----------
plt.imshow(img, cmap='gray')
if unique_vertices:
    vx, vy = zip(*unique_vertices)
    plt.scatter(vx, vy, c='red', s=40)
plt.title("Detected Voronoi vertices (junctions + boundary + extreme corners)")
plt.show()
