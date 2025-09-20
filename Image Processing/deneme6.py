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

# ---------- find intersection vertices ----------
NB8 = [(-1,-1),(-1,0),(-1,1),
       (0,-1),        (0,1),
       (1,-1),(1,0),(1,1)]

vertices = []
h, w = skeleton.shape

for y in range(1,h-1):
    for x in range(1,w-1):
        if skeleton[y,x] > 0:
            # count neighbors
            n = 0
            for dy,dx in NB8:
                if skeleton[y+dy, x+dx] > 0:
                    n += 1
            # keep only true junctions
            if n >= 3:
                vertices.append((x,y))

# cluster nearby points (merge close vertices)
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
plt.title("Detected Voronoi vertices after skeletonization")
plt.show()
