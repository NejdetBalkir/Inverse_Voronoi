# generate a voronoi diagram
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi

num_points = 10
points = np.random.rand(num_points, 2)

# Create Voronoi diagram
vor = Voronoi(points)

fig,ax = plt.subplots(figsize=(6,6))

 # Plot Voronoi vertices
ax.plot(vor.vertices[:,0], vor.vertices[:,1], 'ro', label="Vertices")

# Plot Voronoi edges (ridges)
for ridge in vor.ridge_vertices:
    ridge = np.asarray(ridge)
    if np.all(ridge >= 0):  # only finite ridges
        ax.plot(vor.vertices[ridge,0], vor.vertices[ridge,1], 'b-')

ax.set_title("Voronoi Diagram")
ax.legend()
plt.show()