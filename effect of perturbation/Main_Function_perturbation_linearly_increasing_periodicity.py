# import numpy as np
# import matplotlib.pyplot as plt

# def fun_vor_main_perturbation_linearly_increasing(vor,points,perturbation_strength,direction):

#     minx = min(vor.vertices[:,0]) - 0.1
#     maxx = max(vor.vertices[:,0]) + 0.1
#     xValues = [minx, maxx]

#     # Defining minimum and maximum y values and store them in an array for future reference
#     miny = min(vor.vertices[:,1]) - 0.1
#     maxy = max(vor.vertices[:,1]) + 0.1
#     yValues = [miny,maxy]

#     boundary_limits = [minx,maxx,miny,maxy]

#     from Find_Neighbor import find_neighbor
#     neighbor_storage, neighbor_vertices_storage = find_neighbor(vor)

#     from New_Vertices import New_Vertices
#     new_vertices, new_vertices_index, neighbor_vertices_storage_updated = New_Vertices(vor,neighbor_storage,neighbor_vertices_storage)
   

#     from New_Edges import New_Edges
#     boundary_points_count, new_edges, new_vertices_updated = New_Edges(new_vertices,new_vertices_index,minx,maxx,miny,maxy,vor)

#     old_vertices = vor.vertices.copy()
            
#     from Boundary_Connections import Boundary_Connections
#     MNVI,new_edges,new_vertices_updated = Boundary_Connections(boundary_limits, boundary_points_count, new_vertices, new_vertices_index,old_vertices)

#     new_edges_reduced = [[x - len(vor.vertices) for x in sublist] for sublist in new_edges]

#     inside_edges = [sublist[2] for sublist in neighbor_vertices_storage_updated ]

#     total_edges = new_edges + inside_edges

#     Lx = maxx - minx
#     Ly = maxy - miny
#     vertices_coords = vor.vertices
#     for idx, (px, py) in enumerate(vor.vertices):
#         vertices_coords[idx,0] += 0.1 * px/maxx

#     vertices = np.vstack((vertices_coords, np.array(new_vertices_updated)))


#     from ExplicitVoronoi import ExplicitVoronoi
#     explicit_voronoi = ExplicitVoronoi(vor,neighbor_vertices_storage_updated,total_edges)

#     from center_search_perturbation_linearly_increasing import center_search_perturbation_linearly_increasing
#     from center_search import center_search
#     cell_centers, cell_centers_iteration = center_search(explicit_voronoi,vertices)
#     # cell_centers, cell_centers_iteration = center_search_perturbation_linearly_increasing(explicit_voronoi,vertices,perturbation_strength,direction)

#     #rearrange vor.points according to vor.point_region
#     # print('vor.points:',vor.points)
#     # print('points:',points)
#     # print('vor.point_region:',vor.point_region)

#     #rearange vor.points according to vor.point_region
#     original_points_rearranged = np.copy(points)
#     # for index in  range(len(vor.point_region)):
#     #     original_points_rearranged[vor.point_region[index]] = points[index]


#     #find the distance of the found cell centers and the original points
#     distance_from_found_to_original = []
#     distance = cell_centers-original_points_rearranged
#     distance_original_list = np.sqrt(np.sum(distance**2,axis=1))

#     # for iteration in range(len(cell_centers_iteration)):
#     #     distance_original_list = []
#     #     for k in range(len(cell_centers_iteration[f'iteration_{iteration}'])):
#     #         dist_original = ( (cell_centers_iteration[f'iteration_{iteration}'][k][0]-original_points_rearranged[k][0])**2 + (cell_centers_iteration[f'iteration_{iteration}'][k][1]-original_points_rearranged[k][1])**2 )**(1/2)
#     #         distance_original_list.append(dist_original)
#     #     distance_from_found_to_original[f'iteration_{iteration}'] = distance_original_list

#     #find the distance of the found cell centers in each iteration to the previous cell centers, starting from the second iteration
#     distance_from_found_to_previous = {}
#     for iteration in range(len(cell_centers_iteration)):
#         if iteration > 0:
#             distance_previous_list = []
#             for k in range(len(cell_centers_iteration[f'iteration_{iteration}'])):
#                 dist_previous = ( (cell_centers_iteration[f'iteration_{iteration}'][k][0]-cell_centers_iteration[f'iteration_{iteration-1}'][k][0])**2 + (cell_centers_iteration[f'iteration_{iteration}'][k][1]-cell_centers_iteration[f'iteration_{iteration-1}'][k][1])**2 )**(1/2)
#                 distance_previous_list.append(dist_previous)
#             distance_from_found_to_previous[f'iteration_{iteration}'] = distance_previous_list


  


#     # fig, ax = plt.subplots() # initialize the plot

#     # ax.plot(list(zip(*vertices))[0],list(zip(*vertices))[1],'o',color='red')

#     # for lineI in range(len(new_edges_reduced)):
#     #     firstP = new_edges_reduced[lineI][0]
#     #     secondP = new_edges_reduced[lineI][1]

#     #     linex = [new_vertices_updated[firstP][0],new_vertices_updated[secondP][0]]
#     #     liney = [new_vertices_updated[firstP][1],new_vertices_updated[secondP][1]]

#     #     ax.plot(linex,liney,'-',color='black')

#     # for lineI in range(len(inside_edges)):
#     #     firstP = inside_edges[lineI][0]
#     #     secondP = inside_edges[lineI][1]

#     #     linex = [vertices[firstP][0],vertices[secondP][0]]
#     #     liney = [vertices[firstP][1],vertices[secondP][1]]

#     #     ax.plot(linex,liney,'-',color='black')

#     # for index, (x, y) in enumerate(vertices):
#     #     ax.text(x, y, str(index), color='blue', fontsize=12)


#     # for index, (x,y) in enumerate(vor.points):
#     #     region = vor.point_region[index]
#     #     xy_text = f"[{x:.2f}, {y:.2f}]"
#     #     ax.plot(x,y,marker='o',markersize=5,color='green')
#     #     ax.text(x,y,str(region),color='black')
#     #     ax.text(x,y,xy_text,color='black',horizontalalignment='right',verticalalignment='top')
#     # for findex, (x,y) in enumerate(cell_centers):
#     #     ax.plot(x,y,marker='o',markersize=2,color='red')
#     #     xy_text = f"[{x:.2f}, {y:.2f}]"
#     #     ax.text(x,y,str(findex),color='red',horizontalalignment='left',verticalalignment='top')
#     #     ax.text(x,y,xy_text,color='red',horizontalalignment='left',verticalalignment='bottom')


#     # plt.xlim((minx-0.1,maxx+0.1))
#     # plt.ylim((miny-0.1,maxy+0.1))

#     # plt.show()
    

#     # return explicit_voronoi, vertices, cell_centers, mean_centers, distance_from_found_to_original, distance_from_found_to_previous,  first_three_cell_centers
#     return explicit_voronoi, vertices, cell_centers, distance_original_list, distance_from_found_to_previous



import numpy as np
from scipy.spatial import Voronoi
import matplotlib.pyplot as plt


def fun_vor_main_perturbation_linearly_increasing(points,
                                                  perturbation_strength=0.1,
                                                  direction=np.array([1.0, 0.0]),
                                                  buffer=0.1):
    """
    Stable Voronoi–perturbation function.
    Prevents runaway vertices, ensures bounded tessellation, and
    preserves compatibility with ExplicitVoronoi and center_search.

    Parameters
    ----------
    points : (N,2) ndarray
        Original seed coordinates.
    perturbation_strength : float
        Maximum perturbation magnitude (fraction of domain width).
    direction : (2,) ndarray
        Perturbation direction (e.g. np.array([1,0])).
    buffer : float
        Thickness of ghost layer around the domain.

    Returns
    -------
    explicit_voronoi : dict
        Explicit Voronoi structure.
    vertices : ndarray
        Combined vertex array (original + boundary vertices).
    cell_centers : ndarray
        Recovered cell centers after iteration.
    distance_original_list : ndarray
        Per-seed RMS distance to original points.
    distance_from_found_to_previous : dict
        Per-iteration convergence record.
    """

    # ------------------------------------------------------------
    # STEP 1. Add ghost (buffer) points around domain
    # ------------------------------------------------------------
    xmin, ymin = np.min(points, axis=0) - buffer
    xmax, ymax = np.max(points, axis=0) + buffer

    ghost_x = np.linspace(xmin, xmax, 10)
    ghost_y = np.linspace(ymin, ymax, 10)
    ghosts = np.array([
        [x, y]
        for x in ghost_x for y in ghost_y
        if (x < np.min(points[:, 0]) or x > np.max(points[:, 0]) or
            y < np.min(points[:, 1]) or y > np.max(points[:, 1]))
    ])

    all_points = np.vstack([points, ghosts])
    N_real = len(points)

    # ------------------------------------------------------------
    # STEP 2. Build bounded Voronoi
    # ------------------------------------------------------------
    vor = Voronoi(all_points, qhull_options="Qbb Qx Qc A0.9999999")

    # geometric bounds
    minx, maxx = np.min(vor.vertices[:, 0]), np.max(vor.vertices[:, 0])
    miny, maxy = np.min(vor.vertices[:, 1]), np.max(vor.vertices[:, 1])

    # ------------------------------------------------------------
    # STEP 3. Perturb finite vertices safely
    # ------------------------------------------------------------
    vertices_coords = vor.vertices.copy()
    finite_mask = np.all(np.isfinite(vertices_coords), axis=1)
    Lx = maxx - minx

    for idx in np.where(finite_mask)[0]:
        px, py = vertices_coords[idx]
        magnitude = (px - minx) / Lx * perturbation_strength
        vertices_coords[idx, :] += magnitude * direction

    # ------------------------------------------------------------
    # STEP 4. Build explicit Voronoi data
    # ------------------------------------------------------------
    from Find_Neighbor import find_neighbor
    neighbor_storage, neighbor_vertices_storage = find_neighbor(vor)

    from New_Vertices import New_Vertices
    new_vertices, new_vertices_index, neighbor_vertices_storage_updated = \
        New_Vertices(vor, neighbor_storage, neighbor_vertices_storage)

    from New_Edges import New_Edges
    boundary_points_count, new_edges, new_vertices_updated = \
        New_Edges(new_vertices, new_vertices_index, minx, maxx, miny, maxy, vor)

    old_vertices = vor.vertices.copy()

    from Boundary_Connections import Boundary_Connections
    MNVI, new_edges, new_vertices_updated = Boundary_Connections(
        [minx, maxx, miny, maxy],
        boundary_points_count,
        new_vertices,
        new_vertices_index,
        old_vertices
    )

    # combine inside + boundary edges
    inside_edges = [sublist[2] for sublist in neighbor_vertices_storage_updated]
    total_edges = new_edges + inside_edges

    from ExplicitVoronoi import ExplicitVoronoi
    explicit_voronoi = ExplicitVoronoi(vor, neighbor_vertices_storage_updated, total_edges)

    # ------------------------------------------------------------
    # STEP 5. Combine vertices for consistent indexing
    # ------------------------------------------------------------
    vertices = np.vstack((vertices_coords, np.array(new_vertices_updated)))

    # ------------------------------------------------------------
    # STEP 6. Iterative center search
    # ------------------------------------------------------------
    from center_search import center_search
    cell_centers, cell_centers_iteration = center_search(explicit_voronoi, vertices)

    # ------------------------------------------------------------
    # STEP 7. Distance metrics (fixed for ghost points)
    # ------------------------------------------------------------
    # Only compare true seeds, not ghost points
    distance = cell_centers[:N_real] - points
    distance_original_list = np.sqrt(np.sum(distance ** 2, axis=1))

    distance_from_found_to_previous = {}
    for iteration in range(1, len(cell_centers_iteration)):
        prev = cell_centers_iteration[f'iteration_{iteration-1}']
        curr = cell_centers_iteration[f'iteration_{iteration}']
        # match lengths in case ghosts appear in later iterations
        nmin = min(len(prev), len(curr), N_real)
        dist_prev = np.sqrt(np.sum((curr[:nmin] - prev[:nmin]) ** 2, axis=1))
        distance_from_found_to_previous[f'iteration_{iteration}'] = dist_prev.tolist()

    # ------------------------------------------------------------
    # Optional visualization (debug)
    # ------------------------------------------------------------
    # fig, ax = plt.subplots(figsize=(6, 6))
    # from scipy.spatial import voronoi_plot_2d
    # voronoi_plot_2d(vor, ax=ax, show_points=False, show_vertices=False,
    #                 line_colors='black', line_width=0.3, line_alpha=0.5)
    # ax.scatter(points[:, 0], points[:, 1], c='red', s=10, label='Original seeds')
    # ax.scatter(cell_centers[:N_real, 0], cell_centers[:N_real, 1],
    #            c='blue', s=10, label='Recovered centers')
    # ax.legend()
    # ax.set_aspect('equal')
    # plt.show()

    # ------------------------------------------------------------
    return explicit_voronoi, vertices, cell_centers, distance_original_list, distance_from_found_to_previous,all_points


