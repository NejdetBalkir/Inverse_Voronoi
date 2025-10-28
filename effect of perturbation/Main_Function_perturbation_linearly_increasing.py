import numpy as np
import matplotlib.pyplot as plt




def fun_vor_main_perturbation_linearly_increasing(vor,points,perturbation_strength):

    minx = min(vor.vertices[:,0]) - 0.5
    maxx = max(vor.vertices[:,0]) + 0.5
    xValues = [minx, maxx]

    # Defining minimum and maximum y values and store them in an array for future reference
    miny = min(vor.vertices[:,1]) - 0.5
    maxy = max(vor.vertices[:,1]) + 0.5
    yValues = [miny,maxy]

    boundary_limits = [minx,maxx,miny,maxy]

    from Find_Neighbor import find_neighbor
    neighbor_storage, neighbor_vertices_storage = find_neighbor(vor)

    from New_Vertices import New_Vertices
    new_vertices, new_vertices_index, neighbor_vertices_storage_updated = New_Vertices(vor,neighbor_storage,neighbor_vertices_storage)
   

    from New_Edges import New_Edges
    boundary_points_count, new_edges, new_vertices_updated = New_Edges(new_vertices,new_vertices_index,minx,maxx,miny,maxy,vor)

    old_vertices = vor.vertices.copy()
            
    from Boundary_Connections import Boundary_Connections
    MNVI,new_edges,new_vertices_updated = Boundary_Connections(boundary_limits, boundary_points_count, new_vertices, new_vertices_index,old_vertices)


    inside_edges = [sublist[2] for sublist in neighbor_vertices_storage_updated ]

    total_edges = new_edges + inside_edges

    vertices_coords = vor.vertices.copy()
    finite_mask = np.all(np.isfinite(vertices_coords), axis=1)
    Lx = maxx - minx

    for idx in np.where(finite_mask)[0]:
        px, py = vertices_coords[idx]
        if px > 0 and px < 1 and py > 0 and py < 1:
            vertices_coords[idx, :] += perturbation_strength * np.array([px/maxx,py/maxx])

    vertices = np.vstack((vertices_coords, np.array(new_vertices_updated)))


    from ExplicitVoronoi import ExplicitVoronoi
    explicit_voronoi = ExplicitVoronoi(vor,neighbor_vertices_storage_updated,total_edges)

    from center_search_perturbation_linearly_increasing import center_search_perturbation_linearly_increasing
    from center_search import center_search
    cell_centers, cell_centers_iteration = center_search(explicit_voronoi,vertices)

    #rearange vor.points according to vor.point_region
    original_points_rearranged = np.copy(points)



    #find the distance of the found cell centers and the original points
    distance_from_found_to_original = []
    distance = cell_centers-original_points_rearranged
    distance_original_list = np.sqrt(np.sum(distance**2,axis=1))


    distance_from_found_to_previous = {}
    for iteration in range(len(cell_centers_iteration)):
        if iteration > 0:
            distance_previous_list = []
            for k in range(len(cell_centers_iteration[f'iteration_{iteration}'])):
                dist_previous = ( (cell_centers_iteration[f'iteration_{iteration}'][k][0]-cell_centers_iteration[f'iteration_{iteration-1}'][k][0])**2 + (cell_centers_iteration[f'iteration_{iteration}'][k][1]-cell_centers_iteration[f'iteration_{iteration-1}'][k][1])**2 )**(1/2)
                distance_previous_list.append(dist_previous)
            distance_from_found_to_previous[f'iteration_{iteration}'] = distance_previous_list


    return explicit_voronoi, vertices, cell_centers, distance_original_list, distance_from_found_to_previous



