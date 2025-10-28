import numpy as np
import math
from mirror import mirror_point

def center_search_perturbation_linearly_increasing(explicit_voronoi, vertices,perturbation_strength,direction):
    # This function finds the seed points of voronoi seeds iteratively
    # DEFINITION OF INPUTS:
        #   explicit_voronoi    :   dictionary item that stores all the necessary information about voronoi cells
        #   vertices            :   coordinates of vertices
    
    # direction = direction / np.linalg.norm(direction)  # normalize the direction vector
    # # list the indices of vertices according to their x-coordinate
    # sorted_indices = np.argsort(vertices[:, 0])
    # # number of vertices
    # num_vertices = len(vertices)
    # # create linearly spaced peturbation magnitudes with maximum value of 'perturbation_strength'
    # perturbation_magnitudes = np.linspace(0, perturbation_strength, num_vertices)
    # # create a perturbation array
    # delta = np.zeros_like(vertices)
    # for i, index in enumerate(sorted_indices):
    #     delta[index] = perturbation_magnitudes[i] * direction

    # vertices += delta


    # inside_domain_vertices = []
    # for i in range(len(vertices)):
    #     if (0.2 <= vertices[i][0] <= 0.8) and (0 <= vertices[i][1] <= 1):
    #         inside_domain_vertices.append(i)

    # # sort the inside_domain_vertices based on the increasing x-coordinate
    # inside_domain_vertices = sorted(inside_domain_vertices, key=lambda idx: vertices[idx][0])

    # perturbation = np.linspace(0, perturbation_strength, len(inside_domain_vertices))
    # for idx, vertex_index in enumerate(inside_domain_vertices):
    #     print(vertices[vertex_index],perturbation[idx])
    #     delta = perturbation[idx] * direction
    #     vertices[vertex_index] += delta


    # linearly increasing perturbation along x direction
    # delta = np.zeros_like(vertices)
    # x_min, x_max = np.min(vertices[:, 0]), np.max(vertices[:, 0])
    # for i in range(len(vertices)):
    #     perturbation_magnitude = (vertices[i][0] - x_min) / (x_max - x_min) * perturbation_strength
    #     perturbation_magnitude = max(0, min(perturbation_magnitude, perturbation_strength))  # clamp to [0, perturbation_strength]
    #     delta[i] = perturbation_magnitude * direction

    # vertices += delta

#     
# 
# inside_domain_vertices = [
#     i for i in range(len(vertices))
#     if (0.05 <= vertices[i][0] <= 0.95) and (0.05 <= vertices[i][1] <= 0.95)
# ]

#     # linearly increasing perturbation along x
#     x_min, x_max = np.min(vertices[:,0]), np.max(vertices[:,0])
#     for i in inside_domain_vertices:
#         magnitude = ((vertices[i][0] - x_min) / (x_max - x_min)) * perturbation_strength
#         vertices[i] += magnitude * direction

    # for idx, (px, py) in enumerate(vertices):
    #     if px < 1 and py < 1 and px > 0 and py > 0:
    #         vertices[idx] += 0.1*np.array([px/1, py/1])
        



    


    cell_average_center = np.empty((0,2))
    cell_centers = np.empty((0,2))
    for sub_dict in explicit_voronoi:
        sum_coordinates = np.array([0,0])
        for vertice_index in explicit_voronoi[sub_dict]['cell vertices']:
            sum_coordinates = [sum_coordinates[0] + vertices[vertice_index][0],sum_coordinates[1] + vertices[vertice_index][1]]
            # average_of_vertex_coord = [number / len(explicit_voronoi[sub_dict]['cell vertices']) for number in sum_coord_of_vertices]
            # cell_centers = np.vstack([cell_centers,average_of_vertex_coord])
        average_of_vertex_coord = [sum_coordinates[0]/len(explicit_voronoi[sub_dict]['cell vertices']),sum_coordinates[1]/len(explicit_voronoi[sub_dict]['cell vertices'])]
        cell_centers = np.vstack([cell_centers,average_of_vertex_coord])


        

    mean_centers = cell_centers.copy()
    
    old_cell_centers = cell_centers.copy()


    iteration_0 = cell_centers.tolist()




    # for i in range(len(explicit_voronoi)):
    #     cell_index = explicit_voronoi[f'sub_dict_{i}']['seed number']
    #     neighbors = explicit_voronoi[f'sub_dict_{i}']['neighbors'] # find the neighbor cells
    #     mirrored_neighbor_centers = []
    #     for nb in neighbors: # take a neighbors from the 'neighbors' array
    #         for sub_dictionary in explicit_voronoi: # look at each dictionary item to find the corresponding cell
    #             if (explicit_voronoi[sub_dictionary]['seed number'] == nb): # find the dictionary item corresponding to that neighbor
    #                 #     vertices_of_neighbor = explicit_voronoi[sub_dictionary]['cell vertices']
    #                 #     sum_coordinates = [0,0] # initialize the 'sum_coordinates' list
    #                 #     for vertice_index in explicit_voronoi[sub_dictionary]['cell vertices']: 
    #                 #         sum_coordinates = [a + b for a,b in zip(sum_coordinates,vertices[vertice_index])] # add the vertice coordinates
    #                 #     average_of_coordinates = [number / len(vertice_indices) for number in sum_coordinates] # calculate the average of vertices
    #                 cell_number = explicit_voronoi[sub_dictionary]['seed number']
    #                 average_neighbor_cell_center = cell_centers[cell_number,:]
    #                 neighbor_edges = explicit_voronoi[sub_dictionary]['edges'] #edges of the neighbor cell
    #                 for neighbor_edge_index in neighbor_edges:
    #                     for current_cell_edge_index in explicit_voronoi[f'sub_dict_{i}']['edges']:
    #                         if neighbor_edge_index == current_cell_edge_index:
    #                             common_edge = current_cell_edge_index
    #                 # extract the vertices of the common edge
    #                 common_vertex_1 = vertices[common_edge[0]]
    #                 common_vertex_2 = vertices[common_edge[1]]
    #                 # find the mirror of the average of vertice points with respect to the common edge
    #                 mirror_seed_point = mirror_point(average_neighbor_cell_center,common_vertex_2,common_vertex_1)
    #                 mirrored_neighbor_centers.append(mirror_seed_point)

    #     mirrored_neighbor_centers = np.array(mirrored_neighbor_centers)
    #     average_of_mirrored_neighbor_centers = [np.mean(mirrored_neighbor_centers[:,0]),np.mean(mirrored_neighbor_centers[:,1])]
        
    #     cell_centers[i,:] = average_of_mirrored_neighbor_centers
    #     old_cell_centers = cell_centers.copy()
    deviation = 100
        
    iteration = 0

    distance_from_found_to_original = {}
    distace_from_found_to_previous = {}

    cell_centers_iteration = {}
    cell_centers_iteration['iteration_0'] = cell_centers
   
    while (iteration < 300):
        new_cell_centers = np.empty((0,2))
        for i in range(len(explicit_voronoi)):
            cell_index = explicit_voronoi[f'sub_dict_{i}']['seed number']
            neighbors = explicit_voronoi[f'sub_dict_{i}']['neighbors'] # find the neighbor cells
            mirrored_neighbor_centers = []
            for nb in neighbors: # take a neighbors from the 'neighbors' array
                for sub_dictionary in explicit_voronoi: # look at each dictionary item to find the corresponding cell
                    if (explicit_voronoi[sub_dictionary]['seed number'] == nb): # find the dictionary item corresponding to that neighbor
                        cell_number = explicit_voronoi[sub_dictionary]['seed number']
                        neighbor_cell_center = old_cell_centers[cell_number,:]
                        neighbor_edges = explicit_voronoi[sub_dictionary]['edges'] #edges of the neighbor cell
                        for neighbor_edge_index in neighbor_edges:
                            for current_cell_edge_index in explicit_voronoi[f'sub_dict_{i}']['edges']:
                                if neighbor_edge_index == current_cell_edge_index:
                                    common_edge = current_cell_edge_index
                        # extract the vertices of the common edge
                        common_vertex_1 = vertices[common_edge[0]]
                        common_vertex_2 = vertices[common_edge[1]]
                        # find the mirror of the average of vertice points with respect to the common edge
                        mirror_seed_point = mirror_point(neighbor_cell_center,common_vertex_2,common_vertex_1)
                        mirrored_neighbor_centers.append(mirror_seed_point)

            # Filter out None or NaN values
            mirrored_neighbor_centers = [
                pt for pt in mirrored_neighbor_centers 
                if pt is not None and not any(math.isnan(c) for c in pt)
            ]

            if len(mirrored_neighbor_centers) == 0:
                # Fallback: no valid neighbors → keep the old center
                average_point = old_cell_centers[cell_index, :]
                if np.any(np.isnan(average_point)):
                    # final fallback: use polygon centroid if available
                    verts = explicit_voronoi[f'sub_dict_{cell_index}']['cell vertices']
                    if len(verts) > 0:
                        valid_verts = [v for v in verts if v >= 0]
                        average_point = np.mean(vertices[valid_verts], axis=0)
                    else:
                        average_point = np.array([0.5, 0.5])  # or domain midpoint
            else:
                x_average = sum(row[0] for row in mirrored_neighbor_centers) / len(mirrored_neighbor_centers)
                y_average = sum(row[1] for row in mirrored_neighbor_centers) / len(mirrored_neighbor_centers)
                average_point = [x_average, y_average]

            # x_average = sum(row[0] for row in mirrored_neighbor_centers) / len(mirrored_neighbor_centers)
            # y_average = sum(row[1] for row in mirrored_neighbor_centers) / len(mirrored_neighbor_centers)
            # average_point = [x_average, y_average]

            new_cell_centers= np.vstack([new_cell_centers,average_point])

        distance_previous_list = []

        
        for k in range(len(new_cell_centers)):
            dist_previous = ( (new_cell_centers[k][0]-old_cell_centers[k][0])**2 + (new_cell_centers[k][1]-old_cell_centers[k][1])**2 )**(1/2)
            distance_previous_list.append(dist_previous)



        distance_from_found_to_original['iteration_'+str(iteration)] = distance_previous_list

        old_cell_centers = new_cell_centers.copy()

        if iteration == 1:
            iteration_1 = new_cell_centers.tolist()
        elif iteration == 2:
            iteration_2 = new_cell_centers.tolist()

        cell_centers_iteration['iteration_'+str(iteration+1)] = new_cell_centers

        iteration += 1

        mean_change_distance = np.mean(distance_previous_list)
        maximum_change_distance = np.max(distance_previous_list)
        if maximum_change_distance< 1e-15:
            print(f'            Converged after {iteration} iterations')
            break

    first_three_cell_centers = [iteration_0,iteration_1,iteration_2]

        

    return new_cell_centers, cell_centers_iteration #, mean_centers, distance_from_found_to_original, first_three_cell_centers