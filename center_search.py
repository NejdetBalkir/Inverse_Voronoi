import numpy as np
import math
from mirror import mirror_point

def center_search(explicit_voronoi, vertices):
    # This function finds the seed points of voronoi seeds iteratively
    # DEFINITION OF INPUTS:
        #   explicit_voronoi    :   dictionary item that stores all the necessary information about voronoi cells
        #   vertices            :   coordinates of vertices
    
    # take a cell from the explicit_voronoi dictionary

    

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
   
    while (iteration < 100):
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
            x_average = sum(row[0] for row in mirrored_neighbor_centers)/len(mirrored_neighbor_centers)
            y_average = sum(row[1] for row in mirrored_neighbor_centers)/len(mirrored_neighbor_centers)
            average_point = [x_average,y_average]
            new_cell_centers= np.vstack([new_cell_centers,average_point])
        distance = []

        
        for k in range(len(new_cell_centers)):
            dist = ( (new_cell_centers[k][0]-old_cell_centers[k][0])**2 + (new_cell_centers[k][1]-old_cell_centers[k][1])**2 )**(1/2)
            distance.append(dist)

        distance_from_found_to_original['iteration_'+str(iteration)] = distance

        deviation = max(distance)
        old_cell_centers = new_cell_centers.copy()

        if iteration == 1:
            iteration_1 = new_cell_centers.tolist()
        elif iteration == 2:
            iteration_2 = new_cell_centers.tolist()

        iteration += 1


    first_three_cell_centers = [iteration_0,iteration_1,iteration_2]

        

    return new_cell_centers, mean_centers, distance_from_found_to_original, first_three_cell_centers