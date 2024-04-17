def ExplicitVoronoi(vor,neighbor_vertices_storage_updated,total_edges):
    seed_neighbor_info = {} # initialize a dictionary and np array to store all the neighbor storage
    explicit_voronoi = {}
    for seed in range(len(vor.points)): # iterate over the indices of points
        seed_neighbors = [] # initialize the list. This list will contain the neighbors of each considered cell
        seed_edges = []
        # seed_neighbors.append(seed) # the first element of each list is the base cell seed
        for j in range(len(neighbor_vertices_storage_updated)): # iterate over neighborhood storage
        
            if (seed == neighbor_vertices_storage_updated[j][0]): # look at all neighbor information's first position that contain the same seed that we consider
                seed_neighbors.append(neighbor_vertices_storage_updated[j][1]) # add the seed information at the second position
                seed_edges.append(neighbor_vertices_storage_updated[j][2])
            elif (seed == neighbor_vertices_storage_updated[j][1]): # look at all neighbor information's second position that contain the same seed that we consider
                seed_neighbors.append(neighbor_vertices_storage_updated[j][0]) # add this seed point information at the first position
                seed_edges.append(neighbor_vertices_storage_updated[j][2])
        # seed_neighbors = list(set(seed_neighbors)) # delete the duplicate elements
        # seed_neighbor_info.append(seed_neighbors)
        sub_dict = {'seed number': seed , 'seed coordinates': vor.points[seed].tolist(), 'neighbors': seed_neighbors, 'edges': seed_edges }
        explicit_voronoi[f'sub_dict_{seed}'] = sub_dict



    ############ ADDING THE NEW CONNECTIONS ON THE BOUNDARY CORNERS ###########

    number_of_old_vertices = len(vor.vertices)


    for sub_dictionary in explicit_voronoi:
        cell_edges = explicit_voronoi[sub_dictionary]['edges']
        number_of_edges = len(cell_edges)

        boundary_vertices = [number for sublist in cell_edges for number in sublist if number > (number_of_old_vertices-1) ]

        for i in range(len(boundary_vertices)):
            vertex1 = boundary_vertices[0]
            vertex1_connections = []
            vertex2 = boundary_vertices[1]
            vertex2_connections = []
            for j in range(len(total_edges)):
                if (total_edges[j][0] == vertex1):
                    vertex1_connections.append(total_edges[j][1])
                elif (total_edges[j][1] == vertex1):
                    vertex1_connections.append(total_edges[j][0])
                elif (total_edges[j][0] == vertex2):
                    vertex2_connections.append(total_edges[j][1])
                elif (total_edges[j][1] == vertex2):
                    vertex2_connections.append(total_edges[j][0])
            
            same_connection = list(set(vertex1_connections) & set(vertex2_connections))

            if (len(same_connection) == 0):
                explicit_voronoi[sub_dictionary]['edges'].append([vertex1,vertex2])
            elif (len(same_connection) != 0):
                explicit_voronoi[sub_dictionary]['edges'].append([vertex1,same_connection[0]])
                explicit_voronoi[sub_dictionary]['edges'].append([vertex2,same_connection[0]])

        unique_numbers = set()
        for sublist in explicit_voronoi[sub_dictionary]['edges']:
            unique_numbers.update(sublist)
        cell_edges = list(unique_numbers)
        explicit_voronoi[sub_dictionary]['cell vertices'] = cell_edges


    return explicit_voronoi