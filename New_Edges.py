def New_Edges(new_vertices,new_vertices_index,minx,maxx,miny,maxy,vor):
        #CREATING NEW EDGES
    # This part of the code will create a new edges from the intersection points that are found in the previous code cell

    # First look for at which boundary line does the new vertex lie on
    number_of_vertices = len(vor.vertices)
    corners_to_vertices = []
    new_edges_storage = []
    number_of_new_vertices_0 = 0
    number_of_new_vertices_1 = 0 
    number_of_new_vertices_2 = 0
    number_of_new_vertices_3 = 0
    vertex_indices_on_boundary_0 = []
    vertex_indices_on_boundary_1 = []
    vertex_indices_on_boundary_2 = []
    vertex_indices_on_boundary_3 = []

    # Initialize boundary counts so they're always defined
    boundary_0_count = [0, 0, []]
    boundary_1_count = [1, 0, []]
    boundary_2_count = [2, 0, []]
    boundary_3_count = [3, 0, []]

    boundaries_with_intersection = []

    for vertexI in range(len(new_vertices)):
        # There are four possibilities: 1- left bound. 2- right bound. 3- bottom bound. 4- top bound.
        # For the left and right boundries, x coordinate of the new vertice should be equal to minx and maxx respectively.
        # And for the bottom and top boundaries, y coordinate of the new vertice should be equal to miny and maxy respectively.
        # In this perspective first look at the coordinates
        
        if (new_vertices[vertexI][0] == minx): # 1st condition
            # if in the left boundary, the corner points are:
            verticeC = new_vertices[vertexI]
            corner1 = [minx,miny]
            corner2 = [minx,maxy]

            number_of_new_vertices_0 +=1
            vertex_indices_on_boundary_0.append(new_vertices_index[vertexI])
            boundary_0_count = [0,number_of_new_vertices_0,vertex_indices_on_boundary_0]

            #distance to these corners
            distance1 = ((verticeC[0] - corner1[0])**2 + (verticeC[1]-corner1[1])**2)**0.5
            distance2 = ((verticeC[0] - corner2[0])**2 + (verticeC[1]-corner2[1])**2)**0.5

            if (distance1 < distance2): # checking which corner is closer
                corners_to_vertices.append(corner1)
                index_of_corner = number_of_vertices+1
                number_of_vertices += 1
                edge_connection = [new_vertices_index[vertexI] , index_of_corner]
                new_edges_storage.append(edge_connection)
            elif (distance2 < distance1): # checking which corner is closer
                corners_to_vertices.append(corner2) # add to a list
                index_of_corner = number_of_vertices+1 # assign a number of new vertice (right now corner)
                number_of_vertices += 1 # increase the number of vertices by 1
                edge_connection = [new_vertices_index[vertexI] , index_of_corner] # define a new connection
                new_edges_storage.append(edge_connection) # add the new edge to a storage list

            if number_of_new_vertices_0==1:
                boundaries_with_intersection.append(0)

        elif (new_vertices[vertexI][0] == maxx): # 2nd condition
            # if in the right boundary, the corner points are:
            verticeC = new_vertices[vertexI]
            corner1 = [maxx,miny]
            corner2 = [maxx,maxy]

            number_of_new_vertices_1 +=1
            vertex_indices_on_boundary_1.append(new_vertices_index[vertexI])
            boundary_1_count = [1,number_of_new_vertices_1,vertex_indices_on_boundary_1]

            #distance to these corners
            distance1 = ((verticeC[0] - corner1[0])**2 + (verticeC[1]-corner1[1])**2)**0.5
            distance2 = ((verticeC[0] - corner2[0])**2 + (verticeC[1]-corner2[1])**2)**0.5

            if (distance1 < distance2): # checking which corner is closer
                corners_to_vertices.append(corner1)
                index_of_corner = number_of_vertices+1
                number_of_vertices += 1
                edge_connection = [new_vertices_index[vertexI] , index_of_corner]
                new_edges_storage.append(edge_connection)
            elif (distance2 < distance1): # checking which corner is closer
                corners_to_vertices.append(corner2) # add to a list
                index_of_corner = number_of_vertices+1 # assign a number of new vertice (right now corner)
                number_of_vertices += 1 # increase the number of vertices by 1
                edge_connection = [new_vertices_index[vertexI] , index_of_corner] # define a new connection
                new_edges_storage.append(edge_connection) # add the new edge to a storage list

            if number_of_new_vertices_1==1:
                boundaries_with_intersection.append(1)

        elif (new_vertices[vertexI][1] == miny): # 3rd condition
            # if in the bottom boundary, the corner points are:
            verticeC = new_vertices[vertexI]
            corner1 = [minx,miny]
            corner2 = [maxx,miny]

            number_of_new_vertices_2 +=1
            vertex_indices_on_boundary_2.append(new_vertices_index[vertexI])
            boundary_2_count = [2,number_of_new_vertices_2,vertex_indices_on_boundary_2]

            #distance to these corners
            distance1 = ((verticeC[0] - corner1[0])**2 + (verticeC[1]-corner1[1])**2)**0.5
            distance2 = ((verticeC[0] - corner2[0])**2 + (verticeC[1]-corner2[1])**2)**0.5

            if (distance1 < distance2): # checking which corner is closer
                corners_to_vertices.append(corner1)
                index_of_corner = number_of_vertices+1
                number_of_vertices += 1
                edge_connection = [new_vertices_index[vertexI] , index_of_corner]
                new_edges_storage.append(edge_connection)
            elif (distance2 < distance1): # checking which corner is closer
                corners_to_vertices.append(corner2) # add to a list
                index_of_corner = number_of_vertices+1 # assign a number of new vertice (right now corner)
                number_of_vertices += 1 # increase the number of vertices by 1
                edge_connection = [new_vertices_index[vertexI] , index_of_corner] # define a new connection
                new_edges_storage.append(edge_connection) # add the new edge to a storage list

            if number_of_new_vertices_2==1:
                boundaries_with_intersection.append(2)

        elif (new_vertices[vertexI][1] == maxy): # 4th condition
            # if in the top boundary, the corner points are:
            verticeC = new_vertices[vertexI]
            corner1 = [minx,maxy]
            corner2 = [maxx,maxy]

            number_of_new_vertices_3 += 1
            vertex_indices_on_boundary_3.append(new_vertices_index[vertexI])
            boundary_3_count = [3,number_of_new_vertices_3,vertex_indices_on_boundary_3]

            #distance to these corners
            distance1 = ((verticeC[0] - corner1[0])**2 + (verticeC[1]-corner1[1])**2)**0.5
            distance2 = ((verticeC[0] - corner2[0])**2 + (verticeC[1]-corner2[1])**2)**0.5

            if (distance1 < distance2): # checking which corner is closer
                corners_to_vertices.append(corner1)
                index_of_corner = number_of_vertices+1
                number_of_vertices += 1
                edge_connection = [new_vertices_index[vertexI] , index_of_corner]
                new_edges_storage.append(edge_connection)
            elif (distance2 < distance1): # checking which corner is closer
                corners_to_vertices.append(corner2) # add to a list
                index_of_corner = number_of_vertices+1 # assign a number of new vertice (right now corner)
                number_of_vertices += 1 # increase the number of vertices by 1
                edge_connection = [new_vertices_index[vertexI] , index_of_corner] # define a new connection
                new_edges_storage.append(edge_connection) # add the new edge to a storage list
            if number_of_new_vertices_3==1:
                boundaries_with_intersection.append(3)

    boundary_limits = [minx,maxx,miny,maxy]

    if (len(boundaries_with_intersection) < 4):
        sum_boundary_numbers = sum(boundaries_with_intersection)
        missing_boundary_number = 6 - sum_boundary_numbers
        # if missing_boundary_number == 0:
        #     boundary_points_count=[boundary_1_count,boundary_2_count,boundary_3_count]
        # elif missing_boundary_number == 1:
        #     boundary_points_count=[boundary_0_count,boundary_2_count,boundary_3_count]
        # elif missing_boundary_number == 2:
        #     boundary_points_count=[boundary_0_count,boundary_1_count,boundary_3_count]
        # elif missing_boundary_number == 3:
        #     boundary_points_count=[boundary_0_count,boundary_1_count,boundary_2_count]

        # boundary_points_count.append([missing_boundary_number,'empty','no intersection'])
        all_counts = [boundary_0_count, boundary_1_count, boundary_2_count, boundary_3_count]
        boundary_points_count = [all_counts[i] for i in boundaries_with_intersection]
        boundary_points_count.append([missing_boundary_number, 'empty', 'no intersection'])
    
    else:
        boundary_points_count = [boundary_0_count, boundary_1_count, boundary_2_count, boundary_3_count]

    # elif (len(boundaries_with_intersection) >= 4):
    #         boundary_points_count=[boundary_0_count,boundary_1_count,boundary_2_count,boundary_3_count]
        
    # boundary_points_count = [boundary_0_count,boundary_1_count,boundary_2_count,boundary_3_count]


    return boundary_points_count,new_edges_storage,new_vertices