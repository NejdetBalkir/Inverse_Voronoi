import numpy as np

def Boundary_Connections(boundary_limits, boundary_points_count, new_vertices, new_vertices_index,old_vertices):
    # boundary_limits: [minx,maxx,miny,maxy]
    # boundary_points_count: [[IOB,NoVOB,[IoVs]]]
    #   IOB     : index of boundary
    #   NoVOB   : number of points on boundary
    #   IoPs    : index of vertices on boundary
    #   new_vertices  : newly added vertices on boundary
    #   new_vertices_index  : index of the new vertices
    #   new_edges_storage   : stores which vertices or corners are linked to each other
    NONV = len(new_vertices) #number of new vertices
    MNVI = max(new_vertices_index) #maximum number vertex index
    new_edges = []
    new_vertices_updated = new_vertices.copy()

    corner_bottom_left_index = MNVI + 1
    corner_bottom_left = [boundary_limits[0],boundary_limits[2]]
    new_vertices_updated.append(corner_bottom_left)

    corner_top_left_index = MNVI + 2
    corner_top_left = [boundary_limits[0],boundary_limits[3]]
    new_vertices_updated.append(corner_top_left)

    corner_bottom_right_index = MNVI + 3   
    corner_bottom_right = [boundary_limits[1],boundary_limits[2]]
    new_vertices_updated.append(corner_bottom_right)

    corner_top_right_index = MNVI + 4
    corner_top_right = [boundary_limits[1],boundary_limits[3]]
    new_vertices_updated.append(corner_top_right)

    for BI in range(len(boundary_points_count)): # BI: boundary index
        boundary_number = boundary_points_count[BI][0]
        NoVOB = boundary_points_count[1]
        if (boundary_number==0) or (boundary_number==1) : # for vertical boundaries

            if (boundary_points_count[BI][2] != 'no intersection'):

                vertex_indices = boundary_points_count[BI][2]
                vertex_indices_adjusted = [x- (len(old_vertices)) for x in vertex_indices]
                vertex_coordinates = [new_vertices[i] for i in vertex_indices_adjusted]
                if (len(vertex_indices) >= 2):
                    combined_list = list(zip(vertex_indices,vertex_coordinates))
                    sorted_combined_list = sorted(combined_list,key=lambda x:x[1])
                    sorted_indices , sorted_coordinates = zip(*sorted_combined_list)
                    if (boundary_number == 0):
                        edges = [[corner_bottom_left_index,sorted_indices[0]],[sorted_indices[0],sorted_indices[1]],
                                    [sorted_indices[1],corner_top_left_index]]
                        new_edges.append(edges)
                    elif (boundary_number == 1):
                        edges = [[corner_bottom_right_index,sorted_indices[0]],[sorted_indices[0],sorted_indices[1]],
                                    [sorted_indices[1],corner_top_right_index]]
                        new_edges.append(edges)
                
                elif (len(vertex_indices)==1):
                    if (boundary_number == 0):
                        edges = [ [corner_bottom_left_index,vertex_indices[0]] , [vertex_indices[0],corner_top_left_index] ]
                        new_edges.append(edges)
                    elif (boundary_number == 1):
                        edges = [ [corner_bottom_right_index,vertex_indices[0]] , [vertex_indices[0],corner_top_right_index] ]
                        new_edges.append(edges)

            elif (boundary_points_count[BI][2] == 'no intersection'):
                if (boundary_number == 0):
                        edges = [ [corner_bottom_left_index,corner_top_left_index] ]
                        new_edges.append(edges)
                elif (boundary_number == 1):
                    edges = [ [corner_bottom_right_index,corner_top_right_index] ]
                    new_edges.append(edges)
                

        elif (boundary_number==2) or (boundary_number==3) : # for vertical boundaries

            if (boundary_points_count[BI][2] != 'no intersection'):

                vertex_indices = boundary_points_count[BI][2]
                vertex_indices_adjusted = [x- (len(old_vertices)) for x in vertex_indices]
                vertex_coordinates = [new_vertices[i] for i in vertex_indices_adjusted]
                if (len(vertex_indices) >= 2):
                    combined_list = list(zip(vertex_indices,vertex_coordinates))
                    sorted_combined_list = sorted(combined_list,key=lambda x:x[0])
                    sorted_indices , sorted_coordinates = zip(*sorted_combined_list)
                    if (boundary_number == 2):
                        edges = [[corner_bottom_left_index,sorted_indices[0]],[sorted_indices[0],sorted_indices[1]],
                                    [sorted_indices[1],corner_bottom_right_index]]
                        new_edges.append(edges)
                    elif (boundary_number == 3):
                        edges = [[corner_top_left_index,sorted_indices[0]],[sorted_indices[0],sorted_indices[1]],
                                    [sorted_indices[1],corner_top_right_index]]
                        new_edges.append(edges)
                    
                elif (len(vertex_indices)==1):
                    if (boundary_number == 2):
                        edges = [ [corner_bottom_left_index,vertex_indices[0]] , [vertex_indices[0],corner_bottom_right_index] ]
                        new_edges.append(edges)
                    elif (boundary_number == 3):
                        edges = [ [corner_top_left_index,vertex_indices[0]] , [vertex_indices[0],corner_top_right_index] ]
                        new_edges.append(edges)

            elif (boundary_points_count[BI][2] == 'no intersection'):
                if (boundary_number == 0):
                        edges = [ [corner_bottom_left_index,corner_bottom_right_index] ]
                        new_edges.append(edges)
                elif (boundary_number == 1):
                    edges = [ [corner_top_left_index,corner_top_right_index] ]
                    new_edges.append(edges)

    new_edges_simplified = [item for sublist in new_edges for item in sublist]





    return MNVI, new_edges_simplified,new_vertices_updated