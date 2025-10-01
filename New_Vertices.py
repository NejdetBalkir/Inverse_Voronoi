import numpy as np
import math
def New_Vertices(vor, neighbor_storage, neighbor_vertices_storage):

    from intersection import line_intersection_vertical, line_intersection_horizontal
    minx = min(vor.vertices[:,0]) - 0.5
    maxx = max(vor.vertices[:,0]) + 0.5
    xValues = [minx, maxx]

    # Defining minimum and maximum y values and store them in an array for future reference
    miny = min(vor.vertices[:,1]) - 0.5
    maxy = max(vor.vertices[:,1]) + 0.5
    yValues = [miny,maxy]

    points = vor.points
    # Finding the midpoints between each neighbor seed point
    number_of_edges = len(neighbor_vertices_storage)
    number_of_vertices = len(vor.vertices) - 1
    neighbor_vertices_storage_updated = neighbor_vertices_storage.copy()
    new_vertices_index = []
    new_vertices=[]
    boundary_line_index = []
    for i in range(number_of_edges):
        seed1_index = neighbor_vertices_storage[i][0]
        seed2_index = neighbor_vertices_storage[i][1]

        seed1_index_LN = np.where(vor.point_region == seed1_index)[0][0]
        seed2_index_LN = np.where(vor.point_region == seed2_index)[0][0]

        edge_points = neighbor_vertices_storage[i][2]
        for j in range(2):
            if (edge_points[j] == -1): # look for the infinite vertex
                if (j==0):
                    finite_vertice_index = edge_points[1]
                elif (j==1):
                    finite_vertice_index = edge_points[0]

                finite_vertice = vor.vertices[finite_vertice_index]

                midpoint = vor.points[[seed1_index_LN,seed2_index_LN]].mean(axis=0) # calculate the midpoint between two neighboring seed points

                seed1_coordinates = vor.points[seed1_index_LN] # get the coordinates of the seed 1
                seed2_coordinates = vor.points[seed2_index_LN] # get the coordinates of the seed 2


                linebtw = seed2_coordinates - seed1_coordinates # vector between seed1 and seed2
                linebtw /= np.linalg.norm(linebtw) # make the line between seed1 and seed2 unit vector
                normal = np.array([-linebtw[1],linebtw[0]]) # calculate unit vector normal to the line between

                ##### These are for simulation purposes, can be deleted after the code is constructed ########
                simulation_point1 = midpoint + linebtw
                simulation_point2 = midpoint + normal
                ################################################################################################

                # put a far point from the midpoint in the direction of calculated normal #
                far_point = midpoint + 3*normal 
                

                #calculate the slope and the constant of the line in between far point and middle point
                # dx = far_point[0] - midpoint[0]
                # dy = far_point[1] - midpoint[1]
                # if abs(dx) < 1e-12: # vertical line
                #     m = None
                # elif abs(dy) < 1e-12: # horizontal line
                #     m = 0.0
                # else:
                #     m = dy/dx
                m = (far_point[1] - midpoint[1] ) / ( [far_point[0] - midpoint[0] ]) #slope
                b = midpoint[1] - (m*midpoint[0])

                # CALCULATION OF THE INTERSECTION POINTS WITH THE BOUNDARY LINES
                #There are 4 boundary lines, therefore we need to iterate one and find the closest intersection points            
                # 0: left boundary      (x = minx)  (m0 = 1 & b0 = minx)
                # 1: right boundary     (x = maxx)  (m1 = 1 & b1 = maxx)
                # 2: bottom boundary    (y = miny)  (m2 = 0 & b2 = miny)
                # 3: top boundary       (y = maxy)  (m3 = 0 & b3 = maxy)

                # Slope and constant of the boundary lines inside a nested list
                boundary_lines = [[1 , minx] , [1 , maxx] , [0 , miny] , [0 , maxy]]

                intersection_points_list = []
                distance_list = []

                centroid_point = mean_point = (sum(x for x, _ in points) / len(points), sum(y for _, y in points) / len(points))

                for k in range(4): # iterate over each boundary line
                    if (k == 0 ) or (k == 1): # 0: left boundary , 1: right boundary
                        # dist_to_centroid_left = math.sqrt((centroid_point[0] - boundary_lines[0][1])**2)
                        # dist_to_centroid_right = math.sqrt((centroid_point[0] - boundary_lines[1][1])**2)
                        # Calculate the intersection point between the lines
                        intersection_point = line_intersection_vertical(boundary_lines[k][1] , m[0] , b[0])

                        # Calculate the distance between the finite vertice and intersection point
                        dist = math.sqrt((finite_vertice[0] - intersection_point[0])**2 + (finite_vertice[1] - intersection_point[1])**2)
                        intersection_points_list.append(intersection_point) # add the intersection point to a list
                        distance_list.append(dist) # add the distance to a list

                    elif (k==2) or (k==3): # 2: bottom boundary, 3: top boundary
                        intersection_point = line_intersection_horizontal(boundary_lines[k][1] , m[0] , b[0]) #calculate the intersection point
                        dist = math.sqrt((finite_vertice[0] - intersection_point[0])**2 + (finite_vertice[1] - intersection_point[1])**2) # calculate the distance
                        intersection_points_list.append(intersection_point) # add the intersection point to a list
                        distance_list.append(dist) # add the distance to a list

                min_index , min_value = min(enumerate(distance_list), key=lambda x:x[1]) # find the index of the minimum distance
                closest_intersection_point = intersection_points_list[min_index] # find the point with associated index

                

                new_vertices.append(closest_intersection_point) # add this point to a new vertices list

                edge_points_updated = edge_points.copy()
                edge_points_updated[j] = number_of_vertices+1 
                
                neighbor_vertices_storage_updated[i][2] = edge_points_updated
                new_vertices_index.append(number_of_vertices+1)
                number_of_vertices += 1              

                ################# PLOTTING #########################

                # ax.plot(simulation_point1[0] , simulation_point1[1] , 'go' , 'markersize',10)
                # ax.plot(simulation_point2[0] , simulation_point2[1] , 'o', color='purple')
                # ax.plot(closest_intersection_point[0],closest_intersection_point[1],'o',color='yellow')
                # ax.plot(finite_vertice[0],finite_vertice[1],'o',color='grey')

                # ax.plot(seed1_coordinates[0] , seed1_coordinates[1] , 'ro' , "markersize",10)
                # ax.plot(seed2_coordinates[0] , seed2_coordinates[1] , 'ro' , "markersize",10)
                # ax.plot(midpoint[0] , midpoint[1] , 'ko' , "markersize",10)
                #######################################################

    return new_vertices, new_vertices_index, neighbor_vertices_storage_updated


# import numpy as np
# import math
# from intersection import line_intersection_vertical, line_intersection_horizontal

# def New_Vertices(vor, neighbor_storage, neighbor_vertices_storage):
#     """
#     Replace infinite Voronoi ridge vertices (-1) with intersections
#     on a bounding box, creating finite vertices for explicit Voronoi.
#     """

#     # bounding box
#     minx = np.min(vor.vertices[:, 0]) - 0.5
#     maxx = np.max(vor.vertices[:, 0]) + 0.5
#     miny = np.min(vor.vertices[:, 1]) - 0.5
#     maxy = np.max(vor.vertices[:, 1]) + 0.5

#     points = vor.points
#     number_of_edges = len(neighbor_vertices_storage)
#     number_of_vertices = len(vor.vertices) - 1

#     neighbor_vertices_storage_updated = neighbor_vertices_storage.copy()
#     new_vertices_index = []
#     new_vertices = []

#     for i in range(number_of_edges):
#         seed1_index = neighbor_vertices_storage[i][0]
#         seed2_index = neighbor_vertices_storage[i][1]

#         seed1_index_LN = np.where(vor.point_region == seed1_index)[0][0]
#         seed2_index_LN = np.where(vor.point_region == seed2_index)[0][0]

#         edge_points = neighbor_vertices_storage[i][2]

#         for j in range(2):
#             if edge_points[j] == -1:  # infinite vertex
#                 finite_vertice_index = edge_points[1 - j]
#                 finite_vertice = vor.vertices[finite_vertice_index]

#                 # midpoint of seeds
#                 midpoint = vor.points[[seed1_index_LN, seed2_index_LN]].mean(axis=0)
#                 seed1_coordinates = vor.points[seed1_index_LN]
#                 seed2_coordinates = vor.points[seed2_index_LN]

#                 # normal to the line between seeds
#                 linebtw = seed2_coordinates - seed1_coordinates
#                 linebtw /= np.linalg.norm(linebtw)
#                 normal = np.array([-linebtw[1], linebtw[0]])

#                 far_point = midpoint + 3 * normal

#                 dx = far_point[0] - midpoint[0]
#                 dy = far_point[1] - midpoint[1]

#                 if abs(dx) < 1e-12:
#                     m, b = None, None   # vertical
#                 elif abs(dy) < 1e-12:
#                     m, b = 0.0, midpoint[1]   # horizontal
#                 else:
#                     m = dy / dx
#                     b = midpoint[1] - m * midpoint[0]

#                 # bounding box lines: (type, value)
#                 boundary_lines = [
#                     ("v", minx),
#                     ("v", maxx),
#                     ("h", miny),
#                     ("h", maxy),
#                 ]

#                 intersection_points_list = []
#                 distance_list = []

#                 for typ, val in boundary_lines:
#                     if typ == "v":
#                         ip = line_intersection_vertical(val, m, b)
#                     else:
#                         ip = line_intersection_horizontal(val, m, b)

#                     if ip is not None:
#                         dist = math.dist(finite_vertice, ip)
#                         intersection_points_list.append(ip)
#                         distance_list.append(dist)

#                 if not intersection_points_list:
#                     continue  # skip if no valid intersections

#                 # pick the closest intersection
#                 min_index = int(np.argmin(distance_list))
#                 closest_intersection_point = intersection_points_list[min_index]

#                 # add to list of new vertices
#                 new_vertices.append(closest_intersection_point)

#                 # update storage
#                 edge_points_updated = edge_points.copy()
#                 edge_points_updated[j] = number_of_vertices + 1
#                 neighbor_vertices_storage_updated[i][2] = edge_points_updated

#                 new_vertices_index.append(number_of_vertices + 1)
#                 number_of_vertices += 1

#     return new_vertices, new_vertices_index, neighbor_vertices_storage_updated



