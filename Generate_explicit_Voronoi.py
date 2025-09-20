import numpy as np
import matplotlib.pyplot as plt

def ExplicitVoronoi(vor):

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

    new_edges_reduced = [[x - len(vor.vertices) for x in sublist] for sublist in new_edges]

    inside_edges = [sublist[2] for sublist in neighbor_vertices_storage_updated ]

    total_edges = new_edges + inside_edges

    vertices = np.vstack((vor.vertices, np.array(new_vertices_updated)))

    from ExplicitVoronoi import ExplicitVoronoi
    explicit_voronoi = ExplicitVoronoi(vor,neighbor_vertices_storage_updated,total_edges)

    return explicit_voronoi, vertices

  


    