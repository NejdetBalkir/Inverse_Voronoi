import numpy as np
import matplotlib.pyplot as plt

def fun_vor_main(vor):

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

    from center_search import center_search
    cell_centers, mean_centers, distance_from_found_to_old, first_three_cell_centers = center_search(explicit_voronoi,vertices)

  


    # fig, ax = plt.subplots() # initialize the plot

    # ax.plot(list(zip(*vertices))[0],list(zip(*vertices))[1],'o',color='red')

    # for lineI in range(len(new_edges_reduced)):
    #     firstP = new_edges_reduced[lineI][0]
    #     secondP = new_edges_reduced[lineI][1]

    #     linex = [new_vertices_updated[firstP][0],new_vertices_updated[secondP][0]]
    #     liney = [new_vertices_updated[firstP][1],new_vertices_updated[secondP][1]]

    #     ax.plot(linex,liney,'-',color='black')

    # for lineI in range(len(inside_edges)):
    #     firstP = inside_edges[lineI][0]
    #     secondP = inside_edges[lineI][1]

    #     linex = [vertices[firstP][0],vertices[secondP][0]]
    #     liney = [vertices[firstP][1],vertices[secondP][1]]

    #     ax.plot(linex,liney,'-',color='black')

    # for index, (x, y) in enumerate(vertices):
    #     ax.text(x, y, str(index), color='blue', fontsize=12)


    # for index, (x,y) in enumerate(vor.points):
    #     region = vor.point_region[index]
    #     xy_text = f"[{x:.2f}, {y:.2f}]"
    #     ax.plot(x,y,marker='o',markersize=5,color='green')
    #     ax.text(x,y,str(region),color='black')
    #     ax.text(x,y,xy_text,color='black',horizontalalignment='right',verticalalignment='top')
    # for findex, (x,y) in enumerate(cell_centers):
    #     ax.plot(x,y,marker='o',markersize=2,color='red')
    #     xy_text = f"[{x:.2f}, {y:.2f}]"
    #     ax.text(x,y,str(findex),color='red',horizontalalignment='left',verticalalignment='top')
    #     ax.text(x,y,xy_text,color='red',horizontalalignment='left',verticalalignment='bottom')


    # plt.xlim((minx-0.1,maxx+0.1))
    # plt.ylim((miny-0.1,maxy+0.1))

    # plt.show()
    

    return explicit_voronoi, vertices, cell_centers, mean_centers, distance_from_found_to_old, first_three_cell_centers,