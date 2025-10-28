def PlotVoronoi(cell_centers,explicit_voronoi, vertices,ax):
    import matplotlib.pyplot as plt
    import numpy as np
    print(cell_centers)
    for sub_dict in explicit_voronoi:
        edges_of_the_cell = explicit_voronoi[sub_dict]['edges']
        for i in range(len(edges_of_the_cell)):
            vertice1_of_edge = vertices[edges_of_the_cell[i][0]]
            vertice2_of_edge = vertices[edges_of_the_cell[i][1]]
            ax.plot([vertice1_of_edge[0],vertice2_of_edge[0]],[vertice1_of_edge[1],vertice2_of_edge[1]],'k-')

    # limit according to seed points
    # x_min = np.min(cell_centers[:,0]) - 0.1
    # x_max = np.max(cell_centers[:,0]) + 0.1
    # y_min = np.min(cell_centers[:,1]) - 0.1
    # y_max = np.max(cell_centers[:,1]) + 0.1
    # ax.set_xlim(x_min, x_max)
    # ax.set_ylim(y_min, y_max)
    # ax.set_aspect('equal', adjustable='box')
    




    

