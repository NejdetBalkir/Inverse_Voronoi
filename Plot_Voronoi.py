def PlotVoronoi(explicit_voronoi, vertices,ax):
    import matplotlib.pyplot as plt
    import numpy as np
    for sub_dict in explicit_voronoi:
        edges_of_the_cell = explicit_voronoi[sub_dict]['edges']
        for i in range(len(edges_of_the_cell)):
            vertice1_of_edge = vertices[edges_of_the_cell[i][0]]
            vertice2_of_edge = vertices[edges_of_the_cell[i][1]]
            ax.plot([vertice1_of_edge[0],vertice2_of_edge[0]],[vertice1_of_edge[1],vertice2_of_edge[1]],'k-')


    

