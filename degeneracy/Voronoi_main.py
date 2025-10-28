from scipy.spatial import Voronoi, voronoi_plot_2d
import numpy as np
import math
import matplotlib.pyplot as plt

# Creating random points
np.random.seed(10)
points = np.random.rand(5, 2)

# points =  np.array([[0.22199317, 0.87073231], [0.20671916 ,0.91861091], [0.48841119, 0.61174386], [0.48841119, 0.61174386], [0.2968005,  0.18772123]])

# np.random.seed(10)
# points = np.random.rand(100, 2)

# Creating the Voronoi diagram
vor = Voronoi(points,qhull_options="Qc")

minx = min(vor.vertices[:,0]) - 0.5
maxx = max(vor.vertices[:,0]) + 0.5
xValues = [minx, maxx]

# Defining minimum and maximum y values and store them in an array for future reference
miny = min(vor.vertices[:,1]) - 0.5
maxy = max(vor.vertices[:,1]) + 0.5
yValues = [miny,maxy]

# from Main_Function import fun_vor_main
# explicit_voronoi,vertices, cell_centers, mean_centers, distance_from_found_to_old, first_three_cell_centers= fun_vor_main(vor)
from Main_Function_for_Visualization import fun_vor_main_visualization
explicit_voronoi, vertices, first_three_cell_centers, mirrored_centers_to_cell = fun_vor_main_visualization(vor)


from Plot_Voronoi import PlotVoronoi
fig, ax = plt.subplots() # initialize the plot
PlotVoronoi(explicit_voronoi, vertices,ax)
markerstyles = ['o','^','s','p','*']

ax.plot(explicit_voronoi['sub_dict_2']['seed coordinates'][0],explicit_voronoi['sub_dict_2']['seed coordinates'][1],color='red', marker=markerstyles[1])
# ax.text(explicit_voronoi[sub_dict]['seed coordinates'][0],explicit_voronoi[sub_dict]['seed coordinates'][1],f'{explicit_voronoi[sub_dict]["seed number"]}',fontsize=12)

first_iteration_centers = first_three_cell_centers[0]
second_iteration_centers = first_three_cell_centers[1]
third_iteration_centers = first_three_cell_centers[2]
cell_arrangement = []

cell_centers_arranged = np.array([cell_centers[i] for i in vor.point_region])
mean_centers_arranged = np.array([mean_centers[i] for i in vor.point_region])
second_iteration_centers_arranged = np.array([second_iteration_centers[i] for i in vor.point_region])
third_iteration_centers_arranged = np.array([third_iteration_centers[i] for i in vor.point_region])




# fig,ax = plt.subplots()
# from Plot_Voronoi import PlotVoronoi
# PlotVoronoi(explicit_voronoi, vertices,ax)

# markerstyles = ['o','^','s','p','*','.']
# for i in range(len(cell_centers_arranged)):
#     ax.plot(cell_centers_arranged[i][0],cell_centers_arranged[i][1],markerstyles[i],color='red')
#     ax.plot(first_iteration_centers[i][0],first_iteration_centers[i][1],markerstyles[i],markerfacecolor='none',color='green')
#     ax.plot(second_iteration_centers_arranged[i][0],second_iteration_centers_arranged[i][1],markerstyles[i],markerfacecolor='none',color='blue')
#     ax.plot(third_iteration_centers_arranged[i][0],third_iteration_centers_arranged[i][1],markerstyles[i], markerfacecolor='none',color='magenta')
#     # ax.plot(mean_centers_arranged[i][0],mean_centers_arranged[i][1],markerstyles[i],color='blue')
# plt.show()

# print(distance_from_found_to_old)
# normalized_distance = {}

# for i in range(len(distance_from_found_to_old)):
#     normalized_distance[f'iteration_{i}'] = [x/distance_from_found_to_old[f'iteration_{i}'][0] for x in distance_from_found_to_old[f'iteration_{i}']]



# mean_distance = []
# for i in range(len(normalized_distance)):
#     mean_distance.append(np.mean(normalized_distance[f'iteration_{i}']))

# total_mean_distance = np.mean(mean_distance)
# print('total_mean_distance:',total_mean_distance)

# fig,ax = plt.subplots()
# ax.plot(total_mean_distance)
# plt.show()

















    









