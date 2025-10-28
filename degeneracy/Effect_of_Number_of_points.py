from scipy.spatial import Voronoi, voronoi_plot_2d
import numpy as np
import math
import matplotlib.pyplot as plt

List_of_NoP= [10,50,100,250,500,1000,2000,3000,4000,5000]  # number of points in each Voronoi cell

# Creating random points
# np.random.seed(10)
dist_to_original_storage ={}
dist_to_previous_storage = {}

# Number_of_Dif_Voronoi_Diagrams = 10
number_of_trial = 100



# for NoP in List_of_NoP: # Loop over the number of points
sum_of_rms_distances = 0  # Initialize sum of RMS distances
rms_list = []  # Initialize list to store RMS distances for each trial
NoP = 10
for i in range(number_of_trial):
    # print(f'Trial number: {i}')
    points = np.random.rand(NoP, 2)

    vor = Voronoi(points,qhull_options="Qc")


    from Main_Function import fun_vor_main
    explicit_voronoi,vertices, cell_centers, distance_from_found_to_original, distance_from_found_to_previous = fun_vor_main(vor, points)

    distance = distance_from_found_to_original
    # distance_at_last_iteration = distance_from_found_to_original[f'iteration_{len(distance_from_found_to_original)-1}']

    # rarrange original points according to vor.point_region
    original_points_rearranged = np.zeros((len(vor.points),2))
    for index in  range(len(vor.point_region)):
        original_points_rearranged[vor.point_region[index]] = points[index]

    # rms distance calculation
    distance = cell_centers-original_points_rearranged
    distance = np.sqrt(np.sum(distance**2,axis=1))
    rms_distance = math.sqrt(np.sum(np.array(distance)**2)/len(distance))
    rms_list.append(rms_distance)

    print('RMS of distances from original:', rms_distance)

# Calculate mean log 10 of RMS distances
mean_log_rms_distance = np.mean(np.log10(rms_list))
print(f'Mean log RMS for {NoP} points: {mean_log_rms_distance}')





    









