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
number_of_trial = 1000



# for NoP in List_of_NoP: # Loop over the number of points
sum_of_rms_distances = 0  # Initialize sum of RMS distances
rms_list = []  # Initialize list to store RMS distances for each trial
NoP = 500
for i in range(number_of_trial):
    # print(f'Trial number: {i}')
    points = np.random.rand(NoP, 2)

    vor = Voronoi(points,qhull_options="Qc")


    from Main_Function import fun_vor_main
    explicit_voronoi,vertices, cell_centers, distance_from_found_to_original, distance_from_found_to_previous = fun_vor_main(vor, points)

    distance_at_last_iteration = distance_from_found_to_original[f'iteration_{len(distance_from_found_to_original)-1}']

    # Distance from original at last iteration
    # print('Distance from original at last iteration:', distance_at_last_iteration)

    # RMS of distances
    rms_distance = math.sqrt(np.sum(np.array(distance_at_last_iteration)**2)/len(distance_at_last_iteration))
    rms_list.append(rms_distance)
    # print('RMS of distances from original:', rms_distance)
    sum_of_rms_distances += rms_distance

    # #log of RMS error
    # log_rms_distance = math.log10(rms_distance)
    # print('Log of RMS error:', log_rms_distance)

    # mean RMS error
mean_rms_distance = np.mean(np.log10(rms_list))
print(f'Mean log RMS for {NoP} points: {mean_rms_distance}')









    









