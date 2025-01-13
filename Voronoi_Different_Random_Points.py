from scipy.spatial import Voronoi, voronoi_plot_2d
import numpy as np
import math
import matplotlib.pyplot as plt

NoP = 100 # number of points in each Voronoi cell

# Creating random points
# np.random.seed(10)
dist_to_original_storage ={}

Number_of_Dif_Voronoi_Diagrams = 10


for i in range(Number_of_Dif_Voronoi_Diagrams):
    print(i)
    points = np.random.rand(NoP, 2)

    vor = Voronoi(points,qhull_options="Qc")


    from Main_Function import fun_vor_main
    explicit_voronoi,vertices, cell_centers, mean_centers, distance_from_found_to_original, first_three_cell_centers= fun_vor_main(vor)

    dist_to_original_storage[f'Random_Voronoi_{i}'] = distance_from_found_to_original




mean_storage = {}
max_storage = {}
min_storage = {}

for i in range(len(dist_to_original_storage)):
    mean_storage_of_distribution = []
    max_storage_of_distribution = []
    min_storage_of_distribution = []
    for j in range(len(dist_to_original_storage[f'Random_Voronoi_{i}'])):
        mean_of_iteration = np.mean(dist_to_original_storage[f'Random_Voronoi_{i}'][f'iteration_{j}'])
        max_of_iteration = np.max(dist_to_original_storage[f'Random_Voronoi_{i}'][f'iteration_{j}'])
        min_of_iteration = np.min(dist_to_original_storage[f'Random_Voronoi_{i}'][f'iteration_{j}'])

        mean_storage_of_distribution.append(mean_of_iteration)
        max_storage_of_distribution.append(max_of_iteration)
        min_storage_of_distribution.append(min_of_iteration)

    # mean_storage_of_distribution = [number/mean_storage_of_distribution[0] for number in mean_storage_of_distribution]
    # max_storage_of_distribution = [number/max_storage_of_distribution[0] for number in max_storage_of_distribution]
    # min_storage_of_distribution = [number/min_storage_of_distribution[0] for number in min_storage_of_distribution]

    mean_storage[f'Random_Voronoi_{i}'] = mean_storage_of_distribution
    max_storage[f'Random_Voronoi_{i}'] = max_storage_of_distribution
    min_storage[f'Random_Voronoi_{i}'] = min_storage_of_distribution



mean_storage_summation = np.zeros(len(mean_storage[f'Random_Voronoi_0']))
max_storage_summation = np.zeros(len(max_storage[f'Random_Voronoi_0']))
min_storage_summation = np.zeros(len(min_storage[f'Random_Voronoi_0']))

for k in range(len(mean_storage)):
    mean_storage_summation = np.add(mean_storage_summation,mean_storage[f'Random_Voronoi_{k}'])
    max_storage_summation = np.add(max_storage_summation,max_storage[f'Random_Voronoi_{k}'])
    min_storage_summation = np.add(min_storage_summation,min_storage[f'Random_Voronoi_{k}'])

    
mean_of_mean_storage_summation = [number/Number_of_Dif_Voronoi_Diagrams for number in mean_storage_summation]
mean_of_max_storage_summation = [number/Number_of_Dif_Voronoi_Diagrams for number in max_storage_summation]
mean_of_min_storage_summation = [number/Number_of_Dif_Voronoi_Diagrams for number in min_storage_summation]


iteration = np.arange(1,100+1,1)




fig1,ax1 = plt.subplots()
ax1.semilogy(iteration,mean_of_mean_storage_summation,label='Mean of the Error of all Voronoi diagrams')
ax1.semilogy(iteration,mean_of_max_storage_summation,label='Max of the Error of all Voronoi diagrams')
ax1.semilogy(iteration,mean_of_min_storage_summation,label='Min of the Error of all Voronoi diagrams')
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Mean Disrance from Original Seed Points')
ax1.set_title('Semilog plot of the Mean of the Error of all Voronoi diagrams')
# add legend
ax1.legend()

fig1.savefig('/Users/balkirgoka/Documents/Phd/Projects/Voronoi/Inverse_Voronoi/Plots/semilog_Convergence.png',dpi=300)



























    









