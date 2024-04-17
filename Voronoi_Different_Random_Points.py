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

    # minx = min(vor.vertices[:,0]) - 0.5
    # maxx = max(vor.vertices[:,0]) + 0.5
    # xValues = [minx, maxx]

    # # Defining minimum and maximum y values and store them in an array for future reference
    # miny = min(vor.vertices[:,1]) - 0.5
    # maxy = max(vor.vertices[:,1]) + 0.5
    # yValues = [miny,maxy]

    from Main_Function import fun_vor_main
    explicit_voronoi,vertices, cell_centers, mean_centers, distance_from_found_to_original, first_three_cell_centers= fun_vor_main(vor)

    dist_to_original_storage[f'Random_Voronoi_{i}'] = distance_from_found_to_original

# print(np.mean(dist_to_original_storage['iteration_0']['iteration_0']))
# print(np.mean(dist_to_original_storage['iteration_0']['iteration_1']))
# print(np.mean(dist_to_original_storage['iteration_0']['iteration_2']))


mean_storage = {}

for i in range(len(dist_to_original_storage)):
    mean_storage_of_distribution = []
    for j in range(len(dist_to_original_storage[f'Random_Voronoi_{i}'])):
        mean_of_iteration = np.mean(dist_to_original_storage[f'Random_Voronoi_{i}'][f'iteration_{j}'])
        mean_storage_of_distribution.append(mean_of_iteration)
    mean_storage_of_distribution = [number/mean_storage_of_distribution[0] for number in mean_storage_of_distribution]
    mean_storage[f'Random_Voronoi_{i}'] = mean_storage_of_distribution

mean_storage_summation = np.empty(())

for k in range(len(mean_storage)):
    mean_storage_summation = np.add(mean_storage_summation,mean_storage[f'Random_Voronoi_{k}'])

    
mean_of_mean_storage_summation = [number/Number_of_Dif_Voronoi_Diagrams for number in mean_storage_summation]
# print(mean_storage)
# np.savetxt('/Users/balkirgoka/Documents/Phd/Projects/Voronoi/convergence_type_trial/array.txt', mean_of_mean_storage_summation, fmt='%.15f')

fig1,ax1 = plt.subplots()

iteration = np.arange(1,101,1)
print(iteration)

for i in range(len(mean_storage)):
    ax1.loglog(iteration,mean_storage[f'Random_Voronoi_{i}'],label=f'Random_Voronoi_{i}')
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Mean distance from original')
ax1.set_title('Log-Log plot of Convergence of Each Voronoi diagrams')

fig1.savefig('/Users/balkirgoka/Documents/Phd/Projects/Voronoi/Voronoi_6/Plots/Covergence_of_Each_Voronoi_Diagram.png', dpi=300)



fig2,ax2 = plt.subplots()

ax2.loglog(iteration,mean_of_mean_storage_summation,label='Mean of the Error of all Voronoi diagrams')
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Mean distance from Original Seed Points')
ax2.set_title('Log-Log plot of the Mean of the Error of all Voronoi diagrams')

fig2.savefig('/Users/balkirgoka/Documents/Phd/Projects/Voronoi/Voronoi_6/Plots/Log_Log_Convergence.png',dpi=300)



fig3,ax3 = plt.subplots()
ax3.semilogy(iteration,mean_of_mean_storage_summation,label='Mean of the Error of all Voronoi diagrams')
ax3.set_xlabel('Iteration')
ax3.set_ylabel('Mean Disrance from Original Seed Points')
ax3.set_title('Semilog plot of the Mean of the Error of all Voronoi diagrams')

fig3.savefig('/Users/balkirgoka/Documents/Phd/Projects/Voronoi/Voronoi_6/Plots/semilog_Convergence.png',dpi=300)


fig4,ax4 = plt.subplots()
ax4.plot(iteration,mean_of_mean_storage_summation,label='Mean of the Error of all Voronoi diagrams')
ax4.set_xlabel('Iteration')
ax4.set_ylabel('Mean Disrance from Original Seed Points')
ax4.set_title('Plot of the Mean of the Error of all Voronoi diagrams')
fig4.savefig('/Users/balkirgoka/Documents/Phd/Projects/Voronoi/Voronoi_6/Plots/plot_Convergence.png',dpi=300)

plt.show()
























    









