import numpy as np
from scipy.spatial import Voronoi
from joblib import Parallel, delayed
from tqdm import tqdm
from Main_Function import fun_vor_main
from scipy.stats import binned_statistic

def nonvoronoiness_per_seed(cell_centers_perturbed, explicit_voronoi,vertices_perturbed):
    from mirror import mirror_point
    # This function calculates non-voronoiness for each seed based on the perturbed cell centers and explicit voronoi structure.
    # for each central seed n and for each neighboring seed m of n
    # reflect m's final position across the common edge with n
    # numerator = ||reflected_m - n_final||
    # denominator = || reflected_m - m_final||
    # return the mean ratio of neighbors
    nonvoronoiness = []
    for key in explicit_voronoi.keys():
        n_index = explicit_voronoi[key]['seed number']
        n_final = cell_centers_perturbed[n_index]
        neighbors = explicit_voronoi[key]['neighbors']
        ratios = []
        for m_index in neighbors:
            m_final = cell_centers_perturbed[m_index]
            # print('n_index:',n_index,'m_index:',m_index)
            # find common edge
            edges_of_n = explicit_voronoi[key]['edges']
            edges_of_m = explicit_voronoi[f'sub_dict_{m_index}']['edges']
            common_edges = [edge for edge in edges_of_n if edge in edges_of_m]
            # print('common edge:', common_edges)
            # mirror m_final across the common edge
            common_vertex1_coord = vertices_perturbed[common_edges[0][0]]
            common_vertex2_coord = vertices_perturbed[common_edges[0][1]]
            
            reflected_m = mirror_point(m_final, common_vertex1_coord, common_vertex2_coord)
            # calculate distances
            distance_to_n_final = np.linalg.norm(reflected_m - n_final)
            distance_to_m_final = np.linalg.norm(reflected_m - m_final)
            ratio = distance_to_n_final / distance_to_m_final
            ratios.append(ratio)
        nonvoronoiness.append(float(np.mean(ratios)))
    return nonvoronoiness

def single_test(i, N_points=1000, Lx=1.0, Ly=1.0):
    # --- Random points (can be replaced by Poisson disk)
    points = np.random.rand(N_points, 2)

    # --- Generate Voronoi diagram
    vor = Voronoi(points)

    # --- Perturbation setup
    perturbation_strength = 1 / np.sqrt(N_points) * 1e-1
    vertices_coords = vor.vertices.copy()

    # --- Find central vertex (as perturbation origin)
    center_of_box = np.array([Lx / 2, Ly / 2])
    center_idx = np.argmin(np.linalg.norm(vertices_coords - center_of_box[None, :], axis=1))

    # --- Random perturbation direction
    random_direction = np.random.randn(2)
    random_direction /= np.linalg.norm(random_direction)
    vertices_coords[center_idx] += perturbation_strength * random_direction
    perturbed_vertex = vertices_coords[center_idx].copy()


    # --- Inverse Voronoi reconstruction
    (explicit_voronoi_perturbed,
     vertices_perturbed,
     cell_centers_perturbed,
     distance_original_list_perturbed,
     distance_from_found_to_previous_perturbed) = fun_vor_main(vor, points, vertices_coords)

    # --- Compute non-voronoiness per seed
    nonvoronoiness = nonvoronoiness_per_seed(cell_centers_perturbed,
                                             explicit_voronoi_perturbed,
                                             vertices_perturbed)

    # --- Return data for later spatial decay analysis
    result = {
        "cell_centers": np.array(points),
        "nonvoronoiness": np.array(nonvoronoiness),
        "perturbed_vertex": perturbed_vertex,
    }
    return result

number_of_tests = 1000
N_points = 1000

print(f"Running {number_of_tests} trials with {N_points} points each...")

results = Parallel(n_jobs=-1, backend="loky")(
    delayed(single_test)(i, N_points)
    for i in tqdm(range(number_of_tests))
)

# Save for later postprocessing
np.save("voronoi_decay_data.npy", results, allow_pickle=True)

print("✅ All data saved to voronoi_decay_data.npy")