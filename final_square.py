import numpy as np
import pandas as pd
from scipy.optimize import minimize
import time

# ---------------------- CONFIGURATION ------------------------
np.random.seed(41)
num_points = 5
start_time = time.time()

# ---------------------- LOAD DATA ------------------------
quantities_df = pd.read_csv('quantities_all.csv')
hubs_df = pd.read_csv('hubs.csv')

ij_pairs = quantities_df[['Start_Lat', 'Start_Lon', 'Destination_Lat', 'Destination_Lon']].values
kl_array = hubs_df[['Latitude', 'Longitude']].values
num_kl = len(kl_array)

# ---------------------- DISTANCE UTILITY ------------------------
def fast_squared_distance(a, b):
    return np.sum((a - b) ** 2, axis=1)

def total_squared_distance_batch(i, j, k, l, m, n):
    return (
        fast_squared_distance(i, m) +
        fast_squared_distance(m, k) +
        fast_squared_distance(k, l) +
        fast_squared_distance(l, n) +
        fast_squared_distance(n, j)
    )

# ---------------------- OBJECTIVE FUNCTION ------------------------
def objective(mn_flat, ij_pairs, kl_array, num_points, beta, epsilon=1e-8):
    mn = mn_flat.reshape((num_points, 2))
    m_idx, n_idx = np.meshgrid(np.arange(num_points), np.arange(num_points))
    m_points = mn[m_idx.ravel()]
    n_points = mn[n_idx.ravel()]
    mn_len = len(m_points)

    k_idx, l_idx = np.meshgrid(np.arange(num_kl), np.arange(num_kl))
    k_points = kl_array[k_idx.ravel()]
    l_points = kl_array[l_idx.ravel()]
    kl_len = len(k_points)

    m_tile = np.repeat(m_points, kl_len, axis=0)
    n_tile = np.repeat(n_points, kl_len, axis=0)
    k_tile = np.tile(k_points, (mn_len, 1))
    l_tile = np.tile(l_points, (mn_len, 1))

    total_logsum = 0.0

    for i_j in ij_pairs:
        i = np.full_like(m_tile, i_j[:2])
        j = np.full_like(m_tile, i_j[2:])

        dists = total_squared_distance_batch(i, j, k_tile, l_tile, m_tile, n_tile)
        max_val = -beta * np.min(dists)
        logsum = max_val + np.log(np.sum(np.exp(-beta * dists - max_val)) + epsilon)
        total_logsum += logsum
    total_logsum=(1/beta)*total_logsum

    return -total_logsum

# ---------------------- OPTIMIZATION WRAPPER ------------------------
def gradient_descent(ij_pairs, kl_array, num_points, beta, initial_mn, max_iter=300):
    result = minimize(
        objective,
        initial_mn,
        args=(ij_pairs, kl_array, num_points, beta),
        method='L-BFGS-B',
        options={'maxiter': max_iter, 'disp': False}
    )
    return result.x, result.fun, result.nit

# ---------------------- INITIALIZATION ------------------------
lat_min, lat_max = hubs_df['Latitude'].min(), hubs_df['Latitude'].max()
lon_min, lon_max = hubs_df['Longitude'].min(), hubs_df['Longitude'].max()

initial_mn = np.zeros(num_points * 2)
initial_mn[::2] = np.random.uniform(lat_min, lat_max, num_points)
initial_mn[1::2] = np.random.uniform(lon_min, lon_max, num_points)

# ---------------------- OPTIMIZATION LOOP ------------------------
beta = 0.001
beta_max = 1000
k = 1.85

while beta < beta_max:
    print(f"Current beta: {beta:.4f}")
    hubs = np.round(initial_mn.reshape(-1, 2), 3)
    kl_array=np.vstack((kl_array,hubs))
    optimized_mn, final_obj_val, num_iter = gradient_descent(
        ij_pairs, kl_array, num_points, beta, initial_mn
    )
    print(f"Final objective: {final_obj_val:.4f} after {num_iter} iterations")
    initial_mn = optimized_mn
    beta *= k
    


total_time = time.time() - start_time
print(f"Computation time: {total_time:.2f} seconds")

# Save results to file
with open("computation_results.txt", "w") as f:
    f.write(f"Computation time: {total_time:.6f} seconds\n")
    f.write(f"Final objective value: {final_obj_val:.6f}\n")
    f.write(f"Final beta: {beta:.6f}\n")
    f.write("Optimized hub coordinates:\n")
    for i in range(num_points):
        lat = optimized_mn[2*i]
        lon = optimized_mn[2*i+1]
        f.write(f"Hub {i+1}: Lat = {lat:.6f}, Lon = {lon:.6f}\n")

# ---------------------- POSTPROCESSING WITH SQUARED DISTANCE ------------------------
from scipy.special import softmax
from itertools import product

ij_pairs = [
    (np.array([row['Start_Lat'], row['Start_Lon']]), np.array([row['Destination_Lat'], row['Destination_Lon']]))
    for _, row in quantities_df.iterrows()
]

hubs = np.round(optimized_mn.reshape(-1, 2), 3)
kl_array=np.vstack((kl_array,hubs))
stations = kl_array
results = []

combs = list(product(hubs, stations, stations, hubs))
num_combs = len(combs)

m_arr = np.array([c[0] for c in combs])
k_arr = np.array([c[1] for c in combs])
l_arr = np.array([c[2] for c in combs])
n_arr = np.array([c[3] for c in combs])

for idx, (i, j) in enumerate(ij_pairs):
    print(f"\nProcessing pair {idx+1}: i={i}, j={j}")
    
    i_arr = np.repeat(i[np.newaxis, :], num_combs, axis=0)
    j_arr = np.repeat(j[np.newaxis, :], num_combs, axis=0)

    dist = (
        np.sum((i_arr - m_arr) ** 2, axis=1) +
        np.sum((m_arr - k_arr) ** 2, axis=1) +
        np.sum((k_arr - l_arr) ** 2, axis=1) +
        np.sum((l_arr - n_arr) ** 2, axis=1) +
        np.sum((n_arr - j_arr) ** 2, axis=1)
    )

    weights = softmax(-beta * dist)

    results.append({
        'i_x': i[0], 'i_y': i[1],
        'j_x': j[0], 'j_y': j[1],
        'm_x': np.dot(weights, m_arr[:, 0]),
        'm_y': np.dot(weights, m_arr[:, 1]),
        'k_x': np.dot(weights, k_arr[:, 0]),
        'k_y': np.dot(weights, k_arr[:, 1]),
        'l_x': np.dot(weights, l_arr[:, 0]),
        'l_y': np.dot(weights, l_arr[:, 1]),
        'n_x': np.dot(weights, n_arr[:, 0]),
        'n_y': np.dot(weights, n_arr[:, 1]),
        'distance': np.dot(weights, dist),
        'limit': 1
    })

results_df = pd.DataFrame(results)
results_df.to_csv("postprocessed_results_squared.csv", index=False)

print("\n? Optimization and squared-distance-based post-processing completed.")
