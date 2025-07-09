import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
# ----------------------------------------------------------------------------
# 0. Parameters and Data Loading
# ----------------------------------------------------------------------------
start_time = time.time()
np.random.seed(42)

# Hyperparameters
M = 5
beta_initial = 0.001
beta_max = 1000.0
k = 1.85
max_inner_iters = 500
tol_inner = 1e-4
epsilon = 1e-12

# Load data
quantities_df = pd.read_csv('quantities_all.csv', encoding='utf-8-sig')
hubs_df      = pd.read_csv('hubs.csv', encoding='utf-8-sig')

origins      = quantities_df[['Start_Lat', 'Start_Lon']].to_numpy()
destinations = quantities_df[['Destination_Lat', 'Destination_Lon']].to_numpy()
rails        = hubs_df[['Latitude', 'Longitude']].to_numpy()

N = origins.shape[0]
R = rails.shape[0]


# ----------------------------------------------------------------------------
# 1. Helper: Enforce Minimum Distance Between Hubs (conditionally)
# ----------------------------------------------------------------------------
def enforce_min_distance(Y, min_dist=15.0, strength=0.1):
    for i in range(len(Y)):
        for j in range(i + 1, len(Y)):
            diff = Y[i] - Y[j]
            dist = np.linalg.norm(diff)
            if dist < min_dist:
                adjustment = strength * (diff / (dist + 1e-12))
                Y[i] += adjustment
                Y[j] -= adjustment
    return Y

# ----------------------------------------------------------------------------
# 2. DA Update Function with cdist
# ----------------------------------------------------------------------------
def compute_updated_hubs(hubs, origins, destinations, rails, beta):
    hubs_biased = hubs + np.random.normal(scale=1e-4, size=hubs.shape)

    d0 = cdist(origins, hubs) + epsilon
    d1 = cdist(hubs_biased, rails) + epsilon
    d2 = cdist(rails, rails) + epsilon
    d3 = cdist(destinations, hubs)[:, None, :] + cdist(rails, hubs_biased)[None, :, :] + epsilon
    d4 = cdist(destinations, hubs) + epsilon

    L3 = -beta * d3
    p3 = np.exp(L3 - np.max(L3, axis=2, keepdims=True))
    p3 /= np.sum(p3, axis=2, keepdims=True)
    Z3 = np.sum(p3, axis=2)

    L2 = -beta * d2[None, :, :] + np.log(Z3[:, :, None] + epsilon)
    p2 = np.exp(L2 - np.max(L2, axis=2, keepdims=True))
    p2 /= np.sum(p2, axis=2, keepdims=True)
    Z2 = np.sum(p2, axis=2)

    L1 = -beta * d1[None, :, :] + np.log(Z2[:, None, :] + epsilon)
    p1 = np.exp(L1 - np.max(L1, axis=2, keepdims=True))
    p1 /= np.sum(p1, axis=2, keepdims=True)
    Z1 = np.sum(p1, axis=2)

    L0 = -beta * d0 + np.log(Z1 + epsilon)
    p0 = np.exp(L0 - np.max(L0, axis=1, keepdims=True))
    p0 /= np.sum(p0, axis=1, keepdims=True)

    inv_d1 = 1.0 / d1
    inv_d4 = 1.0 / d4

    A1 = np.sum(p0 / d0, axis=0)
    A1_hat = np.sum(p0[:, :, None] * p1 / d1[None, :, :], axis=(0, 2))

    p3_T = np.transpose(p3, (0, 2, 1))
    A2 = np.einsum('nm,nmr,nrs,nfr,fr->f', p0, p1, p2, p3_T, inv_d1)
    A2_hat = np.einsum('nm,nmr,nrs,nfr,nf->f', p0, p1, p2, p3_T, inv_d4)

    B1 = (p0 / d0).T @ origins
    w_nf = np.einsum('nm,nmr,nrs,nfr->nf', p0, p1, p2, p3_T) / d4
    B2 = (w_nf[:, :, None] * destinations[:, None, :]).sum(axis=0)

    C1 = (np.einsum('nf,nfr->fr', p0, p1) / d1) @ rails
    C2 = (np.einsum('nm,nmr,nrs,nfr->fr', p0, p1, p2, p3_T) / d1) @ rails

    denom = A1 + A1_hat + A2 + A2_hat + epsilon
    Y = np.vstack([
        (B1[f] + B2[f] + C1[f] + C2[f]) / denom[f]
        for f in range(M)
    ])

    if 0.1 <= beta <= 10:
        Y = enforce_min_distance(Y)

    return Y

# ----------------------------------------------------------------------------
# 3. Deterministic Annealing Main Loop
# ----------------------------------------------------------------------------
lat_min, lon_min = rails.min(axis=0)
lat_max, lon_max = rails.max(axis=0)

hubs = np.random.uniform([lat_min, lon_min], [lat_max, lon_max], size=(M, 2))
print("Initial hubs:\n", hubs)
beta = beta_initial

while beta <= beta_max:
    print(f"\n=== beta={beta:.4f} ===")
    for it in range(max_inner_iters):
        new_hubs = compute_updated_hubs(hubs, origins, destinations, rails, beta)
        shift = np.max(np.linalg.norm(new_hubs - hubs, axis=1))
        hubs = new_hubs
        if shift < tol_inner:
            print("Converged.")
            break
    print('Hubs are ', hubs)
    beta *= k

print("\nFinal hubs:\n", hubs)
print("Time:", time.time() - start_time)

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# 4. Visualization
# ----------------------------------------------------------------------------
plt.figure(figsize=(8, 6))  # Optional: control size
initial_rails=rails
static_rails=rails
plt.scatter(initial_rails[:, 1], initial_rails[:, 0], label='Initial Rails', c='gray')
plt.scatter(origins[:, 1], origins[:, 0], label='Origins', marker='x', c='blue')
plt.scatter(destinations[:, 1], destinations[:, 0], label='Destinations', marker='^', c='green')
plt.scatter(hubs[:, 1], hubs[:, 0], label='Final Hubs', marker='o', c='red')

plt.legend()
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Hub Optimization via Deterministic Annealing")
plt.grid(True)

# Save the plot to a file instead of showing it
plt.savefig("hub_optimization_plot.png", bbox_inches='tight', dpi=300)
plt.close()

print("Plot saved as '2hub_optimization_lastplot.png'")
... 

# ----------------------------------------------------------------------------
# 7. Objective Function (NumPy-based Log-Sum-Exp using final_hubs)
# ----------------------------------------------------------------------------

# Final hub coordinates after optimization (assumed to be numpy array)
#final_hubs = mn.detach().cpu().numpy()
final_hubs=hubs
# Reload input data to numpy
quantities_df = pd.read_csv('quantities_all.csv')
hubs_df = pd.read_csv('hubs.csv')

origins = quantities_df[['Start_Lat', 'Start_Lon']].to_numpy()
destinations = quantities_df[['Destination_Lat', 'Destination_Lon']].to_numpy()
static_rails = hubs_df[['Latitude', 'Longitude']].to_numpy()

railupdate = np.vstack([static_rails, final_hubs])
final_hubs=railupdate
beta_final=1000
# Compute distances
d0 = np.linalg.norm(origins[:, np.newaxis, :] - final_hubs[np.newaxis, :, :], axis=2)        # [N, M]
d1 = np.linalg.norm(final_hubs[:, np.newaxis, :] - railupdate[np.newaxis, :, :], axis=2)          # [M, R]
d2 = np.linalg.norm(railupdate[:, np.newaxis, :] - railupdate[np.newaxis, :, :], axis=2)               # [R, R]
d3 = np.linalg.norm(railupdate[:, np.newaxis, :] - final_hubs[np.newaxis, :, :], axis=2)          # [R, M]
d4 = np.linalg.norm(final_hubs[:, np.newaxis, :] - destinations[np.newaxis, :, :], axis=2)   # [M, N]

# Reshape for broadcasting
D0 = d0[:, :, np.newaxis, np.newaxis, np.newaxis]        # [N, M, 1, 1, 1]
D1 = d1[np.newaxis, :, :, np.newaxis, np.newaxis]        # [1, M, R, 1, 1]
D2 = d2[np.newaxis, np.newaxis, :, :, np.newaxis]        # [1, 1, R, R, 1]
D3 = d3[np.newaxis, np.newaxis, np.newaxis, :, :]        # [1, 1, 1, R, M]
D4 = d4.T[:, np.newaxis, np.newaxis, np.newaxis, :]      # [N, 1, 1, 1, M]

total_distance = D0 + D1 + D2 + D3 + D4                  # [N, M, R, R, M]
L = -beta_final * total_distance
L_max = np.max(L, axis=(1,2,3,4), keepdims=True)
sum_exp = np.sum(np.exp(L - L_max), axis=(1,2,3,4))
logsumexp = np.log(sum_exp) + L_max.squeeze()

objective = -1 / beta_final * np.sum(logsumexp)
print(f"Objective (sum over i of log-sum-exp): {objective:.6f}")
