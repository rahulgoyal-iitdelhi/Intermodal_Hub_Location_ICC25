import numpy as np
import pandas as pd
import time

###############################################################################
# 0. Parameters and Data Loading
###############################################################################
a = time.time()
np.random.seed(42)
M = 3     # number of hubs
beta_initial = 0.001
beta_max = 1000.0
k = 1.85

max_inner_iters = 500
tol_inner = 1e-4

# Load data
quantities_df = pd.read_csv('quantities_new.csv', encoding='utf-8-sig')
hubs_df = pd.read_csv('hubs.csv', encoding='utf-8-sig')

origins = quantities_df[['Start_Lat', 'Start_Lon']].to_numpy()
destinations = quantities_df[['Destination_Lat', 'Destination_Lon']].to_numpy()
rails = hubs_df[['Latitude', 'Longitude']].to_numpy()

N = len(origins)
R = len(rails)

###############################################################################
# 1. Stable DA Update Function (Squared Euclidean distances here)
###############################################################################
def compute_updated_hubs(hubs, origins, destinations, rails, beta):
    N, _ = origins.shape
    M, _ = hubs.shape
    R, _ = rails.shape
    bias = np.random.normal(scale=1e-6, size=(M, 2))
    biased_hubs = hubs + bias

    d0 = np.sum((origins[:, np.newaxis, :] - hubs[np.newaxis, :, :]) ** 2, axis=2)
    d1 = np.sum((biased_hubs[:, np.newaxis, :] - rails[np.newaxis, :, :]) ** 2, axis=2)
    d2 = np.sum((rails[:, np.newaxis, :] - rails[np.newaxis, :, :]) ** 2, axis=2)
    dist_rf = np.sum((rails[:, np.newaxis, :] - hubs[np.newaxis, :, :]) ** 2, axis=2)
    dist_fd = np.sum((destinations[:, np.newaxis, :] - hubs[np.newaxis, :, :]) ** 2, axis=2)

    d3 = dist_fd[:, np.newaxis, :] + dist_rf[np.newaxis, :, :]

    L3 = -beta * d3
    max_L3 = np.max(L3, axis=2, keepdims=True)
    exp_L3 = np.exp(L3 - max_L3)
    sum_exp_L3 = np.sum(exp_L3, axis=2, keepdims=True)
    p3 = exp_L3 / sum_exp_L3
    Z3 = sum_exp_L3[..., 0] * np.exp(max_L3[..., 0])

    L2_base = -beta * d2
    logZ3 = np.log(Z3 + 1e-300)
    L2 = L2_base[np.newaxis, :, :] + logZ3[:, np.newaxis, :]
    max_L2 = np.max(L2, axis=2, keepdims=True)
    exp_L2 = np.exp(L2 - max_L2)
    sum_exp_L2 = np.sum(exp_L2, axis=2, keepdims=True)
    p2 = exp_L2 / sum_exp_L2
    Z2 = sum_exp_L2[..., 0] * np.exp(max_L2[..., 0])

    L1_base = -beta * d1
    logZ2 = np.log(Z2 + 1e-300)
    L1 = L1_base[np.newaxis, :, :] + logZ2[:, np.newaxis, :]
    max_L1 = np.max(L1, axis=2, keepdims=True)
    exp_L1 = np.exp(L1 - max_L1)
    sum_exp_L1 = np.sum(exp_L1, axis=2, keepdims=True)
    p1 = exp_L1 / sum_exp_L1
    Z1 = sum_exp_L1[..., 0] * np.exp(max_L1[..., 0])

    L0_base = -beta * d0
    logZ1 = np.log(Z1 + 1e-300)
    L0 = L0_base + logZ1
    max_L0 = np.max(L0, axis=1, keepdims=True)
    exp_L0 = np.exp(L0 - max_L0)
    sum_exp_L0 = np.sum(exp_L0, axis=1, keepdims=True)
    p0 = exp_L0 / sum_exp_L0

    A1_diag = np.sum(p0, axis=0)

    U = np.einsum('nf,nfr->nr', p0, p1)
    V = np.einsum('nr,nrs->ns', U, p2)
    W = np.einsum('ns,nsm->nm', V, p3)

    A2_diag = np.sum(W, axis=0)

    B1 = p0.T
    B1Xi = B1 @ origins

    B2 = W.T
    B2Xj = B2 @ destinations

    C1 = np.einsum('nm,nmr->mr', p0, p1)
    C1R = C1 @ rails

    C2 = np.einsum('nr,nrm->mr', V, p3)
    C2R = C2 @ rails

    A_diag = A1_diag + A2_diag
    RHS = B1Xi + B2Xj + C1R + C2R
    Y = 0.5 * (RHS / (A_diag[:, np.newaxis] + 1e-4))

    return Y

###############################################################################
# 2. DA Loop
###############################################################################
lat_min, lon_min = np.min(rails, axis=0)
lat_max, lon_max = np.max(rails, axis=0)

hubs = np.array([
    [np.random.uniform(lat_min, lat_max), np.random.uniform(lon_min, lon_max)]
    for _ in range(M)
])

print("Initial hubs:\n", hubs)
initial_hubs = hubs.copy()
beta = beta_initial
outer_iteration = 0

while beta <= beta_max:
    outer_iteration += 1
    print(f"\n=== Outer iteration {outer_iteration}: beta = {beta:.4f} ===")
    for inner_iter in range(1, max_inner_iters + 1):
        new_hubs = compute_updated_hubs(hubs, origins, destinations, rails, beta)
        max_change = np.max(np.linalg.norm(new_hubs - hubs, axis=1))
        print(f"  [beta={beta:.4f}] Inner iter {inner_iter:3d}: max hub shift = {max_change:.6f}")
        hubs = new_hubs
        if max_change < tol_inner:
            print(f"  Converged at inner iteration {inner_iter} (tol={tol_inner}).")
            break
    else:
        print(f"  Reached max_inner_iters={max_inner_iters} without inner convergence.")
    print(" ? hubs after this beta:\n", hubs)
    beta *= k

final_hubs = hubs.copy()
print("\n*** Final hub coordinates after annealing (M x 2): ***")
print(final_hubs)
print("Computational time is ", time.time() - a)

###############################################################################
# 3. Post-DA: Objective Calculation (uses SQUARED Euclidean, unchanged)
###############################################################################
beta_final = beta / k  # last beta used

d0 = np.sum((origins[:, np.newaxis, :] - final_hubs[np.newaxis, :, :]) ** 2, axis=2)        # [N, M]
d1 = np.sum((final_hubs[:, np.newaxis, :] - rails[np.newaxis, :, :]) ** 2, axis=2)          # [M, R]
d2 = np.sum((rails[:, np.newaxis, :] - rails[np.newaxis, :, :]) ** 2, axis=2)               # [R, R]
d3 = np.sum((rails[:, np.newaxis, :] - final_hubs[np.newaxis, :, :]) ** 2, axis=2)          # [R, M]
d4 = np.sum((final_hubs[:, np.newaxis, :] - destinations[np.newaxis, :, :]) ** 2, axis=2)   # [M, N]

D0 = d0[:, :, np.newaxis, np.newaxis, np.newaxis]
D1 = d1[np.newaxis, :, :, np.newaxis, np.newaxis]
D2 = d2[np.newaxis, np.newaxis, :, :, np.newaxis]
D3 = d3[np.newaxis, np.newaxis, np.newaxis, :, :]
D4 = d4.T[:, np.newaxis, np.newaxis, np.newaxis, :]

total_distance = D0 + D1 + D2 + D3 + D4

L = -beta_final * total_distance
L_max = np.max(L, axis=(1, 2, 3, 4), keepdims=True)
sum_exp = np.sum(np.exp(L - L_max), axis=(1, 2, 3, 4))
logsumexp = np.log(sum_exp) + L_max.squeeze()

objective = -1 / beta_final * np.sum(logsumexp)
print(f"Objective (sum over i of log-sum-exp): {objective:.6f}")
