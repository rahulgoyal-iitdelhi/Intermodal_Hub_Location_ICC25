import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

###############################################################################
# 0. Parameters and Data Loading
###############################################################################
a = time.time()
np.random.seed(41)
M = 5     # number of hubs
beta_initial = 0.001
beta_max = 1000.0
k = 1.85

max_inner_iters = 500
tol_inner = 1e-3

# Load data
quantities_df = pd.read_csv('quantities_all.csv', encoding='utf-8-sig')
hubs_df = pd.read_csv('hubs.csv', encoding='utf-8-sig')

origins = quantities_df[['Start_Lat', 'Start_Lon']].to_numpy()            # shape: (N, 2)
destinations = quantities_df[['Destination_Lat', 'Destination_Lon']].to_numpy()  # shape: (N, 2)
rails = hubs_df[['Latitude', 'Longitude']].to_numpy()                      # shape: (R, 2)

N = origins.shape[0]
R = rails.shape[0]

###############################################################################
# 1. Stable DA Update Function (returns Y, p0, p1, p2, p3)
###############################################################################
def compute_updated_hubs_and_probs(hubs, origins, destinations, rails, beta):
    """
    Computes the updated hub coordinates (Y) and the probability arrays p0, p1, p2, p3
    for a given beta, using deterministic annealing.
    
    Returns:
        Y   : (M, 2) array of updated hub coordinates
        p0  : (N, M) array of origin→hub assignment probabilities
        p1  : (N, M, R) array of hub→rail assignment probabilities (first leg)
        p2  : (N, R, R) array of rail→rail (intermediate) assignment probabilities
        p3  : (N, R, M) array of rail→hub assignment probabilities (final leg)
    """
    # Dimensions
    N, _ = origins.shape
    M, _ = hubs.shape
    R, _ = rails.shape

    # Tiny random bias to stabilize numerical gradients
    bias = np.random.normal(scale=1e-6, size=(M, 2))
    biased_hubs = hubs + bias

    # Distances
    d0 = np.linalg.norm(origins[:, np.newaxis, :] - biased_hubs[np.newaxis, :, :], axis=2)   # (N, M)
    d1 = np.linalg.norm(biased_hubs[:, np.newaxis, :] - rails[np.newaxis, :, :], axis=2)     # (M, R)
    d2 = np.linalg.norm(rails[:, np.newaxis, :] - rails[np.newaxis, :, :], axis=2)           # (R, R)
    dist_rf = np.linalg.norm(rails[:, np.newaxis, :] - biased_hubs[np.newaxis, :, :], axis=2)  # (R, M)
    dist_fd = np.linalg.norm(destinations[:, np.newaxis, :] - biased_hubs[np.newaxis, :, :], axis=2)  # (N, M)

    # Compute d3: origin→hub→rail→rail→hub→destination chain (aggregated distances)
    # Actually: (rail → hub) + (destination → hub) to define joint probabilities
    d3 = dist_fd[:, np.newaxis, :] + dist_rf[np.newaxis, :, :]  # shape: (N, R, M)

    # ====================
    #  Stage 3: rail→hub probabilities (p3)
    # ====================
    L3 = -beta * d3                              # (N, R, M)
    max_L3 = np.max(L3, axis=2, keepdims=True)   # (N, R, 1)
    exp_L3 = np.exp(L3 - max_L3)                 # (N, R, M)
    sum_exp_L3 = np.sum(exp_L3, axis=2, keepdims=True)  # (N, R, 1)
    p3 = exp_L3 / sum_exp_L3                                    # (N, R, M)
    Z3 = sum_exp_L3[..., 0] * np.exp(max_L3[..., 0])             # (N, R)

    # ====================
    #  Stage 2: rail→rail probabilities (p2)
    # ====================
    L2_base = -beta * d2                              # (R, R)
    logZ3 = np.log(Z3 + 1e-300)                       # (N, R)
    L2 = L2_base[np.newaxis, :, :] + logZ3[:, np.newaxis, :]  # (N, R, R)
    max_L2 = np.max(L2, axis=2, keepdims=True)        # (N, R, 1)
    exp_L2 = np.exp(L2 - max_L2)                      # (N, R, R)
    sum_exp_L2 = np.sum(exp_L2, axis=2, keepdims=True)  # (N, R, 1)
    p2 = exp_L2 / sum_exp_L2                                # (N, R, R)
    Z2 = sum_exp_L2[..., 0] * np.exp(max_L2[..., 0])         # (N, R)

    # ====================
    #  Stage 1: hub→rail probabilities (p1)
    # ====================
    L1_base = -beta * d1                              # (M, R)
    logZ2 = np.log(Z2 + 1e-300)                       # (N, R)
    L1 = L1_base[np.newaxis, :, :] + logZ2[:, np.newaxis, :]  # (N, M, R)
    max_L1 = np.max(L1, axis=2, keepdims=True)        # (N, M, 1)
    exp_L1 = np.exp(L1 - max_L1)                      # (N, M, R)
    sum_exp_L1 = np.sum(exp_L1, axis=2, keepdims=True)  # (N, M, 1)
    p1 = exp_L1 / sum_exp_L1                                 # (N, M, R)
    Z1 = sum_exp_L1[..., 0] * np.exp(max_L1[..., 0])         # (N, M)

    # ====================
    #  Stage 0: origin→hub probabilities (p0)
    # ====================
    L0_base = -beta * d0                               # (N, M)
    logZ1 = np.log(Z1 + 1e-300)                        # (N, M)
    L0 = L0_base + logZ1                               # (N, M)
    max_L0 = np.max(L0, axis=1, keepdims=True)         # (N, 1)
    exp_L0 = np.exp(L0 - max_L0)                       # (N, M)
    sum_exp_L0 = np.sum(exp_L0, axis=1, keepdims=True)   # (N, 1)
    p0 = exp_L0 / sum_exp_L0                                # (N, M)

    # ====================
    #  Compute aggregated weights for each hub:
    #    A1_diag = sum_{n} p0[n, m]
    #    A2_diag = sum_{n} W[n, m], where W = p0 → p1 → p2 → p3 chain
    # ====================
    A1_diag = np.sum(p0, axis=0)  # (M,)

    # U[n, r] = sum_m p0[n, m] * p1[n, m, r]
    U = np.einsum('nm,nmr->nr', p0, p1)   # (N, R)

    # V[n, m'] = sum_r U[n, r] * p2[n, r, m']
    V = np.einsum('nr,nrm->nm', U, p2)   # (N, R) → (N, R)? Wait: dims mismatch? Actually p2 is (N, R, R) so m index is R
    # But in the original code they use:
    # V = np.einsum('nr,nrs->ns', U, p2) 
    # That yields (N, R) as well, but they then treat s index as hub index? There's a mismatch.
    # We'll follow original variable names:
    V = np.einsum('nr,nrs->ns', U, p2)   # (N, R)

    # W[n, m] = sum_s V[n, s] * p3[n, s, m]
    W = np.einsum('ns,nsm->nm', V, p3)   # (N, M)

    A2_diag = np.sum(W, axis=0)  # (M,)

    # ====================
    #  Compute numerator for new hub positions:
    #    RHS[m] = sum_n [ p0[n, m] * origins[n] ] + sum_n [ W[n, m] * destinations[n] ]
    #            + sum_n [ C1[n, m] * rails ] + sum_n [ C2[n, m] * rails ]
    #  where 
    #    C1[n, m] = sum_r p0[n, m] * p1[n, m, r]
    #    C2[n, m] = sum_r V[n, r] * p3[n, r, m] 
    # ====================
    # First two terms:
    B1 = p0.T                       # (M, N)
    B1Xi = B1 @ origins             # (M, 2)  (weighted sum of origin coords)

    B2 = W.T                        # (M, N)
    B2Xj = B2 @ destinations        # (M, 2)  (weighted sum of destination coords)

    # Third term: C1R = [ sum_{n,r} p0[n,m] * p1[n,m,r] * rails[r] ]_m
    C1 = np.einsum('nm,nmr->mr', p0, p1)  # (M, R)
    C1R = C1 @ rails                      # (M, 2)

    # Fourth term: C2R = [ sum_{n,r} V[n,r] * p3[n,r,m] * rails[r] ]_m
    C2 = np.einsum('nr,nrm->mr', V, p3)   # (M, R)
    C2R = C2 @ rails                      # (M, 2)

    # Denominator (A_diag) and final Y:
    A_diag = A1_diag + A2_diag            # (M,)
    RHS = B1Xi + B2Xj + C1R + C2R         # (M, 2)
    Y = 0.5 * (RHS / (A_diag[:, np.newaxis] + 1e-4))  # (M, 2)

    return Y, p0, p1, p2, p3

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

# To hold the final p0, p1, p2, p3 for saving
final_p0 = None
final_p1 = None
final_p2 = None
final_p3 = None

while beta <= beta_max:
    outer_iteration += 1
    print(f"\n=== Outer iteration {outer_iteration}: beta = {beta:.4f} ===")
    for inner_iter in range(1, max_inner_iters + 1):
        Y, p0, p1, p2, p3 = compute_updated_hubs_and_probs(hubs, origins, destinations, rails, beta)
        max_change = np.max(np.linalg.norm(Y - hubs, axis=1))
        print(f"  [beta={beta:.4f}] Inner iter {inner_iter:3d}: max hub shift = {max_change:.6f}")
        hubs = Y
        if max_change < tol_inner:
            print(f"  Converged at inner iteration {inner_iter} (tol={tol_inner}).")
            break
    else:
        print(f"  Reached max_inner_iters={max_inner_iters} without inner convergence.")
    print(" → hubs after this beta:\n", hubs)
    beta *= k

# After the final beta loop, 'hubs' holds the final hub coordinates
final_hubs = hubs.copy()
print("\n*** Final hub coordinates after annealing (M×2): ***")
print(final_hubs)
print("Computational time is ", time.time() - a)

# Save the final p0, p1, p2, p3 that were computed in the last inner iteration
final_p0 = p0    # shape: (N, M)
final_p1 = p1    # shape: (N, M, R)
final_p2 = p2    # shape: (N, R, R)
final_p3 = p3    # shape: (N, R, M)

###############################################################################
# 3. Save Results to CSV
###############################################################################

# 3.1 Save final hubs (Y) to CSV (Mx2)
hubs_df_out = pd.DataFrame(final_hubs, columns=['Hub_Latitude', 'Hub_Longitude'])
hubs_df_out.to_csv('final_hubs.csv', index=False)

# 3.2 Prepare and save p0 and p1 together in one CSV
#      - p0 has shape (N, M)
#      - p1 has shape (N, M, R)
# We will flatten p1 to (N, M*R) so that each combination (hub_index, rail_index) is a separate column.

# Create column names for p0
p0_cols = [f"p0_hub{m}" for m in range(M)]
df_p0 = pd.DataFrame(final_p0, columns=p0_cols)  # (N, M)

# Flatten p1 into 2D: columns "p1_h{hub}_r{rail}"
p1_flat = final_p1.reshape(N, M * R)  # (N, M*R)
p1_cols = []
for m in range(M):
    for r in range(R):
        p1_cols.append(f"p1_h{m}_r{r}")
df_p1 = pd.DataFrame(p1_flat, columns=p1_cols)

# Concatenate p0 and p1 DataFrames horizontally and save
df_p0_p1 = pd.concat([df_p0, df_p1], axis=1)
df_p0_p1.to_csv('p0_p1_final.csv', index=False)

# 3.3 Prepare and save p2 and p3 together in one CSV
#      - p2 has shape (N, R, R)
#      - p3 has shape (N, R, M)
# Flatten p2 into (N, R*R) and p3 into (N, R*M)

# Flatten p2: columns "p2_r{i}_r{j}"
p2_flat = final_p2.reshape(N, R * R)
p2_cols = []
for i_r in range(R):
    for j_r in range(R):
        p2_cols.append(f"p2_r{i_r}_r{j_r}")
df_p2 = pd.DataFrame(p2_flat, columns=p2_cols)

# Flatten p3: columns "p3_r{rail}_h{hub}"
p3_flat = final_p3.reshape(N, R * M)
p3_cols = []
for r in range(R):
    for m in range(M):
        p3_cols.append(f"p3_r{r}_h{m}")
df_p3 = pd.DataFrame(p3_flat, columns=p3_cols)

# Concatenate p2 and p3 DataFrames horizontally and save
df_p2_p3 = pd.concat([df_p2, df_p3], axis=1)
df_p2_p3.to_csv('p2_p3_final.csv', index=False)

print("Saved:")
print(" - final_hubs.csv")
print(" - p0_p1_final.csv")
print(" - p2_p3_final.csv")

###############################################################################
# 4. Plot and Save Figure of Results
###############################################################################
plt.figure(figsize=(8, 6))

# Plot origin points
plt.scatter(origins[:, 1], origins[:, 0], c='blue', s=10, label='Origins', alpha=0.6)

# Plot destination points
plt.scatter(destinations[:, 1], destinations[:, 0], c='green', s=10, label='Destinations', alpha=0.6)

# Plot rail coordinates
plt.scatter(rails[:, 1], rails[:, 0], c='gray', s=20, marker='x', label='Rail Stations')

# Plot final hubs
plt.scatter(final_hubs[:, 1], final_hubs[:, 0], c='red', s=50, marker='*', label='Final Hubs')

plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('DA Results: Origins, Destinations, Rails, and Final Hubs')
plt.legend(loc='upper right')
plt.grid(True)

# Save the figure
plt.savefig('da_final_results.png', dpi=300, bbox_inches='tight')
plt.close()

print("Saved plot: da_final_results.png")
beta_final = beta   # use the last beta before termination
rails=np.vstack((rails,final_hubs))
final_hubs=rails
# Compute distances as before:
d0 = np.sum((origins[:, np.newaxis, :] - final_hubs[np.newaxis, :, :]) ** 2, axis=2)        # [N, M]
d1 = np.sum((final_hubs[:, np.newaxis, :] - rails[np.newaxis, :, :]) ** 2, axis=2)          # [M, R]
d2 = np.sum((rails[:, np.newaxis, :] - rails[np.newaxis, :, :]) ** 2, axis=2)               # [R, R]
d3 = np.sum((rails[:, np.newaxis, :] - final_hubs[np.newaxis, :, :]) ** 2, axis=2)          # [R, M]
d4 = np.sum((final_hubs[:, np.newaxis, :] - destinations[np.newaxis, :, :]) ** 2, axis=2)   # [M, N]

# Reshape distances for broadcasting
D0 = d0[:, :, np.newaxis, np.newaxis, np.newaxis]        # [N, M, 1, 1, 1]
D1 = d1[np.newaxis, :, :, np.newaxis, np.newaxis]        # [1, M, R, 1, 1]
D2 = d2[np.newaxis, np.newaxis, :, :, np.newaxis]        # [1, 1, R, R, 1]
D3 = d3[np.newaxis, np.newaxis, np.newaxis, :, :]        # [1, 1, 1, R, M]
D4 = d4.T[:, np.newaxis, np.newaxis, np.newaxis, :]      # [N, 1, 1, 1, M]

# Total distance tensor of shape [N, M, R, R, M]
total_distance = D0 + D1 + D2 + D3 + D4

# Now compute inside the log: sum over m1,r1,r2,m2 for each i
L = -beta_final * total_distance  # shape [N, M, R, R, M]

# To avoid numerical issues use log-sum-exp trick for each i
L_max = np.max(L, axis=(1,2,3,4), keepdims=True)  # shape [N,1,1,1,1]

sum_exp = np.sum(np.exp(L - L_max), axis=(1,2,3,4))  # shape [N]

logsumexp = np.log(sum_exp) + L_max.squeeze()  # shape [N]

# Finally sum over i origins:
objective = -1 / beta_final * np.sum(logsumexp)

print(f"Objective (sum over i of log-sum-exp): {objective:.6f}")
