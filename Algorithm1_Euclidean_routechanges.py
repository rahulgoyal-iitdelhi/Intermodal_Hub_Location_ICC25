import torch
import pandas as pd
import time
from scipy.spatial.distance import cdist
dtype = torch.float32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import numpy as np

# ---------------------- LOAD DATA ------------------------
quantities = pd.read_csv('quantities_new.csv')
hubs       = pd.read_csv('stations.csv')

# OD pairs as [P,2]
I = torch.tensor(
    quantities[['Start_Lat','Start_Lon']].values,
    dtype=dtype, device=device
)
J = torch.tensor(
    quantities[['Destination_Lat','Destination_Lon']].values,
    dtype=dtype, device=device
)

# Static hubs as [K0,2]
K0 = torch.tensor(
    hubs[['Latitude','Longitude']].values,
    dtype=dtype, device=device
)

P = I.size(0)
M = 3 # number of moving hubs

# Bounding box for initialization
lat_min, lat_max = float(hubs['Latitude'].min()), float(hubs['Latitude'].max())
lon_min, lon_max = float(hubs['Longitude'].min()), float(hubs['Longitude'].max())

# ---------------------- LOSS FUNCTION ------------------------
def loss_fn(mn, K0, beta):
    K_update = torch.cat([K0, mn], dim=0)  # combine static K0 and current mn
    K = K_update.size(0)

    d1 = torch.cdist(I, mn)           # [P, M]
    d2 = torch.cdist(mn, K_update)    # [M, K]
    d3 = torch.cdist(K_update, K_update)  # [K, K]
    d4 = torch.cdist(K_update, mn)    # [K, M]
    d5 = torch.cdist(mn, J)           # [M, P]

    D1 = d1.view(P, M, 1, 1, 1)
    D2 = d2.view(1, M, K, 1, 1)
    D3 = d3.view(1, 1, K, K, 1)
    D4 = d4.view(1, 1, 1, K, M)
    D5 = d5.t().view(P, 1, 1, 1, M)

    D = D1 + D2 + D3 + D4 + D5
    T = K * K
    D = D.view(P, M, T, M)

    lse = torch.logsumexp(-beta * D, dim=(1,2,3))
    return (-1.0/beta) * lse.sum()

# ---------------------- MAIN beta-LOOP w/ Adam + Scheduler ------------------------
beta      = 0.001
beta_max  = 1000
k         = 1.25

start = time.time()

while beta < beta_max:
    print(f"Current beta: {beta:.4f}")

    # Initialize fresh moving hubs (mn) randomly
    mn = torch.stack([
        torch.rand(M, device=device) * (lat_max - lat_min) + lat_min,
        torch.rand(M, device=device) * (lon_max - lon_min) + lon_min
    ], dim=1).requires_grad_(True)

    optimizer = torch.optim.Adam([mn], lr=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=20
    )

    num_steps = 500

    for step in range(1, num_steps+1):
        optimizer.zero_grad()
        loss = loss_fn(mn, K0, beta)
        loss.backward()
        optimizer.step()
        scheduler.step(loss.item())

        if step % 50 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"    inner iter {step:3d}: obj = {loss.item():.4f}, lr = {current_lr:.2e}")

    final_obj = loss_fn(mn, K0, beta).item()
    print(f"Final objective: {final_obj:.4f} after {num_steps} iterations")

    # Optionally print K_update size
    K_update = torch.cat([K0, mn.detach()], dim=0)
    print(f"K_update size: {K_update.shape} (static {K0.shape[0]} + moving {mn.shape[0]})\n")

    beta *= k

# Final message
print(f"Total time: {time.time() - start:.2f}s")
print(mn)
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# 7. Objective Function (NumPy-based Log-Sum-Exp using final_hubs)
# ----------------------------------------------------------------------------

# Final hub coordinates after optimization (assumed to be numpy array)
final_hubs = mn.detach().cpu().numpy()
#hubs=np.array(mn)
#final_hubs=hubs
# Reload input data to numpy
quantities_df = pd.read_csv('quantities_new.csv')
hubs_df = pd.read_csv('stations.csv')

origins = quantities_df[['Start_Lat', 'Start_Lon']].to_numpy()
destinations = quantities_df[['Destination_Lat', 'Destination_Lon']].to_numpy()
static_rails = hubs_df[['Latitude', 'Longitude']].to_numpy()

railupdate = np.vstack([static_rails, final_hubs])
#final_hubs=railupdate
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
