# Load libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from pyomo.environ import *
from scipy.spatial.distance import cdist

# Load data
quantities_df = pd.read_csv("quantities_all.csv")
hubs = pd.read_csv("big_hubs.csv")

# Parameters
alpha = 1  # Discount factor for inter-hub transport
P = 10      # Number of hubs to open

# Start timer
start_time = time.time()

# Extract coordinates
sources = quantities_df[['Start_Lat', 'Start_Lon']].to_numpy()
destinations = quantities_df[['Destination_Lat', 'Destination_Lon']].to_numpy()
hub_coords = hubs[['Latitude', 'Longitude']].to_numpy()

num_pairs = sources.shape[0]
num_hubs = hub_coords.shape[0]

# Compute pairwise Euclidean distances (vectorized)
dist_ik = cdist(sources, hub_coords)                  # Source to hub
dist_km = cdist(hub_coords, hub_coords)               # Hub to hub
dist_mj = cdist(destinations, hub_coords)             # Hub to destination

# Total distance computation: d_ijkm
# d_ijkm[p, k, m] = dist_ik[p, k] + alpha * dist_km[k, m] + dist_mj[p, m]
d_ijkm = dist_ik[:, :, None] + alpha * dist_km[None, :, :] + dist_mj[:, None, :]

# Pyomo model
model = ConcreteModel()

# Decision variables
model.X = Var(range(num_pairs), range(num_hubs), range(num_hubs), domain=NonNegativeReals)
model.y = Var(range(num_hubs), domain=Binary)

# Objective function (weighted by quantity)
model.obj = Objective(
    expr=sum(quantities_df.iloc[p]['Quantity'] * d_ijkm[p, k, m] * model.X[p, k, m]
             for p in range(num_pairs)
             for k in range(num_hubs)
             for m in range(num_hubs)),
    sense=minimize
)

# Constraints
model.flow_constraint = ConstraintList()
for p in range(num_pairs):
    model.flow_constraint.add(
        sum(model.X[p, k, m] for k in range(num_hubs) for m in range(num_hubs)) == 1
    )

model.routing_constraint_1 = ConstraintList()
for p in range(num_pairs):
    for m in range(num_hubs):
        model.routing_constraint_1.add(
            sum(model.X[p, k, m] for k in range(num_hubs)) <= model.y[m]
        )

model.routing_constraint_2 = ConstraintList()
for p in range(num_pairs):
    for k in range(num_hubs):
        model.routing_constraint_2.add(
            sum(model.X[p, k, m] for m in range(num_hubs)) <= model.y[k]
        )

model.hub_opening_constraint = Constraint(
    expr=sum(model.y[k] for k in range(num_hubs)) == P
)

# Solve
solver = SolverFactory('gurobi')
solver.solve(model)

# Extract and print results
y_values = [model.y[k]() for k in range(num_hubs)]
selected_hubs = [hubs.iloc[k] for k in range(num_hubs) if y_values[k] > 0.5]

print("Selected Hubs:")
for k in range(num_hubs):
    if y_values[k] > 0.5:
        print(f"Hub {k} ({hubs.iloc[k]['Latitude']}, {hubs.iloc[k]['Longitude']}) is OPEN")

print("\nFlow Fractions (X values):")
for p in range(num_pairs):
    for k in range(num_hubs):
        for m in range(num_hubs):
            flow_value = model.X[p, k, m]()
            if flow_value > 0:
                print(f"Source {p} -> Hub1 {k} -> Hub2 {m}: Flow = {flow_value}")

print("\nTotal Objective Value (Total Cost):")
print(model.obj())

# Save results
results = []
for p in range(num_pairs):
    for k in range(num_hubs):
        for m in range(num_hubs):
            flow = model.X[p, k, m]()
            if flow > 0:
                results.append([
                    sources[p, 0], sources[p, 1],
                    destinations[p, 0], destinations[p, 1],
                    hub_coords[k, 0], hub_coords[k, 1],
                    hub_coords[m, 0], hub_coords[m, 1],
                    flow
                ])

df_results = pd.DataFrame(results, columns=[
    'Source_Lat', 'Source_Lon', 'Dest_Lat', 'Dest_Lon',
    'Hub1_Lat', 'Hub1_Lon', 'Hub2_Lat', 'Hub2_Lon', 'Flow_Fraction'
])
df_results.to_csv('optimized_flow_results_fixed_quantity.csv', index=False)

# Save summary
end_time = time.time()
print(end_time-start_time)
output_txt_path = "hub_optimization_summary.txt"
with open(output_txt_path, "w") as f:
    f.write("Selected Hubs (Lat, Lon):\n")
    for k in range(num_hubs):
        if y_values[k] > 0.5:
            lat = hubs.iloc[k]['Latitude']
            lon = hubs.iloc[k]['Longitude']
            f.write(f"Hub {k}: Latitude = {lat:.6f}, Longitude = {lon:.6f}\n")

    f.write("\nTotal Objective Value (Total Cost):\n")
    f.write(f"{model.obj():.6f}\n")

    total_time = end_time - start_time
    f.write("\nComputation Time (seconds):\n")
    f.write(f"{total_time:.2f}\n")

print(f"\nSaved summary to {output_txt_path}")
