# Load libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from pyomo.environ import *

# Load data
quantities_df = pd.read_csv("quantities_all.csv")  # Still using for locations
hubs = pd.read_csv("hubs.csv")

# Parameters
alpha = 1  # Discount factor for inter-hub transport
P = 5      # Number of hubs to open

# Start timer
start_time = time.time()

# Squared Euclidean distance function
def squared_euclidean(lat1, lon1, lat2, lon2):
    return ((lat2 - lat1) ** 2 + (lon2 - lon1) ** 2)

# Compute distances
num_pairs = len(quantities_df)
num_hubs = len(hubs)

dist_ik = np.zeros((num_pairs, num_hubs))  # Source to hub
dist_km = np.zeros((num_hubs, num_hubs))  # Hub to hub
dist_mj = np.zeros((num_pairs, num_hubs))  # Hub to destination

# Source to hub distances
for p, row in quantities_df.iterrows():
    for k in range(num_hubs):
        dist_ik[p, k] = squared_euclidean(row['Start_Lat'], row['Start_Lon'],
                                          hubs.iloc[k]['Latitude'], hubs.iloc[k]['Longitude'])

# Hub to hub distances
for k in range(num_hubs):
    for m in range(num_hubs):
        dist_km[k, m] = squared_euclidean(hubs.iloc[k]['Latitude'], hubs.iloc[k]['Longitude'],
                                          hubs.iloc[m]['Latitude'], hubs.iloc[m]['Longitude'])

# Hub to destination distances
for p, row in quantities_df.iterrows():
    for m in range(num_hubs):
        dist_mj[p, m] = squared_euclidean(hubs.iloc[m]['Latitude'], hubs.iloc[m]['Longitude'],
                                          row['Destination_Lat'], row['Destination_Lon'])

# Total distance computation: d_ijkm
d_ijkm = np.zeros((num_pairs, num_hubs, num_hubs))
for p in range(num_pairs):
    for k in range(num_hubs):
        for m in range(num_hubs):
            d_ijkm[p, k, m] = dist_ik[p, k] + alpha * dist_km[k, m] + dist_mj[p, m]

# Pyomo model
model = ConcreteModel()

# Decision variables
model.X = Var(range(num_pairs), range(num_hubs), range(num_hubs), domain=NonNegativeReals)
model.y = Var(range(num_hubs), domain=Binary)

# Objective function: Minimize total transportation cost (with quantity = 1 for all)
model.obj = Objective(
    expr=sum(d_ijkm[p, k, m] * model.X[p, k, m]
             for p in range(num_pairs)
             for k in range(num_hubs)
             for m in range(num_hubs)),
    sense=minimize
)

# Flow constraint: total flow for each p must be 1
model.flow_constraint = ConstraintList()
for p in range(num_pairs):
    model.flow_constraint.add(
        sum(model.X[p, k, m] for k in range(num_hubs) for m in range(num_hubs)) == 1
    )

# Routing constraint: Exit hub m must be open
model.routing_constraint_1 = ConstraintList()
for p in range(num_pairs):
    for m in range(num_hubs):
        model.routing_constraint_1.add(
            sum(model.X[p, k, m] for k in range(num_hubs)) <= model.y[m]
        )

# Routing constraint: Entry hub k must be open
model.routing_constraint_2 = ConstraintList()
for p in range(num_pairs):
    for k in range(num_hubs):
        model.routing_constraint_2.add(
            sum(model.X[p, k, m] for m in range(num_hubs)) <= model.y[k]
        )

# Hub opening constraint: Exactly P hubs must be opened
model.hub_opening_constraint = Constraint(
    expr=sum(model.y[k] for k in range(num_hubs)) == P
)

# New constraint: Entry hub cannot be the same as exit hub (k != m)

# Solve the model
solver = SolverFactory('gurobi')
solver.solve(model)

# Extract hub selection results
y_values = [model.y[k]() for k in range(num_hubs)]
selected_hubs = [hubs.iloc[k] for k in range(num_hubs) if y_values[k] > 0.5]

# Print selected hubs
print("Selected Hubs:")
for k in range(num_hubs):
    if y_values[k] > 0.5:
        print(f"Hub {k} ({hubs.iloc[k]['Latitude']}, {hubs.iloc[k]['Longitude']}) is OPEN")

# Print the flow fractions (X values)
print("\nFlow Fractions (X values):")
for p in range(num_pairs):
    for k in range(num_hubs):
        for m in range(num_hubs):
            flow_value = model.X[p, k, m]()
            if flow_value > 0:
                print(f"Source {p} -> Hub1 {k} -> Hub2 {m}: Flow = {flow_value}")

# Print the total objective value (total cost)
print("\nTotal Objective Value (Total Cost):")
print(model.obj())

# Save results to CSV
results = []
for p in range(num_pairs):
    for k in range(num_hubs):
        for m in range(num_hubs):
            flow = model.X[p, k, m]()
            if flow > 0:
                results.append([
                    quantities_df.iloc[p]['Start_Lat'], quantities_df.iloc[p]['Start_Lon'],
                    quantities_df.iloc[p]['Destination_Lat'], quantities_df.iloc[p]['Destination_Lon'],
                    hubs.iloc[k]['Latitude'], hubs.iloc[k]['Longitude'],
                    hubs.iloc[m]['Latitude'], hubs.iloc[m]['Longitude'],
                    flow
                ])

df_results = pd.DataFrame(results, columns=[
    'Source_Lat', 'Source_Lon', 'Dest_Lat', 'Dest_Lon',
    'Hub1_Lat', 'Hub1_Lon', 'Hub2_Lat', 'Hub2_Lon', 'Flow_Fraction'
])
df_results.to_csv('optimized_flow_results_fixed_quantity.csv', index=False)

# Time taken
end_time = time.time()
print(f"Computation time: {end_time - start_time:.2f} seconds")
# ---------------- Save Hubs, Objective, and Time to TXT ----------------
output_txt_path = "hub_optimization_summary.txt"

with open(output_txt_path, "w") as f:
    # Selected hubs
    f.write("Selected Hubs (Lat, Lon):\n")
    for k in range(num_hubs):
        if y_values[k] > 0.5:
            lat = hubs.iloc[k]['Latitude']
            lon = hubs.iloc[k]['Longitude']
            f.write(f"Hub {k}: Latitude = {lat:.6f}, Longitude = {lon:.6f}\n")

    # Objective value
    f.write("\nTotal Objective Value (Total Cost):\n")
    f.write(f"{model.obj():.6f}\n")

    # Computation time
    total_time = end_time - start_time
    f.write("\nComputation Time (seconds):\n")
    f.write(f"{total_time:.2f}\n")

print(f"\nSaved summary to {output_txt_path}")
