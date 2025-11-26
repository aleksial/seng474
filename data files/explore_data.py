"""
Quick data exploration script to understand the available datasets
for bus ridership prediction.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Set up paths
data_dir = Path("data files")

print("=" * 80)
print("BUS RIDERSHIP PREDICTION - DATA EXPLORATION")
print("=" * 80)

# 1. Transit Demand Data
print("\n1. TRANSIT DEMAND DATA")
print("-" * 80)
try:
    demand_df = pd.read_csv(data_dir / "demand" / "streamlined_transit_data.csv")
    print(f"Shape: {demand_df.shape}")
    print(f"\nColumns: {list(demand_df.columns)}")
    print(f"\nFirst few rows:")
    print(demand_df.head(3))
    print(f"\nDemand metrics summary:")
    demand_cols = [col for col in demand_df.columns if 'Demand' in col and '_distance' not in col]
    print(demand_df[demand_cols].describe())
    print(f"\nMissing values:")
    print(demand_df.isnull().sum()[demand_df.isnull().sum() > 0])
except Exception as e:
    print(f"Error loading demand data: {e}")

# 2. Bus Network Nodes
print("\n\n2. BUS NETWORK NODES (STOPS)")
print("-" * 80)
try:
    nodes_df = pd.read_csv(data_dir / "roads" / "bus_network_nodes.csv")
    print(f"Shape: {nodes_df.shape}")
    print(f"Columns: {list(nodes_df.columns)}")
    print(f"\nCoordinate ranges:")
    print(f"  X: {nodes_df['x'].min():.2f} to {nodes_df['x'].max():.2f}")
    print(f"  Y: {nodes_df['y'].min():.2f} to {nodes_df['y'].max():.2f}")
    print(f"\nNote: Coordinates appear to be in a projected system (not lat/lon)")
except Exception as e:
    print(f"Error loading nodes data: {e}")

# 3. Bus Network Edges
print("\n\n3. BUS NETWORK EDGES (ROUTES)")
print("-" * 80)
try:
    edges_df = pd.read_csv(data_dir / "roads" / "bus_network_edges.csv")
    print(f"Shape: {edges_df.shape}")
    print(f"Unique routes: {edges_df['from_node'].nunique()} to {edges_df['to_node'].nunique()}")
    print(f"Average edge length: {edges_df['length'].mean():.2f} units")
except Exception as e:
    print(f"Error loading edges data: {e}")

# 4. Population Data
print("\n\n4. POPULATION DATA")
print("-" * 80)
try:
    pop_df = pd.read_csv(data_dir / "population" / "victoria_census_da.csv")
    print(f"Shape: {pop_df.shape}")
    print(f"Columns: {list(pop_df.columns)}")
    if 'pop' in pop_df.columns:
        print(f"\nPopulation statistics:")
        print(f"  Total population: {pop_df['pop'].sum():,}")
        print(f"  Mean per DA: {pop_df['pop'].mean():.1f}")
        print(f"  Median per DA: {pop_df['pop'].median():.1f}")
except Exception as e:
    print(f"Error loading population data: {e}")

# 5. Healthcare Facilities
print("\n\n5. HEALTHCARE FACILITIES (POI)")
print("-" * 80)
try:
    health_df = pd.read_csv(data_dir / "POI" / "healthcare_facilities.csv")
    print(f"Shape: {health_df.shape}")
    print(f"Facilities with coordinates: {health_df[['longitude', 'latitude']].notna().all(axis=1).sum()}")
    if 'longitude' in health_df.columns and 'latitude' in health_df.columns:
        print(f"\nLocation bounds:")
        print(f"  Longitude: {health_df['longitude'].min():.4f} to {health_df['longitude'].max():.4f}")
        print(f"  Latitude: {health_df['latitude'].min():.4f} to {health_df['latitude'].max():.4f}")
except Exception as e:
    print(f"Error loading healthcare data: {e}")

# 6. Schools
print("\n\n6. SCHOOLS (POI)")
print("-" * 80)
try:
    schools_df = pd.read_csv(data_dir / "POI" / "schools.csv")
    print(f"Shape: {schools_df.shape}")
    print(f"Schools with coordinates: {schools_df[['longitude', 'latitude']].notna().all(axis=1).sum()}")
    if 'LOCALITY' in schools_df.columns:
        print(f"\nTop localities:")
        print(schools_df['LOCALITY'].value_counts().head(10))
except Exception as e:
    print(f"Error loading schools data: {e}")

# 7. Data Integration Opportunities
print("\n\n7. DATA INTEGRATION OPPORTUNITIES")
print("-" * 80)
print("""
Potential feature engineering:
1. Map DA-level demand metrics to bus stops (spatial join)
2. Calculate distance from each stop to:
   - Nearest healthcare facility
   - Nearest school
   - Nearest high-population DA
3. Count POIs within buffer zones (500m, 1km, 2km)
4. Calculate network features:
   - Number of routes per stop
   - Connectivity metrics
   - Distance to city center
5. Aggregate population data to stop-level features
""")

print("\n" + "=" * 80)
print("EXPLORATION COMPLETE")
print("=" * 80)

