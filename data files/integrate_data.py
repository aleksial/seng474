"""
Data Integration Script for Bus Ridership Prediction
Maps DA-level demand metrics to bus stops and calculates key features
Focus: Distance to schools/healthcare (priority features)
"""

import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from sklearn.neighbors import BallTree
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("BUS RIDERSHIP PREDICTION - DATA INTEGRATION")
print("=" * 80)

# Set up paths
data_dir = Path("data files")
output_dir = Path("processed_data")
output_dir.mkdir(exist_ok=True)

# ============================================================================
# 1. LOAD AND PREPARE BUS STOPS
# ============================================================================
print("\n1. Loading and converting bus stops...")
nodes_df = pd.read_csv(data_dir / "roads" / "bus_network_nodes.csv")
print(f"   Loaded {len(nodes_df)} bus stops")

# Convert from Web Mercator (EPSG:3857) to WGS84 (EPSG:4326) for lat/lon
# Create GeoDataFrame with projected coordinates
gdf_nodes = gpd.GeoDataFrame(
    nodes_df,
    geometry=[Point(x, y) for x, y in zip(nodes_df['x'], nodes_df['y'])],
    crs='EPSG:3857'  # Web Mercator
)

# Convert to lat/lon
gdf_nodes = gdf_nodes.to_crs('EPSG:4326')
nodes_df['longitude'] = gdf_nodes.geometry.x
nodes_df['latitude'] = gdf_nodes.geometry.y
print(f"   [OK] Converted coordinates to lat/lon")

# ============================================================================
# 2. LOAD DEMAND DATA (DA-LEVEL)
# ============================================================================
print("\n2. Loading demand data...")
demand_df = pd.read_csv(data_dir / "demand" / "streamlined_transit_data.csv")
print(f"   Loaded {len(demand_df)} census areas")

# Create GeoDataFrame for spatial operations
gdf_demand = gpd.GeoDataFrame(
    demand_df,
    geometry=[Point(lon, lat) for lon, lat in zip(demand_df['longitude'], demand_df['latitude'])],
    crs='EPSG:4326'
)

# ============================================================================
# 3. MAP DA DEMAND TO BUS STOPS (SPATIAL JOIN)
# ============================================================================
print("\n3. Mapping DA-level demand to bus stops...")

# Convert to projected CRS for accurate distance calculations (meters)
gdf_nodes_proj = gdf_nodes.to_crs('EPSG:32610')  # UTM Zone 10N (Victoria area)
gdf_demand_proj = gdf_demand.to_crs('EPSG:32610')

# Create BallTree for efficient nearest neighbor search
# Use projected coordinates (centroids after projection)
demand_centroids = gdf_demand_proj.geometry.centroid
demand_coords_proj = np.array([[p.x, p.y] for p in demand_centroids])

node_centroids = gdf_nodes_proj.geometry
node_coords_proj = np.array([[p.x, p.y] for p in node_centroids])

tree = BallTree(demand_coords_proj, metric='euclidean')
distances, indices = tree.query(node_coords_proj, k=1)

# Convert distances from meters to kilometers
distances_km = distances.flatten() / 1000

# Map demand features to each bus stop
demand_cols_to_map = [
    'Population_Density_Demand',
    'Commute_PT_Demand',
    'Bus_Stop_Proximity',
    'Route_Coverage_Percent',
    'Bus_Stop_Density',
    'Income_Demand',
    'Overall_Demand',
    'Employment_Demand',
    'Commute_Duration_Demand',
    'Overall_Accessibility'
]

# Initialize new columns in nodes_df
for col in demand_cols_to_map:
    nodes_df[f'nearest_{col}'] = demand_df.iloc[indices.flatten()][col].values
nodes_df['distance_to_nearest_da_km'] = distances_km

print(f"   [OK] Mapped demand metrics to {len(nodes_df)} bus stops")
print(f"   Average distance to nearest DA: {distances_km.mean():.3f} km")

# ============================================================================
# 4. CALCULATE DISTANCE TO SCHOOLS (PRIORITY FEATURE)
# ============================================================================
print("\n4. Calculating distance to schools (PRIORITY)...")

# Load schools data
schools_df = pd.read_csv(data_dir / "POI" / "schools.csv")

# Filter for Victoria area only (based on lat/lon bounds from exploration)
victoria_bounds = {
    'lon_min': -123.45,
    'lon_max': -123.30,
    'lat_min': 48.40,
    'lat_max': 48.50
}

victoria_schools = schools_df[
    (schools_df['longitude'] >= victoria_bounds['lon_min']) &
    (schools_df['longitude'] <= victoria_bounds['lon_max']) &
    (schools_df['latitude'] >= victoria_bounds['lat_min']) &
    (schools_df['latitude'] <= victoria_bounds['lat_max'])
].copy()

print(f"   Found {len(victoria_schools)} schools in Victoria area")

if len(victoria_schools) > 0:
    # Create GeoDataFrame for schools
    gdf_schools = gpd.GeoDataFrame(
        victoria_schools,
        geometry=[Point(lon, lat) for lon, lat in 
                 zip(victoria_schools['longitude'], victoria_schools['latitude'])],
        crs='EPSG:4326'
    )
    gdf_schools_proj = gdf_schools.to_crs('EPSG:32610')
    
    # Calculate distances
    school_coords_proj = np.array([[p.x, p.y] for p in gdf_schools_proj.geometry])
    tree_schools = BallTree(school_coords_proj, metric='euclidean')
    distances_schools, _ = tree_schools.query(node_coords_proj, k=1)
    distances_schools_km = distances_schools.flatten() / 1000
    
    nodes_df['distance_to_nearest_school_km'] = distances_schools_km
    
    # Count schools within buffer zones
    for buffer_km in [0.5, 1.0, 2.0]:
        buffer_m = buffer_km * 1000
        counts = []
        for node_point in node_centroids:
            buffer_zone = node_point.buffer(buffer_m)
            count = gdf_schools_proj.geometry.within(buffer_zone).sum()
            counts.append(count)
        nodes_df[f'schools_within_{buffer_km}km'] = counts
    
    print(f"   [OK] Calculated school distances")
    print(f"   Average distance to nearest school: {distances_schools_km.mean():.3f} km")
else:
    print("   [WARNING]  No schools found in Victoria area - check filtering")
    nodes_df['distance_to_nearest_school_km'] = np.nan
    for buffer_km in [0.5, 1.0, 2.0]:
        nodes_df[f'schools_within_{buffer_km}km'] = 0

# ============================================================================
# 5. CALCULATE DISTANCE TO HEALTHCARE (PRIORITY FEATURE)
# ============================================================================
print("\n5. Calculating distance to healthcare facilities (PRIORITY)...")

healthcare_df = pd.read_csv(data_dir / "POI" / "healthcare_facilities.csv")
print(f"   Found {len(healthcare_df)} healthcare facilities")

# Create GeoDataFrame for healthcare
gdf_healthcare = gpd.GeoDataFrame(
    healthcare_df,
    geometry=[Point(lon, lat) for lon, lat in 
             zip(healthcare_df['longitude'], healthcare_df['latitude'])],
    crs='EPSG:4326'
)
gdf_healthcare_proj = gdf_healthcare.to_crs('EPSG:32610')

# Calculate distances
healthcare_coords_proj = np.array([[p.x, p.y] for p in gdf_healthcare_proj.geometry])
tree_healthcare = BallTree(healthcare_coords_proj, metric='euclidean')
distances_healthcare, _ = tree_healthcare.query(node_coords_proj, k=1)
distances_healthcare_km = distances_healthcare.flatten() / 1000

nodes_df['distance_to_nearest_healthcare_km'] = distances_healthcare_km

# Count healthcare facilities within buffer zones
for buffer_km in [0.5, 1.0, 2.0]:
    buffer_m = buffer_km * 1000
    counts = []
    for node_point in node_centroids:
        buffer_zone = node_point.buffer(buffer_m)
        count = gdf_healthcare_proj.geometry.within(buffer_zone).sum()
        counts.append(count)
    nodes_df[f'healthcare_within_{buffer_km}km'] = counts

print(f"   [OK] Calculated healthcare distances")
print(f"   Average distance to nearest healthcare: {distances_healthcare_km.mean():.3f} km")

# ============================================================================
# 6. CALCULATE DISTANCE TO POPULATION CENTERS
# ============================================================================
print("\n6. Calculating distance to population centers...")

pop_df = pd.read_csv(data_dir / "population" / "victoria_census_da.csv")

# Create GeoDataFrame for population
gdf_pop = gpd.GeoDataFrame(
    pop_df,
    geometry=[Point(lon, lat) for lon, lat in 
             zip(pop_df['centroid_long'], pop_df['centroid_lat'])],
    crs='EPSG:4326'
)
gdf_pop_proj = gdf_pop.to_crs('EPSG:32610')

# Calculate distances to high-population areas (top 25% by population)
pop_threshold = pop_df['pop'].quantile(0.75)
high_pop = pop_df[pop_df['pop'] >= pop_threshold]
gdf_high_pop_proj = gdf_pop_proj[pop_df['pop'] >= pop_threshold]

if len(gdf_high_pop_proj) > 0:
    high_pop_coords_proj = np.array([[p.x, p.y] for p in gdf_high_pop_proj.geometry])
    tree_pop = BallTree(high_pop_coords_proj, metric='euclidean')
    distances_pop, _ = tree_pop.query(node_coords_proj, k=1)
    distances_pop_km = distances_pop.flatten() / 1000
    nodes_df['distance_to_high_pop_area_km'] = distances_pop_km
    
    # Calculate weighted population within buffers (using all population points)
    pop_coords_proj = np.array([[p.x, p.y] for p in gdf_pop_proj.geometry])
    for buffer_km in [0.5, 1.0, 2.0]:
        buffer_m = buffer_km * 1000
        weighted_pop = []
        for node_point in node_centroids:
            buffer_zone = node_point.buffer(buffer_m)
            within = gdf_pop_proj.geometry.within(buffer_zone)
            total_pop = pop_df.loc[within, 'pop'].sum() if within.any() else 0
            weighted_pop.append(total_pop)
        nodes_df[f'population_within_{buffer_km}km'] = weighted_pop
    
    print(f"   [OK] Calculated population features")
else:
    nodes_df['distance_to_high_pop_area_km'] = np.nan
    for buffer_km in [0.5, 1.0, 2.0]:
        nodes_df[f'population_within_{buffer_km}km'] = 0

# ============================================================================
# 7. NETWORK FEATURES
# ============================================================================
print("\n7. Calculating network features...")

# Load edges to calculate route connectivity
edges_df = pd.read_csv(data_dir / "roads" / "bus_network_edges.csv")

# Count how many edges connect to each node
from_counts = edges_df['from_node'].value_counts()
to_counts = edges_df['to_node'].value_counts()
all_counts = pd.concat([from_counts, to_counts]).groupby(level=0).sum()

nodes_df['route_connections'] = nodes_df['node_id'].map(all_counts).fillna(0).astype(int)

# Calculate bus stop density (stops per km² in surrounding area)
# Use a 1km buffer to count nearby stops
buffer_1km = 1000  # meters
stop_density = []
for node_point in node_centroids:
    buffer_zone = node_point.buffer(buffer_1km)
    nearby_stops = node_centroids.within(buffer_zone).sum() - 1  # -1 to exclude self
    # Calculate area in km² (approximate)
    area_km2 = (np.pi * buffer_1km**2) / 1_000_000
    density = nearby_stops / area_km2 if area_km2 > 0 else 0
    stop_density.append(density)

nodes_df['bus_stop_density_per_km2'] = stop_density

print(f"   [OK] Calculated network features")

# ============================================================================
# 8. CREATE TARGET VARIABLE (RIDERSHIP PROXY)
# ============================================================================
print("\n8. Creating target variable (ridership proxy)...")

# Use Overall_Demand as primary proxy, but scale it to be more interpretable
# We'll create a "predicted_ridership" that's based on demand metrics
# This is a proxy since we don't have actual ridership data

# Combine multiple demand signals
nodes_df['ridership_proxy'] = (
    nodes_df['nearest_Overall_Demand'] * 10 +  # Base demand (scaled)
    nodes_df['nearest_Commute_PT_Demand'] * 5 +  # Transit commuters
    nodes_df['nearest_Population_Density_Demand'] * 3  # Population density
)

# Add inverse distance penalties (closer to high-demand areas = higher ridership)
nodes_df['ridership_proxy'] = nodes_df['ridership_proxy'] / (1 + nodes_df['distance_to_nearest_da_km'])

# Normalize to reasonable range (0-100 passengers per day estimate)
nodes_df['ridership_proxy'] = (
    (nodes_df['ridership_proxy'] - nodes_df['ridership_proxy'].min()) /
    (nodes_df['ridership_proxy'].max() - nodes_df['ridership_proxy'].min()) * 100
)

print(f"   [OK] Created ridership proxy (0-100 scale)")
print(f"   Mean ridership proxy: {nodes_df['ridership_proxy'].mean():.2f}")
print(f"   Min: {nodes_df['ridership_proxy'].min():.2f}, Max: {nodes_df['ridership_proxy'].max():.2f}")

# ============================================================================
# 9. SAVE PROCESSED DATA
# ============================================================================
print("\n9. Saving processed data...")

output_file = output_dir / "bus_stops_with_features.csv"
nodes_df.to_csv(output_file, index=False)
print(f"   [OK] Saved to {output_file}")
print(f"   Total features: {len(nodes_df.columns)}")
print(f"   Total stops: {len(nodes_df)}")

# Create a summary
print("\n" + "=" * 80)
print("FEATURE SUMMARY")
print("=" * 80)
print("\nPriority Features (Distance-based):")
print(f"  - distance_to_nearest_school_km: {nodes_df['distance_to_nearest_school_km'].notna().sum()} stops")
print(f"  - distance_to_nearest_healthcare_km: {nodes_df['distance_to_nearest_healthcare_km'].notna().sum()} stops")
print(f"  - schools_within_0.5km: {nodes_df['schools_within_0.5km'].sum()} total")
print(f"  - healthcare_within_0.5km: {nodes_df['healthcare_within_0.5km'].sum()} total")

print("\nDemand Features (from DA-level data):")
demand_features = [col for col in nodes_df.columns if col.startswith('nearest_')]
print(f"  - {len(demand_features)} demand metrics mapped")

print("\nNetwork Features:")
print(f"  - route_connections: {nodes_df['route_connections'].sum()} total connections")
print(f"  - bus_stop_density_per_km2: avg {nodes_df['bus_stop_density_per_km2'].mean():.2f}")

print("\nTarget Variable:")
print(f"  - ridership_proxy: range {nodes_df['ridership_proxy'].min():.2f} - {nodes_df['ridership_proxy'].max():.2f}")

print("\n" + "=" * 80)
print("[OK] DATA INTEGRATION COMPLETE")
print("=" * 80)

