import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import warnings
warnings.filterwarnings('ignore')
from sklearn.neighbors import BallTree

# getting data frames of relevant features
demand_df = pd.read_csv("data files/demand/streamlined_transit_data.csv")
nodes_df = pd.read_csv("data files/roads/bus_network_nodes.csv")
edges_df = pd.read_csv("data files/roads/bus_network_edges.csv")
pop_df = pd.read_csv("data files/population/victoria_census_da.csv")
healthcare_df = pd.read_csv("data files/POI/healthcare_facilities.csv")
schools_df = pd.read_csv("data files/POI/schools.csv")

#---------------------------------------------------------------------------
# PREPARING GEODATAFRAMES FOR SPATIAL OPERATIONS

# PREPARING BUS STOPS
# convert from Web Mercator: x,y METERS system for web maps to 
# WGS84: (x,y) tuple geometry objects for GPS coordinate systems
# which will be our new lat/lon

# Web Mercator (EPSG:3857) to WGS84 (EPSG:4326)

gdf_nodes = gpd.GeoDataFrame( 
    nodes_df,
    geometry=[Point(x, y) for x, y in zip(nodes_df['x'], nodes_df['y'])],
    crs='EPSG:3857' 
)

gdf_nodes = gdf_nodes.to_crs('EPSG:4326')
nodes_df['longitude'] = gdf_nodes.geometry.x # new cols
nodes_df['latitude'] = gdf_nodes.geometry.y

# PREPARING DEMAND DATA
gdf_demand = gpd.GeoDataFrame(
    demand_df,
    geometry=[Point(lon, lat) for lon, lat in zip(demand_df['longitude'], demand_df['latitude'])],
    crs='EPSG:4326'
)

#---------------------------------------------------------------------------
# MAP DEMAND TO ITS NEAREST BUS STOP (SPATIAL JOIN EUCLIDEAN DISTANCE)

# convert demands and bus stops to projected Coordinate Reference System (CRS)
# for distance calculations
# UTM Zone 10N (Victoria area) is EPSG:32610 for accurate real distance
# as Web Mercator measurements distort actual distances to fit on the map
gdf_nodes_proj = gdf_nodes.to_crs('EPSG:32610') 
gdf_demand_proj = gdf_demand.to_crs('EPSG:32610')

# #GeoPandas sjoin_nearest to join demand to its closest stop 
# # (euclidean distance: meters)
nearest_demand = gdf_nodes_proj.sjoin_nearest(gdf_demand_proj, how="left", distance_col="demand_dist")
nearest_demand['demand_dist'] = nearest_demand['demand_dist'] / 1000 # meters to km

demand_cols_to_map = [
    'DAUID',
    'Population_Density_Demand',
    'Commute_PT_Demand',
    'Bus_Stop_Proximity',
    'Route_Coverage_Percent',
    'Bus_Stop_Density',
    'Income_Demand',
    'Overall_Demand',
    'Employment_Demand',
    'Commute_Duration_Demand',
    'Overall_Accessibility',
    'demand_dist'
]

# copy relevant cols back to nodes_df
for col in demand_cols_to_map:
    nodes_df[f'nearest_{col}'] = nearest_demand[f'{col}'].values

#---------------------------------------------------------------------------
# DISTANCE TO SCHOOLS FROM BUS STOP

# Filter for Victoria area only (based on lat/lon bounds from exploration)
victoria_bounds = {
    'lon_min': -123.45,
    'lon_max': -123.30,
    'lat_min': 48.40,
    'lat_max': 48.50
}

# Copy schools within Victoria's bounds
victoria_schools = schools_df[
    (schools_df['longitude'] >= victoria_bounds['lon_min']) &
    (schools_df['longitude'] <= victoria_bounds['lon_max']) &
    (schools_df['latitude'] >= victoria_bounds['lat_min']) &
    (schools_df['latitude'] <= victoria_bounds['lat_max'])
].copy()

if len(victoria_schools) > 0:
    # Create GeoDataFrame for schools within Victoria
    gdf_schools = gpd.GeoDataFrame(
        victoria_schools,
        geometry=[Point(lon, lat) for lon, lat in 
                 zip(victoria_schools['longitude'], victoria_schools['latitude'])],
        crs='EPSG:4326'
    )
    gdf_schools_proj = gdf_schools.to_crs('EPSG:32610')
    
    # Calculate distance to nearest school and map to node_df
    nearest_school = gdf_nodes_proj.sjoin_nearest(gdf_schools_proj, how="left", distance_col="school_dist")
    nearest_school['school_dist'] = nearest_school['school_dist'] / 1000
    nodes_df['nearest_school_dist'] = nearest_school['school_dist']
    
    # Count schools within buffer zones
    buffer_distances_km = [0.5, 1.0, 2.0]  # zone radius ADJUSTABLE

    for buffer_km in buffer_distances_km:
        buffer_m = buffer_km * 1000  # km to meters
        
        # Create buffer circle for each node
        buffers = gdf_nodes_proj.copy()
        buffers['geometry'] = buffers.geometry.buffer(buffer_m)
        
        # Spatial join to add the buffer zones to the school if it is within it
        # (school + buffer it's in for all schools)
        contained_school = gpd.sjoin(gdf_schools_proj, buffers, how='inner', predicate='within')

        # Count schools per bus stop
        # group by index of bus stops in 'buffers'
        counts = contained_school.groupby('index_right').size()
        
        # Map counts back to nodes_df
        # 0 if no school within the buffer zone
        nodes_df[f'schools_within_{buffer_km}km'] = nodes_df.index.map(counts).fillna(0).astype(int)
    
else:
    print("No schools found in Victoria")

#---------------------------------------------------------------------------
# DISTANCE TO HEALTHCARE FROM BUS STOP

# Copy healthcare within Victoria's bounds
victoria_healthcare = healthcare_df[
    (healthcare_df['longitude'] >= victoria_bounds['lon_min']) &
    (healthcare_df['longitude'] <= victoria_bounds['lon_max']) &
    (healthcare_df['latitude'] >= victoria_bounds['lat_min']) &
    (healthcare_df['latitude'] <= victoria_bounds['lat_max'])
].copy()

if len(victoria_healthcare) > 0:
    # Create GeoDataFrame for healthcare facilities within Victoria
    gdf_healthcare = gpd.GeoDataFrame(
        victoria_healthcare,
        geometry=[Point(lon, lat) for lon, lat in 
                 zip(victoria_healthcare['longitude'], victoria_healthcare['latitude'])],
        crs='EPSG:4326'
    )
    gdf_healthcare_proj = gdf_healthcare.to_crs('EPSG:32610')
    
    # Calculate distance to nearest healthcare and map to node_df
    nearest_healthcare = gdf_nodes_proj.sjoin_nearest(gdf_healthcare_proj, how="left", distance_col="healthcare_dist")
    nearest_healthcare['healthcare_dist'] = nearest_healthcare['healthcare_dist'] / 1000 # meters to km
    nodes_df['nearest_healthcare_dist'] = nearest_healthcare['healthcare_dist']
    
    # Count healthcare facilities within buffer zones
    buffer_distances_km = [0.5, 1.0, 2.0]  # zone radius ADJUSTABLE

    for buffer_km in buffer_distances_km:
        buffer_m = buffer_km * 1000  # km to meters
        
        # Create buffer circle for each node
        buffers = gdf_nodes_proj.copy()
        buffers['geometry'] = buffers.geometry.buffer(buffer_m)
        
        # Spatial join to add the buffer zones to the healthcare if it is within it
        # (healthcare + buffer it's in for all healthcare)
        contained_healthcare = gpd.sjoin(gdf_healthcare_proj, buffers, how='inner', predicate='within')

        # Count healthcare per bus stop
        # group by index of bus stops in 'buffers'
        counts = contained_healthcare.groupby('index_right').size()
        
        # Map counts back to nodes_df
        # 0 if no healthcare within the buffer zone
        nodes_df[f'healthcare_within_{buffer_km}km'] = nodes_df.index.map(counts).fillna(0).astype(int)

else:
    print("No healthcare facilities found in Victoria")

#---------------------------------------------------------------------------
# DISTANCE TO POPULATION CENTERS FROM BUS STOP

# Create GeoDataFrame for population
gdf_pop = gpd.GeoDataFrame(
    pop_df,
    geometry=[Point(lon, lat) for lon, lat in 
             zip(pop_df['centroid_long'], pop_df['centroid_lat'])],
    crs='EPSG:4326'
)
gdf_pop_proj = gdf_pop.to_crs('EPSG:32610')

# Only consider high population areas (top 25% by population)
pop_threshold = pop_df['pop'].quantile(0.75)
gdf_high_pop_proj = gdf_pop_proj[pop_df['pop'] >= pop_threshold]

if len(gdf_high_pop_proj) > 0:
    # Calculate distance to nearest high pop centroid and map to node_df
    nearest_pop = gdf_nodes_proj.sjoin_nearest(gdf_high_pop_proj, how="left", distance_col="pop_dist")
    nearest_pop['pop_dist'] = nearest_pop['pop_dist'] / 1000 # meters to km
    nodes_df['nearest_pop_dist'] = nearest_pop['pop_dist']
    
    # Calculate weighted population within buffers (using all population points)
    buffer_distances_km = [0.5, 1.0, 2.0]  # zone radius ADJUSTABLE
    for buffer_km in buffer_distances_km:
        buffer_m = buffer_km * 1000
        
        # Create buffer circle for each node
        buffers = gdf_nodes_proj.copy()
        buffers['geometry'] = buffers.geometry.buffer(buffer_m)
        
        # Spatial join to find population points within each buffer
        contained_pop = gpd.sjoin(gdf_pop_proj, buffers, how='inner', predicate='within')
        
        # Sum population per bus stop
        pop_sum = contained_pop.groupby('index_right')['pop'].sum()
        
        # Map total population back to nodes_df
        # 0 if no population within the buffer zone
        nodes_df[f'population_within_{buffer_km}km'] = nodes_df.index.map(pop_sum).fillna(0).astype(int)
    
else:
    print("No high population centroids found in Victoria")

#---------------------------------------------------------------------------
# NETWORK FEATURES OF EACH BUS STOP

# Count how many edges connect to each node
from_counts = edges_df['from_node'].value_counts()
to_counts = edges_df['to_node'].value_counts()
all_counts = pd.concat([from_counts, to_counts]).groupby(level=0).sum()

nodes_df['route_connections'] = nodes_df['node_id'].map(all_counts).fillna(0).astype(int)

# Calculate bus stop density (stops per km² in surrounding area)
# Use a 1km buffer to count nearby stops
# Density in this case is a better measure due to scalability (when buffers vary)
# especially when checking for coverage vs availability
# but since we are using the same 1km2 buffer size, using count works as well
buffer_1km = 1000  # meters

# Create 1 km buffer around each bus stop
buffers = gdf_nodes_proj.copy()
buffers['geometry'] = buffers.geometry.buffer(buffer_1km)

# Spatial join to find all bus stops within each buffer
contained_stops = gpd.sjoin(gdf_nodes_proj, buffers, how='inner', predicate='within')

# Count nearby stops per bus stop (subtract 1 to exclude self)
stop_counts = contained_stops.groupby('index_right').size() - 1

# Calculate density: number of stops / area of buffer (km²)
area_km2 = np.pi * (buffer_1km**2) / 1_000_000  # circle area in km²
stop_density = (stop_counts / area_km2).astype(float)

# 5. Map back to nodes_df
nodes_df['bus_stop_density_per_km2'] = nodes_df.index.map(stop_density).fillna(0)

#---------------------------------------------------------------------------
# CREATE TARGET VARIABLE
# Use Overall_Demand as primary proxy, but scale it to be more interpretable
# We'll create a "predicted_ridership" that's based on demand metrics
# This is a proxy since we don't have actual ridership data

# Combine multiple demand signals
# Chosen weights are arbitrary
nodes_df['ridership_proxy'] = (
    nodes_df['nearest_Overall_Accessibility'] * 10 + 
    nodes_df['nearest_Commute_PT_Demand'] * 5 + 
    nodes_df['nearest_Population_Density_Demand'] * 5 +
    nodes_df['nearest_Bus_Stop_Proximity'] * 3 +
    nodes_df['population_within_1.0km'] * 2 +
    nodes_df['schools_within_0.5km'] +
    nodes_df['healthcare_within_0.5km'] +
    nodes_df['route_connections']
)

# Add inverse distance penalties 
# Closer to high-demand areas = higher ridership
nodes_df['ridership_proxy'] = nodes_df['ridership_proxy'] / (1 + nodes_df['nearest_demand_dist'])

# Normalize to a scale of 0-100 for readability
nodes_df['ridership_proxy'] = (
    (nodes_df['ridership_proxy'] - nodes_df['ridership_proxy'].min()) /
    (nodes_df['ridership_proxy'].max() - nodes_df['ridership_proxy'].min()) * 100
)

#---------------------------------------------------------------------------
# SAVE PROCESSED DATA FOR ML
output_file = "training_data/bus_stops_with_features.csv"
nodes_df.to_csv(output_file, index=False)
