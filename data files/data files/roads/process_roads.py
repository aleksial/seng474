import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point, Polygon
import warnings
warnings.filterwarnings('ignore')

def create_complete_network(roads_geojson_path, zones_geojson_path, output_prefix="bus_network"):
    """
    Complete processing: Convert roads to graph CSV and associate with zoning data
    """
    print("🚀 Starting complete network processing...")
    
    # Step 1: Process Roads into Graph
    print("\n📊 Step 1: Processing roads into network graph...")
    roads_edges, roads_nodes = process_roads_to_graph(roads_geojson_path, output_prefix)
    
    # Step 2: Process Zoning Data
    print("\n🏘️  Step 2: Processing zoning data...")
    zones_gdf = process_zoning_data(zones_geojson_path)
    
    # Step 3: Associate Zoning with Roads
    print("\n🔗 Step 3: Associating zoning with roads...")
    roads_with_zoning = associate_zoning_with_roads(roads_edges, roads_nodes, zones_gdf, output_prefix)
    
    print("\n✅ Processing complete!")
    return roads_with_zoning, roads_nodes, zones_gdf

def process_roads_to_graph(geojson_path, output_prefix):
    """Convert roads GeoJSON to network graph CSVs"""
    gdf = gpd.read_file(geojson_path)
    print(f"   Loaded {len(gdf)} road segments")
    
    # Convert to projected CRS for accurate lengths
    if gdf.crs and gdf.crs.is_geographic:
        gdf = gdf.to_crs('EPSG:3857')
    
    # Create unique node mapping
    nodes = {}
    node_counter = 0
    
    def get_or_create_node(coord):
        nonlocal node_counter
        key = (round(coord[0], 6), round(coord[1], 6))
        if key not in nodes:
            nodes[key] = f"N{node_counter}"
            node_counter += 1
        return nodes[key]
    
    # Build edges dataframe
    edges_data = []
    for idx, row in gdf.iterrows():
        geom = row.geometry
        
        if geom.geom_type == 'LineString':
            coords = list(geom.coords)
            if len(coords) >= 2:
                start_node = get_or_create_node(coords[0])
                end_node = get_or_create_node(coords[-1])
                
                # Extract attributes with proper typing
                edge_attrs = {
                    'edge_id': f"E{idx}",
                    'from_node': start_node,
                    'to_node': end_node,
                    'length': float(geom.length),
                    'geometry_wkt': geom.wkt
                }
                
                # Add all other attributes from original data
                for col in gdf.columns:
                    if col not in ['geometry'] and not pd.isna(row[col]):
                        value = row[col]
                        if pd.api.types.is_numeric_dtype(gdf[col]):
                            edge_attrs[col] = float(value) if not pd.isna(value) else 0.0
                        else:
                            edge_attrs[col] = value
                
                edges_data.append(edge_attrs)
    
    # Create edges DataFrame
    edges_df = pd.DataFrame(edges_data)
    
    # Ensure numeric types
    numeric_columns = ['length', 'volume', 'distance']
    for col in numeric_columns:
        if col in edges_df.columns:
            edges_df[col] = pd.to_numeric(edges_df[col], errors='coerce').fillna(0.0)
    
    # Create nodes DataFrame
    nodes_data = []
    for coord, node_id in nodes.items():
        node_attrs = {
            'node_id': node_id,
            'x': float(coord[0]),
            'y': float(coord[1])
        }
        nodes_data.append(node_attrs)
    
    nodes_df = pd.DataFrame(nodes_data)
    
    # Save initial graph files
    edges_df.to_csv(f"{output_prefix}_edges.csv", index=False)
    nodes_df.to_csv(f"{output_prefix}_nodes.csv", index=False)
    
    print(f"   ✅ Exported {len(edges_df)} edges and {len(nodes_df)} nodes")
    return edges_df, nodes_df

def process_zoning_data(geojson_path):
    """Process zoning data and calculate areas"""
    zones_gdf = gpd.read_file(geojson_path)
    print(f"   Loaded {len(zones_gdf)} zoning areas")
    
    # Convert to projected CRS for accurate areas
    if zones_gdf.crs.is_geographic:
        zones_gdf = zones_gdf.to_crs('EPSG:3857')
    
    # Calculate area if not present
    if 'shape_area' not in zones_gdf.columns:
        zones_gdf['shape_area'] = zones_gdf.geometry.area
    else:
        zones_gdf['shape_area'] = pd.to_numeric(zones_gdf['shape_area'], errors='coerce')
    
    # Ensure required columns exist
    if 'locality' not in zones_gdf.columns:
        zones_gdf['locality'] = 'Unknown'
    if 'title' not in zones_gdf.columns:
        zones_gdf['title'] = 'No Title'
    
    print(f"   Zoning localities: {zones_gdf['locality'].unique()}")
    return zones_gdf

def associate_zoning_with_roads(roads_edges, roads_nodes, zones_gdf, output_prefix):
    """Associate zoning data with roads, picking larger zones when multiple are nearby"""
    print("   Finding zoning for each road segment...")
    
    # Create dictionary for node coordinates
    node_coords = {}
    for idx, row in roads_nodes.iterrows():
        node_coords[row['node_id']] = (row['x'], row['y'])
    
    # Create road segment centroids
    road_centroids = []
    for idx, row in roads_edges.iterrows():
        from_coords = node_coords.get(row['from_node'])
        to_coords = node_coords.get(row['to_node'])
        
        if from_coords and to_coords:
            mid_x = (from_coords[0] + to_coords[0]) / 2
            mid_y = (from_coords[1] + to_coords[1]) / 2
            centroid = Point(mid_x, mid_y)
            
            road_centroids.append({
                'edge_id': row['edge_id'],
                'geometry': centroid
            })
    
    # Create GeoDataFrame for road centroids
    road_centroids_gdf = gpd.GeoDataFrame(road_centroids, crs=zones_gdf.crs)
    
    # Spatial join to find ALL zoning areas that contain each road centroid
    print("   Performing spatial analysis...")
    all_matches = gpd.sjoin(road_centroids_gdf, 
                           zones_gdf[['locality', 'title', 'shape_area', 'geometry']], 
                           how='left', 
                           predicate='within')
    
    # For roads that have multiple zoning matches, pick the one with largest area
    print("   Resolving multiple zoning matches...")
    zoning_attributes = []
    
    for edge_id in all_matches['edge_id'].unique():
        edge_matches = all_matches[all_matches['edge_id'] == edge_id]
        
        if len(edge_matches) == 0:
            # No zoning found
            zoning_attributes.append({
                'edge_id': edge_id,
                'locality': 'Unknown',
                'title': 'No Zone',
                'zone_area': 0.0
            })
        elif len(edge_matches) == 1:
            # Single zoning match
            match = edge_matches.iloc[0]
            zoning_attributes.append({
                'edge_id': edge_id,
                'locality': match['locality'] if pd.notna(match['locality']) else 'Unknown',
                'title': match['title'] if pd.notna(match['title']) else 'No Title',
                'zone_area': float(match['shape_area']) if pd.notna(match['shape_area']) else 0.0
            })
        else:
            # Multiple matches - pick the one with largest area
            largest_zone = edge_matches.loc[edge_matches['shape_area'].idxmax()]
            zoning_attributes.append({
                'edge_id': edge_id,
                'locality': largest_zone['locality'] if pd.notna(largest_zone['locality']) else 'Unknown',
                'title': largest_zone['title'] if pd.notna(largest_zone['title']) else 'No Title',
                'zone_area': float(largest_zone['shape_area']) if pd.notna(largest_zone['shape_area']) else 0.0
            })
    
    # Convert to DataFrame and merge with roads
    zoning_df = pd.DataFrame(zoning_attributes)
    roads_with_zoning = roads_edges.merge(zoning_df, on='edge_id', how='left')
    
    # Fill any remaining NaN values
    roads_with_zoning['locality'] = roads_with_zoning['locality'].fillna('Unknown')
    roads_with_zoning['title'] = roads_with_zoning['title'].fillna('No Zone')
    roads_with_zoning['zone_area'] = roads_with_zoning['zone_area'].fillna(0.0)
    
    # Save the final enhanced network
    final_output_path = f"{output_prefix}_with_zoning.csv"
    roads_with_zoning.to_csv(final_output_path, index=False)
    
    # Print statistics
    zone_stats = roads_with_zoning['locality'].value_counts()
    print(f"\n📈 Final road segment distribution:")
    for zone, count in zone_stats.items():
        print(f"   {zone}: {count} segments")
    
    multiple_zones = len(all_matches['edge_id'].value_counts()[all_matches['edge_id'].value_counts() > 1])
    print(f"   {multiple_zones} road segments had multiple zoning matches (resolved by area)")
    
    print(f"   ✅ Saved enhanced network to: {final_output_path}")
    
    return roads_with_zoning

# Main execution
if __name__ == "__main__":
    # Just run this one function with your file names
    roads_with_zoning, roads_nodes, zones_gdf = create_complete_network(
        roads_geojson_path="roads_ungraph.geojson",
        zones_geojson_path="zoning.geojson",
        output_prefix="bus_network"
    )
    
    # Print final summary
    print("\n🎉 PROCESSING COMPLETE!")
    print("=" * 50)
    print(f"Final output files:")
    print(f"  - bus_network_edges.csv (original graph)")
    print(f"  - bus_network_nodes.csv (original graph)") 
    print(f"  - bus_network_with_zoning.csv (enhanced with zoning)")
    print(f"\nEnhanced network columns: {list(roads_with_zoning.columns)}")
    
    print(f"\nSample of final data:")
    sample = roads_with_zoning.head(3)[['edge_id', 'from_node', 'to_node', 'length', 'locality', 'title', 'zone_area']]
    print(sample)
    
    '''
# =============================================================================
# ML DATA PREPARATION FROM PROCESSED ROAD NETWORK
# =============================================================================

# Load the enhanced road network with zoning data
roads_df = pd.read_csv('bus_network_with_zoning.csv')

print("Available features for ML modeling:")
for col in roads_df.columns:
    print(f"  - {col}")

# =============================================================================
# FEATURE EXTRACTION FOR BUS ROUTE OPTIMIZATION
# =============================================================================

# Numeric features (continuous values)
numeric_features = ['length', 'volume', 'zone_area']  # Add other numeric columns from your data
# Note: 'length' = road segment length, 'volume' = traffic volume, 'zone_area' = size of associated zone

# Categorical features (will need encoding)
categorical_features = ['locality', 'title']  # Zoning categories

# =============================================================================
# FEATURE ENGINEERING EXAMPLES
# =============================================================================

# Create density features
roads_df['traffic_density'] = roads_df['volume'] / roads_df['length']  # Vehicles per meter

# Create zone-based features (if you have multiple zone types)
zone_dummies = pd.get_dummies(roads_df['locality'], prefix='zone')
title_dummies = pd.get_dummies(roads_df['title'], prefix='zoning_type')

# Combine all features
ml_features = pd.concat([
    roads_df[numeric_features],  # Numeric features
    zone_dummies,                # One-hot encoded zones
    title_dummies               # One-hot encoded zoning types
], axis=1)

print(f"Final feature matrix shape: {ml_features.shape}")

# =============================================================================
# TARGET VARIABLE PREPARATION (EXAMPLE - ADJUST BASED ON YOUR ML TASK)
# =============================================================================

# Example 1: For route optimization - travel time estimation
# roads_df['travel_time'] = roads_df['length'] / roads_df['speed_limit']  # If you have speed data

# Example 2: For traffic prediction - use historical volume as target
# target = roads_df['volume']

# Example 3: For route popularity - binary classification
# roads_df['is_main_road'] = (roads_df['volume'] > roads_df['volume'].median()).astype(int)
# target = roads_df['is_main_road']

# =============================================================================
# DATA SPLITTING FOR ML (IF APPLICABLE)
# =============================================================================

from sklearn.model_selection import train_test_split

# If you have a target variable for supervised learning:
# X_train, X_test, y_train, y_test = train_test_split(
#     ml_features, 
#     target, 
#     test_size=0.2, 
#     random_state=42
# )

# For unsupervised learning (like route clustering):
# X = ml_features.values  # Direct feature matrix

# =============================================================================
# FEATURE SCALING (RECOMMENDED FOR MOST ML MODELS)
# =============================================================================

from sklearn.preprocessing import StandardScaler

# Scale numeric features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(ml_features)

print("Feature matrix ready for ML modeling!")
print(f"Scaled features shape: {scaled_features.shape}")

# =============================================================================
# GRAPH-BASED FEATURES (FOR NETWORK ANALYSIS)
# =============================================================================

import networkx as nx

# Convert back to graph for network analysis
G = nx.from_pandas_edgelist(
    roads_df, 
    'from_node', 
    'to_node', 
    edge_attr=['length', 'volume', 'locality']  # Include relevant attributes
)

# Calculate graph metrics for each node
degree_centrality = nx.degree_centrality(G)
betweenness_centrality = nx.betweenness_centrality(G, weight='length')

# These graph metrics can be added as additional features for each road segment
# by mapping them back through the node connections

# =============================================================================
# USAGE EXAMPLES FOR DIFFERENT ML TASKS:
# =============================================================================

# 1. Route Optimization:
#    Features: length, volume, locality, zone_area
#    Target: travel_time or route_efficiency

# 2. Traffic Prediction:  
#    Features: historical volumes, zoning types, road length
#    Target: future_traffic_volume

# 3. Zone Classification:
#    Features: road attributes, connectivity metrics
#    Target: zone_type or development_potential

# 4. Route Recommendation:
#    Features: all available + graph metrics
#    Target: route_score or user_preference
    
'''