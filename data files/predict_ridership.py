"""
Predict ridership for new bus stop locations
Usage: python predict_ridership.py <latitude> <longitude>
"""

import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from sklearn.neighbors import BallTree
import joblib
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def calculate_features_for_location(lat, lon, data_dir):
    """Calculate all features for a new bus stop location"""
    
    # Create point
    point = Point(lon, lat)
    gdf_point = gpd.GeoDataFrame([1], geometry=[point], crs='EPSG:4326')
    gdf_point_proj = gdf_point.to_crs('EPSG:32610')
    point_proj = gdf_point_proj.geometry.iloc[0]
    point_coords_proj = np.array([[point_proj.x, point_proj.y]])
    
    features = {}
    
    # 1. Load demand data and find nearest DA
    demand_df = pd.read_csv(data_dir / "demand" / "streamlined_transit_data.csv")
    gdf_demand = gpd.GeoDataFrame(
        demand_df,
        geometry=[Point(lon, lat) for lon, lat in zip(demand_df['longitude'], demand_df['latitude'])],
        crs='EPSG:4326'
    )
    gdf_demand_proj = gdf_demand.to_crs('EPSG:32610')
    demand_centroids = gdf_demand_proj.geometry.centroid
    demand_coords_proj = np.array([[p.x, p.y] for p in demand_centroids])
    
    tree = BallTree(demand_coords_proj, metric='euclidean')
    distances, indices = tree.query(point_coords_proj, k=1)
    features['distance_to_nearest_da_km'] = distances[0][0] / 1000
    
    # Map demand features
    nearest_da = demand_df.iloc[indices[0][0]]
    for col in ['Population_Density_Demand', 'Commute_PT_Demand', 'Bus_Stop_Proximity',
                'Route_Coverage_Percent', 'Bus_Stop_Density', 'Income_Demand',
                'Overall_Demand', 'Employment_Demand', 'Commute_Duration_Demand',
                'Overall_Accessibility']:
        features[f'nearest_{col}'] = nearest_da[col]
    
    # 2. Distance to schools
    schools_df = pd.read_csv(data_dir / "POI" / "schools.csv")
    victoria_bounds = {'lon_min': -123.45, 'lon_max': -123.30, 'lat_min': 48.40, 'lat_max': 48.50}
    victoria_schools = schools_df[
        (schools_df['longitude'] >= victoria_bounds['lon_min']) &
        (schools_df['longitude'] <= victoria_bounds['lon_max']) &
        (schools_df['latitude'] >= victoria_bounds['lat_min']) &
        (schools_df['latitude'] <= victoria_bounds['lat_max'])
    ].copy()
    
    if len(victoria_schools) > 0:
        gdf_schools = gpd.GeoDataFrame(
            victoria_schools,
            geometry=[Point(lon, lat) for lon, lat in zip(victoria_schools['longitude'], victoria_schools['latitude'])],
            crs='EPSG:4326'
        )
        gdf_schools_proj = gdf_schools.to_crs('EPSG:32610')
        school_coords_proj = np.array([[p.x, p.y] for p in gdf_schools_proj.geometry])
        tree_schools = BallTree(school_coords_proj, metric='euclidean')
        distances_schools, _ = tree_schools.query(point_coords_proj, k=1)
        features['distance_to_nearest_school_km'] = distances_schools[0][0] / 1000
        
        # Count schools in buffers
        for buffer_km in [0.5, 1.0, 2.0]:
            buffer_m = buffer_km * 1000
            buffer_zone = point_proj.buffer(buffer_m)
            count = gdf_schools_proj.geometry.within(buffer_zone).sum()
            features[f'schools_within_{buffer_km}km'] = count
    else:
        features['distance_to_nearest_school_km'] = np.nan
        for buffer_km in [0.5, 1.0, 2.0]:
            features[f'schools_within_{buffer_km}km'] = 0
    
    # 3. Distance to healthcare
    healthcare_df = pd.read_csv(data_dir / "POI" / "healthcare_facilities.csv")
    gdf_healthcare = gpd.GeoDataFrame(
        healthcare_df,
        geometry=[Point(lon, lat) for lon, lat in zip(healthcare_df['longitude'], healthcare_df['latitude'])],
        crs='EPSG:4326'
    )
    gdf_healthcare_proj = gdf_healthcare.to_crs('EPSG:32610')
    healthcare_coords_proj = np.array([[p.x, p.y] for p in gdf_healthcare_proj.geometry])
    tree_healthcare = BallTree(healthcare_coords_proj, metric='euclidean')
    distances_healthcare, _ = tree_healthcare.query(point_coords_proj, k=1)
    features['distance_to_nearest_healthcare_km'] = distances_healthcare[0][0] / 1000
    
    # Count healthcare in buffers
    for buffer_km in [0.5, 1.0, 2.0]:
        buffer_m = buffer_km * 1000
        buffer_zone = point_proj.buffer(buffer_m)
        count = gdf_healthcare_proj.geometry.within(buffer_zone).sum()
        features[f'healthcare_within_{buffer_km}km'] = count
    
    # 4. Population features
    pop_df = pd.read_csv(data_dir / "population" / "victoria_census_da.csv")
    gdf_pop = gpd.GeoDataFrame(
        pop_df,
        geometry=[Point(lon, lat) for lon, lat in zip(pop_df['centroid_long'], pop_df['centroid_lat'])],
        crs='EPSG:4326'
    )
    gdf_pop_proj = gdf_pop.to_crs('EPSG:32610')
    
    pop_threshold = pop_df['pop'].quantile(0.75)
    high_pop = pop_df[pop_df['pop'] >= pop_threshold]
    gdf_high_pop_proj = gdf_pop_proj[pop_df['pop'] >= pop_threshold]
    
    if len(gdf_high_pop_proj) > 0:
        high_pop_coords_proj = np.array([[p.x, p.y] for p in gdf_high_pop_proj.geometry])
        tree_pop = BallTree(high_pop_coords_proj, metric='euclidean')
        distances_pop, _ = tree_pop.query(point_coords_proj, k=1)
        features['distance_to_high_pop_area_km'] = distances_pop[0][0] / 1000
        
        # Population within buffers
        for buffer_km in [0.5, 1.0, 2.0]:
            buffer_m = buffer_km * 1000
            buffer_zone = point_proj.buffer(buffer_m)
            within = gdf_pop_proj.geometry.within(buffer_zone)
            total_pop = pop_df.loc[within, 'pop'].sum() if within.any() else 0
            features[f'population_within_{buffer_km}km'] = total_pop
    else:
        features['distance_to_high_pop_area_km'] = np.nan
        for buffer_km in [0.5, 1.0, 2.0]:
            features[f'population_within_{buffer_km}km'] = 0
    
    # 5. Network features (simplified - would need actual network data)
    # For new locations, we'll use default/estimated values
    features['route_connections'] = 1  # Default for new stop
    features['bus_stop_density_per_km2'] = 50.0  # Average density
    
    return features

def predict_ridership(lat, lon):
    """Predict ridership for a given location"""
    
    data_dir = Path("data files")
    model_dir = Path("models")
    
    # Load model and feature list
    model = joblib.load(model_dir / "ridership_predictor.pkl")
    with open(model_dir / "feature_list.txt", 'r') as f:
        feature_list = [line.strip() for line in f.readlines()]
    
    # Calculate features
    print(f"Calculating features for location ({lat}, {lon})...")
    features = calculate_features_for_location(lat, lon, data_dir)
    
    # Create feature vector in correct order
    feature_vector = []
    for feat in feature_list:
        if feat in features:
            feature_vector.append(features[feat])
        else:
            # Fill missing with median from training data
            feature_vector.append(0.0)  # Simplified - should load training medians
    
    # Handle NaN values
    feature_vector = np.array(feature_vector)
    feature_vector = np.nan_to_num(feature_vector, nan=0.0)
    
    # Predict
    prediction = model.predict([feature_vector])[0]
    
    return prediction, features

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python predict_ridership.py <latitude> <longitude>")
        print("\nExample: python predict_ridership.py 48.4284 -123.3656")
        sys.exit(1)
    
    try:
        lat = float(sys.argv[1])
        lon = float(sys.argv[2])
    except ValueError:
        print("Error: Latitude and longitude must be numbers")
        sys.exit(1)
    
    print("=" * 80)
    print("BUS RIDERSHIP PREDICTION")
    print("=" * 80)
    print(f"\nLocation: ({lat}, {lon})")
    
    prediction, features = predict_ridership(lat, lon)
    
    print(f"\nPredicted Ridership Score: {prediction:.2f} (0-100 scale)")
    print(f"\nKey Features:")
    print(f"  Distance to nearest school: {features.get('distance_to_nearest_school_km', 'N/A'):.3f} km")
    print(f"  Distance to nearest healthcare: {features.get('distance_to_nearest_healthcare_km', 'N/A'):.3f} km")
    print(f"  Schools within 0.5km: {features.get('schools_within_0.5km', 0)}")
    print(f"  Healthcare within 0.5km: {features.get('healthcare_within_0.5km', 0)}")
    print(f"  Overall Demand (nearest DA): {features.get('nearest_Overall_Demand', 'N/A')}")
    print(f"  Population within 1km: {features.get('population_within_1.0km', 0):.0f}")
    
    print("\n" + "=" * 80)

