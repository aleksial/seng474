"""
Bus Ridership Prediction
Takes a latitude and longitude as input, computes necessary features,
and predicts ridership using a pre-trained Random Forest Regressor model.
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point
from sklearn.ensemble import RandomForestRegressor
import joblib
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def calculate_features_for_coordinates(lat, lon, n_neighbors=5, max_distance_km=1.0):
    """
    Calculate features for a given coordinate point by analyzing similar nearby bus stops.
    Uses weighted average based on distance to nearby stops.
    
    Parameters:
    -----------
    lat : float
        Latitude of the point
    lon : float
        Longitude of the point
    n_neighbors : int
        Number of nearby bus stops to consider (default: 5)
    max_distance_km : float
        Maximum distance in km to consider stops (default: 1.0 km)
    
    Returns:
    --------
    dict : Dictionary containing calculated features
    """
    # Create point geometry
    point = Point(lon, lat)  # Note: shapely uses (lon, lat) order
    
    # Create GeoDataFrame with the point
    point_gdf = gpd.GeoDataFrame(
        {'geometry': [point]},
        crs='EPSG:4326'
    )
    
    # Convert to projected CRS for accurate distance calculations
    # Using BC Albers (EPSG:3005) - the standard for British Columbia
    point_gdf = point_gdf.to_crs('EPSG:3005')
    point_geom = point_gdf.geometry.iloc[0]
    
    # print("IN calculate_features_for_coordinates")

    # Load bus stops data with features
    bus_stops_df = pd.read_csv(Path("training_data") / "bus_stops_with_features.csv")
    
    # Convert to GeoDataFrame
    if 'latitude' in bus_stops_df.columns and 'longitude' in bus_stops_df.columns:
        bus_stops_gdf = gpd.GeoDataFrame(
            bus_stops_df,
            geometry=gpd.points_from_xy(bus_stops_df.longitude, bus_stops_df.latitude),
            crs='EPSG:4326'
        )
    else:
        bus_stops_gdf = gpd.GeoDataFrame(bus_stops_df, crs='EPSG:4326')
    
    # Convert to projected CRS
    bus_stops_gdf = bus_stops_gdf.to_crs('EPSG:3005')
    
    # Priority features to calculate
    priority_features = [
        'nearest_Overall_Accessibility',
        'nearest_Commute_PT_Demand',
        'nearest_Population_Density_Demand',
        'nearest_Bus_Stop_Proximity',
        'population_within_1.0km',
        'schools_within_0.5km',
        'healthcare_within_0.5km',
        'route_connections'
    ]
    
    # Calculate Cartesian distances to all bus stops
    bus_stops_gdf['distance'] = bus_stops_gdf.geometry.distance(point_geom)
    
    # Filter stops within max distance (0.5 longitudinal km ~ 500 meters)
    max_distance_meters = max_distance_km * 1000
    nearby_stops = bus_stops_gdf[bus_stops_gdf['distance'] <= max_distance_meters].copy()
    
    # If no stops within max distance, use the n_neighbors closest stops
    if len(nearby_stops) == 0:
        nearby_stops = bus_stops_gdf.nsmallest(n_neighbors, 'distance').copy()
    else:
        # Use up to n_neighbors from the nearby stops
        nearby_stops = nearby_stops.nsmallest(min(n_neighbors, len(nearby_stops)), 'distance')
    
    # Calculate inverse distance weights (closer stops have more influence) 
    # Formula: weight = 1 / distance
    # Add small epsilon to avoid division by zero
    epsilon = 1e-10
    nearby_stops['weight'] = 1 / (nearby_stops['distance'] + epsilon)
    nearby_stops['weight'] = nearby_stops['weight'] / nearby_stops['weight'].sum()   # Normalize weights to sum to 1
    
    # Calculate weighted average for each feature
    features = {}
    
    for feature in priority_features:
        if feature == 'nearest_Bus_Stop_Proximity':
            # This should be the actual distance to the nearest stop
            features[feature] = nearby_stops['distance'].min()
        elif feature in nearby_stops.columns:
            # Calculate weighted average of the feature from nearby stops (n neighbors)
            weighted_value = (nearby_stops[feature] * nearby_stops['weight']).sum()
            features[feature] = weighted_value
        else:
            # Feature doesn't exist in the data
            print(f"Warning: Feature '{feature}' not found in bus_stops_with_features.csv")
            features[feature] = 0
    
    return features


def calculate_features_simple(lat, lon):
    """
    Simpler version: Calculate features by using nearest stop values.
    This is a fallback method if weighted averaging is not desired.
    
    Parameters:
    -----------
    lat : float
        Latitude of the point
    lon : float
        Longitude of the point
    
    Returns:
    --------
    dict : Dictionary containing calculated features
    """
    # Create point geometry
    point = Point(lon, lat) # Note: shapely uses (lon, lat) order
    
    # Create GeoDataFrame with the point
    point_gdf = gpd.GeoDataFrame(
        {'geometry': [point]},
        crs='EPSG:4326'
    )
    
    # Convert to projected CRS (Coordinate Reference System)
    point_gdf = point_gdf.to_crs('EPSG:3005')
    point_geom = point_gdf.geometry.iloc[0]
    
    # Load bus stops data with features
    bus_stops_df = pd.read_csv(Path("training_data") / "bus_stops_with_features.csv")
    
    if 'latitude' in bus_stops_df.columns and 'longitude' in bus_stops_df.columns:
        bus_stops_gdf = gpd.GeoDataFrame(
            bus_stops_df,
            geometry=gpd.points_from_xy(bus_stops_df.longitude, bus_stops_df.latitude),
            crs='EPSG:4326'
        )
    else:
        bus_stops_gdf = gpd.GeoDataFrame(bus_stops_df, crs='EPSG:4326')
    
    bus_stops_gdf = bus_stops_gdf.to_crs('EPSG:3005')
    
    # Priority features
    priority_features = [
        'nearest_Overall_Accessibility',
        'nearest_Commute_PT_Demand',
        'nearest_Population_Density_Demand',
        'nearest_Bus_Stop_Proximity',
        'population_within_1.0km',
        'schools_within_0.5km',
        'healthcare_within_0.5km',
        'route_connections'
    ]
    
    # Find nearest stop
    distances = bus_stops_gdf.geometry.distance(point_geom)
    nearest_idx = distances.idxmin()
    nearest_stop = bus_stops_gdf.loc[nearest_idx]
    
    # Extract features
    features = {}
    for feature in priority_features:
        if feature == 'nearest_Bus_Stop_Proximity':
            features[feature] = distances.min()
        elif feature in bus_stops_gdf.columns:
            features[feature] = nearest_stop[feature]
        else:
            features[feature] = 0
    
    return features


def ridership_prediction(lat, lon, use_weighted=True, n_neighbors=5, max_distance_km=0.5):
    """
    Predict ridership for a given coordinate.
    
    Parameters:
    -----------
    lat : float
        Latitude of the point
    lon : float
        Longitude of the point
    model_path : str
        Path to the saved Random Forest Regressor model
    use_weighted : bool
        If True, uses weighted average from nearby stops. If False, uses nearest stop only.
    n_neighbors : int
        Number of nearby stops to consider for weighted average
    max_distance_km : float
        Maximum distance to consider stops for weighted average - use 500 meters by default due to urban density.
    
    Returns:
    --------
    float : Predicted ridership value
    """
    # Load the trained Random Forest Regressor model
    model_package = joblib.load(Path("models") / "random_forest_model.pkl")

    # Extract the model from the package
    if isinstance(model_package, dict):
        model = model_package['model']
        # Optionally verify feature names match
        if 'feature_names' in model_package:
            saved_features = model_package['feature_names']
            print(f"Model trained with features: {saved_features}")
    else:
        # If it's not a dict, assume it's the model directly
        model = model_package
    
    # Calculate features for the coordinates
    # if use_weighted:
    features = calculate_features_for_coordinates(lat, lon, n_neighbors, max_distance_km)
    # else:
    #    features = calculate_features_simple(lat, lon)
    
    # Convert features to DataFrame for prediction
    feature_df = pd.DataFrame([features])
    
    # Ensure features are in the correct order
    priority_features = [
        'nearest_Overall_Accessibility',
        'nearest_Commute_PT_Demand',
        'nearest_Population_Density_Demand',
        'nearest_Bus_Stop_Proximity',
        'population_within_1.0km',
        'schools_within_0.5km',
        'healthcare_within_0.5km',
        'route_connections'
    ]
    
    # Reorder columns to match training data
    feature_df = feature_df[priority_features]
    
    # Make prediction
    prediction = model.predict(feature_df)[0]
    
    return prediction


# Example usage
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
    
    # Make prediction
    print("\nMaking prediction...")
    try:
        prediction = ridership_prediction(lat, lon)
        print(f"Predicted ridership: {prediction:.2f}")
    except FileNotFoundError:
        print("Model file not found. Please ensure 'random_forest_model.pkl' exists.")