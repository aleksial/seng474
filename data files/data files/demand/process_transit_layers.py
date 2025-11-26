# =============================================================================
# STREAMLINED SPATIAL JOIN - ONLY KEY COLUMNS WITH MEANINGFUL NAMES
# =============================================================================

import geopandas as gpd
import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
import os

# Define the key column for each layer (the rightmost data column)
LAYER_KEY_COLUMNS = {
    '1': 'PopDensDemandSuitability',
    '2': 'ComPTDemandSuitability', 
    '25': 'AddressBusStopProximity',
    '26': 'PercentRouteCoverage',
    '3': 'ComATDemandSuitability',
    '34': 'BusStopDensity',
    '4': 'IncomeDemandSuitability',
    '40': 'OverallAccessibilitySuitabili',
    '41': 'OverallDemandSuitability',
    '5': 'EmployDemandSuitability',
    '6': 'ComDurDemandSuitability'
}

# Define meaningful names for each layer
LAYER_MEANINGFUL_NAMES = {
    '1': 'Population_Density_Demand',
    '2': 'Commute_PT_Demand', 
    '25': 'Bus_Stop_Proximity',
    '26': 'Route_Coverage_Percent',
    '3': 'Commute_AT_Demand',
    '34': 'Bus_Stop_Density',
    '4': 'Income_Demand',
    '40': 'Overall_Accessibility',
    '41': 'Overall_Demand',
    '5': 'Employment_Demand',
    '6': 'Commute_Duration_Demand'
}

def create_streamlined_spatial_join(base_layer_path, other_layers_folder, output_csv="streamlined_transit_data.csv", max_distance_km=10):
    """
    Streamlined version: Only extract key columns with meaningful names
    """
    print("🎯 Starting streamlined spatial join...")
    
    # Load base layer and project to meters for accurate distances
    print(f"📁 Loading base layer: {base_layer_path}")
    base_gdf = gpd.read_file(base_layer_path)
    
    # Project to UTM (meters) for accurate distance calculations
    base_gdf = base_gdf.to_crs('EPSG:32610')
    print(f"   📊 {len(base_gdf)} DA polygons loaded")
    
    # Calculate centroids in projected CRS
    print("📍 Calculating DA centroids...")
    base_gdf['centroid'] = base_gdf.geometry.centroid
    base_gdf['centroid_x'] = base_gdf.centroid.x
    base_gdf['centroid_y'] = base_gdf.centroid.y
    
    # Get all other layer files
    other_layer_files = [f for f in os.listdir(other_layers_folder) 
                        if f.startswith('layer_') and f != os.path.basename(base_layer_path)]
    
    print(f"🔍 Processing {len(other_layer_files)} transit layers...")
    
    # Process each layer and extract only the key column
    for layer_file in other_layer_files:
        layer_path = os.path.join(other_layers_folder, layer_file)
        layer_id = layer_file.replace('layer_', '').replace('.geojson', '')
        
        # Skip if we don't have a mapping for this layer
        if layer_id not in LAYER_KEY_COLUMNS:
            print(f"⚠️  Skipping layer {layer_id} - no key column mapping")
            continue
            
        key_column = LAYER_KEY_COLUMNS[layer_id]
        meaningful_name = LAYER_MEANINGFUL_NAMES[layer_id]
        
        print(f"\n🔄 Processing {meaningful_name} (layer {layer_id})...")
        
        try:
            # Load the layer and project to same CRS
            layer_gdf = gpd.read_file(layer_path)
            layer_gdf = layer_gdf.to_crs('EPSG:32610')
            print(f"   📊 {len(layer_gdf)} features in layer")
            
            # Calculate centroids for this layer
            layer_gdf['centroid'] = layer_gdf.geometry.centroid
            layer_gdf['layer_centroid_x'] = layer_gdf.centroid.x
            layer_gdf['layer_centroid_y'] = layer_gdf.centroid.y
            
            # Verify the key column exists
            if key_column not in layer_gdf.columns:
                print(f"   ❌ Key column '{key_column}' not found in layer {layer_id}")
                print(f"   Available columns: {list(layer_gdf.columns)}")
                continue
            
            # Create BallTree for efficient nearest neighbor search
            layer_coords = layer_gdf[['layer_centroid_x', 'layer_centroid_y']].values
            tree = BallTree(layer_coords, metric='euclidean')
            
            # For each DA centroid, find closest feature in this layer
            base_coords = base_gdf[['centroid_x', 'centroid_y']].values
            distances, indices = tree.query(base_coords, k=1)
            
            # Convert distances from meters to kilometers
            distances_km = distances / 1000
            
            # Attach only the key column from closest features
            features_added = 0
            for i in range(len(base_gdf)):
                closest_idx = indices[i][0]
                distance_km = distances_km[i][0]
                
                if closest_idx < len(layer_gdf) and distance_km <= max_distance_km:
                    closest_feature = layer_gdf.iloc[closest_idx]
                    
                    # Add only the key column with meaningful name
                    base_gdf.loc[base_gdf.index[i], meaningful_name] = closest_feature[key_column]
                    base_gdf.loc[base_gdf.index[i], f"{meaningful_name}_distance_km"] = distance_km
                    features_added += 1
            
            print(f"   ✅ Added {meaningful_name} to {features_added} DAs")
            
        except Exception as e:
            print(f"   ❌ Error processing layer {layer_id}: {e}")
            continue
    
    # Convert back to geographic CRS for final coordinates
    base_gdf = base_gdf.to_crs('EPSG:4326')
    base_gdf['longitude'] = base_gdf.centroid.x
    base_gdf['latitude'] = base_gdf.centroid.y
    
    # Convert to regular DataFrame and save
    print(f"\n💾 Preparing final dataset...")
    
    # Select only the columns we want in the final output
    final_columns = ['DAUID', 'longitude', 'latitude']  # Base columns
    
    # Add all our meaningful data columns
    for meaningful_name in LAYER_MEANINGFUL_NAMES.values():
        if meaningful_name in base_gdf.columns:
            final_columns.append(meaningful_name)
        if f"{meaningful_name}_distance_km" in base_gdf.columns:
            final_columns.append(f"{meaningful_name}_distance_km")
    
    # Add any original columns from base layer that might be useful
    for col in ['Shape__Area', 'Shape__Length']:
        if col in base_gdf.columns:
            final_columns.append(col)
    
    final_df = pd.DataFrame(base_gdf[final_columns])
    
    # Calculate data coverage statistics
    print("\n📊 FINAL DATA COVERAGE:")
    print("=" * 50)
    
    data_columns = [col for col in final_df.columns 
                   if col in LAYER_MEANINGFUL_NAMES.values()]
    
    for data_col in data_columns:
        coverage = final_df[data_col].notna().sum()
        coverage_pct = (coverage / len(final_df)) * 100
        print(f"📍 {data_col}: {coverage} DAs ({coverage_pct:.1f}%)")
    
    # Save to CSV
    final_df.to_csv(output_csv, index=False)
    print(f"\n✅ SAVED: {output_csv}")
    print(f"   📊 {len(final_df)} DA polygons")
    print(f"   📋 {len(final_df.columns)} total columns")
    print(f"   🗂️  Data columns: {data_columns}")
    
    return final_df

# =============================================================================
# VERIFICATION FUNCTION
# =============================================================================

def verify_layer_columns(layers_folder):
    """
    Verify that each layer has the expected key column
    """
    print("🔍 Verifying layer columns...")
    
    for layer_id, expected_column in LAYER_KEY_COLUMNS.items():
        layer_file = f"layer_{layer_id}.geojson"
        layer_path = os.path.join(layers_folder, layer_file)
        
        if os.path.exists(layer_path):
            layer_gdf = gpd.read_file(layer_path)
            
            if expected_column in layer_gdf.columns:
                print(f"✅ Layer {layer_id}: '{expected_column}' found")
                # Show sample values
                sample_value = layer_gdf[expected_column].iloc[0] if len(layer_gdf) > 0 else "No data"
                print(f"   Sample value: {sample_value}")
            else:
                print(f"❌ Layer {layer_id}: '{expected_column}' NOT FOUND")
                print(f"   Available columns: {list(layer_gdf.columns)}")
        else:
            print(f"⚠️  Layer {layer_id}: File not found")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

print("🎯 STREAMLINED TRANSIT DATA PROCESSING")
print("=" * 60)

# First verify all layers have the expected columns
verify_layer_columns("qgis_layers")

print("\n🚀 STARTING STREAMLINED SPATIAL JOIN")
print("=" * 60)

# Run the streamlined spatial join
final_data = create_streamlined_spatial_join(
    base_layer_path="qgis_layers/layer_9.geojson",
    other_layers_folder="qgis_layers",
    output_csv="streamlined_transit_data.csv",
    max_distance_km=10
)

print(f"\n🎉 PROCESSING COMPLETE!")

# Show a comprehensive sample of the final data
if final_data is not None and len(final_data) > 0:
    print(f"\n🔍 COMPREHENSIVE SAMPLE OF FINAL DATA:")
    print("=" * 50)
    
    # Show the first few rows with all data columns
    sample_data = final_data.head(3)
    
    for idx, row in sample_data.iterrows():
        print(f"\n📍 DA {row['DAUID']}:")
        print(f"   Coordinates: ({row['longitude']:.6f}, {row['latitude']:.6f})")
        
        # Show all data values (excluding coordinate and distance columns)
        data_values = {}
        for col in final_data.columns:
            if col not in ['DAUID', 'longitude', 'latitude', 'Shape__Area', 'Shape__Length'] and pd.notna(row[col]):
                data_values[col] = row[col]
        
        for col, value in data_values.items():
            print(f"   {col}: {value}")
    
    print(f"\n📋 FINAL COLUMNS IN CSV:")
    for col in final_data.columns:
        print(f"   - {col}")

print(f"\n📁 Output file: streamlined_transit_data.csv")
print("   This file is now ready for ML modeling!")