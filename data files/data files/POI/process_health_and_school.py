# =============================================================================
# HEALTHCARE FACILITIES PROCESSING
# =============================================================================

def process_healthcare_facilities(geojson_path, output_csv="healthcare_facilities.csv"):
    """
    Process healthcare facilities GeoJSON to CSV, preserving all data
    """
    import geopandas as gpd
    import pandas as pd
    
    # Load healthcare data
    healthcare_gdf = gpd.read_file(geojson_path)
    print(f"Loaded {len(healthcare_gdf)} healthcare facilities")
    
    # Convert to consistent CRS and extract coordinates
    if healthcare_gdf.crs.is_geographic:
        healthcare_gdf = healthcare_gdf.to_crs('EPSG:4326')  # Standard lat/lon
    
    # Extract latitude and longitude from point geometry
    healthcare_gdf['longitude'] = healthcare_gdf.geometry.x
    healthcare_gdf['latitude'] = healthcare_gdf.geometry.y
    
    # Preserve ALL original columns - they might be useful for ML
    print("Healthcare columns:", list(healthcare_gdf.columns))
    
    # Ensure 'healthcare' column exists (your key variable)
    if 'healthcare' not in healthcare_gdf.columns:
        print("Warning: 'healthcare' column not found. Available columns:", list(healthcare_gdf.columns))
        # You might need to identify which column contains the healthcare type
    
    # Save to CSV, dropping the geometry column since we have lat/long
    healthcare_df = healthcare_gdf.drop('geometry', axis=1)
    healthcare_df.to_csv(output_csv, index=False)
    
    print(f"✅ Healthcare facilities saved to {output_csv}")
    print(f"📊 Total facilities: {len(healthcare_df)}")
    
    # Show sample of healthcare types
    if 'healthcare' in healthcare_df.columns:
        print("Healthcare types distribution:")
        print(healthcare_df['healthcare'].value_counts().head(10))
    
    return healthcare_df

# =============================================================================
# SCHOOLS PROCESSING FROM REST API
# =============================================================================

def process_schools_from_rest(rest_url, output_csv="schools.csv", max_records=5000):
    """
    Process schools data from ArcGIS REST API, preserving all data
    """
    import requests
    import pandas as pd
    
    print("Fetching schools data from REST API...")
    
    # Query parameters - adjust based on API documentation if needed
    params = {
        'where': '1=1',  # Get all records
        'outFields': '*',  # Get all fields
        'returnGeometry': 'true',
        'f': 'json',
        'outSR': '4326'  # Output in lat/lon
    }
    
    try:
        response = requests.get(rest_url + '/query', params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if 'features' not in data:
            print("Error: No features found in response")
            return None
        
        features = data['features']
        print(f"Retrieved {len(features)} school records")
        
        # Extract attributes and geometry
        schools_data = []
        for feature in features:
            attributes = feature.get('attributes', {})
            geometry = feature.get('geometry', {})
            
            # Extract coordinates - adjust based on actual geometry structure
            if 'x' in geometry and 'y' in geometry:
                attributes['longitude'] = geometry['x']
                attributes['latitude'] = geometry['y']
            elif 'rings' in geometry:
                # Handle polygon centroids if needed
                pass
            
            schools_data.append(attributes)
        
        # Create DataFrame
        schools_df = pd.DataFrame(schools_data)
        
        # Ensure OCCUPANT_NAME exists
        if 'OCCUPANT_NAME' not in schools_df.columns:
            print("Warning: 'OCCUPANT_NAME' column not found. Available columns:", list(schools_df.columns))
        
        print("Schools columns:", list(schools_df.columns))
        
        # Save to CSV
        schools_df.to_csv(output_csv, index=False)
        print(f"✅ Schools data saved to {output_csv}")
        print(f"📊 Total schools: {len(schools_df)}")
        
        # Show sample of school names
        if 'OCCUPANT_NAME' in schools_df.columns:
            print("Sample school names:")
            print(schools_df['OCCUPANT_NAME'].head(10))
        
        return schools_df
        
    except Exception as e:
        print(f"Error fetching schools data: {e}")
        return None

# =============================================================================
# COMPLETE FACILITIES PROCESSING
# =============================================================================

def process_all_facilities():
    """
    Process both healthcare and schools facilities
    """
    print("🏥 Processing Healthcare Facilities...")
    healthcare_df = process_healthcare_facilities("Healthcare.geojson")
    
    print("\n🏫 Processing Schools...")
    schools_df = process_schools_from_rest(
        "https://delivery.maps.gov.bc.ca/arcgis/rest/services/whse/bcgw_pub_whse_imagery_and_base_maps/MapServer/56"
    )
    
    print("\n📋 Summary:")
    if healthcare_df is not None:
        print(f"  Healthcare facilities: {len(healthcare_df)}")
        print(f"  Healthcare file: healthcare_facilities.csv")
    
    if schools_df is not None:
        print(f"  Schools: {len(schools_df)}")
        print(f"  Schools file: schools.csv")
    
    return healthcare_df, schools_df

# Run the complete processing
healthcare_df, schools_df = process_all_facilities()