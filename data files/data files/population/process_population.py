# =============================================================================
# CITY LIMITS PROCESSING
# =============================================================================

def get_city_limits(rest_url, output_geojson="city_limits.geojson"):
    """
    Fetch city boundaries from REST API
    """
    import requests
    import geopandas as gpd
    from shapely.geometry import shape
    
    print("Fetching city limits from REST API...")
    
    params = {
        'where': '1=1',
        'outFields': '*',
        'returnGeometry': 'true',
        'f': 'geojson',
        'outSR': '4326'
    }
    
    try:
        response = requests.get(rest_url + '/query', params=params, timeout=30)
        response.raise_for_status()
        
        # Save as GeoJSON
        with open(output_geojson, 'w') as f:
            f.write(response.text)
        
        # Load as GeoDataFrame to verify
        city_gdf = gpd.read_file(output_geojson)
        print(f"✅ City limits loaded: {len(city_gdf)} feature(s)")
        print(f"City area: {city_gdf.geometry.area.sum():.2f} sq units")
        
        return city_gdf
        
    except Exception as e:
        print(f"Error fetching city limits: {e}")
        return None

# Usage:
# city_limits = get_city_limits("https://maps.victoria.ca/server/rest/services/OpenData/OpenData_Land/MapServer/2")

# =============================================================================
# DA-LEVEL CENSUS DATA PROCESSING
# =============================================================================

def process_census_data(census_geojson_path, city_limits_gdf, output_csv="census_da.csv"):
    """
    Process DA-level census data and clip to city boundaries
    Preserve centroids and all census attributes
    """
    import geopandas as gpd
    import pandas as pd
    
    print("Loading census data...")
    census_gdf = gpd.read_file(census_geojson_path)
    print(f"Original census features: {len(census_gdf)}")
    
    # Ensure same CRS
    if census_gdf.crs != city_limits_gdf.crs:
        census_gdf = census_gdf.to_crs(city_limits_gdf.crs)
    
    # Clip census data to city boundaries
    print("Clipping to city boundaries...")
    census_clipped = gpd.clip(census_gdf, city_limits_gdf)
    print(f"Features after clipping: {len(census_clipped)}")
    
    # Calculate centroids
    census_clipped['centroid_long'] = census_clipped.geometry.centroid.x
    census_clipped['centroid_lat'] = census_clipped.geometry.centroid.y
    
    # Calculate area (useful for density calculations)
    census_clipped['area_sq_m'] = census_clipped.geometry.area
    
    # Preserve ALL census attributes - these are valuable for ML!
    print("Census columns available:")
    for col in census_clipped.columns:
        print(f"  - {col}")
    
    # Convert to DataFrame (drop geometry, keep centroids)
    census_df = census_clipped.drop('geometry', axis=1)
    
    # Ensure numeric types for census variables
    numeric_columns = []
    for col in census_df.columns:
        if col not in ['centroid_long', 'centroid_lat', 'area_sq_m']:
            if pd.api.types.is_numeric_dtype(census_df[col]):
                numeric_columns.append(col)
            else:
                # Try to convert to numeric
                try:
                    census_df[col] = pd.to_numeric(census_df[col], errors='ignore')
                    if pd.api.types.is_numeric_dtype(census_df[col]):
                        numeric_columns.append(col)
                except:
                    pass
    
    print(f"Identified {len(numeric_columns)} numeric census variables")
    
    # Save to CSV
    census_df.to_csv(output_csv, index=False)
    print(f"✅ Census data saved to {output_csv}")
    
    # Show key statistics
    if len(numeric_columns) > 0:
        print("Sample census statistics:")
        sample_col = numeric_columns[0]
        print(f"  {sample_col}: mean={census_df[sample_col].mean():.2f}")
    
    return census_df

# Usage:
# census_df = process_census_data("census_da.geojson", city_limits, "victoria_census_da.csv")

# =============================================================================
# COMPLETE CENSUS PROCESSING PIPELINE
# =============================================================================

def process_complete_census_data(census_geojson_path, 
                               city_limits_url,
                               output_csv="census_da_city.csv"):
    """
    Complete pipeline: Get city limits + process census data
    """
    print("🏙️  Starting complete census data processing...")
    
    # Step 1: Get city boundaries
    city_limits = get_city_limits(city_limits_url)
    if city_limits is None:
        print("Failed to get city limits")
        return None
    
    # Step 2: Process and clip census data
    census_data = process_census_data(census_geojson_path, city_limits, output_csv)
    
    print(f"\n✅ Complete! Processed {len(census_data)} DA regions within city limits")
    return census_data

# Run everything:
census_data = process_complete_census_data(
    census_geojson_path="populationDA.geojson",
    city_limits_url="https://maps.victoria.ca/server/rest/services/OpenData/OpenData_Land/MapServer/2",
    output_csv="victoria_census_da.csv"
)


'''
# =============================================================================
# ML INTEGRATION - CENSUS + ROADS + FACILITIES
# =============================================================================

def create_complete_urban_features(roads_df, healthcare_df, schools_df, census_df):
    """
    Combine all datasets for complete urban understanding
    """
    # Road-level features (from existing processing)
    urban_features = roads_df[['edge_id', 'from_node', 'to_node', 'length', 'volume', 'locality']].copy()
    
    # Add centroid coordinates for spatial joining
    urban_features['road_centroid_long'] = (urban_features['from_node_coord_long'] + urban_features['to_node_coord_long']) / 2
    urban_features['road_centroid_lat'] = (urban_features['from_node_coord_lat'] + urban_features['to_node_coord_lat']) / 2
    
    # Spatial joins to attach census data to roads
    # This is conceptual - you'd implement proper spatial joining
    urban_features = spatial_join_census_to_roads(urban_features, census_df)
    
    # Add facility densities (from previous processing)
    urban_features = add_facility_densities(urban_features, healthcare_df, schools_df)
    
    return urban_features

def spatial_join_census_to_roads(roads_df, census_df):
    """
    Attach census demographics to road segments based on spatial proximity
    """
    # This would use spatial libraries to find which DA each road segment falls into
    # For now, conceptual implementation:
    
    # roads_df['da_population'] = ...  # Population of containing DA
    # roads_df['da_income'] = ...      # Income level of containing DA  
    # roads_df['da_density'] = ...     # Population density of containing DA
    
    return roads_df

# =============================================================================
# KEY CENSUS VARIABLES FOR BUS OPTIMIZATION
# =============================================================================

"""
Your ML model will now have access to:

POPULATION DEMOGRAPHICS:
- Total population
- Population density
- Age distribution (children, working age, seniors)
- Household income levels
- Education levels
- Car ownership rates
- Commuting patterns

URBAN FORM:
- Housing density
- Land use mix
- Employment density

SOCIAL INDICATORS:
- Low income indicators
- Immigration patterns
- Language diversity
"""

def identify_key_census_variables(census_df):
    """
    Identify the most useful census variables for bus route optimization
    """
    key_variables = {}
    
    # Look for common census variable names
    population_vars = [col for col in census_df.columns if 'pop' in col.lower() or 'population' in col.lower()]
    income_vars = [col for col in census_df.columns if 'income' in col.lower() or 'inc' in col.lower()]
    age_vars = [col for col in census_df.columns if 'age' in col.lower()]
    density_vars = [col for col in census_df.columns if 'density' in col.lower()]
    
    print("Key census variables identified:")
    print(f"  Population: {population_vars}")
    print(f"  Income: {income_vars}") 
    print(f"  Age: {age_vars}")
    print(f"  Density: {density_vars}")
    
    return {
        'population': population_vars[0] if population_vars else None,
        'income': income_vars[0] if income_vars else None,
        'density': density_vars[0] if density_vars else None
    }

# Usage in ML pipeline:
# census_vars = identify_key_census_variables(census_df)
# roads_df['da_population'] = attach_census_variable(roads_df, census_df, census_vars['population'])

'''