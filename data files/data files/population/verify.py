# =============================================================================
# QUICK CITY LIMITS VERIFICATION
# =============================================================================

def verify_census_in_city_limits(census_csv_path, city_limits_rest_url):
    """
    Quick verification that census data points are within city limits
    Returns: "YES" or "NO" with error details
    """
    import geopandas as gpd
    import pandas as pd
    import requests
    from shapely.geometry import Point
    
    print("🔍 Verifying census data against city limits...")
    
    # Load census data
    census_df = pd.read_csv(census_csv_path)
    print(f"Checking {len(census_df)} census areas...")
    
    # Get city boundaries
    params = {
        'where': '1=1',
        'outFields': '*',
        'returnGeometry': 'true',
        'f': 'geojson',
        'outSR': '4326'
    }
    
    try:
        response = requests.get(city_limits_rest_url + '/query', params=params, timeout=30)
        response.raise_for_status()
        
        # Create temporary file for city boundaries
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.geojson', delete=False) as f:
            f.write(response.text)
            temp_geojson = f.name
        
        city_gdf = gpd.read_file(temp_geojson)
        
        # Create GeoDataFrame from census centroids
        if 'centroid_long' in census_df.columns and 'centroid_lat' in census_df.columns:
            census_points = gpd.GeoDataFrame(
                census_df,
                geometry=[Point(xy) for xy in zip(census_df['centroid_long'], census_df['centroid_lat'])],
                crs="EPSG:4326"
            )
        else:
            print("❌ No centroid columns found in census data")
            return "NO - Missing centroid coordinates"
        
        # Ensure same CRS
        census_points = census_points.to_crs(city_gdf.crs)
        
        # Check which points are within city boundaries
        points_within_city = gpd.sjoin(census_points, city_gdf, how='inner', predicate='within')
        points_outside_city = gpd.sjoin(census_points, city_gdf, how='left', predicate='within')
        points_outside_city = points_outside_city[points_outside_city.index_right.isna()]
        
        total_points = len(census_points)
        points_inside = len(points_within_city)
        points_outside = len(points_outside_city)
        
        print(f"📊 Verification Results:")
        print(f"  Total census areas: {total_points}")
        print(f"  Within city limits: {points_inside}")
        print(f"  Outside city limits: {points_outside}")
        
        if points_outside == 0:
            print("✅ YES - All census areas are within city limits!")
            return "YES"
        else:
            # Calculate how "badly" we're outside
            outside_percentage = (points_outside / total_points) * 100
            print(f"❌ NO - {outside_percentage:.1f}% of census areas are outside city limits")
            
            # Show the worst offenders (farthest points)
            if points_outside > 0:
                print(f"🔍 Sample of outside points:")
                outside_samples = points_outside_city.head(3)
                for idx, row in outside_samples.iterrows():
                    print(f"   - DA {row.name}: ({row['centroid_long']:.4f}, {row['centroid_lat']:.4f})")
            
            return f"NO - {outside_percentage:.1f}% outside ({points_outside} areas)"
            
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return f"NO - Error: {str(e)}"

# Even quicker version - just check bounding box
def quick_bounds_check(census_csv_path, city_limits_rest_url):
    """
    Ultra-fast verification using bounding box comparison
    """
    import pandas as pd
    import requests
    
    print("⚡ Quick bounds verification...")
    
    # Load census data
    census_df = pd.read_csv(census_csv_path)
    
    # Get city bounding box from REST API
    params = {
        'where': '1=1',
        'geometryType': 'esriGeometryEnvelope',
        'returnGeometry': 'true',
        'f': 'json'
    }
    
    try:
        response = requests.get(city_limits_rest_url + '/query', params=params, timeout=30)
        data = response.json()
        
        # Extract city bounding box
        if 'extent' in data:
            city_bounds = data['extent']
            city_minx, city_miny = city_bounds['xmin'], city_bounds['ymin']
            city_maxx, city_maxy = city_bounds['xmax'], city_bounds['ymax']
        else:
            # Alternative: use first feature's geometry
            city_bounds = data['features'][0]['geometry']['extent']
            city_minx, city_miny = city_bounds[0], city_bounds[1]
            city_maxx, city_maxy = city_bounds[2], city_bounds[3]
        
        print(f"City bounds: [{city_minx:.4f}, {city_miny:.4f}] to [{city_maxx:.4f}, {city_maxy:.4f}]")
        
        # Check census points against bounds
        if 'centroid_long' in census_df.columns and 'centroid_lat' in census_df.columns:
            outside_points = census_df[
                (census_df['centroid_long'] < city_minx) | 
                (census_df['centroid_long'] > city_maxx) |
                (census_df['centroid_lat'] < city_miny) | 
                (census_df['centroid_lat'] > city_maxy)
            ]
            
            total_points = len(census_df)
            outside_count = len(outside_points)
            
            if outside_count == 0:
                print("✅ YES - All census centroids within city bounding box!")
                return "YES"
            else:
                outside_percentage = (outside_count / total_points) * 100
                print(f"❌ NO - {outside_percentage:.1f}% outside city bounds ({outside_count} areas)")
                
                # Show worst coordinate violations
                if outside_count > 0:
                    worst_long = outside_points['centroid_long'].iloc[0]
                    worst_lat = outside_points['centroid_lat'].iloc[0]
                    print(f"   Worst point: ({worst_long:.4f}, {worst_lat:.4f})")
                
                return f"NO - {outside_percentage:.1f}% outside bounds"
        else:
            print("❌ No centroid columns found")
            return "NO - Missing centroids"
            
    except Exception as e:
        print(f"❌ Quick check failed: {e}")
        return f"NO - Error: {str(e)}"

# =============================================================================
# RUN VERIFICATION
# =============================================================================

def run_verification():
    """
    Run both verification methods for confidence
    """
    print("🚨 RUNNING CENSUS DATA VERIFICATION")
    print("=" * 50)
    
    result1 = quick_bounds_check(
        census_csv_path="victoria_census_da.csv",  # Your output CSV
        city_limits_rest_url="https://maps.victoria.ca/server/rest/services/OpenData/OpenData_Land/MapServer/2"
    )
    
    print("\n" + "=" * 50)
    
    result2 = verify_census_in_city_limits(
        census_csv_path="victoria_census_da.csv",  # Your output CSV  
        city_limits_rest_url="https://maps.victoria.ca/server/rest/services/OpenData/OpenData_Land/MapServer/2"
    )
    
    print("\n" + "=" * 50)
    print("🎯 FINAL VERDICT:")
    
    if "YES" in result1 and "YES" in result2:
        print("✅ CONFIRMED: Census data is properly within city limits!")
    else:
        print("❌ ISSUE: Some census data may be outside city limits")
        print(f"   Quick check: {result1}")
        print(f"   Detailed check: {result2}")

# Run it immediately:
run_verification()