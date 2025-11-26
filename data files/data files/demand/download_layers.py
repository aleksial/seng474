# =============================================================================
# DOWNLOAD ALL LAYERS AS GEOJSON FOR QGIS
# =============================================================================

import requests
import json

def download_all_layers_for_qgis(layer_urls, output_folder="qgis_layers"):
    """
    Download each layer as GeoJSON for import into QGIS
    """
    import os
    os.makedirs(output_folder, exist_ok=True)
    
    print("📥 Downloading all layers for QGIS processing...")
    
    for layer_url in layer_urls:
        layer_id = layer_url.split('/')[-1]
        output_file = f"{output_folder}/layer_{layer_id}.geojson"
        
        print(f"🔄 Downloading Layer {layer_id}...")
        
        try:
            # Get layer metadata to get name
            metadata_url = f"{layer_url}?f=json"
            metadata_response = requests.get(metadata_url, timeout=30)
            layer_info = metadata_response.json()
            layer_name = layer_info.get('name', f'Layer_{layer_id}')
            
            # Download all features as GeoJSON
            query_url = f"{layer_url}/query"
            params = {
                'where': '1=1',
                'outFields': '*',
                'returnGeometry': 'true',
                'outSR': '4326',
                'f': 'geojson',
                'resultRecordCount': 10000
            }
            
            response = requests.get(query_url, params=params, timeout=60)
            response.raise_for_status()
            
            # Save as GeoJSON
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(response.text)
            
            # Verify the download
            with open(output_file, 'r') as f:
                geojson_data = json.load(f)
                feature_count = len(geojson_data.get('features', []))
            
            print(f"   ✅ {layer_name}: {feature_count} features -> {output_file}")
            
        except Exception as e:
            print(f"   ❌ Error downloading layer {layer_id}: {e}")
    
    print(f"\n🎉 All layers downloaded to '{output_folder}' folder!")
    print("📁 Now import these into QGIS for processing")

# Download all layers
layer_urls = [
    "https://services.arcgis.com/9PtzeAadJyclx9t7/ArcGIS/rest/services/Accessibility_and_Demand_of_Public_Transit_in_Victoria_BC_WFL1/FeatureServer/9",
    "https://services.arcgis.com/9PtzeAadJyclx9t7/ArcGIS/rest/services/Accessibility_and_Demand_of_Public_Transit_in_Victoria_BC_WFL1/FeatureServer/40",
    "https://services.arcgis.com/9PtzeAadJyclx9t7/ArcGIS/rest/services/Accessibility_and_Demand_of_Public_Transit_in_Victoria_BC_WFL1/FeatureServer/34",
    "https://services.arcgis.com/9PtzeAadJyclx9t7/ArcGIS/rest/services/Accessibility_and_Demand_of_Public_Transit_in_Victoria_BC_WFL1/FeatureServer/26",
    "https://services.arcgis.com/9PtzeAadJyclx9t7/ArcGIS/rest/services/Accessibility_and_Demand_of_Public_Transit_in_Victoria_BC_WFL1/FeatureServer/25",
    "https://services.arcgis.com/9PtzeAadJyclx9t7/ArcGIS/rest/services/Accessibility_and_Demand_of_Public_Transit_in_Victoria_BC_WFL1/FeatureServer/41",
    "https://services.arcgis.com/9PtzeAadJyclx9t7/ArcGIS/rest/services/Accessibility_and_Demand_of_Public_Transit_in_Victoria_BC_WFL1/FeatureServer/6",
    "https://services.arcgis.com/9PtzeAadJyclx9t7/ArcGIS/rest/services/Accessibility_and_Demand_of_Public_Transit_in_Victoria_BC_WFL1/FeatureServer/5",
    "https://services.arcgis.com/9PtzeAadJyclx9t7/ArcGIS/rest/services/Accessibility_and_Demand_of_Public_Transit_in_Victoria_BC_WFL1/FeatureServer/4",
    "https://services.arcgis.com/9PtzeAadJyclx9t7/ArcGIS/rest/services/Accessibility_and_Demand_of_Public_Transit_in_Victoria_BC_WFL1/FeatureServer/3",
    "https://services.arcgis.com/9PtzeAadJyclx9t7/ArcGIS/rest/services/Accessibility_and_Demand_of_Public_Transit_in_Victoria_BC_WFL1/FeatureServer/2",
    "https://services.arcgis.com/9PtzeAadJyclx9t7/ArcGIS/rest/services/Accessibility_and_Demand_of_Public_Transit_in_Victoria_BC_WFL1/FeatureServer/1"
]

download_all_layers_for_qgis(layer_urls)