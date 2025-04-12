import os
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point
import logging
from pathlib import Path
from tqdm import tqdm
import math
import rasterio
from rasterio.mask import mask
from rasterstats import zonal_stats

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

def count_points_in_polygons(polygons_gdf, points_gdf, feature_name):
    """
    Count points that fall within each polygon.
    
    Args:
        polygons_gdf: GeoDataFrame with polygons
        points_gdf: GeoDataFrame with points
        feature_name: Name for the count column to be created
        
    Returns:
        Updated polygons_gdf with counts
    """
    logging.info(f"Counting {feature_name} in each census tract...")
    
    # Ensure both dataframes have the same CRS
    if polygons_gdf.crs != points_gdf.crs:
        logging.info(f"Converting points from {points_gdf.crs} to {polygons_gdf.crs}")
        points_gdf = points_gdf.to_crs(polygons_gdf.crs)
    
    # Initialize count column with zeros
    polygons_gdf[feature_name] = 0
    
    if len(points_gdf) > 0:
        if feature_name == 'data_center_count':
            # For data centers, use sjoin_nearest to ensure each point is assigned to exactly one polygon
            logging.info(f"Using nearest join approach for {len(points_gdf)} data centers")
            joined = gpd.sjoin_nearest(points_gdf, polygons_gdf, how='left', distance_col='dist')
            
            # Count occurrences of each polygon index
            counts = joined.groupby('index_right').size()
            polygons_gdf.loc[counts.index, feature_name] = counts
            
            logging.info(f"Assigned {len(joined)} data centers to {len(counts)} census tracts")
        else:
            # For other features, use the original method with spatial join
            joined = gpd.sjoin(polygons_gdf, points_gdf, how='left', predicate='intersects')
            if not joined.empty:
                counts = joined.groupby(joined.index).size()
                polygons_gdf.loc[counts.index, feature_name] = counts
    
    # Ensure the count is an integer
    polygons_gdf[feature_name] = polygons_gdf[feature_name].fillna(0).astype(int)
    
    # Now calculate the has_feature column after we've properly counted
    if feature_name == 'data_center_count':
        polygons_gdf['has_data_center'] = (polygons_gdf[feature_name] > 0).astype(int)
        logging.info(f"Found {polygons_gdf['has_data_center'].sum()} census tracts with at least one data center")
        logging.info(f"Total data centers counted: {polygons_gdf[feature_name].sum()}")
    
    return polygons_gdf

def calculate_min_distance(polygons_gdf, features_gdf, feature_name):
    """
    Calculate minimum distance from each polygon centroid to the nearest feature.
    
    Args:
        polygons_gdf: GeoDataFrame with polygons
        features_gdf: GeoDataFrame with features to calculate distance to
        feature_name: Name for the distance column to be created
        
    Returns:
        Updated polygons_gdf with distances
    """
    logging.info(f"Calculating minimum distance to {feature_name} for each census tract...")
    
    # Ensure both dataframes have the same CRS
    if polygons_gdf.crs != features_gdf.crs:
        logging.info(f"Converting features from {features_gdf.crs} to {polygons_gdf.crs}")
        features_gdf = features_gdf.to_crs(polygons_gdf.crs)
    
    # Use UTM zone 23S for São Paulo (EPSG:31983)
    utm_crs = 'EPSG:31983'
    
    # First convert both datasets to UTM for accurate distances and proper centroids
    logging.info(f"Converting data to {utm_crs} for accurate calculations")
    polygons_utm = polygons_gdf.to_crs(utm_crs)
    features_utm = features_gdf.to_crs(utm_crs)
    
    # Calculate centroids AFTER projecting to UTM to avoid the warning
    logging.info("Calculating centroids in projected CRS")
    polygons_utm['centroid'] = polygons_utm.geometry.centroid
    centroids_gdf = gpd.GeoDataFrame(
        geometry=polygons_utm['centroid'],
        index=polygons_utm.index,
        crs=utm_crs
    )
    
    # Use spatial index for faster nearest neighbor calculations
    logging.info("Building spatial index for efficient distance calculations")
    from shapely.strtree import STRtree
    
    # Build spatial index for features
    if len(features_utm) > 0:
        tree = STRtree(features_utm.geometry.values)
        
        # Process in batches to manage memory
        batch_size = 5000
        num_batches = (len(centroids_gdf) // batch_size) + 1
        
        min_distances = []
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(centroids_gdf))
            
            batch_centroids = centroids_gdf.iloc[start_idx:end_idx]
            if len(batch_centroids) == 0:
                continue
                
            logging.info(f"Processing batch {batch_idx+1}/{num_batches} ({start_idx}-{end_idx})")
            batch_distances = []
            
            # Find nearest neighbor for each centroid
            for idx, centroid in tqdm(enumerate(batch_centroids.geometry), 
                                    total=len(batch_centroids),
                                    desc=f"Finding nearest {feature_name}"):
                # Get nearest neighbor
                nearest_idx = tree.nearest(centroid)
                
                # Check if we got a valid result
                if nearest_idx is not None:
                    # Calculate distance (in meters) and convert to km
                    nearest_feature = features_utm.geometry.iloc[nearest_idx]
                    dist = centroid.distance(nearest_feature) / 1000
                    batch_distances.append(dist)
                else:
                    # No nearest feature found (shouldn't happen with STRtree)
                    batch_distances.append(np.nan)
            
            min_distances.extend(batch_distances)
    else:
        min_distances = [np.nan] * len(centroids_gdf)
    
    # Add calculated metrics to original polygons
    polygons_gdf[f'min_dist_to_{feature_name}_km'] = min_distances
    
    return polygons_gdf

def calculate_distance_to_lines(polygons_gdf, lines_gdf, feature_name):
    """
    Calculate minimum distance from each polygon centroid to the nearest line.
    
    Args:
        polygons_gdf: GeoDataFrame with polygons
        lines_gdf: GeoDataFrame with lines
        feature_name: Name for the distance column to be created
        
    Returns:
        Updated polygons_gdf with distances
    """
    logging.info(f"Calculating minimum distance to {feature_name} for each census tract...")
    
    # Ensure both dataframes have the same CRS
    if polygons_gdf.crs != lines_gdf.crs:
        logging.info(f"Converting lines from {lines_gdf.crs} to {polygons_gdf.crs}")
        lines_gdf = lines_gdf.to_crs(polygons_gdf.crs)
    
    # Use UTM zone 23S for São Paulo (EPSG:31983)
    utm_crs = 'EPSG:31983'
    
    # First convert both datasets to UTM for accurate distances and proper centroids
    logging.info(f"Converting data to {utm_crs} for accurate calculations")
    polygons_utm = polygons_gdf.to_crs(utm_crs)
    lines_utm = lines_gdf.to_crs(utm_crs)
    
    # Calculate centroids AFTER projecting to UTM to avoid the warning
    logging.info("Calculating centroids in projected CRS")
    polygons_utm['centroid'] = polygons_utm.geometry.centroid
    centroids_gdf = gpd.GeoDataFrame(
        geometry=polygons_utm['centroid'],
        index=polygons_utm.index,
        crs=utm_crs
    )
    
    # Process in batches to manage memory
    batch_size = 5000
    num_batches = (len(centroids_gdf) // batch_size) + 1
    
    min_distances = []
    
    # Use spatial indexing for lines
    if len(lines_utm) > 0:
        from shapely.strtree import STRtree
        tree = STRtree(lines_utm.geometry.values)
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(centroids_gdf))
            
            batch_centroids = centroids_gdf.iloc[start_idx:end_idx]
            if len(batch_centroids) == 0:
                continue
                
            logging.info(f"Processing batch {batch_idx+1}/{num_batches} ({start_idx}-{end_idx})")
            batch_distances = []
            
            # Calculate distances for this batch
            for idx, centroid in tqdm(enumerate(batch_centroids.geometry), 
                                    total=len(batch_centroids),
                                    desc=f"Finding nearest {feature_name}"):
                # Get nearest indices (may return multiple)
                nearest_idxs = tree.query(centroid, predicate='dwithin', distance=50000)  # 50km initial search
                
                if len(nearest_idxs) > 0:
                    # Calculate actual distances to nearby lines and find minimum
                    distances = [
                        centroid.distance(lines_utm.geometry.iloc[i]) / 1000  # Convert m to km
                        for i in nearest_idxs
                    ]
                    min_dist = min(distances)
                    batch_distances.append(min_dist)
                else:
                    # If no lines found within 50km, do a full search
                    distances = [
                        centroid.distance(line) / 1000  # Convert m to km
                        for line in lines_utm.geometry
                    ]
                    
                    if not distances:
                        batch_distances.append(np.nan)
                    else:
                        min_dist = min(distances)
                        batch_distances.append(min_dist)
            
            min_distances.extend(batch_distances)
    else:
        min_distances = [np.nan] * len(centroids_gdf)
    
    # Add calculated metrics to original polygons dataframe
    polygons_gdf[f'min_dist_to_{feature_name}_km'] = min_distances
    
    return polygons_gdf

def calculate_viirs_stats(polygons_gdf, viirs_path):
    """
    Calculate mean VIIRS nighttime lights value for each polygon.
    
    Args:
        polygons_gdf: GeoDataFrame with polygons
        viirs_path: Path to the VIIRS TIF file
        
    Returns:
        Updated polygons_gdf with VIIRS mean value
    """
    logging.info("Calculating VIIRS nighttime lights mean value for each census tract...")
    
    # Ensure polygons are in the same CRS as VIIRS data (usually EPSG:4326)
    with rasterio.open(viirs_path) as src:
        viirs_crs = src.crs
        if polygons_gdf.crs != viirs_crs:
            logging.info(f"Converting polygons from {polygons_gdf.crs} to {viirs_crs}")
            polygons_gdf = polygons_gdf.to_crs(viirs_crs)
    
    # Calculate zonal statistics (mean only)
    stats = zonal_stats(
        polygons_gdf.geometry,
        str(viirs_path),  # Convert Path to string for zonal_stats
        stats=['mean'],
        nodata=None,
        all_touched=True
    )
    
    # Convert stats to DataFrame
    stats_df = pd.DataFrame(stats)
    
    # Add mean value to polygons_gdf
    polygons_gdf['viirs_mean'] = stats_df['mean'].fillna(0)
    
    return polygons_gdf

def main():
    """
    Process census tracts for São Paulo state.
    """
    logging.info("Starting processing of São Paulo census tracts...")
    
    # Define paths using pathlib
    base_dir = Path(__file__).parent.parent.parent
    
    # Make sure we're using the correct Google Drive path
    google_drive_path = "GoogleDrive-leodavinci550@gmail.com"
    if google_drive_path not in str(base_dir):
        base_dir = Path(str(base_dir).replace("GoogleDrive-", google_drive_path))
    logging.info(f"Using Google Drive path: {google_drive_path}")
    
    # Load census tracts
    logging.info("Loading census tracts shapefile...")
    
    # Correct path with "brasil" (with s)
    census_tracts_path = base_dir / "data/raw/brasil/BR_setores_CD2022/BR_setores_CD2022.shp"
    
    if not census_tracts_path.exists():
        raise FileNotFoundError(f"Census tracts shapefile not found at: {census_tracts_path}")
    else:
        logging.info(f"Found census tracts file at: {census_tracts_path}")
    
    # Read only São Paulo state to reduce memory usage (CD_UF = 35)
    census_tracts = gpd.read_file(
        census_tracts_path,
        where="CD_UF = '35'"  # Pre-filter to São Paulo state
    )
    
    # Log columns to verify data
    logging.info(f"Census tract columns: {census_tracts.columns.tolist()}")
    
    # Find state column and filter for São Paulo
    state_col = None
    for col in ['CD_UF', 'UF', 'ESTADO']:
        if col in census_tracts.columns:
            state_col = col
            break
    
    if not state_col:
        raise ValueError("Could not find state column in census tracts data")
    
    logging.info(f"Found state column: {state_col}")
    logging.info(f"Found states: {census_tracts[state_col].unique()}")
    
    # São Paulo state code is 35
    sp_code = '35'
    logging.info(f"Found São Paulo state code: {sp_code}")
    
    # Filter for São Paulo
    census_tracts = census_tracts[census_tracts[state_col] == sp_code].copy()
    logging.info(f"Loaded {len(census_tracts)} census tracts for São Paulo with CRS: {census_tracts.crs}")
    
    # --- Select desired columns ---
    logging.info("Selecting relevant initial columns...")
    # Keep geometry for spatial operations, CD_SETOR as ID, CD_SIT for urban/rural, and original AREA_KM2 for comparison
    columns_to_keep = ['geometry', 'CD_SETOR', 'CD_SIT', 'AREA_KM2']

    # Check if all required columns exist
    missing_cols = [col for col in columns_to_keep if col not in census_tracts.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in census data: {missing_cols}. Available columns: {census_tracts.columns.tolist()}")
        
    census_tracts = census_tracts[columns_to_keep].copy()
    logging.info(f"Selected initial columns: {census_tracts.columns.tolist()}")
    # --- End column selection ---

    # Add VIIRS path
    viirs_path = base_dir / "data/raw/global/VIIRS/SVDNB_npp_20241201-20241231_global_vcmcfg_v10_c202501131100.avg_rade9h.tif"
    
    # Load infrastructure data
    logging.info("Loading processed infrastructure data...")
    processed_dir = base_dir / "data/processed/brasil"
    
    # Load data centers
    data_centers = gpd.read_file(processed_dir / "data_centers.geojson")
    logging.info(f"Loaded {len(data_centers)} data centers with CRS: {data_centers.crs}")
    
    # Load cell towers
    cell_towers = gpd.read_file(processed_dir / "cell_towers.geojson")
    logging.info(f"Loaded {len(cell_towers)} cell towers with CRS: {cell_towers.crs}")
    
    # Load electrical infrastructure
    substations = gpd.read_file(processed_dir / "substations.geojson")
    logging.info(f"Loaded {len(substations)} substations with CRS: {substations.crs}")
    
    transmission_lines = gpd.read_file(processed_dir / "transmission_lines.geojson")
    logging.info(f"Loaded {len(transmission_lines)} transmission lines with CRS: {transmission_lines.crs}")
    
    # Use SIRGAS 2000 (EPSG:4674) as target CRS for all data
    target_crs = "EPSG:4674"
    logging.info(f"Using {target_crs} as the target CRS for all data")
    
    # Reproject all data to target CRS if needed
    if data_centers.crs != target_crs:
        logging.info(f"Reprojecting data centers from {data_centers.crs} to {target_crs}")
        data_centers = data_centers.to_crs(target_crs)
    
    if cell_towers.crs != target_crs:
        logging.info(f"Reprojecting cell towers from {cell_towers.crs} to {target_crs}")
        cell_towers = cell_towers.to_crs(target_crs)
    
    if substations.crs != target_crs:
        logging.info(f"Reprojecting substations from {substations.crs} to {target_crs}")
        substations = substations.to_crs(target_crs)
    
    if transmission_lines.crs != target_crs:
        logging.info(f"Reprojecting transmission lines from {transmission_lines.crs} to {target_crs}")
        transmission_lines = transmission_lines.to_crs(target_crs)
    
    # Count cell towers in each census tract
    logging.info("Counting cell_tower_count in each census tract...")
    census_tracts = count_points_in_polygons(census_tracts, cell_towers, 'cell_tower_count')
    
    # Calculate minimum distance to substations
    logging.info("Calculating minimum distance to substation for each census tract...")
    census_tracts = calculate_min_distance(census_tracts, substations, 'substation')
    
    # Calculate minimum distance to transmission lines
    logging.info("Calculating minimum distance to transmission_line for each census tract...")
    census_tracts = calculate_distance_to_lines(census_tracts, transmission_lines, 'transmission_line')
    
    # Count data centers in each census tract
    logging.info("Counting data_center_count in each census tract...")
    census_tracts = count_points_in_polygons(census_tracts, data_centers, 'data_center_count')
    
    # Calculate VIIRS statistics
    logging.info("Processing VIIRS nighttime lights data...")
    census_tracts = calculate_viirs_stats(census_tracts, viirs_path)
    
    # Calculate cell tower density using original area
    logging.info("Calculating cell tower density...")
    census_tracts['cell_tower_density'] = census_tracts['cell_tower_count'] / census_tracts['AREA_KM2']
    
    # Save output
    output_dir = base_dir / "data/model_input/brasil"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Select final columns to keep
    logging.info("Selecting final columns for output...")
    final_columns = [
        'CD_SETOR',  # tract ID
        'CD_SIT',    # urban/rural status
        'AREA_KM2',  # original area from source
        'data_center_count',
        'has_data_center',
        'cell_tower_count',
        'cell_tower_density',
        'min_dist_to_substation_km',
        'min_dist_to_transmission_line_km',
        'viirs_mean'
    ]
    
    # Verify all columns exist
    missing_cols = [col for col in final_columns if col not in census_tracts.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in final data: {missing_cols}")
    
    # Rename AREA_KM2 to area_km2 for consistency
    census_tracts = census_tracts[final_columns].rename(columns={'AREA_KM2': 'area_km2'})
    logging.info(f"Final columns: {census_tracts.columns.tolist()}")
    
    # Save full dataset with descriptive name
    output_path = output_dir / "sao_paulo_census_tracts_full.geojson"
    output_path_csv = output_dir / "sao_paulo_census_tracts_full.csv"
    
    logging.info(f"Created {len(census_tracts)} records")
    
    # Save non-geometric data to CSV first
    census_tracts.to_csv(output_path_csv, index=False)
    
    # Add geometry for GeoJSON output
    if 'geometry' in census_tracts.columns:
        census_tracts = census_tracts.drop(columns=['geometry'])
    census_tracts_geo = census_tracts.copy()
    census_tracts_geo['geometry'] = census_tracts.geometry
    census_tracts_geo = gpd.GeoDataFrame(census_tracts_geo, geometry='geometry', crs=census_tracts.crs)
    census_tracts_geo.to_file(output_path, driver='GeoJSON')
    
    logging.info(f"Saved full census tracts dataset to {output_path} and {output_path_csv}")
    logging.info("Census tract processing completed successfully!")

if __name__ == "__main__":
    main() 