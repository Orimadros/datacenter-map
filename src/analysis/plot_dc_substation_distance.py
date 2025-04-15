import logging
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import numpy as np
from shapely.ops import nearest_points
from shapely.strtree import STRtree
from scipy.spatial import KDTree
from tqdm import tqdm
from typing import Union

# Add project root to Python path
root_dir = str(Path(__file__).resolve().parent.parent.parent)
if root_dir not in sys.path:
    sys.path.append(root_dir)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Utility Functions --- 

def get_project_root() -> Path:
    """Find the project root based on a marker file or directory."""
    current_file_path = Path(__file__).resolve()
    for parent in current_file_path.parents:
        if (parent / 'README.md').exists() or (parent / '.git').exists() or (parent / 'setup.py').exists():
            # logging.info(f"Project root identified as: {parent}")
            return parent
    logging.warning("Could not automatically determine project root via markers. Defaulting to 2 levels up.")
    project_root = current_file_path.parent.parent.parent
    if (project_root / 'src').is_dir() and (project_root / 'data').is_dir():
         logging.info(f"Using default project root: {project_root}")
         return project_root
    else:
         logging.error("Project root structure not as expected. Defaulting to CWD, paths might be incorrect.")
         return Path.cwd()

def load_and_reproject(file_path: Path, target_crs: str, filter_query: str = None) -> Union[gpd.GeoDataFrame, None]:
    """Load a GeoJSON/Shapefile, optionally filter, and reproject."""
    logging.info(f"Loading and reprojecting {file_path.name} to {target_crs}...")
    if not file_path.exists():
        logging.error(f"File not found: {file_path}")
        return None
    try:
        if filter_query:
            gdf = gpd.read_file(file_path, where=filter_query)
        else:
            gdf = gpd.read_file(file_path)
        
        if gdf.empty:
            logging.warning(f"Loaded GeoDataFrame is empty: {file_path.name} (Query: {filter_query})")
            # Return an empty GDF with target CRS if possible, or None
            try:
                return gpd.GeoDataFrame(geometry=[], crs=target_crs)
            except:
                return None 
            
        if gdf.crs is None:
            logging.warning(f"CRS not found for {file_path.name}. Assuming EPSG:4326.")
            gdf.set_crs("EPSG:4326", inplace=True)
            
        gdf_proj = gdf.to_crs(target_crs)
        logging.info(f"Loaded and reprojected {len(gdf_proj)} features from {file_path.name}.")
        return gdf_proj
    except Exception as e:
        logging.error(f"Error loading/reprojecting {file_path.name}: {e}")
        return None

def calculate_min_distance_points(source_gdf, target_gdf, feature_name):
    """Calculate min distance from source points to target points using STRtree."""
    logging.info(f"Calculating minimum distance from {len(source_gdf)} points to {len(target_gdf)} {feature_name} points...")
    if target_gdf.empty or source_gdf.empty:
        logging.warning(f"Source or target GDF for {feature_name} is empty. Returning NaNs.")
        return pd.Series([np.nan] * len(source_gdf), index=source_gdf.index)

    tree = STRtree(target_gdf.geometry.values)
    min_distances = []

    for idx, point in tqdm(source_gdf.geometry.items(), total=len(source_gdf), desc=f"Finding nearest {feature_name}"):
        try:
            # Query for nearest geometry index in the tree
            nearest_geom_idx = tree.nearest(point)
            if nearest_geom_idx is not None:
                nearest_feature = target_gdf.geometry.iloc[nearest_geom_idx]
                dist = point.distance(nearest_feature) / 1000  # convert meters to km
                min_distances.append(dist)
            else:
                min_distances.append(np.nan)
        except Exception as e:
            logging.error(f"Error calculating distance for index {idx} to {feature_name}: {e}")
            min_distances.append(np.nan)

    return pd.Series(min_distances, index=source_gdf.index)

def calculate_min_distance_lines(source_gdf, target_lines_gdf, feature_name):
    """Calculate min distance from source points to target lines using STRtree."""
    logging.info(f"Calculating minimum distance from {len(source_gdf)} points to {len(target_lines_gdf)} {feature_name} lines...")
    if target_lines_gdf.empty or source_gdf.empty:
        logging.warning(f"Source or target GDF for {feature_name} is empty. Returning NaNs.")
        return pd.Series([np.nan] * len(source_gdf), index=source_gdf.index)

    tree = STRtree(target_lines_gdf.geometry.values)
    min_distances = []

    for idx, point in tqdm(source_gdf.geometry.items(), total=len(source_gdf), desc=f"Finding nearest {feature_name}"):
        try:
            # Query returns indices of geometries intersecting the point's buffer (or potentially nearest)
            # A direct nearest query is often sufficient and simpler with STRtree for point-to-line
            nearest_line_idx = tree.nearest(point)
            if nearest_line_idx is not None:
                nearest_line = target_lines_gdf.geometry.iloc[nearest_line_idx]
                dist = point.distance(nearest_line) / 1000 # Convert meters to km
                min_distances.append(dist)
            else:
                min_distances.append(np.nan)
        except Exception as e:
            logging.error(f"Error calculating distance for index {idx} to {feature_name}: {e}")
            min_distances.append(np.nan)

    return pd.Series(min_distances, index=source_gdf.index)

def categorize_provider(name):
    """Categorizes provider based on name."""
    if pd.isna(name):
        return "Outros"
    name_lower = name.lower()
    # Adjusted to capture AWS specifically as well
    if "amazon" in name_lower or "aws" in name_lower:
        return "Amazon/AWS"
    elif "equinix" in name_lower:
        return "Equinix"
    elif "ascenty" in name_lower:
        return "Ascenty"
    elif "scala" in name_lower:
        return "Scala"
    else:
        return "Outros"

def count_towers_in_radius(source_gdf, tower_gdf, radius_meters):
    """Counts towers within a given radius of source points using KDTree."""
    radius_km = int(radius_meters / 1000)
    logging.info(f"Counting towers within {radius_km}km ({radius_meters}m) radius...")
    if tower_gdf.empty or source_gdf.empty:
        logging.warning("Source or tower GDF is empty. Returning zeros.")
        return pd.Series([0] * len(source_gdf), index=source_gdf.index)

    # Prepare coordinates for KDTree
    source_coords = np.array(list(zip(source_gdf.geometry.x, source_gdf.geometry.y)))
    tower_coords = np.array(list(zip(tower_gdf.geometry.x, tower_gdf.geometry.y)))

    if tower_coords.size == 0:
        logging.warning("No tower coordinates available. Returning zeros.")
        return pd.Series([0] * len(source_gdf), index=source_gdf.index)

    # Build KDTree on tower coordinates
    logging.info("Building KDTree on tower coordinates...")
    tower_tree = KDTree(tower_coords)

    # Query the tree for points within the radius for each source point
    logging.info(f"Querying KDTree for towers within {radius_meters}m...")
    try:
        indices_within_radius = tower_tree.query_ball_point(source_coords, r=radius_meters, workers=-1)
        # Count the number of points (indices) found for each source point
        counts = [len(indices) for indices in indices_within_radius]
    except Exception as e:
        logging.error(f"Error querying KDTree for radius {radius_meters}m: {e}")
        counts = [0] * len(source_gdf) # Assign zero counts on error

    logging.info(f"Finished counting towers within {radius_km}km. Mean count: {np.mean(counts):.2f}")
    return pd.Series(counts, index=source_gdf.index)

def plot_ranked_data(gdf: gpd.GeoDataFrame, value_col: str, output_path: Path, x_label: str, is_distance: bool):
    """Creates an elegant, color-coded scatter plot for ranked data (distance or count)."""
    if gdf.empty or value_col not in gdf.columns or gdf[value_col].isnull().all():
        logging.warning(f"No valid data or value column '{value_col}' to plot for '{x_label}'.")
        return

    plot_gdf = gdf.copy()
    plot_gdf.dropna(subset=[value_col], inplace=True)
    if plot_gdf.empty:
        logging.warning(f"No non-NaN values left for '{value_col}' to plot.")
        return
        
    # --- Outlier Removal (Specific to Hyperscaler Distance Plot) ---
    is_hyperscaler_dist_plot = is_distance and (value_col == 'min_dist_hyperscaler_km')
    num_outliers_to_remove = 4
    if is_hyperscaler_dist_plot and len(plot_gdf) > num_outliers_to_remove:
        outlier_indices = plot_gdf.nlargest(num_outliers_to_remove, value_col).index
        plot_gdf = plot_gdf.drop(outlier_indices)
        logging.info(f"Removed {num_outliers_to_remove} outliers from hyperscaler distance plot based on largest distance.")
        if plot_gdf.empty:
            logging.warning(f"No data points left after removing outliers for {value_col}.")
            return
            
    # --- Prepare data for plotting (continued) ---
    plot_gdf['provider_category'] = plot_gdf['name'].apply(categorize_provider)
    # Sort by the value column (distance or count)
    plot_gdf = plot_gdf.sort_values(by=value_col).reset_index(drop=True)
    plot_gdf['rank'] = np.arange(len(plot_gdf), 0, -1)

    # --- Define Colors --- 
    color_palette = {
        "Amazon/AWS": "orange", 
        "Equinix": "red",
        "Ascenty": "blue", 
        "Scala": "black", 
        "Outros": "grey"
    }
    hue_order = [cat for cat in color_palette.keys() if cat in plot_gdf['provider_category'].unique()]

    # --- Styling --- 
    plt.style.use('seaborn-v0_8-whitegrid') 
    fig, ax = plt.subplots(figsize=(12, 8)) 

    # Create scatter plot
    sns.scatterplot(
        data=plot_gdf,
        x=value_col,
        y='rank',
        hue='provider_category',
        hue_order=hue_order,
        palette=color_palette,
        s=70,  
        alpha=0.9, 
        edgecolor="darkgrey", 
        linewidth=0.6,
        ax=ax
    )

    # --- Customize Axes, Spines, Legend --- 
    ax.yaxis.set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xlabel(f"{x_label}", color='black', fontsize=13, labelpad=10) # Use provided x_label directly
    ax.tick_params(axis='x', colors='black', labelsize=11)
    ax.spines['bottom'].set_color('darkgrey')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(False)

    min_val = plot_gdf[value_col].min()
    max_val = plot_gdf[value_col].max()
    xlim_right_padding = 0.1 if is_hyperscaler_dist_plot else 0.05 
    ax.set_xlim(left=max(0, min_val - (max_val - min_val)*0.1), right=max_val * (1 + xlim_right_padding))
    
    # Generate title based on whether it's distance or count
    if is_distance:
        plot_title = f"Proximidade de Data Centers ao {x_label.split(' (')[0]} mais Próximo por Provedor"
        if is_hyperscaler_dist_plot:
            plot_title += " (Outliers Removidos)"
    else:
        plot_title = f"Classificação de Data Centers por {x_label.split(' (')[0]} Próximos por Provedor"
        
    ax.set_title(plot_title, fontsize=16, color='black', pad=20)

    # Legend customization
    legend = ax.legend(title='Provedor', title_fontsize='12', 
                       labelcolor='black', frameon=True, 
                       edgecolor='lightgrey',
                       facecolor='white', 
                       bbox_to_anchor=(1.02, 1), loc='upper left') 
    for text in legend.get_texts():
        text.set_color('black') 
    legend.get_title().set_color('black') 

    # --- Save Plot --- 
    plt.tight_layout(rect=[0, 0, 0.9, 1]) 
    try:
        plt.savefig(output_path, facecolor='white', dpi=300, bbox_inches='tight') 
        logging.info(f"Plot saved to {output_path}")
    except Exception as e:
        logging.error(f"Failed to save plot: {e}")
    plt.close()
    plt.style.use('default') 

# --- Main Execution --- 

def main():
    project_root = get_project_root()
    logging.info(f"Project root: {project_root}")

    # --- Configuration ---
    data_processed_dir = project_root / "data/processed/brasil"
    data_raw_dir = project_root / "data/raw/brasil"
    data_input_dir = project_root / "data/model_input/brasil"
    output_plot_dir = project_root / "outputs/figures"
    output_plot_dir.mkdir(parents=True, exist_ok=True)

    dc_path = data_processed_dir / "data_centers.geojson"
    substation_path = data_processed_dir / "substations.geojson"
    trans_line_path = data_processed_dir / "transmission_lines.geojson"
    hyperscaler_path = data_raw_dir / "hyperscaler_onramps.geojson" # Raw path as per previous scripts
    cell_tower_path = data_processed_dir / "cell_towers.geojson" # Added cell tower path
    census_tract_path = data_raw_dir / "BR_setores_CD2022/BR_setores_CD2022.shp"
    
    projected_crs = "EPSG:31983" # SIRGAS 2000 / UTM zone 23S (suitable for SP)

    # --- Load and Prepare Data ---
    data_centers_proj = load_and_reproject(dc_path, projected_crs)
    substations_proj = load_and_reproject(substation_path, projected_crs)
    trans_lines_proj = load_and_reproject(trans_line_path, projected_crs)
    hyperscalers_proj = load_and_reproject(hyperscaler_path, projected_crs)
    # Load and filter LTE cell towers
    cell_towers_proj = load_and_reproject(cell_tower_path, projected_crs, filter_query="radio = 'LTE'")
    census_tracts = load_and_reproject(census_tract_path, projected_crs, filter_query="CD_UF = '35'")

    if any(gdf is None for gdf in [data_centers_proj, substations_proj, trans_lines_proj, hyperscalers_proj, census_tracts]):
        logging.error("Failed to load one or more essential datasets. Exiting.")
        sys.exit(1)

    # Create São Paulo boundary
    logging.info("Creating São Paulo state boundary...")
    sp_boundary = census_tracts.dissolve()
    if sp_boundary.empty:
        logging.error("Failed to create São Paulo boundary. Exiting.")
        sys.exit(1)
    logging.info("São Paulo boundary created.")

    # Filter Data Centers within São Paulo
    logging.info("Filtering data centers within São Paulo boundary...")
    # Ensure DCs have valid geometries before spatial join
    data_centers_proj = data_centers_proj[data_centers_proj.geometry.is_valid & ~data_centers_proj.geometry.is_empty]
    if data_centers_proj.empty:
        logging.error("No valid data center geometries found after initial load/reprojection. Exiting.")
        sys.exit(1)
        
    # Perform spatial join - Use 'within' predicate
    # Create a temporary GeoDataFrame for the boundary to avoid index issues
    sp_boundary_gdf = gpd.GeoDataFrame([1], geometry=[sp_boundary.iloc[0].geometry], crs=projected_crs)
    # Keep relevant columns including 'name'
    data_centers_sp = gpd.sjoin(data_centers_proj[['name', 'geometry']], sp_boundary_gdf, how='inner', predicate='within')
    data_centers_sp = data_centers_sp.drop(columns=['index_right'], errors='ignore')
    logging.info(f"Filtered to {len(data_centers_sp)} data centers within São Paulo.")

    if data_centers_sp.empty:
        logging.error("No data centers found within the São Paulo boundary. Cannot proceed.")
        sys.exit(1)

    # --- Calculate Distances --- 
    data_centers_sp['min_dist_substation_km'] = calculate_min_distance_points(
        data_centers_sp, substations_proj, "Substation"
    )
    data_centers_sp['min_dist_trans_line_km'] = calculate_min_distance_lines(
        data_centers_sp, trans_lines_proj, "Transmission Line"
    )
    data_centers_sp['min_dist_hyperscaler_km'] = calculate_min_distance_points(
        data_centers_sp, hyperscalers_proj, "Hyperscaler"
    )

    # --- Calculate LTE Tower Counts --- 
    if cell_towers_proj is not None and not cell_towers_proj.empty:
        logging.info("Calculating LTE tower counts within specified radii...")
        data_centers_sp['lte_towers_10km_count'] = count_towers_in_radius(
            data_centers_sp, cell_towers_proj, radius_meters=10000
        )
        data_centers_sp['lte_towers_20km_count'] = count_towers_in_radius(
            data_centers_sp, cell_towers_proj, radius_meters=20000
        )
        data_centers_sp['lte_towers_100km_count'] = count_towers_in_radius(
            data_centers_sp, cell_towers_proj, radius_meters=100000
        )
    else:
         logging.warning("Skipping LTE tower count calculation due to missing/empty cell tower data.")
         # Add empty columns so plotting function doesn't fail if called
         data_centers_sp['lte_towers_10km_count'] = np.nan
         data_centers_sp['lte_towers_20km_count'] = np.nan
         data_centers_sp['lte_towers_100km_count'] = np.nan

    # --- Plotting --- 
    logging.info("Generating plots...")
    plot_substation_path = output_plot_dir / 'dc_ranked_substation_distance_color_white_scatter.png'
    plot_ranked_data(data_centers_sp, 
                     'min_dist_substation_km',
                     plot_substation_path, 
                     "Distância da Subestação mais Próxima (km)",
                     is_distance=True)
    
    plot_trans_line_path = output_plot_dir / 'dc_ranked_trans_line_distance_color_white_scatter.png'
    plot_ranked_data(data_centers_sp, 
                     'min_dist_trans_line_km',
                     plot_trans_line_path, 
                     "Distância da Linha de Transmissão mais Próxima (km)",
                     is_distance=True)
                          
    # Update filename for hyperscaler plot
    plot_hyperscaler_path = output_plot_dir / 'dc_ranked_hyperscaler_distance_color_white_no_outliers_scatter.png'
    plot_ranked_data(data_centers_sp, 
                     'min_dist_hyperscaler_km',
                     plot_hyperscaler_path, 
                     "Distância do On-Ramp de Hyperscaler mais Próximo (km)",
                     is_distance=True)

    # --- Generate LTE Tower Count Plots ---
    logging.info("Generating LTE tower count plots...")
    # Check if counts exist before plotting
    if 'lte_towers_10km_count' in data_centers_sp.columns:
        plot_lte_10km_path = output_plot_dir / 'dc_ranked_lte_towers_10km_color_white_scatter.png'
        plot_ranked_data(data_centers_sp,
                         'lte_towers_10km_count',
                         plot_lte_10km_path,
                         "Torres LTE em um Raio de 10km",
                         is_distance=False)
    
    if 'lte_towers_20km_count' in data_centers_sp.columns:
        plot_lte_20km_path = output_plot_dir / 'dc_ranked_lte_towers_20km_color_white_scatter.png'
        plot_ranked_data(data_centers_sp,
                         'lte_towers_20km_count',
                         plot_lte_20km_path,
                         "Torres LTE em um Raio de 20km",
                         is_distance=False)
                         
    if 'lte_towers_100km_count' in data_centers_sp.columns:                 
        plot_lte_100km_path = output_plot_dir / 'dc_ranked_lte_towers_100km_color_white_scatter.png'
        plot_ranked_data(data_centers_sp,
                         'lte_towers_100km_count',
                         plot_lte_100km_path,
                         "Torres LTE em um Raio de 100km",
                         is_distance=False)

    # --- Save Data with Distances (Optional) --- 
    # output_data_path = data_input_dir / "sao_paulo_datacenters_with_distances.geojson"
    # logging.info(f"Saving SP data centers with calculated distances to {output_data_path}...")
    # try:
    #     data_centers_sp.to_file(output_data_path, driver='GeoJSON')
    #     logging.info("Data saved successfully.")
    # except Exception as e:
    #     logging.error(f"Failed to save data: {e}")

    logging.info("Distance calculation and plotting script finished.")

if __name__ == "__main__":
    main() 