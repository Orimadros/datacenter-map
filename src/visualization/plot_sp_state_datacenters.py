import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

def main():
    """Plot all data centers in the state of São Paulo."""
    logging.info("Starting São Paulo state data centers visualization...")
    
    # Load data
    logging.info("Loading data...")
    census_tracts = gpd.read_file('data/model_input/brasil/sao_paulo_census_tracts_full.geojson')
    data_centers = gpd.read_file('data/processed/brasil/data_centers.geojson')
    
    logging.info(f"Total census tracts in São Paulo state: {len(census_tracts)}")
    logging.info(f"Total data centers in Brazil: {len(data_centers)}")
    
    # Create a dissolved boundary for São Paulo state
    logging.info("Creating São Paulo state boundary...")
    sp_state_boundary = gpd.GeoDataFrame(geometry=[census_tracts.unary_union], crs=census_tracts.crs)
    
    # Ensure same CRS for data centers and state boundary
    if data_centers.crs != sp_state_boundary.crs:
        logging.info(f"Converting data centers from {data_centers.crs} to {sp_state_boundary.crs}")
        data_centers = data_centers.to_crs(sp_state_boundary.crs)
    
    # Filter data centers to those within the São Paulo state boundary
    sp_data_centers = data_centers[data_centers.geometry.within(sp_state_boundary.unary_union)]
    logging.info(f"São Paulo state data centers: {len(sp_data_centers)}")
    
    # Convert to Web Mercator for plotting with contextily basemap
    if sp_state_boundary.crs != 'EPSG:3857':
        sp_state_boundary = sp_state_boundary.to_crs('EPSG:3857')
        sp_data_centers = sp_data_centers.to_crs('EPSG:3857')
    
    # Plot
    logging.info("Creating plot...")
    fig, ax = plt.subplots(figsize=(15, 15))
    
    # Plot state boundary
    sp_state_boundary.plot(ax=ax, color='none', edgecolor='gray', linewidth=1, alpha=0.5)
    
    # Plot data centers
    sp_data_centers.plot(
        ax=ax,
        marker='o',
        color='blue',
        markersize=100,
        edgecolor='white',
        label=f'Data Centers ({len(sp_data_centers)})'
    )
    
    # Add basemap
    try:
        ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
    except Exception as e:
        logging.warning(f"Could not add basemap: {e}")
    
    # Add title and legend
    ax.set_title(f"Data Centers in São Paulo State (Total: {len(sp_data_centers)})", fontsize=16)
    ax.legend(loc='upper left')
    
    # Remove axes
    ax.set_axis_off()
    
    # Save plot
    output_dir = Path('outputs/figures')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'sao_paulo_state_datacenters.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logging.info(f"Saved plot to {output_path}")
    
    plt.show()
    logging.info("Visualization completed!")

if __name__ == "__main__":
    main() 