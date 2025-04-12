import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
import numpy as np
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

def main():
    """
    Create visualizations of São Paulo census tracts and data centers.
    """
    logging.info("Starting visualization of census tracts and data centers...")
    
    # Define paths
    base_dir = Path.cwd()
    
    # Make sure we're using the correct Google Drive path
    google_drive_path = "GoogleDrive-leodavinci550@gmail.com"
    if google_drive_path not in str(base_dir):
        base_dir = Path(str(base_dir).replace("GoogleDrive-", google_drive_path))
    logging.info(f"Using base path: {base_dir}")
    
    # Load data centers
    processed_dir = base_dir / "data/processed/brasil"
    data_centers_path = processed_dir / "data_centers.geojson"
    
    if not data_centers_path.exists():
        data_centers_path = Path("/Users/leonardogomes/Library/CloudStorage/GoogleDrive-leodavinci550@gmail.com/My Drive/Elementum/GCB/Data Centers/dc_map_project/data/processed/brasil/data_centers.geojson")
    
    data_centers = gpd.read_file(data_centers_path)
    logging.info(f"Loaded {len(data_centers)} data centers with CRS: {data_centers.crs}")
    
    # Load census tracts with data centers
    model_input_dir = base_dir / "data/model_input/brasil"
    census_tracts_path = model_input_dir / "sao_paulo_census_tracts_full.geojson"
    
    if not census_tracts_path.exists():
        census_tracts_path = Path("/Users/leonardogomes/Library/CloudStorage/GoogleDrive-leodavinci550@gmail.com/My Drive/Elementum/GCB/Data Centers/dc_map_project/data/model_input/brasil/sao_paulo_census_tracts_full.geojson")
    
    census_tracts = gpd.read_file(census_tracts_path)
    logging.info(f"Loaded {len(census_tracts)} census tracts with CRS: {census_tracts.crs}")
    
    # Ensure both dataframes have the same CRS
    target_crs = "EPSG:4674"  # SIRGAS 2000
    if data_centers.crs != target_crs:
        data_centers = data_centers.to_crs(target_crs)
    if census_tracts.crs != target_crs:
        census_tracts = census_tracts.to_crs(target_crs)
    
    # Filter census tracts that have data centers
    tracts_with_dcs = census_tracts[census_tracts['has_data_center'] > 0].copy()
    logging.info(f"Found {len(tracts_with_dcs)} census tracts with data centers")
    logging.info(f"Total data centers in these tracts: {tracts_with_dcs['data_center_count'].sum()}")
    
    # Plot 1: Overview of all São Paulo census tracts and data centers
    logging.info("Creating overview plot...")
    
    # Convert to WebMercator for basemap compatibility
    census_tracts_web = census_tracts.to_crs(epsg=3857)
    data_centers_web = data_centers.to_crs(epsg=3857)
    tracts_with_dcs_web = tracts_with_dcs.to_crs(epsg=3857)
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(15, 15))
    
    # Plot all census tracts with light gray color
    census_tracts_web.plot(ax=ax, color='lightgray', edgecolor='gray', linewidth=0.1, alpha=0.5)
    
    # Plot census tracts with data centers with red color
    tracts_with_dcs_web.plot(ax=ax, color='red', edgecolor='darkred', linewidth=0.2, alpha=0.6)
    
    # Plot data centers with blue markers
    data_centers_web.plot(ax=ax, color='blue', markersize=20, marker='o', alpha=0.7)
    
    # Add basemap
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
    
    # Add title and description
    plt.title('Census Tracts and Data Centers in São Paulo', fontsize=16)
    ax.set_axis_off()
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label=f'Data Centers ({len(data_centers)})', alpha=0.7),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='red', markersize=10, label=f'Census Tracts with DCs ({len(tracts_with_dcs)})', alpha=0.6),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='lightgray', markersize=10, label=f'Other Census Tracts', alpha=0.5)
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12)
    
    # Save figure
    figures_dir = base_dir / "reports/figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    overview_path = figures_dir / "sao_paulo_census_tracts_dcs_overview.png"
    plt.savefig(overview_path, dpi=300, bbox_inches='tight')
    logging.info(f"Saved overview map to {overview_path}")
    
    # Plot 2: Focus on census tracts with data centers
    logging.info("Creating detailed plot of tracts with data centers...")
    
    fig, ax = plt.subplots(figsize=(15, 15))
    
    # Plot census tracts with data centers with color based on number of data centers
    tracts_with_dcs_web.plot(
        column='data_center_count',
        ax=ax,
        cmap='OrRd',
        legend=True,
        legend_kwds={'label': "Data Centers Count", 'orientation': "horizontal"},
        alpha=0.7
    )
    
    # Plot data centers
    data_centers_web.plot(ax=ax, color='blue', markersize=30, marker='o', alpha=0.7)
    
    # Add basemap
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
    
    # Add title
    plt.title('Census Tracts with Data Centers in São Paulo', fontsize=16)
    ax.set_axis_off()
    
    # Add legend for data centers
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label=f'Data Centers ({len(data_centers)})', alpha=0.7)
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12)
    
    # Save figure
    detail_path = figures_dir / "sao_paulo_census_tracts_dcs_detail.png"
    plt.savefig(detail_path, dpi=300, bbox_inches='tight')
    logging.info(f"Saved detailed map to {detail_path}")
    
    logging.info("Visualization completed!")

if __name__ == "__main__":
    main() 