import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from pathlib import Path
import contextily as ctx

def main():
    # 1. Construct the path to the cell tower data
    # Get the current file's directory
    current_dir = Path(__file__).resolve().parent
    # Go up one level to the project root
    project_root = current_dir.parent
    # Construct path to the data file
    towers_path = project_root / 'data' / 'raw' / 'cell_towers.csv'

    # 2. Read the CSV data
    print(f"Attempting to load data from: {towers_path}")
    try:
        # Only load necessary columns to save memory
        towers_df = pd.read_csv(towers_path, usecols=['lat', 'lon'])
        print(f"Successfully loaded {len(towers_df):,} records")
    except FileNotFoundError:
        print(f"Error: File not found at {towers_path}")
        print("Please ensure the cell towers data file exists at the specified location")
        return
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    # 3. Drop rows with missing or invalid coordinates
    initial_count = len(towers_df)
    towers_df.dropna(subset=['lat', 'lon'], inplace=True)
    # Remove potential non-numeric values and invalid coordinates
    towers_df = towers_df[pd.to_numeric(towers_df['lat'], errors='coerce').notnull()]
    towers_df = towers_df[pd.to_numeric(towers_df['lon'], errors='coerce').notnull()]
    towers_df = towers_df[(towers_df['lat'] >= -90) & (towers_df['lat'] <= 90)]
    towers_df = towers_df[(towers_df['lon'] >= -180) & (towers_df['lon'] <= 180)]

    final_count = len(towers_df)
    print(f"Proceeding with {final_count:,} records after dropping {initial_count - final_count:,} invalid/missing coordinate rows")

    if final_count == 0:
        print("Error: No valid coordinate data found")
        return

    # 4. Convert to GeoDataFrame
    try:
        gdf_towers = gpd.GeoDataFrame(
            towers_df,
            geometry=gpd.points_from_xy(towers_df["lon"], towers_df["lat"]),
            crs="EPSG:4326"  # WGS84 projection
        )
        print("Successfully converted to GeoDataFrame")
    except Exception as e:
        print(f"Error converting to GeoDataFrame: {e}")
        return

    # 5. Load a world map outline
    try:
        # Path to the local Natural Earth shapefile
        world_map_path = project_root / 'data' / 'raw' / '110m_cultural' / 'ne_110m_admin_0_countries.shp'
        if not world_map_path.exists():
            print(f"Error: World map shapefile not found at {world_map_path}")
            print("Please ensure the Natural Earth 110m admin 0 countries shapefile exists.")
            return
        world = gpd.read_file(world_map_path)
        print(f"Successfully loaded world map from {world_map_path}")
    except Exception as e:
        print(f"Error loading world map from local file: {e}")
        return

    # 6. Create the plot
    print("Generating plot...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 20))

    # Plot 1: Global Distribution
    world.plot(ax=ax1, color='lightgray', edgecolor='black', linewidth=0.5)
    gdf_towers.plot(ax=ax1, marker='.', color='blue', markersize=1, alpha=0.1)
    ax1.set_title('Global Distribution of Cell Towers', fontsize=16, pad=20)
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.set_xlim([-180, 180])
    ax1.set_ylim([-90, 90])
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Plot 2: Heatmap using hexbin
    hb = ax2.hexbin(
        gdf_towers.geometry.x, 
        gdf_towers.geometry.y,
        gridsize=50,
        cmap='YlOrRd',
        bins='log'
    )
    world.plot(ax=ax2, color='none', edgecolor='black', linewidth=0.5, alpha=0.5)
    ax2.set_title('Cell Tower Density Heatmap', fontsize=16, pad=20)
    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Latitude')
    ax2.set_xlim([-180, 180])
    ax2.set_ylim([-90, 90])
    ax2.grid(True, linestyle='--', alpha=0.6)
    plt.colorbar(hb, ax=ax2, label='Log10(count)')

    # Add some statistics
    stats_text = (
        f"Total Towers: {final_count:,}\n"
        f"Invalid/Missing Coordinates: {initial_count - final_count:,}"
    )
    fig.text(0.02, 0.98, stats_text, fontsize=10, va='top', ha='left')

    plt.tight_layout()
    
    # Save the plot
    output_path = project_root / 'cell_towers_distribution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    
    plt.show()
    print("Script finished")

if __name__ == "__main__":
    main() 