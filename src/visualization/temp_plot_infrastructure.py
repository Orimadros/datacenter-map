import matplotlib.pyplot as plt
import geopandas as gpd
from pathlib import Path
import contextily as ctx

def plot_census_tracts():
    """
    Create a visualization of all São Paulo census tracts with only boundaries.
    """
    print("Loading data files...")
    # Set paths
    base_dir = Path(__file__).parent.parent.parent
    
    # Census tracts
    census_tracts_path = base_dir / "data/model_input/brazil/sao_paulo_census_tracts_full.geojson"
    
    # Load data
    print("Loading all census tracts...")
    census_tracts = gpd.read_file(census_tracts_path)
    print(f"Loaded {len(census_tracts)} census tracts")
    
    # Create the plot
    print("Creating plot...")
    fig, ax = plt.subplots(figsize=(20, 20))
    
    # Plot only the boundaries of census tracts with no fill
    census_tracts.boundary.plot(
        color='black',
        linewidth=0.2,  # Thin lines since we're plotting all tracts
        ax=ax
    )
    
    # Add basemap 
    try:
        ctx.add_basemap(ax, crs=census_tracts.crs.to_string(), 
                       source=ctx.providers.CartoDB.Positron)
        print("Added basemap")
    except Exception as e:
        print(f"Could not add basemap: {e}")
    
    # Add a title
    ax.set_title('All São Paulo Census Tract Boundaries', fontsize=16)
    
    # Save the plot
    output_path = base_dir / "reports/figures/sao_paulo_all_tracts_boundaries.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved figure to {output_path}")
    
    # Show the plot path
    print(f"Figure created and saved to: {output_path}")
    
    plt.close()

if __name__ == "__main__":
    plot_census_tracts() 