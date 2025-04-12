import os
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import contextily as ctx
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
from shapely.geometry import Point
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

def load_data():
    """Load census tract data and data centers."""
    project_root = Path(__file__).resolve().parent.parent.parent
    
    # Load census tracts for São Paulo state
    logging.info("Loading census tract data...")
    model_input_path = project_root / 'data' / 'model_input' / 'brasil'
    processed_path = project_root / 'data' / 'processed' / 'brasil'
    
    # Load census tracts with geometric data
    census_tracts_path = model_input_path / 'sao_paulo_census_tracts_full.geojson'
    logging.info(f"Loading census tracts from: {census_tracts_path}")
    
    if not census_tracts_path.exists():
        raise FileNotFoundError(f"Census tract data not found at: {census_tracts_path}")
    
    census_tracts_gdf = gpd.read_file(census_tracts_path)
    logging.info(f"Loaded {len(census_tracts_gdf)} census tracts")
    
    # Load data centers
    data_centers_path = processed_path / 'data_centers.geojson'
    logging.info(f"Loading data centers from: {data_centers_path}")
    
    if not data_centers_path.exists():
        raise FileNotFoundError(f"Data centers file not found at: {data_centers_path}")
    
    data_centers_gdf = gpd.read_file(data_centers_path)
    logging.info(f"Loaded {len(data_centers_gdf)} data centers with CRS: {data_centers_gdf.crs}")
    
    return census_tracts_gdf, data_centers_gdf

def filter_sao_paulo_city(census_tracts_gdf, data_centers_gdf):
    """
    Filter census tracts and data centers to include only São Paulo city.
    São Paulo city code (CD_MUN) is 3550308.
    """
    logging.info("Filtering data for São Paulo city...")
    
    # Extract first 7 characters of CD_SETOR to get CD_MUN
    if 'CD_SETOR' in census_tracts_gdf.columns:
        census_tracts_gdf['CD_MUN'] = census_tracts_gdf['CD_SETOR'].astype(str).str[:7]
    else:
        raise ValueError("CD_SETOR column not found in census tracts data")
    
    # Filter for São Paulo city
    sp_code = '3550308'
    sp_tracts = census_tracts_gdf[census_tracts_gdf['CD_MUN'] == sp_code].copy()
    logging.info(f"Filtered to {len(sp_tracts)} census tracts in São Paulo city")
    
    if len(sp_tracts) == 0:
        raise ValueError(f"No census tracts found for São Paulo city (code: {sp_code})")
    
    # Ensure same CRS for data centers and census tracts
    if data_centers_gdf.crs != sp_tracts.crs:
        logging.info(f"Converting data centers from {data_centers_gdf.crs} to {sp_tracts.crs}")
        data_centers_gdf = data_centers_gdf.to_crs(sp_tracts.crs)
    
    # Create a dissolved boundary for São Paulo city
    sp_boundary = sp_tracts.geometry.unary_union
    
    # Filter data centers within the boundary
    data_centers_in_sp = data_centers_gdf[data_centers_gdf.geometry.within(sp_boundary)].copy()
    logging.info(f"Found {len(data_centers_in_sp)} data centers in São Paulo city")
    
    # List data center names if available
    if len(data_centers_in_sp) > 0 and 'name' in data_centers_in_sp.columns:
        logging.info("Data centers in São Paulo city:")
        for name in data_centers_in_sp['name']:
            logging.info(f"- {name}")
    
    return sp_tracts, data_centers_in_sp

def load_model_and_predict(tracts_gdf):
    """Load trained model and predict probabilities for São Paulo city census tracts."""
    logging.info("Loading model and making predictions...")
    
    project_root = Path(__file__).resolve().parent.parent.parent
    models_dir = project_root / 'models'
    
    # Load model and scaler
    model_path = models_dir / 'sao_paulo_dc_model.h5'
    scaler_path = models_dir / 'sao_paulo_scaler.pkl'
    
    if not model_path.exists() or not scaler_path.exists():
        raise FileNotFoundError(f"Model or scaler not found at {models_dir}")
    
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)
    
    # Prepare features
    features = [
        'cell_tower_count',
        'cell_tower_density',
        'min_dist_to_substation_km',
        'min_dist_to_transmission_line_km',
        'viirs_mean'
    ]
    
    # Check if required features are available
    missing_features = [col for col in features if col not in tracts_gdf.columns]
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")
    
    X = tracts_gdf[features].fillna(0)
    
    # Scale features and predict
    X_scaled = scaler.transform(X)
    predictions = model.predict(X_scaled).flatten()  # Flatten to 1D array
    
    # Add predictions to GeoDataFrame
    tracts_gdf['predicted_prob'] = predictions
    
    # Find the top 5 tracts with highest probability
    top_tracts = tracts_gdf.sort_values('predicted_prob', ascending=False).head(5)
    logging.info("Top 5 census tracts with highest probability:")
    for idx, tract in top_tracts.iterrows():
        logging.info(f"- Tract {tract['CD_SETOR']}: {tract['predicted_prob']:.4f}")
    
    return tracts_gdf

def plot_heatmap(tracts_gdf, data_centers_gdf, save_path=None):
    """Plot heatmap of predicted probabilities with data center locations overlaid."""
    logging.info("Plotting heatmap of predicted probabilities with data center locations...")
    
    # Ensure both geodataframes have the same CRS and convert to Web Mercator for basemap
    web_mercator_crs = 'EPSG:3857'
    
    # Convert to Web Mercator for basemap compatibility
    if tracts_gdf.crs != web_mercator_crs:
        tracts_gdf = tracts_gdf.to_crs(web_mercator_crs)
    
    if data_centers_gdf.crs != web_mercator_crs:
        data_centers_gdf = data_centers_gdf.to_crs(web_mercator_crs)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(15, 12))
    
    # Plot census tracts colored by predicted probability
    tracts_gdf.plot(column='predicted_prob', cmap='Reds', linewidth=0.2, 
                    edgecolor='black', legend=True, alpha=0.7, ax=ax)
    
    # Overlay actual data center locations
    if len(data_centers_gdf) > 0:
        data_centers_gdf.plot(ax=ax, marker='o', color='blue', markersize=80, 
                             edgecolor='white', label='Data Centers')
    
    # Try to add basemap for context
    try:
        ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
    except Exception as e:
        logging.warning(f"Could not add basemap: {e}")
    
    # Set plot title and labels
    ax.set_title("Predicted Probability of Data Center Construction in São Paulo City\nwith Actual Data Center Locations", fontsize=16)
    
    # Add legend for data centers if applicable
    if len(data_centers_gdf) > 0:
        ax.legend()
    
    plt.tight_layout()
    
    # Save plot if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"Saved plot to {save_path}")
    
    plt.show()

def main():
    """Main function to generate heatmap for São Paulo city."""
    logging.info("Starting São Paulo city heatmap generation...")
    
    # Load data
    census_tracts_gdf, data_centers_gdf = load_data()
    
    # Filter for São Paulo city
    sp_tracts, sp_data_centers = filter_sao_paulo_city(census_tracts_gdf, data_centers_gdf)
    
    # Load model and predict
    sp_tracts = load_model_and_predict(sp_tracts)
    
    # Create output directory if it doesn't exist
    project_root = Path(__file__).resolve().parent.parent.parent
    output_dir = project_root / 'outputs' / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot heatmap
    plot_path = output_dir / 'sao_paulo_city_heatmap_vs_actual.png'
    plot_heatmap(sp_tracts, sp_data_centers, save_path=plot_path)
    
    logging.info("São Paulo city heatmap generation completed successfully!")

if __name__ == "__main__":
    main() 